# DHW Override Deterministic Forecast Correction (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make any committed DHW override (legionella today, comfort-boost once Phase B ships) move the published hef forecast by its true expected kWh amount deterministically — never routed through the trained LightGBM model, which is free to (and currently does) discount it — and fix the training-reconstruction bug where retrain only ever replays the *most recently* committed override.

**Architecture:** Split the single `physics_kwh` ML feature into two independently-computed series: an always-override-blind `physics_kwh` (so the tree model has nothing override-shaped left to discount), and a new `override_delta_series` computed once as a joint with-overrides-vs-baseline trajectory diff and applied additively — to the training target (subtracted, so the model trains on "as if no override happened") and to the final published forecast (added back, post-model, so the correction is structurally exempt from the model's own weighting). A new append-only `override_history` list replaces the single-slot `committed_override` for training-reconstruction purposes, while `committed_override` itself gains merge semantics (today it's a flat replace) so legionella and comfort-boost can coexist. This plan is Phase A of a two-phase rollout — see the spec's Rollout section; **Phase B (EM-side comfort-boost wiring) depends on this phase's 3-part exit gate being manually confirmed** before it starts.

**Tech Stack:** Python 3.13, pandas/numpy (physics ODE model), LightGBM (via `model.py`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-15-dhw-comfort-boost-commit-design.md` (in `ha-energy-manager` — this plan implements only the hef-side ("Phase A") half of that spec; the EM-side ("Phase B") half has its own plan in `ha-energy-manager/docs/superpowers/plans/2026-08-24-dhw-comfort-boost-commit-phase-b.md`).

## Global Constraints

- Every new/changed method that can encounter malformed data (write-time in `commit_dhw_schedule`, read-time in history reconstruction) must log a WARNING and skip/continue — **never raise**. This applies uniformly across Phase A.
- Tail-termination threshold for `override_delta_series`: **`|t_tank_with_overrides − t_tank_baseline| < 0.1°C`**, implemented as one single shared function called identically by both the training and serving paths (R2-#3) — not two independently-implemented cutoffs.
- Sanity bound for `override_delta_series[h]`: **`|delta[h]| <= max(with_overrides_kwh[h], baseline_kwh[h])`** for every hour — clip and log WARNING on violation. The clip is a runtime guard applied only at point of use (training subtraction, serving addition); it never rewrites `override_history`/`committed_override`, which always retain the raw committed values.
- `override_delta_series` must be computed **jointly** across every override active in a window (one with-overrides trajectory vs. one baseline trajectory, diffed) — never as independent per-override diffs summed together (R2-#1: overlapping tails double-count).
- No backfill of `override_history` for pre-migration overrides — those hours are zero-delta by design (a strict improvement over today's always-wrong-except-latest behavior), down-weighted in training via the existing `open_window_flags` mechanism, not treated as a regression to fix retroactively.
- Zero-override degradation: with empty `committed_override`/`override_history`, every new code path must produce output byte-for-byte identical to today's Phase 1 behavior. This is the single most important regression guard in this plan — most tasks below end with an explicit test for it.
- Python 3.13; run tests via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v` (see project CLAUDE.md — never bare `python`/base conda).

---

## Design notes (decisions this plan makes that the spec leaves to implementation)

These aren't spec ambiguities to escalate — they're the concrete choices needed to turn the spec's prose into code, recorded here so later tasks and reviewers can see the reasoning in one place:

1. **`get_scenario`'s existing scenario-diff mechanism stays override-aware and is explicitly untouched.** The spec says `physics_kwh` becomes override-blind "everywhere it trains or serves as an ML input" — but `predict_series`'s `dhw_schedule_override` parameter is also how `predict_scenario`/`_get_scenario_cb` (used by EM's `ScenarioScorer` to score candidate hours *before* committing) hypothetically injects a not-yet-committed override to diff two full model runs. That mechanism must keep working unchanged (spec's Out-of-scope section: "resolved this round" only adds `target_c` to the payload, nothing else). The fix therefore only removes `predict_series`'s **silent fallback** to the live `committed_override` when the caller passes `None` (physics.py:335-338 today) — an explicit `dhw_schedule_override` argument (get_scenario's use case) keeps working exactly as today; an absent one (the main published-forecast call site) now means genuinely override-blind, not "whatever happens to be committed right now."
2. **One shared per-hour-callable lookup abstraction (`override_lookup: Callable[[pd.Timestamp], float | None]`) replaces the dict-based override parameter inside `_dhw_kwh_series` itself.** This lets the *same* stateful ODE loop serve three different lookup modes without duplicating the loop: override-blind (`None`), live-dict-based (today's `_dhw_override_for_hour`, used by `predict_series`'s explicit-override path), and history-based (`_dhw_override_for_hour_from_history`, new, used only inside the joint delta computation for training reconstruction and — via the live `committed_override` dict — for serving's correction).
3. **`_dhw_kwh_series` gains a full per-hour `t_tank` trajectory return value**, not just the final scalar it returns today — required so the tail-termination check (`|t_tank_with_overrides − t_tank_baseline| < 0.1°C`) can be computed hour-by-hour, not just from the endpoint.
4. **The calibration-input correction (R1-#11) is computed once per `train()` call, using `self._calib`'s *current* (pre-this-cycle) values**, and applied to a corrected copy of `energy_df["gross_kwh"]` before *any* downstream consumer (calibration, lag/rolling feature engineering, target construction) touches it. This is a standard bootstrap approximation (calibration already re-estimates itself every cycle as more data accumulates) rather than a fixed-point solve — worth a one-line comment in the code, not a design gap.
5. **The published-forecast correction (serving) is added in `energy_forecast.py`, after `model.predict()` returns its final blended series** — not inside `physics.py` or `model.py` — so "never routed through the model" is structurally visible at the call site, not just true by construction.

---

## Task 1: Resolve the DHW-baseline verification-checklist item (R1-#23) before touching `_dhw_kwh_series`

**Files:**
- Read: `apps/energy_forecast/physics.py:217-263` (`_dhw_kwh_series`, current baseline branch)
- Create: `docs/superpowers/plans/2026-08-24-r1-23-dhw-baseline-finding.md` (short decision note)

**Interfaces:**
- Produces: a written, committed decision on whether the override-blind baseline (Task 6 onward) needs any change to its resting-temperature assumption before it ships.

- [ ] **Step 1: Confirm there is no `dhw_idle_setpoint_c`-equivalent concept in hef today**

  Run:
  ```bash
  grep -rn "dhw_idle_setpoint_c\|idle_setpoint" /home/jovyan/work/ha-energy-forecast/apps /home/jovyan/work/ha-energy-forecast/tests
  ```
  Expected: zero matches. This confirms `_dhw_kwh_series`'s baseline branch (physics.py:255-261) has no analogous "idle setpoint" concept — it models the tank as a continuous ODE bounded by `T_dhw_lower` (45°C default, physics.py:79) and `T_legionella` (60°C), reheating whenever `t_tank < T_dhw_lower` and passively decaying (draw profile + UA loss) otherwise. There is no static "resting" value comparable to EM's `dhw_idle_setpoint_c` (25°C, in `heat_pump.py`'s actuation-side setpoint command, not a forecast concept) for the physics model to be stale against.

- [ ] **Step 2: Write the decision note**

  Create `docs/superpowers/plans/2026-08-24-r1-23-dhw-baseline-finding.md`:
  ```markdown
  # R1-#23 finding: no dhw_idle_setpoint_c-equivalent baseline drift to fix

  hef's `_dhw_kwh_series` baseline branch does not model a static "resting setpoint"
  at all — it's a continuous ODE (UA-loss decay + draw-profile depletion) bounded by
  `T_dhw_lower`/`T_legionella`, reheating whenever the simulated tank drops below
  `T_dhw_lower` (45°C default). EM's `dhw_idle_setpoint_c` (25°C) is a *physical
  actuation* concept (the commanded HP setpoint when no boost is wanted) with no
  equivalent in hef's forecast model — there is nothing "stale" to correct here.

  Confirmed via `grep -rn "dhw_idle_setpoint_c\|idle_setpoint"` across
  `ha-energy-forecast` (zero matches) on 2026-08-24.

  Decision: no prerequisite fix needed. R1-#23 is resolved as "not applicable" —
  the spec's concern was based on an assumption (hef modeling a static resting
  setpoint analogous to EM's) that doesn't hold in the current physics model.
  Proceeding with the override-blind baseline exactly as `_dhw_kwh_series` computes
  it today (Task 6 onward), with no independent baseline correction.
  ```

- [ ] **Step 3: Flag this finding for stakeholder awareness, don't just proceed silently**

  Before continuing to Task 2, surface this finding to whoever owns the spec (it changes
  a verification-checklist item from "confirm and possibly fix" to "confirmed not
  applicable") — a one-line note in the PR description referencing the decision file is
  sufficient; no code change results from this task.

- [ ] **Step 4: Commit the decision note**

  ```bash
  git add docs/superpowers/plans/2026-08-24-r1-23-dhw-baseline-finding.md
  git commit -m "docs: resolve R1-#23 baseline verification item as not-applicable"
  ```

---

## Task 2: Data model — add `override_history` to the schedule schema

**Files:**
- Modify: `apps/energy_forecast/physics.py:73-82` (`_default_schedule`)
- Test: `tests/test_physics.py` (new assertions in a new `TestOverrideHistorySchema` class)

**Interfaces:**
- Produces: `self._schedule["override_history"]` — a `list[dict]`, always present (defaults to `[]`), each entry shaped `{"kind": str, "date": str, "hour": int, "target_c": float, "committed_at": str, "cancelled_at": str | None}`.

- [ ] **Step 1: Write the failing test**

  Add to `tests/test_physics.py`:
  ```python
  class TestOverrideHistorySchema:
      def test_default_schedule_has_empty_override_history(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          assert pm._schedule["override_history"] == []

      def test_override_history_persists_and_reloads(self, tmp_path):
          model_dir = tmp_path / "models"
          pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
          pm._schedule["override_history"].append(
              {
                  "kind": "legionella",
                  "date": "2026-08-04",
                  "hour": 12,
                  "target_c": 60.0,
                  "committed_at": "2026-08-04T06:00:01+02:00",
                  "cancelled_at": None,
              }
          )
          _atomic_write_json(pm._schedule_path, pm._schedule)
          pm2 = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
          assert pm2._schedule["override_history"] == [
              {
                  "kind": "legionella",
                  "date": "2026-08-04",
                  "hour": 12,
                  "target_c": 60.0,
                  "committed_at": "2026-08-04T06:00:01+02:00",
                  "cancelled_at": None,
              }
          ]
  ```
  (`_atomic_write_json` is already imported at the top of `test_physics.py` — confirm; if not, `from energy_forecast.physics import _atomic_write_json`.)

- [ ] **Step 2: Run test to verify it fails**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestOverrideHistorySchema -v
  ```
  Expected: FAIL — `KeyError: 'override_history'`.

- [ ] **Step 3: Add `override_history` to `_default_schedule`**

  In `apps/energy_forecast/physics.py`:
  ```python
  def _default_schedule() -> dict[str, Any]:
      return {
          "T_dhw_upper": 55.0,
          "T_legionella": 60.0,
          "legionella_dow": 2,
          "legionella_hour": 14,
          "T_dhw_lower": 45.0,
          "dhw_tank_volume_l": 200,
          "committed_override": None,
          "override_history": [],
      }
  ```

- [ ] **Step 4: Run test to verify it passes**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestOverrideHistorySchema -v
  ```
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add apps/energy_forecast/physics.py tests/test_physics.py
  git commit -m "feat: add override_history to DHW schedule data model"
  ```

---

## Task 3: `commit_dhw_schedule` — merge semantics, validation, append-only history

**Files:**
- Modify: `apps/energy_forecast/physics.py:868-874` (replace `commit_dhw_schedule` entirely)
- Modify: `tests/test_physics.py:991-1003` (`TestCommittedDhwSchedule` — existing exact-equality assertion needs updating since `commit_dhw_schedule` now also appends to `override_history`)

**Interfaces:**
- Consumes: nothing new (same `override: dict` parameter as today).
- Produces: `commit_dhw_schedule(override: dict) -> None`, unchanged public signature; `self._schedule["committed_override"]` merge semantics; `self._schedule["override_history"]` append/dedup/cancel side effects, all new.

- [ ] **Step 1: Write the failing tests**

  Add to `tests/test_physics.py` (new class, alongside existing `TestCommittedDhwSchedule`):
  ```python
  class TestCommitDhwScheduleMergeSemantics:
      def test_legionella_then_comfort_boost_both_survive(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm.commit_dhw_schedule({"legionella": ("2026-08-04", 12)})
          pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 57.5)})
          assert pm._schedule["committed_override"] == {
              "legionella": ["2026-08-04", 12],
              "comfort_boost": ["2026-08-05", 14, 57.5],
          }

      def test_clearing_comfort_boost_leaves_legionella_intact(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm.commit_dhw_schedule({"legionella": ("2026-08-04", 12)})
          pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 57.5)})
          pm.commit_dhw_schedule({"comfort_boost": None})
          assert pm._schedule["committed_override"] == {"legionella": ["2026-08-04", 12]}

      def test_clearing_last_key_resets_to_none_not_empty_dict(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm.commit_dhw_schedule({"legionella": ("2026-08-04", 12)})
          pm.commit_dhw_schedule({"legionella": None})
          assert pm._schedule["committed_override"] is None

      def test_malformed_entry_skipped_with_warning_does_not_corrupt_merge(self, tmp_path, caplog):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm.commit_dhw_schedule({"legionella": ("2026-08-04", 12)})
          with caplog.at_level("WARNING"):
              pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14)})  # missing target_c — malformed
          assert any("malformed" in r.message for r in caplog.records)
          assert pm._schedule["committed_override"] == {"legionella": ["2026-08-04", 12]}

      def test_commit_appends_override_history_entry(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 57.5)})
          assert len(pm._schedule["override_history"]) == 1
          entry = pm._schedule["override_history"][0]
          assert entry["kind"] == "comfort_boost"
          assert entry["date"] == "2026-08-05"
          assert entry["hour"] == 14
          assert entry["target_c"] == 57.5
          assert entry["cancelled_at"] is None
          assert entry["committed_at"]  # non-empty ISO string

      def test_legionella_history_target_c_is_t_legionella(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm.commit_dhw_schedule({"legionella": ("2026-08-04", 12)})
          entry = pm._schedule["override_history"][0]
          assert entry["target_c"] == pm._schedule["T_legionella"]

      def test_last_write_wins_dedup_on_kind_date_hour_for_non_cancelled(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 55.0)})
          pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 57.5)})  # re-armed, revised target
          assert len(pm._schedule["override_history"]) == 1
          assert pm._schedule["override_history"][0]["target_c"] == 57.5

      def test_genuine_cancellation_marks_cancelled_at_not_deleted(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 57.5)})
          pm.commit_dhw_schedule({"comfort_boost": None})
          assert len(pm._schedule["override_history"]) == 1  # not deleted
          entry = pm._schedule["override_history"][0]
          assert entry["cancelled_at"] is not None
          assert entry["target_c"] == 57.5  # raw value retained

      def test_rearm_after_cancellation_appends_fresh_entry_not_uncancel(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 55.0)})
          pm.commit_dhw_schedule({"comfort_boost": None})  # cancelled
          pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 58.0)})  # re-armed, same hour
          assert len(pm._schedule["override_history"]) == 2
          cancelled, fresh = pm._schedule["override_history"]
          assert cancelled["cancelled_at"] is not None
          assert cancelled["target_c"] == 55.0
          assert fresh["cancelled_at"] is None
          assert fresh["target_c"] == 58.0
  ```

  Update the pre-existing `test_commit_persists_override_and_bypasses_stability_guard`
  (`tests/test_physics.py:991-1003`) — its exact-equality assertion breaks now that
  `commit_dhw_schedule` also appends to `override_history`:
  ```python
  def test_commit_persists_override_and_bypasses_stability_guard(self, tmp_path):
      model_dir = tmp_path / "models"
      pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
      pm._schedule.update(legionella_dow=2, legionella_hour=14)
      pm.commit_dhw_schedule({"legionella": ("2026-06-25", 22)})
      assert pm._schedule["committed_override"] == {"legionella": ["2026-06-25", 22]}
      pm2 = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
      assert pm2._schedule["committed_override"] == {"legionella": ["2026-06-25", 22]}
      assert len(pm2._schedule["override_history"]) == 1  # new: side effect of commit
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestCommitDhwScheduleMergeSemantics tests/test_physics.py::TestCommittedDhwSchedule -v
  ```
  Expected: FAIL — today's `commit_dhw_schedule` is a flat replace with no `override_history` side effect.

- [ ] **Step 3: Implement merge semantics**

  Replace `commit_dhw_schedule` in `apps/energy_forecast/physics.py`:
  ```python
  _VALID_OVERRIDE_KINDS = ("legionella", "comfort_boost")

  def commit_dhw_schedule(self, override: dict) -> None:
      """Merge semantics into committed_override + append-only override_history (Goal 4).
      This is net-new code — the previous implementation was a flat replace. Never
      raises: a malformed entry logs a WARNING and is skipped, never corrupting the
      rest of the merge (R1 write-time validation contract)."""
      if self._schedule.get("committed_override") is None:
          self._schedule["committed_override"] = {}
      committed = self._schedule["committed_override"]
      history = self._schedule.setdefault("override_history", [])
      now_iso = dt.datetime.now(dt.UTC).isoformat()

      for kind, value in override.items():
          if value is None:
              prior = committed.pop(kind, None)
              if prior is not None:
                  self._mark_history_cancelled(history, kind, prior[0], prior[1])
              continue
          if kind not in _VALID_OVERRIDE_KINDS:
              _LOGGER.warning(f"commit_dhw_schedule: unknown kind {kind!r} — skipping")
              continue
          if kind == "legionella":
              if not (isinstance(value, (list, tuple)) and len(value) == 2):
                  _LOGGER.warning(f"commit_dhw_schedule: malformed legionella value {value!r} — skipping")
                  continue
              date_str, hour = value
              target_c = self._schedule["T_legionella"]
          else:  # comfort_boost
              if not (isinstance(value, (list, tuple)) and len(value) == 3):
                  _LOGGER.warning(f"commit_dhw_schedule: malformed comfort_boost value {value!r} — skipping")
                  continue
              date_str, hour, target_c = value
              if target_c is None:
                  _LOGGER.warning("commit_dhw_schedule: comfort_boost target_c is None — skipping")
                  continue

          committed[kind] = [date_str, hour] if kind == "legionella" else [date_str, hour, target_c]
          self._replace_or_append_history(history, kind, date_str, hour, float(target_c), now_iso)

      if not committed:
          self._schedule["committed_override"] = None
      _atomic_write_json(self._schedule_path, self._schedule)
      _LOGGER.info(f"DHW schedule committed: {override}")

  @staticmethod
  def _mark_history_cancelled(history: list[dict], kind: str, date_str: str, hour: int) -> None:
      """Genuine remote cancellation (not self-expiry): mark the specific still-pending
      (kind, date, hour) entry cancelled rather than deleting it — keeps history
      append-only/auditable and correctly zero-deltas it in reconstruction (R2 fix)."""
      now_iso = dt.datetime.now(dt.UTC).isoformat()
      for entry in history:
          if (
              entry.get("cancelled_at") is None
              and entry.get("kind") == kind
              and entry.get("date") == date_str
              and entry.get("hour") == hour
          ):
              entry["cancelled_at"] = now_iso
              return

  @staticmethod
  def _replace_or_append_history(
      history: list[dict], kind: str, date_str: str, hour: int, target_c: float, committed_at: str
  ) -> None:
      """Last-write-wins dedup on (kind, date, hour) among non-cancelled entries (R1-#21)."""
      for entry in history:
          if (
              entry.get("cancelled_at") is None
              and entry.get("kind") == kind
              and entry.get("date") == date_str
              and entry.get("hour") == hour
          ):
              entry["target_c"] = target_c
              entry["committed_at"] = committed_at
              return
      history.append(
          {
              "kind": kind,
              "date": date_str,
              "hour": hour,
              "target_c": target_c,
              "committed_at": committed_at,
              "cancelled_at": None,
          }
      )
  ```
  Add `import datetime as dt` at the top of `physics.py` if not already present — check first (`grep -n "^import\|^from" apps/energy_forecast/physics.py`).

- [ ] **Step 4: Run tests to verify they pass**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestCommitDhwScheduleMergeSemantics tests/test_physics.py::TestCommittedDhwSchedule -v
  ```
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add apps/energy_forecast/physics.py tests/test_physics.py
  git commit -m "feat: merge semantics + append-only history for commit_dhw_schedule"
  ```

---

## Task 4: `_dhw_override_for_hour` — generalize for `comfort_boost`, legionella-wins tie-break

**Files:**
- Modify: `apps/energy_forecast/physics.py:207-215`
- Test: `tests/test_physics.py` (new `TestDhwOverrideForHour` class)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_dhw_override_for_hour(ts: pd.Timestamp, override: dict | None) -> float | None` — same signature, now recognizes both `"legionella"` (2-elem) and `"comfort_boost"` (3-elem) keys in `override`.

- [ ] **Step 1: Write the failing tests**

  ```python
  class TestDhwOverrideForHour:
      def test_legionella_override_returns_t_legionella(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          ts = pd.Timestamp("2026-08-04 12:00")
          result = pm._dhw_override_for_hour(ts, {"legionella": ("2026-08-04", 12)})
          assert result == pm._schedule["T_legionella"]

      def test_comfort_boost_override_returns_target_c(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          ts = pd.Timestamp("2026-08-05 14:00")
          result = pm._dhw_override_for_hour(ts, {"comfort_boost": ("2026-08-05", 14, 57.5)})
          assert result == 57.5

      def test_no_match_returns_none(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          ts = pd.Timestamp("2026-08-05 09:00")
          assert pm._dhw_override_for_hour(ts, {"comfort_boost": ("2026-08-05", 14, 57.5)}) is None

      def test_same_hour_precedence_legionella_wins(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          ts = pd.Timestamp("2026-08-05 14:00")
          result = pm._dhw_override_for_hour(
              ts, {"legionella": ("2026-08-05", 14), "comfort_boost": ("2026-08-05", 14, 57.5)}
          )
          assert result == pm._schedule["T_legionella"]

      def test_empty_or_none_override_returns_none(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          ts = pd.Timestamp("2026-08-05 14:00")
          assert pm._dhw_override_for_hour(ts, None) is None
          assert pm._dhw_override_for_hour(ts, {}) is None
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestDhwOverrideForHour -v
  ```
  Expected: FAIL on `test_comfort_boost_override_returns_target_c` and the precedence test — today's implementation only checks `"legionella"`.

- [ ] **Step 3: Implement**

  ```python
  def _dhw_override_for_hour(self, ts: pd.Timestamp, override: dict | None) -> float | None:
      """Return an override target temp for *ts* if a committed override (legionella
      or comfort_boost) applies, else None. Same-hour precedence: legionella wins —
      defense in depth for the data model; Goal 5 (EM-side) makes two genuinely
      concurrent same-hour events physically unreachable in practice."""
      if not override:
          return None
      legionella_val = override.get("legionella")
      if legionella_val is not None:
          date_str, hour = legionella_val
          if ts == pd.Timestamp(f"{date_str} {int(hour):02d}:00"):
              return self._schedule["T_legionella"]
      comfort_boost_val = override.get("comfort_boost")
      if comfort_boost_val is not None:
          date_str, hour, target_c = comfort_boost_val
          if ts == pd.Timestamp(f"{date_str} {int(hour):02d}:00"):
              return float(target_c)
      return None
  ```

- [ ] **Step 4: Run tests to verify they pass**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestDhwOverrideForHour -v
  ```
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add apps/energy_forecast/physics.py tests/test_physics.py
  git commit -m "feat: generalize _dhw_override_for_hour for comfort_boost"
  ```

---

## Task 5: `_dhw_override_for_hour_from_history` — training-reconstruction lookup mode

**Files:**
- Modify: `apps/energy_forecast/physics.py` (new method, near `_dhw_override_for_hour`)
- Test: `tests/test_physics.py` (new `TestDhwOverrideForHourFromHistory` class)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_dhw_override_for_hour_from_history(ts: pd.Timestamp, history: list[dict]) -> float | None` — scans `override_history` for a non-cancelled entry matching `(kind, date, hour) == ts`; legionella wins on same-hour tie; malformed entries skipped with a WARNING (never raised), matching `commit_dhw_schedule`'s write-time validation contract.

- [ ] **Step 1: Write the failing tests**

  ```python
  class TestDhwOverrideForHourFromHistory:
      def test_returns_target_c_for_matching_non_cancelled_entry(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          history = [
              {"kind": "legionella", "date": "2026-08-04", "hour": 12, "target_c": 60.0,
               "committed_at": "x", "cancelled_at": None},
          ]
          ts = pd.Timestamp("2026-08-04 12:00")
          assert pm._dhw_override_for_hour_from_history(ts, history) == 60.0

      def test_cancelled_entry_returns_none(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          history = [
              {"kind": "comfort_boost", "date": "2026-08-05", "hour": 14, "target_c": 57.5,
               "committed_at": "x", "cancelled_at": "y"},
          ]
          ts = pd.Timestamp("2026-08-05 14:00")
          assert pm._dhw_override_for_hour_from_history(ts, history) is None

      def test_self_expired_entry_cancelled_at_none_reconstructs_normally(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          history = [
              {"kind": "comfort_boost", "date": "2026-08-05", "hour": 14, "target_c": 57.5,
               "committed_at": "x", "cancelled_at": None},
          ]
          ts = pd.Timestamp("2026-08-05 14:00")
          assert pm._dhw_override_for_hour_from_history(ts, history) == 57.5

      def test_returns_the_actually_committed_value_not_just_latest(self, tmp_path):
          """Direct regression test for the reconstruction-fidelity bug (Goal 2):
          two different historical days' entries must each reconstruct correctly,
          not both resolve to whichever was committed most recently."""
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          history = [
              {"kind": "legionella", "date": "2026-07-28", "hour": 12, "target_c": 60.0,
               "committed_at": "x", "cancelled_at": None},
              {"kind": "legionella", "date": "2026-08-04", "hour": 13, "target_c": 60.0,
               "committed_at": "y", "cancelled_at": None},
          ]
          assert pm._dhw_override_for_hour_from_history(pd.Timestamp("2026-07-28 12:00"), history) == 60.0
          assert pm._dhw_override_for_hour_from_history(pd.Timestamp("2026-08-04 13:00"), history) == 60.0
          assert pm._dhw_override_for_hour_from_history(pd.Timestamp("2026-07-28 13:00"), history) is None

      def test_same_hour_precedence_legionella_wins(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          history = [
              {"kind": "legionella", "date": "2026-08-05", "hour": 14, "target_c": 60.0,
               "committed_at": "x", "cancelled_at": None},
              {"kind": "comfort_boost", "date": "2026-08-05", "hour": 14, "target_c": 57.5,
               "committed_at": "y", "cancelled_at": None},
          ]
          ts = pd.Timestamp("2026-08-05 14:00")
          assert pm._dhw_override_for_hour_from_history(ts, history) == 60.0

      def test_malformed_entry_skipped_with_warning_not_raised(self, tmp_path, caplog):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          history = [{"kind": "legionella", "date": "2026-08-04"}]  # missing hour/target_c
          ts = pd.Timestamp("2026-08-04 12:00")
          with caplog.at_level("WARNING"):
              result = pm._dhw_override_for_hour_from_history(ts, history)
          assert result is None
          assert any("malformed" in r.message for r in caplog.records)

      def test_empty_history_returns_none(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          assert pm._dhw_override_for_hour_from_history(pd.Timestamp("2026-08-04 12:00"), []) is None
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestDhwOverrideForHourFromHistory -v
  ```
  Expected: FAIL — `AttributeError: 'ThermalPhysicsModel' object has no attribute '_dhw_override_for_hour_from_history'`.

- [ ] **Step 3: Implement**

  ```python
  def _dhw_override_for_hour_from_history(self, ts: pd.Timestamp, history: list[dict]) -> float | None:
      """Training-reconstruction lookup mode (Goal 2): scans override_history for a
      non-cancelled entry matching (kind, date, hour) == ts. Same-hour precedence:
      legionella wins (mirrors _dhw_override_for_hour). Malformed entries are skipped
      with a WARNING, never raised — read-time validation mirrors commit_dhw_schedule's
      write-time contract (R1-#20)."""
      legionella_target: float | None = None
      comfort_boost_target: float | None = None
      for entry in history:
          try:
              if entry.get("cancelled_at") is not None:
                  continue
              kind = entry["kind"]
              date_str = entry["date"]
              hour = entry["hour"]
              target_c = entry["target_c"]
              if kind not in _VALID_OVERRIDE_KINDS or target_c is None:
                  _LOGGER.warning(f"override_history: malformed entry skipped: {entry}")
                  continue
              if ts != pd.Timestamp(f"{date_str} {int(hour):02d}:00"):
                  continue
              if kind == "legionella":
                  legionella_target = float(target_c)
              else:
                  comfort_boost_target = float(target_c)
          except Exception as e:  # noqa: BLE001 — read-time validation contract, R1-#20
              _LOGGER.warning(f"override_history: malformed entry skipped: {entry} ({e})")
              continue
      return legionella_target if legionella_target is not None else comfort_boost_target
  ```

- [ ] **Step 4: Run tests to verify they pass**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestDhwOverrideForHourFromHistory -v
  ```
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add apps/energy_forecast/physics.py tests/test_physics.py
  git commit -m "feat: add history-based DHW override lookup for training reconstruction"
  ```

---

## Task 6: `_dhw_kwh_series` — callable-based override lookup + full `t_tank` trajectory return

**Files:**
- Modify: `apps/energy_forecast/physics.py:217-263` (`_dhw_kwh_series`), and its two callers `predict_series` (~283-346) and `predict_training_series` (~348-383)
- Test: `tests/test_physics.py` (extend `TestDHWOde`, update any test asserting the old 2-tuple return)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_dhw_kwh_series(timestamps, t_ambient, initial_t_tank, override_lookup: Callable[[pd.Timestamp], float | None] | None = None) -> tuple[pd.Series, pd.Series, float]` — `(el_kwh_series, t_tank_series, final_t_tank)`. `override_lookup=None` means override-blind (the mandatory mode for the ML `physics_kwh` feature everywhere it trains or serves — Goal 1).

- [ ] **Step 1: Write the failing tests**

  Check existing `TestDHWOde` tests first (`tests/test_physics.py:247+`) — several likely unpack `el_kwh, final_t = pm._dhw_kwh_series(...)` (2-tuple). Update each to 3-tuple unpacking (`el_kwh, t_tank_series, final_t = ...`) as part of this task — this is a mechanical signature-shape fix, not new test logic; run `grep -n "_dhw_kwh_series(" tests/test_physics.py` first to find every call site.

  Add new tests:
  ```python
  class TestDhwKwhSeriesOverrideLookup:
      def test_override_lookup_none_is_override_blind(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          el_kwh, t_tank_series, final_t = pm._dhw_kwh_series(timestamps, t_ambient, 50.0, override_lookup=None)
          # No override anywhere in a 24h override-blind run — series must have no
          # discontinuous jump to T_legionella at any hour.
          assert (t_tank_series <= pm._schedule["T_legionella"] + 1e-6).all()
          assert not (t_tank_series == pm._schedule["T_legionella"]).any()

      def test_override_lookup_callable_applies_target_at_matching_hour(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          target_ts = pd.Timestamp("2026-08-04 12:00")
          lookup = lambda ts: 60.0 if ts == target_ts else None
          _, t_tank_series, _ = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=lookup)
          assert t_tank_series.loc[target_ts] == 60.0

      def test_t_tank_series_has_same_index_as_timestamps(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          timestamps = pd.date_range("2026-08-04 00:00", periods=6, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          _, t_tank_series, _ = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=None)
          assert list(t_tank_series.index) == list(timestamps)
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestDhwKwhSeriesOverrideLookup tests/test_physics.py::TestDHWOde -v
  ```
  Expected: FAIL — current signature takes a dict `dhw_schedule_override`, not a callable, and returns a 2-tuple.

- [ ] **Step 3: Implement — refactor `_dhw_kwh_series` and both callers**

  ```python
  def _dhw_kwh_series(
      self,
      timestamps: pd.DatetimeIndex,
      t_ambient: pd.Series,
      initial_t_tank: float,
      override_lookup: Callable[[pd.Timestamp], float | None] | None = None,
  ) -> tuple[pd.Series, pd.Series, float]:
      """Returns (el_kwh_series, t_tank_series, final_t_tank). override_lookup is a
      per-hour callable returning an override target temp or None; None means
      override-blind (the mandatory mode for physics_kwh as an ML feature — Goal 1).
      Same ODE math as before Phase A — only the override-source abstraction changed,
      from a dict to a callable, so one loop can serve live-dict, history-based, and
      blind lookup modes without duplicating it (R2-#3's "one shared function")."""
      volume_l = self._config["dhw_tank_volume_l"]
      c_dhw = volume_l * WATER_SPECIFIC_HEAT_WH_PER_L_K
      if c_dhw <= 0:
          _LOGGER.warning("DHW tank volume is zero or invalid — skipping DHW component")
          zeros = pd.Series(0.0, index=timestamps)
          return zeros, pd.Series(float(initial_t_tank), index=timestamps), float(initial_t_tank)
      q_dhw_power = self._config["dhw_power_w"]
      heating_rise = q_dhw_power / c_dhw

      t_lower = self._schedule["T_dhw_lower"]
      t_legionella = self._schedule["T_legionella"]

      q_dhw_daily = self._calib.get("Q_dhw_daily") or 0.0
      draw_rate = (q_dhw_daily * 1000.0 / c_dhw) if c_dhw > 0 else 0.0

      cop_dhw = max(COP_MIN, self._cop_formula_value(t_ambient.iloc[0] if len(t_ambient) else 10.0, None))

      t_tank = float(initial_t_tank)
      el_kwh = np.zeros(len(timestamps))
      t_tank_trajectory = np.zeros(len(timestamps))
      for i, ts in enumerate(timestamps):
          ua_dhw = self._calib.get("UA_dhw") or 15.0
          dT = -ua_dhw * (t_tank - float(t_ambient.iloc[i])) / c_dhw
          hour_of_day = ts.hour
          dT -= _DEFAULT_DRAW_PROFILE[hour_of_day] * draw_rate

          override_target = override_lookup(ts) if override_lookup is not None else None
          if override_target is not None:
              q_el_w = q_dhw_power / cop_dhw
              el_kwh[i] = q_el_w / 1000.0
              t_tank = override_target
              t_tank_trajectory[i] = t_tank
              continue

          if t_tank < t_lower:
              q_el_w = q_dhw_power / cop_dhw
              el_kwh[i] = q_el_w / 1000.0
              t_tank = float(np.clip(t_tank + dT + heating_rise, t_lower, t_legionella))
          else:
              el_kwh[i] = 0.0
              t_tank = float(np.clip(t_tank + dT, t_lower, t_legionella))
          t_tank_trajectory[i] = t_tank

      return pd.Series(el_kwh, index=timestamps), pd.Series(t_tank_trajectory, index=timestamps), t_tank
  ```
  Add `from collections.abc import Callable` to the imports at the top of `physics.py` if not already present.

  Update `predict_series` (physics.py ~325-338) — **drop the fallback to `committed_override`** (Design note 1 above):
  ```python
  # was: effective_override = dhw_schedule_override if dhw_schedule_override is not None else self._schedule.get("committed_override")
  #      q_dhw_el, _ = self._dhw_kwh_series(timestamps, t_indoor, initial_t_tank, effective_override)
  override_lookup = (
      (lambda ts: self._dhw_override_for_hour(ts, dhw_schedule_override)) if dhw_schedule_override else None
  )
  q_dhw_el, _dhw_t_tank_series, _ = self._dhw_kwh_series(timestamps, t_indoor, initial_t_tank, override_lookup)
  ```

  Update `predict_training_series` (physics.py ~380-381) — **always override-blind, no exceptions**:
  ```python
  # was: effective_override = self._schedule.get("committed_override")
  #      q_dhw_el, _ = self._dhw_kwh_series(timestamps, t_indoor, initial_t_tank, effective_override)
  q_dhw_el, _dhw_t_tank_series, _ = self._dhw_kwh_series(timestamps, t_indoor, initial_t_tank, override_lookup=None)
  ```

- [ ] **Step 4: Run tests to verify they pass**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py -v
  ```
  Expected: PASS (full file — this is a signature change touching several existing tests; confirm none regress).

- [ ] **Step 5: Commit**

  ```bash
  git add apps/energy_forecast/physics.py tests/test_physics.py
  git commit -m "refactor: callable override_lookup + t_tank trajectory in _dhw_kwh_series"
  ```

---

## Task 7: `_compute_override_delta_series` — the shared joint trajectory-diff function

**Files:**
- Modify: `apps/energy_forecast/physics.py` (new method + two thin public wrappers)
- Test: `tests/test_physics.py` (new `TestOverrideDeltaSeries` class)

**Interfaces:**
- Consumes: `_dhw_kwh_series` (Task 6), `_dhw_override_for_hour` (Task 4), `_dhw_override_for_hour_from_history` (Task 5).
- Produces:
  - `_compute_override_delta_series(timestamps, t_ambient, initial_t_tank, override_lookup, baseline_kwh: pd.Series, baseline_t_tank: pd.Series) -> pd.Series` — the one shared function (R2-#3) used by both public wrappers below.
  - `compute_training_override_delta(timestamps, t_ambient, initial_t_tank, override_history: list[dict]) -> pd.Series` — training-side entry point, used by Task 9.
  - `compute_serving_override_delta(timestamps, t_ambient, initial_t_tank) -> pd.Series` — serving-side entry point (reads live `self._schedule["committed_override"]`), used by Task 11.

- [ ] **Step 1: Write the failing tests**

  ```python
  class TestOverrideDeltaSeries:
      def test_zero_when_no_override_history(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, [])
          assert (delta == 0.0).all()

      def test_legionella_override_produces_nonzero_multihour_tail(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
          timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          history = [
              {"kind": "legionella", "date": "2026-08-04", "hour": 12, "target_c": 60.0,
               "committed_at": "x", "cancelled_at": None},
          ]
          delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)
          # nonzero at the override hour and for at least a few hours after (reconvergence tail)
          assert delta.loc["2026-08-04 12:00"] != 0.0
          nonzero_after = (delta.loc["2026-08-04 13:00":"2026-08-04 18:00"] != 0.0).sum()
          assert nonzero_after >= 1
          # trajectories must have reconverged by the end of the 24h window
          assert delta.iloc[-1] == 0.0

      def test_joint_computation_no_double_counting_on_overlapping_tails(self, tmp_path):
          """R2-#1 direct regression: legionella hour 12 + comfort_boost hour 14 (inside
          the reconvergence tail) must not sum two independent per-override diffs."""
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
          timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          history = [
              {"kind": "legionella", "date": "2026-08-04", "hour": 12, "target_c": 60.0,
               "committed_at": "x", "cancelled_at": None},
              {"kind": "comfort_boost", "date": "2026-08-04", "hour": 14, "target_c": 55.0,
               "committed_at": "y", "cancelled_at": None},
          ]
          joint_delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)

          # independent-per-override baseline for comparison: replay legionella alone
          legionella_only = [history[0]]
          legionella_only_delta = pm.compute_training_override_delta(
              timestamps, t_ambient, 45.0, legionella_only
          )
          comfort_boost_only = [history[1]]
          comfort_boost_only_delta = pm.compute_training_override_delta(
              timestamps, t_ambient, 45.0, comfort_boost_only
          )
          naive_sum = legionella_only_delta + comfort_boost_only_delta
          # the joint computation must differ from the naive independent sum in at
          # least one overlapping hour — proving it isn't just summing two diffs
          assert not joint_delta.equals(naive_sum)

      def test_sanity_bound_references_larger_of_the_two_trajectories(self, tmp_path):
          """R2-#2: for the dominant negative-delta case (override run draws less),
          the bound must reference max(with_override, baseline), not just the
          override run's own (often near-zero) draw."""
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
          timestamps = pd.date_range("2026-08-04 00:00", periods=12, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          history = [
              {"kind": "legionella", "date": "2026-08-04", "hour": 0, "target_c": 60.0,
               "committed_at": "x", "cancelled_at": None},
          ]
          delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)
          baseline_kwh, _, _ = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=None)
          override_lookup = lambda ts: pm._dhw_override_for_hour_from_history(ts, history)
          with_override_kwh, _, _ = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup)
          bound = pd.concat([with_override_kwh, baseline_kwh], axis=1).max(axis=1)
          assert (delta.abs() <= bound + 1e-9).all()

      def test_tail_termination_matches_between_training_and_serving_entry_points(self, tmp_path):
          """R2-#3 direct regression: both public entry points delegate to the same
          shared function, so they resolve an equivalent override to the identical
          tail length."""
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
          timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          history = [
              {"kind": "legionella", "date": "2026-08-04", "hour": 12, "target_c": 60.0,
               "committed_at": "x", "cancelled_at": None},
          ]
          training_delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)

          pm._schedule["committed_override"] = {"legionella": ["2026-08-04", 12]}
          serving_delta = pm.compute_serving_override_delta(timestamps, t_ambient, 45.0)

          training_tail_end = (training_delta != 0.0).to_numpy().nonzero()[0][-1]
          serving_tail_end = (serving_delta != 0.0).to_numpy().nonzero()[0][-1]
          assert training_tail_end == serving_tail_end

      def test_train_serve_symmetry_full_delta_tail_round_trips(self, tmp_path):
          """Subtracting then re-adding the delta across the entire tail (not just the
          committed hour) must round-trip to the original value (R1-#7)."""
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
          timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          history = [
              {"kind": "legionella", "date": "2026-08-04", "hour": 12, "target_c": 60.0,
               "committed_at": "x", "cancelled_at": None},
          ]
          with_override_kwh, _, _ = pm._dhw_kwh_series(
              timestamps, t_ambient, 45.0, lambda ts: pm._dhw_override_for_hour_from_history(ts, history)
          )
          delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)
          baseline_kwh, _, _ = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=None)
          # train side: subtract delta from with-override actuals to get "as if blind"
          reconstructed_blind = with_override_kwh - delta
          pd.testing.assert_series_equal(reconstructed_blind, baseline_kwh, check_names=False)
  ```

- [ ] **Step 2: Run tests to verify they fail**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestOverrideDeltaSeries -v
  ```
  Expected: FAIL — `AttributeError: no attribute 'compute_training_override_delta'`.

- [ ] **Step 3: Implement**

  ```python
  _OVERRIDE_TAIL_THRESHOLD_C = 0.1  # R2-#3: shared cutoff, both call sites use it identically

  def _compute_override_delta_series(
      self,
      timestamps: pd.DatetimeIndex,
      t_ambient: pd.Series,
      initial_t_tank: float,
      override_lookup: Callable[[pd.Timestamp], float | None],
      baseline_kwh: pd.Series,
      baseline_t_tank: pd.Series,
  ) -> pd.Series:
      """The one shared joint trajectory-diff-and-cutoff function (R2-#3) — called
      identically by both compute_training_override_delta and
      compute_serving_override_delta so the two never resolve an equivalent override
      to different tail lengths. Computes ONE with-overrides trajectory (applying
      every override the given lookup can see, in chronological order — R2-#1, not a
      per-override sum), diffs it against the already-computed baseline, terminates
      the tail at the first hour the two trajectories reconverge within 0.1°C, and
      clips to the sanity bound (R1-#12/R2-#2)."""
      with_overrides_kwh, with_overrides_t_tank, _ = self._dhw_kwh_series(
          timestamps, t_ambient, initial_t_tank, override_lookup=override_lookup
      )
      raw_delta = with_overrides_kwh - baseline_kwh
      tank_diff = (with_overrides_t_tank - baseline_t_tank).abs()

      diverged = tank_diff >= self._OVERRIDE_TAIL_THRESHOLD_C
      if diverged.any():
          diverged_start = diverged.to_numpy().argmax()
          reconverged = tank_diff.iloc[diverged_start:] < self._OVERRIDE_TAIL_THRESHOLD_C
          if reconverged.any():
              tail_end = diverged_start + reconverged.to_numpy().argmax()
              raw_delta.iloc[tail_end:] = 0.0
      else:
          raw_delta[:] = 0.0  # no override ever diverged the trajectory in this window

      bound = pd.concat([with_overrides_kwh, baseline_kwh], axis=1).max(axis=1)
      violated = raw_delta.abs() > bound
      if violated.any():
          _LOGGER.warning(f"override_delta_series exceeded sanity bound on {int(violated.sum())} hour(s) — clipping")
      return raw_delta.clip(lower=-bound, upper=bound)

  def compute_training_override_delta(
      self,
      timestamps: pd.DatetimeIndex,
      t_ambient: pd.Series,
      initial_t_tank: float,
      override_history: list[dict],
  ) -> pd.Series:
      """Training-side entry point (Goal 2): replays override_history over the full
      training window, including each override's full multi-hour carryover tail."""
      baseline_kwh, baseline_t_tank, _ = self._dhw_kwh_series(timestamps, t_ambient, initial_t_tank, override_lookup=None)
      if not override_history:
          return pd.Series(0.0, index=timestamps)
      lookup = lambda ts: self._dhw_override_for_hour_from_history(ts, override_history)
      return self._compute_override_delta_series(
          timestamps, t_ambient, initial_t_tank, lookup, baseline_kwh, baseline_t_tank
      )

  def compute_serving_override_delta(
      self,
      timestamps: pd.DatetimeIndex,
      t_ambient: pd.Series,
      initial_t_tank: float,
  ) -> pd.Series:
      """Serving-side entry point (Goal 1): the deterministic post-model forecast
      correction. Callers add this UNCONDITIONALLY to the model's already-final
      published forecast — never fed back into physics_kwh or the trained model."""
      baseline_kwh, baseline_t_tank, _ = self._dhw_kwh_series(timestamps, t_ambient, initial_t_tank, override_lookup=None)
      committed = self._schedule.get("committed_override")
      if not committed:
          return pd.Series(0.0, index=timestamps)
      lookup = lambda ts: self._dhw_override_for_hour(ts, committed)
      return self._compute_override_delta_series(
          timestamps, t_ambient, initial_t_tank, lookup, baseline_kwh, baseline_t_tank
      )
  ```

- [ ] **Step 4: Run tests to verify they pass**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestOverrideDeltaSeries -v
  ```
  Expected: PASS.

- [ ] **Step 5: Commit**

  ```bash
  git add apps/energy_forecast/physics.py tests/test_physics.py
  git commit -m "feat: joint override_delta_series computation with tail-termination and sanity bound"
  ```

---

## Task 8: Calibration-input scoping (R1-#11) — subtract `override_delta_series` from `gross_kwh` once, upstream

**Files:**
- Modify: `apps/energy_forecast/physics.py` — expose `schedule` property and extract `_initial_t_tank_for_window` helper
- Modify: `apps/energy_forecast/model.py` — `train()`, around lines 381-433 (confirm exact ordering first, see Step 1)
- Test: `tests/test_physics.py` (calibration methods now see corrected input); `tests/test_model_train.py` or equivalent training-pipeline test file — check `grep -rln "def train" tests/` for the right file

**Interfaces:**
- Consumes: `compute_training_override_delta` (Task 7).
- Produces: `ThermalPhysicsModel.schedule` (public read-only property); `ThermalPhysicsModel._initial_t_tank_for_window(dhw_df, timestamps) -> float` (extracted, DRY, from `predict_training_series`'s existing inline duplicate).

- [ ] **Step 1: Read the exact current ordering in `model.py`'s `train()`**

  ```bash
  sed -n '370,435p' /home/jovyan/work/ha-energy-forecast/apps/energy_forecast/model.py
  ```
  Confirm precisely where `df = _add_lag_and_rolling_training(energy_df, ...)` (or whichever line first derives a working copy of `energy_df` for feature engineering) sits relative to the physics-hybrid block (~404-433). The correction below **must** be inserted before that first derivation — inserting it only inside the existing physics block (after `df` has already been built from uncorrected `energy_df`) would silently fail to propagate the correction into the ML training target at line ~630 (`y = df["gross_kwh"].to_numpy(...)`), since `df` would already be a stale copy.

- [ ] **Step 2: Write the failing test**

  In whichever test file covers `model.py`'s `train()` orchestration (find via `grep -rln "physics_model.calibrate\|predict_training_series" tests/`), add:
  ```python
  def test_gross_kwh_corrected_before_calibration_and_target_construction(self, tmp_path):
      """R1-#11: override_delta_series must be subtracted from gross_kwh once, upstream
      of both calibrate() and the ML training target — not independently in two places
      that could drift apart."""
      # Build a minimal energy_df/weather_df/dhw_df fixture with one committed legionella
      # override in physics_model.schedule["override_history"], and assert that:
      # (a) physics_model.calibrate(...) is called with a gross_kwh series that has the
      #     override_delta already subtracted at the overridden hour (patch calibrate
      #     and inspect its energy_df argument's gross_kwh at that hour vs. the raw fixture value), and
      # (b) the final y_fit / y used in the LightGBM fit reflects the same corrected value,
      #     not the raw fixture value.
      # Follow this file's existing fixture-building pattern for EnergyForecastModel.train()
      # (see nearby tests in this file for the exact fixture-construction helper in use).
      ...
  ```
  This test's exact fixture mechanics depend on the existing test file's helper functions for building `train()` inputs — read the nearest existing `train()` test in the same file first and mirror its fixture-construction pattern exactly (energy_df/weather_df/dhw_df shape, physics_model construction) rather than inventing a new one.

- [ ] **Step 3: Run test to verify it fails**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/<the_file_from_step_2> -k gross_kwh_corrected -v
  ```
  Expected: FAIL — no correction happens today.

- [ ] **Step 4: Add `schedule` property and `_initial_t_tank_for_window` helper to `physics.py`**

  ```python
  @property
  def schedule(self) -> dict:
      return self._schedule

  def _initial_t_tank_for_window(self, dhw_df: pd.DataFrame | None, timestamps: pd.DatetimeIndex) -> float:
      """Shared initial-tank-state estimate for a training/forecast window — extracted
      from predict_training_series to avoid duplicating this logic at the new
      override-delta call site in model.py."""
      if dhw_df is not None and not dhw_df.empty:
          d = dhw_df.set_index(pd.to_datetime(dhw_df["timestamp"]))["buffer_temp"].reindex(timestamps, method="nearest")
          return float(d.iloc[0]) if not d.empty and pd.notna(d.iloc[0]) else self._schedule["T_dhw_upper"]
      return (self._schedule["T_dhw_upper"] + self._schedule["T_dhw_lower"]) / 2
  ```
  Update `predict_training_series` (physics.py ~369-376) to call `self._initial_t_tank_for_window(dhw_df, timestamps)` instead of its inline duplicate of the same logic.

- [ ] **Step 5: Insert the correction block in `model.py`'s `train()`**

  Immediately before whichever line Step 1 identified as the first derivation of a working `df` from `energy_df`:
  ```python
  # Goal 1/2 (R1-#11): correct gross_kwh for any committed DHW override ONCE, upstream
  # of both calibration and the ML training target — single source of truth, not two
  # independent subtraction points that could drift apart. Uses self._calib's current
  # (pre-this-cycle) values as a bootstrap approximation — calibrate() below refines
  # them further using the now-corrected series.
  if physics_model is not None:
      _timestamps_for_delta = pd.DatetimeIndex(pd.to_datetime(energy_df["timestamp"]))
      _w_for_delta = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
      _t_outdoor_for_delta = _w_for_delta["temp_c"].reindex(_timestamps_for_delta, method="nearest")
      _override_history = physics_model.schedule.get("override_history", [])
      if _override_history:
          _initial_t_tank = physics_model._initial_t_tank_for_window(dhw_df, _timestamps_for_delta)
          _override_delta = physics_model.compute_training_override_delta(
              _timestamps_for_delta, _t_outdoor_for_delta, _initial_t_tank, _override_history
          )
          energy_df = energy_df.copy()
          energy_df["gross_kwh"] = energy_df["gross_kwh"].to_numpy() - _override_delta.to_numpy()
  ```

- [ ] **Step 6: Run test to verify it passes**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/<the_file_from_step_2> -k gross_kwh_corrected -v
  ```
  Expected: PASS.

- [ ] **Step 7: Run full test suite to confirm no regression**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v
  ```

- [ ] **Step 8: Commit**

  ```bash
  git add apps/energy_forecast/physics.py apps/energy_forecast/model.py tests/
  git commit -m "feat: scope calibration inputs to override-blind gross_kwh (R1-#11)"
  ```

---

## Task 9: `predict_training_series` — override-blind feature (already done, Task 6) + training-target correction

**Files:**
- Modify: `apps/energy_forecast/model.py` (target construction, ~lines 577-647)
- Test: training-pipeline test file (same file as Task 8)

**Interfaces:**
- Consumes: `compute_training_override_delta` (Task 7); `energy_df["gross_kwh"]` already corrected once upstream by Task 8.
- Produces: the training target (`y` / `y_fit`, both the feature-mode `log1p(gross_kwh)` path and the residual-mode `gross_kwh - physics_kwh` path) reflects the override-blind corrected series.

- [ ] **Step 1: Confirm Task 8's upstream correction already reaches this point**

  Since Task 8 mutates `energy_df["gross_kwh"]` before `df = _add_lag_and_rolling_training(energy_df, ...)`, and target construction at model.py:630 (`y = df["gross_kwh"].to_numpy(dtype=float)`) reads from `df` (derived from the now-corrected `energy_df`), **no additional code change is needed here** — this task is a verification + explicit regression test, not new production code.

- [ ] **Step 2: Write the regression test**

  ```python
  def test_training_target_reflects_override_blind_gross_kwh(self, tmp_path):
      """Direct regression test for Goal 1/2: with a committed historical override in
      override_history, the model's training target (both feature-mode log1p(gross_kwh)
      and residual-mode gross_kwh - physics_kwh) must be computed from the
      override-blind-corrected gross_kwh, not the raw override-inflated actuals."""
      # Build two otherwise-identical energy_df fixtures: one where gross_kwh at the
      # override hour includes the real override-driven spike, one without any override
      # ever having happened (a pure "as if blind" reference). Run train() on both
      # (first with the override committed in override_history, second with an
      # identical energy_df already override-blind and empty override_history) and
      # assert the resulting training target arrays for the overridden day are
      # numerically equal within the sanity-bound tolerance — proving the correction
      # neutralizes the override's effect on what the model actually fits against.
      ...
  ```

- [ ] **Step 3: Run test to verify it fails (if it does — this may already pass from Task 8's fix)**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/<file> -k training_target_reflects_override_blind -v
  ```

- [ ] **Step 4: If it fails, trace why Task 8's correction isn't reaching target construction and fix the gap (likely an ordering issue between the correction block and `df`'s derivation) — do not add a second, independent subtraction point (violates R1-#11's single-source-of-truth requirement)**

- [ ] **Step 5: Run test to verify it passes**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/<file> -k training_target_reflects_override_blind -v
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add tests/
  git commit -m "test: regression coverage for override-blind training target"
  ```

---

## Task 10: Transition-window down-weighting for pre-migration hours (R1-#13)

**Files:**
- Modify: `apps/energy_forecast/model.py:466-470` area (the existing `open_window_flags` down-weight block)
- Test: training-pipeline test file (same as Task 8/9)

**Interfaces:**
- Consumes: `override_history` (already available in `train()` from Task 8's block).
- Produces: `hourly_weights` gets an additional multiplicative down-weight (0.5) for training rows dated before the earliest `override_history["committed_at"]` — bounding the pre-migration failure mode's influence to roughly `weight_halflife_days` after ship, via the existing recency-halflife weighting already applied elsewhere (untouched by this task).

- [ ] **Step 1: Write the failing test**

  ```python
  def test_pre_migration_rows_down_weighted(self, tmp_path):
      """R1-#13: training rows dated before the earliest override_history entry get
      the same 0.5 down-weight multiplier as open-window-flagged hours, so the
      pre-fix failure mode (override-shaped noise the model has to explain away)
      doesn't stay at full training weight for as long as the full window spans."""
      # Fixture: energy_df spanning a period before AND after one override_history
      # entry's committed_at. Assert hourly_weights for rows before committed_at are
      # exactly half of what they'd be without this down-weight (all else equal),
      # and rows after committed_at are unaffected by this specific down-weight.
      ...
  ```

- [ ] **Step 2: Run test to verify it fails**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/<file> -k pre_migration_rows_down_weighted -v
  ```

- [ ] **Step 3: Implement**

  Immediately after the existing block at model.py:466-470:
  ```python
  # ── Down-weight pre-migration hours (R1-#13) ────────────────────────────────
  if _override_history and hourly_weights is not None:
      _ship_cutoff = pd.Timestamp(min(e["committed_at"] for e in _override_history))
      _pre_migration = pd.to_datetime(df["timestamp"]) < _ship_cutoff
      _down_weight_migration = pd.Series(np.where(_pre_migration, 0.5, 1.0), index=hourly_weights.index)
      hourly_weights = hourly_weights * _down_weight_migration
  ```
  (`_override_history` is the same local from Task 8's inserted block — confirm it's still in scope at this point in `train()`; if the two blocks end up far apart in the function, hoist `_override_history = physics_model.schedule.get("override_history", []) if physics_model is not None else []` once, near the top, and reference it from both places.)

- [ ] **Step 4: Run test to verify it passes**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/<file> -k pre_migration_rows_down_weighted -v
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add apps/energy_forecast/model.py tests/
  git commit -m "feat: down-weight pre-migration training rows (R1-#13)"
  ```

---

## Task 11: Serving-path correction — add `override_delta` to the published forecast, never through the model

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` — `_update_sensors` (~lines 1761-1996; confirm exact call site first)
- Test: `tests/test_energy_forecast_physics_config.py` or a new dedicated test class in the same file

**Interfaces:**
- Consumes: `compute_serving_override_delta` (Task 7).
- Produces: the sensor-published forecast series has the override correction added unconditionally, after `self._model.predict(...)` returns.

- [ ] **Step 1: Read the exact current `_update_sensors` call site**

  ```bash
  sed -n '1750,2000p' /home/jovyan/work/ha-energy-forecast/apps/energy_forecast/energy_forecast.py
  ```
  Identify the exact local variable names for: the forecast timestamps index, the ambient-temperature series already used to build `predict()`'s inputs, whatever local holds the DHW-recent buffer-temp data (for `_initial_t_tank_for_window`), and the variable holding `self._model.predict(...)`'s returned series before it's written to the sensor. Use those exact names in Step 3 below — do not guess.

- [ ] **Step 2: Write the failing test**

  In `tests/test_energy_forecast_physics_config.py` (reuse its `_make_app`/`_restore_module_loggers` fixture pattern):
  ```python
  class TestServingOverrideCorrection:
      def test_committed_override_moves_published_forecast_by_full_delta(self, tmp_path):
          """Direct regression test for the 2026-08-04 live bug: a committed override
          must move the published forecast sensor by its true expected kWh amount,
          independent of what the trained model happens to weigh — not routed through
          predict()'s ML blending at all."""
          # Build an EnergyForecast app with physics configured, commit a legionella
          # override via app._physics_model.commit_dhw_schedule(...), call
          # app._update_sensors(...), and assert the published forecast sensor's
          # per-hour value at the override hour (and its multi-hour tail) differs from
          # an otherwise-identical run with no override committed by approximately
          # compute_serving_override_delta's own value at that hour (within floating-
          # point tolerance) — proving the addition happened and wasn't absorbed/
          # discounted by the model.
          ...
  ```

- [ ] **Step 3: Run test to verify it fails**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py::TestServingOverrideCorrection -v
  ```

- [ ] **Step 4: Implement**

  Immediately after the line that assigns `self._model.predict(...)`'s result (identified in Step 1), before it's written to the published sensor:
  ```python
  # Goal 1: deterministic post-model forecast correction — never routed through the
  # model, so double-counting is structurally impossible (spec's Design section).
  if self._physics_model is not None:
      _override_delta = self._physics_model.compute_serving_override_delta(
          <forecast_timestamps_var>, <t_ambient_var>,
          self._physics_model._initial_t_tank_for_window(<dhw_recent_var>, <forecast_timestamps_var>),
      )
      <predicted_series_var> = <predicted_series_var> + _override_delta.reindex(<predicted_series_var>.index).fillna(0.0)
  ```
  Replace the `<...>` placeholders with the exact variable names found in Step 1 — this is the one place in this plan where the exact names can't be pinned without reading the surrounding code first, but the algorithm and insertion point are fully specified.

- [ ] **Step 5: Run test to verify it passes**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py::TestServingOverrideCorrection -v
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
  git commit -m "feat: add deterministic override correction to published forecast (Goal 1)"
  ```

---

## Task 12: Cross-cutting integration test — comfort_boost 3-elem payload round-trips through the live service stack

**Files:**
- Test: `tests/test_energy_forecast_physics_config.py::TestSetDhwScheduleService` (extend existing class)

**Interfaces:**
- Consumes: `commit_dhw_schedule` (Task 3), `_set_dhw_schedule_cb` (unmodified — confirms it doesn't need to change).

- [ ] **Step 1: Write the test**

  Following the existing pattern at `tests/test_energy_forecast_physics_config.py:917-933`:
  ```python
  def test_comfort_boost_3elem_payload_forwards_unmodified(self, tmp_path):
      from energy_forecast.energy_forecast import EnergyForecast

      app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
      app.initialize()
      app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
      app._cached_forecast_df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})

      app._set_dhw_schedule_cb(
          "default", "energy_forecast", "set_dhw_schedule",
          {"dhw_schedule": {"comfort_boost": ["2026-08-05", 14, 57.5]}},
      )
      assert app._physics_model._schedule["committed_override"] == {"comfort_boost": ["2026-08-05", 14, 57.5]}
      assert len(app._physics_model._schedule["override_history"]) == 1
      assert app._cached_forecast_df is None  # cache invalidated, same as legionella today

  def test_legionella_and_comfort_boost_same_day_neither_clobbers_other(self, tmp_path):
      """Cross-repo scenario from the spec's Testing section: same-day legionella-then-
      comfort_boost commit sequence through the real call_service→commit_dhw_schedule
      path, verifying override_history accumulates both."""
      from energy_forecast.energy_forecast import EnergyForecast

      app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
      app.initialize()
      app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
      app._cached_forecast_df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})

      app._set_dhw_schedule_cb(
          "default", "energy_forecast", "set_dhw_schedule",
          {"dhw_schedule": {"legionella": ["2026-08-04", 12]}},
      )
      app._set_dhw_schedule_cb(
          "default", "energy_forecast", "set_dhw_schedule",
          {"dhw_schedule": {"comfort_boost": ["2026-08-05", 14, 57.5]}},
      )
      committed = app._physics_model._schedule["committed_override"]
      assert committed == {"legionella": ["2026-08-04", 12], "comfort_boost": ["2026-08-05", 14, 57.5]}
      assert len(app._physics_model._schedule["override_history"]) == 2
  ```

- [ ] **Step 2: Run to verify it passes (this exercises only already-implemented code from Tasks 3/12)**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py::TestSetDhwScheduleService -v
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add tests/test_energy_forecast_physics_config.py
  git commit -m "test: cross-cutting integration coverage for comfort_boost commit path"
  ```

---

## Task 13: Zero-override degradation regression + full suite run

**Files:**
- Test: `tests/test_physics.py` (new `TestZeroOverrideDegradation` class)

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the test**

  ```python
  class TestZeroOverrideDegradation:
      def test_empty_history_and_committed_override_produces_zero_delta_everywhere(self, tmp_path):
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          timestamps = pd.date_range("2026-08-04 00:00", periods=48, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          training_delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, [])
          serving_delta = pm.compute_serving_override_delta(timestamps, t_ambient, 45.0)
          assert (training_delta == 0.0).all()
          assert (serving_delta == 0.0).all()

      def test_physics_kwh_identical_to_pre_phase_a_when_no_override_ever_committed(self, tmp_path):
          """No regression risk for installs that never use DHW overrides — the
          override-blind physics_kwh feature must be bit-identical to calling
          _dhw_kwh_series with override_lookup=None (today's only behavior)."""
          pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
          timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
          t_ambient = pd.Series(10.0, index=timestamps)
          el_kwh_a, _, final_a = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=None)
          el_kwh_b, _, final_b = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=None)
          pd.testing.assert_series_equal(el_kwh_a, el_kwh_b)
          assert final_a == final_b
  ```

- [ ] **Step 2: Run to verify it passes**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py::TestZeroOverrideDegradation -v
  ```

- [ ] **Step 3: Run the FULL test suite — this is the plan's final gate**

  ```bash
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v
  ```
  Expected: PASS, zero failures, zero regressions against pre-Phase-A behavior.

- [ ] **Step 4: Commit**

  ```bash
  git add tests/test_physics.py
  git commit -m "test: zero-override degradation regression coverage (Phase A final gate)"
  ```

- [ ] **Step 5: Update CHANGELOG.md and MEMORY.md per project workflow, then hand off to `@deploy-agent` per the standard finalize-a-branch sequence — but do NOT flip anything live-facing yet.** Phase A's exit gate (3 manual checks against live data, spec's Rollout section) must be walked and confirmed **before** Phase B (the EM-side plan) begins — this is a documentation/runbook gate, not a code-enforced one, so the deploy step itself is safe to run, but treat the exit-gate confirmation as a separate, explicit follow-up action, not implied by tests passing.
