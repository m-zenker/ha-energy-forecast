# #122 Trim the DHW Override Serving Correction to the Override Hour — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

OWNER-APPROVED docs/superpowers/plans/2026-11-23-122-override-hour-trim.md 2026-11-23

**Goal:** The published forecast keeps only the override-hour leg of the DHW override correction; the multi-hour tail (second +1 kWh ~23 h later, negative "natural reheat" legs) is dropped at serving time.

**Architecture:** `ThermalPhysicsModel.compute_serving_override_delta` computes the delta exactly as today, then zeroes every hour whose committed-override lookup returns `None`. `compute_training_override_delta` is untouched: training still removes each override's full tail from the target. No config key (v1.0 gate: config schema, services and sensor names stay unchanged); rollback = previous tag.

**Tech Stack:** Python 3.13, pandas, pytest; env `/home/jovyan/my_envs/ha-energy-forecast`.

**Spec:** ROADMAP `### #122`; evidence `memory/project_122_override_trim_measurement_2026_11_23.md` (gitignored; numbers repeated below).

**Base branch:** `dev` (386fa6b). No dependency on unshipped train items. Branch: `fix/override-hour-trim`, worktree `.worktrees/override-hour-trim`.

**Size:** S. **Risk tags:** `BEHAVIOUR` (published hourly forecast changes on updates with a pending committed override). Not `CONFIG-STEP`, not `UI`, not `REMOVES`. Forecast-changing → own deploy window (ROADMAP rule "one forecast-changing step per deploy window").

**Data-simulation evidence (policy §10.6):** live captures 2026-11-08..11-23, 47 updates with a pending override, 6 episodes; trim vs live: 3 h-block MAE −0.021 kWh [95 % CI −0.080, −0.003], hourly −0.024 [−0.034, −0.017], negative in every leave-one-episode-out; exact-0.0 published hours 45 → 0. Daily day-ahead MAE +0.11 kWh (6.61 → 6.73, n = 47, within noise). Script: local `scripts/measure_override_trim_122.py`.

## Global Constraints

- Training path (`compute_training_override_delta`, `model.py` call site) must produce bit-identical output to today.
- No new config key, service parameter, sensor or attribute.
- Same-hour precedence stays: legionella wins over comfort_boost (`_dhw_override_for_hour`).
- The app's existing floor (`np.maximum(0.0, …)` in `energy_forecast.py` ~L2273) stays.

## Review Focus

1. Both a legionella and a comfort_boost pending in the same 48 h window → both hours keep their own leg (Task 1 test `test_serving_trim_keeps_both_pending_override_hours`).
2. Committed override already in the past (before the first serving timestamp) → all-zero delta, as today (Task 1 test `test_serving_trim_past_override_is_zero`).
3. Delta value at the override hour is exactly the untrimmed value (incl. sanity-bound clip), not recomputed (Task 1 test `test_serving_trim_override_hour_value_unchanged`).
4. Training delta still carries the full multi-hour tail (Task 1 test `test_training_delta_keeps_full_tail`, replaces the tail-equality test).
5. No negative serving delta remains anywhere → the correction can no longer create exact-0.0 published hours (Task 1 test `test_serving_trim_has_no_negative_legs`).

---

### Task 1: Trim the serving delta to the committed override hours

**Files:**
- Modify: `apps/energy_forecast/physics.py` — `compute_serving_override_delta` (~L500–521) and its docstring; `_compute_override_delta_series` docstring line "called identically by both … tail lengths" (~L456–459) to say serving trims afterwards.
- Test: `tests/test_physics.py` — replace `test_tail_termination_matches_between_training_and_serving_entry_points` (~L412–437); add the tests below in the same class.

**Interfaces:**
- Produces: `ThermalPhysicsModel.compute_serving_override_delta(timestamps: pd.DatetimeIndex, t_ambient: pd.Series, initial_t_tank: float) -> pd.Series` — same signature and index as today; non-zero only at hours where `self._dhw_override_for_hour(ts, committed) is not None`.

Fixture for all tests (as in the existing tail test): `pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)`, `pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)`, `timestamps = pd.date_range("2026-08-04 00:00", periods=48, freq="h")`, `t_ambient = pd.Series(10.0, index=timestamps)`, initial tank 45.0.

- [ ] **Step 1: Write the failing tests**

```python
def test_training_delta_keeps_full_tail(self, tmp_path):
    # history = one legionella 2026-08-04 12:00 (as in the replaced test)
    training = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)
    assert (training != 0.0).sum() > 1          # tail beyond the override hour survives in training

def test_serving_trim_keeps_only_override_hour(self, tmp_path):
    pm._schedule["committed_override"] = {"legionella": ["2026-08-04", 12]}
    serving = pm.compute_serving_override_delta(timestamps, t_ambient, 45.0)
    nz = serving[serving != 0.0]
    assert list(nz.index) == [pd.Timestamp("2026-08-04 12:00")]
    assert nz.iloc[0] > 0

def test_serving_trim_override_hour_value_unchanged(self, tmp_path):
    pm._schedule["committed_override"] = {"legionella": ["2026-08-04", 12]}
    untrimmed = pm._compute_override_delta_series(...)   # same inputs, lookup=_dhw_override_for_hour, baseline from _dhw_kwh_series(override_lookup=None)
    serving = pm.compute_serving_override_delta(timestamps, t_ambient, 45.0)
    assert serving[pd.Timestamp("2026-08-04 12:00")] == untrimmed[pd.Timestamp("2026-08-04 12:00")]

def test_serving_trim_keeps_both_pending_override_hours(self, tmp_path):
    pm._schedule["committed_override"] = {"legionella": ["2026-08-04", 12],
                                          "comfort_boost": ["2026-08-05", 6, 55.0]}
    serving = pm.compute_serving_override_delta(timestamps, t_ambient, 45.0)
    assert set(serving[serving != 0.0].index) <= {pd.Timestamp("2026-08-04 12:00"), pd.Timestamp("2026-08-05 06:00")}
    assert serving[pd.Timestamp("2026-08-04 12:00")] > 0

def test_serving_trim_past_override_is_zero(self, tmp_path):
    pm._schedule["committed_override"] = {"legionella": ["2026-08-03", 12]}
    assert (pm.compute_serving_override_delta(timestamps, t_ambient, 45.0) == 0.0).all()

def test_serving_trim_has_no_negative_legs(self, tmp_path):
    pm._schedule["committed_override"] = {"legionella": ["2026-08-04", 12]}
    assert (pm.compute_serving_override_delta(timestamps, t_ambient, 45.0) >= 0.0).all()
```

In `test_serving_trim_keeps_both_pending_override_hours`, the comfort_boost hour may legitimately be 0 if the tank is already at/above 55 °C after the legionella cycle; the assertion is "no other hour", plus the legionella leg > 0.

- [ ] **Step 2: Run them to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics.py -k "serving_trim or keeps_full_tail" -v`
Expected: `keeps_full_tail` and `past_override_is_zero` PASS (current behaviour); the four `serving_trim_*` others FAIL (tail hours non-zero / negative legs).

- [ ] **Step 3: Implement the trim in `compute_serving_override_delta`**

After the existing `_compute_override_delta_series(...)` call, build `on = timestamps.map(lambda ts: lookup(ts) is not None)` and return `delta.where(on, 0.0)`. Keep the early `return pd.Series(0.0, index=timestamps)` for no committed override. Docstring: state that serving keeps only the override-hour leg (#122, measured 2026-11-23) and training keeps the full tail.

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -q`
Expected: all pass. `tests/test_energy_forecast_physics_config.py` serving tests build their expected value from `compute_serving_override_delta` itself and must stay green unchanged; if one asserts a non-zero hour other than the override hour, that assertion encoded the old tail — update it and say so in the commit body.

- [ ] **Step 5: Commit**

```bash
git -C <worktree> add apps/energy_forecast/physics.py tests/test_physics.py
git -C <worktree> commit -m "fix(physics): trim DHW override serving correction to the override hour (#122)"
```

---

### Task 2: Verify on live captures and document

**Files:**
- Modify: `CHANGELOG.md` (`[Unreleased]` → `### Changed`), `README.md` Stage 5 "Persisted DHW schedule commits" bullet (~L160–168), `ROADMAP.md` `### #122` status line and the Priority Queue row 122.

- [ ] **Step 1: Capture cross-check (branch code reproduces the measured trim arm)**

With the live env, for every predict capture in `/home/jovyan/work/ha-energy-forecast/data/captures/predict` that has a pending committed override: load the snapshot, take its `physics_model`, and recompute `compute_serving_override_delta` with the branch code on the serving timestamps and the same `t_ambient`/initial tank the app used — or, simpler and sufficient, assert that the branch's trimmed delta is non-zero only at the committed hour(s) and that `published − model` at those hours equals the branch delta there (± 1e-9). Run: `PYTHONPATH=apps:. /home/jovyan/my_envs/ha-energy-forecast-live/bin/python <snippet>` from the worktree. Expected: 47/47 captures match at the override hour; report the count. If the app-side inputs can't be reconstructed exactly, fall back to the second assertion only and say so.

- [ ] **Step 2: Docs**

CHANGELOG `### Changed`: "DHW override correction: the published forecast now adds only the override cycle's energy at the committed hour; the knock-on tail later in the day is no longer applied (it removed exact-0.0 forecast hours and cut 3 h-block error on override days, #122). Training is unchanged."
README: one sentence after the `set_dhw_schedule` paragraph saying the same in user terms.
ROADMAP #122: status "implemented on `fix/override-hour-trim` 2026-11-23, measured: 3 h-block −0.021 kWh, hourly −0.024, exact-0.0 hours 45 → 0"; Priority Queue row 122 status cell to match.

- [ ] **Step 3: Commit**

```bash
git -C <worktree> add CHANGELOG.md README.md ROADMAP.md
git -C <worktree> commit -m "docs: #122 override-hour trim"
```

- [ ] **Step 4: Hand-off**

Full suite green in the worktree; report the commit hashes and the capture cross-check count. Stop at ready-to-merge; merging into `train/next` is an owner click (risk tag `BEHAVIOUR`).

---

## Out of scope (decided 2026-11-23)

- Dropping the serving correction entirely, or gating it on an EM execution signal: better point estimate (3 h-block −0.052) but CI spans 0 on 6 episodes, and most wrong override-hour legs were phantom commits (denied opportunistic legionella 11-15, expired comfort boosts) that EM could not cancel before the `clear` fix shipped 2026-11-19 (hef alpha-44 / EM alpha-164). Re-measure the override-hour leg with ≥ 2 weeks of post-11-19 captures before deciding.
- Any change to `compute_training_override_delta` or #124's tank-verified replay.
- Legionella cycle *energy*: the override-hour leg models 60 °C as heat-pump work (~1 kWh, one hour), but the HP stops at ~54 °C tank and the unmetered electric heater does the rest — live cycles cost ~4–5 kWh over ~2 h (11-09, 11-18). Separate follow-up item; this plan only removes the tail.
