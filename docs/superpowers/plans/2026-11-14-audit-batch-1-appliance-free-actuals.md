# Appliance-Free Stored Actuals (ROADMAP #117) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the actuals stored for MAE, bias and anomaly scoring use the same definition as the training target, so in `baseline_mode` they no longer include dishwasher/washer/tumbler energy.

**Architecture:** two pure module-level helpers in `energy_forecast.py` replace the inline loop in `_update_sensors`: one builds the scoring actuals for the recent window, one migrates the 30 days already stored. A marker in `pred_history.json` makes the migration run exactly once.

**Tech Stack:** Python 3.13, pandas, pytest. AppDaemon app (`hassapi` is stubbed by `tests/conftest.py`).

**Spec:** `ROADMAP.md` #117 (on `train/next`); `docs/analysis/2026-11-13-forecast-improvement-audit.md`; index `docs/superpowers/plans/2026-11-14-audit-batch-index.md` (in the main checkout).

## Global Constraints

- Base branch `train/next`; branch `fix/appliance-free-actuals`; worktree `/home/jovyan/work/ha-energy-forecast/.worktrees/appliance-free-actuals`.
- Tests only via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest` (below: `$PY -m pytest`).
- The published forecast must not change. Only `_actuals_history` and what is computed from it (`mae_7d`, `mae_30d`, relative MAE, anomaly sensor) may change.
- With `baseline_mode: false`, or with no `sub_energy_sensors`, stored values must be bit-identical to today.
- Sensors listed in `baseline_included_sensors` stay in the actuals (they stay in the training target). Use the existing `self._subtract_only_dict()`.
- EV handling is unchanged: threshold-inferred EV hours ±1 h are not stored; wallbox-sensor hours are stored net of the wallbox kWh.
- `pred_history.json` stays loadable by older code (only a new top-level key is added).

## Review Focus

1. A sub-meter fetch fails in the hourly cycle (its prefix is missing from `sub_sensors_recent`): the migration must not run on partial data; it retries next cycle. → Task 2 test.
2. `pred_history.json` written by the old version (no `actuals_basis` key) must be migrated once; a file written by the new version must never be migrated again (double subtraction). → Task 2 and Task 3 tests.
3. Fresh install without `pred_history.json`: nothing to migrate, marker is written as `target` on first save. → Task 3 test.
4. An hour where sub-meter kWh exceeds the corrected consumption (meter timing skew): stored value is 0.0, never negative. → Task 1 test.
5. Appliance run inside a wallbox EV hour: both are subtracted, floor at 0. → Task 1 test.

---

## Session setup

```bash
REPO=/home/jovyan/work/ha-energy-forecast
git -C $REPO worktree add $REPO/.worktrees/appliance-free-actuals -b fix/appliance-free-actuals train/next
cd $REPO/.worktrees/appliance-free-actuals
PY=/home/jovyan/my_envs/ha-energy-forecast/bin/python
$PY -m pytest tests/ -q          # baseline: 0 failed
```

## File Structure

- Modify `apps/energy_forecast/energy_forecast.py`: two new module-level functions next to `_subtract_sub_sensors` (≈ line 3629); `_update_sensors` block "Populate rolling actuals history" (≈ 2306–2330); `initialize` (≈ 510); `_load_pred_history` / `_save_pred_history` (≈ 3336–3394).
- Modify `tests/test_energy_forecast.py`: new class `TestScoringActuals`; replace one obsolete test.

Background you need: `_subtract_sub_sensors(df, sub_sensors_dict, column="gross_kwh") -> (df, removed_kwh)` already exists and clips at 0. `ha_data.fetch_recent_sub_sensor()` returns the **full** cached sub-meter history (not just 2 days), so `sub_sensors_recent` covers all 30 stored days. `full_actuals` covers only the last ~2 days.

---

### Task 1: `_scoring_actuals` helper

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (add after `_subtract_sub_sensors`)
- Test: `tests/test_energy_forecast.py`

**Interfaces:**
- Produces: `_scoring_actuals(full_actuals: pd.DataFrame, sub_sensors_dict: dict | None, ev_excluded: set, ev_sensor_kwh: dict, completed_cutoff: pd.Timestamp) -> dict[pd.Timestamp, float]`

- [ ] **Step 1: Write the failing tests** (new class at the end of `tests/test_energy_forecast.py`)

```python
class TestScoringActuals:
    """ROADMAP #117: stored actuals use the training-target definition."""

    T0 = pd.Timestamp("2026-11-10 08:00")

    def _full(self, kwh):
        return pd.DataFrame({"timestamp": pd.date_range(self.T0, periods=len(kwh), freq="1h"), "gross_kwh": kwh})

    def _sub(self, kwh):
        return pd.DataFrame({"timestamp": pd.date_range(self.T0, periods=len(kwh), freq="1h"), "kwh": kwh})

    def test_appliance_run_is_subtracted(self):
        from energy_forecast.energy_forecast import _scoring_actuals

        out = _scoring_actuals(
            self._full([1.0, 2.5, 1.0]), {"sub_dw": self._sub([0.0, 1.2, 0.0])}, set(), {}, self.T0 + pd.Timedelta(hours=3)
        )
        assert out == pytest.approx({self.T0: 1.0, self.T0 + pd.Timedelta(hours=1): 1.3, self.T0 + pd.Timedelta(hours=2): 1.0})

    def test_no_sub_sensors_keeps_values(self):
        from energy_forecast.energy_forecast import _scoring_actuals

        out = _scoring_actuals(self._full([1.0, 2.5]), None, set(), {}, self.T0 + pd.Timedelta(hours=2))
        assert list(out.values()) == [1.0, 2.5]

    def test_floor_at_zero_and_ev_sensor_also_subtracted(self):
        from energy_forecast.energy_forecast import _scoring_actuals

        t1 = self.T0 + pd.Timedelta(hours=1)
        out = _scoring_actuals(
            self._full([0.5, 9.0]), {"sub_dw": self._sub([0.8, 1.0])}, set(), {t1: 7.0}, self.T0 + pd.Timedelta(hours=2)
        )
        assert out[self.T0] == 0.0
        assert out[t1] == pytest.approx(1.0)

    def test_ev_excluded_and_incomplete_hours_are_not_stored(self):
        from energy_forecast.energy_forecast import _scoring_actuals

        t1 = self.T0 + pd.Timedelta(hours=1)
        out = _scoring_actuals(self._full([1.0, 6.0, 1.0]), None, {t1}, {}, self.T0 + pd.Timedelta(hours=2))
        assert list(out) == [self.T0]  # t1 is EV-excluded, t2 is the open hour
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_energy_forecast.py::TestScoringActuals -q`
Expected: 4 errors, `ImportError: cannot import name '_scoring_actuals'`

- [ ] **Step 3: Implement `_scoring_actuals`**

Call `_subtract_sub_sensors(full_actuals, sub_sensors_dict)` first, then loop exactly like the current inline code (floor timestamps to 1 h; skip `ts >= completed_cutoff`; skip `ts in ev_excluded`; store `max(0.0, kwh - ev_sensor_kwh.get(ts, 0.0))`). Return a new dict; do not mutate inputs.

- [ ] **Step 4: Run to verify they pass**

Run: `$PY -m pytest tests/test_energy_forecast.py::TestScoringActuals -q` → 4 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast.py
git commit -m "feat(scoring): helper for appliance-free scoring actuals (#117)"
```

---

### Task 2: One-time migration helper

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (add after `_scoring_actuals`)
- Test: `tests/test_energy_forecast.py` (class `TestScoringActuals`)

**Interfaces:**
- Produces: `_migrate_actuals_to_target(actuals_history: dict, sub_sensors_dict: dict | None, expected_prefixes: set[str]) -> tuple[dict, bool]`. Returns `(new_history, done)`. `done` is False (and the history is returned unchanged) when any prefix in `expected_prefixes` is missing from `sub_sensors_dict` or has an empty frame.

- [ ] **Step 1: Write the failing tests** (add to `TestScoringActuals`)

```python
    def test_migration_subtracts_cached_sub_history(self):
        from energy_forecast.energy_forecast import _migrate_actuals_to_target

        hist = {self.T0: 2.0, self.T0 + pd.Timedelta(hours=1): 0.3}
        new, done = _migrate_actuals_to_target(hist, {"sub_dw": self._sub([1.5, 0.5])}, {"sub_dw"})
        assert done is True
        assert new == pytest.approx({self.T0: 0.5, self.T0 + pd.Timedelta(hours=1): 0.0})
        assert hist[self.T0] == 2.0  # input not mutated

    def test_migration_waits_for_missing_or_empty_sub_sensor(self):
        from energy_forecast.energy_forecast import _migrate_actuals_to_target

        hist = {self.T0: 2.0}
        for subs in ({}, None, {"sub_dw": self._sub([])}, {"sub_other": self._sub([1.0])}):
            new, done = _migrate_actuals_to_target(hist, subs, {"sub_dw"})
            assert done is False and new == hist

    def test_migration_with_nothing_to_subtract_is_done(self):
        from energy_forecast.energy_forecast import _migrate_actuals_to_target

        new, done = _migrate_actuals_to_target({self.T0: 2.0}, None, set())
        assert done is True and new == {self.T0: 2.0}
```

- [ ] **Step 2: Run to verify they fail** — `ImportError: cannot import name '_migrate_actuals_to_target'`

- [ ] **Step 3: Implement `_migrate_actuals_to_target`**

Empty `expected_prefixes` → `(dict(actuals_history), True)`. Otherwise check completeness, build a `[timestamp, gross_kwh]` frame from the dict, run `_subtract_sub_sensors` with only the expected prefixes, and return the result as a dict keyed by the original timestamps.

- [ ] **Step 4: Run** — `$PY -m pytest tests/test_energy_forecast.py::TestScoringActuals -q` → 7 passed

- [ ] **Step 5: Commit** — `git commit -am "feat(scoring): one-time migration of stored actuals to the target basis (#117)"`

---

### Task 3: Wire into the app and persist the marker

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py`
- Test: `tests/test_energy_forecast.py`

**Interfaces:**
- Consumes: `_scoring_actuals`, `_migrate_actuals_to_target` (Tasks 1–2).
- Produces: attribute `self._actuals_on_target_basis: bool`; JSON key `"actuals_basis"` in `pred_history.json` with value `"target"` or `"gross"`. Phase 3 of the batch reads this key.

- [ ] **Step 1: Write the failing persistence tests**

Find the existing load/save tests (search `_load_pred_history` in `tests/test_energy_forecast.py`, ≈ lines 2760–2970) and reuse their fixture pattern (fake app + monkeypatched `PRED_HISTORY_PATH`) for three new tests in the same class:

```python
    def test_save_writes_actuals_basis_marker(self, ...):
        # app._actuals_on_target_basis = True; app._save_pred_history()
        assert json.loads(path.read_text())["actuals_basis"] == "target"
        # app._actuals_on_target_basis = False; save again
        assert json.loads(path.read_text())["actuals_basis"] == "gross"

    def test_load_old_file_without_marker_needs_migration(self, ...):
        # file content: {"pred": {}, "actuals": {"<recent ISO ts>": 1.0}}   (no actuals_basis key)
        # app._actuals_on_target_basis = True before the load
        assert app._actuals_on_target_basis is False

    def test_load_marked_file_and_missing_file_need_no_migration(self, ...):
        # file with "actuals_basis": "target"  -> True after load
        # no file at all                       -> stays True
```

- [ ] **Step 2: Run to verify they fail** (KeyError `actuals_basis` / AttributeError)

- [ ] **Step 3: Implement persistence**

- `initialize`: next to `self._actuals_history: dict[Any, float] = {}` (≈ line 510) add `self._actuals_on_target_basis: bool = True`.
- `_load_pred_history`: after the JSON is read successfully, set `self._actuals_on_target_basis = data.get("actuals_basis") == "target"`. The early return for a missing file leaves it True.
- `_save_pred_history`: add `"actuals_basis": "target" if self._actuals_on_target_basis else "gross"` to `data`.
- Any test fake that calls these methods without going through `initialize` needs the attribute; add it where the fakes set `_actuals_history`.

- [ ] **Step 4: Replace the inline loop in `_update_sensors`**

Anchor: the comment `# ── Populate rolling actuals history for mae_7d / mae_30d sensors (#41) ─`. Replace the block from that comment down to (not including) `actuals_cutoff = now_ts.floor("1h") - pd.Timedelta(days=30)` with, in this order:

1. `_score_subs = self._subtract_only_dict(sub_sensors_recent) if (self._baseline_mode and sub_sensors_recent) else None`
2. Migration, only `if not self._actuals_on_target_basis:` — expected prefixes are `set()` when `baseline_mode` is off, else `{self._sub_sensor_prefix(e) for e in self._sub_energy_sensors} - {self._sub_sensor_prefix(e) for e in self._baseline_included_sensors}`. Call `_migrate_actuals_to_target(self._actuals_history, _score_subs, expected)`; when `done`, assign the history, set the flag True and log once at INFO: `"Stored actuals migrated to the training-target basis (%d hours, %.2f kWh appliance energy removed)"`.
3. The existing retroactive eviction (`for _ts in ev_mae_excluded: self._actuals_history.pop(_ts, None)`), unchanged, guarded by the same `full_actuals is not None and not full_actuals.empty` condition.
4. `self._actuals_history.update(_scoring_actuals(full_actuals, _score_subs, ev_mae_excluded, ev_sensor_kwh, now_ts.floor("1h")))`

Rewrite the comment above it: stored kWh = target-corrected consumption minus the `baseline_mode` sub-meters (same set as the training target) minus wallbox EV kWh.

The migration must run before step 4, so recent hours are then overwritten with freshly computed values and never subtracted twice.

- [ ] **Step 5: Replace the obsolete test**

Delete `test_actuals_history_stores_raw_kwh_not_sub_sensor_subtracted` (it pins the bug and only simulates the loop inline). `TestScoringActuals` covers the new behaviour.

- [ ] **Step 6: Run the full suite** — `$PY -m pytest tests/ -q` → 0 failed. Then `ruff check . && ruff format --check .`

- [ ] **Step 7: Commit** — `git commit -am "fix(scoring): store appliance-free actuals for MAE, bias and anomaly (#117)"`

---

### Task 4: Docs and merge to the train

- [ ] **Step 1:** CHANGELOG `[Unreleased]` via the `/changelog` skill. Must say: in `baseline_mode`, `mae_7d`/`mae_30d`, relative MAE and the anomaly sensor now score against appliance-free actuals; readings step once at the first update after deploy; stored history is migrated once; tag BEHAVIOUR.
- [ ] **Step 2:** ROADMAP: #117 section status → "implemented on `train/next` (date), not deployed"; Pending Summary row 117 likewise.
- [ ] **Step 3:** README: if the MAE sensor description says what the actuals contain, correct it (search `mae_7d`).
- [ ] **Step 4:** `/home/jovyan/work/ha-energy-forecast/MEMORY.md`: one line under "Next" item 2d that #117 is on the train, with the post-deploy check below.
- [ ] **Step 5:** Merge and push per the index's rule 7:
  ```bash
  git -C /home/jovyan/work/ha-energy-forecast/.worktrees/train merge --no-ff fix/appliance-free-actuals
  cd /home/jovyan/work/ha-energy-forecast/.worktrees/train && $PY -m pytest tests/ -q
  git push origin train/next
  ```

**Post-deploy check (owner/PM, after the train ships):** the log shows the "migrated to the training-target basis" line once; `pred_history.json` on Samba contains `"actuals_basis": "target"`; expected appliance energy removed is roughly 0.8 kWh/day × 30 days.
