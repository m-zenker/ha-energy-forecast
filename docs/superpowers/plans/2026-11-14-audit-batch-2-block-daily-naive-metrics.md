# Block, Daily and Naive-Guardrail Accuracy (ROADMAP #114) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** publish 3 h-block MAE and daily bias as live sensors, with the naive same-hour mean as a guardrail, and add the same columns to the lean backtest kit's summary.

**Architecture:** a new pure module `apps/energy_forecast/accuracy.py` computes all metrics from the two dicts the app already keeps (`_pred_history`, `_actuals_history`). `_update_sensors` calls it for a 7-day and a 30-day window; `_publish` writes four sensors. Definitions mirror `backtest/wf_score.py`, pinned by a parity test. The lean kit (`backtest/runner.py`, `backtest/score.py`) gets `mae3` and naive columns.

**Tech Stack:** Python 3.13, pandas, numpy, pytest, AppDaemon (MQTT discovery or `set_state`).

**Spec:** `ROADMAP.md` #114 and Design Decisions "Primary accuracy metric" (on `train/next`); index `docs/superpowers/plans/2026-11-14-audit-batch-index.md`.

## Global Constraints

- **Prerequisite:** phase 1 (`fix/appliance-free-actuals`) is merged into `train/next`. Check: `git log train/next --oneline | grep -i "appliance-free"` prints a line. If not, stop.
- Base branch `train/next`; branch `feat/block-daily-accuracy`; worktree `.worktrees/block-daily-accuracy`.
- Tests only via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest` (`$PY`).
- Existing sensors `mae_7d`, `mae_30d`, `relative_mae_*` and `_compute_live_mae` keep their values and code. This phase only adds.
- Metric definitions (identical to `backtest/wf_score.py`): error `e = predicted − actual`; block = 3 h, clock-aligned (`floor("3h")`), counted only with all 3 hours paired; day counted only with ≥ 22 paired hours; naive = mean of the same hour on days −2..−8, NaN-skipping.
- Units: block MAE in kWh per 3 h block; daily bias in kWh/day; `bias_pct` in % of actual consumption on the scored days.
- A metric with no data is NaN internally, published as `unavailable` (state) or `null` (attribute). Never publish 0.0 for "no data".
- `accuracy.py` must not import from `backtest/` (not deployed to HA). `scripts/deploy.py` uploads every `apps/energy_forecast/*.py`, so the new module ships without a deploy change.

## Review Focus

1. EV days: EV hours are absent from `_actuals_history`, so such a day often has < 22 paired hours and is not a scored day; its complete 3 h blocks still count. → Task 1 test.
2. First week after install: fewer than 2 days of actuals, naive is all-NaN; skill values must be NaN, not a division error. → Task 1 test.
3. Naive MAE of exactly 0 (constant load): skill is NaN, no ZeroDivisionError. → Task 1 test.
4. `pred_history` keys loaded from JSON are `pd.Timestamp`, but may be unfloored or strings in tests; pairing must floor to the hour. → Task 1 test.
5. MQTT mode: `_mqtt_set_sensor` turns NaN into 0.0. A bias of 0.0 would read as "perfect". The new sensors must use the raw `unavailable` path. → Task 3 test.

---

## Session setup

```bash
REPO=/home/jovyan/work/ha-energy-forecast
git -C $REPO worktree add $REPO/.worktrees/block-daily-accuracy -b feat/block-daily-accuracy train/next
cd $REPO/.worktrees/block-daily-accuracy
PY=/home/jovyan/my_envs/ha-energy-forecast/bin/python
$PY -m pytest tests/ -q          # baseline: 0 failed
```

## File Structure

- Create `apps/energy_forecast/accuracy.py` — metric functions only, no AppDaemon imports.
- Create `tests/test_accuracy.py`.
- Modify `apps/energy_forecast/energy_forecast.py` — compute in `_update_sensors`, discovery in `_mqtt_publish_all_discovery`, publish in `_publish`, ids in `_cleanup_legacy_states`.
- Modify `backtest/runner.py`, `backtest/score.py`, `tests/test_backtest_score.py`, `tests/test_backtest_runner.py`.
- Modify `README.md`, `backtest/README.md`, `CHANGELOG.md`, `ROADMAP.md`.

---

### Task 1: `accuracy.py`

**Files:**
- Create: `apps/energy_forecast/accuracy.py`
- Test: `tests/test_accuracy.py`

**Interfaces:**
- Produces:
  - constants `BLOCK_HOURS = 3`, `MIN_DAY_HOURS = 22`, `NAIVE_DAYS = range(2, 9)`
  - `naive_same_hour_mean(actual: pd.Series) -> pd.Series` (hourly-indexed in, same index out)
  - `paired_errors(pred_history: dict, actuals_history: dict, since: pd.Timestamp | None = None) -> pd.DataFrame` — index = hour (`pd.Timestamp`, floored, sorted), columns `pred`, `actual`, `naive`; one row per hour present in both dicts with `ts >= since`. `naive` is computed from **all** of `actuals_history`, before the `since` filter.
  - `summarize(pairs: pd.DataFrame) -> dict[str, float]` with exactly these keys: `n_hours`, `mae`, `naive_mae`, `skill_pct`, `n_blocks`, `block_mae`, `naive_block_mae`, `block_skill_pct`, `n_days`, `daily_abs_error`, `daily_bias`, `daily_bias_pct`.

Definitions for `summarize` (`e = pred − actual`, `en = naive − actual`):

| Key | Definition |
|---|---|
| `mae` | mean \|e\| over all rows |
| `naive_mae` | mean \|en\| over rows with naive present |
| `skill_pct` | `100 × (1 − mean\|e\| / mean\|en\|)`, both on rows with naive present; NaN if none or naive MAE is 0 |
| `n_blocks`, `block_mae` | blocks with 3 rows; mean \|Σe\| per block |
| `naive_block_mae` | mean \|Σen\| over complete blocks whose 3 naive values are present |
| `block_skill_pct` | as `skill_pct`, on those blocks |
| `n_days` | calendar days with ≥ 22 rows |
| `daily_bias` | mean of Σe per scored day (kWh/day) |
| `daily_abs_error` | mean of \|Σe\| per scored day |
| `daily_bias_pct` | `100 × Σe / Σactual` over all rows of scored days |

Counts are ints (0 when empty); every other value is NaN when it has no data.

- [ ] **Step 1: Write the failing tests** — `tests/test_accuracy.py`

```python
import math

import numpy as np
import pandas as pd
import pytest
from energy_forecast import accuracy

from backtest import wf_score

START = pd.Timestamp("2026-11-01")


def _hist(values, start=START):
    return dict(zip(pd.date_range(start, periods=len(values), freq="1h"), values))


def test_constant_offset_two_days():
    actual = _hist([2.0] * 240)
    pred = {ts: 2.5 for ts in list(actual)[-48:]}
    s = accuracy.summarize(accuracy.paired_errors(pred, actual))
    assert s["n_hours"] == 48 and s["n_blocks"] == 16 and s["n_days"] == 2
    assert s["mae"] == pytest.approx(0.5)
    assert s["block_mae"] == pytest.approx(1.5)
    assert s["daily_bias"] == pytest.approx(12.0) and s["daily_abs_error"] == pytest.approx(12.0)
    assert s["daily_bias_pct"] == pytest.approx(25.0)
    assert s["naive_mae"] == pytest.approx(0.0)
    assert math.isnan(s["skill_pct"]) and math.isnan(s["block_skill_pct"])  # naive error is 0


def test_errors_cancel_inside_a_block():
    actual = _hist([2.0] * 240)
    last = list(actual)[-24:]
    pred = {ts: 2.0 + (1.0, -1.0, 0.0)[i % 3] for i, ts in enumerate(last)}
    s = accuracy.summarize(accuracy.paired_errors(pred, actual))
    assert s["mae"] == pytest.approx(2 / 3)
    assert s["block_mae"] == pytest.approx(0.0)
    assert s["daily_bias"] == pytest.approx(0.0)


def test_incomplete_day_and_block_are_not_scored():
    actual = _hist([2.0] * 240)
    last = list(actual)[-24:-3]  # 21 hours of the final day
    s = accuracy.summarize(accuracy.paired_errors({ts: 2.5 for ts in last}, actual))
    assert s["n_days"] == 0 and math.isnan(s["daily_bias"]) and math.isnan(s["daily_bias_pct"])
    assert s["n_blocks"] == 7 and s["mae"] == pytest.approx(0.5)


def test_skill_against_naive():
    actual = _hist([1.0] * 216 + [3.0] * 24)  # last day jumps to 3.0; naive says 1.0
    last = list(actual)[-24:]
    s = accuracy.summarize(accuracy.paired_errors({ts: 3.0 for ts in last}, actual))
    assert s["naive_mae"] == pytest.approx(2.0) and s["skill_pct"] == pytest.approx(100.0)
    assert s["naive_block_mae"] == pytest.approx(6.0) and s["block_skill_pct"] == pytest.approx(100.0)
    half = accuracy.summarize(accuracy.paired_errors({ts: 2.0 for ts in last}, actual))
    assert half["skill_pct"] == pytest.approx(50.0)


def test_naive_matches_backtest_wf_score():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(0, 3, 24 * 20), index=pd.date_range(START, periods=24 * 20, freq="1h"))
    s = s.drop(s.index[[30, 31, 200, 333]])  # gaps (EV-excluded hours)
    pd.testing.assert_series_equal(accuracy.naive_same_hour_mean(s), wf_score.naive_same_hour_mean(s))


def test_since_filter_keeps_naive_history_and_floors_keys():
    actual = _hist([1.0] * 240)
    keys = list(actual)
    pred = {str(ts + pd.Timedelta(minutes=5)): 1.0 for ts in keys[-72:]}  # unfloored string keys
    pairs = accuracy.paired_errors(pred, actual, since=keys[-24])
    assert len(pairs) == 24 and pairs.index[0] == keys[-24]
    assert pairs["naive"].notna().all()  # computed from the full history, not the filtered window


def test_empty_and_young_history_give_nan_not_errors():
    s = accuracy.summarize(accuracy.paired_errors({}, {}))
    assert s["n_hours"] == 0 and s["n_blocks"] == 0 and s["n_days"] == 0
    assert all(math.isnan(v) for k, v in s.items() if not k.startswith("n_"))
    young = _hist([1.0] * 30)
    s2 = accuracy.summarize(accuracy.paired_errors({ts: 1.2 for ts in young}, young))
    assert s2["mae"] == pytest.approx(0.2) and math.isnan(s2["naive_mae"]) and math.isnan(s2["skill_pct"])
```

- [ ] **Step 2: Run to verify they fail** — `$PY -m pytest tests/test_accuracy.py -q` → `ImportError: cannot import name 'accuracy'`

- [ ] **Step 3: Implement `accuracy.py`** to the Interfaces block and the definitions table. `naive_same_hour_mean` is the same three lines as in `backtest/wf_score.py` (reindex to a full hourly grid, mean of `shift(24 * k)` for `k` in `NAIVE_DAYS`, reindex back); return an empty float Series for empty input. Module docstring: one line saying definitions mirror `backtest/wf_score.py` and are pinned by `tests/test_accuracy.py`.

- [ ] **Step 4: Run** — `$PY -m pytest tests/test_accuracy.py -q` → 7 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/accuracy.py tests/test_accuracy.py
git commit -m "feat(accuracy): 3 h-block, daily and naive-guardrail metrics (#114)"
```

---

### Task 2: Compute in `_update_sensors`

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py`
- Test: `tests/test_energy_forecast.py`

**Interfaces:**
- Consumes: `accuracy.paired_errors`, `accuracy.summarize`.
- Produces: `aggregated["accuracy_7d"]` and `aggregated["accuracy_30d"]`, each the dict returned by `summarize`. Task 3 reads these keys from `data` in `_publish`.

- [ ] **Step 1: Write the failing test**

Add a module-level helper so the wiring is testable without a full app:

```python
def test_accuracy_windows_helper():
    from energy_forecast.energy_forecast import _accuracy_windows

    now = pd.Timestamp("2026-11-14 10:00")
    hours = pd.date_range(now - pd.Timedelta(days=20), now - pd.Timedelta(hours=1), freq="1h")
    actual = {ts: 2.0 for ts in hours}
    pred = {ts: 2.5 for ts in hours}
    out = _accuracy_windows(pred, actual, now)
    assert set(out) == {"accuracy_7d", "accuracy_30d"}
    assert out["accuracy_7d"]["n_hours"] == 7 * 24
    assert out["accuracy_30d"]["n_hours"] == len(hours)
    assert out["accuracy_7d"]["block_mae"] == pytest.approx(1.5)
```

- [ ] **Step 2: Run to verify it fails** — `ImportError: cannot import name '_accuracy_windows'`

- [ ] **Step 3: Implement**

- Add `from . import accuracy` next to the other package imports at the top of `energy_forecast.py` (follow the form the file already uses for `ha_data`, `capture`).
- Add module-level `_accuracy_windows(pred_history: dict, actuals_history: dict, now_ts: pd.Timestamp) -> dict[str, dict]`: 7-day window uses `since = now_ts.floor("1h") - pd.Timedelta(days=7)` (same cutoff as `cutoff_7d`), 30-day uses `since=None`.
- In `_update_sensors`, where `aggregated["mae_7d"] = mae_7d` is assigned (≈ line 2424), add `aggregated.update(_accuracy_windows(self._pred_history, self._actuals_history, now_ts))` inside a `try/except Exception` that logs a warning (`"Accuracy metrics failed: %s"`); a failure here must never stop the forecast publish.

- [ ] **Step 4: Run** — the new test passes; full suite 0 failed.

- [ ] **Step 5: Commit** — `git commit -am "feat(accuracy): compute 7d/30d accuracy windows each update (#114)"`

---

### Task 3: Publish four sensors

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (`_mqtt_publish_all_discovery`, `_publish`, `_cleanup_legacy_states`)
- Test: `tests/test_energy_forecast.py` (class `TestRollingMaeSensors`)

**Interfaces:**
- Consumes: `data["accuracy_7d"]`, `data["accuracy_30d"]` (Task 2).
- Produces these sensors (`{w}` = `7d`, `30d`):

| unique_id | Friendly name | Unit | Icon | State | Attributes |
|---|---|---|---|---|---|
| `energy_forecast_block_mae_{w}` | Energy Forecast Block MAE {w} | kWh | `mdi:chart-timeline-variant` | `block_mae` (4 decimals) | `n_blocks`, `naive_block_mae`, `skill_vs_naive_pct` (= `block_skill_pct`), `hourly_mae` (= `mae`), `hourly_naive_mae`, `hourly_skill_vs_naive_pct` (= `skill_pct`), `block_hours: 3` |
| `energy_forecast_daily_bias_{w}` | Energy Forecast Daily Bias {w} | kWh | `mdi:scale-unbalanced` | `daily_bias` (3 decimals) | `bias_pct` (= `daily_bias_pct`), `daily_abs_error`, `n_days` |

State class `measurement`, device class `None`. Floats in attributes are rounded to 3 decimals; NaN becomes `None`.

- [ ] **Step 1: Write the failing tests**

In `TestRollingMaeSensors`, copy the setup of `test_publish_mae_sensors_set_state_mode`, `test_publish_mae_sensors_mqtt_mode` and `test_mqtt_discovery_includes_mae_sensors` and add:

```python
    def test_publish_accuracy_sensors_set_state_mode(self):
        # data["accuracy_7d"] = {"block_mae": 0.61234, "n_blocks": 50, "naive_block_mae": 0.7, "block_skill_pct": 12.5,
        #                         "mae": 0.3, "naive_mae": 0.31, "skill_pct": 3.2, "daily_bias": -1.2345,
        #                         "daily_bias_pct": -5.1, "daily_abs_error": 2.0, "n_days": 6, "n_hours": 160}
        # data["accuracy_30d"] = every metric NaN, counts 0
        # after _publish(data):
        #   sensor.energy_forecast_block_mae_7d   state "0.6123", attributes n_blocks == 50, skill_vs_naive_pct == 12.5
        #   sensor.energy_forecast_daily_bias_7d  state "-1.234" or "-1.235" (round), attributes bias_pct == -5.1, n_days == 6
        #   sensor.energy_forecast_block_mae_30d  state "unavailable", attributes naive_block_mae is None
        #   sensor.energy_forecast_daily_bias_30d state "unavailable"

    def test_publish_accuracy_sensors_mqtt_mode(self):
        # NaN window  -> _mqtt_set_sensor_raw(uid, "unavailable") is called, _mqtt_set_sensor is NOT called for that uid
        # valid window -> _mqtt_set_sensor(uid, value) and _mqtt_publish_sensor_attributes(uid, attrs)

    def test_mqtt_discovery_includes_accuracy_sensors(self):
        # the four unique_ids are registered, each with a json_attributes_topic ending in f"/sensor/{uid}/attributes"

    def test_publish_without_accuracy_keys_does_not_raise(self):
        # data has no "accuracy_7d"/"accuracy_30d" (e.g. Task 2's try/except fired): all four publish as unavailable
```

Write them as real assertions against the fakes those three existing tests use.

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement**

- Discovery: after the "Relative MAE sensors (#54)" loop (≈ line 1058), register the four sensors with `json_attributes_topic=f"{self._mqtt_discovery_prefix}/energy_forecast/sensor/{uid}/attributes"` (same pattern as `energy_forecast_thermal_pressure_net` a few lines below).
- Publish: in `_publish`, after the "Relative MAE sensors (#54)" block (≈ line 3111), add a block "Block / daily accuracy sensors (#114)". Read with `data.get(f"accuracy_{w}") or {}`. MQTT mode: NaN/missing state → `self._mqtt_set_sensor_raw(uid, "unavailable")`, otherwise `self._mqtt_set_sensor(uid, value)`; always `self._mqtt_publish_sensor_attributes(uid, attrs)`. `set_state` mode: same attribute dict plus `unit_of_measurement`, `friendly_name`, `unique_id`, `icon`, `attribution`, with `replace=True`, as the MAE block does.
- `_cleanup_legacy_states`: add the four `sensor.energy_forecast_*` ids to `legacy_ids`.

- [ ] **Step 4: Run** — new tests pass; full suite 0 failed; `ruff check . && ruff format --check .`

- [ ] **Step 5: Commit** — `git commit -am "feat(accuracy): block MAE and daily bias sensors with naive guardrail (#114)"`

---

### Task 4: Lean backtest kit columns

**Files:**
- Modify: `backtest/runner.py` (`_row`, `run_arm`), `backtest/score.py` (`_segment_table`, `format_summary`)
- Test: `tests/test_backtest_runner.py`, `tests/test_backtest_score.py`

**Interfaces:**
- Produces: result rows gain float columns `mae3`, `naive_mae`, `naive_mae3` (NaN when not computable). `_segment_table` gains columns `mae3`, `mae3_pct`, `naive_mae`, `naive_mae3`. Result pickles written before this change (no `mae3` column) must still summarize; the new columns are then NaN.

- [ ] **Step 1: Write the failing tests**

`tests/test_backtest_score.py` (extend the file's row-builder so rows carry `mae3`, `naive_mae`, `naive_mae3`):

```python
def test_mae3_and_naive_columns_in_summary():
    # base rows: mae3 == 1.0, naive_mae == 0.5, naive_mae3 == 1.2 ; arm "x": mae3 == 0.9
    t = summarize(df)["ALL"]
    assert t.loc["base", "mae3"] == pytest.approx(1.0)
    assert t.loc["x", "mae3_pct"] == pytest.approx(-10.0)
    assert t.loc["base", "naive_mae3"] == pytest.approx(1.2)
    text = format_summary(summarize(df))
    assert "mae3" in text and "dMAE3%" in text and "naive" in text


def test_old_pickle_without_mae3_still_summarizes():
    t = summarize(df.drop(columns=["mae3", "naive_mae", "naive_mae3"]))["ALL"]
    assert np.isnan(t.loc["base", "mae3"])
    format_summary(summarize(df.drop(columns=["mae3", "naive_mae", "naive_mae3"])))  # must not raise
```

`tests/test_backtest_runner.py`:

```python
def test_row_block_and_naive_metrics():
    from backtest.runner import _row
    idx = pd.date_range("2026-11-10", periods=24, freq="1h")
    a = pd.Series(1.0, index=idx)
    p = pd.Series(1.5, index=idx)
    n = pd.Series(1.25, index=idx)
    row = _row(arm, fold, idx[0], 18, False, a, p, n)      # arm/fold: reuse this file's existing fixtures
    assert row["mae3"] == pytest.approx(1.5)
    assert row["naive_mae"] == pytest.approx(0.25) and row["naive_mae3"] == pytest.approx(0.75)
    assert np.isnan(_row(arm, fold, idx[0], 18, False, a, p)["naive_mae"])   # n defaults to None
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement**

- `runner._row(..., a, p, n: pd.Series | None = None)`: `mae3` = mean \|Σ(p − a)\| over clock-aligned 3 h blocks whose 3 hours are all scored (mask `m`), NaN if none; `naive_mae` = mean \|n − a\| over scored hours with `n` present; `naive_mae3` likewise per complete block with all 3 naive values.
- `runner.run_arm`: compute `naive = wf_score.naive_same_hour_mean(actual)` once (import from `.wf_score`), pass `naive.reindex(a.index)` to `_row` in both the `lead25` and the origin branch.
- `score._segment_table`: add `mae3`, `naive_mae`, `naive_mae3` as column means when present (else NaN); `mae3_pct` from `paired_bootstrap(df, arm, ref, col="mae3")["pct"]` for non-reference arms when `mae3` exists.
- `score.format_summary`: add `mae3` and `dMAE3%` columns to the table, and one line per segment under the header, built from the reference arm's row: `  naive same-hour mean (ref hours): mae 0.xxx  mae3 0.xxx`. Skip that line when both are NaN.

- [ ] **Step 4: Run** — full suite 0 failed; ruff clean.

- [ ] **Step 5: Commit** — `git commit -am "feat(backtest): 3 h-block and naive columns in the lean kit summary (#114)"`

---

### Task 5: Docs and merge to the train

- [ ] **Step 1:** `README.md` sensor table: add the four sensors with the definitions from Task 3 and one sentence that 3 h-block MAE and daily bias are the primary accuracy numbers, hourly MAE secondary.
- [ ] **Step 2:** `backtest/README.md`, section "Reading the summary": document `mae3`, `dMAE3%` and the naive line.
- [ ] **Step 3:** CHANGELOG `[Unreleased]` via `/changelog`.
- [ ] **Step 4:** ROADMAP #114 section and Pending Summary row → "implemented on `train/next` (date), not deployed".
- [ ] **Step 5:** `/home/jovyan/work/ha-energy-forecast/MEMORY.md`: one line (sensors added, entity ids).
- [ ] **Step 6:** Merge and push:
  ```bash
  git -C /home/jovyan/work/ha-energy-forecast/.worktrees/train merge --no-ff feat/block-daily-accuracy
  cd /home/jovyan/work/ha-energy-forecast/.worktrees/train && $PY -m pytest tests/ -q
  git push origin train/next
  ```

**Out of scope:** dashboard cards for the new sensors; changing `mae_7d`'s definition; interval calibration (#121).
