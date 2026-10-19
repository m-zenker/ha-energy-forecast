# Deficit Fade (Indoor Projection) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a transient hour-0 room deficit (e.g. the ~04–05 h morning setpoint step-up) from being held for the whole 48 h projection, which swings the tomorrow forecast by ±3–4 kWh between consecutive hourly runs.

**Architecture:** `_project_indoor_temps` (model.py) holds each heated room at `setpoint − d0` for all 48 h, where `d0` is the deficit at hour 0. Replace the constant `d0` with a per-hour floor deficit `d(i) = d_ss + (d0 − d_ss)·exp(−i/τ_r)`:
- `d_ss` is the room's typical deficit over the last 24 h, taken only from observations at the **same setpoint** as now;
- `τ_r` is a recovery time constant measured from the climate caches.

Hour 0 is unchanged (`d(0) = d0`), so the published sensors that read hour 0 don't move. While touching the same lines, skip NaN (entity-unavailable) climate rows when picking the latest observation; today a trailing NaN row turns the whole room projection into NaN.

**Tech Stack:** Python 3.13, pandas/numpy, pytest, AppDaemon app `apps/energy_forecast`.

**Spec:** No separate spec. The design evidence is in `memory/project_forecast_jumps_snapshot_broadcast.md`:
- Hourly-origin replay with fixed weather on 10-17: the 05 h origin sees wohnzimmer 20.5/21 °C and esszimmer 19.5/20 °C at 04:00 (setpoint just raised). That gives `thermal_pressure_max` 0.5 (training max 0.72) and +3.0–3.75 kWh on the next-day total. The 06 h origin drops back to 0.
- `scripts/backtest_tank_state_features.py`: arm B (no tank features) still swings 0.30 kWh/h on average, so this mechanism is independent of the tank features.

**Base branch:** `dev` (currently v0.12.0-alpha-27 + docs). Branch: `fix/deficit-fade` (create with `/feat fix/deficit-fade ha-energy-forecast`).

## Global Constraints

- Run tests only with `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`, never bare `python`; the base env silently degrades results.
- Ruff: line length 120; `ruff check` and `ruff format` must be clean.
- Predict-time only: no feature added, removed or renamed, and training is unchanged (training thermal features come from the measured climate history, not from `_project_indoor_temps`). No retrain or model-schema bump is needed.
- Hour 0 of every room's projection must stay identical to today's (`current_temp[0]`, `deficit[0]`). There is one intended exception: a room whose **latest** row is NaN now starts from its last valid row instead of NaN.
- A room with **no** valid row in `climate_recent` must still come out as an all-NaN frame (today's behaviour), not be skipped. `_engineer_features` counts every room's area in the `thermal_pressure` denominator, and its NaN delta drops out of the sum, which matches how training treats unavailable rooms (model.py ~L3260–3270). Skipping the room would raise pressure and change `thermal_pressure_std` against the training convention (review finding #6).
- **Deploy is HELD.** This changes the C (heating-aware projection) behaviour that the C gate is measuring. Don't merge to `dev`: `dev` = deployed snapshot. Don't deploy until the user decides the sequencing; the default proposal is to ship together with A+B after the C verdict, then re-baseline all three. See `docs/superpowers/plans/2026-10-15-cold-onset-forecast-fixes-index.md`.
- `scripts/` is gitignored: the replay gate script is local-only and is never committed.

## Review Focus

1. **Trailing unavailable row:** the latest climate row has NaN `current_temp` and `setpoint` (HA entity unavailable; ≈ 430 such hours live across 9 rooms). Expected: use the last valid row (still subject to the 2 h staleness rule) and never return NaN. Test: `test_trailing_nan_row_uses_last_valid_observation` (Task 2).
2. **Room with only NaN rows in the window:** no crash; it is kept as an all-NaN frame so the area-weighted `thermal_pressure` denominator matches training. Test: `test_all_nan_rows_room_kept_as_nan_frame` (Task 2).
3. **Room that genuinely can't reach its setpoint** (persistent deficit at an unchanged setpoint): the deficit must be kept, not faded away, because that's real heat demand. Test: `test_persistent_deficit_at_same_setpoint_is_kept` (Task 2).
4. **Night-setback start** (setpoint just lowered, room above it, with yesterday's daytime deficits measured against a different setpoint): no phantom deficit may appear against the new low setpoint. Test: `test_setpoint_change_ignores_other_setpoint_history` (Task 2).
5. **Hour-0 invariance:** for every case the published hour-0 values are unchanged. Test: `test_hour_zero_unchanged_by_fade` (Task 2).

---

### Task 1: Measure the deficit recovery time constant τ_r

Analysis only; nothing is committed. It fixes the value of `DEFICIT_RECOVERY_TAU_H` used in Task 2.

**Files:**
- Create (local, gitignored): `scripts/measure_deficit_recovery.py`

**Interfaces:**
- Produces: one number, the median e-folding recovery time in hours, written into Task 2's `const.py` value. If fewer than 10 usable events are found, use **2.0 h** and record that the default was used.

- [ ] **Step 1: Refresh data**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python scripts/pull_ha_data.py`
Expected: `data/climate_*.csv` updated (mtime today).

- [ ] **Step 2: Write the measurement script**

```python
#!/usr/bin/env python3
"""Measure how fast a room closes a deficit after a setpoint step-up (plan 2026-10-19-deficit-fade, Task 1).

Event: setpoint rises by >= 0.5 K between consecutive hourly rows and the room starts >= 0.3 K below the
new setpoint. Recovery time = first hour h (1..12) where deficit <= d0 / e; events that never recover
within 12 h, or whose setpoint changes again inside the window, are dropped.
"""

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).parent.parent / "data"

times = []
for f in sorted(DATA.glob("climate_*.csv")):
    d = pd.read_csv(f)
    d["timestamp"] = pd.to_datetime(d["timestamp"], format="mixed")
    d = d.dropna(subset=["current_temp", "setpoint"]).groupby("timestamp").last().asfreq("1h")
    sp, t = d["setpoint"].values, d["current_temp"].values
    for i in range(1, len(d) - 12):
        if not (np.isfinite(sp[i - 1]) and np.isfinite(sp[i]) and sp[i] - sp[i - 1] >= 0.5):
            continue
        d0 = sp[i] - t[i]
        if not np.isfinite(d0) or d0 < 0.3:
            continue
        win_sp, win_t = sp[i : i + 13], t[i : i + 13]
        if not np.all(win_sp[np.isfinite(win_sp)] == sp[i]):
            continue
        defs = win_sp - win_t
        hit = np.where(np.isfinite(defs[1:]) & (defs[1:] <= d0 / np.e))[0]
        if len(hit):
            times.append(int(hit[0]) + 1)
        print(f"{f.stem} {d.index[i]} d0={d0:.2f} recovered_h={int(hit[0]) + 1 if len(hit) else None}")

print(f"\nevents with recovery: {len(times)}")
if times:
    print(f"median e-folding hours: {np.median(times):.1f}  p25 {np.percentile(times, 25):.1f}  p75 {np.percentile(times, 75):.1f}")
```

- [ ] **Step 3: Run it**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python scripts/measure_deficit_recovery.py | tail -5`
Expected: a median in hours. Hourly resolution means the result is an integer ≥ 1.
Decision: `DEFICIT_RECOVERY_TAU_H = max(1.0, median)` if ≥ 10 events, else `2.0`. Write the number and the event count into this plan under "Task 1 result" below before starting Task 2.

**Task 1 result (2026-10-19 14:23 pull):** 64 events, median 2.0 h (p25 1, p75 3) → `DEFICIT_RECOVERY_TAU_H = 2.0`. Review dry run on 10-19 data: 64 events, median 2.0 h (p25 1, p75 3). Caveat: room temperatures are quantised to 0.5 K and 35/64 events have d0 = 0.5, so "≤ d0/e" in practice means "back on target". Hourly resolution also rounds up, so treat τ_r as coarse (±1 h).

---

### Task 2: Fade the floor deficit and skip NaN climate rows

**Files:**
- Modify: `apps/energy_forecast/const.py:41-43` (Thermal Projection block). Add three constants after `DEFAULT_TAU`.
- Modify: `apps/energy_forecast/model.py:52-61` (`from .const import` block) and `apps/energy_forecast/model.py:2895-3007` (`_project_indoor_temps`).
- Test: `tests/test_setpoint_projection.py`
  - modify `TestHeatingAwareProjection.test_heating_on_holds_initial_deficit` (~L497): its single-observation fixture now fades by design, so give it real same-setpoint history;
  - add the new class `TestDeficitFade`.

**Interfaces:**
- Consumes: τ_r from Task 1.
- Produces (const.py):
  - `DEFICIT_RECOVERY_TAU_H: float`, the e-folding time of a transient hour-0 deficit;
  - `DEFICIT_STEADY_WINDOW_H = 24`, the look-back window for the steady deficit;
  - `DEFICIT_STEADY_MIN_ROWS = 3`, the minimum number of same-setpoint rows before the steady deficit is trusted.
- Unchanged: the `_project_indoor_temps(...)` signature and its return shape `{entity_id: DataFrame[timestamp, current_temp, setpoint, deficit]}`.
- Replay hook: Task 3 sets `model.DEFICIT_RECOVERY_TAU_H = float("inf")` to reproduce the old behaviour (`exp(-i/inf) = 1` ⇒ `d(i) = d0`). The function must therefore read the module-level name at call time and must not bind it as a default argument.

- [ ] **Step 1: Write the failing tests**

Add this helper below `_heating_active_series` in `tests/test_setpoint_projection.py`:

```python
def _history(ts0: pd.Timestamp, rows: list[tuple[float, float]]) -> dict:
    """climate_recent for one room: hourly (current_temp, setpoint) rows ending at ts0 (last row = ts0)."""
    n = len(rows)
    stamps = pd.date_range(end=ts0, periods=n, freq="1h")
    return {
        "climate.room": pd.DataFrame(
            {"timestamp": stamps, "current_temp": [r[0] for r in rows], "setpoint": [r[1] for r in rows]}
        )
    }
```

Replace the body of `test_heating_on_holds_initial_deficit` (keep its name, update the docstring):

```python
    def test_heating_on_holds_initial_deficit(self):
        """A deficit seen at the SAME setpoint over the last 24 h is real demand and is held (deficit fade keeps d_ss)."""
        ts = _future_ts()
        p = self._proj(_history(ts[0], [(20.5, 21.0)] * 24), ts, [1] * 48)
        assert p["deficit"].values == pytest.approx([0.5] * 48)
```

Add the new class at the end of the file:

```python
class TestDeficitFade:
    """Transient hour-0 deficits fade with DEFICIT_RECOVERY_TAU_H; persistent same-setpoint deficits are kept (2026-10-19)."""

    def _proj(self, cr, ts, states=None, setpoint_on=19.4, setpoint_off=12.0, t_out=5.0):
        return _project_indoor_temps(
            cr,
            ts,
            _outdoor_series(ts, t_out),
            tau_hours=11.64,
            heating_active_series=_heating_active_series(ts, states or [1] * len(ts)),
            setpoint_on=setpoint_on,
            setpoint_off=setpoint_off,
        )["climate.room"]

    def test_transient_step_up_deficit_fades(self):
        from apps.energy_forecast.const import DEFICIT_RECOVERY_TAU_H

        ts = _future_ts()
        # 23 h at 21/21 (no deficit), then the latest hour 0.5 K below → transient
        p = self._proj(_history(ts[0], [(21.0, 21.0)] * 23 + [(20.5, 21.0)]), ts)
        d = p["deficit"].values
        assert d[0] == pytest.approx(0.5)
        assert d[1] == pytest.approx(0.5 * np.exp(-1 / DEFICIT_RECOVERY_TAU_H), abs=1e-6)
        assert np.all(np.diff(d) <= 1e-12)
        assert d[24] == pytest.approx(0.5 * np.exp(-24 / DEFICIT_RECOVERY_TAU_H), abs=1e-6)

    def test_morning_step_up_with_yesterdays_daytime_history_fades(self):
        """Real 10-17 05 h shape: yesterday 06–22 h at 21/21, setback 19 overnight, setpoint back to 21 with the
        room 0.5 K below → 18 same-setpoint rows, median deficit 0 → transient fades (live: wohnzimmer n=17)."""
        ts = _future_ts()
        p = self._proj(_history(ts[0], [(21.0, 21.0)] * 17 + [(20.0, 19.0)] * 6 + [(20.5, 21.0)]), ts)
        assert p["deficit"].values[0] == pytest.approx(0.5)
        assert p["deficit"].values[24] < 0.01

    def test_too_few_same_setpoint_rows_fades_to_zero(self):
        from apps.energy_forecast.const import DEFICIT_RECOVERY_TAU_H

        ts = _future_ts()
        # night setback 19 (room at 20), setpoint just raised to 21 with the room at 20.5:
        # only 1 row at setpoint 21 (< DEFICIT_STEADY_MIN_ROWS) → d_ss = 0
        p = self._proj(_history(ts[0], [(20.0, 19.0)] * 6 + [(20.5, 21.0)]), ts)
        assert p["deficit"].values[0] == pytest.approx(0.5)
        assert p["deficit"].values[24] == pytest.approx(0.5 * np.exp(-24 / DEFICIT_RECOVERY_TAU_H), abs=1e-6)

    def test_persistent_deficit_at_same_setpoint_is_kept(self):
        ts = _future_ts()
        p = self._proj(_history(ts[0], [(20.0, 21.0)] * 24), ts)
        assert p["deficit"].values == pytest.approx([1.0] * 48)

    def test_current_better_than_usual_relaxes_to_steady_deficit(self):
        ts = _future_ts()
        # usually 1 K below at setpoint 21, right now on target → floor relaxes to the usual 1 K
        p = self._proj(_history(ts[0], [(20.0, 21.0)] * 23 + [(21.0, 21.0)]), ts)
        d = p["deficit"].values
        assert d[0] == pytest.approx(0.0)
        assert d[24] == pytest.approx(1.0, abs=0.01)

    def test_setpoint_change_ignores_other_setpoint_history(self):
        ts = _future_ts()
        # daytime 0.5 K deficit at setpoint 21; setback just started: setpoint 17, room at 20.5 (above it)
        p = self._proj(_history(ts[0], [(20.5, 21.0)] * 20 + [(20.5, 17.0)]), ts)
        assert p["deficit"].max() == pytest.approx(0.0)  # no phantom 0.5 K against the new setpoint

    def test_stale_observation_has_no_steady_deficit(self):
        ts = _future_ts()
        cr = _history(ts[0] - pd.Timedelta(hours=3), [(20.0, 21.0)] * 24)  # latest row 3 h old → stale
        p = self._proj(cr, ts)
        assert p["current_temp"].values[0] == pytest.approx(21.0)
        assert np.all(p["deficit"].values == pytest.approx(0.0))

    def test_trailing_nan_row_uses_last_valid_observation(self):
        ts = _future_ts()
        cr = _history(ts[0], [(21.0, 21.0)] * 22 + [(20.5, 21.0), (np.nan, np.nan)])
        p = self._proj(cr, ts)
        assert not p.isna().any().any()
        assert p["current_temp"].values[0] == pytest.approx(20.5)  # last valid row is 1 h old → fresh
        assert p["deficit"].values[0] == pytest.approx(0.5)

    def test_all_nan_rows_room_kept_as_nan_frame(self):
        ts = _future_ts()
        cr = _history(ts[0], [(np.nan, np.nan)] * 5)
        out = _project_indoor_temps(cr, ts, _outdoor_series(ts), tau_hours=11.64)
        p = out["climate.room"]  # kept: thermal_pressure's area denominator must match training
        assert len(p) == len(ts)
        assert p[["current_temp", "setpoint", "deficit"]].isna().all().all()

    def test_hour_zero_unchanged_by_fade(self, monkeypatch):
        import apps.energy_forecast.model as model_mod

        ts = _future_ts()
        cases = [
            _history(ts[0], [(21.0, 21.0)] * 23 + [(20.5, 21.0)]),
            _history(ts[0], [(20.0, 21.0)] * 24),
            _history(ts[0], [(20.5, 21.0)] * 20 + [(20.5, 17.0)]),
            _history(ts[0], [(22.0, 18.0)] * 24),
        ]
        for cr in cases:
            faded = self._proj(cr, ts)
            monkeypatch.setattr(model_mod, "DEFICIT_RECOVERY_TAU_H", float("inf"))
            held = self._proj(cr, ts)
            monkeypatch.undo()
            assert faded["current_temp"].values[0] == pytest.approx(held["current_temp"].values[0])
            assert faded["deficit"].values[0] == pytest.approx(held["deficit"].values[0])
        # the replay hook really switches behaviour (fails if the constant were bound as a default argument)
        cr = cases[0]
        monkeypatch.setattr(model_mod, "DEFICIT_RECOVERY_TAU_H", float("inf"))
        assert self._proj(cr, ts)["deficit"].values[24] == pytest.approx(0.5)
        monkeypatch.undo()
        assert self._proj(cr, ts)["deficit"].values[24] < 0.01
```

Note: the tests import from `apps.energy_forecast...` (the file's existing import style). `monkeypatch.setattr(model_mod, ...)` therefore has to patch that same module object. The tail of `test_hour_zero_unchanged_by_fade` pins the Task 3 hook: the constant must be read at call time. Its `< 0.01` assumes τ_r ≤ 4 h, true for the measured 2.0 h; if Task 1 yields more, use `0.5 * np.exp(-24 / DEFICIT_RECOVERY_TAU_H) + 1e-6`.
Several of the new tests (persistent, setpoint change, stale) pass on today's code too. That is intentional: they guard behaviour the change must keep.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_setpoint_projection.py -v`
Expected: the module still imports, because the constants are imported inside the tests.
- These **6** fail: `test_transient_step_up_deficit_fades` and `test_too_few_same_setpoint_rows_fades_to_zero` (both `ImportError` on `DEFICIT_RECOVERY_TAU_H`), plus `test_morning_step_up_with_yesterdays_daytime_history_fades`, `test_current_better_than_usual_relaxes_to_steady_deficit`, `test_trailing_nan_row_uses_last_valid_observation` and `test_hour_zero_unchanged_by_fade`.
- Everything else passes, including the rewritten `test_heating_on_holds_initial_deficit` and the guard tests (persistent, setpoint change, stale, all-NaN).
- Verified on a scratch copy on 2026-10-19: with the constants present, the same 6 fail on old code; with the Task 2 code, the file gives 35 passed and the full suite 1165 passed, 11 skipped.

- [ ] **Step 3: Add the constants**

In `apps/energy_forecast/const.py`, directly after `DEFAULT_TAU = 12.0 ...` (Thermal Projection block):

```python
# Thermostat recovery of a transient hour-0 deficit in the indoor projection (2026-10-19): the floor
# deficit decays from d0 to the room's steady same-setpoint deficit with this e-folding time.
DEFICIT_RECOVERY_TAU_H = 2.0  # hours — value from plan 2026-10-19-deficit-fade Task 1
DEFICIT_STEADY_WINDOW_H = 24  # hours of climate history used for the steady deficit
DEFICIT_STEADY_MIN_ROWS = 3  # fewer same-setpoint rows → steady deficit 0 (setpoint just changed)
```

Replace `2.0` with the Task 1 result.

In `apps/energy_forecast/model.py`, extend the `from .const import (...)` block alphabetically:

```python
from .const import (
    APPLIANCE_MAX_WINDOW_HOURS,
    DEFAULT_ROOM_AREA_M2,
    DEFAULT_TAU,
    DEFICIT_RECOVERY_TAU_H,
    DEFICIT_STEADY_MIN_ROWS,
    DEFICIT_STEADY_WINDOW_H,
    HOLDOUT_FRACTION,
    MIN_CV_ROWS,
    MIN_TRAINING_ROWS,
    SENSOR_BLEND_HOURS,
    SENSOR_FULL_TRUST_HOURS,
)
```

- [ ] **Step 4: Implement in `_project_indoor_temps`**

(a) Replace

```python
        cdf = cdf.copy()
        cdf["timestamp"] = pd.to_datetime(cdf["timestamp"])
        latest = cdf.sort_values("timestamp").iloc[-1]

        age = now - latest["timestamp"]
        if age <= stale_threshold:
            t_in_start = float(latest["current_temp"])
        else:
            t_in_start = float(latest["setpoint"])
```

with

```python
        cdf = cdf.copy()
        cdf["timestamp"] = pd.to_datetime(cdf["timestamp"])
        # Entity-unavailable rows carry NaN temp + setpoint (~430 h across 9 rooms live, 2026-10-19). A
        # trailing one used to make the whole room projection NaN — use the last valid row instead.
        cdf = cdf.dropna(subset=["current_temp", "setpoint"]).sort_values("timestamp")
        if cdf.empty:
            # No valid reading at all: keep an all-NaN frame (pre-existing behaviour) so the room's area stays
            # in thermal_pressure's denominator exactly as in training, where unavailable rooms count too.
            nan_col = np.full(len(future_timestamps), np.nan)
            result[eid] = pd.DataFrame(
                {"timestamp": future_timestamps, "current_temp": nan_col, "setpoint": nan_col, "deficit": nan_col}
            )
            continue
        latest = cdf.iloc[-1]

        age = now - latest["timestamp"]
        steady_deficit = 0.0
        if age <= stale_threshold:
            t_in_start = float(latest["current_temp"])
            # Typical deficit at the CURRENT setpoint over the last DEFICIT_STEADY_WINDOW_H: a room that
            # persistently sits below target keeps that demand; rows at other setpoints (night setback,
            # just-raised morning setpoint) don't count — they'd invent or hide deficit vs the new target.
            window = cdf[cdf["timestamp"] > latest["timestamp"] - pd.Timedelta(hours=DEFICIT_STEADY_WINDOW_H)]
            same_sp = window[(window["setpoint"] - float(latest["setpoint"])).abs() < 0.05]
            if len(same_sp) >= DEFICIT_STEADY_MIN_ROWS:
                steady_deficit = float(np.median(np.maximum(0.0, same_sp["setpoint"] - same_sp["current_temp"])))
        else:
            t_in_start = float(latest["setpoint"])
```

(b) Replace

```python
        initial_deficit = max(0.0, room_setpoint - t_in_start) if on_arr[0] else 0.0
        t_in_arr = np.empty(n)
        t_in_arr[0] = t_in_start
        for i in range(1, n):
            free = t_in_arr[i - 1] + (t_out_arr[i - 1] - t_in_arr[i - 1]) / tau
            t_in_arr[i] = max(free, setpoint_arr[i] - initial_deficit) if on_arr[i] else free
```

with

```python
        # The hour-0 deficit is often transient (setpoint just stepped up, e.g. ~04–05 h): holding it for
        # 48 h swung the tomorrow forecast by +3–4 kWh for one hourly run (2026-10-19 replay). It now
        # decays to the room's steady same-setpoint deficit with DEFICIT_RECOVERY_TAU_H; d(0) = d0 keeps
        # hour 0 (what the published sensors read) unchanged. Read the module global at call time — the
        # replay gate patches it to inf to reproduce the old constant-d0 floor.
        if on_arr[0]:
            initial_deficit = max(0.0, room_setpoint - t_in_start)
            decay = np.exp(-np.arange(n, dtype=float) / DEFICIT_RECOVERY_TAU_H)
            floor_deficit = steady_deficit + (initial_deficit - steady_deficit) * decay
        else:
            floor_deficit = np.zeros(n)
        t_in_arr = np.empty(n)
        t_in_arr[0] = t_in_start
        for i in range(1, n):
            free = t_in_arr[i - 1] + (t_out_arr[i - 1] - t_in_arr[i - 1]) / tau
            t_in_arr[i] = max(free, setpoint_arr[i] - floor_deficit[i]) if on_arr[i] else free
```

(c) In the docstring, replace the line `while heating is on (thermostat floor), else the free term alone; ``d0`` is the deficit at hour 0 (0 when heating only switches on later).` with:

```
    while heating is on (thermostat floor), else the free term alone. The floor deficit is
    ``d(i) = d_ss + (d0 − d_ss)·exp(−i / DEFICIT_RECOVERY_TAU_H)``: ``d0`` is the deficit at hour 0,
    ``d_ss`` the median deficit at the current setpoint over the last DEFICIT_STEADY_WINDOW_H (0 with
    fewer than DEFICIT_STEADY_MIN_ROWS such rows); both are 0 when heating only switches on later or
    the observation is stale. NaN (entity-unavailable) climate rows are skipped; a room with no valid row
    yields an all-NaN frame.
```

Also update the formula line above it to `T_in[t+1] = max(T_in[t] + (T_out[t] - T_in[t]) / tau, setpoint[t] - d(t))`.

- [ ] **Step 5: Run the file's tests**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_setpoint_projection.py -v`
Expected: all PASS.
- If `test_transient_step_up_deficit_fades` fails on `d[1]`, check that the free term doesn't bind: T_out 5, τ 11.64 gives about −1.3 K/h, which is below the floor.
- If `test_current_better_than_usual_relaxes_to_steady_deficit` fails on `d[24]`, the room must cool to `21 − d(24)` via the free term. It does, at −1.3 K/h. Don't loosen the tolerance without understanding the failure.

- [ ] **Step 6: Run the full suite and ruff**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -q 2>&1 | tail -5`
Expected: all pass, **1176 collected (1165 passed, 11 skipped)**. The `dev` baseline is 1166 collected (1155 passed, 11 skipped), plus 10 new tests; the rewritten existing test doesn't add to the count.
- `tests/test_physics.py::...predict_series_uses_climate_recent_projection_path` and the `test_model.py` projection tests go through `_project_indoor_temps`. If one fails, read it before changing anything: a single-observation fixture with a deficit now fades by design. Update the fixture with same-setpoint history (as in Step 1), not the assertion's intent, and say so in the commit body.

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/ruff check apps/ tests/ && /home/jovyan/my_envs/ha-energy-forecast/bin/ruff format --check apps/ tests/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add apps/energy_forecast/const.py apps/energy_forecast/model.py tests/test_setpoint_projection.py
git commit -m "fix: fade transient hour-0 room deficit in indoor projection and skip NaN climate rows"
```
(Add the co-author trailer per the repo's commit convention.)

---

### Task 3: Replay gate on the live model (local, not committed)

Paired comparison of the old constant-`d0` floor (`DEFICIT_RECOVERY_TAU_H = inf`) against the new fade, on the **live** model. It replays hourly forecast starts over the cold period with fixed archive weather. The live model was trained through 10-19, so the ML part is in-sample; both arms share it, so only the difference is meaningful.

**Files:**
- Create (local, gitignored): `scripts/replay_deficit_fade.py`

**Interfaces:**
- Consumes: `model.DEFICIT_RECOVERY_TAU_H` (Task 2). Helpers `backtest_heating_feature.load_target / _csv / freeze_now / SWITCH / _REAL_NOW / hs` (existing local script `scripts/backtest_heating_feature.py`).

- [ ] **Step 1: Refresh data and models**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python scripts/pull_ha_data.py`
Expected: `data/models/energy_model.pkl` has today's mtime.

- [ ] **Step 2: Write the script**

```python
#!/usr/bin/env python3
"""Replay gate for plan 2026-10-19-deficit-fade: constant-d0 floor (hold) vs faded floor (fade) on the live model.

Hourly origins, archive weather as a perfect 48 h forecast, live tank/climate/sub-sensor inputs as the app
passes them. Reports next-day-total stability, MAE/bias by lead on cold target days, and a paired daily
sign test. In-sample for the ML model (trained through the pull date) — read the arm DIFFERENCE only.

Known representativeness gaps (review 2026-10-19), all shared by both arms:
- Climate history here is continuous, so the stale branch (> 2 h old row → start at setpoint, d = 0) never fires. Live,
  the cache edge ages because HA records only changes: at the 10-19 pull 6/10 rooms were > 2 h stale.
- physics_model isn't passed (physics_kwh filled with 0.0; 0 gain in the live model).
- heating_active_series is the actual switch state (oracle), not hs.project_heating_active.
- Weather is capped at today, so the last origins have < 48 h of forecast (next-day totals stay complete).

Usage:
  /home/jovyan/my_envs/ha-energy-forecast/bin/python -W ignore scripts/replay_deficit_fade.py \
      [--start "2026-10-09 00:00"] [--end "2026-10-18 23:00"]
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile
from datetime import date, timedelta

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import backtest_heating_feature as bh  # noqa: E402  (stubs hassapi, puts apps/ on sys.path)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from energy_forecast import ha_data  # noqa: E402
from energy_forecast import model as mm  # noqa: E402
from energy_forecast.weather import fetch_historical_weather  # noqa: E402

DATA = REPO / "data"
TZ = "Europe/Zurich"
COLD_BELOW_C = 13.5


def _asof(series: pd.Series, when: pd.DatetimeIndex) -> np.ndarray:
    idx = series.index.searchsorted(when, side="right") - 1
    vals = series.to_numpy(dtype=float)
    return np.where(idx >= 0, vals[np.clip(idx, 0, None)], np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-10-09 00:00")
    ap.add_argument("--end", default="2026-10-18 23:00")
    args = ap.parse_args()
    pd.set_option("display.width", 200)

    cfg = yaml.safe_load((DATA / "apps.yaml").read_text())["energy_forecast"]
    target = bh.load_target()
    evk = bh._csv("ev_se_one_ev_charger_192_168_1_128_total_energy.csv", "kwh")
    base, _ = ha_data.split_ev_charging_from_sensor(target, evk)
    clim = {}
    for eid in cfg["climate_entities"]:
        d = pd.read_csv(DATA / f"climate_{eid.split('.', 1)[1]}.csv")
        d["timestamp"] = pd.to_datetime(d["timestamp"], format="mixed")
        clim[eid] = d.groupby("timestamp").last().reset_index()
    dhw = bh._csv("dhw_em_kermi_bridge_kermi_hot_water_temperature.csv", "buffer_temp")
    hbuf = bh._csv("physics_heating_buffer_em_kermi_bridge_kermi_buffer_temperature.csv", "heating_buffer_temp")
    switch = bh._csv(bh.SWITCH, "heating_active").set_index("timestamp")["heating_active"].sort_index()
    subs = {f.stem: bh._csv(f.name, "kwh") for f in DATA.glob("sub_*.csv")}
    w_end = min(pd.Timestamp(args.end).date() + timedelta(days=2), date.today())
    weather = fetch_historical_weather(cfg["latitude"], cfg["longitude"], date(2026, 9, 25), w_end, timezone=TZ)
    day_means = bh.hs.daily_mean_temp(weather)
    day_means = pd.Series(day_means.values, index=pd.to_datetime(day_means.index))
    areas = cfg.get("climate_room_areas") or {}
    act = base.set_index("timestamp")["gross_kwh"]
    origins = pd.date_range(args.start, args.end, freq="1h")
    fade_tau = mm.DEFICIT_RECOVERY_TAU_H

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        mdir = pathlib.Path(tmp) / "models"
        shutil.copytree(DATA / "models", mdir)
        model = mm.EnergyForecastModel(mdir, timezone=TZ)
        for arm, tau_r in (("hold", float("inf")), ("fade", fade_tau)):
            mm.DEFICIT_RECOVERY_TAU_H = tau_r
            for origin in origins:
                bh.freeze_now(origin)
                fc = weather[(weather["timestamp"] >= origin) & (weather["timestamp"] < origin + pd.Timedelta(hours=48))]
                recent = base[(base["timestamp"] < origin) & (base["timestamp"] >= origin - pd.Timedelta(days=30))]
                cr = {
                    e: d[(d["timestamp"] < origin) & (d["timestamp"] >= origin - pd.Timedelta(hours=48))]
                    for e, d in clim.items()
                }
                sps = [d["setpoint"].dropna().iloc[-1] for d in cr.values() if d["setpoint"].notna().any()]
                fts = pd.DatetimeIndex(fc["timestamp"])
                p = model.predict(
                    fc,
                    live_temp=None,
                    recent_actuals=recent,
                    sub_sensors_recent={k: v[v["timestamp"] < origin] for k, v in subs.items()},
                    climate_recent=cr,
                    room_areas=areas,
                    heating_active_series=pd.Series(_asof(switch, fts), index=fts).fillna(0),
                    setpoint_on=float(np.mean(sps)),
                    setpoint_off=12.0,
                    dhw_recent=dhw[dhw["timestamp"] < origin].tail(48),
                    heating_buffer_temp_recent=hbuf[hbuf["timestamp"] < origin].tail(48),
                ).set_index("timestamp")["predicted_kwh"]
                for lead, (ts, pv) in enumerate(p.items()):
                    rows.append((arm, origin, ts, lead, pv, act.get(ts, np.nan)))
            print(f"arm {arm}: {len(origins)} origins done", flush=True)
    mm.DEFICIT_RECOVERY_TAU_H = fade_tau
    pd.Timestamp.now = bh._REAL_NOW

    r = pd.DataFrame(rows, columns=["arm", "origin", "ts", "lead", "pred", "act"])
    r["tday"] = r["ts"].dt.normalize()

    # 1) stability of the next-day total between consecutive hourly origins (midnight rollover excluded)
    nxt = r[r["tday"] == r["origin"].dt.normalize() + pd.Timedelta(days=1)]
    tot = nxt.groupby(["arm", "origin"])["pred"].sum().unstack("arm")
    dd = tot.diff()[tot.index.hour != 0].abs()
    print("\n══ Next-day total: |Δ| between consecutive hourly origins (kWh) ══")
    print(pd.DataFrame({"mean": dd.mean(), "p90": dd.quantile(0.9), "max": dd.max()}).round(2))
    print("\nlargest hold jumps and the fade value at the same origin:")
    print(pd.concat([dd["hold"].nlargest(8), dd["fade"].reindex(dd["hold"].nlargest(8).index)], axis=1).round(2))

    # 2) accuracy on cold target days
    s = r.dropna(subset=["act"]).copy()
    s["err"] = s["pred"] - s["act"]
    s["bucket"] = np.where(s["lead"] < 24, "0-23h", "24-47h")
    s["cold"] = s["tday"].map(day_means).lt(COLD_BELOW_C)
    c = s[s["cold"]]
    print("\n══ Cold target days: MAE / bias (kWh/h) ══")
    print(
        c.groupby(["bucket", "arm"])["err"]
        .agg(MAE=lambda e: e.abs().mean(), bias="mean", n="size")
        .unstack("arm")
        .round(4)
        .to_string()
    )
    if c.empty:
        print("\nno cold target days with actuals in range — accuracy gate not evaluable; widen --start/--end")
        return
    per_day = c[c["bucket"] == "24-47h"].groupby(["tday", "arm"])["err"].apply(lambda e: e.abs().mean()).unstack("arm")
    worse = int((per_day["fade"] > per_day["hold"] + 1e-9).sum())
    print(f"\npaired daily (24-47h): fade worse on {worse}/{len(per_day)} cold days")

    # 3) gate
    stab_gain = 1 - dd["fade"].mean() / dd["hold"].mean()
    m = c[c["bucket"] == "24-47h"].groupby("arm")["err"].apply(lambda e: e.abs().mean())
    mae_rel = m["fade"] / m["hold"] - 1
    ok = stab_gain >= 0.25 and mae_rel <= 0.02 and worse <= len(per_day) // 2
    print(f"\nstability gain {stab_gain:.0%} (need ≥ 25%), cold 24-47h MAE {mae_rel:+.1%} (need ≤ +2%), "
          f"fade worse on {worse}/{len(per_day)} days (need ≤ half)")
    print("GATE:", "GO" if ok else "STOP")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -W ignore scripts/replay_deficit_fade.py 2>&1 | grep -v "signature has high variability\|reindex introduces\|NaN values" | tail -40`
Expected (sanity):
- The `hold` arm's largest jumps include origins at about 04–05 h with |Δ| ≥ 2 kWh, the mechanism being fixed. If they don't, stop: either the hook isn't taking effect or the mechanism wasn't captured.
- The `fade` arm's value at those origins is much smaller.
- Runtime is about 2 × 240 origins at ~0.25 s each.
- Other swing sources remain and appear identically in both arms. The main one is `heating_buffer_temp`, the latest reading copied to all hours, e.g. 10-17 08 h ≈ 3.4 kWh in both. A GO means this mechanism is fixed, not that the tomorrow forecast is stable; list the largest remaining shared jumps in the Task 3 result. (Review dry run, 10-16 22 h to 10-17 08 h: 05 h hold 3.02 vs fade 0.91, 06 h 1.83 vs 0.28, stability gain 34 %.)

Gate: `GO` requires all of:
- stability gain ≥ 25 %;
- cold-day 24–47 h MAE no more than 2 % worse;
- fade worse on at most half of the cold days.

`STOP` → don't merge. Record the output and revisit τ_r or the `d_ss` rule with the user.

- [ ] **Step 4: Record the result**

Paste the stability table, the cold-day table and the gate line into this plan under "Task 3 result", and into `memory/project_forecast_jumps_snapshot_broadcast.md`.

**Task 3 result (2026-10-19, live model pulled 14:23, origins 10-09 00 h – 10-18 23 h):** written gate **STOP**. Stability gain was 18 % against the 25 % bar. Accuracy passed: cold 24–47 h MAE −0.9 % (0.2485 → 0.2462), 0–23 h 0.2162 → 0.2123, fade worse on 1 of 9 cold days.

| next-day total, hourly \|Δ\| (kWh) | mean | p90 | max |
|---|---|---|---|
| hold | 1.06 | 3.40 | 13.00 |
| fade | 0.86 | 3.26 | 6.87 |

Largest hold jumps and the fade value at the same origin: 10-12 04 h 13.00 → 6.87, 10-13 08 h 8.07 → 3.36, 10-13 09 h 6.87 → 2.16, 10-12 06 h 6.62 → 1.40, 10-12 05 h 6.24 → 5.32, 10-18 08 h 5.69 → 4.12, 10-09 01 h 5.27 → 5.27, 10-18 17 h 5.02 → 4.81.

Attribution, from a feature diff at the 10-12 02–07 h origins with the fade code: no `thermal_pressure*` feature changes at any remaining jump. The remaining jumps follow `heating_buffer_temp` (29.6 → 35.9 °C ⇒ +6.87; 35.9 → 32.5 ⇒ −5.32; 30.3 → 38.6 ⇒ +3.58). So the targeted mechanism is removed. The 25 % bar was set on total stability without knowing the buffer's share in the live model (#7 gain), which is the plan defect. See the ledger ruling.

---

### Task 4: Docs and hand-off (deploy held)

**Files:**
- Modify: `CHANGELOG.md` (via `@changelog-writer`), `ROADMAP.md`, `docs/superpowers/plans/2026-10-15-cold-onset-forecast-fixes-index.md`, `MEMORY.md` + `memory/project_forecast_jumps_snapshot_broadcast.md`

- [ ] **Step 0: Commit the pre-existing ROADMAP edits first**

`ROADMAP.md` already carries uncommitted dev-status edits from 2026-10-19: the alpha-27 / v0.11.16 status line, the #105 Step 3b text and the ✅ on sequencing step 0. They travel with the branch when it's created. Commit them on their own **before** adding #107, so this branch's history stays readable (`git add -p` isn't available here):

```bash
git diff --stat ROADMAP.md     # expect 3 changed lines, none about #107
git add ROADMAP.md
git commit -m "docs: roadmap status alpha-27/v0.11.16 and #105 Step 3b note"
```

- [ ] **Step 1: CHANGELOG**

Delegate to `@changelog-writer`: an Unreleased "Fixed" entry. The indoor projection no longer holds a transient hour-0 room deficit for 48 h: it decays with the measured τ_r to the room's same-setpoint steady deficit, which removes ±3–4 kWh hour-to-hour swings in the tomorrow forecast. Unavailable (NaN) climate rows are skipped. Hour-0 sensors are unchanged; no retrain.

- [ ] **Step 2: ROADMAP**

Add row `| 107 | Deficit fade in indoor projection (`fix/deficit-fade`) | removes ±3–4 kWh hourly swings of the tomorrow forecast (transient hour-0 deficit held 48 h) | done on branch, deploy held | ships with A+B after the C verdict; restarts the C gate count |` directly after row 106, using the table's existing column layout.

- [ ] **Step 3: Cold-onset index**

Under "Post-deploy verification & attribution" add a line: deficit fade (ROADMAP #107) changes C's projection; deploy with A+B, and only target days predicted entirely after that deploy count towards the re-baseline.

- [ ] **Step 4: Memory**

Update `memory/project_forecast_jumps_snapshot_broadcast.md` (fix status, branch, gate result) and its `MEMORY.md` index line.

- [ ] **Step 5: Commit docs on the branch**

```bash
git add CHANGELOG.md ROADMAP.md docs/superpowers/plans/2026-10-15-cold-onset-forecast-fixes-index.md docs/superpowers/plans/2026-10-19-deficit-fade.md
git commit -m "docs: deficit fade changelog, roadmap #107, cold-onset sequencing"
```
(Co-author trailer per convention. `MEMORY.md` and `memory/` are gitignored; never `git add` them.)

- [ ] **Step 6: Push the branch, don't merge**

```bash
git push -u origin fix/deficit-fade
```
Then stop. Merge, alpha tag and deploy follow the user's sequencing decision through `@deploy-agent` Workflow A, together with A (`fix/counter-reset-increments`) and B (`fix/ev-sensor-trust`).
