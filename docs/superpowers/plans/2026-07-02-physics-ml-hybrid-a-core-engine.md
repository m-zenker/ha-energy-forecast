# Physics-ML Hybrid — Plan A: ThermalPhysicsModel Core Engine + Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ThermalPhysicsModel` — a standalone physics-based hourly electricity predictor (space heating, DHW, base load) with self-calibration from history — with zero coupling to `model.py`/`energy_forecast.py`. Plan B wires it in.

**Architecture:** One new module `apps/energy_forecast/physics.py` exposing the `ThermalPhysicsModel` class per the index doc's interface contract, plus a module-level `_find_passive_windows()` utility added to `model.py` and consumed by both `_calibrate_tau()` (refactored) and this plan's UA_eff calibration.

**Tech Stack:** pandas, numpy (OLS via `np.polyfit`/`np.linalg.lstsq`), stdlib `json`/`os` for atomic writes. No new dependencies.

**Base branch:** `dev`. Branch name: `feat/physics-core-engine ha-energy-forecast`.

## Global Constraints

- `COP_min = 1.1` (floor covering defrost cycles and cold-snap degradation) — spec §3.1.
- `η_carnot = 0.45` (typical ASHP second-law efficiency) — spec §3.1.
- Specific heat of water: `1.163 Wh/(L·K)` — spec §3.1.
- All calibration windows must satisfy `timestamp < holdout_cutoff` (no leakage into the ML holdout) — spec §4.1.
- `physics_calibration.json` and `physics_schedule.json` are written via atomic write-then-rename, never in place — spec §2.2, §4.1.
- Every calibration step returns `None` (not an exception) on insufficient data, logs WARNING, and the caller falls back to the config default — spec §6.
- `_find_passive_windows()` signature is fixed by the index doc's interface contract — do not change it in a way Plan B/C don't expect.

## Assumptions Not Fully Specified in the Design Doc (flag for review before merge)

The spec's DHW ODE (§3.1) references `draw_profile[h]` and `draw_rate` and `T_ambient` without defining their source. Two implementation decisions fill this gap; both keep `physics_calibration.json`'s field list exactly as specified in §2.2 (no new persisted fields invented):

1. **`draw_profile`** is a fixed, non-calibrated 24-value shape constant in `physics.py` (module-level `_DEFAULT_DRAW_PROFILE`), representing a typical household draw pattern (morning + evening peaks). `draw_rate` is derived at runtime from the calibrated `Q_dhw_daily` so that integrating the draw term over 24h reproduces that daily energy target — i.e. `draw_profile` is shape-only, `draw_rate` is the calibrated magnitude. If the real household's draw pattern differs materially (e.g. no evening peak), this under/over-estimates hourly Q_dhw_el timing while keeping the daily total correct; LightGBM's residual (Phase 2) or the feature itself (Phase 1) absorbs the difference.
2. **`T_ambient`** for the DHW tank insulation-loss term is the area-weighted `T_indoor` series (same one used for space heating) — physically correct for an indoors tank — falling back to a config constant (`20.0°C`) when no room thermostats are configured.

Both are implemented in Task 4 with named constants so they're trivially adjustable later.

---

### Task 1: Module skeleton, config dataclass, atomic JSON I/O

**Files:**
- Create: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Produces: `ThermalPhysicsModel.__init__(model_dir: Path, config: dict)`, `_atomic_write_json(path, data)`, `_read_json_or_default(path, default)`, `_default_calibration()`, `_default_schedule()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_physics.py
"""Tests for physics.py — ThermalPhysicsModel."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from energy_forecast.physics import ThermalPhysicsModel, _atomic_write_json, _read_json_or_default


DEFAULT_CONFIG = {
    "cop_sensor": None,
    "dhw_tank_temp_sensor": None,
    "heating_buffer_temp_sensor": None,
    "heating_curve_sensor": None,
    "cop_formula": {"a": 2.5, "b": 0.07},
    "dhw_tank_volume_l": 200,
    "dhw_power_w": 4000,
    "internal_gains_fraction": 0.8,
    "heating_curve_points": [[-20, 55.5], [-5, 46.0], [5, 39.5], [20, 25.0]],
    "room_thermostats": [],
    "use_physics_residual": False,
}


class TestSkeletonAndIO:
    def test_init_creates_model_dir(self, tmp_path):
        model_dir = tmp_path / "models"
        ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        assert model_dir.exists()

    def test_missing_calibration_file_uses_defaults(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        assert pm._calib["UA_eff"] is None
        assert pm._calib["Q_base_el"] == 0.35
        assert pm._calib["n_calibration_windows_ua_eff"] == 0

    def test_missing_schedule_file_uses_defaults(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        assert pm._schedule["T_dhw_upper"] == 55.0
        assert pm._schedule["T_legionella"] == 60.0
        assert pm._schedule["dhw_tank_volume_l"] == 200

    def test_atomic_write_then_read_roundtrip(self, tmp_path):
        path = tmp_path / "calib.json"
        _atomic_write_json(path, {"UA_eff": 150.5, "calibrated_at": "2026-07-02T00:00:00"})
        assert not path.with_suffix(".json.tmp").exists()
        data = _read_json_or_default(path, {})
        assert data["UA_eff"] == 150.5

    def test_read_corrupt_json_falls_back_to_default(self, tmp_path, caplog):
        path = tmp_path / "calib.json"
        path.write_text("{not valid json")
        data = _read_json_or_default(path, {"UA_eff": None})
        assert data == {"UA_eff": None}

    def test_calibration_json_missing_calibrated_at_is_always_stale(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        assert pm.calibration_stale is True

    def test_calibration_json_fresh_is_not_stale(self, tmp_path):
        model_dir = tmp_path / "models"
        model_dir.mkdir(parents=True)
        _atomic_write_json(
            model_dir / "physics_calibration.json",
            {**pytest.importorskip("energy_forecast.physics")._default_calibration(),
             "calibrated_at": pd.Timestamp.now().isoformat()},
        )
        pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        assert pm.calibration_stale is False

    def test_is_cold_start_gated_when_windows_below_30(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib["n_calibration_windows_ua_eff"] = 29
        assert pm.is_cold_start_gated is True
        pm._calib["n_calibration_windows_ua_eff"] = 30
        assert pm.is_cold_start_gated is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'energy_forecast.physics'`

- [ ] **Step 3: Write minimal implementation**

```python
# apps/energy_forecast/physics.py
"""Physics-based hourly electricity predictor (space heating, DHW, base load).

See docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md for the design.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger("energy_forecast.physics")

COP_MIN = 1.1
ETA_CARNOT = 0.45
WATER_SPECIFIC_HEAT_WH_PER_L_K = 1.163
DEFAULT_T_FLOW_C = 45.0
DEFAULT_AMBIENT_C = 20.0
COLD_START_MIN_WINDOWS = 30
STALE_AFTER_DAYS = 30

# Fixed 24h draw-timing shape (sums to 1.0) — see Plan A "Assumptions" section.
# Morning peak (06-08h) + evening peak (18-22h), flat baseline otherwise.
_DEFAULT_DRAW_PROFILE = np.array(
    [0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.08, 0.10, 0.06, 0.03, 0.02, 0.02,
     0.03, 0.02, 0.02, 0.02, 0.03, 0.05, 0.09, 0.11, 0.10, 0.07, 0.04, 0.02]
)
_DEFAULT_DRAW_PROFILE = _DEFAULT_DRAW_PROFILE / _DEFAULT_DRAW_PROFILE.sum()


def _default_calibration() -> dict[str, Any]:
    return {
        "calibrated_at": None,
        "n_calibration_windows_ua_eff": 0,
        "UA_eff": None,
        "solar_gain_area": 0.0,
        "Q_base_el": 0.35,
        "Q_dhw_daily": 3.5,
        "UA_dhw": 15.0,
        "cop_formula": None,  # None → caller falls back to config cop_formula
    }


def _default_schedule() -> dict[str, Any]:
    return {
        "T_dhw_upper": 55.0,
        "T_legionella": 60.0,
        "legionella_dow": 2,
        "legionella_hour": 14,
        "T_dhw_lower": 45.0,
        "dhw_tank_volume_l": 200,
    }


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(tmp_path, path)


def _read_json_or_default(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("root is not an object")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as e:
        _LOGGER.warning(f"Failed to read {path.name}: {e} — using defaults")
        return dict(default)


class ThermalPhysicsModel:
    """Calibrated physics baseline for hourly household electricity consumption."""

    def __init__(self, model_dir: Path, config: dict) -> None:
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._calibration_path = model_dir / "physics_calibration.json"
        self._schedule_path = model_dir / "physics_schedule.json"
        self._config = config

        calib_defaults = _default_calibration()
        self._calib = {**calib_defaults, **_read_json_or_default(self._calibration_path, calib_defaults)}
        schedule_defaults = _default_schedule()
        self._schedule = {**schedule_defaults, **_read_json_or_default(self._schedule_path, schedule_defaults)}

        self._tau_hours: float | None = None  # set externally by Plan B from EnergyForecastModel._tau_hours

    @property
    def calibration_stale(self) -> bool:
        raw = self._calib.get("calibrated_at")
        if not raw:
            return True
        try:
            calibrated_at = pd.Timestamp(raw)
        except (ValueError, TypeError):
            return True
        age_days = (pd.Timestamp.now() - calibrated_at.tz_localize(None)).total_seconds() / 86400
        return age_days > STALE_AFTER_DAYS

    @property
    def is_cold_start_gated(self) -> bool:
        return self._calib.get("n_calibration_windows_ua_eff", 0) < COLD_START_MIN_WINDOWS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add ThermalPhysicsModel skeleton with atomic JSON I/O"
```

---

### Task 2: COP model (Carnot-bounded formula + sensor priority)

**Files:**
- Modify: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Consumes: `COP_MIN`, `ETA_CARNOT`, `DEFAULT_T_FLOW_C` from Task 1.
- Produces: `ThermalPhysicsModel._t_flow_c(t_outdoor_c, live_shift_k) -> float`, `ThermalPhysicsModel._cop_series(timestamps, t_outdoor, cop_sensor_series) -> pd.Series`.

- [ ] **Step 1: Write the failing test**

```python
class TestCOPModel:
    def test_t_flow_from_curve_interpolates(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        # curve points: [[-20, 55.5], [-5, 46.0], [5, 39.5], [20, 25.0]]
        assert pm._t_flow_c(-5, None) == pytest.approx(46.0)
        # midpoint between -5 (46.0) and 5 (39.5) is 0 -> linear interp
        assert pm._t_flow_c(0, None) == pytest.approx(42.75)

    def test_t_flow_applies_live_parallel_shift(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        assert pm._t_flow_c(-5, 3.0) == pytest.approx(49.0)

    def test_t_flow_clamps_outside_curve_domain(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        assert pm._t_flow_c(-30, None) == pytest.approx(55.5)
        assert pm._t_flow_c(30, None) == pytest.approx(25.0)

    def test_cop_carnot_bound_at_minus_15(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        ts = pd.date_range("2026-01-15 00:00", periods=3, freq="1h")
        cop = pm._cop_series(ts, t_outdoor=pd.Series([-15.0, -15.0, -15.0], index=ts), cop_sensor_series=None)
        # Carnot: 0.45 * T_flow_K / (T_flow_K - T_out_K); linear a+b*T_out with a=2.5,b=0.07 -> 2.5+0.07*-15=1.45
        # linear (1.45) < Carnot bound here, so min() picks linear, but floored at COP_MIN=1.1
        assert (cop >= COP_MIN).all()
        assert (cop <= 3.0).all()  # sanity: well below Carnot ceiling at this delta

    def test_cop_floor_never_below_min(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        ts = pd.date_range("2026-01-15 00:00", periods=1, freq="1h")
        # extreme cold where linear formula goes negative
        cop = pm._cop_series(ts, t_outdoor=pd.Series([-40.0], index=ts), cop_sensor_series=None)
        assert cop.iloc[0] == COP_MIN

    def test_cop_sensor_overrides_formula_where_present(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        ts = pd.date_range("2026-01-15 00:00", periods=2, freq="1h")
        sensor = pd.Series([4.2, np.nan], index=ts)  # only hour 0 has a live reading
        cop = pm._cop_series(ts, t_outdoor=pd.Series([-5.0, -5.0], index=ts), cop_sensor_series=sensor)
        assert cop.iloc[0] == pytest.approx(4.2)
        assert cop.iloc[1] != pytest.approx(4.2)  # falls back to formula
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestCOPModel -v`
Expected: FAIL with `AttributeError: 'ThermalPhysicsModel' object has no attribute '_t_flow_c'`

- [ ] **Step 3: Write minimal implementation**

Add to `ThermalPhysicsModel` in `physics.py`:

```python
    def _t_flow_c(self, t_outdoor_c: float, live_shift_k: float | None) -> float:
        points = self._config.get("heating_curve_points") or []
        if not points:
            return DEFAULT_T_FLOW_C
        shift = live_shift_k if live_shift_k is not None else 0.0
        xs = [p[0] for p in points]
        ys = [p[1] + shift for p in points]
        return float(np.interp(t_outdoor_c, xs, ys))

    def _cop_formula_value(self, t_outdoor_c: float, live_shift_k: float | None) -> float:
        formula = self._calib.get("cop_formula") or self._config["cop_formula"]
        a, b = formula["a"], formula["b"]
        t_flow_k = self._t_flow_c(t_outdoor_c, live_shift_k) + 273.15
        t_outdoor_k = t_outdoor_c + 273.15
        denom = t_flow_k - t_outdoor_k
        carnot = ETA_CARNOT * t_flow_k / denom if denom > 0 else COP_MIN
        linear = a + b * t_outdoor_c
        return max(COP_MIN, min(carnot, linear))

    def _cop_series(
        self,
        timestamps: pd.DatetimeIndex,
        t_outdoor: pd.Series,
        cop_sensor_series: pd.Series | None,
        live_shift_series: pd.Series | None = None,
    ) -> pd.Series:
        formula_vals = np.array(
            [
                self._cop_formula_value(
                    t_o, None if live_shift_series is None else live_shift_series.reindex(timestamps).iloc[i]
                )
                for i, t_o in enumerate(t_outdoor.reindex(timestamps).values)
            ]
        )
        result = pd.Series(formula_vals, index=timestamps)
        if cop_sensor_series is not None:
            aligned = cop_sensor_series.reindex(timestamps)
            result = aligned.combine_first(result)
        return result.clip(lower=COP_MIN)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add Carnot-bounded COP model with sensor priority"
```

---

### Task 3: Space heating component

**Files:**
- Modify: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Consumes: `_cop_series()` from Task 2, `self._calib["UA_eff"]`/`solar_gain_area`/`Q_base_el` from Task 1.
- Produces: `ThermalPhysicsModel._space_heating_kwh(t_indoor, t_outdoor, ghi, cop) -> pd.Series`.

- [ ] **Step 1: Write the failing test**

```python
class TestSpaceHeating:
    def test_matches_spec_worked_example(self, tmp_path):
        # spec §7: UA_eff=150, ΔT=10°C, COP=3.0 -> 0.5 kWh/h (no solar/gains/mass)
        config = {**DEFAULT_CONFIG, "internal_gains_fraction": 0.0}
        pm = ThermalPhysicsModel(tmp_path / "models", config)
        pm._calib["UA_eff"] = 150.0
        pm._calib["solar_gain_area"] = 0.0
        pm._calib["Q_base_el"] = 0.0
        ts = pd.date_range("2026-01-15 00:00", periods=2, freq="1h")
        t_indoor = pd.Series([20.0, 20.0], index=ts)  # constant -> Q_mass = 0
        t_outdoor = pd.Series([10.0, 10.0], index=ts)
        ghi = pd.Series([0.0, 0.0], index=ts)
        cop = pd.Series([3.0, 3.0], index=ts)
        q_heat_el = pm._space_heating_kwh(t_indoor, t_outdoor, ghi, cop)
        assert q_heat_el.iloc[0] == pytest.approx(0.5, abs=1e-6)

    def test_solar_offset_reduces_heating_load(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_eff=150.0, solar_gain_area=10.0, Q_base_el=0.0)
        ts = pd.date_range("2026-01-15 12:00", periods=2, freq="1h")
        t_indoor = pd.Series([20.0, 20.0], index=ts)
        t_outdoor = pd.Series([10.0, 10.0], index=ts)
        cop = pd.Series([3.0, 3.0], index=ts)
        no_sun = pm._space_heating_kwh(t_indoor, t_outdoor, pd.Series([0.0, 0.0], index=ts), cop)
        with_sun = pm._space_heating_kwh(t_indoor, t_outdoor, pd.Series([200.0, 200.0], index=ts), cop)
        assert with_sun.iloc[0] < no_sun.iloc[0]

    def test_internal_gains_reduce_q_heat(self, tmp_path):
        config = {**DEFAULT_CONFIG, "internal_gains_fraction": 0.8}
        pm = ThermalPhysicsModel(tmp_path / "models", config)
        pm._calib.update(UA_eff=150.0, solar_gain_area=0.0, Q_base_el=0.35)
        ts = pd.date_range("2026-01-15 00:00", periods=2, freq="1h")
        t_indoor = pd.Series([20.0, 20.0], index=ts)
        t_outdoor = pd.Series([10.0, 10.0], index=ts)
        ghi = pd.Series([0.0, 0.0], index=ts)
        cop = pd.Series([3.0, 3.0], index=ts)
        with_gains = pm._space_heating_kwh(t_indoor, t_outdoor, ghi, cop).iloc[0]
        config_no_gains = {**DEFAULT_CONFIG, "internal_gains_fraction": 0.0}
        pm2 = ThermalPhysicsModel(tmp_path / "models2", config_no_gains)
        pm2._calib.update(UA_eff=150.0, solar_gain_area=0.0, Q_base_el=0.35)
        no_gains = pm2._space_heating_kwh(t_indoor, t_outdoor, ghi, cop).iloc[0]
        assert with_gains < no_gains

    def test_rising_indoor_temp_increases_q_heat_falling_decreases(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_eff=150.0, solar_gain_area=0.0, Q_base_el=0.0)
        pm._tau_hours = 8.0
        ts = pd.date_range("2026-01-15 00:00", periods=3, freq="1h")
        t_outdoor = pd.Series([10.0] * 3, index=ts)
        ghi = pd.Series([0.0] * 3, index=ts)
        cop = pd.Series([3.0] * 3, index=ts)

        rising = pd.Series([19.0, 20.0, 21.0], index=ts)
        falling = pd.Series([21.0, 20.0, 19.0], index=ts)
        q_rising = pm._space_heating_kwh(rising, t_outdoor, ghi, cop)
        q_falling = pm._space_heating_kwh(falling, t_outdoor, ghi, cop)
        assert q_rising.iloc[0] > q_falling.iloc[0]

    def test_ua_eff_none_skips_heating_component(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib["UA_eff"] = None
        ts = pd.date_range("2026-01-15 00:00", periods=2, freq="1h")
        t_indoor = pd.Series([20.0, 20.0], index=ts)
        t_outdoor = pd.Series([10.0, 10.0], index=ts)
        ghi = pd.Series([0.0, 0.0], index=ts)
        cop = pd.Series([3.0, 3.0], index=ts)
        q_heat_el = pm._space_heating_kwh(t_indoor, t_outdoor, ghi, cop)
        assert (q_heat_el == 0.0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestSpaceHeating -v`
Expected: FAIL with `AttributeError: 'ThermalPhysicsModel' object has no attribute '_space_heating_kwh'`

- [ ] **Step 3: Write minimal implementation**

Add to `ThermalPhysicsModel`:

```python
    def _space_heating_kwh(
        self,
        t_indoor: pd.Series,
        t_outdoor: pd.Series,
        ghi: pd.Series,
        cop: pd.Series,
    ) -> pd.Series:
        ua = self._calib.get("UA_eff")
        if ua is None:
            return pd.Series(0.0, index=t_indoor.index)

        solar_area = self._calib.get("solar_gain_area") or 0.0
        q_base_el = self._calib.get("Q_base_el") or 0.0
        gains_fraction = self._config["internal_gains_fraction"]

        q_loss = ua * (t_indoor - t_outdoor).clip(lower=0.0)
        q_solar = solar_area * ghi.fillna(0.0)
        q_gain_int = q_base_el * gains_fraction * 1000.0

        tau = self._tau_hours or 8.0
        c_building = self._config.get("c_building_wh_k") or (ua * tau)
        t_indoor_next = t_indoor.shift(-1).bfill()
        q_mass = c_building * (t_indoor_next - t_indoor)

        q_heat = (q_loss - q_solar - q_gain_int + q_mass).clip(lower=0.0)
        q_heat_el = q_heat / cop.clip(lower=COP_MIN) / 1000.0
        return q_heat_el
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 19 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add space heating physics component with thermal mass term"
```

---

### Task 4: DHW tank ODE

**Files:**
- Modify: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Consumes: `WATER_SPECIFIC_HEAT_WH_PER_L_K`, `_DEFAULT_DRAW_PROFILE`, `DEFAULT_AMBIENT_C` from Task 1; `_cop_formula_value` (DHW uses a separate `COP_dhw` — spec doesn't distinguish it from space-heating COP explicitly beyond "COP_dhw"; this plan reuses `_cop_formula_value` at the DHW tank's target flow temp, since no separate DHW COP calibration path exists in §4).
- Produces: `ThermalPhysicsModel._dhw_kwh_series(timestamps, t_ambient, initial_t_tank, dhw_schedule_override) -> tuple[pd.Series, float]` (returns hourly Q_dhw_el kWh/h series and the final T_tank for state continuity).

- [ ] **Step 1: Write the failing test**

```python
class TestDHWOde:
    def test_cycle_triggers_at_lower_stops_at_upper(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm._schedule.update(T_dhw_lower=45.0, T_dhw_upper=55.0, T_legionella=60.0)
        ts = pd.date_range("2026-01-15 00:00", periods=24, freq="1h")
        t_ambient = pd.Series([20.0] * 24, index=ts)
        q_dhw_el, final_temp = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=44.0, dhw_schedule_override=None)
        assert (q_dhw_el >= 0.0).all()
        assert 45.0 <= final_temp <= 60.0  # clamp bounds enforced

    def test_heating_rise_derived_not_constant(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        c_dhw = pm._config["dhw_tank_volume_l"] * WATER_SPECIFIC_HEAT_WH_PER_L_K
        expected_rise = pm._config["dhw_power_w"] / c_dhw
        ts = pd.date_range("2026-01-15 00:00", periods=2, freq="1h")
        t_ambient = pd.Series([20.0, 20.0], index=ts)
        # start just below T_lower to force a reheat on hour 0
        q_dhw_el, _ = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=44.0, dhw_schedule_override=None)
        assert q_dhw_el.iloc[0] > 0.0
        # different tank volume -> different heating_rise -> different resulting series (not hardcoded)
        pm2 = ThermalPhysicsModel(tmp_path / "models2", {**DEFAULT_CONFIG, "dhw_tank_volume_l": 300})
        pm2._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        q_dhw_el2, _ = pm2._dhw_kwh_series(ts, t_ambient, initial_t_tank=44.0, dhw_schedule_override=None)
        assert not q_dhw_el.equals(q_dhw_el2)

    def test_post_legionella_silence(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm._schedule.update(T_dhw_lower=45.0, T_dhw_upper=55.0, T_legionella=60.0)
        ts = pd.date_range("2026-01-15 00:00", periods=12, freq="1h")
        t_ambient = pd.Series([20.0] * 12, index=ts)
        q_dhw_el, _ = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=60.0, dhw_schedule_override=None)
        # tank starts at legionella temp -> several hours of zero electricity before it cools to T_lower
        assert q_dhw_el.iloc[0] == 0.0
        assert q_dhw_el.iloc[1] == 0.0

    def test_dhw_schedule_override_shifts_electricity_to_specified_hour(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm._schedule.update(T_dhw_lower=45.0, T_dhw_upper=55.0, T_legionella=60.0, legionella_dow=2, legionella_hour=14)
        ts = pd.date_range("2026-06-24 00:00", periods=48, freq="1h")  # Wed 2026-06-24 is dow=2
        t_ambient = pd.Series([20.0] * 48, index=ts)
        override = {"legionella": ("2026-06-25", 10)}  # move to Thursday 10:00
        q_dhw_el, _ = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=50.0, dhw_schedule_override=override)
        thu_10 = pd.Timestamp("2026-06-25 10:00")
        # a legionella boost (heating to T_legionella) must occur at/after the overridden hour
        assert q_dhw_el.loc[q_dhw_el.index >= thu_10].max() > 0

    def test_ode_edge_case_zero_delta_t(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        ts = pd.date_range("2026-01-15 00:00", periods=2, freq="1h")
        t_ambient = pd.Series([50.0, 50.0], index=ts)  # T_ambient == T_tank -> no insulation loss
        q_dhw_el, final_temp = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=50.0, dhw_schedule_override=None)
        assert np.isfinite(final_temp)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestDHWOde -v`
Expected: FAIL with `AttributeError: 'ThermalPhysicsModel' object has no attribute '_dhw_kwh_series'`

- [ ] **Step 3: Write minimal implementation**

Add to `ThermalPhysicsModel`:

```python
    def _dhw_override_for_hour(self, ts: pd.Timestamp, override: dict | None) -> float | None:
        """Return an override target temp (T_legionella) for *ts* if a legionella override applies, else None."""
        if not override or "legionella" not in override:
            return None
        date_str, hour = override["legionella"]
        target = pd.Timestamp(f"{date_str} {hour:02d}:00")
        if ts == target:
            return self._schedule["T_legionella"]
        return None

    def _dhw_kwh_series(
        self,
        timestamps: pd.DatetimeIndex,
        t_ambient: pd.Series,
        initial_t_tank: float,
        dhw_schedule_override: dict | None,
    ) -> tuple[pd.Series, float]:
        volume_l = self._config["dhw_tank_volume_l"]
        c_dhw = volume_l * WATER_SPECIFIC_HEAT_WH_PER_L_K
        q_dhw_power = self._config["dhw_power_w"]
        heating_rise = q_dhw_power / c_dhw  # K/h, derived each call — not a stored constant

        t_lower = self._schedule["T_dhw_lower"]
        t_legionella = self._schedule["T_legionella"]

        q_dhw_daily = self._calib.get("Q_dhw_daily") or 0.0
        draw_rate = (q_dhw_daily * 1000.0 / c_dhw) if c_dhw > 0 else 0.0  # K-equivalent/day, scaled by shape below

        cop_dhw = max(COP_MIN, self._cop_formula_value(t_ambient.iloc[0] if len(t_ambient) else 10.0, None))

        t_tank = float(initial_t_tank)
        el_kwh = np.zeros(len(timestamps))
        for i, ts in enumerate(timestamps):
            ua_dhw = self._calib.get("UA_dhw") or 15.0
            dT = -ua_dhw * (t_tank - float(t_ambient.iloc[i])) / c_dhw
            hour_of_day = ts.hour
            dT -= _DEFAULT_DRAW_PROFILE[hour_of_day] * draw_rate

            override_target = self._dhw_override_for_hour(ts, dhw_schedule_override)
            if override_target is not None:
                q_el_w = q_dhw_power / cop_dhw
                el_kwh[i] = q_el_w / 1000.0
                t_tank = override_target
                continue

            if t_tank < t_lower:
                q_el_w = q_dhw_power / cop_dhw
                el_kwh[i] = q_el_w / 1000.0
                t_tank = float(np.clip(t_tank + dT + heating_rise, t_lower, t_legionella))
            else:
                el_kwh[i] = 0.0
                t_tank = float(np.clip(t_tank + dT, t_lower, t_legionella))

        return pd.Series(el_kwh, index=timestamps), t_tank
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 24 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add DHW tank ODE with legionella schedule override"
```

---

### Task 5: Base electrical + predict_series() / predict_training_series() assembly + fallback hierarchy

**Files:**
- Modify: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Consumes: `_space_heating_kwh`, `_cop_series`, `_dhw_kwh_series` from Tasks 2-4.
- Produces: `ThermalPhysicsModel.predict_series(...)`, `ThermalPhysicsModel.predict_training_series(...)` — exact signatures from the index doc's interface contract.

- [ ] **Step 1: Write the failing test**

```python
class TestPredictSeries:
    def test_predict_series_returns_series_aligned_to_forecast_df(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_eff=150.0, solar_gain_area=5.0, Q_base_el=0.35, UA_dhw=15.0, Q_dhw_daily=3.5)
        ts = pd.date_range("2026-01-15 00:00", periods=48, freq="1h")
        forecast_df = pd.DataFrame(
            {"timestamp": ts, "temp_c": np.linspace(-2, 8, 48), "direct_radiation_wm2": np.zeros(48)}
        )
        result = pm.predict_series(forecast_df)
        assert isinstance(result, pd.Series)
        assert len(result) == 48
        assert (result >= 0).all()

    def test_predict_series_no_calibration_returns_zeros_no_exception(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)  # fresh, UA_eff=None, Q_base_el=0.35 default
        ts = pd.date_range("2026-01-15 00:00", periods=48, freq="1h")
        forecast_df = pd.DataFrame(
            {"timestamp": ts, "temp_c": np.linspace(-2, 8, 48), "direct_radiation_wm2": np.zeros(48)}
        )
        result = pm.predict_series(forecast_df)
        assert len(result) == 48
        assert (result >= 0).all()  # Q_base_el default still contributes; no crash

    def test_predict_series_missing_ghi_column_solar_zero(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_eff=150.0, solar_gain_area=5.0)
        ts = pd.date_range("2026-01-15 00:00", periods=4, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 4})  # no direct_radiation_wm2
        result = pm.predict_series(forecast_df)
        assert len(result) == 4  # no KeyError

    def test_predict_training_series_uses_actual_climate_readings(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35, UA_dhw=15.0, Q_dhw_daily=3.5)
        ts = pd.date_range("2026-01-15 00:00", periods=10, freq="1h")
        energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 10})
        weather_df = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 10, "direct_radiation_wm2": [0.0] * 10})
        climate_dfs = {"climate.living_room": pd.DataFrame({"timestamp": ts, "current_temp": [20.0] * 10})}
        result = pm.predict_training_series(energy_df, weather_df, climate_dfs=climate_dfs)
        assert len(result) == 10
        assert list(result.index) == list(ts)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestPredictSeries -v`
Expected: FAIL with `AttributeError: 'ThermalPhysicsModel' object has no attribute 'predict_series'`

- [ ] **Step 3: Write minimal implementation**

Add to `ThermalPhysicsModel`:

```python
    def _area_weighted_t_indoor(
        self, climate_data: dict[str, pd.DataFrame], timestamps: pd.DatetimeIndex, room_areas: dict[str, float] | None
    ) -> pd.Series | None:
        if not climate_data:
            return None
        parts, weights = [], []
        for eid, cdf in climate_data.items():
            if cdf.empty:
                continue
            c = cdf.set_index("timestamp")["current_temp"].reindex(timestamps, method="nearest")
            parts.append(c)
            weights.append((room_areas or {}).get(eid, 20.0))
        if not parts:
            return None
        stacked = pd.concat(parts, axis=1)
        w = np.array(weights)
        return pd.Series(np.average(stacked.values, axis=1, weights=w), index=timestamps)

    def predict_series(
        self,
        forecast_df: pd.DataFrame,
        climate_recent: dict[str, pd.DataFrame] | None = None,
        dhw_recent: pd.DataFrame | None = None,
        room_areas: dict[str, float] | None = None,
        heating_active_series: pd.Series | None = None,
        setpoint_on: float | None = None,
        setpoint_off: float | None = None,
        dhw_schedule_override: dict | None = None,
    ) -> pd.Series:
        from .model import _project_indoor_temps  # local import avoids a circular import at module load time

        timestamps = pd.DatetimeIndex(forecast_df["timestamp"])
        t_outdoor = pd.Series(forecast_df["temp_c"].values, index=timestamps)
        ghi = (
            pd.Series(forecast_df["direct_radiation_wm2"].values, index=timestamps)
            if "direct_radiation_wm2" in forecast_df.columns
            else pd.Series(0.0, index=timestamps)
        )

        try:
            if climate_recent:
                projected = _project_indoor_temps(
                    climate_recent,
                    timestamps,
                    t_outdoor,
                    tau_hours=self._tau_hours,
                    heating_active_series=heating_active_series,
                    setpoint_on=setpoint_on,
                    setpoint_off=setpoint_off,
                )
                t_indoor = self._area_weighted_t_indoor(projected, timestamps, room_areas)
            else:
                t_indoor = None

            if t_indoor is None:
                t_indoor = pd.Series(setpoint_on or DEFAULT_AMBIENT_C, index=timestamps)

            cop = self._cop_series(timestamps, t_outdoor, cop_sensor_series=None)
            q_heat_el = self._space_heating_kwh(t_indoor, t_outdoor, ghi, cop)

            if dhw_recent is not None and not dhw_recent.empty:
                latest = dhw_recent.sort_values("timestamp").iloc[-1]
                age = timestamps[0] - pd.Timestamp(latest["timestamp"])
                initial_t_tank = (
                    float(latest["buffer_temp"])
                    if age <= pd.Timedelta(hours=2)
                    else (self._schedule["T_dhw_upper"] + self._schedule["T_dhw_lower"]) / 2
                )
            else:
                initial_t_tank = (self._schedule["T_dhw_upper"] + self._schedule["T_dhw_lower"]) / 2

            q_dhw_el, _ = self._dhw_kwh_series(timestamps, t_indoor, initial_t_tank, dhw_schedule_override)

            q_base_el = self._calib.get("Q_base_el") or 0.35
            physics_kwh = q_heat_el + q_dhw_el + q_base_el
            return physics_kwh.clip(lower=0.0)
        except Exception as e:
            _LOGGER.warning(f"physics predict_series failed: {e} — returning zeros")
            return pd.Series(0.0, index=timestamps)

    def predict_training_series(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        climate_dfs: dict[str, pd.DataFrame] | None = None,
        dhw_df: pd.DataFrame | None = None,
        room_areas: dict[str, float] | None = None,
    ) -> pd.Series:
        timestamps = pd.DatetimeIndex(pd.to_datetime(energy_df["timestamp"]))
        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        t_outdoor = w["temp_c"].reindex(timestamps, method="nearest")
        ghi = (
            w["direct_radiation_wm2"].reindex(timestamps, method="nearest")
            if "direct_radiation_wm2" in w.columns
            else pd.Series(0.0, index=timestamps)
        )

        t_indoor = self._area_weighted_t_indoor(climate_dfs or {}, timestamps, room_areas)
        if t_indoor is None:
            t_indoor = pd.Series(DEFAULT_AMBIENT_C, index=timestamps)

        cop = self._cop_series(timestamps, t_outdoor, cop_sensor_series=None)
        q_heat_el = self._space_heating_kwh(t_indoor, t_outdoor, ghi, cop)

        if dhw_df is not None and not dhw_df.empty:
            d = dhw_df.set_index(pd.to_datetime(dhw_df["timestamp"]))["buffer_temp"].reindex(
                timestamps, method="nearest"
            )
            initial_t_tank = float(d.iloc[0]) if not d.empty and pd.notna(d.iloc[0]) else self._schedule["T_dhw_upper"]
        else:
            initial_t_tank = (self._schedule["T_dhw_upper"] + self._schedule["T_dhw_lower"]) / 2

        q_dhw_el, _ = self._dhw_kwh_series(timestamps, t_indoor, initial_t_tank, dhw_schedule_override=None)
        q_base_el = self._calib.get("Q_base_el") or 0.35
        return (q_heat_el + q_dhw_el + q_base_el).clip(lower=0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 29 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: assemble ThermalPhysicsModel.predict_series/predict_training_series"
```

---

### Task 6: `_find_passive_windows()` shared utility + `_calibrate_tau()` refactor

**Files:**
- Modify: `apps/energy_forecast/model.py` (add module-level function above `_project_indoor_temps` at line 2342; refactor `_calibrate_tau` at lines 1396-1623 to use it)
- Test: `tests/test_model.py` (regression), `tests/test_physics.py` (new unit)

**Interfaces:**
- Produces: module-level `_find_passive_windows(df, *, min_delta_t=8.0, min_hp_off_hours=2) -> pd.Index` in `model.py`, importable as `from energy_forecast.model import _find_passive_windows`.

This is the one task in Plan A that touches `model.py`. It is scoped tightly: extract the off-block/ΔT filter from `_calibrate_tau()` (lines 1470-1489 today) into a standalone function, then make `_calibrate_tau()` call it as a pre-filter before its existing per-block OLS/scoring logic. `_calibrate_tau()`'s output must be unchanged for existing test fixtures — this is a refactor, not a behavior change, and is covered by the existing `tests/test_model.py` τ-calibration tests as a regression gate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_physics.py
from energy_forecast.model import _find_passive_windows


class TestFindPassiveWindows:
    def test_excludes_hp_on_rows(self):
        ts = pd.date_range("2026-01-15 00:00", periods=6, freq="1h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "T_outdoor": [0.0] * 6,
                "T_indoor": [20.0] * 6,
                "hp_running": [False, False, True, False, False, False],
                "dhw_tank_temp": [np.nan] * 6,
            }
        )
        idx = _find_passive_windows(df, min_delta_t=8.0, min_hp_off_hours=2)
        assert 2 not in idx  # hp_running row excluded

    def test_excludes_delta_t_below_threshold(self):
        ts = pd.date_range("2026-01-15 00:00", periods=3, freq="1h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "T_outdoor": [18.0, 18.0, 18.0],  # ΔT = 2K, below 8K threshold
                "T_indoor": [20.0, 20.0, 20.0],
                "hp_running": [False, False, False],
                "dhw_tank_temp": [np.nan] * 3,
            }
        )
        idx = _find_passive_windows(df, min_delta_t=8.0, min_hp_off_hours=2)
        assert len(idx) == 0

    def test_excludes_rising_dhw_tank_temp_hours(self):
        ts = pd.date_range("2026-01-15 00:00", periods=3, freq="1h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "T_outdoor": [0.0, 0.0, 0.0],
                "T_indoor": [20.0, 20.0, 20.0],
                "hp_running": [False, False, False],
                "dhw_tank_temp": [45.0, 50.0, 50.0],  # rising 45->50 at row 1 = active DHW cycle
            }
        )
        idx = _find_passive_windows(df, min_delta_t=8.0, min_hp_off_hours=2)
        assert 1 not in idx

    def test_requires_min_consecutive_off_hours(self):
        ts = pd.date_range("2026-01-15 00:00", periods=4, freq="1h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "T_outdoor": [0.0] * 4,
                "T_indoor": [20.0] * 4,
                "hp_running": [True, False, True, False],  # never 2 consecutive off hours
                "dhw_tank_temp": [np.nan] * 4,
            }
        )
        idx = _find_passive_windows(df, min_delta_t=8.0, min_hp_off_hours=2)
        assert len(idx) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestFindPassiveWindows -v`
Expected: FAIL with `ImportError: cannot import name '_find_passive_windows'`

- [ ] **Step 3: Write minimal implementation**

Insert into `model.py` immediately before `_project_indoor_temps` (line 2342):

```python
def _find_passive_windows(
    df: pd.DataFrame,
    *,
    min_delta_t: float = 8.0,
    min_hp_off_hours: int = 2,
) -> pd.Index:
    """Return the index of rows suitable for passive-cooling calibration (τ, UA_eff).

    A row qualifies when: the heat pump has been off for at least
    ``min_hp_off_hours`` consecutive hours ending at that row, ΔT = T_indoor −
    T_outdoor ≥ ``min_delta_t``, and ``dhw_tank_temp`` is not rising into that
    row (rising tank temp indicates an active DHW cycle, which would inflate
    UA_eff / shorten apparent τ if included).
    """
    d = df.sort_values("timestamp").reset_index(drop=True)

    off = (~d["hp_running"].astype(bool)).astype(int)
    off_run_length = off.groupby((off != off.shift()).cumsum()).cumcount() + 1
    off_run_length = off_run_length.where(off == 1, 0)
    enough_off = off_run_length >= min_hp_off_hours

    delta_t = d["T_indoor"] - d["T_outdoor"]
    enough_delta = delta_t >= min_delta_t

    dhw_rising = d["dhw_tank_temp"].diff() > 0
    not_dhw_active = ~dhw_rising.fillna(False)

    mask = enough_off & enough_delta & not_dhw_active
    return d.index[mask]
```

Now refactor `_calibrate_tau()` to pre-filter with it. In the existing method (around line 1465, right after `combined = pd.DataFrame(...)`), add:

```python
        combined_reset = combined.reset_index().rename(columns={"index": "timestamp"})
        passive_df = pd.DataFrame(
            {
                "timestamp": combined_reset["timestamp"],
                "T_outdoor": combined_reset["T_outdoor"],
                "T_indoor": combined_reset["T_indoor"],
                "hp_running": combined_reset["heating_active"].astype(bool),
                "dhw_tank_temp": np.nan,  # τ calibration has no DHW filter input today — unchanged behaviour
            }
        )
        passive_idx = _find_passive_windows(passive_df, min_delta_t=0.0, min_hp_off_hours=1)
        # min_delta_t=0.0/min_hp_off_hours=1 here preserve τ's existing (looser) block semantics —
        # τ's own per-block ΔT>0 and declining-delta filters below are unchanged and still authoritative.
        combined = combined.iloc[passive_idx].copy() if len(passive_idx) else combined.iloc[0:0]
```

This makes `_calibrate_tau` route its off-block rows through the shared utility without changing its existing `min_delta_t`/decline semantics (τ's own scoring logic downstream is untouched). Run the full existing τ test suite to confirm no behavior drift.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py::TestFindPassiveWindows tests/test_model.py -v -k "tau or Tau or calibrate"`
Expected: all pass, including pre-existing `_calibrate_tau` tests unchanged

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_physics.py
git commit -m "refactor: extract _find_passive_windows() as shared calibration utility"
```

---

### Task 7: Calibration — `Q_base_el` and `Q_dhw_daily`

**Files:**
- Modify: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ThermalPhysicsModel._calibrate_base_load(energy_df, away_df, ev_df, holdout_cutoff) -> float | None`, `ThermalPhysicsModel._calibrate_dhw_daily(energy_df, q_base_el, holdout_cutoff) -> float | None`.

- [ ] **Step 1: Write the failing test**

```python
class TestCalibrateBaseLoad:
    def test_recovers_median_within_5_percent(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        rng = np.random.default_rng(0)
        # 30 summer nights, 01-05h, no EV/away, median load 0.4 kWh/h + noise
        rows = []
        for day in pd.date_range("2026-06-01", periods=30, freq="1D"):
            for hour in range(1, 5):
                rows.append({"timestamp": day + pd.Timedelta(hours=hour), "gross_kwh": 0.4 + rng.normal(0, 0.02)})
        energy_df = pd.DataFrame(rows)
        result = pm._calibrate_base_load(energy_df, away_df=None, ev_df=None, holdout_cutoff=pd.Timestamp("2026-07-01"))
        assert result == pytest.approx(0.4, rel=0.05)

    def test_insufficient_data_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        energy_df = pd.DataFrame(
            {"timestamp": pd.date_range("2026-06-01 01:00", periods=5, freq="1h"), "gross_kwh": [0.4] * 5}
        )
        result = pm._calibrate_base_load(energy_df, away_df=None, ev_df=None, holdout_cutoff=pd.Timestamp("2026-07-01"))
        assert result is None

    def test_excludes_post_holdout_rows(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        rows = []
        for day in pd.date_range("2026-06-01", periods=14, freq="1D"):
            for hour in range(1, 5):
                rows.append({"timestamp": day + pd.Timedelta(hours=hour), "gross_kwh": 0.4})
        for day in pd.date_range("2026-07-15", periods=14, freq="1D"):  # post-holdout, wildly different value
            for hour in range(1, 5):
                rows.append({"timestamp": day + pd.Timedelta(hours=hour), "gross_kwh": 99.0})
        energy_df = pd.DataFrame(rows)
        result = pm._calibrate_base_load(energy_df, away_df=None, ev_df=None, holdout_cutoff=pd.Timestamp("2026-07-01"))
        assert result == pytest.approx(0.4, abs=0.01)


class TestCalibrateDhwDaily:
    def test_recovers_from_summer_daily_mean(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        ts = pd.date_range("2026-06-01", periods=14 * 24, freq="1h")
        # base load 0.4 kWh/h all day + 3.5 kWh spread over one DHW hour/day
        vals = np.full(len(ts), 0.4)
        vals[3::24] += 3.5  # hour 3 each day gets the DHW reheat
        energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": vals})
        result = pm._calibrate_dhw_daily(energy_df, q_base_el=0.4, holdout_cutoff=pd.Timestamp("2026-07-01"))
        assert result == pytest.approx(3.5, rel=0.05)

    def test_insufficient_data_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        energy_df = pd.DataFrame(
            {"timestamp": pd.date_range("2026-06-01", periods=10, freq="1h"), "gross_kwh": [0.4] * 10}
        )
        result = pm._calibrate_dhw_daily(energy_df, q_base_el=0.4, holdout_cutoff=pd.Timestamp("2026-07-01"))
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestCalibrateBaseLoad tests/test_physics.py::TestCalibrateDhwDaily -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

Add to `ThermalPhysicsModel`:

```python
    def _calibrate_base_load(
        self, energy_df: pd.DataFrame, away_df: pd.DataFrame | None, ev_df: pd.DataFrame | None, holdout_cutoff: pd.Timestamp
    ) -> float | None:
        df = energy_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"] < holdout_cutoff]
        df = df[(df["timestamp"].dt.hour >= 1) & (df["timestamp"].dt.hour < 5)]
        df = df[df["timestamp"].dt.month.isin([6, 7, 8])]

        if away_df is not None and not away_df.empty:
            away_ts = set(pd.to_datetime(away_df.loc[away_df["is_away"] > 0, "timestamp"]).dt.floor("1h"))
            df = df[~df["timestamp"].dt.floor("1h").isin(away_ts)]
        if ev_df is not None and not ev_df.empty:
            ev_ts = set(pd.to_datetime(ev_df["timestamp"]).dt.floor("1h"))
            df = df[~df["timestamp"].dt.floor("1h").isin(ev_ts)]

        n_nights = df["timestamp"].dt.date.nunique()
        if n_nights < 14 or len(df) < 14:
            _LOGGER.warning(f"Q_base_el calibration: only {n_nights} summer nights available (need 14) — skipping")
            return None
        return float(df["gross_kwh"].median())

    def _calibrate_dhw_daily(
        self, energy_df: pd.DataFrame, q_base_el: float, holdout_cutoff: pd.Timestamp
    ) -> float | None:
        df = energy_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"] < holdout_cutoff]
        df = df[df["timestamp"].dt.month.isin([6, 7, 8])]
        if df.empty:
            _LOGGER.warning("Q_dhw_daily calibration: no summer data available — skipping")
            return None

        daily = df.set_index("timestamp")["gross_kwh"].resample("1D").sum().dropna()
        if len(daily) < 14:
            _LOGGER.warning(f"Q_dhw_daily calibration: only {len(daily)} summer days available (need 14) — skipping")
            return None

        result = float(daily.mean()) - 24 * q_base_el
        return max(0.0, result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 36 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add Q_base_el and Q_dhw_daily calibration"
```

---

### Task 8: Calibration — `UA_eff` (cold-start gate, R² gate, DHW filtering)

**Files:**
- Modify: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Consumes: `_find_passive_windows()` from Task 6 (`from .model import _find_passive_windows` — local import to avoid a circular import at module load, matching the pattern used in Task 5's `predict_series`).
- Produces: `ThermalPhysicsModel._calibrate_ua_eff(energy_df, weather_df, climate_dfs, dhw_df, holdout_cutoff) -> tuple[float | None, int]` (UA_eff, n_windows_used).

- [ ] **Step 1: Write the failing test**

```python
class TestCalibrateUAEff:
    def _synthetic_winter_data(self, ua_eff_true=150.0, cop=3.0, n_nights=35, with_dhw_sensor=True):
        rows = []
        rng = np.random.default_rng(1)
        for day in pd.date_range("2025-11-01", periods=n_nights, freq="1D"):
            for hour in range(22, 24 - 24 % 24 + 6):  # crude 22-06 wraparound handled by modulo below
                pass
        # build explicit 22:00-06:00 hourly rows per night
        energy_rows, weather_rows, climate_rows, dhw_rows = [], [], [], []
        t_indoor = 20.0
        for night in range(n_nights):
            base_day = pd.Timestamp("2025-11-01") + pd.Timedelta(days=night)
            for h in list(range(22, 24)) + list(range(0, 6)):
                ts = base_day + pd.Timedelta(hours=h) if h >= 22 else base_day + pd.Timedelta(days=1, hours=h)
                t_outdoor = 0.0 + rng.normal(0, 0.3)
                delta_t = t_indoor - t_outdoor
                q_heat_w = ua_eff_true * delta_t
                q_heat_el = q_heat_w / cop / 1000.0
                energy_rows.append({"timestamp": ts, "gross_kwh": q_heat_el + 0.35})
                weather_rows.append({"timestamp": ts, "temp_c": t_outdoor, "direct_radiation_wm2": 0.0})
                climate_rows.append({"timestamp": ts, "current_temp": t_indoor})
                dhw_rows.append({"timestamp": ts, "buffer_temp": 50.0 if with_dhw_sensor else np.nan})
        return (
            pd.DataFrame(energy_rows),
            pd.DataFrame(weather_rows),
            {"climate.living_room": pd.DataFrame(climate_rows)},
            pd.DataFrame(dhw_rows) if with_dhw_sensor else pd.DataFrame(columns=["timestamp", "buffer_temp"]),
        )

    def test_recovers_ua_eff_within_20_percent(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        energy_df, weather_df, climate_dfs, dhw_df = self._synthetic_winter_data(ua_eff_true=150.0, n_nights=35)
        ua_eff, n_windows = pm._calibrate_ua_eff(
            energy_df, weather_df, climate_dfs, dhw_df, holdout_cutoff=pd.Timestamp("2026-06-01")
        )
        assert ua_eff is not None
        assert ua_eff == pytest.approx(150.0, rel=0.20)
        assert n_windows >= 30

    def test_fewer_than_30_windows_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        energy_df, weather_df, climate_dfs, dhw_df = self._synthetic_winter_data(n_nights=10)
        ua_eff, n_windows = pm._calibrate_ua_eff(
            energy_df, weather_df, climate_dfs, dhw_df, holdout_cutoff=pd.Timestamp("2026-06-01")
        )
        assert ua_eff is None

    def test_excludes_post_holdout_rows(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        energy_df, weather_df, climate_dfs, dhw_df = self._synthetic_winter_data(ua_eff_true=150.0, n_nights=35)
        # holdout_cutoff before all data -> nothing left
        ua_eff, n_windows = pm._calibrate_ua_eff(
            energy_df, weather_df, climate_dfs, dhw_df, holdout_cutoff=pd.Timestamp("2025-11-01")
        )
        assert ua_eff is None
        assert n_windows == 0

    def test_dhw_sensor_absent_raises_min_delta_t_to_12(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        energy_df, weather_df, climate_dfs, _ = self._synthetic_winter_data(
            ua_eff_true=150.0, n_nights=35, with_dhw_sensor=False
        )
        empty_dhw = pd.DataFrame(columns=["timestamp", "buffer_temp"])
        ua_eff, n_windows = pm._calibrate_ua_eff(
            energy_df, weather_df, climate_dfs, empty_dhw, holdout_cutoff=pd.Timestamp("2026-06-01")
        )
        # ΔT = 20K in this synthetic data, comfortably above the raised 12K bar
        assert ua_eff is not None

    def test_low_r_squared_discards_result(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        rng = np.random.default_rng(2)
        energy_df, weather_df, climate_dfs, dhw_df = self._synthetic_winter_data(ua_eff_true=150.0, n_nights=35)
        # destroy the relationship with pure noise
        energy_df["gross_kwh"] = rng.uniform(0, 10, size=len(energy_df))
        ua_eff, n_windows = pm._calibrate_ua_eff(
            energy_df, weather_df, climate_dfs, dhw_df, holdout_cutoff=pd.Timestamp("2026-06-01")
        )
        assert ua_eff is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestCalibrateUAEff -v`
Expected: FAIL with `AttributeError: 'ThermalPhysicsModel' object has no attribute '_calibrate_ua_eff'`

- [ ] **Step 3: Write minimal implementation**

Add to `ThermalPhysicsModel`:

```python
    def _calibrate_ua_eff(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        climate_dfs: dict[str, pd.DataFrame] | None,
        dhw_df: pd.DataFrame | None,
        holdout_cutoff: pd.Timestamp,
    ) -> tuple[float | None, int]:
        from .model import _find_passive_windows

        if not climate_dfs:
            return None, 0

        e = energy_df.copy()
        e["timestamp"] = pd.to_datetime(e["timestamp"])
        e = e[e["timestamp"] < holdout_cutoff]
        e = e[(e["timestamp"].dt.hour >= 22) | (e["timestamp"].dt.hour < 6)]
        e = e[e["timestamp"].dt.month.isin([11, 12, 1, 2, 3])]
        if e.empty:
            return None, 0

        t_indoor = self._area_weighted_t_indoor(climate_dfs, pd.DatetimeIndex(e["timestamp"]), room_areas=None)
        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        t_outdoor = w["temp_c"].reindex(e["timestamp"], method="nearest").values

        has_dhw_sensor = dhw_df is not None and not dhw_df.empty
        if has_dhw_sensor:
            d = dhw_df.set_index(pd.to_datetime(dhw_df["timestamp"]))["buffer_temp"].reindex(
                e["timestamp"], method="nearest"
            )
            dhw_tank_temp = d.values
            min_delta_t = 8.0
        else:
            dhw_tank_temp = np.full(len(e), np.nan)
            min_delta_t = 12.0
            _LOGGER.warning("DHW tank sensor absent — UA_eff calibration may be inflated")

        passive_df = pd.DataFrame(
            {
                "timestamp": e["timestamp"].values,
                "T_outdoor": t_outdoor,
                "T_indoor": t_indoor.values if t_indoor is not None else np.nan,
                "hp_running": False,  # already filtered to heating-off... actually this is nighttime window,
                # not HP-off; UA_eff wants the observed heating demand itself, so hp_running=False here
                # would incorrectly zero out min_hp_off_hours=2 requirement. Set min_hp_off_hours=0 below.
                "dhw_tank_temp": dhw_tank_temp,
            }
        )
        passive_idx = _find_passive_windows(passive_df, min_delta_t=min_delta_t, min_hp_off_hours=0)
        n_windows = len(passive_idx)
        if n_windows < 30:
            return None, n_windows

        sub = e.iloc[passive_idx]
        sub_t_indoor = t_indoor.iloc[passive_idx].values
        sub_t_outdoor = t_outdoor[passive_idx]
        delta_t = sub_t_indoor - sub_t_outdoor

        cop = np.array([self._cop_formula_value(t, None) for t in sub_t_outdoor])
        q_heat_obs = sub["gross_kwh"].values  # gross consumption proxy for heating demand in this window

        x = (delta_t / cop).reshape(-1, 1)
        y = q_heat_obs
        # OLS through origin (Q_heat_obs ≈ UA_eff/1000 × ΔT/COP): slope = sum(xy)/sum(xx)
        slope = float(np.sum(x.flatten() * y) / np.sum(x.flatten() ** 2))
        y_hat = slope * x.flatten()
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        if r2 < 0.5:
            _LOGGER.warning(f"UA_eff calibration: R²={r2:.2f} < 0.5 — discarding result, using config default")
            return None, n_windows

        ua_eff = slope * 1000.0
        return ua_eff, n_windows
```

Note the inline comment on `hp_running=False` / `min_hp_off_hours=0`: UA_eff calibration filters on nighttime hours and ΔT/DHW-activity, not on a heat-pump-off state (unlike τ calibration, which specifically wants *passive cooling* windows). Passing `min_hp_off_hours=0` makes the shared utility's off-run-length check always pass, so only the ΔT and DHW-rising filters apply — this matches spec §4.2's UA_eff row filter exactly (nighttime, ΔT gate, DHW exclusion; no HP-off requirement).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 42 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add UA_eff calibration with cold-start gate and R2 discard"
```

---

### Task 9: Calibration — `solar_gain_area` and `UA_dhw`

**Files:**
- Modify: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Consumes: `_find_passive_windows()`, `_area_weighted_t_indoor()`.
- Produces: `ThermalPhysicsModel._calibrate_solar_gain_area(energy_df, weather_df, climate_dfs, ua_eff, holdout_cutoff) -> float | None`, `ThermalPhysicsModel._calibrate_ua_dhw(dhw_df, weather_df, heating_active_df, holdout_cutoff) -> float | None`.

- [ ] **Step 1: Write the failing test**

```python
class TestCalibrateSolarGainArea:
    def test_recovers_known_ghi_offset(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        rng = np.random.default_rng(3)
        rows_e, rows_w, rows_c = [], [], []
        true_area = 8.0
        ua_eff = 150.0
        cop = 3.0
        for day in pd.date_range("2025-12-01", periods=20, freq="1D"):
            for h in range(10, 15):  # daytime hours
                ts = day + pd.Timedelta(hours=h)
                t_outdoor = 2.0
                ghi = rng.uniform(60, 400)
                q_loss = ua_eff * (20.0 - t_outdoor)
                q_solar = true_area * ghi
                q_heat_el = max(0.0, q_loss - q_solar) / cop / 1000.0
                rows_e.append({"timestamp": ts, "gross_kwh": q_heat_el + 0.35})
                rows_w.append({"timestamp": ts, "temp_c": t_outdoor, "direct_radiation_wm2": ghi})
                rows_c.append({"timestamp": ts, "current_temp": 20.0})
        energy_df, weather_df = pd.DataFrame(rows_e), pd.DataFrame(rows_w)
        climate_dfs = {"climate.living_room": pd.DataFrame(rows_c)}
        result = pm._calibrate_solar_gain_area(
            energy_df, weather_df, climate_dfs, ua_eff=ua_eff, holdout_cutoff=pd.Timestamp("2026-06-01")
        )
        assert result == pytest.approx(true_area, rel=0.3)

    def test_insufficient_sunny_days_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        ts = pd.date_range("2025-12-01 12:00", periods=3, freq="1D")
        energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 3})
        weather_df = pd.DataFrame({"timestamp": ts, "temp_c": [2.0] * 3, "direct_radiation_wm2": [100.0] * 3})
        climate_dfs = {"climate.living_room": pd.DataFrame({"timestamp": ts, "current_temp": [20.0] * 3})}
        result = pm._calibrate_solar_gain_area(
            energy_df, weather_df, climate_dfs, ua_eff=150.0, holdout_cutoff=pd.Timestamp("2026-06-01")
        )
        assert result is None

    def test_ua_eff_none_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        result = pm._calibrate_solar_gain_area(
            pd.DataFrame({"timestamp": [], "gross_kwh": []}),
            pd.DataFrame({"timestamp": [], "temp_c": [], "direct_radiation_wm2": []}),
            {},
            ua_eff=None,
            holdout_cutoff=pd.Timestamp("2026-06-01"),
        )
        assert result is None


class TestCalibrateUADhw:
    def test_passive_decay_regression_recovers_ua_dhw(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        true_ua_dhw = 15.0
        c_dhw = 200 * WATER_SPECIFIC_HEAT_WH_PER_L_K
        ts = pd.date_range("2026-01-15 00:00", periods=12, freq="1h")
        t_ambient = 20.0
        t_tank = 55.0
        rows_dhw, rows_heat = [], []
        for t in ts:
            rows_dhw.append({"timestamp": t, "buffer_temp": t_tank})
            rows_heat.append({"timestamp": t, "heating_active": 0})
            dT = -true_ua_dhw * (t_tank - t_ambient) / c_dhw
            t_tank += dT
        dhw_df = pd.DataFrame(rows_dhw)
        heating_active_df = pd.DataFrame(rows_heat)
        weather_df = pd.DataFrame({"timestamp": ts, "temp_c": [t_ambient] * len(ts)})
        result = pm._calibrate_ua_dhw(dhw_df, weather_df, heating_active_df, holdout_cutoff=pd.Timestamp("2026-06-01"))
        assert result == pytest.approx(true_ua_dhw, rel=0.25)

    def test_no_dhw_sensor_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        result = pm._calibrate_ua_dhw(
            pd.DataFrame(columns=["timestamp", "buffer_temp"]),
            pd.DataFrame(columns=["timestamp", "temp_c"]),
            pd.DataFrame(columns=["timestamp", "heating_active"]),
            holdout_cutoff=pd.Timestamp("2026-06-01"),
        )
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestCalibrateSolarGainArea tests/test_physics.py::TestCalibrateUADhw -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

Add to `ThermalPhysicsModel`:

```python
    def _calibrate_solar_gain_area(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        climate_dfs: dict[str, pd.DataFrame] | None,
        ua_eff: float | None,
        holdout_cutoff: pd.Timestamp,
    ) -> float | None:
        if ua_eff is None or not climate_dfs:
            return None

        e = energy_df.copy()
        e["timestamp"] = pd.to_datetime(e["timestamp"])
        e = e[e["timestamp"] < holdout_cutoff]
        e = e[e["timestamp"].dt.month.isin([11, 12, 1, 2])]

        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        ghi = w["direct_radiation_wm2"].reindex(e["timestamp"], method="nearest")
        e = e[ghi.values > 50]
        if e.empty:
            return None

        n_days = e["timestamp"].dt.date.nunique()
        if n_days < 14:
            _LOGGER.warning(f"solar_gain_area calibration: only {n_days} sunny winter days (need 14) — skipping")
            return None

        t_indoor = self._area_weighted_t_indoor(climate_dfs, pd.DatetimeIndex(e["timestamp"]), room_areas=None)
        t_outdoor = w["temp_c"].reindex(e["timestamp"], method="nearest").values
        ghi_sub = w["direct_radiation_wm2"].reindex(e["timestamp"], method="nearest").values
        cop = np.array([self._cop_formula_value(t, None) for t in t_outdoor])

        q_loss_w = ua_eff * np.clip((t_indoor.values if t_indoor is not None else 20.0) - t_outdoor, 0, None)
        q_heat_el_observed = e["gross_kwh"].values
        # residual after removing UA_eff-predicted loss (in electrical kWh/h) attributed to solar offset
        residual_w = q_loss_w - q_heat_el_observed * cop * 1000.0
        # residual_w ≈ -solar_gain_area * ghi  (residual is negative where solar reduced demand)
        x = ghi_sub
        y = -residual_w
        if np.sum(x**2) < 1e-9:
            return None
        slope = float(np.sum(x * y) / np.sum(x**2))
        return max(0.0, slope)

    def _calibrate_ua_dhw(
        self,
        dhw_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        heating_active_df: pd.DataFrame,
        holdout_cutoff: pd.Timestamp,
    ) -> float | None:
        if dhw_df is None or dhw_df.empty:
            return None

        d = dhw_df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d[d["timestamp"] < holdout_cutoff].sort_values("timestamp").reset_index(drop=True)

        ha = heating_active_df.copy() if heating_active_df is not None else pd.DataFrame()
        if not ha.empty:
            ha["timestamp"] = pd.to_datetime(ha["timestamp"])
            hp_off = ha.set_index("timestamp")["heating_active"].reindex(d["timestamp"], method="nearest").fillna(0) == 0
        else:
            hp_off = pd.Series(True, index=d.index)

        d["hp_running"] = ~hp_off.values
        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        t_ambient = w["temp_c"].reindex(d["timestamp"], method="nearest").fillna(DEFAULT_AMBIENT_C).values

        # passive decay = tank cooling with no reheat and no rising trend (a draw would also make it fall,
        # so restrict to windows where the temp decline matches a smooth exponential, i.e. no big single-step drops)
        d["is_falling_smooth"] = d["buffer_temp"].diff().between(-1.0, 0.0)
        mask = (~d["hp_running"]) & d["is_falling_smooth"].fillna(False)
        sub = d[mask]
        if len(sub) < 6:
            return None

        delta_t = sub["buffer_temp"].values - t_ambient[sub.index]
        dT_dt = -sub["buffer_temp"].diff().fillna(0).values  # K lost per hour, positive
        c_dhw = self._config["dhw_tank_volume_l"] * WATER_SPECIFIC_HEAT_WH_PER_L_K
        # dT/dt = -UA_dhw * deltaT / C_dhw  ->  UA_dhw = (dT/dt * C_dhw) / deltaT
        valid = delta_t > 0.5
        if valid.sum() < 6:
            return None
        ua_dhw_samples = (dT_dt[valid] * c_dhw) / delta_t[valid]
        ua_dhw_samples = ua_dhw_samples[(ua_dhw_samples > 0) & (ua_dhw_samples < 200)]
        if len(ua_dhw_samples) < 3:
            return None
        return float(np.median(ua_dhw_samples))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 47 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add solar_gain_area and UA_dhw calibration"
```

---

### Task 10: DHW schedule inference (§4.3) + instability guard

**Files:**
- Modify: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Produces: `ThermalPhysicsModel._infer_dhw_schedule(dhw_df) -> dict | None` (returns `{T_dhw_upper, T_legionella, legionella_dow, legionella_hour, T_dhw_lower, dhw_tank_volume_l}` or `None` on insufficient data), `ThermalPhysicsModel._check_legionella_stability(new_dow, new_hour) -> bool` (returns `True` if stable, `False` and logs WARNING + suspends autonomous learning if the shift exceeds ±2h week-over-week).

- [ ] **Step 1: Write the failing test**

```python
class TestInferDhwSchedule:
    def test_four_synthetic_peaks_wednesday_1400(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        rows = []
        for week in range(4):
            base = pd.Timestamp("2026-01-07") + pd.Timedelta(weeks=week)  # a Wednesday
            for h in range(24):
                ts = base + pd.Timedelta(hours=h)
                if h == 14:
                    temp = 62.0  # legionella peak, well above normal upper (55)
                elif 10 <= h < 20:
                    temp = 50.0  # normal cycling upper-ish
                else:
                    temp = 45.0
                rows.append({"timestamp": ts, "buffer_temp": temp})
        dhw_df = pd.DataFrame(rows)
        result = pm._infer_dhw_schedule(dhw_df)
        assert result is not None
        assert result["legionella_dow"] == 2  # Wednesday
        assert result["legionella_hour"] == 14
        assert result["T_legionella"] > result["T_dhw_upper"]

    def test_insufficient_data_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        dhw_df = pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-01", periods=5, freq="1h"), "buffer_temp": [50.0] * 5}
        )
        result = pm._infer_dhw_schedule(dhw_df)
        assert result is None


class TestLegionellaStabilityGuard:
    def test_shift_within_2h_is_stable(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._schedule.update(legionella_dow=2, legionella_hour=14)
        assert pm._check_legionella_stability(new_dow=2, new_hour=15) is True

    def test_shift_beyond_2h_suspends_autonomous_learning(self, tmp_path, caplog):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._schedule.update(legionella_dow=2, legionella_hour=14)
        result = pm._check_legionella_stability(new_dow=2, new_hour=17)
        assert result is False

    def test_dow_change_treated_as_unstable(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._schedule.update(legionella_dow=2, legionella_hour=14)
        assert pm._check_legionella_stability(new_dow=3, new_hour=14) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestInferDhwSchedule tests/test_physics.py::TestLegionellaStabilityGuard -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Write minimal implementation**

Add to `ThermalPhysicsModel`:

```python
    def _infer_dhw_schedule(self, dhw_df: pd.DataFrame) -> dict | None:
        if dhw_df is None or dhw_df.empty or len(dhw_df) < 7 * 24:
            _LOGGER.warning("DHW schedule inference: insufficient history (need ≥7 days) — skipping")
            return None

        d = dhw_df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d.sort_values("timestamp").reset_index(drop=True)

        # local peaks: value higher than both neighbours
        is_peak = (d["buffer_temp"] > d["buffer_temp"].shift(1)) & (d["buffer_temp"] > d["buffer_temp"].shift(-1))
        peaks = d[is_peak.fillna(False)]
        if peaks.empty:
            return None

        t_dhw_upper = float(peaks["buffer_temp"].quantile(0.90))
        legionella_peaks = peaks[peaks["buffer_temp"] > t_dhw_upper + 3.0]
        if legionella_peaks.empty:
            return None

        t_legionella = float(legionella_peaks["buffer_temp"].max())
        legionella_dow = int(legionella_peaks["timestamp"].dt.dayofweek.mode().iloc[0])
        legionella_hour = int(legionella_peaks["timestamp"].dt.hour.mode().iloc[0])

        # cycle-start local minima -> T_dhw_lower
        is_trough = (d["buffer_temp"] < d["buffer_temp"].shift(1)) & (d["buffer_temp"] < d["buffer_temp"].shift(-1))
        troughs = d[is_trough.fillna(False) & (d["buffer_temp"] < t_dhw_upper)]
        t_dhw_lower = float(troughs["buffer_temp"].quantile(0.5)) if not troughs.empty else t_dhw_upper - 10.0

        return {
            "T_dhw_upper": t_dhw_upper,
            "T_legionella": t_legionella,
            "legionella_dow": legionella_dow,
            "legionella_hour": legionella_hour,
            "T_dhw_lower": t_dhw_lower,
            "dhw_tank_volume_l": self._schedule["dhw_tank_volume_l"],
        }

    def _check_legionella_stability(self, new_dow: int, new_hour: int) -> bool:
        old_dow = self._schedule.get("legionella_dow")
        old_hour = self._schedule.get("legionella_hour")
        if old_dow is None or old_hour is None:
            return True
        if new_dow != old_dow:
            _LOGGER.warning(
                f"Legionella timing day-of-week shifted ({old_dow} -> {new_dow}) — suspending autonomous learning"
            )
            return False
        if abs(new_hour - old_hour) > 2:
            _LOGGER.warning(
                f"Legionella timing shifted {abs(new_hour - old_hour)}h week-over-week — suspending autonomous learning"
            )
            return False
        return True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 52 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add DHW schedule inference and legionella stability guard"
```

---

### Task 11: `calibrate()` orchestration + atomic persistence

**Files:**
- Modify: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Consumes: all calibration methods from Tasks 7-10.
- Produces: `ThermalPhysicsModel.calibrate(energy_df, weather_df, climate_dfs, dhw_df, holdout_cutoff, heating_active_df=None, ev_df=None, away_df=None) -> None` — the index doc's contract method. Writes both JSON files atomically on completion.

- [ ] **Step 1: Write the failing test**

```python
class TestCalibrateOrchestration:
    def test_calibrate_writes_both_files_atomically(self, tmp_path):
        model_dir = tmp_path / "models"
        pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        rng = np.random.default_rng(4)
        ts = pd.date_range("2025-06-01", periods=20 * 24, freq="1h")
        energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": 0.4 + rng.normal(0, 0.02, len(ts))})
        weather_df = pd.DataFrame({"timestamp": ts, "temp_c": 15.0, "direct_radiation_wm2": 100.0})
        pm.calibrate(
            energy_df, weather_df, climate_dfs=None, dhw_df=None, holdout_cutoff=pd.Timestamp("2025-07-01")
        )
        assert (model_dir / "physics_calibration.json").exists()
        assert (model_dir / "physics_schedule.json").exists()
        assert not (model_dir / "physics_calibration.json.tmp").exists()

    def test_calibrate_updates_calibrated_at(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        assert pm._calib["calibrated_at"] is None
        ts = pd.date_range("2025-06-01", periods=20 * 24, freq="1h")
        energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": [0.4] * len(ts)})
        weather_df = pd.DataFrame({"timestamp": ts, "temp_c": [15.0] * len(ts), "direct_radiation_wm2": [0.0] * len(ts)})
        pm.calibrate(energy_df, weather_df, climate_dfs=None, dhw_df=None, holdout_cutoff=pd.Timestamp("2025-07-01"))
        assert pm._calib["calibrated_at"] is not None
        assert pm.calibration_stale is False

    def test_calibrate_persists_n_calibration_windows_ua_eff_across_reload(self, tmp_path):
        model_dir = tmp_path / "models"
        pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        pm._calib["n_calibration_windows_ua_eff"] = 35
        _atomic_write_json(pm._calibration_path, pm._calib)
        pm2 = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        assert pm2._calib["n_calibration_windows_ua_eff"] == 35
        assert pm2.is_cold_start_gated is False

    def test_calibration_failure_falls_back_to_config_defaults_no_exception(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        empty = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        empty_w = pd.DataFrame(columns=["timestamp", "temp_c", "direct_radiation_wm2"])
        pm.calibrate(empty, empty_w, climate_dfs=None, dhw_df=None, holdout_cutoff=pd.Timestamp("2025-07-01"))
        assert pm._calib["Q_base_el"] == 0.35  # unchanged default, no crash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestCalibrateOrchestration -v`
Expected: FAIL with `AttributeError: 'ThermalPhysicsModel' object has no attribute 'calibrate'`

- [ ] **Step 3: Write minimal implementation**

Add to `ThermalPhysicsModel`:

```python
    def calibrate(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        climate_dfs: dict[str, pd.DataFrame] | None,
        dhw_df: pd.DataFrame | None,
        holdout_cutoff: pd.Timestamp,
        heating_active_df: pd.DataFrame | None = None,
        ev_df: pd.DataFrame | None = None,
        away_df: pd.DataFrame | None = None,
    ) -> None:
        try:
            q_base_el = self._calibrate_base_load(energy_df, away_df, ev_df, holdout_cutoff)
            if q_base_el is not None:
                self._calib["Q_base_el"] = q_base_el

            q_dhw_daily = self._calibrate_dhw_daily(energy_df, self._calib["Q_base_el"], holdout_cutoff)
            if q_dhw_daily is not None:
                self._calib["Q_dhw_daily"] = q_dhw_daily

            ua_eff, n_windows = self._calibrate_ua_eff(energy_df, weather_df, climate_dfs, dhw_df, holdout_cutoff)
            self._calib["n_calibration_windows_ua_eff"] = n_windows
            if ua_eff is not None:
                self._calib["UA_eff"] = ua_eff
            else:
                _LOGGER.warning("UA_eff calibration unavailable — heating component will be skipped")

            solar_area = self._calibrate_solar_gain_area(
                energy_df, weather_df, climate_dfs, self._calib["UA_eff"], holdout_cutoff
            )
            if solar_area is not None:
                self._calib["solar_gain_area"] = solar_area

            ua_dhw = self._calibrate_ua_dhw(dhw_df, weather_df, heating_active_df, holdout_cutoff)
            if ua_dhw is not None:
                self._calib["UA_dhw"] = ua_dhw

            self._calib["calibrated_at"] = pd.Timestamp.now().isoformat()
            _atomic_write_json(self._calibration_path, self._calib)

            if dhw_df is not None and not dhw_df.empty:
                inferred = self._infer_dhw_schedule(dhw_df)
                if inferred is not None:
                    if self._check_legionella_stability(inferred["legionella_dow"], inferred["legionella_hour"]):
                        self._schedule.update(inferred)
                    else:
                        self._schedule.update({k: v for k, v in inferred.items() if not k.startswith("legionella")})
            _atomic_write_json(self._schedule_path, self._schedule)
        except Exception as e:
            _LOGGER.warning(f"Physics calibration failed: {e} — retaining previous/default parameters")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 56 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add calibrate() orchestration with atomic persistence"
```

---

### Task 12: Zone-boundary consistency check + open-window detection

**Files:**
- Modify: `apps/energy_forecast/physics.py`
- Test: `tests/test_physics.py`

**Interfaces:**
- Produces: `ThermalPhysicsModel.check_zone_boundary(current_thermostat_entities: list[str]) -> None` (logs WARNING if the entity list differs from the one recorded at `calibrated_at`; call this from Plan B's `train()` hook before calibration), `ThermalPhysicsModel.detect_open_windows(climate_dfs, weather_df, room_areas) -> pd.Series` (returns a boolean Series indexed by timestamp — the `open_window_hour` flag Plan B's sample-weighting step consumes).

`calibrate()` (Task 11) is extended to record which thermostat entities were used, so `check_zone_boundary()` has something to compare against.

- [ ] **Step 1: Write the failing test**

```python
class TestZoneBoundaryAndOpenWindows:
    def test_zone_boundary_warns_on_thermostat_list_change(self, tmp_path, caplog):
        model_dir = tmp_path / "models"
        pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        pm._calib["room_thermostats_at_calibration"] = ["climate.living_room"]
        with caplog.at_level("WARNING"):
            pm.check_zone_boundary(["climate.living_room", "climate.bedroom"])
        assert any("zone boundary" in r.message.lower() or "thermostat" in r.message.lower() for r in caplog.records)

    def test_zone_boundary_silent_when_unchanged(self, tmp_path, caplog):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib["room_thermostats_at_calibration"] = ["climate.living_room"]
        with caplog.at_level("WARNING"):
            pm.check_zone_boundary(["climate.living_room"])
        assert not any("zone boundary" in r.message.lower() for r in caplog.records)

    def test_zone_boundary_silent_when_not_yet_calibrated(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm.check_zone_boundary(["climate.living_room"])  # no calibration recorded yet -> no crash, no warning

    def test_open_window_flags_large_residual(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_eff=150.0)
        pm._tau_hours = 8.0
        ts = pd.date_range("2026-01-15 00:00", periods=6, freq="1h")
        weather_df = pd.DataFrame({"timestamp": ts, "temp_c": [0.0] * 6, "direct_radiation_wm2": [0.0] * 6})
        # room temp crashes at hour 3 (open window) — far from the smooth ODE projection
        actual_temps = [20.0, 19.9, 19.8, 12.0, 19.5, 19.4]
        climate_dfs = {"climate.living_room": pd.DataFrame({"timestamp": ts, "current_temp": actual_temps})}
        flags = pm.detect_open_windows(climate_dfs, weather_df, room_areas=None)
        assert flags.iloc[3] == True  # noqa: E712 — explicit bool comparison reads clearer here
        assert flags.iloc[0] == False  # noqa: E712

    def test_open_window_threshold_from_passive_windows_only(self, tmp_path):
        # regression guard: a globally noisy dataset with no genuine open-window event
        # should not blow the 2-sigma threshold out and flag everything
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_eff=150.0)
        pm._tau_hours = 8.0
        rng = np.random.default_rng(5)
        ts = pd.date_range("2026-01-15 00:00", periods=48, freq="1h")
        weather_df = pd.DataFrame({"timestamp": ts, "temp_c": rng.uniform(-5, 5, 48), "direct_radiation_wm2": [0.0] * 48})
        actual_temps = 20.0 + rng.normal(0, 0.1, 48)
        climate_dfs = {"climate.living_room": pd.DataFrame({"timestamp": ts, "current_temp": actual_temps})}
        flags = pm.detect_open_windows(climate_dfs, weather_df, room_areas=None)
        assert flags.sum() < 5  # not everything flagged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestZoneBoundaryAndOpenWindows -v`
Expected: FAIL with `AttributeError: 'ThermalPhysicsModel' object has no attribute 'check_zone_boundary'`

- [ ] **Step 3: Write minimal implementation**

Add to `ThermalPhysicsModel`:

```python
    def check_zone_boundary(self, current_thermostat_entities: list[str]) -> None:
        recorded = self._calib.get("room_thermostats_at_calibration")
        if not recorded:
            return
        if sorted(recorded) != sorted(current_thermostat_entities):
            _LOGGER.warning(
                "Zone boundary changed: room_thermostats at calibration time %s differs from current %s — "
                "UA_eff may no longer be consistent with Q_loss until recalibration",
                recorded,
                current_thermostat_entities,
            )

    def detect_open_windows(
        self,
        climate_dfs: dict[str, pd.DataFrame],
        weather_df: pd.DataFrame,
        room_areas: dict[str, float] | None,
    ) -> pd.Series:
        from .model import _find_passive_windows, _project_indoor_temps

        if not climate_dfs:
            return pd.Series(dtype=bool)

        timestamps = pd.DatetimeIndex(next(iter(climate_dfs.values()))["timestamp"])
        t_actual = self._area_weighted_t_indoor(climate_dfs, timestamps, room_areas)
        if t_actual is None:
            return pd.Series(False, index=timestamps)

        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        t_outdoor = w["temp_c"].reindex(timestamps, method="nearest")

        # ODE projection anchored at the first actual reading, stepped forward with tau
        tau = self._tau_hours or 8.0
        t_ode = np.empty(len(timestamps))
        t_ode[0] = t_actual.iloc[0]
        t_out_arr = t_outdoor.values
        for i in range(1, len(timestamps)):
            t_ode[i] = t_ode[i - 1] + (t_out_arr[i - 1] - t_ode[i - 1]) / tau
        t_ode_series = pd.Series(t_ode, index=timestamps)

        residual = t_actual - t_ode_series

        passive_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "T_outdoor": t_out_arr,
                "T_indoor": t_actual.values,
                "hp_running": False,
                "dhw_tank_temp": np.nan,
            }
        )
        passive_idx = _find_passive_windows(passive_df, min_delta_t=0.0, min_hp_off_hours=1)
        if len(passive_idx) >= 5:
            sigma = float(residual.iloc[passive_idx].std())
        else:
            sigma = float(residual.std()) if len(residual) > 1 else 0.0

        if sigma <= 1e-9:
            return pd.Series(False, index=timestamps)

        return (residual.abs() > 2 * sigma)
```

Wire `room_thermostats_at_calibration` into Task 11's `calibrate()` by adding, right before writing `physics_calibration.json`:

```python
            if climate_dfs:
                self._calib["room_thermostats_at_calibration"] = sorted(climate_dfs.keys())
```

(Insert this line in `calibrate()` immediately before `self._calib["calibrated_at"] = ...`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: 61 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add zone-boundary consistency check and open-window detection"
```

---

### Task 13: Full-suite regression check + self-review pass

**Files:** none new — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass, including the pre-existing `tests/test_model.py` τ-calibration tests (regression gate for Task 6's refactor).

- [ ] **Step 2: Confirm no unrelated files changed**

Run: `git diff dev --stat`
Expected: only `apps/energy_forecast/physics.py`, the `_find_passive_windows` addition + `_calibrate_tau` refactor in `apps/energy_forecast/model.py`, and `tests/test_physics.py`.

- [ ] **Step 3: Commit if any cleanup was needed, otherwise proceed to PR**

Follow `superpowers:finishing-a-development-branch` to open a PR against `dev` titled `feat: physics core engine (ThermalPhysicsModel)`. Do not merge until Plan B is ready to land immediately after — Plan A alone adds a dead module with no caller, which is fine to sit on `dev` briefly but should not go to `main` on its own.
