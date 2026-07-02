"""Physics-based hourly electricity predictor (space heating, DHW, base load).

See docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md for the design.
"""

from __future__ import annotations

import json
import logging
import os
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
    [
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.02,
        0.08,
        0.10,
        0.06,
        0.03,
        0.02,
        0.02,
        0.03,
        0.02,
        0.02,
        0.02,
        0.03,
        0.05,
        0.09,
        0.11,
        0.10,
        0.07,
        0.04,
        0.02,
    ]
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

    def _t_flow_c(self, t_outdoor_c: float, live_shift_k: float | None) -> float:
        points = self._config.get("heating_curve_points") or []
        if not points:
            return DEFAULT_T_FLOW_C
        shift = live_shift_k if live_shift_k is not None else 0.0
        xs = [p[0] for p in points]
        ys = [p[1] + shift for p in points]
        return float(np.interp(t_outdoor_c, xs, ys))

    def _cop_formula_value(self, t_outdoor_c: float, live_shift_k: float | None) -> float:
        formula = self._calib.get("cop_formula") or self._config.get("cop_formula", {"a": 2.5, "b": 0.07})
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
        t_indoor_next = t_indoor.shift(-1).fillna(t_indoor)
        q_mass = c_building * (t_indoor_next - t_indoor)

        q_heat = (q_loss - q_solar - q_gain_int + q_mass).clip(lower=0.0)
        q_heat_el = q_heat / cop.clip(lower=COP_MIN) / 1000.0
        return q_heat_el

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
