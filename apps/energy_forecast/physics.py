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
