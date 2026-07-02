"""Tests for physics.py — ThermalPhysicsModel."""

from __future__ import annotations

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
            {
                **pytest.importorskip("energy_forecast.physics")._default_calibration(),
                "calibrated_at": pd.Timestamp.now().isoformat(),
            },
        )
        pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        assert pm.calibration_stale is False

    def test_is_cold_start_gated_when_windows_below_30(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib["n_calibration_windows_ua_eff"] = 29
        assert pm.is_cold_start_gated is True
        pm._calib["n_calibration_windows_ua_eff"] = 30
        assert pm.is_cold_start_gated is False
