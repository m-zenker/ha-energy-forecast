"""Tests for physics.py — ThermalPhysicsModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_forecast.physics import COP_MIN, ThermalPhysicsModel, _atomic_write_json, _read_json_or_default

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
