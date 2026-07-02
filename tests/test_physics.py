"""Tests for physics.py — ThermalPhysicsModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_forecast.model import _find_passive_windows
from energy_forecast.physics import (
    COP_MIN,
    WATER_SPECIFIC_HEAT_WH_PER_L_K,
    ThermalPhysicsModel,
    _atomic_write_json,
    _read_json_or_default,
)

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

    def test_cop_formula_missing_from_config_uses_default(self, tmp_path):
        # Regression test: config dict missing cop_formula should not raise KeyError
        config_without_formula = {
            "cop_sensor": None,
            "dhw_tank_temp_sensor": None,
            "heating_buffer_temp_sensor": None,
            "heating_curve_sensor": None,
            # NOTE: cop_formula is intentionally omitted
            "dhw_tank_volume_l": 200,
            "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8,
            "heating_curve_points": [[-20, 55.5], [-5, 46.0], [5, 39.5], [20, 25.0]],
            "room_thermostats": [],
            "use_physics_residual": False,
        }
        pm = ThermalPhysicsModel(tmp_path / "models", config_without_formula)
        # Should not raise KeyError; should return a sane COP value
        cop_value = pm._cop_formula_value(-5.0, None)
        assert COP_MIN <= cop_value
        # Also test via _cop_series
        ts = pd.date_range("2026-01-15 00:00", periods=1, freq="1h")
        cop_series = pm._cop_series(ts, t_outdoor=pd.Series([-5.0], index=ts), cop_sensor_series=None)
        assert cop_series.iloc[0] >= COP_MIN


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

    def test_last_row_not_nan_regression(self, tmp_path):
        """Regression test: last row of _space_heating_kwh should be finite, not NaN.

        Previously, shift(-1).bfill() would leave trailing NaN which propagates
        through the calculation. This test ensures the fix (using fillna(t_indoor))
        correctly handles the boundary.
        """
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_eff=150.0, solar_gain_area=0.0, Q_base_el=0.0)
        ts = pd.date_range("2026-01-15 00:00", periods=4, freq="1h")
        t_indoor = pd.Series([20.0, 20.0, 20.0, 20.0], index=ts)
        t_outdoor = pd.Series([10.0, 10.0, 10.0, 10.0], index=ts)
        ghi = pd.Series([0.0, 0.0, 0.0, 0.0], index=ts)
        cop = pd.Series([3.0, 3.0, 3.0, 3.0], index=ts)
        q_heat_el = pm._space_heating_kwh(t_indoor, t_outdoor, ghi, cop)

        # Last row should be finite, not NaN
        assert np.isfinite(q_heat_el.iloc[-1]), f"Last row is NaN: {q_heat_el.iloc[-1]}"
        # Last row should be non-negative (clipping ensures this)
        assert q_heat_el.iloc[-1] >= 0.0
        # All rows should be finite
        assert q_heat_el.isna().sum() == 0, f"Found NaN values in result: {q_heat_el}"


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
        ts = pd.date_range("2026-01-15 00:00", periods=2, freq="1h")
        t_ambient = pd.Series([20.0, 20.0], index=ts)
        # start just below T_lower to force a reheat on hour 0
        q_dhw_el, final_temp = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=44.0, dhw_schedule_override=None)
        assert q_dhw_el.iloc[0] > 0.0
        # different tank volume -> different heating_rise -> different final tank temp (not hardcoded)
        pm2 = ThermalPhysicsModel(tmp_path / "models2", {**DEFAULT_CONFIG, "dhw_tank_volume_l": 300})
        pm2._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        q_dhw_el2, final_temp2 = pm2._dhw_kwh_series(ts, t_ambient, initial_t_tank=44.0, dhw_schedule_override=None)
        # within 2-hour window, electricity series is identical (same reheat power/COP hour 0, silent hour 1)
        # but final tank temperature diverges due to different heating_rise values
        assert final_temp != pytest.approx(final_temp2, abs=0.5)

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

    def test_zero_dhw_tank_volume_skips_dhw_component(self, tmp_path, caplog):
        """Regression test: zero/invalid dhw_tank_volume_l should not raise ZeroDivisionError.

        Previously, dividing by c_dhw=0 would crash. Now, the method gracefully
        returns an all-zero series with initial tank temperature preserved.
        """
        config_zero_volume = {**DEFAULT_CONFIG, "dhw_tank_volume_l": 0}
        pm = ThermalPhysicsModel(tmp_path / "models", config_zero_volume)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        ts = pd.date_range("2026-01-15 00:00", periods=24, freq="1h")
        t_ambient = pd.Series([20.0] * 24, index=ts)
        initial_t = 50.0

        # Should not raise; should return zeros
        q_dhw_el, final_temp = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=initial_t, dhw_schedule_override=None)

        # All electricity should be zero
        assert (q_dhw_el == 0.0).all(), f"Expected all zeros, got {q_dhw_el.values}"
        # Tank temperature should remain unchanged (returned as initial)
        assert final_temp == pytest.approx(initial_t)
        # Should have logged a warning
        assert "DHW tank volume is zero or invalid" in caplog.text


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

    def test_predict_series_uses_climate_recent_projection_path(self, tmp_path):
        """Regression: climate_recent path should change results vs. flat fallback."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_eff=150.0, solar_gain_area=5.0, Q_base_el=0.35, UA_dhw=15.0, Q_dhw_daily=3.5)

        # Future forecast window
        future_ts = pd.date_range("2026-01-15 12:00", periods=24, freq="1h")
        forecast_df = pd.DataFrame(
            {
                "timestamp": future_ts,
                "temp_c": np.linspace(8, 12, 24),
                "direct_radiation_wm2": np.linspace(100, 200, 24),
            }
        )

        # Recent history: 3 hours before forecast starts, with realistic indoor temp around 20°C
        recent_ts = pd.date_range("2026-01-15 09:00", periods=3, freq="1h")
        climate_recent = {
            "climate.living_room": pd.DataFrame(
                {
                    "timestamp": recent_ts,
                    "current_temp": [20.5, 20.3, 20.1],
                    "setpoint": [21.0, 21.0, 21.0],
                }
            )
        }

        # Call predict_series: once with climate_recent, once without
        result_with_climate = pm.predict_series(
            forecast_df, climate_recent=climate_recent, room_areas={"climate.living_room": 35.0}
        )
        result_without_climate = pm.predict_series(forecast_df, climate_recent=None)

        # Both should be valid Series of correct length
        assert isinstance(result_with_climate, pd.Series)
        assert isinstance(result_without_climate, pd.Series)
        assert len(result_with_climate) == 24
        assert len(result_without_climate) == 24

        # The critical assertion: results should differ because climate_recent changes t_indoor
        assert not result_with_climate.equals(result_without_climate), (
            "climate_recent path produced identical results to fallback; "
            "_project_indoor_temps may not be called or has no effect"
        )

    def test_predict_series_exception_fallback_returns_zeros_with_warning(self, tmp_path, monkeypatch, caplog):
        """Regression: internal exceptions should return all-zero Series and log WARNING."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_eff=150.0, solar_gain_area=5.0, Q_base_el=0.35, UA_dhw=15.0, Q_dhw_daily=3.5)

        ts = pd.date_range("2026-01-15 00:00", periods=48, freq="1h")
        forecast_df = pd.DataFrame(
            {"timestamp": ts, "temp_c": np.linspace(-2, 8, 48), "direct_radiation_wm2": np.zeros(48)}
        )

        # Monkeypatch an internal method to raise an exception
        def raise_error(*args, **kwargs):
            raise RuntimeError("Simulated physics calculation failure")

        monkeypatch.setattr(pm, "_space_heating_kwh", raise_error)

        # Capture log output
        with caplog.at_level("WARNING"):
            result = pm.predict_series(forecast_df)

        # Assert fallback behavior: all zeros with correct length
        assert isinstance(result, pd.Series)
        assert len(result) == 48
        assert (result == 0.0).all(), "Exception fallback should return all zeros"
        assert list(result.index) == list(ts), "Result index should match forecast_df timestamps"

        # Assert warning was logged
        assert "physics predict_series failed" in caplog.text, (
            f"Expected warning about predict_series failure in log; got: {caplog.text}"
        )


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

    def test_all_nan_gross_kwh_returns_none_regression(self, tmp_path):
        """Regression test: _calibrate_base_load with all NaN gross_kwh should return None, not NaN.

        Previously, 14+ rows of NaN gross_kwh would pass the length check, and
        df["gross_kwh"].median() returns NaN, which float(nan) preserves. This would silently
        poison downstream calibration. After fix: dropna filters out NaN rows before counting,
        and the result median is checked for finiteness.
        """
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        rows = []
        for day in pd.date_range("2026-06-01", periods=14, freq="1D"):
            for hour in range(1, 5):
                rows.append({"timestamp": day + pd.Timedelta(hours=hour), "gross_kwh": np.nan})
        energy_df = pd.DataFrame(rows)
        result = pm._calibrate_base_load(energy_df, away_df=None, ev_df=None, holdout_cutoff=pd.Timestamp("2026-07-01"))
        assert result is None, f"Expected None for all-NaN data, got {result}"


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

    def test_excludes_post_holdout_rows(self, tmp_path):
        """Holdout-exclusion test: verify post-holdout anomalies don't skew DHW daily calibration.

        Construct 14+ summer days of normal DHW daily consumption (before holdout_cutoff)
        plus a batch of post-holdout days with wildly inflated values. Verify the result
        matches what the pre-holdout data alone would produce, i.e. post-holdout anomalies
        are correctly excluded.
        """
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        q_base_el = 0.4

        # Pre-holdout: 14 summer days, consistent pattern (base 0.4 kWh/h * 24 + DHW 3.5 = 12.1 kWh/day)
        ts_pre = pd.date_range("2026-06-01", periods=14 * 24, freq="1h")
        vals_pre = np.full(len(ts_pre), q_base_el)
        vals_pre[3::24] += 3.5  # DHW reheat hour 3 each day
        energy_df_pre = pd.DataFrame({"timestamp": ts_pre, "gross_kwh": vals_pre})

        # Post-holdout: 14 days with wildly inflated values (to verify they are excluded)
        ts_post = pd.date_range("2026-07-15", periods=14 * 24, freq="1h")
        vals_post = np.full(len(ts_post), q_base_el * 10)  # 10x normal
        vals_post[3::24] += 35.0  # DHW scaled 10x
        energy_df_post = pd.DataFrame({"timestamp": ts_post, "gross_kwh": vals_post})

        # Combine into full dataset
        energy_df = pd.concat([energy_df_pre, energy_df_post], ignore_index=True)

        # Calibrate with holdout_cutoff that excludes the post-holdout anomaly
        result = pm._calibrate_dhw_daily(energy_df, q_base_el=q_base_el, holdout_cutoff=pd.Timestamp("2026-07-01"))

        # Expected: daily mean from pre-holdout only: 0.4*24 + 3.5 = 12.1 kWh/day
        # Q_dhw_daily = daily_mean - 24*q_base_el = 12.1 - 24*0.4 = 12.1 - 9.6 = 2.5 kWh/day
        expected = 3.5
        assert result == pytest.approx(expected, rel=0.05), (
            f"Expected {expected}, got {result} — holdout exclusion may have failed"
        )


class TestCalibrateUAEff:
    def _synthetic_winter_data(self, ua_eff_true=150.0, cop=3.0, n_nights=35, with_dhw_sensor=True):
        rng = np.random.default_rng(1)
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
        # 3 nights x 8 rows/night = 24 rows, all of which pass the passive-window filter in
        # this synthetic dataset (constant DHW temp, ΔT always well above the 8K gate) -> 24
        # windows, genuinely below the 30-window cold-start threshold. (n_nights=10 would give
        # 80 rows, all passing the filter, which would exercise the R² gate instead of the
        # cold-start gate this test is meant to isolate.)
        energy_df, weather_df, climate_dfs, dhw_df = self._synthetic_winter_data(n_nights=3)
        ua_eff, n_windows = pm._calibrate_ua_eff(
            energy_df, weather_df, climate_dfs, dhw_df, holdout_cutoff=pd.Timestamp("2026-06-01")
        )
        assert ua_eff is None
        assert n_windows < 30

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
