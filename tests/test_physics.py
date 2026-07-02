"""Tests for physics.py — ThermalPhysicsModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_forecast.model import _find_passive_windows
from energy_forecast.physics import (
    COP_MIN,
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
