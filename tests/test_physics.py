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
        q_dhw_el, _, final_temp = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=44.0, override_lookup=None)
        assert (q_dhw_el >= 0.0).all()
        assert 45.0 <= final_temp <= 60.0  # clamp bounds enforced

    def test_reheat_cycles_multiple_times_not_permanently_pinned_after_first_floor_touch(self, tmp_path):
        """Regression: the reheat trigger was `t_tank < t_lower` (strict), but the passive
        branch's clip() floors any crossing to exactly t_lower. Once that happened, the
        strict `<` check could never be true again (t_tank == t_lower forever), permanently
        disabling reheat for the rest of any run — el_kwh stuck at 0.0 no matter how long
        the window. Fixed to `t_tank <= t_lower` so a tank sitting exactly at the floor
        re-triggers reheat. Over 24h with a default calibration and initial_t_tank at the
        floor, the tank must cycle (reheat, coast down, re-touch the floor, reheat again)
        multiple times, not just once."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm._schedule.update(T_dhw_lower=45.0, T_dhw_upper=55.0, T_legionella=60.0)
        ts = pd.date_range("2026-08-04 00:00", periods=24, freq="1h")
        t_ambient = pd.Series(10.0, index=ts)
        q_dhw_el, t_tank_series, _ = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=45.0, override_lookup=None)
        nonzero_hours = (q_dhw_el > 0.0).sum()
        assert nonzero_hours >= 2, f"expected multiple reheat cycles in 24h, got {nonzero_hours}"
        # tank must actually revisit the floor between reheats, not just heat once and coast
        assert (t_tank_series == pm._schedule["T_dhw_lower"]).sum() >= 2

    def test_normal_reheat_caps_at_t_dhw_upper_not_t_legionella(self, tmp_path):
        """Final-review #3: the NORMAL (non-override) reheat branch used to clip to
        T_legionella (60 degC) as its ceiling, so every ordinary reheat cycle over-shot
        the real setpoint T_dhw_upper (55 degC) by 5 K — inflating stored energy, and
        therefore standing losses and the silent-coast time that follows, for every
        install. Legionella/comfort_boost overrides are unaffected: they bypass this
        branch entirely via the `override_target is not None` branch above it.
        """
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm._schedule.update(T_dhw_lower=45.0, T_dhw_upper=55.0, T_legionella=60.0)
        ts = pd.date_range("2026-01-15 00:00", periods=24 * 5, freq="1h")
        t_ambient = pd.Series(20.0, index=ts)
        q_dhw_el, t_tank_series, _ = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=45.0, override_lookup=None)

        assert t_tank_series.max() <= pm._schedule["T_dhw_upper"] + 1e-9, (
            f"normal reheat overshot T_dhw_upper: max={t_tank_series.max()}"
        )
        # ... and the ceiling is actually reachable (not a vacuous bound): with a
        # 17.2 K/h heating rise from the 45 degC floor, a full reheat hour saturates it.
        assert t_tank_series.max() == pytest.approx(pm._schedule["T_dhw_upper"], abs=1e-6)
        # ... while the tank still genuinely cycles (reheat / coast / reheat).
        assert (q_dhw_el > 0.0).sum() >= 2

    def test_reheat_energy_prorated_to_actual_delta_when_ceiling_clips_mid_hour(self, tmp_path):
        """Regression for the +29% DHW electricity side effect discovered after the
        T_dhw_upper ceiling fix above: shrinking the T_dhw_lower..T_dhw_upper buffer
        makes reheat cycles more frequent, and each one used to bill a flat full hour
        at rated power even when the tank hit the ceiling partway through the hour.
        Electricity must instead be prorated to the delta actually achieved by
        heating (14-day scenario: 50.26 -> 64.62 kWh with flat billing, ~neutral with
        prorating)."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        # 1 K buffer guarantees the ~17.2 K/h heating rise clips well within the hour.
        pm._schedule.update(T_dhw_lower=45.0, T_dhw_upper=46.0, T_legionella=60.0)
        ts = pd.date_range("2026-01-15 00:00", periods=1, freq="1h")
        t_ambient = pd.Series([20.0], index=ts)
        q_dhw_el, t_tank_series, _ = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=45.0, override_lookup=None)

        cop_dhw = pm._cop_formula_value(20.0, None)
        full_hour_kwh = (DEFAULT_CONFIG["dhw_power_w"] / cop_dhw) / 1000.0
        assert t_tank_series.iloc[0] == pytest.approx(46.0, abs=1e-6)  # ceiling reached mid-hour
        assert 0.0 < q_dhw_el.iloc[0] < full_hour_kwh, (
            f"expected prorated billing below the full-hour amount ({full_hour_kwh:.3f} kWh), "
            f"got {q_dhw_el.iloc[0]:.3f} kWh"
        )

    def test_override_target_still_bypasses_the_reheat_ceiling(self, tmp_path):
        """Companion to the above: a legionella override must still be able to drive the
        tank to T_legionella (60 degC), above the normal-reheat T_dhw_upper ceiling."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm._schedule.update(T_dhw_lower=45.0, T_dhw_upper=55.0, T_legionella=60.0)
        ts = pd.date_range("2026-01-15 00:00", periods=24, freq="1h")
        t_ambient = pd.Series(20.0, index=ts)
        target_ts = pd.Timestamp("2026-01-15 12:00")
        _, t_tank_series, _ = pm._dhw_kwh_series(
            ts, t_ambient, initial_t_tank=45.0, override_lookup=lambda t: 60.0 if t == target_ts else None
        )
        assert t_tank_series.loc[target_ts] == pytest.approx(60.0)

    def test_heating_rise_derived_not_constant(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        ts = pd.date_range("2026-01-15 00:00", periods=2, freq="1h")
        t_ambient = pd.Series([20.0, 20.0], index=ts)
        # start just below T_lower to force a reheat on hour 0
        q_dhw_el, _, final_temp = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=44.0, override_lookup=None)
        assert q_dhw_el.iloc[0] > 0.0
        # different tank volume -> different heating_rise -> different final tank temp (not hardcoded).
        # 500 L, not 300 L: once the normal-reheat ceiling was corrected from T_legionella
        # (60 degC) to T_dhw_upper (55 degC) — final-review #3 — a 300 L tank's 11.5 K/h rise
        # from the 44 degC start still saturates that lower ceiling within the single reheat
        # hour, just as the 200 L tank's 17.2 K/h does, so both land on 55.0 and the volume
        # difference is masked by the clip rather than by any hardcoding. 500 L (6.9 K/h)
        # stays below the ceiling, so heating_rise still visibly drives the result.
        pm2 = ThermalPhysicsModel(tmp_path / "models2", {**DEFAULT_CONFIG, "dhw_tank_volume_l": 500})
        pm2._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        q_dhw_el2, _, final_temp2 = pm2._dhw_kwh_series(ts, t_ambient, initial_t_tank=44.0, override_lookup=None)
        # within 2-hour window, electricity series is identical (same reheat power/COP hour 0, silent hour 1)
        # but final tank temperature diverges due to different heating_rise values
        assert final_temp != pytest.approx(final_temp2, abs=0.5)

    def test_post_legionella_silence(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm._schedule.update(T_dhw_lower=45.0, T_dhw_upper=55.0, T_legionella=60.0)
        ts = pd.date_range("2026-01-15 00:00", periods=12, freq="1h")
        t_ambient = pd.Series([20.0] * 12, index=ts)
        q_dhw_el, _, _ = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=60.0, override_lookup=None)
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
        q_dhw_el, _, _ = pm._dhw_kwh_series(
            ts, t_ambient, initial_t_tank=50.0, override_lookup=lambda ts: pm._dhw_override_for_hour(ts, override)
        )
        thu_10 = pd.Timestamp("2026-06-25 10:00")
        # a legionella boost (heating to T_legionella) must occur at/after the overridden hour
        assert q_dhw_el.loc[q_dhw_el.index >= thu_10].max() > 0

    def test_ode_edge_case_zero_delta_t(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        ts = pd.date_range("2026-01-15 00:00", periods=2, freq="1h")
        t_ambient = pd.Series([50.0, 50.0], index=ts)  # T_ambient == T_tank -> no insulation loss
        q_dhw_el, _, final_temp = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=50.0, override_lookup=None)
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
        q_dhw_el, _, final_temp = pm._dhw_kwh_series(ts, t_ambient, initial_t_tank=initial_t, override_lookup=None)

        # All electricity should be zero
        assert (q_dhw_el == 0.0).all(), f"Expected all zeros, got {q_dhw_el.values}"
        # Tank temperature should remain unchanged (returned as initial)
        assert final_temp == pytest.approx(initial_t)
        # Should have logged a warning
        assert "DHW tank volume is zero or invalid" in caplog.text


class TestDhwKwhSeriesOverrideLookup:
    def test_override_lookup_none_is_override_blind(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
        t_ambient = pd.Series(10.0, index=timestamps)
        el_kwh, t_tank_series, final_t = pm._dhw_kwh_series(timestamps, t_ambient, 50.0, override_lookup=None)
        # No override anywhere in a 24h override-blind run — series must have no
        # discontinuous jump to T_legionella at any hour.
        assert (t_tank_series <= pm._schedule["T_legionella"] + 1e-6).all()
        assert not (t_tank_series == pm._schedule["T_legionella"]).any()

    def test_override_lookup_callable_applies_target_at_matching_hour(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
        t_ambient = pd.Series(10.0, index=timestamps)
        target_ts = pd.Timestamp("2026-08-04 12:00")

        def lookup(ts):
            return 60.0 if ts == target_ts else None

        _, t_tank_series, _ = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=lookup)
        assert t_tank_series.loc[target_ts] == 60.0

    def test_t_tank_series_has_same_index_as_timestamps(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        timestamps = pd.date_range("2026-08-04 00:00", periods=6, freq="h")
        t_ambient = pd.Series(10.0, index=timestamps)
        _, t_tank_series, _ = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=None)
        assert list(t_tank_series.index) == list(timestamps)


class TestOverrideDeltaSeries:
    def test_zero_when_no_override_history(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
        t_ambient = pd.Series(10.0, index=timestamps)
        delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, [])
        assert (delta == 0.0).all()

    def test_legionella_override_produces_nonzero_multihour_tail(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
        timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
        # 20.0, not the plan's literal 10.0: _dhw_kwh_series's reheat-trigger permanent-pin
        # bug (t_tank < t_lower never re-firing once clipped exactly to the floor) has now
        # been fixed (t_tank <= t_lower), so the tank genuinely cycles at either ambient.
        # But at 10.0 the passive cool-down between cycles takes longer, so the reheat
        # cycle period is long enough that the legionella boost's one-cycle phase-shift
        # hasn't fully resynced with the un-boosted baseline by hour 23 (delta.iloc[-1]
        # != 0) — empirically verified. 20.0 gives a short enough cycle period to
        # reconverge within this 24h window while still exercising a genuine multi-hour
        # nonzero tail.
        t_ambient = pd.Series(20.0, index=timestamps)
        history = [
            {
                "kind": "legionella",
                "date": "2026-08-04",
                "hour": 12,
                "target_c": 60.0,
                "committed_at": "x",
                "cancelled_at": None,
            },
        ]
        delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)
        # nonzero at the override hour and for at least a few hours after (reconvergence tail)
        assert delta.loc["2026-08-04 12:00"] != 0.0
        nonzero_after = (delta.loc["2026-08-04 13:00":"2026-08-04 18:00"] != 0.0).sum()
        assert nonzero_after >= 1
        # trajectories must have reconverged by the end of the 24h window
        assert delta.iloc[-1] == 0.0

    def test_joint_computation_no_double_counting_on_overlapping_tails(self, tmp_path):
        """R2-#1 direct regression: legionella hour 12 + comfort_boost hour 14 (inside
        the reconvergence tail) must not sum two independent per-override diffs."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
        timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
        t_ambient = pd.Series(10.0, index=timestamps)
        history = [
            {
                "kind": "legionella",
                "date": "2026-08-04",
                "hour": 12,
                "target_c": 60.0,
                "committed_at": "x",
                "cancelled_at": None,
            },
            {
                "kind": "comfort_boost",
                "date": "2026-08-04",
                "hour": 14,
                "target_c": 55.0,
                "committed_at": "y",
                "cancelled_at": None,
            },
        ]
        joint_delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)

        # independent-per-override baseline for comparison: replay legionella alone
        legionella_only = [history[0]]
        legionella_only_delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, legionella_only)
        comfort_boost_only = [history[1]]
        comfort_boost_only_delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, comfort_boost_only)
        naive_sum = legionella_only_delta + comfort_boost_only_delta
        # the joint computation must differ from the naive independent sum in at
        # least one overlapping hour — proving it isn't just summing two diffs
        assert not joint_delta.equals(naive_sum)

    def test_sanity_bound_references_larger_of_the_two_trajectories(self, tmp_path):
        """R2-#2: for the dominant negative-delta case (override run draws less),
        the bound must reference max(with_override, baseline), not just the
        override run's own (often near-zero) draw."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
        timestamps = pd.date_range("2026-08-04 00:00", periods=12, freq="h")
        t_ambient = pd.Series(10.0, index=timestamps)
        history = [
            {
                "kind": "legionella",
                "date": "2026-08-04",
                "hour": 0,
                "target_c": 60.0,
                "committed_at": "x",
                "cancelled_at": None,
            },
        ]
        delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)
        baseline_kwh, _, _ = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=None)

        def override_lookup(ts):
            return pm._dhw_override_for_hour_from_history(ts, history)

        with_override_kwh, _, _ = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup)
        bound = pd.concat([with_override_kwh, baseline_kwh], axis=1).max(axis=1)
        assert (delta.abs() <= bound + 1e-9).all()

    def test_tail_termination_matches_between_training_and_serving_entry_points(self, tmp_path):
        """R2-#3 direct regression: both public entry points delegate to the same
        shared function, so they resolve an equivalent override to the identical
        tail length."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
        timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
        t_ambient = pd.Series(10.0, index=timestamps)
        history = [
            {
                "kind": "legionella",
                "date": "2026-08-04",
                "hour": 12,
                "target_c": 60.0,
                "committed_at": "x",
                "cancelled_at": None,
            },
        ]
        training_delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)

        pm._schedule["committed_override"] = {"legionella": ["2026-08-04", 12]}
        serving_delta = pm.compute_serving_override_delta(timestamps, t_ambient, 45.0)

        training_tail_end = (training_delta != 0.0).to_numpy().nonzero()[0][-1]
        serving_tail_end = (serving_delta != 0.0).to_numpy().nonzero()[0][-1]
        assert training_tail_end == serving_tail_end

    def test_two_separate_episodes_both_keep_their_own_tail(self, tmp_path):
        """Final-review regression (#1): the tail-termination logic used to find only the
        FIRST divergence→reconvergence pair and then zero raw_delta from that reconvergence
        point to the END of the array — silently destroying every later override episode's
        legitimate, already-correctly-computed delta. Two legionella overrides two weeks
        apart in a month-long training window must BOTH produce a nonzero delta."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
        timestamps = pd.date_range("2026-08-01 00:00", periods=24 * 30, freq="h")
        # 20.0 ambient keeps the reheat cycle short enough that the first episode fully
        # reconverges long before the second one starts (same rationale as the
        # single-episode test above), so the two episodes are genuinely independent.
        t_ambient = pd.Series(20.0, index=timestamps)
        history = [
            {
                "kind": "legionella",
                "date": "2026-08-04",
                "hour": 12,
                "target_c": 60.0,
                "committed_at": "x",
                "cancelled_at": None,
            },
            {
                "kind": "legionella",
                "date": "2026-08-18",
                "hour": 12,
                "target_c": 60.0,
                "committed_at": "y",
                "cancelled_at": None,
            },
        ]
        delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)

        first = delta.loc["2026-08-04 12:00":"2026-08-04 23:00"]
        second = delta.loc["2026-08-18 12:00":"2026-08-18 23:00"]
        assert (first != 0.0).any(), "first episode must produce a nonzero delta"
        assert (second != 0.0).any(), "second episode must produce a nonzero delta too"
        assert delta.loc["2026-08-18 12:00"] != 0.0
        # ... and the quiet gap between the two episodes must still be fully zeroed
        # (episode 1's tank trajectories reconverge at 2026-08-07 06:00, so the gap
        # starts after that — the point is that the SECOND episode is not swallowed,
        # not that the first one's legitimate tail is cut short)
        gap = delta.loc["2026-08-07 07:00":"2026-08-18 11:00"]
        assert (gap == 0.0).all(), "hours between the two episodes must stay zeroed"
        # ... as must the tail after the last episode reconverges
        assert delta.iloc[-1] == 0.0

    def test_train_serve_symmetry_full_delta_tail_round_trips(self, tmp_path):
        """Subtracting then re-adding the delta across the entire tail (not just the
        committed hour) must round-trip to the original value (R1-#7)."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5, Q_base_el=0.35)
        timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
        t_ambient = pd.Series(10.0, index=timestamps)
        history = [
            {
                "kind": "legionella",
                "date": "2026-08-04",
                "hour": 12,
                "target_c": 60.0,
                "committed_at": "x",
                "cancelled_at": None,
            },
        ]
        with_override_kwh, _, _ = pm._dhw_kwh_series(
            timestamps, t_ambient, 45.0, lambda ts: pm._dhw_override_for_hour_from_history(ts, history)
        )
        delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, history)
        baseline_kwh, _, _ = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=None)
        # train side: subtract delta from with-override actuals to get "as if blind"
        reconstructed_blind = with_override_kwh - delta
        pd.testing.assert_series_equal(reconstructed_blind, baseline_kwh, check_names=False)


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

    def test_accepts_precomputed_dhw_rising_mask(self):
        ts = pd.date_range("2026-01-15 00:00", periods=3, freq="1h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "T_outdoor": [0.0, 0.0, 0.0],
                "T_indoor": [20.0, 20.0, 20.0],
                "hp_running": [False, False, False],
                "dhw_tank_temp": [45.0, 50.0, 50.0],  # rising 45->50 at row 1 -- would exclude
                #                                        row 1 if the default .diff() path ran
            }
        )
        # Precomputed mask says nothing is rising -> overrides the (otherwise-excluding)
        # dhw_tank_temp column entirely. min_hp_off_hours=0 matches how
        # _calibrate_ua_eff always calls this function, isolating the dhw-rising
        # behavior from the off-run-length gate.
        precomputed = pd.Series([False, False, False], index=ts)
        idx = _find_passive_windows(df, min_delta_t=8.0, min_hp_off_hours=0, dhw_rising_precomputed=precomputed)
        assert list(idx) == [0, 1, 2]

    def test_precomputed_mask_reindexed_by_timestamp_not_position(self):
        # Out of order, plus an extra timestamp df doesn't have -- reindexing by real
        # timestamp value (not position) must still find each row's correct entry.
        ts = pd.date_range("2026-01-15 00:00", periods=3, freq="1h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "T_outdoor": [0.0, 0.0, 0.0],
                "T_indoor": [20.0, 20.0, 20.0],
                "hp_running": [False, False, False],
                "dhw_tank_temp": [np.nan] * 3,
            }
        )
        precomputed = pd.Series(
            [True, False, True, False],
            index=[ts[2], ts[0], pd.Timestamp("2026-01-15 05:00"), ts[1]],
        )
        idx = _find_passive_windows(df, min_delta_t=8.0, min_hp_off_hours=0, dhw_rising_precomputed=precomputed)
        assert list(idx) == [0, 1]  # ts[2] -> True (rising) -> excluded; ts[0], ts[1] -> False -> included

    def test_default_behavior_unchanged_when_precomputed_mask_omitted(self):
        """Regression test for the spec §3.5 default-preserves-behavior claim: identical
        input/output to the pre-existing test_excludes_rising_dhw_tank_temp_hours."""
        ts = pd.date_range("2026-01-15 00:00", periods=3, freq="1h")
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "T_outdoor": [0.0, 0.0, 0.0],
                "T_indoor": [20.0, 20.0, 20.0],
                "hp_running": [False, False, False],
                "dhw_tank_temp": [45.0, 50.0, 50.0],
            }
        )
        idx = _find_passive_windows(df, min_delta_t=8.0, min_hp_off_hours=2)
        assert 1 not in idx


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


class TestInferDhwSchedule:
    def test_four_synthetic_peaks_wednesday_1400(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        rows = []
        # Generate 4 weeks of continuous hourly data with oscillating DHW temperature
        # Tank heats up to peaks at ~08:00 and ~14:00, cools between peaks
        # Normal peak: ~50°C; Wednesday 14:00 boost: 62°C
        base = pd.Timestamp("2026-01-07")  # a Wednesday
        for day in range(28):  # 4 weeks = 28 days
            for h in range(24):
                ts = base + pd.Timedelta(days=day, hours=h)
                dow = ts.dayofweek  # 0=Monday, 2=Wednesday
                # Temperature pattern: rises to peak during heating, falls during off periods
                # Morning peak around 08:00, afternoon peak around 14:00
                if h < 6:
                    temp = 45.0  # Night: baseline
                elif h < 8:
                    temp = 45.0 + (h - 6) * 2.5  # Morning ramp up
                elif h == 8:
                    temp = 50.0  # Morning peak
                elif h < 14:
                    temp = 50.0 - (h - 8) * 0.5  # Gradual cooling
                elif h == 14:
                    # Wednesday gets legionella boost to 62°C; others stay at ~49°C
                    temp = 62.0 if dow == 2 else 49.0
                elif h < 18:
                    temp = 49.0 - (h - 14) * 1.0  # Cooling from afternoon
                elif h < 20:
                    temp = 45.0 + (h - 18) * 2.5  # Evening ramp up
                elif h == 20:
                    temp = 50.0  # Evening peak
                else:
                    temp = 50.0 - (h - 20) * 1.5  # Evening cool-down
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

    def test_flat_buffer_temp_no_peaks_logs_warning(self, tmp_path, caplog):
        """Test that flat buffer_temp (no local peaks) logs WARNING and returns None."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        rows = []
        base = pd.Timestamp("2026-01-01")
        # Generate 7 days of constant buffer_temp (no local peaks)
        for day in range(7):
            for h in range(24):
                ts = base + pd.Timedelta(days=day, hours=h)
                rows.append({"timestamp": ts, "buffer_temp": 50.0})
        dhw_df = pd.DataFrame(rows)

        with caplog.at_level("WARNING"):
            result = pm._infer_dhw_schedule(dhw_df)

        assert result is None
        assert "no local peaks found in temperature history" in caplog.text

    def test_peaks_exist_but_no_legionella_boost_logs_warning(self, tmp_path, caplog):
        """Test that peaks exist but none exceed T_dhw_upper+3.0K logs WARNING and returns None."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        rows = []
        base = pd.Timestamp("2026-01-01")
        # Generate 4 weeks with peaks all at ~50°C, no legionella boost
        # This ensures T_dhw_upper will be ~50°C, and no peak will exceed 50 + 3.0 = 53°C
        for day in range(28):
            for h in range(24):
                ts = base + pd.Timedelta(days=day, hours=h)
                # Create temperature oscillation with peaks all at 50°C
                if h < 6:
                    temp = 45.0
                elif h < 8:
                    temp = 45.0 + (h - 6) * 2.5  # Ramp to peak at h=8
                elif h == 8:
                    temp = 50.0  # Peak 1 at exactly 50°C
                elif h < 14:
                    temp = 50.0 - (h - 8) * 0.5  # Cool down
                elif h == 14:
                    temp = 50.0  # Peak 2 at exactly 50°C (no boost)
                elif h < 20:
                    temp = 50.0 - (h - 14) * 0.5  # Cool down
                elif h == 20:
                    temp = 50.0  # Peak 3 at exactly 50°C
                else:
                    temp = 50.0 - (h - 20) * 1.5  # Cool down
                rows.append({"timestamp": ts, "buffer_temp": temp})
        dhw_df = pd.DataFrame(rows)

        with caplog.at_level("WARNING"):
            result = pm._infer_dhw_schedule(dhw_df)

        assert result is None
        assert "no legionella boost peak found" in caplog.text


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


class TestCalibrateOrchestration:
    def test_calibrate_writes_both_files_atomically(self, tmp_path):
        model_dir = tmp_path / "models"
        pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        rng = np.random.default_rng(4)
        ts = pd.date_range("2025-06-01", periods=20 * 24, freq="1h")
        energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": 0.4 + rng.normal(0, 0.02, len(ts))})
        weather_df = pd.DataFrame({"timestamp": ts, "temp_c": 15.0, "direct_radiation_wm2": 100.0})
        pm.calibrate(energy_df, weather_df, climate_dfs=None, dhw_df=None, holdout_cutoff=pd.Timestamp("2025-07-01"))
        assert (model_dir / "physics_calibration.json").exists()
        assert (model_dir / "physics_schedule.json").exists()
        assert not (model_dir / "physics_calibration.json.tmp").exists()

    def test_calibrate_updates_calibrated_at(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        assert pm._calib["calibrated_at"] is None
        ts = pd.date_range("2025-06-01", periods=20 * 24, freq="1h")
        energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": [0.4] * len(ts)})
        weather_df = pd.DataFrame(
            {"timestamp": ts, "temp_c": [15.0] * len(ts), "direct_radiation_wm2": [0.0] * len(ts)}
        )
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
        weather_df = pd.DataFrame(
            {"timestamp": ts, "temp_c": rng.uniform(-5, 5, 48), "direct_radiation_wm2": [0.0] * 48}
        )
        actual_temps = 20.0 + rng.normal(0, 0.1, 48)
        climate_dfs = {"climate.living_room": pd.DataFrame({"timestamp": ts, "current_temp": actual_temps})}
        flags = pm.detect_open_windows(climate_dfs, weather_df, room_areas=None)
        assert flags.sum() < 5  # not everything flagged


class TestCommittedDhwSchedule:
    def test_commit_persists_override_and_bypasses_stability_guard(self, tmp_path):
        model_dir = tmp_path / "models"
        pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        pm._schedule.update(legionella_dow=2, legionella_hour=14)
        # a >2h shift would normally be rejected by _check_legionella_stability
        pm.commit_dhw_schedule({"legionella": ("2026-06-25", 22)})
        assert pm._schedule["committed_override"] == {
            "legionella": ["2026-06-25", 22]
        }  # JSON round-trips tuples as lists
        # reload from disk to confirm persistence
        pm2 = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        assert pm2._schedule["committed_override"] == {"legionella": ["2026-06-25", 22]}
        assert len(pm2._schedule["override_history"]) == 1  # new: side effect of commit

    def test_predict_series_is_override_blind_by_default_even_with_committed_override(self, tmp_path):
        """As of Phase A Task 6, predict_series's silent fallback to
        self._schedule["committed_override"] was intentionally removed — the
        ML-facing physics_kwh feature must never silently pick up whatever
        happens to be currently committed (Goal 1). Formerly named
        test_natural_baseline_applies_committed_override_by_default and
        asserted the opposite; that fallback no longer exists anywhere in
        predict_series. Reflecting a committed override into a "natural
        baseline" is now the caller's explicit responsibility — see
        predict_scenario()'s natural_baseline_df construction (model.py) and
        TestScenarioDhwDelta::test_set_dhw_schedule_then_get_scenario_with_different_override,
        which exercises that call site."""
        # Model WITH committed override
        pm_with_override = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm_with_override._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm_with_override._schedule.update(T_dhw_lower=45.0, T_dhw_upper=55.0, T_legionella=60.0)
        pm_with_override.commit_dhw_schedule({"legionella": ("2026-06-25", 10)})

        # Fresh model WITHOUT committed override for comparison
        pm_no_override = ThermalPhysicsModel(tmp_path / "models2", DEFAULT_CONFIG)
        pm_no_override._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm_no_override._schedule.update(T_dhw_lower=45.0, T_dhw_upper=55.0, T_legionella=60.0)
        # Note: no commit_dhw_schedule call for pm_no_override

        ts = pd.date_range("2026-06-25 00:00", periods=24, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "temp_c": [10.0] * 24, "direct_radiation_wm2": [0.0] * 24})

        # no explicit dhw_schedule_override passed — must be override-blind
        # regardless of what is committed on the model
        result_with = pm_with_override.predict_series(forecast_df)
        result_without = pm_no_override.predict_series(forecast_df)
        assert result_with.equals(result_without)

        # Proof the committed override is real and would matter if asked for
        # explicitly (same explicit-ask pattern predict_scenario now uses)
        result_with_explicit = pm_with_override.predict_series(
            forecast_df, dhw_schedule_override=pm_with_override._schedule.get("committed_override")
        )
        assert not result_with_explicit.equals(result_with)

    def test_explicit_per_call_override_takes_precedence_over_committed(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm.commit_dhw_schedule({"legionella": ("2026-06-25", 10)})
        ts = pd.date_range("2026-06-25 00:00", periods=24, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "temp_c": [10.0] * 24, "direct_radiation_wm2": [0.0] * 24})
        override = {"legionella": ("2026-06-25", 20)}
        result_a = pm.predict_series(forecast_df, dhw_schedule_override=override)
        # No explicit override: predict_series is override-BLIND — Task 6 removed its
        # silent fallback to committed_override, so result_b applies no override at all
        # (not the committed hour-10 one). result_a therefore differs because it alone
        # carries the explicitly-passed hour-20 boost.
        result_b = pm.predict_series(forecast_df)
        assert not result_a.equals(result_b)


class TestCommitDhwScheduleMergeSemantics:
    def test_legionella_then_comfort_boost_both_survive(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm.commit_dhw_schedule({"legionella": ("2026-08-04", 12)})
        pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 57.5)})
        assert pm._schedule["committed_override"] == {
            "legionella": ["2026-08-04", 12],
            "comfort_boost": ["2026-08-05", 14, 57.5],
        }

    def test_clearing_comfort_boost_leaves_legionella_intact(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm.commit_dhw_schedule({"legionella": ("2026-08-04", 12)})
        pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 57.5)})
        pm.commit_dhw_schedule({"comfort_boost": None})
        assert pm._schedule["committed_override"] == {"legionella": ["2026-08-04", 12]}

    def test_clearing_last_key_resets_to_none_not_empty_dict(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm.commit_dhw_schedule({"legionella": ("2026-08-04", 12)})
        pm.commit_dhw_schedule({"legionella": None})
        assert pm._schedule["committed_override"] is None

    def test_malformed_entry_skipped_with_warning_does_not_corrupt_merge(self, tmp_path, caplog):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm.commit_dhw_schedule({"legionella": ("2026-08-04", 12)})
        with caplog.at_level("WARNING"):
            pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14)})  # missing target_c — malformed
        assert any("malformed" in r.message for r in caplog.records)
        assert pm._schedule["committed_override"] == {"legionella": ["2026-08-04", 12]}

    def test_commit_appends_override_history_entry(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 57.5)})
        assert len(pm._schedule["override_history"]) == 1
        entry = pm._schedule["override_history"][0]
        assert entry["kind"] == "comfort_boost"
        assert entry["date"] == "2026-08-05"
        assert entry["hour"] == 14
        assert entry["target_c"] == 57.5
        assert entry["cancelled_at"] is None
        assert entry["committed_at"]  # non-empty ISO string

    def test_legionella_history_target_c_is_t_legionella(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm.commit_dhw_schedule({"legionella": ("2026-08-04", 12)})
        entry = pm._schedule["override_history"][0]
        assert entry["target_c"] == pm._schedule["T_legionella"]

    def test_last_write_wins_dedup_on_kind_date_hour_for_non_cancelled(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 55.0)})
        pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 57.5)})  # re-armed, revised target
        assert len(pm._schedule["override_history"]) == 1
        assert pm._schedule["override_history"][0]["target_c"] == 57.5

    def test_genuine_cancellation_marks_cancelled_at_not_deleted(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 57.5)})
        pm.commit_dhw_schedule({"comfort_boost": None})
        assert len(pm._schedule["override_history"]) == 1  # not deleted
        entry = pm._schedule["override_history"][0]
        assert entry["cancelled_at"] is not None
        assert entry["target_c"] == 57.5  # raw value retained

    def test_rearm_after_cancellation_appends_fresh_entry_not_uncancel(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 55.0)})
        pm.commit_dhw_schedule({"comfort_boost": None})  # cancelled
        pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, 58.0)})  # re-armed, same hour
        assert len(pm._schedule["override_history"]) == 2
        cancelled, fresh = pm._schedule["override_history"]
        assert cancelled["cancelled_at"] is not None
        assert cancelled["target_c"] == 55.0
        assert fresh["cancelled_at"] is None
        assert fresh["target_c"] == 58.0

    def test_comfort_boost_non_numeric_target_c_skipped_with_warning(self, tmp_path, caplog):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        with caplog.at_level("WARNING"):
            pm.commit_dhw_schedule({"comfort_boost": ("2026-08-05", 14, "not-a-number")})
        assert any("not numeric" in r.message for r in caplog.records)
        assert pm._schedule["committed_override"] is None
        assert pm._schedule["override_history"] == []


class TestOverrideHistorySchema:
    def test_default_schedule_has_empty_override_history(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        assert pm._schedule["override_history"] == []

    def test_override_history_persists_and_reloads(self, tmp_path):
        model_dir = tmp_path / "models"
        pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        pm._schedule["override_history"].append(
            {
                "kind": "legionella",
                "date": "2026-08-04",
                "hour": 12,
                "target_c": 60.0,
                "committed_at": "2026-08-04T06:00:01+02:00",
                "cancelled_at": None,
            }
        )
        _atomic_write_json(pm._schedule_path, pm._schedule)
        pm2 = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        assert pm2._schedule["override_history"] == [
            {
                "kind": "legionella",
                "date": "2026-08-04",
                "hour": 12,
                "target_c": 60.0,
                "committed_at": "2026-08-04T06:00:01+02:00",
                "cancelled_at": None,
            }
        ]


class TestDhwOverrideForHour:
    def test_legionella_override_returns_t_legionella(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        ts = pd.Timestamp("2026-08-04 12:00")
        result = pm._dhw_override_for_hour(ts, {"legionella": ("2026-08-04", 12)})
        assert result == pm._schedule["T_legionella"]

    def test_comfort_boost_override_returns_target_c(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        ts = pd.Timestamp("2026-08-05 14:00")
        result = pm._dhw_override_for_hour(ts, {"comfort_boost": ("2026-08-05", 14, 57.5)})
        assert result == 57.5

    def test_no_match_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        ts = pd.Timestamp("2026-08-05 09:00")
        assert pm._dhw_override_for_hour(ts, {"comfort_boost": ("2026-08-05", 14, 57.5)}) is None

    def test_same_hour_precedence_legionella_wins(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        ts = pd.Timestamp("2026-08-05 14:00")
        result = pm._dhw_override_for_hour(
            ts, {"legionella": ("2026-08-05", 14), "comfort_boost": ("2026-08-05", 14, 57.5)}
        )
        assert result == pm._schedule["T_legionella"]

    def test_empty_or_none_override_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        ts = pd.Timestamp("2026-08-05 14:00")
        assert pm._dhw_override_for_hour(ts, None) is None
        assert pm._dhw_override_for_hour(ts, {}) is None


class TestDhwOverrideForHourFromHistory:
    def test_returns_target_c_for_matching_non_cancelled_entry(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        history = [
            {
                "kind": "legionella",
                "date": "2026-08-04",
                "hour": 12,
                "target_c": 60.0,
                "committed_at": "x",
                "cancelled_at": None,
            },
        ]
        ts = pd.Timestamp("2026-08-04 12:00")
        assert pm._dhw_override_for_hour_from_history(ts, history) == 60.0

    def test_cancelled_entry_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        history = [
            {
                "kind": "comfort_boost",
                "date": "2026-08-05",
                "hour": 14,
                "target_c": 57.5,
                "committed_at": "x",
                "cancelled_at": "y",
            },
        ]
        ts = pd.Timestamp("2026-08-05 14:00")
        assert pm._dhw_override_for_hour_from_history(ts, history) is None

    def test_self_expired_entry_cancelled_at_none_reconstructs_normally(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        history = [
            {
                "kind": "comfort_boost",
                "date": "2026-08-05",
                "hour": 14,
                "target_c": 57.5,
                "committed_at": "x",
                "cancelled_at": None,
            },
        ]
        ts = pd.Timestamp("2026-08-05 14:00")
        assert pm._dhw_override_for_hour_from_history(ts, history) == 57.5

    def test_returns_the_actually_committed_value_not_just_latest(self, tmp_path):
        """Direct regression test for the reconstruction-fidelity bug (Goal 2):
        two different historical days' entries must each reconstruct correctly,
        not both resolve to whichever was committed most recently."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        history = [
            {
                "kind": "legionella",
                "date": "2026-07-28",
                "hour": 12,
                "target_c": 60.0,
                "committed_at": "x",
                "cancelled_at": None,
            },
            {
                "kind": "legionella",
                "date": "2026-08-04",
                "hour": 13,
                "target_c": 60.0,
                "committed_at": "y",
                "cancelled_at": None,
            },
        ]
        assert pm._dhw_override_for_hour_from_history(pd.Timestamp("2026-07-28 12:00"), history) == 60.0
        assert pm._dhw_override_for_hour_from_history(pd.Timestamp("2026-08-04 13:00"), history) == 60.0
        assert pm._dhw_override_for_hour_from_history(pd.Timestamp("2026-07-28 13:00"), history) is None

    def test_same_hour_precedence_legionella_wins(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        history = [
            {
                "kind": "legionella",
                "date": "2026-08-05",
                "hour": 14,
                "target_c": 60.0,
                "committed_at": "x",
                "cancelled_at": None,
            },
            {
                "kind": "comfort_boost",
                "date": "2026-08-05",
                "hour": 14,
                "target_c": 57.5,
                "committed_at": "y",
                "cancelled_at": None,
            },
        ]
        ts = pd.Timestamp("2026-08-05 14:00")
        assert pm._dhw_override_for_hour_from_history(ts, history) == 60.0

    def test_malformed_entry_skipped_with_warning_not_raised(self, tmp_path, caplog):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        history = [{"kind": "legionella", "date": "2026-08-04"}]  # missing hour/target_c
        ts = pd.Timestamp("2026-08-04 12:00")
        with caplog.at_level("WARNING"):
            result = pm._dhw_override_for_hour_from_history(ts, history)
        assert result is None
        assert any("malformed" in r.message for r in caplog.records)

    def test_empty_history_returns_none(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        assert pm._dhw_override_for_hour_from_history(pd.Timestamp("2026-08-04 12:00"), []) is None


class TestZeroOverrideDegradation:
    def test_empty_history_and_committed_override_produces_zero_delta_everywhere(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        timestamps = pd.date_range("2026-08-04 00:00", periods=48, freq="h")
        t_ambient = pd.Series(10.0, index=timestamps)
        training_delta = pm.compute_training_override_delta(timestamps, t_ambient, 45.0, [])
        serving_delta = pm.compute_serving_override_delta(timestamps, t_ambient, 45.0)
        assert (training_delta == 0.0).all()
        assert (serving_delta == 0.0).all()

    def test_physics_kwh_identical_to_pre_phase_a_when_no_override_ever_committed(self, tmp_path):
        """No regression risk for installs that never use DHW overrides — the
        override-blind physics_kwh feature must be bit-identical to calling
        _dhw_kwh_series with override_lookup=None (today's only behavior)."""
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        timestamps = pd.date_range("2026-08-04 00:00", periods=24, freq="h")
        t_ambient = pd.Series(10.0, index=timestamps)
        el_kwh_a, _, final_a = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=None)
        el_kwh_b, _, final_b = pm._dhw_kwh_series(timestamps, t_ambient, 45.0, override_lookup=None)
        pd.testing.assert_series_equal(el_kwh_a, el_kwh_b)
        assert final_a == final_b
