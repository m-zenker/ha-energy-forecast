"""Tests for dynamic setpoint projection via heating-active hysteresis.

Covers _project_indoor_temps() with heating_active_series + thermal_pressure
behaviour in _engineer_features(), plus a predict() smoke test.
"""

import numpy as np
import pandas as pd
import pytest

from apps.energy_forecast.model import (
    _engineer_features,
    _project_indoor_temps,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _future_ts(n: int = 48) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-15 10:00", periods=n, freq="1h")


def _outdoor_series(ts: pd.DatetimeIndex, temp: float = 5.0) -> pd.Series:
    return pd.Series([temp] * len(ts), index=ts)


def _climate_recent(ts: pd.DatetimeIndex, setpoint: float = 21.0, current: float = 19.0) -> dict:
    """One climate entity, observation at ts[0] (fresh)."""
    return {
        "climate.room": pd.DataFrame(
            {
                "timestamp": [ts[0]],
                "current_temp": [current],
                "setpoint": [setpoint],
            }
        )
    }


def _heating_active_series(ts: pd.DatetimeIndex, states: list[int]) -> pd.Series:
    return pd.Series(states, index=ts, dtype=int)


# ── _project_indoor_temps: setpoint projection ─────────────────────────────────


class TestSetpointProjection:
    def test_heating_on_uses_setpoint_on(self):
        """All hours ON → entire projected setpoint array == setpoint_on."""
        ts = _future_ts()
        outdoor = _outdoor_series(ts)
        cr = _climate_recent(ts, setpoint=12.0)  # current entity setpoint is 12 (off state)
        ha_series = _heating_active_series(ts, [1] * 48)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        sp = result["climate.room"]["setpoint"].values
        assert np.all(sp == pytest.approx(21.0)), "All hours ON → setpoint should be 21"

    def test_heating_off_uses_setpoint_off(self):
        """All hours OFF → entire projected setpoint array == setpoint_off."""
        ts = _future_ts()
        outdoor = _outdoor_series(ts)
        cr = _climate_recent(ts, setpoint=21.0)
        ha_series = _heating_active_series(ts, [0] * 48)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        sp = result["climate.room"]["setpoint"].values
        assert np.all(sp == pytest.approx(12.0)), "All hours OFF → setpoint should be 12"

    def test_on_to_off_transition(self):
        """Hours 0–5 ON, hours 6–47 OFF → setpoint switches at hour 6."""
        ts = _future_ts()
        outdoor = _outdoor_series(ts)
        cr = _climate_recent(ts, setpoint=21.0)
        states = [1] * 6 + [0] * 42
        ha_series = _heating_active_series(ts, states)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        sp = result["climate.room"]["setpoint"].values
        assert np.all(sp[:6] == pytest.approx(21.0))
        assert np.all(sp[6:] == pytest.approx(12.0))

    def test_off_to_on_transition(self):
        """Hours 0–9 OFF, hours 10–47 ON → setpoint switches at hour 10."""
        ts = _future_ts()
        outdoor = _outdoor_series(ts)
        cr = _climate_recent(ts, setpoint=12.0)
        states = [0] * 10 + [1] * 38
        ha_series = _heating_active_series(ts, states)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        sp = result["climate.room"]["setpoint"].values
        assert np.all(sp[:10] == pytest.approx(12.0))
        assert np.all(sp[10:] == pytest.approx(21.0))

    def test_no_entity_configured_flat_fallback(self):
        """When heating_active_series=None, setpoint is flat at latest entity value."""
        ts = _future_ts()
        outdoor = _outdoor_series(ts)
        cr = _climate_recent(ts, setpoint=19.5)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=None,
            setpoint_on=None,
            setpoint_off=None,
        )
        sp = result["climate.room"]["setpoint"].values
        assert np.all(sp == pytest.approx(19.5)), "Flat fallback should use entity setpoint"


# ── _project_indoor_temps: deficit blending (live → hysteresis) ────────────────


class TestDeficitBlending:
    """`deficit` blends the live-setpoint deficit (near-term) into the
    hysteresis-projected-setpoint deficit (far-term) over
    SENSOR_FULL_TRUST_HOURS..SENSOR_BLEND_HOURS, mirroring the outdoor-temp
    live/forecast blend in _build_prediction_temp_df."""

    def test_near_term_uses_live_setpoint_even_when_hysteresis_projects_on(self):
        """Live setpoint is 12 (e.g. summer/eco mode); hysteresis projects heating
        ON (setpoint_on=21) for every hour because outdoor temp is cold. The
        full-trust window (h <= SENSOR_FULL_TRUST_HOURS) must still be computed
        against the LIVE setpoint (12), not the hysteresis setpoint (21) — this
        is the bug this plan fixes."""
        ts = _future_ts(n=8)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=12.0, current=19.0)
        ha_series = _heating_active_series(ts, [1] * 8)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=1e6,  # effectively no cooling — indoor stays ~19 for all 8h
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        deficit = result["climate.room"]["deficit"].values
        assert np.allclose(deficit[:3], 0.0), (
            "Full-trust window (h=0..SENSOR_FULL_TRUST_HOURS) must use the live "
            "setpoint (12 < indoor 19 → deficit 0), not the hysteresis setpoint (21)"
        )
        t_in = result["climate.room"]["current_temp"].values
        assert deficit[7] == pytest.approx(max(0.0, 21.0 - t_in[7]), abs=1e-6), (
            "Beyond SENSOR_BLEND_HOURS the deficit must be purely hysteresis-based"
        )
        assert deficit[7] > 0

    def test_interpolates_between_full_trust_and_blend_hours(self):
        """Between SENSOR_FULL_TRUST_HOURS and SENSOR_BLEND_HOURS, deficit is a
        linear blend of the live-setpoint deficit and the hysteresis-setpoint
        deficit."""
        from apps.energy_forecast.const import SENSOR_BLEND_HOURS, SENSOR_FULL_TRUST_HOURS

        ts = _future_ts(n=8)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=12.0, current=19.0)
        ha_series = _heating_active_series(ts, [1] * 8)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=1e6,
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        t_in = result["climate.room"]["current_temp"].values
        deficit = result["climate.room"]["deficit"].values
        mid = (SENSOR_FULL_TRUST_HOURS + SENSOR_BLEND_HOURS) // 2  # hour 4
        alpha = (mid - SENSOR_FULL_TRUST_HOURS) / (SENSOR_BLEND_HOURS - SENSOR_FULL_TRUST_HOURS)
        live_deficit = max(0.0, 12.0 - t_in[mid])
        hyst_deficit = max(0.0, 21.0 - t_in[mid])
        expected = live_deficit * (1 - alpha) + hyst_deficit * alpha
        assert deficit[mid] == pytest.approx(expected, abs=1e-6)

    def test_setpoint_column_unaffected_by_blending(self):
        """The raw 'setpoint' column still reflects the pure hysteresis
        trajectory (kept for diagnostics/backward-compat) — only the new
        'deficit' column is blended."""
        ts = _future_ts(n=8)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=12.0, current=19.0)
        ha_series = _heating_active_series(ts, [1] * 8)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        sp = result["climate.room"]["setpoint"].values
        assert np.all(sp == pytest.approx(21.0))

    def test_no_hysteresis_configured_deficit_matches_flat_live_setpoint(self):
        """Without heating_active_series/setpoint_on/off, 'deficit' equals the
        flat live-setpoint deficit for every hour (blending is a no-op since
        live == hysteresis in this branch)."""
        ts = _future_ts(n=6)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=21.0, current=19.0)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=None,
            setpoint_on=None,
            setpoint_off=None,
        )
        t_in = result["climate.room"]["current_temp"].values
        deficit = result["climate.room"]["deficit"].values
        expected = np.maximum(0.0, 21.0 - t_in)
        np.testing.assert_allclose(deficit, expected, atol=1e-9)


# ── _engineer_features: prefers the precomputed 'deficit' column ───────────────


def _make_bare_df(timestamps) -> pd.DataFrame:
    """Minimal energy df with gross_kwh for _engineer_features input."""
    n = len(timestamps)
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "gross_kwh": [1.5] * n})


class TestThermalPressurePrefersDeficitColumn:
    def test_uses_deficit_column_when_present(self):
        """When climate_dfs carries a precomputed 'deficit' column (as produced
        by _project_indoor_temps' blending), _engineer_features must use it
        directly instead of recomputing (setpoint - current_temp)."""
        ts = pd.date_range("2026-01-15 10:00", periods=2, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        climate_dfs = {
            "climate.room": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": [19.0] * 2,
                    "setpoint": [21.0] * 2,  # would give delta=2.0 if recomputed
                    "deficit": [5.0] * 2,  # pre-blended value — must win
                }
            )
        }
        result = _engineer_features(df, w, None, climate_dfs=climate_dfs)
        np.testing.assert_allclose(result["thermal_pressure"].values, 5.0)

    def test_falls_back_to_setpoint_minus_current_when_no_deficit_column(self):
        """Historical/training-time climate_dfs (no 'deficit' column) keep the
        original setpoint-minus-current_temp calculation, unchanged."""
        ts = pd.date_range("2026-01-15 10:00", periods=2, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        climate_dfs = {
            "climate.room": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": [19.0] * 2,
                    "setpoint": [21.0] * 2,
                }
            )
        }
        result = _engineer_features(df, w, None, climate_dfs=climate_dfs)
        np.testing.assert_allclose(result["thermal_pressure"].values, 2.0)


# ── thermal_pressure via _engineer_features ────────────────────────────────────


def _make_weather_df(ts: pd.DatetimeIndex, temp: float = 5.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": [temp] * len(ts),
            "precipitation_mm": [0.0] * len(ts),
            "sunshine_min": [0.0] * len(ts),
            "wind_kmh": [0.0] * len(ts),
            "cloud_cover_pct": [0.0] * len(ts),
            "direct_radiation_wm2": [0.0] * len(ts),
            "humidity": [60.0] * len(ts),
        }
    )


class TestThermalPressureWithHysteresis:
    def test_pressure_zero_when_off_and_live_setpoint_agrees(self):
        """Setpoint 12°C live AND hysteresis-off, T_indoor ~19°C → genuinely no
        deficit anywhere → thermal_pressure = 0."""
        ts = _future_ts(n=6)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=12.0, current=19.0)
        ha_series = _heating_active_series(ts, [0] * 6)

        climate_dfs = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        future_df = pd.DataFrame({"timestamp": ts, "gross_kwh": [np.nan] * 6})
        weather_df = _make_weather_df(ts, temp=5.0)
        feat = _engineer_features(future_df, weather_df, None, climate_dfs=climate_dfs)
        assert np.allclose(feat["thermal_pressure"].values, 0.0), (
            "Live setpoint (12) and hysteresis-off setpoint (12) agree — no deficit should exist anywhere"
        )

    def test_live_deficit_not_masked_by_hysteresis_off(self):
        """Live setpoint 21°C (thermostat genuinely calling for heat) but
        hysteresis projects OFF (setpoint_off=12) because outdoor temp is mild.
        The full-trust window must surface the real, live deficit instead of
        the hysteresis model's zero — this is the bug from the original
        7.2 °C·m² sensor report (live summer setpoint overridden by a stale
        seasonal projection)."""
        ts = _future_ts(n=6)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=21.0, current=19.0)
        ha_series = _heating_active_series(ts, [0] * 6)

        climate_dfs = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        future_df = pd.DataFrame({"timestamp": ts, "gross_kwh": [np.nan] * 6})
        weather_df = _make_weather_df(ts, temp=5.0)
        feat = _engineer_features(future_df, weather_df, None, climate_dfs=climate_dfs)
        assert feat["thermal_pressure"].iloc[0] == pytest.approx(2.0, abs=1e-6), (
            "Hour 0 (published sensor value) must use the live setpoint (21) "
            "against indoor (~19), giving a real ~2.0 deficit — not 0"
        )

    def test_pressure_positive_when_on(self):
        """Setpoint 21 °C, T_indoor cooling from 19 °C → pressure > 0 for early hours."""
        ts = _future_ts(n=6)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=12.0, current=19.0)
        ha_series = _heating_active_series(ts, [1] * 6)

        climate_dfs = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        future_df = pd.DataFrame({"timestamp": ts, "gross_kwh": [np.nan] * 6})
        weather_df = _make_weather_df(ts, temp=5.0)
        feat = _engineer_features(future_df, weather_df, None, climate_dfs=climate_dfs)
        assert (feat["thermal_pressure"] > 0).any(), (
            "Heating ON → setpoint(21) > T_indoor(19) → thermal_pressure must be > 0"
        )

    def test_pressure_drops_after_off_transition(self):
        """Hours 0–5 ON (setpoint 21), hours 6–11 OFF (setpoint 12) → pressure drops at hour 6."""
        ts = _future_ts(n=12)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=21.0, current=18.0)
        states = [1] * 6 + [0] * 6
        ha_series = _heating_active_series(ts, states)

        climate_dfs = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=ha_series,
            setpoint_on=21.0,
            setpoint_off=12.0,
        )
        future_df = pd.DataFrame({"timestamp": ts, "gross_kwh": [np.nan] * 12})
        weather_df = _make_weather_df(ts, temp=5.0)
        feat = _engineer_features(future_df, weather_df, None, climate_dfs=climate_dfs)

        pressure = feat["thermal_pressure"].values
        # First 6 hours: ON → pressure should be positive (setpoint 21 > T_indoor ~18)
        assert (pressure[:6] > 0).all(), "ON hours should have positive pressure"
        # Last 6 hours: OFF → setpoint 12 < T_indoor (still ~18) → pressure should be 0
        assert np.allclose(pressure[6:], 0.0), "OFF hours should have zero pressure"


# ── Hysteresis projection via _build_heating_active_projection ─────────────────


class TestHeatingActiveProjectionHysteresis:
    """Test the outdoor-temp hysteresis logic directly."""

    def _run_hysteresis(
        self,
        temps: list[float],
        initial_state: int,
        temp_on: float = 14.0,
        temp_off: float = 18.0,
    ) -> list[int]:
        """Replicate the hysteresis loop from _build_heating_active_projection."""
        state = initial_state
        result = []
        for t in temps:
            if t < temp_on:
                state = 1
            elif t > temp_off:
                state = 0
            result.append(state)
        return result

    def test_rising_temp_turns_off(self):
        """Temp rises above temp_off → state switches to OFF."""
        temps = [10.0] * 5 + [20.0] * 5  # rises above 18 at index 5
        states = self._run_hysteresis(temps, initial_state=1)
        assert all(s == 1 for s in states[:5])
        assert all(s == 0 for s in states[5:])

    def test_falling_temp_turns_on(self):
        """Temp falls below temp_on → state switches to ON."""
        temps = [20.0] * 5 + [10.0] * 5  # falls below 14 at index 5
        states = self._run_hysteresis(temps, initial_state=0)
        assert all(s == 0 for s in states[:5])
        assert all(s == 1 for s in states[5:])

    def test_dead_band_holds_state_on(self):
        """Temp stays in dead band (14–18) → initial ON state is preserved."""
        temps = [16.0] * 20
        states = self._run_hysteresis(temps, initial_state=1)
        assert all(s == 1 for s in states)

    def test_dead_band_holds_state_off(self):
        """Temp stays in dead band (14–18) → initial OFF state is preserved."""
        temps = [16.0] * 20
        states = self._run_hysteresis(temps, initial_state=0)
        assert all(s == 0 for s in states)

    def test_multiple_transitions(self):
        """Verify multiple on/off cycles are all handled correctly."""
        # cold → warm → cold → warm
        temps = [10.0] * 4 + [20.0] * 4 + [10.0] * 4 + [20.0] * 4
        states = self._run_hysteresis(temps, initial_state=0)
        assert all(s == 1 for s in states[0:4])  # cold → ON
        assert all(s == 0 for s in states[4:8])  # warm → OFF
        assert all(s == 1 for s in states[8:12])  # cold → ON
        assert all(s == 0 for s in states[12:16])  # warm → OFF
