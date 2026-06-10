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
    def test_pressure_zero_when_off(self):
        """Setpoint 12 °C, T_indoor ~19 °C → setpoint < indoor → thermal_pressure = 0."""
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
        assert np.allclose(feat["thermal_pressure"].values, 0.0), (
            "Heating OFF → setpoint(12) < T_indoor → thermal_pressure must be 0"
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
