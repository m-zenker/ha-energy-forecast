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


def _history(ts0: pd.Timestamp, rows: list[tuple[float, float]]) -> dict:
    """climate_recent for one room: hourly (current_temp, setpoint) rows ending at ts0 (last row = ts0)."""
    n = len(rows)
    stamps = pd.date_range(end=ts0, periods=n, freq="1h")
    return {
        "climate.room": pd.DataFrame(
            {"timestamp": stamps, "current_temp": [r[0] for r in rows], "setpoint": [r[1] for r in rows]}
        )
    }


# ── _project_indoor_temps: setpoint projection ─────────────────────────────────


class TestSetpointProjection:
    def test_heating_on_at_hour_zero_uses_room_setpoint(self):
        """Heating already ON at hour 0 → room's own live setpoint.

        (Production passes setpoint_on = live mean in this state anyway.)
        """
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
        assert np.all(sp == pytest.approx(12.0)), "ON at hour 0 → room setpoint (12)"

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

    def test_near_term_uses_live_setpoint_not_hysteresis_setpoint(self):
        """Live setpoint is 12 (e.g. summer/eco mode); hysteresis setpoint (21)
        differs from live for every hour. The
        full-trust window (h <= SENSOR_FULL_TRUST_HOURS) must still be computed
        against the LIVE setpoint (12), not the hysteresis setpoint (21) — this
        is the bug this plan fixes."""
        from apps.energy_forecast.const import SENSOR_BLEND_HOURS, SENSOR_FULL_TRUST_HOURS

        ts = _future_ts(n=8)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=12.0, current=19.0)
        ha_series = _heating_active_series(ts, [0] * 8)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=1e6,  # effectively no cooling — indoor stays ~19 for all 8h
            heating_active_series=ha_series,
            # Deliberate swap: heating is off, so setpoint_off=21 plays the hysteresis-projected setpoint (live 12 vs 21 diverge).
            setpoint_on=12.0,
            setpoint_off=21.0,
        )
        deficit = result["climate.room"]["deficit"].values
        assert np.allclose(deficit[: SENSOR_FULL_TRUST_HOURS + 1], 0.0), (
            "Full-trust window (h=0..SENSOR_FULL_TRUST_HOURS) must use the live "
            "setpoint (12 < indoor 19 → deficit 0), not the hysteresis setpoint (21)"
        )
        t_in = result["climate.room"]["current_temp"].values
        far = SENSOR_BLEND_HOURS + 1
        assert deficit[far] == pytest.approx(max(0.0, 21.0 - t_in[far]), abs=1e-6), (
            "Beyond SENSOR_BLEND_HOURS the deficit must be purely hysteresis-based"
        )
        assert deficit[far] > 0

    def test_interpolates_between_full_trust_and_blend_hours(self):
        """Between SENSOR_FULL_TRUST_HOURS and SENSOR_BLEND_HOURS, deficit is a
        linear blend of the live-setpoint deficit and the hysteresis-setpoint
        deficit."""
        from apps.energy_forecast.const import SENSOR_BLEND_HOURS, SENSOR_FULL_TRUST_HOURS

        ts = _future_ts(n=8)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=12.0, current=19.0)
        ha_series = _heating_active_series(ts, [0] * 8)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=1e6,
            heating_active_series=ha_series,
            # Deliberate swap: heating is off, so setpoint_off=21 plays the hysteresis-projected setpoint (live 12 vs 21 diverge).
            setpoint_on=12.0,
            setpoint_off=21.0,
        )
        t_in = result["climate.room"]["current_temp"].values
        deficit = result["climate.room"]["deficit"].values
        mid = (SENSOR_FULL_TRUST_HOURS + SENSOR_BLEND_HOURS) // 2  # hour 4
        alpha = (mid - SENSOR_FULL_TRUST_HOURS) / (SENSOR_BLEND_HOURS - SENSOR_FULL_TRUST_HOURS)
        live_deficit = max(0.0, 12.0 - t_in[mid])
        hyst_deficit = max(0.0, 21.0 - t_in[mid])
        expected = live_deficit * (1 - alpha) + hyst_deficit * alpha
        assert deficit[mid] == pytest.approx(expected, abs=1e-6)

    def test_alpha_reaches_one_exactly_at_blend_hours(self):
        """At h == SENSOR_BLEND_HOURS exactly, alpha must equal 1.0 — deficit is
        purely the hysteresis-projected value, with no live-setpoint contribution."""
        from apps.energy_forecast.const import SENSOR_BLEND_HOURS

        ts = _future_ts(n=8)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=12.0, current=19.0)
        ha_series = _heating_active_series(ts, [0] * 8)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=1e6,
            heating_active_series=ha_series,
            # Deliberate swap: heating is off, so setpoint_off=21 plays the hysteresis-projected setpoint (live 12 vs 21 diverge).
            setpoint_on=12.0,
            setpoint_off=21.0,
        )
        t_in = result["climate.room"]["current_temp"].values
        deficit = result["climate.room"]["deficit"].values
        expected = max(0.0, 21.0 - t_in[SENSOR_BLEND_HOURS])
        assert deficit[SENSOR_BLEND_HOURS] == pytest.approx(expected, abs=1e-6)

    def test_setpoint_column_unaffected_by_blending(self):
        """The raw 'setpoint' column still reflects the pure hysteresis
        trajectory (kept for diagnostics/backward-compat) — only the new
        'deficit' column is blended."""
        ts = _future_ts(n=8)
        outdoor = _outdoor_series(ts, temp=5.0)
        cr = _climate_recent(ts, setpoint=12.0, current=19.0)
        ha_series = _heating_active_series(ts, [0] * 8)

        result = _project_indoor_temps(
            cr,
            ts,
            outdoor,
            tau_hours=24.0,
            heating_active_series=ha_series,
            # Deliberate swap: heating is off, so setpoint_off=21 plays the hysteresis-projected setpoint (live 12 vs 21 diverge).
            setpoint_on=12.0,
            setpoint_off=21.0,
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
        hysteresis projects OFF (setpoint_off=12) because outdoor temp is mild
        — the mirror-image direction of the original bug report (there, live
        was LOW and hysteresis projected HIGH; see test_pressure_positive_when_on
        for that exact scenario). This test guards the opposite direction: the
        full-trust window must surface a real live deficit instead of the
        hysteresis model's zero, regardless of which way the mismatch runs."""
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
        """Live setpoint 12°C (e.g. actual summer/eco setpoint) but the hysteresis
        setpoint (21) differs from live — the real-world direction of the original bug report
        (live setpoint low, hysteresis projecting a high heating-season setpoint).
        Hour 0 must show NO phantom deficit (live 12 < indoor 19 → 0); later
        hours transition to the hysteresis-projected setpoint and go positive
        once the blend hands off to it."""
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
            setpoint_on=12.0,
            setpoint_off=21.0,
        )
        future_df = pd.DataFrame({"timestamp": ts, "gross_kwh": [np.nan] * 6})
        weather_df = _make_weather_df(ts, temp=5.0)
        feat = _engineer_features(future_df, weather_df, None, climate_dfs=climate_dfs)
        pressure = feat["thermal_pressure"].values
        assert pressure[0] == pytest.approx(0.0, abs=1e-6), (
            "Hour 0 must use the live setpoint (12 < indoor 19) — no phantom "
            "deficit despite the hysteresis-projected setpoint (21) being higher"
        )
        assert pressure[5] > 0, (
            "By hour 5 (blend window) the hysteresis-projected setpoint (21) should dominate, producing a real deficit"
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


class TestHeatingAwareProjection:
    """Rooms are thermostat-controlled: while heating is on they don't cool toward outdoor (2026-10-15)."""

    def _proj(self, cr, ts, states, setpoint_on=19.4, setpoint_off=12.0, t_out=5.0):
        return _project_indoor_temps(
            cr,
            ts,
            _outdoor_series(ts, t_out),
            tau_hours=11.64,
            heating_active_series=_heating_active_series(ts, states),
            setpoint_on=setpoint_on,
            setpoint_off=setpoint_off,
        )["climate.room"]

    def test_heating_on_holds_room_at_setpoint(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=21.0, current=21.0), ts, [1] * 48)
        assert p["current_temp"].min() == pytest.approx(21.0)
        assert p["deficit"].max() == pytest.approx(0.0)

    def test_heating_on_holds_initial_deficit(self):
        """A deficit seen at the SAME setpoint over the last 24 h is real demand and is held (deficit fade keeps d_ss)."""
        ts = _future_ts()
        p = self._proj(_history(ts[0], [(20.5, 21.0)] * 24), ts, [1] * 48)
        assert p["deficit"].values == pytest.approx([0.5] * 48)

    def test_per_room_setpoint_used_when_on_at_hour_zero(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=18.0, current=18.0), ts, [1] * 48, setpoint_on=19.4)
        assert np.all(p["setpoint"].values == pytest.approx(18.0))
        assert p["deficit"].max() == pytest.approx(0.0)  # no fake 1.4 °C deficit vs the shared mean

    def test_frost_mode_room_stays_at_own_setpoint(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=7.0, current=15.0), ts, [1] * 48)
        assert p["deficit"].max() == pytest.approx(0.0)

    def test_warm_room_cools_to_setpoint_then_holds(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=18.0, current=22.0), ts, [1] * 48)
        t = p["current_temp"].values
        assert t[1] < t[0]
        assert t.min() == pytest.approx(18.0)

    def test_heating_off_still_free_floats(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=21.0, current=21.0), ts, [0] * 48)
        t = p["current_temp"].values
        assert t[-1] < 10.0  # tau 11.6 h, T_out 5 → well below setpoint after 48 h

    def test_off_then_on_uses_shared_setpoint_on_with_zero_d0(self):
        ts = _future_ts()
        # room below its own setpoint at hour 0 (19 vs 20) — that d0 must NOT leak into the later setpoint_on floor
        p = self._proj(_climate_recent(ts, setpoint=20.0, current=19.0), ts, [0] * 10 + [1] * 38, setpoint_on=21.0)
        assert np.all(p["setpoint"].values[10:] == pytest.approx(21.0))
        assert p["current_temp"].values[10:].min() == pytest.approx(21.0)  # floor = setpoint_on − 0
        assert p["current_temp"].values[1:10].max() < 19.0  # off hours still free-float

    def test_no_series_is_thermostat_controlled(self):
        ts = _future_ts()
        p = _project_indoor_temps(
            _climate_recent(ts, setpoint=20.0, current=20.0), ts, _outdoor_series(ts, 0.0), tau_hours=11.64
        )["climate.room"]
        assert p["current_temp"].min() == pytest.approx(20.0)

    def test_stale_observation_starts_at_setpoint_with_zero_deficit(self):
        ts = _future_ts()
        cr = _climate_recent(ts, setpoint=21.0, current=15.0)
        cr["climate.room"]["timestamp"] = [ts[0] - pd.Timedelta(hours=3)]  # > 2 h old -> ignored
        p = self._proj(cr, ts, [1] * 48)
        assert p["current_temp"].values[0] == pytest.approx(21.0)
        assert np.all(p["deficit"].values == pytest.approx(0.0))  # d0 = 0


class TestDeficitFade:
    """Transient hour-0 deficits fade with DEFICIT_RECOVERY_TAU_H; persistent same-setpoint deficits are kept (2026-10-19)."""

    def _proj(self, cr, ts, states=None, setpoint_on=19.4, setpoint_off=12.0, t_out=5.0):
        return _project_indoor_temps(
            cr,
            ts,
            _outdoor_series(ts, t_out),
            tau_hours=11.64,
            heating_active_series=_heating_active_series(ts, states or [1] * len(ts)),
            setpoint_on=setpoint_on,
            setpoint_off=setpoint_off,
        )["climate.room"]

    def test_transient_step_up_deficit_fades(self):
        from apps.energy_forecast.const import DEFICIT_RECOVERY_TAU_H

        ts = _future_ts()
        # 23 h at 21/21 (no deficit), then the latest hour 0.5 K below → transient
        p = self._proj(_history(ts[0], [(21.0, 21.0)] * 23 + [(20.5, 21.0)]), ts)
        d = p["deficit"].values
        assert d[0] == pytest.approx(0.5)
        assert d[1] == pytest.approx(0.5 * np.exp(-1 / DEFICIT_RECOVERY_TAU_H), abs=1e-6)
        assert np.all(np.diff(d) <= 1e-12)
        assert d[24] == pytest.approx(0.5 * np.exp(-24 / DEFICIT_RECOVERY_TAU_H), abs=1e-6)

    def test_morning_step_up_with_yesterdays_daytime_history_fades(self):
        """Real 10-17 05 h shape: yesterday 06–22 h at 21/21, setback 19 overnight, setpoint back to 21 with the
        room 0.5 K below → 18 same-setpoint rows, median deficit 0 → transient fades (live: wohnzimmer n=17)."""
        ts = _future_ts()
        p = self._proj(_history(ts[0], [(21.0, 21.0)] * 17 + [(20.0, 19.0)] * 6 + [(20.5, 21.0)]), ts)
        assert p["deficit"].values[0] == pytest.approx(0.5)
        assert p["deficit"].values[24] < 0.01

    def test_too_few_same_setpoint_rows_fades_to_zero(self):
        from apps.energy_forecast.const import DEFICIT_RECOVERY_TAU_H

        ts = _future_ts()
        # night setback 19 (room at 20), setpoint just raised to 21 with the room at 20.5:
        # only 1 row at setpoint 21 (< DEFICIT_STEADY_MIN_ROWS) → d_ss = 0
        p = self._proj(_history(ts[0], [(20.0, 19.0)] * 6 + [(20.5, 21.0)]), ts)
        assert p["deficit"].values[0] == pytest.approx(0.5)
        assert p["deficit"].values[24] == pytest.approx(0.5 * np.exp(-24 / DEFICIT_RECOVERY_TAU_H), abs=1e-6)

    def test_persistent_deficit_at_same_setpoint_is_kept(self):
        ts = _future_ts()
        p = self._proj(_history(ts[0], [(20.0, 21.0)] * 24), ts)
        assert p["deficit"].values == pytest.approx([1.0] * 48)

    def test_current_better_than_usual_relaxes_to_steady_deficit(self):
        ts = _future_ts()
        # usually 1 K below at setpoint 21, right now on target → floor relaxes to the usual 1 K
        p = self._proj(_history(ts[0], [(20.0, 21.0)] * 23 + [(21.0, 21.0)]), ts)
        d = p["deficit"].values
        assert d[0] == pytest.approx(0.0)
        assert d[24] == pytest.approx(1.0, abs=0.01)

    def test_setpoint_change_ignores_other_setpoint_history(self):
        ts = _future_ts()
        # daytime 0.5 K deficit at setpoint 21; setback just started: setpoint 17, room at 20.5 (above it)
        p = self._proj(_history(ts[0], [(20.5, 21.0)] * 20 + [(20.5, 17.0)]), ts)
        assert p["deficit"].max() == pytest.approx(0.0)  # no phantom 0.5 K against the new setpoint

    def test_stale_observation_has_no_steady_deficit(self):
        ts = _future_ts()
        cr = _history(ts[0] - pd.Timedelta(hours=3), [(20.0, 21.0)] * 24)  # latest row 3 h old → stale
        p = self._proj(cr, ts)
        assert p["current_temp"].values[0] == pytest.approx(21.0)
        assert np.all(p["deficit"].values == pytest.approx(0.0))

    def test_trailing_nan_row_uses_last_valid_observation(self):
        ts = _future_ts()
        cr = _history(ts[0], [(21.0, 21.0)] * 22 + [(20.5, 21.0), (np.nan, np.nan)])
        p = self._proj(cr, ts)
        assert not p.isna().any().any()
        assert p["current_temp"].values[0] == pytest.approx(20.5)  # last valid row is 1 h old → fresh
        assert p["deficit"].values[0] == pytest.approx(0.5)

    def test_all_nan_rows_room_kept_as_nan_frame(self):
        ts = _future_ts()
        cr = _history(ts[0], [(np.nan, np.nan)] * 5)
        out = _project_indoor_temps(cr, ts, _outdoor_series(ts), tau_hours=11.64)
        p = out["climate.room"]  # kept: thermal_pressure's area denominator must match training
        assert len(p) == len(ts)
        assert p[["current_temp", "setpoint", "deficit"]].isna().all().all()

    def test_hour_zero_unchanged_by_fade(self, monkeypatch):
        import apps.energy_forecast.model as model_mod

        ts = _future_ts()
        cases = [
            _history(ts[0], [(21.0, 21.0)] * 23 + [(20.5, 21.0)]),
            _history(ts[0], [(20.0, 21.0)] * 24),
            _history(ts[0], [(20.5, 21.0)] * 20 + [(20.5, 17.0)]),
            _history(ts[0], [(22.0, 18.0)] * 24),
        ]
        for cr in cases:
            faded = self._proj(cr, ts)
            monkeypatch.setattr(model_mod, "DEFICIT_RECOVERY_TAU_H", float("inf"))
            held = self._proj(cr, ts)
            monkeypatch.undo()
            assert faded["current_temp"].values[0] == pytest.approx(held["current_temp"].values[0])
            assert faded["deficit"].values[0] == pytest.approx(held["deficit"].values[0])
        # the replay hook really switches behaviour (fails if the constant were bound as a default argument)
        cr = cases[0]
        monkeypatch.setattr(model_mod, "DEFICIT_RECOVERY_TAU_H", float("inf"))
        assert self._proj(cr, ts)["deficit"].values[24] == pytest.approx(0.5)
        monkeypatch.undo()
        assert self._proj(cr, ts)["deficit"].values[24] < 0.01
