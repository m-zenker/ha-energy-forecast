"""App-level tests for the daily-rule heating projection (plan 2026-09-25-heating-season-projection)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
from energy_forecast.heating_season import HeatingThresholds


def _app(entity_state="off", label_source="entity", meter_entity=None):
    from energy_forecast.energy_forecast import EnergyForecast

    app = MagicMock()
    app._timezone = "Europe/Zurich"
    app._heating_active_entity = "input_boolean.heating"
    app.get_state = MagicMock(return_value=entity_state)
    app._heating_setpoint_on = 20.0
    app._heating_setpoint_off = 12.0
    app._heating_thresholds = HeatingThresholds(12.0, 16.0, "learned", label_source, 150, 0.05)
    app._physics_config = {"heating_sub_meter_sensor": meter_entity}
    for name in (
        "_sub_sensor_prefix",
        "_heating_prior_and_today",
        "_build_heating_active_projection",
        "_heating_season_attr",
    ):
        setattr(app, name, getattr(EnergyForecast, name).__get__(app))
    return app


def _forecast(fn) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
    ts = pd.date_range(now, periods=72, freq="1h")
    return pd.DataFrame({"timestamp": ts, "temp_c": [fn(t) for t in ts]})


def test_cold_nights_warm_days_project_off_without_climate():
    """Sept-2026 regression: entity off, nights 8 °C / days 22 °C -> no projected heating, climate_recent empty."""
    app = _app("off")
    series, s_on, s_off = app._build_heating_active_projection(_forecast(lambda t: 8.0 if t.hour < 9 else 22.0), {})
    assert len(series) == 48 and series.sum() == 0
    assert (s_on, s_off) == (20.0, 12.0)


def test_cold_forecast_switches_on_after_today():
    app = _app("off")
    series, _, _ = app._build_heating_active_projection(_forecast(lambda t: 4.0), {})
    today = series.index[0].normalize()
    assert series[series.index.normalize() == today].sum() == 0  # entity is off today
    assert (series[series.index.normalize() > today] == 1).all()


def test_projection_entity_unavailable():
    app = _app("unavailable")
    series, _, _ = app._build_heating_active_projection(_forecast(lambda t: 20.0), {})
    assert series.sum() == 0


def test_meter_tier_uses_meter_prior():
    meter_entity = "sensor.hp_heating_energy"
    app = _app("off", label_source="meter", meter_entity=meter_entity)
    now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None)
    ts = pd.date_range(now.normalize() - pd.Timedelta(days=1), now.floor("1h"), freq="1h")
    meter = pd.DataFrame({"timestamp": ts, "kwh": 0.2})  # heated yesterday and today
    series, _, _ = app._build_heating_active_projection(
        _forecast(lambda t: 14.0), {}, sub_sensors_recent={"sub_hp_heating_energy": meter}
    )
    assert (series == 1).all()  # today heated; dead-band 14 °C holds ON


def test_heating_season_attr():
    attr = _app()._heating_season_attr()
    assert attr == {
        "on_below": 12.0,
        "off_above": 16.0,
        "source": "learned",
        "label_source": "entity",
        "n_days": 150,
        "mismatch_rate": 0.05,
    }
