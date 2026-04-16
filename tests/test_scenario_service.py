"""Tests for the Stage 4 scenario/what-if service callback.

Covers:
  - _get_scenario_cb: no cache → logs WARNING, returns early
  - _get_scenario_cb: valid cache → fire_event called with forecast data
  - _get_scenario_cb: publish=True → _publish_scenario_forecast called
  - _get_scenario_cb: predict_scenario raises → ERROR logged, fire_event not called
  - _get_scenario_cb: invalid schedule type → WARNING logged, fire_event not called
  - _publish_scenario_forecast: set_state call count, entity IDs, value correctness
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ── Minimal stand-in for EnergyForecast ──────────────────────────────────────

def _make_app(cached_df=None):
    """Build a minimal fake EnergyForecast instance for callback testing."""
    app = MagicMock()
    app._cached_forecast_df    = cached_df
    app._cached_live_temp      = 12.0
    app._cached_recent_actuals = None
    app._cached_sub_sensors    = None
    app._cached_away_series    = None
    app._cached_people_home    = None
    app._cached_climate_recent = None
    app._cached_dhw_recent     = None
    app._timezone              = "Europe/Zurich"
    return app


def _make_baseline_df(start="2024-06-01 10:00", n=48):
    ts = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"timestamp": ts, "predicted_kwh": [1.0] * n})


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestGetScenarioCb:

    def test_no_cache_logs_warning(self, caplog):
        """When _cached_forecast_df is None, should log WARNING and return without error."""
        import logging
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app(cached_df=None)
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._get_scenario_cb(app, "homeassistant", "energy_forecast", "get_scenario", {})

        assert any("before first" in r.message for r in caplog.records), \
            "Expected a WARNING log when cache is None"
        app.fire_event.assert_not_called()

    def test_fires_result_event_with_forecast(self):
        """With valid cache, fire_event should be called with 'forecast' payload."""
        from energy_forecast.energy_forecast import EnergyForecast

        cached_df = _make_baseline_df()
        app = _make_app(cached_df=cached_df)

        # Build a scenario result df (baseline + delta=0 for empty schedule)
        scenario_result = cached_df.copy()
        scenario_result["delta_kwh"] = 0.0
        app._ml_model.predict_scenario.return_value = scenario_result

        EnergyForecast._get_scenario_cb(
            app, "homeassistant", "energy_forecast", "get_scenario",
            {"schedule": {}, "publish": False},
        )

        app.fire_event.assert_called_once()
        event_name, kwargs = app.fire_event.call_args[0][0], app.fire_event.call_args[1]
        assert event_name == "energy_forecast_scenario_result"
        assert "forecast" in kwargs
        records = kwargs["forecast"]
        assert isinstance(records, list)
        assert len(records) == 48

    def test_publish_flag_calls_publish_method(self):
        """publish=True should call _publish_scenario_forecast."""
        from energy_forecast.energy_forecast import EnergyForecast

        cached_df = _make_baseline_df()
        app = _make_app(cached_df=cached_df)

        scenario_result = cached_df.copy()
        scenario_result["delta_kwh"] = 0.0
        app._ml_model.predict_scenario.return_value = scenario_result

        EnergyForecast._get_scenario_cb(
            app, "homeassistant", "energy_forecast", "get_scenario",
            {"schedule": {"sub_dw": "14:00"}, "publish": True},
        )

        app._publish_scenario_forecast.assert_called_once_with(scenario_result)
        app.fire_event.assert_called_once()


class TestGetScenarioCbErrors:

    def test_predict_scenario_exception_logged(self, caplog):
        """When predict_scenario raises, ERROR must be logged and fire_event not called."""
        import logging
        from energy_forecast.energy_forecast import EnergyForecast

        cached_df = _make_baseline_df()
        app = _make_app(cached_df=cached_df)
        app._ml_model.predict_scenario.side_effect = ValueError("boom")

        with caplog.at_level(logging.ERROR, logger="energy_forecast"):
            EnergyForecast._get_scenario_cb(
                app, "homeassistant", "energy_forecast", "get_scenario",
                {"schedule": {}, "publish": False},
            )

        assert any(r.levelno == logging.ERROR for r in caplog.records), \
            "Expected ERROR log when predict_scenario raises"
        app.fire_event.assert_not_called()

    def test_invalid_schedule_type_logs_warning(self, caplog):
        """Passing schedule as a non-dict should log WARNING and not fire_event."""
        import logging
        from energy_forecast.energy_forecast import EnergyForecast

        cached_df = _make_baseline_df()
        app = _make_app(cached_df=cached_df)

        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._get_scenario_cb(
                app, "homeassistant", "energy_forecast", "get_scenario",
                {"schedule": "not-a-dict", "publish": False},
            )

        assert any(r.levelno == logging.WARNING for r in caplog.records), \
            "Expected WARNING log for invalid schedule type"
        app.fire_event.assert_not_called()


class TestPublishScenarioForecast:

    def _make_result_df(self, value_kwh: float = 1.0):
        """Build a 48-row result_df anchored to today midnight (naive UTC)."""
        today = pd.Timestamp.now(tz="Europe/Zurich").normalize().tz_localize(None)
        ts = pd.date_range(today, periods=48, freq="1h")
        return pd.DataFrame({
            "timestamp": ts,
            "predicted_kwh": [value_kwh] * 48,
            "delta_kwh": [0.1] * 48,
        })

    def test_set_state_called_correct_count(self):
        """_publish_scenario_forecast should call set_state exactly 11 times."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app()
        result_df = self._make_result_df()

        EnergyForecast._publish_scenario_forecast(app, result_df)

        assert app.set_state.call_count == 11

    def test_sensor_ids_match_expected_pattern(self):
        """set_state must be called for the 3 summary sensors and all 8 block sensors."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app()
        result_df = self._make_result_df()

        EnergyForecast._publish_scenario_forecast(app, result_df)

        called_ids = {c.args[0] for c in app.set_state.call_args_list}
        expected_ids = {
            "sensor.energy_forecast_scenario_today",
            "sensor.energy_forecast_scenario_tomorrow",
            "sensor.energy_forecast_scenario_delta_today",
            "sensor.energy_forecast_scenario_today_00_03",
            "sensor.energy_forecast_scenario_today_03_06",
            "sensor.energy_forecast_scenario_today_06_09",
            "sensor.energy_forecast_scenario_today_09_12",
            "sensor.energy_forecast_scenario_today_12_15",
            "sensor.energy_forecast_scenario_today_15_18",
            "sensor.energy_forecast_scenario_today_18_21",
            "sensor.energy_forecast_scenario_today_21_24",
        }
        assert called_ids == expected_ids

    def test_state_values_are_correct_sums(self):
        """Sensor states must equal rounded sums of predicted_kwh over each window."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app()
        # 24 today-hours + 24 tomorrow-hours; each hour = 2.0 kWh
        result_df = self._make_result_df(value_kwh=2.0)

        EnergyForecast._publish_scenario_forecast(app, result_df)

        states = {c.args[0]: c.kwargs["state"] for c in app.set_state.call_args_list}
        # 24 hours × 2.0 kWh = 48.0
        assert states["sensor.energy_forecast_scenario_today"]    == "48.0"
        assert states["sensor.energy_forecast_scenario_tomorrow"] == "48.0"
        # Each 3-hour block: 3 × 2.0 = 6.0
        assert states["sensor.energy_forecast_scenario_today_00_03"] == "6.0"
        assert states["sensor.energy_forecast_scenario_today_21_24"] == "6.0"
