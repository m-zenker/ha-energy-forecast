"""Tests for the Stage 4 scenario/what-if service callback.

Covers:
  - _get_scenario_cb: no cache → logs WARNING, returns early
  - _get_scenario_cb: valid cache → fire_event called with forecast data
  - _get_scenario_cb: publish=True → _publish_scenario_forecast called
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

    def test_no_cache_logs_warning(self):
        """When _cached_forecast_df is None, should log WARNING and return without error."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app(cached_df=None)
        EnergyForecast._get_scenario_cb(app, "homeassistant", "energy_forecast", "get_scenario", {})

        # Should have logged a WARNING
        warning_calls = [
            call for call in app.log.call_args_list
            if call.kwargs.get("level") == "WARNING" or (
                len(call.args) >= 1 and "before first" in str(call.args[0])
            )
        ]
        assert warning_calls, "Expected a WARNING log when cache is None"
        # fire_event must NOT have been called
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
