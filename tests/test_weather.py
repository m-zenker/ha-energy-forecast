"""Tests for weather.py fetch functions.

Covers:
  - fetch_open_meteo: sunshine_duration parsed and converted to minutes,
    graceful fallback when the field is absent, column contract, network errors
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from energy_forecast import weather


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_response(hourly: dict) -> MagicMock:
    """Build a mock requests.Response returning the given hourly dict."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"hourly": hourly}
    return mock


def _base_hourly(n: int = 4) -> dict:
    return {
        "time":             [f"2026-03-12T{h:02d}:00" for h in range(n)],
        "temperature_2m":   [5.0] * n,
        "precipitation":    [0.0] * n,
        "windspeed_10m":    [10.0] * n,
    }


# ── fetch_open_meteo ──────────────────────────────────────────────────────────

class TestFetchOpenMeteo:

    def test_sunshine_duration_converted_to_minutes(self):
        """3600 seconds input must produce 60.0 minutes in sunshine_min."""
        hourly = {**_base_hourly(3), "sunshine_duration": [3600, 1800, 0]}
        with patch("requests.get", return_value=_make_response(hourly)):
            df = weather.fetch_open_meteo(47.0, 8.0)
        assert list(df["sunshine_min"]) == [60.0, 30.0, 0.0]

    def test_sunshine_populated_nonzero(self):
        """Regression: sunshine_min must not be a flat zero when API provides data."""
        hourly = {**_base_hourly(4), "sunshine_duration": [1200, 2400, 600, 0]}
        with patch("requests.get", return_value=_make_response(hourly)):
            df = weather.fetch_open_meteo(47.0, 8.0)
        assert df["sunshine_min"].sum() > 0

    def test_sunshine_missing_key_defaults_to_zero(self):
        """If sunshine_duration is absent from the response, sunshine_min defaults to 0."""
        hourly = _base_hourly(3)  # no sunshine_duration key
        with patch("requests.get", return_value=_make_response(hourly)):
            df = weather.fetch_open_meteo(47.0, 8.0)
        assert (df["sunshine_min"] == 0).all()

    def test_returns_expected_columns(self):
        """DataFrame must contain all four expected columns with consistent length."""
        n = 5
        hourly = {**_base_hourly(n), "sunshine_duration": [300] * n}
        with patch("requests.get", return_value=_make_response(hourly)):
            df = weather.fetch_open_meteo(47.0, 8.0)
        assert set(df.columns) >= {"timestamp", "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh"}
        assert len(df) == n

    def test_network_error_returns_empty_dataframe(self):
        """On RequestException fetch_open_meteo must return an empty DataFrame."""
        import requests as req
        with patch("requests.get", side_effect=req.RequestException("timeout")):
            df = weather.fetch_open_meteo(47.0, 8.0)
        assert df.empty

    def test_url_contains_sunshine_duration(self):
        """The outgoing URL must request sunshine_duration from the API."""
        hourly = {**_base_hourly(2), "sunshine_duration": [0, 0]}
        with patch("requests.get", return_value=_make_response(hourly)) as mock_get:
            weather.fetch_open_meteo(47.0, 8.0)
        called_url = mock_get.call_args[0][0]
        assert "sunshine_duration" in called_url
