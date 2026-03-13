"""Tests for weather.py fetch functions.

Covers:
  - fetch_open_meteo: sunshine_duration, cloud_cover_pct, direct_radiation_wm2,
    past_days=3, graceful fallbacks, column contract, network errors
  - fetch_historical_weather: cloud_cover_pct and direct_radiation_wm2 columns
  - _supplement_from_open_meteo: historical tail prepended, SRG values preserved,
    cloud/radiation filled from Open-Meteo
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import pandas as pd

from energy_forecast import weather
from energy_forecast.weather import _supplement_from_open_meteo


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


def _full_hourly(n: int = 4) -> dict:
    """Base hourly dict including all new fields."""
    return {
        **_base_hourly(n),
        "sunshine_duration":  [1800] * n,
        "cloud_cover":        [50]   * n,
        "direct_radiation":   [200]  * n,
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

    def test_url_contains_past_days(self):
        """URL must include past_days=3 to anchor temp_rolling_3d."""
        with patch("requests.get", return_value=_make_response(_full_hourly(2))) as mock_get:
            weather.fetch_open_meteo(47.0, 8.0)
        assert "past_days=3" in mock_get.call_args[0][0]

    def test_url_contains_cloud_cover_and_direct_radiation(self):
        with patch("requests.get", return_value=_make_response(_full_hourly(2))) as mock_get:
            weather.fetch_open_meteo(47.0, 8.0)
        url = mock_get.call_args[0][0]
        assert "cloud_cover" in url
        assert "direct_radiation" in url

    def test_cloud_cover_pct_parsed_correctly(self):
        hourly = {**_base_hourly(3), "sunshine_duration": [0]*3,
                  "cloud_cover": [10, 50, 90], "direct_radiation": [0]*3}
        with patch("requests.get", return_value=_make_response(hourly)):
            df = weather.fetch_open_meteo(47.0, 8.0)
        assert list(df["cloud_cover_pct"]) == [10, 50, 90]

    def test_direct_radiation_wm2_parsed_correctly(self):
        hourly = {**_base_hourly(3), "sunshine_duration": [0]*3,
                  "cloud_cover": [0]*3, "direct_radiation": [100, 200, 300]}
        with patch("requests.get", return_value=_make_response(hourly)):
            df = weather.fetch_open_meteo(47.0, 8.0)
        assert list(df["direct_radiation_wm2"]) == [100, 200, 300]

    def test_new_columns_missing_key_fallback(self):
        """If cloud_cover or direct_radiation absent, columns default to 0."""
        hourly = {**_base_hourly(3), "sunshine_duration": [0]*3}  # no cloud/radiation
        with patch("requests.get", return_value=_make_response(hourly)):
            df = weather.fetch_open_meteo(47.0, 8.0)
        assert (df["cloud_cover_pct"] == 0).all()
        assert (df["direct_radiation_wm2"] == 0).all()

    def test_returns_all_expected_columns(self):
        with patch("requests.get", return_value=_make_response(_full_hourly(3))):
            df = weather.fetch_open_meteo(47.0, 8.0)
        expected = {"timestamp", "temp_c", "precipitation_mm", "sunshine_min",
                    "wind_kmh", "cloud_cover_pct", "direct_radiation_wm2"}
        assert expected <= set(df.columns)


# ── fetch_historical_weather ──────────────────────────────────────────────────

class TestFetchHistoricalWeather:

    def _make_archive_response(self, n: int = 3) -> MagicMock:
        hourly = {
            "time":               [f"2026-01-01T{h:02d}:00" for h in range(n)],
            "temperature_2m":     [2.0] * n,
            "precipitation":      [0.0] * n,
            "sunshine_duration":  [600] * n,
            "windspeed_10m":      [8.0] * n,
            "cloud_cover":        [40]  * n,
            "direct_radiation":   [150] * n,
        }
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hourly": hourly}
        return mock

    def test_returns_new_columns(self):
        from datetime import date
        with patch("requests.get", return_value=self._make_archive_response()):
            df = weather.fetch_historical_weather(47.0, 8.0, date(2026, 1, 1), date(2026, 1, 1))
        assert "cloud_cover_pct"      in df.columns
        assert "direct_radiation_wm2" in df.columns

    def test_new_columns_missing_key_fallback(self):
        """Archive response without cloud/radiation fields defaults to 0."""
        from datetime import date
        hourly = {
            "time":              ["2026-01-01T00:00"],
            "temperature_2m":    [2.0],
            "precipitation":     [0.0],
            "sunshine_duration": [600],
            "windspeed_10m":     [8.0],
            # no cloud_cover, no direct_radiation
        }
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"hourly": hourly}
        with patch("requests.get", return_value=mock):
            df = weather.fetch_historical_weather(47.0, 8.0, date(2026, 1, 1), date(2026, 1, 1))
        assert (df["cloud_cover_pct"] == 0).all()
        assert (df["direct_radiation_wm2"] == 0).all()


# ── _supplement_from_open_meteo ───────────────────────────────────────────────

def _make_srg_df(start: str, periods: int) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="1h")
    return pd.DataFrame({
        "timestamp":        timestamps,
        "temp_c":           [10.0] * periods,
        "precipitation_mm": [0.0]  * periods,
        "sunshine_min":     [30.0] * periods,
        "wind_kmh":         [5.0]  * periods,
    })


def _make_om_df(start: str, periods: int, cloud: float = 60.0, rad: float = 250.0) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=periods, freq="1h")
    return pd.DataFrame({
        "timestamp":            timestamps,
        "temp_c":               [8.0]   * periods,
        "precipitation_mm":     [0.1]   * periods,
        "sunshine_min":         [20.0]  * periods,
        "wind_kmh":             [7.0]   * periods,
        "cloud_cover_pct":      [cloud] * periods,
        "direct_radiation_wm2": [rad]   * periods,
    })


class TestSupplementFromOpenMeteo:

    def test_historical_rows_prepended(self):
        """OM rows before the SRG window must appear in the combined output."""
        srg = _make_srg_df("2026-03-12 08:00", 48)
        # OM covers 72h before + 48h future overlap
        om  = _make_om_df("2026-03-09 08:00", 72 + 48)
        result = _supplement_from_open_meteo(srg, om)
        assert result["timestamp"].min() < srg["timestamp"].min()

    def test_srg_temp_values_preserved(self):
        """SRG temp_c must not be overwritten by Open-Meteo values."""
        srg = _make_srg_df("2026-03-12 08:00", 4)
        om  = _make_om_df("2026-03-12 06:00", 6)  # 2h hist + 4h overlap
        result = _supplement_from_open_meteo(srg, om)
        future_rows = result[result["timestamp"] >= srg["timestamp"].min()]
        assert (future_rows["temp_c"] == 10.0).all()

    def test_cloud_radiation_filled_from_om(self):
        """cloud_cover_pct and direct_radiation_wm2 must come from Open-Meteo."""
        srg = _make_srg_df("2026-03-12 08:00", 4)
        om  = _make_om_df("2026-03-12 06:00", 6, cloud=75.0, rad=300.0)
        result = _supplement_from_open_meteo(srg, om)
        future_rows = result[result["timestamp"] >= srg["timestamp"].min()]
        assert (future_rows["cloud_cover_pct"] == 75.0).all()
        assert (future_rows["direct_radiation_wm2"] == 300.0).all()

    def test_empty_om_returns_srg_unchanged(self):
        """If Open-Meteo call failed (empty df), SRG result is returned as-is."""
        srg = _make_srg_df("2026-03-12 08:00", 4)
        result = _supplement_from_open_meteo(srg, pd.DataFrame())
        assert len(result) == len(srg)
        assert list(result["temp_c"]) == list(srg["temp_c"])

    def test_tz_aware_srg_timestamps_do_not_raise(self):
        """SRG v2 returns ISO timestamps with +01:00 offset; _supplement must not crash."""
        # Reproduce real SRG v2 date_time strings (tz-aware, UTC+1)
        tz_timestamps = pd.to_datetime([
            "2026-03-12T08:00:00+01:00",
            "2026-03-12T09:00:00+01:00",
            "2026-03-12T10:00:00+01:00",
            "2026-03-12T11:00:00+01:00",
        ])
        srg = pd.DataFrame({
            "timestamp":        tz_timestamps,
            "temp_c":           [10.0] * 4,
            "precipitation_mm": [0.0]  * 4,
            "sunshine_min":     [30.0] * 4,
            "wind_kmh":         [5.0]  * 4,
        })
        om = _make_om_df("2026-03-12 06:00", 6)
        result = _supplement_from_open_meteo(srg, om)
        # Must not raise; timestamps in result must be tz-naive
        assert result["timestamp"].dt.tz is None
        assert len(result) > 0


# ── fetch_forecast v2 ─────────────────────────────────────────────────────────

def _make_srg_v2_response(n: int = 3) -> MagicMock:
    """Mock for /forecastpoint/{id} — v2 flat hours array."""
    hours = [
        {
            "date_time": f"2026-03-12T{h:02d}:00:00+01:00",
            "TTT_C": 10, "RRR_MM": 1.5, "SUN_MIN": 30, "FF_KMH": 15,
        }
        for h in range(n)
    ]
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"hours": hours}
    return mock


def _make_geo_plz_response(geo_id: str = "47.2044,7.5581") -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = [{"geolocation": {"id": geo_id}}]
    return mock


def _make_geo_latlon_response(geo_id: str = "47.2263,7.5784") -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = [{"id": geo_id}]
    return mock


def _make_token_response() -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"access_token": "test-token"}
    return mock


def _om_empty() -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"hourly": {
        "time": [], "temperature_2m": [], "precipitation": [],
        "windspeed_10m": [], "sunshine_duration": [], "cloud_cover": [], "direct_radiation": [],
    }}
    return mock


class TestFetchForecastV2:

    def test_geolocation_resolved_via_latlon(self):
        """Geolocation must always be resolved via lat/lon (not PLZ) to match the registered station."""
        with patch("requests.post", return_value=_make_token_response()), \
             patch("requests.get", side_effect=[
                 _make_geo_latlon_response(), _make_srg_v2_response(), _om_empty(),
             ]) as mock_get:
            weather.fetch_forecast("4528", 47.2, 7.5, "key", "secret")
        first_url = mock_get.call_args_list[0][0][0]
        assert "geolocations" in first_url
        assert "latitude" in first_url

    def test_forecastpoint_called_with_id_from_latlon_lookup(self):
        """forecastpoint URL must use the id returned by the lat/lon lookup."""
        geo_id = "47.2263,7.5784"
        with patch("requests.post", return_value=_make_token_response()), \
             patch("requests.get", side_effect=[
                 _make_geo_latlon_response(geo_id), _make_srg_v2_response(), _om_empty(),
             ]) as mock_get:
            weather.fetch_forecast("4528", 47.2, 7.5, "key", "secret")
        forecast_url = mock_get.call_args_list[1][0][0]
        assert f"forecastpoint/{geo_id}" in forecast_url

    def test_empty_geo_response_falls_back_to_open_meteo(self):
        """If geolocation returns no hits, fall back to Open-Meteo."""
        empty_geo = MagicMock()
        empty_geo.raise_for_status = MagicMock()
        empty_geo.json.return_value = []
        with patch("requests.post", return_value=_make_token_response()), \
             patch("requests.get", side_effect=[empty_geo, _om_empty()]):
            df = weather.fetch_forecast("4528", 47.2, 7.5, "key", "secret")
        assert isinstance(df, pd.DataFrame)

    def test_rrr_mm_mapped_to_precipitation_mm(self):
        """v2 field RRR_MM must be read as precipitation_mm (not PRP_MM)."""
        with patch("requests.post", return_value=_make_token_response()), \
             patch("requests.get", side_effect=[
                 _make_geo_latlon_response(), _make_srg_v2_response(), _om_empty(),
             ]):
            df = weather.fetch_forecast("4528", 47.2, 7.5, "key", "secret")
        srg_rows = df[df["precipitation_mm"].notna()]
        assert not srg_rows.empty
        assert (srg_rows["precipitation_mm"] == 1.5).any()

    def test_no_credentials_returns_open_meteo(self):
        """Without credentials, must skip SRG entirely and call Open-Meteo."""
        with patch("requests.get", return_value=_make_response(_full_hourly(3))) as mock_get:
            df = weather.fetch_forecast("4528", 47.2, 7.5, None, None)
        assert not df.empty
        # Only one GET (Open-Meteo), no POST for auth
        assert mock_get.call_count == 1
