"""Weather data fetching — MeteoSwiss (primary) + Open-Meteo (fallback/archive)."""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from .const import (
    METEOSWISS_URL,
    OPENMETEO_FORECAST_URL,
    OPENMETEO_ARCHIVE_URL,
)

_LOGGER = logging.getLogger(__name__)

WEATHER_COLS = ["timestamp", "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh"]
_HEADERS = {"User-Agent": "HAEnergyForecast/1.0"}


# ── Forecast ─────────────────────────────────────────────────────────────────

def fetch_forecast(plz: str, latitude: float, longitude: float) -> pd.DataFrame:
    """
    Return 48h+ hourly forecast DataFrame with columns defined in WEATHER_COLS.
    Tries MeteoSwiss PLZ API first; falls back to Open-Meteo on any failure.
    """
    try:
        resp = requests.get(
            METEOSWISS_URL, params={"plz": plz}, headers=_HEADERS, timeout=15
        )
        resp.raise_for_status()
        df = _parse_meteoswiss(resp.json())
        _LOGGER.debug("MeteoSwiss forecast: %d rows for PLZ %s", len(df), plz)
        return df
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(
            "MeteoSwiss PLZ API failed (%s) — falling back to Open-Meteo", exc
        )
        return _fetch_openmeteo_forecast(latitude, longitude)


def _parse_meteoswiss(data: dict) -> pd.DataFrame:
    fc = data.get("forecast", data)
    timestamps = fc.get("time", [])
    if not timestamps:
        raise ValueError("MeteoSwiss response contains no timestamps")

    def _safe(key: str, default: float) -> list:
        vals = fc.get(key, [])
        return [v if v is not None else default for v in vals]

    rows = []
    for i, ts in enumerate(timestamps):
        dt = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Europe/Zurich")
        rows.append(
            {
                "timestamp": dt,
                "temp_c": _safe("temperature", 10.0)[i] if i < len(_safe("temperature", 10.0)) else 10.0,
                "precipitation_mm": _safe("precipitation", 0.0)[i] if i < len(_safe("precipitation", 0.0)) else 0.0,
                "sunshine_min": _safe("sunshine", 0.0)[i] if i < len(_safe("sunshine", 0.0)) else 0.0,
                "wind_kmh": _safe("wind", 0.0)[i] if i < len(_safe("wind", 0.0)) else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed MeteoSwiss DataFrame is empty")
    return df


def _fetch_openmeteo_forecast(latitude: float, longitude: float) -> pd.DataFrame:
    resp = requests.get(
        OPENMETEO_FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,precipitation,windspeed_10m,direct_radiation",
            "timezone": "Europe/Zurich",
            "forecast_days": 3,
        },
        timeout=15,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(h["time"]),
            "temp_c": h["temperature_2m"],
            "precipitation_mm": h["precipitation"],
            "sunshine_min": [r / 60.0 if r else 0.0 for r in h["direct_radiation"]],
            "wind_kmh": h["windspeed_10m"],
        }
    )
    df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Zurich")
    return df


# ── Historical archive ────────────────────────────────────────────────────────

def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Fetch hourly historical weather from Open-Meteo Archive API.
    Free, no API key, covers Switzerland perfectly.
    """
    resp = requests.get(
        OPENMETEO_ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": "temperature_2m,precipitation,windspeed_10m,direct_radiation",
            "timezone": "Europe/Zurich",
        },
        timeout=90,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(h["time"]),
            "temp_c": h["temperature_2m"],
            "precipitation_mm": h["precipitation"],
            "sunshine_min": [r / 60.0 if r else 0.0 for r in h["direct_radiation"]],
            "wind_kmh": h["windspeed_10m"],
        }
    )
    df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Zurich")
    _LOGGER.debug(
        "Open-Meteo archive: %d rows (%s → %s)", len(df), start, end
    )
    return df
