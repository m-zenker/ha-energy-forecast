"""Weather data — MeteoSwiss (primary) + Open-Meteo (fallback/archive)."""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

import requests

from .const import METEOSWISS_URL, OPENMETEO_FORECAST_URL, OPENMETEO_ARCHIVE_URL

_LOGGER = logging.getLogger(__name__)
_HEADERS = {"User-Agent": "HAEnergyForecast/1.0"}


def fetch_forecast(plz: str, latitude: float, longitude: float) -> Any:
    """Return 48h+ hourly forecast DataFrame. MeteoSwiss → Open-Meteo fallback."""
    try:
        resp = requests.get(METEOSWISS_URL, params={"plz": plz}, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        df = _parse_meteoswiss(resp.json())
        _LOGGER.debug("MeteoSwiss forecast: %d rows for PLZ %s", len(df), plz)
        return df
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("MeteoSwiss failed (%s) — falling back to Open-Meteo", exc)
        return _fetch_openmeteo_forecast(latitude, longitude)


def _parse_meteoswiss(data: dict) -> Any:
    import pandas as pd  # noqa: PLC0415

    fc = data.get("forecast", data)
    timestamps = fc.get("time", [])
    if not timestamps:
        raise ValueError("MeteoSwiss response contains no timestamps")

    def _safe(key: str, default: float) -> list:
        vals = fc.get(key, [])
        return [v if v is not None else default for v in vals]

    temps   = _safe("temperature",   10.0)
    precip  = _safe("precipitation",  0.0)
    sun     = _safe("sunshine",       0.0)
    wind    = _safe("wind",           0.0)

    rows = []
    for i, ts in enumerate(timestamps):
        dt = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Europe/Zurich")
        rows.append({
            "timestamp":        dt,
            "temp_c":           temps[i]  if i < len(temps)  else 10.0,
            "precipitation_mm": precip[i] if i < len(precip) else 0.0,
            "sunshine_min":     sun[i]    if i < len(sun)    else 0.0,
            "wind_kmh":         wind[i]   if i < len(wind)   else 0.0,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Parsed MeteoSwiss DataFrame is empty")
    return df


def _fetch_openmeteo_forecast(latitude: float, longitude: float) -> Any:
    import pandas as pd  # noqa: PLC0415

    resp = requests.get(
        OPENMETEO_FORECAST_URL,
        params={
            "latitude": latitude, "longitude": longitude,
            "hourly": "temperature_2m,precipitation,windspeed_10m,direct_radiation",
            "timezone": "Europe/Zurich", "forecast_days": 3,
        },
        timeout=15,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame({
        "timestamp":        pd.to_datetime(h["time"]),
        "temp_c":           h["temperature_2m"],
        "precipitation_mm": h["precipitation"],
        "sunshine_min":     [r / 60.0 if r else 0.0 for r in h["direct_radiation"]],
        "wind_kmh":         h["windspeed_10m"],
    })
    df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Zurich")
    return df


def fetch_historical_weather(
    latitude: float, longitude: float, start: date, end: date,
) -> Any:
    """Historical hourly weather from Open-Meteo Archive (free, no key)."""
    import pandas as pd  # noqa: PLC0415

    resp = requests.get(
        OPENMETEO_ARCHIVE_URL,
        params={
            "latitude": latitude, "longitude": longitude,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "hourly": "temperature_2m,precipitation,windspeed_10m,direct_radiation",
            "timezone": "Europe/Zurich",
        },
        timeout=90,
    )
    resp.raise_for_status()
    h = resp.json()["hourly"]
    df = pd.DataFrame({
        "timestamp":        pd.to_datetime(h["time"]),
        "temp_c":           h["temperature_2m"],
        "precipitation_mm": h["precipitation"],
        "sunshine_min":     [r / 60.0 if r else 0.0 for r in h["direct_radiation"]],
        "wind_kmh":         h["windspeed_10m"],
    })
    df["timestamp"] = df["timestamp"].dt.tz_localize("Europe/Zurich")
    return df
