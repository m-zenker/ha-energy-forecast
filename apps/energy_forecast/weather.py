import logging

import requests
import pandas as pd
from datetime import datetime, date

_LOGGER = logging.getLogger(__name__)


def fetch_historical_weather(lat: float, lon: float, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch hourly historical weather from the Open-Meteo Archive API.

    Returns a DataFrame with columns:
        timestamp (naive, UTC+local via timezone param), temp_c,
        precipitation_mm, sunshine_min, wind_kmh,
        cloud_cover_pct, direct_radiation_wm2
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
        "&hourly=temperature_2m,precipitation,sunshine_duration,windspeed_10m"
        ",cloud_cover,direct_radiation"
        "&timezone=Europe%2FZurich"
    )
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    h = res.json()["hourly"]
    n = len(h["time"])
    return pd.DataFrame({
        "timestamp":             pd.to_datetime(h["time"]),
        "temp_c":                h["temperature_2m"],
        "precipitation_mm":      h["precipitation"],
        # Open-Meteo returns sunshine_duration in seconds; model expects minutes
        "sunshine_min":          [s / 60.0 for s in h["sunshine_duration"]],
        "wind_kmh":              h["windspeed_10m"],
        "cloud_cover_pct":       h.get("cloud_cover",      [0] * n),
        "direct_radiation_wm2":  h.get("direct_radiation", [0] * n),
    })


def fetch_forecast(plz: str, lat: float, lon: float, client_id: str | None = None, client_secret: str | None = None) -> pd.DataFrame:
    """Fetches high-quality forecast from SRG-SSR API with Open-Meteo fallback.

    When SRG credentials are provided and the fetch succeeds, Open-Meteo is
    called as a supplement to fill in cloud_cover_pct, direct_radiation_wm2,
    and the 3-day historical tail needed to anchor temp_rolling_3d.  SRG
    values for temp_c, precipitation_mm, sunshine_min, wind_kmh are preserved.
    """
    if not client_id or not client_secret:
        return fetch_open_meteo(lat, lon)

    try:
        # 1. Get OAuth Token
        auth_url = "https://api.srgssr.ch/oauth/v1/accesstoken?grant_type=client_credentials"
        auth_res = requests.post(auth_url, auth=(client_id, client_secret), timeout=10)
        auth_res.raise_for_status()
        try:
            token = auth_res.json()["access_token"]
        except (ValueError, KeyError) as exc:
            _LOGGER.warning(
                "SRG-SSR auth failed — HTTP %s, body: %.200r — falling back to Open-Meteo.",
                auth_res.status_code, auth_res.text,
            )
            return fetch_open_meteo(lat, lon)
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

        # 2. Resolve geolocation ID — nearest station by lat/lon.
        # The Freemium plan locks a single registered location; using lat/lon ensures
        # we always resolve the same station that was registered in the developer app.
        geo_res = requests.get(
            f"https://api.srgssr.ch/srf-meteo/v2/geolocations?latitude={lat}&longitude={lon}",
            headers=headers, timeout=10,
        )
        geo_res.raise_for_status()
        geo_hits = geo_res.json()
        geo_id = geo_hits[0]["id"] if geo_hits else None

        if not geo_id:
            _LOGGER.warning("SRG-SSR geolocation lookup returned no results — falling back to Open-Meteo.")
            return fetch_open_meteo(lat, lon)

        # 3. Get Forecast (60min intervals)
        forecast_url = f"https://api.srgssr.ch/srf-meteo/v2/forecastpoint/{geo_id}"
        res = requests.get(forecast_url, headers=headers, timeout=10)
        res.raise_for_status()
        try:
            data = res.json()
        except ValueError as exc:
            import re as _re
            _title = (_re.search(r"<title[^>]*>(.*?)</title>", res.text, _re.I | _re.S) or None)
            title_str = _title.group(1).strip() if _title else res.text[:120].strip()
            _LOGGER.warning(
                "SRG-SSR forecast parse failed — HTTP %s, page: %r — falling back to Open-Meteo.",
                res.status_code, title_str,
            )
            return fetch_open_meteo(lat, lon)

        records = []
        for hour in data.get("hours", []):
            records.append({
                "timestamp":        hour.get("date_time"),
                "temp_c":           float(hour.get("TTT_C", 0)),
                "precipitation_mm": float(hour.get("RRR_MM", 0)),
                "sunshine_min":     float(hour.get("SUN_MIN", 0)),
                "wind_kmh":         float(hour.get("FF_KMH", 0)),
            })

        srg_df = pd.DataFrame(records)

        # 3. Supplement with Open-Meteo: cloud/radiation + 3-day historical tail
        om_df = fetch_open_meteo(lat, lon)
        return _supplement_from_open_meteo(srg_df, om_df)

    except (requests.RequestException, KeyError, ValueError) as exc:
        _LOGGER.warning("SRG-SSR forecast failed (%s) — falling back to Open-Meteo.", exc)
        return fetch_open_meteo(lat, lon)


def _supplement_from_open_meteo(srg_df: pd.DataFrame, om_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Open-Meteo data into an SRG forecast DataFrame.

    Combines two sources into one DataFrame:
      - Historical rows (timestamps before the SRG window) from Open-Meteo only,
        providing the 3-day tail needed to anchor temp_rolling_3d.
      - Future rows (SRG timestamps): SRG values for temp_c, precipitation_mm,
        sunshine_min, wind_kmh; cloud_cover_pct and direct_radiation_wm2 from
        Open-Meteo (joined by timestamp, NaN where OM has no matching row).

    If om_df is empty (Open-Meteo call failed), returns srg_df unchanged.
    """
    if om_df.empty:
        return srg_df

    srg_df = srg_df.copy()
    _ts = pd.to_datetime(srg_df["timestamp"])
    if _ts.dt.tz is not None:
        _ts = _ts.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
    srg_df["timestamp"] = _ts
    om_df  = om_df.copy()
    om_df["timestamp"]  = pd.to_datetime(om_df["timestamp"])

    srg_start = srg_df["timestamp"].min()

    # Historical tail: Open-Meteo rows before the SRG window
    om_hist = om_df[om_df["timestamp"] < srg_start].copy()

    # Cloud/radiation columns to pull from Open-Meteo for future rows
    om_supplement_cols = ["timestamp", "cloud_cover_pct", "direct_radiation_wm2"]
    om_future = om_df[om_df["timestamp"] >= srg_start][om_supplement_cols]

    # Join cloud/radiation onto SRG future rows
    srg_df = srg_df.merge(om_future, on="timestamp", how="left")

    # Stack historical tail above SRG future rows
    combined = pd.concat([om_hist, srg_df], ignore_index=True)
    return combined.sort_values("timestamp").reset_index(drop=True)


def fetch_open_meteo(lat: float, lon: float) -> pd.DataFrame:
    """Forecast (+ 3-day historical tail) using the free Open-Meteo API.

    past_days=3 adds ~72 h of measured history before the forecast window.
    This anchors the temp_rolling_3d feature in _engineer_features so the
    3-day rolling mean is based on real observations, not forecast values.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation,sunshine_duration,windspeed_10m"
        ",cloud_cover,direct_radiation"
        "&past_days=3"
        "&timezone=Europe%2FZurich"
    )
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        h = res.json()["hourly"]
        n = len(h["time"])
        # Open-Meteo returns sunshine_duration in seconds; model expects minutes.
        # Use .get() with zero-filled fallbacks in case any field is absent.
        sunshine_s = h.get("sunshine_duration", [0] * n)
        return pd.DataFrame({
            "timestamp":            pd.to_datetime(h["time"]),
            "temp_c":               h["temperature_2m"],
            "precipitation_mm":     h["precipitation"],
            "sunshine_min":         [s / 60.0 for s in sunshine_s],
            "wind_kmh":             h["windspeed_10m"],
            "cloud_cover_pct":      h.get("cloud_cover",      [0] * n),
            "direct_radiation_wm2": h.get("direct_radiation", [0] * n),
        })
    except (requests.RequestException, KeyError, ValueError):
        return pd.DataFrame()