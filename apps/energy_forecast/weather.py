import requests
import pandas as pd
from datetime import datetime, date


def fetch_historical_weather(lat: float, lon: float, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch hourly historical weather from the Open-Meteo Archive API.

    Returns a DataFrame with columns:
        timestamp (naive, UTC+local via timezone param), temp_c,
        precipitation_mm, sunshine_min, wind_kmh
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
        "&hourly=temperature_2m,precipitation,sunshine_duration,windspeed_10m"
        "&timezone=Europe%2FZurich"
    )
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    h = res.json()["hourly"]
    return pd.DataFrame({
        "timestamp":         pd.to_datetime(h["time"]),
        "temp_c":            h["temperature_2m"],
        "precipitation_mm":  h["precipitation"],
        # Open-Meteo returns sunshine_duration in seconds; model expects minutes
        "sunshine_min":      [s / 60.0 for s in h["sunshine_duration"]],
        "wind_kmh":          h["windspeed_10m"],
    })


def fetch_forecast(plz: str, lat: float, lon: float, client_id: str | None = None, client_secret: str | None = None) -> pd.DataFrame:
    """Fetches high-quality forecast from SRG-SSR API with Open-Meteo fallback."""
    if not client_id or not client_secret:
        return fetch_open_meteo(lat, lon)

    try:
        # 1. Get OAuth Token
        auth_url = "https://api.srgssr.ch/oauth/v1/accesstoken?grant_type=client_credentials"
        auth_res = requests.post(auth_url, auth=(client_id, client_secret), timeout=10)
        auth_res.raise_for_status()
        token = auth_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # 2. Get Forecast (60min intervals)
        forecast_url = f"https://api.srgssr.ch/forecasts/v1.0/weather/7day?latitude={lat}&longitude={lon}"
        res = requests.get(forecast_url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()

        records = []
        for day in data.get("forecast", []):
            for hour in day.get("hours", []):
                records.append({
                    "timestamp": hour.get("date_time"),
                    "temp_c": float(hour.get("TTT_C", 0)),
                    "precipitation_mm": float(hour.get("PRP_MM", 0)),
                    "sunshine_min": float(hour.get("SUN_MIN", 0)),
                    "wind_kmh": float(hour.get("FF_KMH", 0))
                })
        
        df = pd.DataFrame(records)
        return df

    except (requests.RequestException, KeyError, ValueError):
        # SRG failed — fall back to Open-Meteo
        return fetch_open_meteo(lat, lon)

def fetch_open_meteo(lat: float, lon: float) -> pd.DataFrame:
    """Fallback forecast using the free Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation,windspeed_10m&timezone=Europe%2FZurich"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        h = res.json()["hourly"]
        return pd.DataFrame({
            "timestamp": pd.to_datetime(h["time"]),
            "temp_c": h["temperature_2m"],
            "precipitation_mm": h["precipitation"],
            "sunshine_min": 0,  # Open-Meteo free tier doesn't always provide sunshine duration
            "wind_kmh": h["windspeed_10m"]
        })
    except (requests.RequestException, KeyError, ValueError):
        return pd.DataFrame()