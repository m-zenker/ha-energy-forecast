"""Synthetic weather/energy frames shared by the tests/test_backtest_*.py files."""

import numpy as np
import pandas as pd


def synthetic_weather(days: int, start: str = "2026-01-01", seed: int = 1) -> pd.DataFrame:
    """Hourly weather: seasonal ramp + ±5 °C day-to-day noise + diurnal swing, naive local timestamps."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=days * 24, freq="1h")
    day_noise = np.repeat(rng.uniform(-5, 5, days), 24)
    ramp = np.linspace(0, 6, len(ts))
    diurnal = 3 * np.sin(2 * np.pi * (ts.hour - 9) / 24)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": 2 + ramp + day_noise + diurnal,
            "precipitation_mm": 0.0,
            "sunshine_min": np.where((ts.hour >= 8) & (ts.hour <= 16), 30.0, 0.0),
            "wind_kmh": 8.0,
            "humidity": 70.0,
            "cloud_cover_pct": 50.0,
            "direct_radiation_wm2": np.where((ts.hour >= 8) & (ts.hour <= 16), 150.0, 0.0),
        }
    )


def synthetic_energy(weather: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    """Consumption = 0.4 + 0.05·max(0, 15 − temp) + hour shape + noise (kWh/h)."""
    rng = np.random.default_rng(seed)
    ts = weather["timestamp"]
    shape = 0.2 * ((ts.dt.hour >= 7) & (ts.dt.hour <= 21))
    kwh = 0.4 + 0.05 * np.clip(15 - weather["temp_c"], 0, None) + shape + rng.normal(0, 0.05, len(ts))
    return pd.DataFrame({"timestamp": ts, "gross_kwh": np.clip(kwh, 0, None)})
