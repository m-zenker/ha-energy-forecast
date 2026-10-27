"""Shared synthetic training frames for model tests (Plans 2026-10-15 B/C/D)."""

import numpy as np
import pandas as pd


def train_frames(n: int = 960) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": rng.uniform(-5, 25, size=n),
            "precipitation_mm": 0.0,
            "sunshine_min": 30.0,
            "wind_kmh": 10.0,
            "cloud_cover_pct": 50.0,
            "direct_radiation_wm2": 100.0,
        }
    )
    return energy, weather
