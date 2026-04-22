"""Constants for the Energy Forecast app."""

from pathlib import Path
from typing import Any

# Data Validation
MAX_HOURLY_KWH = 50.0  # Filters out meter resets or spikes

# EV Charging Detection
# Hours above this threshold are classified as EV charging and excluded from
# baseline model training.  Normal household max is ~6.5 kWh/h; EV sessions
# start at ~7.3 kWh/h — giving no clean gap.  7 kWh/h is a good midpoint.
# Override per-instance in apps.yaml: ev_charging_threshold_kwh: 7
EV_CHARGING_THRESHOLD_KWH = 7

# Model Training
MIN_TRAINING_ROWS = 100   # Minimum clean rows to proceed with training
HOLDOUT_FRACTION  = 0.9   # Training fraction (first 90% of rows); name kept for backward compat
MIN_CV_ROWS       = 500   # Minimum rows for TimeSeriesSplit cross-validation

# Data Storage
CACHE_PATH = Path(__file__).parent / "energy_history.csv"
PRED_HISTORY_PATH = Path(__file__).parent / "pred_history.json"

# Occupancy Detection
PRESENCE_STATE_HOME = "home"

# Appliance Signature Learning
APPLIANCE_MAX_WINDOW_HOURS = 12  # Hard cap for adaptive cycle detection

# Sensor Blending Logic
# How many hours to trust the local outdoor sensor before
# transitioning fully to the weather forecast.
SENSOR_FULL_TRUST_HOURS = 2
SENSOR_BLEND_HOURS = 6

# Thermal Projection (indoor temperature RC-ODE forward simulation)
DEFAULT_TAU = 12.0        # hours — used when τ has not been calibrated yet
DEFAULT_ROOM_AREA_M2 = 15.0  # m² — default per-room area when not configured


def strip_tz(df: Any, timezone: str = "Europe/Zurich") -> Any:
    """Convert the 'timestamp' column of a DataFrame to naive local time.

    tz-aware timestamps are converted to *timezone* first, then the tzinfo is
    dropped. Naive timestamps pass through unchanged.
    """
    import pandas as pd
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
        df = df.copy()
        df["timestamp"] = ts
    return df
