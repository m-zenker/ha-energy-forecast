"""Constants for the Energy Forecast app."""

from pathlib import Path

# API URLs
METEOSWISS_URL = "https://app-prod-ws.meteoswiss-app.ch/v1/forecast"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

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
HOLDOUT_FRACTION  = 0.9   # Train/test split fraction for holdout MAE
MIN_CV_ROWS       = 500   # Minimum rows for TimeSeriesSplit cross-validation

# Data Storage
CACHE_PATH = Path(__file__).parent / "energy_history.csv"

# Sensor Blending Logic
# How many hours to trust the local outdoor sensor before
# transitioning fully to the weather forecast.
SENSOR_FULL_TRUST_HOURS = 2
SENSOR_BLEND_HOURS = 6
