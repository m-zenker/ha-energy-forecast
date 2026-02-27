"""Constants for HA Energy Forecast integration."""

DOMAIN = "ha_energy_forecast"
PLATFORMS = ["sensor"]

# ── Config entry keys ────────────────────────────────────────────────────────
CONF_ENERGY_SENSOR = "energy_sensor"
CONF_OUTDOOR_TEMP_SENSOR = "outdoor_temp_sensor"
CONF_PLZ = "plz"
CONF_LATITUDE = "latitude"
CONF_LONGITUDE = "longitude"

# ── Hardcoded operational constants ──────────────────────────────────────────
RETRAIN_INTERVAL_HOURS = 168          # Weekly
HISTORY_MONTHS = 12
MIN_HISTORY_HOURS = 720               # ~1 month before first training
UPDATE_INTERVAL_MINUTES = 60
MAX_HOURLY_KWH = 50                   # Spike / meter-reset filter threshold

# ── Temperature blending horizon ─────────────────────────────────────────────
SENSOR_FULL_TRUST_HOURS = 3           # Use live sensor fully for this many hours
SENSOR_BLEND_HOURS = 12              # Blend sensor→forecast up to this horizon

# ── Storage keys ─────────────────────────────────────────────────────────────
STORAGE_VERSION = 1
STORAGE_KEY = f"{DOMAIN}.model"

# ── Sensor entity IDs (relative, without domain prefix) ──────────────────────
SENSOR_NEXT_3H = "next_3h"
SENSOR_TODAY = "today"
SENSOR_TOMORROW = "tomorrow"
SENSOR_MAE = "model_mae"

# 3-hour block slot names, e.g. "00_03", "03_06", ...
BLOCK_SLOTS = [f"{h:02d}_{h+3:02d}" for h in range(0, 24, 3)]

# ── MeteoSwiss ────────────────────────────────────────────────────────────────
METEOSWISS_URL = "https://app-prod-ws.meteoswiss-app.ch/v1/plzDetail"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# ── Attribution ───────────────────────────────────────────────────────────────
ATTRIBUTION = "HA Energy Forecast — LightGBM + MeteoSwiss"
