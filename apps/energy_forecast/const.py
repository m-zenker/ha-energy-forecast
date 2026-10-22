"""Constants for the Energy Forecast app."""

import logging
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
MIN_TRAINING_ROWS = 100  # Minimum clean rows to proceed with training
HOLDOUT_FRACTION = 0.9  # Training fraction (first 90% of rows); name kept for backward compat
MIN_CV_ROWS = 500  # Minimum rows for TimeSeriesSplit cross-validation

# Data Storage
CACHE_PATH = Path(__file__).parent / "energy_history.csv"
CACHE_PATH_15M = Path(__file__).parent / "energy_history_15m.csv"
MAX_15MIN_KWH = MAX_HOURLY_KWH / 4  # 12.5 kWh — per-slot cap for 15-min resolution
PRED_HISTORY_PATH = Path(__file__).parent / "pred_history.json"
GITHUB_RELEASES_URL = "https://api.github.com/repos/m-zenker/ha-energy-forecast/releases/latest"

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
DEFAULT_TAU = 12.0  # hours — used when τ has not been calibrated yet
# Thermostat recovery of a transient hour-0 deficit in the indoor projection (2026-10-19): the floor
# deficit decays from d0 to the room's steady same-setpoint deficit with this e-folding time.
DEFICIT_RECOVERY_TAU_H = 2.0  # hours — value from plan 2026-10-19-deficit-fade Task 1
DEFICIT_STEADY_WINDOW_H = 24  # hours of climate history used for the steady deficit
DEFICIT_STEADY_MIN_ROWS = 3  # fewer same-setpoint rows → steady deficit 0 (setpoint just changed)
# Heating-buffer smoothing (2026-10-20): train and predict `heating_buffer_temp` as a causal trailing mean over
# this many hours instead of a heat-pump-cycle snapshot. ON (24 h, option S24); 0 = off (raw same-hour reading,
# latest broadcast). Models trained before the retrain keep the raw path (window persisted in meta.pkl).
HEATING_BUFFER_SMOOTH_H = 24
DEFAULT_ROOM_AREA_M2 = 15.0  # m² — default per-room area when not configured

# Physics UA_eff calibration eligibility (#92) — deliberately not shared with
# model.py's hp_heating_degree literal, which serves a different purpose (an ML
# feature hinge, not a calibration-eligibility threshold).
HEATING_SUB_METER_MIN_KWH = 0.3  # tier 1: below this, a hour reads as heat-pump-off/short-cycling
UA_EFF_FALLBACK_MAX_OUTDOOR_C = 8.0  # tier 3: colder than the ML feature's 15°C knee (wants precision, not recall)
UA_EFF_FALLBACK_MIN_LOAD_KWH = 0.3  # tier 3: load floor, applied only once Q_base_el is genuinely calibrated
UA_EFF_SANITY_MIN_W_PER_K = 50.0  # implausible-result guard on the recovered UA_eff
UA_EFF_SANITY_MAX_W_PER_K = 500.0

# Supported energy sensor units and their kWh conversion factors.
UNIT_TO_KWH: dict[str, float] = {"kWh": 1.0, "MWh": 1000.0, "Wh": 0.001}


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


def drop_dst_duplicates(df: Any) -> Any:
    """Drop rows whose naive 'timestamp' repeats an earlier row (DST fall-back).

    strip_tz maps 02:00 CEST and 02:00 CET on the last Sunday of October onto
    the same naive 02:00 label; any later Series.reindex on such an index raises
    "cannot reindex on an axis with duplicate labels". The first (CEST) row is
    kept, matching the drop_duplicates(keep="first") on the energy-history paths.
    """
    if df.empty or "timestamp" not in df.columns:
        return df
    dup = df["timestamp"].duplicated(keep="first")
    n_dup = int(dup.sum())
    if not n_dup:
        return df
    logging.getLogger("energy_forecast").info("DST fall-back: dropped %d duplicate naive timestamp row(s)", n_dup)
    return df.loc[~dup].reset_index(drop=True)


def resolve_timezone(configured: str | None, ha_timezone: Any, logger: logging.Logger) -> str:
    """Resolve the effective timezone for an app instance.

    Precedence: an explicit *configured* value (the `timezone:` key in
    apps.yaml) always wins. If *ha_timezone* (from AppDaemon's
    ``get_timezone()``) is known and differs from *configured*, logs a
    WARNING — a mismatch here silently shifts every historical timestamp by
    the UTC-offset difference between the two zones, so daytime consumption
    can appear to occur at night. Falls back to *ha_timezone*, then
    "Europe/Zurich", when *configured* is not set.

    ha_timezone is coerced to str before use: AppDaemon's get_timezone() is
    type-hinted -> str, but the installed AppDaemonConfig.time_zone field is
    actually validated through pytz.timezone(...), so it returns a pytz
    tzinfo object at runtime, not a str. Passing that object on to
    urllib.parse.quote() in weather.py raises
    "TypeError: quote_from_bytes() expected bytes" (GitHub Discussion #15).
    str() on both pytz tzinfo and zoneinfo.ZoneInfo objects recovers the
    IANA zone key, e.g. str(pytz.timezone("Europe/Zurich")) == "Europe/Zurich".
    """
    if ha_timezone is not None:
        ha_timezone = str(ha_timezone)
    if configured:
        if ha_timezone and ha_timezone != configured:
            logger.warning(
                "Configured timezone %r does not match Home Assistant's timezone %r. "
                "If this is unintentional, historical data will be shifted by the offset "
                "between the two zones (e.g. daytime consumption may appear to occur at "
                "night). Set timezone: %s in apps.yaml, or remove the override to use "
                "HA's timezone automatically.",
                configured,
                ha_timezone,
                ha_timezone,
            )
        return configured
    return ha_timezone or "Europe/Zurich"
