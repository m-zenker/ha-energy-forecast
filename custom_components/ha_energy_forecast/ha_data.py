"""
Home Assistant data access — pulls energy history and outdoor temperature
history from the HA recorder via the REST API.

All functions are synchronous (blocking) and must be called via
hass.async_add_executor_job().
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
import requests

from .const import HISTORY_MONTHS, MAX_HOURLY_KWH

_LOGGER = logging.getLogger(__name__)


def fetch_energy_history(ha_url: str, token: str, entity_id: str) -> pd.DataFrame:
    """
    Pull cumulative grid-import history and differentiate to hourly gross kWh.

    Returns DataFrame with columns: [timestamp, gross_kwh]
    """
    end = datetime.now()
    start = end - timedelta(days=30 * HISTORY_MONTHS)

    raw = _fetch_ha_history(ha_url, token, entity_id, start, end)
    if raw.empty:
        raise ValueError(f"No history returned for {entity_id}")

    hourly = (
        raw.set_index("timestamp")["value"]
        .resample("1h")
        .last()
        .ffill()
    )
    diff = hourly.diff().clip(lower=0).reset_index()
    diff.columns = ["timestamp", "gross_kwh"]

    # Drop meter resets and noise
    diff = diff[(diff["gross_kwh"] > 0) & (diff["gross_kwh"] < MAX_HOURLY_KWH)]
    diff = diff.dropna()

    _LOGGER.debug("Energy history: %d hourly records for %s", len(diff), entity_id)
    return diff


def fetch_outdoor_temp_history(
    ha_url: str, token: str, entity_id: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """
    Pull outdoor temperature history and resample to hourly means.

    Returns DataFrame with columns: [timestamp, outdoor_temp_live]
    """
    raw = _fetch_ha_history(ha_url, token, entity_id, start, end)
    if raw.empty:
        _LOGGER.warning("No outdoor temp history for %s", entity_id)
        return pd.DataFrame(columns=["timestamp", "outdoor_temp_live"])

    hourly = (
        raw.set_index("timestamp")["value"]
        .resample("1h")
        .mean()
        .reset_index()
    )
    hourly.columns = ["timestamp", "outdoor_temp_live"]
    hourly = hourly.dropna()

    _LOGGER.debug(
        "Outdoor temp history: %d hourly records for %s", len(hourly), entity_id
    )
    return hourly


# ── Internal ──────────────────────────────────────────────────────────────────

def _fetch_ha_history(
    ha_url: str,
    token: str,
    entity_id: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Call the HA /api/history/period endpoint and return a tidy DataFrame
    with columns [timestamp, value].
    """
    url = (
        f"{ha_url}/api/history/period/{start.isoformat()}"
        f"?filter_entity_id={entity_id}"
        f"&end_time={end.isoformat()}"
        f"&significant_changes_only=false"
        f"&minimal_response=true"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        _LOGGER.error("HA history request failed for %s: %s", entity_id, exc)
        return pd.DataFrame(columns=["timestamp", "value"])

    if not data or not data[0]:
        return pd.DataFrame(columns=["timestamp", "value"])

    rows = []
    for state in data[0]:
        try:
            val = float(state["state"])
            ts = pd.to_datetime(state["last_updated"]).tz_convert("Europe/Zurich")
            rows.append({"timestamp": ts, "value": val})
        except (ValueError, KeyError, TypeError):
            continue

    if not rows:
        return pd.DataFrame(columns=["timestamp", "value"])

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
