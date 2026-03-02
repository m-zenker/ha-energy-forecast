"""Home Assistant data access — pulls energy and temperature history from HA recorder.

Uses the HA recorder's internal Python API so no HTTP calls, no token, and no
hardcoded URL are needed. Works on all HA install types (HAOS, Container, Core).
The functions here are synchronous and intended to be called from an executor
thread via hass.async_add_executor_job().
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from homeassistant.core import HomeAssistant

from .const import HISTORY_MONTHS, MAX_HOURLY_KWH

_LOGGER = logging.getLogger(__name__)


def fetch_energy_history(hass: HomeAssistant, entity_id: str) -> Any:
    """Pull cumulative grid-import history and differentiate to hourly gross kWh."""
    import pandas as pd  # noqa: PLC0415

    end = datetime.now()
    start = end - timedelta(days=30 * HISTORY_MONTHS)
    raw = _fetch_recorder_history(hass, entity_id, start, end)

    if raw.empty:
        raise ValueError(f"No history returned for {entity_id}")

    hourly = raw.set_index("timestamp")["value"].resample("1h").last().ffill()
    diff = hourly.diff().clip(lower=0).reset_index()
    diff.columns = ["timestamp", "gross_kwh"]
    diff = diff[(diff["gross_kwh"] > 0) & (diff["gross_kwh"] < MAX_HOURLY_KWH)].dropna()

    _LOGGER.debug("Energy history: %d hourly records for %s", len(diff), entity_id)
    return diff


def fetch_outdoor_temp_history(
    hass: HomeAssistant,
    entity_id: str,
    start: datetime,
    end: datetime,
) -> Any:
    """Pull outdoor temperature history resampled to hourly means."""
    import pandas as pd  # noqa: PLC0415

    raw = _fetch_recorder_history(hass, entity_id, start, end)
    if raw.empty:
        _LOGGER.warning("No outdoor temp history for %s", entity_id)
        return pd.DataFrame(columns=["timestamp", "outdoor_temp_live"])

    hourly = raw.set_index("timestamp")["value"].resample("1h").mean().reset_index()
    hourly.columns = ["timestamp", "outdoor_temp_live"]
    return hourly.dropna()


def _fetch_recorder_history(
    hass: HomeAssistant,
    entity_id: str,
    start: datetime,
    end: datetime,
) -> Any:
    """Fetch state history directly from the HA recorder.

    No HTTP connection, no auth token, and no hardcoded URL required.
    state_changes_during_period() is thread-safe and designed to be called
    from executor threads.
    """
    import pandas as pd  # noqa: PLC0415
    from homeassistant.components.recorder.history import (  # noqa: PLC0415
        state_changes_during_period,
    )

    try:
        states_by_entity = state_changes_during_period(
            hass,
            start,
            end,
            entity_id=entity_id,
            no_attributes=True,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("Recorder history fetch failed for %s: %s", entity_id, exc)
        return pd.DataFrame(columns=["timestamp", "value"])

    states = states_by_entity.get(entity_id, [])
    if not states:
        _LOGGER.warning("Recorder returned no states for %s", entity_id)
        return pd.DataFrame(columns=["timestamp", "value"])

    rows = []
    for state in states:
        try:
            val = float(state.state)
            rows.append({"timestamp": state.last_updated, "value": val})
        except (ValueError, TypeError):
            continue

    if not rows:
        return pd.DataFrame(columns=["timestamp", "value"])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("Europe/Zurich")
    return df.sort_values("timestamp").reset_index(drop=True)
