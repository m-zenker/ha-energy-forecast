"""
History fetching via AppDaemon's get_history() API.

No HA internals, no HTTP calls, no token. AppDaemon handles the connection
to HA's WebSocket API and returns state history as plain Python dicts.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import hassapi as hass

_LOGGER = logging.getLogger(__name__)

HISTORY_MONTHS  = 12
MAX_HOURLY_KWH  = 50    # Spike / meter-reset filter threshold


def fetch_energy_history(app: "hass.Hass", entity_id: str) -> Any:
    """Pull cumulative grid-import history and differentiate to hourly gross kWh."""
    import pandas as pd

    end   = datetime.now()
    start = end - timedelta(days=30 * HISTORY_MONTHS)
    raw   = _fetch_history(app, entity_id, days=HISTORY_MONTHS * 30)

    if raw.empty:
        raise ValueError(
            f"No history returned for {entity_id}. "
            "Check the entity_id in apps.yaml and that the recorder has data."
        )

    hourly = raw.set_index("timestamp")["value"].resample("1h").last().ffill()
    diff   = hourly.diff().clip(lower=0).reset_index()
    diff.columns = ["timestamp", "gross_kwh"]
    diff   = diff[(diff["gross_kwh"] > 0) & (diff["gross_kwh"] < MAX_HOURLY_KWH)].dropna()

    app.log(f"Energy history: {len(diff)} hourly records for {entity_id}", level="DEBUG")
    return diff


def fetch_outdoor_temp_history(
    app: "hass.Hass",
    entity_id: str,
    start: datetime,
    end: datetime,
) -> Any:
    """Pull outdoor temperature history resampled to hourly means."""
    import pandas as pd

    days = max(1, (end - start).days + 1)
    raw  = _fetch_history(app, entity_id, days=days)

    if raw.empty:
        app.log(f"No outdoor temp history for {entity_id}", level="WARNING")
        return pd.DataFrame(columns=["timestamp", "outdoor_temp_live"])

    # Trim to the requested window
    raw = raw[(raw["timestamp"] >= pd.Timestamp(start, tz="Europe/Zurich")) &
              (raw["timestamp"] <= pd.Timestamp(end,   tz="Europe/Zurich"))]

    hourly = raw.set_index("timestamp")["value"].resample("1h").mean().reset_index()
    hourly.columns = ["timestamp", "outdoor_temp_live"]
    return hourly.dropna()


def _fetch_history(app: "hass.Hass", entity_id: str, days: int) -> Any:
    """
    Call AppDaemon's get_history() and normalise the result into a tidy
    DataFrame with columns [timestamp, value].

    AppDaemon's get_history() returns a list of lists:
        [ [ {state, last_updated, ...}, ... ] ]   ← one inner list per entity
    """
    import pandas as pd

    try:
        raw = app.get_history(entity_id=entity_id, days=days)
    except Exception as exc:  # noqa: BLE001
        app.log(f"get_history failed for {entity_id}: {exc}", level="ERROR")
        return pd.DataFrame(columns=["timestamp", "value"])

    # Normalise: AD may return a list-of-lists or a dict keyed by entity_id
    if isinstance(raw, dict):
        states = raw.get(entity_id, [])
    elif isinstance(raw, list) and raw:
        states = raw[0] if isinstance(raw[0], list) else raw
    else:
        app.log(f"Unexpected get_history format for {entity_id}: {type(raw)}", level="WARNING")
        return pd.DataFrame(columns=["timestamp", "value"])

    rows = []
    for state in states:
        try:
            val = float(state["state"])
            ts  = pd.to_datetime(state["last_updated"]).tz_convert("Europe/Zurich")
            rows.append({"timestamp": ts, "value": val})
        except (ValueError, KeyError, TypeError):
            continue

    if not rows:
        return pd.DataFrame(columns=["timestamp", "value"])

    return (
        pd.DataFrame(rows)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
