"""
History fetching with Local CSV Caching.
This version stores data in a local file so you don't lose history
when Home Assistant purges its database.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import hassapi as hass

_LOGGER = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / "energy_history.csv"
MAX_HOURLY_KWH = 50    # Spike / meter-reset filter threshold


def _merge_energy_frames(df_winner: Any, df_loser: Any) -> Any:
    """Merge two energy DataFrames; df_winner's value wins on duplicate timestamps.

    Concatenates loser first so that keep="last" in drop_duplicates() always
    selects the winner's row.  Sorts by timestamp and drops rows with NaN in
    either key column.
    """
    import pandas as pd
    return (
        pd.concat([df_loser, df_winner])   # winner last → keep="last" selects it
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .dropna(subset=["timestamp", "gross_kwh"])
        .reset_index(drop=True)
    )


def fetch_energy_history(app: "hass.Hass", entity_id: str) -> Any:
    """Pull grid-import history, merging local CSV with fresh HA data."""
    import pandas as pd

    # 1. Load existing cache if it exists
    df_cache = pd.DataFrame(columns=["timestamp", "gross_kwh"])
    if CACHE_PATH.exists():
        try:
            df_cache = pd.read_csv(CACHE_PATH)
            ts = pd.to_datetime(df_cache["timestamp"])
            # CSV may contain tz-aware strings — normalise to naive Europe/Zurich
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
            df_cache["timestamp"] = ts
            app.log(f"Loaded {len(df_cache)} records from local cache.")
        except Exception as e:
            app.log(f"Failed to load cache: {e}", level="WARNING")

    # 2. Fetch fresh data from HA
    raw_ha = _fetch_history(app, entity_id, days=30)

    if raw_ha.empty and df_cache.empty:
        raise ValueError(f"No history found in HA or Cache for {entity_id}")

    # 3. Process HA data into hourly gross kWh
    if not raw_ha.empty:
        hourly = raw_ha.set_index("timestamp")["value"].resample("1h").last().ffill()
        diff = hourly.diff().clip(lower=0).reset_index()
        diff.columns = ["timestamp", "gross_kwh"]
        if hasattr(diff["timestamp"].dtype, "tz") and diff["timestamp"].dt.tz is not None:
            diff["timestamp"] = diff["timestamp"].dt.tz_localize(None)
        df_new = diff[(diff["gross_kwh"] > 0) & (diff["gross_kwh"] < MAX_HOURLY_KWH)].copy()
    else:
        df_new = pd.DataFrame(columns=["timestamp", "gross_kwh"])

    # 4. Merge — fresh HA data wins on timestamp conflicts
    combined = _merge_energy_frames(df_winner=df_new, df_loser=df_cache)

    # 5. Save back to CSV
    try:
        combined.to_csv(CACHE_PATH, index=False)
        app.log(f"Cache updated. Total history: {len(combined)} hours.")
    except Exception as e:
        app.log(f"Failed to save cache: {e}", level="ERROR")

    return combined


def fetch_recent_energy(app: "hass.Hass", entity_id: str, hours: int = 6) -> Any:
    """Lightweight update for hourly sensor refreshes.

    Fetches only the last `hours` of HA history (vs. 30 days in
    fetch_energy_history), merges into the existing CSV cache, and
    returns the full cache for lag-feature use.  Keeps _update_cb
    well within AppDaemon's 10s callback limit.

    _retrain() continues to call fetch_energy_history() for a full
    30-day resync once a week.
    """
    import pandas as pd

    # 1. Load existing cache — this is the bulk of our history
    df_cache = pd.DataFrame(columns=["timestamp", "gross_kwh"])
    if CACHE_PATH.exists():
        try:
            df_cache = pd.read_csv(CACHE_PATH)
            ts = pd.to_datetime(df_cache["timestamp"])
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except Exception as e:
            app.log(f"Failed to load cache: {e}", level="WARNING")

    # 2. Fetch only the last 2 days from HA — enough to cover `hours`
    #    plus a small overlap buffer for the diff() boundary.
    raw_ha = _fetch_history(app, entity_id, days=2)

    if raw_ha.empty and df_cache.empty:
        raise ValueError(f"No history found in HA or Cache for {entity_id}")

    # 3. Process into hourly kWh and keep only the recent window
    if not raw_ha.empty:
        hourly = raw_ha.set_index("timestamp")["value"].resample("1h").last().ffill()
        diff = hourly.diff().clip(lower=0).reset_index()
        diff.columns = ["timestamp", "gross_kwh"]
        if hasattr(diff["timestamp"].dtype, "tz") and diff["timestamp"].dt.tz is not None:
            diff["timestamp"] = diff["timestamp"].dt.tz_localize(None)
        df_new = diff[(diff["gross_kwh"] > 0) & (diff["gross_kwh"] < MAX_HOURLY_KWH)].copy()
    else:
        df_new = pd.DataFrame(columns=["timestamp", "gross_kwh"])

    # 4. Merge — fresh HA data wins on timestamp conflicts
    combined = _merge_energy_frames(df_winner=df_new, df_loser=df_cache)

    try:
        combined.to_csv(CACHE_PATH, index=False)
    except Exception as e:
        app.log(f"Failed to save cache: {e}", level="ERROR")

    return combined


def split_ev_charging(df: Any, threshold_kwh: float) -> tuple[Any, Any]:
    """Split a history DataFrame into baseline and EV charging portions.

    Charging hours are identified by gross_kwh > threshold_kwh.  Your charger
    runs at ~9 kW (constant), so charging hours cluster tightly above 7 kWh/h.
    The normal household ceiling is ~3.9 kWh/h, giving a clean gap at 4.5.

    Returns:
        baseline_df  — all rows retained; EV hours have gross_kwh replaced
                        with (gross_kwh - 9.0), clipped to ≥ 0.  This keeps
                        the true household co-load (lighting, cooking etc.)
                        visible to the model rather than dropping the row, and
                        preserves shift()-based lag alignment in training.
        ev_df        — only the rows classified as EV charging, with the
                        original gross_kwh values, for publishing EV sensors.
    """
    import numpy as np

    df = df.copy()
    ev_mask = df["gross_kwh"] > threshold_kwh

    ev_df = df[ev_mask].copy()

    # Subtract the fixed charger load (~9 kW → 9 kWh/h) from charging hours,
    # leaving the concurrent household baseline intact.
    df.loc[ev_mask, "gross_kwh"] = np.maximum(
        0.0, df.loc[ev_mask, "gross_kwh"] - 9.0
    )

    return df, ev_df


def _fetch_history(app: "hass.Hass", entity_id: str, days: int) -> Any:
    """Internal helper to call AppDaemon's get_history API."""
    import pandas as pd
    try:
        raw = app.get_history(entity_id=entity_id, days=days)
    except Exception as exc:
        app.log(f"get_history failed for {entity_id}: {exc}", level="ERROR")
        return pd.DataFrame()

    if isinstance(raw, dict):
        states = raw.get(entity_id, [])
    elif isinstance(raw, list) and raw:
        states = raw[0] if isinstance(raw[0], list) else raw
    else:
        return pd.DataFrame()

    rows = []
    for state in states:
        try:
            val = float(state["state"])
            ts  = pd.to_datetime(state["last_updated"]).tz_convert("Europe/Zurich")
            rows.append({"timestamp": ts, "value": val})
        except (ValueError, KeyError, TypeError):
            continue

    return pd.DataFrame(rows)
    