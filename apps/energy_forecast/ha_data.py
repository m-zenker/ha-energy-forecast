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
    import pandas as pd
    import hassapi as hass

from .const import CACHE_PATH, MAX_HOURLY_KWH

_LOGGER = logging.getLogger(__name__)


def _check_dst_duplicates(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Warn if the DataFrame contains duplicate naive timestamps.

    Duplicate naive timestamps arise during the DST fall-back transition
    (e.g. Europe/Zurich: last Sunday of October, 03:00 CEST → 02:00 CET).
    After tz_localize(None) the naive hour 02:xx appears twice — once for the
    summer-time reading, once for the winter-time reading.  The merge keeps
    both rows, so callers should be aware that downstream aggregations may
    double-count that hour.

    Spring-forward gaps (e.g. 02:00–02:59 never occurring in March) are filled
    by the resample/ffill in the fetch functions and do NOT produce duplicates;
    they are accepted behaviour and are not flagged here.
    """
    if df.empty or "timestamp" not in df.columns:
        return
    dup_mask = df["timestamp"].duplicated(keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup:
        dup_times = df.loc[dup_mask, "timestamp"].unique()
        logger.warning(
            "DST fall-back: %d rows share %d duplicate naive timestamp(s) after merge "
            "(e.g. %s). The ambiguous hour appears twice — downstream aggregations "
            "may double-count it.",
            n_dup,
            len(dup_times),
            dup_times[0],
        )


def _merge_energy_frames(df_winner: pd.DataFrame, df_loser: pd.DataFrame) -> pd.DataFrame:
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


def fetch_energy_history(
    app: "hass.Hass",
    entity_id: str,
    cache_path: Path = CACHE_PATH,
) -> pd.DataFrame:
    """Pull grid-import history, merging local CSV with fresh HA data."""
    import pandas as pd

    # 1. Load existing cache if it exists
    df_cache = pd.DataFrame(columns=["timestamp", "gross_kwh"])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            # CSV may contain tz-aware strings — normalise to naive Europe/Zurich
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
            df_cache["timestamp"] = ts
            app.log(f"Loaded {len(df_cache)} records from local cache.")
        except (OSError, pd.errors.ParserError, ValueError) as e:
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
        if diff["timestamp"].dt.tz is not None:
            diff["timestamp"] = diff["timestamp"].dt.tz_localize(None)
        df_new = diff[(diff["gross_kwh"] > 0) & (diff["gross_kwh"] < MAX_HOURLY_KWH)].copy()
    else:
        df_new = pd.DataFrame(columns=["timestamp", "gross_kwh"])

    # 4. Merge — fresh HA data wins on timestamp conflicts
    combined = _merge_energy_frames(df_winner=df_new, df_loser=df_cache)
    _check_dst_duplicates(combined, _LOGGER)

    # 5. Compact and save back to CSV (full sort + dedup rewrite; runs weekly).
    # This also corrects any stale values that slipped through fetch_recent_energy's
    # append-only path (HA-wins corrections are applied here on the next retrain).
    try:
        combined.to_csv(cache_path, index=False)
        app.log(f"Cache compacted. Total history: {len(combined)} hours.")
    except OSError as e:
        app.log(f"Failed to save cache: {e}", level="ERROR")

    return combined


def fetch_recent_energy(app: "hass.Hass", entity_id: str, cache_path: Path = CACHE_PATH) -> pd.DataFrame:
    """Lightweight update for hourly sensor refreshes.

    Fetches only the last 2 days of HA history (vs. 30 days in
    fetch_energy_history), merges into the existing CSV cache, and
    returns the full cache for lag-feature use.  Keeps _update_cb
    well within AppDaemon's 10s callback limit.

    _retrain() continues to call fetch_energy_history() for a full
    30-day resync once a week.
    """
    import pandas as pd

    # 1. Load existing cache — this is the bulk of our history
    df_cache = pd.DataFrame(columns=["timestamp", "gross_kwh"])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
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
        if diff["timestamp"].dt.tz is not None:
            diff["timestamp"] = diff["timestamp"].dt.tz_localize(None)
        df_new = diff[(diff["gross_kwh"] > 0) & (diff["gross_kwh"] < MAX_HOURLY_KWH)].copy()
    else:
        df_new = pd.DataFrame(columns=["timestamp", "gross_kwh"])

    # 4. Merge — fresh HA data wins on timestamp conflicts (for return value)
    combined = _merge_energy_frames(df_winner=df_new, df_loser=df_cache)
    _check_dst_duplicates(combined, _LOGGER)

    # 5. Append only genuinely new timestamps to CSV — avoids full rewrite each hour.
    # Timestamps already in the cache are not re-written; any HA-wins corrections for
    # existing rows will be fixed during the next weekly fetch_energy_history compaction.
    existing_ts = set(df_cache["timestamp"]) if not df_cache.empty else set()
    new_rows = combined[~combined["timestamp"].isin(existing_ts)]
    if not new_rows.empty:
        try:
            # Determine header inside the same except block to avoid a TOCTOU race:
            # another process could delete/truncate the file between stat() and to_csv().
            write_header = not cache_path.exists() or cache_path.stat().st_size == 0
            new_rows.to_csv(cache_path, mode="a", header=write_header, index=False)
        except OSError as e:
            app.log(f"Failed to save cache: {e}", level="ERROR")

    return combined


def split_ev_charging(
    df: pd.DataFrame,
    threshold_kwh: float,
    charger_kw: float = 9.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a history DataFrame into baseline and EV charging portions.

    Charging hours are identified by gross_kwh > threshold_kwh.  The charger
    load (charger_kw, default 9 kW) is subtracted from those hours, leaving
    the concurrent household co-load intact.

    Returns:
        baseline_df  — all rows retained; EV hours have gross_kwh replaced
                        with (gross_kwh - charger_kw), clipped to ≥ 0.  This
                        keeps the true household co-load (lighting, cooking
                        etc.) visible to the model rather than dropping the
                        row, and preserves shift()-based lag alignment.
        ev_df        — only the rows classified as EV charging, with the
                        original gross_kwh values, for publishing EV sensors.
    """
    import numpy as np

    df = df.copy()
    ev_mask = df["gross_kwh"] > threshold_kwh

    ev_df = df[ev_mask].copy()

    df.loc[ev_mask, "gross_kwh"] = np.maximum(
        0.0, df.loc[ev_mask, "gross_kwh"] - charger_kw
    )

    return df, ev_df


def _merge_sub_sensor_frames(df_winner: "pd.DataFrame", df_loser: "pd.DataFrame") -> "pd.DataFrame":
    """Merge two sub-sensor DataFrames (column 'kwh'); fresh HA data wins on conflicts.

    Thin wrapper around _merge_energy_frames: temporarily renames 'kwh' to
    'gross_kwh' so the shared helper can be used, then renames back.  This
    avoids duplicating the concat/drop_duplicates/sort logic and keeps both
    sub-sensor fetch paths in lockstep with the main energy merge semantics.

    Unlike fetch_sub_sensor_history (which raises ValueError when both sources
    are empty), sub-sensor functions return an empty DataFrame silently — callers
    should handle both cases.
    """
    import pandas as pd

    rename_to   = {"kwh": "gross_kwh"}
    rename_back = {"gross_kwh": "kwh"}
    combined = _merge_energy_frames(
        df_winner=df_winner.rename(columns=rename_to),
        df_loser=df_loser.rename(columns=rename_to),
    )
    return combined.rename(columns=rename_back)


def fetch_sub_sensor_history(
    app: "hass.Hass",
    entity_id: str,
    cache_path: Path,
) -> pd.DataFrame:
    """Pull sub-sensor kWh history, merging local CSV cache with fresh HA data.

    Analogous to fetch_energy_history but:
    - Column name is 'kwh' (not 'gross_kwh')
    - Zero-diff hours are kept (diff >= 0 instead of > 0): they represent the
      appliance being off and must appear as 0 so lag features return 0 (not NaN)
      during idle hours.
    - Suitable for any cumulative kWh meter (heat pump, dishwasher, etc.)
    """
    import pandas as pd

    df_cache = pd.DataFrame(columns=["timestamp", "kwh"])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            app.log(f"Failed to load sub-sensor cache {cache_path.name}: {e}", level="WARNING")

    raw_ha = _fetch_history(app, entity_id, days=30)

    if raw_ha.empty and df_cache.empty:
        app.log(f"No history found for sub-sensor {entity_id} — skipping.", level="WARNING")
        return pd.DataFrame(columns=["timestamp", "kwh"])

    if not raw_ha.empty:
        hourly = raw_ha.set_index("timestamp")["value"].resample("1h").last().ffill()
        diff = hourly.diff().clip(lower=0).reset_index()
        diff.columns = ["timestamp", "kwh"]
        if diff["timestamp"].dt.tz is not None:
            diff["timestamp"] = diff["timestamp"].dt.tz_localize(None)
        df_new = diff[diff["kwh"] < MAX_HOURLY_KWH].copy()
    else:
        df_new = pd.DataFrame(columns=["timestamp", "kwh"])

    combined = _merge_sub_sensor_frames(df_winner=df_new, df_loser=df_cache)

    try:
        combined.to_csv(cache_path, index=False)
    except OSError as e:
        app.log(f"Failed to save sub-sensor cache {cache_path.name}: {e}", level="ERROR")

    return combined


def fetch_recent_sub_sensor(
    app: "hass.Hass",
    entity_id: str,
    cache_path: Path,
) -> pd.DataFrame:
    """Lightweight update for sub-sensor hourly refreshes.

    Fetches only the last 2 days of HA history, merges into the existing CSV
    cache, and returns the full cache for lag-feature use.  Analogous to
    fetch_recent_energy but for sub-sensors (column name 'kwh', keeps zeros).

    Unlike fetch_energy_history (raises ValueError when both sources empty),
    this function returns an empty DataFrame silently so a missing sub-sensor
    does not abort the hourly update for the main sensor.
    """
    import pandas as pd

    df_cache = pd.DataFrame(columns=["timestamp", "kwh"])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            app.log(f"Failed to load sub-sensor cache {cache_path.name}: {e}", level="WARNING")

    raw_ha = _fetch_history(app, entity_id, days=2)

    if raw_ha.empty and df_cache.empty:
        app.log(f"No recent data for sub-sensor {entity_id}.", level="WARNING")
        return pd.DataFrame(columns=["timestamp", "kwh"])

    if not raw_ha.empty:
        hourly = raw_ha.set_index("timestamp")["value"].resample("1h").last().ffill()
        diff = hourly.diff().clip(lower=0).reset_index()
        diff.columns = ["timestamp", "kwh"]
        if diff["timestamp"].dt.tz is not None:
            diff["timestamp"] = diff["timestamp"].dt.tz_localize(None)
        df_new = diff[diff["kwh"] < MAX_HOURLY_KWH].copy()
    else:
        df_new = pd.DataFrame(columns=["timestamp", "kwh"])

    combined = _merge_sub_sensor_frames(df_winner=df_new, df_loser=df_cache)

    try:
        combined.to_csv(cache_path, index=False)
    except OSError as e:
        app.log(f"Failed to save sub-sensor cache {cache_path.name}: {e}", level="ERROR")

    return combined


def fetch_boolean_entity_history(
    app: "hass.Hass",
    entity_id: str | None,
    days: int = 30,
) -> "pd.DataFrame":
    """Return hourly is_away flags from a boolean entity's state history.

    Fetches up to *days* of history for *entity_id* (e.g. input_boolean.vacation_mode),
    forward-fills state changes onto an hourly grid, and returns a DataFrame with
    one row per hour.

    Returns:
        pd.DataFrame with columns {"timestamp" (naive Europe/Zurich), "is_away" (0/1)}.
        Returns an empty DataFrame (no rows) when entity_id is None or the fetch fails.
    """
    import pandas as pd

    if entity_id is None:
        return pd.DataFrame(columns=["timestamp", "is_away"])

    try:
        raw = app.get_history(entity_id=entity_id, days=days)
    except Exception as exc:  # noqa: BLE001
        app.log(f"get_history failed for away entity {entity_id}: {exc}", level="WARNING")
        return pd.DataFrame(columns=["timestamp", "is_away"])

    if isinstance(raw, dict):
        states = raw.get(entity_id, [])
    elif isinstance(raw, list) and raw:
        states = raw[0] if isinstance(raw[0], list) else raw
    else:
        states = []

    events = []
    for state in states:
        try:
            s = str(state.get("state", "")).lower()
            if s not in ("on", "off"):
                continue
            ts = pd.to_datetime(state["last_changed"]).tz_convert("Europe/Zurich")
            events.append({"timestamp": ts, "state": s})
        except (ValueError, KeyError, TypeError):
            continue

    if not events:
        app.log(
            f"No usable history for away entity {entity_id} — is_away will be 0.",
            level="WARNING",
        )
        return pd.DataFrame(columns=["timestamp", "is_away"])

    events_df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
    ev_ser = events_df.set_index("timestamp")["state"]

    # Hourly grid: span from first to last event
    start = ev_ser.index[0].floor("1h")
    end   = ev_ser.index[-1].floor("1h")
    hourly = pd.date_range(start, end, freq="1h", tz="Europe/Zurich")

    # Forward-fill state changes onto the grid; hours before first event default to "off"
    combined_idx = ev_ser.index.union(hourly)
    filled = ev_ser.reindex(combined_idx).ffill().reindex(hourly).fillna("off")

    # Convert to naive Europe/Zurich (strip tz, local time already correct)
    timestamps_naive = hourly.tz_localize(None)

    return pd.DataFrame({
        "timestamp": timestamps_naive,
        "is_away": (filled == "on").astype(int).values,
    })


def _fetch_history(app: "hass.Hass", entity_id: str, days: int) -> pd.DataFrame:
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
    