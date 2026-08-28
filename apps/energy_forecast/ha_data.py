"""
History fetching with Local CSV Caching.
This version stores data in a local file so you don't lose history
when Home Assistant purges its database.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import hassapi as hass
    import pandas as pd

from .const import CACHE_PATH, CACHE_PATH_15M, MAX_15MIN_KWH, MAX_HOURLY_KWH

_LOGGER = logging.getLogger("energy_forecast")


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


def validate_energy_cache(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Run defensive health checks on the merged energy cache DataFrame.

    Logs WARNINGs for detected issues; never raises.  Intended to catch
    problems in the cached CSV that should have been filtered upstream
    (meter resets, corrupt rows, clock drift).

    Three checks:
      1. Non-monotonic timestamps — rows where timestamp decreases.
      2. Gaps > 2h between consecutive rows (includes DST spring-forward).
      3. Out-of-range gross_kwh: values outside [0, MAX_HOURLY_KWH].
    """
    import pandas as pd

    if df.empty or "timestamp" not in df.columns:
        return
    try:
        diffs = df["timestamp"].diff()

        # Check 1: non-monotonic timestamps (NaT on first row evaluates False, correct)
        n_bad = int((diffs < pd.Timedelta(0)).sum())
        if n_bad:
            example = df.loc[diffs < pd.Timedelta(0), "timestamp"].iloc[0]
            logger.warning(
                "Cache health: %d non-monotonic timestamp(s) detected (e.g. %s). "
                "Cache may have been written out-of-order.",
                n_bad,
                example,
            )

        # Check 2: gaps > 2h (strict)
        gap_mask = diffs > pd.Timedelta(hours=2)
        n_gaps = int(gap_mask.sum())
        if n_gaps:
            first_gap = df.loc[gap_mask, "timestamp"].iloc[0]
            logger.warning(
                "Cache health: %d gap(s) > 2h in energy history. "
                "First gap ends at %s. "
                "DST spring-forward (late March) causes a natural ~2h gap — "
                "check if expected.",
                n_gaps,
                first_gap,
            )

        # Check 3: out-of-range gross_kwh values
        if "gross_kwh" in df.columns:
            bad_mask = ~((df["gross_kwh"] >= 0) & (df["gross_kwh"] <= MAX_HOURLY_KWH))
            n_bad_vals = int(bad_mask.sum())
            if n_bad_vals:
                example_val = df.loc[bad_mask, "gross_kwh"].iloc[0]
                logger.warning(
                    "Cache health: %d row(s) with gross_kwh outside [0, %.1f] "
                    "(e.g. %.4f). Spike filter may have missed these.",
                    n_bad_vals,
                    MAX_HOURLY_KWH,
                    example_val,
                )
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        logger.error("validate_energy_cache raised unexpectedly: %s", exc)


_EXCLUDED_RANGE_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?$")


def _warn_once(logger: logging.Logger, warned: set | None, key: tuple, msg: str, *args) -> None:
    """Log at WARNING the first time `key` is seen, INFO on repeats.

    `warned` is an optional dedup set shared across an app instance's call
    sites (see load_excluded_ranges/filter_excluded_ranges). `warned=None`
    disables dedup — always logs at WARNING, matching pre-#95 behavior.
    """
    if warned is not None and key in warned:
        logger.info(msg, *args)
    else:
        logger.warning(msg, *args)
        if warned is not None:
            warned.add(key)


def load_excluded_ranges(
    path: Path, timezone: str, logger: logging.Logger, warned: set | None = None
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Load hand-edited known-bad training/prediction date ranges.

    Never raises. A missing file or a header-only file (valid CSV, zero data
    rows) both return [] silently — this is the common/default case. A file
    missing the required 'start'/'end' columns, or one that can't be parsed
    as CSV at all (zero-byte file, torn write), logs one WARNING and returns
    []. A single malformed row (bad date format, timezone offset present,
    end < start) logs a WARNING and is skipped; the rest of the file still
    loads.

    Args:
        path: Path to excluded_ranges.csv (same directory as energy_history.csv).
        timezone: IANA timezone name (e.g. self._timezone) used to flag
            start/end values that fall in a nonexistent local time
            (DST spring-forward gap).
        logger: Logger to report malformed rows/files to.
        warned: Optional dedup set shared across calls. When given, a
            condition (unreadable file, missing columns, a specific bad
            row) that already warned once logs at INFO on repeat calls
            instead of WARNING again — "should only fire on system
            startup" (ROADMAP #95). Omit for the pre-existing behavior of
            warning on every call.

    Returns:
        List of (start, end, reason) tuples — naive local pd.Timestamps,
        end inclusive. A bare-date end (no time component in the raw
        string) is expanded to 23:59:59 of that date; an explicit-time end
        (including 00:00) is used exactly as written.
    """
    import pandas as pd

    if not path.exists():
        return []

    try:
        raw = pd.read_csv(path, dtype=str)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        _warn_once(
            logger,
            warned,
            ("csv_unreadable", str(path)),
            "excluded_ranges.csv unreadable (%s) — treating as no exclusions.",
            exc,
        )
        return []

    if raw.empty:
        return []

    if "start" not in raw.columns or "end" not in raw.columns:
        _warn_once(
            logger,
            warned,
            ("missing_columns", tuple(raw.columns)),
            "excluded_ranges.csv missing required 'start'/'end' column(s) (found: %s) — treating as no exclusions.",
            list(raw.columns),
        )
        return []

    if "reason" not in raw.columns:
        raw["reason"] = ""
    raw["reason"] = raw["reason"].fillna("")

    ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []

    for row_num, row in raw.iterrows():
        try:
            start_raw = str(row["start"]).strip()
            end_raw = str(row["end"]).strip()
            reason = str(row["reason"]).strip()

            if not _EXCLUDED_RANGE_TS_RE.match(start_raw) or not _EXCLUDED_RANGE_TS_RE.match(end_raw):
                _warn_once(
                    logger,
                    warned,
                    ("bad_format", row_num, start_raw, end_raw),
                    "excluded_ranges.csv row %d: '%s'/'%s' doesn't match YYYY-MM-DD or "
                    "YYYY-MM-DD HH:MM — skipping row.",
                    row_num + 2,  # +2: 1-indexed data row, plus the header row
                    start_raw,
                    end_raw,
                )
                continue

            end_is_bare_date = " " not in end_raw
            start = pd.Timestamp(start_raw)
            end = pd.Timestamp(end_raw)
            if end_is_bare_date:
                end = end + pd.Timedelta(hours=23, minutes=59, seconds=59)

            if end < start:
                _warn_once(
                    logger,
                    warned,
                    ("end_before_start", row_num, start_raw, end_raw),
                    "excluded_ranges.csv row %d: end (%s) is before start (%s) — skipping row.",
                    row_num + 2,
                    end,
                    start,
                )
                continue

            for label, ts in (("start", start), ("end", end)):
                try:
                    ts.tz_localize(timezone, nonexistent="raise", ambiguous="raise")
                except ValueError as exc:
                    if "nonexistent" in str(exc):
                        _warn_once(
                            logger,
                            warned,
                            ("dst_gap", row_num, label, str(ts)),
                            "excluded_ranges.csv row %d: %s value %s falls in a nonexistent "
                            "local time (DST spring-forward gap in %s) — check for a "
                            "transcription error. Row still applied as written.",
                            row_num + 2,
                            label,
                            ts,
                            timezone,
                        )
                    # Ambiguous (DST fall-back duplicate hour) is expected/documented
                    # behavior (spec §2) — no warning needed.

            ranges.append((start, end, reason))
        except (ValueError, TypeError, AttributeError) as exc:
            _warn_once(
                logger,
                warned,
                ("row_error", row_num, str(exc)),
                "excluded_ranges.csv row %d malformed (%s) — skipping row.",
                row_num + 2,
                exc,
            )
            continue

    return ranges


_EXCLUSION_WARN_ROW_FRACTION = 0.10
_EXCLUSION_WARN_SPAN_DAYS = 14


def filter_excluded_ranges(
    df: pd.DataFrame,
    ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]],
    logger: logging.Logger,
    warned: set | None = None,
) -> pd.DataFrame:
    """Drop rows whose timestamp falls in any hand-configured known-bad range.

    Never raises — on any unexpected failure, logs .error() and returns df
    unfiltered rather than propagating, so a bug here degrades to "no
    filtering happened," never to "the caller's retrain/predict cycle didn't
    happen."

    Args:
        df: DataFrame with a 'timestamp' column (naive local).
        ranges: Output of load_excluded_ranges() — (start, end, reason)
            tuples, end inclusive.
        logger: Logger to report per-range and total drop counts to.
        warned: Optional dedup set shared across calls (e.g. across an
            app instance's hourly + weekly call sites). When given, a
            range whose escalation condition (see below) already fired
            once logs at INFO on repeat calls instead of WARNING again —
            "should only fire on system startup" (ROADMAP #95). Callers
            that don't need dedup can omit it; every call then warns as
            before.

    Returns:
        df with excluded rows removed (a new DataFrame; input is not
        mutated), index reset to a clean RangeIndex. Returns df unchanged
        (same object) if df is empty, has no 'timestamp' column, or ranges
        is empty/falsy.
    """
    import pandas as pd

    if df.empty or "timestamp" not in df.columns or not ranges:
        return df

    try:
        total_rows = len(df)
        union_mask = pd.Series(False, index=df.index)
        for start, end, reason in ranges:
            range_mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            n_dropped = int(range_mask.sum())
            span_days = (end - start).total_seconds() / 86400
            if n_dropped:
                escalate = (
                    total_rows > 0 and n_dropped > total_rows * _EXCLUSION_WARN_ROW_FRACTION
                ) or span_days > _EXCLUSION_WARN_SPAN_DAYS
                msg = "Excluded range %s -> %s (%s): dropped %d row(s)."
                msg_args = (start, end, reason or "no reason given", n_dropped)
                if escalate:
                    _warn_once(logger, warned, ("escalate", start, end, reason), msg, *msg_args)
                else:
                    logger.info(msg, *msg_args)
            union_mask = union_mask | range_mask

        total_dropped = int(union_mask.sum())
        if total_dropped:
            logger.info("Excluded ranges: %d unique row(s) dropped in total.", total_dropped)

        return df.loc[~union_mask].reset_index(drop=True)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        logger.error("filter_excluded_ranges failed (%s) — returning df unfiltered.", exc)
        return df


def _merge_frames(df_winner: pd.DataFrame, df_loser: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Merge two DataFrames; df_winner's value wins on duplicate timestamps.

    Concatenates loser first so that keep="last" in drop_duplicates() always
    selects the winner's row.  Sorts by timestamp and drops rows with NaN in
    either key column.
    """
    import pandas as pd

    result = (
        pd.concat([df_loser, df_winner])  # winner last → keep="last" selects it
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .dropna(subset=["timestamp", value_col])
        .reset_index(drop=True)
    )
    # pandas 3.x promotes concat(float64_df, empty_object_df) to object dtype.
    # Coerce the value column back to float64 so downstream lag features stay numeric.
    result[value_col] = pd.to_numeric(result[value_col], errors="coerce")
    return result


def _merge_energy_frames(df_winner: pd.DataFrame, df_loser: pd.DataFrame) -> pd.DataFrame:
    """Merge two energy DataFrames; df_winner's value wins on duplicate timestamps."""
    return _merge_frames(df_winner, df_loser, "gross_kwh")


def _raw_to_kwh_diff(
    raw_ha: pd.DataFrame,
    resolution: str,
    max_kwh: float,
    timezone: str = "Europe/Zurich",
    unit_multiplier: float = 1.0,
    end_time: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convert raw cumulative meter readings to per-slot kWh differences.

    Args:
        raw_ha:          DataFrame with columns [timestamp (tz-aware), value].
        resolution:      Pandas offset string, e.g. "1h" or "15min".
        max_kwh:         Per-slot upper bound (in kWh, applied after conversion);
                         slots above this are filtered out.
        timezone:        Timezone to convert to before stripping tzinfo.
        unit_multiplier: Factor to convert raw diff values to kWh.
                         1.0 for kWh sensors, 1000.0 for MWh, 0.001 for Wh.
        end_time:        Optional tz-aware timestamp (same tz as raw_ha) through
                         which the resampled series is extended before
                         forward-filling. Without it, resample() stops at the
                         last raw HA state — so a sensor that stops emitting
                         entirely (e.g. a grid-import meter with solar covering
                         100% of household load, which has nothing new to
                         report) produces no rows at all for the silent hours,
                         instead of the genuine 0.0-kWh diffs they represent.

    Returns:
        DataFrame with columns [timestamp (naive local), gross_kwh], non-negative
        values only. A diff of exactly 0.0 is a real reading (e.g. solar covering
        100% of household load for that hour) and is kept. A *negative* raw diff
        (meter reset) is dropped rather than clipped-and-kept — true consumption
        during a reset is unknown, unlike a genuinely flat hour.
    """
    import pandas as pd

    if raw_ha.empty:
        return pd.DataFrame(columns=["timestamp", "gross_kwh"])
    slotted = raw_ha.set_index("timestamp")["value"].resample(resolution).last()
    if end_time is not None:
        full_range = pd.date_range(
            start=slotted.index.min(),
            end=max(slotted.index.max(), end_time),
            freq=resolution,
            tz=slotted.index.tz,
        )
        slotted = slotted.reindex(full_range)
    slotted = slotted.ffill()
    raw_diff = slotted.diff()
    diff = raw_diff.clip(lower=0).reset_index()
    diff.columns = ["timestamp", "gross_kwh"]
    if unit_multiplier != 1.0:
        diff["gross_kwh"] = diff["gross_kwh"] * unit_multiplier
    if diff["timestamp"].dt.tz is not None:
        diff["timestamp"] = diff["timestamp"].dt.tz_convert(timezone).dt.tz_localize(None)
    keep = (raw_diff.to_numpy() >= 0) & (diff["gross_kwh"] < max_kwh)
    return diff[keep].copy()


def fetch_energy_history(
    app: hass.Hass,
    entity_id: str,
    cache_path: Path = CACHE_PATH,
    timezone: str = "Europe/Zurich",
    unit_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Pull grid-import history, merging local CSV with fresh HA data."""
    import pandas as pd

    # 1. Load existing cache if it exists
    df_cache = pd.DataFrame(columns=["timestamp", "gross_kwh"])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            # CSV may contain tz-aware strings — normalise to naive local time
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
            df_cache["timestamp"] = ts
            _LOGGER.info(f"Loaded {len(df_cache)} records from local cache.")
        except (OSError, pd.errors.ParserError, ValueError) as e:
            _LOGGER.warning(f"Failed to load cache: {e}")

    # 2. Fetch fresh data from HA
    raw_ha = _fetch_history(app, entity_id, days=30, timezone=timezone)

    if raw_ha.empty and df_cache.empty:
        raise ValueError(f"No history found in HA or Cache for {entity_id}")

    # 3. Process HA data into hourly gross kWh
    df_new = _raw_to_kwh_diff(
        raw_ha,
        "1h",
        MAX_HOURLY_KWH,
        timezone=timezone,
        unit_multiplier=unit_multiplier,
        end_time=pd.Timestamp.now(tz=timezone),
    )

    # 4. Merge — fresh HA data wins on timestamp conflicts
    combined = _merge_energy_frames(df_winner=df_new, df_loser=df_cache)
    _check_dst_duplicates(combined, _LOGGER)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")
    validate_energy_cache(combined, _LOGGER)

    # 5. Compact and save back to CSV (full sort + dedup rewrite; runs weekly).
    # This also corrects any stale values that slipped through fetch_recent_energy's
    # append-only path (HA-wins corrections are applied here on the next retrain).
    try:
        combined.to_csv(cache_path, index=False)
        _LOGGER.info(f"Cache compacted. Total history: {len(combined)} hours.")
    except OSError as e:
        _LOGGER.error(f"Failed to save cache: {e}")

    # Strip the current (still-open) hourly bucket so training never sees a
    # partial-hour value.  The CSV write above retains it; the correct full-hour
    # value overwrites it on the next weekly compaction via HA-wins merge.
    completed_cutoff = pd.Timestamp.now(tz=timezone).floor("1h").tz_localize(None)
    return combined[combined["timestamp"] < completed_cutoff]


_FETCH_RECENT_TAIL_ROWS = 400  # 336 h max lag + buffer; limits memory use in hourly updates


def fetch_recent_energy(
    app: hass.Hass,
    entity_id: str,
    cache_path: Path = CACHE_PATH,
    timezone: str = "Europe/Zurich",
    unit_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Lightweight update for hourly sensor refreshes.

    Fetches only the last 2 days of HA history (vs. 30 days in
    fetch_energy_history), merges into the existing CSV cache, and
    returns the last _FETCH_RECENT_TAIL_ROWS rows for lag-feature use.
    Keeps _update_cb well within AppDaemon's 10s callback limit.

    Only the tail of the CSV is read into memory (deque-based), reducing
    memory from O(all rows) to O(_FETCH_RECENT_TAIL_ROWS) per hourly call.
    _retrain() continues to call fetch_energy_history() for a full
    30-day resync once a week.
    """
    import collections
    import io

    import pandas as pd

    # 1. Load only the tail of the cache — enough rows to cover all lag features.
    #    deque(fh, maxlen=N) reads the file sequentially but keeps only the last N
    #    lines in memory, reducing peak memory from O(all rows) to O(N).
    df_cache = pd.DataFrame(columns=["timestamp", "gross_kwh"])
    if cache_path.exists():
        try:
            with open(cache_path) as fh:
                header_line = fh.readline()
                tail_lines = list(collections.deque(fh, maxlen=_FETCH_RECENT_TAIL_ROWS))
            df_cache = pd.read_csv(io.StringIO(header_line + "".join(tail_lines)))
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            _LOGGER.warning(f"Failed to load cache: {e}")

    # 2. Fetch only the last 2 days from HA — enough to cover `hours`
    #    plus a small overlap buffer for the diff() boundary.
    raw_ha = _fetch_history(app, entity_id, days=2, timezone=timezone)

    if raw_ha.empty and df_cache.empty:
        raise ValueError(f"No history found in HA or Cache for {entity_id}")

    # 3. Process into hourly kWh and keep only the recent window
    df_new = _raw_to_kwh_diff(
        raw_ha,
        "1h",
        MAX_HOURLY_KWH,
        timezone=timezone,
        unit_multiplier=unit_multiplier,
        end_time=pd.Timestamp.now(tz=timezone),
    )

    # 4. Merge — fresh HA data wins on timestamp conflicts (for return value)
    combined = _merge_energy_frames(df_winner=df_new, df_loser=df_cache)
    _check_dst_duplicates(combined, _LOGGER)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")

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
            _LOGGER.error(f"Failed to save cache: {e}")

    # Strip the current (still-open) hourly bucket — same guard as fetch_energy_history.
    # The CSV append above may write a partial-hour row; it will be superseded by the
    # next hourly HA fetch (HA-wins merge) and corrected on the next weekly compaction.
    completed_cutoff = pd.Timestamp.now(tz=timezone).floor("1h").tz_localize(None)
    return combined[combined["timestamp"] < completed_cutoff]


def fetch_energy_history_15m(
    app: hass.Hass,
    entity_id: str,
    cache_path: Path = CACHE_PATH_15M,
    timezone: str = "Europe/Zurich",
    unit_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Pull grid-import history at 15-min resolution, merging local CSV with fresh HA data.

    Mirrors fetch_energy_history() but resamples to 15-minute slots. The return
    value is not consumed by the production model; the caller's purpose is to grow
    the background 15m cache for future use.
    """
    import pandas as pd

    df_cache = pd.DataFrame(columns=["timestamp", "gross_kwh"])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            _LOGGER.warning("fetch_energy_history_15m: failed to load cache: %s", e)

    raw_ha = _fetch_history(app, entity_id, days=30, timezone=timezone)

    if raw_ha.empty and df_cache.empty:
        raise ValueError(f"No history found in HA or Cache for {entity_id}")

    df_new = _raw_to_kwh_diff(
        raw_ha,
        "15min",
        MAX_15MIN_KWH,
        timezone=timezone,
        unit_multiplier=unit_multiplier,
        end_time=pd.Timestamp.now(tz=timezone),
    )

    combined = _merge_energy_frames(df_winner=df_new, df_loser=df_cache)
    _check_dst_duplicates(combined, _LOGGER)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")

    try:
        combined.to_csv(cache_path, index=False)
    except OSError as e:
        _LOGGER.error("fetch_energy_history_15m: failed to save cache: %s", e)

    completed_cutoff = pd.Timestamp.now(tz=timezone).floor("15min").tz_localize(None)
    return combined[combined["timestamp"] < completed_cutoff]


_FETCH_RECENT_15M_TAIL_ROWS = 500  # 500 × 15 min = ~125 hours; no consumer needs more


def fetch_recent_energy_15m(
    app: hass.Hass,
    entity_id: str,
    cache_path: Path = CACHE_PATH_15M,
    timezone: str = "Europe/Zurich",
    unit_multiplier: float = 1.0,
) -> None:
    """Lightweight 15-minute cache update. Appends new slots without a full resync.

    Called every hour from _update_sensors(). Returns None — the 15m cache is
    not yet consumed by the production model.
    """
    import collections
    import io

    import pandas as pd

    df_cache = pd.DataFrame(columns=["timestamp", "gross_kwh"])
    if cache_path.exists():
        try:
            with open(cache_path) as fh:
                header_line = fh.readline()
                tail_lines = list(collections.deque(fh, maxlen=_FETCH_RECENT_15M_TAIL_ROWS))
            df_cache = pd.read_csv(io.StringIO(header_line + "".join(tail_lines)))
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            _LOGGER.warning("fetch_recent_energy_15m: failed to load cache: %s", e)

    raw_ha = _fetch_history(app, entity_id, days=2, timezone=timezone)

    if raw_ha.empty and df_cache.empty:
        _LOGGER.warning("fetch_recent_energy_15m: no data from HA or cache for %s", entity_id)
        return

    df_new = _raw_to_kwh_diff(
        raw_ha,
        "15min",
        MAX_15MIN_KWH,
        timezone=timezone,
        unit_multiplier=unit_multiplier,
        end_time=pd.Timestamp.now(tz=timezone),
    )

    combined = _merge_energy_frames(df_winner=df_new, df_loser=df_cache)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")

    existing_ts = set(df_cache["timestamp"]) if not df_cache.empty else set()
    new_rows = combined[~combined["timestamp"].isin(existing_ts)]
    if not new_rows.empty:
        try:
            write_header = not cache_path.exists() or cache_path.stat().st_size == 0
            new_rows.to_csv(cache_path, mode="a", header=write_header, index=False)
        except OSError as e:
            _LOGGER.error("fetch_recent_energy_15m: failed to append cache: %s", e)


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

    df.loc[ev_mask, "gross_kwh"] = np.maximum(0.0, df.loc[ev_mask, "gross_kwh"] - charger_kw)

    return df, ev_df


def split_ev_charging_from_sensor(
    energy_df: pd.DataFrame,
    ev_kwh_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split using actual wallbox readings instead of threshold inference.

    Returns (baseline_df, ev_df) with the same contract as split_ev_charging():
      baseline_df — gross_kwh with wallbox kWh subtracted, clipped ≥ 0
      ev_df       — charging rows; gross_kwh column holds the wallbox kWh
                    (for compatibility with downstream logging / model code)
    """
    import numpy as np
    import pandas as pd

    energy_df = energy_df.copy()
    ev_active = ev_kwh_df[ev_kwh_df["kwh"] > 0][["timestamp", "kwh"]].copy()
    ev_active["_ts"] = pd.to_datetime(ev_active["timestamp"]).dt.floor("1h")

    energy_df["_ts"] = pd.to_datetime(energy_df["timestamp"]).dt.floor("1h")
    merged = energy_df.merge(ev_active[["_ts", "kwh"]], on="_ts", how="left")
    ev_kwh = merged["kwh"].fillna(0.0).values

    ev_mask = ev_kwh > 0
    energy_df.loc[ev_mask, "gross_kwh"] = np.maximum(0.0, energy_df.loc[ev_mask, "gross_kwh"] - ev_kwh[ev_mask])
    energy_df.drop(columns=["_ts"], inplace=True)

    ev_df = energy_df[ev_mask].copy()
    ev_df["gross_kwh"] = ev_kwh[ev_mask]

    return energy_df, ev_df


def ev_sensor_coverage(ev_kwh_df: pd.DataFrame | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the (start, end) hour-range a wallbox kWh cache actually covers.

    Both bounds are floored to the hour, matching how split_ev_charging_from_sensor
    aligns timestamps. Returns None when *ev_kwh_df* is None or empty — callers
    should then treat every row as outside coverage (fall back to threshold
    detection everywhere), since the sensor has no data to be a source of truth for.

    fetch_sub_sensor_history()'s on-disk cache only ever contains rows from
    whenever the sensor entity was first configured onward — it never backfills
    dates before that. This coverage window is how callers know which part of a
    longer energy_df the sensor can actually speak for.
    """
    import pandas as pd

    if ev_kwh_df is None or ev_kwh_df.empty:
        return None
    ts = pd.to_datetime(ev_kwh_df["timestamp"]).dt.floor("1h")
    return ts.min(), ts.max()


def split_ev_charging_hybrid(
    energy_df: pd.DataFrame,
    ev_kwh_df: pd.DataFrame,
    threshold_kwh: float,
    charger_kw: float = 9.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split EV charging using the wallbox sensor where it has coverage, threshold
    detection everywhere else. Same (baseline_df, ev_df) contract as
    split_ev_charging() / split_ev_charging_from_sensor().

    Rationale: fetch_sub_sensor_history()'s cache for ev_charging_sensor only
    covers dates from whenever that entity was configured onward.
    split_ev_charging_from_sensor() alone would silently treat every row before
    that as "0 kWh charged" (its left-join fillna(0.0)), un-excluding real
    pre-wallbox EV sessions from training. This function keeps the sensor as the
    source of truth inside its coverage window (exact per-hour kWh — catches
    variable-power solar-surplus charging the threshold would miss) and falls
    back to threshold detection for rows outside it.
    """
    import pandas as pd

    coverage = ev_sensor_coverage(ev_kwh_df)
    if coverage is None:
        return split_ev_charging(energy_df, threshold_kwh, charger_kw=charger_kw)

    cov_start, cov_end = coverage
    ts_floor = pd.to_datetime(energy_df["timestamp"]).dt.floor("1h")
    in_coverage = (ts_floor >= cov_start) & (ts_floor <= cov_end)

    sensor_baseline, sensor_ev = split_ev_charging_from_sensor(energy_df[in_coverage], ev_kwh_df)
    threshold_baseline, threshold_ev = split_ev_charging(energy_df[~in_coverage], threshold_kwh, charger_kw=charger_kw)

    baseline_df = pd.concat([sensor_baseline, threshold_baseline]).sort_values("timestamp").reset_index(drop=True)
    ev_df = pd.concat([sensor_ev, threshold_ev]).sort_values("timestamp").reset_index(drop=True)
    return baseline_df, ev_df


def _merge_sub_sensor_frames(df_winner: pd.DataFrame, df_loser: pd.DataFrame) -> pd.DataFrame:
    """Merge two sub-sensor DataFrames (columns 'kwh', optional 'program').

    Fresh HA data wins on kwh conflicts.  For the 'program' column the rule is
    "keep existing non-empty label if the winner has none" — program history is
    only fetched for the recent window so older cached labels must be preserved.
    """
    import pandas as pd

    combined = _merge_frames(df_winner, df_loser, "kwh")

    # Preserve program labels: if the winner had an empty/NaN program for a row
    # that the loser already labelled, restore the loser's label.
    if "program" in df_winner.columns or "program" in df_loser.columns:
        # Re-merge only the program column from both sides.
        # Build a loser index: timestamp → program (non-empty only)
        loser_prog = (
            df_loser.copy() if "program" in df_loser.columns else pd.DataFrame(columns=["timestamp", "program"])
        )
        loser_prog = loser_prog[loser_prog["program"].notna() & (loser_prog["program"].astype(str) != "")][
            ["timestamp", "program"]
        ]
        loser_prog = loser_prog.set_index("timestamp")["program"]

        winner_prog = (
            df_winner.copy() if "program" in df_winner.columns else pd.DataFrame(columns=["timestamp", "program"])
        )
        winner_prog = winner_prog[winner_prog["program"].notna() & (winner_prog["program"].astype(str) != "")][
            ["timestamp", "program"]
        ]
        winner_prog = winner_prog.set_index("timestamp")["program"]

        # Winner takes precedence; loser fills gaps
        merged_prog = winner_prog.combine_first(loser_prog)
        combined["program"] = combined["timestamp"].map(merged_prog).fillna("")

    return combined


def _resolve_programs_for_series(
    timestamps: pd.Series,
    prog_df: pd.DataFrame,
) -> pd.Series:
    """Assign a program label to each hourly timestamp using last-value-carry-forward.

    Args:
        timestamps: Series of naive hourly timestamps (the energy CSV rows).
        prog_df:    DataFrame with columns ``timestamp`` (naive) and ``program``
                    (str), representing program state-change events.

    Returns:
        Series of str (same length / index as *timestamps*).  Rows with no
        preceding program event get an empty string.

    The primary strategy is backward LVFC: each row gets the last program state
    at or before its timestamp.  A forward-lookup fallback (tolerance: 2 h) is
    applied when the backward result is empty or the idle sentinel
    ``"no_program"``: this handles the common case where a user starts the
    machine at (say) 12:05, the hourly row is stamped 12:00, and the backward
    lookup still sees the previous ``"no_program"`` state — the forward pass
    finds the real ``"eco"`` event at 12:05 and substitutes it.  The forward
    pass is *not* applied when backward already has a real program label, so
    the tail of a running cycle is never overwritten by the next cycle's program.
    """
    import pandas as pd

    if prog_df.empty:
        return pd.Series("", index=timestamps.index)

    # Ensure timestamp dtype is datetime64 — an empty Series created via
    # pd.DataFrame(columns=[...]) has object dtype, which causes merge_asof
    # to raise "Incompatible merge dtype" in pandas 3.x.
    left = pd.DataFrame({"timestamp": pd.to_datetime(timestamps)}).sort_values("timestamp").reset_index(drop=True)
    right = prog_df[["timestamp", "program"]].sort_values("timestamp")

    # Primary: backward LVFC
    merged_back = pd.merge_asof(left, right, on="timestamp", direction="backward")
    prog_back = merged_back["program"].fillna("").astype(str)

    # Fallback: forward lookup within 2 h (catches late-firing program sensors).
    # 1 h was insufficient when the sensor fires just into the following hour,
    # e.g. cycle starts at 21:00, program fires at 22:05 → gap is 65 min.
    merged_fwd = pd.merge_asof(
        left,
        right,
        on="timestamp",
        direction="forward",
        tolerance=pd.Timedelta("2h"),
    )
    prog_fwd = merged_fwd["program"].fillna("").astype(str)

    # Substitute forward result only when backward gives "no signal"
    idle = prog_back.isin(["", "no_program"])
    has_real_fwd = ~prog_fwd.isin(["", "no_program"])
    result = prog_back.copy()
    result[idle & has_real_fwd] = prog_fwd[idle & has_real_fwd]

    result.index = timestamps.index
    return result


def fetch_sub_sensor_history(
    app: hass.Hass,
    entity_id: str,
    cache_path: Path,
    timezone: str = "Europe/Zurich",
    program_entity_id: str | None = None,
) -> pd.DataFrame:
    """Pull sub-sensor kWh history, merging local CSV cache with fresh HA data.

    Analogous to fetch_energy_history but:
    - Column name is 'kwh' (not 'gross_kwh')
    - Zero-diff hours are kept (diff >= 0 instead of > 0): they represent the
      appliance being off and must appear as 0 so lag features return 0 (not NaN)
      during idle hours.
    - Suitable for any cumulative kWh meter (heat pump, dishwasher, etc.)

    When *program_entity_id* is provided the returned DataFrame (and the written
    CSV) will contain a ``program`` column: the program label active at each
    hourly timestamp, resolved via last-value-carry-forward from the program
    sensor's state-change history.  Existing non-empty labels in the cache are
    preserved even when the fresh window does not cover them.
    """
    import pandas as pd

    df_cache = pd.DataFrame(columns=["timestamp", "kwh"])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            _LOGGER.warning(f"Failed to load sub-sensor cache {cache_path.name}: {e}")

    raw_ha = _fetch_history(app, entity_id, days=30, timezone=timezone)

    if raw_ha.empty and df_cache.empty:
        _LOGGER.warning(f"No history found for sub-sensor {entity_id} — skipping.")
        return pd.DataFrame(columns=["timestamp", "kwh"])

    if not raw_ha.empty:
        hourly = raw_ha.set_index("timestamp")["value"].resample("1h").last().ffill()
        diff = hourly.diff().clip(lower=0).reset_index()
        diff.columns = ["timestamp", "kwh"]
        if diff["timestamp"].dt.tz is not None:
            diff["timestamp"] = diff["timestamp"].dt.tz_localize(None)
        df_new = diff[diff["kwh"] < MAX_HOURLY_KWH].copy()
    else:
        # Use explicit dtypes so downstream merge_asof / pd.concat don't see
        # object-typed timestamp columns (pandas 3.x raises on mixed dtypes).
        df_new = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[us]"), "kwh": pd.Series(dtype="float64")})

    if program_entity_id:
        prog_df = fetch_program_sensor_history(app, program_entity_id, days=30, timezone=timezone)
        df_new["program"] = _resolve_programs_for_series(df_new["timestamp"], prog_df)

    combined = _merge_sub_sensor_frames(df_winner=df_new, df_loser=df_cache)

    try:
        combined.to_csv(cache_path, index=False)
    except OSError as e:
        _LOGGER.error(f"Failed to save sub-sensor cache {cache_path.name}: {e}")

    return combined


def fetch_recent_sub_sensor(
    app: hass.Hass,
    entity_id: str,
    cache_path: Path,
    timezone: str = "Europe/Zurich",
    program_entity_id: str | None = None,
) -> pd.DataFrame:
    """Lightweight update for sub-sensor hourly refreshes.

    Fetches only the last 2 days of HA history, merges into the existing CSV
    cache, and returns the full cache for lag-feature use.  Analogous to
    fetch_recent_energy but for sub-sensors (column name 'kwh', keeps zeros).

    Unlike fetch_energy_history (raises ValueError when both sources empty),
    this function returns an empty DataFrame silently so a missing sub-sensor
    does not abort the hourly update for the main sensor.

    When *program_entity_id* is provided the new rows written to the CSV will
    include a ``program`` label resolved from the last 2 days of program sensor
    history.  Existing cached labels for older rows are preserved.
    """
    import pandas as pd

    df_cache = pd.DataFrame(columns=["timestamp", "kwh"])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            _LOGGER.warning(f"Failed to load sub-sensor cache {cache_path.name}: {e}")

    raw_ha = _fetch_history(app, entity_id, days=2, timezone=timezone)

    if raw_ha.empty and df_cache.empty:
        _LOGGER.warning(f"No recent data for sub-sensor {entity_id}.")
        return pd.DataFrame(columns=["timestamp", "kwh"])

    if not raw_ha.empty:
        hourly = raw_ha.set_index("timestamp")["value"].resample("1h").last().ffill()
        diff = hourly.diff().clip(lower=0).reset_index()
        diff.columns = ["timestamp", "kwh"]
        if diff["timestamp"].dt.tz is not None:
            diff["timestamp"] = diff["timestamp"].dt.tz_localize(None)
        df_new = diff[diff["kwh"] < MAX_HOURLY_KWH].copy()
    else:
        # Use explicit dtypes so downstream merge_asof / pd.concat don't see
        # object-typed timestamp columns (pandas 3.x raises on mixed dtypes).
        df_new = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[us]"), "kwh": pd.Series(dtype="float64")})

    if program_entity_id:
        prog_df = fetch_program_sensor_history(app, program_entity_id, days=2, timezone=timezone)
        df_new["program"] = _resolve_programs_for_series(df_new["timestamp"], prog_df)

    combined = _merge_sub_sensor_frames(df_winner=df_new, df_loser=df_cache)

    try:
        combined.to_csv(cache_path, index=False)
    except OSError as e:
        _LOGGER.error(f"Failed to save sub-sensor cache {cache_path.name}: {e}")

    return combined


def fetch_generic_sensor_history(
    app: hass.Hass,
    entity_id: str,
    cache_path: Path,
    column_name: str = "value",
    timezone: str = "Europe/Zurich",
) -> pd.DataFrame:
    """Pull generic sensor history (absolute values), merging local CSV cache with fresh HA data."""
    import pandas as pd

    df_cache = pd.DataFrame(columns=["timestamp", column_name])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            _LOGGER.warning(f"Failed to load generic sensor cache {cache_path.name}: {e}")

    raw_ha = _fetch_history(app, entity_id, days=30, timezone=timezone)

    if raw_ha.empty and df_cache.empty:
        _LOGGER.warning(f"No history found for sensor {entity_id} — skipping.")
        return pd.DataFrame(columns=["timestamp", column_name])

    if not raw_ha.empty:
        hourly = raw_ha.set_index("timestamp")["value"].resample("1h").last().ffill().reset_index()
        hourly.columns = ["timestamp", column_name]
        if hourly["timestamp"].dt.tz is not None:
            hourly["timestamp"] = hourly["timestamp"].dt.tz_localize(None)
        df_new = hourly.copy()
    else:
        df_new = pd.DataFrame(columns=["timestamp", column_name])

    combined = _merge_frames(df_winner=df_new, df_loser=df_cache, value_col=column_name)

    try:
        combined.to_csv(cache_path, index=False)
    except OSError as e:
        _LOGGER.error(f"Failed to save generic sensor cache {cache_path.name}: {e}")

    return combined


def fetch_recent_generic_sensor(
    app: hass.Hass,
    entity_id: str,
    cache_path: Path,
    column_name: str = "value",
    timezone: str = "Europe/Zurich",
    quiet_if_empty: bool = False,
) -> pd.DataFrame:
    """Lightweight update for generic absolute sensor hourly refreshes.

    `quiet_if_empty` logs the "no recent data" case at DEBUG instead of WARNING —
    for sensors with a known, expected idle period (e.g. a heat pump's live COP
    sensor, which only reports while actively heating) an empty result every hour
    is normal, not something to page-noise on.
    """
    import pandas as pd

    df_cache = pd.DataFrame(columns=["timestamp", column_name])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            _LOGGER.warning(f"Failed to load generic sensor cache {cache_path.name}: {e}")

    raw_ha = _fetch_history(app, entity_id, days=2, timezone=timezone)

    if raw_ha.empty and df_cache.empty:
        log = _LOGGER.debug if quiet_if_empty else _LOGGER.warning
        log(f"No recent data for sensor {entity_id}.")
        return pd.DataFrame(columns=["timestamp", column_name])

    if not raw_ha.empty:
        hourly = raw_ha.set_index("timestamp")["value"].resample("1h").last().ffill().reset_index()
        hourly.columns = ["timestamp", column_name]
        if hourly["timestamp"].dt.tz is not None:
            hourly["timestamp"] = hourly["timestamp"].dt.tz_localize(None)
        df_new = hourly.copy()
    else:
        df_new = pd.DataFrame(columns=["timestamp", column_name])

    combined = _merge_frames(df_winner=df_new, df_loser=df_cache, value_col=column_name)

    try:
        combined.to_csv(cache_path, index=False)
    except OSError as e:
        _LOGGER.error(f"Failed to save generic sensor cache {cache_path.name}: {e}")

    return combined


def fetch_climate_history(
    app: hass.Hass,
    entity_id: str,
    cache_path: Path,
    timezone: str = "Europe/Zurich",
) -> pd.DataFrame:
    """Pull climate current_temperature and setpoint history, merging local cache with fresh HA data."""
    import pandas as pd

    df_cache = pd.DataFrame(columns=["timestamp", "current_temp", "setpoint"])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            _LOGGER.warning(f"Failed to load climate cache {cache_path.name}: {e}")

    raw_ha = _fetch_history(app, entity_id, days=30, timezone=timezone, include_attributes=True)

    if raw_ha.empty and df_cache.empty:
        _LOGGER.warning(f"No history found for climate {entity_id} — skipping.")
        return pd.DataFrame(columns=["timestamp", "current_temp", "setpoint"])

    if not raw_ha.empty:
        raw_ha["timestamp"] = pd.to_datetime(raw_ha["timestamp"]).dt.floor("1h")
        # For climate, we need to extract and resample both 'current_temperature' and 'temperature'
        # attributes from the raw attribute rows.
        ha_df = raw_ha[["timestamp", "current_temperature", "temperature"]].copy()
        ha_df.columns = ["timestamp", "current_temp", "setpoint"]
        ha_df = ha_df.sort_values("timestamp")

        # Resample to hourly grid
        hourly = ha_df.set_index("timestamp").resample("1h").last().ffill().reset_index()
        if hourly["timestamp"].dt.tz is not None:
            hourly["timestamp"] = hourly["timestamp"].dt.tz_localize(None)
        df_new = hourly.copy()
    else:
        df_new = pd.DataFrame(columns=["timestamp", "current_temp", "setpoint"])

    # Multi-column merge
    combined = (
        pd.concat([df_cache, df_new])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .dropna(subset=["timestamp"])
        .reset_index(drop=True)
    )

    try:
        combined.to_csv(cache_path, index=False)
    except OSError as e:
        _LOGGER.error(f"Failed to save climate cache {cache_path.name}: {e}")

    return combined


def fetch_recent_climate(
    app: hass.Hass,
    entity_id: str,
    cache_path: Path,
    timezone: str = "Europe/Zurich",
) -> pd.DataFrame:
    """Lightweight update for climate hourly refreshes."""
    import pandas as pd

    df_cache = pd.DataFrame(columns=["timestamp", "current_temp", "setpoint"])
    if cache_path.exists():
        try:
            df_cache = pd.read_csv(cache_path)
            ts = pd.to_datetime(df_cache["timestamp"], format="mixed")
            if ts.dt.tz is not None:
                ts = ts.dt.tz_convert(timezone).dt.tz_localize(None)
            df_cache["timestamp"] = ts
        except (OSError, pd.errors.ParserError, ValueError) as e:
            _LOGGER.warning(f"Failed to load climate cache {cache_path.name}: {e}")

    raw_ha = _fetch_history(app, entity_id, days=2, timezone=timezone, include_attributes=True)

    if raw_ha.empty and df_cache.empty:
        _LOGGER.warning(f"No recent data for climate {entity_id}.")
        return pd.DataFrame(columns=["timestamp", "current_temp", "setpoint"])

    if not raw_ha.empty:
        raw_ha["timestamp"] = pd.to_datetime(raw_ha["timestamp"]).dt.floor("1h")
        ha_df = raw_ha[["timestamp", "current_temperature", "temperature"]].copy()
        ha_df.columns = ["timestamp", "current_temp", "setpoint"]
        ha_df = ha_df.sort_values("timestamp")
        hourly = ha_df.set_index("timestamp").resample("1h").last().ffill().reset_index()
        if hourly["timestamp"].dt.tz is not None:
            hourly["timestamp"] = hourly["timestamp"].dt.tz_localize(None)
        df_new = hourly.copy()
    else:
        df_new = pd.DataFrame(columns=["timestamp", "current_temp", "setpoint"])

    combined = (
        pd.concat([df_cache, df_new])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .dropna(subset=["timestamp"])
        .reset_index(drop=True)
    )

    try:
        combined.to_csv(cache_path, index=False)
    except OSError as e:
        _LOGGER.error(f"Failed to save climate cache {cache_path.name}: {e}")

    return combined


def fetch_boolean_entity_history(
    app: hass.Hass,
    entity_id: str | None,
    days: int = 30,
    timezone: str = "Europe/Zurich",
) -> pd.DataFrame:
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
        _LOGGER.warning(f"get_history failed for away entity {entity_id}: {exc}")
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
            ts = pd.to_datetime(state["last_changed"]).tz_convert(timezone)
            events.append({"timestamp": ts, "state": s})
        except (ValueError, KeyError, TypeError):
            continue

    if not events:
        _LOGGER.warning(f"No usable history for away entity {entity_id} — is_away will be 0.")
        return pd.DataFrame(columns=["timestamp", "is_away"])

    events_df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
    ev_ser = events_df.set_index("timestamp")["state"]

    # Hourly grid: span from first to last event
    start = ev_ser.index[0].floor("1h")
    end = ev_ser.index[-1].floor("1h")
    hourly = pd.date_range(start, end, freq="1h", tz=timezone)

    # Forward-fill state changes onto the grid; hours before first event default to "off"
    combined_idx = ev_ser.index.union(hourly)
    filled = ev_ser.reindex(combined_idx).ffill().reindex(hourly).fillna("off")

    # Convert to naive local time (strip tz, local time already correct)
    timestamps_naive = hourly.tz_localize(None)

    return pd.DataFrame(
        {
            "timestamp": timestamps_naive,
            "is_away": (filled == "on").astype(int).values,
        }
    )


def fetch_presence_history(
    app: hass.Hass,
    entity_ids: list[str] | None,
    days: int = 30,
    timezone: str = "Europe/Zurich",
) -> pd.DataFrame:
    """Return hourly occupancy count from person entity state history.

    Fetches up to *days* of history for each entity in *entity_ids* (e.g. person.alice),
    counts how many are in the "home" state per hour, and returns a DataFrame with
    one row per hour. Person entities use state "home" or "not_home" (not "on"/"off").

    Args:
        app: AppDaemon app instance
        entity_ids: List of HA person entity IDs, or None
        days: Number of days to fetch

    Returns:
        pd.DataFrame with columns {"timestamp" (naive Europe/Zurich), "people_home" (int count)}.
        Returns an empty DataFrame (no rows) when entity_ids is None/empty or all fetches fail.
    """
    import pandas as pd

    if not entity_ids:
        return pd.DataFrame(columns=["timestamp", "people_home"])

    all_per_entity = {}  # entity_id -> DataFrame of hourly 0/1 for that person

    for entity_id in entity_ids:
        try:
            raw = app.get_history(entity_id=entity_id, days=days)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(f"get_history failed for presence entity {entity_id}: {exc}")
            continue

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
                if s not in ("home", "not_home"):
                    continue
                ts = pd.to_datetime(state["last_changed"]).tz_convert(timezone)
                events.append({"timestamp": ts, "state": s})
            except (ValueError, KeyError, TypeError):
                continue

        if not events:
            _LOGGER.warning(f"No usable history for presence entity {entity_id}.")
            continue

        events_df = pd.DataFrame(events).sort_values("timestamp").reset_index(drop=True)
        ev_ser = events_df.set_index("timestamp")["state"]

        # Hourly grid
        start = ev_ser.index[0].floor("1h")
        end = ev_ser.index[-1].floor("1h")
        hourly = pd.date_range(start, end, freq="1h", tz=timezone)

        # Forward-fill state changes; hours before first event default to "not_home"
        combined_idx = ev_ser.index.union(hourly)
        filled = ev_ser.reindex(combined_idx).ffill().reindex(hourly).fillna("not_home")

        all_per_entity[entity_id] = (filled == "home").astype(int)

    if not all_per_entity:
        _LOGGER.warning(f"No usable presence history from {entity_ids} — people_home will be 0.")
        return pd.DataFrame(columns=["timestamp", "people_home"])

    # Union all timestamps from all entities using reduce
    from functools import reduce

    all_ts = reduce(
        lambda idx1, idx2: idx1.union(idx2),
        [s.index for s in all_per_entity.values()],
        pd.DatetimeIndex([]),
    )
    all_ts = pd.DatetimeIndex(all_ts.sort_values())

    # Reindex each series to full union, fill gaps with 0 (not_home)
    stacked = pd.concat(
        [all_per_entity[e].reindex(all_ts, fill_value=0) for e in all_per_entity],
        axis=1,
    )

    # Sum across entities and convert to naive Europe/Zurich
    people_count = stacked.sum(axis=1)
    timestamps_naive = all_ts.tz_localize(None)

    return pd.DataFrame(
        {
            "timestamp": timestamps_naive,
            "people_home": people_count.astype(int).values,
        }
    )


def fetch_program_sensor_history(
    app: hass.Hass,
    entity_id: str,
    days: int = 30,
    timezone: str = "Europe/Zurich",
) -> pd.DataFrame:
    """Return state-change history for a string program sensor.

    Columns: timestamp (naive, local tz), program (str, lowercased).
    States "unavailable"/"unknown"/"" are dropped.
    Returns empty DataFrame on failure — not cached to CSV.
    """
    import pandas as pd

    try:
        raw = app.get_history(entity_id=entity_id, days=days)
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning(f"get_history failed for program sensor {entity_id}: {exc}")
        return pd.DataFrame(columns=["timestamp", "program"])

    if isinstance(raw, dict):
        states = raw.get(entity_id, [])
    elif isinstance(raw, list) and raw:
        states = raw[0] if isinstance(raw[0], list) else raw
    else:
        states = []

    rows = []
    for state in states:
        try:
            s = str(state.get("state", "")).strip().lower()
            if s in ("unavailable", "unknown", ""):
                continue
            ts = pd.to_datetime(state["last_changed"]).tz_convert(timezone)
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            rows.append({"timestamp": ts, "program": s})
        except (ValueError, KeyError, TypeError):
            continue

    return (
        pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["timestamp", "program"])
    )


def _fetch_history(
    app: hass.Hass, entity_id: str, days: int, timezone: str = "Europe/Zurich", include_attributes: bool = False
) -> pd.DataFrame:
    """Internal helper to call AppDaemon's get_history API."""
    import pandas as pd

    try:
        raw = app.get_history(entity_id=entity_id, days=days)
    except Exception as exc:
        _LOGGER.error(f"get_history failed for {entity_id}: {exc}")
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
            ts = pd.to_datetime(state["last_updated"]).tz_convert(timezone)
            if include_attributes:
                # Extract all attributes for specialized callers (e.g. climate)
                attrs = state.get("attributes", {})
                row = {"timestamp": ts}
                row.update(attrs)
                rows.append(row)
            else:
                raw_state = state["state"]
                if raw_state.lower() == "on":
                    val = 1.0
                elif raw_state.lower() == "off":
                    val = 0.0
                else:
                    val = float(raw_state)
                rows.append({"timestamp": ts, "value": val})
        except (ValueError, KeyError, TypeError):
            continue

    return pd.DataFrame(rows)
