# Excluded Training Date Ranges Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an operator hand-edit a CSV of known-bad datetime ranges (hardware fault windows) that gets excluded from both model training and live prediction, without a code deploy.

**Architecture:** Two new pure functions in `ha_data.py` (`load_excluded_ranges` parses the CSV, `filter_excluded_ranges` drops matching rows from any timestamp-indexed DataFrame), wired into both places `energy_df`/`recent_actuals` are assembled (`_retrain()` and `_update_sensors()` in `energy_forecast.py`), plus a one-line fix in `model.py`'s physics holdout-cutoff calculation so it no longer assumes gap-free history.

**Tech Stack:** Python 3.13, pandas (no new dependencies — DST nonexistent/ambiguous-time detection uses `pd.Timestamp.tz_localize(tz, nonexistent="raise", ambiguous="raise")`, not `pytz`; confirmed by direct import that `pytz` is not actually installed in this project's env despite being mentioned in nearby docstrings/comments — it must not be imported anywhere in new code).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-19-excluded-training-ranges-design.md` (rev. 3, multi-stakeholder-reviewed, approved). Every task below implements a specific section of it — cited inline.
- `excluded_ranges.csv` lives in `self._cache_path.parent` (same directory as `energy_history.csv`), columns `start` (required), `end` (required), `reason` (optional, defaults to `""`). Extra columns are ignored.
- `start`/`end` must match `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` exactly (regex-validated) — no lenient/mixed date parsing, no timezone offsets accepted.
- A bare-date `end` (no time component in the raw string) expands to `23:59:59` of that date. An explicit-time `end` (including `00:00`) is used exactly as given.
- Both new functions **never raise**: `load_excluded_ranges` degrades to `[]` on any file/row problem; `filter_excluded_ranges` degrades to returning `df` unfiltered on any unexpected failure.
- `filter_excluded_ranges` computes each range's dropped-row count against the **original** `df` (not sequentially against a shrinking frame), plus logs one final total-unique-rows-dropped line (via the union of all range masks).
- A single range's per-range log line escalates from `INFO` to `WARNING` when it drops >10% of `len(df)` or spans >14 days (typo/fat-finger tripwire).
- Follow `validate_energy_cache()`'s existing conventions in `ha_data.py`: typed `logger: logging.Logger` parameter, Google-style docstring with an explicit "never raises" sentence, `Args:`/`Returns:` sections.
- Test env: always run tests via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest ...` — never bare `python`/`pytest` (base conda env silently produces false failures/degraded codepaths per this project's `CLAUDE.md`).
- Timezone used throughout: production default is `Europe/Zurich`; tests exercise `Europe/Zurich`'s known 2026 DST transitions (spring-forward gap 2026-03-29 02:00–02:59, fall-back ambiguous hour 2026-10-25 02:00–02:59) and, for one test confirming the `timezone` parameter is actually used, `US/Eastern`'s 2026 spring-forward gap (2026-03-08 02:00–02:59).

---

### Task 1: `load_excluded_ranges()` in `ha_data.py`

Implements spec §2 (format rules, end-of-day rule, DST handling) and §3's `load_excluded_ranges` contract.

**Files:**
- Modify: `apps/energy_forecast/ha_data.py` (add after `validate_energy_cache`, i.e. after the closing `return` inside the `except` block that ends that function — currently line 110's `except (KeyError, ValueError, TypeError, AttributeError) as exc:` block, so insert the new code starting right after line 111's blank line before `def _merge_frames`)
- Test: `tests/test_ha_data.py` (add a new `TestLoadExcludedRanges` class after `TestValidateCacheIntegration`, i.e. after the class ending around line 1111 where `TestFetchPresenceHistory` begins — insert immediately before that)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `load_excluded_ranges(path: Path, timezone: str, logger: logging.Logger) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]` — a module-level function in `ha_data.py`, importable as `from energy_forecast.ha_data import load_excluded_ranges` or via `ha_data.load_excluded_ranges(...)`. Task 3 and Task 4 both call this directly.

- [ ] **Step 1: Write the failing tests**

Add this import near the top of `tests/test_ha_data.py`, right after the existing `_LOGGER` module-level logger declaration (line 641, `_LOGGER = _logging.getLogger("energy_forecast.ha_data")`) is too late for this class if placed earlier in the file — instead, add a **second**, file-scoped `_LOGGER` reference is unnecessary: reuse the existing one. Insert this new test class directly before `class TestFetchPresenceHistory:` (search for that exact line):

```python
# ── #new load_excluded_ranges / filter_excluded_ranges ────────────────────────

from energy_forecast.ha_data import load_excluded_ranges  # noqa: E402


class TestLoadExcludedRanges:
    def test_missing_file_returns_empty_no_log(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert not caplog.records

    def test_header_only_file_returns_empty_no_warning(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert not caplog.records

    def test_well_formed_multi_row_file(self, tmp_path):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text(
            "start,end,reason\n"
            "2026-07-19 14:00,2026-07-21 09:30,gplug fault\n"
            "2026-07-25 00:00,2026-07-25 12:00,second fault\n"
        )
        result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == [
            (pd.Timestamp("2026-07-19 14:00"), pd.Timestamp("2026-07-21 09:30"), "gplug fault"),
            (pd.Timestamp("2026-07-25 00:00"), pd.Timestamp("2026-07-25 12:00"), "second fault"),
        ]

    def test_reason_column_absent_defaults_to_empty_string(self, tmp_path):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end\n2026-07-19,2026-07-20\n")
        result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == [(pd.Timestamp("2026-07-19 00:00"), pd.Timestamp("2026-07-20 23:59:59"), "")]

    def test_extra_column_ignored(self, tmp_path):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason,ticket\n2026-07-19,2026-07-20,fault,JIRA-123\n")
        result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert len(result) == 1
        assert result[0][2] == "fault"

    def test_malformed_row_skipped_others_still_load(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text(
            "start,end,reason\n"
            "not-a-date,2026-07-20,bad row\n"
            "2026-07-25,2026-07-26,good row\n"
        )
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert len(result) == 1
        assert result[0][2] == "good row"
        assert any("row" in r.message.lower() for r in caplog.records)

    def test_end_before_start_skipped(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-07-20 10:00,2026-07-19 10:00,backwards\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert caplog.records

    def test_missing_required_columns_returns_empty_with_warning(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("begin,finish\n2026-07-19,2026-07-20\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert any("start" in r.message.lower() or "end" in r.message.lower() for r in caplog.records)

    def test_ambiguous_date_format_rejected(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n07/19/2026,07/20/2026,slash format\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []

    def test_timezone_offset_rejected(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-07-19 14:00+02:00,2026-07-20 14:00,tz offset\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []

    def test_bare_date_end_expands_to_end_of_day(self, tmp_path):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-07-19,2026-07-21,multi-day\n")
        result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result[0][1] == pd.Timestamp("2026-07-21 23:59:59")

    def test_explicit_time_end_used_exactly(self, tmp_path):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-07-19,2026-07-21 00:00,exact midnight\n")
        result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result[0][1] == pd.Timestamp("2026-07-21 00:00:00")

    def test_spring_forward_nonexistent_time_warns_distinctly(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-03-29 02:30,2026-03-29 04:00,gap\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert len(result) == 1  # still loaded, just warned
        assert any("nonexistent" in r.message.lower() for r in caplog.records)

    def test_fall_back_ambiguous_time_loads_without_special_warning(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-10-25 02:00,2026-10-25 03:00,fall-back\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert len(result) == 1
        assert not any("nonexistent" in r.message.lower() for r in caplog.records)

    def test_truncated_csv_returns_empty_with_warning(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_bytes(b"")  # zero-byte file, simulates a torn Samba write
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert caplog.records

    def test_different_timezone_changes_spring_forward_detection(self, tmp_path, caplog):
        """US/Eastern's 2026 spring-forward gap (Mar 8) differs from Europe/Zurich's (Mar 29) —
        proves the timezone parameter is actually used, not hardcoded."""
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-03-08 02:30,2026-03-08 04:00,gap\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            load_excluded_ranges(path, "US/Eastern", _LOGGER)
        assert any("nonexistent" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py::TestLoadExcludedRanges -v`
Expected: FAIL (collection error) with `ImportError: cannot import name 'load_excluded_ranges'`

- [ ] **Step 3: Implement `load_excluded_ranges`**

In `apps/energy_forecast/ha_data.py`, add `import re` to the top-level imports (after `import logging`, before `from pathlib import Path`):

```python
import logging
import re
from pathlib import Path
```

Then insert this after `validate_energy_cache`'s closing lines (after the `except (KeyError, ValueError, TypeError, AttributeError) as exc:` / `logger.error(...)` block that ends that function, before `def _merge_frames`):

```python
_EXCLUDED_RANGE_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?$")


def load_excluded_ranges(
    path: Path, timezone: str, logger: logging.Logger
) -> list[tuple["pd.Timestamp", "pd.Timestamp", str]]:
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
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        logger.warning("excluded_ranges.csv unreadable (%s) — treating as no exclusions.", exc)
        return []

    if raw.empty:
        return []

    if "start" not in raw.columns or "end" not in raw.columns:
        logger.warning(
            "excluded_ranges.csv missing required 'start'/'end' column(s) (found: %s) — "
            "treating as no exclusions.",
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
                logger.warning(
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
                logger.warning(
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
                        logger.warning(
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
            logger.warning("excluded_ranges.csv row %d malformed (%s) — skipping row.", row_num + 2, exc)
            continue

    return ranges
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py::TestLoadExcludedRanges -v`
Expected: `15 passed`

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: all tests pass (same pass count as before this task, plus the 15 new ones)

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/ha_data.py tests/test_ha_data.py
git commit -m "feat: add load_excluded_ranges() for hand-configured bad-data windows"
```

---

### Task 2: `filter_excluded_ranges()` in `ha_data.py`

Implements spec §3's `filter_excluded_ranges` contract (never-raise guard, original-df row counting, escalation threshold, total-unique-dropped logging).

**Files:**
- Modify: `apps/energy_forecast/ha_data.py` (add immediately after `load_excluded_ranges`, the function added in Task 1)
- Test: `tests/test_ha_data.py` (add `TestFilterExcludedRanges` class immediately after `TestLoadExcludedRanges`, added in Task 1)

**Interfaces:**
- Consumes: the `list[tuple[pd.Timestamp, pd.Timestamp, str]]` shape produced by Task 1's `load_excluded_ranges` (though this function accepts any list of 3-tuples — it doesn't call `load_excluded_ranges` itself, callers do).
- Produces: `filter_excluded_ranges(df: pd.DataFrame, ranges: list[tuple[pd.Timestamp, pd.Timestamp, str]], logger: logging.Logger) -> pd.DataFrame`. Task 3 and Task 4 both call this directly, right after calling `load_excluded_ranges`.

- [ ] **Step 1: Write the failing tests**

Add immediately after `TestLoadExcludedRanges` in `tests/test_ha_data.py`:

```python
from energy_forecast.ha_data import filter_excluded_ranges  # noqa: E402


class TestFilterExcludedRanges:
    def _df(self, start="2024-01-01 00:00", periods=48, freq="1h"):
        ts = pd.date_range(start, periods=periods, freq=freq)
        return pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * periods})

    def test_rows_inside_range_dropped_outside_kept(self):
        df = self._df()
        ranges = [(pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-01 14:00"), "fault")]
        result = filter_excluded_ranges(df, ranges, _LOGGER)
        assert len(result) == 48 - 5  # 10:00..14:00 inclusive = 5 hourly rows
        in_range = (result["timestamp"] >= ranges[0][0]) & (result["timestamp"] <= ranges[0][1])
        assert not in_range.any()

    def test_multiple_non_overlapping_ranges_both_apply(self):
        df = self._df()
        ranges = [
            (pd.Timestamp("2024-01-01 02:00"), pd.Timestamp("2024-01-01 03:00"), "a"),
            (pd.Timestamp("2024-01-01 20:00"), pd.Timestamp("2024-01-01 21:00"), "b"),
        ]
        result = filter_excluded_ranges(df, ranges, _LOGGER)
        assert len(result) == 48 - 4  # 2 rows each range

    def test_overlapping_ranges_correct_per_range_and_total_counts(self, caplog):
        df = self._df()
        ranges = [
            (pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-01 14:00"), "a"),  # 5 rows
            (pd.Timestamp("2024-01-01 12:00"), pd.Timestamp("2024-01-01 16:00"), "b"),  # 5 rows, overlaps a
        ]
        with caplog.at_level(logging.INFO, logger="energy_forecast"):
            result = filter_excluded_ranges(df, ranges, _LOGGER)
        # union of [10:00-14:00] and [12:00-16:00] = [10:00-16:00] = 7 unique hourly rows
        assert len(result) == 48 - 7
        messages = [r.message for r in caplog.records]
        assert sum("dropped 5 row" in m for m in messages) == 2  # each range logged independently, 5 each
        assert any("7" in m and ("unique" in m.lower() or "total" in m.lower()) for m in messages)

    def test_empty_ranges_list_is_noop(self):
        df = self._df()
        result = filter_excluded_ranges(df, [], _LOGGER)
        pd.testing.assert_frame_equal(result, df)

    def test_range_entirely_outside_cache_is_noop(self, caplog):
        df = self._df()
        ranges = [(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), "old")]
        with caplog.at_level(logging.INFO, logger="energy_forecast"):
            result = filter_excluded_ranges(df, ranges, _LOGGER)
        assert len(result) == 48
        assert any("dropped 0 row" in r.message for r in caplog.records)

    def test_boundary_exact_match_drops_single_row(self):
        df = self._df()
        exact_ts = df["timestamp"].iloc[10]
        ranges = [(exact_ts, exact_ts, "single hour")]
        result = filter_excluded_ranges(df, ranges, _LOGGER)
        assert len(result) == 47
        assert exact_ts not in result["timestamp"].values

    def test_large_drop_fraction_escalates_to_warning(self, caplog):
        df = self._df(periods=48)
        ranges = [(df["timestamp"].iloc[0], df["timestamp"].iloc[10], "big")]  # 11/48 rows = 23%
        with caplog.at_level(logging.INFO, logger="energy_forecast"):
            filter_excluded_ranges(df, ranges, _LOGGER)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_long_span_escalates_to_warning(self, caplog):
        df = self._df(periods=24 * 20, freq="1h")  # 20 days of hourly data
        ranges = [(df["timestamp"].iloc[0], df["timestamp"].iloc[-1], "long")]  # ~20-day span > 14
        with caplog.at_level(logging.INFO, logger="energy_forecast"):
            filter_excluded_ranges(df, ranges, _LOGGER)
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_short_small_range_stays_info(self, caplog):
        df = self._df()
        ranges = [(pd.Timestamp("2024-01-01 10:00"), pd.Timestamp("2024-01-01 11:00"), "small")]
        with caplog.at_level(logging.INFO, logger="energy_forecast"):
            filter_excluded_ranges(df, ranges, _LOGGER)
        assert not any(r.levelno == logging.WARNING for r in caplog.records)

    def test_malformed_input_returns_df_unfiltered(self, caplog):
        df = pd.DataFrame({"timestamp": ["not", "timestamps"], "gross_kwh": [1.0, 2.0]})
        ranges = [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), "x")]
        with caplog.at_level(logging.ERROR, logger="energy_forecast"):
            result = filter_excluded_ranges(df, ranges, _LOGGER)
        pd.testing.assert_frame_equal(result, df)
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_empty_df_returns_df_unchanged(self):
        df = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        ranges = [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), "x")]
        result = filter_excluded_ranges(df, ranges, _LOGGER)
        assert result.empty

    def test_missing_timestamp_column_returns_df_unchanged(self):
        df = pd.DataFrame({"gross_kwh": [1.0, 2.0]})
        ranges = [(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), "x")]
        result = filter_excluded_ranges(df, ranges, _LOGGER)
        pd.testing.assert_frame_equal(result, df)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py::TestFilterExcludedRanges -v`
Expected: FAIL (collection error) with `ImportError: cannot import name 'filter_excluded_ranges'`

- [ ] **Step 3: Implement `filter_excluded_ranges`**

Insert immediately after `load_excluded_ranges` in `apps/energy_forecast/ha_data.py`:

```python
_EXCLUSION_WARN_ROW_FRACTION = 0.10
_EXCLUSION_WARN_SPAN_DAYS = 14


def filter_excluded_ranges(
    df: "pd.DataFrame", ranges: list[tuple["pd.Timestamp", "pd.Timestamp", str]], logger: logging.Logger
) -> "pd.DataFrame":
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
            escalate = (
                total_rows > 0 and n_dropped > total_rows * _EXCLUSION_WARN_ROW_FRACTION
            ) or span_days > _EXCLUSION_WARN_SPAN_DAYS
            log_fn = logger.warning if escalate else logger.info
            log_fn(
                "Excluded range %s -> %s (%s): dropped %d row(s).",
                start,
                end,
                reason or "no reason given",
                n_dropped,
            )
            union_mask = union_mask | range_mask

        total_dropped = int(union_mask.sum())
        if total_dropped:
            logger.info("Excluded ranges: %d unique row(s) dropped in total.", total_dropped)

        return df.loc[~union_mask].reset_index(drop=True)
    except (KeyError, ValueError, TypeError, AttributeError) as exc:
        logger.error("filter_excluded_ranges failed (%s) — returning df unfiltered.", exc)
        return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py::TestFilterExcludedRanges -v`
Expected: `12 passed`

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/ha_data.py tests/test_ha_data.py
git commit -m "feat: add filter_excluded_ranges() to drop known-bad rows by timestamp"
```

---

### Task 3: Wire exclusion filtering into `_retrain()` (training path)

Implements spec §4.1 (training-path integration, `MIN_HISTORY_HOURS` re-check) and the recency-anchor staleness logging from §4.3.

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (`_retrain()`, currently starting at line 1453)
- Test: `tests/test_energy_forecast.py` (new test class near the existing `TestFifteenMinCache`/`_FakeRetrain` fixtures, around line 3428)

**Interfaces:**
- Consumes: `ha_data.load_excluded_ranges(path, timezone, logger)` and `ha_data.filter_excluded_ranges(df, ranges, logger)` from Tasks 1 and 2.
- Produces: no new public interface — this task changes `_retrain()`'s internal behavior only. Later tasks don't depend on anything new here.

- [ ] **Step 1: Write the failing test**

Add this new class to `tests/test_energy_forecast.py`, immediately after the `class TestFifteenMinCache:` block ends (i.e., right before `class TestRetrainEvCachePathBug:`, currently around line 3475). It reuses the existing `_FakeRetrain` and `_make_energy_df` helpers already defined earlier in this file (around lines 3355-3421):

```python
class TestRetrainExcludedRanges:
    """Excluded date ranges (excluded_ranges.csv) must be filtered out of
    energy_df before it reaches _ml_model.train() — spec §4.1."""

    def _patch_retrain_deps(self, monkeypatch, energy_df):
        import energy_forecast.ha_data as ha_data_mod
        import energy_forecast.weather as weather_mod

        empty_df = pd.DataFrame()

        monkeypatch.setattr(ha_data_mod, "fetch_energy_history", lambda *a, **kw: energy_df)
        monkeypatch.setattr(ha_data_mod, "split_ev_charging", lambda df, *a, **kw: (df, empty_df))
        monkeypatch.setattr(weather_mod, "fetch_historical_weather", lambda *a, **kw: _empty_weather())
        monkeypatch.setattr(weather_mod, "fetch_open_meteo", lambda *a, **kw: _empty_weather())
        monkeypatch.setattr(ha_data_mod, "fetch_boolean_entity_history", lambda *a, **kw: empty_df)
        monkeypatch.setattr(ha_data_mod, "fetch_presence_history", lambda *a, **kw: empty_df)
        monkeypatch.setattr(ha_data_mod, "fetch_energy_history_15m", lambda *a, **kw: None)
        return ha_data_mod

    def test_excluded_window_absent_from_training_frame(self, tmp_path, monkeypatch):
        from energy_forecast.energy_forecast import EnergyForecast

        energy_df = _make_energy_df(200)  # 2024-01-01 00:00, hourly, 200 rows
        self._patch_retrain_deps(monkeypatch, energy_df)

        cache_path = tmp_path / "energy_history.csv"
        (tmp_path / "excluded_ranges.csv").write_text(
            "start,end,reason\n2024-01-05 00:00,2024-01-05 12:00,test fault\n"
        )

        stub = _FakeRetrain(cache_path)
        EnergyForecast._retrain(stub)

        trained_df = stub._ml_model.train.call_args.args[0]
        excluded = (trained_df["timestamp"] >= pd.Timestamp("2024-01-05 00:00")) & (
            trained_df["timestamp"] <= pd.Timestamp("2024-01-05 12:00")
        )
        assert not excluded.any(), "excluded window must not appear in the frame passed to train()"
        assert len(trained_df) == 200 - 13  # 00:00..12:00 inclusive = 13 hourly rows

    def test_stale_anchor_after_filtering_logs_warning(self, tmp_path, monkeypatch, caplog):
        """energy_df fixtures use fixed 2024 dates, always far behind real now() —
        exploiting that to assert the recency-anchor staleness warning (spec §4.3)
        fires without needing to mock pd.Timestamp.now()."""
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        energy_df = _make_energy_df(200)
        self._patch_retrain_deps(monkeypatch, energy_df)
        cache_path = tmp_path / "energy_history.csv"  # no excluded_ranges.csv needed

        stub = _FakeRetrain(cache_path)
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._retrain(stub)

        assert any("behind now" in r.message.lower() for r in caplog.records)

    def test_no_excluded_ranges_file_trains_on_full_history(self, tmp_path, monkeypatch):
        """Absence of excluded_ranges.csv must be a no-op (the common case)."""
        from energy_forecast.energy_forecast import EnergyForecast

        energy_df = _make_energy_df(200)
        self._patch_retrain_deps(monkeypatch, energy_df)
        cache_path = tmp_path / "energy_history.csv"  # no excluded_ranges.csv written

        stub = _FakeRetrain(cache_path)
        EnergyForecast._retrain(stub)

        trained_df = stub._ml_model.train.call_args.args[0]
        assert len(trained_df) == 200

    def test_exclusion_pushing_below_min_history_hours_skips_retrain(self, tmp_path, monkeypatch, caplog):
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        # 60 rows total; MIN_HISTORY_HOURS is 48. Exclude 20 of them so the
        # post-filter count (40) falls below the threshold that the
        # pre-filter count (60) passed.
        energy_df = _make_energy_df(60)
        self._patch_retrain_deps(monkeypatch, energy_df)
        cache_path = tmp_path / "energy_history.csv"
        (tmp_path / "excluded_ranges.csv").write_text(
            "start,end,reason\n2024-01-01 00:00,2024-01-01 19:00,big fault\n"  # 20 hourly rows
        )

        stub = _FakeRetrain(cache_path)
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._retrain(stub)

        stub._ml_model.train.assert_not_called()
        assert any("exclu" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py::TestRetrainExcludedRanges -v`
Expected: FAIL — `test_excluded_window_absent_from_training_frame`, `test_exclusion_pushing_below_min_history_hours_skips_retrain`, and `test_stale_anchor_after_filtering_logs_warning` all fail because neither the filtering nor the anchor-gap logging exists yet (the first two assert rows are missing / `train` wasn't called when they actually are/were; the third asserts a warning that's never logged)

- [ ] **Step 3: Implement the integration**

In `apps/energy_forecast/energy_forecast.py`, replace the current `_retrain()` opening (lines 1453-1469):

```python
    def _retrain(self) -> None:
        import pandas as pd

        _LOGGER.info("Starting model retraining…")
        energy_df = ha_data.fetch_energy_history(
            self,
            self._energy_sensor,
            cache_path=self._cache_path,
            timezone=self._timezone,
            unit_multiplier=self._unit_multiplier,
        )

        if len(energy_df) < MIN_HISTORY_HOURS:
            _LOGGER.warning("Insufficient history (%d h). Skipping.", len(energy_df))
            return

        energy_df = _strip_tz(energy_df, self._timezone)
```

with:

```python
    def _retrain(self) -> None:
        import pandas as pd

        _LOGGER.info("Starting model retraining…")
        energy_df = ha_data.fetch_energy_history(
            self,
            self._energy_sensor,
            cache_path=self._cache_path,
            timezone=self._timezone,
            unit_multiplier=self._unit_multiplier,
        )

        if len(energy_df) < MIN_HISTORY_HOURS:
            _LOGGER.warning("Insufficient history (%d h). Skipping.", len(energy_df))
            return

        energy_df = _strip_tz(energy_df, self._timezone)

        # ── Hand-configured known-bad date ranges (hardware faults, etc.) ────
        # Filtered as early as possible: a correction/EV-detection computed
        # from a corrupted main reading is equally meaningless.
        excluded_ranges = ha_data.load_excluded_ranges(
            self._cache_path.parent / "excluded_ranges.csv", self._timezone, _LOGGER
        )
        energy_df = ha_data.filter_excluded_ranges(energy_df, excluded_ranges, _LOGGER)

        if len(energy_df) < MIN_HISTORY_HOURS:
            _LOGGER.warning(
                "Insufficient history after excluded-range filtering (%d h) — active "
                "exclusions in excluded_ranges.csv reduced the training set below the "
                "%d h minimum. Skipping.",
                len(energy_df),
                MIN_HISTORY_HOURS,
            )
            return

        now_ts = pd.Timestamp.now(tz=self._timezone).tz_localize(None)
        gap_hours = (now_ts - energy_df["timestamp"].max()).total_seconds() / 3600
        if gap_hours > 24:
            _LOGGER.warning(
                "Most recent training data is %.1fh behind now() — an active excluded "
                "range may be freezing the recency-weighting anchor at the fault's onset.",
                gap_hours,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py::TestRetrainExcludedRanges -v`
Expected: `4 passed`

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast.py
git commit -m "feat: filter excluded date ranges out of _retrain()'s training data"
```

---

### Task 4: Wire exclusion filtering into `_update_sensors()` (live prediction path)

Implements spec §4.2 — without this, live predictions during an active fault would keep computing lag/rolling features from the same known-bad readings training is taught to exclude.

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (`_update_sensors()`, currently starting at line 1712)
- Test: `tests/test_energy_forecast.py` (new test class)

**Interfaces:**
- Consumes: `ha_data.load_excluded_ranges` / `ha_data.filter_excluded_ranges` from Tasks 1 and 2 (same functions Task 3 uses).
- Produces: nothing new for later tasks.

**Scope note:** `_update_sensors()` spans ~360 lines (energy_forecast.py:1712-2073) with dozens of external dependencies (weather fetch, several HA history fetches, physics sensors, publishing, anomaly detection, pred-history persistence) — confirmed by grep that no existing test in this suite drives the real method end-to-end; the file's own established pattern for testing a small piece of logic deep inside `_retrain()`/`_update_sensors()` is to replicate that exact logic inline rather than build a full fake for the whole method (see `test_actuals_for_retrain_excludes_ev_adjacent_hours`, `tests/test_energy_forecast.py:1856`, which does exactly this for the EV-adjacency filter). This task follows that same precedent: the test below characterizes the exact snippet Step 3 inserts, rather than exercising all of `_update_sensors()`. It cannot mechanically prove the snippet landed in the right place in the method body — Step 3's code diff and a manual read are what confirm that.

- [ ] **Step 1: Write the failing test**

Add this class anywhere after the `TestRetrainExcludedRanges` class from Task 3. No module-level test logger exists yet in this file, so define one locally alongside the class:

```python
import logging as _logging_for_excluded_ranges_tests  # noqa: E402

_EXCLUDED_RANGES_TEST_LOGGER = _logging_for_excluded_ranges_tests.getLogger("energy_forecast")


class TestUpdateSensorsExcludedRanges:
    """Characterizes the load+filter composition Task 4 inserts into
    _update_sensors() right before self._cached_recent_actuals = recent_actuals
    (spec §4.2). See the Scope note above for why this doesn't drive the full
    method."""

    def test_excluded_window_removed_from_recent_actuals(self, tmp_path):
        from energy_forecast import ha_data

        cache_path = tmp_path / "energy_history.csv"
        (tmp_path / "excluded_ranges.csv").write_text(
            "start,end,reason\n2024-01-01 05:00,2024-01-01 08:00,test fault\n"
        )
        recent_actuals = _make_energy_df(24)  # 2024-01-01 00:00..23:00, hourly

        excluded_ranges = ha_data.load_excluded_ranges(
            cache_path.parent / "excluded_ranges.csv", "Europe/Zurich", _EXCLUDED_RANGES_TEST_LOGGER
        )
        result = ha_data.filter_excluded_ranges(recent_actuals, excluded_ranges, _EXCLUDED_RANGES_TEST_LOGGER)

        in_range = (result["timestamp"] >= pd.Timestamp("2024-01-01 05:00")) & (
            result["timestamp"] <= pd.Timestamp("2024-01-01 08:00")
        )
        assert not in_range.any()
        assert len(result) == 24 - 4  # 05:00..08:00 inclusive = 4 hourly rows

    def test_no_excluded_ranges_file_is_noop(self, tmp_path):
        from energy_forecast import ha_data

        cache_path = tmp_path / "energy_history.csv"  # no excluded_ranges.csv written
        recent_actuals = _make_energy_df(24)

        excluded_ranges = ha_data.load_excluded_ranges(
            cache_path.parent / "excluded_ranges.csv", "Europe/Zurich", _EXCLUDED_RANGES_TEST_LOGGER
        )
        result = ha_data.filter_excluded_ranges(recent_actuals, excluded_ranges, _EXCLUDED_RANGES_TEST_LOGGER)

        assert len(result) == 24
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py::TestUpdateSensorsExcludedRanges -v`
Expected: both tests actually pass already at this point, since they only exercise Tasks 1-2's already-implemented functions — this step is a checkpoint confirming the composition is correct *before* copying it into `_update_sensors()` in Step 3, not a red-green cycle for new production code. Proceed to Step 3.

- [ ] **Step 3: Implement the integration**

In `apps/energy_forecast/energy_forecast.py`, find this block inside `_update_sensors()` (currently lines 1886-1889):

```python
        # Cache inputs for scenario/what-if API (Stage 4)
        self._cached_forecast_df = forecast_df
        self._cached_live_temp = live_temp
        self._cached_recent_actuals = recent_actuals
```

Replace with:

```python
        # ── Hand-configured known-bad date ranges (hardware faults, etc.) ────
        # Applied to recent_actuals (feeds lag/rolling features for live
        # prediction) so an active fault doesn't feed the model corrupted
        # readings it was trained to treat as NaN during the same window.
        # Does NOT extend to full_actuals (anomaly/MAE sensors) — spec §6.
        if recent_actuals is not None and not recent_actuals.empty:
            excluded_ranges = ha_data.load_excluded_ranges(
                self._cache_path.parent / "excluded_ranges.csv", self._timezone, _LOGGER
            )
            recent_actuals = ha_data.filter_excluded_ranges(recent_actuals, excluded_ranges, _LOGGER)

        # Cache inputs for scenario/what-if API (Stage 4)
        self._cached_forecast_df = forecast_df
        self._cached_live_temp = live_temp
        self._cached_recent_actuals = recent_actuals
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py::TestUpdateSensorsExcludedRanges -v`
Expected: `2 passed`

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast.py
git commit -m "feat: filter excluded date ranges out of live prediction lag features"
```

---

### Task 5: Fix physics holdout cutoff to use calendar-day span, not row count

Implements the fix committed to in spec §4.3 ("this is being fixed as part of this feature, not deferred"): `train()`'s physics-calibration holdout cutoff used `len(energy_df) / 24` as a proxy for calendar days, which silently breaks once a multi-day exclusion introduces a real gap between rows.

**Files:**
- Modify: `apps/energy_forecast/model.py` (inside `train()`, the `physics_model.calibrate(...)` call, currently lines 412-424)
- Test: `tests/test_model.py` (add to `class TestTrainWithPhysics`, which already has the `_physics_config()` helper and the `_make_trained_model` pattern this test extends)

**Interfaces:**
- Consumes: nothing from Tasks 1-4 — this is an independent correctness fix inside `model.py`'s existing `train()` signature, unrelated to the CSV-loading path.
- Produces: nothing new for later tasks (this is the last code task before docs).

- [ ] **Step 1: Write the failing test**

Add to `class TestTrainWithPhysics` in `tests/test_model.py` (anywhere after `_physics_config`, e.g. right after `test_calibrate_skipped_when_fresh`):

```python
    def test_holdout_cutoff_uses_calendar_span_not_row_count(self, tmp_path):
        """A multi-day gap in energy_df (e.g. from an excluded date range) must not
        shrink the holdout window via the old len(df)/24 row-count proxy — the
        cutoff must reflect the actual (max_ts - min_ts) calendar span."""
        from energy_forecast.physics import ThermalPhysicsModel

        # 100-day calendar span, but only the first 20 and last 20 days have rows
        # (the middle 60 days are missing — simulates a large excluded range).
        # Old proxy: len(df)/24*0.1 = (40*24)/24*0.1 = 4 days.
        # Fixed calc: (max_ts - min_ts).days*0.1 = 100*0.1 = 10 days.
        full_range = pd.date_range("2024-01-01", periods=100 * 24, freq="1h")
        first_chunk = full_range[: 20 * 24]
        last_chunk = full_range[-20 * 24 :]
        ts = first_chunk.append(last_chunk)
        rng = np.random.default_rng(0)
        energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=len(ts))})
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=len(ts)),
                "precipitation_mm": [0.0] * len(ts),
                "sunshine_min": [30.0] * len(ts),
                "wind_kmh": [10.0] * len(ts),
                "cloud_cover_pct": [50.0] * len(ts),
                "direct_radiation_wm2": [100.0] * len(ts),
            }
        )

        pm = ThermalPhysicsModel(tmp_path / "physics_models", self._physics_config())
        m = EnergyForecastModel(tmp_path / "model", timezone="Europe/Zurich")
        with patch.object(pm, "calibrate", wraps=pm.calibrate) as mock_calibrate:
            m.train(energy_df, weather_df, outdoor_df=None, weight_halflife_days=0, physics_model=pm)
            actual_cutoff = mock_calibrate.call_args.kwargs["holdout_cutoff"]

        expected_cutoff = ts.max() - pd.Timedelta(days=int((ts.max() - ts.min()).days * 0.1) or 1)
        assert actual_cutoff == expected_cutoff
```

Also add this test to a new class in the same file (`tests/test_model.py`), covering the spec §5 integration case where an exclusion shrinks the row count enough to flip `lag_168h` out of `active_lags` between two retrains — confirming graceful degradation, not a crash:

```python
class TestActiveLagsBoundaryAcrossRetrains:
    """A retrain after a new exclusion is added can shift row count across the
    active_lags n_rows - lag >= 100 threshold (model.py:365). This must degrade
    gracefully (the lag feature is simply dropped), never crash — spec §5."""

    def _energy_df(self, n):
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        rng = np.random.default_rng(0)
        return pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})

    def _weather_df(self, ts):
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=len(ts)),
                "precipitation_mm": [0.0] * len(ts),
                "sunshine_min": [30.0] * len(ts),
                "wind_kmh": [10.0] * len(ts),
                "cloud_cover_pct": [50.0] * len(ts),
                "direct_radiation_wm2": [100.0] * len(ts),
            }
        )

    def test_lag_168h_drops_out_without_crash_when_rows_fall_below_threshold(self, tmp_path):
        # lag_168h needs n_rows - 168 >= 100, i.e. n_rows >= 268.
        m = EnergyForecastModel(tmp_path / "model", timezone="Europe/Zurich")

        energy_wide = self._energy_df(280)  # 280 - 168 = 112 >= 100: lag_168h active
        m.train(energy_wide, self._weather_df(energy_wide["timestamp"]), outdoor_df=None, weight_halflife_days=0)
        assert "lag_168h" in m.feature_cols

        # Simulate a retrain after an exclusion range shrank the training set.
        energy_narrow = self._energy_df(260)  # 260 - 168 = 92 < 100: lag_168h now inactive
        m.train(energy_narrow, self._weather_df(energy_narrow["timestamp"]), outdoor_df=None, weight_halflife_days=0)
        assert "lag_168h" not in m.feature_cols
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_model.py::TestTrainWithPhysics::test_holdout_cutoff_uses_calendar_span_not_row_count -v`
Expected: FAIL — `actual_cutoff` reflects the old `len(energy_df)/24*0.1` = 4-day-back calculation, not the expected 10-day-back one

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_model.py::TestActiveLagsBoundaryAcrossRetrains -v`
Expected: this test exercises pre-existing dynamic-lag-selection behavior (unrelated to this task's code change), so it should already pass — its purpose is to lock in that this pre-existing robustness still holds once exclusions can shrink row counts between retrains. Confirm it passes.

- [ ] **Step 3: Implement the fix**

In `apps/energy_forecast/model.py`, replace (currently lines 412-424):

```python
        if physics_model is not None:
            physics_model.check_zone_boundary(list(climate_dfs.keys()) if climate_dfs else [])
            if physics_model.calibration_stale:
                physics_model.calibrate(
                    energy_df,
                    weather_df,
                    climate_dfs,
                    dhw_df,
                    holdout_cutoff=energy_df["timestamp"].max()
                    - pd.Timedelta(days=int(len(energy_df) / 24 * 0.1) or 1),
                    heating_active_df=heating_active_df,
                    away_df=away_df,
                )
```

with:

```python
        if physics_model is not None:
            physics_model.check_zone_boundary(list(climate_dfs.keys()) if climate_dfs else [])
            if physics_model.calibration_stale:
                # Calendar-day span, not row count: a multi-day excluded date range
                # (or any other real gap) leaves len(energy_df)/24 undercounting the
                # true span, silently shrinking/misplacing the holdout window.
                _span_days = (energy_df["timestamp"].max() - energy_df["timestamp"].min()).days
                physics_model.calibrate(
                    energy_df,
                    weather_df,
                    climate_dfs,
                    dhw_df,
                    holdout_cutoff=energy_df["timestamp"].max() - pd.Timedelta(days=int(_span_days * 0.1) or 1),
                    heating_active_df=heating_active_df,
                    away_df=away_df,
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_model.py::TestTrainWithPhysics tests/test_model.py::TestActiveLagsBoundaryAcrossRetrains -v`
Expected: all tests pass, including both new ones

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "fix: physics holdout cutoff uses calendar-day span, not row count"
```

---

### Task 6: Update CHANGELOG.md and memory

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `memory/project_status.md` (or add a new `memory/project_excluded_ranges.md` if the feature needs its own entry — follow the existing index pattern in `MEMORY.md`)

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing (end of plan).

- [ ] **Step 1: Delegate the changelog entry**

Per this project's workflow convention, invoke the `changelog-writer` agent (do not hand-write the entry) to add a CHANGELOG.md entry summarizing: new `excluded_ranges.csv` mechanism for hand-configured known-bad training/prediction date ranges, filtered in both `_retrain()` and `_update_sensors()`; physics holdout-cutoff fix (calendar-day span instead of row count).

- [ ] **Step 2: Add/update a project memory entry**

Write `memory/project_excluded_ranges.md` (type: `project`) documenting: the feature exists to handle the ongoing 2026-07-19 gPlug/SolarEdge hardware fault, where the file lives (`excluded_ranges.csv` in the HA data dir, hand-edited via Samba), the format, and the operational step to apply an exclusion immediately (fire the `RELOAD_ENERGY_MODEL` HA event). Add a one-line pointer to it in `MEMORY.md`'s index table.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md memory/project_excluded_ranges.md
git commit -m "docs: document excluded-training-ranges feature in changelog and memory"
```
