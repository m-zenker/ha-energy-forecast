# Excluded Training Date Ranges — Design Spec

**Date:** 2026-07-19
**Status:** Proposed — pending user review of this document before implementation planning
**Branch base:** `dev`

## 1. Problem & Motivation

Hardware faults on the household's raw grid-import meter (`sensor.gplugk_z_ei`, "gPlug") and the
SolarEdge inverter are currently ongoing (2026-07-19). Both feed directly into training:
`gplugk_z_ei` is `energy_sensor` (the target consumption series itself), and SolarEdge sensors
feed `_apply_target_correction()`'s solar/battery correction. A hardware fault can produce
readings that are **plausible-looking but wrong** — not wild spikes or exact zeros, which the
existing automated filters already catch, but degraded/stale/incorrect values within normal
range. This exact sensor pair has already caused two related incidents
(`memory/project_actuals_history_freeze.md`, `memory/project_zero_diff_cache_gap_fix.md`), both a
flat/stale-during-outage failure mode rather than an out-of-bounds one.

There is currently no pathway to mark "this time window's readings are known-bad, discard
regardless of value" — confirmed by auditing `ha_data.py`, `model.py`, and
`energy_forecast.py::_retrain()`. Existing filters are automated and narrow: out-of-range
clipping in `_raw_to_kwh_diff`, warning-only gap/monotonicity/range checks in
`validate_energy_cache()`, and EV-hour exclusion. None of them can express "these specific
calendar hours are untrustworthy regardless of what value they contain."

## 2. Storage Format

A new file, `excluded_ranges.csv`, lives in the same directory as `energy_history.csv`
(`self._cache_path.parent`) — the data directory on the live HA instance, synced via Samba like
the other cache CSVs, and already pulled locally by `scripts/pull_ha_data.py`. This location was
chosen (over a checked-in repo file or an `apps.yaml` config block) specifically so a newly
noticed hardware fault can be excluded immediately by editing the file directly on HA, with no
deploy or AppDaemon restart required — the next scheduled or triggered retrain picks it up.

Columns:

| Column | Type | Notes |
|---|---|---|
| `start` | naive local timestamp | Same tz convention as `energy_df` after `_strip_tz()` |
| `end` | naive local timestamp | Inclusive |
| `reason` | free text | For humans; not parsed |

Example:

```csv
start,end,reason
2026-07-19 14:00,2026-07-21 09:30,gplug + solaredge hardware fault
```

Ranges may be added, removed, or edited freely; the file is read fresh on every retrain. Datetime
(not just date) granularity is supported so a fault that starts or ends mid-day doesn't force
discarding the whole day.

## 3. Loading & Filtering

Two new functions in `ha_data.py`, placed next to `validate_energy_cache()` (the existing
data-quality section of that module):

- **`load_excluded_ranges(path: Path, logger) -> list[tuple[Timestamp, Timestamp, str]]`**
  Missing file → return `[]`, no warning (this is the common/default case — most retrains have no
  active exclusions). A malformed row (unparseable dates, or `end < start`) is logged as a
  `WARNING` and skipped; the rest of the file still loads. A completely malformed file (e.g.
  missing required columns) logs a `WARNING` and returns `[]` — mirrors `validate_energy_cache`'s
  "never raise" contract, since a bad exclusions file must not block retraining.

- **`filter_excluded_ranges(df: pd.DataFrame, ranges: list[tuple], logger) -> pd.DataFrame`**
  Drops rows from `df` whose `timestamp` falls within `[start, end]` for any range (inclusive).
  Logs one `INFO` line per range with the actual row count dropped — including a count of `0`,
  which signals the range no longer overlaps the cache (useful for noticing a stale entry that
  should be removed from the CSV). Returns `df` unchanged if `ranges` is empty.

## 4. Integration Point

Single call site: `_retrain()` in `energy_forecast.py`, immediately after
`energy_df = _strip_tz(energy_df, self._timezone)` (currently line 1469) — before target
correction and EV-threshold splitting. Rationale: if the main meter reading is corrupt, any
correction or EV-detection computed from it is equally meaningless, so filtering as early as
possible avoids doing wasted/misleading work on rows that are about to be dropped anyway.

Confirmed via `grep` that `fetch_energy_history()` (the function that produces `energy_df`) has
exactly one call site feeding the training pipeline — `_retrain()` — so no other code path needs
touching. `energy_history_backfill.py` (the one-off historical backfill tool) writes into the
cache CSV itself and is out of scope: exclusion is a training-time filter over the cache, not a
cache-mutation step, so backfilled data can stay in `energy_history.csv` untouched and still gets
filtered out at train time.

### Why dropping rows (not NaN-ing values in place) is safe

`_add_lag_and_rolling_training()` (model.py:2168-2172) already reindexes `energy_df` onto a dense
continuous hourly grid *before* computing `shift()`-based lag/rolling features, specifically so
that gaps (from EV-adjacent-hour drops, sensor outages, etc.) don't corrupt lag calculations for
surrounding rows — this is pre-existing, not new. `_add_sub_sensor_lags_training()` does the
equivalent for sub-sensor lags. Consequence for this feature:

- **Hours outside the excluded window**: unaffected — lags/rolling stats compute correctly, same
  gap-handling already exercised today.
- **Hours inside the excluded window**: lose their own training row entirely (correct — there's no
  trustworthy label to train against). Any lag feature on a later row that looks back into the gap
  becomes `NaN`, filled by stored feature medians — same fallback already used for short-history
  and other-gap cases.

Other sensors' data (weather, climate, sub-sensors, presence) for the excluded hours isn't
separately discarded — it's simply unused for those hours because there's no `energy_df` row left
to join it to. Data for all non-excluded hours remains fully usable.

## 5. Testing

- `tests/test_ha_data.py`:
  - `load_excluded_ranges`: missing file → `[]`; well-formed file → correct tuples; one malformed
    row among valid ones → malformed skipped, valid ones still loaded, warning logged; fully
    malformed file → `[]`, warning logged, no exception.
  - `filter_excluded_ranges`: rows inside a range are dropped; rows outside are kept; multiple
    non-overlapping ranges both apply; empty `ranges` list is a no-op; dropped-row count in the
    log matches actual rows removed.
- One integration-style test on `_retrain()` (or the smallest slice of it that's practical) confirming
  a configured excluded window does not appear in the frame ultimately passed to `train()`.

## 6. Out of Scope

- Per-sensor exclusion (nulling only the affected sensor while keeping others for that window) —
  rejected during design; the main-series-only drop is sufficient since gPlug/SolarEdge feed the
  target series and its correction directly.
- Touching `energy_history_backfill.py` or analysis scripts (`analyze_forecast_bias.py`, etc.) —
  can be added later if those scripts turn out to need the same filter; not needed for the
  immediate training-exclusion goal.
- Tooling to manage the CSV (add/remove-range script) — direct hand-editing via Samba is
  sufficient for the expected frequency of hardware faults.
