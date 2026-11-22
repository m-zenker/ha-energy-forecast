# Excluded Training Date Ranges — Design Spec

**Date:** 2026-07-19 (rev. 3 — post multi-stakeholder review, round 2)
**Status:** Proposed — pending user review before implementation planning
**Branch base:** `dev`

**Revision note:** rev. 1 was reviewed by three independent domain experts (Data Scientist,
Software Engineer, Domain/Energy-Systems Engineer) in parallel. That review surfaced 25 issues
(10 High) — most significantly: excluded ranges never reached the live prediction path (only
training), `MIN_HISTORY_HOURS` was checked before filtering instead of after, DST fall-back
duplicate rows were unaddressed, neither new function had a specified exception-handling
contract, overlapping ranges produced misleading log counts, and the spec implied the feature
alone "solves" the incident class despite this project having no fault-detection/alerting
infrastructure. Rev. 2 resolved all High/Medium findings and folded the accepted Low-severity
gaps into an explicit Known Limitations section. The same three reviewers re-checked rev. 2 and
found 27/29 original issues resolved (the remaining two — a hard sanity cap on range size, and
CSV backup/version history — deliberately left as documented limitations rather than fixed) plus
4 new issues from the round-2 pass: `load_excluded_ranges`'s signature was missing the
`timezone` parameter its own spring-forward check depended on, the file-level-vs-row-level
`KeyError` distinction lacked an implementation mechanism (would have warned once per row instead
of once per file), two new model.py behavior changes (holdout-cutoff fix, anchor-staleness
logging) had no corresponding tests, and the live `recent_actuals` path's 2-day fetch window
wasn't checked against exclusion-range size (resolved by pointing at the existing empty-input
fallback in `_add_lag_and_rolling_prediction`, no new code needed). This revision (rev. 3) fixes
all four, plus the two lingering cosmetic items (typed `logger` parameter, docstring convention
callout) and a code-location citation slip. See §3-§5 for the specific changes.

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

**Scope note (added post-review):** this feature addresses *training-time* data quality and, as
of rev. 2, the *live prediction-time* lag-feature path (§4). It explicitly does **not** provide
fault detection/alerting, does not retroactively fix an already-poisoned deployed model or
physics calibration without a follow-up operator action, and does not suppress bogus values on
live anomaly/MAE sensors during an active fault. See §6 for the complete list of accepted gaps.

## 2. Storage Format

A new file, `excluded_ranges.csv`, lives in the same directory as `energy_history.csv`
(`self._cache_path.parent`) — the data directory on the live HA instance, synced via Samba like
the other cache CSVs, and already pulled locally by `scripts/pull_ha_data.py`. This location was
chosen (over a checked-in repo file or an `apps.yaml` config block) specifically so a newly
noticed hardware fault can be excluded immediately by editing the file directly on HA, with no
deploy or AppDaemon restart required.

Columns:

| Column | Required | Notes |
|---|---|---|
| `start` | yes | Naive local timestamp, `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` only |
| `end` | yes | Same format; inclusive; see end-of-day rule below |
| `reason` | no | Free text, defaults to `""` if the column is absent |

Any additional/unrecognized columns are ignored (not treated as malformed).

Example:

```csv
start,end,reason
2026-07-19 14:00,2026-07-21 09:30,gplug + solaredge hardware fault
2026-07-25,2026-07-26,gplug + solaredge hardware fault (day 2)
```

### Format requirements (added post-review — findings #16, #17)

`start`/`end` must match `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` exactly (validated with a regex
before parsing) — not passed through pandas' lenient/mixed date parser. This removes the
day/month ambiguity risk a hand-typed date otherwise carries (e.g. `07/19/2026` vs `19/07/2026`).
A row that doesn't match either pattern is treated as malformed: logged at `WARNING`, skipped.

A timestamp containing a timezone offset or `Z` suffix (e.g. `2026-07-19 14:00+02:00`) is
explicitly rejected as malformed (logged, skipped) rather than silently normalized — this file's
convention is naive local time only, matching `energy_df` post-`_strip_tz()`, and silently
accepting an offset risks a human reasoning in the wrong reference frame.

### End-of-day rule (added post-review — finding #22)

If `end` is given as a bare date (no time component in the raw string, distinguished at the
string level before parsing — `2026-07-21`, not `2026-07-21 00:00`), it is expanded to
`2026-07-21 23:59:59`, i.e. "through the end of that day." This matches the overwhelmingly likely
human intent ("discard the 19th through the 21st") and avoids a silent off-by-23-hours bug where
an explicit-looking date actually only covered the first minute of the last day. An explicit
`end` with a time component (including `00:00`) is used exactly as written — this is how an
operator excludes down to the exact minute.

### DST handling (added post-review — findings #3, #20)

- **Fall-back (e.g. late-October, Europe/Zurich 03:00 CEST → 02:00 CET):** after `_strip_tz()`,
  the ambiguous hour appears as two rows sharing one naive timestamp (`_check_dst_duplicates`).
  A range boundary landing in this hour cannot select only one occurrence — **both rows are
  always dropped together, or neither.** This is a documented limitation, not a bug: splitting
  them would require a schema change (UTC or explicit-offset timestamps) that isn't justified by
  the current use case. If a future fault's boundary lands exactly here, both real hours are
  lost; this is an acceptable, rare cost.
- **Spring-forward (e.g. late-March, 02:00–02:59 never occurs locally):** a human can type a
  nonexistent local time into `start`/`end` (this parses without error). The resulting range
  simply won't match any real row, indistinguishable in the row-count log from a legitimately
  stale range. `load_excluded_ranges` additionally checks each parsed timestamp against the
  known spring-forward gap for the configured timezone (passed in as a parameter — see signature
  below, fixed post-re-review) and logs a distinct `WARNING`
  ("timestamp falls in a nonexistent local time — check for a transcription error") when it does,
  so this failure mode isn't silently conflated with "range no longer needed."

## 3. Loading & Filtering

Two new functions in `ha_data.py`, placed next to `validate_energy_cache()` (the existing
data-quality section of that module), following its docstring convention (one-line summary, an
explicit "never raises" sentence, enumerated malformed-input behaviors, `Args:`/`Returns:`
sections) and its typed-parameter style (`logger: logging.Logger`, fixed post-re-review — rev. 2
draft omitted the type hint).

### `load_excluded_ranges(path: Path, timezone: str, logger: logging.Logger) -> list[tuple[Timestamp, Timestamp, str]]`

- Takes `timezone` explicitly (fixed post-re-review — rev. 2 draft referenced "the configured
  timezone" in the spring-forward check without it being a parameter) so the spring-forward-gap
  check has a timezone to check against; callers pass `self._timezone`, same as every other
  `ha_data.py` function that needs it.
- Missing file → return `[]`, no warning (the common/default case).
- File-level failures (`OSError`, `pd.errors.ParserError`, `pd.errors.EmptyDataError` — covers a
  zero-byte file, a torn Samba write, or a fully corrupt file) are caught around the `read_csv`
  call; logs one `WARNING`, returns `[]`. This is a distinct branch from "missing file" (tested
  separately, per §5) but produces the same externally-visible result.
- **Column presence is checked once, upfront, before any per-row loop** (fixed post-re-review —
  a per-row-only check would raise the same `KeyError` N times, once per row, instead of the one
  clean file-level warning the design intends): if `start` or `end` is missing from the parsed
  header, log one `WARNING` and return `[]` immediately. `reason`'s absence is not an error — see
  below.
- A header-only file (valid CSV, zero data rows) is not an error — returns `[]` silently, same
  as "missing file," since there's nothing malformed about it.
- With required columns confirmed present, per-row validation is wrapped in its own
  `try/except (ValueError, TypeError, AttributeError)` (no `KeyError` here — that failure mode is
  fully handled by the upfront column check above, so it can't recur per-row) — a single bad row
  (unparseable date per the format rule above, tz-offset present, or `end < start` after the
  end-of-day expansion) is logged at `WARNING` and skipped, and the rest of the file still loads.
- `reason` defaults to `""` if the column is absent.

### `filter_excluded_ranges(df: pd.DataFrame, ranges: list[tuple], logger: logging.Logger) -> pd.DataFrame`

- Guard clause: `if df.empty or "timestamp" not in df.columns or not ranges: return df` — mirrors
  `validate_energy_cache`'s pattern.
- Wrapped in `try/except (KeyError, ValueError, TypeError, AttributeError)`: on any unexpected
  failure, logs `.error(...)` and **returns `df` unfiltered** rather than propagating — a bug in
  this new code must degrade to "no filtering happened," never to "retrain didn't happen" (added
  post-review — finding #4; this was previously unspecified and the only exception safety net
  upstream is `_retrain_cb`'s broad `except Exception`, which would have silently skipped the
  *entire* retrain, not just the filter).
- **Row-count accounting (fixed post-review — finding #6):** each range's dropped-row count is
  computed against the *original* `df`, independently of other ranges — not sequentially against
  a shrinking frame. This keeps a `0`-count log line meaningful ("this range doesn't overlap
  anything, might be stale") even when a later-listed range overlaps an earlier one. After all
  per-range lines, one additional line logs the total unique rows actually dropped (via the union
  of all range masks), so overlapping ranges don't inflate the perceived total loss.
- **Escalation threshold (added post-review — finding on typo protection):** if a single range's
  drop count exceeds 10% of `len(df)` or the range spans more than 14 days, the per-range log
  line is emitted at `WARNING` instead of `INFO` — this is the only realistic tripwire for a
  fat-fingered year/typo in a hand-edited file, given the project has no other alerting.

## 4. Integration Points

### 4.1 Training path (primary)

`_retrain()` in `energy_forecast.py`: `filter_excluded_ranges` is called immediately after
`energy_df = _strip_tz(energy_df, self._timezone)` (currently line 1469) — before target
correction and EV-threshold splitting, since a correction or EV-detection computed from a
corrupted main reading is equally meaningless.

**`MIN_HISTORY_HOURS` re-check (fixed post-review — findings #2):** the existing check at
energy_forecast.py:1465 runs *before* `_strip_tz()`/the new filter call, so it validates the
pre-filter row count. A second check is added immediately after `filter_excluded_ranges` runs: if
the post-filter row count falls below `MIN_HISTORY_HOURS`, log a `WARNING` that explicitly
attributes the shortfall to active exclusions (not just "insufficient history," which would be
misleading to whoever edited the CSV) and skip the retrain, same as the existing pre-filter case.

### 4.2 Live prediction path (added post-review — finding #1, was entirely missing from rev. 1)

`_add_lag_and_rolling_prediction()` (model.py:2187) is fed by `recent_actuals`, populated in
`_update_sensors()` via a *separate* call to `ha_data.fetch_recent_energy(...)`
(energy_forecast.py:1732) — not by `_retrain()`'s `energy_df`. Without a fix, every prediction
made while a fault is active computes `lag_24h`/`lag_168h`/rolling stats from the same known-bad
readings the training set is being taught to never see — the opposite of the intended isolation,
and a real train/predict mismatch (the model learns these hours are `NaN→median`; live inference
would otherwise feed it real corrupted values instead).

**Fix:** `filter_excluded_ranges` (the same function, no new code) is also applied to
`recent_actuals` after it passes through EV-splitting/`_subtract_sub_sensors` and is cached
(energy_forecast.py:1889 — added post-final-review, as the precise insertion point; the exact
line is order-independent here since row-masking by timestamp commutes with the upstream
subtraction steps, but an implementer needs a concrete anchor), and before it reaches
`_add_lag_and_rolling_prediction()`. Excluded hours fall back to `NaN`, filled by the same
stored feature medians used elsewhere — consistent with the training-side behavior instead of
diverging from it.

This does **not** extend to the anomaly-detection/live-MAE sensors computed earlier in the same
`_update_sensors()` pass from the same raw fetch — see §6. Verified structurally: `full_actuals`
(feeds `_actuals_history`/anomaly/MAE) and `recent_actuals` (feeds lag features, the only one
filtered here) are genuinely distinct objects in `_update_sensors()`, so this boundary is
implementable exactly as described, not just true in prose.

**Small-window edge case (added post-re-review):** `recent_actuals` is fetched as a 2-day window
(energy_forecast.py:1728-1729). An exclusion range comparable to or larger than 2 days can empty
it entirely for a given prediction cycle. This is not a new failure mode:
`_add_lag_and_rolling_prediction()` already has an explicit empty/`None` branch (model.py:2200)
that falls back to `NaN` for every lag/rolling column, filled by stored training medians exactly
as the "no history yet" cold-start case is handled today. No additional code is needed — noted
here so an implementer doesn't mistake this for an unhandled gap.

### 4.3 Downstream effects requiring operator awareness (added post-review)

These are documented here because they affect what "the exclusion took effect" actually means in
practice — none of them are new code in this iteration, but the spec previously implied adding a
range was a complete fix, which isn't accurate for two subsystems:

- **Physics calibration is not automatically refreshed.** `self._physics_model.calibrate(...)` is
  only invoked from the manual `energy_forecast/recalibrate_physics` HA service — never from
  `_retrain()`. If an excluded window overlaps the period `physics_calibration.json` was last
  calibrated against, adding the exclusion does not fix that calibration; the operator must
  separately call `recalibrate_physics` if the fault materially affected it (finding #10).
- **Physics holdout sizing assumes no large gaps.** `train()`'s holdout cutoff
  (`len(energy_df) / 24 * 0.1` days, ~model.py:420) treats row count as a proxy for calendar
  span. Pre-existing gaps (EV-adjacent hours, an hour or two of outage) are small enough for this
  to not matter; a multi-day exclusion is not. **This is being fixed as part of this feature**
  (not deferred) — the holdout cutoff will be computed from `(max_ts - min_ts).days` instead of
  row count (finding #11).
- **Regime clustering's day-completeness filter is stricter than the hourly model.**
  `DailyProfileClusterer.fit()` requires ≥18 hourly rows to keep a day at all (clustering.py:76-77)
  — unlike EV days (excluded from centroid *fitting* but still labeled via `km.predict`), a day
  more than ~6 hours chopped by an exclusion gets no regime label at all, and
  `mapped_labels.ffill()` (model.py:519, not clustering.py — corrected post-re-review) carries the
  previous day's label forward. The spec's original claim that
  datetime-granularity exclusion "doesn't force discarding the whole day" holds for the GBM's
  lag/rolling features but not for clustering — documented here rather than silently
  inconsistent (finding #13).
- **Gap blast radius extends past the window edges.** `lag_168h`/`lag_336h` and `rolling_mean_7d`
  (min_periods=48/168) mean rows up to 336 hours (14 days) *after* an excluded window can have
  individual features silently `NaN→median`-filled, not just rows strictly inside the window.
  Sizing a multi-day exclusion should account for this wider (though much less severe —
  individual features, not the whole row) degraded radius (finding #14).
- **Recency weighting can freeze during an active, ongoing exclusion.** `weight_halflife_days`
  anchors to `energy_df["timestamp"].max()` — computed post-filter. If the fault is still
  ongoing at retrain time, this anchor becomes the last good pre-fault hour, not "now," and stays
  there across every retrain until the fault ends. `_retrain()` will log the gap between this
  anchor and the actual current time when it exceeds 24h, so a long-running fault is visible
  rather than silently degrading recency weighting retrain after retrain (finding #15).
- **An already-poisoned deployed model isn't retroactively fixed by adding an exclusion** — the
  currently-serving model/`meta.pkl` keeps making live predictions, trained on the bad data,
  until the next successful filtered retrain actually completes (up to 7 days away — see §4.4).
  `rollback_model()` (already available, model.py:1548 — corrected post-final-review) is the recommended immediate stopgap if a
  fault is discovered to have measurably degraded live predictions (finding #12).

### 4.4 Applying an exclusion immediately (added post-review — finding #8)

Editing `excluded_ranges.csv` alone does not take effect until the next retrain, and the only
unconditional automatic retrain is weekly (`RETRAIN_INTERVAL_S`, 168h). The MAE-triggered
adaptive retrain path (`_maybe_adaptive_retrain`) is not a reliable fast path here — this
feature's whole premise is that a hardware fault of this kind produces *plausible-looking* data,
which is exactly the kind of degradation that may not cleanly blow out live MAE past its
threshold. **For an active fault, the operational step is to fire the existing
`RELOAD_ENERGY_MODEL` HA event** (already wired: `self.listen_event(self._retrain_cb,
"RELOAD_ENERGY_MODEL")`, energy_forecast.py:397) immediately after editing the CSV, to force a
retrain under the new exclusion right away rather than waiting up to a week.

## 5. Testing

`tests/test_ha_data.py`:

- `load_excluded_ranges`:
  - missing file → `[]`, no log
  - header-only file (valid, zero rows) → `[]`, no warning (distinct code path from missing file)
  - well-formed multi-row file → correct tuples, `reason` populated
  - `reason` column absent → tuples with `reason=""`
  - extra unrecognized column present → ignored, doesn't affect parsing
  - one malformed row among valid ones (bad format, `end < start`) → malformed skipped, others
    still loaded, `WARNING` logged
  - fully malformed file (missing `start`/`end` columns) → `[]`, `WARNING` logged, no exception
  - ambiguous/non-ISO date format (e.g. `07/19/2026`) → rejected as malformed, skipped
  - timezone-aware timestamp in a row (`+02:00` suffix) → rejected as malformed, skipped, no crash
  - bare-date `end` → expands to `23:59:59` of that date
  - explicit-time `end` (including `00:00`) → used exactly as given, not expanded
  - range boundary landing on a DST fall-back ambiguous hour → both duplicate-timestamp rows
    dropped together (asserted, not just "doesn't crash")
  - range boundary on a spring-forward nonexistent local time → distinct `WARNING` logged,
    doesn't crash
  - truncated/corrupt CSV content → `[]`, `WARNING`, no exception (Samba torn-write scenario)
- `filter_excluded_ranges`:
  - rows inside a range dropped; rows outside kept
  - multiple non-overlapping ranges both apply
  - two overlapping ranges: correct per-range counts (each vs. original `df`) and correct total
    unique-rows-dropped count
  - empty `ranges` list → no-op, `df` unchanged
  - range entirely before/after all cache data → no-op, `0` logged
  - range boundary exactly matching a single row's timestamp → that row dropped (inclusive
    correctness, off-by-one check)
  - a range dropping >10% of rows or spanning >14 days → logged at `WARNING`, not `INFO`
  - malformed input (e.g. non-timestamp dtype in `df["timestamp"]`) → caught, `df` returned
    unfiltered, `.error` logged, no exception

Integration-level (smallest practical slice, likely in `tests/test_energy_forecast.py` or
equivalent):

- `_retrain()`: a configured excluded window does not appear in the frame passed to `train()`,
  and post-filter row count below `MIN_HISTORY_HOURS` produces the exclusion-attributed warning
  and skips retraining.
- `_update_sensors()` / `_add_lag_and_rolling_prediction()`: `recent_actuals` for an excluded
  window is excluded from live lag-feature computation the same way it is in training.
- A retrain where filtering shifts row count across the `active_lags` `n_rows - lag ≈ 100`
  threshold: confirm graceful degradation (feature dropped/median-filled), not a crash, when a
  previously-saved `meta.pkl` expected a lag feature that's no longer active.
- **(Added post-re-review — §4.3's two model.py changes had no corresponding tests):** the
  physics holdout cutoff, with a multi-day exclusion present, is computed from
  `(max_ts - min_ts).days` and produces a holdout window of the expected calendar length — not
  silently shrunk by the row-count proxy the old logic used.
- **(Added post-re-review)**: when an active exclusion touches the tail of history (the ongoing-
  fault case), `_retrain()` logs the gap between the frozen `weight_halflife_days` anchor and the
  actual current time once it exceeds 24h.

`tests/test_ha_data.py` additionally covers `load_excluded_ranges`'s new `timezone` parameter
(fixed post-re-review — the spring-forward-gap check needs it, and the rev. 2 draft's signature
omitted it): passing a different configured timezone changes which local times are flagged as
falling in that timezone's spring-forward gap.

## 6. Known Limitations & Accepted Gaps (expanded post-review)

- **No fault detection or alerting.** This project has no watchdog/killswitch infrastructure
  (per project `CLAUDE.md`); both prior incidents motivating this feature were found by a human
  noticing anomalous behavior during unrelated work, not by an automated alert. This feature is
  the *response* mechanism, not the *detection* mechanism — a fault an operator doesn't notice
  gets no benefit from this feature existing. Explicitly flagged as a standing risk, not solved
  here (finding #9).
- **Live anomaly/MAE sensors remain unfiltered.** The anomaly-detection and live-MAE computation
  inside `_update_sensors()` (energy_forecast.py:1732-1792) uses the same raw
  `fetch_recent_energy`/`_apply_target_correction` path as §4.2's fix, but this spec only filters
  the *lag-feature* consumption of `recent_actuals`, not these other consumers. During an active
  fault, anomaly/MAE sensors will keep surfacing values derived from the faulty readings. This is
  accepted for this iteration; fully suppressing those sensors during an active exclusion window
  is a reasonable follow-up but adds meaningful scope (finding #24).
- **Per-sensor exclusion remains out of scope.** The main-series-only drop is sufficient for the
  motivating incident (both gPlug and SolarEdge are simultaneously bad), but the mechanism as
  designed always discards the whole row. A future fault affecting only one of the two sensors
  (e.g. SolarEdge misbehaving, gPlug fine) would still force discarding an otherwise-valid
  raw-import label rather than just dropping the correction term for that window. Revisit if that
  scenario occurs (finding #19).
- **No backup or version history for the hand-edited CSV.** Unlike `apps.yaml` edits in this
  project (which get an explicit `.bak` file before patching) or code changes (git-reviewed),
  `excluded_ranges.csv` is a bare hand-edit over Samba. A fat-fingered deletion of a still-needed
  range has no recovery path beyond the operator's own care. Pulling this file into
  `scripts/pull_ha_data.py`'s existing sync would give a cheap local copy for diffing, but isn't
  required for this iteration (finding #21).
- **Samba mid-write torn reads are accepted as a non-issue.** A read racing a hand-edit save could
  see a truncated file. Given retrain cadence isn't sub-second and every realistic torn-read
  outcome degrades into the malformed-file or malformed-row handling already specified in §3,
  this is treated as covered by existing error handling rather than needing a separate mitigation
  (finding #23; a truncated-CSV test case is included in §5 to confirm this holds).
- **`energy_history_backfill.py` and analysis scripts remain untouched.** The one-off historical
  backfill tool writes into the cache CSV itself and is unaffected — exclusion is a training-time
  filter over the cache, not a cache-mutation step, so backfilled data can stay in
  `energy_history.csv` untouched and still gets filtered at train time. `analyze_forecast_bias.py`
  and similar scripts can be updated to use the same filter later if needed; not required now.
- **No CSV-management tooling.** Direct hand-editing via Samba is sufficient for the expected
  frequency of hardware faults; an add/remove-range script is not justified by that frequency.
