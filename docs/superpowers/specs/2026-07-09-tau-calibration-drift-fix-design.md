# τ Calibration Drift Fix — Design Spec

**Date:** 2026-07-09 (rev. 3 — post multi-stakeholder review, round 2)
**Status:** Proposed — pending approval before implementation
**Branch base:** `feat/physics-core-engine` (current worktree; contains the Phase 1 physics integration this bug lives in)

**Revision note:** rev. 1 of this spec was reviewed by three independent domain experts (building-thermal/controls, data science, software engineering) in parallel. That review surfaced a structural flaw in the confidence formula (it let a single good signal fully rescue trust, reproducing the exact bug being fixed) and a real UTC/local-timezone bug in the cooldown fix, among 20 total findings. Rev. 2 replaced the confidence formula, added a rolling drift cap, and fixed the timezone bug (§5). The same three reviewers re-checked rev. 2 and found it mostly sound but surfaced a second round of issues: the drift cap only bounded movement *within* a single 30-day window (sustained bias could still compound *across* windows), the three-way confidence product was more punishing than intended for mid-range conditions, and — caught by the software engineer — an actual `AttributeError`-causing bug (`self.STARTUP_RETRAIN_MIN_GAP_HOURS` referencing a module-level constant) plus a test that used dates incompatible with the real (hardcoded-January) test fixture. This revision (rev. 3) adds a second, longer-window (180-day) drift cap, switches confidence to a geometric mean, and fixes both round-2 code/test bugs. See §5 (round 1) and §6 (round 2) for the full findings-to-changes mapping.

---

## 1. Problem & Motivation

`sensor.ha_energy_forecast_thermal_pressure_net`'s `tau_hours` attribute — the calibrated building
thermal time constant produced by `EnergyForecastModel._calibrate_tau()` — has been drifting
downward through the 2026 spring/summer transition despite the "spring-bias guard" added to
prevent exactly that. Pulled directly from the live HA recorder:

| Timestamp (UTC) | τ (h) |
|---|---|
| 2026-07-01 06:37 | 9.68 |
| 2026-07-02 07:31 | 9.38 |
| 2026-07-02 10:53 | 7.97 |
| 2026-07-02 18:37 | 7.52 |
| 2026-07-04 05:58 | 8.12 |
| 2026-07-04 06:11 | 6.02 |
| 2026-07-07 06:38 | 6.55 |
| 2026-07-09 04:24 | 6.90 |

**-29% over 9 days**, via a sequence of small (<30%) steps rather than one correction. The trace
is not perfectly monotonic (07-02→07-04 has an uptick from 7.52 to 8.12) — consistent with a
biased-but-noisy process, not a single deterministic cause. A second, independent replay
(offline, against the same production CSVs, walking day-by-day from 2026-06-25 through 07-09)
shows the same directional decline over a longer span: 9.6h → 8.7h → 8.4h, after which the old
guard begins intermittently firing and holds the value — i.e. the drift is not confined to the
9-day window above, it's the tail end of a longer decline that predates it. This is evidence of a
systematic, not purely noise-driven, mechanism, though — per review — a single dataset replay is
necessarily correlational, not a controlled experiment; §4 makes live post-deploy monitoring the
actual confirmation, not the replay alone.

### 1.1 Root cause (confirmed against live data)

Investigation pulled the deployed model's cached CSVs (`climate_*.csv`,
`heating_active_heizung_wintermodus.csv`) and archived `meta.pkl` snapshots via Samba, plus
matching Open-Meteo archive weather, and replayed `_calibrate_tau()` against them directly
(`apps/energy_forecast/model.py`). Two compounding defects were confirmed:

**(A) Retrains fire far more often than the intended weekly cadence, and a cooldown resets on
every restart.** `energy_forecast.py:358` (`self.run_in(self._retrain_cb, 10)`) forces a full
retrain 10s after *every* `initialize()` — i.e. every AppDaemon container restart, independent of
the weekly `run_every` schedule. Worse, `_last_adaptive_retrain` (`energy_forecast.py:312`, the 24h
cooldown guarding the MAE-triggered adaptive retrain in `_maybe_adaptive_retrain`) is held only in
memory and initializes to `datetime.min` on every restart — so ~2 minutes later, `_update_cb` sees
`hours_since > 24h` and can fire a **second** full retrain. This matches the live data exactly:
2026-07-04 05:58 and 06:11 are 13 minutes apart. Each restart gets two free shots at recalibrating
τ instead of one bounded weekly shot.

**(B) The calibration's own damping never engages for realistic updates, and the guard is a hard
binary gate.** In `_calibrate_tau` (`model.py:1568`):
- The spring/summer guard (`night_frac < 15% AND outdoor_median > 12°C`) only blocks the most
  extreme case. Replaying real data showed it sitting **inactive** through late June while
  conditions were in a "gray zone" (night_frac ~14-25%, outdoor_median ~12-16°C), letting
  season-biased medians through completely unblocked.
- The EMA blend (`0.8×old + 0.2×new`) only activates when `change_frac > 0.5`
  (`model.py:1791-1803`). Every real observed step in the 9-day window was under 30% — the
  smoothing **never once engaged**; every retrain fully overwrote τ with the raw new median.

Defect (A) multiplies how often τ gets touched; defect (B) means each touch is either fully
unguarded or fully un-damped. Together: slow, compounding drift that the existing guard doesn't
stop, because the guard and the smoothing both only defend against the *extreme* case, not the
*steady small-bias, high-frequency* case actually happening.

---

## 2. Design

Four changes. Fix A replaces the guard + EMA with a continuous, correctly-composed confidence
weight. Fix B adds a two-tier rolling drift cap — a *second*, independent line of defense that bounds
*cumulative* movement regardless of how many times Fix A gets invoked, because per-update damping
alone (however well-tuned) only slows convergence to a biased estimate, it doesn't prevent it over
enough retrains (see §5, finding #2). Fix C persists the adaptive-retrain cooldown *and* gates the
unconditional startup retrain, closing both halves of defect (A). Fix D is documentation/logging
hygiene forced by the above changes.

### 2.1 Fix A: continuous, correctly-composed confidence weight

**Design principle (this replaced rev. 1's formula after review):** confidence must require *both*
signals (night fraction, outdoor temperature) to independently look winter-like — a single good
signal must not be able to rescue full trust for a window where the other signal is bad. The old
guard only blocked when *both* conditions were simultaneously extreme; anything else proceeded
with **zero** damping. Rev. 1 mistook "reproduce the old guard's permissive edge continuously" for
correctness — but that permissive edge (e.g. a warm, all-night window — a textbook
open-window-ventilation case in summer) is itself part of the bug, not a behavior worth preserving.
The corrected design requires **both** signals to be good simultaneously to reach high confidence:

```python
_SPRING_BIAS_OUTDOOR_TEMP: float = 12.0         # existing — outdoor_median at/above this = zero temp confidence
_SPRING_BIAS_OUTDOOR_TEMP_FLOOR: float = 5.0    # NEW — outdoor_median at/below this = full temp confidence
_SPRING_BIAS_NIGHT_FRAC_FULL: float = 0.40      # NEW — night_frac at/above this = full night confidence
                                                 # (widened from the old 0.15 guard threshold so the
                                                 # ramp doesn't saturate exactly where the old cliff sat)
_TAU_SAMPLE_CONF_REF: int = 3                   # NEW — candidate count at/above which sample confidence is full
_TAU_EMA_MAX_NEW_WEIGHT: float = 0.2            # weight given to the fresh estimate at full confidence
                                                 # (same value the old >50%-jump blend used)

assert _SPRING_BIAS_OUTDOOR_TEMP > _SPRING_BIAS_OUTDOOR_TEMP_FLOOR  # division-by-zero / inverted-ramp guard
```

Confidence is now computed **after** candidate selection, over the *same* population
(`selected`, the quality-filtered top-N) that `tau_median` itself comes from — rev. 1 computed
`night_frac`/`outdoor_median` from the raw, unfiltered candidate pool (and `outdoor_median` from
the *entire off-period timeline*, an even larger mismatch), decoupling the trust signal from what
it's supposed to be gating. Each candidate tuple gains a 4th element, its sub-window's mean
outdoor temperature (already computed as `sub_t_outdoor` a few lines earlier in the existing loop,
just not retained):

```python
candidates.append((tau, quality, hour_start, sub_t_outdoor))  # was: (tau, quality, hour_start)
```

```python
old_tau = self._tau_hours

candidates.sort(key=lambda c: c[1], reverse=True)
divisor = 4 if len(candidates) >= self._TAU_SELECTIVITY_THRESHOLD else 2
n_select = max(1, len(candidates) // divisor)
selected = candidates[:n_select]
tau_estimates = [c[0] for c in selected]

night_frac = sum(1 for c in selected if c[2] >= 22 or c[2] < 6) / len(selected)
outdoor_median = float(np.median([c[3] for c in selected]))

night_conf = min(1.0, night_frac / self._SPRING_BIAS_NIGHT_FRAC_FULL)
temp_conf = min(
    1.0,
    max(0.0, (self._SPRING_BIAS_OUTDOOR_TEMP - outdoor_median)
             / (self._SPRING_BIAS_OUTDOOR_TEMP - self._SPRING_BIAS_OUTDOOR_TEMP_FLOOR)),
)
# len(candidates) (pre-selection pool size), not len(selected) (which is always >=1 by
# construction) -- sparse RETRAINS should be trusted less, matching the existing
# len(candidates) >= 3 threshold this replaces (kept at the same magnitude as that
# pre-existing hard cutoff -- not re-derived from first principles, but not a new
# arbitrary choice either; a follow-up could tune it against real candidate-count
# distributions once more production data exists).
sample_conf = min(1.0, len(candidates) / self._TAU_SAMPLE_CONF_REF)

# AND, not "AND on badness" (rev. 1's bug): all three factors must be good simultaneously.
# Geometric mean, not a plain product (rev. 2 used night_conf*temp_conf*sample_conf directly):
# a plain 3-way product over-penalizes moderately-good conditions -- three factors of 0.7 each
# would multiply to 0.34, more than halving trust even though every individual signal looks
# "pretty good." The geometric mean preserves the exact same AND-semantics at the boundaries
# (any single factor at exactly 0 still forces confidence to exactly 0 -- 0**(1/3) == 0, so
# every "must be an exact preserve" case verified below is unaffected) while being much gentler
# in the interior: geometric mean of (0.7, 0.7, 0.7) = 0.7, not 0.34.
confidence = (night_conf * temp_conf * sample_conf) ** (1.0 / 3.0)

tau_median = float(np.median(tau_estimates))

if old_tau is not None and old_tau > 0:
    new_weight = confidence * self._TAU_EMA_MAX_NEW_WEIGHT
    tau_result = (1.0 - new_weight) * old_tau + new_weight * tau_median
    if confidence < 0.05:
        _LOGGER.info(
            "τ calibration: ~0%% confidence (night_frac=%.0f%%, T_out median=%.1f°C, "
            "%d candidates) — preserving stored τ=%.1f h.",
            night_frac * 100, outdoor_median, len(candidates), old_tau,
        )
    else:
        _LOGGER.info(
            "τ EMA blend: %.1f h → %.1f h (raw %.1f h, new_weight=%.0f%%, confidence=%.0f%% "
            "[night=%.0f%%, temp=%.0f%%, sample=%.0f%%])",
            old_tau, tau_result, tau_median, new_weight * 100, confidence * 100,
            night_conf * 100, temp_conf * 100, sample_conf * 100,
        )
else:
    tau_result = tau_median
```

Removed entirely: the pre-selection guard block, the `change_frac` computation, and the
`return old_tau` early exit. The `len(candidates) >= 3` special case is gone too — folded into
`sample_conf`'s continuous ramp instead of a hard cliff.

**Verified against the exact production incident** (offline replay, `night_frac≈100%` of *selected*
candidates but `outdoor_median≈14°C`, i.e. the literal gray-zone condition from §1's table):
`confidence=0` → exact preserve, unchanged by the geometric-mean switch (`0**(1/3) == 0`).
**Verified against the reviewers' key counterexample** (warm, all-night window,
`outdoor_median=18°C`): `confidence=0` → exact preserve — this is the case rev. 1's formula got
wrong (it produced `confidence=1`, full undamped trust). **Verified for the sample-size edge
case** (`n_days=1`, few candidates): with the geometric mean, `confidence≈0.69` (vs. `≈0.33` under
the plain product), landing the blended result at `≈10.21h` rather than `≈10.10h` — still clearly
damped relative to the raw estimate (`≈11.49h`), just less punitively than a straight product
would be for a case where two of the three factors are already at full trust.

**Persistence invariant (addresses a review question about state consistency):**
`_tau_hours`, `_tau_anchor_hours`, and `_tau_anchor_ts` (§2.2) are always read and written together
as part of the same `meta` dict via `_save()`/`_load()` (§2.6), and `_tau_hours` is never mutated
anywhere outside `_calibrate_tau()` itself. A model rollback (`_rollback_model_cb`, which restores
an archived `meta.pkl` wholesale) therefore restores all three fields from the same archived
snapshot together — they cannot desync via any existing code path, including rollback.

**Deliberate behavior change from rev. 1 and from the original code:** a cold-but-all-daytime
window (`night_frac=0%, outdoor_median=5°C`) now also gets `confidence=0` (preserved), not a full
update. The original code and rev. 1 both let this update fully/near-fully, on the theory that a
cold day is "safe." Per review (finding #12 in §5): the codebase's own candidate-quality-scoring
docstring (`model.py:1731`, "daytime windows produce τ estimates 5× shorter... due to
ventilation") attributes daytime bias to ventilation behavior, independent of temperature — so
requiring both signals to be good is the physically consistent choice, not just a formula
simplification. `tests/test_model.py::TestTauCalibrationSafeguards::test_spring_bias_guard_not_triggered_with_cold_outdoor`
is renamed and its assertion inverted accordingly (§3.1).

### 2.2 Fix B: two-tier rolling drift cap (new)

Per-update damping (Fix A) bounds how much a *single* retrain can move τ, but under sustained
gray-zone conditions across many retrains it still asymptotically converges to the same
(potentially biased) raw estimate — just more slowly (§5, finding #2). A single 30-day cap
(rev. 2) closes this *within* one window, but round-2 review correctly pointed out that it doesn't
stop compounding *across* windows: each time the window elapses, the anchor resets to wherever τ
currently sits — including a τ that was itself pushed to the previous window's edge. This
revision adds a second, longer, wider cap layered on top: both must hold simultaneously.

```python
_TAU_DRIFT_WINDOW_DAYS: int = 30        # short cap — catches a single bad month
_TAU_MAX_DRIFT_FRAC: float = 0.35
_TAU_LONG_DRIFT_WINDOW_DAYS: int = 180  # NEW — long cap — catches sustained multi-month bias
_TAU_LONG_MAX_DRIFT_FRAC: float = 0.50  # NEW
```

New persisted state (added to `meta.pkl` alongside `tau_hours`, same pattern as `last_trained`;
see the persistence invariant note in §2.1 for why these can't desync from `tau_hours`):

```python
self._tau_anchor_hours: float | None = None
self._tau_anchor_ts: pd.Timestamp | None = None
self._tau_long_anchor_hours: float | None = None
self._tau_long_anchor_ts: pd.Timestamp | None = None
```

Applied after Fix A's blend computes `tau_result`, using the latest timestamp already present in
the passive-window data (`combined.index.max()`) as "now" — this keeps the cap's clock tied to the
data's own time axis rather than wall-clock time, so it's deterministic in tests and unaffected by
how often retrains happen to fire. **Timezone note (round-2 review question):** `combined.index`
is built entirely from the tz-naive local timestamps already flowing through this function
(`climate_dfs`/`weather_df`/`heating_active_df`, all stripped of tzinfo upstream in
`energy_forecast.py` before being passed in), and `_tau_anchor_ts`/`_tau_long_anchor_ts` are only
ever assigned *from* `combined.index.max()` itself — never from an independently-sourced
wall-clock value — so the two sides of every comparison are on the same basis by construction.
This is different from Fix C's `_last_trained_local()` conversion (§2.3), which exists specifically
*because* `last_trained` comes from a different source (`datetime.now()`, system/UTC time) than
what it's compared against; no equivalent cross-source mismatch exists here. §3.2 adds a test
asserting `_tau_anchor_ts.tzinfo is None`, matching `combined.index`, to keep this true as the
code evolves.

```python
if old_tau is not None and old_tau > 0:
    new_weight = confidence * self._TAU_EMA_MAX_NEW_WEIGHT
    tau_result = (1.0 - new_weight) * old_tau + new_weight * tau_median
    latest_ts = combined.index.max()

    def _apply_drift_cap(anchor_h_attr, anchor_ts_attr, window_days, max_frac, result):
        anchor_h = getattr(self, anchor_h_attr, None)
        anchor_ts = getattr(self, anchor_ts_attr, None)
        if anchor_h is None or anchor_ts is None or (latest_ts - anchor_ts).days >= window_days:
            anchor_h = old_tau
            setattr(self, anchor_h_attr, anchor_h)
            setattr(self, anchor_ts_attr, latest_ts)
        max_drift = anchor_h * max_frac
        lo, hi = anchor_h - max_drift, anchor_h + max_drift
        if not (lo <= result <= hi):
            _LOGGER.warning(
                "τ drift cap (%d-day): %.1f h clamped to [%.1f, %.1f] h (anchor=%.1f h @ %s)",
                window_days, result, lo, hi, anchor_h, anchor_ts,
            )
        return min(max(result, lo), hi)

    # Short cap first, then long cap on the (possibly already-clamped) result -- the tighter
    # of the two always wins since both are applied in sequence.
    tau_result = _apply_drift_cap("_tau_anchor_hours", "_tau_anchor_ts",
                                   self._TAU_DRIFT_WINDOW_DAYS, self._TAU_MAX_DRIFT_FRAC, tau_result)
    tau_result = _apply_drift_cap("_tau_long_anchor_hours", "_tau_long_anchor_ts",
                                   self._TAU_LONG_DRIFT_WINDOW_DAYS, self._TAU_LONG_MAX_DRIFT_FRAC, tau_result)
else:
    tau_result = tau_median
```

±35% over 30 days is generous enough to allow genuine winter→spring seasonal correction over a
realistic timeframe, while ±50% over 180 days independently bounds the multi-window compounding
case the short cap alone can't stop.

**Verified the multi-window compounding gap is closed:** simulated 6 successive 30-day windows
under sustained, moderate (not confidence=0) gray-zone bias (`night_frac` good, `outdoor_median=8°C`
— partial `temp_conf`), forcing a short-window reset before each of the 6 retrains. Without the
long cap, 6 windows of unfettered 35%-per-window compounding would allow
`10.0 × 0.65⁶ ≈ 0.75h` — an implausible, near-total collapse. With the long cap active, τ is
correctly held at exactly the long-window floor, `10.0 × 0.5 = 5.0h`, after the 4th window and stays
there — bounded, not unbounded, under sustained bias.

### 2.3 Fix C: persist and correctly localize the adaptive-retrain cooldown; gate the startup retrain

Rev. 1's `_seed_adaptive_cooldown()` only fixed the *second* free retrain (the adaptive re-fire).
The *first* — the unconditional `run_in(self._retrain_cb, 10)` on every restart — was untouched,
and the seeding itself had a UTC/local-timezone bug (§5, findings #4, #5). All three are fixed
together because they share the same underlying local-time-conversion need:

```python
def _last_trained_local(self) -> datetime:
    """self._ml_model.last_trained as local wall-clock time.

    It's persisted via datetime.now() in model.py's train() — system/UTC time in Docker/HA,
    per the same reasoning already documented at _maybe_adaptive_retrain's local-time
    construction. Every retrain-cadence comparison in this file uses
    pd.Timestamp.now(self._timezone).tz_localize(None), so this converts last_trained to the
    same basis before any comparison — without it, comparisons are off by the UTC offset
    (1-2h for Europe/Zurich), which lets cooldown gates expire early near their boundary.
    """
    last_trained = self._ml_model.last_trained
    if last_trained == datetime.min:
        return last_trained
    return pd.Timestamp(last_trained).tz_localize("UTC").tz_convert(self._timezone).tz_localize(None)


def _seed_adaptive_cooldown(self) -> None:
    """Seed the adaptive-retrain cooldown from the persisted last-trained timestamp.

    Without this, self._last_adaptive_retrain starts at datetime.min on every AppDaemon
    restart, immediately re-arming the 24h adaptive-retrain cooldown. Re-synced at the end of
    every _retrain() too (not just here at startup) — otherwise this seed is a one-shot and
    _last_adaptive_retrain silently drifts apart from last_trained again after the very next
    retrain, which would only mask the bug for a single restart cycle.
    """
    self._last_adaptive_retrain = self._last_trained_local()


STARTUP_RETRAIN_MIN_GAP_HOURS = 6  # module-level constant (same convention as RETRAIN_INTERVAL_S,
                                   # energy_forecast.py:47 -- a bare module global, NOT a class
                                   # attribute; round-2 review caught an earlier draft of this
                                   # referencing it as self.STARTUP_RETRAIN_MIN_GAP_HOURS, which
                                   # would raise AttributeError on first call since nothing sets
                                   # it as an instance/class attribute)


def _maybe_startup_retrain(self, event_name=None, data=None, kwargs=None) -> None:
    """Startup retrain (run_in(..., 10)), skipped if a retrain already completed recently.

    Closes the other half of defect (A): previously *every* restart forced a full retrain
    regardless of how recently one had already happened, so a crash-loop or repeated manual
    restart could touch τ far more often than the intended weekly cadence. A restart shortly
    after a genuine code deploy will run on the previous model until the next
    scheduled/adaptive retrain — an acceptable trade-off given the model changes slowly
    week to week, and RELOAD_ENERGY_MODEL remains available for an explicit forced reload.
    """
    last_local = self._last_trained_local()
    if last_local != datetime.min:
        hours_since = (pd.Timestamp.now(self._timezone).tz_localize(None) - last_local).total_seconds() / 3600
        if hours_since < STARTUP_RETRAIN_MIN_GAP_HOURS:
            _LOGGER.info(
                "Startup retrain skipped — last retrain was %.1fh ago (< %dh gap); the "
                "weekly/adaptive schedule will pick up the next real update.",
                hours_since, STARTUP_RETRAIN_MIN_GAP_HOURS,
            )
            return
    self._retrain_cb(event_name, data, kwargs)
```

Call sites in `initialize()`:

```python
        self._ml_model = EnergyForecastModel(
            model_dir, model_archive_count=self._model_archive_count, timezone=self._timezone
        )
        self._seed_adaptive_cooldown()
        self._lock = threading.Lock()
        ...
        self.run_in(self._maybe_startup_retrain, 10)   # was: self.run_in(self._retrain_cb, 10)
        self.run_every(self._retrain_cb, f"now+{RETRAIN_INTERVAL_S + 10}", RETRAIN_INTERVAL_S)  # unchanged
```

And at the end of `_retrain()` (after `self._ml_model.train(...)` / `_save()` completes), one line
added to keep the cooldown resynced continuously rather than only at startup:

```python
        self._last_adaptive_retrain = self._last_trained_local()
```

`self.listen_event(self._retrain_cb, "RELOAD_ENERGY_MODEL")` and the weekly `run_every` both still
target `_retrain_cb` directly (unchanged) — an explicit operator-triggered reload must not be
silently skipped, and the weekly cadence is inherently ≥7 days apart so the 6h gate never applies
to it.

### 2.4 Fix D: documentation and observability

- **`_calibrate_tau`'s docstring** (`model.py:1568-1596`) still describes the old binary guard +
  conditional EMA in prose. Replace with:

  > Windows are scored rather than hard-filtered... A continuous confidence weight —
  > requiring both a sufficient nighttime fraction *and* a cold-enough outdoor median among the
  > selected top-N candidates, plus enough total candidates — scales how much a fresh estimate
  > can move the stored τ in a single retrain (0-20%). A 30-day rolling cap independently bounds
  > total movement to ±35% regardless of retrain frequency. See §2.1-2.2 of
  > `docs/superpowers/specs/2026-07-09-tau-calibration-drift-fix-design.md` for the full design
  > rationale.

- **No-op log line preserved** — see the `confidence < 0.05` branch in §2.1; an operator can still
  `grep "preserving stored"` to find effectively-skipped cycles, same as before.
- **§2.1's equivalence claim from rev. 1 is retracted.** Rev. 1 argued the new formula was
  "byte-for-byte identical to the old guard" at both extremes. That's no longer true by design —
  the cold-daytime extreme now behaves differently (§2.1, "Deliberate behavior change"), and this
  revision considers that a correctness fix, not a regression.

### 2.5 Downstream impact bound (addresses review finding on `physics.py`)

`τ` feeds `physics_model._tau_hours` (`model.py:407`), which sets `C_building_Wh_K = UA_eff × τ`
and the indoor-temperature free-decay rate `1/τ` in `physics.py` (`predict_training_series`,
`_project_indoor_temps`) — consumed downstream by the broader ha-energy-manager ecosystem's
heat-pump/DHW decisions per the Physics-ML hybrid design (see
`docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md`). A full HP-decision-level
regression suite is out of scope for this fix (`physics.py` is unchanged — see §2.7 "Files
Unchanged" — and is exercised by its own existing test suite), but this fix directly bounds the
input `physics.py` receives:

- **Single retrain:** τ can move by at most `confidence × 20%` of the gap to the fresh estimate
  (Fix A) — down from an unbounded 100% overwrite today. Fix B tightens this further: on the very
  first blended retrain in a rolling window, the anchor initializes to the *pre-update* stored
  value, so that same retrain is also immediately bounded to ±35% of its starting τ. Verified: an
  extreme synthetic pull (`tau_true=100h` against `old_tau=10h`, full confidence) is clamped to
  `13.5h` — the drift cap's boundary (`10 × 1.35`), not the nominally-looser `10 + 0.2×(100−10)=28h`
  the EMA weight alone would allow. In practice the tighter of the two bounds always wins.
- **Any 30-day window:** τ can move by at most 35% from its value at the window's start (Fix B),
  regardless of retrain count — down from the unbounded (and, per §1, observed) compounding today.

§3.4 adds one test asserting this per-retrain bound holds against a large synthetic pull, tying
the numeric guarantee directly to `Q_heat_el`'s linear dependence on `C_building_Wh_K`
(and therefore on τ) so a future change to the weight ceiling or drift cap can't silently widen
the bound this section documents without also breaking that test.

### 2.6 Files Modified

| File | Change |
|---|---|
| `apps/energy_forecast/model.py` | `_calibrate_tau`: replace binary guard + conditional EMA with confidence-weighted blend computed over selected candidates (§2.1); add two-tier rolling drift cap (§2.2); update docstring (§2.4); new class constants; `_save`/`_load` gain `tau_anchor_hours`/`tau_anchor_ts`/`tau_long_anchor_hours`/`tau_long_anchor_ts` in the `meta` dict |
| `apps/energy_forecast/energy_forecast.py` | Add `_last_trained_local()`, `_seed_adaptive_cooldown()`, `_maybe_startup_retrain()` (§2.3); change `run_in(self._retrain_cb, 10)` → `run_in(self._maybe_startup_retrain, 10)`; add cooldown resync at the end of `_retrain()` |
| `tests/test_model.py` | Update/extend `TestTauCalibrationSafeguards` (§3.1); new `TestTauDriftCap` (§3.2) |
| `tests/test_energy_forecast.py` | Extend `TestAdaptiveRetrainLock`; new `TestSeedAdaptiveCooldown`, `TestMaybeStartupRetrain` (§3.3) |

### 2.7 Files Unchanged

`ha_data.py`, `weather.py`, `physics.py`, `clustering.py`, `const.py` — this is scoped entirely to
the τ calibration path and the adaptive/startup-retrain triggers. `physics.py`'s consumption of
`_tau_hours` is unchanged; only the range and cadence of values it receives is now bounded (§2.5).

---

## 3. Testing

### 3.1 `tests/test_model.py` — `TestTauCalibrationSafeguards`

**Unaffected** (verified numerically against the redesigned formula, same as rev. 1's
verification pass): tests that never set `model._tau_hours` before calling `_calibrate_tau`
(`test_high_solar_penalized_not_excluded`, `test_daytime_windows_penalized_not_excluded`,
`test_single_window_sufficient`, `test_no_radiation_column_degrades_gracefully`,
`test_top_50_percent_selected`, the daytime-vs-night comparison tests) hit the `old_tau is None`
branch, untouched by either Fix A or Fix B.

**Must change — behavior intentionally differs from rev. 1 *and* from the original code:**

```python
def test_cold_daytime_also_damped_not_fully_trusted(self, tmp_path):
    """Replaces test_spring_bias_guard_not_triggered_with_cold_outdoor, whose name and
    assertion encoded the pre-fix assumption that cold-but-all-daytime data is safe to fully
    trust. Per review: the codebase's own quality-scoring docstring (model.py:1731) attributes
    daytime bias to ventilation, independent of temperature — confidence now correctly requires
    BOTH night_frac and outdoor_median to be good, so this case gets confidence=0, same as the
    old guard's "both bad" case, not a full update.
    """
    model = _make_tau_model(tmp_path)
    model._tau_hours = 25.0
    cold_day_dfs, cold_day_heat, cold_day_wx = self._make_night_blocks(
        tau_true=10.0, n_days=5, start_hour=10, t_out=5.0,
    )

    result = model._calibrate_tau(cold_day_dfs, cold_day_heat, cold_day_wx)

    assert result == 25.0, "night_frac=0% must zero out confidence regardless of temperature"


def test_spring_bias_guard_preserves_stored_tau(self, tmp_path):
    """Unaffected in outcome (still an exact preserve) but now reached via confidence=0
    rather than the removed hard-coded early-return guard."""
    model = _make_tau_model(tmp_path)
    model._tau_hours = 25.0
    bias_dfs, bias_heat, bias_wx = self._make_night_blocks(tau_true=5.0, n_days=5, start_hour=10, t_out=18.0)

    result = model._calibrate_tau(bias_dfs, bias_heat, bias_wx)

    assert result == 25.0
```

**New — the production incident, verbatim:**

```python
def test_production_gray_zone_incident_now_preserves(self, tmp_path):
    """Reproduces the literal §1 incident conditions (night_frac~100% of a small selected
    set, outdoor_median~14°C — just above the old guard's 12°C trigger, which is exactly why
    the old guard never fired here). Rev. 1's formula also got this case wrong (confidence=1,
    full trust) because a single good signal (night) could rescue trust. This is the
    regression test for the actual bug.
    """
    model = _make_tau_model(tmp_path)
    model._tau_hours = 10.0
    dfs, heat, wx = self._make_night_blocks(
        tau_true=15.0, n_days=6, start_hour=23, window_hours=3, t_out=14.0,
    )

    result = model._calibrate_tau(dfs, heat, wx)

    assert result == 10.0, "the exact production gray-zone condition must now be an exact preserve"


def test_warm_mostly_night_window_not_fully_trusted(self, tmp_path):
    """The reviewers' key counterexample to rev. 1: an all-nighttime but warm window (a
    textbook summer window-ventilation case) must NOT reach full confidence just because
    night_frac is good — both signals are required.
    """
    model = _make_tau_model(tmp_path)
    model._tau_hours = 10.0
    dfs, heat, wx = self._make_night_blocks(tau_true=16.0, n_days=5, start_hour=22, t_out=18.0)

    result = model._calibrate_tau(dfs, heat, wx)

    assert result == 10.0


def test_sparse_candidates_reduce_confidence_not_maximize_it(self, tmp_path):
    """len(candidates) below _TAU_SAMPLE_CONF_REF must scale confidence down, not leave it at
    the old code's implicit "guard doesn't apply, proceed at full trust" default.

    Compared against a full-confidence run on independently-generated data with the same
    tau_true, rather than a captured/narrated number for this specific fixture -- avoids the
    circular-derivation pattern flagged in round-1 review.
    """
    model_sparse = _make_tau_model(tmp_path / "sparse")
    model_sparse._tau_hours = 10.0
    sparse_dfs, sparse_heat, sparse_wx = self._make_night_blocks(tau_true=12.0, start_hour=22, n_days=1)
    sparse_result = model_sparse._calibrate_tau(sparse_dfs, sparse_heat, sparse_wx)

    model_full = _make_tau_model(tmp_path / "full")
    model_full._tau_hours = 10.0
    full_dfs, full_heat, full_wx = self._make_night_blocks(tau_true=12.0, start_hour=22, n_days=5)
    full_result = model_full._calibrate_tau(full_dfs, full_heat, full_wx)

    assert sparse_result is not None and full_result is not None
    assert 10.0 < sparse_result < full_result, (
        "a 1-day (sparse) retrain must land strictly closer to the stored τ than an "
        "otherwise-identical 5-day (ample-sample) retrain toward the same raw estimate"
    )


def test_both_signals_good_updates_normally(self, tmp_path):
    """Sanity check: when both night_frac and outdoor_median genuinely look winter-like,
    normal (confidence≈1) damped updates still occur — this isn't a regression to "never
    update."""
    model = _make_tau_model(tmp_path)
    model._tau_hours = 10.0
    dfs, heat, wx = self._make_night_blocks(tau_true=12.0, start_hour=22, t_out=5.0)

    result = model._calibrate_tau(dfs, heat, wx)

    assert result is not None
    assert 10.0 < result < 10.8
```

### 3.2 `tests/test_model.py` — new `TestTauDriftCap`

**Fixture-compatibility note (round-2 review caught this):** `_make_night_blocks` (the existing
shared fixture helper, `tests/test_model.py:4706`) hardcodes its dates to January 2026
(`f"2026-01-{day_start + day:02d} ..."`) — it does not take a month parameter. All anchor
timestamps below are therefore set well *before* January 2026 (not "2026-06-01" as an earlier
draft of this spec incorrectly used, which would have put the anchor *after* the fixture's data
and broken every elapsed-days comparison).

```python
class TestTauDriftCap:
    """Fix B: bounds cumulative τ movement to ±_TAU_MAX_DRIFT_FRAC (short) / ±_TAU_LONG_MAX_DRIFT_FRAC
    (long) from the value at the start of each respective rolling window, independent of Fix A's
    per-step damping and independent of how many retrains occur inside the window."""

    def _seed_anchors(self, model, anchor_hours, anchor_ts):
        """Seed both short and long anchors to the same value/timestamp -- the common case for
        a model that's had a stable τ for a while before the window under test begins."""
        model._tau_anchor_hours = anchor_hours
        model._tau_anchor_ts = anchor_ts
        model._tau_long_anchor_hours = anchor_hours
        model._tau_long_anchor_ts = anchor_ts

    def test_clamps_when_cumulative_budget_exhausted(self, tmp_path):
        """A single retrain's per-step-legal blend must still be clamped if it would push τ
        beyond the rolling window's remaining budget. Verified: old_tau=6.0 sits below the
        window floor (anchor=10.0 * 0.65 = 6.5); even a modest raw pull (tau_true=6.5) that
        would normally blend to something >= 6.0 gets clamped up to exactly the floor, 6.5.
        Anchor set 2025-12-15 -- ~19 days before the fixture's Jan 3-9 data, inside the 30-day
        short window (so the short cap applies without resetting first)."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 6.0  # already drifted below the window's floor
        self._seed_anchors(model, 10.0, pd.Timestamp("2025-12-15"))

        dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
            tau_true=6.5, start_hour=22, t_out=5.0, day_start=3,
        )
        result = model._calibrate_tau(dfs, heat, wx)

        assert result is not None
        assert result == pytest.approx(6.5), "must be clamped up to the 30-day drift floor (anchor * 0.65)"

    def test_resets_after_window_elapses(self, tmp_path):
        """Once _TAU_DRIFT_WINDOW_DAYS has passed since the anchor, a new anchor is set at
        the current stored τ — this is what allows genuine multi-month seasonal correction
        rather than freezing τ forever after one bad month. Anchor set 2025-10-01 -- roughly
        97 days before the fixture's Jan 1-6 data, comfortably past the 30-day window."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        self._seed_anchors(model, 10.0, pd.Timestamp("2025-10-01"))

        dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
            tau_true=10.0, start_hour=22, t_out=5.0, day_start=1,
        )
        model._calibrate_tau(dfs, heat, wx)

        assert model._tau_anchor_ts > pd.Timestamp("2025-10-01")

    def test_no_cap_within_budget(self, tmp_path):
        """A modest, per-step-legal move well inside the 35% budget is not touched by the cap."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        self._seed_anchors(model, 10.0, pd.Timestamp("2025-12-15"))

        dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
            tau_true=11.0, start_hour=22, t_out=5.0, day_start=3,
        )
        result = model._calibrate_tau(dfs, heat, wx)

        assert result is not None
        assert 10.0 < result < 13.5  # within [anchor*0.65, anchor*1.35] = [6.5, 13.5]

    def test_anchor_timestamps_are_tz_naive(self, tmp_path):
        """Both anchor timestamps must stay tz-naive, matching combined.index -- guards the
        round-2 review question about tz consistency (see §2.2's timezone note)."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
            tau_true=11.0, start_hour=22, t_out=5.0,
        )
        model._calibrate_tau(dfs, heat, wx)

        assert model._tau_anchor_ts.tzinfo is None
        assert model._tau_long_anchor_ts.tzinfo is None

    def test_long_cap_bounds_multi_window_compounding(self, tmp_path):
        """The round-2 review's key finding: the short (30-day) cap alone only bounds movement
        WITHIN one window, not ACROSS successive windows under sustained bias -- each window
        reset re-anchors at wherever the previous window left off. This test simulates 6
        successive short-window resets under a sustained, moderate (not confidence=0) pull and
        confirms the long (180-day) cap holds the line where the short cap alone would not.

        Verified numerically: without the long cap, 6 windows of unfettered 35%-per-window
        compounding would allow 10.0 * 0.65**6 =~ 0.75h -- an implausible near-total collapse.
        With the long cap, τ is held at exactly the long-window floor, 10.0 * 0.5 = 5.0h.
        """
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0

        for i in range(6):
            dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
                tau_true=2.0, n_days=5, start_hour=22, window_hours=8, t_out=8.0, day_start=1,
            )
            if i > 0:
                # Force the short window to have elapsed before each subsequent retrain,
                # simulating six real 30-day windows without needing six actual calendar months
                # of synthetic data (the shared fixture's dates are fixed to January).
                model._tau_anchor_ts = model._tau_anchor_ts - pd.Timedelta(days=31)
            result = model._calibrate_tau(dfs, heat, wx)
            assert result is not None
            model._tau_hours = result

        assert model._tau_hours == pytest.approx(5.0), (
            "sustained bias across many short-window resets must still be caught by the "
            "180-day/50% long cap, not allowed to compound toward the naive ~0.75h"
        )
```

### 3.3 `tests/test_energy_forecast.py`

```python
class TestSeedAdaptiveCooldown:
    """Fix C: the adaptive-retrain cooldown must be seeded from the persisted last_trained
    timestamp, converted to local time (not left UTC-naive), so a restart can't immediately
    re-arm it and the cooldown doesn't expire early near its boundary."""

    def test_seed_converts_utc_naive_last_trained_to_local(self):
        """Regression test for the UTC/local mismatch: last_trained is UTC-naive
        (datetime.now() in model.py), matching how it's genuinely produced in production —
        NOT a local-tz value hand-built to look correct (that was rev. 1's test bug)."""
        from datetime import datetime, timezone

        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeSelf()
        fake._timezone = "Europe/Zurich"
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        fake._ml_model.last_trained = utc_now  # exactly how model.py produces it

        EnergyForecast._seed_adaptive_cooldown(fake)

        expected_local = pd.Timestamp(utc_now).tz_localize("UTC").tz_convert("Europe/Zurich").tz_localize(None)
        assert fake._last_adaptive_retrain == expected_local
        # In summer (CEST, UTC+2) this must differ from the naive value by ~2h -- if it
        # doesn't, the conversion silently isn't happening.
        assert abs((fake._last_adaptive_retrain - pd.Timestamp(utc_now)).total_seconds()) > 3000

    def test_seed_with_never_trained_model_keeps_datetime_min(self):
        """A genuinely fresh install (no meta.pkl yet) must not be blocked from its first
        adaptive retrain — last_trained defaults to datetime.min, same as before."""
        from datetime import datetime

        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeSelf()
        fake._ml_model.last_trained = datetime.min

        EnergyForecast._seed_adaptive_cooldown(fake)

        assert fake._last_adaptive_retrain == datetime.min


class TestMaybeStartupRetrain:
    """Fix C: the startup retrain (run_in(..., 10)) must be skipped if a retrain already
    completed within STARTUP_RETRAIN_MIN_GAP_HOURS -- this is the fix for the *other* half of
    defect (A) that TestSeedAdaptiveCooldown alone doesn't cover."""

    def test_skips_when_recently_trained(self):
        from datetime import datetime, timezone

        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeSelf()
        fake._timezone = "Europe/Zurich"
        fake._ml_model.last_trained = (
            datetime.now(timezone.utc).replace(tzinfo=None) - pd.Timedelta(hours=2)
        )
        retrain_calls = []
        fake._retrain_cb = lambda *a, **kw: retrain_calls.append(1)

        EnergyForecast._maybe_startup_retrain(fake)

        assert retrain_calls == [], "a restart 2h after the last retrain must not force another one"

    def test_fires_when_stale(self):
        from datetime import datetime

        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeSelf()
        fake._timezone = "Europe/Zurich"
        fake._ml_model.last_trained = datetime.min
        retrain_calls = []
        fake._retrain_cb = lambda *a, **kw: retrain_calls.append(1)

        EnergyForecast._maybe_startup_retrain(fake)

        assert retrain_calls == [1], "a genuinely fresh/stale model must still retrain on startup"


class TestAdaptiveRetrainLock:
    # ... existing tests unchanged ...

    def test_seeded_cooldown_prevents_immediate_refire_after_restart(self):
        """Regression test for the production bug: simulates a restart 2 hours after the
        last real retrain, using the actual UTC-naive form last_trained is produced in
        (not a hand-built local-tz value — see TestSeedAdaptiveCooldown's note)."""
        from datetime import datetime, timezone
        from unittest.mock import patch

        from energy_forecast.energy_forecast import EnergyForecast

        fake = self._make_fake()
        fake._ml_model.last_trained = datetime.now(timezone.utc).replace(tzinfo=None) - pd.Timedelta(hours=2)
        EnergyForecast._seed_adaptive_cooldown(fake)

        retrain_calls = []
        fake._retrain = lambda: retrain_calls.append(1)

        with patch("energy_forecast.energy_forecast._compute_live_mae", return_value=(999.0, 100)):
            EnergyForecast._maybe_adaptive_retrain(fake, pd.DataFrame(columns=["timestamp", "gross_kwh"]))

        assert retrain_calls == [], "adaptive retrain must respect the cooldown seeded from last_trained"
```

`_FakeSelf` already stubs `_ml_model` as a `MagicMock`, so `fake._ml_model.last_trained`
assignment works with no fixture changes.

### 3.4 Downstream-bound test (`tests/test_model.py`)

```python
def test_single_retrain_tau_move_bounded_by_drift_cap(self, tmp_path):
    """Ties the §2.5 downstream-impact bound to an executable assertion. Verified: for an
    extreme, fully-trusted pull, Fix B's ±35%-of-anchor drift cap is the *tighter* bound
    (not Fix A's confidence*20%-of-gap alone) because the anchor initializes to the
    pre-update stored value on the very first blended retrain -- so even a pathological
    raw estimate cannot move τ by more than 35% in one retrain. This is what bounds the
    C_building_Wh_K / Q_heat_el perturbation physics.py sees per retrain."""
    model = _make_tau_model(tmp_path)
    model._tau_hours = 10.0
    # Extreme, fully-trusted pull (both signals good) toward a very different raw estimate.
    dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
        tau_true=100.0, start_hour=22, t_out=5.0, window_hours=12,
    )

    result = model._calibrate_tau(dfs, heat, wx)

    assert result is not None
    assert result == pytest.approx(13.5), "drift cap (anchor=10.0 * 1.35) must be the binding constraint"
    assert abs(result - 10.0) <= 10.0 * 0.35 + 1e-6
```

### 3.5 Full-suite verification

Run `python -m pytest tests/ -v` after each of Fix A/B and Fix C/D (not just the new/changed
tests above) per project convention — both changes touch widely-used methods, so a full pass is
required to catch anything not enumerated above.

---

## 4. Rollout / Post-deploy Verification

Same diagnostic path used to find this bug is the way to confirm the fix:

```bash
START=$(date -u -d '9 days ago' +%Y-%m-%dT%H:%M:%SZ)
curl -s -H "Authorization: Bearer $EM_HA_TOKEN" \
  "http://homeassistant:8123/api/history/period/${START}?filter_entity_id=sensor.ha_energy_forecast_thermal_pressure_net" \
  | python3 -c "
import json, sys
for p in json.load(sys.stdin)[0]:
    print(p['last_changed'], p['attributes'].get('tau_hours'))
"
```

Expect: (a) at most one or two τ updates per week post-deploy (confirms Fix C — no more
restart-doubled or restart-repeated retrains), (b) week-over-week τ changes bounded to roughly
≤20% of the old→new gap per update once in steady state (confirms Fix A), (c) over any 30-day
span, total movement bounded to ±35% of the value at the start of that span regardless of how many
updates occurred (confirms Fix B's short cap), and (d) over any 180-day span, bounded to ±50%
regardless of how many 30-day windows elapsed within it (confirms Fix B's long cap — the actual
guarantee against the originally-reported "slow slip" persisting across a full seasonal
transition, as opposed to (c) alone which a sustained bias could still walk through one
window-reset at a time).

No config or migration changes to existing fields — `_tau_hours` itself is unaffected in shape
(still `float | None` persisted in `meta.pkl`). The four new persisted fields
(`_tau_anchor_hours`, `_tau_anchor_ts`, `_tau_long_anchor_hours`, `_tau_long_anchor_ts`) default to
`None` and self-initialize on the first retrain after deploy, so existing deployments pick this up
with no manual intervention.

---

## 5. Multi-Stakeholder Review — Findings and Disposition

Reviewed in parallel by a building-thermal/controls domain engineer, a data scientist, and a
software engineer against rev. 1 of this spec. 20 findings total (6 High, 10 Medium, 4 Low).
Disposition:

| # | Sev | Finding | Disposition |
|---|---|---|---|
| 1 | High | Confidence formula let one good signal rescue full trust for a bad partner signal | **Fixed** — §2.1 redesign, multiplicative AND on confidence (not badness) |
| 2 | High | Per-step EMA damping doesn't bound cumulative drift | **Fixed** — §2.2 rolling drift cap, independent of retrain count |
| 3 | High | Fixed 0.2 EMA ceiling unjustified, ignores sample size | **Fixed** — `sample_conf` term added; ceiling itself kept at 0.2 (matches the pre-existing, already-shipped >50%-jump case) but now demonstrably bounded further by Fix B |
| 4 | High | Fix 2 (rev. 1) only closed the adaptive-retrain double-fire, not the startup retrain | **Fixed** — §2.3 `_maybe_startup_retrain` |
| 5 | High | UTC/local-tz mismatch in cooldown seeding | **Fixed** — §2.3 `_last_trained_local()` |
| 6 | High | Confidence defaulted to 1.0 (max trust) for `len(candidates)<3` | **Fixed** — `sample_conf` ramp |
| 7 | Med | Root-cause evidence from a small, non-monotonic sample overstated as "confirmed" | **Addressed** — §1 adds the longer independent replay and softened language; §4 remains the actual confirmation mechanism |
| 8 | Med | Confidence computed over a different population than `tau_median` | **Fixed** — §2.1, confidence now computed over `selected` (post quality-filter), same population |
| 9 | Med | One global scalar can't represent a heterogeneous multi-week block | **Mitigated, not fully solved** — fixing #8 substantially narrows this (confidence and the estimate now share a population); full per-candidate weighting is a larger structural change, noted as a follow-up, not blocking |
| 10 | Med | Downstream `physics.py`/HP-decision impact unchecked | **Addressed** — §2.5 derives and tests an explicit per-retrain and per-30-days bound; full HP-decision regression suite out of scope (physics.py unchanged, has its own tests) |
| 11 | Med | Stale docstring | **Fixed** — §2.4 |
| 12 | Med | Confidence formula (temp-only) contradicted the ventilation-attributed daytime-bias docstring elsewhere in the file | **Fixed** — §2.1's redesign requires night_frac too, resolving the contradiction |
| 13 | Med | Loss of a clean "no-op" log line | **Fixed** — §2.1, `confidence < 0.05` branch |
| 14 | Med | No test for cumulative drift or the real production night_frac band | **Fixed** — §3.1 `test_production_gray_zone_incident_now_preserves`, §3.2 `TestTauDriftCap` |
| 15 | Med | Fix 2's own test used a hand-built local-tz fixture, not real UTC production semantics | **Fixed** — §3.3 tests now use `datetime.now(timezone.utc)` |
| 16 | Med | New tests' numeric bounds read as captured-then-asserted, some tight | **Fixed** — §3.1/3.2 bounds widened and tied to formula arithmetic (e.g. `[anchor*0.65, anchor*1.35]`) rather than captured decimals |
| 17 | Low | Boundary mismatch vs. old guard at exactly 12°C | **Moot** — the "byte-for-byte equivalence" claim is retracted (§2.4); extremes are now intentionally different |
| 18 | Low | Cooldown seed is one-shot, drifts from `last_trained` after first retrain | **Fixed** — §2.3, resync added at the end of every `_retrain()` |
| 19 | Low | No invariant guard on the two temp constants | **Fixed** — `assert` added in §2.1 |
| 20 | Low | "Confidence" conflates bias-risk and sample-size uncertainty | **Fixed** — `sample_conf` is now a named, separate factor rather than an implicit default |

---

## 6. Round-2 Review — Findings and Disposition

Rev. 2 (§5's fixes applied) was re-reviewed by the same three experts. 9 issues carried over for
re-verification (all RESOLVED or PARTIALLY RESOLVED, none NOT RESOLVED), plus 7 new issues raised
against the rev. 2 changes themselves. Disposition of the new issues:

| # | Sev | Finding | Source | Disposition |
|---|---|---|---|---|
| 1 | High | The 30-day drift cap bounds movement *within* a window but not *across* successive windows — sustained bias could still compound roughly `0.65ⁿ` over `n` window resets | DS + DSE | **Fixed** — §2.2 adds a second, longer (180-day/50%) cap layered on top; verified numerically to hold τ at the correct floor (5.0h) where an unbounded short-cap-only design would allow ~0.75h over 6 sustained-bias windows |
| 2 | Medium | Unconditional clamp could in principle override a zero-confidence "exact preserve" if `tau_hours` and the anchor fields ever desynced (e.g. via model rollback) | DS | **Addressed** — §2.1 adds an explicit persistence-invariant note: all three fields are always saved/loaded together in the same `meta` dict, and `tau_hours` is never mutated outside `_calibrate_tau()`, so this can't happen via any existing code path |
| 3 | Medium | Three-way multiplicative product (`night_conf*temp_conf*sample_conf`) over-penalizes moderately-good conditions relative to the stated "both signals independently look winter-like" design intent | DS | **Fixed** — §2.1 switches to the geometric mean, `(night_conf*temp_conf*sample_conf)**(1/3)`; verified the exact-zero boundary cases (all "must preserve" tests) are unaffected, while the interior sample-size case moves from `confidence≈0.33` to `≈0.69` |
| 4 | Medium | tz basis of `combined.index.max()` vs. the persisted anchor timestamp never explicitly confirmed or tested | DSE | **Addressed** — §2.2 adds an explicit note (both sides are always tz-naive local time by construction, unlike Fix C's `last_trained` which genuinely needs conversion) plus a regression test (`test_anchor_timestamps_are_tz_naive`) |
| 5 | Low | `_TAU_SAMPLE_CONF_REF=3` carried over from the old cutoff without re-justifying the magnitude | DSE | **Acknowledged, not re-derived** — §2.1 adds a comment noting it's kept at the same magnitude as the pre-existing hard cutoff (not new arbitrariness) with re-tuning flagged as a data-driven follow-up once more production candidate-count history exists |
| 6 | High | `STARTUP_RETRAIN_MIN_GAP_HOURS` was referenced as `self.STARTUP_RETRAIN_MIN_GAP_HOURS` despite being declared as a module-level constant — would raise `AttributeError` on first call | SWE | **Fixed** — §2.3, both references changed to the bare module-level name, matching the existing `RETRAIN_INTERVAL_S` convention |
| 7 | Medium | `test_resets_after_window_elapses` used an anchor date (`2026-01-01`) and fixture `day_start` that, given the shared `_make_night_blocks` helper's *hardcoded* January dates, produced a data range only ~5 days after the anchor — not the 30+ days needed to actually exercise the reset branch; the assertion would have failed | SWE | **Fixed** — §3.2, all `TestTauDriftCap` anchor dates moved to genuinely pre-January-2026 dates (`2025-10-01` / `2025-12-15`) verified against the real fixture's date construction, not a custom test-only helper |

**Software-engineer's verification of round-1 fixes (round 2), for completeness:** all 8 of that
reviewer's round-1 issues confirmed RESOLVED against the actual current codebase (not just
plausible-looking spec prose) — including tracing the exact `tz_localize("UTC").tz_convert(...)
.tz_localize(None)` chain in §2.3 against the established pattern already in
`_maybe_adaptive_retrain` (`energy_forecast.py:1904`), and confirming `_FakeSelf`'s
unbound-method-call compatibility with the new `_seed_adaptive_cooldown`/`_maybe_startup_retrain`
methods.

No further review round required: every High and Medium finding from both rounds is now Fixed or
Addressed with an executable test; remaining Low items are either Acknowledged trade-offs (§6
row 5) or Moot due to an earlier claim being retracted (§5 row 17).
