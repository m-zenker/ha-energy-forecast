# τ Calibration Drift Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the confirmed τ (building thermal time constant) calibration drift bug — replace the binary spring-bias guard and conditional EMA in `_calibrate_tau()` with a continuous, correctly-composed confidence weight and a two-tier rolling drift cap, and fix the two defects (unbounded startup retrain, UTC/local-tz cooldown bug) that let retrains fire far more often than intended.

**Architecture:** Two independent, layered defenses inside `EnergyForecastModel._calibrate_tau()` (`apps/energy_forecast/model.py`): a confidence weight (0-100%, requiring both a sufficient nighttime fraction *and* a cold-enough outdoor median among the quality-selected candidates, geometric-mean-composed with a sample-size term) scales how much a single retrain can move τ, and a two-tier rolling cap (±35%/30 days, ±50%/180 days, anchored to τ's value at each window's start) independently bounds cumulative movement regardless of retrain frequency. Separately, `EnergyForecast` (`apps/energy_forecast/energy_forecast.py`) gains a persisted, correctly-localized adaptive-retrain cooldown and a gate on the unconditional startup retrain, so a container restart can no longer fire two uncapped retrains in quick succession.

**Tech Stack:** Python 3.13, pandas, numpy, AppDaemon `hassapi.Hass` (stubbed in tests), pytest — no new dependencies.

**Full design rationale, numeric verification, and the two-round multi-stakeholder review this plan implements:** `docs/superpowers/specs/2026-07-09-tau-calibration-drift-fix-design.md` (rev. 3, APPROVED). Every code block in this plan is copied verbatim from that spec's §2; this plan only adds exact file paths, line numbers, and step-by-step TDD sequencing.

## Global Constraints

- Base branch: **`feat/physics-core-engine`** (current worktree). `_calibrate_tau`'s `_find_passive_windows`-based structure this fix modifies only exists on this branch, not on `dev`/`main` — confirmed via `git show origin/dev:apps/energy_forecast/model.py | grep _find_passive_windows` (0 matches).
- Branch name: `fix/tau-calibration-drift`, created off `feat/physics-core-engine`.
- Run `python -m pytest tests/ -v` after **every task**, not just the new/changed tests (per project CLAUDE.md and spec §3.5) — both changed files are widely used elsewhere in the test suite.
- No new runtime dependencies. No changes to `pyproject.toml`.
- Follow existing repo/file conventions exactly: `pd`/`np` are imported **locally inside each method** in both `model.py` and `energy_forecast.py` (there is no module-level `import pandas as pd` at runtime in `energy_forecast.py` — it's only under `TYPE_CHECKING` at line 33), module-level constants are bare (not `self.`-prefixed) matching `RETRAIN_INTERVAL_S` at `energy_forecast.py:47`, and `_FakeSelf`/unbound-method-call test patterns from `tests/test_energy_forecast.py:596`.
- `scripts/` is gitignored in this repo (`.gitignore:36`) — the spec's §4.1 one-time seed procedure is a manual deploy-time step, not a tracked deliverable of this plan (see "Deployment" section at the end of this document).
- This is a bugfix scoped entirely to `_calibrate_tau` and the adaptive/startup-retrain triggers. Do not touch `ha_data.py`, `weather.py`, `physics.py`, `clustering.py`, or `const.py` (spec §2.7).

---

### Task 1: Branch setup

**Files:** none (git operation only)

- [ ] **Step 1: Create the fix branch off the current branch**

```bash
git checkout feat/physics-core-engine
git pull
git checkout -b fix/tau-calibration-drift
```

- [ ] **Step 2: Confirm the baseline test suite passes before making any changes**

Run: `python -m pytest tests/ -v 2>&1 | tail -20`
Expected: all tests pass (or the same pre-existing failures noted in project memory, none new).

---

### Task 2: Fix A — confidence formula (class constants + candidate tuple)

**Files:**
- Modify: `apps/energy_forecast/model.py:245-247` (class constants)
- Modify: `apps/energy_forecast/model.py` — the line appending to `candidates` inside `_calibrate_tau`'s per-sub-window loop (currently `candidates.append((tau, quality, hour_start))`, one occurrence)
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `EnergyForecastModel._SPRING_BIAS_OUTDOOR_TEMP_FLOOR`, `_SPRING_BIAS_NIGHT_FRAC_FULL`, `_TAU_SAMPLE_CONF_REF`, `_TAU_EMA_MAX_NEW_WEIGHT` class constants, consumed by Task 3.
- Candidate tuples become 4-element `(tau, quality, hour_start, sub_t_outdoor)`, consumed by Task 3.

This task only lands the constants and the extra tuple element — no test can observe them yet in isolation (they're inert until Task 3 wires them into the confidence calculation), so it's combined with Task 3's tests below rather than given its own. Do Steps 1-2 here, then continue directly into Task 3.

- [ ] **Step 1: Replace the class constants block**

Current (`model.py:245-247`):
```python
    _TAU_SELECTIVITY_THRESHOLD: int = 32  # ≥ this many candidates → top-25%; below → top-50%
    _SPRING_BIAS_NIGHT_FRAC: float = 0.15  # fraction of nighttime candidates below which guard fires
    _SPRING_BIAS_OUTDOOR_TEMP: float = 12.0  # °C outdoor median above which guard fires
```

Replace with:
```python
    _TAU_SELECTIVITY_THRESHOLD: int = 32  # ≥ this many candidates → top-25%; below → top-50%
    _SPRING_BIAS_OUTDOOR_TEMP: float = 12.0  # °C outdoor median at/above which temp confidence is zero
    _SPRING_BIAS_OUTDOOR_TEMP_FLOOR: float = 5.0  # °C outdoor median at/below which temp confidence is full
    _SPRING_BIAS_NIGHT_FRAC_FULL: float = 0.40  # night_frac at/above which night confidence is full
    _TAU_SAMPLE_CONF_REF: int = 3  # candidate count at/above which sample confidence is full
    _TAU_EMA_MAX_NEW_WEIGHT: float = 0.2  # weight given to the fresh τ estimate at full confidence
    _TAU_DRIFT_WINDOW_DAYS: int = 30  # short drift-cap window (days)
    _TAU_MAX_DRIFT_FRAC: float = 0.35  # short drift-cap max fractional move
    _TAU_LONG_DRIFT_WINDOW_DAYS: int = 180  # long drift-cap window (days)
    _TAU_LONG_MAX_DRIFT_FRAC: float = 0.50  # long drift-cap max fractional move

    assert _SPRING_BIAS_OUTDOOR_TEMP > _SPRING_BIAS_OUTDOOR_TEMP_FLOOR
```

(`_SPRING_BIAS_NIGHT_FRAC` — the old 0.15 threshold — is removed entirely; it has exactly one other use in the file, inside the guard block Task 3 deletes. The drift-cap constants are added here too, alongside Fix A's, even though Task 4 is what wires them in — keeps all class constants in one place.)

- [ ] **Step 2: Extend the candidate tuple**

Find the line (inside `_calibrate_tau`, in the `for s, e in zip(starts, ends):` loop, right after `quality = r2 * outdoor_temp_score * n_score * solar_score * hour_score`):
```python
                candidates.append((tau, quality, hour_start))
```
Replace with:
```python
                candidates.append((tau, quality, hour_start, sub_t_outdoor))
```
(`sub_t_outdoor` is already computed a few lines earlier in this same loop iteration, at `sub_t_outdoor = float(group["T_outdoor"].iloc[s:e_cap].mean())` — nothing else to add.)

- [ ] **Step 3: Run tests to confirm nothing broke yet**

Run: `python -m pytest tests/test_model.py -k TauCalibration -v`
Expected: all still PASS (the 4th tuple element and unused new constants are inert until Task 3).

---

### Task 3: Fix A — confidence-weighted blend replaces the binary guard + conditional EMA

**Files:**
- Modify: `apps/energy_forecast/model.py:1568-1596` (docstring — done in Task 6, skip for now) and `model.py:1749-1813` (the guard block, candidate selection, and EMA blend)
- Test: `tests/test_model.py` (`TestTauCalibrationSafeguards`)

**Interfaces:**
- Consumes: constants and 4-element candidate tuples from Task 2.
- Produces: `_calibrate_tau` now computes `confidence` (float, 0.0-1.0) and uses it to blend — consumed by Task 4 (drift cap applies to the same `tau_result`).

- [ ] **Step 1: Write the failing tests — new tests**

Add to `tests/test_model.py`, inside `class TestTauCalibrationSafeguards:` (after `test_top_50_percent_selected`, i.e. anywhere before line 5128 where the old guard-specific tests start — exact position doesn't matter within the class):

```python
    def test_production_gray_zone_incident_now_preserves(self, tmp_path):
        """Reproduces the literal incident conditions from the design spec §1 (night_frac~100%
        of a small selected set, outdoor_median~14°C — just above the old guard's 12°C trigger,
        which is exactly why the old guard never fired here). The rejected first-draft formula
        also got this case wrong (confidence=1, full trust) because a single good signal (night)
        could rescue trust. This is the regression test for the actual production bug.
        """
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        dfs, heat, wx = self._make_night_blocks(
            tau_true=15.0, n_days=6, start_hour=23, window_hours=3, t_out=14.0,
        )

        result = model._calibrate_tau(dfs, heat, wx)

        assert result == 10.0, "the exact production gray-zone condition must now be an exact preserve"

    def test_warm_mostly_night_window_not_fully_trusted(self, tmp_path):
        """Key counterexample the rejected first-draft formula got wrong: an all-nighttime but
        warm window (a textbook summer window-ventilation case) must NOT reach full confidence
        just because night_frac is good — both signals are required.
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
        tau_true, rather than a captured/narrated number for this specific fixture.
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

    def test_cold_daytime_also_damped_not_fully_trusted(self, tmp_path):
        """Deliberate behavior change from the original code (which fully trusted this case):
        the codebase's own quality-scoring docstring (model.py:1731) attributes daytime bias to
        ventilation, independent of temperature — confidence now correctly requires BOTH
        night_frac and outdoor_median to be good, so this case gets confidence=0, same as the
        "both bad" case, not a full update.
        """
        model = _make_tau_model(tmp_path)
        model._tau_hours = 25.0
        cold_day_dfs, cold_day_heat, cold_day_wx = self._make_night_blocks(
            tau_true=10.0, n_days=5, start_hour=10, t_out=5.0,
        )

        result = model._calibrate_tau(cold_day_dfs, cold_day_heat, cold_day_wx)

        assert result == 25.0, "night_frac=0% must zero out confidence regardless of temperature"
```

- [ ] **Step 2: Delete the two tests whose assertions the redesign intentionally inverts**

Delete `test_no_ema_on_small_change` (`tests/test_model.py:4819-4828`, right before `test_no_radiation_column_degrades_gracefully`) — it asserted the pre-fix behavior (small changes bypassed damping entirely), which is the exact bug being fixed; superseded by `test_both_signals_good_updates_normally` above.

Delete `test_spring_bias_guard_not_triggered_with_cold_outdoor` (`tests/test_model.py:5155-5168`, the last test in the class) — its assertion (`result != 25.0`) is the opposite of the new, deliberately-changed behavior; superseded by `test_cold_daytime_also_damped_not_fully_trusted` added in Step 1.

`test_spring_bias_guard_preserves_stored_tau` (`tests/test_model.py:5128-5142`) and
`test_spring_bias_guard_not_triggered_without_stored_tau` (`tests/test_model.py:5144-5153`) are
**left unchanged** — both still pass with the new code (verified in the spec's round-1 review
pass): the first is an exact-preserve case (`confidence=0` either way), the second never sets
`model._tau_hours` so it hits the untouched `old_tau is None` branch.

- [ ] **Step 3: Run the new/changed tests to verify they fail correctly**

Run: `python -m pytest tests/test_model.py -k "production_gray_zone_incident or warm_mostly_night or sparse_candidates_reduce or both_signals_good or cold_daytime_also_damped" -v`
Expected: FAIL — `test_production_gray_zone_incident_now_preserves` and
`test_warm_mostly_night_window_not_fully_trusted` fail because the current code's `outdoor_median`
is measured over the *entire off-period timeline* (not the selected candidates) and the guard's
threshold (0.15/12°C) doesn't match these fixtures' construction; `test_cold_daytime_also_damped_not_fully_trusted`
fails because the current guard requires *both* bad conditions to block, and this fixture has only
one. This confirms the tests exercise the pre-fix behavior.

- [ ] **Step 4: Replace the guard block, candidate selection, and EMA blend**

Current (`model.py`, starting at the line `old_tau = self._tau_hours` — roughly line 1756 in the
pre-Task-2 file, shifted slightly by Task 2's edits — search for this exact text):

```python
        # Spring-bias guard: if almost no nighttime candidates exist and outdoor temps
        # are warm, the estimates are dominated by open-window ventilation rather than
        # structural thermal mass.  Preserve the stored winter τ in that case.
        old_tau = self._tau_hours
        if old_tau is not None and old_tau > 0 and len(candidates) >= 3:
            night_frac = sum(1 for _, _, h in candidates if h >= 22 or h < 6) / len(candidates)
            outdoor_median = float(combined[combined["off"] == 1]["T_outdoor"].median())
            if night_frac < self._SPRING_BIAS_NIGHT_FRAC and outdoor_median > self._SPRING_BIAS_OUTDOOR_TEMP:
                _LOGGER.info(
                    "τ calibration skipped — spring/summer bias (%.0f%% nighttime candidates, "
                    "T_out median=%.1f°C); preserving stored τ=%.1f h.",
                    night_frac * 100,
                    outdoor_median,
                    old_tau,
                )
                return old_tau

        # With few candidates use top-50%; with many (≥ threshold) tighten to top-25%
        # to reduce bias from lower-quality windows in data-rich retrains.
        candidates.sort(key=lambda c: c[1], reverse=True)
        divisor = 4 if len(candidates) >= self._TAU_SELECTIVITY_THRESHOLD else 2
        n_select = max(1, len(candidates) // divisor)
        selected = candidates[:n_select]
        tau_estimates = [c[0] for c in selected]

        _LOGGER.debug(
            "τ calibration: %d candidates, using top %d (%d%%) (quality %.2f–%.2f, τ range %.1f–%.1f h)",
            len(candidates),
            n_select,
            round(100 * n_select / len(candidates)),
            selected[-1][1],
            selected[0][1],
            min(tau_estimates),
            max(tau_estimates),
        )

        tau_median = float(np.median(tau_estimates))

        if old_tau is not None and old_tau > 0:
            change_frac = abs(tau_median - old_tau) / old_tau
            if change_frac > 0.5:
                tau_result = 0.8 * old_tau + 0.2 * tau_median
                _LOGGER.info(
                    "τ EMA blend: %.1f h → %.1f h (raw %.1f h, Δ=%.0f%%)",
                    old_tau,
                    tau_result,
                    tau_median,
                    change_frac * 100,
                )
            else:
                tau_result = tau_median
        else:
            tau_result = tau_median
```

Replace the whole block above with:

```python
        old_tau = self._tau_hours

        # With few candidates use top-50%; with many (≥ threshold) tighten to top-25%
        # to reduce bias from lower-quality windows in data-rich retrains.
        candidates.sort(key=lambda c: c[1], reverse=True)
        divisor = 4 if len(candidates) >= self._TAU_SELECTIVITY_THRESHOLD else 2
        n_select = max(1, len(candidates) // divisor)
        selected = candidates[:n_select]
        tau_estimates = [c[0] for c in selected]

        _LOGGER.debug(
            "τ calibration: %d candidates, using top %d (%d%%) (quality %.2f–%.2f, τ range %.1f–%.1f h)",
            len(candidates),
            n_select,
            round(100 * n_select / len(candidates)),
            selected[-1][1],
            selected[0][1],
            min(tau_estimates),
            max(tau_estimates),
        )

        # Confidence: requires BOTH a sufficient nighttime fraction AND a cold-enough outdoor
        # median among the SELECTED (quality-filtered) candidates -- computed over the same
        # population tau_median comes from, not the raw candidate pool. A single good signal
        # must not be able to rescue full trust for a window where the other signal is bad
        # (e.g. a warm, all-night window is a textbook summer ventilation case, not safe data).
        night_frac = sum(1 for c in selected if c[2] >= 22 or c[2] < 6) / len(selected)
        outdoor_median = float(np.median([c[3] for c in selected]))

        night_conf = min(1.0, night_frac / self._SPRING_BIAS_NIGHT_FRAC_FULL)
        temp_conf = min(
            1.0,
            max(0.0, (self._SPRING_BIAS_OUTDOOR_TEMP - outdoor_median)
                     / (self._SPRING_BIAS_OUTDOOR_TEMP - self._SPRING_BIAS_OUTDOOR_TEMP_FLOOR)),
        )
        # len(candidates) (pre-selection pool size) -- sparse retrains should be trusted less.
        sample_conf = min(1.0, len(candidates) / self._TAU_SAMPLE_CONF_REF)

        # Geometric mean, not a plain product: a plain 3-way product over-penalizes moderately
        # -good conditions (three factors of 0.7 would multiply to 0.34); the geometric mean
        # preserves the same AND-semantics at the boundaries (any single factor at exactly 0
        # still forces confidence to exactly 0) while being much gentler in the interior.
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

- [ ] **Step 5: Run the new/changed tests to verify they now pass**

Run: `python -m pytest tests/test_model.py -k "production_gray_zone_incident or warm_mostly_night or sparse_candidates_reduce or both_signals_good or cold_daytime_also_damped" -v`
Expected: all 5 PASS.

- [ ] **Step 6: Run the full `TestTauCalibrationSafeguards` class**

Run: `python -m pytest tests/test_model.py -k TauCalibrationSafeguards -v`
Expected: all PASS, including the unaffected tests (`test_high_solar_penalized_not_excluded`,
`test_daytime_windows_penalized_not_excluded`, `test_single_window_sufficient`,
`test_no_radiation_column_degrades_gracefully`, `test_top_50_percent_selected`, and the two
tests kept unchanged from Step 2) and `test_ema_blend_on_large_change` (`model.py:4807-4817`,
also kept unchanged — verified in the spec: `confidence=1` at full-confidence conditions gives
the same `10 < result < 22` range the old `>50%`-jump blend did).

- [ ] **Step 7: Run the full test suite**

Run: `python -m pytest tests/ -v 2>&1 | tail -20`
Expected: no new failures vs. Task 1 Step 2's baseline.

- [ ] **Step 8: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "fix: replace binary spring-bias guard with continuous confidence weight

Requires both nighttime fraction and outdoor temperature to
independently look winter-like (geometric-mean composed with a
sample-size term) before trusting a fresh tau estimate, computed over
the same quality-filtered candidate population tau_median comes from.
Fixes the case where a single good signal (e.g. an all-night but warm
window) could fully rescue trust and reproduce the original drift bug."
```

---

### Task 4: Fix B — two-tier rolling drift cap

**Files:**
- Modify: `apps/energy_forecast/model.py:282` (add 4 new instance attributes in `__init__`)
- Modify: `apps/energy_forecast/model.py:1456-1473` (`_save()`'s `meta` dict)
- Modify: `apps/energy_forecast/model.py:1839-1854` (`_load()`'s meta restoration)
- Modify: `apps/energy_forecast/model.py` — end of the `if old_tau is not None and old_tau > 0:` branch added in Task 3
- Test: `tests/test_model.py` (new `TestTauDriftCap` class + one test in `TestTauCalibrationSafeguards`)

**Interfaces:**
- Consumes: `tau_result` computed by Task 3's blend, `old_tau`, `combined` (the existing DataFrame indexed by timestamp), `confidence` (only used for the `if old_tau is not None` guard, already established).
- Produces: `self._tau_anchor_hours`, `self._tau_anchor_ts`, `self._tau_long_anchor_hours`, `self._tau_long_anchor_ts` — persisted fields, not consumed by any other task in this plan.

- [ ] **Step 1: Write the failing tests — new `TestTauDriftCap` class**

Add to `tests/test_model.py`, immediately after `TestTauCalibrationSafeguards` ends (after line
5169's closing test, before the `# ── Auto-K Regime Selection ──` separator at line 5171):

```python
class TestTauDriftCap:
    """Fix B: bounds cumulative τ movement to ±_TAU_MAX_DRIFT_FRAC (short) / ±_TAU_LONG_MAX_DRIFT_FRAC
    (long) from the value at the start of each respective rolling window, independent of Fix A's
    per-step damping and independent of how many retrains occur inside the window.

    _make_night_blocks (tests/test_model.py:4749) hardcodes its dates to January 2026 -- all
    anchor timestamps below are set well before January 2026 accordingly.
    """

    def _seed_anchors(self, model, anchor_hours, anchor_ts):
        model._tau_anchor_hours = anchor_hours
        model._tau_anchor_ts = anchor_ts
        model._tau_long_anchor_hours = anchor_hours
        model._tau_long_anchor_ts = anchor_ts

    def test_clamps_when_cumulative_budget_exhausted(self, tmp_path):
        """A single retrain's per-step-legal blend must still be clamped if it would push τ
        beyond the rolling window's remaining budget. old_tau=6.0 sits below the window floor
        (anchor=10.0 * 0.65 = 6.5); a modest raw pull (tau_true=6.5) gets clamped up to exactly
        the floor. Anchor set 2025-12-15 -- ~19 days before the fixture's Jan 3-9 data, inside
        the 30-day short window."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 6.0
        self._seed_anchors(model, 10.0, pd.Timestamp("2025-12-15"))

        dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
            tau_true=6.5, start_hour=22, t_out=5.0, day_start=3,
        )
        result = model._calibrate_tau(dfs, heat, wx)

        assert result is not None
        assert result == pytest.approx(6.5), "must be clamped up to the 30-day drift floor (anchor * 0.65)"

    def test_resets_after_window_elapses(self, tmp_path):
        """Once _TAU_DRIFT_WINDOW_DAYS has passed since the anchor, a new anchor is set at the
        current stored τ -- this is what allows genuine multi-month seasonal correction. Anchor
        set 2025-10-01 -- roughly 97 days before the fixture's Jan 1-6 data, comfortably past
        the 30-day window."""
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
        """Both anchor timestamps must stay tz-naive, matching combined.index."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
            tau_true=11.0, start_hour=22, t_out=5.0,
        )
        model._calibrate_tau(dfs, heat, wx)

        assert model._tau_anchor_ts.tzinfo is None
        assert model._tau_long_anchor_ts.tzinfo is None

    def test_long_cap_bounds_multi_window_compounding(self, tmp_path):
        """The short (30-day) cap alone only bounds movement WITHIN one window, not ACROSS
        successive windows under sustained bias -- each window reset re-anchors at wherever the
        previous window left off. Simulates 6 successive short-window resets under a sustained,
        moderate (not confidence=0) pull and confirms the long (180-day) cap holds the line.

        Without the long cap, 6 windows of unfettered 35%-per-window compounding would allow
        10.0 * 0.65**6 =~ 0.75h. With the long cap, τ is held at exactly 10.0 * 0.5 = 5.0h.
        """
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0

        for i in range(6):
            dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
                tau_true=2.0, n_days=5, start_hour=22, window_hours=8, t_out=8.0, day_start=1,
            )
            if i > 0:
                model._tau_anchor_ts = model._tau_anchor_ts - pd.Timedelta(days=31)
            result = model._calibrate_tau(dfs, heat, wx)
            assert result is not None
            model._tau_hours = result

        assert model._tau_hours == pytest.approx(5.0), (
            "sustained bias across many short-window resets must still be caught by the "
            "180-day/50% long cap, not allowed to compound toward the naive ~0.75h"
        )
```

Also add to `TestTauCalibrationSafeguards` (this one's a Fix-A/Fix-B interaction check, placed
alongside the Task 3 tests):

```python
    def test_single_retrain_tau_move_bounded_by_drift_cap(self, tmp_path):
        """For an extreme, fully-trusted pull, the drift cap's ±35%-of-anchor bound is the
        *tighter* constraint (not the EMA weight's confidence*20%-of-gap alone), because the
        anchor initializes to the pre-update stored value on the very first blended retrain."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        dfs, heat, wx = self._make_night_blocks(
            tau_true=100.0, start_hour=22, t_out=5.0, window_hours=12,
        )

        result = model._calibrate_tau(dfs, heat, wx)

        assert result is not None
        assert result == pytest.approx(13.5), "drift cap (anchor=10.0 * 1.35) must be the binding constraint"
        assert abs(result - 10.0) <= 10.0 * 0.35 + 1e-6
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_model.py -k "TauDriftCap or single_retrain_tau_move_bounded" -v`
Expected: FAIL with `AttributeError: 'EnergyForecastModel' object has no attribute '_tau_anchor_hours'`.

- [ ] **Step 3: Add the four new instance attributes**

In `model.py`'s `__init__`, right after (`model.py:282`):
```python
        # Building thermal time constant (hours) — calibrated from passive-cooling windows
        self._tau_hours: float | None = None
```
Add:
```python
        # Rolling drift-cap anchors for _tau_hours (short: 30-day/35%, long: 180-day/50%)
        self._tau_anchor_hours: float | None = None
        self._tau_anchor_ts: pd.Timestamp | None = None
        self._tau_long_anchor_hours: float | None = None
        self._tau_long_anchor_ts: pd.Timestamp | None = None
```

- [ ] **Step 4: Persist the new fields in `_save()`**

In `model.py`'s `_save()` (`model.py:1456-1473`), the `meta` dict currently ends:
```python
            "tau_hours": self._tau_hours,
            "enable_regimes": self._enable_regimes,
            "regime_count": self._regime_count,
            "weather_tail": self._weather_tail,
        }
```
Replace with:
```python
            "tau_hours": self._tau_hours,
            "tau_anchor_hours": self._tau_anchor_hours,
            "tau_anchor_ts": self._tau_anchor_ts,
            "tau_long_anchor_hours": self._tau_long_anchor_hours,
            "tau_long_anchor_ts": self._tau_long_anchor_ts,
            "enable_regimes": self._enable_regimes,
            "regime_count": self._regime_count,
            "weather_tail": self._weather_tail,
        }
```

- [ ] **Step 5: Restore the new fields in `_load()`**

In `model.py`'s `_load()` (`model.py:1839-1854`), find:
```python
                    self._tau_hours = meta.get("tau_hours", None)
```
Add right after it:
```python
                    self._tau_hours = meta.get("tau_hours", None)
                    self._tau_anchor_hours = meta.get("tau_anchor_hours", None)
                    self._tau_anchor_ts = meta.get("tau_anchor_ts", None)
                    self._tau_long_anchor_hours = meta.get("tau_long_anchor_hours", None)
                    self._tau_long_anchor_ts = meta.get("tau_long_anchor_ts", None)
```

- [ ] **Step 6: Add the drift-cap logic after the Task-3 blend**

Find the code added in Task 3, Step 4's replacement — the `if old_tau is not None and old_tau > 0:` branch. Its body currently ends with:
```python
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
Insert the drift-cap application between the `if confidence < 0.05: / else:` block and the
`else: tau_result = tau_median` at the end (i.e. still inside `if old_tau is not None and old_tau > 0:`):
```python
            else:
                _LOGGER.info(
                    "τ EMA blend: %.1f h → %.1f h (raw %.1f h, new_weight=%.0f%%, confidence=%.0f%% "
                    "[night=%.0f%%, temp=%.0f%%, sample=%.0f%%])",
                    old_tau, tau_result, tau_median, new_weight * 100, confidence * 100,
                    night_conf * 100, temp_conf * 100, sample_conf * 100,
                )

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

            tau_result = _apply_drift_cap("_tau_anchor_hours", "_tau_anchor_ts",
                                           self._TAU_DRIFT_WINDOW_DAYS, self._TAU_MAX_DRIFT_FRAC, tau_result)
            tau_result = _apply_drift_cap("_tau_long_anchor_hours", "_tau_long_anchor_ts",
                                           self._TAU_LONG_DRIFT_WINDOW_DAYS, self._TAU_LONG_MAX_DRIFT_FRAC, tau_result)
        else:
            tau_result = tau_median
```

- [ ] **Step 7: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_model.py -k "TauDriftCap or single_retrain_tau_move_bounded" -v`
Expected: all 6 PASS.

- [ ] **Step 8: Run the full test suite**

Run: `python -m pytest tests/ -v 2>&1 | tail -20`
Expected: no new failures.

- [ ] **Step 9: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "fix: add two-tier rolling drift cap for tau calibration

Per-update damping alone only slows convergence to a biased estimate,
it doesn't bound it over many retrains. A 30-day/35% cap bounds
movement within a window; a 180-day/50% cap independently bounds
compounding across successive window resets under sustained bias."
```

---

### Task 5: Fix C — persist and correctly-localize the adaptive-retrain cooldown; gate the startup retrain

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:844` (insert new methods after `_rollback_model_cb`)
- Modify: `apps/energy_forecast/energy_forecast.py:344-360` (`initialize()` call sites)
- Modify: `apps/energy_forecast/energy_forecast.py:1410` (resync at end of `_retrain()`)
- Modify: `apps/energy_forecast/energy_forecast.py:47` area (new module-level constant)
- Test: `tests/test_energy_forecast.py`

**Interfaces:**
- Produces: `EnergyForecast._last_trained_local()`, `_seed_adaptive_cooldown()`, `_maybe_startup_retrain()`, module-level `STARTUP_RETRAIN_MIN_GAP_HOURS`. Not consumed by any other task in this plan.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_energy_forecast.py`, immediately after `TestAdaptiveRetrainLock` closes
(after line 2597, before the `# ── _subtract_sub_sensors ──` separator at line 2600):

```python
class TestSeedAdaptiveCooldown:
    """Fix C: the adaptive-retrain cooldown must be seeded from the persisted last_trained
    timestamp, converted to local time (not left UTC-naive), so a restart can't immediately
    re-arm it and the cooldown doesn't expire early near its boundary."""

    def test_seed_converts_utc_naive_last_trained_to_local(self):
        """last_trained is UTC-naive (datetime.now() in model.py) -- this uses the actual
        production form, not a local-tz value hand-built to look correct."""
        from datetime import datetime, timezone

        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeSelf()
        fake._timezone = "Europe/Zurich"
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        fake._ml_model.last_trained = utc_now

        EnergyForecast._seed_adaptive_cooldown(fake)

        expected_local = pd.Timestamp(utc_now).tz_localize("UTC").tz_convert("Europe/Zurich").tz_localize(None)
        assert fake._last_adaptive_retrain == expected_local
        # In summer (CEST, UTC+2) this must differ from the naive value by ~2h -- if it
        # doesn't, the conversion silently isn't happening.
        assert abs((fake._last_adaptive_retrain - pd.Timestamp(utc_now)).total_seconds()) > 3000

    def test_seed_with_never_trained_model_keeps_datetime_min(self):
        """A genuinely fresh install (no meta.pkl yet) must not be blocked from its first
        adaptive retrain -- last_trained defaults to datetime.min, same as before."""
        from datetime import datetime

        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeSelf()
        fake._ml_model.last_trained = datetime.min

        EnergyForecast._seed_adaptive_cooldown(fake)

        assert fake._last_adaptive_retrain == datetime.min


class TestMaybeStartupRetrain:
    """Fix C: the startup retrain (run_in(..., 10)) must be skipped if a retrain already
    completed within STARTUP_RETRAIN_MIN_GAP_HOURS."""

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
```

And append this method inside `class TestAdaptiveRetrainLock:` (after line 2597, before the
class closes):

```python
    def test_seeded_cooldown_prevents_immediate_refire_after_restart(self):
        """Regression test for the production bug: simulates a restart 2 hours after the last
        real retrain, using the actual UTC-naive form last_trained is produced in."""
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
assignment works with no fixture changes (verified in spec round-2 review).

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/test_energy_forecast.py -k "SeedAdaptiveCooldown or MaybeStartupRetrain or seeded_cooldown_prevents" -v`
Expected: FAIL with `AttributeError: type object 'EnergyForecast' has no attribute '_seed_adaptive_cooldown'` (and similarly for `_maybe_startup_retrain`).

- [ ] **Step 3: Add the module-level constant**

In `energy_forecast.py`, right after (`energy_forecast.py:47`):
```python
RETRAIN_INTERVAL_S = 168 * 3600  # weekly
```
Add:
```python
STARTUP_RETRAIN_MIN_GAP_HOURS = 6  # skip the unconditional startup retrain if one already
                                    # happened this recently (e.g. a crash-loop restart)
```

- [ ] **Step 4: Add the three new methods**

In `energy_forecast.py`, right after `_rollback_model_cb` ends (`energy_forecast.py:844`, right
before `def _get_scenario_cb`), insert:

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
        import pandas as pd

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

    def _maybe_startup_retrain(self, event_name=None, data=None, kwargs=None) -> None:
        """Startup retrain (run_in(..., 10)), skipped if a retrain already completed recently.

        Closes the other half of defect (A): previously *every* restart forced a full retrain
        regardless of how recently one had already happened, so a crash-loop or repeated manual
        restart could touch τ far more often than the intended weekly cadence. A restart shortly
        after a genuine code deploy will run on the previous model until the next
        scheduled/adaptive retrain — an acceptable trade-off given the model changes slowly
        week to week, and RELOAD_ENERGY_MODEL remains available for an explicit forced reload.
        """
        import pandas as pd

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

- [ ] **Step 5: Wire the new methods into `initialize()`**

In `energy_forecast.py`, find (`energy_forecast.py:344-348`):
```python
        model_dir = Path(__file__).parent / "models"
        self._ml_model = EnergyForecastModel(
            model_dir, model_archive_count=self._model_archive_count, timezone=self._timezone
        )
        self._lock = threading.Lock()
```
Replace with:
```python
        model_dir = Path(__file__).parent / "models"
        self._ml_model = EnergyForecastModel(
            model_dir, model_archive_count=self._model_archive_count, timezone=self._timezone
        )
        self._seed_adaptive_cooldown()
        self._lock = threading.Lock()
```

Then find (`energy_forecast.py:358`):
```python
        self.run_in(self._retrain_cb, 10)
```
Replace with:
```python
        self.run_in(self._maybe_startup_retrain, 10)
```

(The next line, `self.run_every(self._retrain_cb, f"now+{RETRAIN_INTERVAL_S + 10}", RETRAIN_INTERVAL_S)`,
and `self.listen_event(self._retrain_cb, "RELOAD_ENERGY_MODEL")` at line 350, are both left
unchanged — they must keep targeting `_retrain_cb` directly.)

- [ ] **Step 6: Add the resync at the end of `_retrain()`**

In `energy_forecast.py`, find (`energy_forecast.py:1410`):
```python
        _LOGGER.info("Retrained. MAE: %s", self._ml_model.last_mae)

        try:
            ha_data.fetch_energy_history_15m(
```
Replace with:
```python
        _LOGGER.info("Retrained. MAE: %s", self._ml_model.last_mae)
        self._last_adaptive_retrain = self._last_trained_local()

        try:
            ha_data.fetch_energy_history_15m(
```

- [ ] **Step 7: Run the new tests to verify they pass**

Run: `python -m pytest tests/test_energy_forecast.py -k "SeedAdaptiveCooldown or MaybeStartupRetrain or seeded_cooldown_prevents" -v`
Expected: all 5 PASS.

- [ ] **Step 8: Run the full test suite**

Run: `python -m pytest tests/ -v 2>&1 | tail -20`
Expected: no new failures.

- [ ] **Step 9: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast.py
git commit -m "fix: persist and correctly-localize the adaptive-retrain cooldown, gate startup retrain

_last_adaptive_retrain previously reset to datetime.min on every
restart (UTC-naive vs. the local-tz comparisons everywhere else),
letting a restart re-arm the 24h adaptive cooldown and, combined with
the unconditional startup retrain, fire two uncapped tau recalibrations
within ~2 minutes of every AppDaemon restart. Both are now gated by
the same correctly-localized last_trained timestamp, resynced at the
end of every retrain so the fix doesn't degrade to a one-shot startup
patch."
```

---

### Task 6: Fix D — docstring update

**Files:**
- Modify: `apps/energy_forecast/model.py:1568-1596` (`_calibrate_tau`'s docstring)

**Interfaces:** none — documentation only.

- [ ] **Step 1: Replace the docstring**

Current (`model.py:1568-1596`):
```python
    def _calibrate_tau(
        self,
        climate_dfs: dict[str, pd.DataFrame],
        heating_active_df: pd.DataFrame,  # cols: timestamp, heating_active (0/1)
        weather_df: pd.DataFrame,  # cols: timestamp, temp_c
    ) -> float | None:
        """Estimate building thermal time constant τ (hours) from passive-cooling windows.

        Fits log-linear OLS on ``ln(T_indoor − T_outdoor) = ln(ΔT₀) − t/τ`` for each
        contiguous sub-sequence where indoor is warmer than outdoor and cooling.

        Windows are scored rather than hard-filtered.  Each candidate receives a
        composite quality score (0–1):

        * ``r²``             — goodness of OLS log-linear fit (primary signal quality indicator;
                               candidates with r² ≤ 0 are dropped as physics failures)
        * ``outdoor_temp``   — ``exp(−max(T_out_mean − 10, 0) / 8)``; penalises warm-weather
                               windows where open-window ventilation shortens apparent τ
        * ``n score``        — length bonus, capped at 6 points
        * ``solar``          — ``exp(−max_radiation / 400)``; continuous penalty, no hard cut-off
        * ``hour``           — 1.0 nighttime (22–06), 0.7 shoulder (06–09, 16–22), 0.1 daytime

        A spring-bias guard is applied before selection: if fewer than 15 % of candidates
        are nighttime AND the heating-off outdoor median exceeds 12 °C, the stored τ is
        preserved (spring open-window data is not representative of building thermal mass).

        The top 50 % of candidates by quality (minimum 1) are used to compute the
        median τ, which is then EMA-smoothed against the stored value.
        """
```

Replace the docstring (keep the function signature) with:
```python
    def _calibrate_tau(
        self,
        climate_dfs: dict[str, pd.DataFrame],
        heating_active_df: pd.DataFrame,  # cols: timestamp, heating_active (0/1)
        weather_df: pd.DataFrame,  # cols: timestamp, temp_c
    ) -> float | None:
        """Estimate building thermal time constant τ (hours) from passive-cooling windows.

        Fits log-linear OLS on ``ln(T_indoor − T_outdoor) = ln(ΔT₀) − t/τ`` for each
        contiguous sub-sequence where indoor is warmer than outdoor and cooling.

        Windows are scored rather than hard-filtered.  Each candidate receives a
        composite quality score (0–1):

        * ``r²``             — goodness of OLS log-linear fit (primary signal quality indicator;
                               candidates with r² ≤ 0 are dropped as physics failures)
        * ``outdoor_temp``   — ``exp(−max(T_out_mean − 10, 0) / 8)``; penalises warm-weather
                               windows where open-window ventilation shortens apparent τ
        * ``n score``        — length bonus, capped at 6 points
        * ``solar``          — ``exp(−max_radiation / 400)``; continuous penalty, no hard cut-off
        * ``hour``           — 1.0 nighttime (22–06), 0.7 shoulder (06–09, 16–22), 0.1 daytime

        The top 50 % of candidates by quality (minimum 1) are used to compute the median τ.

        A continuous confidence weight — requiring both a sufficient nighttime fraction *and*
        a cold-enough outdoor median among the selected top-N candidates, plus enough total
        candidates — scales how much a fresh estimate can move the stored τ in a single retrain
        (0–20 %). A two-tier rolling cap independently bounds total movement to ±35 % over any
        30 days and ±50 % over any 180 days, regardless of retrain frequency. See
        `docs/superpowers/specs/2026-07-09-tau-calibration-drift-fix-design.md` §2.1–2.2 for the
        full design rationale.
        """
```

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest tests/ -v 2>&1 | tail -20`
Expected: no new failures (docstring-only change).

- [ ] **Step 3: Commit**

```bash
git add apps/energy_forecast/model.py
git commit -m "docs: update _calibrate_tau docstring for the confidence-weight/drift-cap design"
```

---

### Task 7: Lint and full verification

**Files:** none (verification only)

- [ ] **Step 1: Run ruff**

```bash
ruff check apps/energy_forecast/model.py apps/energy_forecast/energy_forecast.py tests/test_model.py tests/test_energy_forecast.py --fix
ruff format apps/energy_forecast/model.py apps/energy_forecast/energy_forecast.py tests/test_model.py tests/test_energy_forecast.py
```
Expected: zero violations after `--fix` (project's ruff config: `target-version = "py313"`, `line-length = 120`, `select = ["E", "F", "I", "UP"]`). If `--fix`/`format` change anything, review the diff before committing — auto-format shouldn't alter behavior, but confirm no unintended reflow of the nested-function/comment blocks added in Task 4.

- [ ] **Step 2: Run the full test suite one more time**

Run: `python -m pytest tests/ -v 2>&1 | tail -30`
Expected: all pass (same baseline as Task 1, Step 2, plus all new tests from Tasks 3-5).

- [ ] **Step 3: Commit if ruff changed anything**

```bash
git add -A
git diff --cached --quiet || git commit -m "style: ruff format"
```

---

### Task 8: Changelog

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Invoke the changelog-writer agent**

Per project CLAUDE.md workflow, use the `changelog-writer` subagent to add an entry summarizing
this fix (binary spring-bias guard → continuous confidence weight; new two-tier drift cap;
adaptive-retrain cooldown persistence + startup-retrain gate) rather than writing the entry by
hand — it knows the project's changelog format/version conventions.

- [ ] **Step 2: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog entry for tau calibration drift fix"
```

---

## Deployment (not part of this plan's tracked deliverables — requires separate explicit approval)

`scripts/` is gitignored in this repo, so the spec's §4.1 one-time seed procedure is not a file
this plan creates or commits. Once Tasks 1-8 are merged and ready to ship:

1. Deploy the code fix (`apps/energy_forecast/model.py`, `apps/energy_forecast/energy_forecast.py`)
   to the live HA instance via the existing Samba upload procedure (`memory/reference_samba_deploy.md`).
2. Run the spec's §4.1 seed procedure once, against the live `models/` directory, to seed
   `tau_hours` and all four anchor fields to `11.64` (derived in spec §4.1 from the deployed
   model's own April 2026 data — the last stable, high-confidence window before the spring
   transition biased later calibrations).
3. Restart AppDaemon and monitor via the spec's §4.2 diagnostic query for at least one week to
   confirm the expected behavior (a/b/c/d in §4.2).

Steps 1-3 touch the live production system and should be run explicitly, not automatically as
part of implementing this plan — flag back to the user before executing them.

---

## Self-Review

**Spec coverage:** §2.1 (Fix A) → Tasks 2-3. §2.2 (Fix B) → Task 4. §2.3 (Fix C) → Task 5. §2.4
(Fix D) → Task 6. §2.5 (downstream bound) → the `test_single_retrain_tau_move_bounded_by_drift_cap`
test in Task 4. §3.1-3.4 (all spec test code) → reproduced verbatim across Tasks 3-5. §4.1 (seed
procedure) → documented in "Deployment" above, explicitly excluded from tracked tasks since
`scripts/` is gitignored. §4.2 (post-deploy verification) → "Deployment" step 3. No gaps found.

**Placeholder scan:** no TBD/TODO/"add appropriate"/"similar to Task N" patterns — every step has
complete, spec-verbatim code.

**Type consistency:** `_calibrate_tau`'s signature is unchanged throughout (Tasks 2-4 only modify
its body). `_last_trained_local() -> datetime`, `_seed_adaptive_cooldown() -> None`,
`_maybe_startup_retrain(self, event_name=None, data=None, kwargs=None) -> None` in Task 5 match
the calling convention already established by `_retrain_cb`/`_rollback_model_cb` in the same file
(unbound-method-callable with the AppDaemon timer/event dual signature). The four new
`EnergyForecastModel` attributes (`_tau_anchor_hours`, `_tau_anchor_ts`, `_tau_long_anchor_hours`,
`_tau_long_anchor_ts`) are named identically across Task 4's `__init__`/`_save`/`_load`/blend-logic
edits and Task 4's tests.
