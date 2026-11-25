# Cold-Bias Candidate Arms (walk-forward kit) Implementation Plan

OWNER-APPROVED docs/superpowers/plans/2026-11-25-cold-bias-candidate-arms.md 2026-11-25 (native execution)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the tracked walk-forward kit can run three cold-bias candidates and score them with one pre-registered verdict, on days after 2026-11-25 only.

**Architecture:** `WfArm` gains a `patch` (context manager around `train()`) and an `energy_transform` (rewrites the training target per cutoff). `backtest/regime_arms.py` holds the regime patches and the target rescale. `wf_score.candidate_verdict()` applies the criteria below. No file under `apps/` changes; nothing is deployed.

**Tech Stack:** Python 3.13, pandas, numpy, LightGBM, scikit-learn, pytest.

**Spec:** `memory/project_overforecast_diagnosis_2026_11_25.md` (findings, numbers) and the owner decision of 2026-11-25 (arms approved in outline; west facade insulated with 160 mm).

## Global Constraints

- Base `dev`, branch `feat/cold-bias-arms`, worktree `.worktrees/cold-bias-arms`. Tooling only, risk tag NONE, v1.0-safe.
- Tests: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest` (`$PY`). Real captures: `/home/jovyan/my_envs/ha-energy-forecast-live/bin/python`, `PYTHONPATH=apps:.`.
- Arms, fixed now: `shapecl` (KMeans on profiles divided by their daily mean, mean floored at 0.1 kWh/h); `clrecent` (centroids and k from days >= 2026-03-01, all days still assigned); `rescale` (training target before 2026-10-01 scaled; insulation finished during September 2026). Dates are not tuned later.
- Rescale rule: `base` = median daily kWh of days with mean temp >= 15 °C. Heat excess `x = max(0, kWh - base)`, `hdd = max(0, 15 - T)`. `r = (sum x_new / sum hdd_new) / (sum x_old / sum hdd_old)` using days with T < 15 and T inside the 10th..90th percentile of the new heating days, where new = days from 2026-10-01 to the cutoff. Clip r to [0.5, 1.0]. EV days are excluded from the fit, not from the scaling. Each old heating day is multiplied by `(base + r*x) / kWh` when `x > 0`; hours keep their shape. Fewer than 14 new or 14 old fit days: `r = 1.0` (no change).
- Criteria, fixed now (arm vs `base`, target days >= 2026-11-26):
  - INCONCLUSIVE below 14 scored days or 5 days <= 6 °C.
  - `|bias|` on days <= 6 °C at least 30 % lower than base.
  - `|bias|` on days 6-10 °C at most base + 0.5 kWh/day (no such day: INCONCLUSIVE).
  - `mae3` no more than 2 % worse than base.
  - `next_day_swing` range at most 10 % above base.
  - PASS = all; a PASS only makes the arm a candidate for its own plan. Three arms are scored on one window, so the winner still needs a second, later window before any default changes. Days <= 0 °C are not covered by the criteria.
- The #120 `droptp` arms keep `shadow_verdict`; both runs use the same window and `jit`.

## Review Focus

1. `clrecent` must leave every old day labelled (no `regime_kwh = 0` rows from missing labels). Task 1 test.
2. A patch leaking out of `train()` would alter the next arm in the same process. Task 1 test.
3. `rescale` fit must not see rows at or after the cutoff. Task 2 test.
4. Too few new heating days must be a no-op, not a crash or an extreme ratio. Task 2 test.
5. `r > 1` evidence (colder house than last winter) must clip to 1.0, never inflate old load. Task 2 test.

## File Structure

- Create `backtest/regime_arms.py` — `shape_only()`, `centroids_since(since)`, `rescale_target(...)`.
- Modify `backtest/walkforward.py` — `WfArm` fields, `run_arm`, three `ARMS` entries.
- Modify `backtest/wf_score.py` — `candidate_verdict`.
- Modify `backtest/README.md`, `CHANGELOG.md`.
- Tests: create `tests/test_backtest_regime_arms.py`; modify `tests/test_backtest_wf_score.py`, `tests/test_backtest_capture_frame.py`.

### Task 1: regime patches and `WfArm.patch`

**Interfaces — Produces:** `shape_only() -> ContextManager`; `centroids_since(since: pd.Timestamp) -> ContextManager`; both rebind `DailyProfileClusterer.fit` and `clustering.find_optimal_k` and restore them on exit. `WfArm.patch: Callable[[], ContextManager] | None = None`; `run_arm` passes it to `train()` only and not to `predict()` (`drop` keeps today's behaviour at both).

- [ ] **Step 1: failing tests**
  - `test_shape_only_centroids_have_unit_mean`: two synthetic day groups, same shape, levels 10 and 40 kWh → every centroid mean is 1.0 ± 1e-6.
  - `test_centroids_since_labels_all_days`: 60 old + 60 new days → returned labels index equals all 120 dates; centroids equal a fit on the new days only.
  - `test_centroids_since_k_uses_recent_days`: `find_optimal_k` receives no row before `since`.
  - `test_patches_restore_on_exit_and_on_error`: `fit` and `find_optimal_k` are the originals after a normal exit and after an exception inside the block.
  - `test_run_arm_patch_active_in_train_not_predict` (use the existing synthetic frame in `tests/test_backtest_capture_frame.py`).
- [ ] **Step 2:** run them → FAIL (module missing).
- [ ] **Step 3:** implement; `centroids_since` adds the old dates to `ev_day_dates` so they are excluded from the fit but still assigned.
- [ ] **Step 4:** run them → PASS, then the full suite → 0 failed.
- [ ] **Step 5:** commit `feat(backtest): regime arm patches`.

### Task 2: target rescale

**Interfaces — Produces:** `rescale_target(train_kwargs: dict, before: pd.Timestamp, min_days: int = 14) -> tuple[pd.DataFrame, float]` (new `energy_df`, ratio). Uses `energy_df`, `weather_df`, `ev_df` from kwargs. `WfArm.energy_transform: Callable[[dict], pd.DataFrame] | None`; `run_arm` applies it to `frame.train_kwargs(cutoff)` and passes the result as `train_kwargs["energy_df"]`.

- [ ] **Step 1: failing tests**
  - `test_ratio_recovers_known_factor`: new heating load 0.6 × old, same temperatures → `r` = 0.6 ± 0.02; old heating-day totals equal `base + 0.6*x` ± 1e-6; hours keep their shape; rows >= `before` and days with T >= 15 unchanged.
  - `test_ratio_clips`: factor 1.4 → 1.0 and an unchanged frame; factor 0.2 → 0.5.
  - `test_too_few_days_is_noop`: 13 new heating days → r = 1.0, frame equal to the input.
  - `test_ev_days_excluded_from_fit`: an EV day with +30 kWh does not move `r`.
  - `test_only_rows_before_cutoff`: through `run_arm` with a recording transform, `energy_df.timestamp.max() < cutoff`.
  - `test_input_not_mutated`.
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS, full suite. **Step 5:** commit `feat(backtest): last-winter target rescale arm`.

### Task 3: arms, verdict, docs

**Interfaces — Produces:** `ARMS["shapecl"|"clrecent"|"rescale"]`; `candidate_verdict(table: pd.DataFrame, swing: pd.DataFrame, daily_temp: pd.Series, arm: str, ref: str = "base") -> dict` with keys `days, n_le6, bias6_pct, bias610_delta, mae3_pct, swing_pct, checks, verdict`. `table` is a `day_table`; rows are used as given (the caller filters dates).

- [ ] **Step 1: failing tests**
  - `test_candidate_verdict_pass` / `_fail_each_criterion` (parametrised, one criterion broken at a time, thresholds exactly at 30 %, +0.5, 2 %, 10 % pass).
  - `test_candidate_verdict_inconclusive`: 13 days; 4 days <= 6 °C; no 6-10 °C day; arm missing.
  - `test_arm_registry`: the three names exist, carry the dates above, and `base`/`jit`/`droptp` are unchanged.
- [ ] **Step 2:** FAIL. **Step 3:** implement; README section "Cold-bias candidates" with the run command, the window `--start 2026-11-19` (burn-in) and scoring from `2026-11-26`; correct the #120 section's "after 2026-11-12" to "after 2026-11-25". **Step 4:** PASS, full suite. CHANGELOG via the `/changelog` skill. **Step 5:** commit `feat(backtest): cold-bias candidate arms and verdict`.

### Task 4: parity check on real captures (no merge gate on accuracy)

- [ ] Run `base shapecl clrecent rescale` for 2026-11-01..11-24 with the live env; `shapecl` and `clrecent` p50 must match the pickles in `data/backtest/overforecast_2026_11_25/wf/` within 1e-6 (same capture: pass `--capture-id 20261124T212830744257Z`). Record `rescale`'s ratio per cutoff and its seen-data result as information only. A mismatch is a finding: stop and report.

## After this plan

Merge to `train/next` on an owner click. **2026-12-10 or later:** pull `data/`, run `base jit droptp droptp_nb shapecl clrecent rescale`, report both verdicts. No app change follows without a separate plan.
