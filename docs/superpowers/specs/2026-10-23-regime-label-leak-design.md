# Regime Label: Train/Serve Leak + Hard-Label Flip — Design

**Date:** 2026-10-23 (rev. 2: after multi-stakeholder review)
**Status:** Draft, awaiting approval
**Branch base:** `dev` (all branches below)
**Roadmap item:** #110 (unblocks #83)
**Findings so far:** `memory/project_regime_flip_weather_source.md`

---

## 1. Intent

`regime_kwh` is the model's #1 feature, but its value is partly an artefact:

- **Leak.** Training fills `regime_kwh` from the clusterer's *true* daily labels (`model.py:604–639`). Live prediction uses the RandomForest's *predicted* label (`model.py:1176–1216`, inside `_prepare_prediction_X`). In harness models `regime_kwh` holds 31–33 % of gain; with out-of-bag (OOB) soft training labels it drops to #10–12 (2–2.5 %).
- **Hard-label flip.** The label is an argmax, so a small change in daily sunshine near a regime boundary swaps the whole 24 h centroid. On 2026-10-21 an SRG-SSR 503 → Open-Meteo fallback flipped tomorrow's label 0 → 2 (7.9 → 20.9 kWh/day) and moved `tomorrow` 11.06 → 16.64 → 11.06 kWh. ha-energy-manager schedules off the tomorrow 3 h block sensors, which flip the same way.
- **2026-10-23 evidence.** The 10-22 actual was 15.82 kWh. Regime 2 (16.6) was right and the live regime-0 forecast (10.7) was wrong, so on cold days the hard label can land on the wrong side, not only flicker.

## 2. Goals, non-goals, anti-goals

**Goals:**
1. Measure `tomorrow` stability in production, before and after the fix.
2. Decide between H (current), T (OOB-soft train + soft predict) and N (regimes off) with a backtest that runs the shipping code on a frame proven to match the live model.
3. Ship the winner without confounding the cold-onset sequencing.

**Non-goals:** changing KMeans, auto-K or the regime feature set; SRG/OM sunshine alignment; label locking; the 8 °C / 168 h temperature gate; HA-visible regime attributes; #83.

**Anti-goals:**
- trading cold-day accuracy for stability;
- a forecast-changing deploy in the same window as A+B (#101/#102) or #105;
- deciding on a harness that doesn't reproduce the live model.

## 3. Constraints

- **Live config:** `enable_regimes: true`, `regime_count: 0` (auto-K, currently k = 3), `baseline_mode: true`, `mqtt_discovery: true`.
- KMeans cluster IDs are arbitrary per retrain, so anything comparing labels across time must key on `last_trained`.
- **Every AppDaemon restart retrains** (`_maybe_startup_retrain`, `energy_forecast.py:445`) unless the last retrain was < 6 h ago (`STARTUP_RETRAIN_MIN_GAP_HOURS`). A restart also resets the weekly retrain timer. A "code-neutral" deploy is therefore forecast-neutral only inside that 6 h window.
- **Rollback:** `rollback_model()` restores only the newest of 3 archive snapshots, and the next retrain overwrites it. The real rollback is to revert the merge on `dev`, redeploy and let it retrain.
- `data/energy_history.csv` `gross_kwh` is not the training target. Use `pred_history.json` actuals for daily checks.
- `data/` history starts 2025-10-10, so last season's cold onset can't be a fold.
- Tests run through `/home/jovyan/my_envs/ha-energy-forecast/bin/python`, full suite after every change.

---

## 4. Step 1: Stability log (prediction-neutral)

**Branch:** `fix/regime-log`. **Effort:** ~1–2 h.

- **`RegimePredictor.predict_proba(daily_features, n_clusters)`** returns a DataFrame with rows = dates and columns `0..n_clusters-1`, mapped from `classes_`; absent classes get 0. It returns `None` if unfitted or on error.
- **`_prepare_prediction_X`** resets `self._last_regime_info = {}` at entry. After the hard labels are computed, it fills, per forecast date:
  - `label`;
  - `proba` (rounded to 3);
  - `soft_day_kwh` = Σ p_k · sum(centroid_k), taken before the temperature gate. This gives a free live shadow of the soft forecast.

  The fill sits in its **own nested `try/except`** that only logs. A failure there must never reach the outer `except`, which zeroes `regime_kwh`. The hard path feeding `feat_df["regime_kwh"]` stays unchanged.
- **App:** after the app's own `_prepare_prediction_X` call (`energy_forecast.py:2068`, before any SHAP or scenario call that would overwrite the info), each sensor update appends one line to `regime_log.jsonl`, written next to `pred_history.json` in the app directory:

  ```
  {ts, last_trained, tomorrow_kwh, tomorrow: {label, proba, soft_day_kwh}}
  ```

  The log is append-only with no trim (≈ 3 MB/year). A write failure logs a warning and carries on.
- **`scripts/pull_ha_data.py`:** add one `FIXED_FILES` entry for the log.
- **Tests:**
  - A **golden test, committed first, before any `model.py` change.** It uses `enable_regimes=True`, a fixed k and a frozen clock. It asserts that `regime_kwh` is in `feature_cols` and non-zero for at least one hour after the gate, and runs `assert_array_equal` on `X["regime_kwh"]` and the predictions, via both `predict()` and `_prepared=`.
  - A fault-injection test: when `predict_proba` raises, `X["regime_kwh"]` is unchanged and the info dict is empty.
  - `predict_proba` mapping with a missing class.
  - Log append and write-failure tolerance.
- **Deploy:** only within 6 h after a scheduled retrain, or after the C verdict closes. Acceptance: the AppDaemon log shows "Startup retrain skipped", and the log file grows by one line per update.

## 5. Step 2: Soft path as shipping code, default off

**Branch:** `fix/regime-soft-labels`. **Effort:** ~1 day. Not merged or deployed until Step 4.

- **Private `regime_mode` keyword on `train()`** with values `"hard"` (default), `"soft_predict"` (P, diagnostic only) and `"soft_oob"` (T). It is not exposed in config. Step 3 runs this exact code.
- **One helper used in both places** (training `model.py:623–639`, predict `model.py:1196–1216`):
  - Input: a per-date probability matrix (dates × k), where `k = centroids.shape[0]`, not `_regime_count`.
  - Hard mode uses one-hot rows; soft modes use probabilities.
  - It reindexes to the hourly dates and forward-fills (keeping today's partial-day semantics). Leading gaps get 0.
  - It returns `Σ_k P[date(t), k] · centroid_k[hour(t)]`.
- **Training probabilities for T:** `RegimePredictor.fit` keeps the OOB matrix (`oob_decision_function_`, rows = `common_idx`) in memory for this retrain only; nothing new is pickled.
  - NaN rows (never out-of-bag, ≈ 0 at 100 trees) fall back to the one-hot label.
  - Days with a clusterer label but no daily features get `regime_kwh = 0`, not the hard label (which would re-introduce the leak), and their count is logged.
- **Versioning:** `meta.pkl` gets `regime_mode`, written only when the soft series was actually built. Missing key → `"hard"`.
- **Tests:**
  - the helper on a toy 2-cluster fixture (hard vs soft vs edge cases);
  - the OOB mapping;
  - meta read and write, including the fallback when fit fails;
  - the Step 1 golden test still passes with the default.

## 6. Step 3: Reproduce, then backtest

**Script:** `scripts/backtest_regime.py` (on the Step 2 branch). **Effort:** ~1 day. **Run date:** once ≥ 21 complete days exist from 2026-10-15, so about 2026-11-05.

### 6.1 Live-equivalent loader
`load_live_equivalent_training_frame(cutoff)` drives the real `EnergyForecast._retrain()` through a stub app (pattern: `tests/test_energy_forecast.py:4025`). Its `model.train()` captures the arguments instead of training (user decision 2026-10-23). Setup:
- copy `data/` to a scratch directory and point every cache path there, because the fetches write back to their caches;
- stub `get_history` to return `[]`, so `is_away` = 0, since `input_boolean.abwesend` history has no cache (accepted difference);
- serve weather from the Open-Meteo archive, cached locally;
- freeze the clock at `cutoff`;
- stub `heating_season.save_thresholds` and the post-train 15 min fetch.

The code runs on dev + A+B (#101/#102), using a local, never-pushed scratch merge if A+B aren't on dev yet.

### 6.2 Reproduction gate (blocks the backtest)
At the live `meta.pkl` cutoff, with the live `clusterer.pkl` and model fetched via Samba (as in `scripts/analyze_regimes.py`):
- same k;
- ≥ 90 % daily label agreement with the live clusterer (after sorting IDs by centroid day-sum);
- harness-model vs live-model daily predictions within 3 % on the same stored inputs.

If it fails, check auto-K and whether the live clusterer is stale, then **stop and report** to the user. The harness-vs-live mismatch (35/64/10.6 vs 7.9/37/20.9 kWh/day) must be explained here.

### 6.3 Backtest
- **Arms:** H, T and N (`enable_regimes: false`) decide. P is a report-only column.
- **Folds:**
  - primary: 2026-10-15 → last complete day (≥ 21 days);
  - guard: a 3–4 week January 2026 window with ≥ 60 days of history before it.
- **Protocol:** expanding window, weekly retrain, archive weather as a perfect forecast. Origin 17:00, scoring the next calendar day (what EM reads).
- **Metrics:** daily-total MAE, primary and pooled across both folds. Source-switch sensitivity: re-predict each origin with `sunshine_min` ×0.66 and ×1.5, and report max |Δ daily total| and the count of days with |Δ| > 2 kWh. The `regime_kwh` gain share is report only.

### 6.4 Decision
1. A candidate (T or N) must beat H on sensitivity (lower max |Δ| and fewer days > 2 kWh). Its pooled daily MAE must be ≤ 1.02 × H, and its primary-fold MAE ≤ 1.05 × H.
2. **T unless N clearly wins** (user decision 2026-10-23). N is chosen only if it passes (1) and beats T by ≥ 3 % on pooled daily MAE, or is clearly better in the January fold.
3. If neither passes, keep H, record the numbers, and #110 stays open.

Results go into the memory file and ROADMAP #110 before Step 4.

## 7. Step 4: Ship

- **T:** change the default `regime_mode` to `"soft_oob"` on `fix/regime-soft-labels`.
- **N:** set `enable_regimes: false` in the live `apps.yaml` only. The shipped default is unchanged (user decision 2026-10-23), and the Step 2 branch is left unmerged.
- **Rollout:**
  - It is forecast-changing, so it follows Cold-onset sequencing (`ROADMAP.md`): after A+B are deployed and their re-baseline holds, never in a window with A+B or #105.
  - Deploy via `@deploy-agent` Workflow A. The startup retrain applies it; fire `RELOAD_ENERGY_MODEL` only if the log shows "Startup retrain skipped".
  - Rollback: revert the merge, redeploy and retrain (§3). A code revert with a soft-trained model on disk also needs a retrain.
- **Post-deploy check (14 days), comparing against the 14 days before deploy** (that window starts after the A+B re-baseline; if it's shorter, use what exists and say so):
  - hard criterion: the share of hourly `tomorrow` updates with |Δ| > 2 kWh, from `regime_log.jsonl`, goes down;
  - sanity floor: daily MAE ≤ 1.10 × pre-deploy, from `pred_history.json`.
- **Docs:** CHANGELOG (`@changelog-writer`), ROADMAP (#110 closed, #83 unblocked with the new feature rank), MEMORY.

---

## 8. Accepted risks
- The OOB bootstrap isn't temporal, so T keeps a small residual leak. The 31 % → 2 % gain drop shows it's small.
- Perfect archive weather favours H, so T or N roughly equal to H on MAE reads as "not worse live".
- Sunshine scaling is only a relative proxy for real source switches; it doesn't move zero-sun days.
- The harness has `is_away` = 0 throughout.
- 21 days can't resolve 1–2 % MAE differences. Sensitivity plus the user's T-default make the call.

**Calls made without explicit sign-off** are marked inline: P as a diagnostic arm, the numeric gates (6.2, 6.4, 7), zero `regime_kwh` on no-feature days, and the log fields.
