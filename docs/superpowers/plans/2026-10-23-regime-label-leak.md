# Regime Label Leak + Hard-Label Flip (#110) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log `tomorrow` regime stability in production, build the OOB-soft regime path behind a private switch, decide H/T/N with a backtest that is proven to reproduce the live model, then ship the winner.

**Architecture:** Four parts on three branches, executed at different times.
- **Part A** (`fix/regime-log`, now) adds a prediction-neutral JSONL log.
- **Part B** (`fix/regime-soft-labels`) refactors both `regime_kwh` builders onto one probability-matrix helper and adds `regime_mode` (`hard` | `soft_oob`) persisted in `meta.pkl`.
- **Part C** (same branch, scripts only) drives the real `_retrain()` through a stub app to get the live training frame, gates on reproducing the live clusterer and model, then backtests.
- **Part D** ships per the decision rule.

**Tech Stack:** Python 3.13, pandas, scikit-learn (KMeans, RandomForest OOB), LightGBM, AppDaemon 4, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-regime-label-leak-design.md` (rev. 3, approved 2026-10-23). Section refs (§) below point there.

## Global Constraints

- Tests: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`. Run the full suite after every task, never bare `python`.
- All branches are cut from `dev`; never commit to `dev` or `main` directly. Commit subject is one line, ending with the session's Co-Authored-By line.
- `regime_mode` values are exactly `"hard"` and `"soft_oob"`. The meta key is `regime_mode`; a missing key means `"hard"`. It is not exposed in `apps.yaml`.
- `k` is always `clusterer.centroids.shape[0]`, never `_regime_count`.
- The log file is `regime_log.jsonl`, next to `pred_history.json` (`const.PRED_HISTORY_PATH.parent`). It is append-only with no trim. Line schema, exact keys: `{"ts", "last_trained", "target_date", "tomorrow_kwh", "label", "proba"}`.
- Part A must not change any predicted value. The golden test (Task A1) is the proof and must pass unchanged through Parts A and B.
- **Deploy rules (§3, §4, §7):**
  - Part A deploys only within 6 h after a retrain (`STARTUP_RETRAIN_MIN_GAP_HOURS = 6`), or after the C verdict closes.
  - Part D obeys ROADMAP "Cold-onset sequencing": after A+B (#101/#102) are deployed and their re-baseline holds, and never in a window with A+B or #105.
- **Decision numbers (§6.2, §6.4, §7), verbatim:**
  - Reproduction gate: same k; ≥ 90 % label agreement; harness vs live daily predictions within 3 %.
  - Candidate passes vs H: strictly lower max |Δ|, count of days > 2 kWh ≤ H's, pooled daily MAE ≤ 1.03 × H, primary-fold MAE ≤ 1.05 × H.
  - T unless N clearly wins: if both pass, N only if ≥ 3 % better pooled, or ≥ 5 % better in the January fold. If only one passes, ship that one. If none passes, keep H.
  - Post-deploy check: the |Δ| > 2 kWh share goes down (hard criterion); daily MAE ≤ 1.10 × pre-deploy.

## Review Focus

1. **Model trained without regimes** (arm N, or `enable_regimes: false`, or clustering failed): the log line is still written, with `label`/`proba` = `null`, and nothing raises. Test in Task A3.
2. **Forecast frame doesn't reach tomorrow's date** (short SRG frame), or the regime classifier returns `-1`: the log line is written with `label` = `null`, not a stale value from the previous update. Test in Task A3.
3. **`target_date` around midnight and on DST days:** `target_date` is the local calendar date of the `tomorrow` window used by `_aggregate`, not a UTC date. Test in Task A3, with a 2026-10-24 23:30 local clock (the night before the 10-25 fall-back).
4. **Old pickles under new code:** a `meta.pkl` without `regime_mode` and a `RegimePredictor` pickled before Part B both give the hard path and predictions identical to before. Test in Task B3.
5. **Clusterer k > classes the RF saw** (a cluster with no feature-days): the probability columns for the missing class are 0, and the soft series stays finite. Tests in Tasks A2 and B1.

---

## Part A — Stability log (`fix/regime-log`)

### Task A1: Golden test pinning current regime/prediction output

**Files:**
- Test: `tests/test_model.py` (new class `TestRegimeGolden`)
- Create: `tests/fixtures/regime_golden.npz`

**Interfaces:**
- Produces: `TestRegimeGolden._train_and_prepare(tmp_path, monkeypatch) -> tuple[EnergyForecastModel, pd.DataFrame, pd.DataFrame]`, returning `(model, forecast_df, X)`. Tasks A3 and B1–B3 reuse it.

- [ ] **Step 1: Write the test.**
  - Reuse `TestAutoKRegimeSelection()._make_regime_train_data(n=24*60, n_regimes=3)`.
  - `train(..., weight_halflife_days=0, enable_regimes=True, regime_count=3)`.
  - Freeze the clock: `monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: pd.Timestamp("2026-10-23 17:00", tz=tz)))` (pattern at `tests/test_model.py:2864`). `forecast_df` = 48 h from that hour, built like the end-to-end test at ~6370, but with `temp_c` alternating between two levels per day so `regime_kwh` varies.
  - `test_golden_regime_kwh_and_predictions`:
    - `"regime_kwh" in model.feature_cols`;
    - `(X["regime_kwh"] > 0).any()`;
    - `np.testing.assert_array_equal(X["regime_kwh"].to_numpy(), golden["regime_kwh"])`;
    - via `predict(forecast_df, live_temp=10.0)` and via `predict(..., _prepared=model._prepare_prediction_X(forecast_df, 10.0, None))`: `np.testing.assert_allclose(pred, golden["pred"], rtol=0, atol=1e-9)` for both.
  - If `REGEN_REGIME_GOLDEN=1` is set, write the npz and `pytest.skip`.
- [ ] **Step 2: Generate the fixture on unmodified code.** Run `REGEN_REGIME_GOLDEN=1 <env python> -m pytest tests/test_model.py::TestRegimeGolden -v`. Expected: SKIPPED, and the npz exists.
- [ ] **Step 3: Run without the env var.** Expected: PASS. Run it 3× to confirm determinism; if LightGBM jitters, pin `num_threads=1` in the test-only train call, never in product code.
- [ ] **Step 4: Commit** the test and npz as the branch's **first commit**: `test: golden regime_kwh/prediction fixture (#110)`.

### Task A2: `RegimePredictor.predict_proba`

**Files:**
- Modify: `apps/energy_forecast/clustering.py` (`RegimePredictor`, after `predict` ~line 214)
- Test: `tests/test_clustering.py`

**Interfaces:**
- Produces: `RegimePredictor.predict_proba(self, daily_features: pd.DataFrame, n_clusters: int) -> pd.DataFrame | None`. Index = `daily_features.index`; columns = `range(n_clusters)`; rows sum to 1. Classes absent from `self.model.classes_` are all 0. Returns `None` if unfitted or on any exception (logged at warning).

- [ ] **Step 1: Write the tests.**
  - `test_predict_proba_maps_classes_to_columns`: fit on labels `{0, 2}` only, call with `n_clusters=3`. Assert `list(df.columns) == [0, 1, 2]`, `(df[1] == 0).all()`, `np.allclose(df.sum(axis=1), 1)`.
  - `test_predict_proba_unfitted_returns_none`.
  - `test_predict_proba_argmax_matches_predict`: `df.idxmax(axis=1).to_numpy()` equals `predict(...)`.
- [ ] **Step 2: Run them.** Expected: FAIL (AttributeError).
- [ ] **Step 3: Implement.** Use `self.model.predict_proba(X)` and place column `j` at `int(self.model.classes_[j])`.
- [ ] **Step 4: Run them.** Expected: PASS, and the full suite is green.
- [ ] **Step 5: Commit:** `feat: RegimePredictor.predict_proba (#110)`.

### Task A3: Capture regime info in `_prepare_prediction_X` and write the log

**Files:**
- Modify: `apps/energy_forecast/model.py` (`__init__` ~line 312; regime block `1176–1216`)
- Modify: `apps/energy_forecast/const.py` (next to `PRED_HISTORY_PATH`, line 26)
- Modify: `apps/energy_forecast/energy_forecast.py` (after the `_prepared` call ~line 2068; after `aggregated` is built ~line 2254; new module-level helper near the other `_`-prefixed helpers)
- Modify: `scripts/pull_ha_data.py` (`FIXED_FILES`, line 33)
- Test: `tests/test_model.py`, `tests/test_energy_forecast.py`

**Interfaces:**
- Consumes: `RegimePredictor.predict_proba` (A2).
- Produces:
  - `EnergyForecastModel._last_regime_info: dict[datetime.date, dict]`. Each value is `{"label": int, "proba": list[float]}`, with `proba` rounded to 3 decimals and of length k.
  - `const.REGIME_LOG_PATH = PRED_HISTORY_PATH.parent / "regime_log.jsonl"`.
  - `energy_forecast._append_regime_log(path: Path, record: dict) -> None`, which never raises.

- [ ] **Step 1: Write the model tests** in `TestRegimeGolden`, using A1's helper.
  - `test_last_regime_info_filled`: both forecast dates present; `len(proba) == model._clusterer.centroids.shape[0]`; `label == argmax(proba)`.
  - `test_regime_info_failure_is_isolated`: monkeypatch `RegimePredictor.predict_proba` to raise. `X["regime_kwh"]` equals golden, and `model._last_regime_info == {}`.
  - `test_regime_info_reset_each_call`: a second `_prepare_prediction_X` with a 12 h `forecast_df` (today only) drops tomorrow's key (Review Focus 2).
- [ ] **Step 2: Write the app tests** in `tests/test_energy_forecast.py`.
  - `test_append_regime_log_appends_json_line`: two calls give two lines, each parseable with exactly the 6 schema keys.
  - `test_append_regime_log_swallows_oserror`: path in a nonexistent read-only dir → no raise, warning logged.
  - `test_regime_log_record_without_regimes`: `_last_regime_info == {}` gives `label`/`proba` = `None` (Review Focus 1).
  - `test_regime_log_target_date_local`: clock 2026-10-24 23:30 Europe/Zurich gives `target_date == "2026-10-25"` (Review Focus 3).

  Factor the record construction into `_build_regime_log_record(ts, last_trained, tomorrow_date, tomorrow_kwh, regime_info) -> dict`, which is pure and tested directly, so the app tests don't need `_update_sensors`.
- [ ] **Step 3: Run them.** Expected: FAIL.
- [ ] **Step 4: Implement the model side.**
  - `self._last_regime_info = {}` in `__init__` and as the first statement of `_prepare_prediction_X`.
  - In the regime block, after `predicted_labels`: a **nested** `try/except Exception` that only `_LOGGER.warning`s. It calls `predict_proba(daily_features, centroids.shape[0])` and fills the dict, skipping dates whose label is `-1`. The outer block is otherwise untouched.
- [ ] **Step 5: Implement the app side.**
  - Right after the `_prepared = ...` call: `_regime_info = dict(self._ml_model._last_regime_info)`.
  - After `aggregated = self._aggregate(...)`: build the record. `ts` = now local, ISO. `last_trained` = `self._ml_model.last_trained` as ISO or `None`. `target_date` = the same `tomorrow` local date `_aggregate` uses. `tomorrow_kwh` = `aggregated["tomorrow"]`, rounded to 3. Then `_append_regime_log(REGIME_LOG_PATH, record)`.
  - Add `(f"{FORECAST_REMOTE}/regime_log.jsonl", "regime_log.jsonl")` to `FIXED_FILES`.
- [ ] **Step 6: Run** the new tests, the golden test and the full suite. Expected: all PASS.
- [ ] **Step 7: Commit:** `feat: regime_log.jsonl — tomorrow regime label/proba per update (#110)`.

### Task A4: Docs and finalize Part A

- [ ] **Step 1:** CHANGELOG entry via `@changelog-writer`: "regime stability log, prediction-neutral". No README change, since nothing in HA changes.
- [ ] **Step 2: Check the deploy window.** `RELOAD`/retrain time from the live `meta.pkl` `last_trained` (via `pull_ha_data.py`) must be < 6 h ago, or the C verdict must be closed. If neither holds, **stop and tell the user** the next eligible window. Don't deploy outside it.
- [ ] **Step 3:** Finalize via `@deploy-agent` Workflow A (merge → dev, next `v0.12.0-alpha-N` tag, deploy, push, cleanup).
- [ ] **Step 4: Verify.**
  - On the 6 h-window path, the AppDaemon log shows "Startup retrain skipped".
  - After ≥ 2 hourly updates, `pull_ha_data.py` brings down `data/regime_log.jsonl` with ≥ 2 lines and a non-null `label`.
- [ ] **Step 5:** Update `memory/project_regime_flip_weather_source.md`: Part A live, plus the date the log started (that date bounds the §7 baseline).

---

## Part B — Soft path, default off (`fix/regime-soft-labels`, cut from `dev` after Part A merges)

### Task B1: One `regime_kwh` builder for training and predict

**Files:**
- Modify: `apps/energy_forecast/clustering.py` (new module function)
- Modify: `apps/energy_forecast/model.py` (`623–639` training; `1200–1216` predict)
- Test: `tests/test_clustering.py`

**Interfaces:**
- Produces: `clustering.regime_kwh_from_proba(proba: pd.DataFrame, timestamps: pd.Series, centroids: np.ndarray) -> np.ndarray`.
  - Input `proba`: index = `datetime.date`, columns `0..k-1`.
  - It reindexes to `timestamps.dt.date` and forward-fills. Leading NaN rows become 0, and explicit all-zero rows stay 0 (they are not NaN, so the forward-fill leaves them alone).
  - Returns `(P_hourly * centroids[:, hours].T).sum(axis=1)`, one value per timestamp.
- Produces: `clustering.one_hot(labels: pd.Series, k: int) -> pd.DataFrame`. Labels `< 0` give an all-NaN row, which keeps today's ffill-over-`-1` behaviour (`model.py:1206`).

- [ ] **Step 1: Write the tests.**
  - `test_one_hot_matches_centroid_lookup`: `regime_kwh_from_proba(one_hot(labels, k), ts, C)` equals `C[labels[date(t)], hour(t)]`.
  - `test_soft_is_weighted_sum`: proba `[0.25, 0.75]` gives `0.25*C[0,h] + 0.75*C[1,h]`.
  - `test_ffill_and_leading_zero`: a gap date takes the previous day's row; a date before the first row gives 0.
  - `test_explicit_zero_row_not_ffilled`.
  - `test_missing_class_column_zero_finite` (Review Focus 5).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement both functions.** Replace the bodies of both builders in `model.py` with `regime_kwh_from_proba(one_hot(labels, centroids.shape[0]), ts, centroids)`. Training labels come from the clusterer; predict labels from the RF.
- [ ] **Step 4: Run** the new tests and **the A1 golden test, which must pass unchanged**, then the full suite.
- [ ] **Step 5: Commit:** `refactor: single regime_kwh builder from a probability matrix (#110)`.

### Task B2: OOB probabilities from `RegimePredictor.fit`

**Files:**
- Modify: `apps/energy_forecast/clustering.py` (`RegimePredictor.fit`, ~138–212)
- Test: `tests/test_clustering.py`

**Interfaces:**
- Produces: `RegimePredictor.fit(...) -> pd.DataFrame | None`. It now **returns** the OOB matrix, with index = `common_idx`, columns `range(n_clusters)` and the same `classes_` mapping as A2. NaN rows (never out-of-bag) become one-hot of `y` at that date. Return `None` on the early `< 14` return or an exception. Add the keyword `n_clusters: int | None = None`; `None` means `max(labels) + 1`. Nothing new is stored on `self`, so the pickle format is unchanged.

- [ ] **Step 1: Write the tests.**
  - `test_fit_returns_oob_matrix_aligned`: index equals the intersection of features and labels; rows sum to 1.
  - `test_fit_oob_nan_row_falls_back_to_one_hot`: monkeypatch `oob_decision_function_` to include a NaN row.
  - `test_fit_short_returns_none`.
  - `test_fit_does_not_add_attributes`: `vars(rp)` keys are unchanged.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS, and the full suite is green.
- [ ] **Step 3: Commit:** `feat: RegimePredictor.fit returns OOB class probabilities (#110)`.

### Task B3: `regime_mode` on `train()`, persisted, read at predict

**Files:**
- Modify: `apps/energy_forecast/model.py`:
  - `train()` signature (~327–360);
  - training block (~583–639);
  - predict block (~1196–1216);
  - `__init__` (add `self._regime_mode = "hard"`);
  - `_save` meta dict (~1721);
  - `_load` (~2188).
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: B1 `regime_kwh_from_proba`, `one_hot`; B2 `fit` return value; A2 `predict_proba`.
- Produces: `EnergyForecastModel.train(..., regime_mode: str = "hard")`, with `ValueError` for any other value, and `self._regime_mode` saved as meta `regime_mode`.
  - **Training, `soft_oob`:** `P` = the fit's OOB matrix, reindexed to all clusterer-label dates. Dates with a clusterer label but no features get explicit 0 rows, and their count is logged at info. `self._regime_mode = "soft_oob"` is set **only** if that series was built; otherwise `"hard"` (fit returned `None`, or regimes disabled).
  - **Predict:** if `self._regime_mode == "soft_oob"`, `P` = `predict_proba(...)`; otherwise `one_hot(predicted_labels)`. If `predict_proba` returns `None`, fall back to one-hot.
  - `_load`: `self._regime_mode = meta.get("regime_mode", "hard")`.

- [ ] **Step 1: Write the tests.**
  - `test_default_mode_is_hard_and_golden`: the A1 golden test still passes, and meta has `regime_mode == "hard"`.
  - `test_soft_oob_training_matches_hand_computed`: for 3 sampled training hours, `regime_kwh == Σ P_oob[date,k]·C[k,hour]`. Check against the series passed into `_engineer_features` by spying on it, or recompute from the returned feature frame.
  - `test_soft_oob_no_feature_days_zero`: drop 2 dates from weather so they lack daily features; their training `regime_kwh` is 0.
  - `test_soft_oob_fit_failure_writes_hard`: monkeypatch `RegimePredictor.fit` to return `None`, then `model._regime_mode == "hard"`.
  - `test_meta_roundtrip_and_missing_key`: save, then reload gives `"soft_oob"`. Deleting the key from `meta.pkl` and reloading gives `"hard"`, and predictions equal the hard path (Review Focus 4).
  - `test_soft_predict_uses_proba`: for a `soft_oob` model, X `regime_kwh` for tomorrow equals `Σ p·C` from `predict_proba`.
  - `test_invalid_mode_raises`.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS, and the full suite is green.
- [ ] **Step 3: Commit:** `feat: regime_mode soft_oob behind private train() switch (#110)`. **Do not merge or deploy** this branch (§5).

---

## Part C — Reproduce, then backtest (scripts on `fix/regime-soft-labels`; earliest run ≈ 2026-11-05)

**Codebase for every Part C run (§6.1):** dev + A+B. If `fix/counter-reset-increments` and `fix/ev-sensor-trust` aren't on `dev` yet, run from a local scratch branch that merges both into `fix/regime-soft-labels`. **Never push the scratch branch.** Record which commit the run used in the results JSON.

### Task C1: Live-equivalent loader

**Files:**
- Create: `scripts/regime_harness.py`
- Test: `tests/test_regime_harness.py` (importlib idiom as in `tests/test_check_ha_status.py:24`)

**Interfaces:**
- Produces:
  - `trim_caches(src: Path, dst: Path, cutoff: pd.Timestamp) -> None`. It copies every `*.csv`/`*.json` from `src` to `dst`. For CSVs with a `timestamp` column it keeps rows `≤ cutoff`. Other files are copied unchanged.
  - `load_live_equivalent_training_frame(cutoff: pd.Timestamp, data_dir: Path, scratch: Path) -> dict`, returning the exact kwargs plus the positional args (as `"_args"`) that `_retrain()` passed to `model.train()`.
- Approach (§6.1):
  - Build the app via `EnergyForecast.__new__(EnergyForecast)`. Set every `self._*` attribute `_retrain` reads, with values from `data/apps.yaml` (start from `_FakeRetrain`, `tests/test_energy_forecast.py:4024`, and grep `_retrain` for the rest). Point `_cache_path` at `scratch/energy_history.csv` so every derived cache path lands in `scratch`.
  - `get_history` → `[]`.
  - Patch `energy_forecast.energy_forecast.weather.fetch_historical_weather` and `fetch_open_meteo` to read and write `scratch/../weather_archive.csv`, filled once from the Open-Meteo archive API up to `cutoff`.
  - Patch `pd.Timestamp.now` and `date.today` to `cutoff`.
  - Stub `heating_season.save_thresholds`, `ha_data.fetch_energy_history_15m`, and `self._ml_model` with an object whose `train(*a, **kw)` records its arguments and whose `last_trained` = `datetime.min`.

- [ ] **Step 1: Write the tests.**
  - `test_trim_caches_drops_future_rows`.
  - `test_trim_caches_copies_non_timestamp_files`.
  - `test_loader_captures_train_kwargs`: tiny `data_dir` fixture (100 h energy CSV, minimal apps.yaml, weather patched to an in-memory frame). Captured kwargs include `enable_regimes=True` and the energy frame's max timestamp ≤ cutoff.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS.
- [ ] **Step 3: Smoke-run** against the real `data/`, with the cutoff set to the live `meta.pkl` `last_trained` (`data/models/meta.pkl` after `pull_ha_data.py`). Expected: returns without error. Print the energy rows, date span and kwargs keys.
- [ ] **Step 4: Commit:** `feat: regime_harness live-equivalent training loader (#110)`.

### Task C2: Reproduction gate

**Files:**
- Create: `scripts/backtest_regime.py` (subcommand `gate`)
- Test: `tests/test_regime_harness.py`

**Interfaces:**
- Consumes: C1 loader; the live `data/models/{clusterer,regime_model,energy_model,meta}.pkl` from `pull_ha_data.py` (it already pulls `models/*.pkl`).
- Produces:
  - `label_agreement(live_centroids, harness_centroids, live_labels, harness_labels) -> float`. Both ID sets are sorted by centroid day-sum, then the share of equal labels is returned.
  - `gate(results: dict) -> tuple[bool, list[str]]`, applying: same k, agreement ≥ 0.90, and max |harness − live| daily prediction / live ≤ 0.03.

- [ ] **Step 1: Write the tests.**
  - `test_label_agreement_permutation_invariant`.
  - `test_gate_thresholds`: 0.90 passes, 0.899 fails; 3.0 % passes, 3.01 % fails; a k mismatch fails.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS.
- [ ] **Step 3: Run the gate.**
  - Train a harness model with `regime_mode="hard"` on the C1 frame at the live cutoff.
  - Compare against the live pickles. Label agreement is computed on the training days; predictions use both models' `_prepare_prediction_X` on the same stored inputs from the last 7 days.
  - Print the sorted centroid day-sums for both (they explain the 35/64/10.6 vs 7.9/37/20.9 mismatch).
  - **If the gate fails:** check auto-K, and whether the live clusterer is older than `meta.last_trained`. Then **stop and report to the user**; don't run C3.
- [ ] **Step 4: Commit** the script and record the gate output in `memory/project_regime_flip_weather_source.md`.

### Task C3: Backtest and decision

**Files:**
- Modify: `scripts/backtest_regime.py` (subcommand `backtest`)
- Test: `tests/test_regime_harness.py`

**Interfaces:**
- Produces: `decide(r: dict) -> str` returning `"T"`, `"N"` or `"H"`. Input `r[arm]` = `{"pooled_mae", "primary_mae", "jan_mae", "max_delta", "n_over_2"}` for `arm in ("H", "T", "N")`. It implements the §6.4 rules exactly as in Global Constraints.
- Protocol (§6.3):
  - Arms: H = `regime_mode="hard"`; T = `"soft_oob"`; N = `enable_regimes=False`.
  - Folds: primary = 2026-10-15 → last complete day (assert ≥ 21 days); guard = 2026-01-05 → 2026-01-31.
  - Expanding window: C1 loader at each weekly cutoff, then `train`.
  - Origin 17:00 local, scoring the next calendar day's total against `pred_history.json` actuals, falling back to the corrected target frame for days not in `pred_history`.
  - Sensitivity: re-predict each origin with `sunshine_min` ×0.66 and ×1.5, giving max |Δ daily total| and the count of days with |Δ| > 2 kWh.
  - Also print per-fold MAE and the `regime_kwh` gain share.

- [ ] **Step 1: Write the tests.** One per rule:
  - `test_decide_t_when_both_pass_and_n_not_3pct_better`;
  - `test_decide_n_when_3pct_pooled`;
  - `test_decide_n_when_5pct_january`;
  - `test_decide_only_passing_candidate`;
  - `test_decide_h_when_none_pass`;
  - `test_sensitivity_requires_strictly_lower_max`: equal max fails;
  - `test_count_over_2_allows_equal`: H = 0 and T = 0 passes.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS.
- [ ] **Step 3: Run the backtest** (≥ 2026-11-05). Save `data/backtest_regime_results.json` and print the per-fold table and `decide()` output.
- [ ] **Step 4: Record the results.** Write the results and the decision into the memory file and ROADMAP #110. **Show the user the table and decision before starting Part D.**
- [ ] **Step 5: Commit:** `feat: regime backtest H/T/N + decision rule (#110)`.

---

## Part D — Ship (only after the user has seen C3, and the sequencing gate holds)

### Task D1: Stability check script

**Files:**
- Create: `scripts/check_regime_stability.py`
- Test: `tests/test_regime_harness.py`

**Interfaces:**
- Produces: `delta_share(lines: list[dict], threshold: float = 2.0) -> float`. Δ is taken between consecutive lines with the same `target_date` **and** the same `last_trained`; the share is `|Δ| > threshold` over those pairs.
- CLI: `--deploy <ISO>` prints the pre and post share over 14 days each. The pre window starts no earlier than the A+B re-baseline date; if it's shorter, print its length. It also prints daily MAE pre/post from `pred_history.json` and PASS/FAIL against the §7 criteria.

- [ ] **Step 1: Write the tests.**
  - `test_delta_share_skips_rollover`;
  - `test_delta_share_skips_retrain_boundary`;
  - `test_delta_share_basic`.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS.
- [ ] **Step 3: Commit:** `feat: regime stability check script (#110)`.

### Task D2a (decision T): ship soft_oob

- [ ] **Step 1: Change the test first.** Change the `train()` default to `regime_mode="soft_oob"`. Write `test_default_mode_is_soft_oob` before the change and run it red. Then regenerate the A1 golden npz with `REGEN_REGIME_GOLDEN=1` and commit it **with an explicit note that the prediction change is intended**.
- [ ] **Step 2: Run the full suite.** Expected: green. Commit: `feat: soft OOB regime labels by default (#110)`.
- [ ] **Step 3: Confirm the sequencing gate.** Read ROADMAP "Cold-onset sequencing" and confirm A+B are deployed and re-baselined, and that no #105 deploy shares the window. If not, **stop and tell the user**.
- [ ] **Step 4: Finalize** via `@deploy-agent` Workflow A. Deploy only. The startup retrain applies it (the last retrain is > 6 h old); fire `RELOAD_ENERGY_MODEL` only if the log shows "Startup retrain skipped". Verify the live `meta.pkl` `regime_mode == "soft_oob"`.

### Task D2b (decision N): disable regimes live

- [ ] **Step 1:** On a new branch `chore/regime-backtest-scripts`, cherry-pick the Part C/D1 script commits (`scripts/regime_harness.py`, `scripts/backtest_regime.py`, `scripts/check_regime_stability.py` and their tests). Also cherry-pick B1/B2 only if the scripts import them; otherwise leave `fix/regime-soft-labels` unmerged. Finalize via `@deploy-agent`.
- [ ] **Step 2: Disable regimes on the live instance.** Set `enable_regimes: false` in the **live** `apps.yaml` via Samba, with a backup copy, then restart AppDaemon. The shipped default stays unchanged. Verify the live `meta.pkl` `enable_regimes == False` after the startup retrain.

### Task D3: Post-deploy check and docs (deploy + 14 days)

- [ ] **Step 1:** Run `pull_ha_data.py`, then `check_regime_stability.py --deploy <ts>`. Expected: PASS on both §7 criteria. On FAIL, report to the user. Rollback = revert the merge on `dev`, redeploy, and let it retrain; for N, restore the `apps.yaml` backup.
- [ ] **Step 2: Update the docs.**
  - CHANGELOG via `@changelog-writer`.
  - ROADMAP: #110 closed, and #83 unblocked with the new `regime_kwh` rank.
  - Memory: retire `project_regime_flip_weather_source` or mark it resolved.
