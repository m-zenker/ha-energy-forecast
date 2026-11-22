# Regime Label Leak + Hard-Label Flip (#110) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log `tomorrow` regime stability in production, build the out-of-fold soft regime path behind a private switch, decide H/T/N with a backtest proven to reproduce the live model, then ship the winner.

**Architecture:** Four parts on three branches, executed at different times.
- **Part A** (`fix/regime-log`, now) adds a prediction-neutral JSONL log.
- **Part B** (`fix/regime-soft-labels`) moves both `regime_kwh` builders onto one probability-matrix helper and adds `regime_mode` (`hard` | `soft_oof`), persisted in `meta.pkl`.
- **Part C** (same branch, scripts only) drives the real `_retrain()` through a stub app to get the live training frame, gates on reproducing the live clusterer and model, then backtests.
- **Part D** ships per the decision rule.

**Tech Stack:** Python 3.13, pandas, scikit-learn 1.9 (KMeans, RandomForest, `cross_val_predict`), LightGBM, AppDaemon 4, pytest.

**Spec:** `docs/superpowers/specs/2026-10-23-regime-label-leak-design.md` (rev. 4, approved 2026-10-23; out-of-fold change 2026-10-24). Section refs (§) below point there.

**Revision note:** rev. 2 (2026-10-24) applies the 21-item independent plan review. That review also changed the spec's T probabilities from OOB to 5-fold out-of-fold, a user decision.

## Global Constraints

- Tests: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`. Run the full suite after every task, never bare `python`.
- All branches are cut from `dev`, except Part B, which is cut from `fix/regime-log` if Part A hasn't merged yet and rebased later. Never commit to `dev` or `main` directly. Commit subject is one line, ending with the session's Co-Authored-By line.
- `regime_mode` values are exactly `"hard"` and `"soft_oof"`. The meta key is `regime_mode`; a missing key means `"hard"`. It is not exposed in `apps.yaml`.
- `k` is always `clusterer.centroids.shape[0]`, never `_regime_count`. Pass it explicitly wherever a function takes `n_clusters`.
- Regime-info dict keys are `datetime.date` (that is what `_prepare_daily_regime_features(...).index` holds). Values are JSON-native: `int(label)` and `[float(round(p, 3)), ...]`.
- **Times:** `meta.last_trained` is naive **local** (Europe/Zurich) on this install, verified 2026-10-24: the live value matches the local-time AppDaemon "Retrained" log line. Caches and `pred_history.json` are naive local too. Compare them directly, with **no** UTC conversion. (Corrected from rev. 2, which assumed UTC.)
- The log file is `regime_log.jsonl`, next to `pred_history.json` (`const.PRED_HISTORY_PATH.parent`). It is append-only with no trim. Line schema, exact keys: `{"ts", "last_trained", "target_date", "tomorrow_kwh", "label", "proba"}`.
- Part A must not change any predicted value. The golden test (Task A1) is the proof and must pass unchanged through Parts A and B.
- **Deploy rules (§3, §4, §7):**
  - Part A deploys only if the last retrain is < 5 h old **at the start** of the deploy (a 1 h margin on the 6 h gate), or after the C verdict closes.
  - Every Part D deploy, **T or N**, obeys ROADMAP "Cold-onset sequencing": after A+B (#101/#102) are deployed and their re-baseline holds, and never in a window with A+B or #105.
- **Decision numbers (§6.2, §6.4, §7), verbatim:**
  - Reproduction gate: same k; ≥ 90 % label agreement; harness vs live daily predictions within 3 %.
  - Candidate passes vs H: strictly lower max |Δ|, count of days > 2 kWh ≤ H's, pooled daily MAE ≤ 1.03 × H, primary-fold MAE ≤ 1.05 × H.
  - T unless N clearly wins: if both pass, N only if ≥ 3 % better pooled, or ≥ 5 % better in the January fold. If only one passes, ship that one. If none passes, keep H.
  - Post-deploy check: the |Δ| > 2 kWh share goes down (hard criterion); daily MAE ≤ 1.10 × pre-deploy.

## Review Focus

1. **Regimes disabled while stale pickles are on disk** (arm N live: `_load` reloads `clusterer.pkl`/`regime_model.pkl` regardless of `enable_regimes`, `model.py:2195–2209`): the log line has `label`/`proba` = `null`. The info fill is gated on `self._enable_regimes`. Test: model-level, in Task A3.
2. **Forecast frame doesn't reach tomorrow's date, or the label is `-1`:** the log line has `label` = `null`, not a stale value. Test in Task A3.
3. **`target_date` on the night before the DST fall-back** (2026-10-24 23:30 local): `_tomorrow_local_date` gives 2026-10-25, using the same rule as `_aggregate` (`energy_forecast.py:3014–3017`). Test in Task A3.
4. **Old pickles under new code:** a `meta.pkl` without `regime_mode` (re-hashed so it actually loads), and a `RegimePredictor` pickled before Part B, both give the hard path and predictions equal to the golden fixture. Test in Task B3.
5. **Clusterer k > classes the RF saw, or a stale label ≥ k:** missing-class columns are 0, labels ≥ k are clipped to k − 1 (today's `np.clip`), and the series stays finite. Tests in Tasks A2 and B1.

---

## Part A — Stability log (`fix/regime-log`)

### Task A1: Golden test pinning current regime/prediction output

**Files:**
- Test: `tests/test_model.py` (new class `TestRegimeGolden`)
- Create: `tests/fixtures/regime_golden.npz`

**Interfaces:**
- Produces: `TestRegimeGolden._train_and_prepare(tmp_path, monkeypatch, forecast_hours: int = 48) -> tuple[EnergyForecastModel, pd.DataFrame, pd.DataFrame]`, returning `(model, forecast_df, X)`. Tasks A3 and B1–B3 reuse it.

- [ ] **Step 1: Write the test.**
  - Reuse `TestAutoKRegimeSelection()._make_regime_train_data(n=24*60, n_regimes=3)`.
  - `train(..., weight_halflife_days=0, enable_regimes=True, regime_count=3)`.
  - Freeze the clock: `monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: pd.Timestamp("2026-10-23 17:00", tz=tz)))` (pattern at `tests/test_model.py:2864`). `forecast_df` = `forecast_hours` h from that hour, built like the end-to-end test at ~6370, with `temp_c` alternating between two levels per day. The reviewer verified this gives mean `regime_kwh` 0.645 and labels `[2, 0, 2]`.
  - `test_golden_regime_kwh_and_predictions`:
    - `(X["regime_kwh"] > 0).any()`. This is the real guard; `"regime_kwh" in feature_cols` is always true via `_FEATURES_BASE`, so don't assert it.
    - `np.testing.assert_array_equal(X["regime_kwh"].to_numpy(), golden["regime_kwh"])`.
    - via `predict(forecast_df, live_temp=10.0)` and via `predict(..., _prepared=model._prepare_prediction_X(forecast_df, 10.0, None))`: `np.testing.assert_allclose(pred, golden["pred"], rtol=0, atol=1e-9)` for both.
  - If `REGEN_REGIME_GOLDEN=1` is set, write the npz and `pytest.skip`.
- [ ] **Step 2: Generate the fixture on unmodified code.** `REGEN_REGIME_GOLDEN=1 <env python> -m pytest tests/test_model.py::TestRegimeGolden -v`. Expected: SKIPPED, and the npz exists.
- [ ] **Step 3: Run without the env var**, 3×. Expected: PASS each time. If LightGBM jitters, monkeypatch the LightGBM params construction in `model.py` (grep `LGBMRegressor(` / `params`) to add `num_threads=1` inside the test only.
- [ ] **Step 4: Commit** as the branch's **first commit**: `test: golden regime_kwh/prediction fixture (#110)`.

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
- Modify: `apps/energy_forecast/energy_forecast.py`:
  - after the `_prepared` call (~line 2068);
  - after `aggregated` is built (~line 2254);
  - new module-level helpers near the other `_`-prefixed helpers.
- Modify: `scripts/pull_ha_data.py` (`FIXED_FILES`, line 33)
- Test: `tests/test_model.py`, `tests/test_energy_forecast.py`

**Interfaces:**
- Consumes: `RegimePredictor.predict_proba` (A2).
- Produces:
  - `EnergyForecastModel._last_regime_info: dict[datetime.date, dict]`. Each value is `{"label": int, "proba": list[float]}`, JSON-native per Global Constraints.
  - `const.REGIME_LOG_PATH = PRED_HISTORY_PATH.parent / "regime_log.jsonl"`.
  - `energy_forecast._tomorrow_local_date(now_local: pd.Timestamp) -> datetime.date`: `(now_local.normalize() + 1 day).date()`, the same rule as `_aggregate`.
  - `energy_forecast._build_regime_log_record(ts: str, last_trained: datetime | None, tomorrow_date: datetime.date, tomorrow_kwh: float, regime_info: dict) -> dict`. It is pure. `last_trained` is `None` or `datetime.min`, giving `None`; otherwise local ISO per Global Constraints. A missing date gives `label`/`proba` = `None`.
  - `energy_forecast._append_regime_log(path: Path, record: dict) -> None`, which never raises.

- [ ] **Step 1: Write the model tests** in `TestRegimeGolden`, using A1's helper.
  - `test_last_regime_info_filled`:
    - keys are exactly the 3 dates 2026-10-23/24/25, of type `datetime.date`;
    - `len(proba) == model._clusterer.centroids.shape[0]`;
    - `label` equals `model._regime_model.predict(...)` for that date (not the argmax of the rounded proba, which can tie).
  - `test_regime_info_failure_is_isolated`: monkeypatch `RegimePredictor.predict_proba` to raise. `X["regime_kwh"]` equals golden, and `model._last_regime_info == {}`.
  - `test_regime_info_reset_each_call`: a second call with `forecast_hours=7` (17:00–23:00, 10-23 only) drops the 10-24 and 10-25 keys (Review Focus 2).
  - `test_regime_info_empty_when_regimes_disabled`:
    - train with regimes on and save;
    - in the same `tmp_path`, train a fresh model with `enable_regimes=False` (stale pickles stay on disk), then `_prepare_prediction_X`;
    - `_last_regime_info == {}` (Review Focus 1).
- [ ] **Step 2: Write the app tests** in `tests/test_energy_forecast.py`.
  - `test_tomorrow_local_date_dst_eve`: 2026-10-24 23:30 gives `date(2026, 10, 25)`; 2026-10-25 00:10 gives `date(2026, 10, 26)` (Review Focus 3).
  - `test_build_record_schema_and_nulls`: exactly the 6 keys. A missing date gives `label`/`proba` = `None`. `datetime.min` gives `last_trained` = `None`.
  - `test_record_from_real_model_is_json_serializable`: `json.dumps(_build_regime_log_record(..., regime_info=model._last_regime_info))` succeeds on the A1 model. This catches numpy types, which `_append_regime_log` would otherwise swallow, silently dropping every line.
  - `test_append_regime_log_appends_json_line`: two calls give two parseable lines.
  - `test_append_regime_log_swallows_oserror`: monkeypatch `Path.open` (or `builtins.open`) to raise `OSError`. No raise, and a warning is logged.
- [ ] **Step 3: Run them.** Expected: FAIL.
- [ ] **Step 4: Implement the model side.**
  - `self._last_regime_info = {}` in `__init__` and as the first statement of `_prepare_prediction_X`.
  - In the regime block, after `predicted_labels`, if `self._enable_regimes`: a **nested** `try/except Exception` that only `_LOGGER.warning`s. It calls `predict_proba(daily_features, self._clusterer.centroids.shape[0])` once and fills the dict, skipping labels `< 0`, with values cast to `int`/`float`. The outer block is otherwise untouched.
- [ ] **Step 5: Implement the app side.**
  - Right after `_prepared = ...`: `_regime_info = dict(self._ml_model._last_regime_info)`.
  - After `aggregated = self._aggregate(...)`: `_append_regime_log(REGIME_LOG_PATH, _build_regime_log_record(...))`, with `tomorrow_kwh = round(float(aggregated["tomorrow"]), 3)` and `_tomorrow_local_date(pd.Timestamp.now(self._timezone).tz_localize(None))`.
  - Add `(f"{FORECAST_REMOTE}/regime_log.jsonl", "regime_log.jsonl")` to `FIXED_FILES`.
- [ ] **Step 6: Run** the new tests, the golden test and the full suite. Expected: all PASS.
- [ ] **Step 7: Commit:** `feat: regime_log.jsonl — tomorrow regime label/proba per update (#110)`.

### Task A4: Docs and finalize Part A

- [ ] **Step 1:** CHANGELOG entry via `@changelog-writer`: "regime stability log, prediction-neutral". No README change.
- [ ] **Step 2: Check the deploy window.**
  - Run `pull_ha_data.py`, then read `data/models/meta.pkl` `last_trained` (naive local, no conversion).
  - Deploy only if it is < 5 h before now, or the C verdict is closed.
  - Otherwise **stop and tell the user** the next eligible window, i.e. the next scheduled retrain = last retrain + 168 h. The user can also choose to trigger a retrain deliberately; that retrain is itself forecast-changing, so that's their call.
- [ ] **Step 3:** Finalize via `@deploy-agent` Workflow A (merge → dev, next `v0.12.0-alpha-N` tag, deploy, push, cleanup).
- [ ] **Step 4: Verify.**
  - On the window path, the AppDaemon log shows "Startup retrain skipped".
  - After ≥ 2 hourly updates, `pull_ha_data.py` brings down `data/regime_log.jsonl` with ≥ 2 lines and a non-null `label`.
- [ ] **Step 5:** Update `memory/project_regime_flip_weather_source.md`: Part A live, plus the date the log started (that date bounds the §7 baseline).

---

## Part B — Soft path, default off (`fix/regime-soft-labels`)

### Task B1: One `regime_kwh` builder for training and predict

**Files:**
- Modify: `apps/energy_forecast/clustering.py` (new module functions)
- Modify: `apps/energy_forecast/model.py` (`623–639` training; `1200–1216` predict)
- Test: `tests/test_clustering.py`

**Interfaces:**
- Produces: `clustering.regime_kwh_from_proba(proba: pd.DataFrame, timestamps: pd.Series, centroids: np.ndarray) -> np.ndarray`.
  - Input `proba`: index = `datetime.date`, columns `0..k-1`.
  - It reindexes to `timestamps.dt.date` and forward-fills over dates **absent** from the index. Leading absent dates give 0. Rows present in the index (including all-zero rows) are used as-is.
  - Returns `(P_hourly * centroids[:, hours].T).sum(axis=1)`, one value per timestamp.
- Produces: `clustering.one_hot(labels: pd.Series, k: int) -> pd.DataFrame`. Labels `< 0` give an **all-zero row**, which matches today: `map` yields `-1`, which gives 0 (`model.py:1206`). Labels `≥ k` are clipped to `k − 1`, matching today's `np.clip`.

- [ ] **Step 1: Write the tests.**
  - `test_one_hot_matches_centroid_lookup`: `regime_kwh_from_proba(one_hot(labels, k), ts, C)` equals `C[labels[date(t)], hour(t)]`.
  - `test_soft_is_weighted_sum`: proba `[0.25, 0.75]` gives `0.25*C[0,h] + 0.75*C[1,h]`.
  - `test_ffill_absent_dates_and_leading_zero`.
  - `test_negative_label_gives_zero_row`.
  - `test_label_ge_k_clipped`.
  - `test_missing_class_column_zero_finite` (Review Focus 5).
- [ ] **Step 2: Run them.** Expected: FAIL.
- [ ] **Step 3: Implement both functions.** Replace both builders' lookup bodies in `model.py` with `regime_kwh_from_proba(one_hot(labels, centroids.shape[0]), ts, centroids)`, keeping each builder's existing label source and date mapping.
- [ ] **Step 4: Run** the new tests and **the A1 golden test, which must pass unchanged**, then the full suite.
- [ ] **Step 5: Commit:** `refactor: single regime_kwh builder from a probability matrix (#110)`.

### Task B2: Out-of-fold probabilities from `RegimePredictor.fit`

**Files:**
- Modify: `apps/energy_forecast/clustering.py` (`RegimePredictor.fit`, ~138–212)
- Test: `tests/test_clustering.py`

**Interfaces:**
- Produces: `RegimePredictor.fit(daily_features, labels, sample_weight=None, verbose=True, n_clusters: int | None = None) -> pd.DataFrame | None`. The return value is not used anywhere today (`model.py:621`), so the change is safe.
  - Returns the **5-fold out-of-fold** matrix: `cross_val_predict(<same RF config, no oob>, X, y, cv=KFold(5, shuffle=True, random_state=42), method="predict_proba", params={"sample_weight": w} if weighted else {})`.
  - Index = `common_idx`. Columns `range(n_clusters)`, mapped from `np.unique(y)` (= the order `cross_val_predict` uses). Missing classes are 0.
  - Computed **right after** `self.model.fit` (`clustering.py:173`), in its own `try`, before the `cross_val_score` block, so a CV failure can't lose it.
  - Returns `None` on the early `< 14` return or if the out-of-fold step raises (logged at warning).
  - Nothing new is stored on `self`, so the pickle format is unchanged. `n_clusters=None` means `max(labels) + 1`; callers pass it explicitly.

- [ ] **Step 1: Write the tests.**
  - `test_fit_returns_oof_matrix_aligned`: index equals the intersection of features and labels; rows sum to 1; `list(columns) == list(range(n_clusters))`.
  - `test_fit_oof_no_zero_rows_with_heavy_weights`: weights `np.linspace(1, 40, n)` give no row with sum 0. This is the regression against the sklearn-1.9 weighted-bootstrap OOB failure (spec §5).
  - `test_fit_oof_rare_class_column`: a class present in only 3 days still gets a finite column.
  - `test_fit_short_returns_none`.
  - `test_fit_cv_failure_keeps_oof`: monkeypatch `clustering.cross_val_score` to raise; `fit` still returns the matrix and `is_fitted` is True.
  - `test_fit_does_not_add_attributes`: `vars(rp)` keys are unchanged.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS, and the full suite is green.
- [ ] **Step 3: Commit:** `feat: RegimePredictor.fit returns 5-fold out-of-fold probabilities (#110)`.

### Task B3: `regime_mode` on `train()`, persisted, read at predict

**Files:**
- Modify: `apps/energy_forecast/model.py`:
  - `train()` signature (~327–360);
  - training block (~583–639);
  - predict block (~1196–1216);
  - `__init__` (add `self._regime_mode = "hard"`);
  - `_save` meta dict (~1721–1745);
  - `_load` (~2188).
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: B1 `regime_kwh_from_proba`, `one_hot`; B2 `fit(..., n_clusters=centroids.shape[0])` return value; A2 `predict_proba`.
- Produces: `EnergyForecastModel.train(..., regime_mode: str = "hard")`, with `ValueError` for any other value, and `self._regime_mode` saved as meta `regime_mode`.
  - **Training, `soft_oof`:** `P` = the fit's out-of-fold matrix, reindexed to all clusterer-label dates. Dates with a clusterer label but no features get explicit 0 rows, and their count is logged at info. `self._regime_mode = "soft_oof"` is set **only** if that series was built; otherwise `"hard"` (fit returned `None`, or regimes disabled).
  - **Predict:** call `predict_proba` once (reuse the A3 result). If `self._regime_mode == "soft_oof"` and it isn't `None`, `P` = that; otherwise `one_hot(predicted_labels)`.
  - `_load`: `self._regime_mode = meta.get("regime_mode", "hard")`.

- [ ] **Step 1: Write the tests.**
  - `test_default_mode_is_hard_and_golden`: the A1 golden test still passes, and meta has `regime_mode == "hard"`. D2a rewrites this test.
  - `test_soft_oof_training_matches_hand_computed`:
    - spy `RegimePredictor.fit` (wrap, keep its return as `P_oof`);
    - spy `model._engineer_features` via monkeypatch, capturing its `regime_kwh_series` kwarg;
    - for 3 sampled training hours, `series[t] == Σ_k P_oof[date,k]·C[k,hour]`.
  - `test_soft_oof_no_feature_days_zero`: monkeypatch `model._prepare_daily_regime_features` (as imported in `model.py`) to drop 2 dates from its result (weather stays intact, so the rows survive into training). The captured series is 0 at those dates' hours.
  - `test_soft_oof_fit_failure_writes_hard`: monkeypatch `RegimePredictor.fit` to return `None`, then `model._regime_mode == "hard"`.
  - `test_meta_roundtrip_and_missing_key` (Review Focus 4):
    - save a `soft_oof` model; reload gives `"soft_oof"`;
    - for a hard-trained model, strip the key from `meta.pkl`, then **`_write_hash(meta_path)`** (otherwise `_load` discards the whole meta, `model.py:2160–2165`);
    - reload: assert `last_mae` / `feature_cols` loaded (non-default), `_regime_mode == "hard"`, and predictions `assert_allclose` to the A1 golden.
  - `test_soft_predict_uses_proba`: for a `soft_oof` model, X `regime_kwh` for tomorrow equals `Σ p·C` from `predict_proba`.
  - `test_invalid_mode_raises`.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS, and the full suite is green.
- [ ] **Step 3: Commit:** `feat: regime_mode soft_oof behind private train() switch (#110)`. **Do not merge or deploy** this branch (§5).

---

## Part C — Reproduce, then backtest (scripts on `fix/regime-soft-labels`; earliest run ≈ 2026-11-05)

**Codebase for every Part C run (§6.1):** dev + A+B. If `fix/counter-reset-increments` and `fix/ev-sensor-trust` aren't on `dev` yet, run from a local scratch branch that merges both into `fix/regime-soft-labels`. **Never push the scratch branch.** Record the commit in the results JSON.

**Scripts must run without Part B:** `backtest_regime.py` feature-detects `regime_mode` via `inspect.signature(EnergyForecastModel.train)`. Without it, the script skips arm T with a printed message. `tests/test_regime_harness.py` passes on `dev` without B's commits (needed for D2b).

**Single import path:** scripts and tests import `energy_forecast.energy_forecast` and `energy_forecast.model`, via `apps/` on `sys.path` as in `scripts/backtest_heating_feature.py`. Never use `apps.energy_forecast...`, so monkeypatches land on the module `_retrain` runs in.

### Task C1: Live-equivalent loader and prediction inputs

**Files:**
- Create: `scripts/regime_harness.py`
- Test: `tests/test_regime_harness.py` (importlib idiom as in `tests/test_check_ha_status.py:24`)

**Interfaces:**
- Produces:
  - `trim_caches(src: Path, dst: Path, cutoff: pd.Timestamp) -> None`. It copies every `*.csv`/`*.json` from `src` to `dst`. For CSVs with a `timestamp` column it keeps rows `≤ cutoff` (local); other files are copied unchanged.
  - `load_live_equivalent_training_frame(cutoff: pd.Timestamp, data_dir: Path, scratch: Path) -> dict`, returning the exact kwargs plus the positional args (as `"_args"`) that `_retrain()` passed to `model.train()`.
  - `build_prediction_inputs(origin: pd.Timestamp, data_dir: Path, scratch: Path) -> dict`, returning the kwargs for `_prepare_prediction_X` at `origin`, built the way `_update_sensors` builds them (`energy_forecast.py` ~1990–2085):
    - `forecast_df` = archive weather for origin … origin + 48 h, as a perfect forecast;
    - recents (actuals, sub-sensors, climate, DHW, heating-active, buffer) = caches trimmed to `≤ origin`;
    - the same heating/physics projection inputs;
    - the post-predict DHW override is **excluded** (the backtest scores model output only).
- Approach (§6.1):
  - Build the app via `EnergyForecast.__new__(EnergyForecast)`. Set every `self._*` attribute `_retrain` reads, with values from `data/apps.yaml` (start from `_FakeRetrain`, `tests/test_energy_forecast.py:4024`, and grep `_retrain` for the rest). Point `_cache_path` at `scratch/energy_history.csv` so every derived path lands in `scratch`.
  - `get_history` → `[]`.
  - Patch `weather.fetch_historical_weather` and `fetch_open_meteo` with a stub that serves the requested `[start, end]` window clipped to `≤ cutoff` from `scratch/weather_archive.csv`. That file is filled from the Open-Meteo archive API once per run, kept inside `scratch`, and never shared across cutoffs.
  - Patch `pd.Timestamp.now` → `cutoff`. Patch the **module attribute** `energy_forecast.energy_forecast.date` with a `date` subclass whose `today()` returns `cutoff.date()`. (`datetime.date.today` itself can't be patched; it's used at `energy_forecast.py:1688`.)
  - Stub `heating_season.save_thresholds`, `ha_data.fetch_energy_history_15m`, and `self._ml_model` with a recorder whose `train(*a, **kw)` stores its arguments and whose `last_trained` = `datetime.min`.

- [ ] **Step 1: Write the tests.**
  - `test_trim_caches_drops_future_rows`.
  - `test_trim_caches_copies_non_timestamp_files`.
  - `test_weather_stub_clips_to_cutoff`.
  - `test_loader_captures_train_kwargs`: tiny `data_dir` fixture (100 h energy CSV, minimal apps.yaml, weather stub in-memory). Captured kwargs include `enable_regimes=True`, and the energy frame's max timestamp ≤ cutoff.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS, and the tests also pass on `dev` without Part B.
- [ ] **Step 3: Smoke-run** against the real `data/`. Cutoff = live `meta.pkl` `last_trained` (naive local, no conversion). Expected: no error. Print the energy rows, date span and kwargs keys, and one `build_prediction_inputs` call's keys.
- [ ] **Step 4: Commit:** `feat: regime_harness live-equivalent loader + prediction inputs (#110)`.

### Task C2: Reproduction gate

**Files:**
- Create: `scripts/backtest_regime.py` (subcommand `gate`)
- Test: `tests/test_regime_harness.py`

**Interfaces:**
- Consumes: C1; the live `data/models/{clusterer,regime_model,energy_model,meta}.pkl` from `pull_ha_data.py` (it already pulls `models/*.pkl`).
- Produces:
  - `nearest_centroid_labels(energy_df: pd.DataFrame, centroids: np.ndarray) -> pd.Series`. It builds day profiles exactly as `DailyProfileClusterer.fit` does (`clustering.py:86–89`): days with ≥ 18 h, pivot, `reindex(range(24))`, `interpolate(axis=1, limit=6)`, ffill/bfill. It returns the argmin Euclidean distance per day. Live labels = this with the **live** centroids, because `clusterer.pkl` stores no labels.
  - `label_agreement(live_centroids, harness_centroids, live_labels, harness_labels) -> float`. Both ID sets are sorted by centroid day-sum, then the share of equal labels is returned.
  - `gate(results: dict) -> tuple[bool, list[str]]`, applying: same k, agreement ≥ 0.90, and max |harness − live| daily prediction / live ≤ 0.03.

- [ ] **Step 1: Write the tests.**
  - `test_nearest_centroid_labels_matches_fit`: on a toy frame, it equals the labels returned by `DailyProfileClusterer.fit`.
  - `test_label_agreement_permutation_invariant`.
  - `test_gate_thresholds`: 0.90 passes, 0.899 fails; 3.0 % passes, 3.01 % fails; a k mismatch fails.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS.
- [ ] **Step 3: Run the gate.**
  - Train a harness model (`regime_mode="hard"`) on the C1 frame at the live cutoff.
  - Label agreement on the harness training days.
  - Predictions: both models' `_prepare_prediction_X` → `predict(_prepared=...)` on `build_prediction_inputs` for the **7 daily 17:00 origins before the cutoff**, scoring next-day totals.
  - Print the sorted centroid day-sums for both, which explain 35/64/10.6 vs 7.9/37/20.9.
  - Record `sklearn`/`lightgbm`/`pandas`/`numpy` versions, local and live (read the live versions from the AppDaemon log, or via `check_ha_logs.py`), in the results JSON.
  - **If the gate fails:** check auto-K, whether the live clusterer is older than `meta.last_trained`, and library-version differences. Then **stop and report to the user**; don't run C3.
- [ ] **Step 4: Commit** the script, and record the gate output in `memory/project_regime_flip_weather_source.md`.

### Task C3: Backtest and decision

**Files:**
- Modify: `scripts/backtest_regime.py` (subcommand `backtest`)
- Test: `tests/test_regime_harness.py`

**Interfaces:**
- Produces: `decide(r: dict) -> str` returning `"T"`, `"N"` or `"H"`. Input `r[arm]` = `{"pooled_mae", "primary_mae", "jan_mae", "max_delta", "n_over_2"}` for `arm in ("H", "T", "N")`; T may be absent, which means it doesn't pass. It implements the §6.4 rules exactly as in Global Constraints.
- Protocol (§6.3):
  - Arms: H = `regime_mode="hard"`; T = `"soft_oof"`; N = `enable_regimes=False`.
  - Folds: primary = 2026-10-15 → last complete day (assert ≥ 21 days); guard = 2026-01-05 → 2026-01-31.
  - Expanding window: C1 loader at each weekly cutoff, then `train`.
  - Origin 17:00 local via `build_prediction_inputs`, scoring the next calendar day's total against `pred_history.json` actuals, falling back to the corrected target frame for days not in `pred_history`.
  - Sensitivity: re-predict each origin with `sunshine_min` ×0.66 and ×1.5, giving max |Δ daily total| and the count of days with |Δ| > 2 kWh.
  - Also print per-fold MAE and the `regime_kwh` gain share.

- [ ] **Step 1: Write the tests.** One per rule:
  - `test_decide_t_when_both_pass_and_n_not_3pct_better`;
  - `test_decide_n_when_3pct_pooled`;
  - `test_decide_n_when_5pct_january`;
  - `test_decide_only_passing_candidate`;
  - `test_decide_h_when_none_pass`;
  - `test_decide_without_t_arm`;
  - `test_sensitivity_requires_strictly_lower_max`: equal max fails;
  - `test_count_over_2_allows_equal`: H = 0 and T = 0 passes.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS.
- [ ] **Step 3: Run the backtest** (≥ 2026-11-05). Save `data/backtest_regime_results.json` (including commit and library versions) and print the per-fold table and `decide()` output.
- [ ] **Step 4: Record the results.** Write them into the memory file and ROADMAP #110. **Show the user the table and decision before starting Part D.**
- [ ] **Step 5: Commit:** `feat: regime backtest H/T/N + decision rule (#110)`.

---

## Part D — Ship (only after the user has seen C3, and the sequencing gate holds)

### Task D1: Stability check script

**Files:**
- Create: `scripts/check_regime_stability.py`
- Test: `tests/test_regime_harness.py`

**Interfaces:**
- Produces: `delta_share(lines: list[dict], threshold: float = 2.0) -> float`. Δ is taken between consecutive lines with the same `target_date` **and** the same `last_trained`; the share is `|Δ| > threshold` over those pairs.
- CLI: `--deploy <ISO local>` and `--rebaseline <ISO local>`. It prints:
  - the pre share over `max(rebaseline, deploy − 14 d)` … deploy, with that window's length;
  - the post share over deploy … deploy + 14 d;
  - daily MAE pre/post from `pred_history.json`;
  - PASS/FAIL against the §7 criteria.

- [ ] **Step 1: Write the tests.**
  - `test_delta_share_skips_rollover`;
  - `test_delta_share_skips_retrain_boundary`;
  - `test_delta_share_basic`.
- [ ] **Step 2: Run, implement, run.** Expected: FAIL, then PASS.
- [ ] **Step 3: Commit:** `feat: regime stability check script (#110)`.

### Task D2a (decision T): ship soft_oof

- [ ] **Step 1: Change the test first.** Change the `train()` default to `regime_mode="soft_oof"`. **Rewrite** B3's `test_default_mode_is_hard_and_golden` into `test_default_mode_is_soft_oof` and run it red first. Then regenerate the A1 golden npz with `REGEN_REGIME_GOLDEN=1` and commit it **with an explicit note that the prediction change is intended**.
- [ ] **Step 2: Run the full suite.** Expected: green. Commit: `feat: soft out-of-fold regime labels by default (#110)`.
- [ ] **Step 3: Confirm the sequencing gate.** Read ROADMAP "Cold-onset sequencing" and confirm A+B are deployed and re-baselined, and that no #105 deploy shares the window. If not, **stop and tell the user**.
- [ ] **Step 4: Finalize** via `@deploy-agent` Workflow A.
  - The startup retrain applies the change if the last retrain is > 6 h old. Fire `RELOAD_ENERGY_MODEL` only if the log shows "Startup retrain skipped".
  - Verify the live `meta.pkl` `regime_mode == "soft_oof"`.

### Task D2b (decision N): disable regimes live

- [ ] **Step 1: Confirm the sequencing gate.** Same check as D2a Step 3; if it doesn't hold, **stop and tell the user**.
- [ ] **Step 2: Land the scripts on dev.**
  - On a new branch `chore/regime-backtest-scripts`, cherry-pick **only** the Part C/D1 script commits (`scripts/regime_harness.py`, `scripts/backtest_regime.py`, `scripts/check_regime_stability.py`, `tests/test_regime_harness.py`). **Never cherry-pick B1–B3** (§7 N).
  - Run the full suite on that branch. Expected: green, with arm T skipped by feature detection.
  - Finalize via `@deploy-agent`. Arm T stays reproducible only from the unmerged `fix/regime-soft-labels`; note that in the memory file.
- [ ] **Step 3: Disable regimes on the live instance.**
  - Set `enable_regimes: false` in the **live** `apps.yaml` via Samba, keeping a backup copy, then restart AppDaemon. The shipped default stays unchanged.
  - Fire `RELOAD_ENERGY_MODEL` if the log shows "Startup retrain skipped".
  - Verify the live `meta.pkl` `enable_regimes == False`, and that the regime log shows `label: null` (Review Focus 1).

### Task D3: Post-deploy check and docs (deploy + 14 days)

- [ ] **Step 1:** Run `pull_ha_data.py`, then `check_regime_stability.py --deploy <ts> --rebaseline <A+B re-baseline ts>`. Expected: PASS on both §7 criteria. On FAIL, report to the user. Rollback = revert the merge on `dev`, redeploy, and let it retrain; for N, restore the `apps.yaml` backup.
- [ ] **Step 2: Update the docs.**
  - CHANGELOG via `@changelog-writer`.
  - ROADMAP: #110 closed, and #83 unblocked with the new `regime_kwh` rank.
  - Memory: mark `project_regime_flip_weather_source` resolved.
