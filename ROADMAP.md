# Forecast Accuracy Roadmap

Current: **v0.11.4** — 2026-06-16, main. 627 tests.

---

## Current Status

**dev:** v0.11.4-alpha-2 (same codebase — release commit was made on main only). **main:** v0.11.4 released 2026-06-16.

Recent releases:
- v0.11.4 — 15-minute energy history cache (#85), strip partial day from clustering (#86), tomorrow block P10/P90 interval sensors, code review batch 1–5 (all open findings closed).
- v0.11.0 — Daily Regime Clustering, EV-subtracted clustering input (#82 — live since 2026-04-23).

**SHAP check due:** #82 (EV-clean clustering) has been live for ~2 months. Regime clusters should now reflect genuine intra-day shape patterns rather than EV charge timing. Run a SHAP summary to see whether `regime_kwh` ranks high (shape signal) or low (redundant with temperature features) — this determines whether #83 adds value.

**MAE trajectory:** 0.7 → 0.52 kWh/h (as of April). Current value readable from `sensor.ha_energy_forecast_mae_30d`.

**Physics Phase 2 interval-coverage check due (post-deployment):** after Phase 2 (`use_physics_residual: true`) has been live for ≥30 days, verify empirical prediction-interval coverage on gross kWh matches the target (80% by default, per `_calibrate_intervals()`'s conformal quantile). If coverage has drifted, the CQR calibration on the residual distribution may need a wider correction — see `docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md` §5.2. Not yet applicable — Phase 2 stays dormant behind the cold-start gate until ≥30 winter UA_eff calibration windows exist (not expected before winter 2026/27).

**Deploy freeze lifted (2026-07-16):** solar panel commissioning completed 2026-07-16 (hardware confirmed live). `#89` (physics sensor cache dedup, merged to `dev` @ `fc78113`) deployed the same day, and `apps.yaml`'s solar PV + battery target-correction block was enabled with live SolarEdge/gPlugK entity IDs — see `memory/project_solar_feature_pending.md`. `#89`'s Task 6 manual cache-cleanup is also done (11 orphaned CSV files deleted from HA, confirmed via re-list). Still open: re-verifying the battery charge/discharge sensor direction once the battery has visibly cycled (SOE was still 0% at commissioning). `#40` (battery SoC as a feature) is now unblocked and worth revisiting.

**Solar/battery live-path correction bug found and fixed (2026-07-16):** same evening as commissioning, `_update_sensors()` (live hourly path) was found never to apply the solar/grid-export/battery target correction that `_retrain()` uses to train the model — every battery charge cycle inflated raw grid import in `recent_actuals`/`full_actuals`, firing false "unusual consumption" alerts and pushing forecasts up. Fixed via `docs/superpowers/plans/2026-07-16-live-target-correction-fix.md` (shared `_fetch_correction_dfs()` extracted from `_retrain()`, wired into `_update_sensors()` too).

**Physics Phase 1 + τ-seed accuracy check due (~2026-07-17):** Plan A-D (physics-ML hybrid) and the τ-calibration one-time seed (11.64h) both went live 2026-07-10. Check `sensor.ha_energy_forecast_energy_forecast_mae_7d` (should fully reflect the post-deployment period by then) and `_mae_30d` (partial blend, directionally informative) to see whether forecast accuracy improved, held steady, or regressed. Expect a limited effect this early: `UA_eff` (space heating) failed to calibrate on the first post-deploy retrain (R²=0.09, insufficient summer heating-cycle data) and stays at its default, so only the DHW/base-load physics component (`Q_base_el`, `Q_dhw_daily`, `UA_dhw` — all calibrated successfully) and the τ fix are in play right now. A bigger signal is expected once winter data lets `UA_eff` calibrate for real.

---

## Design Decisions

| Topic | Decision |
|---|---|
| Primary goal | **Forecast accuracy** + visibility/dashboards |
| Solar PV | Planned — target correction done (v0.8.0); solar forecast feature out of scope |
| Home battery | SoC as feature deferred until panels installed (#40) |
| Tariff | Fixed flat rate — price optimisation **out of scope** |
| Load shifting | **Out of scope** — handled by a separate system |
| Audience | Personal-first; HACS nice-to-have, never at cost of accuracy |

> **Critical definition:** *Consumption* = `grid_import − grid_export + solar_production − battery_charge + battery_discharge`. Not net load, not grid-only import.

---

## Deployment Workflow

1. Feature branch → implement + tests pass (`python -m pytest tests/ -v`)
2. PR → code review → merge to `dev`
3. Smoke-test on local HA instance (watch AppDaemon log; confirm sensors update)
4. Stable period on `dev` → merge to `main`
5. Update CHANGELOG.md, create semver tag, push → Forgejo release

---

## Backlog

### #83 — Add `predicted_day_total` Feature (Temperature Regression)

**Priority:** Medium — consider after #82.

**Context**: Empirical analysis (2026-04-22) shows non-EV daily totals range 14→40 kWh across seasons, strongly temperature-driven. After #82, clean regime clusters will capture shape. A separate `predicted_day_total` feature from a lightweight regression (`heating_deg_sum_24h`, `temp_ewma_24h`, `is_away`, `people_home` → daily total kWh) would give the main model an explicit scale signal independent of the regime shape.

**Note**: May not add much if the cleaned `regime_kwh` already encodes enough scale information. Evaluate after #82 is live and SHAP importance is re-checked.

**Prerequisite**: #82.

**Effort:** ~3 h. **Impact:** MEDIUM.

---

### #15 — HVAC / Boiler State: Projected Flow Setpoint

**Priority:** escalate if sub-sensor bouncing persists after 2026-04-20; otherwise long-term.

**Signal:** Derive a `flow_setpoint` feature from the Kermi heating curve — this allows accurate 48-hour forward projection using forecast outdoor temps, rather than relying on stale sensor values.

**Projection formula (per future hour h):**
```
flow_setpoint(h) = np.interp(outdoor_temp[h], curve_x, curve_y)
                   + parallel_shift          # current HA entity value, projected flat
                   - 2  if 21 ≤ hour < 24
                       or  0 ≤ hour < 6     # night setback
                   → NaN if outdoor_temp[h] ≥ 20  # heating cutoff
```

**Heating curve breakpoints (from Kermi UI):**

| Outdoor °C | Flow °C |
|---|---|
| -20 | 55.5 |
| -15 | 52.5 |
| -10 | 49.5 |
| -5 | 46.0 |
| 0 | 43.0 |
| 5 | 39.5 |
| 10 | 35.5 |
| 15 | 31.0 |
| 20 | 25.0 |

**Config keys:**
```yaml
heating_curve_sensor: sensor.kermi_parallel_shift
heating_curve_points:
  - [-20, 55.5]
  - [ -5, 46.0]
  - [  5, 39.5]
  - [ 20, 25.0]
heating_cutoff_temp: 20
night_setback_delta: -2
night_setback_start: 21
night_setback_end: 6
```

**Effort:** ~3 h. **Impact:** HIGH for heat pump buildings.

---

### #10 — School Holiday Feature

Swiss Schulferien dates are canton-specific but stable year-to-year. During school holidays daytime consumption rises. Implement a static lookup table per canton via `apps.yaml`; add `is_school_holiday` to `_FEATURES_BASE`.

**Effort:** ~4 h. **Impact:** MEDIUM.

---

### #46 — Dashboard: Replace Personal Entity IDs

`dashboard/dashboard.yaml` and `dashboard/energy-today.yaml` contain user-specific entity IDs (`sensor.skoda_enyaq_*`, `sensor.kermi_*`, etc.) that will break on other installs. Required before HACS or wider sharing:

- Replace personal entity IDs with commented-out placeholders.
- Add `# EDIT: replace entity IDs below with your own` header comment in each file.

**Effort:** ~30 min. **Impact:** UX / sharing pre-requisite.

---

### #16 — HACS Support

Make the app installable via [HACS](https://hacs.xyz/) (AppDaemon category). No code changes needed — `apps/energy_forecast/` is already in the correct location.

Required:
- Add `hacs.json` at repo root.
- Add `info.md` (HACS install panel; must warn that `apps.yaml` setup is still manual).
- Add "Install via HACS" section to README.
- Set repo topics: `appdaemon`, `home-assistant`, `hacs`.

**Effort:** ~1 h. **Prerequisite:** #46 (entity ID cleanup).

---

### #18 — Custom Component Config Flow *(long-term)*

A full HA custom component with UI-driven setup wizard (entity picker, lat/lon auto-populated, optional fields). Writes `apps.yaml` and patches AppDaemon add-on dependencies via Supervisor API. Significant effort; only path to zero-manual-step install.

**Effort:** 8+ h. **Impact:** UX / install.

---

### #87 — Recent Consumption Trend Feature (`trend_deviation`)

**Priority:** Low-Medium — standalone, no prerequisites.

**Context:** Simulation (2026-06-21) on last-30-day holdout shows ~18% daily MAE improvement from adding `trend_deviation = rolling_mean_24h − rolling_mean_7d` to the feature set. The feature ranks 14th of 19 in importance — it adds signal the model cannot derive by itself from individual rolling stats because tree splits on pair-wise differences are expensive to find without an explicit feature. Confirmed by LightGBM simulation without weather features (weather-only simulation; absolute numbers not directly comparable to live model).

**Design:**
```python
# In _engineer_features(), after rolling stats are computed:
df["trend_deviation"] = df["rolling_mean_24h"] - df["rolling_mean_7d"]
df["trend_z_score"]   = df["trend_deviation"] / (df["rolling_std_24h"].clip(lower=0.05))
```
Both columns added to `_FEATURES_BASE`. At prediction time, `trend_deviation` and `trend_z_score` are derived from already-computed rolling stats, so no new data fetching is needed.

**Note:** Does not fix the thermal-transition cluster (Jun 4-12 type days) — those require shorter halflife or a temperature-similarity weighting scheme (see #88). Adds signal on ordinary days.

**Effort:** ~1 h (code + 2 tests). **Impact:** LOW-MEDIUM (+18% daily MAE in simulation; smaller real-world gain expected with weather features present).

---

### #88 — Temperature-Similarity Sample Weighting

**Priority:** Low-Medium — more effective as warm-weather history grows.

**Context:** Simulation (2026-06-21) compared three weighting schemes on 30-day holdout (May 22 – Jun 21):

| Scheme | Daily MAE | Daily MBE | Weighted mean train temp |
|--------|-----------|-----------|--------------------------|
| time-60 (current) | 3.52 kWh | −3.20 kWh | 8.5 °C |
| time-30 (shorter halflife) | **3.20 kWh** | −2.84 kWh | 10.5 °C |
| tempsim (time-60 × Gaussian kernel σ=5°C) | 3.35 kWh | −3.01 kWh | 12.8 °C |

Holdout period mean outdoor temp: **19.9 °C**. Temperature-similarity shifts the effective training distribution toward warmer data, but can only shift 4°C (8.5→12.8) with current history — still 7°C below the holdout mean. The simpler time-30 halflife outperforms the combined approach.

**Key finding:** Both improvements are modest; neither fixes the thermal-transition cluster (Jun 4-12 type days). The root cause of those outlier days is that the model has never seen warm-month data (history started Oct 2025). **As summer 2026 data accumulates, temperature-similarity weighting will become meaningfully effective.**

**Design (when ready to implement):**
```python
# In train(), after existing exponential decay weights:
temp_sigma = self._cfg.get("temp_weight_sigma_c", 5.0)  # configurable, 0 = off
if temp_sigma > 0 and predict_temp is not None:
    temp_sim = np.exp(-((train_temps - predict_temp)**2) / (2 * temp_sigma**2))
    h_weights = h_weights * temp_sim
```
`predict_temp` = mean of last 24h outdoor temperature. New config key `temp_weight_sigma_c` (default 0 = off; suggest 5.0 once ≥ 12 months of history exists).

**Prerequisite:** Revisit after first full year of data is available (earliest: Oct 2026).

**Effort:** ~3 h. **Impact:** LOW now, MEDIUM once a full year of history is available.

---

### #84 — Legionella / DHW Boost Hour Feature

**Prerequisite:** DHW sub-sensor infrastructure (related to #22).

**Problem**: The weekly legionella DHW protection cycle (heat buffer to ~60 °C, ~1–2 h) creates a predictable spike that the model has no dedicated signal for. Currently relies entirely on `lag_168h` (1-week lag), which takes 2–3 weeks to establish after a schedule change. The schedule was shifted from Tuesday ~23 h to Wednesday ~14 h on 2026-04-22, so the transition period is live now.

Lag-feature pollution to the following day is modest (~0.1–0.3 kWh/h for 24–48 h), so this is not urgent.

**Design (when implemented)**:
- New `_compute_likely_legionella_hours()`: detect HOW slots where `dhw_buffer_temp > 58 °C` within a 30-day rolling window
- New binary feature `is_legionella_hour` (mirrors `likely_ev_hour` pattern)
- Optional `legionella_schedule_reset_date` config key to prune pre-change data and accelerate transition
- Falls back gracefully to 0 when `dhw_buffer_sensor` is not configured

**Effort:** ~3 h. **Impact:** LOW-MEDIUM (primarily useful after schedule changes; lag features self-correct within ~3 weeks otherwise).

---

### #90 — Fill Gaps in `_SHAP_FEATURE_LABELS` Dashboard Narrative Dictionary

**Found:** 2026-07-15, while auditing physics/thermal feature additions (#89 review).

**Problem:** `_build_shap_narrative()` (`energy_forecast.py:143-151`) falls back to the raw internal feature name (`label = _SHAP_FEATURE_LABELS.get(feat, feat)`) whenever a top-SHAP feature isn't in `_SHAP_FEATURE_LABELS` (`energy_forecast.py:62-140`). Several features that are actually in `_FEATURES_BASE`/`_FEATURES_WITH_SENSOR` (`model.py:79-160`) or added dynamically to `feature_cols` have no entry, so if they ever rank in the top SHAP features, the dashboard text card shows the raw column name instead of a readable phrase:

- `hp_heating_degree` (HP-calibrated heating cutoff)
- `temp_in_neutral_zone` (dead-band flag, 15–22 °C)
- `heating_active` (seasonal heating on/off)
- `lag_24h_tgated`, `lag_168h_tgated`, `lag_336h_tgated` (temp-delta-gated lag features — labels dict currently only has the untagged `lag_24h`/`lag_168h`/`lag_336h`, which are **not** real feature columns; those three dict entries are dead and should be replaced by the `_tgated` names actually in `_FEATURES_BASE`)
- `heating_buffer_temp` (added to `base_features` when `heating_buffer_temp_sensor` is configured — `model.py:578`)
- `physics_kwh` (added to `base_features` when the physics residual model is active — `model.py:576`)

**Recommended fix (not yet implemented):** add the six missing labels above to `_SHAP_FEATURE_LABELS`, and replace the three stale untagged `lag_*h` entries with their `_tgated` equivalents. Add a test asserting `set(_SHAP_FEATURE_LABELS) ⊇ set(_FEATURES_WITH_SENSOR) ∪ {"physics_kwh", "heating_buffer_temp"}` (minus any deliberately-excluded internal-only columns) so this can't silently drift again as new features are added.

**Effort:** ~1 h. **Impact:** LOW (cosmetic — dashboard narrative readability only, no forecast-accuracy effect).

---

### Deferred

| # | Item | Reason |
|---|------|--------|
| #22 | EV SoC / charging state feature | EV hours are subtracted from training target — SoC has no signal to learn. Revisit if EV load is re-included. |
| #40 | Home battery SoC as feature | Deferred until solar panels installed; revisit if residuals show SoC correlation. |
| #24 | Electricity spot price feature | Fixed flat tariff — out of scope. |

---

## Pending Summary

| # | Item | Impact | Effort | Priority |
|---|------|--------|--------|----------|
| 82 | Fix EV contamination in clustering | high (regime_kwh #1 feature) | 2 h | ✅ done (v0.11.0-alpha-16) |
| 83 | `predicted_day_total` scale feature | medium | 3 h | SHAP check first — may be redundant |
| 84 | Legionella/DHW boost hour feature | low-medium | 3 h | low urgency — lag features self-corrected |
| 87 | `trend_deviation` feature (recent vs baseline) | low-medium | 1 h | ready |
| 88 | Temperature-similarity sample weighting | low-medium | 3 h | simulated — see #88 detail |
| 90 | Fill gaps in SHAP narrative label dictionary | low | 1 h | cosmetic — dashboard text card only |
| 15 | HVAC flow setpoint | high (heat pump) | 3 h | escalate if bouncing |
| 10 | School holidays | medium | 4 h | long-term |
| 46 | Dashboard entity ID cleanup | UX / sharing | 30 min | partial — interval entity IDs fixed in fix/review-critical |
| 16 | HACS support | distribution | 1 h | long-term |
| 18 | Config flow | UX / install | 8+ h | long-term |
| 22 | EV SoC | high (EV) | 4 h | deferred |
| 40 | Battery SoC | medium (battery) | 1 h | deferred |
| 24 | Spot price | n/a | — | out of scope |

---

## Done

### Release History

| Version | Date | Highlights |
|---------|------|------------|
| v0.11.4-alpha | 2026-06-13 | 15-minute energy history cache (#85); strip partial day from clustering input (#86). 643 tests. |
| v0.11.0-alpha-16 | 2026-04-23 | Fix EV day exclusion from centroid fitting (#82). 535 tests. |
| v0.11.0-alpha-15 | 2026-04-22 | Regime logging improvements (#82 alpha-15 prep). 535 tests. |
| v0.11.0-alpha-14 | 2026-04-22 | Algorithmic correctness (#64–#69), code quality (#68, #71–#73), test coverage (#74–#78), documentation (#70, #80–#81). 535 tests. |
| v0.11.0 | 2026-04-17 | Daily Regime Clustering (optional module), K-Means 24h profiles, secondary regime predictor model |
| v0.10.0 | 2026-04-10 | Baseline mode (Stages 1–4), thermal/DHW intent, appliance signatures, scenario API, physics features (#55–#58), τ calibration, RC-ODE indoor projection |
| v0.9.0 | 2026-04-10 | Thermal modelling (#49–#52), occupancy (`people_home`), SHAP narrative, relative MAE sensors, rolling MAE persistence |
| v0.8.0 | 2026-03-31 | Solar/battery target correction, model versioning + rollback, CSV health checks, temperature bias-fade |
| v0.7.1 | 2026-03-24 | 404 DELETE fix, MQTT anomaly attrs, dashboard cards (anomaly + SHAP) |
| v0.7.0 | 2026-03-23 | 48 h weather features, anomaly detection, SHAP importance, prediction intervals, ApexCharts dashboard |
| v0.6.0 | — | MQTT Discovery (entity registry, area assignment, labels) |
| ≤v0.5.x | — | Core app, EV subtraction, lag features, adaptive retraining, holiday calendar |

### Completed Items

| # | Item | Done in |
|---|------|---------|
| 1 | Fix missing sunshine in Open-Meteo fallback | ≤v0.5.x |
| 2 | Add `temp_rolling_3d` to prediction horizon | ≤v0.5.x |
| 3 | Pre/post-holiday bridge day features | ≤v0.5.x |
| 4 | Cloud cover / solar irradiance feature | ≤v0.5.x |
| 5 | Fix training/prediction mismatch in rolling features | ≤v0.5.x |
| 6 | LightGBM early stopping + validation-set tuning | ≤v0.5.x |
| 7 | Log-transform the target | ≤v0.5.x |
| 8 | Adaptive retraining trigger | ≤v0.5.x |
| 9 | Cantonal public holidays | ≤v0.5.x |
| 11 | Additional lag: `lag_72h` | ≤v0.5.x |
| 12 | EV charge session probability feature | ≤v0.5.x |
| 13 | Prediction intervals as HA sensors | ≤v0.5.x |
| 14 | Intra-day actuals substitution | ≤v0.5.x |
| 17 | Setup checker sensor (`energy_forecast_setup_status`) | ≤v0.5.x |
| 19 | CSV cache: append-only writes | ≤v0.5.x |
| 20 | Config validation: warn when EV threshold ≥ charger_kw | ≤v0.5.x |
| 25 | Vacation / away flag (`is_away`) | v0.7.0 |
| 26 | Sub-energy sensors (`sub_energy_sensors`) | v0.7.0 |
| 27 | Short-horizon lags (`lag_1h`–`lag_12h`) | v0.7.0 |
| 28 | `num_leaves` hyperparameter sweep | v0.7.0 |
| 29 | Feature importance logging after training | v0.7.0 |
| 30 | CV fold std logging alongside mean | v0.7.0 |
| 31 | Per-hour-of-week NaN fill medians | v0.7.0 |
| 32 | Holiday `apply` → `np.searchsorted` vectorization | v0.7.0 |
| 33 | Day-of-year cyclical feature (`doy_sin` / `doy_cos`) | v0.7.0 |
| 34 | `hours_ahead` feature for horizon-aware prediction | v0.7.0 |
| 35 | Sub-sensor binary activity flag (`{prefix}_active_24h`) | v0.7.0 |
| 36 | Sub-sensor rolling run count (`{prefix}_runs_7d`) | v0.7.0 |
| 37 | MQTT Discovery for entity registry | v0.6.0 |
| 38 | Full 48 h weather forecast features | v0.7.0 |
| 39 | Anomaly detection on forecast residuals | v0.7.0 |
| 41 | Rolling accuracy history sensors (7d / 30d MAE) | v0.7.0 |
| 42 | SHAP feature importance per prediction | v0.7.0 |
| 43 | ApexCharts / Lovelace config snippet | v0.7.0 |
| 44 | Model versioning — keep last N, rollback | v0.8.0 |
| 45 | CSV health checks + gap repair | v0.8.0 |
| 46 | Fix 404 DELETE spam in `_cleanup_legacy_states()` | v0.7.1 |
| 47 | Anomaly binary sensor MQTT attrs + discovery fix | v0.7.1 |
| 21 | Occupancy feature (`people_home`) | v0.9.0 |
| 23 | Solar PV target correction (B1) | v0.8.0 |
| 49 | Exponentially weighted moving average temperature (`temp_ewma_24h/72h`) | v0.9.0 |
| 50 | Rolling accumulated heating degree-hours (`heating_deg_sum_24h/168h`) | v0.9.0 |
| 51 | Temperature rate of change (`temp_delta_1h/24h`) | v0.9.0 |
| 52 | Temperature lag features (`temp_lag_24h/168h`) | v0.9.0 |
| 53 | "Why today?" SHAP narrative attribute | v0.9.0 |
| 54 | Relative MAE sensors (7d / 30d) | v0.9.0 |
| 55 | Verified Passive Decay — τ calibration (OLS passive-cooling windows) | v0.10.0 |
| 56 | Solar-Compensated Thermal Pressure (`thermal_pressure_net`) | v0.10.0 |
| 57 | Wind-Driven Infiltration Feature (`infiltration_pressure`) | v0.10.0 |
| 58 | Humidity-Aware Defrost Proxy (`defrost_risk`) | v0.10.0 |
| 60 | Calibrated default thermal time constant (`DEFAULT_TAU = 12 h`) | v0.10.0 |
| 61 | Daily Regime Clustering (`regime_kwh`) | v0.11.0 |
| 63 | Fix RegimePredictor overfitting (OOB score + constraints + occupancy features) | v0.11.0-alpha-8 |
| 59 | Relaxed τ calibration — quality-scored windows replace hard daytime/solar filters | v0.11.0-alpha-9 |
| 62 | Adaptive Regime Selection (Auto-K) — inertia elbow, K ∈ [2, 8], `regime_count: 0` | v0.11.0-alpha-10 |
| 64 | CQR calibration: random holdout split (rng seed 42) for valid exchangeability guarantee | v0.11.0-alpha-14 |
| 65 | RegimePredictor: TimeSeriesSplit CV logged alongside OOB; warning uses TSCV mean | v0.11.0-alpha-14 |
| 66 | Inertia normalization: bail out to k_lo when range < 1e-6 (homogeneous data guard) | v0.11.0-alpha-14 |
| 67 | Regime label ffill in prediction path — matches training semantics for gap days | v0.11.0-alpha-14 |
| 68 | `strip_tz()` moved to `const.py` as shared utility; weather.py and energy_forecast.py deduped | v0.11.0-alpha-14 |
| 69 | EWMA temperature resets at weather gaps > 2h via NaN sentinels before `.ewm()` | v0.11.0-alpha-14 |
| 70 | Physics feature scaling constants (0.01, 10.0) documented with empirical basis | v0.11.0-alpha-14 |
| 71 | Sub-sensor quality demoted to "fair" when energy_cov > 0.5; CoV stored in signature dict | v0.11.0-alpha-14 |
| 72 | `get_scenario` validates schedule keys and HH:MM format; drops invalid entries with WARNING | v0.11.0-alpha-14 |
| 73 | `__version__` in `__init__.py` is single source of truth for MQTT `sw_version` | v0.11.0-alpha-14 |
| 74 | Gaussian noise in `_make_energy_df()` — KMeans ConvergenceWarnings reduced from 18 → 9 | v0.11.0-alpha-14 |
| 75 | Pickle corruption recovery test for `clusterer.pkl` | v0.11.0-alpha-14 |
| 76 | K=1 fallback test — homogeneous data hits inertia bail-out | v0.11.0-alpha-14 |
| 77 | `train()` edge cases: empty DataFrame, below MIN_TRAINING_ROWS, constant values | v0.11.0-alpha-14 |
| 78 | Network failure tests for `fetch_open_meteo` (404, 500, Timeout, ConnectionError, bad JSON) | v0.11.0-alpha-14 |
| 79 | Timezone-aware fixture audit — confirmed existing tests use naive timestamps correctly; no changes needed | v0.11.0-alpha-14 |
| 80 | `find_optimal_k()` docstring fully documents normalization, bail-out, smoothing, tolerance band, OOB note | v0.11.0-alpha-14 |
| 81 | `_project_indoor_temps()` stale-sensor threshold already documented — confirmed, no change needed | v0.11.0-alpha-14 |
| 82 | Fix EV contamination in regime clustering — EV days excluded from `DailyProfileClusterer.fit()` | v0.11.0-alpha-16 |
| 89 | Dedup physics sensor history fetches — DHW tank temp and room thermostat temp now reuse the ML pipeline's already-fetched data instead of redundant HA history API calls | Unreleased |
