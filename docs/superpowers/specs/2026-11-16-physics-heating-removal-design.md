# Physics Heating Removal — Design

**Date:** 2026-11-16 · **Status:** reviewed (multi-stakeholder, round 1 applied) · **Repos:** ha-energy-forecast (hef), ha-energy-manager (EM)
**Base branch:** `train/next` (carries #113 and #124, which touch `physics.py` / `model.py`)
**Supersedes:** the heating and residual parts of `2026-06-22-physics-ml-hybrid-design.md`; `2026-11-14-105-go-live-criteria.md` (#105 continues offline only).

## 1. Goals

1. Simplify the code base by removing features that add no value today.
2. Simplify future development: no dormant modes or 0-gain features to keep compiling and testing.
3. Ship a clean public `physics:` config surface in v0.12.0, the first `main` release that contains physics. Nothing to deprecate later.
4. Unblock v0.12.0 and set a reachable v1.0.0 criterion.

**Non-goals:** improving forecast accuracy; changing the DHW tank model maths; renaming the `physics:` block; making `model.py` independent of `physics.py`; evaluating the override correction or open-window down-weighting.

## 2. Evidence

| Part | Live state | Evidence |
|---|---|---|
| `physics_kwh` ML feature | live | 0 splits in the 2026-11-15 model (rank 88/88). Training series constant at about 0.144: `UA_eff` is null, and one room's NaN `current_temp` (from 2026-04-12) freezes the DHW simulation. Its reheats correlate about 0 with the DHW sub-meter. The model already has `dhw_buffer_temp` (rank 2). |
| Phase 2 residual | dormant | After #113: autumn replay worse (hourly +13.8 %, cold-day bias +4.6 kWh/day); winter only −3.1 % on 3 h blocks. Intervals not conformally calibrated. |
| `UA_eff`, `solar_gain_area` | live, null | Only consumers are the heating term → `physics_kwh` / Phase 2. Never passed calibration. Removing the live code does not move the forecast. |
| `physics_base_today`, `ml_adjustment_today` | live | Diagnostic; misleading while `physics_kwh` has 0 gain. |
| Override correction, serving add-back | live | The only physics output that moves the published forecast. |
| Override correction, training subtraction | live code, **inactive** | The latest retrain capture replays 34 overrides and the delta sums to 0.0. The same NaN indoor temperature sets the mask all False. |
| `get_scenario` DHW delta | live, **always 0** | `dhw_schedule_override` reaches `predict()` only through `physics_kwh` (`model.py:1160`). Because of that, EM's `_legionella_physics_present()` (true whenever `model_phase` exists) leaves its 5 kWh cycle out of the v2 coverage denominator (`heat_pump.py:4207`), so v2 is overstated. v1 (the live gate) uses `sum(delta_kwh)` ≡ 0. |

## 3. Decisions

| Part | Decision |
|---|---|
| `physics_kwh`; `predict_series`, `predict_training_series`; Phase 2 (all branches, cold-start gate, `_effective_use_physics_residual`); `UA_eff` / `solar_gain_area` / heating eligibility / heating COP series / zone-boundary check; `_validate_physics_phase1` | **Remove** |
| `physics_base_today`, `ml_adjustment_today` (hef + EM mirrors); `model_phase` attribute | **Remove, no replacement** |
| `cop_sensor` (fetched every retrain, never read) | **Remove** |
| DHW tank simulation, override lookup, training and serving override delta, `override_history`, `set_dhw_schedule`, legionella schedule inference, `recalibrate_physics`, calibration of `Q_base_el` (needed: `Q_dhw_daily` = summer mean − 24·`Q_base_el`), `Q_dhw_daily`, `UA_dhw`; `_cop_formula_value` + `_t_flow_c` (DHW COP); `_t_indoor_for_window` / `_area_weighted_t_indoor` (tank ambient); open-window detection | **Keep, unchanged** |
| `get_scenario` `dhw_schedule` argument | **Accepted and ignored** (DEBUG log). EM always adds its own cycle energy (§5). |

## 4. Design — hef

### 4.1 `apps/energy_forecast/physics.py`

Remove the symbols in §3 and the heating-only branches of `calibrate()`. `calibrate()` keeps the inputs the kept calibrations read (`energy_df`, `weather_df`, `heating_active_df`, `dhw_df`, `away_df`, `ev_df`, `holdout_cutoff`). It drops the `climate_dfs` and `heating_sub_meter_df` inputs, and stops writing `q_base_el_calibrated`, `n_calibration_windows_ua_eff` and `room_thermostats_at_calibration` (their only readers are removed). `train()` keeps `climate_dfs` for the override delta, τ and open-window detection. Leftover keys in `physics_calibration.json` are harmless (merged and written back) and stay.

### 4.2 `apps/energy_forecast/model.py`

- `train()` keeps `physics_model` and today's order: override delta (old calibration) → subtract → τ calibration (`:484`) → `calibrate()` on the corrected frame (`:506`) → `detect_open_windows` (reads τ via `physics._tau_hours`) → weights. It drops the `use_physics_residual` and `heating_sub_meter_df` parameters (caller: `energy_forecast.py:~2005`). Delete the `check_zone_boundary` call (`:500`), the `predict_training_series` call, the `physics_kwh` feature (`_engineer_features` merge, `feature_cols` entry), the residual branches (CV folds, non-log fold, holdout, `_log_transform` toggle → always log), and `physics_phase1_diagnostic`.
- `predict()`, `predict_intervals()`, `_prepare_prediction_X()`: drop `physics_model`, `dhw_schedule_override` and `_last_physics_kwh_series`. This also removes the #113 `_get_scenario_cb` race.
- `predict_scenario()`: one `predict()` call. The DHW block is removed.
- Saved models: the existing fill-0.0 line for `physics_kwh` (`model.py:~1229`) stays, with its log downgraded from WARNING to DEBUG. It is exact because the feature has 0 splits. The weekly retrain drops the column. The `use_physics_residual` meta key is ignored on load.

### 4.3 `apps/energy_forecast/energy_forecast.py`

- `_get_scenario_cb`: stop passing `physics_model` / `dhw_schedule_override`; accept `dhw_schedule` and ignore it.
- `_recalibrate_physics_cb` (`:~1591`): stop passing `climate_dfs` / `heating_sub_meter_df` to `calibrate()`.
- Remove `_publish_physics_sensors`, the discovery entries for both sensors, `_effective_use_physics_residual`, `_model_phase_attr` and the `model_phase` attribute. At every startup, publish an empty retained payload to both sensors' discovery config topics so HA deletes the entities (the existing `_cleanup_legacy_states` does not cover MQTT entities).
- Config defaults: drop `use_physics_residual`, `internal_gains_fraction`, `cop_sensor` (and its fetch, `_physics_cop_df`). Drop any further key only if the plan's key→consumer table shows no reader. `heating_sub_meter_sensor` stays (`heating_season.py`). Stale keys in `apps.yaml` are already ignored silently; remove them from the live `apps.yaml` at deploy.

### 4.4 Other hef files

- `backtest/capture_replay.py`, `backtest/capture_frame.py`, `backtest/audit_features.py`, `backtest/arms.py`, `backtest/walkforward.py` (`WfArm.physics`, `:126`), `backtest/README.md`; tracked `scripts/backtest_regime.py` (`:394`) and `scripts/regime_harness.py`: drop `physics_model` / `physics` from `predict*` / `_prepare_prediction_X` calls and `physics_kwh` handling. The legacy `use_physics_residual` key is dropped where stored `train_kwargs` are loaded (shared by `capture_replay` and `capture_frame`), so existing captures still replay. `train()` call sites are unchanged.
- `const.py`: remove heating-only constants (`UA_EFF_*` …).
- Delete `scripts/analyze_cop_scale.py` and `tests/test_analyze_cop_scale.py` (#105 COP-scale gate tooling). Untracked scripts are left alone.
- Tests: delete tests of removed behaviour (`test_model.py`, `test_physics.py`, `test_energy_forecast_physics_config.py`, `test_physics_features.py`, `test_backtest_arms.py`); keep DHW / override / open-window tests.
- Docs: README (`physics:` = DHW and override modelling; removed keys listed), CHANGELOG `[Unreleased]` (Removed; Fixed: EM scenario contract), ROADMAP (close #92, #103, #105 → offline only, a PASS re-enters as an ML feature; the Phase 2 coverage check; Phase 2 rows; v1.0 section), a "superseded in part" line on the 2026-06-22 spec, MEMORY (`project_v1_release_criteria`; archive `project_physics_kwh_low_importance`).

## 5. Design — EM (ha-energy-manager)

- `loads/legionella_coverage.py` (`v2_coverage`, `:77`) and the duplicate denominator in `loads/heat_pump.py:4207`: drop the `physics` parameter; v2 always adds `cycle_kwh` to the denominator. Delete `_legionella_physics_present` (`:4220`). Drop the `physics_present` detail key and update its consumers (`:4032`, `:5606`) and tests (`test_heat_pump.py` `test_physics_present_*`, `test_legionella_coverage.py`). v1 and comfort-boost coverage are unchanged (`delta_kwh` stays 0). **v2 gates the boost hand-off live** (`heat_pump.py:5583`, `:5612`), so the hand-off becomes stricter from the hef deploy on. That is a correction, owner-accepted 2026-11-16. A replay of the 3 past hand-offs: 10-15 5.54→1.39 and 10-31 3.34→0.72 still pass; 11-09 0.94→0.20 flips to below_min. Restart the v1/v2 review count and watch the first hand-offs with `legionella_report.py`.
- `forecast/consumption_forecast_app.py`: remove the two sensor mirrors and the `model_phase` read, log and pass-through. Add the current-format ids that the publish at `:356-357` creates to the existing `_mqtt_clear_discovery_topics` call (`:272`; its list already holds the legacy ids).
- `dashboard/ha-energy-manager.yaml`: remove the two sensor cards and the "hef Modellphase" row (`:1128`); deploy with `scripts/deploy_dashboard.py`. Delete the now-dead `hef_forecast_today` key (`config.example.yaml:564-566`). EM CLAUDE.md hef-physics section: the compatibility signal becomes the `set_dhw_schedule` / `get_scenario` services, not `model_phase`.
- Deploy order does not matter. Without `model_phase`, the current EM already adds the cycle, which is now correct. The EM change is cleanup.

## 6. Verification

1. Full hef and EM suites green in the dedicated envs.
2. New tests:
   - hef: `get_scenario` with a `dhw_schedule` returns the same frame as without one.
   - hef: replay of an existing capture with legacy `train_kwargs`.
   - hef: the discovery-clear publish runs at every startup (idempotent; no persisted marker).
   - EM: v2 always includes `cycle_kwh`.
3. Capture replay:
   - Predict round trip on saved pre-change models: max |Δ `predicted_kwh`| ≤ 1e-9.
   - Train round trip: expected to diverge (88 → 87 columns); judged as normal retrain jitter.
   - The training override-delta sum on the latest retrain capture stays 0.0.
   - The serving override add-back is outside `predict()` and unchanged; the existing tests cover it.
4. After deploy:
   - `meta.pkl` has no `physics_kwh` after the next retrain.
   - The four entities are gone from HA.
   - Logs are clean (ha-deploy-reviewer).
   - The retrain is judged with the #118 gate and the #114 sensors like any retrain.

## 7. Rollout and release path

1. #124 (already on `train/next`) ships first, in its own window.
2. This hef change ships at the next `/pm evening`, alone.
3. The EM change ships at the next EM window.
4. v0.12.0: merge `dev` → `main` (separate release plan; 13 conflicted files on a trial merge 2026-11-16).
5. v1.0.0: config schema, service signatures (`get_scenario`, `set_dhw_schedule`, `recalibrate_physics`) and sensor names unchanged for 4 weeks after v0.12.0 on `main`. This replaces the Phase 2 criterion.

## 8. Out of scope (follow-ups)

- NaN-robust indoor averaging. It would switch the training subtraction on for 34 overrides and move the target.
- Measured vs simulated override correction: hef models about 1 kWh for a legionella hour, EM measures 5 kWh.
- Open-window down-weighting on/off backtest.
- DHW draw calibration (`Q_dhw_daily` 2.79 vs sub-meter about 0.83 kWh/day).
- Moving open-window detection into `model.py` / decoupling `model.py` from `physics.py`.

## 9. Accepted risks

- The per-hour override correction is no longer visible in HA (it was visible through `physics_base_today`'s `hourly_kwh`); logs only.
- Leftover heating keys stay in `physics_calibration.json`.
- Stricter legionella hand-offs (§5): some boosts move to a later candidate or attempt.
- After the hef deploy, EM may read a stale retained `model_phase` for minutes (until hef's first attribute publish).
- The post-retrain feature-set change (88 → 87 columns) re-draws `colsample_bytree` subsets; this counts as normal retrain jitter.
