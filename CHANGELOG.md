# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

---

## [0.10.2-alpha-16] — 2026-04-16

### Fixed
- `apps/energy_forecast/energy_forecast.py` — replaced all `self.log()` calls with `_LOGGER = logging.getLogger("energy_forecast")`. All messages now appear under the `energy_forecast` logger name instead of `AppDaemon`, making log filtering and structured log routing consistent.
- `apps/energy_forecast/energy_history_backfill.py` — same `self.log()` → `_LOGGER` migration.

### Tests
- 474 passing (unchanged). `test_energy_forecast.py` and `test_scenario_service.py` updated to assert on `caplog` records rather than mock `self.log` calls.

### Deployment
- Merged `fix/logging-consistency` into `dev`. Deployed to HA (Samba upload + AppDaemon restart 2026-04-16).

---

## [0.10.2-alpha-15] — 2026-04-16

### Fixed
- `apps/energy_forecast/const.py` — `DEFAULT_TAU` reduced from 24.0 h to 12.0 h. The previous value overestimated the thermal time-constant before calibration data is available, causing the RC-ODE indoor projection to respond too slowly to setpoint changes. 12 h is a better prior for typical residential buildings.

### Tests
- 474 passing (unchanged).

### Deployment
- Merged `fix/default-tau-12h` into `dev`. Deployed to HA (Samba upload + AppDaemon restart 2026-04-16).

---

## [0.10.2-alpha-14] — 2026-04-15

### Added
- `apps/energy_forecast/weather.py` — `humidity` column fetched from Open-Meteo (`relativehumidity_2m`). Added to `_FEATURES_BASE` in `model.py` (#55). Defaults to 70.0 % when the weather API omits the field.
- `apps/energy_forecast/model.py` — **`thermal_pressure_net`**: solar-compensated thermal pressure — raw thermal pressure reduced by `weighted_solar_gain` (#56). Captures how passive solar gain offsets the building heat deficit before the heat pump acts.
- `apps/energy_forecast/model.py` — **`infiltration_pressure`**: wind × thermal gradient interaction term (#57). High wind combined with large indoor–outdoor temperature difference drives cold-air infiltration, increasing heating demand independently of the modelled U-value losses.
- `apps/energy_forecast/model.py` — **`defrost_risk`**: humidity-scaled Gaussian centred at +2°C (#58). Acts as a proxy for heat-pump defrost cycles, which briefly reverse the refrigerant loop and spike power draw.
- `apps/energy_forecast/energy_forecast.py` — SHAP labels for all four new physics features.
- `tests/test_physics_features.py` — 4 new tests covering each physics feature independently (humidity default, `thermal_pressure_net`, `infiltration_pressure`, `defrost_risk` Gaussian peak).

### Tests
- 474 passing (4 new in `test_physics_features.py`).

### Deployment
- Merged `feat/physics-features` into `dev`. Deployed to HA (Samba upload + AppDaemon restart 2026-04-15).

---

## [0.10.2-alpha-13] — 2026-04-14

### Fixed
- `apps/energy_forecast/ha_data.py` — `fetch_recent_sub_sensor` / `fetch_sub_sensor_history`: pandas 3.x `merge_asof` raised `Incompatible merge dtype, dtype('O') and dtype('<M8[us]')` whenever the raw HA fetch returned empty data and `program_entity_id` was configured. Fix on three levels: (1) `_resolve_programs_for_series` wraps timestamps in `pd.to_datetime()` defensively; (2) both fetch functions now build their empty fallback `DataFrame` with explicit `dtype` per column instead of column-names-only. Washer and dryer were failing every hour since 13:01 today.

### Tests
- 470 passing (2 new regression tests covering both fetch paths with empty raw HA + cache + `program_entity_id`).

### Deployment
- Merged `fix/subsensor-dtype-empty-raw` into `dev`. Deployed to HA.

---

## [0.10.2-alpha-12] — 2026-04-14

### Changed
- `apps/energy_forecast/energy_forecast.py` — Extended `_SHAP_FEATURE_LABELS` to cover all 62 model features (up from 23). Constant-only change; no behaviour difference.

### Deployment
- Merged `feat/shap-labels-complete` into `dev` (fast-forward). Deployed to HA. 468 tests passing.

---

## [0.10.2-alpha-11] — 2026-04-14

### Deployment
- Merged `feat/thermal-cop-safeguards` into `dev`. Deployed to HA; AppDaemon restart clean, 0 `energy_forecast` errors. 468 tests passing.

---

## [0.10.2-alpha-10] — 2026-04-14

### Added
- `apps/energy_forecast/model.py` — **`weighted_solar_gain`**: `direct_radiation_wm2` scaled by a half-cosine window peaking at 13:00, zero outside 09:00–17:00. Captures south-facing passive solar gain that reduces actual heating demand even when `thermal_pressure` looks high.
- `apps/energy_forecast/model.py` — **`thermal_pressure_cop`**: `thermal_pressure / max(0.5, 0.11 × T_out + 3.0)`. Expresses heat debt as electrical urgency — the same 1°C deficit costs ~2× more electricity at −10°C than at +10°C outdoors. Denominator clamped at 0.5 to guard against extreme negative temperatures.
- `apps/energy_forecast/energy_forecast.py` — SHAP labels for `thermal_pressure_cop` and `weighted_solar_gain`.

### Changed
- `apps/energy_forecast/model.py` — **τ calibration safeguards** in `_calibrate_tau()`:
  - *Daytime exclusion*: passive-cooling windows that contain any timestamp in 09:00–15:00 are now rejected (solar gain and occupancy patterns during these hours corrupt the passive-cooling ODE fit).
  - *Solar radiation mask*: windows where `direct_radiation_wm2 > 150 W/m²` at any point are rejected. Degrades gracefully when `weather_df` lacks the radiation column (mask skipped, no crash).
  - *EMA smoothing*: when a new τ estimate differs from the stored value by more than 50%, the result is blended as `0.8 × old + 0.2 × new` to prevent single-batch jitter from propagating into the 48-hour forecast.

### Tests
- 468 passing (15 new covering solar gain shape, COP denominator clamp, daytime/radiation window exclusion, EMA blend thresholds; 2 existing tests updated for the new daytime exclusion filter).

---

## [0.10.2-alpha-9] — 2026-04-14

### Added
- `apps/energy_forecast/model.py` — **Indoor Temperature Projection**: new `_project_indoor_temps()` pure function uses an RC-ODE (Euler forward integration) to project indoor temperatures for all 48 forecast hours based on outdoor forecast. Eliminates the zero-fill problem where `thermal_pressure` defaulted to 0 for hours >2h ahead due to stale `climate.current_temperature` sensor data. Starting indoor temperature uses the most-recent `current_temp` if <2h old; falls back to setpoint for stale sensors.
- `apps/energy_forecast/model.py` — **Area-Weighted Thermal Pressure**: rooms are now weighted by floor area via new `climate_room_areas` config key (dict mapping entity_id → m²). Rooms without explicit area default to 15 m². Thermal pressure is expressed in °C·h (area-weighted cumulative deficit; τ-division removed for clearer physical interpretation).
- `apps/energy_forecast/model.py` — **Secondary thermal features**: `thermal_pressure_max` (largest per-room deficit) and `thermal_pressure_std` (room temperature imbalance spread across rooms).
- `apps/energy_forecast/const.py` — constants `DEFAULT_TAU = 24.0` (building thermal time constant fallback) and `DEFAULT_ROOM_AREA_M2 = 15.0` (per-room area default).
- `apps/energy_forecast/energy_forecast.py` — SHAP labels for all three thermal pressure features; new `_parse_room_areas()` helper validates `climate_room_areas` config entries.
- `apps.yaml.example` — `climate_room_areas` example configuration.

### Tests
- 453 passing (12 new covering RC-ODE convergence, stale-sensor fallback, area weighting, max/std features, zero-fill elimination; 1 existing test updated for removed τ-division).

---

## [0.10.2-alpha-8] — 2026-04-12

### Fixed
- `apps/energy_forecast/ha_data.py` — `_resolve_programs_for_series()`: increased forward-lookup tolerance from 1 h to 2 h. The 1 h window was insufficient when the program sensor fires just into the following hour (e.g. cycle starts at 21:00, sensor fires at 22:05 → 65-minute gap). The forward fallback still only activates when backward LVFC gives `""` or `"no_program"`, so labels for running cycles are unaffected.

---

## [0.10.2-alpha-7] — 2026-04-12

### Fixed
- `apps/energy_forecast/ha_data.py` — `_resolve_programs_for_series()`: program sensor events that fire slightly *after* the hour boundary (e.g. user starts machine at 12:05, hourly row stamped 12:00) were previously mislabelled with the preceding idle state (`"no_program"`). A 1-hour forward-lookup pass now substitutes the real program label when backward LVFC gives `""` or `"no_program"` and a non-idle program event exists within the next hour. The forward pass is skipped when backward already carries a real label, so the tail of a running cycle is never overwritten by the next cycle's program. (4 new tests.)

---

## [0.10.2-alpha-6] — 2026-04-12

### Changed
- `apps/energy_forecast/ha_data.py` — sub-sensor CSVs (`sub_<id>.csv`) now include a `program` column: the program label active at each hourly timestamp, resolved via last-value-carry-forward (`pd.merge_asof`). Hours before the first program event get an empty string. Backward-compatible: existing CSVs without the column are loaded and extended transparently.
- `apps/energy_forecast/ha_data.py` — new `_resolve_programs_for_series()` helper encapsulates the last-value-carry-forward logic; `fetch_sub_sensor_history()` and `fetch_recent_sub_sensor()` accept a new optional `program_entity_id` param that triggers program label resolution and persistence.
- `apps/energy_forecast/ha_data.py` — `_merge_sub_sensor_frames()` extended to preserve non-empty program labels from the cache when the fresh window doesn't cover older rows (winner's label wins; loser's label fills gaps; column only added when at least one frame has it).
- `apps/energy_forecast/energy_forecast.py` — `_retrain()` now derives `program_histories` from the persisted CSV `program` column instead of a separate 30-day `fetch_program_sensor_history()` call. Program attribution accumulates indefinitely rather than being capped to the last 30 days.
- `apps/energy_forecast/energy_forecast.py` — `_update_sensors()` threads `program_entity_id` into `fetch_recent_sub_sensor()` so every hourly write also captures the current program state.

### Tests
- 436 passing (11 new: `TestResolveProgramsForSeries` × 5 covering LVFC correctness, empty input, index preservation; `TestFetchSubSensorHistoryWithProgram` × 6 covering column written to CSV, label resolution, backward compat, cached-label preservation, recent fetch).

---

## [0.10.2-alpha-5] — 2026-04-12

### Added
- `apps/energy_forecast/model.py` — **Program-type sensor support**: `_learn_appliance_signatures()` now accepts a `program_histories` dict mapping timestamps to program labels; cycles are grouped per program and per-program hourly profiles stored under `sig["programs"][<label>]` (with `hourly_profile`, `std_profile`, `total_kwh`, `n_cycles`). Requires ≥ 2 cycles per program to avoid overfitting single runs. Backward-compatible: without program states, behavior is identical to alpha-4.
- `apps/energy_forecast/model.py` — `_composite_forecast()` scenario API accepts dict-form schedule entries `{"start": "HH:MM", "program": "<label>"}`. When a program label is provided the matching per-program profile is used; falls back gracefully to the combined signature when unknown or not specified.
- `apps/energy_forecast/energy_forecast.py` — `train()` fetches program-type sensor history for all sub-sensors that have `program_type_sensor` configured and passes it to `_learn_appliance_signatures()`.
- `apps/energy_forecast/ha_data.py` — new `fetch_program_type_history()` retrieves categorical state history for program-type sensors (e.g. `sensor.washing_machine_program`) over the training window, returning a DataFrame of timestamp → program label for cycle labelling.

### Tests
- 425 passing (19 new covering per-program signature learning, minimum-cycle filtering, per-program scenario profile selection, graceful fallback to aggregate signature, and config schema validation for the new `program_type_sensor` key).

---

## [0.10.2-alpha-4] — 2026-04-11

### Added
- `apps/energy_forecast/model.py` — `_learn_appliance_signatures()`: each signature now includes quality metadata: `cov` (coefficient of variation of cycle total_kwh), `reliability` (`"high"` / `"medium"` / `"low"` based on CoV thresholds), `n_cycles` is now also stored per cluster. Duration clusters (`short`, `long`) each carry `cov` and `reliability` independently, so mixed-program appliances (e.g., quick-wash vs. cotton) report accurate reliability per mode.
- `apps/energy_forecast/model.py` — CoV-based clustering: cycles are split into CoV groups and a Jenks-like median-split heuristic assigns `"high"` (CoV < 0.25), `"medium"` (0.25–0.5), or `"low"` (> 0.5) reliability labels, enabling downstream consumers (scenario API, dashboards) to suppress or weight signatures by confidence.

### Tests
- 406 passing (5 new tests: `TestApplianceSignaturesMetadata` covering CoV computation, reliability labelling thresholds, per-cluster metadata, and JSON serialization round-trip).

---

## [0.10.2-alpha-3] — 2026-04-11

### Added
- `apps/energy_forecast/const.py` — new `APPLIANCE_MAX_WINDOW_HOURS = 12` constant caps the adaptive cycle window to prevent runaway detection.
- `apps/energy_forecast/model.py` — `_learn_appliance_signatures()`: **adaptive window** — cycle end is now detected at the first return to idle instead of a fixed 4-hour cutoff. The `window_hours` parameter is removed. Fixes truncated profiles for long-duration appliances (tumble dryer cotton, DHW heating).
- `apps/energy_forecast/model.py` — `_learn_appliance_signatures()`: **demand-surge detection** — appliances with < 5 % zero-hours (e.g. heat pump heating in winter) use the 25th-percentile consumption as idle baseline; cycle starts/ends are upward/downward threshold crossings instead of strict 0→>0 transitions. Eliminates "no cycles found" for always-on devices.
- `apps/energy_forecast/model.py` — `_learn_appliance_signatures()`: **outlier cycle rejection** — cycles with total_kwh > median + 2σ are discarded before averaging (≥ 4 cycles required). Stored as `n_rejected` in the signature JSON.
- `apps/energy_forecast/model.py` — `_learn_appliance_signatures()`: **duration clustering** — cycles with median duration ≥ 2 h are split into `short` and `long` clusters via median-duration split. Each cluster stores its own `hourly_profile`, `std_profile`, and `total_kwh`. CoV warnings are evaluated per cluster, eliminating false positives caused by mixed programs (quick wash vs. cotton 90°, ECO dishwasher vs. intensive). Top-level keys preserved for backward compatibility; `_composite_forecast()` unchanged.

### Tests
- 401 passing (11 new in `TestApplianceSignatures` covering adaptive window, demand-surge detection, outlier rejection, duration clustering, CoV warning suppression, and backward-compatible JSON schema).

---

## [0.10.2-alpha-2] — 2026-04-11

### Fixed
- `apps/energy_forecast/ha_data.py` — `_fetch_history()` now maps `"on"` → `1.0` and `"off"` → `0.0` before attempting `float()`. Previously, `input_boolean` entities (e.g. `heating_system_active_entity: input_boolean.heizung_wintermodus`) caused `float("on")` to raise `ValueError` which was silently swallowed, returning an empty DataFrame and logging "No history found" despite HA having the data. τ calibration was therefore never receiving any `heating_active_df`. Two regression tests added.
- `apps/energy_forecast/energy_forecast.py` — `_update_sensors()` now calls `fetch_recent_generic_sensor()` for `heating_system_active_entity` each prediction cycle, mirroring the DHW pattern. Previously the CSV cache was only written during training; HA's short history window (~10 days) meant the cache would drift stale between weekly retrains.
- `apps/energy_forecast/model.py` — `_calibrate_tau()`: passive-cooling windows are now trimmed at the first hour where T_indoor ≤ T_outdoor (instead of rejecting the whole block). In spring, a single warm afternoon hour where outdoor temp briefly reaches indoor temp was discarding otherwise-valid multi-hour cooling curves.
- `apps/energy_forecast/model.py` — `_calibrate_tau()`: windows are additionally trimmed at the first hour where ΔT stops declining (instead of rejecting on slope ≥ 0). Solar gain during off-periods pushes indoor temps up mid-block; the initial decay portion remains physically valid for the OLS fit.
- `apps/energy_forecast/model.py` — `_calibrate_tau()`: replaced prefix-trim approach with a full sub-sequence scan. Each off-block is now scanned for all maximal contiguous intervals where `delta > 0` AND `diff(delta) < 0` simultaneously; each qualifying run of ≥ 2 hours is evaluated independently. This captures evening cooling windows (the most valuable signal when heating turns off in the morning and solar gain dominates the early hours) that the prefix-trim approach discarded entirely.

### Tests
- 390 passing (3 new: `test_fetch_history_boolean_states`, `test_fetch_history_numeric_states_unchanged`, `test_evening_cooling_window_used`)

---

## [0.10.2-alpha-1] — 2026-04-11

### Added
- `apps/energy_forecast/model.py` — **#55 Verified Passive Decay (τ calibration)**: new `_calibrate_tau()` method fits log-linear OLS on passive-cooling windows (contiguous periods where the heating system is confirmed off) to estimate the building thermal time constant τ (hours). Result is the median τ across all qualifying windows (minimum 3). τ is used to scale `thermal_pressure` by `1/τ`, expressing heating urgency in °C/h rather than a static °C delta — more physically meaningful and transfer-learned between climates. τ persisted in `meta.pkl` so it survives AppDaemon restarts without re-fitting.
- `apps/energy_forecast/energy_forecast.py` — fetches `heating_system_active_entity` history (30 days, binary 0/1) and passes it to `train()` as `heating_active_df` for τ calibration. Fully backward-compatible: if `heating_system_active_entity` is not configured, calibration is skipped and `thermal_pressure` remains unchanged.

### Tests
- 387 passing (5 new since v0.10.1: `TestCalibrateTau` covering the OLS fit, sanity bounds, minimum-window guard, non-positive-delta skip, and rising-temperature skip)

---

## [0.10.1] — 2026-04-10

### Added
- `apps/energy_forecast/energy_forecast.py` — new `baseline_included_sensors` config key (list of HA sensor entity IDs). When `baseline_mode: true` is set, only sub-sensors NOT in `baseline_included_sensors` are subtracted from the training target; sensors listed there (e.g., heating and DHW) remain in the baseline model target. Previously all `sub_energy_sensors` were subtracted. This allows the model to learn heating/DHW consumption as a baseline pattern (leveraging thermal/weather/occupancy features) while still removing schedulable appliances for clean scenario modeling.

### Fixed
- `apps/energy_forecast/ha_data.py` — `_merge_frames()` now explicitly coerces the value column back to `float64` after `pd.concat()`. In pandas 3.0.2, `pd.concat([float64_df, empty_object_df])` promotes the result to `object` dtype even when the empty DataFrame contributes no rows. This caused all lag feature columns to become object dtype, which LightGBM rejected with "pandas dtypes must be int, float or bool". Fix: `result[value_col] = pd.to_numeric(result[value_col], errors="coerce")` applied after concat in `_merge_frames`.
- `apps/energy_forecast/model.py` — coerce feature matrix to `float64` via `.astype(float)` before fitting, providing a belt-and-suspenders guard against dtype incompatibility errors when `int32` or pandas extension types appear in the feature columns.
- `apps/energy_forecast/energy_forecast.py` — retrain exception handler now logs the full traceback (`traceback.format_exc()`) alongside the error message, making dtype and other unexpected retrain failures diagnosable from HA logs without needing a debugger.

### Tests
- 382 passing (5 new since v0.10.0: `TestBaselineIncludedSensors` covering selective inclusion logic, and `test_empty_winner_preserves_float_dtype` in `TestMergeEnergyFrames` for the pandas 3.x dtype regression)

---

## [0.10.0] — 2026-04-10

### Added
- `apps/energy_forecast/energy_forecast.py` — **Stage 1: Passive House Baseline mode**: new `baseline_mode` config flag (default `false`). When enabled, the model trains on and predicts only the household baseline by subtracting all controllable `sub_energy_sensors` from the total consumption target before training, and mirrors that subtraction at prediction time. Keeps appliance noise out of the baseline model so `predict_scenario()` deltas are meaningful. Also wires `presence_sensors` config into the hourly update and retrain cycle (model feature `people_home` was added in v0.9.0; Stage 1 connects the config key end-to-end).
- `apps/energy_forecast/ha_data.py` + `model.py` — **Stage 2: Intent-Driven Thermal & DHW Modeling**: new `fetch_climate_history()` reads setpoint and current temperature from any HA `climate` entities listed in `climate_entities`; new `fetch_generic_sensor_history()` reads a DHW buffer temperature sensor. `model.py` derives two new features: `thermal_pressure` (mean of setpoint − current-temp across all configured climate entities; 0 when none configured) and `dhw_pressure` (heat-loss urgency score `1 / (max(0.5, buffer_temp − 40) + 1)²`; 0 when not configured). Both features are added to `_FEATURES_BASE` and the NaN-fill loop. Config keys: `climate_entities` (list), `dhw_buffer_sensor`, `heating_system_active_entity`.
- `apps/energy_forecast/model.py` — **Stage 3: Automated Load Signature Discovery**: new `_learn_appliance_signatures()` scans sub-sensor histories for run cycles (0→>0 kWh transitions) and computes an average hourly energy profile per appliance prefix. Signatures persisted to `models/appliance_signatures.json`, reloaded on startup, and ready for Stage 4 scenario modeling. Per-prefix log: `Appliance signature | <prefix> | cycles=N | total=X.XX kWh | peak_hour=H`. 8 new unit + integration tests in `TestApplianceSignatures`.
- `apps/energy_forecast/model.py` — **Stage 4: Scenario Modeling & What-If API**: new `_composite_forecast()` pure function overlays learned appliance hourly run profiles onto a baseline 48h forecast given a schedule dict `{prefix: "HH:MM" | "off" | None}`. New `predict_scenario()` method on `EnergyForecastModel` wraps `predict()` + composite overlay, returning `[timestamp, predicted_kwh, delta_kwh]`. Enables the Energy Manager to ask "what is the expected import if dishwasher runs at 14:00?" and receive a delta-annotated composite forecast instantly.
- `apps/energy_forecast/energy_forecast.py` — **Scenario service**: AppDaemon service `energy_forecast/get_scenario` registered in `initialize()`. Callback `_get_scenario_cb()` validates cached inputs, calls `predict_scenario()`, fires `energy_forecast_scenario_result` event with full 48h records. `publish=True` additionally writes `sensor.energy_forecast_scenario_today`, `_tomorrow`, `_delta_today`, and 8 block sensors via `_publish_scenario_forecast()`. Input cache (`_cached_forecast_df`, `_cached_live_temp`, etc.) populated each `_update_sensors()` cycle for zero-latency scenario re-use.

### Fixed
- `apps/energy_forecast/requirements.txt` — added `holidays>=0.46` explicitly; absence caused a silent holiday-feature fallback and `sensor.energy_forecast_setup_status` showing `missing_packages` on fresh installs.
- `apps/energy_forecast/model.py` / `energy_forecast.py` — replaced deprecated `datetime.utcnow()` calls with `datetime.now(timezone.utc)` throughout.
- `apps/energy_forecast/model.py` — eliminated pandas `PerformanceWarning` in `_add_lag_and_rolling_training()` and `_add_sub_sensor_lags_training()` by accumulating new columns in a dict and calling `pd.concat` once instead of assigning per-column in a loop.
- `apps/energy_forecast/model.py` — **DS-1**: appliance signatures now store `std_profile` (per-hour standard deviation across all observed cycles alongside `hourly_profile`); logs a `WARNING` when the coefficient of variation exceeds 0.5, flagging high-variability appliances (e.g. oven) where a fixed mean profile is unreliable.
- `apps/energy_forecast/model.py` — **DS-2**: scenario time-slot offset now uses `int()` floor instead of `round()`, ensuring deterministic slot assignment (e.g. "00:30" always maps to offset 0, not randomly 0 or 1 depending on floating-point rounding).
- `apps/energy_forecast/ha_data.py` / `model.py` — **SE-3**: replaced three broad `except Exception` clauses with narrower types: `OSError`/`ValueError` in `validate_energy_cache`, `KeyError`/`TypeError` in `_add_holiday_feature`, and specific LightGBM exception types in the early-stopping fallback paths.

### Tests
- 377 passing (25 new since Stage 3: 8 for Stage 3 `TestApplianceSignatures`, 13 for Stage 4 `TestCompositeForecast` × 10 + `TestGetScenarioCb` × 3, plus T3b/T4/T5 regression tests and `TestConcurrency` from DS-1/DS-2/SE-3 review fixes)

---

## [0.10.0-alpha] — 2026-04-09

### Added
- `apps/energy_forecast/energy_forecast.py` — **Timezone generalisation + holiday country**: `timezone` apps.yaml key is now fully wired up (was silently ignored); all 18 hardcoded `"Europe/Zurich"` strings replaced with `self._timezone`. New `holiday_country` config key (ISO 3166-1 alpha-2, default `CH`) propagated through `train()` → `_engineer_features()` → `_add_holiday_feature()`, replacing three hardcoded `"CH"` references.

### Fixed
- `apps/energy_forecast/energy_forecast.py` — **Concurrency**: `_update_cb` no longer acquires the training lock; hourly sensor updates always run against the last-good model during a background retrain, eliminating the ~60-second sensor-silent window that previously occurred when retraining and an update raced.
- `apps/energy_forecast/ha_data.py` — **CSV tail read**: `fetch_recent_energy()` now uses `deque(maxlen=400)` to read the last 400 rows, reducing memory from O(all rows) to O(400) per hourly call.
- `apps/energy_forecast/energy_forecast.py` — **Relative MAE sensors volatility** (`mae_7d_pct`, `mae_30d_pct`): relative MAE sensors now remain stable across AppDaemon restarts. Root cause was loss of `_pred_history` and `_actuals_history` dicts on restart, causing ~24h recovery period with very low `n_pairs` and high volatility. Implemented JSON persistence layer: `_load_pred_history()` reads `pred_history.json` at startup (with 30-day pruning), `_save_pred_history()` atomically writes JSON after each forecast cycle. Includes 7 new tests covering roundtrip, pruning, keep-first semantics, error handling, and atomic writes.
- `apps/energy_forecast/mqtt_mixin.py` — **MQTT NaN handling in relative MAE**: relative MAE percentages are now safely published even when mean consumption is zero or undefined (e.g., first day of month). Prevents NaN/inf from propagating to MQTT and breaking Lovelace graphs. Explicitly sets value to 0.0 and logs WARNING.
- `apps/energy_forecast/mqtt_mixin.py` — **UID slicing fragility**: topic splitting for sub-entity UID extraction was fragile to extra colons or missing segments. Replaced naive `split(':')[X]` with robust parsing that validates segment count before slicing and logs DEBUG for dropped malformed UIDs.

---

## [0.9.1-alpha] — 2026-04-07

### Added
- `apps/energy_forecast/energy_forecast.py` — relative MAE sensors (`mae_7d_pct`, `mae_30d_pct`) express rolling MAE as a percentage of mean consumption, providing a normalized accuracy metric independent of consumption scale. Useful for comparing forecast accuracy across seasons (heating/cooling) and across households. Implements same persistence logic as absolute MAE sensors.
- `apps/energy_forecast/shap_analysis.py` — SHAP narrative attribute (`explanation`) on `sensor.energy_forecast_shap` provides a human-readable "Why today?" summary via SHAP `force_plot` interpretation. Formats feature contributions (base value + top N positive/negative pushes) in a single-line narrative, published as an entity attribute for display in automations/notifications.

### Fixed
- `apps/energy_forecast/energy_forecast.py` — rolling MAE sensors (`mae_7d`, `mae_30d`) now remain stable across AppDaemon restarts. Root cause was loss of `_pred_history` and `_actuals_history` dicts on restart, causing ~24h recovery period with very low `n_pairs` and high volatility. Implemented JSON persistence layer: `_load_pred_history()` reads `pred_history.json` at startup (with 30-day pruning), `_save_pred_history()` atomically writes JSON after each forecast cycle. Includes 7 new tests covering roundtrip, pruning, keep-first semantics, error handling, and atomic writes.
- `apps/energy_forecast/mqtt_mixin.py` — relative MAE percentages are now safely published even when mean consumption is zero or undefined (e.g., first day of month). Prevents NaN/inf from propagating to MQTT and breaking Lovelace graphs. Explicitly sets value to 0.0 and logs WARNING.
- `apps/energy_forecast/mqtt_mixin.py` — topic splitting for sub-entity UID extraction was fragile to extra colons or missing segments. Replaced naive `split(':')[X]` with robust parsing that validates segment count before slicing and logs DEBUG for dropped malformed UIDs.

---

## [0.9.0] — 2026-04-10

Promoted from `dev` → `main`. First stable release with thermal modelling, occupancy, and rolling MAE persistence.

### Summary
- Thermal modelling features (#49–#52): `temp_ewma_24h/72h`, `heating_deg_sum_24h/168h`, `temp_delta_1h/24h`, `temp_lag_24h/168h`
- Occupancy feature (#21): `people_home` integer count via `presence_sensors`
- Relative MAE sensors (`mae_7d_pct`, `mae_30d_pct`) — normalized accuracy independent of consumption scale
- SHAP narrative attribute (`shap_narrative`) on `sensor.energy_forecast_today`
- Rolling MAE persistence via `pred_history.json` — survives AppDaemon restarts
- MQTT NaN guard for relative MAE and UID-slicing hardening

See [[0.9.1-alpha]] and [[0.9.0-alpha]] for detailed per-change descriptions.

---

## [0.9.0-alpha] — 2026-04-02

### Added
- **Thermal modeling features** (`model.py`): eight new engineered features improve heating season
  accuracy by capturing thermal inertia, outdoor temperature trends, and lagged temperature effects:
  `temp_ewma_24h`, `temp_ewma_72h` (exponential weighted moving averages),
  `heating_deg_sum_24h`, `heating_deg_sum_168h` (heating degree sums below 15°C base),
  `temp_delta_1h`, `temp_delta_24h` (short/medium-term deltas),
  `temp_lag_24h`, `temp_lag_168h` (multi-day temperature lags). Documented in #49–#52.
- **Occupancy sensor** (`model.py`): new optional `people_home` feature counts number of people
  present using Home Assistant `person` or `device_tracker` entities (configurable via
  `presence_sensors` in `apps.yaml`). Enables occupancy-correlated consumption patterns (lighting,
  HVAC, cooking). Documented in #21.

### Tests
- 307 tests passing (7 new tests covering thermal features, occupancy sensor, and feature
  engineering edge cases).

### Known Issues
- **SolarEdge entity lookup warnings** (out of scope): entity ID resolution errors for SolarEdge
  sensors originate from `ha-energy-manager` config and do not affect forecast operation.
- **SRG-SSR API 429 error** (observation): one rate-limit error logged during testing despite
  geolocation caching (v0.8.1). Cache verification and monitoring ongoing.

---

## [0.8.1] — 2026-04-01

### Fixed
- **SRG-SSR API quota over-consumption** (`weather.py`): geolocation lookups are now cached at
  module scope (aux storage alongside OAuth token), reducing daily API quota from ~48 calls
  (uncached) to ~24 calls (cached). Fixes repeated 429 rate-limit errors during normal operation
  and aligns with Freemium tier cap (50 calls/day). Includes 1 regression test.

### Tests
- `test_srg_geolocation_caching`: verifies that repeated calls to `_get_srg_geo_id` reuse the
  cached `geo_id` instead of making duplicate API requests.

---

## [0.8.0] — 2026-03-31

### Fixed
- **Temperature sensor blending bias-fade** (`model.py`): the 6-hour outdoor sensor blend now uses
  bias-fade semantics (`temp = forecast[h] + bias * (1 - alpha)`) instead of linear interpolation,
  preserving the forecast's temperature trajectory while smoothly fading the current sensor offset.
  This prevents loss of forecast warming/cooling signals in the blend zone.

### Added
- **Model versioning** (`model.py`, `energy_forecast.py`): before overwriting model files each
  weekly retrain, the previous snapshot is archived to `models/archive/<timestamp>/`. Configurable
  via `model_archive_count` (default 3); set to 0 to disable. Roll back by firing HA event
  `energy_forecast_rollback_model` — sensors update automatically. Includes 6 unit tests
  (`TestModelVersioning`).
- **CSV health checks** (`ha_data.py`): `validate_energy_cache()` runs after every weekly
  retrain merge and logs WARNINGs for: non-monotonic timestamps, gaps > 2 h (with DST note),
  and out-of-range `gross_kwh` values that survived the spike filter. Never raises. Includes 7 unit
  tests + 1 integration test (`TestValidateEnergyCache`, `TestValidateCacheIntegration`).
- **Solar PV + battery target correction** (`energy_forecast.py`): four new optional config keys
  (`solar_production_sensor`, `grid_export_sensor`, `battery_charge_sensor`,
  `battery_discharge_sensor`) allow the training target to be corrected from raw grid import
  to true household consumption:
  `total_consumption = grid_import − grid_export + solar_production − battery_charge + battery_discharge`.
  All sensors must be cumulative kWh entities (`state_class: total_increasing`). Any subset of
  the four sensors can be configured independently; no hardware is required to merge the branch.
  Documented in `apps.yaml.example` with SolarEdge Modbus Multi and Enphase Envoy examples.
  Includes 9 unit tests for `_apply_target_correction`.

### Tests
- `TestBuildPredictionTempDf`: 7 new tests covering temperature blending zones (full-trust, blend,
  forecast), trajectory preservation, bias fade semantics, and fallback behavior.
- `TestModelVersioning`: 6 tests covering model archive creation, rollback, and pruning.
- `TestValidateEnergyCache` + `TestValidateCacheIntegration`: 8 tests for CSV health validation.
- `TestApplyTargetCorrection`: 9 tests for solar/battery target correction logic.

---

## [0.7.2] — 2026-03-27

### Fixed
- **Sub-sensor `object` dtype crash** (`model.py`): sub-sensor reindex now wrapped in
  `pd.to_numeric(..., errors='coerce')` so sensors with mostly-NaN history (e.g. tumbler)
  produce `float64` lag columns instead of `object`, unblocking LightGBM retraining.
- **SRG mixed-timezone error at DST boundary** (`weather.py`): `ValueError` from
  mixed-offset SRG timestamps (spring-forward +01:00/+02:00 mix) is now caught and
  re-parsed with `utc=True`; existing naive-timestamp path preserved.

### Tests
- `test_mixed_offset_srg_timestamps_do_not_raise`: covers the spring-forward timestamp
  mix that triggered the live error.

### Docs
- **README.md `init_commands` section**: expanded with Alpine/armv7 context, pip cache
  note, and corrected formatting.

---

## [0.7.1] — 2026-03-24

### Fixed
- **404 DELETE spam on startup** (`energy_forecast.py`): `_cleanup_legacy_states` now guards
  each `remove_entity` call with `entity_exists`, eliminating ~30 `[404] HTTP DELETE: Not Found`
  log errors on fresh installs where legacy entities were never created (fixes #47).
- **Anomaly binary sensor attributes missing in MQTT mode** (`energy_forecast.py`): `_publish`
  now publishes the four anomaly attributes (`residual_kwh`, `residual_std_kwh`,
  `sigma_threshold`, `n_pairs`) to a dedicated `binary_sensor/.../attributes` MQTT topic.
  Discovery payload for `energy_forecast_unusual_consumption` now includes
  `json_attributes_topic`. State topic path corrected from `sensor/` to `binary_sensor/`.
- **`_mqtt_publish_sensor_attributes` category param**: method now accepts `category`
  (default `"sensor"`) so it can be routed to `binary_sensor/` paths.

### Added
- **Dashboard card — anomaly detection** (`dashboard/anomaly-detection.yaml`): standalone
  vertical-stack with mushroom state card + conditional attribute detail (expands when ON).
- **Dashboard card — SHAP feature importance** (`dashboard/shap-importance.yaml`): native
  Lovelace markdown card with Jinja2 template; no custom card dependency.
- **`dashboard/dashboard.yaml`**: anomaly mushroom card inserted after MAE mini-graph card.

### Tests
- `test_publish_mqtt_mode`: extended to verify anomaly attributes topic and payload keys.
- `test_mqtt_discovery_includes_anomaly_sensor`: extended to verify `json_attributes_topic`
  is present in the binary sensor discovery config payload.

### Docs
- **`assets/` folder**: logo (light + dark), icon (light + dark), and two dashboard screenshots
  added to the repository.
- **README.md**: project logo at top (dark-mode aware via `<picture>`), version badge updated
  to v0.7.1, new **Dashboard** section with side-by-side screenshots of the forecast overview
  card and the SHAP feature importance table.

---

## [0.7.0] — 2026-03-23

### Fixed
- **`_load_interval_correction` stale-value bug** (`model.py`): `_interval_correction` is now
  reset to `0.0` before attempting to parse the JSON file, so a corrupted or unreadable file
  can never leave a stale value from a previous run in place.  `json.JSONDecodeError` added
  explicitly to the except tuple (it is a `ValueError` subclass, so behaviour is unchanged,
  but intent is clearer).
- **SHAP summary early-day fallback** (`model.py`): `shap_summary` now falls back to all 48
  prediction rows when fewer than 3 rows match today's date slice (previously fell back only
  when zero rows matched, producing a misleading 1-hour average late in the day).

### Tests
- `TestPredictIntervals::test_calibrated_intervals_wider_than_raw`: added `assert m._log_transform`
  guard so the interval-widening assertion cannot pass vacuously if log-transform is disabled.
- `TestAwayFeature::test_predict_with_away_series`: rewritten to verify `is_away` propagation
  at the feature-matrix level via `_prepare_prediction_X`, proving a regression that silently
  ignores `away_series` would be caught.



### Added
- **Quantile interval calibration (CQR)** (`model.py`): prediction intervals (10th–90th percentile)
  are now calibrated via split conformal prediction (Conformalized Quantile Regression).  The last
  15% of training rows (≥ 20) are held out as a calibration split; q10/q90 are trained on the
  remaining 85%.  A conformity score `max(q10(x)−y, y−q90(x))` is computed in log-space for each
  calibration row, and the empirical `⌈(n+1)·0.8⌉/n` quantile (`q_hat`) is applied as a symmetric
  additive correction before `expm1` at predict time.  This gives ≥80% marginal coverage on
  held-out data.  `q_hat` is persisted to `energy_model_interval_correction.json` and reloaded on
  startup.  A log line `CQR correction: q_hat=<value> (cal_n=<N>)` is emitted after every retrain.
- **SHAP feature importance (#42)** (`model.py`, `energy_forecast.py`): the top-N driving features
  behind each prediction are exposed as a `shap_top_features` attribute on
  `sensor.energy_forecast_today`.  LightGBM uses native TreeSHAP (`pred_contrib=True`) for
  per-prediction values; sklearn GBR falls back to global `feature_importances_`.  Features are
  ranked by mean absolute contribution over today's prediction slice and returned as a
  `{feature_name: importance}` dict (descending).  New config key: `shap_top_n` (int ≥ 0,
  default 5; set to 0 to disable).  MQTT Discovery mode publishes attributes via
  `json_attributes_topic` on the `energy_forecast_today` discovery payload.
- **Anomaly detection sensor (#39)** (`energy_forecast.py`): new binary sensor
  `binary_sensor.energy_forecast_unusual_consumption` fires when the latest actual consumption
  deviates more than `anomaly_sigma_threshold` (default 3.0) standard deviations from the stored
  day-ahead prediction.  State is `off` during cold-start (< 10 matched pairs).  Attributes:
  `residual_kwh`, `residual_std_kwh`, `sigma_threshold`, `n_pairs`.  Published in both set_state
  and MQTT Discovery modes (`binary_sensor/<uid>/config` topic).  New config key:
  `anomaly_sigma_threshold` (float > 0, default 3.0, validated at startup).
- **Rolling MAE sensors (#41)** (`energy_forecast.py`): two new sensors track live forecast accuracy
  over a rolling window using stored prediction-vs-actual pairs:
  - `sensor.energy_forecast_mae_7d` — mean absolute error over the last 7 days (n_pairs attribute)
  - `sensor.energy_forecast_mae_30d` — mean absolute error over the last 30 days (n_pairs attribute)
  Both sensors are published in set_state and MQTT Discovery modes; state is `"0.0"` until enough
  history accumulates (~15 days to fill the 30d window).  The `_pred_history` prune window is
  extended from 7 to 30 days to support the longer sensor.  Adaptive retrain behaviour is unchanged.
- **Vacation / away flag (#25)** (`model.py`, `ha_data.py`, `energy_forecast.py`): new binary
  `is_away` feature lets the model learn lower consumption during vacations and predict accordingly.
  Two optional config keys: `away_mode_entity` (e.g. `input_boolean.vacation_mode`) and
  `away_return_entity` (e.g. `input_datetime.vacation_return`).  Both are independent and fully
  backward-compatible — when unconfigured `is_away` is 0 everywhere.
  - Training: 30-day state history of `away_mode_entity` is fetched via `fetch_boolean_entity_history`
    and joined to the training DataFrame as hourly `is_away` flags.
  - Prediction: `_build_away_prediction_series` projects `is_away` across the 48-hour window;
    if `away_return_entity` holds a future datetime, `is_away` flips to 0 at the return hour.

---

## [0.6.0] — 2026-03-23

### Fixed
- **Doubled "Energy Forecast" prefix in MQTT Discovery sensor names** (`energy_forecast.py`): HA
  prepends the device name ("HA Energy Forecast") to the sensor `name` field, so names like
  `"Energy Forecast Model MAE"` were displayed as `"HA Energy Forecast Energy Forecast Model MAE"`.
  Discovery `name` values are now short labels (`"Model MAE"`, `"Today"`, `"Setup Status"`, etc.);
  `set_state()` paths are unchanged as they have no device grouping.
- **Doubled sensors after enabling MQTT Discovery** (`energy_forecast.py`): on startup when
  `mqtt_discovery=True`, `_cleanup_legacy_states()` now calls `remove_entity()` for every
  entity_id previously created by the `set_state` path.  Ghost entities from a prior
  `mqtt_discovery=False` run are removed without requiring an HA restart.
- **MQTT publish broken on HASS apps** (`energy_forecast.py`): replaced `self.mqtt_publish()` (only
  available on MQTT-namespace apps) with `self.call_service("mqtt/publish", ...)`, which works from
  any AppDaemon HASS app.  Discovery, state, and availability publishes now succeed at startup and
  after retraining.
- **numpy 2.x retraining error** (`model.py`): `np.log1p` on object-dtype arrays (Python floats)
  raised `"loop of ufunc does not support argument 0 of type float"` on numpy 2.x.
  Fix: `df["gross_kwh"].to_numpy(dtype=float)` forces float64 before the log transform.

### Added
- **MQTT Discovery (#37)** (`energy_forecast.py`): opt-in entity registration via MQTT Discovery.
  Set `mqtt_discovery: true` in `apps.yaml` to register all ~29 sensors in the HA entity registry,
  enabling area assignment and labels.  Requires the AppDaemon MQTT plugin and a running MQTT broker.
  Config keys: `mqtt_discovery` (default `false`), `mqtt_namespace` (default `mqtt`),
  `mqtt_discovery_prefix` (default `homeassistant`).  All sensors grouped under a single
  `HA Energy Forecast` device.  Prediction interval sensors (`*_low`/`*_high`) are registered
  lazily on the first update cycle where quantile models exist.  Availability topic publishes
  `"online"` at startup and `"offline"` on AppDaemon shutdown.  Existing `set_state()` behaviour
  is unchanged when `mqtt_discovery: false`.

### Changed
- README: added MQTT Discovery section (prerequisites, `appdaemon.yaml` snippet, `apps.yaml` example, sensor count table, availability behaviour, revert instructions); added `mqtt_discovery` / `mqtt_namespace` / `mqtt_discovery_prefix` to parameter reference; updated Published sensors intro and Features bullet; added Contents entry

### Fixed
- Align hourly sensor updates to XX:01:00 wall-clock time using `run_hourly`; eliminates startup-time drift
- Downgrade prediction-time sub-sensor NaN log from WARNING to DEBUG; training-time WARNING (weekly) is sufficient

---

## [0.5.2] — 2026-03-20

### Fixed
- **pandas 3.x mixed-format timestamp parse failure** (`ha_data.py`): all four CSV-cache
  `pd.to_datetime()` calls now pass `format="mixed"`, preventing a `ValueError` when a
  date-only midnight entry (e.g. `"2026-03-20"`) appeared alongside full datetime strings
  in `energy_history.csv`.  Without this fix every hourly update after midnight ran with
  `recent_actuals = None`, degrading all lag/rolling features to training medians for the
  rest of the day.  The inner `except` clauses at each load site also widen to include
  `ValueError` so any future parse error degrades gracefully (empty cache, WARNING logged)
  rather than silently losing lag features.

### Added
- **MDI icons on all published sensors** (`energy_forecast.py`): every `set_state()` call
  now carries a `"unique_id"` attribute (stable identifier = entity_id minus `sensor.` prefix,
  reserved for future MQTT Discovery integration) and an `"icon"` attribute.  Icons:
  `mdi:lightning-bolt` (forecast totals + unavailable placeholders), `mdi:arrow-down-bold` /
  `mdi:arrow-up-bold` (prediction interval low/high), `mdi:calendar-clock` (3-hour block
  sensors), `mdi:car-electric` (EV sensors), `mdi:chart-bell-curve-cumulative` (model MAE).

  **Note:** AppDaemon's `set_state()` writes to HA's state machine only; it does not register
  entities in the entity registry.  Area assignment and labels require MQTT Discovery (roadmap
  item #37).

---

## [0.5.1] — 2026-03-19

### Fixed
- **Adaptive retrain cooldown timezone** (`energy_forecast.py`, H1): `_maybe_adaptive_retrain`
  now uses `pd.Timestamp.now("Europe/Zurich").tz_localize(None)` instead of `datetime.now()`
  (system local time), preventing the 24-hour cooldown from firing ±2 h early/late on
  UTC-based Docker/HA systems and across DST transitions.
- **Duplicate numpy import** (`energy_forecast.py`, H2): removed redundant `import numpy as np`
  inside the EV block of `_aggregate`; numpy was already imported at the top of the method.
- **CSV header TOCTOU race** (`ha_data.py`, H3): `stat()` + `to_csv(mode="a")` are now
  wrapped in a single `except OSError` block, preventing a potential race where another
  process deletes/truncates the file between the stat check and the write.
- **Sub-sensor merge deduplication** (`ha_data.py`, H4): both `fetch_sub_sensor_history` and
  `fetch_recent_sub_sensor` now use the shared `_merge_sub_sensor_frames` helper (backed by
  `_merge_energy_frames`) instead of duplicated inline `pd.concat/drop_duplicates` chains.
- **Missing cloud/radiation defaults** (`weather.py`, M2): absent `cloud_cover` / `direct_radiation`
  keys now fall back to `[np.nan]` instead of `[0]`; `0` was interpreted as "perfectly clear sky"
  and biased training. NaN triggers the safety-net median fill in `_engineer_features`.
- **SRG OAuth token cached** (`weather.py`, M1): token is now reused for 55 minutes, reducing
  SRG token-endpoint calls from 24+/day to ~1/day and removing silent Open-Meteo fallbacks
  caused by rate-limit errors.

### Added
- **Sunshine clamp + warning** (`weather.py`, M4): `_parse_sunshine_min` helper converts
  sunshine_duration (seconds → minutes) and clamps values > 60 min/h with a WARNING log.
- **Column guard in `_supplement_from_open_meteo`** (`weather.py`, M3): if Open-Meteo omits
  `cloud_cover_pct`/`direct_radiation_wm2` (API schema drift), the function logs a WARNING
  and returns the SRG DataFrame unchanged instead of raising `KeyError`.
- **No-lag WARNING** (`model.py`, M7): logs a WARNING when all autoregressive lags are
  skipped (history too short for even `lag_1h`), making it visible that the model is
  training without its core predictive features.
- **EV config in apps.yaml.example** (C3): `ev_charging_threshold_kwh` and `ev_charger_kw`
  are now documented with default values in the config template.

### Changed
- `_check_setup` exception narrowed from `except Exception` to
  `except (AttributeError, TypeError, RuntimeError)` (`energy_forecast.py`, L3).
- Redundant `hasattr(col.dtype, "tz")` guards removed; idiomatic `col.dt.tz is not None`
  used directly at all four sites (`ha_data.py`, M8).
- `HOLDOUT_FRACTION` clarified with inline comment that it is the *training* fraction
  (`const.py`, L1).
- `conftest.py` hassapi stub comment expanded to explain the purpose.
- README features table rewritten to cover stages 2–5 additions; sub-sensor feature list
  updated to include `active_24h` and `runs_7d`; activation-threshold wording corrected.
- README Installation step numbering fixed (duplicate step 3 renumbered to 4/5).
- README Published Sensors: `sensor.energy_forecast_setup_status` documented in Model diagnostics table.
- README Sub-energy sensors: feature table expanded to show all four features (`lag_24h`, `lag_168h`, `active_24h`, `runs_7d`) with activation thresholds.
- README Parameter reference: deprecated `plz` parameter removed from table; replaced with a brief callout note.
- README Troubleshooting: MAE guidance reframed as a percentage of average hourly consumption rather than a fixed threshold.
- CHANGELOG version comparison links added for v0.4.0–v0.5.0; `[Unreleased]` pointer corrected to `v0.5.0...HEAD`.
- `apps.yaml.example`: `timezone` line annotated with a change hint for non-Swiss users.

---

## [0.5.0] — 2026-03-19

### Added
- **Setup checker sensor** (`energy_forecast.py`, #17): `_check_setup()` is called on
  `initialize()` and publishes `sensor.energy_forecast_setup_status` (state: `ok` or
  `missing_packages`).  The `missing_packages` attribute lists which pip packages failed
  to import, so users can diagnose install issues directly from HA Developer Tools without
  reading AppDaemon logs.

### Changed
- **CSV append-only writes** (`ha_data.py`, #19): `fetch_recent_energy` (hourly) now
  appends only genuinely new timestamps to the cache CSV instead of rewriting the entire
  file on every sensor update.  `fetch_energy_history` (weekly retrain) continues to do a
  full sort + dedup compaction rewrite, which also corrects any stale values that bypassed
  the append-only path.

---

## [0.4.5] — 2026-03-19

### Added
- **Per-hour-of-week NaN fill medians** (`model.py`, #31): during training, per-HOW
  (168-cell) medians are computed for all lag and rolling columns and stored as
  `feature_medians_by_how` in `meta.pkl`.  At predict time, NaN values in these columns
  are filled using the HOW-specific median for the matching `hour_of_week` slot, falling
  back to the global median when the HOW bucket is empty.  Backward compatible — old
  `meta.pkl` without this key silently defaults to global-median behaviour.

---

## [0.4.4] — 2026-03-19

### Added
- **`{prefix}_active_24h` binary flag** (`model.py`, #35): 1 when the sub-sensor had
  any non-zero reading in the 24h window before each training/prediction row, else 0.
  Provides a "was the appliance recently active?" signal for sparse sensors (~95% zero).
- **`{prefix}_runs_7d` rolling run count** (`model.py`, #36): count of appliance start
  events (0 → >0 transitions) in the past 168h during training.  At predict time the
  count is computed from recent actuals and held constant across the 48-hour horizon
  (future starts are unknown).  Helps the model distinguish heavy-use from idle periods.

---

## [0.4.3] — 2026-03-19

### Added
- **Day-of-year cyclical features** (`model.py`, #33): `doy_sin` and `doy_cos`
  (period 365) added to `_FEATURES_BASE` and `_engineer_features`.  Gives the model
  a smooth, continuous seasonal signal independent of month/season buckets.
- **`hours_ahead` horizon feature** (`model.py`, #34): set to 0 for all training rows
  (actuals) and overwritten with 0–47 in `_prepare_prediction_X` so the model can
  learn horizon-specific bias without distributional leakage.
- **`num_leaves` sweep** (`model.py`, #28): on the last CV fold (LightGBM only), values
  `[16, 31, 63]` are evaluated; the best is selected and used for the final model.
  Results are logged at INFO level.  Falls back to 31 on sklearn GBR.

---

## [0.4.2] — 2026-03-19

### Added
- **Short-horizon lag features** (`model.py`, #27): `lag_1h`, `lag_2h`, `lag_6h`, and
  `lag_12h` added to `LAG_HOURS`.  The existing dynamic-selection gate (`n_rows - lag ≥ 100`)
  activates each as history grows (lag_1h at 101 rows, lag_12h at 112).  At predict time
  only the first `L` future hours carry real lag values; later hours receive the training
  median, which is intentional — the model learns horizon-specific weighting.  Expected
  accuracy improvement is concentrated on hours 1–12 ahead.

---

## [0.4.1] — 2026-03-19

### Added
- **Feature importance logging** (`model.py`, #29): after every training run the top-10
  feature importances (by gain) are logged at INFO level for quick diagnostics.
- **CV fold std logging** (`model.py`, #30): the CV MAE log line now includes
  `mean ± std` across the three TimeSeriesSplit folds alongside the per-fold values.
- **EV threshold / charger_kw mismatch warning** (`energy_forecast.py`, #20):
  `_validate_config` now logs a WARNING when `ev_charging_threshold_kwh ≥ ev_charger_kw`,
  which would prevent any EV session from being detected.

### Changed
- **Holiday distance vectorisation** (`model.py`, #32): `days_to_next_holiday` and
  `days_since_last_holiday` are now computed via `np.searchsorted` on date ordinals
  instead of a per-row Python `bisect` + `.map()` call. Semantics are identical.

---

## [0.4.0] — 2026-03-19

### Fixed
- **NaN warning in `_add_sub_sensor_lags_training`** (`model.py`): mirrors the
  prediction-side check — logs WARNING when sub-sensor reindex introduces >50% NaN
  values, surfacing gap/alignment issues during training.

### Added
- **`sub_energy_sensors` config key** (`ha_data.py`, `model.py`, `energy_forecast.py`):
  Track hourly consumption of custom HA cumulative kWh sensors (e.g. heat pump,
  dishwasher) as `lag_24h` / `lag_168h` features to improve forecast accuracy.
  Sub-sensors must be `total_increasing` kWh meters; zero-kWh hours (appliance off)
  are preserved so lag features correctly return 0 instead of NaN during idle periods.
  All parameters are optional and default to `[]` — no behaviour change for existing
  deployments.

---

## [0.3.0] — 2026-03-13

### Fixed
- **Forecast weather features silently imputed from medians** (`energy_forecast.py`):
  `pd.to_datetime(forecast_df["timestamp"], utc=True)` reinterpreted tz-naive local
  timestamps as UTC, then `tz_convert("Europe/Zurich")` shifted them +1h. The weather
  merge in `_engineer_features` consequently found zero timestamp matches, causing all
  weather features to fall back to training-set medians on every sensor update.
  Replaced with `_strip_tz(forecast_df)` — a no-op for tz-naive input, correct for
  tz-aware. Pre-existing on the Open-Meteo path; also affected SRG after the fix below.
- **`_supplement_from_open_meteo` crash under pandas 3.x** (`weather.py`): SRG-SSR v2
  returns timestamps with UTC offset (e.g. `+01:00`). Comparing that tz-aware Series
  against tz-naive Open-Meteo timestamps raised `"Invalid comparison between
  dtype=datetime64[us] and Timestamp"`, crashing every sensor update. SRG timestamps
  are now stripped to naive Europe/Zurich before the comparison, consistent with the
  rest of the pipeline.
- **SRG-SSR API migrated to v2** (`weather.py`): v1 endpoint
  (`/forecasts/v1.0/weather/7day`) was decommissioned. Updated flow: resolve station
  via `GET /srf-meteo/v2/geolocations?latitude=...&longitude=...`, fetch forecast via
  `GET /srf-meteo/v2/forecastpoint/{id}`. Precipitation field renamed `PRP_MM` →
  `RRR_MM`; response structure changed from nested `forecast[].hours[]` to flat
  `hours[]`. Geolocation now uses lat/lon (not PLZ) to reliably match the registered
  Freemium station.
- **`_retrain_cb` / `_update_cb` crash on `RELOAD_ENERGY_MODEL` event**: Both
  callbacks now accept `(event_name=None, data=None, kwargs=None)` so they work when
  fired by `listen_event` (three positional args) as well as the scheduler (one arg).
- **EV charger power ignored in hourly lag features**: `_update_sensors` was passing
  the default 9.0 kW to `split_ev_charging` regardless of the configured
  `ev_charger_kw`, causing lag-feature drift for non-default charger powers.
- **EV kWh sensors reported threshold energy instead of charger energy**: `ev_today`
  / `ev_yesterday` now subtract `ev_charger_kw` (not `ev_charging_threshold_kwh`)
  from gross import, giving the correct net charger energy estimate.
- **`adaptive_retrain_threshold` accepts negative values at startup**: `_validate_config`
  now raises `ValueError` for values < 0; a negative threshold would have triggered
  retraining on every hourly update.
- **Open-Meteo sunshine hardcoded to zero**: `fetch_open_meteo` previously set
  `sunshine_min = 0` unconditionally, silently degrading forecasts for all
  installations without SRG-SSR credentials. `sunshine_duration` is now requested
  from the API and converted from seconds to minutes.
- **Rolling feature train/predict mismatch**: `rolling_mean_24h`, `rolling_mean_7d`,
  and `rolling_std_24h` were broadcast as a single scalar across all 48 prediction
  hours. These features now slide over an extended actuals + fill series with
  `shift(1)`, mirroring the training computation exactly.

### Changed
- **`_empty_weather_df`** now includes `cloud_cover_pct` and `direct_radiation_wm2`,
  matching the full column contract of the real weather fetchers.
- **`fetch_recent_energy`**: removed unused `hours: int = 6` parameter.
- **Dead URL constants** `METEOSWISS_URL`, `OPENMETEO_FORECAST_URL`,
  `OPENMETEO_ARCHIVE_URL` removed from `const.py` (never imported).
- **`is_public_holiday`** computation vectorised in `_add_holiday_feature`.
- **SRG-SSR fallback** now logged at WARNING so operators know forecast quality may
  be reduced when the SRG API is unavailable.
- **Lock contention** (skipped retrain / sensor-update cycles) now logged at DEBUG.

### Added
- **Prediction intervals**: quantile regression models (α=0.1, α=0.9) trained
  alongside the point-estimate model. Six new sensors:
  `sensor.energy_forecast_{next_3h,today,tomorrow}_{low,high}`. Today's bounds blend
  actuals for elapsed hours with quantile forecasts for remaining hours; the interval
  collapses to zero width for hours that have already passed.
- **Intra-day actuals blending**: `sensor.energy_forecast_today` and the
  `today_HH_HH` block sensors now substitute measured consumption for elapsed hours
  instead of relying on predictions alone.
- **Adaptive retraining**: each hourly update stores the 48h prediction (keep-first,
  preserving the ≈24h-ahead forecast). If live day-ahead MAE exceeds
  `adaptive_retrain_threshold × cv_MAE` with ≥ 24 matched pairs and a 24h cooldown
  has elapsed, an early retrain is triggered. Configure via `adaptive_retrain_threshold`
  in `apps.yaml` (default 2.0).
- **Log-transform target**: `gross_kwh` is `log1p`-transformed before training and
  `expm1`-inverted at prediction time, reducing the influence of high-energy outliers.
  MAE is still reported in kWh. Backward-compatible: existing pickles default to no
  transform until the next retrain.
- **LightGBM early stopping**: CV folds use `stopping_rounds=50`; `best_iteration_`
  from the last fold sets `n_estimators` for the final model.
- **Cantonal holidays**: configure `holiday_canton` (e.g. `"ZH"`, `"BE"`) in
  `apps.yaml`; invalid codes fall back to federal-only with a warning.
- **EV charging pattern feature**: `likely_ev_hour` binary feature marks
  `hour_of_week` slots where EV charging occurred in ≥ 15% of historical occurrences.
- **Cloud cover and direct radiation** (`cloud_cover_pct`, `direct_radiation_wm2`)
  added to archive fetcher, Open-Meteo forecast fetcher, and `_FEATURES_BASE`. SRG
  users receive these via `_supplement_from_open_meteo`.
- **`lag_72h`** autoregressive feature (same time 3 days ago); activates dynamically
  at ≥ 172 h of history.
- **Bridge-day proximity features**: `days_to_next_holiday` and
  `days_since_last_holiday` (integers, capped at 3; 0 on a holiday itself).
- **`fetch_open_meteo` `past_days=3`**: 72 h of measured history anchors
  `temp_rolling_3d` in real observations for both Open-Meteo and SRG users.
- `ROADMAP.md` — forecast accuracy improvement roadmap (15 items across 4 tiers)

---

## [0.2.1] — 2026-03-10

### Added
- `LICENSE` file (MIT)
- `ha_appdaemon_config.yaml` — ready-to-paste AppDaemon add-on dependency config
- `.gitignore` entries for model artifacts (`*.pkl`, `*.sha256`, `energy_history.csv`) and HA database files

### Changed
- README: consolidated `INSTALL.md` content (add-on install step, directory tree, entity ID tip); fixed Licence section to link to `LICENSE`

### Removed
- `INSTALL.md` — content merged into README

### Security
- Rewrote git history to remove SRG-SSR API credentials committed in early history

---

## [0.2.0] — 2026-03-10

### Added
- **Security**: SHA-256 integrity sidecars for model pickle files; integrity mismatch triggers warning and cold-start retrain
- **Correctness**: unified `_merge_energy_frames()` helper — HA data always wins on timestamp conflicts; `apps.yaml` config validation on startup; empty weather guard before model training
- **Observability**: specific exception types throughout (replacing bare `except Exception` in inner catches); ML engine logged at startup (`LightGBM` or `sklearn GBR`); warning when lag features contain NaN
- **Code quality**: all shared constants moved to `const.py` (single source of truth); type hints added; timezone fix (`Europe/Zurich` throughout); vectorised lag feature computation; config-driven cache path
- **DST hardening**: `_check_dst_duplicates()` detects fall-back duplicate timestamps after every merge and logs a warning; spring-forward gap filled by resample/ffill
- **Test suite**: 25 pytest tests covering merge semantics, HA/cache fetch integration, and DST edge cases — no live HA or AppDaemon required
- **README**: full installation, configuration, sensor reference, architecture, backfill, weather sources, EV detection, troubleshooting, and security notes

### Fixed
- SQL injection in `get_history` query (parameterised input)
- Fall-back to Open-Meteo when SRG-SSR forecast is unavailable was silently discarded; now logged as a warning

---

## [0.1.0] — 2026-03-09

### Added
- Initial AppDaemon app (`EnergyForecast`) publishing 48-hour hourly energy forecasts as HA sensor entities
- LightGBM model with automatic scikit-learn GBR fallback for platforms without a C compiler (e.g. armv7)
- Open-Meteo archive weather fetcher for training; SRG-SSR + Open-Meteo forecast fetcher for prediction
- Persistent CSV history cache — survives HA database purges
- EV charging detection: hours above threshold have fixed charger load subtracted before training; detected kWh published as separate sensors
- Live outdoor temperature sensor blending for near-term forecast hours
- Exponential sample weighting (`weight_halflife_days`, default 90 days)
- One-off SQLite backfill tool (`energy_history_backfill.py`) to import up to one year of HA recorder history
- `apps.yaml.example` configuration template

[Unreleased]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-12...HEAD
[0.10.2-alpha-12]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-11...v0.10.2-alpha-12
[0.10.2-alpha-11]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-10...v0.10.2-alpha-11
[0.10.2-alpha-10]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-9...v0.10.2-alpha-10
[0.10.2-alpha-9]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-8...v0.10.2-alpha-9
[0.10.2-alpha-8]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-7...v0.10.2-alpha-8
[0.10.2-alpha-7]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-6...v0.10.2-alpha-7
[0.10.2-alpha-6]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-5...v0.10.2-alpha-6
[0.10.2-alpha-5]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-4...v0.10.2-alpha-5
[0.10.2-alpha-4]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-3...v0.10.2-alpha-4
[0.10.2-alpha-3]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-2...v0.10.2-alpha-3
[0.10.2-alpha-2]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.2-alpha-1...v0.10.2-alpha-2
[0.10.2-alpha-1]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.1...v0.10.2-alpha-1
[0.10.1]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.10.0...v0.10.1
[0.10.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.9.0...v0.10.0
[0.10.0-alpha]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.9.0...v0.10.0-alpha
[0.9.1-alpha]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.9.0-alpha...v0.9.1-alpha
[0.9.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.8.1...v0.9.0
[0.9.0-alpha]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.8.1...v0.9.0-alpha
[0.8.1]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.8.0...v0.8.1
[0.8.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.7.2...v0.8.0
[0.7.2]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.7.1...v0.7.2
[0.7.1]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.7.0...v0.7.1
[0.7.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.6.0...v0.7.0
[0.6.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.5.2...v0.6.0
[0.5.2]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.5.1...v0.5.2
[0.5.1]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.5.0...v0.5.1
[0.5.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.4.5...v0.5.0
[0.4.5]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.4.4...v0.4.5
[0.4.4]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.4.3...v0.4.4
[0.4.3]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.4.2...v0.4.3
[0.4.2]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.4.1...v0.4.2
[0.4.1]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.4.0...v0.4.1
[0.4.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.3.0...v0.4.0
[0.3.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.2.1...v0.3.0
[0.2.1]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.2.0...v0.2.1
[0.2.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.1.0...v0.2.0
[0.1.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/releases/tag/v0.1.0
