# Physics-ML Hybrid Energy Forecast Model — Design Spec

**Date:** 2026-06-22
**Status:** Approved for implementation planning
**Branch base:** `dev` (v0.11.4-alpha-2)

---

## 1. Problem & Motivation

The current model is a pattern recogniser: LightGBM finds correlations between weather, time, occupancy, and consumption. It has reached ~0.52 kWh/h MAE (30-day) but faces a structural ceiling: warm-weather accuracy suffers because the model has never seen summer data (history started Oct 2025), and the only remedy — more time — is slow. Adding yet more engineered features has diminishing returns; the thermal modelling pack (#49–#58) already covers most of that ground.

The step-change improvement requires the model to understand *why* energy is consumed, not just *when*. This spec describes a hybrid architecture where a calibrated physics model provides a baseline prediction grounded in building thermodynamics, and LightGBM learns only the residual (occupancy patterns, behavioural quirks, edge cases).

---

## 2. Architecture

### 2.1 Two-Phase Rollout

Both phases share the same `ThermalPhysicsModel` class. The rollout is gated by `use_physics_residual` in config.

**Phase 1 — Physics as feature (de-risking)**
- `physics_kwh` added to `_FEATURES_BASE` **only when `physics_model is not None`** — a zero-filled constant column adds no signal and pollutes SHAP rankings and saved model artifacts
- LightGBM trains on `gross_kwh` as today; learns when to trust or override the physics signal
- Validate via SHAP: if `physics_kwh` ranks in top-5 features **and** OLS of `physics_kwh` vs `gross_kwh` on the holdout yields slope ∈ [0.8, 1.2], calibration is good; proceed to Phase 2. High SHAP rank alone is insufficient — a systematically biased physics signal can still be predictive while miscalibrated.
- **Cold-start gate**: do not trigger Phase 2 until UA_eff has been calibrated from at least 30 valid nighttime windows (ΔT ≥ 8 K, pre-holdout). At initial deployment (spring 2026), UA_eff will not be available until winter 2026/27. Log a WARNING and hold at Phase 1 until this condition is met.

**Phase 2 — Explicit residual split (structural)**
- LightGBM trains on `gross_kwh − physics_kwh`
- `predict()` returns `(physics_kwh + lgbm_residual).clip(lower=0)` — clip applied consistently in both phases
- The **log1p target transform is disabled** for Phase 2: the residual can be zero or negative when physics over-predicts. Train on the raw residual; accept mildly heteroskedastic variance.
- Holdout MAE is always reported on **gross_kwh** regardless of phase (see §5.2). Residual MAE is an internal diagnostic only.
- Two new HA sensors: `physics_base_today`, `ml_adjustment_today`
- SHAP narrative gains opening line: *"Physics expects X kWh; behaviour adds Y kWh"*

```
ThermalPhysicsModel.predict_series()
        │
        │ physics_kwh_series (same pattern as regime_kwh_series)
        ▼
_engineer_features()
        │
        ├─ Phase 1: physics_kwh is one feature among many
        │           LightGBM target = gross_kwh  (log1p active)
        │
        └─ Phase 2: LightGBM target = gross_kwh − physics_kwh  (log1p disabled)
                    predict() = (physics_kwh + lgbm_residual).clip(lower=0)
```

### 2.2 New Files

| File | Purpose |
|---|---|
| `apps/energy_forecast/physics.py` | `ThermalPhysicsModel` class |
| `models/physics_calibration.json` | Stable calibrated parameters: `calibrated_at`, `n_calibration_windows_ua_eff`, `UA_eff`, `solar_gain_area`, `Q_base_el`, `Q_dhw_daily`, `UA_dhw`, `cop_formula`; written atomically via write-then-rename |
| `models/physics_schedule.json` | Frequently-updated operational state: `T_dhw_upper`, `T_legionella`, `legionella_dow`, `legionella_hour`, `T_dhw_lower`, `dhw_tank_volume_l`; written via atomic write-then-rename |

Stable calibrated constants and frequently-written operational state are kept in separate files to avoid corrupt writes when AppDaemon restarts mid-update. The original proposed single `physics_params.json` is replaced by these two files.

### 2.3 Files Modified

| File | Change |
|---|---|
| `model.py` | `_engineer_features()` gets `physics_kwh_series` param; `train()` / `predict()` Phase 2 logic; `_find_passive_windows()` promoted to shared utility; log1p gate; gross-kWh MAE logging |
| `energy_forecast.py` | Config ingest; new sensors; `recalibrate_physics` service; Scenario API `dhw_schedule` extension; `set_dhw_schedule` cache invalidation |

### 2.4 Files Unchanged

`ha_data.py`, `weather.py`, `clustering.py`, `const.py`, all MQTT sensor IDs, all existing tests.

---

## 3. Physics Model

### 3.1 Components

**Space heating**

`T_indoor_h` and `T_indoor_{h+1}` are the indoor temperatures for hour h and h+1, sourced differently depending on path:
- **Training**: actual historical readings from `room_thermostats[*].temp_sensor`, area-weighted across rooms. Consecutive actual readings give the real-world thermal mass term.
- **Prediction**: output of `_project_indoor_temps()` ODE projection — **not** the raw thermostat setpoint.

In both cases, when T_indoor exceeds the setpoint, `Q_heat = 0` (building is coasting).

```
Q_loss    [W]     = UA_eff × max(0, T_indoor_h − T_outdoor)
Q_solar   [W]     = solar_gain_area × GHI
Q_gain_int [W]    = Q_base_el × internal_gains_fraction × 1000   # default fraction = 0.8 (config: internal_gains_fraction); calibration-period representative — does not scale with instantaneous consumption; above-baseline gains absorbed by LightGBM residual
Q_mass    [W]     = C_building_Wh_K × (T_indoor_{h+1} − T_indoor_h)   # thermal mass charge/discharge
Q_heat    [W]     = max(0, Q_loss − Q_solar − Q_gain_int + Q_mass)
Q_heat_el [kWh/h] = Q_heat / COP(T_outdoor) / 1000
```

`C_building_Wh_K` defaults to `UA_eff × τ` where τ comes from `_calibrate_tau()` (typically 6–10 h for residential buildings). Can be overridden in config. The thermal mass term corrects the steady-state Q_loss assumption during transients (morning warm-up, cold snaps), which otherwise causes 30–60% errors at the hourly resolution.

When `UA_eff = None` (insufficient winter data), the entire heating component is skipped (`Q_heat_el = 0`), so `C_building_Wh_K` and `Q_mass` are not evaluated. No fallback value is needed for `C_building_Wh_K` in that case.

**GHI/POA note**: `solar_gain_area` is calibrated on winter days; the calibrated value implicitly absorbs a winter tilt factor and winter-specific shading geometry (obstructions, neighbouring buildings at low sun angles). In summer, GHI under-represents irradiance on a tilted south-facing surface and shading patterns differ — expect a 20–40% over-estimate of solar gain in summer months. This is conservative (reduces predicted heating demand in summer) and acceptable at this forecast horizon.

**Infiltration note**: UA_eff as calibrated via OLS implicitly absorbs infiltration losses at normal operating conditions. Wind-driven infiltration variability is not explicitly modelled; the existing `wind_kmh` feature in `_FEATURES_BASE` allows LightGBM to learn this residual correction.

**COP model** — priority: live sensor (past/current hours) → Carnot-bounded formula (future hours) → constant fallback:
```
COP(T_outdoor) = max(
    COP_min,
    min(
        η_carnot × T_flow_K / (T_flow_K − T_outdoor_K),   # Carnot upper bound
        a + b × T_outdoor                                   # linear fit from sensor OLS
    )
)
COP_min = 1.1   # floor covering defrost cycles and cold-snap degradation
```
`η_carnot = 0.45` (typical ASHP second-law efficiency); `T_flow_K` = mean flow temperature in Kelvin from `heating_curve_points` lookup (or 318 K / 45°C default). The linear formula alone overestimates COP at T_outdoor < −5°C; the Carnot bound prevents this. `a`, `b` priority: OLS of COP sensor vs T_outdoor → config `cop_formula` → Carnot-only with η = 0.40.

**DHW — tank ODE**

State-simulated forward per hour from current `T_tank`. Timestep Δt = 1 h throughout.

```
C_dhw [Wh/K]      = dhw_tank_volume_l × 1.163    # specific heat of water
Q_dhw_power [W]   = dhw_power_w                   # HP rated DHW output — see source priority below

dT = −UA_dhw × (T_tank − T_ambient) / C_dhw      # insulation loss [K/h]
   − draw_profile[h] × draw_rate                  # hot-water draws [K/h]

if T_tank < T_lower:
    Q_dhw_el    = Q_dhw_power / COP_dhw           # [W]
    heating_rise = Q_dhw_power / C_dhw            # [K/h] — derived each step, not a config constant
else:
    Q_dhw_el    = 0
    heating_rise = 0

T_tank = clamp(T_tank + dT + heating_rise, T_lower, T_legionella)
```

**`Q_dhw_power` source priority:**
1. Calibrated from energy meter: median HP electrical power during confirmed DHW-active hours × COP_dhw → stored in `physics_calibration.json`
2. Config: `dhw_power_w` (explicit override)
3. Fallback: 4000 W (typical mid-range ASHP DHW output for a 200L tank)

`heating_rise` is computed as `Q_dhw_power / C_dhw` each timestep, not stored as a fixed constant.

Post-legionella silence emerges naturally: tank initialises at `T_legionella` (60°C) and does not trigger heating until it cools below `T_lower`. No special-casing required.

Initial state:
- `dhw_tank_temp_sensor` reading if <2h old
- Else: estimated from time elapsed since last legionella via passive cooling ODE

**Heating buffer**

Used as a direct feature only. `heating_buffer_temp_sensor` (current reading) tells the model whether the HP is about to coast or run — a near-term signal for hours 0–1. No forward ODE, no volume parameter. Steady-state physics handles hours 2–48.

The column `heating_buffer_temp` is added to the feature matrix via the same hourly merge pattern as other sensor readings (see §5.1).

**Base electrical**

`Q_base_el` [kWh/h] — calibrated standing load; optionally scaled by `people_home`.

**Total:** `physics_kwh = Q_heat_el + Q_dhw_el + Q_base_el`

### 3.2 Room Thermostat Integration

Config:
```yaml
room_thermostats:
  - climate_entity: climate.living_room
    temp_sensor: sensor.netatmo_living_room_temp
    area_m2: 35
  - climate_entity: climate.bedroom
    temp_sensor: sensor.netatmo_bedroom_temp
    area_m2: 20
```

- **Training:** historical per-room setpoints and actuals replace the static `T_setpoint` assumption in the heating demand formula
- **Prediction:** 48h setpoint schedule projected forward from thermostat weekly programs (same per-room ODE as `_project_indoor_temps()`, which already handles this). The projected indoor temperature series feeds `T_indoor_h` in the Q_loss formula — not the raw thermostat setpoint. Using the setpoint directly would assign heating demand during coasting hours, double-counting with the thermal mass term.
- **Zone boundary consistency**: `UA_eff` is calibrated over the thermostat list present at calibration time. If `room_thermostats` changes between calibrations (rooms added or removed), the effective zone boundary shifts and UA_eff is no longer consistent with Q_loss. Log a WARNING if the thermostat entity list differs from the one recorded in `physics_calibration.json` at `calibrated_at`.
- **Open window detection:** per-room residual `dT_actual − dT_ODE` > 2σ → `open_window_hour` flag stored in the training frame. The 2σ threshold is computed from **passive-window residuals only** (HP-off, stable ΔT periods) — not from all training hours. This avoids the threshold being inflated by miscalibrated UA_eff or τ, which would suppress legitimate flags. Affected hours are down-weighted during training (analogous to how EV-adjacent hours are down-weighted for MAE; not excluded outright). Not used in the prediction path — windows cannot be forecast.

### 3.3 DHW Schedule — Operational vs Calibrated

DHW and legionella timing is **operational** (can change any cycle) rather than calibrated (stable physical property). It is stored in `physics_schedule.json` (see §2.2) and injectable at runtime:

```python
physics_model.predict_series(..., dhw_schedule_override={"legionella": ("2026-06-25", 10)})
```

**Stability guard:** if inferred legionella timing shifts >±2h week-over-week, log WARNING and suspend **autonomous** schedule learning. This guard applies to autonomous inference only. When `set_dhw_schedule` receives an explicit confirmation from the energy manager, `physics_schedule.json` is updated immediately without going through the instability guard — the guard exists to detect unexpected external shifts, not to block intentional rescheduling by the energy manager. Failure to bypass the guard here would cause the energy manager's own schedule shifts to be detected as instability, permanently suspending learning on the second week.

---

## 4. Calibration

### 4.1 Trigger

- Startup, if `physics_calibration.json` absent or `calibrated_at` >30 days old
- On demand: AppDaemon service `energy_forecast/recalibrate_physics`
- Triggered inside `train()` (which already has all required data in scope)

The staleness check reads `calibrated_at` from `physics_calibration.json` at the top of `train()`, before any data loading. `calibrated_at` missing or unparseable → treat as epoch zero (always stale). The `recalibrate_physics` service updates the same `calibrated_at` on completion. Both `physics_calibration.json` and `physics_schedule.json` are written via **atomic write-then-rename** to prevent partial reads on AppDaemon restart. Concurrent calls (service + scheduled train overlap) are safe: the later writer wins and the file is always valid.

`physics_calibration.json` also stores `n_calibration_windows_ua_eff` (the count of qualifying nighttime windows used to produce UA_eff). The cold-start gate reads this value directly; if it falls below 30 on a re-calibration (e.g. after a data gap), Phase 2 is blocked again even if a prior calibration had satisfied the threshold.

`_find_passive_windows()` is promoted from `_calibrate_tau()` to a shared module-level utility used by both τ and UA_eff calibration.

**Signature:** `_find_passive_windows(df: pd.DataFrame, *, min_delta_t: float = 8.0, min_hp_off_hours: int = 2) -> pd.Index`
- `df` must contain: `timestamp`, `T_outdoor`, `T_indoor` (or setpoint proxy), `hp_running` (bool), `dhw_tank_temp` (float or NaN — from `dhw_tank_temp_sensor`, **not** `heating_buffer_temp_sensor` which is the space heating circuit buffer, a distinct sensor)
- Returns an index of rows where HP has been off for ≥ `min_hp_off_hours` and ΔT ≥ `min_delta_t` K; additionally excludes hours where `dhw_tank_temp` is rising (active DHW cycle)
- When `dhw_tank_temp_sensor` is not configured, `dhw_tank_temp` column is all-NaN. In this case DHW-active hours **cannot** be filtered; compensate by raising `min_delta_t` to 12.0 K to reduce the probability of including DHW-heating hours, and log WARNING "DHW tank sensor absent — UA_eff calibration may be inflated"
- Callers must pass only rows with `timestamp < holdout_cutoff` — calibration windows must **exclude the ML holdout period** to prevent target leakage in Phase 2 (UA_eff calibrated on holdout data would pre-subtract signal LightGBM is evaluated on)

### 4.2 Parameters Calibrated from History

| Parameter | Method | Data required | Fallback |
|---|---|---|---|
| `Q_base_el` [kWh/h] | Median of summer nights (01–05h, Jun–Aug, no EV, no away) | 14 nights | 0.35 kWh/h |
| `Q_dhw_daily` [kWh] | Summer daily mean − 24 × Q_base_el | 14 days | 3.5 kWh |
| `UA_eff` [W/K] | Nighttime OLS: `Q_heat_obs ≈ UA_eff/1000 × ΔT/COP` (Nov–Mar, 22–06h, **excluding hours where `dhw_tank_temp` is rising**, ΔT ≥ 8 K or ≥ 12 K if DHW sensor absent, pre-holdout only); accepted only if R² ≥ 0.5 | 30 winter nights with ΔT ≥ 8 K | None — log WARNING; skip heating component |
| `solar_gain_area` [m²] | Daytime OLS residual with UA known (GHI >50 W/m²) | 14 sunny winter days | 0 m² (conservative) |
| `UA_dhw` [W/K] | Passive decay regression on DHW sensor during HP-off periods | DHW sensor + HP-off windows | 15 W/K (200L spec) |

**UA_eff calibration note**: DHW reheating and legionella cycles frequently run in the 22–06h window and inflate UA_eff if included. The filter uses `dhw_tank_temp` (from `dhw_tank_temp_sensor`) to detect active DHW cycles — distinct from `heating_buffer_temp_sensor` which tracks the space-heating circuit. Enforce ΔT ≥ 8 K (12 K if DHW sensor absent) to remove low-gradient nights where noise dominates. If OLS R² < 0.5, discard the result and log WARNING; fall back to config default.

### 4.3 Parameters Inferred from Sensor Data

All inferred values override config defaults and are stored in `physics_schedule.json`. Config override keys are available as escape hatches (commented-out in config example).

| Parameter | Method | Confidence | Sensor |
|---|---|---|---|
| `T_dhw_upper` | 90th percentile of local peaks in DHW sensor | High | DHW sensor |
| `T_legionella` | Outlier peaks >T_dhw_upper + 3°C | High | DHW sensor |
| `legionella_dow` | Mode of day-of-week at legionella peak starts | High | DHW sensor |
| `legionella_hour` | Mode of hour at legionella peak starts | High | DHW sensor |
| `T_dhw_lower` | Tank temp at cycle-start local minima | Medium | DHW sensor |
| `dhw_tank_volume_l` | `C_dhw = Q_in / ΔT_rise` per heating cycle | Medium | DHW sensor + sub-sensor |
| `cop_formula` a, b | OLS of COP sensor vs T_outdoor | High | COP sensor |
| `legionella_dow/hour` (fallback) | Weekly outlier in summer consumption residuals | Low–medium | None |

### 4.4 Must-Configure

Only entity IDs and spec-sourced defaults require manual configuration:

```yaml
physics:
  # Sensor entity IDs — all optional
  cop_sensor: sensor.kermi_cop
  dhw_tank_temp_sensor: sensor.kermi_dhw_buffer_temp
  heating_buffer_temp_sensor: sensor.kermi_heating_buffer
  heating_curve_sensor: sensor.kermi_parallel_shift

  # Spec fallbacks — used only when sensor/inference unavailable
  cop_formula: {a: 2.5, b: 0.07}
  dhw_tank_volume_l: 200
  dhw_power_w: 4000              # HP rated DHW heating output [W]; inferred from energy meter if available
  internal_gains_fraction: 0.8   # fraction of Q_base_el [kWh/h] that becomes room heat; override if known

  # Override escapes (inferred by default when DHW sensor configured)
  # T_legionella: 60
  # legionella_dow: 2
  # legionella_hour: 14

  # Heating curve (for flow setpoint projection; subsumes roadmap #15)
  heating_curve_points: [[-20, 55.5], [-5, 46.0], [5, 39.5], [20, 25.0]]

  # Room thermostats
  room_thermostats:
    - climate_entity: climate.living_room
      temp_sensor: sensor.netatmo_living_room_temp
      area_m2: 35

  # Phase gate
  use_physics_residual: false
```

Absent `physics:` block → model behaviour identical to current v0.11.4.

---

## 5. Integration with Existing Model

### 5.1 Feature Integration

`physics_kwh_series` follows the same pattern as `regime_kwh_series`:
- Computed upstream in `train()` and `predict()` before `_engineer_features()`
- Passed as `physics_kwh_series: pd.Series | None` parameter
- Inside `_engineer_features()`: merged onto df by timestamp (lines ~2749 pattern)
- `physics_kwh` is added to `_FEATURES_BASE` **only when `physics_model is not None`** — when physics is disabled no column is added (a constant-zero column adds no signal, pollutes SHAP output, and breaks model artifact portability between physics-enabled and physics-disabled environments)
- **Model artifact portability**: a model trained with `physics_kwh` stores it in its feature list. If `physics_model` is later set to `None` (e.g. sensor outage, config change), prediction must not fail. At prediction time, if `physics_kwh` is in the saved feature list but `physics_model is None`, fill the column with 0.0 and log WARNING. This is a graceful degradation, not an error.
- `heating_buffer_temp` is added to `_FEATURES_BASE` when `heating_buffer_temp_sensor` is configured; merged from the most recent sensor reading per hour (same merge pattern as other direct sensor features)

**Collinearity note (post-Phase-1 cleanup):** `physics_kwh` shares signal with `hp_heating_degree`, `heating_deg_sum_24h`, and `thermal_pressure` (all ΔT-derived). After Phase 1 validation, audit SHAP for these features and consider consolidating or removing redundant thermal degree features. This is a post-Phase-1 task, not a Phase 1 blocker — the multicollinearity reduces SHAP interpretability but does not harm prediction accuracy.

### 5.2 Residual Split (Phase 2)

```python
# train()
if self._use_physics_residual:
    physics_aligned = physics_kwh_series.reindex(df["timestamp"])
    n_nans = physics_aligned.isna().sum()
    if n_nans > 0:
        self._log(f"WARNING: {n_nans} hours have no physics prediction — set to 0 in residual target (check weather data gaps)")
    physics_vals = physics_aligned.fillna(0).values  # NaN → 0, never propagate into target
    df["_target"] = df["gross_kwh"] - physics_vals
    use_log_transform = False  # residual can be negative; log1p is invalid
else:
    df["_target"] = df["gross_kwh"]
    use_log_transform = True   # Phase 1 / v0.11.4 behaviour unchanged

# predict()
lgbm_raw = self._model.predict(X)
if self._use_physics_residual:
    return (physics_baseline + lgbm_raw).clip(lower=0)
return lgbm_raw.clip(lower=0)   # clip applied consistently in both phases
```

**Gross MAE reporting** — always computed on gross_kwh regardless of phase:
```python
gross_pred = (physics_baseline + lgbm_holdout_raw) if self._use_physics_residual else lgbm_holdout_raw
holdout_mae_gross = mean_absolute_error(y_gross_holdout, gross_pred)
self._log(f"Holdout MAE (gross kWh): {holdout_mae_gross:.4f}")
# residual MAE logged separately in Phase 2 as an internal diagnostic
```

CV, sample weighting, and interval calibration operate on `_target` as today. After Phase 2 deployment, verify empirical interval coverage on gross kWh — the quantile models retrain on the lower-variance residual distribution and reconstructed intervals may differ in width from Phase 1.

### 5.3 New HA Sensors (Phase 2)

Published via existing MQTT discovery path:
- `sensor.ha_energy_forecast_physics_base_today` — hourly physics baseline [kWh/h]
  - State: current-hour value (float, kWh/h)
  - Attributes: `{"hourly_kwh": [v0, v1, ..., v47]}` — 48-value list from midnight, matching the `sensor.ha_energy_forecast_consumption_today` payload structure
- `sensor.ha_energy_forecast_ml_adjustment_today` — behavioural residual [kWh/h, may be negative]
  - Same payload structure as above

The existing main consumption forecast sensor gains one new attribute in both phases: `model_phase` = `"phase1"` or `"phase2"` (reflects `use_physics_residual` at last train time). This lets external consumers (e.g. the LP scheduler in ha-energy-manager) detect a phase transition and log a recalibration prompt without polling config.

### 5.4 Scenario API Extension

`get_scenario` gains an optional `dhw_schedule` parameter:

```python
call_service("energy_forecast/get_scenario",
    schedule={"sub_waschmaschine": "10:00"},   # appliance overlay — unchanged
    dhw_schedule={"boost": "13:00"},            # physics baseline override — new
    publish=True)
```

When `dhw_schedule` is present: triggers fresh `predict_series(dhw_schedule_override=...)` rather than using `_cached_forecast_df`. Appliance overlay applied on top as today. Delta computed vs the **natural baseline** — defined as `predict_series()` using the current committed schedule from `physics_schedule.json` at the time of the call (including any prior `set_dhw_schedule` updates). The delta shows the incremental impact of the proposed `dhw_schedule` argument relative to whatever schedule is already in force; it is never computed relative to zero or the original factory schedule.

New service `energy_forecast/set_dhw_schedule`: confirmed intent from energy manager; updates `physics_schedule.json` (atomic write-then-rename), **immediately invalidates `_cached_forecast_df`** so the next HA sensor publish reflects the updated schedule, and bypasses the legionella instability guard (see §3.3). Uses same `dhw_schedule_override` mechanism. Together with `get_scenario(dhw_schedule=...)`, this forms the oracle pattern for energy manager optimisation (see §8).

**Caller cache responsibility**: ha-energy-forecast invalidates its own `_cached_forecast_df` on `set_dhw_schedule`, but callers that maintain a secondary scenario cache (e.g. `ConsumptionForecastApp._scenario_cache` / `_scenario_store`) must flush their own cache immediately after a successful `set_dhw_schedule` call — all prior `delta_kwh` entries were computed against the old DHW baseline and are now stale.

**`dhw_schedule` caching rule**: `get_scenario` calls that include `dhw_schedule` bypass ha-energy-forecast's internal cache. Callers must either (a) exclude results from their own scenario cache when `dhw_schedule` is present, or (b) include the `dhw_schedule` dict in their cache key hash. The safest default: do not cache any scenario result computed with a non-None `dhw_schedule`.

**Latency**: physics-enabled `predict_series()` adds O(1ms) for the 48-step DHW ODE; total p99 latency including feature engineering and LightGBM inference is expected < 500ms on typical HA hardware. The EM's existing 2-second scenario timeout requires no change.

### 5.5 Connections to Existing Infrastructure

| Existing | Role in new design |
|---|---|
| `_calibrate_tau()` passive windows | Refactored into shared `_find_passive_windows()` |
| `thermal_pressure` (area-weighted setpoint − actual) | Feeds T_indoor_h; already correct signal |
| `weighted_solar_gain` (radiation × cosine) | Reused as GHI input to solar_gain_area OLS |
| COP proxy `0.11×T + 3.0` | Replaced by Carnot-bounded formula + sensor |
| `dhw_buffer_temp` + `dhw_pressure` | Extended: buffer_temp becomes ODE initial state; rising buffer_temp flags DHW-active hours excluded from UA_eff OLS |
| `_project_indoor_temps()` (Euler ODE with τ) | Unchanged; T_indoor_h series sourced from here for Q_loss |

---

## 6. Error Handling and Fallbacks

### 6.1 Fallback Hierarchy

```
physics_kwh for any hour
  ├─ calibration fresh + valid      → full ODE prediction
  ├─ calibration stale (>30 days)   → stale params + WARNING
  ├─ calibration failed / missing   → config defaults + WARNING
  └─ defaults missing               → 0 (ML-only, silent)

Phase 2 specifically:
  any exception in predict_series() → ML-only for that cycle (lgbm_raw unmodified)
```

### 6.2 Specific Cases

| Failure | Handler |
|---|---|
| `physics_calibration.json` missing or corrupt | Re-run calibration; if fails, use config defaults |
| `physics_schedule.json` missing, corrupt, or schema-mismatched | Use `.get(key, default)` for all fields; missing keys fall back to config defaults without raising; log WARNING |
| `calibrated_at` key absent or unparseable | Treat as epoch zero (always stale); trigger re-calibration |
| <30 winter nights with ΔT ≥ 8 K for UA_eff | `UA_eff = None`; skip heating component; WARNING |
| DHW sensor unavailable at predict time | Assume `(T_upper + T_lower) / 2`; continue |
| COP sensor unavailable | Fall through: sensor → Carnot-bounded formula → 2.8 constant |
| DHW ODE temp outside `[T_lower, T_legionella]` | Clamp; log DEBUG |
| Legionella timing unstable >±2h/week (autonomous inference) | WARNING; suspend autonomous schedule learning; require explicit `set_dhw_schedule` |
| GHI missing from weather forecast | Solar contribution = 0 for those hours |
| Phase 2 physics baseline negative | Clip to 0; WARNING |
| `physics_kwh_series.reindex()` produces NaN (timestamp gap or missing hours) | `.fillna(0)` before residual subtraction; NaN count logged as WARNING if > 0; NaN never propagates into LightGBM target |
| `dhw_tank_temp_sensor` not configured | DHW-active hours cannot be filtered from UA_eff OLS; `min_delta_t` raised to 12 K automatically; WARNING logged |
| Model trained with `physics_kwh` but `physics_model is None` at predict time | Fill `physics_kwh = 0.0` for all rows; log WARNING; continue (graceful degradation) |

---

## 7. Testing

New test file: `tests/test_physics.py`.

**Unit — ThermalPhysicsModel**
- `predict_series()` with `UA_eff=150, ΔT=10°C, COP=3.0` → `0.5 kWh/h`
- Solar offset reduces heating load proportionally
- Internal gains reduce Q_heat: `Q_gain_int = Q_base_el × 800` subtracted from Q_loss
- Thermal mass term: rising projected indoor temp reduces Q_heat; falling indoor temp increases it
- COP priority: sensor for past hours; Carnot-bounded formula for future; constant when neither
- COP Carnot bound: at T_outdoor = −15°C, result ≤ Carnot limit and ≥ COP_min = 1.1
- DHW ODE: cycle triggers at T_lower; stops at T_upper; clamps enforced
- `heating_rise` derived from `Q_dhw_power / C_dhw` — confirmed not a fixed constant
- Post-legionella silence: tank initialised at 60°C → correct days of zero DHW electricity
- `dhw_schedule_override`: shifts DHW electricity to specified hour
- ODE edge cases: temp clamping, zero ΔT, heating cutoff in summer

**Unit — Calibration**
- `_calibrate_base_load()`: synthetic summer nights → median recovered within 5%
- `_calibrate_ua()`: generate data from known `UA_eff=150` → OLS recovers within ±20%; hours with rising `dhw_buffer_temp` excluded; ΔT < 8 K rows excluded; post-holdout rows excluded
- `_calibrate_solar_gain()`: inject known GHI offset → `solar_gain_area` recovered
- `_infer_dhw_schedule()`: 4 synthetic legionella peaks, Wednesday 14:00 → correct dow/hour/T_legionella
- Insufficient data: each step returns `None` + WARNING (not exception)
- Instability guard: peaks spread across 3 days → WARNING, timing not updated
- `_find_passive_windows()`: excludes post-holdout timestamps; excludes ΔT < 8 K rows; excludes HP-on rows

**Integration — model.py**
- `physics_kwh` column present in feature matrix when `physics_model` configured; **absent** (not zero-filled) when `None`
- `heating_buffer_temp` column present in feature matrix when sensor configured
- Phase 2: LightGBM target is `gross_kwh − physics_kwh` with `.fillna(0)` applied before subtraction
- Phase 2: `predict()` returns `(physics_kwh + lgbm_raw).clip(lower=0)`
- Phase 1: `predict()` returns `lgbm_raw.clip(lower=0)` — clip behaviour consistent across phases
- Phase 2: log1p transform disabled; Phase 1: log1p transform active
- Both phases: holdout MAE logged on gross_kwh; Phase 2 additionally logs residual MAE separately
- Phase 2 → Phase 1 regression: `use_physics_residual=False` → identical output to pre-physics model

**Fallback**
- `predict_series()` with no calibration → Series of zeros, no exception
- Phase 2 with physics exception → ML-only output (lgbm_raw unmodified)
- Missing GHI column → solar contribution silently zero
- `physics_kwh_series` with NaN values → `.fillna(0)` applied; WARNING logged; LightGBM target contains no NaN
- `physics_schedule.json` with missing keys → `.get()` fallback, no KeyError
- `physics_schedule.json` with invalid JSON → WARNING + config defaults, no crash
- `UA_eff = None` → `C_building_Wh_K` not evaluated; `Q_heat_el = 0`; no exception
- Model trained with `physics_kwh` + `physics_model=None` at predict time → `physics_kwh` filled with 0.0; WARNING logged; predict succeeds
- `_calibrate_ua()` with R² < 0.5 → result discarded; WARNING logged; UA_eff = config default
- `dhw_tank_temp_sensor` absent → `min_delta_t` raised to 12 K; WARNING logged

**Scenario API**
- `get_scenario` with `dhw_schedule` → fresh prediction (not cache)
- `get_scenario` without `dhw_schedule` → cache used (no regression)
- Delta computed vs natural physics baseline, not vs zero
- `set_dhw_schedule(A)` then `get_scenario(dhw_schedule=B)` → delta computed vs baseline with schedule A (not B); natural baseline uses updated standing schedule A
- `set_dhw_schedule` → `_cached_forecast_df` invalidated; next HA sensor publish reflects new schedule

---

## 8. ha-energy-manager Integration Review

*Reviewed from the perspective of ha-energy-manager as a consumer of ha-energy-forecast results. Issues are listed with their resolution status and the spec section where the fix was applied.*

### Issue 1 — `set_dhw_schedule` silently invalidates EM scenario cache — **RESOLVED (§5.4)**

§5.4 now explicitly mandates: "Callers that maintain a secondary scenario cache must flush their own cache immediately after a successful `set_dhw_schedule` call." For `ConsumptionForecastApp` this means clearing `_scenario_cache` and saving `_scenario_store` after each successful service call. The spec also clarifies that ha-energy-forecast invalidates only its own `_cached_forecast_df`; secondary caches are the caller's responsibility.

### Issue 2 — `dhw_schedule` not part of the scenario cache key — **RESOLVED (§5.4)**

§5.4 now states the caching rule: callers must either (a) exclude results computed with a non-None `dhw_schedule` from their scenario cache, or (b) include the `dhw_schedule` dict in their cache key hash. The safest default is (a). This prevents false cache hits when the same appliance schedule is paired with different DHW schedules.

### Issue 3 — Scenario response timeout may be too tight with physics — **RESOLVED (§5.4)**

§5.4 now states the latency expectation: "physics-enabled `predict_series()` adds O(1ms) for the 48-step DHW ODE; total p99 latency including feature engineering and LightGBM inference is expected < 500ms on typical HA hardware. The EM's existing 2-second scenario timeout requires no change."

### Issue 4 — `physics_base_today`/`ml_adjustment_today` payload format undefined — **RESOLVED (§5.3)**

§5.3 now specifies: state = current-hour value (float, kWh/h); attributes = `{"hourly_kwh": [v0, v1, ..., v47]}`, matching the `sensor.ha_energy_forecast_consumption_today` payload structure. The bridge can expose these as `em_` entities using the same JSON-attribute parsing path as the existing consumption sensor.

### Issue 5 — No observable phase indicator for LP scheduler risk calibration — **RESOLVED (§5.3)**

§5.3 now specifies that the existing main consumption forecast sensor gains a `model_phase` attribute (`"phase1"` or `"phase2"`), updated at each train cycle. The EM can read this attribute to detect the Phase 1→2 transition and log a recalibration prompt for `risk_aversion`.

---

## 9. Future Extensions

These are noted for awareness; none are in scope for the initial implementation.

**Energy manager feedback loop**

The `set_dhw_schedule` / `get_scenario(dhw_schedule=...)` pair forms the oracle pattern: the energy manager queries the forecast for optimal scheduling, then confirms its decision back. Full integration requires a stable interface contract between `ha-energy-manager` and `ha-energy-forecast`. The current architecture keeps this path open via the injectable `dhw_schedule_override` parameter.

**15-minute model**

`energy_history_15m.csv` is accumulating. Revisit once ≥12 months of data are available (earliest: Oct 2026).

**Community accuracy network**

`UA_eff` and `solar_gain_area` are anonymisable home fingerprints — the building's thermal leakiness and solar aperture. These are natural seeds for cross-home benchmarking and cold-start acceleration for new installs.

**Foundation model pre-training**

Academic datasets (REFIT, IDEAL) as cold-start primer; fine-tune on home-specific data to cut the learning period from months to days.

**Temperature-similarity sample weighting (#88)**

Already simulated (see ROADMAP). Revisit once first full year of data is available (Oct 2026).
