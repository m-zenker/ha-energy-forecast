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
- `physics_kwh` added to `_FEATURES_BASE` alongside existing features
- LightGBM trains on `gross_kwh` as today; learns when to trust or override the physics signal
- Validate via SHAP: if `physics_kwh` ranks high, calibration is good; proceed to Phase 2

**Phase 2 — Explicit residual split (structural)**
- LightGBM trains on `gross_kwh − physics_kwh`
- `predict()` returns `physics_kwh + lgbm_residual`
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
        │           LightGBM target = gross_kwh
        │
        └─ Phase 2: LightGBM target = gross_kwh − physics_kwh
                    predict() = physics_kwh + lgbm_residual
```

### 2.2 New Files

| File | Purpose |
|---|---|
| `apps/energy_forecast/physics.py` | `ThermalPhysicsModel` class |
| `models/physics_params.json` | Calibrated + inferred parameters |

### 2.3 Files Modified

| File | Change |
|---|---|
| `model.py` | `_engineer_features()` gets `physics_kwh_series` param; `train()` / `predict()` Phase 2 logic; `_find_passive_windows()` promoted to shared utility |
| `energy_forecast.py` | Config ingest; new sensors; `recalibrate_physics` service; Scenario API `dhw_schedule` extension |

### 2.4 Files Unchanged

`ha_data.py`, `weather.py`, `clustering.py`, `const.py`, all MQTT sensor IDs, all existing tests.

---

## 3. Physics Model

### 3.1 Components

**Space heating**
```
T_setpoint_avg    = area-weighted mean of room thermostat setpoints
Q_loss   [W]      = UA_eff × max(0, T_setpoint_avg − T_outdoor)
Q_solar  [W]      = solar_gain_area × GHI
Q_heat   [W]      = max(0, Q_loss − Q_solar)
Q_heat_el [kWh/h] = Q_heat / COP(T_outdoor) / 1000
```

COP resolution (priority order):
1. Live HA sensor (`cop_sensor`) — past and current hours
2. Configurable formula `COP(T) = a + b × T` — future hours
3. Constant fallback (default 2.8)

**DHW — tank ODE**

State-simulated forward per hour from current `T_tank`:
```
dT = −UA_dhw × (T_tank − T_ambient) / C_dhw   # insulation loss
   − draw_profile[h] × draw_rate               # hot-water draws

if T_tank < T_lower:  Q_dhw_el = Q_dhw_power / COP_dhw
else:                 Q_dhw_el = 0

T_tank = clamp(T_tank + dT + heating_rise, T_lower, T_legionella)
```

Post-legionella silence emerges naturally: tank initialises at `T_legionella` (60°C) and does not trigger heating until it cools below `T_lower`. No special-casing required.

Initial state:
- `dhw_tank_temp_sensor` reading if <2h old
- Else: estimated from time elapsed since last legionella via passive cooling ODE

**Heating buffer**

Used as a direct feature only. `heating_buffer_temp_sensor` (current reading) tells the model whether the HP is about to coast or run — a near-term signal for hours 0–1. No forward ODE, no volume parameter. Steady-state physics handles hours 2–48.

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
- **Prediction:** 48h setpoint schedule projected forward from thermostat weekly programs (same per-room ODE as `_project_indoor_temps()`, which already handles this)
- **Open window detection:** per-room residual `dT_actual − dT_ODE` > 2σ → `open_window_hour` flag stored in the training frame. Affected hours are down-weighted during training (analogous to how EV-adjacent hours are down-weighted for MAE; not excluded outright). Not used in the prediction path — windows cannot be forecast.

### 3.3 DHW Schedule — Operational vs Calibrated

DHW and legionella timing is **operational** (can change any cycle) rather than calibrated (stable physical property). It is stored as a writable slot in `physics_params.json` and injectable at runtime:

```python
physics_model.predict_series(..., dhw_schedule_override={"legionella": ("2026-06-25", 10)})
```

**Stability guard:** if inferred legionella timing shifts >±2h week-over-week, log WARNING and suspend schedule learning. This is the signal that an energy manager is actively shifting cycles.

---

## 4. Calibration

### 4.1 Trigger

- Startup, if `physics_params.json` absent or `calibrated_at` >30 days old
- On demand: AppDaemon service `energy_forecast/recalibrate_physics`
- Triggered inside `train()` (which already has all required data in scope)

`_find_passive_windows()` is promoted from `_calibrate_tau()` to a shared module-level utility used by both τ and UA_eff calibration.

### 4.2 Parameters Calibrated from History

| Parameter | Method | Data required | Fallback |
|---|---|---|---|
| `Q_base_el` [kWh/h] | Median of summer nights (01–05h, Jun–Aug, no EV, no away) | 14 nights | 0.35 kWh/h |
| `Q_dhw_daily` [kWh] | Summer daily mean − 24 × Q_base_el | 14 days | 3.5 kWh |
| `UA_eff` [W/K] | Nighttime OLS: `Q_heat_obs ≈ UA_eff/1000 × ΔT/COP` (Nov–Mar, 22–06h) | 30 winter nights | None — log WARNING; skip heating component |
| `solar_gain_area` [m²] | Daytime OLS residual with UA known (GHI >50 W/m²) | 14 sunny winter days | 0 m² (conservative) |
| `UA_dhw` [W/K] | Passive decay regression on DHW sensor during HP-off periods | DHW sensor + HP-off windows | 15 W/K (200L spec) |

### 4.3 Parameters Inferred from Sensor Data

All inferred values override config defaults and are stored in `physics_params.json`. Config override keys are available as escape hatches (commented-out in config example).

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
- Added to `_FEATURES_BASE`

### 5.2 Residual Split (Phase 2)

```python
# train()
if self._use_physics_residual:
    df["_target"] = df["gross_kwh"] - physics_kwh_series.reindex(df["timestamp"]).values
else:
    df["_target"] = df["gross_kwh"]

# predict()
lgbm_raw = self._model.predict(X)
if self._use_physics_residual:
    return (physics_baseline + lgbm_raw).clip(lower=0)
return lgbm_raw
```

CV, sample weighting, holdout MAE, and interval calibration are all unchanged — they operate on `_target` regardless of what it represents.

### 5.3 New HA Sensors (Phase 2)

Published via existing MQTT discovery path:
- `sensor.ha_energy_forecast_physics_base_today` — hourly physics baseline [kWh/h]
- `sensor.ha_energy_forecast_ml_adjustment_today` — behavioural residual [kWh/h, may be negative]

### 5.4 Scenario API Extension

`get_scenario` gains an optional `dhw_schedule` parameter:

```python
call_service("energy_forecast/get_scenario",
    schedule={"sub_waschmaschine": "10:00"},   # appliance overlay — unchanged
    dhw_schedule={"boost": "13:00"},            # physics baseline override — new
    publish=True)
```

When `dhw_schedule` is present: triggers fresh `predict_series(dhw_schedule_override=...)` rather than using `_cached_forecast_df`. Appliance overlay applied on top as today. Delta computed vs natural physics baseline.

New service `energy_forecast/set_dhw_schedule`: confirmed intent from energy manager; updates standing forecast. Uses same `dhw_schedule_override` mechanism. Together with `get_scenario(dhw_schedule=...)`, this forms the oracle pattern for energy manager optimisation (see §7).

### 5.5 Connections to Existing Infrastructure

| Existing | Role in new design |
|---|---|
| `_calibrate_tau()` passive windows | Refactored into shared `_find_passive_windows()` |
| `thermal_pressure` (area-weighted setpoint − actual) | Feeds T_setpoint_avg; already correct signal |
| `weighted_solar_gain` (radiation × cosine) | Reused as GHI input to solar_gain_area OLS |
| COP proxy `0.11×T + 3.0` | Replaced by configurable formula + sensor |
| `dhw_buffer_temp` + `dhw_pressure` | Extended: buffer_temp becomes ODE initial state |
| `_project_indoor_temps()` (Euler ODE with τ) | Unchanged; setpoint schedule projection already handled |

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
| `physics_params.json` missing or corrupt | Re-run calibration; if fails, use defaults |
| <30 winter nights for UA_eff | `UA_eff = None`; skip heating component; WARNING |
| DHW sensor unavailable at predict time | Assume `(T_upper + T_lower) / 2`; continue |
| COP sensor unavailable | Fall through: sensor → formula → 2.8 constant |
| DHW ODE temp outside `[T_lower, T_legionella]` | Clamp; log DEBUG |
| Legionella timing unstable (>±2h/week shift) | WARNING; suspend schedule learning; require explicit override |
| GHI missing from weather forecast | Solar contribution = 0 for those hours |
| Phase 2 physics baseline negative | Clip to 0; WARNING |

---

## 7. Testing

New test file: `tests/test_physics.py`.

**Unit — ThermalPhysicsModel**
- `predict_series()` with `UA_eff=150, ΔT=10°C, COP=3.0` → `0.5 kWh/h`
- Solar offset reduces heating load proportionally
- COP priority: sensor for past hours; formula for future; constant when neither
- DHW ODE: cycle triggers at T_lower; stops at T_upper; clamps enforced
- Post-legionella silence: tank initialised at 60°C → correct days of zero DHW electricity
- `dhw_schedule_override`: shifts DHW electricity to specified hour
- ODE edge cases: temp clamping, zero ΔT, heating cutoff in summer

**Unit — Calibration**
- `_calibrate_base_load()`: synthetic summer nights → median recovered within 5%
- `_calibrate_ua()`: generate data from known `UA_eff=150` → OLS recovers within ±20%
- `_calibrate_solar_gain()`: inject known GHI offset → `solar_gain_area` recovered
- `_infer_dhw_schedule()`: 4 synthetic legionella peaks, Wednesday 14:00 → correct dow/hour/T_legionella
- Insufficient data: each step returns `None` + WARNING (not exception)
- Instability guard: peaks spread across 3 days → WARNING, timing not updated

**Integration — model.py**
- `physics_kwh` column present in feature matrix when `physics_model` configured; zero when `None`
- Phase 2: LightGBM target is `gross_kwh − physics_kwh`; `predict()` restores total
- Phase 2 → Phase 1 regression: `use_physics_residual=False` → identical output to pre-physics model

**Fallback**
- `predict_series()` with no calibration → Series of zeros, no exception
- Phase 2 with physics exception → ML-only output
- Missing GHI column → solar contribution silently zero

**Scenario API**
- `get_scenario` with `dhw_schedule` → fresh prediction (not cache)
- `get_scenario` without `dhw_schedule` → cache used (no regression)
- Delta vs natural physics baseline, not vs zero

---

## 8. Future Extensions

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
