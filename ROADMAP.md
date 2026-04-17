# Forecast Accuracy Roadmap

Current: **v0.11.0-alpha-1** — 2026-04-17, main. 474 tests.

---

## Current Status

Sub-sensor history for the heat pump (integrated 2026-03-18) was expected to reach full lag-feature activation (~672 rows) around 2026-04-20. MAE has improved from 0.7 → 0.52 kWh/h.

**v0.11.0-alpha-1 Update:** Daily Regime Clustering implemented as an optional module.
 This explicitly extracts 24h consumption patterns and uses a secondary model to predict the expected "regime" for tomorrow, providing a stable physics-informed prior to the main model.

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

### #59 — Relaxed Thermal Calibration Constraints

**Signal:** The current τ-calibration is conservative to avoid solar/daytime corruption. Relaxing slightly may allow faster calibration in spring/autumn.

**Proposed changes:**
- Raise max solar radiation mask from 150 → 250 W/m².
- Allow 2 qualifying windows (down from 3) when historical variance is low.

**Effort:** ~30 min. **Impact:** MEDIUM — faster τ convergence.

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

### #62 — Adaptive Regime Selection (Auto-K)

**Signal:** Instead of a fixed `regime_count`, automatically find the optimal number of clusters ($K$) that maximizes the balance between clustering quality (Silhouette Score) and weather-based predictability.

**Proposed changes:**
- Iterate $K \in [2, 8]$ during the clustering stage.
- Calculate `silhouette_score` for each $K$ via `sklearn.metrics`.
- Run internal cross-validation on `RegimePredictor` to measure predictability.
- Select $K$ that maximizes $\text{Silhouette} \times \text{Accuracy}$.
- Activate when `regime_count: 0`.

**Effort:** ~2 h. **Impact:** MEDIUM — zero-config optimization.

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
| 15 | HVAC flow setpoint | high (heat pump) | 3 h | escalate if bouncing |
| 59 | Relaxed τ calibration | medium | 30 min | backlog |
| 10 | School holidays | medium | 4 h | long-term |
| 46 | Dashboard entity ID cleanup | UX / sharing | 30 min | pre-HACS |
| 16 | HACS support | distribution | 1 h | long-term |
| 18 | Config flow | UX / install | 8+ h | long-term |
| 22 | EV SoC | high (EV) | 4 h | deferred |
| 40 | Battery SoC | medium (battery) | 1 h | deferred |
| 24 | Spot price | n/a | — | out of scope |

---

---

## Done

### Release History

| Version | Date | Highlights |
|---------|------|------------|
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
