# Forecast Accuracy Roadmap

Proposed improvements to `ha-energy-forecast`, ordered by impact tier.
Current baseline: **v0.10.2-alpha-12** (2026-04-14, on dev). Stable (main): v0.9.0 (released 2026-04-10).

---

## Current Status — Sub-sensor Feature Maturation (Path A: Monitor)

**Date (2026-04-14):** Sub-sensor history (~27 days, ~648 rows) approaching the 672-row threshold for full lag feature activation. MAE improvement from 0.7 → 0.52 kWh/h observed so far. No action required until after ~2026-04-20 if bouncing pattern persists.

**Context:** Sub-sensor integration for heat pump (added 2026-03-18) shows expected settling behavior — lag features like `lag_48h` and `lag_168h` only activate when `n_rows >= MIN_CV_ROWS` (500 rows). Activation is now in progress; bouncing pattern expected to flatten as the model learns temporal stability across multi-day windows.

**Timeline:** By ~2026-04-20, sub-sensor history reaches 4+ weeks (≥672 rows), triggering full lag feature activation.

**If bouncing persists after 2026-04-20:** Revisit Path B — HVAC/thermostat state as an anticipatory signal (item #15).

**No action required** until after 2026-04-20 if bouncing remains.

---

## Staged Development & Deployment Plan

### Release milestones

| Milestone | Version | Contents | Status |
|-----------|---------|----------|--------|
| Hotfix merge | v0.5.3 | Merge `dev` → `main`: log noise reduction, XX:01 hourly alignment | done |
| Entity registry | v0.6.0 | #37 MQTT Discovery (entity registry, area assignment, labels) | ✓ done |
| Accuracy + visibility + explainability | v0.7.0 | #38 Full 48h weather features (✓ done), #25 Vacation flag (✓ done), #41 Rolling MAE sensor (✓ done), #39 Anomaly detection sensor (✓ done), #42 SHAP feature importance (✓ done), quantile interval calibration (✓ done), #43 ApexCharts dashboard (✓ done) | ✓ done |
| Bug-fix + dashboard polish | v0.7.1 | #47 entity_exists guard (404 DELETE spam), #48 MQTT anomaly sensor attrs | ✓ done |
| Solar + battery + ops safety | v0.8.0 | #23 B1 target correction; #44 model versioning + rollback; #45 CSV health checks | ✓ done (on main) |
| Occupancy + thermal modelling | v0.9.0 | #21 Occupancy (`people_home`), #49 EWMA temperature, #50 Rolling degree-hour sums, #51 Temperature rate of change, #52 Temperature lags, #53 SHAP narrative, #54 Relative MAE sensors | ✓ done (on main, released 2026-04-10) |
| Baseline + scenario API | v0.10.0 | Stage 1 baseline_mode, Stage 2 thermal/DHW intent, Stage 3 appliance signatures, Stage 4 scenario/what-if API | ✓ done (on dev) |
| Selective baseline + dtype fix | v0.10.1 | `baseline_included_sensors`, pandas 3.x dtype coercion fix in `_merge_frames` | ✓ done (on dev) |
| Thermal accuracy suite | v0.10.2-alpha | τ calibration (OLS passive-decay), RC-ODE indoor projection, area-weighted pressure, `thermal_pressure_cop`, `weighted_solar_gain`, program-type appliance signatures, 62 SHAP labels | in progress (on dev, alpha-12) |
| Long-term | v1.x+ | #16 HACS, #10 School holidays, #15 HVAC, #18 Config flow | backlog |

### Deployment workflow (per release)

1. Feature branch → implement + tests pass (`python -m pytest tests/ -v`)
2. PR → code review → merge to `dev`
3. Smoke-test on local HA instance (watch AppDaemon log; confirm sensors update)
4. PR `dev` → `main` after stable period on local instance
5. Update CHANGELOG.md (close `[Unreleased]` → `vX.Y.Z`)
6. Create semver tag (`git tag vX.Y.Z`) → push tag → GitHub release with notes
7. After #16: HACS auto-picks up new semver tag for AppDaemon category listing

---

## User priorities & design decisions

Captured from user interview (2026-03-24). These constrain scope and feature design.

| Topic | Decision |
|---|---|
| Primary goal | **Forecast accuracy** + visibility/dashboards |
| Solar PV | Planned soon — design features for it now (#23, #40) |
| Home battery | Coming with solar — SoC as feature (#40) |
| Tariff | Fixed flat rate — price optimisation is **out of scope** |
| Load shifting | **Explicitly out of scope** — handled by a separate system |
| Audience | Personal-first; HACS nice-to-have but never at cost of accuracy |

> **Critical definition:** *Consumption* = total household consumption
> (`grid_import − grid_export + solar_production − battery_charge + battery_discharge`).
> **Not** net load. **Not** grid-only import. This definition applies now and
> once solar/battery arrive — the forecast target never changes.

**Out of scope (per above):** all load-shifting / scheduling features,
spot-price feed (#24 moved to backlog), net-load forecast,
Docker standalone, HA Energy dashboard integration.

---

## Tier 1 — High impact, low effort (quick wins)

### ~~1. Fix missing sunshine in Open-Meteo fallback~~ ✓ done
### ~~2. Add `temp_rolling_3d` to the prediction horizon~~ ✓ done
### ~~3. Pre/post-holiday bridge day features~~ ✓ done
### ~~4. Cloud cover / solar irradiance feature~~ ✓ done

---

## Tier 2 — High impact, medium effort

### ~~5. Fix training/prediction mismatch in rolling features~~ ✓ done
### ~~6. LightGBM early stopping + validation-set tuning~~ ✓ done
### ~~7. Log-transform the target~~ ✓ done
### ~~8. Adaptive retraining trigger~~ ✓ done

---

## Tier 3 — Medium impact, higher effort

### ~~9. Cantonal public holidays~~ ✓ done

### 10. School holiday feature *(long-term backlog)*
Swiss Schulferien dates are canton-specific but stable year-to-year. During
school holidays household daytime consumption rises (children at home). None of
the current features capture this. Implement a static lookup table per canton,
configurable via `apps.yaml`, and add `is_school_holiday` to `_FEATURES_BASE`.

### ~~11. Additional lag: `lag_72h`~~ ✓ done
### ~~12. EV charge session probability feature~~ ✓ done

---

## Tier 4 — Longer-term / architectural

### ~~13. Prediction intervals as HA sensors~~ ✓ done
### ~~14. Intra-day actuals substitution~~ ✓ done

### 15. HVAC / boiler state — projected flow setpoint *(long-term backlog; escalate if Path A bouncing persists after 2026-04-20)*

**Signal:** Derive a `flow_setpoint` feature from the Kermi heating curve rather than reading a raw
sensor — this allows accurate 48-hour forward projection using forecast outdoor temps.

**Projection formula (per future hour h):**
```
flow_setpoint(h) = np.interp(outdoor_temp[h], curve_x, curve_y)
                   + parallel_shift          # current HA entity value, projected flat
                   - 2  if 21 ≤ hour < 24
                       or  0 ≤ hour < 6     # night setback
                   → NaN if outdoor_temp[h] ≥ 20  # heating cutoff
```

**Heating curve breakpoints (from Kermi UI):**

| Outdoor °C | Flow setpoint °C |
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

Slope steepens at the warm end — `np.interp` handles piecewise linear exactly; a single slope
approximation would lose accuracy at the extremes.

**Parallel shift:** Available as a live HA sensor entity (`sensor.kermi_parallel_shift` or similar).
Read at prediction time; project flat across 48h (changes rarely — a few times per season at most).

**Night setback:** −2°C, 21:00–06:00. Captured as a feature, so the model learns that heat pump
load drops during setback hours even on cold nights.

**Heating cutoff:** outdoor temp ≥ 20°C → heating circuit off → `flow_setpoint = NaN` (median-filled
at predict time, same as other NaN features).

**Training path:** apply same formula to historical `temp_c` column + fetched parallel shift history
→ `flow_setpoint` column in training frame.

**Additional signals available (not in scope for first implementation):**
- Buffer temperature (h=0 current-state signal; poor projection beyond first hour)
- Flow temperature (similar to buffer)
- DHW buffer temperature (usage-driven; poor projection)
These can be added as a second pass if `flow_setpoint` alone shows limited gain.

**`apps.yaml` config keys:**
```yaml
heating_curve_sensor: sensor.kermi_parallel_shift   # HA entity for curve offset
heating_curve_points:                               # piecewise lookup; user-configurable
  - [-20, 55.5]
  - [-15, 52.5]
  - [-10, 49.5]
  - [ -5, 46.0]
  - [  0, 43.0]
  - [  5, 39.5]
  - [ 10, 35.5]
  - [ 15, 31.0]
  - [ 20, 25.0]
heating_cutoff_temp: 20       # °C — NaN above this
night_setback_delta: -2       # °C
night_setback_start: 21       # hour, inclusive
night_setback_end: 6          # hour, exclusive
```

**Effort:** ~3 hours (similar architecture to #21 occupancy — history fetch + feature engineering +
prediction projection + tests).

**Impact:** HIGH for heat pump buildings. Directly captures system intent rather than just ambient
temperature pressure. Night setback signal explains the 21:00–06:00 consumption reduction without
relying on time-of-day features alone.

### 55. Verified Passive Decay — building thermal time constant (τ) *(backlog)*

**Origin:** Stage 2, task [2.4] from `temp/STAGED_ROADMAP_V1.md` — planned but not implemented.

**Idea:** Observe the indoor temperature drop rate during verified heating-off periods (gated by
`heating_system_active_entity`) to calibrate the building's thermal time constant τ (hours). A low
τ means the house cools fast → the heat pump turns on sooner after a setpoint raise → predictable
consumption spike.

**Current gap:** `heating_system_active_entity` is read from config and stored in
`self._heating_active_entity` (`energy_forecast.py:139`) but is never used. The `thermal_pressure`
feature is a raw `setpoint − current_temp` delta with no τ weighting.

**Implementation sketch:**
- During each retrain, scan the climate + `heating_system_active_entity` history for windows where
  heating was confirmed off for ≥ 2 h.
- Fit an exponential decay `T_indoor(t) = T_outdoor + (T0 − T_outdoor) · e^(−t/τ)` on those
  windows (scipy `curve_fit` or a simple log-linear OLS).
- Store τ in `meta.pkl` alongside NaN medians.
- Weight `thermal_pressure` by `1/τ` (or derive `hours_to_equilibrium = τ · ln(ΔT/threshold)`)
  to give the model a physically meaningful "urgency" signal instead of a raw degree delta.

**Dependencies:** `heating_system_active_entity` config key already wired up (`energy_forecast.py:139`).
Climate history fetch already implemented (`fetch_climate_history` in `ha_data.py`).

**Effort:** Medium (~4 h). Requires careful handling of summer/transition periods where heating is
off for weeks (τ estimation becomes noisy without an outdoor temp anchor).

**Impact:** Medium. Sharpens `thermal_pressure` accuracy, especially on buildings with fast thermal
decay. Pairs with #15 (HVAC state).

### 56. Solar-Compensated Thermal Pressure *(backlog)*
**Signal:** Integrate passive solar gain directly into the thermal demand signal. Passive solar heating reduces the actual energy required to maintain a setpoint, even if the temperature delta is still high. Integrating this helps the model learn to "wait" for the sun.

**Formula:**
`thermal_pressure_net = thermal_pressure - (k * weighted_solar_gain)`
Where `k` is a learnable coefficient (or captured via feature interaction in the ML model).

**Impact:** HIGH. Reduces forecast over-shooting on sunny winter days.

### 57. Wind-Driven Infiltration Feature *(backlog)*
**Signal:** Model the accelerated building heat loss caused by wind-driven air exchange and convection.

**Physics:** Infiltration loss is roughly proportional to $WindSpeed \cdot (T_{indoor} - T_{outdoor})$.

**Feature:** `infiltration_pressure = wind_kmh * thermal_pressure`

**Impact:** MEDIUM. Captures the "wind chill" effect on the building envelope.

### 58. Humidity-Aware Defrost Proxy (Heat Pump Specific) *(backlog)*
**Signal:** Predict energy spikes caused by air-source heat pump evaporator defrosting. Defrosting is most frequent when $T_{outdoor}$ is between $-2^\circ C$ and $+5^\circ C$ and humidity is high.

**Feature:** `defrost_risk = humidity * exp(-((T_out - 2)^2) / 10)`

**Impact:** MEDIUM. Explains "unexplained" energy spikes during foggy, near-freezing winter days.

---

## Distribution

### 16. HACS support *(long-term backlog)*

Make the app installable via [HACS](https://hacs.xyz/) (AppDaemon category).
Deprioritised in favour of accuracy and visibility work (v0.7.0–v0.9.0).

Required changes:
- Add `hacs.json` at repo root (HACS manifest; `render_readme: true`).
- Add `info.md` — shown in the HACS detail panel before installation; must prominently warn that HACS only copies app files and that the AppDaemon add-on dependency config and `apps.yaml` creation are still manual steps.
- Add a "Install via HACS" subsection to `README.md` under `## Installation`, explaining what HACS does and doesn't do, with links to the dependency config and `apps.yaml.example` steps.
- Set GitHub repo topics: `appdaemon` (required for HACS category), `home-assistant`, `hacs`.

No code changes required — `apps/energy_forecast/` is already in the correct location for HACS AppDaemon installs. Semver tags are already present.

### 17. Setup checker sensor ✓ done
Bake a startup self-check into the main app that surfaces setup problems as a visible HA entity rather than silent log failures.

- On initialisation, attempt `import pandas, numpy, lightgbm, sklearn, requests, holidays` and log a clear error for each missing package.
- Validate required config keys (`energy_sensor`, `latitude`, `longitude`); validate optional keys have sane types.
- Publish `sensor.energy_forecast_setup_status` with states `ok`, `missing_packages`, `missing_config`, or `invalid_config`, and an `issues` attribute listing specific problems.
- Self-silences (removes sensor) once all checks pass and the main app is running normally.

This converts silent failure after a fresh HACS install into actionable, user-visible feedback without requiring any Supervisor access.

### 19. CSV cache: append-only writes ✓ done
For long-running installs with months of history, `fetch_recent_energy` rewrites the entire `energy_history.csv` on every hourly update. At ~8 760 rows/year this is already measurable I/O and will compound over time.

Improvement: write only new rows using `df.to_csv(..., mode='a', header=False)` in `fetch_recent_energy`, and perform a periodic compaction (dedup + sort) in `fetch_energy_history` (the weekly full-read path) rather than on every update. Requires care around the merge-winner rule to avoid duplicating rows that already exist in the CSV.

### 20. Config validation: warn when `ev_charging_threshold_kwh >= ev_charger_kw` ✓ done
When the detection threshold is set at or above the charger power (e.g. threshold=10, charger=9), every detected EV hour produces `max(0, gross - charger_kw) = 0`, so the EV sensors always read zero while the model still strips those hours from training data. The combination is silently confusing.

Add a validation check in `_validate_config` that logs a `WARNING` when `self._ev_threshold >= self._ev_charger_kw`, explaining that the EV sensor will report 0 kWh for all detected sessions in that configuration.

### 21. Occupancy feature (`people_home`) *(✓ done — v0.9.0, on main)*
Implemented in v0.9.0. See MEMORY.md for details on presence history fetching and feature integration.

### 22. EV charging state + SoC feature *(DEFERRED)*
EV charging hours are subtracted from the training target during model fitting, so the model never sees EV load. Adding SoC/charging state as a feature provides no direct signal to learn from. Any value as an occupancy proxy is better covered by #21 (`people_home`). Revisit only if EV load is re-included in the target or if there is evidence of improved forecast accuracy.

### 49. Exponentially weighted moving average temperature *(✓ done — v0.9.0, on main)*
Implemented in v0.9.0: `temp_ewma_24h` and `temp_ewma_72h` with physically-motivated half-lives. See MEMORY.md for implementation details.

### 50. Rolling accumulated heating degree-hours *(✓ done — v0.9.0, on main)*
Implemented in v0.9.0: `heating_deg_sum_24h` and `heating_deg_sum_168h` for multi-day thermal patterns. See MEMORY.md for details.

### 51. Temperature rate of change feature *(✓ done — v0.9.0, on main)*
Implemented in v0.9.0: `temp_delta_1h` and `temp_delta_24h` for HVAC anticipation signals. See MEMORY.md for details.

### 52. Temperature lag features *(✓ done — v0.9.0, on main)*
Implemented in v0.9.0: `temp_lag_24h` and `temp_lag_168h` for day/week temperature patterns. See MEMORY.md for details.

### 23. Solar PV integration *(✓ merged to dev — v0.8.0)*
Target: `total_consumption = grid_import − grid_export + solar_production − battery_charge + battery_discharge`.
The model must always forecast **total consumption**, not net grid import.

Three sub-items (B1, B2, B9):

1. **B1 — Target correction** *(✓ done, merged v0.8.0)*: four optional config
   keys (`solar_production_sensor`, `grid_export_sensor`, `battery_charge_sensor`,
   `battery_discharge_sensor`) correct the training target before `model.train()` is called.
   No new model features — solar/battery influence is captured via the corrected target only.
   Solar production does not drive consumption; only the measurement needs correction.
2. **B2 — Solar forecast as prediction feature** *(deferred — out of scope for now)*: solar
   production prediction is explicitly out of scope. `direct_radiation_wm2` already in
   `_FEATURES_BASE` gives the model the solar irradiance signal.
3. **B9 — Battery SoC as feature** *(deferred — out of scope for now)*: battery SoC prediction
   is out of scope. Revisit if residuals show SoC correlation after panels are installed.

### 24. Electricity spot price feature *(out of scope — fixed tariff)*
~~Households on dynamic tariffs (Tibber, Nordpool) actively shift deferrable loads — dishwasher, washing machine, EV charging — to cheap hours. The model currently cannot learn this behaviour because it sees no price signal. Add an optional `price_sensor` config key; include the hourly price (or a `is_cheap_hour` binary derived from a configurable threshold) as a feature. The Tibber and Nordpool HA integrations already expose standardised hourly price sensors.~~

User is on a fixed flat tariff; price-driven load shifting is out of scope. Retained for reference only.

### 25. Vacation / away flag *(✓ done — v0.7.0)*
Multi-day absences cause baseline drops that look like anomalies to the rolling lag features and bias the model until history catches up.

Two optional, independent config keys:

- `away_mode_entity` (e.g. `input_boolean.vacation_mode`) → binary `is_away` feature. Sufficient on its own to give the model the basic home/away signal.
- `away_return_entity` (e.g. `input_datetime.vacation_return`) → used to flip `is_away` to 0 at the stored return hour during prediction, so the model sees the correct home/away state for future hours. The simpler binary approach was chosen deliberately. A `hours_until_return` numeric feature (pre-return consumption spike signal) can be added as a future enhancement if the pattern proves significant in residuals.

Both keys are optional and independent; `away_mode_entity` alone is enough for the basic feature.

### 18. Custom component config flow *(long-term backlog)*
A full HA custom component (lives in `custom_components/energy_forecast/`) that provides a UI-driven setup wizard via HA's config flow:

- Step 1: entity picker for `energy_sensor`; lat/lon auto-populated from HA's own location config.
- Step 2: optional fields (SRG credentials, outdoor temp sensor, canton, EV threshold).
- On completion: writes the `energy_forecast:` stanza into `appdaemon/apps/apps.yaml` via the HA filesystem API.
- Calls the Supervisor REST API (`/supervisor/addons/<appdaemon_slug>/options`) to patch `python_packages` and `init_commands` with the required dependencies, then triggers an add-on restart.

This is the only path to fully zero-manual-step installation. Significant effort: requires maintaining a separate HA integration type alongside the AppDaemon app, Supervisor API integration, and config flow UI.

---

## Tier 5 — Diagnostics, Performance & Minor Features

### 27. Short-horizon lags (`lag_1h`, `lag_2h`, `lag_6h`, `lag_12h`) ✓ done
The current lag set jumps from `lag_24h` to `lag_48h`, leaving a blind spot in the
1–12 h range that matters most for same-day intra-day prediction. Adding `lag_1h`,
`lag_2h`, `lag_6h`, and `lag_12h` to `_add_lag_features` (training) and
`_add_lags_prediction` (inference) is a direct feature-engineering win at negligible
data-volume cost — all four are available as soon as a single day of history exists.
Expected impact: **HIGH** for hours 1–6 ahead; Low effort.

### 28. `num_leaves` hyperparameter sweep — complete ROADMAP #6 ✓ done
ROADMAP item #6 added early stopping but left the `num_leaves` sweep (`16 / 31 / 63`)
as a follow-up. A narrow grid search on the final CV split can be wired into the
existing `_cross_validate` path without changing the training API. Prevents the model
from being locked into the LightGBM default of 31 leaves regardless of data volume.
Expected impact: **MEDIUM**; Low effort (sweep is already sketched in the #6 description).

### 29. Feature importance logging after training ✓ done
After `model.fit()` in `_train_model`, log `model.feature_importances_` sorted by
weight. Currently there is no visibility into which features the model actually uses.
One `logger.debug` call with the sorted list adds zero runtime cost and makes
under-contributing features immediately visible in the AppDaemon log.
Expected impact: Diagnostic; Trivial effort.

### 30. CV fold std logging alongside mean ✓ done
`_cross_validate` currently logs only the mean MAE across folds. Adding the standard
deviation (and optionally the per-fold breakdown at DEBUG level) surfaces high-variance
training runs — an early signal of insufficient data or a degraded feature — without
changing any model logic.
Expected impact: Diagnostic; Trivial effort.

### 31. Per-hour-of-week NaN fill medians ✓ done
NaN values in rolling/lag features are currently filled with the global training
median. A per-`hour_of_week` median (168 cells) is a much tighter imputation for the
warm-up period at install time, where the model would otherwise use a "typical any
hour" stand-in for a specifically 3 a.m. Tuesday slot. Requires computing and caching
a `(168,)` lookup table during training and applying it in `_build_prediction_features`.
Expected impact: **LOW–MEDIUM** (mainly during first week of data); Medium effort.

### 32. Holiday `apply` → `np.searchsorted` vectorization ✓ done
`_add_holiday_feature` calls `pd.Series.apply(lambda ts: ts.date() in holiday_set)`,
which is a Python loop over every row in the training frame. Replacing it with
`np.searchsorted` on a sorted date array (or a boolean index join) reduces the
holiday computation from O(n) Python-level iterations to a vectorised C operation.
Expected impact: Performance (training speed); Low effort.

### 33. Day-of-year cyclical feature (`doy_sin` / `doy_cos`) ✓ done
The model captures seasonality through rolling temperature features and calendar
proxies, but has no smooth cyclic encoding of position within the year. Adding
`doy_sin = sin(2π·doy/365)` and `doy_cos = cos(2π·doy/365)` to `_FEATURES_BASE`
gives the model a continuous signal for seasonal baseline that avoids the
discontinuity at New Year's Day.
Expected impact: **LOW**; Low effort.

### 34. `hours_ahead` feature for horizon-aware prediction ✓ done
All 48 prediction rows currently receive identical feature vectors; the model cannot
distinguish whether it is predicting 1 h ahead or 47 h ahead. Adding `hours_ahead`
(0–47) as a numeric feature lets the model learn horizon-specific biases — e.g. that
rolling features decay in reliability with distance. Requires adding the feature during
`_build_prediction_features` and including a `hours_ahead = 0` column in training rows
(or omitting from training to avoid leakage — needs careful scoping).
Expected impact: **LOW**; Low effort.

### 35. Sub-sensor binary activity flag (`{prefix}_active_24h`) ✓ done
With ~95% zero hours in dishwasher/washer data, the raw `{prefix}_lag_24h` feature is
almost always 0 and carries near-zero signal for those appliances. A binary
"was it used in the last 24 hours?" flag is more robust to sparsity and provides a
useful signal from the very first recorded event during the warm-up period.
Implementation: in `_add_sub_sensor_lags_training` and `_add_sub_sensor_lags_prediction`,
compute `(kwh_lag > 0).astype(int)` for each sub-sensor prefix and add
`{prefix}_active_24h` to the feature list.
Expected impact: **LOW–MEDIUM** (mainly during warm-up); Low effort.

### 36. Sub-sensor rolling run count (`{prefix}_runs_7d`) ✓ done
Weekly appliance usage frequency (dishwasher 1–2×/day, washer 1–2×/week) is more
predictable than a point-in-time lag. A count of non-zero hours in the trailing 7 days
captures the weekly rhythm more stably than `lag_168h` alone, especially during the
warm-up phase when the 168 h window is partially empty.
Implementation: in `_add_sub_sensor_lags_training`, compute
`(kwh_series > 0).astype(int).rolling(168, min_periods=1).sum().shift(1)` and add
`{prefix}_runs_7d` to the feature list.
Expected impact: **LOW–MEDIUM**; Low effort.

### ~~37. MQTT Discovery for entity registry~~ ✓ done (feature/mqtt-discovery)
~~Publish `homeassistant/sensor/<id>/config` payloads on `initialize()` so HA registers all
`energy_forecast_*` sensors in the entity registry (enables area assignment, labels, UI
renaming). Requires Mosquitto add-on or any MQTT broker. State updates switch from
`set_state()` to `mqtt_publish()` on the corresponding state topics. Optional: falls back
to `set_state()` if `mqtt_host` not configured. Stable `unique_id` values are already
embedded in sensor attributes as preparation.~~

---

## Tier 6 — Planned (v0.7.0–v0.10.0)

### 38. Full 48 h weather forecast features *(✓ done — v0.7.0)*
Per-hour weather merge was already in place via `_engineer_features` timestamp join
(implemented as a side-effect of #4, commit c9513b8). All six weather columns
(`temp_c`, `cloud_cover_pct`, `direct_radiation_wm2`, `sunshine_min`,
`precipitation_mm`, `wind_kmh`) are merged onto the 48-row prediction frame by
timestamp — no scalar broadcast.

Regression tests added in `TestWeatherPerHourVariation` (`tests/test_model.py`) to
guard against future regressions: each column must have `nunique() > 1` across 48 h,
and `temp_c` at h=0 and h=47 must match the input forecast values exactly.

### 39. Anomaly detection on forecast residuals *(✓ done — v0.7.0)*
Publish `binary_sensor.energy_forecast_unusual_consumption` that fires when the latest
actual reading deviates by more than N standard deviations from the model's prediction
made 1 h earlier. Uses the existing `_compute_live_mae` residual series; threshold N
is configurable via `apps.yaml` (`anomaly_sigma_threshold`, default 3.0).

Pairs naturally with the rolling MAE sensor (#41) for diagnostic visibility.
Expected impact: Diagnostic / UX; Low effort.

### 40. Home battery SoC as feature *(deferred — revisit if residuals show SoC correlation after panels installed)*
When a home battery is present, its state of charge (SoC) shapes consumption: a low SoC
during a sunny forecast triggers aggressive solar charging (raising consumption); a full
SoC suppresses it. Add optional `battery_soc_sensor` config key; include current SoC %
as a feature at training and prediction time (forward-fill at constant for horizon).
Expected impact: **MEDIUM** (battery households only); Low effort once solar is live.

### 41. Rolling accuracy history sensor (7d / 30d MAE) *(✓ done — v0.7.0)*
The current `sensor.energy_forecast_model_mae` reflects the latest training CV MAE —
a static snapshot. Add a persistent rolling-window MAE computed from `_pred_history`
vs actuals: publish `sensor.energy_forecast_mae_7d` and `sensor.energy_forecast_mae_30d`
on each hourly update. Enables a trend chart of model quality over time in Lovelace.
Expected impact: Visibility / diagnostic; Low effort.

### 42. SHAP feature importance per prediction *(✓ done — v0.7.0)*
LightGBM has native SHAP support (`model.predict(X, pred_contrib=True)`). After each
48 h prediction, compute the top-N driving features and expose them as attributes on
`sensor.energy_forecast_today` (e.g. `shap_top_features: ["temp_c", "lag_24h", ...]`).
Answers "why did the forecast spike?" directly from the sensor in HA.
Expected impact: Explainability / UX; Medium effort (SHAP call + attribute serialisation).

### 54. Relative MAE sensors (7d / 30d) *(✓ done — v0.9.0, on main)*

Companion sensors to the existing `sensor.energy_forecast_mae_7d` / `_mae_30d` (#41), expressing
accuracy as a percentage of mean consumption over the same window rather than in absolute kWh/hour.

**New sensors:**
- `sensor.energy_forecast_relative_mae_7d` — `mae_7d / mean_consumption_7d * 100` (%)
- `sensor.energy_forecast_relative_mae_30d` — `mae_30d / mean_consumption_30d * 100` (%)

**Why:** 0.57 kWh/hour MAE is hard to interpret without knowing average consumption. "4.8% error"
is immediately meaningful. Relative MAE also allows fair comparison across seasons (winter
consumption is 2–3× summer) and across households.

**Implementation note:** Use `MAE / mean(actuals)` rather than true MAPE
(`mean(|error| / actual)`). MAPE is unstable when actuals approach zero (night hours with very low
consumption cause division-by-near-zero, inflating the metric). The mean-normalised form is robust.

Mean actuals are already available from `_actuals_history` (same dict used to compute rolling MAE).
No new data fetching required — purely a derived metric from existing state.

**Effort:** ~30 min (two extra sensor publishes in `_publish_mae_sensors`, two new MQTT discovery
registrations). **No model changes.**

### 53. "Why today?" SHAP narrative attribute *(✓ done — v0.9.0, on main)*
Generate a short human-readable narrative from the top SHAP features, explaining what's
driving today's forecast. Example: "Today's forecast is shaped primarily by time-of-day
patterns and a cold 24-hour thermal window (accumulated heating demand is high). Yesterday's
same-hour consumption is also a strong signal."

Expose as a `shap_narrative` attribute on `sensor.energy_forecast_today`. Compute after
`shap_summary()` in `energy_forecast.py` using simple template-based substitution (no LLM).
Map feature names to human-readable descriptions (hour → "time-of-day", heating_deg_sum_24h
→ "accumulated heating demand", lag_24h → "yesterday's same-hour consumption", etc.).

Expected impact: Explainability / UX; Low effort (template strings + feature name mapping).

### 55. Fix SHAP: pass climate context to shap_summary() *(backlog)*

`shap_summary()` (model.py:776) lacks `climate_recent`, `dhw_recent`, `people_home_series`,
and `room_areas` params. As a result `_prepare_prediction_X()` receives `climate_recent=None`
and `room_areas=None`, so `thermal_pressure_*` and `weighted_solar_gain` are always zero
when SHAP values are computed. Predictions themselves are unaffected; only the SHAP narrative
is wrong — it cannot attribute consumption to thermal load even when it dominates.

Fix: add the four params to `shap_summary()`, thread through to `_prepare_prediction_X()`,
and update the call in `energy_forecast.py:1209` to pass `self._cached_climate_recent`,
`self._cached_dhw_recent`, `self._cached_people_home`, `self._climate_room_areas`.

Expected impact: SHAP narrative correctness; Low effort (~1 h, mostly plumbing + tests).

### 46. Dashboard: personalise entity IDs + icon cleanup *(backlog)*
`dashboard/dashboard.yaml` and `dashboard/energy-today.yaml` contain user-specific entity
IDs (`sensor.skoda_enyaq_battery_percentage`, `sensor.kermi_*`, `sensor.gplugk_z_ei`, etc.)
that will not exist on other installations. Before any wider sharing or HACS inclusion:
- Replace all personal entity IDs with commented-out placeholders or a README note
  directing users to substitute their own entities.
- Add a header comment in each file: `# EDIT: replace entity IDs below with your own`.

### 47. Fix 404 DELETE spam in `_cleanup_legacy_states()` *(pre-merge gate — dev → main)*

On startup with `mqtt_discovery: true`, `_cleanup_legacy_states()` calls
`self.remove_entity()` for each of ~30 legacy entity IDs unconditionally. AppDaemon
issues an HTTP DELETE to HA for every call; on a fresh install (or when the entities
never existed), HA returns 404 and AppDaemon logs `ERROR HASS: [404] HTTP DELETE:
Not Found {}` ~30 times. The app-level `except Exception: pass` does not suppress
the HTTP-layer log.

Fix: guard each deletion with `self.entity_exists(entity_id)` before calling
`self.remove_entity()`. Prevents the HTTP round-trip entirely when the entity is absent.
Also add a test asserting `remove_entity` is **not** called when `entity_exists`
returns False.

Observed in v0.7.0 test deploy 2026-03-24. Must fix before merging `dev` → `main`.
Expected impact: Log cleanliness; Trivial effort (~5 min).

### 43. ApexCharts / Lovelace config snippet *(long-term backlog)*
A documented, copy-paste YAML config for an ApexCharts card showing forecast vs actual
consumption. Not a custom card — uses `sensor.energy_forecast_*` sensors that already
exist. Include a sample screenshot and instructions in README under a new "Dashboard"
section.
Expected impact: Visibility / UX; Low effort (docs only).

### 44. Model versioning — keep last N, rollback *(✓ done — v0.8.0)*
When a new model is trained, archive the previous `energy_model.pkl` / `meta.pkl` pair
under a timestamped filename (e.g. `energy_model_20260324T1200.pkl`). Keep the last N
versions (configurable, default 3). Add a `rollback_model()` helper that loads the
previous version and logs a WARNING. Useful when experimenting with new features causes
accuracy regression.
Expected impact: Ops safety; Low effort.

### 45. CSV health checks + gap repair *(✓ done — v0.8.0)*
On startup (and optionally on each weekly retrain), validate `energy_history.csv` for:
- Monotonically increasing timestamps (detect clock resets or duplicated rows).
- Gaps > 2 h that are not explained by DST (log WARNING; optionally back-fill from HA).
- Values outside `[0, MAX_HOURLY_KWH]` that survived the spike filter.

Prevents silent data corruption from propagating into training without detection.
Expected impact: Correctness / defensive; Low effort.

**Storage format note (assessed 2026-03-24):** At current data volumes (~2–5 MB total,
≈8 760 rows/year) CSV has no meaningful performance or storage problem. Parquet/Feather
require `pyarrow` (~50 MB compiled dep, fragile on Alpine/armv7). **SQLite is the right
long-term direction** — no new dependency (stdlib), ACID upserts eliminate the full-rewrite
dedup path, and gap queries become trivial. Migrate when this item is implemented.

Migration reference:
- Critical files: `apps/energy_forecast/ha_data.py` (8 read/write sites),
  `apps/energy_forecast/const.py` (`CACHE_PATH`), `tests/test_ha_data.py` (40+ tests)
- Pattern already proven in `energy_history_backfill.py` (SQLite→DataFrame path)
- Migration path: on startup, if `.db` absent but `.csv` present, import CSV once
- No new dependencies; atomic upsert replaces append+dedup complexity

---

## Summary

| # | Change | Expected MAE impact | Effort | Status |
|---|--------|--------------------:|--------|--------|
| 1 | Fix Open-Meteo sunshine | high (non-SRG installs) | 15 min | ✓ done |
| 2 | Forward-roll `temp_rolling_3d` | medium | 1 h | ✓ done |
| 3 | Pre/post holiday bridge features | medium | 1 h | ✓ done |
| 4 | Cloud cover / radiation feature | medium | 2 h | ✓ done |
| 5 | Per-hour rolling prediction features | **high** | 3 h | ✓ done |
| 6 | LightGBM early stopping | medium | 2 h | ✓ done |
| 7 | Log-transform target | medium | 1 h | ✓ done |
| 8 | Adaptive retraining trigger | medium | 3 h | ✓ done |
| 9 | Cantonal holidays config | low | 30 min | ✓ done |
| 10 | School holiday feature | medium | 4 h | long-term backlog |
| 11 | `lag_72h` | low | 30 min | ✓ done |
| 12 | EV session probability feature | medium | 4 h | ✓ done |
| 13 | Prediction intervals (HA sensors) | UX value | 4 h | ✓ done |
| 14 | Intra-day actuals substitution | high (late-day sensor) | 2 h | ✓ done |
| 15 | HVAC state feature | high (if available) | 3 h | long-term backlog |
| 16 | HACS support | distribution | 1 h | long-term backlog |
| 17 | Setup checker sensor | UX / install | 2 h | ✓ done |
| 18 | Custom component config flow | UX / install | 8+ h | long-term backlog |
| 19 | CSV append-only writes | performance | 2 h | ✓ done |
| 20 | Warn when EV threshold ≥ charger_kw | correctness / UX | 30 min | ✓ done |
| 21 | Occupancy feature (`people_home`) | **high** | 4 h | ✓ done (on dev) |
| 22 | EV SoC + charging state feature | high (EV households) | 4 h | deferred |
| 23 | Solar PV target correction (B1 — grid_import/export + solar + battery) | correctness (solar households) | 2 h | ✓ done (on dev) |
| 24 | Electricity spot price feature | n/a (fixed tariff) | — | out of scope |
| 25 | Vacation / away flag | medium | 2 h | ✓ done |
| 26 | Sub-energy sensors (`sub_energy_sensors`) | medium | 4 h | ✓ done |
| 27 | Short-horizon lags (`lag_1h`–`lag_12h`) | **high** | 1 h | ✓ done |
| 28 | `num_leaves` sweep (complete #6) | medium | 1 h | ✓ done |
| 29 | Feature importance logging | diagnostic | 15 min | ✓ done |
| 30 | CV fold std logging | diagnostic | 15 min | ✓ done |
| 31 | Per-hour-of-week NaN fill medians | low–medium | 2 h | ✓ done |
| 32 | Holiday `apply` → `np.searchsorted` | performance | 30 min | ✓ done |
| 33 | Day-of-year cyclical feature (`doy_sin/cos`) | low | 30 min | ✓ done |
| 34 | `hours_ahead` horizon feature | low | 1 h | ✓ done |
| 35 | Sub-sensor binary activity flag (`{prefix}_active_24h`) | low–medium | 30 min | ✓ done |
| 36 | Sub-sensor rolling run count (`{prefix}_runs_7d`) | low–medium | 30 min | ✓ done |
| 37 | MQTT Discovery for entity registry | UX / install | 4 h | ✓ done |
| 38 | Full 48 h weather forecast features | **high** (tail accuracy) | 2 h | ✓ done |
| 39 | Anomaly detection on forecast residuals | diagnostic / UX | 1 h | ✓ done v0.7.0 |
| 40 | Home battery SoC as feature | medium (battery households) | 1 h | deferred |
| 41 | Rolling accuracy history sensor (7d/30d MAE) | visibility | 1 h | ✓ done v0.7.0 |
| 42 | SHAP feature importance per prediction | explainability | 3 h | ✓ done v0.7.0 |
| 43 | ApexCharts / Lovelace config snippet | visibility / UX | 1 h | ✓ done v0.7.0 |
| 44 | Model versioning (keep last N, rollback) | ops safety | 2 h | ✓ done v0.8.0 |
| 45 | CSV health checks + gap repair | correctness / defensive | 2 h | ✓ done v0.8.0 |
| 46 | Dashboard: personalise entity IDs + icon cleanup | UX / sharing | 30 min | pre-v0.7.0-release |
| 47 | Fix 404 DELETE spam in `_cleanup_legacy_states()` | log cleanliness | 5 min | ✓ done v0.7.1 |
| 48 | Anomaly binary sensor MQTT attrs + discovery fix | correctness / UX | 30 min | ✓ done v0.7.1 |
| 49 | Exponentially weighted moving average temperature | **high** (thermal model) | 1 h | ✓ done (on dev) |
| 50 | Rolling accumulated heating degree-hours | high (thermal model) | 1.5 h | ✓ done (on dev) |
| 51 | Temperature rate of change feature | medium (thermal model) | 30 min | ✓ done (on dev) |
| 52 | Temperature lag features (24h, 168h) | medium (thermal model) | 30 min | ✓ done (on dev) |
| 53 | "Why today?" SHAP narrative attribute | explainability / UX | 2 h | ✓ done (on dev) |
| 54 | Relative MAE sensors (7d / 30d) | visibility / UX | 30 min | ✓ done (on dev) |
| 55 | Fix SHAP: pass climate_recent / room_areas to shap_summary() | explainability correctness | 1 h | ✓ done (on feat/physics-features) |
| 56 | Solar-Compensated Thermal Pressure | **high** (sunny winter days) | 2 h | ✓ done (on feat/physics-features) |
| 57 | Wind-Driven Infiltration Feature | medium | 1 h | ✓ done (on feat/physics-features) |
| 58 | Humidity-Aware Defrost Proxy (Heat Pump) | medium | 1 h | ✓ done (on feat/physics-features) |
