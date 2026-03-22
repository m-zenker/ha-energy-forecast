# Forecast Accuracy Roadmap

Proposed improvements to `ha-energy-forecast`, ordered by impact tier.
Current baseline: v0.5.1 on `feature/stage6-distribution-ux` (next milestone: #16 HACS support).

---

## Staged Development & Deployment Plan

### Release milestones

| Milestone | Version | Contents | Status |
|-----------|---------|----------|--------|
| Hotfix merge | v0.5.3 | Merge `dev` → `main`: log noise reduction, XX:01 hourly alignment | done |
| Entity registry | v0.6.0 | #37 MQTT Discovery (entity registry, area assignment, labels) | ✓ done |
| HACS distribution | v0.7.0 | #16 HACS support (hacs.json, info.md, README section, GitHub topics) | planned |
| Long-term | v1.x+ | #10 School holidays, #15 HVAC, #21 Occupancy, #22 EV SoC, #23 Solar, #24 Spot price, #25 Vacation, #18 Config flow | backlog |

### Deployment workflow (per release)

1. Feature branch → implement + tests pass (`python -m pytest tests/ -v`)
2. PR → code review → merge to `dev`
3. Smoke-test on local HA instance (watch AppDaemon log; confirm sensors update)
4. PR `dev` → `main` after stable period on local instance
5. Update CHANGELOG.md (close `[Unreleased]` → `vX.Y.Z`)
6. Create semver tag (`git tag vX.Y.Z`) → push tag → GitHub release with notes
7. After #16: HACS auto-picks up new semver tag for AppDaemon category listing

---

## Tier 1 — High impact, low effort (quick wins)

### ~~1. Fix missing sunshine in Open-Meteo fallback~~ ✓ done (144de78)
~~`weather.py` hardcodes `sunshine_min = 0` for the Open-Meteo forecast fallback.
The free Open-Meteo API *does* provide `sunshine_duration`. This is a bug that
silently degrades any installation without SRG-SSR credentials — summer cooling
and daylighting-driven consumption patterns become invisible to the model for all
48 prediction hours. Fix: add `sunshine_duration` to the Open-Meteo forecast URL
and convert from seconds to minutes, matching the archive fetcher.~~

### ~~2. Add `temp_rolling_3d` to the prediction horizon~~ ✓ done (c9513b8)
`temp_rolling_3d` is computed only over historical weather. During prediction it
is populated at the current timestamp but never rolled forward using forecast
temperatures. Add a forward-projection of this feature using forecast `temp_c`
so the model sees the *incoming* thermal trend, not just the past one. Particularly
impactful at the start of cold spells and thaws.

### ~~3. Pre/post-holiday bridge day features~~ ✓ done (d47477f)
The model uses only a binary `is_public_holiday` flag. Adding
`days_to_next_holiday` and `days_since_last_holiday` (capped at ±3) captures the
Brückentag (bridging day) pattern common in Swiss households — consumption
typically drops the day before a multi-day break and recovers the day after.

### ~~4. Cloud cover / solar irradiance feature~~ ✓ done (c9513b8)
Open-Meteo provides `cloudcover` and `direct_radiation` at no extra cost.
Direct solar radiation is a stronger predictor of cooling load than sunshine
minutes, and cloud cover captures gloom-driven lighting load that the current
features miss. Add both to `fetch_historical_weather` and `fetch_open_meteo`,
and include them in `_FEATURES_BASE`.

---

## Tier 2 — High impact, medium effort

### ~~5. Fix training/prediction mismatch in rolling features~~ ✓ done (144de78)
~~`rolling_mean_24h`, `rolling_mean_7d`, and `rolling_std_24h` are computed as a
*single scalar* applied to all 48 prediction hours. During training these are
time-varying per row. This creates a systematic bias: hours 1–2 ahead get a
reasonably calibrated feature value, but hours 24–48 ahead get a stale one.

Fix: project these features forward hour-by-hour using the predicted values for
intermediate hours (auto-regressive unrolling), or at minimum decay them toward
the training-set median as `hours_ahead` increases.~~

### ~~6. LightGBM early stopping + validation-set tuning~~ ✓ done
~~Fixed hyperparameters (`n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`)
cause the model to under-fit or over-fit depending on data volume. Adding early
stopping against the last CV fold (already computed during training) gives the
right tree count for free without grid search. Also consider a narrow sweep of
`num_leaves` (16 / 31 / 63) on the final CV split.~~

### ~~7. Log-transform the target~~ ✓ done
~~Household energy consumption is right-skewed (heating peaks, near-threshold EV
hours). Training on `log1p(gross_kwh)` and exponentiating predictions reduces the
outsized influence of rare high-energy hours on the loss, typically improving MAE
on typical hours at the cost of negligible degradation on peaks.~~

### ~~8. Adaptive retraining trigger~~ ✓ done
~~Weekly retraining ignores model drift. A rolling MAPE computed on the last 24
hourly actuals vs. predictions made 24 h earlier (trackable cheaply in the CSV
cache) would trigger an early retrain when live error exceeds ~2× the CV MAE.
Most impactful after seasonal transitions or household behaviour changes.~~

---

## Tier 3 — Medium impact, higher effort

### ~~9. Cantonal public holidays~~ ✓ done
~~The `holidays` package supports `canton=`. Exposing this as a `holiday_canton`
key in `apps.yaml` is a one-line change in `_add_holiday_feature` but captures
cantonal school and bank holidays that differ significantly from federal ones.~~

### 10. School holiday feature *(long-term backlog)*
Swiss Schulferien dates are canton-specific but stable year-to-year. During
school holidays household daytime consumption rises (children at home). None of
the current features capture this. Implement a static lookup table per canton,
configurable via `apps.yaml`, and add `is_school_holiday` to `_FEATURES_BASE`.

### ~~11. Additional lag: `lag_72h`~~ ✓ done (d47477f)
A 72 h lag captures the "same-time-3-days-ago" pattern, useful for transitions
at the start/end of weekends and public-holiday bridges. It slots between the
existing `lag_48h` and `lag_168h` with no impact on data-volume requirements
(comparable to `lag_48h`).

### ~~12. EV charge session probability feature~~ ✓ done
~~The current EV subtraction assumes a fixed 9 kW charger load for every detected
hour. Improvements:~~
~~- Detect EV session start and compute session-average power from actuals.~~
~~- Add a `likely_ev_hour` binary feature derived from recurring time-of-use
  patterns so the model can adjust the household-baseline prediction for known
  charging windows.~~

---

## Tier 4 — Longer-term / architectural

### ~~13. Prediction intervals as HA sensors~~ ✓ done
~~Publish `sensor.energy_forecast_today_low` / `_high` using LightGBM quantile
regression (`objective='quantile'`, `alpha=0.1/0.9`). Enables automations like
"charge EV only if forecast upper bound is below 15 kWh" — higher end-user value
than marginal point-estimate MAE improvement.~~

### ~~14. Intra-day actuals substitution~~ ✓ done
~~As today's hours tick by, `_aggregate()` should replace predicted values for
elapsed hours with actuals from the CSV cache. Currently the `today` sensor sums
raw predictions across the full day including already-elapsed hours. Substituting
actuals makes the `today` total significantly more accurate as the day progresses.~~

### 15. HVAC / boiler state feature *(long-term backlog)*
If a thermostat or boiler entity is available in HA (configurable via `apps.yaml`
as `hvac_sensor`), including its current setpoint or on/off state as a feature
directly explains heating/cooling load variance that outdoor temperature alone
cannot capture.

---

## Distribution

### 16. HACS support *(planned)*

Make the app installable via [HACS](https://hacs.xyz/) (AppDaemon category).

Required changes:
- Add `hacs.json` at repo root (HACS manifest; `render_readme: true`).
- Add `info.md` — shown in the HACS detail panel before installation; must prominently warn that HACS only copies app files and that the AppDaemon add-on dependency config and `apps.yaml` creation are still manual steps.
- Add a "Install via HACS" subsection to `README.md` under `## Installation`, explaining what HACS does and doesn't do, with links to the dependency config and `apps.yaml.example` steps.
- Set GitHub repo topics: `appdaemon` (required for HACS category), `home-assistant`, `hacs`.

No code changes required — `apps/energy_forecast/` is already in the correct location for HACS AppDaemon installs. Semver tags are already present.

### 17. Setup checker sensor ✓ done (feature branches)
Bake a startup self-check into the main app that surfaces setup problems as a visible HA entity rather than silent log failures.

- On initialisation, attempt `import pandas, numpy, lightgbm, sklearn, requests, holidays` and log a clear error for each missing package.
- Validate required config keys (`energy_sensor`, `latitude`, `longitude`); validate optional keys have sane types.
- Publish `sensor.energy_forecast_setup_status` with states `ok`, `missing_packages`, `missing_config`, or `invalid_config`, and an `issues` attribute listing specific problems.
- Self-silences (removes sensor) once all checks pass and the main app is running normally.

This converts silent failure after a fresh HACS install into actionable, user-visible feedback without requiring any Supervisor access.

### 19. CSV cache: append-only writes ✓ done (feature branches)
For long-running installs with months of history, `fetch_recent_energy` rewrites the entire `energy_history.csv` on every hourly update. At ~8 760 rows/year this is already measurable I/O and will compound over time.

Improvement: write only new rows using `df.to_csv(..., mode='a', header=False)` in `fetch_recent_energy`, and perform a periodic compaction (dedup + sort) in `fetch_energy_history` (the weekly full-read path) rather than on every update. Requires care around the merge-winner rule to avoid duplicating rows that already exist in the CSV.

### 20. Config validation: warn when `ev_charging_threshold_kwh >= ev_charger_kw` ✓ done (feature branches)
When the detection threshold is set at or above the charger power (e.g. threshold=10, charger=9), every detected EV hour produces `max(0, gross - charger_kw) = 0`, so the EV sensors always read zero while the model still strips those hours from training data. The combination is silently confusing.

Add a validation check in `_validate_config` that logs a `WARNING` when `self._ev_threshold >= self._ev_charger_kw`, explaining that the EV sensor will report 0 kWh for all detected sessions in that configuration.

### 21. Occupancy feature (`people_home`) *(long-term backlog)*
Home vs. away is the single largest unmodelled driver of energy consumption — a weekday with everyone out can draw 30–50% less than a day at home. Add an optional `presence_sensors` list in `apps.yaml` (e.g. `person.alice`, `person.bob`); derive a `people_home` integer feature at each hour by counting how many are in the `home` state. Requires joining HA history for each person entity alongside the energy history fetch.

### 22. EV charging state + SoC feature *(long-term backlog)*
The current `likely_ev_hour` feature is pattern-derived from past sessions. Two optional config keys — `ev_battery_sensor` (SoC %) and `ev_charging_sensor` (binary) — would let the model know *today* whether the car is plugged in and how much charge it needs. At predict time: if the car is home and SoC is low, boost the probability of an EV session tonight; if SoC is full, suppress it. Requires forward-filling sensor state into the prediction horizon.

### 23. Solar PV integration *(long-term backlog)*
Households with panels train the model on net grid import, which conflates household consumption with solar self-consumption — on a sunny day the model sees low import and learns the wrong signal. Two sub-items:

1. **Production offset during training**: subtract a `solar_production_sensor` reading from `gross_kwh` to recover true household consumption before feature engineering.
2. **Solar forecast as feature**: derive an expected production curve for the prediction horizon from PV system parameters (`pv_kwp`, `pv_azimuth`, `pv_tilt`) combined with the already-fetched `direct_radiation_wm2` forecast.

### 24. Electricity spot price feature *(long-term backlog)*
Households on dynamic tariffs (Tibber, Nordpool) actively shift deferrable loads — dishwasher, washing machine, EV charging — to cheap hours. The model currently cannot learn this behaviour because it sees no price signal. Add an optional `price_sensor` config key; include the hourly price (or a `is_cheap_hour` binary derived from a configurable threshold) as a feature. The Tibber and Nordpool HA integrations already expose standardised hourly price sensors.

### 25. Vacation / away flag *(long-term backlog)*
Multi-day absences cause baseline drops that look like anomalies to the rolling lag features and bias the model until history catches up. An optional `away_mode_entity` config key (e.g. `input_boolean.vacation_mode` or a `calendar` entity) adds a binary `is_away` feature. When set, the model can learn the reduced baseline explicitly rather than treating it as noise.

### 18. Custom component config flow *(long-term backlog)*
A full HA custom component (lives in `custom_components/energy_forecast/`) that provides a UI-driven setup wizard via HA's config flow:

- Step 1: entity picker for `energy_sensor`; lat/lon auto-populated from HA's own location config.
- Step 2: optional fields (SRG credentials, outdoor temp sensor, canton, EV threshold).
- On completion: writes the `energy_forecast:` stanza into `appdaemon/apps/apps.yaml` via the HA filesystem API.
- Calls the Supervisor REST API (`/supervisor/addons/<appdaemon_slug>/options`) to patch `python_packages` and `init_commands` with the required dependencies, then triggers an add-on restart.

This is the only path to fully zero-manual-step installation. Significant effort: requires maintaining a separate HA integration type alongside the AppDaemon app, Supervisor API integration, and config flow UI.

---

## Tier 5 — Diagnostics, Performance & Minor Features

### 27. Short-horizon lags (`lag_1h`, `lag_2h`, `lag_6h`, `lag_12h`) ✓ done (feature branches)
The current lag set jumps from `lag_24h` to `lag_48h`, leaving a blind spot in the
1–12 h range that matters most for same-day intra-day prediction. Adding `lag_1h`,
`lag_2h`, `lag_6h`, and `lag_12h` to `_add_lag_features` (training) and
`_add_lags_prediction` (inference) is a direct feature-engineering win at negligible
data-volume cost — all four are available as soon as a single day of history exists.
Expected impact: **HIGH** for hours 1–6 ahead; Low effort.

### 28. `num_leaves` hyperparameter sweep — complete ROADMAP #6 ✓ done (feature branches)
ROADMAP item #6 added early stopping but left the `num_leaves` sweep (`16 / 31 / 63`)
as a follow-up. A narrow grid search on the final CV split can be wired into the
existing `_cross_validate` path without changing the training API. Prevents the model
from being locked into the LightGBM default of 31 leaves regardless of data volume.
Expected impact: **MEDIUM**; Low effort (sweep is already sketched in the #6 description).

### 29. Feature importance logging after training ✓ done (feature branches)
After `model.fit()` in `_train_model`, log `model.feature_importances_` sorted by
weight. Currently there is no visibility into which features the model actually uses.
One `logger.debug` call with the sorted list adds zero runtime cost and makes
under-contributing features immediately visible in the AppDaemon log.
Expected impact: Diagnostic; Trivial effort.

### 30. CV fold std logging alongside mean ✓ done (feature branches)
`_cross_validate` currently logs only the mean MAE across folds. Adding the standard
deviation (and optionally the per-fold breakdown at DEBUG level) surfaces high-variance
training runs — an early signal of insufficient data or a degraded feature — without
changing any model logic.
Expected impact: Diagnostic; Trivial effort.

### 31. Per-hour-of-week NaN fill medians ✓ done (feature branches)
NaN values in rolling/lag features are currently filled with the global training
median. A per-`hour_of_week` median (168 cells) is a much tighter imputation for the
warm-up period at install time, where the model would otherwise use a "typical any
hour" stand-in for a specifically 3 a.m. Tuesday slot. Requires computing and caching
a `(168,)` lookup table during training and applying it in `_build_prediction_features`.
Expected impact: **LOW–MEDIUM** (mainly during first week of data); Medium effort.

### 32. Holiday `apply` → `np.searchsorted` vectorization ✓ done (feature branches)
`_add_holiday_feature` calls `pd.Series.apply(lambda ts: ts.date() in holiday_set)`,
which is a Python loop over every row in the training frame. Replacing it with
`np.searchsorted` on a sorted date array (or a boolean index join) reduces the
holiday computation from O(n) Python-level iterations to a vectorised C operation.
Expected impact: Performance (training speed); Low effort.

### 33. Day-of-year cyclical feature (`doy_sin` / `doy_cos`) ✓ done (feature branches)
The model captures seasonality through rolling temperature features and calendar
proxies, but has no smooth cyclic encoding of position within the year. Adding
`doy_sin = sin(2π·doy/365)` and `doy_cos = cos(2π·doy/365)` to `_FEATURES_BASE`
gives the model a continuous signal for seasonal baseline that avoids the
discontinuity at New Year's Day.
Expected impact: **LOW**; Low effort.

### 34. `hours_ahead` feature for horizon-aware prediction ✓ done (feature branches)
All 48 prediction rows currently receive identical feature vectors; the model cannot
distinguish whether it is predicting 1 h ahead or 47 h ahead. Adding `hours_ahead`
(0–47) as a numeric feature lets the model learn horizon-specific biases — e.g. that
rolling features decay in reliability with distance. Requires adding the feature during
`_build_prediction_features` and including a `hours_ahead = 0` column in training rows
(or omitting from training to avoid leakage — needs careful scoping).
Expected impact: **LOW**; Low effort.

### 35. Sub-sensor binary activity flag (`{prefix}_active_24h`) ✓ done (feature branches)
With ~95% zero hours in dishwasher/washer data, the raw `{prefix}_lag_24h` feature is
almost always 0 and carries near-zero signal for those appliances. A binary
"was it used in the last 24 hours?" flag is more robust to sparsity and provides a
useful signal from the very first recorded event during the warm-up period.
Implementation: in `_add_sub_sensor_lags_training` and `_add_sub_sensor_lags_prediction`,
compute `(kwh_lag > 0).astype(int)` for each sub-sensor prefix and add
`{prefix}_active_24h` to the feature list.
Expected impact: **LOW–MEDIUM** (mainly during warm-up); Low effort.

### 36. Sub-sensor rolling run count (`{prefix}_runs_7d`) ✓ done (feature branches)
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
| 16 | HACS support | distribution | 1 h | planned |
| 17 | Setup checker sensor | UX / install | 2 h | ✓ done |
| 18 | Custom component config flow | UX / install | 8+ h | long-term backlog |
| 19 | CSV append-only writes | performance | 2 h | ✓ done |
| 20 | Warn when EV threshold ≥ charger_kw | correctness / UX | 30 min | ✓ done |
| 21 | Occupancy feature (`people_home`) | **high** | 4 h | long-term backlog |
| 22 | EV SoC + charging state feature | high (EV households) | 4 h | long-term backlog |
| 23 | Solar PV integration | high (solar households) | 6 h | long-term backlog |
| 24 | Electricity spot price feature | medium (dynamic tariff) | 2 h | long-term backlog |
| 25 | Vacation / away flag | medium | 2 h | long-term backlog |
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
