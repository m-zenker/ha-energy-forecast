# Forecast Accuracy Roadmap

Proposed improvements to `ha-energy-forecast`, ordered by impact tier.
Current baseline: v0.2.1 on `main`.

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

### 6. LightGBM early stopping + validation-set tuning
Fixed hyperparameters (`n_estimators=500`, `learning_rate=0.05`, `num_leaves=31`)
cause the model to under-fit or over-fit depending on data volume. Adding early
stopping against the last CV fold (already computed during training) gives the
right tree count for free without grid search. Also consider a narrow sweep of
`num_leaves` (16 / 31 / 63) on the final CV split.

### 7. Log-transform the target
Household energy consumption is right-skewed (heating peaks, near-threshold EV
hours). Training on `log1p(gross_kwh)` and exponentiating predictions reduces the
outsized influence of rare high-energy hours on the loss, typically improving MAE
on typical hours at the cost of negligible degradation on peaks.

### 8. Adaptive retraining trigger
Weekly retraining ignores model drift. A rolling MAPE computed on the last 24
hourly actuals vs. predictions made 24 h earlier (trackable cheaply in the CSV
cache) would trigger an early retrain when live error exceeds ~2× the CV MAE.
Most impactful after seasonal transitions or household behaviour changes.

---

## Tier 3 — Medium impact, higher effort

### 9. Cantonal public holidays
The `holidays` package supports `canton=`. Exposing this as a `holiday_canton`
key in `apps.yaml` is a one-line change in `_add_holiday_feature` but captures
cantonal school and bank holidays that differ significantly from federal ones.

### 10. School holiday feature
Swiss Schulferien dates are canton-specific but stable year-to-year. During
school holidays household daytime consumption rises (children at home). None of
the current features capture this. Implement a static lookup table per canton,
configurable via `apps.yaml`, and add `is_school_holiday` to `_FEATURES_BASE`.

### ~~11. Additional lag: `lag_72h`~~ ✓ done (d47477f)
A 72 h lag captures the "same-time-3-days-ago" pattern, useful for transitions
at the start/end of weekends and public-holiday bridges. It slots between the
existing `lag_48h` and `lag_168h` with no impact on data-volume requirements
(comparable to `lag_48h`).

### 12. EV charge session probability feature
The current EV subtraction assumes a fixed 9 kW charger load for every detected
hour. Improvements:
- Detect EV session start and compute session-average power from actuals.
- Add a `likely_ev_hour` binary feature derived from recurring time-of-use
  patterns so the model can adjust the household-baseline prediction for known
  charging windows.

---

## Tier 4 — Longer-term / architectural

### 13. Prediction intervals as HA sensors
Publish `sensor.energy_forecast_today_low` / `_high` using LightGBM quantile
regression (`objective='quantile'`, `alpha=0.1/0.9`). Enables automations like
"charge EV only if forecast upper bound is below 15 kWh" — higher end-user value
than marginal point-estimate MAE improvement.

### 14. Intra-day actuals substitution
As today's hours tick by, `_aggregate()` should replace predicted values for
elapsed hours with actuals from the CSV cache. Currently the `today` sensor sums
raw predictions across the full day including already-elapsed hours. Substituting
actuals makes the `today` total significantly more accurate as the day progresses.

### 15. HVAC / boiler state feature
If a thermostat or boiler entity is available in HA (configurable via `apps.yaml`
as `hvac_sensor`), including its current setpoint or on/off state as a feature
directly explains heating/cooling load variance that outdoor temperature alone
cannot capture.

---

## Summary

| # | Change | Expected MAE impact | Effort | Status |
|---|--------|--------------------:|--------|--------|
| 1 | Fix Open-Meteo sunshine | high (non-SRG installs) | 15 min | ✓ done |
| 2 | Forward-roll `temp_rolling_3d` | medium | 1 h | ✓ done |
| 3 | Pre/post holiday bridge features | medium | 1 h | ✓ done |
| 4 | Cloud cover / radiation feature | medium | 2 h | ✓ done |
| 5 | Per-hour rolling prediction features | **high** | 3 h | ✓ done |
| 6 | LightGBM early stopping | medium | 2 h | |
| 7 | Log-transform target | medium | 1 h | |
| 8 | Adaptive retraining trigger | medium | 3 h | |
| 9 | Cantonal holidays config | low | 30 min | |
| 10 | School holiday feature | medium | 4 h | |
| 11 | `lag_72h` | low | 30 min | ✓ done |
| 12 | EV session probability feature | medium | 4 h | |
| 13 | Prediction intervals (HA sensors) | UX value | 4 h | |
| 14 | Intra-day actuals substitution | high (late-day sensor) | 2 h | |
| 15 | HVAC state feature | high (if available) | 3 h | |
