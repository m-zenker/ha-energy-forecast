# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Fixed
- **SRG-SSR forecast migrated to v2 API** (`weather.py`): Updated endpoint from
  deprecated v1 (`/forecasts/v1.0/weather/7day`) to v2. Flow: resolve nearest station
  via `GET /srf-meteo/v2/geolocations?latitude=...&longitude=...`, then fetch
  `GET /srf-meteo/v2/forecastpoint/{id}`. Response parsing updated for v2's flat `hours`
  array (was nested `forecast[].hours[]`) and renamed precipitation field `PRP_MM` → `RRR_MM`.
  lat/lon lookup used (not PLZ) to ensure the registered Freemium location is matched.
- **`TestEvKwhSensorCalc` hardcoded date** (`tests/test_energy_forecast.py`): Two tests
  used `2026-03-12`; replaced with `pd.Timestamp.now().normalize()` so they don't rot
  as real time passes.

### Fixed
- **`_retrain_cb` / `_update_cb` event-callback signature** (#review-1): Both callbacks
  now accept `(event_name=None, data=None, kwargs=None)` so they work correctly when
  triggered via `listen_event` (which passes three positional args) as well as from
  timer callbacks (which pass one). Previously a `RELOAD_ENERGY_MODEL` event raised
  `TypeError` before the fault-boundary `except Exception` was reached.
- **EV charger power used in `_update_sensors` lag features** (#review-2): The hourly
  sensor update path now passes `charger_kw=self._ev_charger_kw` to
  `split_ev_charging`, consistent with the weekly retrain path. Previously the default
  9.0 kW was hardcoded regardless of `ev_charger_kw` configuration, causing
  lag-feature drift for custom charger powers.
- **EV kWh sensor reports charger energy, not threshold energy** (#review-10): The
  `ev_today` / `ev_yesterday` sensors now subtract `ev_charger_kw` from gross import
  (clipped at 0) rather than the detection `ev_charging_threshold_kwh`. This gives the
  correct net charger energy estimate.
- **`adaptive_retrain_threshold` validated at startup** (#review-7): A negative value
  would have triggered retraining on every hourly update; `_validate_config` now
  raises `ValueError` for values < 0.
- **SRG-SSR fallback now logged** (#review-6): When the SRG API call fails and the
  forecast falls back to Open-Meteo, a `WARNING` is emitted so operators know forecast
  quality may be reduced.
- **Lock contention logged at DEBUG** (#review-8): Skipped retrain and sensor-update
  cycles (normal during long retrains) are now visible at DEBUG level instead of being
  silently swallowed.

### Changed
- **`_empty_weather_df` column set extended** (#review-5): Now includes
  `cloud_cover_pct` and `direct_radiation_wm2`, matching the full column contract of
  the real fetchers. The `_engineer_features` safety-net NaN fill remains in place.
- **`fetch_recent_energy` signature cleaned up** (#review-3): Removed the unused
  `hours: int = 6` parameter which was never read by the function body.
- **Dead URL constants removed from `const.py`** (#review-4): `METEOSWISS_URL`,
  `OPENMETEO_FORECAST_URL`, and `OPENMETEO_ARCHIVE_URL` were defined but never
  imported; actual URLs are hardcoded inline in `weather.py`.
- **`model.py` module docstring rewritten** (#review-11): Replaced the "Changes from
  original" framing (git-history content) with a description of the current feature
  set, model architecture, and persistence scheme.
- **`is_public_holiday` flag vectorised** (#review-9): Replaced per-row
  `dates.map(lambda d: ...)` with `dates.isin(set(...)).astype(int)` in
  `_add_holiday_feature`.
- **`tz_convert(None)` replaces `tz_localize(None)`** (#review-12): Both work in
  pandas but `tz_convert` better expresses "convert tz-aware timestamp to wall time,
  drop tz annotation".

### Added
- 12 new tests across `test_energy_forecast.py`:
  `TestAggregate` (6) — next_3h sum, tomorrow sum, EV sensors zero without actuals,
  interval keys present/absent, 8 block slots; `TestEvKwhSensorCalc` (2) — charger_kw
  vs threshold subtraction; `TestCallbackSignature` (4) — timer and event calling
  conventions for both callbacks.



### Added
- **EV session probability feature** (#12): `likely_ev_hour` binary feature added to
  `_FEATURES_BASE`. After training, `_compute_likely_ev_hours()` inspects which
  `hour_of_week` slots (0-167) appeared as EV charging hours in ≥ 15% of their
  historical occurrences and stores the result as `self._likely_ev_hours` (persisted
  in `meta.pkl`). The feature lets the model explicitly account for recurring EV
  charging windows when predicting household baseline consumption. `ev_df` is now
  passed from `_retrain()` to `train()` to enable this computation.
- **Fix `split_ev_charging` charger power** (#12): `ev_charger_kw` configured in
  `apps.yaml` is now correctly forwarded to `ha_data.split_ev_charging()` via a
  new `charger_kw` parameter. Previously the function hardcoded 9.0 regardless of
  configuration.
- `tests/test_ha_data.py` — 2 new tests: `TestSplitEvCharging` (custom charger_kw
  subtracted, default 9.0 preserved)
- `tests/test_model.py` — 3 new tests: `TestLikelyEvHour` (hours identified after
  train, binary column, empty set without ev_df)

- **Prediction intervals** (#13): two quantile regression models (α=0.1, α=0.9)
  are trained alongside the point-estimate model using the same feature matrix,
  log-transformed target, and n_estimators from early stopping. Quantile training
  is wrapped in a broad `try/except` so a failure never interrupts normal operation.
  Six new HA sensors are published once quantile models are available:
  `sensor.energy_forecast_{next_3h,today,tomorrow}_{low,high}`. Today's bounds
  blend actuals for elapsed hours with quantile forecasts for remaining hours —
  the interval collapses to zero width for hours that have already passed.
  Models are persisted as `energy_model_q10.pkl` / `energy_model_q90.pkl` with
  SHA-256 sidecars (same integrity pattern as the main model). A new
  `_prepare_prediction_X()` helper eliminates the shared feature-engineering
  duplication between `predict()` and `predict_intervals()`.
- **Intra-day actuals substitution** (#14): `sensor.energy_forecast_today` and
  the `today_HH_HH` block sensors now blend measured actuals (elapsed hours) with
  model predictions (remaining hours). Previously the today total was based on
  predictions alone, which missed all consumption before the current hour. A new
  module-level helper `_blend_today_totals` handles the blending; `"tomorrow"` and
  `"next_3h"` sensors are unchanged.
- **Adaptive retraining trigger** (#8): each hourly sensor update stores the
  48-hour prediction in an in-memory prediction history (keep-first semantics,
  preserving the ≈24h-ahead prediction for each future hour). After every update
  the live day-ahead MAE is compared against `adaptive_retrain_threshold × cv_MAE`;
  if exceeded with ≥ 24 matched pairs and a 24h cooldown has elapsed, an early
  retrain is triggered and logged at WARNING level. Configurable via
  `adaptive_retrain_threshold` in `apps.yaml` (default 2.0; 0 disables).
- `conftest.py`: `hassapi` stub so `energy_forecast.py` module-level helpers are
  importable in the test environment without AppDaemon installed.
- `tests/test_energy_forecast.py` — 7 new tests: `TestBlendTodayTotals` (4),
  `TestComputeLiveMae` (3)
- **Log-transform target** (#7): `gross_kwh` is now `log1p`-transformed before
  training and `expm1`-inverted at prediction time. Reduces the outsized influence
  of rare high-energy hours on the loss, improving MAE on typical hours. MAE
  continues to be reported in kWh (not log space). A `log_transform` flag is
  persisted in `meta.pkl`; old installs without the key default to False (no
  behaviour change until the next retrain).
- **LightGBM early stopping** (#6): CV fold fits now use
  `lgb.early_stopping(stopping_rounds=50)` with the fold's validation set as
  `eval_set`. The `best_iteration_` from the last fold is used as `n_estimators`
  for the final model, eliminating fixed-count under/over-fitting. Falls back to
  the default 500 estimators if early stopping raises an exception or LightGBM is
  unavailable. `_build_model` accepts an `n_estimators` override.
- **Cantonal holidays** (#9): `_add_holiday_feature` accepts a `canton` keyword
  (e.g. `"ZH"`, `"BE"`) passed as `subdiv` to `holidays.country_holidays`. Invalid
  codes fall back to federal-only with a warning. Configure via `holiday_canton`
  in `apps.yaml`; defaults to federal holidays only if omitted.
- `tests/test_model.py` — 8 new tests: `TestLogTransform` (3), `TestBuildModel`
  (2), `TestCantonalHolidays` (3)

### Fixed
- **Open-Meteo fallback sunshine**: `fetch_open_meteo` previously hardcoded
  `sunshine_min = 0`, silently degrading forecasts for all installations without
  SRG-SSR credentials. `sunshine_duration` is now requested from the API and
  converted from seconds to minutes, matching the archive fetcher. A safe `.get()`
  fallback handles any API response that omits the field.
- **Rolling feature train/predict mismatch**: `rolling_mean_24h`, `rolling_mean_7d`,
  and `rolling_std_24h` were broadcast as a single scalar across all 48 prediction
  hours, diverging from their time-varying per-row semantics during training. These
  features now slide forward over an extended series (known actuals + fill values at
  future timestamps) with `shift(1)` applied to exactly mirror the training
  computation. At h=0 the value matches `mean(actuals[-24:])` exactly; beyond h=24
  it stabilises at the recent household baseline.

### Added
- **Cloud cover and direct radiation features** (`cloud_cover_pct`, `direct_radiation_wm2`)
  added to archive fetcher, Open-Meteo forecast fetcher, and `_FEATURES_BASE`. SRG users
  receive these from an automatic Open-Meteo supplement call (`_supplement_from_open_meteo`),
  so the full feature set is available regardless of weather source.
- **`fetch_open_meteo` now includes `past_days=3`**, returning 72 h of measured history
  alongside the 7-day forecast. This anchors `temp_rolling_3d` in real observations for
  both Open-Meteo and SRG users, eliminating the single-value initialisation that
  previously caused the 3-day rolling mean to be based on forecast-only data.
- **`_supplement_from_open_meteo` helper** in `weather.py`: merges the Open-Meteo
  historical tail and cloud/radiation columns into SRG forecast results, preserving
  SRG values for all fields it provides.
- **`lag_72h`** autoregressive feature — captures the same-time-3-days-ago pattern,
  useful for weekend/holiday bridge transitions. Activates dynamically once ≥172 h
  of history is available, consistent with the existing dynamic lag selection logic.
- **Bridge-day holiday features** — `days_to_next_holiday` and `days_since_last_holiday`
  (integers, capped at 3). Both are 0 on a holiday itself; fallback to 3 if the
  `holidays` package is unavailable. Holiday years extended by ±1 to handle dates
  near year boundaries (e.g. Dec 31 seeing Jan 1).
- `tests/test_weather.py` — 6 tests covering `fetch_open_meteo` sunshine parsing,
  unit conversion, missing-key fallback, column contract, and network errors
- `tests/test_model.py` — 23 tests covering rolling feature variance, exact h=0
  boundary semantics, smooth transition at h=24, short/empty actuals edge cases,
  lag_72h correctness, and bridge-day feature range/values/fallback
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

[Unreleased]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.2.1...HEAD
[0.2.1]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.2.0...v0.2.1
[0.2.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.1.0...v0.2.0
[0.1.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/releases/tag/v0.1.0
