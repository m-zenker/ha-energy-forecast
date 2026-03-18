# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

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

[Unreleased]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.3.0...HEAD
[0.3.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.2.1...v0.3.0
[0.2.1]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.2.0...v0.2.1
[0.2.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.1.0...v0.2.0
[0.1.0]: https://forgejo.walzen.me/martin/ha-energy-forecast/releases/tag/v0.1.0
