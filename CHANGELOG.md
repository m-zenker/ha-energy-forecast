# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Fixed
- fix(scheduler): align hourly sensor updates to XX:01:00 wall-clock time using `run_hourly`; eliminates startup-time drift
- fix(model): downgrade prediction-time sub-sensor NaN log from WARNING to DEBUG; training-time WARNING (weekly) is sufficient

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

[Unreleased]: https://forgejo.walzen.me/martin/ha-energy-forecast/compare/v0.5.2...HEAD
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
