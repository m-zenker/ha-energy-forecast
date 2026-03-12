# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
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
