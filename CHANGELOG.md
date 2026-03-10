# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

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
