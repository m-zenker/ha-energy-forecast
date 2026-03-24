# HA Energy Forecast

*Know your electricity bill before the day begins.*

![Version](https://img.shields.io/badge/version-v0.7.0-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-236%20passing-brightgreen) ![AppDaemon](https://img.shields.io/badge/AppDaemon-4.x-orange)

Plan EV charging, avoid bill surprises, and know your daily energy use before the day starts — using a machine-learning model trained on *your own* historical grid-import data and local weather. Forecasts are published as native Home Assistant sensor entities and update every hour. The model retrains weekly to adapt to seasonal patterns and changes in your household.

> **Note:** Designed for Home Assistant power users with a smart meter (`total_increasing` kWh sensor). Requires Home Assistant 2023.x+ and AppDaemon 4.x.

---

## Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Published sensors](#published-sensors)
- [How it works](#how-it-works)
- [Backfilling history](#backfilling-history)
- [Weather sources](#weather-sources)
- [EV charging detection](#ev-charging-detection)
- [Sub-energy sensors](#sub-energy-sensors)
- [Vacation / Away mode](#vacation--away-mode)
- [MQTT Discovery](#mqtt-discovery-optional)
- [Troubleshooting](#troubleshooting)
- [Security notes](#security-notes)
- [Licence](#licence)

---

## Quick Start

These four steps get forecasts running. Skip MQTT Discovery, sub-sensors, and backfill for now — link to each is in the relevant section.

**1. Install AppDaemon and configure dependencies.**
In HA go to **Settings → Add-ons → Add-on Store**, install **AppDaemon**, then paste the dependency block from [Requirements → AppDaemon add-on configuration](#appdaemon-add-on-configuration) into the add-on's Configuration tab and save.

**2. Copy the app files** into your AppDaemon apps directory:
```
<config>/appdaemon/apps/
├── apps.yaml                    ← create in step 3
└── energy_forecast/
    ├── __init__.py
    ├── energy_forecast.py
    ├── ha_data.py
    ├── model.py
    ├── weather.py
    └── const.py
```

**3. Create `apps.yaml`** from the example and set the three required keys:
```bash
cp apps/apps.yaml.example /config/appdaemon/apps/apps.yaml
```
Open the file and fill in `energy_sensor`, `latitude`, and `longitude`. This file stays in place permanently — it is your live configuration.

**4. Restart AppDaemon.** Watch the log for:
```
HA Energy Forecast ready.
```
Within a minute, `sensor.energy_forecast_setup_status` will read `ok` and forecasts will begin publishing. If you have fewer than 48 hours of history, see [Backfilling history](#backfilling-history).

---

## Features

- **48-hour hourly forecast** — trained on your own consumption history, not generic averages
- **Works on any hardware** — including armv7 Raspberry Pi (LightGBM with automatic scikit-learn fallback when no C compiler is available)
- **High-resolution local weather** — SRG-SSR forecast (Switzerland) with automatic Open-Meteo fallback, so a forecast is always available
- **EV charging detection** — EV sessions are identified and subtracted from the training signal so they don't distort household baseline forecasts; detected kWh are published as separate sensors
- **Appliance-level context** — optional sub-energy sensors (heat pump, dishwasher, etc.) give the model lag features per appliance
- **Local outdoor temperature blending** — if you have an outdoor sensor, its live reading is blended with the weather forecast for the first few hours
- **Exponential sample weighting** — recent data influences the model more than old data
- **Self-healing** — graceful fallbacks at every external dependency (weather API, HA history, ML packages)
- **MQTT Discovery** (optional) — registers all sensors in the HA entity registry so you can assign them to areas, add labels, and rename them in the UI

---

## Requirements

**Home Assistant 2023.x+ · AppDaemon 4.x**

### Home Assistant side
- Home Assistant with a cumulative grid-import energy sensor (`state_class: total_increasing`, unit `kWh`)
- [AppDaemon 4.x](https://github.com/AppDaemon/appdaemon) installed as an HA add-on or standalone

### AppDaemon add-on configuration
The HA AppDaemon add-on does **not** read `requirements.txt`. Dependencies must be declared in the add-on's own configuration, which is edited via the HA UI under **Settings → Add-ons → AppDaemon → Configuration**:

```yaml
system_packages:
  - build-base
  - gfortran
  - openblas-dev
  - python3-dev
python_packages:
  - requests>=2.31.0
  - holidays>=0.46
init_commands:
  - >-
    pip install --extra-index-url https://alpine-wheels.github.io/index pandas
    numpy scikit-learn lightgbm
```

`system_packages` provides the Alpine build toolchain needed to compile numpy/scipy extensions. `python_packages` handles pure-Python packages. `init_commands` runs the pip install with the extra index URL required for pre-built Alpine/ARM wheels of pandas, numpy, scikit-learn, and LightGBM.

> **Note:** If LightGBM fails to build on your platform (e.g. armv7 without a C compiler), remove `lightgbm` from the `init_commands` line. The app will automatically fall back to scikit-learn's GradientBoostingRegressor.

This configuration is also available as [`ha_appdaemon_config.yaml`](ha_appdaemon_config.yaml) in the repository root — it is identical to the block above; copy either source.

### Python packages reference

| Package | Notes |
|---------|-------|
| `pandas` ≥ 2.0.0 | |
| `numpy` ≥ 1.24.0 | |
| `requests` ≥ 2.31.0 | |
| `holidays` ≥ 0.46 | Swiss public holiday feature |
| `scikit-learn` ≥ 1.4.0 | Required — GBR fallback engine |
| `lightgbm` ≥ 4.0.0 | Optional — primary engine |

---

## Installation

1. **Install the AppDaemon add-on** if you haven't already:
   - Go to **Settings → Add-ons → Add-on Store**, search for **AppDaemon**, install and start it.
   - Configure dependencies as shown in the [Requirements](#appdaemon-add-on-configuration) section above, then restart the add-on.

2. **Copy the app** into your AppDaemon apps directory so the structure looks like this:
   ```
   <config>/
   └── appdaemon/
       └── apps/
           ├── apps.yaml                    ← create from apps.yaml.example
           └── energy_forecast/
               ├── __init__.py
               ├── energy_forecast.py
               ├── ha_data.py
               ├── model.py
               ├── weather.py
               └── const.py
   ```

3. **Create `apps.yaml`** from the provided example and keep it in place permanently — it is your live configuration:
   ```bash
   cp apps/apps.yaml.example /config/appdaemon/apps/apps.yaml
   ```
   Then edit it with your values (see [Configuration](#configuration) below).

   > **Warning:** `apps.yaml` is **gitignored** in this repo because it contains API credentials. Never commit it.

4. **Restart AppDaemon.** The add-on will run the `init_commands` to install dependencies, then start the app. Watch the AppDaemon log for:
   ```
   HA Energy Forecast initialising…
   ML engine: LightGBM
   Config validated — lat=…
   HA Energy Forecast ready.
   ```

5. **Initial training** runs ~10 seconds after startup. If you have fewer than 48 hours of history the app will log a warning and skip training until more data accumulates. See [Backfilling history](#backfilling-history) to import years of history from the HA SQLite database.

   **Verify it's working:** after ~2 minutes, check `sensor.energy_forecast_setup_status` in **Developer Tools → States** — it should read `ok`. See also the [Troubleshooting quick sanity check](#troubleshooting).

---

## Configuration

All configuration lives in `apps.yaml`. Copy `apps/apps.yaml.example` as your starting point.

```yaml
energy_forecast:
  module: energy_forecast.energy_forecast
  class: EnergyForecast

  # ── Required ──────────────────────────────────────────────────────────────
  energy_sensor: sensor.your_grid_import_sensor
  latitude: 47.0     # decimal degrees
  longitude: 8.0     # decimal degrees

  # ── SRG-SSR weather API (optional) ───────────────────────────────────────
  # High-quality Swiss forecast. If omitted, Open-Meteo is used instead.
  # Obtain credentials free at https://developer.srgssr.ch
  # The nearest weather station is resolved automatically from latitude/longitude.
  # srg_client_id: YOUR_CLIENT_ID
  # srg_client_secret: YOUR_CLIENT_SECRET

  # ── Optional ──────────────────────────────────────────────────────────────
  # outdoor_temp_sensor: sensor.outdoor_temperature
  timezone: Europe/Zurich

  # Exponential sample weight half-life in days (default: 90).
  # Lower = recent data weighted more heavily. 0 = disable weighting.
  weight_halflife_days: 90

  # EV charging detection thresholds (defaults shown).
  # ev_charging_threshold_kwh: 7    # hours above this are classified as EV
  # ev_charger_kw: 9.0              # fixed charger load subtracted from those hours

  # Path override for the energy history CSV (default: next to energy_forecast.py).
  # cache_path: /config/appdaemon/apps/energy_forecast/energy_history.csv

  # Optional: cumulative kWh sub-sensors tracked as lag features (see below).
  # sub_energy_sensors:
  #   - sensor.heat_pump_energy_kwh
  #   - sensor.dishwasher_energy_kwh

  # Vacation / away mode (optional).
  # away_mode_entity: input_boolean.vacation_mode
  # away_return_entity: input_datetime.vacation_return

  # Anomaly detection threshold (optional, default: 3.0).
  # binary_sensor.energy_forecast_unusual_consumption fires when the latest
  # actual consumption deviates more than this many std-deviations from the
  # day-ahead prediction. Requires ≥10 matched hours (cold-start safe).
  # anomaly_sigma_threshold: 3.0
```

> **Note:** To find your `energy_sensor` entity ID, go to **Developer Tools → States**, filter by `energy` or `kwh`, and look for your grid-import meter — a sensor whose state increases continuously and never resets to zero each day.

### Parameter reference

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `energy_sensor` | Yes | — | Entity ID of your cumulative grid-import kWh meter (`state_class: total_increasing`) |
| `latitude` | Yes | — | Home latitude in decimal degrees |
| `longitude` | Yes | — | Home longitude in decimal degrees |
| `srg_client_id` | No | — | SRG-SSR API client ID. If absent, Open-Meteo is used |
| `srg_client_secret` | No | — | SRG-SSR API client secret |
| `outdoor_temp_sensor` | No | — | Entity ID of an outdoor temperature sensor. Blended with forecast for hours 0–6 |
| `timezone` | No | `Europe/Zurich` | IANA timezone name |
| `weight_halflife_days` | No | `90` | Sample weight half-life. `0` disables exponential weighting |
| `ev_charging_threshold_kwh` | No | `7` | Hours above this value (kWh/h) are treated as EV charging |
| `ev_charger_kw` | No | `9.0` | Fixed charger power subtracted from EV hours (kW) |
| `cache_path` | No | Next to `energy_forecast.py` | Override path for the energy history CSV file |
| `holiday_canton` | No | — | Two-letter Swiss canton code (e.g. `ZH`, `BE`, `GE`). Adds cantonal holidays to the `is_public_holiday` feature in addition to federal ones |
| `adaptive_retrain_threshold` | No | `2.0` | Ratio of live day-ahead MAE to CV MAE that triggers an early retrain. Set to `0` to disable. |
| `sub_energy_sensors` | No | `[]` | List of cumulative kWh sub-sensor entity IDs (heat pump, dishwasher, etc.) to track as `lag_24h`/`lag_168h` features. Must be `total_increasing` kWh meters. See [Sub-energy sensors](#sub-energy-sensors). |
| `away_mode_entity` | No | — | Entity ID of a boolean entity (e.g. `input_boolean.vacation_mode`). When `"on"`, the model learns lower vacation-period consumption from history and predicts accordingly via the `is_away` feature. |
| `away_return_entity` | No | — | Entity ID of a datetime entity (e.g. `input_datetime.vacation_return`). When set, `is_away` flips to 0 at the return hour within the 48-hour forecast window. Requires `away_mode_entity`. |
| `anomaly_sigma_threshold` | No | `3.0` | Std-deviation multiplier for `binary_sensor.energy_forecast_unusual_consumption`. Fires when the latest actual–prediction residual exceeds this multiple of the historical residual std. Must be `> 0`. Silent until ≥ 10 matched hours accumulate. |
| `mqtt_discovery` | No | `false` | Enable MQTT Discovery mode. Registers all sensors in the HA entity registry (area assignment, labels). Requires a running MQTT broker and the AppDaemon MQTT plugin. See [MQTT Discovery](#mqtt-discovery-optional) |
| `mqtt_namespace` | No | `mqtt` | AppDaemon MQTT plugin namespace. Must match the `namespace:` key in the MQTT plugin block of `appdaemon.yaml` |
| `mqtt_discovery_prefix` | No | `homeassistant` | HA MQTT discovery prefix. Change only if your HA instance uses a non-default discovery prefix |

> **Note:** `plz` (Swiss postal code) is silently accepted for backward compatibility but has no effect. The nearest SRG-SSR weather station is resolved from `latitude`/`longitude`. You can remove it from existing configs.

---

## Published sensors

After install you will see sensors in **Developer Tools → States** under the `sensor.energy_forecast_*` prefix. With [MQTT Discovery](#mqtt-discovery-optional) enabled they appear as a single **HA Energy Forecast** device in the entity registry.

All sensors have `unit_of_measurement: kWh` and carry `attribution`, `model_engine`, and `last_trained` attributes.

> **Note — MQTT Discovery entity IDs:** When `mqtt_discovery: true` is set, Home Assistant
> creates entities under the device "HA Energy Forecast". Entity IDs take the form
> `sensor.ha_energy_forecast_<unique_id>` (e.g. `sensor.ha_energy_forecast_energy_forecast_today`).
> The `sensor.energy_forecast_*` IDs in the table below reflect the `set_state()` path; update
> any automations accordingly when switching modes.

### Forecast totals

| Entity ID | Description |
|-----------|-------------|
| `sensor.energy_forecast_next_1h` | Predicted consumption for the next hour |
| `sensor.energy_forecast_next_3h` | Predicted consumption for the next 3 hours |
| `sensor.energy_forecast_today` | Total for today (midnight to midnight): actuals for elapsed hours + forecast for remaining hours |
| `sensor.energy_forecast_tomorrow` | Predicted total for tomorrow |

### Prediction intervals (80% confidence)

Published once quantile models are trained (first retrain cycle after install). Elapsed hours use actuals for both bounds; the interval applies only to the forecast portion.

| Entity ID | Description |
|-----------|-------------|
| `sensor.energy_forecast_next_3h_low` | 10th-percentile forecast for the next 3 hours |
| `sensor.energy_forecast_next_3h_high` | 90th-percentile forecast for the next 3 hours |
| `sensor.energy_forecast_today_low` | 10th-percentile total for today |
| `sensor.energy_forecast_today_high` | 90th-percentile total for today |
| `sensor.energy_forecast_tomorrow_low` | 10th-percentile total for tomorrow |
| `sensor.energy_forecast_tomorrow_high` | 90th-percentile total for tomorrow |

### 3-hour block forecasts

One sensor per 3-hour block, for both today and tomorrow:

| Entity ID pattern | Example | Description |
|-------------------|---------|-------------|
| `sensor.energy_forecast_today_HH_HH` | `sensor.energy_forecast_today_06_09` | Today 06:00–09:00 kWh |
| `sensor.energy_forecast_tomorrow_HH_HH` | `sensor.energy_forecast_tomorrow_18_21` | Tomorrow 18:00–21:00 kWh |

Slots: `00_03`, `03_06`, `06_09`, `09_12`, `12_15`, `15_18`, `18_21`, `21_24` (8 slots × 2 days = 16 sensors)

### EV charging actuals

| Entity ID | Description |
|-----------|-------------|
| `sensor.energy_forecast_ev_today` | EV kWh detected in grid import today |
| `sensor.energy_forecast_ev_yesterday` | EV kWh detected in grid import yesterday |

These sensors carry `ev_threshold_kwh` and `ev_charger_kw` as attributes.

### Model diagnostics

| Entity ID | Description |
|-----------|-------------|
| `sensor.energy_forecast_model_mae` | Model mean absolute error (kWh). Attributes include `cv_mae`, `model_engine`, `last_trained` |
| `sensor.energy_forecast_mae_7d` | Rolling mean absolute error over the last 7 days. Attribute `n_pairs` shows how many prediction–actual pairs were used. State is `"0.0"` until enough history accumulates. |
| `sensor.energy_forecast_mae_30d` | Rolling MAE over the last 30 days (`n_pairs` attribute). Reaches full depth after ~30 days. |
| `sensor.energy_forecast_setup_status` | Setup health check. State is `ok` when all packages loaded correctly, or `missing_packages` when one or more pip packages failed to import. The `missing_packages` attribute lists the affected package names — use it to diagnose install issues directly from **Developer Tools → States** without reading AppDaemon logs. |

### Anomaly detection

| Entity ID | Description |
|-----------|-------------|
| `binary_sensor.energy_forecast_unusual_consumption` | `on` when the latest actual consumption deviates more than `anomaly_sigma_threshold` std-deviations from the stored day-ahead prediction. `off` during cold-start (< 10 matched hours). Attributes: `residual_kwh`, `residual_std_kwh`, `sigma_threshold`, `n_pairs`. |

---

## How it works

### Data pipeline

```
HA get_history()  ──┐
                    ├── _merge_energy_frames() ──► energy_history.csv (cache)
energy_history.csv ─┘         (HA wins on conflict)
        │
        ▼
EV detection  ──► baseline_df (EV hours have charger load subtracted)
        │
        ├── fetch_historical_weather()  [Open-Meteo archive]
        │
        ▼
feature engineering + exponential weighting
        │
        ▼
LightGBM / sklearn GBR  ──► model saved to models/energy_model.pkl
```

LightGBM is the primary engine. On platforms without a C compiler (e.g. armv7 Raspberry Pi), it falls back automatically to scikit-learn's GradientBoostingRegressor, which produces equivalent accuracy.

### Prediction pipeline (hourly)

```
fetch_forecast()  [SRG-SSR → Open-Meteo fallback]
        │
        ├── fetch_recent_energy()  [last 2 days, for lag features]
        │
        ├── live outdoor temp  [blended with forecast for h=0..6]
        │
        ▼
48-hour feature matrix  ──► model.predict()  ──► publish sensors
```

### Schedule

| Event | Timing |
|-------|--------|
| Initial training | ~10 seconds after startup |
| Sensor update | ~2 minutes after startup |
| Retrain | Every 7 days (168 hours) |
| Sensor update | Every hour |
| Adaptive retrain | Any hourly update where live day-ahead MAE exceeds `adaptive_retrain_threshold` × CV MAE (≥ 24 matched pairs required; 24h cooldown between triggers) |

### Features used

| Category | Features |
|----------|----------|
| Calendar | hour, day-of-week, month, season, hour-of-week |
| Cyclical encodings | sin/cos of hour, day-of-week, month, day-of-year (`doy_sin`/`doy_cos`) |
| Horizon | `hours_ahead` (0–47, how far into the future the row is) |
| Weather | temp, precipitation, sunshine, wind, cloud cover, direct solar radiation, heating/cooling degree hours, 3-day rolling temperature anchored in measured data |
| Autoregressive lags | `lag_1h`, `lag_2h`, `lag_6h`, `lag_12h` (short horizon); `lag_24h`, `lag_48h`, `lag_72h`, `lag_168h`, `lag_336h` (daily/weekly) |
| Rolling consumption | 24 h mean, 24 h std, 7-day mean |
| Holidays | Swiss public holiday flag; days to/since nearest holiday (capped at 3); configurable cantonal holidays |
| EV probability | `likely_ev_hour` — binary flag per hour-of-week slot where EV sessions were historically ≥ 15% frequent |
| Away / vacation | `is_away` — binary flag; 1 during periods when `away_mode_entity` is "on"; teaches the model lower vacation-period consumption |

When `sub_energy_sensors` is configured, each sub-sensor adds four features: `lag_24h` (same hour yesterday), `lag_168h` (same hour last week, requires ≥ 268 rows of sub-sensor history), `{prefix}_active_24h` (was the appliance active in the past 24 h?), and `{prefix}_runs_7d` (how many on/off cycles in the past 7 days).

Lag features are dynamically enabled as history grows — short-horizon lags (`lag_1h` etc.) activate at ≥ 101 rows; the full autoregressive feature set is active at ≥ 436 rows (≈ 18 days).

### Model persistence

The trained model and metadata are saved as pickle files in `apps/energy_forecast/models/`. Each file has a SHA-256 sidecar (`.sha256`) for integrity verification. A corrupted or missing sidecar triggers a warning and cold-start retrain.

---

## Backfilling history

If you have existing Home Assistant data you want to import before the first training run, use the included backfill tool. It reads directly from the HA SQLite database (no REST API, no token required) and can import up to one year of hourly data.

**1. Add to `apps.yaml` temporarily:**
```yaml
energy_history_backfill:
  module: energy_forecast.energy_history_backfill
  class: EnergyHistoryBackfill
  energy_sensor: sensor.your_grid_import_sensor
  ha_db_path: /homeassistant/home-assistant_v2.db  # adjust path for your setup
```

Common database paths:

| Setup | Path |
|-------|------|
| HAOS add-on | `/homeassistant/home-assistant_v2.db` |
| HAOS (older) | `/config/home-assistant_v2.db` |
| Docker | `/config/home-assistant_v2.db` |

**2. Restart AppDaemon.** Watch the log for:
```
Retrieved N raw statistic rows from DB.
After diff & filtering: N clean hourly rows.
Saved N rows to energy_history.csv (+N rows added). Range: YYYY-MM-DD → YYYY-MM-DD.
Backfill complete — remove 'energy_history_backfill' from apps.yaml and delete energy_history_backfill.py.
```

**3. Remove the backfill entry** from `apps.yaml` and delete `apps/energy_forecast/energy_history_backfill.py` from your AppDaemon apps directory. The main app will now have a full training set.

> **Note:** The backfill tool requires the energy sensor to have `state_class: total_increasing` and to have been tracked by the HA recorder. The `statistics` table (never purged by HA) is used — not the short-lived `states` table.

---

## Weather sources

| Source | Used for | Requires |
|--------|----------|----------|
| [Open-Meteo Archive](https://open-meteo.com/) | Historical weather for training | Nothing (free, no key) |
| [SRG-SSR Forecast API](https://developer.srgssr.ch) | 7-day hourly forecast | Free API key |
| [Open-Meteo Forecast](https://open-meteo.com/) | Forecast fallback | Nothing (free, no key) |

Both SRG-SSR and Open-Meteo provide temperature, precipitation, sunshine, and wind data. SRG-SSR offers higher spatial resolution for Swiss locations. When SRG-SSR credentials are configured, the app resolves the nearest weather station from your `latitude`/`longitude` (via the v2 `/geolocations` endpoint), then fetches a 7-day hourly forecast from `/forecastpoint/{id}`. Open-Meteo is supplemented for cloud cover and direct radiation, and to anchor the 3-day rolling temperature feature with measured history. If credentials are not configured, or if any SRG-SSR request fails, Open-Meteo is used automatically — the app will never fail to produce a forecast due to a weather API issue.

---

## EV charging detection

Any hour where gross grid import exceeds `ev_charging_threshold_kwh` (default 7 kWh/h) is classified as an EV charging session. The fixed charger load (`ev_charger_kw`, default 9 kW) is subtracted from those hours before training, leaving the concurrent household baseline (lighting, cooking, etc.) intact.

This means the model trains on the true household signal even on days with EV sessions. The raw detected EV kWh are published separately as `sensor.energy_forecast_ev_today` and `sensor.energy_forecast_ev_yesterday`.

Tune the threshold in `apps.yaml` to match your charger and household ceiling. The default 7 kWh/h suits a 9–11 kW charger with a household ceiling below 6.5 kWh/h.

---

## Sub-energy sensors

The optional `sub_energy_sensors` list lets you give the model appliance-level context — instead of seeing only the aggregate grid import, it can also see how much the heat pump or dishwasher consumed at the same hour yesterday and last week.

```yaml
sub_energy_sensors:
  - sensor.heat_pump_energy_kwh
  - sensor.dishwasher_energy_kwh
```

For each sensor the model gains four features:

| Feature | Value | Activation |
|---------|-------|------------|
| `sub_<name>_lag_24h` | kWh consumed at the same hour yesterday | always |
| `sub_<name>_lag_168h` | kWh consumed at the same hour last week | ≥ 268 h of sub-sensor history |
| `sub_<name>_active_24h` | 1 if the appliance had any non-zero reading in the past 24 h, else 0 | always |
| `sub_<name>_runs_7d` | Number of on/off cycles (0 → >0 transitions) in the past 7 days | always |

**Requirements:**
- The sensor must be a `total_increasing` cumulative kWh meter (same type as `energy_sensor`). Power sensors (W or kW) must first be integrated into a kWh template helper in HA.
- Hours when the appliance is off appear as 0 kWh (not excluded), so the model correctly learns that a zero-lag means the appliance was idle.
- Each sub-sensor gets its own CSV cache file (`sub_<name>.csv`) in the same directory as the main energy cache.

**How many sensors?** 3–5 is a practical limit — each sensor adds 2 feature columns and a separate HA history fetch on every retrain and hourly update.

**Backward compatibility:** Omitting `sub_energy_sensors` (or leaving it commented out) produces no behaviour change. Old model files without sub-sensor features load cleanly and continue to work.

---

## Vacation / Away mode

When you go on holiday your household consumption drops significantly — the model would otherwise see those low-consumption days as noise and regress toward them over time. The vacation/away feature teaches the model that these are distinct conditions by adding an `is_away` binary flag to every training row and every prediction row.

### Configuration

Add the optional keys to `apps.yaml`:

```yaml
  # Vacation / away mode (optional).
  away_mode_entity: input_boolean.vacation_mode
  away_return_entity: input_datetime.vacation_return   # optional; requires away_mode_entity
```

| Key | Required | Description |
|-----|----------|-------------|
| `away_mode_entity` | No | Entity ID of a `input_boolean` (or any boolean-like entity). When its state is `"on"`, `is_away` is set to 1 for training rows and prediction rows alike. |
| `away_return_entity` | No | Entity ID of an `input_datetime`. When set, `is_away` flips back to 0 at the configured return hour within the 48-hour forecast window. Requires `away_mode_entity`. |

### How it works

- During training, every historical hour when `away_mode_entity` was `"on"` is labelled `is_away = 1`. The model learns that those hours have characteristically lower consumption.
- During prediction, if `away_mode_entity` is currently `"on"`, all 48 forecast hours are initially marked `is_away = 1`.
- If `away_return_entity` is also set and contains a valid future datetime within the 48-hour window, the hours from the return time onward are flipped back to `is_away = 0` — so the forecast transitions from vacation-level to normal consumption at the expected return hour.

**Backward compatibility:** Omitting both keys (or leaving them commented out) produces no behaviour change — `is_away` defaults to 0 for all rows and the model behaves identically to prior versions.

---

## MQTT Discovery (optional)

By default the app publishes all sensors via AppDaemon's `set_state()` API, which writes values to the HA **state machine** only. This means sensors appear in **Developer Tools → States** and can be used in dashboards and automations, but they are **not** registered in the **entity registry** — so you cannot assign them to an area, add labels, or rename them from the HA UI.

Enabling MQTT Discovery registers every sensor as a proper HA entity under a single **HA Energy Forecast** device, unlocking:
- Area assignment (persists across restarts)
- Labels and aliases
- UI renaming without breaking automations

### Prerequisites

1. **MQTT broker** — Mosquitto is the standard choice. Install it as an HA add-on via **Settings → Add-ons → Mosquitto broker**.
2. **AppDaemon MQTT plugin** — add the following block to `appdaemon/appdaemon.yaml`:

```yaml
plugins:
  MQTT:
    type: mqtt
    namespace: mqtt
    client_host: 192.168.1.x   # ← your broker IP or hostname
    client_port: 1883
```

> **Note:** If your broker requires authentication add `client_user` and `client_password` to the plugin block. See the [AppDaemon MQTT plugin docs](https://appdaemon.readthedocs.io/en/latest/AD_API_REFERENCE.html#mqtt) for all options.

### Enabling MQTT Discovery

Add these keys to `apps.yaml`:

```yaml
energy_forecast:
  # … existing keys …

  # ── MQTT Discovery ──────────────────────────────────────────────────────
  mqtt_discovery: true
  # mqtt_namespace: mqtt            # must match the namespace in appdaemon.yaml
  # mqtt_discovery_prefix: homeassistant   # change only if HA uses a custom prefix
```

Restart AppDaemon. After a few seconds, the device **HA Energy Forecast** appears in **Settings → Devices & Services → MQTT → Devices**.

### What gets registered

All sensors are registered at startup. The 6 prediction-interval sensors (`*_low` / `*_high`) are registered on the **first hourly update after the quantile models finish training** — typically within an hour of the initial retrain.

| Sensor group | Count |
|---|---|
| Forecast totals (`next_1h`, `next_3h`, `today`, `tomorrow`) | 4 |
| 3-hour blocks (today + tomorrow) | 16 |
| EV actuals (`ev_today`, `ev_yesterday`) | 2 |
| Model diagnostics (`mae`, `mae_7d`, `mae_30d`) | 3 |
| Setup status | 1 |
| Anomaly detection (`unusual_consumption`) | 1 |
| Prediction intervals (`*_low`/`*_high`) | 6 (lazy) |
| **Total** | **33** |

### Availability

When AppDaemon starts it publishes `online` to the availability topic. When AppDaemon stops cleanly, it publishes `offline` and all sensors show **Unavailable** in the HA UI automatically.

### Reverting to set_state() mode

Set `mqtt_discovery: false` (or remove the key). The app reverts to writing directly to the HA state machine. Previously registered MQTT entities remain in the entity registry until you delete them manually from **Settings → Devices & Services → MQTT**.

---

## Troubleshooting

**Quick sanity check:**

| Check | Expected |
|-------|----------|
| `sensor.energy_forecast_setup_status` | `ok` |
| `sensor.energy_forecast_model_mae` | a numeric value (kWh) |
| AppDaemon log | `HA Energy Forecast ready.` |

---

**App starts but sensors are `unavailable` for more than 5 minutes**
- Check the AppDaemon log for `Retraining failed` or `Sensor update failed` errors.
- Verify `energy_sensor` is correct and returns a numeric state.

**`No history found in HA or Cache` error**
- The sensor has no state history in HA and the CSV cache is empty or missing.
- Use the [backfill tool](#backfilling-history) to import history from the HA SQLite DB.

**`Insufficient history (N h). Skipping.`**
- Fewer than 48 hours of data are available. The app will retry on the next training cycle. Normal on a new install — history accumulates automatically.

**`ML engine: sklearn GBR` instead of LightGBM**
- LightGBM failed to install (no C compiler on the host, e.g. armv7).
- The sklearn fallback is fully functional. If you want LightGBM, ensure the build toolchain system packages are present in the add-on configuration.
- On Alpine ARM without a C compiler: remove `lightgbm` from the `init_commands` line in the add-on configuration to avoid a failed install attempt.

**Forecast accuracy is poor in the first few weeks**
- The model needs at least a few weeks of data to learn daily and weekly patterns. Use the backfill tool to give it a head start.
- Check `sensor.energy_forecast_model_mae` — as a rough guide, MAE below ~15% of your average hourly consumption suggests a well-fitted model (e.g. below 0.3 kWh for a household averaging ~2 kWh/h). Larger homes or those with EV/heat-pump loads will have proportionally higher absolute MAE.

**DST fall-back warning in the log**
- `DST fall-back: N rows share M duplicate naive timestamp(s) after merge` is expected on the last Sunday of October. It is informational — the merge still completes correctly.

**`Could not fetch recent actuals for lag features`**
- HA history fetch failed. The sensor update proceeds without lag features; the model fills them with training-set medians. No action required.

---

## Security notes

- `apps.yaml` is **gitignored**. Copy from `apps.yaml.example` and fill in credentials locally. Never commit the live file.
- SRG-SSR credentials are optional. If you don't use them, no credentials are needed anywhere.
- The backfill tool accesses the HA SQLite database directly (read-only) — ensure AppDaemon has file read access to the DB path.

---

## Licence

[MIT](LICENSE) © 2026 Martin Zenker
