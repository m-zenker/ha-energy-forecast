# HA Energy Forecast

An [AppDaemon](https://appdaemon.readthedocs.io/) app for Home Assistant that forecasts household electricity consumption for the next 48 hours using a machine-learning model trained on your own historical grid-import data and local weather.

Forecasts are published as native Home Assistant sensor entities and update every hour. The model retrains weekly so it adapts to seasonal patterns and changes in your household.

---

## Features

- **48-hour hourly forecast** — trained on your own consumption history, not generic averages
- **LightGBM model** (auto-falls back to scikit-learn GBR on platforms without a C compiler, e.g. armv7)
- **Swiss weather integration** — SRG-SSR high-resolution forecast with automatic Open-Meteo fallback
- **EV charging detection** — EV sessions are identified and subtracted from the training signal so they don't distort household baseline forecasts; detected kWh are published as separate sensors
- **Local outdoor temperature blending** — if you have an outdoor sensor, its live reading is blended with the weather forecast for the first few hours of the prediction window
- **Exponential sample weighting** — recent data influences the model more than old data
- **Persistent CSV cache** — energy history survives Home Assistant database purges
- **Self-healing** — graceful fallbacks at every external dependency (weather API, HA history, ML packages)

---

## Requirements

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

> If LightGBM fails to build on your platform (e.g. armv7 without a C compiler), remove `lightgbm` from the `init_commands` line. The app will automatically fall back to scikit-learn's GradientBoostingRegressor.

The above configuration is also available as [`ha_appdaemon_config.yaml`](ha_appdaemon_config.yaml) in the repository root for easy copy-paste.

### Python packages reference

| Package | Notes |
|---------|-------|
| `pandas` ≥ 2.0.0 | |
| `numpy` ≥ 1.24.0 | |
| `requests` ≥ 2.31.0 | |
| `holidays` ≥ 0.46 | Swiss public holiday feature |
| `scikit-learn` == 1.8.0 | Required — GBR fallback engine |
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

3. **Create `apps.yaml`** from the provided example:
   ```bash
   cp apps/apps.yaml.example /config/appdaemon/apps/apps.yaml
   ```
   Then edit it with your values (see [Configuration](#configuration) below).

   > `apps.yaml` is **gitignored** in this repo because it contains API credentials. Never commit it.

3. **Restart AppDaemon.** The add-on will run the `init_commands` to install dependencies, then start the app. Watch the AppDaemon log for:
   ```
   HA Energy Forecast initialising…
   ML engine: LightGBM
   Config validated — lat=…
   HA Energy Forecast ready.
   ```

4. **Initial training** runs ~10 seconds after startup. If you have fewer than 48 hours of history the app will log a warning and skip training until more data accumulates. See [Backfilling history](#backfilling-history) to import years of history from the HA SQLite database.

---

## Configuration

All configuration lives in `apps.yaml`. Copy `apps/apps.yaml.example` as your starting point.

```yaml
energy_forecast:
  module: energy_forecast.energy_forecast
  class: EnergyForecast

  # ── Required ──────────────────────────────────────────────────────────────
  energy_sensor: sensor.your_grid_import_sensor
  plz: 1234          # Swiss postal code — used for SRG-SSR weather lookup
  latitude: 47.0     # decimal degrees
  longitude: 8.0     # decimal degrees

  # ── SRG-SSR weather API (optional) ───────────────────────────────────────
  # High-quality Swiss forecast. If omitted, Open-Meteo is used instead.
  # Obtain credentials free at https://developer.srgssr.ch
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
```

> **Finding your `energy_sensor` entity ID:** In HA go to **Developer Tools → States**, filter by `energy` or `kwh`, and look for your grid-import meter. It should be a sensor whose state increases continuously (never resets to zero each day).

### Parameter reference

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `energy_sensor` | Yes | — | Entity ID of your cumulative grid-import kWh meter (`state_class: total_increasing`) |
| `plz` | Yes | — | Swiss postal code for SRG-SSR weather lookup |
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

---

## Published sensors

All sensors have `unit_of_measurement: kWh` and carry `attribution`, `model_engine`, and `last_trained` attributes.

### Forecast totals

| Entity ID | Description |
|-----------|-------------|
| `sensor.energy_forecast_next_3h` | Predicted consumption for the next 3 hours |
| `sensor.energy_forecast_today` | Predicted total for today (midnight to midnight) |
| `sensor.energy_forecast_tomorrow` | Predicted total for tomorrow |

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
| Initial training | 10 seconds after startup |
| Sensor update | 130 seconds after startup |
| Retrain | Every 7 days (168 hours) |
| Sensor update | Every hour |

### Features used

Calendar, cyclical encodings (hour, day-of-week, month, hour-of-week), weather (temp, precipitation, sunshine, wind, cloud cover, direct solar radiation, heating/cooling degree hours, 3-day rolling temperature anchored in measured history), autoregressive lags (24 h, 48 h, 72 h, 168 h, 336 h), rolling consumption stats (24 h mean/std, 7-day mean), Swiss public holidays, bridge-day proximity (days to/since nearest holiday, capped at 3).

Lag features are dynamically enabled as history grows — the full feature set is active once you have ≥ 436 hours (≈18 days) of data.

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

**3. Remove the backfill entry** from `apps.yaml` and delete `energy_history_backfill.py`. The main app will now have a full training set.

> The backfill tool requires the energy sensor to have `state_class: total_increasing` and to have been tracked by the HA recorder. The `statistics` table (never purged by HA) is used — not the short-lived `states` table.

---

## Weather sources

| Source | Used for | Requires |
|--------|----------|----------|
| [Open-Meteo Archive](https://open-meteo.com/) | Historical weather for training | Nothing (free, no key) |
| [SRG-SSR Forecast API](https://developer.srgssr.ch) | 7-day hourly forecast | Free API key |
| [Open-Meteo Forecast](https://open-meteo.com/) | Forecast fallback | Nothing (free, no key) |

Both SRG-SSR and Open-Meteo provide temperature, precipitation, sunshine, and wind data. SRG-SSR offers higher spatial resolution for Swiss locations. If credentials are not configured, or if the SRG-SSR request fails for any reason, Open-Meteo is used automatically — the app will never fail to produce a forecast due to a weather API issue.

---

## EV charging detection

Any hour where gross grid import exceeds `ev_charging_threshold_kwh` (default 7 kWh/h) is classified as an EV charging session. The fixed charger load (`ev_charger_kw`, default 9 kW) is subtracted from those hours before training, leaving the concurrent household baseline (lighting, cooking, etc.) intact.

This means the model trains on the true household signal even on days with EV sessions. The raw detected EV kWh are published separately as `sensor.energy_forecast_ev_today` and `sensor.energy_forecast_ev_yesterday`.

Tune the threshold in `apps.yaml` to match your charger and household ceiling. The default 7 kWh/h suits a 9–11 kW charger with a household ceiling below 6.5 kWh/h.

---

## Troubleshooting

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
- Check `sensor.energy_forecast_model_mae` — a MAE below 0.3 kWh indicates a well-fitted model.

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
