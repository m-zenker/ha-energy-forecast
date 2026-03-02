# HA Energy Forecast — AppDaemon Installation Guide

## Prerequisites

- Home Assistant with the **AppDaemon** add-on installed
  (HAOS: Settings → Add-ons → Add-on Store → search "AppDaemon")
- A cumulative grid-import energy sensor in HA (e.g. from a Shelly EM, Fronius, or similar)
- At least ~1 month of recorded history for that sensor (720 hourly records minimum)

---

## Step 1 — Install the AppDaemon add-on

If you haven't already:

1. Go to **Settings → Add-ons → Add-on Store**
2. Search for **AppDaemon** and install it
3. In the AppDaemon add-on configuration tab, leave defaults as-is
4. Start the add-on and wait for it to show **Started**

---

## Step 2 — Copy the app files

Your Home Assistant config folder has a folder called `appdaemon/`. Copy the files from this zip so the structure looks like this:

```
<config>/
└── appdaemon/
    └── apps/
        ├── apps.yaml                          ← edit this
        └── energy_forecast/
            ├── __init__.py
            ├── energy_forecast.py
            ├── ha_data.py
            ├── model.py
            ├── weather.py
            └── requirements.txt
```

If you already have an `apps.yaml`, don't replace it — instead copy just the `energy_forecast:` block from the provided `apps.yaml` into your existing one.

---

## Step 3 — Configure the app

Open `appdaemon/apps/apps.yaml` and fill in your values:

```yaml
energy_forecast:
  module: energy_forecast.energy_forecast
  class: EnergyForecast

  energy_sensor: sensor.grid_import_kwh   # your cumulative kWh sensor
  plz: "8001"                             # Swiss postal code for MeteoSwiss
  latitude: 47.3769                       # your home latitude
  longitude: 8.5417                       # your home longitude

  # Optional — improves near-term accuracy
  # outdoor_temp_sensor: sensor.outdoor_temperature
```

**Finding your energy sensor entity ID:**
In HA, go to Developer Tools → States, filter by `energy` or `kwh`, and find your grid import meter. It should be a sensor whose state increases continuously (not resets to zero each day).

---

## Step 4 — Install Python dependencies

AppDaemon manages its own pip environment, entirely separate from HA. To install the required packages:

**Option A — via the AppDaemon add-on UI (recommended for HAOS):**

1. Open the AppDaemon add-on → **Configuration** tab
2. Find the `python_packages` list and add:

```yaml
python_packages:
  - pandas>=2.0.0
  - numpy>=1.24.0
  - requests>=2.31.0
  - scikit-learn>=1.3.0
  - lightgbm>=4.0.0
```

3. Save and restart the add-on

**Option B — via requirements.txt (AppDaemon 4.4+):**

AppDaemon automatically installs packages listed in any `requirements.txt` found in the apps directory. The file is already included at `apps/energy_forecast/requirements.txt` — no action needed beyond restarting AppDaemon.

> **Note on lightgbm:** On ARM devices (Raspberry Pi 3/4 with 32-bit OS), lightgbm may fail to install because no pre-built wheel exists for armv7l. If this happens, remove `lightgbm>=4.0.0` from `python_packages` / `requirements.txt`. The app will fall back to scikit-learn's GradientBoostingRegressor automatically — predictions still work, training just takes ~2× longer.

---

## Step 5 — Restart AppDaemon

Restart the AppDaemon add-on. Within 10–15 seconds you should see the sensor entities appear in HA:

- `sensor.energy_forecast_next_3h`
- `sensor.energy_forecast_today`
- `sensor.energy_forecast_tomorrow`
- `sensor.energy_forecast_today_00_03` … `_21_24` (3-hour blocks)
- `sensor.energy_forecast_tomorrow_00_03` … `_21_24` (3-hour blocks)
- `sensor.energy_forecast_model_mae` (diagnostic)

They will show `unavailable` until the first training run completes (typically 2–10 minutes depending on how much history you have).

---

## Step 6 — Check the logs

In the AppDaemon add-on, go to the **Log** tab and look for:

```
INFO AppDaemon: HA Energy Forecast initialising…
INFO AppDaemon: Starting model retraining…
INFO AppDaemon: Retraining complete — engine: LightGBM, MAE: 0.12 kWh/h
```

If you see warnings about insufficient history, check that your `energy_sensor` is correct and that the HA recorder has been running long enough.

---

## Sensors reference

| Entity ID | Description | Unit |
|-----------|-------------|------|
| `sensor.energy_forecast_next_3h` | Predicted consumption in the next 3 hours | kWh |
| `sensor.energy_forecast_today` | Predicted total for today | kWh |
| `sensor.energy_forecast_tomorrow` | Predicted total for tomorrow | kWh |
| `sensor.energy_forecast_today_HH_HH` | 3-hour block for today (×8) | kWh |
| `sensor.energy_forecast_tomorrow_HH_HH` | 3-hour block for tomorrow (×8) | kWh |
| `sensor.energy_forecast_model_mae` | Model mean absolute error | kWh |

---

## Schedule

| Event | Frequency |
|-------|-----------|
| Model retraining | Weekly (every 168 hours) |
| Sensor update (prediction) | Every hour |
| First retrain after startup | 10 seconds after AppDaemon starts |
| First sensor update | ~2 minutes after AppDaemon starts |

---

## Troubleshooting

**Sensors stay `unavailable` after 10 minutes**
Check AppDaemon logs for training warnings. Most likely causes:
- Wrong `energy_sensor` entity ID
- Fewer than 720 hours of recorder history for that sensor
- `scikit-learn` not installed (check logs for import errors)

**`ModuleNotFoundError: No module named 'lightgbm'`**
Remove `lightgbm` from `python_packages` and restart AppDaemon. The app falls back to scikit-learn automatically.

**`get_history` returns empty data**
Make sure the `recorder` integration is enabled in HA (it is by default) and that the entity has been recording for at least a few weeks.

**MeteoSwiss forecast fails**
This only affects users outside Switzerland, or if MeteoSwiss is temporarily down. The app automatically falls back to Open-Meteo, which is free and requires no API key.
