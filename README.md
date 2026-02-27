# HA Energy Forecast

[![hacs_badge](https://img.shields.io/badge/HACS-Custom-orange.svg)](https://github.com/hacs/integration)
[![HA Version](https://img.shields.io/badge/Home%20Assistant-2023.6%2B-blue.svg)](https://www.home-assistant.io/)

A Home Assistant custom integration that predicts your **gross household electricity consumption** using a locally-trained LightGBM model, MeteoSwiss weather forecasts, and your own historical energy data.

Designed for Swiss households with heat pumps and/or electric vehicles.

---

## Features

- 📊 **Local ML model** — LightGBM trained on your own HA energy history (no cloud AI)
- 🌤️ **MeteoSwiss forecast** — Swiss PLZ API with Open-Meteo fallback
- 🌡️ **Live outdoor sensor** — blends your real-time temperature into near-term predictions
- ⚡ **Gross consumption** — predicts total grid draw, independent of solar production
- 🔄 **Weekly auto-retrain** — model updates silently in a background thread
- 📉 **MAE diagnostic sensor** — tracks holdout prediction accuracy after each retrain
- 🖥️ **Config Flow UI** — fully configured through Settings → Integrations, no YAML

---

## Sensors

| Entity | Description |
|---|---|
| `sensor.gross_forecast_next_3h` | kWh predicted for the next 3 hours |
| `sensor.gross_forecast_today` | kWh predicted total for today |
| `sensor.gross_forecast_tomorrow` | kWh predicted total for tomorrow |
| `sensor.gross_forecast_today_00_00_03_00` | Today 00:00–03:00 |
| *(+ 7 more today blocks)* | … through 21:00–24:00 |
| `sensor.gross_forecast_tomorrow_*` | 8 × 3-hour blocks for tomorrow |
| `sensor.forecast_model_mae` | Model mean absolute error (kWh/h) — diagnostic |

All energy sensors have `device_class: energy`, `unit_of_measurement: kWh`.

---

## Requirements

- Home Assistant 2023.6 or later
- [HACS](https://hacs.xyz/) installed
- A cumulative kWh grid import sensor tracked in the **Energy Dashboard**
- Internet access for MeteoSwiss / Open-Meteo APIs
- 1–3 months of energy history (more = better model accuracy)

---

## Installation

### 1. Add as custom HACS repository

1. In HA: open **HACS → Integrations**
2. Click the three-dot menu (⋮) → **Custom repositories**
3. Enter your repository URL: `https://github.com/YOUR_USERNAME/ha-energy-forecast`
4. Category: **Integration**
5. Click **Add**

### 2. Install

1. Search for **HA Energy Forecast** in HACS → Integrations
2. Click **Download**
3. Restart Home Assistant

### 3. Configure

1. Go to **Settings → Devices & Services → Add Integration**
2. Search for **HA Energy Forecast**
3. Fill in the form:

| Field | Description |
|---|---|
| Grid import energy sensor | Your cumulative kWh grid sensor (from Energy Dashboard) |
| Outdoor temperature sensor | Optional — improves next-3h accuracy |
| MeteoSwiss PLZ | Your 4-digit Swiss postal code (e.g. `8001`) |
| Latitude / Longitude | Your home coordinates (pre-filled from HA settings) |

4. Click **Submit** — the integration will appear and begin fetching data immediately

> **First training:** The model trains in the background after the first hourly update. You need at least ~720 hours (~1 month) of energy data. Sensors will show `unavailable` until training completes.

---

## How It Works

```
Every 60 minutes:
  ├─ Weekly: retrain LightGBM on HA energy history + archive weather (executor thread)
  ├─ Fetch MeteoSwiss 48h forecast (PLZ API → Open-Meteo fallback)
  ├─ Read live outdoor temperature from HA sensor
  ├─ Build 48-hour feature matrix:
  │     time features + weather + live temp blend
  ├─ Predict hourly gross kWh
  └─ Aggregate → 3 summary sensors + 16 block sensors + MAE
```

### Live temperature blending

When an outdoor sensor is configured, the live reading is used as follows:

| Horizon | Temperature source |
|---|---|
| 0–3 hours | 100% live sensor |
| 3–12 hours | Linear blend: sensor → forecast |
| 12+ hours | 100% MeteoSwiss forecast |

### Model features

Time features (hour, block, weekday, month, season, cyclical encodings), MeteoSwiss temperature / precipitation / sunshine / wind, heating & cooling degree-hours, plus `outdoor_temp_live` and `temp_bias` when the outdoor sensor is configured.

---

## Troubleshooting

**Sensors show `unavailable` after setup**
→ Normal — the model needs ~1 month of hourly history to train. Check the HA log for `"Retraining model..."`.

**`sensor_not_found` error in setup**
→ The entity ID doesn't exist in HA. Go to Settings → Entities and search for your energy meter.

**MeteoSwiss fetch warnings in log**
→ The integration automatically falls back to Open-Meteo. No action needed.

**MAE sensor shows a high value**
→ Normal for the first few weeks. The model improves as it sees more seasonal variation.

**Model not retraining**
→ Check that HA has internet access and that the energy sensor has `state_class: total_increasing`.

---

## Reconfiguration

To change sensors or location: **Settings → Integrations → HA Energy Forecast → Configure**

Changing the energy sensor will trigger a model retrain on the next hourly cycle.

---

## Contributing

Pull requests welcome. Please open an issue first for significant changes.
