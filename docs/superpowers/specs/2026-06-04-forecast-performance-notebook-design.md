# Forecast Performance Notebook — Design Spec

**Date**: 2026-06-04  
**Status**: Approved

## Goal

A self-contained Jupyter notebook (`notebooks/forecast_performance.ipynb`) that pulls live data from Home Assistant and visualises prediction accuracy for the last 30 days using Altair.

## Data Sources

| Source | How fetched | What we get |
|--------|-------------|-------------|
| `pred_history.json` | SMB (inline, like `pull_ha_data.py`) | Per-hour predicted vs actual kWh pairs |
| `energy_history.csv` | SMB (same pull) | Reference for gross_kwh; not used directly in charts |
| `apps.yaml` | SMB (same pull) | lat/lon for Open-Meteo query |
| Open-Meteo archive API | HTTP (requests) | Hourly `temperature_2m` for last 30 days |

Credentials: `SMB_PASSWORD` from env var; `SMB_USER` defaults to `martin`. Notebook prints a clear error and halts if `SMB_PASSWORD` is missing.

## Data Model

After fetch and merge, the primary frame is **`error_df`** (one row per hour, last 30 days):

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime (Europe/Zurich, naive) | Hour start |
| `pred_kwh` | float | Model prediction |
| `actual_kwh` | float | Measured consumption |
| `error` | float | `pred_kwh − actual_kwh` |
| `abs_error` | float | `abs(error)` |
| `temp_c` | float | Outdoor temperature (Open-Meteo) |
| `is_ev` | bool | `actual_kwh > 7 kWh` (EV charging hour) |

Daily aggregate **`daily_df`**:

| Column | Description |
|--------|-------------|
| `date` | Calendar date |
| `daily_pred` | Sum of pred_kwh |
| `daily_actual` | Sum of actual_kwh |
| `daily_mae` | Mean of abs_error |

EV hours are included in all frames but visually flagged (distinct colour/shape).

## Notebook Structure

```
Cell 0:  Markdown header
Cell 1:  Imports
Cell 2:  Configuration (env vars, constants, TZ)
Cell 3:  SMB fetch — pred_history.json, energy_history.csv, apps.yaml
Cell 4:  Open-Meteo archive fetch — hourly temp_2m, last 30 days
Cell 5:  Data preparation — parse, merge, compute error_df + daily_df
Cell 6:  [Markdown] Chart 1 — Daily MAE trend
Cell 7:  Chart 1 code
Cell 8:  [Markdown] Chart 2 — Daily totals
Cell 9:  Chart 2 code
Cell 10: [Markdown] Chart 3 — Predicted vs actual scatter
Cell 11: Chart 3 code
Cell 12: [Markdown] Chart 4 — Hour-of-day heatmap
Cell 13: Chart 4 code
Cell 14: [Markdown] Chart 5 — Error distribution
Cell 15: Chart 5 code
Cell 16: [Markdown] Chart 6 — Temperature correlation
Cell 17: Chart 6 code
```

## Charts

| # | Title | Chart type | X | Y | Colour/detail |
|---|-------|-----------|---|---|---------------|
| 1 | Daily MAE trend | Line (two layers) | date | MAE kWh | Layer 1: daily MAE; Layer 2: 7-day rolling avg |
| 2 | Daily totals | Grouped bar | date | kWh | Two bars per day: predicted (blue) vs actual (orange) |
| 3 | Predicted vs actual | Scatter | actual_kwh | pred_kwh | EV hours = distinct shape; 45° reference line overlay |
| 4 | Hour-of-day heatmap | Rect | hour (0–23) | weekday (Mon–Sun, top→bottom) | Fill = mean abs_error |
| 5 | Error distribution | Bar (histogram) | error (pred−actual) | count | Zero reference line; bin width = 0.1 kWh |
| 6 | Temperature correlation | Scatter | temp_c | abs_error | EV hours = distinct shape |

All charts: `width=700`, `height=300` (heatmap: `height=200`). No vegafusion required (max ~720 hourly rows for 30 days).

## Error Handling

- `SMB_PASSWORD` missing → `raise RuntimeError` in config cell; notebook cannot proceed
- `pred_history.json` has 0 entries → print warning, skip all charts gracefully
- Open-Meteo request fails → `temp_c` column filled with `NaN`; Chart 6 omitted with a printed note
- No matched pred/actual pairs after merge → print summary and skip charts

## Constraints & Conventions

- EV threshold: 7 kWh (matches `EV_CHARGING_THRESHOLD_KWH` in `const.py`)
- Location: `open_meteo_latitude` / `open_meteo_longitude` from `apps.yaml`; fallback to Zurich (47.376, 8.541) if keys absent
- Timezone: `Europe/Zurich` throughout; all timestamps remain naive (stripped tz)
- No new dependencies beyond what's already in the environment (`pysmb`, `altair`, `pandas`, `requests`)
- SMB host: `homeassistant`, port 445, share `addon_configs` (same as `pull_ha_data.py`)
