# Forecast Performance Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `notebooks/forecast_performance.ipynb` — a self-contained notebook that pulls live data from HA via SMB and renders 6 Altair charts of forecast accuracy over the last 30 days.

**Architecture:** Single notebook; no helper modules. SMB pull at the top fetches `pred_history.json`, `energy_history.csv`, and `apps.yaml`. Open-Meteo archive API provides hourly outdoor temperature. Six Altair charts follow in sequential cells.

**Tech Stack:** Python 3.13, Altair 6.x, pandas, pysmb, requests, PyYAML, Jupyter (nb CLI via `notebook-cli` skill)

---

> **IMPORTANT — Notebook tooling:** Always invoke the `notebook-cli` skill before any `.ipynb` work. Use `nb` commands to create, insert, and execute cells. Never write raw notebook JSON directly.

## File Map

| Path | Action | Purpose |
|------|--------|---------|
| `notebooks/forecast_performance.ipynb` | Create | The entire notebook |

---

### Task 1: Create notebook with imports and config cells

**Files:**
- Create: `notebooks/forecast_performance.ipynb`

- [ ] **Step 1: Invoke notebook-cli skill**

  Run: invoke the `notebook-cli` skill so `nb` is available in this session.

- [ ] **Step 2: Create the notebooks directory and blank notebook**

```bash
mkdir -p notebooks
nb new notebooks/forecast_performance.ipynb
```

- [ ] **Step 3: Insert title markdown cell (index 0)**

Content:
```markdown
# Energy Forecast — Prediction Performance (last 30 days)

Pulls live data from Home Assistant via SMB, fetches weather from Open-Meteo, and renders accuracy charts for the last 30 days.

**Pre-requisite:** `SMB_PASSWORD` environment variable must be set before starting the kernel.
```

- [ ] **Step 4: Insert imports code cell (index 1)**

Content:
```python
from __future__ import annotations

import io
import json
import os
from datetime import date, timedelta

import altair as alt
import pandas as pd
import requests
import yaml
from smb.SMBConnection import SMBConnection
```

- [ ] **Step 5: Insert configuration code cell (index 2)**

Content:
```python
# ── Credentials ──────────────────────────────────────────────────────────────
SMB_USER = os.getenv("SMB_USER", "martin")
SMB_PASSWORD = os.getenv("SMB_PASSWORD")
if not SMB_PASSWORD:
    raise RuntimeError(
        "SMB_PASSWORD environment variable is not set. "
        "Set it before starting the kernel: export SMB_PASSWORD=<password>"
    )

# ── Constants ─────────────────────────────────────────────────────────────────
HA_HOST = "homeassistant"
SMB_SHARE = "addon_configs"
AD_BASE = "a0d7b954_appdaemon/apps"
FORECAST_REMOTE = f"{AD_BASE}/energy_forecast"

TZ = "Europe/Zurich"
CUTOFF_DAYS = 30
EV_THRESHOLD_KWH = 7.0  # matches EV_CHARGING_THRESHOLD_KWH in const.py

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

print("Config OK — SMB_PASSWORD is set")
```

- [ ] **Step 6: Run all three cells**

  Use `nb` to execute cells 0–2 (or execute each individually). Expected output from cell 2: `Config OK — SMB_PASSWORD is set`

  If `RuntimeError` is raised, the user must set `SMB_PASSWORD` before continuing.

- [ ] **Step 7: Commit**

```bash
git add notebooks/forecast_performance.ipynb
git commit -m "feat: add forecast performance notebook — scaffold and config"
```

---

### Task 2: SMB fetch section

**Files:**
- Modify: `notebooks/forecast_performance.ipynb` (add cells 3–4)

- [ ] **Step 1: Insert SMB helper markdown cell (index 3)**

Content:
```markdown
## 1 · Fetch data from Home Assistant (SMB)
```

- [ ] **Step 2: Insert SMB fetch code cell (index 4)**

Content:
```python
def _smb_read(remote_path: str) -> bytes:
    conn = SMBConnection(SMB_USER, SMB_PASSWORD, "notebook", HA_HOST, use_ntlm_v2=True)
    assert conn.connect(HA_HOST, 445), f"SMB connection to {HA_HOST}:445 failed"
    buf = io.BytesIO()
    conn.retrieveFile(SMB_SHARE, remote_path, buf)
    conn.close()
    return buf.getvalue()


# Pull files
print("Fetching pred_history.json …")
_pred_raw = json.loads(_smb_read(f"{FORECAST_REMOTE}/pred_history.json"))
print(f"  pred entries : {len(_pred_raw.get('pred', {}))}")
print(f"  actual entries: {len(_pred_raw.get('actuals', {}))}")

print("Fetching apps.yaml …")
_apps_yaml = yaml.safe_load(_smb_read(f"{AD_BASE}/apps.yaml"))

if not _pred_raw.get("pred"):
    print("WARNING: pred_history.json has 0 prediction entries — charts will be empty.")
    print("The app needs at least one update cycle on the HA system to populate this file.")
```

- [ ] **Step 3: Execute the cell**

  Expected output (example):
  ```
  Fetching pred_history.json …
    pred entries : 712
    actual entries: 698
  Fetching apps.yaml …
  ```

  If connection fails, verify `HA_HOST` resolves and SMB port 445 is reachable from this environment.

- [ ] **Step 4: Commit**

```bash
git add notebooks/forecast_performance.ipynb
git commit -m "feat: forecast notebook — SMB fetch section"
```

---

### Task 3: Open-Meteo temperature fetch

**Files:**
- Modify: `notebooks/forecast_performance.ipynb` (add cells 5–6)

- [ ] **Step 1: Insert weather markdown cell (index 5)**

Content:
```markdown
## 2 · Fetch outdoor temperature from Open-Meteo archive
```

- [ ] **Step 2: Insert weather fetch code cell (index 6)**

Content:
```python
# Read lat/lon from apps.yaml; fall back to Zurich city centre
_ef_cfg = _apps_yaml.get("energy_forecast", {})
_lat = _ef_cfg.get("open_meteo_latitude", 47.376)
_lon = _ef_cfg.get("open_meteo_longitude", 8.541)

_start = (date.today() - timedelta(days=CUTOFF_DAYS)).isoformat()
_end = date.today().isoformat()

_url = (
    f"https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={_lat}&longitude={_lon}"
    f"&start_date={_start}&end_date={_end}"
    f"&hourly=temperature_2m&timezone=Europe%2FZurich"
)

try:
    _resp = requests.get(_url, timeout=30)
    _resp.raise_for_status()
    _weather = _resp.json()
    temp_df = pd.DataFrame({
        "timestamp": pd.to_datetime(_weather["hourly"]["time"]),
        "temp_c": _weather["hourly"]["temperature_2m"],
    })
    temp_df["timestamp"] = temp_df["timestamp"].dt.floor("h")
    print(f"Weather fetched: {len(temp_df)} hourly rows, {_start} → {_end}")
except Exception as exc:
    temp_df = pd.DataFrame(columns=["timestamp", "temp_c"])
    print(f"WARNING: Open-Meteo fetch failed ({exc}). Chart 6 (temp correlation) will be skipped.")
```

- [ ] **Step 3: Execute the cell**

  Expected output:
  ```
  Weather fetched: 721 hourly rows, 2026-05-05 → 2026-06-04
  ```

- [ ] **Step 4: Commit**

```bash
git add notebooks/forecast_performance.ipynb
git commit -m "feat: forecast notebook — Open-Meteo temperature fetch"
```

---

### Task 4: Data preparation — error_df and daily_df

**Files:**
- Modify: `notebooks/forecast_performance.ipynb` (add cells 7–8)

- [ ] **Step 1: Insert data-prep markdown cell (index 7)**

Content:
```markdown
## 3 · Data preparation
```

- [ ] **Step 2: Insert data-prep code cell (index 8)**

Content:
```python
def _history_to_df(d: dict, col: str) -> pd.DataFrame:
    rows = [(pd.Timestamp(ts), float(v)) for ts, v in d.items()]
    if not rows:
        return pd.DataFrame(columns=["timestamp", col])
    df = pd.DataFrame(rows, columns=["timestamp", col])
    df["timestamp"] = df["timestamp"].dt.floor("h")
    return df.sort_values("timestamp").reset_index(drop=True)


pred_df = _history_to_df(_pred_raw.get("pred", {}), "pred_kwh")
actuals_df = _history_to_df(_pred_raw.get("actuals", {}), "actual_kwh")

# Merge on matched hours only
error_df = pd.merge(pred_df, actuals_df, on="timestamp", how="inner")

# Filter to last 30 days
_cutoff = pd.Timestamp.now() - pd.Timedelta(days=CUTOFF_DAYS)
error_df = error_df[error_df["timestamp"] >= _cutoff].copy()

# Derived columns
error_df["error"] = error_df["pred_kwh"] - error_df["actual_kwh"]
error_df["abs_error"] = error_df["error"].abs()
error_df["is_ev"] = error_df["actual_kwh"] > EV_THRESHOLD_KWH
error_df["date"] = pd.to_datetime(error_df["timestamp"].dt.date)
error_df["hour"] = error_df["timestamp"].dt.hour
error_df["weekday"] = error_df["timestamp"].dt.day_name()

# Merge outdoor temperature
error_df = error_df.merge(temp_df, on="timestamp", how="left")

# Daily aggregates
daily_df = (
    error_df.groupby("date", as_index=False)
    .agg(daily_pred=("pred_kwh", "sum"), daily_actual=("actual_kwh", "sum"), daily_mae=("abs_error", "mean"))
    .sort_values("date")
)
daily_df["rolling_mae_7d"] = daily_df["daily_mae"].rolling(7, min_periods=1).mean()

print(f"error_df : {len(error_df)} hourly rows, {error_df['timestamp'].min()} → {error_df['timestamp'].max()}")
print(f"daily_df : {len(daily_df)} days")
print(f"EV hours : {error_df['is_ev'].sum()}")
print(f"Overall MAE: {error_df['abs_error'].mean():.3f} kWh")
print(f"Temp coverage: {error_df['temp_c'].notna().sum()}/{len(error_df)} hours")

if error_df.empty:
    print("\nWARNING: No matched pred/actual pairs in the last 30 days. Charts will be empty.")
```

- [ ] **Step 3: Execute the cell**

  Expected output (values will vary):
  ```
  error_df : 698 hourly rows, 2026-05-05 07:00:00 → 2026-06-03 23:00:00
  daily_df : 30 days
  EV hours : 42
  Overall MAE: 0.432 kWh
  Temp coverage: 698/698 hours
  ```

- [ ] **Step 4: Commit**

```bash
git add notebooks/forecast_performance.ipynb
git commit -m "feat: forecast notebook — data preparation (error_df + daily_df)"
```

---

### Task 5: Chart 1 — Daily MAE trend

**Files:**
- Modify: `notebooks/forecast_performance.ipynb` (add cells 9–10)

- [ ] **Step 1: Insert markdown cell (index 9)**

Content:
```markdown
## 4 · Charts

### Chart 1 — Daily MAE trend

Mean absolute error per day. The dashed red line is a 7-day rolling average.
```

- [ ] **Step 2: Insert chart code cell (index 10)**

Content:
```python
_line_daily = (
    alt.Chart(daily_df)
    .mark_line(point=True, color="steelblue")
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("daily_mae:Q", title="MAE (kWh)", scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("daily_mae:Q", title="MAE (kWh)", format=".3f")],
    )
)

_line_rolling = (
    alt.Chart(daily_df)
    .mark_line(strokeDash=[6, 3], color="crimson")
    .encode(
        x=alt.X("date:T"),
        y=alt.Y("rolling_mae_7d:Q"),
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("rolling_mae_7d:Q", title="7d rolling MAE", format=".3f")],
    )
)

chart1 = (_line_daily + _line_rolling).properties(
    title="Daily MAE — last 30 days  (blue = daily, red dashed = 7-day rolling avg)",
    width=700,
    height=300,
)
chart1
```

- [ ] **Step 3: Execute the cell and verify the chart renders**

  Two lines should appear: a solid blue daily MAE line and a dashed red 7-day rolling average.

- [ ] **Step 4: Commit**

```bash
git add notebooks/forecast_performance.ipynb
git commit -m "feat: forecast notebook — chart 1 daily MAE trend"
```

---

### Task 6: Chart 2 — Daily totals bar

**Files:**
- Modify: `notebooks/forecast_performance.ipynb` (add cells 11–12)

- [ ] **Step 1: Insert markdown cell (index 11)**

Content:
```markdown
### Chart 2 — Daily totals: Predicted vs Actual

Side-by-side bars of summed predicted and actual kWh per day.
```

- [ ] **Step 2: Insert chart code cell (index 12)**

Content:
```python
_daily_long = daily_df.melt(
    id_vars=["date"],
    value_vars=["daily_pred", "daily_actual"],
    var_name="series",
    value_name="kwh",
)
_daily_long["series"] = _daily_long["series"].map({"daily_pred": "Predicted", "daily_actual": "Actual"})

chart2 = (
    alt.Chart(_daily_long)
    .mark_bar()
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("kwh:Q", title="kWh"),
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=["Predicted", "Actual"], range=["steelblue", "orange"]),
            legend=alt.Legend(title="Series"),
        ),
        xOffset=alt.XOffset("series:N"),
        tooltip=[alt.Tooltip("date:T", title="Date"), "series:N", alt.Tooltip("kwh:Q", format=".2f")],
    )
    .properties(title="Daily totals — Predicted vs Actual", width=700, height=300)
)
chart2
```

- [ ] **Step 3: Execute the cell and verify the chart renders**

  Grouped bars (blue = predicted, orange = actual) per day.

- [ ] **Step 4: Commit**

```bash
git add notebooks/forecast_performance.ipynb
git commit -m "feat: forecast notebook — chart 2 daily totals bar"
```

---

### Task 7: Chart 3 — Predicted vs actual scatter

**Files:**
- Modify: `notebooks/forecast_performance.ipynb` (add cells 13–14)

- [ ] **Step 1: Insert markdown cell (index 13)**

Content:
```markdown
### Chart 3 — Predicted vs Actual (per hour)

Each point is one hour. The dashed diagonal is perfect prediction. Red points are EV charging hours (actual > 7 kWh).
```

- [ ] **Step 2: Insert chart code cell (index 14)**

Content:
```python
_max_kwh = float(max(error_df["actual_kwh"].max(), error_df["pred_kwh"].max())) * 1.05

_scatter = (
    alt.Chart(error_df)
    .mark_point(opacity=0.45, size=30)
    .encode(
        x=alt.X("actual_kwh:Q", title="Actual (kWh)", scale=alt.Scale(domain=[0, _max_kwh])),
        y=alt.Y("pred_kwh:Q", title="Predicted (kWh)", scale=alt.Scale(domain=[0, _max_kwh])),
        color=alt.Color(
            "is_ev:N",
            scale=alt.Scale(domain=[False, True], range=["steelblue", "crimson"]),
            legend=alt.Legend(title="EV hour"),
        ),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Hour"),
            alt.Tooltip("actual_kwh:Q", format=".3f", title="Actual (kWh)"),
            alt.Tooltip("pred_kwh:Q", format=".3f", title="Predicted (kWh)"),
            "is_ev:N",
        ],
    )
)

_ref_line = (
    alt.Chart(pd.DataFrame({"v": [0, _max_kwh]}))
    .mark_line(color="black", strokeDash=[5, 3])
    .encode(x=alt.X("v:Q"), y=alt.Y("v:Q"))
)

chart3 = (_scatter + _ref_line).properties(
    title="Predicted vs Actual (per hour) — red = EV charging hours",
    width=700,
    height=400,
)
chart3
```

- [ ] **Step 3: Execute the cell and verify the chart renders**

  Points should cluster around the diagonal; EV hours (high actual kWh) appear in red.

- [ ] **Step 4: Commit**

```bash
git add notebooks/forecast_performance.ipynb
git commit -m "feat: forecast notebook — chart 3 scatter pred vs actual"
```

---

### Task 8: Chart 4 — Hour-of-day heatmap

**Files:**
- Modify: `notebooks/forecast_performance.ipynb` (add cells 15–16)

- [ ] **Step 1: Insert markdown cell (index 15)**

Content:
```markdown
### Chart 4 — Error heatmap by hour × weekday

Mean absolute error for each hour-of-day / weekday combination. Darker red = higher average error.
```

- [ ] **Step 2: Insert chart code cell (index 16)**

Content:
```python
_heatmap_df = (
    error_df.groupby(["hour", "weekday"], as_index=False)["abs_error"].mean()
)

chart4 = (
    alt.Chart(_heatmap_df)
    .mark_rect()
    .encode(
        x=alt.X("hour:O", title="Hour of day (0–23)"),
        y=alt.Y("weekday:O", sort=WEEKDAY_ORDER, title="Day of week"),
        color=alt.Color(
            "abs_error:Q",
            scale=alt.Scale(scheme="reds"),
            title="Mean abs error (kWh)",
        ),
        tooltip=[
            "hour:O",
            "weekday:N",
            alt.Tooltip("abs_error:Q", format=".3f", title="Mean abs error (kWh)"),
        ],
    )
    .properties(title="Mean absolute error by hour × weekday", width=700, height=220)
)
chart4
```

- [ ] **Step 3: Execute the cell and verify the chart renders**

  A 24 × 7 grid with red intensity indicating error magnitude.

- [ ] **Step 4: Commit**

```bash
git add notebooks/forecast_performance.ipynb
git commit -m "feat: forecast notebook — chart 4 hour-of-day heatmap"
```

---

### Task 9: Chart 5 — Error distribution histogram

**Files:**
- Modify: `notebooks/forecast_performance.ipynb` (add cells 17–18)

- [ ] **Step 1: Insert markdown cell (index 17)**

Content:
```markdown
### Chart 5 — Error distribution

Distribution of `predicted − actual` across all hourly pairs. A distribution centred at 0 means no systematic bias. A right shift means the model over-predicts; left shift means under-prediction.
```

- [ ] **Step 2: Insert chart code cell (index 18)**

Content:
```python
_hist = (
    alt.Chart(error_df)
    .mark_bar()
    .encode(
        x=alt.X("error:Q", bin=alt.Bin(step=0.1), title="Error (predicted − actual, kWh)"),
        y=alt.Y("count()", title="Count"),
        tooltip=["count()"],
    )
)

_zero_rule = (
    alt.Chart(pd.DataFrame({"x": [0.0]}))
    .mark_rule(color="crimson", strokeDash=[5, 3], strokeWidth=2)
    .encode(x=alt.X("x:Q"))
)

chart5 = (_hist + _zero_rule).properties(
    title="Error distribution (predicted − actual)  — red line = zero",
    width=700,
    height=300,
)
chart5
```

- [ ] **Step 3: Execute the cell and verify the chart renders**

  A histogram centred around 0 with a red dashed reference line.

- [ ] **Step 4: Commit**

```bash
git add notebooks/forecast_performance.ipynb
git commit -m "feat: forecast notebook — chart 5 error distribution"
```

---

### Task 10: Chart 6 — Temperature correlation

**Files:**
- Modify: `notebooks/forecast_performance.ipynb` (add cells 19–20)

- [ ] **Step 1: Insert markdown cell (index 19)**

Content:
```markdown
### Chart 6 — Absolute error vs outdoor temperature

Shows whether forecast errors are correlated with temperature. A rising trend on the left (cold) or right (hot) side suggests seasonal bias. Red = EV charging hours.
```

- [ ] **Step 2: Insert chart code cell (index 20)**

Content:
```python
_temp_error_df = error_df.dropna(subset=["temp_c"])

if _temp_error_df.empty:
    print("Chart 6 skipped — no temperature data available (Open-Meteo fetch failed).")
else:
    chart6 = (
        alt.Chart(_temp_error_df)
        .mark_point(opacity=0.4, size=28)
        .encode(
            x=alt.X("temp_c:Q", title="Outdoor temperature (°C)"),
            y=alt.Y("abs_error:Q", title="Absolute error (kWh)", scale=alt.Scale(zero=True)),
            color=alt.Color(
                "is_ev:N",
                scale=alt.Scale(domain=[False, True], range=["steelblue", "crimson"]),
                legend=alt.Legend(title="EV hour"),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Hour"),
                alt.Tooltip("temp_c:Q", format=".1f", title="Temp (°C)"),
                alt.Tooltip("abs_error:Q", format=".3f", title="Abs error (kWh)"),
                "is_ev:N",
            ],
        )
        .properties(
            title="Absolute error vs outdoor temperature — red = EV charging hours",
            width=700,
            height=300,
        )
    )
    chart6
```

- [ ] **Step 3: Execute the cell and verify the chart renders**

  A scatter of hourly abs_error vs temperature; EV hours in red. If `_temp_error_df` is empty, a skip message is printed instead.

- [ ] **Step 4: Final commit**

```bash
git add notebooks/forecast_performance.ipynb
git commit -m "feat: forecast notebook — chart 6 temperature correlation"
```

---

## Self-Review Checklist

- [x] **Spec coverage**: SMB pull ✓, Open-Meteo weather ✓, error_df with all columns ✓, daily_df ✓, all 6 charts ✓, EV flagging ✓, graceful empty/failure handling ✓
- [x] **No placeholders**: all code cells are complete, no TBDs
- [x] **Type consistency**: `error_df`, `daily_df`, `temp_df`, `pred_df`, `actuals_df` — used consistently across tasks; column names match everywhere
- [x] **Altair 6 compatibility**: `xOffset=alt.XOffset(...)` for grouped bars; `mark_rect()` for heatmap — all valid in Altair 6.x
- [x] **EV threshold**: hardcoded 7.0 to match `const.py` as specified
- [x] **Weekday order**: `WEEKDAY_ORDER` defined in config cell, referenced in Chart 4 — consistent
