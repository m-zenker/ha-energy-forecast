# Diagnosis: Heat Pump Sub-Sensor Feature Integration Issue

**Date**: 2026-04-02
**Issue**: Consumption forecast MAE increase starting 2026-03-18 (~2 weeks ago)
**Root Cause**: Feature integration problem, not cold-start data issue

---

## Executive Summary

The model learned to predict **total household consumption (including heat pump) as a single aggregated signal** over 4000+ hours of history. When the heat pump sub-sensor was explicitly separated on **2026-03-18**, it was added as a **separate input feature** (`sub_hp_lag_24h`, `sub_hp_active_24h`, `sub_hp_runs_7d`, etc.) to the training pipeline.

**This created a feature integration problem:** the model now has access to both:
1. Implicit heat pump consumption embedded in `gross_kwh` 
2. Explicit heat pump consumption as separate lag features

This is **not a multicollinearity issue** (which would cause overfitting, not higher MAE). Rather, it's an **inconsistency in signal representation** that the model must learn to handle, breaking the learned patterns from the pre-split training data.

---

## Timeline

| Date | Event |
|------|-------|
| ~2026-01-01 | Heat pump consumption already in total (implicit, mixed signal) |
| 2026-03-18 13:10:37 | Sub-sensor feature introduced (`feat: add sub_energy_sensors`) |
| 2026-03-19 07:14:19 | Sub-sensor feature merged to main |
| 2026-03-22 onwards | Deployed to dev and live |
| ~2026-03-18 onwards | **MAE degradation observed** |

---

## Current Integration Architecture

### How Sub-Sensors Are Used (NOT Subtracted)

```
gross_kwh (training target) = grid_import (unchanged, includes heat pump)
                           - EV charger (explicitly subtracted, ~9 kW)
                           ± solar/battery corrections (if configured)

sub_hp_lag_24h, sub_hp_active_24h, sub_hp_runs_7d = additional input features
```

**Key finding:** Sub-sensors are **feature inputs only**. They are **NOT subtracted** from the training target.

Evidence from code:
- `ha_data.py::split_ev_charging()` subtracts EV load from `gross_kwh` before training
- No equivalent function exists for sub-sensors — they remain as separate features
- `model.py::_add_sub_sensor_lags_training()` creates lag columns (lines 1073–1088)
- A warning is logged if >50% NaN (line 1066), but columns are still created and filled with 0 (lines 298–300)

### What Happens During Training

1. **Before 2026-03-18**: Model trained on 4000h of data where heat pump consumption is **embedded in gross_kwh**
   - No explicit heat pump features exist
   - Model learns: "consumption at hour H depends on weather + patterns + recent consumption"
   - Recent consumption (`lag_24h`) implicitly includes heat pump behavior

2. **After 2026-03-18**: Model sees **both**:
   - `gross_kwh` (still includes heat pump)
   - `sub_hp_lag_24h` (24h-lagged heat pump consumption)
   - `sub_hp_active_24h` (binary: was heat pump active in last 24h?)
   - `sub_hp_runs_7d` (count of heat pump start events in past 7d)

3. **The problem**: The model must now disentangle:
   - "How much of recent consumption was heat pump?" (from `lag_24h`)
   - "How much of recent consumption was **everything else**?" (from `sub_hp_lag_24h`)
   - But `gross_kwh` conflates both!

---

## Root Cause Analysis

### Hypothesis 1: Feature Integration Problem ✅ LIKELY

**Problem**: The model learned on a conflated signal, then suddenly had to work with a split signal.

**Impact**:
- Lag features that previously captured household baseline + heat pump patterns must now isolate household baseline
- The relationship between `lag_24h` (mixed) and `sub_hp_lag_24h` (heat pump only) is **novel** to the model
- Feature importance weights learned on the old signal don't apply to the new signal

**Example**:
```
Old: "lag_24h = 3.5 kWh means expect consumption ≈ 2.2 kWh (learned relationship)"
     (but lag_24h included 0–2 kWh heat pump)

New: "lag_24h = 3.5 kWh, sub_hp_lag_24h = 1.8 kWh means expect consumption ≈ ???"
     (model must re-learn the split relationship)
```

### Hypothesis 2: Sparse Sub-Sensor Data 

**Evidence**: Sub-sensor added only 2 weeks ago → limited history
- `sub_hp_lag_24h` filled with 0 for hours when heat pump was off
- `sub_hp_active_24h` and `sub_hp_runs_7d` estimated from ~14 days of real data
- Training medians (used to fill NaN at predict time) may be unstable

**However**: The code has defensive measures:
- `_add_sub_sensor_lags_training()` fills NaN with 0 (safe: means appliance off)
- Sub-sensor columns are **not dropped** if >50% NaN — a warning is logged, but columns are created anyway
- NaN values in sub-sensor columns are filled with 0 before training dropna() (lines 298–300), so sparse data doesn't break training

### Hypothesis 3: Model Not Being Retrained

**Unlikely**: Memory indicates adaptive retraining is enabled and working. Weekly full retrains should apply new data.

**Check needed**: Did model retrain after 2026-03-22 when sub-sensor became live?

---

## Clarifying Questions (Critical Path)

### 1. **Did accuracy degrade visibly after 2026-03-18?**
   - Check prediction logs or metrics dashboard
   - Look for a "step change" in MAE around that date
   - If MAE is smooth degradation (gradual), suggests hypothesis #2 (sparse data)
   - If MAE jumped on 2026-03-19, suggests hypothesis #1 (feature integration)

### 2. **Is the sub-sensor configured in production?**
   - Check `apps.yaml` on the HA instance
   - If `sub_energy_sensors` is NOT configured, the drift is unrelated to sub-sensors
   - If configured, was it enabled exactly on 2026-03-18?

### 3. **Did the model retrain after the sub-sensor was deployed?**
   - Check `energy_forecast.py` logs for "Retrained" messages after 2026-03-22
   - Count retrains before vs after 2026-03-18
   - If no retrains occurred between 2026-03-22 and 2026-04-02, the model is still using old weights

### 4. **How is heat pump consumption being measured?**
   - Is it a **cumulative kWh meter** (state_class: total_increasing)?
   - Or a **power sensor** (W) integrated in HA?
   - If power → W integrated, resolution and timing matter for lag feature accuracy

### 5. **What does `sub_hp_lag_24h` actually contain?**
   - 24-hour-old heat pump consumption (kWh)
   - Or 24-hour-old power reading (W) incorrectly scaled?
   - Check a sample prediction log entry

---

## Diagnosis Path (Step-by-Step)

### Step 1: Verify Sub-Sensor is Actually Configured
```bash
grep -A5 "sub_energy_sensors" /homeassistant/appdaemon/apps/apps.yaml
```
Expected output:
```yaml
sub_energy_sensors:
  - sensor.heat_pump_energy_kwh
```

### Step 2: Check for MAE Step-Change Around 2026-03-18
Look at logs or a MAE timeseries chart:
- Is there a visible discontinuity on 2026-03-19?
- Or gradual drift starting 2026-03-18 and leveling off?

**If step-change on 2026-03-19**: Feature integration problem (hypothesis #1)
**If gradual drift**: Sparse data warm-up (hypothesis #2)

### Step 3: Inspect Recent Retrains
```bash
grep "Retrained. MAE:" /var/log/appdaemon.log | tail -20
```
Look for dates and MAE values. Expected:
- At least one retrain between 2026-03-22 and 2026-04-02
- MAE improving if sparse data is the issue (more heat pump history = better estimates)

### Step 4: Check Sub-Sensor Data Completeness
```python
import pandas as pd
df = pd.read_csv("energy_history.csv")  # main cache
sub_df = pd.read_csv("energy_forecast_sub_hp_energy_kwh.csv")  # sub-sensor cache
print(f"Main history rows: {len(df)}")
print(f"Sub-sensor history rows: {len(sub_df)}")
print(f"Sub-sensor coverage: {len(sub_df) / len(df) * 100:.1f}%")
print(f"Sub-sensor date range: {sub_df['timestamp'].min()} to {sub_df['timestamp'].max()}")
```

If coverage < 70%, sparse data is the issue.

### Step 5: Manual Prediction Inspection
Collect one recent prediction + actual pair:
```
Predicted: 2.5 kWh
Actual: 3.2 kWh
Error: +0.7 kWh (+28%)
Recent features used:
  - lag_24h: 3.1 kWh
  - sub_hp_lag_24h: 0.8 kWh
  - sub_hp_active_24h: 1
  - sub_hp_runs_7d: 2
```

Ask: Does the model know how to combine these new signals?

---

## Solution Path (Once Root Cause Confirmed)

### If Hypothesis #1 (Feature Integration):
**Solution**: Force full retraining to let the model learn the split signal

```bash
# In HA Developer Tools → Services → energy_forecast.perform_retrain
# Or via AppDaemon console:
python
import datetime
self.energy_forecast._retrain_cb(datetime.datetime.now())
```

Expected outcome:
- New model trained with ~2 weeks of split-signal history
- MAE should improve by 5–15% as model learns the new relationships

### If Hypothesis #2 (Sparse Sub-Sensor Data):
**Solution**: Wait + monitor. Sub-sensor features need ~4 weeks of history to be stable

- After ~4 weeks (2026-04-15), recheck MAE
- If improving steadily, no action needed
- If flat or degrading, then pursue hypothesis #1 (force retrain)

### If Hypothesis #3 (Model Not Retraining):
**Solution**: Check AppDaemon logs for retraining callback failures

```bash
grep -i "retrain\|error\|exception" /var/log/appdaemon.log | tail -100
```

---

## Code Locations (Reference)

| File | Function | Line | Purpose |
|------|----------|------|---------|
| `energy_forecast.py` | `_sub_sensor_prefix()` | 242 | Extract prefix from entity_id |
| `energy_forecast.py` | `_retrain_cb()` | ~550 | Manual retrain entry point |
| `ha_data.py` | `fetch_sub_sensor_history()` | 302 | Load sub-sensor from cache + HA DB |
| `ha_data.py` | `fetch_recent_sub_sensor()` | 355 | Quick fetch for lag features |
| `model.py` | `_add_sub_sensor_lags_training()` | 1010 | Create lag/activity/run features during training |
| `model.py` | `_add_sub_sensor_lags_prediction()` | 1072 | Fill lag features during prediction |

---

## Expected Behavior (Correct Integration)

When configured correctly:

1. **Training phase** (weekly or on-demand):
   - Sub-sensor history loaded (~2 weeks to present)
   - Lag columns added: `sub_hp_lag_24h`, `sub_hp_lag_168h` (if enough rows)
   - Activity columns added: `sub_hp_active_24h`, `sub_hp_runs_7d`
   - Model trains: learns relationship between mixed `gross_kwh` and split features
   - Old weights adjust to account for new signal

2. **Prediction phase** (hourly):
   - Sub-sensor recent actuals fetched (~24h window)
   - Lag features populated for next 48 hours
   - Model predicts using updated features
   - MAE gradually improves as sub-sensor history accumulates

---

## Monitoring Checklist

- [ ] Confirm sub-sensor is configured in `apps.yaml`
- [ ] Check if MAE degraded sharply on 2026-03-19 (feature problem) or gradually (data problem)
- [ ] Inspect retraining logs for success/failure after 2026-03-22
- [ ] Check sub-sensor cache for data completeness (>70% coverage)
- [ ] Manual prediction inspection: do feature values make sense?
- [ ] If confident in hypothesis #1: force full retrain and re-measure MAE

---

## References

- CLAUDE.md: Feature engineering architecture
- MEMORY.md: Sub-sensor testing coverage (7 tests in test_model.py)
- CHANGELOG.md v0.4.0–v0.5.1: Sub-sensor feature rollout details
- Code: `model.py` lines 1010–1142 (sub-sensor lag functions)

