---
name: project_excluded_ranges
description: excluded_ranges.csv mechanism for hand-configured known-bad training/prediction date ranges — how to apply it live for the 2026-07-19 gPlug/SolarEdge fault
metadata:
  type: project
---

Implemented on `feat/excluded-training-ranges` (2026-07-19) to handle an ongoing hardware fault: the gPlug/SolarEdge integration has been producing known-bad readings since 2026-07-19, and this feature lets an operator hand-exclude the affected date range from both model training and live prediction without a code deploy.

**File:** `excluded_ranges.csv`, hand-edited via Samba, lives in the same directory as `energy_history.csv` (i.e. `self._cache_path.parent` on the live HA instance).

**Format:** CSV columns `start` (required), `end` (required), `reason` (optional). `start`/`end` must be `YYYY-MM-DD` or `YYYY-MM-DD HH:MM` exactly — no other formats, no timezone offsets. A bare-date `end` (no time component) expands to `23:59:59` of that date; an explicit time (including `00:00`) is used exactly as written. Extra columns are ignored. Example:

```csv
start,end,reason
2026-07-19 14:00,2026-07-21 09:30,gPlug/SolarEdge fault
```

**Applying an exclusion immediately:** after editing the CSV, fire the `RELOAD_ENERGY_MODEL` HA event to pick it up without waiting for the next scheduled retrain — the file is re-read on every `_retrain()` and every `_update_sensors()` call, so no restart is needed.

**Implementation:** `ha_data.load_excluded_ranges()` / `ha_data.filter_excluded_ranges()` — both never raise (degrade to no-op on any malformed input), applied in `_retrain()` (training) and `_update_sensors()`'s `recent_actuals` (live prediction lag features), but deliberately NOT applied to `full_actuals`/anomaly-MAE sensors. See [[reference_architecture]] for where these fit in the broader data pipeline.

**Related fix in the same branch:** `model.py`'s physics holdout-cutoff calculation used to derive calendar-day span from `len(energy_df) / 24`, which breaks once an exclusion introduces a real multi-day gap between rows — fixed to use `(max_ts - min_ts).days` directly.
