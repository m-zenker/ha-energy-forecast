---
name: solar-feature-pending
description: Solar correction config deployed and battery direction empirically verified 2026-07-16 — one known transient data-quality gap during pre/post-commissioning transition
metadata:
  type: project
---

**Deployed 2026-07-16, battery direction confirmed 2026-07-16 evening.** SolarEdge hardware (order #25440) was commissioned 2026-07-16. `ha-energy-forecast`'s solar PV + battery target-correction feature (`solar_production_sensor`/`grid_export_sensor`/`battery_charge_sensor`/`battery_discharge_sensor` in `energy_forecast.py:272-275`) was enabled in the live `apps.yaml` (patched directly via Samba, remote backup at `apps.yaml.pre-solar-2026-07-16.bak`) alongside deploying `#89` (physics sensor cache dedup) — bundled into one `scripts/deploy.py` run + AppDaemon restart.

Final mapping, all four confirmed live and correctly directioned:

```yaml
solar_production_sensor:  sensor.solaredge_i1_ac_energy         # lifetime cumulative kWh, confirmed live
grid_export_sensor:       sensor.gplugk_z_eo                    # gPlugK house meter export counterpart to energy_sensor
battery_charge_sensor:    sensor.solaredge_i1_b1_energy_import  # CONFIRMED 2026-07-16 evening — see below
battery_discharge_sensor: sensor.solaredge_i1_b1_energy_export  # CONFIRMED 2026-07-16 evening — see below
```

**Battery direction verification (2026-07-16, ~16:00-18:30 UTC / 18:00-20:30 CEST)**: battery SOE climbed 1.1% → 77.8% during a real charge cycle. `sensor.solaredge_i1_b1_energy_import` climbed in lockstep (0.08 → 11.91 kWh) while `sensor.solaredge_i1_b1_energy_export` stayed flat at 7.594 kWh throughout — `import`=charge, `export`=discharge, exactly as mapped. No further verification needed.

**Known transient data-quality gap (2026-07-16, one-time, will not recur)**: `_apply_target_correction`'s "missing correction data = zero flow" assumption (`energy_forecast.py:2969`) broke during the pre-commissioning → post-commissioning transition window. `grid_export_sensor` (`gplugk_z_eo`, pre-existing meter) had real non-zero hourly diffs (8.7-10.4 kWh/hour) for 08:00-11:00 UTC today, while `solar_production_sensor`/battery sensors didn't exist in HA yet (SolarEdge entities weren't created until ~11:12 UTC) — so those hours' corrections went strongly negative (real export with no offsetting production/battery term) and got `clip(lower=0)`'d to exactly 0.000 kWh, zeroing 4 hours of real training data. Confirmed via manual replication of `ha_data.py`'s exact hourly-resample-and-diff logic against live `/api/history` data. From 12:00 UTC onward (all four sensors coexisting) the correction produced plausible small values (0.46, 0.27 kWh) — no ongoing issue.

Later the same evening (~15:42-18:28 UTC), `gplugk_z_eo` went flat for ~3h while the battery charged heavily (SOE 1%→78%) and grid import rose — investigated and concluded this is real physics (solar tapering toward dusk, all surplus absorbed by battery charging + house load, nothing left to export, so grid import picks up the shortfall), not a stuck sensor or a repeat of the earlier gap. No corrective action taken.

**Why:** The zeroed 08:00-11:00 window is a one-time artifact of today's specific commissioning timing — the entities now all exist permanently, so this exact failure mode (a correction sensor with real flow but a sibling sensor with no entity history yet) cannot recur for this particular sensor set. Not worth surgically repairing those 4 hours: `weight_halflife_days: 60` dilutes the impact of any single day's data over time, and the underlying design gap (any correction sensor going stale/unavailable gets silently treated as "no flow", biasing the target rather than erroring) is a standing, generic risk for this feature — worth knowing about if training-target quality is ever investigated later, but not an active bug today.

**How to apply:** No further action needed on the mapping — it's final and verified. If forecast accuracy looks off in a way traceable to a specific day, check whether any of the four correction sensors had an availability gap that day before assuming a model problem.

**Live-path correction bug found + fixed, same evening (2026-07-16):** hours after the four sensors above went live, `binary_sensor.ha_energy_forecast_unusual_consumption` fired during a real battery charge cycle (`residual_kwh: 5.04` vs `residual_std_kwh: 0.499`, ~10σ). Root cause: `_retrain()` (training path) applies `_apply_target_correction()` to convert raw grid import into true household consumption before training, but `_update_sensors()` (the hourly live path) never did — its `recent_actuals`/`full_actuals` were built from raw grid import only, so every battery charge cycle (real grid draw routed to storage) looked like a consumption spike to both the anomaly detector and the lag features, and forecasts chased it upward. This was a genuine architecture gap (the live path was simply missing a step the training path had), not a config or sensor-mapping problem — no `apps.yaml` change needed. Fixed via `docs/superpowers/plans/2026-07-16-live-target-correction-fix.md`: extracted the correction-sensor-fetch loop out of `_retrain()` into a shared `_fetch_correction_dfs()` (parameterized by fetch function — full 30-day for training, lightweight 2-day for the live path), then wired it into `_update_sensors()` too. Per the plan's self-healing note, once deployed, today's mis-recorded raw actuals get overwritten with corrected values within ~2 days of normal hourly operation (rolling keep-last-wins cache) — no manual `pred_history.json` repair needed.
