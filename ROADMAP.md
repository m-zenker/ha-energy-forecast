# Forecast Accuracy Roadmap

Current: **dev v0.12.0-alpha-39** (2026-11-13, deployed, 1463 tests) · **main v0.11.18** (2026-10-26).

---

## Current Status

**dev:** v0.12.0-alpha-39 (2026-11-13): `get_scenario` accepts dict schedule values + returns `delta_kwh_hi` (upper-quartile appliance profile, `fix/scenario-hi-profile-program`); alpha-38 (2026-11-11): pre-train snapshot for capture replay (`fix/capture-pre-train-seed`); alpha-37 (2026-11-08): backtest kit, cop-scale analysis, live input capture; alpha-36 (2026-11-07): `set_dhw_schedule` keeps the scenario cache; alpha-35 (2026-11-05): #110 soft OOF regime labels; alpha-34 (2026-11-04, deploy 16:11Z, startup retrain 16:14Z): DHW override past-tail fix + today-block P10/P90 blend, on top of alpha-33 (2026-11-03): #110 Part A regime stability log, and alpha-32 (2026-10-27, A+B: #101 reset-aware increments + #102 EV sensor trust), alpha-31 (2026-10-25, DST ambiguous-floor fix) and alpha-30/29 (DST duplicate hour, #109 heating-buffer smoothing). **main:** v0.11.18 released 2026-10-26 (DST ambiguous-floor fix only; v0.11.17 DST duplicate hour 2026-10-22). Rule-3 curated releases off main; dev carries the unreleased physics-ML / UA_eff / DHW-override / heating-season / cold-onset work.

**Cold-onset over-forecast (2026-10-15): C verdict PASSED 2026-10-27 (pooled); A+B re-baseline FAILED the bias criterion 2026-11-03 — see the re-baseline paragraph below.** `mae_7d` jumped 0.196 → 0.477 after heating onset on 2026-10-08. Root cause: the 48 h indoor projection had no heating term and compared every room to one shared setpoint. **C Task 1** (alpha-26, 2026-10-16): thermostat floor + per-room setpoint; replay 2026-10-08..14 MAE 24–47 h 0.485 → 0.255. **D Task 1** (alpha-25, 2026-10-16): `heating_power_request` valve demand cached in `climate_*.csv`; the ≥ 14 valve-closed-night data gate was met 2026-10-21.
- **C check 2026-10-21: NOT MET.** 10-17 −2.2 (pass), 10-18 +6.2 and 10-19 +4.7 kWh/day (fail); pooled ≈ +3.5 vs +5.9 before C; point-in-band 87.5 % (< 90 %). Night hours (22–03 h) over-forecast ≈ 0.3–0.5 kWh/h.
- **Count re-anchored to 2026-10-20 22:33 local** (retrain with #109 smoothing), so the verdict now measures C + smoothing. Earliest recheck ~10-25 (target days 10-22..24).
- Success criterion: cold-day 24–47 h bias in [−3, +1] kWh/day over ≥ 3 cold days, point forecast inside p10–p90 for ≥ 90 % of 3 h blocks. Everything after it is ordered under **Cold-onset sequencing** below. Plans: `docs/superpowers/plans/2026-10-15-cold-onset-forecast-fixes-index.md`; findings: `memory/project_cold_onset_overforecast_2026_10_15.md`.

**C verdict (2026-10-27): PASS on the pooled criterion (accepted by the user).** Anchor 2026-10-20 22:33 (C + #109 smoothing). Pooled post cold-day bias −0.19 kWh/day (101 h; pre +4.45); point inside p10–p90 51/56 = 91.1 %. The per-day reading would fail (2/5 complete days pass), from daytime noise in both directions. Pooled stays the criterion for the A+B re-baseline.

**A+B deployed (2026-10-27, alpha-32); re-baseline FAILED on bias (2026-11-03).** Anchor = forced retrain 2026-10-27 05:40. Pooled cold-day bias post +2.79 kWh/day over 117 h (pre −0.26), criterion [−3, +1] → FAIL; bands 37/40 = 92.5 % → PASS. Over-forecast on 6 of 7 post days (+1.8 … +9.3 kWh/day; 11-01 is partly EV-masked). The three complete non-EV days (10-29, 10-30, 11-02) alone: +2.4 kWh/day, spread across the day with peaks at night (22–01 h, 05–06 h). The model's 08 h forecast jumped from 0–0.9 to 1.6–2.5 kWh since 10-28 while actuals scatter (0.1–1.8). `cv_MAE` 0.3052 vs 0.2924 at the 10-20 retrain (not like-for-like: window and target changed). A raises actuals by ≈ +0.5 kWh/day, so A+B is not the cause. Leading unconfirmed hypothesis: #110 regime flip into the 20.9 kWh/day regime on the 5–8 °C days (predicted daily totals 21.7 / 23.6 / 22.1 on 10-31..11-02). Alternatives: newly learned 08 h peak, real cold-onset load. Rechecks: 2026-11-06 +2.43, 11-08 +2.54, 11-12 +1.89 kWh/day (still FAIL; alpha-35 cold days −1.79, beats naive); #110 soft labels shipped alpha-35 (2026-11-05); next rescore at ≥ 5 alpha-35 cold days. **#105 deploy held** until the re-baseline passes.

**DST fall-back:** resolved. Duplicate-hour fix (alpha-30 / v0.11.17) and ambiguous-floor fix (alpha-31 / v0.11.18) are live; logs clean after the 2026-10-25 fall-back. See `memory/project_dst_fallback_duplicate_hour.md`.

**Heating-season projection (#98, 2026-09-26):** superseded as an open check by the cold-onset work. The mild-weather recheck was met 2026-10-06 (`mae_7d` 0.196); the cold-weather test failed 2026-10-15 and is now covered by the C verdict. Remaining minors: #99.

**DHW override past-tail fix (2026-11-04, `fix/override-delta-past-tail`, deployed in v0.12.0-alpha-34 on 2026-11-04 together with the today-block interval blend fix):** the post-model override correction simulated over `forecast_df`'s past tail (`past_days=3`), so finished overrides shifted future hours by about ±1 kWh, floored at 0 (since v0.12.0-alpha-13). It now covers only hours from the first predicted hour on. Estimated on 2026-10-22..11-03 (simulated, lead 25 h): hourly MAE 0.500 → 0.463 (+7.3 %), 3 h blocks 0.888 → 0.822, daily error worse 2.46 → 2.77; full 30 days only +1.8 % hourly. Not the cause of the failed re-baseline (~0.24 kWh/day; 0 of 7 forecasts at 08 h affected). Tradeoff: a finished override's modelled tail is dropped. Bug is dormant without a recent override, so verification waits for the next override to expire. Not fixed here: `predict_series` and `predict_scenario` also simulate over the tail (no forecast effect at 0 `physics_kwh` gain; matters for Phase 2), and `physics_base_today` can still show negative values while an override is pending. Plan: `docs/superpowers/plans/2026-11-04-override-delta-past-tail-fix.md`; findings: `memory/project_forecast_gap_2026_11_04.md`.

**Forecast-inaccuracy investigation (2026-11-04):** target of 20 % better hourly MAE NOT reached; best candidate +14.1 % vs live on a 13-day holdout that had already been looked at once. Most hourly error is timing inside 3 h windows: perfect daily totals barely help (−4/−3/+3 % winter/Sep/Oct), perfect 3 h totals give −21/−33/−18 %. Follow-ups are #111–#115 below. **The 11-06 re-baseline recheck now spans the alpha-34 boundary:** the 11-04 retrain moves P50, so compare with the anchor in mind.

**Forecast improvement audit (2026-11-13, report `docs/analysis/2026-11-13-forecast-improvement-audit.md`):** on hourly MAE the current code is at its day-ahead noise floor (0.320 kWh/h replayed on 10-01..11-03; perfect daily totals with the usual hourly shape give 0.326). Seven model-structure arms were inside the noise floor or worse. The yardstick is the larger problem: the stored MAE/bias actuals include appliance energy the model excludes (0.79 kWh/day, #117), and hourly MAE rewards under-forecasting (#114). Heat-pump cycle timing has no signal from 3 h lead; the DHW hour is known to `ha-energy-manager` only a few hours ahead, so day-ahead timing-aware DHW/heat-pump modelling is not pursued. Follow-ups: #117–#123, order under **Audit sequencing** below.

**Regime label (#110, new 2026-10-21; Part A log deployed alpha-33 2026-11-03 → `regime_log.jsonl` next to `pred_history.json`):** `regime_kwh` is the top feature but is partly a train/serve leak, and its hard label can flip on a weather-source switch (+5.6 kWh tomorrow spike on 10-21). Needs a backtest before any plan. This also answers the old "SHAP check for #82": `regime_kwh` ranks #1, so #83's value depends on #110.

**Physics Phase 2 interval-coverage check (dormant):** once `use_physics_residual: true` has been live ≥ 30 days, verify empirical interval coverage on gross kWh matches the target (80 %, `_calibrate_intervals()`). Phase 2 stays behind the cold-start gate until ≥ 30 winter UA_eff windows (not before winter 2026/27; see #105). Ref: `docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md` §5.2. **2026-11-13 audit:** Phase 2 cannot be enabled as coded (#113: the reconstruction fails on every live-shaped frame). Even with that corrected it was worse in autumn (10-01..11-03 hourly +13.8 %, cold-day bias +4.6 kWh/day) and better only in winter (3 h blocks −3.1 %, bias ≈ 0), so activation needs a per-season check on the capture frame (#119), not just the window count.

---

## Design Decisions

| Topic | Decision |
|---|---|
| Primary goal | **Forecast accuracy** + visibility/dashboards |
| Primary accuracy metric | **3 h-block MAE and daily bias**; hourly MAE is secondary (2026-11-13; hourly MAE rewards under-forecasting, and `ha-energy-manager` consumes 3 h blocks) |
| Point forecast | **Median-like** (current log-space fit), decided 2026-11-13. Known cost: summed hourly values under-count energy by about 5–7 % (winter replay bias −2.4 kWh/day) |
| Cold-day bias gate | Pooled cold-day bias within **−12 % to +6 % of mean daily consumption over ≥ 14 cold days**, on appliance-free actuals; fewer days = inconclusive (2026-11-13, replaces [−3, +1] kWh/day; #118) |
| Solar PV | Live since 2026-07-16 (SolarEdge) — target correction active (v0.8.0, live-path fix 2026-07-16); solar forecast feature out of scope |
| Home battery | Live since 2026-07-16; charge/discharge in target correction (direction verified). SoC as feature unblocked but deferred until residuals show SoC correlation (#40) |
| Tariff | Fixed flat rate — price optimisation **out of scope** |
| Load shifting | **Out of scope** — handled by a separate system |
| Audience | Personal-first; HACS nice-to-have, never at cost of accuracy |

> **Critical definition:** *Consumption* = `grid_import − grid_export + solar_production − battery_charge + battery_discharge`. Not net load, not grid-only import.

---

## Deployment Workflow

1. Feature branch → implement + tests pass (`python -m pytest tests/ -v`)
2. PR → code review → merge to `dev`
3. Smoke-test on local HA instance (watch AppDaemon log; confirm sensors update)
4. Stable period on `dev` → merge to `main`
5. Update CHANGELOG.md, create semver tag, push → Forgejo release

---

## Backlog

### #111 — Horizon-Consistent Model: Shadow Trial

**Priority:** Parked (2026-11-13). **Status:** simulated, not implemented.

**Parked 2026-11-13:** the audit found current code at the hourly noise floor, and the earlier gain was against a buggy live forecast (about 0 % against replayed current code in winter). Reopen only if it wins on 3 h-block metrics on the capture frame (#119).

Train with features computed only from data available at the forecast origin (each row gets a lead `L`; `hod_med7`, same-hour lags, trailing means, heat/DHW sub-meter history), instead of today's 1 h lags served at 24 h+ (`lag_1h`, `rolling_std_24h` = 0, `<sub>_active_24h` = 0). Candidate E5 + trailing 7-day level correction ("k7"): LightGBM, log1p, 120-day half-life, leads 24–47 h.

| Simulation (lead ≈ 25 h, Open-Meteo day-ahead weather) | Hourly MAE | 3 h blocks | Daily error | Daily bias |
|---|---|---|---|---|
| Holdout 10-22..11-03: live | 0.500 | 0.888 | 2.46 | +0.77 |
| Holdout: current code, replayed | 0.448 | 0.775 | 2.32 | −0.80 |
| Holdout: E5 alone (the one clean holdout test; FAILED the bias guardrail) | 0.425 | 0.727 | 3.04 | −3.04 |
| Holdout: E5 + k7 (second look, not clean) | 0.429 | 0.712 | 2.15 | −0.81 |

E5 + k7 vs live: hourly +14.1 % (95 % CI +8.8..+19.5), 12 of 13 days better. **Against the replayed current model, gains shrink or vanish:** E5 alone −0.7 % winter, +17.5 % Sep, +6.7 % Oct; with k7 −1.1 % / +9.1 % / −1.5 %. So most of the win over live may not survive winter.

**Next:** 2–3 week shadow run in cold weather, on SRG inputs and all leads, logged next to the live forecast, before any switch-over. Short leads (0–23 h) were not scored.

### #112 — Live vs Replay Gap: Remaining Causes

**Priority:** — . **Status:** CLOSED 2026-11-13.

**Closed 2026-11-13 (audit, `docs/analysis/2026-11-13-forecast-improvement-audit.md`):** the gap is an input difference. Live captures reproduce live exactly, including three retrains seeded from `.pre/`. Live SRG weather against perfect weather costs 1–2 %. The exact-0.0 forecasts are the floored negative leg of a pending DHW override (#122). The frozen weather tail corrupts 168 h look-backs once the model is older than 3 days, a small effect (#123). The 08 h over-forecast is about 60 % reproduced by current code (replay 1.28, live 1.67, actual 0.81 over 7 mornings), so it is mostly model-side.

The same code replayed offline beats live: 0.359 vs 0.463 (686 h); 0.448 vs 0.500 on the holdout (+10.4 %). The override past-tail bug (fixed alpha-34) explains about half of the holdout gap. Unmeasured: SRG vs Open-Meteo day-ahead weather (perfect vs day-ahead weather moves replay only 0.352 → 0.359, but SRG has no stored history), and retrain timing. Also unexplained: 12 of 27 exact-0.0 live forecasts; the model's 08 h forecast of 1.67 vs 0.81 actual since 10-28 (0 of 7 affected by the override bug).

**Next:** recheck the gap after alpha-34; store each run's SRG frame so the weather effect can be measured.

### #113 — Physics Past Tail in `predict_series` / `predict_scenario`

**Priority:** **Blocker for Phase 2** (raised 2026-11-13). **Status:** confirmed by replay; not fixed.

**Added 2026-11-13:** `predict()` adds a physics baseline indexed on the whole forecast frame to the 48 predictions (`model.py:1395-1405`). With any frame that is not 48 rows it raises `operands could not be broadcast together with shapes (170,) (48,)`, falls back to "ML-only" and publishes the bare residual: 26.4 instead of 36.4 kWh over 48 h in the replay. `predict_intervals()` never adds the baseline. Unit tests pass because they use 48-row frames. Fix both and add a regression test with a 170-row frame before `use_physics_residual` is turned on anywhere.

Both still simulate over `forecast_df`'s past tail, seeded with the latest tank reading (`predict_series`), or with `committed_override` passed in (`predict_scenario`, `model.py:1501`). No forecast effect while `physics_kwh` has 0 gain (not measured for scenarios). `physics_base_today` also stays unfloored and can show negative values while an override is pending (23 of 24 simulated start hours). Fix before `use_physics_residual` or EM scenario scoring relies on it.

### #114 — Report Block and Daily Accuracy, Not Just Hourly MAE

**Priority:** **High** (raised 2026-11-13; do right after #117). **Status:** proposed.

**Extended 2026-11-13:** also publish the naive same-hour mean (days −2..−8) MAE as a guardrail, and add the block, daily and naive columns to `backtest/score.py`. Evidence: scaling the replayed forecast by 0.8 improves hourly MAE by 6 % while bias goes to −2.5 kWh/day; daily error is best at scale 1.0. The naive mean ties current code on hourly MAE and is 10–16 % worse on 3 h blocks.

Hourly MAE is mostly timing noise, and optimising it rewards median-style forecasts: an L1 arm gained +22 % in September while reaching −3.5 kWh/day bias in October; a 7-day same-hour median scored 0.356 vs live 0.475 with about −4 kWh/day. Battery scheduling depends on totals. Add 3 h block MAE and daily error/bias sensors, and judge model changes on those with hourly MAE as a secondary number.

### #115 — Trailing Level Correction on the Current Model

**Priority:** — . **Status:** CLOSED 2026-11-13.

**Closed 2026-11-13:** the point forecast stays median-like by owner decision, so the level is not corrected.

Scaling the replayed current model by the trailing 7-day actual/forecast ratio (clipped 0.7–1.3, causal) mostly removes daily bias (winter −2.23 → −0.18, Oct −0.96 → +0.05; holdout −0.80 → −0.05) but costs hourly MAE (winter −1.7 %, Sep −0.6 %, Oct −5.9 %) and doesn't reliably reduce daily error (Sep 1.74 → 1.90, Oct 2.37 → 2.56). Consider only if bias, not hourly error, becomes the priority.

### #116 — Retrain Age Off by the UTC Offset (`last_trained` Is Local, Read as UTC)

**Priority:** Low. **Status:** confirmed by code reading and the live log (2026-11-09); not fixed.

**Note 2026-11-13:** in the audit's walk-forward, daily against weekly retraining made no accuracy difference on the dev windows (hourly −1.2 %, CI spans 0). Weekly had smaller next-day swings (0.88 against 1.41 kWh) but a worse 6 h-block result on the 9-day holdout (+15.6 %). No change to the cadence is planned.

`model.py:961` stamps `last_trained` with naive `datetime.now()`, which is local time in the add-on container. `_last_trained_local()` (`energy_forecast.py:1145-1160`) treats that stamp as UTC and converts it to local time (`.tz_localize("UTC").tz_convert(tz)`), which applies the offset a second time. The computed age therefore reads 1 h short in CET and 2 h short in CEST. Live log 2026-11-08 21:26: "Startup retrain skipped — last retrain was -0.9h ago" (true age ≈ 0.15 h). Introduced in 1010a9f (2026-07-09).

**Effect:** the startup-retrain skip gate (`energy_forecast.py:1187`, < 6 h) and the adaptive-retrain cooldown seed (`:1171`, `:1978`) run 1–2 h late. Never early, so there is no spurious retrain and no crash. `model.py:1685` compares naive with naive and is consistent.

**Fix options:** store `last_trained` as tz-aware UTC (needs a load-time shim for existing `meta.pkl` stamps), or have `_last_trained_local()` treat a naive stamp as system-local. Add a regression test that pins both CET and CEST. Size S; tag BEHAVIOUR (retrain timing).

### #117 — Scoring Bug: Stored Actuals Include Appliance Energy

**Priority:** High. **Status:** confirmed 2026-11-13; not fixed. Size S; tag BEHAVIOUR (`mae_*` and bias readings step once).

With `baseline_mode: true` the model trains on consumption minus the dishwasher, washer and tumbler sub-meters, but `_actuals_history` is filled from `full_actuals` before that subtraction (`energy_forecast.py:2306-2326`; the comment there says they match). The difference correlates 0.995 with the appliance sub-meters on 771 of 772 hours and is 0.79 kWh/day. Hourly MAE barely moves (0.474 → 0.468), but every bias reading is about 0.8 kWh/day too low: live bias over 32 days is +2.26 against the true target, not +1.47. The naive comparisons made on `pred_history` actuals carry the same error.

**Fix:** subtract the `baseline_mode` sub-sensors before storing actuals; test with a day that contains an appliance run. Re-read the A+B gate afterwards (#118).

### #118 — Replace the Cold-Day Bias Gate

**Priority:** High. **Status:** decided 2026-11-13; gate script not updated.

New gate: pooled cold-day bias within −12 % to +6 % of mean daily consumption, over at least 14 cold days, on appliance-free actuals (#117). Fewer days is INCONCLUSIVE. It replaces [−3, +1] kWh/day, which was shaped around an expected −2 kWh/day residual from fix C. Reasons: load runs from about 16 kWh/day in autumn to over 30 in winter; a median-like forecast is expected to sit 5–7 % low; daily error is about 2 kWh, so a 7-day mean has a standard error near 0.75 kWh/day, and the retrain jitter alone moves daily |bias| by up to 13 %. Current code replayed: +0.49 kWh/day on 16 cold days in 10-01..11-03 (about +3 %); +1.76 on 7 cold days in 11-04..11-12 (about +9 %, inconclusive).

**Work:** update `scripts/check_cold_onset_gate.py` (local) and the sequencing below; the bias sensors from #114 should report the relative value.

### #119 — Backtest Capture-Frame Mode

**Priority:** High (enabler for #105, #120 and any feature touching climate, tank or sub-meter inputs). **Status:** scratch version built and validated in the 2026-11-13 audit (`scripts/forecast_audit_2026_11_13/`, local); not in `backtest/`.

Walk-forward on a retrain capture: truncate the captured live inputs at each cutoff, synthesize the serve inputs, run the real `train()` / `predict()` / `predict_intervals()`. Validated against live: serve inputs within 0.007 kWh/h, retrain at the capture time identical. Scores 3 h-block and daily error, cold-day bias, interval coverage, next-day swing and a jitter arm (same model retrained 1 h earlier) as the noise floor: ±1 % hourly, ±2 % on 3 h blocks, up to ±13 % on daily |bias|. Replaces plan 2 of the parked live-equivalent-frame spec (`docs/superpowers/specs/2026-11-11-live-equivalent-frame-105-gate-design.md`). Needs `/home/jovyan/my_envs/ha-energy-forecast-live`. Also: `scripts/pull_ha_data.py` must pull `.pre/`.

### #120 — Drop the Thermal-Pressure Feature Family

**Priority:** Medium. **Status:** simulated 2026-11-13; not implemented. Needs #119.

`thermal_pressure`, `_max`, `_std`, `_cop`, `_net` and `infiltration_pressure`: the served values have about zero correlation with the values later trained on (−0.06 at 24–47 h, −0.03 at 1–5 h), and a 0.4 K projected room deficit moved the 11-14 forecast between 25.0 and 31.4 kWh. Dropping them: hourly +0.2 % (CI −1.0..+1.3) and 3 h −0.4 % pooled over dev; holdout hourly −0.8 %, cold-day bias +0.95 against +1.76; spread of the next-day total across the previous day's four forecasts 1.25 → 0.57 kWh (Sep), 1.41 → 1.33 (Oct), 2.44 → 1.39 (holdout). Not testable on last winter (no climate data then).

**Plan:** `train()` flag, default off; shadow ≥ 14 days. Success: hourly and 3 h MAE non-inferior at 2 %, swing at least 25 % lower, cold-day bias not higher. Second arm: also drop `heating_buffer_temp` (served value correlates −0.68 with the trained one at day-ahead lead; the NB fallback from #109). If this ships, #108 is moot.

### #121 — Interval Calibration on Served Errors

**Priority:** Medium-low. **Status:** hourly part simulated 2026-11-13; daily band has no method yet.

Hourly P10–P90 covers 59 % on live captures and 69–76 % in the autumn replay (80 % in winter). The conformal correction is 0.009 in log space because the calibration split is a random 15 % of training rows; in-sample coverage is 86 %. A trailing correction on served day-ahead errors gives 79–84 % at 3–11 % more width; the upper tail stays under-covered. Block and daily bands are sums of hourly quantiles (`energy_forecast.py:3227-3268`): 3 h blocks cover 85–97 % with P90 about right in autumn and too high in winter; daily bands cover 97–100 % at 19–43 kWh width. Empirical block and daily error quantiles were unreliable (69–84 % and 58–85 %), so the daily band needs a different method before it is changed.

### #122 — Trim the DHW Override Correction to the Override Hour

**Priority:** Medium-low. **Status:** mechanism confirmed 2026-11-13; accuracy cost not measured.

With an override pending, each update adds about +1 kWh at the override hour, a second +1 kWh about 23 h later, and subtracts 0.8–1.05 kWh at a "natural reheat" hour that drifts from update to update. The ML forecast holds no discrete reheat at those hours, so the negative leg floors to an exact 0.0 where the forecast is low (4 of 7 affected updates on 11-13). Since `ha-energy-manager` knows the DHW hour only a few hours ahead, this intra-day correction is the only DHW lever left.

**Next:** after ≥ 5 overrides in the captures, compare published forecasts with model output plus only the override-hour leg. Change only if that is no worse. Related: #93, #113.

### #123 — Training-Data Hygiene

**Priority:** Low. **Status:** confirmed 2026-11-13; expected accuracy effect inside the noise floor.

1. **NaN-lag cascade:** `dropna` on lag columns (`model.py:731`) deletes 1 675 of 9 139 rows (23–27 % of each winter month) because one missing hour blanks up to nine later rows. LightGBM handles missing values; keep the rows.
2. **Weather look-backs:** temperature lags and rolling sums are row shifts over the frozen tail plus a 3-day live frame (`model.py:1109-1119`, `3250-3282`), so they land on the wrong day once the model is older than 3 days (0.3–0.5 kWh/day). Make them time-based or fetch a longer live tail.
3. **Imputation:** short lags are filled with same-hour means, `lag_24h` and the sub-meter lags with medians (heating `lag_24h` served 0.001 against 0.185 trained), and `rolling_std_24h` is served as exactly 0. Use one rule.
4. **Dead features:** 13 have zero gain (`physics_kwh`, `hours_ahead`, `heating_active`, `is_away`, `people_home`, the three holiday features, `likely_ev_hour`, `season`, `month_cos`, two sub-meter columns). `away_df` holds 10 days of zeros.
5. **Weather source:** SRG wind runs about 45 % above the archive wind used in training, and SRG sunshine at 0–40 % of it. Accuracy effect 1–2 %; record only.

### #83 — Add `predicted_day_total` Feature (Temperature Regression)

**Priority:** Low (lowered 2026-11-13).

**2026-11-13:** in the audit a degree-day daily-total model was worse than current code in winter (hourly +6 %) and blending it in gained nothing outside the noise floor. Retest on the capture frame (#119) before building.

**Context**: Empirical analysis (2026-04-22) shows non-EV daily totals range 14→40 kWh across seasons, strongly temperature-driven. After #82, clean regime clusters will capture shape. A separate `predicted_day_total` feature from a lightweight regression (`heating_deg_sum_24h`, `temp_ewma_24h`, `is_away`, `people_home` → daily total kWh) would give the main model an explicit scale signal independent of the regime shape.

**Note**: May not add much if the cleaned `regime_kwh` already encodes enough scale information. Evaluate after #82 is live and SHAP importance is re-checked.

**Prerequisite**: #82.

**Effort:** ~3 h. **Impact:** MEDIUM.

---

### #15 — HVAC / Boiler State: Projected Flow Setpoint

**Priority:** escalate if sub-sensor bouncing persists after 2026-04-20; otherwise long-term.

**Signal:** Derive a `flow_setpoint` feature from the Kermi heating curve — this allows accurate 48-hour forward projection using forecast outdoor temps, rather than relying on stale sensor values.

**Projection formula (per future hour h):**
```
flow_setpoint(h) = np.interp(outdoor_temp[h], curve_x, curve_y)
                   + parallel_shift          # current HA entity value, projected flat
                   - 2  if 21 ≤ hour < 24
                       or  0 ≤ hour < 6     # night setback
                   → NaN if outdoor_temp[h] ≥ 20  # heating cutoff
```

**Heating curve breakpoints (from Kermi UI):**

| Outdoor °C | Flow °C |
|---|---|
| -20 | 55.5 |
| -15 | 52.5 |
| -10 | 49.5 |
| -5 | 46.0 |
| 0 | 43.0 |
| 5 | 39.5 |
| 10 | 35.5 |
| 15 | 31.0 |
| 20 | 25.0 |

**Config keys:**
```yaml
heating_curve_sensor: sensor.kermi_parallel_shift
heating_curve_points:
  - [-20, 55.5]
  - [ -5, 46.0]
  - [  5, 39.5]
  - [ 20, 25.0]
heating_cutoff_temp: 20
night_setback_delta: -2
night_setback_start: 21
night_setback_end: 6
```

**Effort:** ~3 h. **Impact:** HIGH for heat pump buildings.

---

### #10 — School Holiday Feature

Swiss Schulferien dates are canton-specific but stable year-to-year. During school holidays daytime consumption rises. Implement a static lookup table per canton via `apps.yaml`; add `is_school_holiday` to `_FEATURES_BASE`.

**Effort:** ~4 h. **Impact:** MEDIUM.

---

### #46 — Dashboard: Replace Personal Entity IDs

`dashboard/dashboard.yaml` and `dashboard/energy-today.yaml` contain user-specific entity IDs (`sensor.skoda_enyaq_*`, `sensor.kermi_*`, etc.) that will break on other installs. Required before HACS or wider sharing:

- Replace personal entity IDs with commented-out placeholders.
- Add `# EDIT: replace entity IDs below with your own` header comment in each file.

**Effort:** ~30 min. **Impact:** UX / sharing pre-requisite.

---

### #16 — HACS Support

Make the app installable via [HACS](https://hacs.xyz/) (AppDaemon category). No code changes needed — `apps/energy_forecast/` is already in the correct location.

Required:
- Add `hacs.json` at repo root.
- Add `info.md` (HACS install panel; must warn that `apps.yaml` setup is still manual).
- Add "Install via HACS" section to README.
- Set repo topics: `appdaemon`, `home-assistant`, `hacs`.

**Effort:** ~1 h. **Prerequisite:** #46 (entity ID cleanup).

---

### #18 — Custom Component Config Flow *(long-term)*

A full HA custom component with UI-driven setup wizard (entity picker, lat/lon auto-populated, optional fields). Writes `apps.yaml` and patches AppDaemon add-on dependencies via Supervisor API. Significant effort; only path to zero-manual-step install.

**Effort:** 8+ h. **Impact:** UX / install.

---

### #91 — Daily Update-Check + Notification

**Context:** Other users are expected to stay on `main`, which only receives stable releases (merged from `dev` after a local test period) — so a GitHub-release-based check gives them a quiet, reliable "you're behind" signal without needing to watch the repo themselves.

**Design:**
- Daily `run_daily()` hook (new, added in `initialize()`) hits the unauthenticated GitHub API: `GET https://api.github.com/repos/m-zenker/ha-energy-forecast/releases/latest`. 1 req/day is negligible against the 60/hr unauth rate limit — no token needed.
- Parses `tag_name`, compares against local `__version__` (`apps/energy_forecast/__init__.py`).
- **Pre-release guard:** skip the check entirely when local `__version__` contains `-alpha`/`-beta` — the maintainer's own dev system runs ahead of `main` and shouldn't nag itself; this is purely for main-track users.
- **Dedup:** persists `{"last_notified_tag": ...}` in a small JSON file next to the existing `pred_history.json` (reusing `self._cache_path`), so a version triggers at most one notification, not a daily repeat.
- **Notify:** `self.call_service("persistent_notification/create", title=..., message=..., notification_id="hef_update_available")` — fixed `notification_id` means a later check replaces rather than stacks.
- **Failure handling:** network/parse errors log a warning and skip silently; retried on the next daily run. No new dependency (`requests` already used by `weather.py`).

**Effort:** ~1–1.5 h. **Impact:** distribution/UX, same bucket as #16.

---

### #87 — Recent Consumption Trend Feature (`trend_deviation`)

**Priority:** Low (lowered 2026-11-13) — retest before building.

**2026-11-13:** the inputs are degenerate at day-ahead lead (`rolling_std_24h` is served as exactly 0, `rolling_mean_24h` as a constant fill), so this feature can only carry signal inside the first 24 h. The 2026-06-21 simulation predates that finding.

**Context:** Simulation (2026-06-21) on last-30-day holdout shows ~18% daily MAE improvement from adding `trend_deviation = rolling_mean_24h − rolling_mean_7d` to the feature set. The feature ranks 14th of 19 in importance — it adds signal the model cannot derive by itself from individual rolling stats because tree splits on pair-wise differences are expensive to find without an explicit feature. Confirmed by LightGBM simulation without weather features (weather-only simulation; absolute numbers not directly comparable to live model).

**Design:**
```python
# In _engineer_features(), after rolling stats are computed:
df["trend_deviation"] = df["rolling_mean_24h"] - df["rolling_mean_7d"]
df["trend_z_score"]   = df["trend_deviation"] / (df["rolling_std_24h"].clip(lower=0.05))
```
Both columns added to `_FEATURES_BASE`. At prediction time, `trend_deviation` and `trend_z_score` are derived from already-computed rolling stats, so no new data fetching is needed.

**Note:** Does not fix the thermal-transition cluster (Jun 4-12 type days) — those require shorter halflife or a temperature-similarity weighting scheme (see #88). Adds signal on ordinary days.

**Effort:** ~1 h (code + 2 tests). **Impact:** LOW-MEDIUM (+18% daily MAE in simulation; smaller real-world gain expected with weather features present).

---

### #88 — Temperature-Similarity Sample Weighting

**Priority:** Low-Medium — re-scoped 2026-11-13: test on the capture frame (#119) with the cold-onset stability metric.

**2026-11-13:** the full-year prerequisite is met. For: rows at ≤ 3 °C hold 2.3 % of training weight (about 56 effective rows), and one more day of data moved the 2 °C forecast by 45 %. Against: this autumn runs lower than last winter at the same temperature (18.9 against 24.5 kWh/day at 3–6 °C, 5 days), so up-weighting old cold rows may over-forecast.

**Context:** Simulation (2026-06-21) compared three weighting schemes on 30-day holdout (May 22 – Jun 21):

| Scheme | Daily MAE | Daily MBE | Weighted mean train temp |
|--------|-----------|-----------|--------------------------|
| time-60 (current) | 3.52 kWh | −3.20 kWh | 8.5 °C |
| time-30 (shorter halflife) | **3.20 kWh** | −2.84 kWh | 10.5 °C |
| tempsim (time-60 × Gaussian kernel σ=5°C) | 3.35 kWh | −3.01 kWh | 12.8 °C |

Holdout period mean outdoor temp: **19.9 °C**. Temperature-similarity shifts the effective training distribution toward warmer data, but can only shift 4°C (8.5→12.8) with current history — still 7°C below the holdout mean. The simpler time-30 halflife outperforms the combined approach.

**Key finding:** Both improvements are modest; neither fixes the thermal-transition cluster (Jun 4-12 type days). The root cause of those outlier days is that the model has never seen warm-month data (history started Oct 2025). **As summer 2026 data accumulates, temperature-similarity weighting will become meaningfully effective.**

**Design (when ready to implement):**
```python
# In train(), after existing exponential decay weights:
temp_sigma = self._cfg.get("temp_weight_sigma_c", 5.0)  # configurable, 0 = off
if temp_sigma > 0 and predict_temp is not None:
    temp_sim = np.exp(-((train_temps - predict_temp)**2) / (2 * temp_sigma**2))
    h_weights = h_weights * temp_sim
```
`predict_temp` = mean of last 24h outdoor temperature. New config key `temp_weight_sigma_c` (default 0 = off; suggest 5.0 once ≥ 12 months of history exists).

**Prerequisite:** Revisit after first full year of data is available (earliest: Oct 2026).

**Effort:** ~3 h. **Impact:** LOW now, MEDIUM once a full year of history is available.

---

### #84 — Legionella / DHW Boost Hour Feature — **SUPERSEDED (2026-08-22)**

**Superseded by the #93 spec revision** (`ha-energy-manager/docs/superpowers/specs/2026-08-15-dhw-comfort-boost-commit-design.md`) — a deterministic post-model forecast correction replaces the plan below entirely; no `is_legionella_hour` ML feature will be built. See `memory/project_dhw_comfort_boost_commit_spec.md`. Original item kept below for history/context.

**Prerequisite:** DHW sub-sensor infrastructure (related to #22).

**Priority: raised to MEDIUM-HIGH (2026-08-04)** — this is no longer just a lag-pollution smoothing item. Confirmed live evidence: a legionella DHW commit (`ha-energy-manager` → `energy_forecast/set_dhw_schedule`) landed correctly in the physics layer (`physics_base_today` showed the expected +3.6 kWh spike) but was ~94% cancelled by the ML layer (`ml_adjustment_today` = -3.4 kWh at the same hour) — `physics_kwh` didn't even place in the top-5 SHAP features that day. A dedicated binary feature is a direct, independent fix for this specific failure mode, regardless of whether/when `UA_eff` (see #92) or Phase 2 residual mode ever land. See `memory/project_physics_kwh_low_importance.md`.

**Problem**: The weekly legionella DHW protection cycle (heat buffer to ~60 °C, ~1–2 h) creates a predictable spike that the model has no dedicated signal for. Currently relies entirely on `lag_168h` (1-week lag), which takes 2–3 weeks to establish after a schedule change. The schedule was shifted from Tuesday ~23 h to Wednesday ~14 h on 2026-04-22, so the transition period is live now.

Lag-feature pollution to the following day is modest (~0.1–0.3 kWh/h for 24–48 h), so this is not urgent **on its own** — but combined with the cancellation evidence above, the feature is worth doing sooner.

**Design (when implemented)**:
- New `_compute_likely_legionella_hours()`: detect HOW slots where `dhw_buffer_temp > 58 °C` within a 30-day rolling window
- New binary feature `is_legionella_hour` (mirrors `likely_ev_hour` pattern)
- **Revisit vs. original design**: prefer sourcing this from `physics_schedule.json`'s `committed_override` (hef already knows the exact committed date/hour from `ha-energy-manager`, when present) over inferring from a `dhw_buffer_temp > 58°C` threshold — more precise, and available immediately rather than needing 30 days of buffer-temp history to infer the pattern. Fall back to the threshold-inference approach when no override has ever been committed (e.g. Phase 1 users without `ha-energy-manager`'s scenario-gate wired up).
- Optional `legionella_schedule_reset_date` config key to prune pre-change data and accelerate transition
- Falls back gracefully to 0 when `dhw_buffer_sensor` is not configured

**Effort:** ~3 h. **Impact:** MEDIUM (was LOW-MEDIUM — see priority note above).

---

### #92 — Temperature-Based UA_eff Calibration Window (replace hardcoded month list)

**Status (2026-09-03): implemented on `fix/ua-eff-calibration-window`, not yet merged to `dev`.** Design spec (`docs/superpowers/specs/2026-09-03-ua-eff-calibration-window-design.md`, rev. 3, 3 review rounds) and implementation plan (`docs/superpowers/plans/2026-09-03-ua-eff-calibration-window.md`) both landed. Built via subagent-driven-development: 3 tasks + a final whole-branch review + one fix wave, all clean. Full suite 1060 passed / 11 skipped / 0 failed. See `memory/project_ua_eff_calibration_window_fix.md` for the summary and the manual apps.yaml step still needed to activate tier 1 on the live instance.

**Priority:** Medium — land before/during this coming winter so the switch is in effect when there's real cold-weather data to calibrate from.

**Problem**: `_calibrate_ua_eff` (`physics.py:437`) gated its input rows to `timestamp.dt.month.isin([11, 12, 1, 2, 3])` — a hardcoded Northern-Hemisphere heating-season proxy. It's brittle in two ways: it can't react to genuinely cold shoulder-season data (a cold October or April), and it silently assumes a hemisphere/climate. Confirmed 2026-08-04 as a contributing cause of `UA_eff` staying `null` (space-heating term hard-zeroed, `physics.py:186-188`) — no calibration run had yet fallen inside a matching month since this feature went live. See `memory/project_physics_kwh_low_importance.md`.

**Design (implemented, superseding the original draft below)**: a three-tier eligibility resolver (sub-meter → `heating_active` → temperature-fallback), not the single temperature-threshold sketch originally proposed here — a multi-stakeholder spec review found the temperature-only approach too weak on its own. See the spec for the full rationale, the real-data validation run (§4 — floors clear, but R² doesn't yet with ~5 months of data; expected to actually calibrate once real winter data arrives), and the explicitly deferred items (§8: defrost exclusion, live COP sensor, thermal-mass regression redesign, out-of-sample R² eval, the other 3 NH-hardcoded gates).

~~**Design (draft — needs its own short brainstorm before implementing)**: replace the calendar-month filter with a temperature/heating-degree threshold applied to the same candidate rows (e.g. `T_outdoor` below some margin under the heating setpoint) so eligibility tracks actual weather rather than the calendar. Keep the existing acceptance gates (≥30 valid passive windows, R²≥0.5) unchanged — this only changes which rows are eligible to be counted.~~ (superseded, see above)

**Not required for this**: the local climate CSV cache (`climate_<entity>.csv`) already accumulates indefinitely via merge-on-fetch (verified live: `climate_wohnzimmer.csv` has been growing since 2026-03-31, independent of HA's own recorder — confirmed via `configuration.yaml` and the HA history API that the live recorder itself only retains ~10 days, default `purge_keep_days`). No separate data-retention work is needed here.

**Effort:** ~~~2 h (filter logic + tests)~~ — actual: a multi-day spec (3 review rounds) + a 3-task subagent-driven implementation (found and fixed 3 real bugs across per-task and final review) — the original estimate badly undersized the eligibility-tier design and plumbing work once reviewed. **Impact:** MEDIUM — removes a real hardcoded-assumption smell and closes the plumbing gaps so `UA_eff` will now be *evaluated* against real winter data as it arrives, but whether it actually calibrates to a value is only verifiable once real winter rows exist, ~Dec 2026/Jan 2027.

---

### #93 — DHW Override Commit + Deterministic Forecast Correction (cross-repo, mirrors Plan 2; supersedes #84)

**Status (2026-08-27): both phases implemented and deployed live — kill-switch pending exit-gate observation.** Spec at `ha-energy-manager/docs/superpowers/specs/2026-08-15-dhw-comfort-boost-commit-design.md`. Phase A (this repo's hef-side deterministic-correction half) deployed 2026-08-27 as `v0.12.0-alpha-13`. Phase B (`ha-energy-manager`'s EM-side commit-triggering half) deployed the same day as `v0.15.1-alpha-51`, with its `dhw_comfort_boost_commit_enabled` kill-switch left `false` — comfort-boost commits don't actually reach hef yet. Activation requires a manual exit-gate check (≥3 observed legionella cycles against Phase A's live correction, ~2-3 weeks) before the kill-switch flips; see `ha-energy-manager/memory/project_dhw_comfort_boost_cross_repo_rollout.md` for the runbook and `memory/project_dhw_comfort_boost_commit_spec.md` for what the spec revision changed and why. This item supersedes #84 entirely.

**Prerequisite context:** `ha-energy-manager`'s hef-physics-adoption Plan 2 (`docs/superpowers/specs/2026-07-10-hef-physics-adoption-design.md` / `plan2-legionella-scenario-gate.md`, live on both repos' `dev`) wired `_check_legionella`'s four branches to commit their chosen hour to hef via `energy_forecast/set_dhw_schedule`. It deliberately scoped out everything else.

**Problem**: `ha-energy-manager`'s DHW Two-Tier Comfort-Boost Solar Scheduling (shipped 2026-07-31, `v0.15.1-alpha-26`) now *also* picks a scheduled hour ahead of time — `heat_pump.py::_arm_dhw_schedule()` → `_rank_dhw_boost_candidates()`, exposed as `dhw_boost_scheduled_hour` — but never calls `set_dhw_schedule`. hef has zero visibility into routine comfort-boosts, the same blind spot Plan 2 fixed for legionella, except this one fires far more often (near-daily on sunny days vs. weekly). Raised 2026-08-04 alongside #84/#92 but is its own item — different code path, different fix shape, not an ML feature-engineering change like #84.

**Design (resolved by the spec above — summary, read the spec for the full/current design):**
- EM: `_commit_dhw_schedule` generalized (kind + optional date), called from a new async `ScenarioScorer`-gated candidate walk off `_arm_dhw_schedule`'s ranked list, with a re-entrancy guard and a desirability re-check before commit. `_clear_dhw_schedule` gains a required `notify_hef` param so only genuine cancellation (not the boost-starting path) notifies hef. **New in the revision:** the commit payload also carries the real, dynamically-computed `_dhw_boost_target_c` (was previously omitted; hef approximated it).
- hef (**replaced 2026-08-22, was**: merge-semantics + `T_dhw_upper` approximation): `physics_kwh` (ML feature) is always computed override-blind; a new `override_delta_series` captures the override's marginal kWh independently and is subtracted from the training target / added back unconditionally to the live forecast — a deterministic guarantee, bypassing the ML layer entirely rather than relying on a learned feature weight. `committed_override` gains a parallel append-only `override_history` log so retrain reconstruction replays every past commit accurately, not just the most recent one — this directly fixes what the previous design left as a deferred Known Limitation.
- Resolved: full `ScenarioScorer`/`get_scenario` gating (mirroring legionella) — unchanged from the original decision.
- Open question left for the next review: whether the `ScenarioScorer`'s per-candidate `get_scenario` scoring payload should also carry a would-be `target_c`, or only the winning candidate needs it at commit time.

**Effort:** ~1 day (cross-repo, EM + hef, tests both sides) — original rough estimate; likely runs longer given the revised hef-side scope (override-blind feature split, target/serve-time correction, append-only history) on top of the original EM-side scope (re-entrancy guard, desirability re-check, entity-key fix, staleness mechanism). **Impact:** MEDIUM-HIGH — the deterministic-correction mechanism is specifically designed to avoid the physics_kwh-cancellation failure mode confirmed live for legionella, rather than merely hoping comfort-boost's higher frequency helps the model learn it.

---

### #96 — Cooling Mode / AC Support (Tropical/Hot-Climate Thermal Pressure)

**Priority:** Low — long-term, community-contribution candidate.

**Source:** [GitHub Discussion #20](https://github.com/m-zenker/ha-energy-forecast/discussions/20), opened 2026-08-17 by @gabrieldelboniz.

**Problem**: The physics thermal-pressure model (`setpoint − indoor_temp`) is heating-only. In cooling-dominated climates, thermal pressure goes negative once AC engages, which effectively disables the thermal signal exactly when cooling load is highest. Passive cooling is already modeled via `infiltration_pressure` (#57); active AC is not.

**Proposed additions (from the discussion)**:
- `cooling_system_active_entity` — config flag/entity marking active cooling mode
- Inverted thermal pressure while cooling: `indoor_temp − setpoint` (positive once above setpoint)
- Cooling-degree metrics, mirroring the existing heating-degree-hour features (`heating_deg_sum_24h/168h`, #50)
- Cooling-specific hysteresis config (`cooling_temp_on`/`cooling_temp_off`, paralleling existing heating hysteresis)

**Maintainer response (2026-08-18)**: acknowledged, added to roadmap; deprioritized behind scenario-API work (see #93), no timeline committed; open to community PRs.

**Note**: doesn't fit the current Design Decisions — this deployment and its calibration (τ, `UA_eff`, hysteresis defaults) are tuned for a heating-dominated Swiss climate. Should land as an opt-in mode behind a config flag so heating-mode behavior for existing users is provably unaffected, not a default-on branch in the shared thermal-pressure calculation.

**Effort:** unscoped — likely 1 day+ (new features, config, dual hysteresis paths, tests, docs). **Impact:** N/A for the primary personal deployment (heating-dominated); relevant mainly for community/HACS adoption in hot climates.

---

### #97 — Open-Meteo Archive Retry/Backoff on Transient 5xx

**Source:** GitHub Issue #21 (2026-09-17) — a user hit a 503 from the Open-Meteo archive API during backfill/retrain. The reporter's diagnosis (`sunshine_duration` being daily-only) was verified wrong by replaying the exact request; the real gap is that `weather.fetch_historical_weather()` has no retry on transient 5xx/timeouts, so one flaky response fails the whole fetch.

**Proposal:** bounded retry (e.g. 3 attempts, exponential backoff 2/4/8 s) on HTTP 5xx and connection timeouts only; 4xx fails immediately. Log each retry at INFO and the final failure at WARNING (existing fallback path unchanged).

**Effort:** ~1 h incl. tests (mock `requests.get` raising/returning 503 then 200).

---

### #99 — Heating-Season Projection Follow-ups

Deferred from #98's final whole-branch review (plan `docs/superpowers/plans/2026-09-25-heating-season-projection.md`, local). None block correctness on the live system; most matter for other users.

1. **Bias recheck (~2026-10-03):** confirm the 5–15 °C forecast bias ≈ 0 and `mae_7d` < ~0.24 after a clean week. Confounders: any other deploy in the window. **Partial (2026-10-06):** `mae_7d` 0.196 (target met; 0.298 on 09-25); 10–15 °C bias +0.205 → +0.065–0.085 (n=77–80); 5–10 °C bin has n ≤ 3 post-deploy. Still open: no day since the deploy had a daily mean < 13.5 °C, so the cold-weather behaviour is untested. Rerun once one occurs, and look at the 5–10 °C bin and 01–09 h bias. The residual +0.07 at 10–15 °C with heating off points at something besides `heating_active` (physics `heating_buffer_temp`, lag features, solar correction).
2. **Zero-width threshold band:** the learner's grid allows `on == off`, and ties resolve to the lowest/narrowest pair (the switch label learned 10.5/11.0 in the backtest). Add a minimum band width (e.g. 1 K) or break ties toward the widest band, so the projection can't flip daily around a single threshold.
3. **Stale config override:** `_retrain` persists the *resolved* thresholds, so a removed `heating_temp_on/off` override is reloaded at init until the next retrain. Persist `learned_thresholds` instead.
4. **Unavailable entity prior:** an `unknown`/`unavailable` switch falls back to prior 0; use the last recorded `heating_active` value (already fetched hourly).
5. **Skip learning when unused:** users with no sub-meter and no switch still learn, save and log default thresholds on every retrain.
6. **Test gaps:** no `_retrain` wiring test per label tier; no serve-gate test for a meter-only user without a switch; `test_thresholds_default_when_unset` reads the real `models/heating_thresholds.json` (conftest only redirects writes).
7. **Docs:** the CHANGELOG says the frost override is < 4 °C, but `docs/examples/heating_season_automation.yaml` uses 3 °C. The CHANGELOG also mentions the gitignored `scripts/backtest_heating_feature.py`.

**Effort:** 2–3 h total; items 2–4 are small TDD fixes in `heating_season.py` / `energy_forecast.py`.

---

### #100 — Source Cumulative-kWh History from HA Long-Term Statistics *(long-term)*

**Source:** GitHub Issue #24 (2026-09-27). A grid-import sensor updating every ~10 s produced ~120k recorder states; the 30-day `get_history` resync exceeded AppDaemon's hard 10 s HA REST timeout and silently returned nothing, which let partial-hour cache rows survive three retrains and collapse the forecast to ~0. The short-term fix (shipped in v0.11.15, GitHub reply posted 2026-10-06; plan `docs/superpowers/plans/2026-09-28-partial-bucket-cache-poisoning.md`, local) never caches the open bucket and splits the fetch into 2-day chunks. The payload still grows with sensor update rate, though: about 8.6k states/day for such a sensor, when the app only needs 24 hourly values.

**Proposal:** for `total_increasing` kWh sensors (grid import, solar, export, battery, EV, sub-meters), read hourly `sum`/`state` from HA's recorder long-term statistics instead of raw states. The recorder already aggregates these hourly, so 30 days is ~720 rows regardless of update rate. Statistics are also kept indefinitely, while raw states are purged after `purge_keep_days` (default 10), so a fresh install could backfill far more than 30 days.

**Open questions:**
- AppDaemon 4.5 has no statistics API. It would need the `recorder/statistics_during_period` websocket command, either through the HA plugin's internal websocket or a separate client using the existing long-lived token. Spike this first.
- The hourly granularity rules out the 15-minute cache (`fetch_energy_history_15m`). Keep raw-state fetches there, or use 5-minute short-term statistics (kept only ~10 days).
- Sensors without `state_class` have no statistics, so the raw-state path must stay as a fallback.
- Statistics buckets are labelled by hour start in UTC. Verify they align with the naive-local `timestamp` convention and the DST handling in `_check_dst_duplicates`.

**Effort:** ~1 day after a ~2 h feasibility spike. **Impact:** removes the whole class of history-size/timeout failures for high-frequency meters and speeds up retrain fetches for all users.

---

### #109 — Heating-Buffer Fade: Live Value Near-Term, Steady-State Further Out

**Status (2026-10-20): smoothing option (point 4) IMPLEMENTED on `feature/buffer-smoothing` (S24, W = 24 h), DEPLOYED 2026-10-20** ahead of the C verdict by user decision (C count restarts at the first retrain; A+B gate must be re-anchored to that). Takes effect at the next retrain; old models keep the raw path (window persisted in `meta.pkl` as `heating_buffer_smooth_h`). A code downgrade with a smoothed model on disk needs a retrain or restoring the previous model archive. **The fade framing below is superseded** (history kept): the response is a step (34 -> 35 C), so a fade target must satisfy f(level) ~ E[f(buffer)], which a 24 h median does not.
- **Gate 1 (backtest, 4 folds, pooled MAE kWh/h, A / NB / S24 / S48 / L1F):** 0-5 h .1927/.1904/.1914/.1959/.1912; 0-23 h .2029/.1990/.2009/.2047/.2014; 24-47 h .2115/.2088/.2090/.2139/.2085; cold 0-23 h .2721/.2604/.2674/.2691/.2643. Stability (last fold, mean/max): A .13/.80, NB .06/.36, S24 .07/.67, S48 .05/.24, L1F .08/.78. NB, S24, S48 pass.
- **Gate 2 (refit replay, origins 10-09..10-19, in-sample, A/NB/S24):** A single-reading sweep range 5.47 kWh (precondition ok), level-sweep max step 1.85; NB and S24 flat (S24 responds only beyond +-3 C level shift). Next-day total |delta| mean/max: A 0.90/5.89, NB 0.05/0.58, S24 0.07/0.71. Cold-day MAE ratio vs A (0-5/6-23/24-47 h): NB .935/.951/.95, S24 .933/.951/.934. Daily bias about -0.6 kWh/day vs A (replay script's figure is mis-scaled), within bar.
- **Lag variant (L1F, point 5):** evaluated in the backtest, did not beat every smoothing arm on both buckets + stability -> not pursued. NB (drop the buffer from training) was an equal-accuracy alternative and stays the fallback if S24 misbehaves live.
- **Follow-up:** the heating ladder (ha-energy-manager) has never engaged and is being reworked. S24 is a cheap pre-heating hook (a 24 h buffer level would register it) but the model has seen no ladder days, so claim no benefit before a ladder-day backtest. An explicit ladder-level / pre-heated-energy feature or HEM's planned schedule as a forecast input is likely better; forecast-vs-manager feedback (ladder consumption depends on the solar forecast) needs a deliberate design.
- **After deploy:** confirm the first nightly retrain wrote `heating_buffer_smooth_h` into `meta.pkl`, re-run `check_cold_onset_gate.py` and the buffer sweep on live data; the C count restarts at that retrain.

**Source:** 2026-10-19 follow-up to #107. After the deficit fade, the remaining hour-to-hour jumps in the tomorrow forecast all trace to `heating_buffer_temp` (see `memory/project_forecast_jumps_snapshot_broadcast.md`). `_prepare_prediction_X` (`model.py:1115`) holds the latest buffer reading flat across the whole 48 h horizon. The buffer cycles 28↔36 °C with the heat pump, so the forecast follows the cycle (replay: 30 → 36 °C ⇒ +3.4 kWh tomorrow; pinning it to 32 °C removed 4 of 5 jumps).

**Proposal (initial idea):** use the live reading for the first ~2 h, then blend to a steady-state level (e.g. the 24 h median) by ~6 h, mirroring #107's deficit fade. The numbers are placeholders to tune.

**Design points to settle before a spec:**
1. **Shape and time constant:** an exponential decay (one parameter, like #107's τ_r) is preferred over hold-then-linear. Fit τ from the buffer data, e.g. the lag at which the reading stops predicting the next hours' consumption, rather than fixing 2 h/6 h. Within one HP cycle the reading may already be stale after < 1 h.
2. **Near-term sign:** in training, the feature is measured in the same hour as the target, so a hot buffer means "the HP is running now". Physically, a just-charged buffer means the HP is about to switch off. The live value may therefore add little, or point the wrong way, even at 1–2 h.
3. **Steady-state target:** a flat 24 h median ignores the heating curve: colder hours need a hotter buffer. Better candidates are a same-hour median over recent days, or the 24 h median shifted by the heating-curve difference (`heating_curve_sensor`).
4. **Alternative (training-side):** train on a smoothed buffer level (e.g. rolling 24 h median), so training and prediction use the same signal and the feature stops standing in for "HP running now". Needs a retrain; the predict-side fade does not.
5. **Variant (2026-10-20): 1 h-lagged training feature + fade at predict.** Train `heating_buffer_temp` as the reading at *t−1* for target hour *t*. The model then learns the causal sign: a cooled-down buffer means the HP switches on, so consumption is high. At predict, the live reading is the true 1 h lag only for the first forecast hour; beyond that it fades to the steady-state target (point 3). A plain 1 h lag on its own does **not** fix the jumps: for hour +k the *t+k−1* reading is unknown, so the current reading would again be held flat across the horizon. Limit it to the heating buffer. For `dhw_buffer_temp`, the 2026-10-19 backtest showed same-hour beats lagged on MAE (arm C, origin-lagged with random lead: 0–23 h 0.219 vs 0.204, 24–47 h 0.226 vs 0.212, A better on 59/79 days; stability 0.19 vs 0.34), because same-hour DHW explains reheat spikes away in training. Test as arm D in `scripts/backtest_tank_state_features.py` (heating buffer lagged + fade, DHW unchanged). Expect a weak verdict while the buffer ranks #50 in fold models; the live-model replay (point 6) decides.
6. **Validation:** the offline backtest can't judge this (`heating_buffer_temp` ranks #50 in fold models because the sensor starts 06-30; it's #7 in the live model). Gate it with a live-model hourly-origin replay like `scripts/replay_deficit_fade.py`: next-day total stability (mean |Δ|), MAE at 0–6 h and 24–47 h, and daily bias.

**Related:** `dhw_buffer_temp` is a constant 50 °C for future hours, which skews forecasts (lower outside reheats) but doesn't cause jumps. The 2026-10-19 backtest showed this arrangement still wins on MAE, so it's out of scope here.

**Effort:** ~½–1 day (τ fit + replay + implementation); +½ day for the lag variant's arm D and retrain. **Impact:** removes the remaining buffer-driven swings of the tomorrow forecast, which EM's battery and DHW scheduling consume.

---

### #110 — Regime Label: Train/Serve Leak + Hard-Label Flip

**Priority:** Medium — needs a backtest before planning; not in the cold-onset deploy windows.

**2026-11-13 audit:** regimes off is worse than the shipped soft labels (3 h blocks +3.1 % on dev, +12.4 % on the holdout), so regimes stay. Watch item, no action: on 4 capture days `regime_kwh` was served about 36 % above the value later trained on for the same hours. Run the 11-19..21 stability check, then close.

**Problem:** training builds `regime_kwh` from the clusterer's *true* daily labels (model.py ~605–632), but live prediction uses the RandomForest's *predicted* label (model.py ~1197–1215). In harness models `regime_kwh` is the #1 feature (31–33 % gain); with out-of-bag soft training labels it drops to #10–12 (2–2.5 %). It's a hard label, so a small change in daily sunshine near a regime boundary switches centroids. On 2026-10-21 an SRG-SSR 503 → Open-Meteo fallback flipped tomorrow's label 0 → 2 (7.9 → 20.9 kWh/day centroid) and the `tomorrow` forecast jumped 11.06 → 16.64 → 11.06 kWh. Training uses Open-Meteo sunshine only, while the live SRG path uses SRG `SUN_MIN` (skew not yet shown to be systematic). Forecast days ≤ ~8 °C will start tipping into regime 2.

**Options simulated 2026-10-21:** (P) soft regime at predict only: kills the flip (+5.93 → +0.45 kWh) but lifts the level +3.75 kWh and costs 5–9 % October MAE, so rejected. (T) out-of-bag soft labels in training + soft at predict: removes the leak, ~4–6 % worse in October, better in March; coin flip on paired daily wins. Not simulated: lock tomorrow's label once per day (stops flicker, not a source-switch flip), or take sunshine from Open-Meteo in the SRG path too (train-consistent, narrow).

**Backtest 2026-10-27** (`scripts/backtest_regime_label.py`, label-level, no LightGBM retrain; rebuilds the exact live clustering input):
- **Centroid mismatch explained.** The old harness (35/64/10.6) clustered the target without subtracting EV energy, which makes a 63.6 kWh/day evening-peak cluster. Live input = corrected target − EV (hybrid sensor/threshold) − dishwasher/washer/tumbler; the harness reproduces live exactly (7.89/19.76/36.47 kWh/day). The 10-21 numbers (7.9/37/20.9) were the pre-wallbox threshold-EV path (8.07/20.68/37.02); A+B moved them only slightly. Label ids are not stable across retrains — compare by centroid kWh, never by id.
- **Sunshine is not the flip driver.** Sunshine ×0.7 (the SRG vs Open-Meteo gap) flips 1.7 % of days ≤ 10 °C and 0 of 27 October days. The RF is temperature-driven (importance temp_mean 0.40, temp_max 0.27, temp_min 0.21, sun 0.09). Forecast temperature error is: ±1 °C flips 15–19 % of days ≤ 10 °C, ±2 °C flips 24–36 %. So the 10-21 flip more likely came from the source switch changing the forecast temperature, not sunshine — **the "take sunshine from Open-Meteo in the SRG path" option is dead** (untested for temperature).
- **The train/serve leak is real in the cold.** Out-of-bag label accuracy is 84.5 % overall but 74.7 % on days ≤ 10 °C, while training uses perfect labels. 23 % of October days have a top-2 probability margin < 0.2. The label flipped 1 → 0 (7.9 → 19.8 kWh/day) on 10-22 as the mean temperature fell to ~7.5 °C; actuals 10-22..26 were 13–17 kWh/day, so regime 0's centroid is ~3–6 kWh high there.
- **Option comparison** (blocked CV, shared day bias σ 1.0 °C + per-origin σ 0.5 °C, 24 origins/day, days ≤ 10 °C, n = 148): label changes per day — hard (live) 1.61, hysteresis(0.15) 0.39, running-mean probabilities 0.16, lock-first 0.00. Label accuracy stays 73–76 % and centroid MAE 5.3–5.7 kWh/day for all four — stabilising removes flicker but does not fix the accuracy gap. Soft centroid has lower centroid MAE (4.74) but a −1.6 kWh bias, consistent with the 10-21 (P) result.
- **Caveats:** simulated i.i.d. temperature noise, not the real SRG/Open-Meteo switch (a persistent mid-day shift, which lock-first cannot undo if the first origin is the bad one); centroid error, not final LightGBM MAE; October n = 26.
- **LightGBM-level replay** (`scripts/backtest_regime_replay.py`; fresh `EnergyForecastModel` per fold, k=3, 8 hourly origins per target day with identical noisy temperature forecasts, σ_day 1.0 °C + σ_origin 0.5 °C; 64 days: Jan26 23, Mar26 23, Oct26 18): **no policy changes MAE** — hourly MAE hard 0.559, lock 0.561, running-mean 0.558, hysteresis 0.558, oracle (true label) 0.545 kWh/h (diffs ≤ 0.002; cold days the same). **Even a perfect label gains only −2.6 % MAE**, so label accuracy is not a lever; stability is. Origin-to-origin jump of the day's total (mean |Δ| kWh): hard 1.23 → lock 0.78, running-mean 0.83, hysteresis 0.96, oracle 0.76; on the Oct26 fold (full-year training, the relevant one) hard 0.74 → **lock 0.21**, running-mean 0.38, hysteresis 0.35, oracle 0.19 — i.e. label flicker is nearly all of the October jumps and lock-per-day removes ~72 % of them. Bias differences are noise (n = 18). Caveats: Jan/Mar folds train on only 87/140 days (daily bias −12/+1 kWh), so only their relative numbers mean anything; i.i.d. origin noise means nearer origins are not more accurate here, whereas real forecasts improve toward the target hour — a lock taken at the first (24–48 h) origin trades freshness for stability, and a bad first origin stays for the day (rng_max 10.7 kWh in Jan).
- **Leaning (revised):** lock-per-day beats running-mean at the model level (the label-level simulation favoured running-mean); candidate to plan: lock the label per target date at a fixed hour, persisted across restarts. Open design questions before a plan: when to take the lock (earliest origin vs a late-evening one with a fresher forecast), how to persist the per-date label, and whether to refresh it when the weather source switches. Still open: expose the label as a sensor attribute so flips can be counted from HA history.

**Backtest result (2026-11-05, branch `fix/regime-soft-labels`, not merged):** the harness reproduces the live clusterer exactly (k=4, same centroids) but not live daily totals (up to 14 %; pure retrain jitter is ~9 %). Hard labels are clearly worst on stability (max day-total change under a sunshine ×0.66/×1.5 perturbation: H 3.1 kWh, soft out-of-fold 0.5, regimes off 0.25) and on primary-fold daily MAE (H 1.92, soft 1.57, off 1.65 kWh/day; jitter 3.5 %). Soft and off are indistinguishable within jitter; the plan's original rule would pick "off", the jitter-adjusted rule picks soft. The January guard fold is uninformative (all arms ~10 kWh/day low). It does not reproduce the live +2.79 kWh/day bias, so it neither confirms nor refutes the regime-flip cause. Part D is waiting for a decision and for the cold-onset deploy sequencing.

**Decision (2026-11-05): ship T with blocked folds** (`regime_mode="soft_oof"`, `regime_oof_cv="blocked"` are now the `train()` defaults on `fix/regime-soft-labels`; hard path kept for old pickles and explicit `regime_mode="hard"`). N (regimes off) and the accuracy fallback (never fired: out-of-fold accuracy 0.76–0.85 vs majority 0.50–0.56) were rejected; evidence in `memory/project_regime_flip_weather_source.md`. Takes effect at the next retrain after deploy. **Sequencing:** the cold-onset gate is circular for this change (the regime flip is the suspected cause of the failed re-baseline), so the user waived it for #110 (2026-11-05): deploy, then measure the re-baseline. Never in the same window as #105. **Post-deploy check at +14 days:** `python scripts/check_regime_stability.py --deploy <ts> --rebaseline <ts>` (share of same-model forecast pairs with |Δ| > 2 kWh must not rise; daily MAE ≤ 1.10 × pre; run no later than deploy + 16 d). Rollback is not durable by itself (next retrain trains soft again): revert on `dev` or pin `regime_mode="hard"` in `_retrain()`, then redeploy. Plan: `docs/superpowers/plans/2026-11-05-soft-regime-blocked-folds-ship.md`.

**Effort:** ~½ day backtest + 1 day implementation. **Impact:** MEDIUM — tomorrow-forecast stability (consumed by EM scheduling) and an honest feature ranking; decides #83. See `memory/project_regime_flip_weather_source.md`.

---

### Deferred

| # | Item | Reason |
|---|------|--------|
| #22 | EV SoC / charging state feature | EV hours are subtracted from training target — SoC has no signal to learn. Revisit if EV load is re-included. |
| #40 | Home battery SoC as feature | Unblocked (battery live since 2026-07-16); revisit if residuals show SoC correlation. |
| #24 | Electricity spot price feature | Fixed flat tariff — out of scope. |
| #94 | Remove vestigial `dhw_tank_volume_l` in `physics_schedule.json` | Dead duplicate of the `physics:` config value (`self._config`, correctly 400L, actually used by `_dhw_kwh_series`) — the schedule file's own copy (currently 200) is only ever echoed back into itself by the autonomous legionella-schedule-learning code, never read by the computation. No functional impact, but misleading if the file is inspected directly (caught 2026-08-18 during hem's DHW boost spec review). Clean up opportunistically next time `physics.py`'s schedule-writing code (`_check_legionella_stability`/schedule-learning path, ~line 684) is touched — not worth a dedicated pass on its own. |

---

## Pending Summary

| # | Item | Impact | Effort | Priority |
|---|------|--------|--------|----------|
| 83 | `predicted_day_total` scale feature | medium | 3 h | low — degree-day total model was worse in winter in the 2026-11-13 audit; retest on #119 first |
| 84 | Legionella/DHW boost hour feature | — | — | **SUPERSEDED (2026-08-22) by #93** — see below |
| 93 | DHW override commit + deterministic forecast correction (cross-repo, supersedes #84) | medium-high | ~1 day+ | done — Phase A deployed 2026-08-27 (hef `v0.12.0-alpha-13`), exit gate passed 2026-09-18, Phase B live (EM). Open: 10 phantom overrides (7 legionella, 3 comfort boost; never executed, never cancelled) replayed in training reconstruction — needs a cross-repo plan, see `memory/project_dhw_override_phase_a.md` |
| 87 | `trend_deviation` feature (recent vs baseline) | low-medium | 1 h | low — inputs degenerate at day-ahead lead; retest first |
| 88 | Temperature-similarity sample weighting | low-medium | 3 h | low-medium — re-scoped: test on #119 with the cold-onset stability metric |
| 15 | HVAC flow setpoint | high (heat pump) | 3 h | escalate if bouncing |
| 10 | School holidays | medium | 4 h | long-term |
| 46 | Dashboard entity ID cleanup | UX / sharing | 30 min | partial — interval entity IDs fixed in fix/review-critical |
| 16 | HACS support | distribution | 1 h | long-term |
| 18 | Config flow | UX / install | 8+ h | long-term |
| 22 | EV SoC | high (EV) | 4 h | deferred |
| 40 | Battery SoC | medium (battery) | 1 h | deferred |
| 24 | Spot price | n/a | — | out of scope |
| 94 | Remove vestigial `dhw_tank_volume_l` duplicate | none (dead field) | 10 min | opportunistic — clean up next time adjacent code is touched |
| 96 | Cooling mode / AC support (tropical climates) | n/a for personal use; HACS-relevant | 1 day+ | long-term — community PR candidate, see Discussion #20 |
| 99 | Heating-season follow-ups (bias recheck + deferred review minors) | low-medium | 2–3 h | bias recheck folded into the cold-onset C verdict (10-21 check not met; recheck ≥ 10-25); minors opportunistically |
| 100 | Cumulative-kWh history from HA long-term statistics | robustness for high-frequency meters (community-reported) | ~1 day + 2 h spike | long-term — after the GitHub #24 short-term fix; spike websocket access first |
| 101 | Cold-onset A: reset-aware hourly kWh for cumulative sub/correction sensors (`fix/counter-reset-increments`) | medium (clean target at 06/07/20/21 h, ≈ −0.05 `mae_7d`) | 2–3 h | **deployed 2026-10-27** (alpha-32, via `integration/ab-deploy`; forced retrain 05:40 = re-baseline anchor). Re-baseline check 2026-11-03: bias criterion FAIL, not attributed to A (see Current Status) |
| 102 | Cold-onset B: EV wallbox-sensor hours no longer treated as contamination (`fix/ev-sensor-trust`) | small MAE gain, more training/MAE coverage | 2–3 h | **deployed 2026-10-27** with A (alpha-32); `mae_*` jumped at deploy by construction |
| 103 | Cold-onset C Task 4: opt-in `heat_loss_deg` feature (steady-state heat-loss gradient) | targets the expected ≈ −2 kWh/day residual | 3–4 h | gated — own branch, backtest + paired sign test, flag on only after ≥ 14 cold days of C baseline **2026-11-13:** demoted to low. The ≈ −2 kWh/day residual it targets is not there (current code replays at +0.5 on corrected actuals); re-justify first. |
| 104 | Cold-onset D Task 3: pooled valve-closed τ estimator behind `tau_estimator` flag | low (τ only matters in heating-off hours) | 1 day | gated — needs ≥ 14 valve-closed nights in the D1 cache |
| 105 | Cold-onset D Task 4: daily heat-balance `UA_eff` behind `physics.ua_eff_method` flag (`feat/ua-eff-daily`); incl. Step 3b NaN-robust indoor averaging + DHW ODE guard (training `physics_kwh` constant since 2026-04-12, 0 gain) | activates physics heating for the first time (live `UA_eff` is null: hourly night fit fails the R² ≥ 0.5 gate) | 1 day | gated — not on D3 (Task 4 decouples `c_building` from the ML τ); hard gates: (1) C (+ #109 smoothing) verified over ≥ 3 cold days, because non-null UA shifts `physics_kwh` (ML feature) at the first retrain; (2) full-feature backtest harness `scripts/backtest_heat_loss_feature.py` (Plan C Task 4 Step 5) exists; (3) COP scale resolved (formula ≈ 3.2 vs live sensor ≈ 5.6, UA uncertain up to ×1.75). Gate = cold-fold MAE not > 2 % worse + sign test, physics daily heating MAE < 50 % of meter daily mean, `solar_gain_area` in [0, 20]. Phase 2 cold-start counter changes unit to heating days (~57 live) **2026-11-13:** hold stays. Success criterion changed: forecast stability and bias on the first days at a new temperature low, judged on #119 with the #118 gate. |
| 106 | Cold-onset D Task 5: `heating_intensity_7d` ML feature behind `heating_intensity_feature` flag | lever for the 2025 Oct–Nov regime shift (16–26 kWh/day at 5–9 °C) | 1 day | gated — cold-day MAE −5 % vs 14-day baseline **2026-11-13:** demoted to low; same reason as #103. |
| 107 | Deficit fade in indoor projection (`fix/deficit-fade`): transient hour-0 room deficit decays (τ_r 2 h) to the room's same-setpoint steady deficit instead of being held 48 h; NaN climate rows skipped | removes the deficit-driven hourly swings of the tomorrow forecast (10-12 04 h replay jump 13.0 → 6.9 kWh); cold 24–47 h MAE −0.9 % (live-model replay, in-sample) | deployed alpha-28 (2026-10-19) | written gate STOP on total stability (18 % < 25 %) overridden: remaining swings are `heating_buffer_temp` (see memory); shipped ahead of the C verdict as bias-neutral (−0.15 kWh/day replay), C count continues |
| 108 | Extend the last climate state to the current hour when the entity is still available (`fetch_recent_climate`) — live projection treats > 2 h without a recorded change as stale (start at setpoint, deficit 0) while training forward-fills between changes | train/serve consistency; today 58 % of room-hours are stale but 0 of 4372 stale room-hours (since 10-01) had a deficit, so no measurable effect yet | ~½ day | gated: re-run the stale-with-deficit check after the first real cold spell; do as its own plan if non-zero **2026-11-13:** superseded if #120 (drop thermal-pressure features) passes. |
| 109 | **Deployed 2026-10-20 (S24 trailing 24 h buffer mean, ahead of the C verdict; fade framing superseded).** Original: Heating-buffer fade: live `heating_buffer_temp` near-term, decaying to a heating-curve-aware steady-state level instead of the latest reading being held 48 h; variant: train on the 1 h-lagged reading for the causal sign (see #109 detail) | removes the remaining tomorrow-forecast swings after #107 (replay: buffer pinned to 32 °C removed 4 of 5 jumps) | deployed alpha-29 (2026-10-20) | live check 10-21: hourly tomorrow \|Δ\| 1.33 → 0.65 kWh (≈ 0.2 excl. one weather-source flip, see #110), > 2 kWh jumps 23 % → 8 %; old models keep the raw path, rollback = previous model archive |
| 110 | Regime label train/serve leak + hard-label flip on weather-source switch | medium (tomorrow stability, feature-ranking honesty) | ~½ day backtest + 1 day | Part A (regime log) deployed alpha-33 2026-11-03; Parts B–C done 2026-11-05 (backtest: T with blocked folds chosen); deployed alpha-35 2026-11-05 (gate waived by user) — see #110 detail. Leading suspect for the post-A+B over-forecast **2026-11-13:** regimes off is worse (3 h +12.4 % on the audit holdout); close after the 11-19..21 stability check. |
| 111 | Horizon-consistent model shadow trial | — | — | **parked 2026-11-13** — reopen only on a 3 h-block win on #119 |
| 112 | Live vs replay gap | — | — | **closed 2026-11-13** — input difference; round trips exact |
| 113 | Phase 2 reconstruction + intervals on live-shaped frames; physics past tail | prevents a ≈ −5 kWh/day failure at activation | S–M | **blocker for Phase 2** |
| 114 | Block, daily and naive-guardrail accuracy sensors and kit columns | makes model changes judgeable | M | **high** — after #117 |
| 115 | Trailing level correction | — | — | **closed 2026-11-13** — forecast stays median-like |
| 116 | Retrain age off by the UTC offset | retrain timing | S | low |
| 117 | Store appliance-free actuals for MAE/bias | bias readings +0.8 kWh/day | S | **high — first** |
| 118 | Cold-day gate: −12 % to +6 % of daily consumption, ≥ 14 cold days | trustworthy gate | S | **high** — decided; script update pending |
| 119 | Backtest capture-frame mode | enabler for #105, #120 | M | **high** |
| 120 | Drop thermal-pressure features behind a flag | next-day swing −43 % on the holdout, accuracy neutral | S–M | medium — needs #119, 14-day shadow |
| 121 | Interval calibration on served errors; daily band method | hourly coverage 69–73 % → 79–84 % | S + open design | medium-low |
| 122 | Trim DHW override correction to the override hour | removes exact-0.0 forecasts; gain unmeasured | S | medium-low — measure on captures first |
| 123 | Training-data hygiene (NaN-lag rows, time-based look-backs, imputation, dead features) | inside the noise floor | M | low |

**Cold-onset sequencing (agreed 2026-10-17).** Every forecast-changing step waits for the one before it, so each effect stays attributable:
0. ✅ v0.11.16 on main (weather dtype fix, Rule 3) — released 2026-10-19 (cf28443), independent of everything below.
1. ✅ Implement A (#101) and B (#102) on separate branches; deploys held until step 2 (done 2026-10-17).
2. ✅ (PASS, pooled, 2026-10-27) C verdict: ≥ 3 cold target days on C + #109 smoothing, counted from the 2026-10-20 22:33 retrain (10-21 check on alpha-26 alone: not met); earliest ~10-25. D1 valve data gate already met (2026-10-21).
3. ✅ Deployed A + B together (alpha-32, 2026-10-27); re-baseline **not yet holding** (bias +2.79 kWh/day on 2026-11-03, +2.43 on 11-06, +2.54 on 11-08, +1.89 on 2026-11-12 (alpha-35 cold days −1.79, beats naive −9 %; `memory/project_deploy_accuracy_review_2026_11_08.md`); #110 shipped alpha-35 2026-11-05; rescore at ≥ 5 alpha-35 cold days). (A shifts actuals ≈ +0.5 kWh/day; B changes the `mae_*` pair population). **2026-11-13:** these readings are on actuals that include appliance energy (about 0.8 kWh/day too low, #117) and on fewer than 14 cold days. Re-read with the #118 gate once #117 is live and ≥ 14 cold days have accumulated.
4. Implement #105 (can overlap steps 2–3; needs the backtest harness and the COP-scale question settled first). **2026-11-13:** the harness requirement is #119.
5. Deploy #105 only after step 3's re-baseline holds under the #118 gate — never in the same window as A/B (first non-null `UA_eff` shifts `physics_kwh`). Judge it on forecast stability and bias on the first days at a new temperature low (#119), not on average MAE.
6. #106, #104, #103 after ≥ 14 cold days of baseline (or ≥ 14 valve-closed nights for #104). Gate #103 and #105 separately — both target the same residual, so don't default both on. **2026-11-13:** #103 and #106 are demoted to low and need a new justification first (see the table).
Open decision at the C verdict: physics release strategy (v0.12.0 opt-in vs further Rule 3 cherry-picks); alpha-26 likely does not port to main.

**Audit sequencing (agreed 2026-11-13).** 1. #117, then #114 and #118 (the yardstick). 2. #119 (tooling). 3. #113. 4. #120 with a 14-day shadow. 5. #121, #122, #123 as capacity allows. #105 stays on hold until 1 and 2 are done.

---

## Done

### Release History

| Version | Date | Highlights |
|---------|------|------------|
| v0.11.4-alpha | 2026-06-13 | 15-minute energy history cache (#85); strip partial day from clustering input (#86). 643 tests. |
| v0.11.0-alpha-16 | 2026-04-23 | Fix EV day exclusion from centroid fitting (#82). 535 tests. |
| v0.11.0-alpha-15 | 2026-04-22 | Regime logging improvements (#82 alpha-15 prep). 535 tests. |
| v0.11.0-alpha-14 | 2026-04-22 | Algorithmic correctness (#64–#69), code quality (#68, #71–#73), test coverage (#74–#78), documentation (#70, #80–#81). 535 tests. |
| v0.11.0 | 2026-04-17 | Daily Regime Clustering (optional module), K-Means 24h profiles, secondary regime predictor model |
| v0.10.0 | 2026-04-10 | Baseline mode (Stages 1–4), thermal/DHW intent, appliance signatures, scenario API, physics features (#55–#58), τ calibration, RC-ODE indoor projection |
| v0.9.0 | 2026-04-10 | Thermal modelling (#49–#52), occupancy (`people_home`), SHAP narrative, relative MAE sensors, rolling MAE persistence |
| v0.8.0 | 2026-03-31 | Solar/battery target correction, model versioning + rollback, CSV health checks, temperature bias-fade |
| v0.7.1 | 2026-03-24 | 404 DELETE fix, MQTT anomaly attrs, dashboard cards (anomaly + SHAP) |
| v0.7.0 | 2026-03-23 | 48 h weather features, anomaly detection, SHAP importance, prediction intervals, ApexCharts dashboard |
| v0.6.0 | — | MQTT Discovery (entity registry, area assignment, labels) |
| ≤v0.5.x | — | Core app, EV subtraction, lag features, adaptive retraining, holiday calendar |

### Completed Items

| # | Item | Done in |
|---|------|---------|
| 1 | Fix missing sunshine in Open-Meteo fallback | ≤v0.5.x |
| 2 | Add `temp_rolling_3d` to prediction horizon | ≤v0.5.x |
| 3 | Pre/post-holiday bridge day features | ≤v0.5.x |
| 4 | Cloud cover / solar irradiance feature | ≤v0.5.x |
| 5 | Fix training/prediction mismatch in rolling features | ≤v0.5.x |
| 6 | LightGBM early stopping + validation-set tuning | ≤v0.5.x |
| 7 | Log-transform the target | ≤v0.5.x |
| 8 | Adaptive retraining trigger | ≤v0.5.x |
| 9 | Cantonal public holidays | ≤v0.5.x |
| 11 | Additional lag: `lag_72h` | ≤v0.5.x |
| 12 | EV charge session probability feature | ≤v0.5.x |
| 13 | Prediction intervals as HA sensors | ≤v0.5.x |
| 14 | Intra-day actuals substitution | ≤v0.5.x |
| 17 | Setup checker sensor (`energy_forecast_setup_status`) | ≤v0.5.x |
| 19 | CSV cache: append-only writes | ≤v0.5.x |
| 20 | Config validation: warn when EV threshold ≥ charger_kw | ≤v0.5.x |
| 25 | Vacation / away flag (`is_away`) | v0.7.0 |
| 26 | Sub-energy sensors (`sub_energy_sensors`) | v0.7.0 |
| 27 | Short-horizon lags (`lag_1h`–`lag_12h`) | v0.7.0 |
| 28 | `num_leaves` hyperparameter sweep | v0.7.0 |
| 29 | Feature importance logging after training | v0.7.0 |
| 30 | CV fold std logging alongside mean | v0.7.0 |
| 31 | Per-hour-of-week NaN fill medians | v0.7.0 |
| 32 | Holiday `apply` → `np.searchsorted` vectorization | v0.7.0 |
| 33 | Day-of-year cyclical feature (`doy_sin` / `doy_cos`) | v0.7.0 |
| 34 | `hours_ahead` feature for horizon-aware prediction | v0.7.0 |
| 35 | Sub-sensor binary activity flag (`{prefix}_active_24h`) | v0.7.0 |
| 36 | Sub-sensor rolling run count (`{prefix}_runs_7d`) | v0.7.0 |
| 37 | MQTT Discovery for entity registry | v0.6.0 |
| 38 | Full 48 h weather forecast features | v0.7.0 |
| 39 | Anomaly detection on forecast residuals | v0.7.0 |
| 41 | Rolling accuracy history sensors (7d / 30d MAE) | v0.7.0 |
| 42 | SHAP feature importance per prediction | v0.7.0 |
| 43 | ApexCharts / Lovelace config snippet | v0.7.0 |
| 44 | Model versioning — keep last N, rollback | v0.8.0 |
| 45 | CSV health checks + gap repair | v0.8.0 |
| 46 | Fix 404 DELETE spam in `_cleanup_legacy_states()` | v0.7.1 |
| 47 | Anomaly binary sensor MQTT attrs + discovery fix | v0.7.1 |
| 21 | Occupancy feature (`people_home`) | v0.9.0 |
| 23 | Solar PV target correction (B1) | v0.8.0 |
| 49 | Exponentially weighted moving average temperature (`temp_ewma_24h/72h`) | v0.9.0 |
| 50 | Rolling accumulated heating degree-hours (`heating_deg_sum_24h/168h`) | v0.9.0 |
| 51 | Temperature rate of change (`temp_delta_1h/24h`) | v0.9.0 |
| 52 | Temperature lag features (`temp_lag_24h/168h`) | v0.9.0 |
| 53 | "Why today?" SHAP narrative attribute | v0.9.0 |
| 54 | Relative MAE sensors (7d / 30d) | v0.9.0 |
| 55 | Verified Passive Decay — τ calibration (OLS passive-cooling windows) | v0.10.0 |
| 56 | Solar-Compensated Thermal Pressure (`thermal_pressure_net`) | v0.10.0 |
| 57 | Wind-Driven Infiltration Feature (`infiltration_pressure`) | v0.10.0 |
| 58 | Humidity-Aware Defrost Proxy (`defrost_risk`) | v0.10.0 |
| 60 | Calibrated default thermal time constant (`DEFAULT_TAU = 12 h`) | v0.10.0 |
| 61 | Daily Regime Clustering (`regime_kwh`) | v0.11.0 |
| 63 | Fix RegimePredictor overfitting (OOB score + constraints + occupancy features) | v0.11.0-alpha-8 |
| 59 | Relaxed τ calibration — quality-scored windows replace hard daytime/solar filters | v0.11.0-alpha-9 |
| 62 | Adaptive Regime Selection (Auto-K) — inertia elbow, K ∈ [2, 8], `regime_count: 0` | v0.11.0-alpha-10 |
| 64 | CQR calibration: random holdout split (rng seed 42) for valid exchangeability guarantee | v0.11.0-alpha-14 |
| 65 | RegimePredictor: TimeSeriesSplit CV logged alongside OOB; warning uses TSCV mean | v0.11.0-alpha-14 |
| 66 | Inertia normalization: bail out to k_lo when range < 1e-6 (homogeneous data guard) | v0.11.0-alpha-14 |
| 67 | Regime label ffill in prediction path — matches training semantics for gap days | v0.11.0-alpha-14 |
| 68 | `strip_tz()` moved to `const.py` as shared utility; weather.py and energy_forecast.py deduped | v0.11.0-alpha-14 |
| 69 | EWMA temperature resets at weather gaps > 2h via NaN sentinels before `.ewm()` | v0.11.0-alpha-14 |
| 70 | Physics feature scaling constants (0.01, 10.0) documented with empirical basis | v0.11.0-alpha-14 |
| 71 | Sub-sensor quality demoted to "fair" when energy_cov > 0.5; CoV stored in signature dict | v0.11.0-alpha-14 |
| 72 | `get_scenario` validates schedule keys and HH:MM format; drops invalid entries with WARNING | v0.11.0-alpha-14 |
| 73 | `__version__` in `__init__.py` is single source of truth for MQTT `sw_version` | v0.11.0-alpha-14 |
| 74 | Gaussian noise in `_make_energy_df()` — KMeans ConvergenceWarnings reduced from 18 → 9 | v0.11.0-alpha-14 |
| 75 | Pickle corruption recovery test for `clusterer.pkl` | v0.11.0-alpha-14 |
| 76 | K=1 fallback test — homogeneous data hits inertia bail-out | v0.11.0-alpha-14 |
| 77 | `train()` edge cases: empty DataFrame, below MIN_TRAINING_ROWS, constant values | v0.11.0-alpha-14 |
| 78 | Network failure tests for `fetch_open_meteo` (404, 500, Timeout, ConnectionError, bad JSON) | v0.11.0-alpha-14 |
| 79 | Timezone-aware fixture audit — confirmed existing tests use naive timestamps correctly; no changes needed | v0.11.0-alpha-14 |
| 80 | `find_optimal_k()` docstring fully documents normalization, bail-out, smoothing, tolerance band, OOB note | v0.11.0-alpha-14 |
| 81 | `_project_indoor_temps()` stale-sensor threshold already documented — confirmed, no change needed | v0.11.0-alpha-14 |
| 82 | Fix EV contamination in regime clustering — EV days excluded from `DailyProfileClusterer.fit()` | v0.11.0-alpha-16 |
| 89 | Dedup physics sensor history fetches — DHW tank temp and room thermostat temp now reuse the ML pipeline's already-fetched data instead of redundant HA history API calls | Unreleased |
| 90 | Fill gaps in `_SHAP_FEATURE_LABELS` dashboard narrative dictionary — 5 missing labels added, 3 stale untagged `lag_*h` entries replaced with `_tgated` equivalents | Unreleased |
| 91 | Daily update-check + notification — compares `__version__` against latest GitHub release tag daily at 09:00, fires `persistent_notification` when `main`-track users are behind; `update_check_enabled` config flag | v0.12.0-alpha-10 |
| 92 | Temperature-based UA_eff calibration window — tiered eligibility instead of a hardcoded month list; tier 1 (`heating_sub_meter_sensor`) activated live 2026-09-26 | v0.12.0-alpha-19 |
| 95 | Dedupe hourly excluded-range escalation/malformed-CSV warnings — `_warn_once()` in `ha_data.py` fires WARNING once per condition per AppDaemon process lifetime via `self._excluded_range_warned`, then INFO on repeats; resets on restart. **Follow-up correction:** alpha-11 only demoted severity — `filter_excluded_ranges()` still logged an INFO line every hourly cycle for a range matching zero rows (e.g. once it ages out of the data window), which was the actual source of ongoing noise reported live. Now gated on `n_dropped > 0`; a no-op range logs nothing. | v0.12.0-alpha-11 |
| 97 | Open-Meteo archive retry/backoff on transient 5xx (GitHub #21) | v0.12.0-alpha-24 / main v0.11.15 |
| 98 | Daily-mean heating-season projection — `heating_active` projected once per calendar day from the forecast daily mean instead of hourly outdoor temps (which projected ON on every cold night while heating stayed off); per-installation thresholds learned at retrain from a daily heating label (sub-meter > switch > defaults); meter heating-day label as training feature; projection no longer requires climate entities and is passed to `get_scenario`; `heating_season` attribute on `sensor.energy_forecast_today`; example daily HA automation in `docs/examples/` | v0.12.0-alpha-23 |
