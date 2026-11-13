# Forecast improvement audit — 2026-11-13

Audit of the live consumption forecaster (dev v0.12.0-alpha-39). Nothing was deployed, committed or changed under
`apps/`. Experiments ran in a session scratchpad; the only repo writes are this file and the memory notes.

All numbers are kWh/h for MAE and kWh/day for bias unless stated. "Day-ahead" means the forecast issued 24–29 h
before the target hour, which matches what `mae_7d` scores (25 h).

## 1. Verdict

1. **On hourly MAE the current code is at its day-ahead noise floor.** Replayed on Oct 1–Nov 3 it scores 0.320;
   a forecast with *perfect daily totals* and the usual hourly shape would score 0.326. Seven model-structure arms
   were tried and none beat it outside the noise floor (±1 % hourly, ±2 % on 3 h blocks).
2. **The metric is the bigger problem.** Hourly MAE rewards under-forecasting (scaling by 0.8 improves it 6 % while
   bias goes to −2.5 kWh/day), and the live "actuals" include appliance energy the model excludes, so every bias
   gate reads 0.8 kWh/day too low. The consumer reads 3 h blocks with P10/P90, where the model already beats the
   naive mean by 10–16 %.
3. **Recoverable headroom, in order:** (a) about −17 % on 3 h-block error if the DHW *hour* were known, which only
   `ha-energy-manager` can supply because it decides that hour; (b) 43 % smaller forecast swings, at no accuracy
   cost, by dropping the thermal-pressure features; (c) hourly interval coverage from 70 % to about 80 %.
4. **Heat-pump cycle timing is not forecastable beyond 2 h** and is worth only 5 % at block level. The earlier
   "only route with real headroom" does not hold for the heat pump.
5. **Single best next move:** fix the scoring (appliance-free actuals, 3 h-block and daily metrics, naive
   guardrail) before any further model work. Separately, do not enable Phase 2: its reconstruction fails on every
   live-shaped input.

### Owner decisions after review (2026-11-13)

- **DHW:** `ha-energy-manager` knows the DHW hour only a few hours ahead. Candidate C2 and Plan 3 are therefore
  dropped; verdict point 3(a) is not reachable. The remaining DHW item is C6 (ROADMAP #122).
- **Point forecast stays median-like.** C7's winter under-forecast (−2.4 kWh/day) is accepted as a known cost.
- **Cold-day gate:** pooled cold-day bias within −12 % to +6 % of mean daily consumption over ≥ 14 cold days, on
  appliance-free actuals; fewer days is inconclusive. Replaces [−3, +1] kWh/day (ROADMAP #118).
- Roadmap follow-ups: #117–#123, plus changes to #83, #87, #88, #103, #105, #106, #108, #110–#116.
- **Agreed order of work:**
  1. #117 (appliance-free actuals), then #114 (block, daily and naive-guardrail metrics) and #118 (new gate).
  2. #119 (backtest capture-frame mode).
  3. #113 (Phase 2 reconstruction and intervals).
  4. #120 (drop thermal-pressure features) with a 14-day shadow.
  5. #121, #122, #123 as capacity allows.

  #105 stays on hold until steps 1 and 2 are done. This supersedes the plan order in section 7.

## 2. Method

**Frame.** The tracked kit uses a lean frame (energy, weather, EV) and scores point forecasts at two origins. It
cannot express intervals, sub-meter/climate/DHW features, Phase 2 or model-selection changes, so I built a scratch
walk-forward on the live captures instead: the real `train()` / `predict()` driven by the 2026-11-13 retrain
capture, truncated at each cutoff. Daily retrain at 18:00, origins every 6 h, 48 h horizon, perfect (archive)
weather. That harness has since been promoted into the kit as `backtest/capture_frame.py`,
`backtest/walkforward.py` and `backtest/wf_score.py` (ROADMAP #119); the promoted code reproduces the audit's
forecasts exactly (max difference 0.0 on the first days of the autumn window, base and jitter arms).

**Reproduction (done before any comparison).**

| Check | Result |
|---|---|
| `backtest.capture_replay`, predict round trip | 113 / 113 PASS (max diff 0) |
| `backtest.capture_replay`, train round trip, seeded from `.pre/` | 3 / 3 PASS (0.0000 %), 1 SKIP (no seed) |
| Scratch frame, serve inputs synthesized from history vs live | mean abs diff 0.007 kWh/h |
| Scratch frame, retrain at the 11-13 capture time vs live model | identical (232 trees, diff 0) |
| Live SRG weather vs perfect weather, same live models, 4 days | MAE 0.347 vs 0.343 (day-ahead 0.367 vs 0.360) |

The train round trip passing is new since the 11-11 note. Captured inputs fully determine the live forecast.

**Windows and guardrails (fixed before running arms).** Dev: WINTER 2026-01-12..03-15, SEP 09-01..09-30, ONSET
10-01..11-03. Holdout: 11-04..11-12, scored once. Guardrails: hourly MAE, 3 h-block MAE, cold-day pooled bias in
[−3, +1], P10–P90 coverage (80 %), and the naive same-hour mean (days −2..−8). CIs are block bootstraps by ISO week.
The noise floor is the same model retrained one hour earlier ("jitter").

**Caveats.** Perfect weather (measured cost of real SRG weather: 1–2 %). WINTER has no sub-meter, climate or tank
data, as live had none then. The holdout is 9 days with 7 cold days; its CIs are wide. Physics state is the
2026-11-13 state at every cutoff (`physics_kwh` has zero gain, so this does not affect the forecast).

## 3. Error decomposition

### 3.1 Live as issued versus current code

Dev overlap 10-12..11-03, 540 h at 25 h lead, scored against the model's true target (appliances removed).

| Forecast | Hourly MAE | Bias |
|---|---|---|
| Live as issued (older code) | 0.470 | +2.66 |
| Current code, replayed | 0.377 | +0.08 |
| Naive same-hour mean | 0.362 | −1.88 |

The fixes shipped since October explain about 20 % of the live error on these days. On the live captures
(11-09..11-13, current model, exact inputs) live scores 0.35 against 0.465 for naive.

### 3.2 Current code by window (day-ahead slice)

| Window | Days | Hourly MAE | 3 h MAE | Daily abs error | Bias | Cold-day bias | Coverage | Naive hourly / 3 h |
|---|---|---|---|---|---|---|---|---|
| WINTER | 50 | 0.578 | 1.009 | 4.63 | −2.37 | −2.23 | 79.5 % | +8.3 % / +16.0 % |
| SEP | 29 | 0.166 | 0.399 | 1.50 | −0.28 | n/a | 75.6 % | −3.7 % / −4.3 % |
| ONSET | 34 | 0.320 | 0.550 | 1.53 | −0.14 | +0.49 | 69.5 % | −2.5 % / +3.5 % |
| HOLDOUT | 9 | 0.447 | 0.789 | 1.79 | +1.21 | +1.76 | 73.1 % | +5.5 % / +13.2 % |

"Naive" columns give naive relative to current code (positive means naive is worse). Pooled over dev, naive is
+4.4 % hourly (CI −0.2..+8.6), +10.2 % on 3 h blocks (+1.5..+18.7), +14.5 % on 6 h blocks, +27 % on daily |bias|.
The holdout cold-day bias of +1.76 is outside the [−3, +1] band; the jitter arm reads +1.57 on the same 7 days.

### 3.3 Where the error sits (ONSET, current code)

| Segment | Share of hours | Actual | MAE | Bias per hour | Share of abs error |
|---|---|---|---|---|---|
| No heat-pump load | 73 % | 0.26 | 0.220 | +0.17 | 50 % |
| Heating low (0.1–0.5) | 6 % | 0.65 | 0.323 | +0.07 | 6 % |
| Heating high (> 0.5) | 18 % | 1.20 | 0.526 | −0.48 | 29 % |
| DHW cycle (> 0.3) | 3 % | 2.32 | 1.694 | −1.69 | 15 % |

This is smearing: the forecast spreads cycle energy over all hours, so it over-forecasts quiet hours and
under-forecasts cycle hours. It is not a level error (overall bias −0.14).

By temperature, 45 % of absolute error falls on 6–9 °C days (MAE 0.435); by hour, 12–14 h is the worst block (0.576,
22 % of error, DHW hours). Weekday and weekend do not differ (0.320 vs 0.331). The DST fall-back day is not an
outlier. Error is flat across leads: 0.29–0.35 from 0 h to 47 h in ONSET, so intra-day updates add no skill.

### 3.4 Noise floor

ONSET, same hours, hourly MAE:

| Forecast | Hourly MAE |
|---|---|
| Current code | 0.323 |
| Oracle: perfect daily total × recent hourly shape | 0.326 |
| Oracle: perfect 6 h totals, flat inside | 0.321 |
| Oracle: perfect 3 h totals, flat inside | 0.266 |

**Estimate:** the day-ahead hourly floor is about 0.30–0.32 in this window. Getting below it needs knowledge of
timing inside 3 h blocks. Current code is within 0–7 % of it.

Oracle swaps of one component (forecast's expected component replaced by the truth):

| Oracle | Hourly MAE | 3 h-block MAE |
|---|---|---|
| Perfect DHW, hourly | −12 % | −18 % |
| Perfect heating, hourly | −29 % | −5 % |
| Perfect heating, 3 h totals | 0 % | −5 % |

### 3.5 Timing feasibility

**Heat pump (32 heating days).** On in 38 % of hours; ON runs median 1 h, 5.7 starts per day. Hourly residual
autocorrelation is −0.46 at lag 1, +0.30 at lag 2, about 0 from 6 h. Predicting "on in hour t" from buffer
temperature and recent heating energy, test AUC by lead: 1 h 0.88, 2 h 0.72, 3 h 0.63, 6 h 0.61, 24 h 0.64; hour and
temperature alone give 0.67. There is no timing signal from 3 h on.

**DHW.** 49 cycles in 105 days, 0.82 kWh/day, 99 % of DHW energy in cycle hours. Whether a cycle happens tomorrow is
predictable from the tank temperature at 18:00 the evening before: 88 % accuracy on Oct 1–Nov 3 with a threshold
fitted on Aug–Sep (AUC 0.89–0.94). A rule built on that changes 3 h-block error by −0.8 % (CI −6.5..+4.7): the day
is not enough, the hour is. Start hours moved from 12 h (Aug–Sep) to 10–14 h (Oct) to 16 h (Nov). 73 % of cycle
days carry an energy-manager commit, made a median of 5.3 h (comfort boost) or 6 h (legionella) before the hour.

## 4. Findings from the audit

Grades: **Confirmed** = predicted something specific and it held; **Supported**; **Speculative**.

| # | Finding | Evidence | Grade |
|---|---|---|---|
| F1 | Live MAE/bias "actuals" include dishwasher, washer and tumbler energy; the model's target excludes them (`baseline_mode`). | Difference correlates 0.995 with the appliance sub-meters, 771 / 772 hours; 0.79 kWh/day. Live bias over 32 days is +2.26 against the true target, not +1.47. MAE barely moves (0.474 → 0.468). `energy_forecast.py:2306-2326`. | Confirmed |
| F2 | Phase 2 reconstruction fails on every live-shaped frame and publishes the bare residual. | `predict()` adds a baseline indexed on the whole forecast frame to 48 predictions: `operands could not be broadcast together with shapes (170,) (48,)`, falls back to ML-only. 26.4 instead of 36.4 kWh over 48 h. Works with a 48-row frame. `model.py:1395-1405`. `predict_intervals()` never adds the baseline. | Confirmed |
| F3 | Hourly P10–P90 under-covers outside winter. | 59 % on live captures (4 days), 69.5 % ONSET, 75.6 % SEP, 79.5 % WINTER, 73 % holdout. The conformal correction is 0.009 in log space; in-sample coverage is 86 % (93 % on the last 30 days). The calibration split is a random 15 % of training rows with real 1 h lags. | Confirmed |
| F4 | Block and daily bands are sums of hourly quantiles. | 3 h blocks cover 85–97 %; P90 exceeded 10.9 % (ONSET), 13.8 % (SEP), 1.3 % (WINTER), 5.6 % (holdout). Daily bands cover 97–100 % at 19–43 kWh width. `energy_forecast.py:3227-3268`. | Confirmed |
| F5 | The exact-0.0 forecasts are the negative leg of a pending DHW override, floored. | With the 11-14 12:00 boost pending, each update adds +1.02 at 11-14 12:00 and +1.02 at 11-15 11:00, and subtracts 0.8–1.05 at an hour that drifts (21:00, 07:00, 06:00, 08:00, 09:00). 4 of 7 affected updates published a zero. The 11-09 05:00 zero fits (stored one day after the 11-07 05:04 commit, boost still pending). | Confirmed |
| F6 | One extra hour of training data moves the forecast by several percent; at a new temperature low by far more. | Truncated history with one more row: 190 → 235 trees, 48 h total +7.7 % (range +3..+14). The 11-13 retrain moved the 2 °C forecast from 24.7 to 35.7 kWh on identical inputs. Daily retrains switch between 16 / 31 / 63 leaves and 3 / 4 regimes. | Confirmed |
| F7 | Cold conditions carry almost no training weight. | 60-day half-life: Nov–Mar holds 3.2 % of weight; rows ≤ 3 °C 2.3 % (about 56 effective rows); ≤ 0 °C 0.55 % (about 13). | Confirmed |
| F8 | Missing hours cascade through lag columns and delete training rows. | 1 995 of 9 139 rows dropped: 320 warm-up, 1 675 with a NaN lag, 0 other. 23–27 % of each winter month. Zero since September. `model.py:731`. | Confirmed |
| F9 | Served thermal-pressure features bear no relation to the trained values and swing the forecast. | Served-vs-trained correlation −0.06 at 24–47 h and −0.03 at 1–5 h. Forecast for 11-14 flips 25.0 ↔ 31.4 kWh with `thermal_pressure` 0.00 ↔ 0.47. | Confirmed |
| F10 | Weather-memory features use row shifts across a frozen tail; they break when the model is older than 3 days. | Ageing only the tail: exactly 0 effect up to 2 days, then 0.3–0.5 kWh/day mean abs change (max 1.7), no consistent sign. `model.py:1109-1119`, `3250-3282`. | Confirmed, small |
| F11 | Several top features are degenerate or inconsistently imputed at day-ahead lead. | `rolling_std_24h` (6.2 % gain) served as exactly 0.000, trained mean 0.48. `lag_1h/2h/6h/12h` filled with same-hour means (correlation 0.33–0.37 with truth); `lag_24h` and heating `lag_24h` filled with medians (0.001 served vs 0.185 trained). DHW tank features constant 50 °C (8 % gain). | Confirmed; effect known (#111) |
| F12 | `regime_kwh` (21 % gain) is served 36 % above the value trained on for the same hours. | 0.390 vs 0.286 on 11-10..11-13, correlation 0.93. Four days in a cooling trend; centroids differ between the serving and the later model. | Supported |
| F13 | SRG weather differs from the archive weather used in training. | SRG wind +45 % (10.7 vs 7.4 km/h); SRG sunshine 0–40 % of archive. Accuracy effect 1–2 %. | Supported |
| F14 | The point forecast sits between median and mean. | In-sample predicted / actual mean 0.906 (all rows), 0.95 (last 30 days); 58 % of actuals below the forecast. WINTER bias −2.37. | Supported |
| F15 | The 08 h over-forecast since 10-28 is mostly model-side. | Live 1.67, current code replayed 1.28, actual 0.81 (7 mornings). | Supported |
| F16 | 13 features have zero gain. | `physics_kwh`, `hours_ahead`, `heating_active`, `is_away`, `people_home`, three holiday features, `likely_ev_hour`, `season`, `month_cos`, two sub-meter features. `away_df` holds 10 days, all zero; 5 public holidays in the span. | Confirmed |
| F17 | `scripts/pull_ha_data.py` does not pull `.pre/`. | Sidecars say `pre_state: true`; the directories arrived only with a separate read-only pull. | Confirmed |

On the live-vs-replay gap (#112): it is an input difference, not code or libraries (round trips exact). Weather
source accounts for 1–2 %. The exact zeros are F5. F10 is a real live-only skew but small. The 08 h item is F15.

## 5. Ranked candidates

Changes are relative to current code on the day-ahead slice. "Floor" is the jitter arm.

### C1. Fix the yardstick (extends #114)

- **Hypothesis:** model changes are being judged on a metric that misranks them and on actuals that do not match
  the target.
- **Evidence:** F1 (Confirmed). Scaling current code by 0.8 improves hourly MAE (0.320 → 0.302 ONSET, 0.447 →
  0.428 holdout) while bias goes to −2.5; 3 h MAE is best at 0.9; daily error is best at 1.0 (1.05–1.1 in
  WINTER). Naive ties on hourly MAE and loses by 10–16 % on blocks. The energy manager consumes 3 h blocks
  (`soc_predictor.py`, `price_timing_allocator.py:797`).
- **Expected gain:** no accuracy change. Bias gates shift by +0.8 kWh/day (the 11-12 reading of +1.89 becomes about
  +2.7; the alpha-35 cold-day mean of −1.79 becomes about −1.0).
- **Cost / risk:** small. `mae_*` sensors step once at deploy.
- **Falsified if:** block and daily rankings of candidates match hourly rankings on the next 14 days.
- **Needs the live-capture frame:** no.

### C2. Give the forecaster the DHW hour, or take DHW out of the target (cross-repo)

- **Hypothesis:** the DHW cycle hour is the energy manager's decision; the forecaster cannot infer it day-ahead.
- **Evidence:** perfect DHW knowledge gives −12 % hourly and −17..−18 % on 3 h blocks at three origins (Supported,
  34 days). Knowing only the day gives −0.8 % (CI −6.5..+4.7). Commits arrive a median of 5–6 h ahead.
- **Expected gain:** up to −17 % on 3 h-block MAE and −12 % hourly, scaled by how early and how accurately the
  energy manager knows its plan. Bias and coverage unchanged.
- **Cost / risk:** medium, two repos. Risk of a feedback loop if the manager plans DHW from a forecast that
  already contains its plan.
- **Falsified if:** the manager's planned hour, logged the evening before, matches the realised cycle hour on
  fewer than about 60 % of cycle days.
- **Needs the live-capture frame:** yes, to score on real commits.

### C3. Drop the thermal-pressure feature family

- **Hypothesis:** these six features add swings and no skill, because what is served is unrelated to what is
  trained (F9).
- **Evidence:** pooled dev hourly +0.2 % (−1.0..+1.3), 3 h −0.4 % (−2.3..+1.1); ONSET alone hourly +2.3 %
  (0.0..+5.2). Holdout hourly −0.8 %, 3 h −1.3 %, cold-day bias +0.95 against +1.76. Spread of the next-day total
  across the previous day's four forecasts: SEP 1.25 → 0.57 kWh, ONSET 1.41 → 1.33, holdout 2.44 → 1.39 (−43 %).
  Grade: Supported.
- **Expected gain:** hourly and block MAE unchanged within the floor; forecast swings −40 to −55 % when rooms sit
  near their setpoints; cold-day bias possibly lower.
- **Cost / risk:** small. ONSET hourly reading is at the edge of the floor; WINTER could not be tested (no
  climate data then).
- **Falsified if:** on 14 new days hourly or 3 h MAE is worse by more than 2 %, or swings do not shrink.
- **Needs the live-capture frame:** yes (the lean kit frame holds these features constant).

### C4. Recalibrate hourly intervals on served errors; replace the daily band

- **Hypothesis:** calibration on training-time rows certifies nothing about served forecasts (F3).
- **Evidence:** a trailing conformal correction on day-ahead errors (same log-space form as today) gives ONSET
  68.9 → 79.0 %, SEP 77.1 → 81.0 %, WINTER 81.8 → 82.8 %, holdout 73.1 → 83.8 %, at 3–11 % more width. The upper
  tail stays under-covered (12–17 % above P90).
- **Expected gain:** hourly coverage to about 80 %. Point metrics unchanged.
- **Cost / risk:** small for hourly. Block P90, which the manager reserves on, is already about right in autumn
  and too high in winter; my empirical block and daily alternatives were unreliable (69–84 % and 58–85 % across windows), so those
  need a better method than I tested.
- **Falsified if:** coverage on 14 new days is outside 75–85 %.
- **Needs the live-capture frame:** no; `pred_history` plus stored bounds would do.

### C5. Fix Phase 2 before it can activate (#113)

- **Evidence:** F2 (Confirmed). With the reconstruction corrected, Phase 2 is still not ready: WINTER 3 h −3.1 %
  (−7.1..+1.2) and bias −0.01, but ONSET hourly +13.8 % and cold-day bias +4.6, SEP +11 %, holdout bias +3.8.
- **Expected gain:** none now; prevents a −5 kWh/day failure at activation.
- **Needs the live-capture frame:** a test with a 170-row frame is enough.

### C6. Trim the override correction to the override hour

- **Hypothesis:** the second positive leg and the drifting negative leg add error, because the ML forecast holds
  no discrete reheat at those hours (F5).
- **Evidence:** mechanism Confirmed; accuracy cost not measured (no affected target hours with actuals yet).
  Grade for the gain: Speculative.
- **Falsified if:** on captures after 11-15, published forecasts score no worse than model output plus only the
  override-hour leg.
- **Needs the live-capture frame:** yes.

### C7. Winter level and cold-onset stability

- **Hypothesis:** the log-space fit under-forecasts the mean in winter (F14), and the cold response rests on too
  few effective rows (F6, F7).
- **Evidence:** WINTER bias −2.37; daily error is best at scale 1.05–1.1. A mean objective removes it (+0.46) but
  over-forecasts autumn (ONSET +1.88, cold-day +3.6; holdout +3.65). No tested arm fixes the cold-onset jumps.
  Grade: Supported for the problem, no solution found.
- **Next:** this is what #105 (daily UA_eff) should be judged on: forecast stability and bias on the first days
  at a new temperature low, not average MAE.

### C8. Hygiene (small, low risk)

Keep NaN-lag rows (F8); make weather look-backs time-based or extend the live tail (F10); impute short and long
lags the same way (F11); remove dead features (F16); pull `.pre/` (F17). Expected accuracy effect is inside the
noise floor.

## 6. Rejected or not worth pursuing

| Idea | Result | Reason |
|---|---|---|
| Timing-aware heat-pump forecast | AUC 0.63 at 3 h lead; 3 h-block ceiling −5 % | No day-ahead signal; irrelevant for blocks |
| DHW rule from tank temperature | −0.8 % on 3 h blocks (−6.5..+4.7) | Day is predictable, hour is not |
| Fixed hyper-parameters | hourly −0.6 %, 3 h +0.2 %; swings larger | Inside the floor |
| Averaging the last 3 daily models, or model + jitter twin | −0.3 % / −0.1 % hourly | Inside the floor |
| Blend with naive mean, or with a degree-day profile | −1.3 % / −0.8 % hourly, CIs span 0 | Inside the floor; adds negative bias in autumn |
| Degree-day profile alone | WINTER hourly +6.2 %, 3 h +10 % | Worse |
| Regimes off | 3 h +3.1 % dev, +12.4 % holdout; WINTER bias −3.25 | Worse. Repeated despite the #110 backtest because `regime_kwh` is the top feature with a measured serve skew (F12) |
| Mean objective (raw L2) | ONSET +9.3 % hourly, +16.9 % 3 h; holdout +19 % 3 h | Worse outside winter. Repeated despite the "tried" list because it had only been judged on hourly MAE |
| Phase 2 residual-on-physics, corrected | see C5 | Worse in autumn |
| Weekly instead of daily retrain | dev −1.2 % hourly (ns); holdout +2.5 % hourly, +15.6 % on 6 h | Mixed; swings smaller (1.41 → 0.88 kWh) but no case to change |
| Weather source | 1–2 % | Not a lever |
| Horizon-consistent model (#111) | not re-run | Already tested; F11 explains why it gains little in winter |

## 7. Proposed experiment plan (proposal only; needs owner approval)

### Plan 1 — Scoring and metrics (C1)

- **Base branch:** `dev`.
- **Files:**
  - `apps/energy_forecast/energy_forecast.py`: subtract the `baseline_mode` sub-sensors before storing
    `_actuals_history` (and fix the comment at 2306-2311); add 3 h-block MAE, daily abs error, daily bias and a
    naive same-hour-mean MAE to the published attributes.
  - `backtest/score.py`, `backtest/runner.py`: add 3 h-block and daily columns and a `naive` pseudo-arm to the
    summary.
  - `scripts/check_cold_onset_gate.py` (local): read the corrected actuals.
  - `tests/`: actuals equal the training target on a day with an appliance run; block metrics on a fixed vector.
  - `CHANGELOG.md`, `ROADMAP.md` (#114), `README.md` (new attributes).
- **Success criteria:** stored actuals match the captured training target on ≥ 99 % of hours over 7 days; new
  sensors populate; suite green.
- **Holdout:** none needed (no forecast change). Re-read the A+B gate on corrected actuals afterwards.

### Plan 2 — Drop thermal-pressure features (C3)

- **Base branch:** `dev`, after Plan 1.
- **Files:**
  - `apps/energy_forecast/model.py`: remove the six names from `_FEATURES_BASE` behind a `train()` flag, default
    off; keep the serve path able to load old models.
  - `backtest/experiments/drop_thermal_pressure.py`: arm on the capture frame (needs the frame mode from the
    parked live-equivalent spec; until then the scratch walk-forward from this audit is the reference).
  - `tests/`: feature list with the flag on; old pickle still predicts.
  - `CHANGELOG.md`, `ROADMAP.md`.
- **Success criteria (fixed now):** on ≥ 14 new days after 2026-11-13, hourly and 3 h MAE non-inferior at 2 %;
  spread of the next-day total across the previous day's forecasts at least 25 % lower; cold-day bias not higher.
- **Holdout:** the next ≥ 14 days, scored once, with the jitter arm as floor. Shadow first: train both, log both.

### Plan 3 — DHW plan sharing (C2), feasibility step only

- **Base branches:** `dev` in both repos.
- **Files:**
  - `ha-energy-manager`: log (not act on) the DHW hour it expects for tomorrow at its evening and dawn planning
    points, with a timestamp.
  - `ha-energy-forecast`: no forecast change; a scratch analysis joins that log with the DHW sub-meter.
- **Success criteria:** planned hour within ±1 h of the realised cycle on ≥ 60 % of cycle days over 3 weeks, and
  planned-but-not-run days below 20 %. Only then design the forecast change (pass the plan as
  `dhw_schedule_override`, or remove DHW from the target and let the manager add it).
- **Holdout:** the 3 weeks of logs are the test; the replay with the logged plan is scored once on them.

C4 and C5 are small enough to ride with Plan 1 if wanted; C5 needs one regression test with a 170-row frame.

## 8. Open questions for the owner

1. Does `ha-energy-manager` know tomorrow's DHW hour when it plans (evening or dawn), or only a few hours ahead?
   C2 depends on this.
2. Should the forecast aim at the mean (energy totals) or stay median-like? Winter under-forecasts by 2.4 kWh/day;
   the reservation logic already uses `max(P90, P50)`.
3. Is the [−3, +1] cold-day band still the right gate once actuals are corrected by +0.8 kWh/day? Current code
   reads +0.49 (ONSET) and +1.76 (holdout, 7 days).
4. Is anything downstream using the daily low/high band? It covers 100 % at 19–43 kWh width.
5. Daily retrains currently happen (deploys plus adaptive retrain). Weekly gave the same accuracy on dev and
   smaller swings, but a worse 6 h-block result on the holdout. Leave as is?
6. May the pull script fetch `.pre/` (F17), and may the scratch walk-forward be promoted into `backtest/` as the
   capture frame mode, replacing the parked spec's plan 2?
