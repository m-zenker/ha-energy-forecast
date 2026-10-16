# Cold-onset forecast fixes — index (2026-10-15)

**Trigger:** live `mae_7d` 0.477 (0.196 on 10-06), relative 115 %, since heating-season onset 10-08.
Findings: `memory/project_cold_onset_overforecast_2026_10_15.md`.

**Base branch for all:** `dev` (v0.12.0-alpha-24 + docs). Each sub-plan (and each gated task in D) gets its own branch, is rebased on `dev` after the previous alpha lands, and ships via deploy-agent Workflow A as its own alpha. Workflow A deletes the branch after merge, so follow-up tasks (C Task 4, D Tasks 2–5) start fresh branches. Run `@changelog-writer` **after** the rebase (CHANGELOG conflicts every time otherwise).

**Order and spacing:**
1. **D Task 1** (`heating_power_request` cache). Zero forecast impact; it starts the data clock (≥ 14 valve-closed nights needed, HA keeps ~10 days).
2. **C Task 1** (dominant error). Then **≥ 3 full cold days** of verification before the next forecast-changing deploy.
3. **A + B** can ship in the same window. A forces a retrain (target changes); re-baseline after it (see Verification).
4. **C Task 4, D Tasks 2–5**: flag-guarded. Code may merge any time; flags are only flipped after their gate *and* the verification rules below.

τ (D Task 3) is not on the critical path: after C, τ only affects heating-off hours and warm rooms cooling down.

| Plan | Branch | What | Evidence | Expected effect |
|---|---|---|---|---|
| [A](2026-10-15-cold-onset-a-counter-reset.md) | `fix/counter-reset-increments` | Reset-aware hourly kWh for cumulative sub/correction sensors + one-off cache repair | `solaredge_i1_b1_energy_export` resets ~06:50 and ~20:55 daily; hef zeroes the reset hour → actual ≈ 0 at 06h/20h every day, ~0.2–0.4 kWh lost per reset | ≈ −0.05 `mae_7d`; clean target + lag features at 06/07/20/21h |
| [B](2026-10-15-cold-onset-b-ev-sensor-trust.md) | `fix/ev-sensor-trust` | Stop treating wallbox-sensor EV hours as contamination (clustering, `likely_ev_hour`, MAE tracking) | 10/14 Oct days dropped from regime clustering; EV hours ±1 dropped from MAE pairs. Scratch backtest (R1): Sept fold MAE 0.1885 → 0.1812 (−3.9 %), cold fold 0.3025 → 0.3030 (flat) | Small MAE gain, more training/MAE coverage; user-requested |
| [C](2026-10-15-cold-onset-c-heating-aware-projection.md) | `fix/heating-aware-indoor-projection` | Indoor projection: thermostat floor while heating on + per-room setpoint (was pure cool-down ODE vs one shared mean setpoint) | Projected pressure 4.6 °C vs 0.02 °C real at 24–35 h lead (training max 0.72). Live-model replay 10-08..10-14: MAE 24–47 h 0.485 → **0.255**, bias +6.9 → −2.2 kWh/day | **Dominant fix** — ship first; validated on live model (in-sample caveat). Task 4 (opt-in `heat_loss_deg` feature, own branch, backtest-gated) targets the expected residual ≈ −2 kWh/day |
| [D](2026-10-15-cold-onset-d-tau-ua-estimation.md) | `feat/climate-valve-cache` → `feat/tau-ua-estimation` | Cache Netatmo `heating_power_request`; pooled valve-closed per-room τ with a cross-room airing filter (no extra sensor) (+ method switch); daily rolling heat-balance `UA_eff`; gated `heating_intensity_7d` feature | Live τ 11.6 h biased by log-ΔT fit on morning warm-up; pooled valve-closed ODE ≈ 92 h area-weighted (biased high); hourly UA_eff null (buffer lag + target selection), daily UA/COP spring 21–28 vs Oct 8–9 W/K | After C. Task 1 ships ASAP (data accumulation); τ/UA/feature each gated |

## Shared test helpers
`_train_frames(n=960)` (synthetic hourly energy + weather frames) is used by B, C Task 4 and D Task 5. The first plan to merge creates it as `train_frames()` in **`tests/_frames.py`** (`tests/` is a package). The others import it: `from tests._frames import train_frames`. Never redefine it in a test module (ruff F811 / merge conflicts).

## Cross-plan interactions (expected, not bugs)
- **A raises actuals ≈ +0.5 kWh/day** (06:00/20:00 battery-reset hours) and forces a retrain on corrected targets. Every bias number measured before A (incl. C's replay −2.15 kWh/day) shifts. Compare like with like (Verification §2).
- **C's expected outcome is a bias near −2 kWh/day, not 0** (validated candidate). C's live criterion is therefore "24–47 h bias moved from ≈ +6.9 to within [−3, +1] kWh/day". C also lowers the DHW-override tank-ambient correction (rooms no longer projected at ~11 °C), which pushes override days slightly lower.
- **C's off→on transition** gives deficit 0 when heating switches on during the window (floor snaps to `setpoint_on`). This is a known limitation that hides warm-up demand on the first cold day of a season; it matters next autumn, not now.
- **B changes the MAE pair population:** EV-day hours are scored again, netted of wallbox kWh. `mae_*` sensors jump at B's deploy regardless of forecast quality. Use the fixed hour mask below.
- **D Task 3 (τ 11.6 → ~60–90 h)** affects C's projection only in heating-off hours, physics `detect_open_windows` (median/MAD-centred, so the shift is mostly absorbed; log flag counts before/after), and **not** physics `c_building` (decoupled in D Task 4). ha-energy-manager reads only the thermal-pressure sensor *state*, not `tau_hours` (verified, `heat_pump.py:861`).
- **D Task 4 (UA_eff non-null)** activates physics heating for the first time: `physics_kwh` (an ML feature) shifts at the first retrain, and `solar_gain_area` calibration runs live for the first time on the next full calibration. D Task 4's gate checks both.

## Gate rigor
All pre-deploy gates use ≤ 7 cold days (no earlier cold data with climate exists; the climate cache starts 2026-03-31). They are **sanity checks, not proof**. Each requires the MAE threshold **and** a paired daily sign test (variant better on ≥ 5 of 7 days). Opt-in features (C Task 4, D Task 5) are switched on live only after the verification baseline below holds ≥ 14 cold days, and are then judged on the next ≥ 14 cold days.

## Post-deploy verification & attribution (all plans)
1. **Deploy log** (append here per deploy): plan/task, alpha tag, deploy timestamp (local), expected effect.
### Deploy log
| Plan/task | Tag | Deploy (local, 2026-10-16) | Expected effect |
|---|---|---|---|
| D Task 1 (`heating_power_request` cache) | v0.12.0-alpha-25 | ~05:42 | none on forecast; data clock starts (≥ 14 valve-closed nights) |
| C Task 1 (+ tidy commit) (heating-aware projection) | v0.12.0-alpha-26 | ~05:51 | cold-day 24–47 h bias +6.9 → about −2 kWh/day; tomorrow forecast ~19–20 → ~10–13 kWh. Only target days whose predictions were issued after 05:51 on 2026-10-16 count for C. |

2. **Metric** (not the rolling `mae_7d`): from `pred_history.json` (24–25 h lead, keep-first), per **target day**: bias (Σpred − Σact, kWh/day) and MAE (kWh/h), with a **fixed hour mask** that excludes the 06/07/20/21 h hours (A's repair window) and all EV-charging hours (B's population change). Split the days by deploy timestamps. Only days whose predictions were all issued after a deploy count for it (predictions are made ~24 h ahead).
3. **Cold day** = daily mean outdoor temp < 13.5 °C (the learned heating-season `on_below`).
4. **Success criteria:**
   - C: cold-day bias in [−3, +1] kWh/day over ≥ 3 cold days; point forecast inside [p10, p90] for ≥ 90 % of 3 h blocks.
   - A: 06/20 h actuals no longer ≈ 0.
   - B: EV-day actuals at household levels (netted).
   - D3: `tau_hours` switched, open-window flag count logged.
   - D4: UA_eff logged, `physics_kwh` daily heating vs meter.
   - D5: cold-day MAE −5 % vs the 14-day baseline.
5. **Re-baseline after A's retrain** before judging anything later.
6. Still open from the investigation: Oct–Nov 2025 at 5–9 °C used 16–26 kWh/day vs ≈ 10 now. Once C is live, check whether the remaining cold-day bias is explained by this regime shift (D Task 5's heating-intensity feature is the lever if it is).

## Workflow notes
- Commits end with the session's `Co-Authored-By` and `Claude-Session` trailers.
- ROADMAP.md: update #99 (heating-season bias recheck) with C's verification result, and add rows for C Task 4 / D Tasks 3–5 as gated items.
- MEMORY.md + `memory/project_cold_onset_overforecast_2026_10_15.md`: record each deploy (tag, date) and gate table.

**Ruled out (don't re-investigate):** EV over-subtraction in actuals (EV hours ±1 are just excluded), the date-only
`energy_history.csv` row (parsed with `format="mixed"`), median-fill inflation of `lag_*` (medians ≈ 0.27 kWh),
regime centroids as primary driver (R2 no-regimes ablation didn't fix cold bias in reduced-feature backtest).
Superseded: "thermal_pressure = 0 before 2026-03-31" as primary cause — now only C's fallback if bias persists after C.

**Out of scope, flagged:** the main `energy_sensor` path (`_raw_to_kwh_diff`) drops reset hours instead of
recovering them — the SolarEdge m1 grid meter doesn't reset, so not touched. The wallbox cache tail stops at the
last state change (10-14 13:00 at pull time) — benign (missing = 0) but `ev_sensor_coverage()` end shrinks with it.
