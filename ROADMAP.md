# Forecast Accuracy Roadmap

Current: **v0.11.4** — 2026-06-16, main. 627 tests.

---

## Current Status

**dev:** v0.11.4-alpha-2 (same codebase — release commit was made on main only). **main:** v0.11.4 released 2026-06-16.

Recent releases:
- v0.11.4 — 15-minute energy history cache (#85), strip partial day from clustering (#86), tomorrow block P10/P90 interval sensors, code review batch 1–5 (all open findings closed).
- v0.11.0 — Daily Regime Clustering, EV-subtracted clustering input (#82 — live since 2026-04-23).

**SHAP check due:** #82 (EV-clean clustering) has been live for ~2 months. Regime clusters should now reflect genuine intra-day shape patterns rather than EV charge timing. Run a SHAP summary to see whether `regime_kwh` ranks high (shape signal) or low (redundant with temperature features) — this determines whether #83 adds value.

**MAE trajectory:** 0.7 → 0.52 kWh/h (as of April). Current value readable from `sensor.ha_energy_forecast_mae_30d`.

**Physics Phase 2 interval-coverage check due (post-deployment):** after Phase 2 (`use_physics_residual: true`) has been live for ≥30 days, verify empirical prediction-interval coverage on gross kWh matches the target (80% by default, per `_calibrate_intervals()`'s conformal quantile). If coverage has drifted, the CQR calibration on the residual distribution may need a wider correction — see `docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md` §5.2. Not yet applicable — Phase 2 stays dormant behind the cold-start gate until ≥30 winter UA_eff calibration windows exist (not expected before winter 2026/27).

**Deploy freeze lifted (2026-07-16):** solar panel commissioning completed 2026-07-16 (hardware confirmed live). `#89` (physics sensor cache dedup, merged to `dev` @ `fc78113`) deployed the same day, and `apps.yaml`'s solar PV + battery target-correction block was enabled with live SolarEdge/gPlugK entity IDs — see `memory/project_solar_feature_pending.md`. `#89`'s Task 6 manual cache-cleanup is also done (11 orphaned CSV files deleted from HA, confirmed via re-list). Still open: re-verifying the battery charge/discharge sensor direction once the battery has visibly cycled (SOE was still 0% at commissioning). `#40` (battery SoC as a feature) is now unblocked and worth revisiting.

**Solar/battery live-path correction bug found and fixed (2026-07-16):** same evening as commissioning, `_update_sensors()` (live hourly path) was found never to apply the solar/grid-export/battery target correction that `_retrain()` uses to train the model — every battery charge cycle inflated raw grid import in `recent_actuals`/`full_actuals`, firing false "unusual consumption" alerts and pushing forecasts up. Fixed via `docs/superpowers/plans/2026-07-16-live-target-correction-fix.md` (shared `_fetch_correction_dfs()` extracted from `_retrain()`, wired into `_update_sensors()` too), merged to `dev` @ `52a201b` and deployed the same evening.

**OPEN — actuals-history freeze found during the above deploy's live verification (2026-07-16 night):** `_actuals_history` (and the raw, pre-correction `energy_history.csv` cache) stopped advancing past a fixed hour across 2+ post-deploy hourly cycles, despite the underlying HA sensor actively updating. Confirmed unrelated to the target-correction fix (the raw cache it touches is untouched by that plan's code). Not yet root-caused — needs a dedicated debugging session. See `memory/project_actuals_history_freeze.md`. Blocks getting a clean read on whether the anomaly-detector fix above actually resolved live false positives.

**Physics Phase 1 + τ-seed accuracy check due (~2026-07-17):** Plan A-D (physics-ML hybrid) and the τ-calibration one-time seed (11.64h) both went live 2026-07-10. Check `sensor.ha_energy_forecast_energy_forecast_mae_7d` (should fully reflect the post-deployment period by then) and `_mae_30d` (partial blend, directionally informative) to see whether forecast accuracy improved, held steady, or regressed. Expect a limited effect this early: `UA_eff` (space heating) failed to calibrate on the first post-deploy retrain (R²=0.09, insufficient summer heating-cycle data) and stays at its default, so only the DHW/base-load physics component (`Q_base_el`, `Q_dhw_daily`, `UA_dhw` — all calibrated successfully) and the τ fix are in play right now. A bigger signal is expected once winter data lets `UA_eff` calibrate for real.

---

## Design Decisions

| Topic | Decision |
|---|---|
| Primary goal | **Forecast accuracy** + visibility/dashboards |
| Solar PV | Planned — target correction done (v0.8.0); solar forecast feature out of scope |
| Home battery | SoC as feature deferred until panels installed (#40) |
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

### #83 — Add `predicted_day_total` Feature (Temperature Regression)

**Priority:** Medium — consider after #82.

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

**Priority:** Low-Medium — standalone, no prerequisites.

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

**Priority:** Low-Medium — more effective as warm-weather history grows.

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

### Deferred

| # | Item | Reason |
|---|------|--------|
| #22 | EV SoC / charging state feature | EV hours are subtracted from training target — SoC has no signal to learn. Revisit if EV load is re-included. |
| #40 | Home battery SoC as feature | Deferred until solar panels installed; revisit if residuals show SoC correlation. |
| #24 | Electricity spot price feature | Fixed flat tariff — out of scope. |
| #94 | Remove vestigial `dhw_tank_volume_l` in `physics_schedule.json` | Dead duplicate of the `physics:` config value (`self._config`, correctly 400L, actually used by `_dhw_kwh_series`) — the schedule file's own copy (currently 200) is only ever echoed back into itself by the autonomous legionella-schedule-learning code, never read by the computation. No functional impact, but misleading if the file is inspected directly (caught 2026-08-18 during hem's DHW boost spec review). Clean up opportunistically next time `physics.py`'s schedule-writing code (`_check_legionella_stability`/schedule-learning path, ~line 684) is touched — not worth a dedicated pass on its own. |

---

## Pending Summary

| # | Item | Impact | Effort | Priority |
|---|------|--------|--------|----------|
| 82 | Fix EV contamination in clustering | high (regime_kwh #1 feature) | 2 h | ✅ done (v0.11.0-alpha-16) |
| 83 | `predicted_day_total` scale feature | medium | 3 h | SHAP check first — may be redundant |
| 84 | Legionella/DHW boost hour feature | — | — | **SUPERSEDED (2026-08-22) by #93** — see below |
| 92 | Temperature-based UA_eff calibration window | medium | 2 h (est.) / multi-day (actual) | **implemented on `fix/ua-eff-calibration-window`, not yet merged to `dev`** (2026-09-03) |
| 93 | DHW override commit + deterministic forecast correction (cross-repo, supersedes #84) | medium-high | ~1 day+ | both phases deployed live 2026-08-27 (hef `v0.12.0-alpha-13`, EM `v0.15.1-alpha-51`) — kill-switch pending exit-gate observation, see `memory/project_dhw_comfort_boost_commit_spec.md` |
| 87 | `trend_deviation` feature (recent vs baseline) | low-medium | 1 h | ready |
| 88 | Temperature-similarity sample weighting | low-medium | 3 h | simulated — see #88 detail |
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
| 95 | Dedupe hourly excluded-range escalation/malformed-CSV warnings — `_warn_once()` in `ha_data.py` fires WARNING once per condition per AppDaemon process lifetime via `self._excluded_range_warned`, then INFO on repeats; resets on restart. **Follow-up correction:** alpha-11 only demoted severity — `filter_excluded_ranges()` still logged an INFO line every hourly cycle for a range matching zero rows (e.g. once it ages out of the data window), which was the actual source of ongoing noise reported live. Now gated on `n_dropped > 0`; a no-op range logs nothing. | v0.12.0-alpha-11 |
