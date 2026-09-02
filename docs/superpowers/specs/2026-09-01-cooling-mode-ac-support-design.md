# Cooling Mode / AC Support — Design Spec

**Date:** 2026-09-01 (rev. 3 — post multi-stakeholder review, round 2)
**Status:** Proposed — community-contribution candidate (#96), no implementation timeline committed
**Branch base:** `dev`
**Source:** [GitHub Discussion #20](https://github.com/m-zenker/ha-energy-forecast/discussions/20), opened 2026-08-17 by @gabrieldelboniz

**Revision note:** rev. 1 was reviewed by three independent domain experts (Domain Systems
Engineer, Data Scientist, Software Engineer) in parallel. That review surfaced 22 issues (6 High)
— most significantly: rev. 1's per-row snippets assumed a `cooling_active_at_ts` signal that
nothing in the real code chain produces (heating's `setpoint_on`/`setpoint_off`/
`heating_active_series` are threaded through ~15 function signatures, including the 48h-ahead
prediction-time projection, none of which rev. 1 extended); τ-calibration and regime-clustering
would both silently ingest AC-active hours as if they were passive/unlabeled data, the same
contamination class as the historical EV-day clustering bug (#82); unvalidated placeholder
constants shipped with no sanity check or production warning; and a code-only rollback against a
newer trained model would `KeyError`. Rev. 2 resolved all 22 findings — see §2, §4.1a, §4.9–4.11,
§6 for the specific changes — and cross-checked against the original Discussion #20 text in §10.
The same three reviewers re-checked rev. 2 and found all original issues resolved (two by the
Domain Systems Engineer — solar-lag timing and the conflict-default mitigation — correctly left as
disclosed limitations rather than fixed without real data to fix them against) plus 3 new issues
from the round-2 pass: the conflict-precedence rule in §4.2 wasn't actually wired into §4.3/§4.4's
per-row formulas, `_calibrate_tau()`'s own call site wasn't in §4.1a's threading inventory, and
§4.6's sanity bound was named without a concrete config key. This revision (rev. 3) fixes all
three, plus two convergent minor findings (a default config value for the sanity bound, aggregate-
SHAP label scoping) that two reviewers independently raised.

**Note on process:** this spec was written directly from the codebase and Discussion #20, without
a synthetic simulated dataset or an impact-on-accuracy simulation. This deployment is heating-only
(Swiss climate, no AC) — there is no real cooling-climate history to hold out against, so a "full
set of realistic simulation data covering all scenarios" would encode unvalidated assumptions
rather than test them. See §6 and §7 for how testing is scoped instead.

## 1. Problem & Motivation

The physics thermal-pressure model (`setpoint − indoor_temp`, clipped at 0) is heating-only. In
cooling-dominated climates, once AC engages, thermal pressure computed this way goes to 0 —
the signal disables itself exactly when cooling load is highest. Passive cooling is already
modeled via `infiltration_pressure` (#57); active AC is not. Confirmed against
`_engineer_features()` in `model.py`: `thermal_pressure` is computed once, heating-direction only
(`model.py:3177`, `delta = (c["setpoint"] - c["current_temp"]).clip(lower=0.0)`), and every
downstream feature (`thermal_pressure_max`, `thermal_pressure_std`, `thermal_pressure_net`,
`thermal_pressure_cop`, `infiltration_pressure`) inherits that direction. Matches Discussion #20's
description exactly (@gabrieldelboniz: "the current thermal pressure formula
`(Setpoint - Current Temp)` produces negative values during cooling... a critical issue in
tropical regions").

## 2. Scope & Non-Goals

**In scope (v1):** the ML-feature layer — `model.py::_engineer_features` and the
`thermal_pressure` family, cooling-degree features, cooling hysteresis config — **plus**, per
round-1 review, everything needed for that layer to be *correct* end-to-end rather than
partially wired:
- Threading a cooling-equivalent of `setpoint_on`/`setpoint_off`/`heating_active_series` through
  the same call chain heating already uses, including the 48h-ahead prediction-time projection
  (§4.1a) — without this, `cooling_mode_enabled: true` would silently produce heating-flavored
  *forecasts* even though *historical* features were fixed.
- A passive-window guard in `_calibrate_tau()` so AC-active hours aren't fit as unconditioned
  decay (§4.9).
- A cooling-day exclusion in `DailyProfileClusterer.fit()`, mirroring the existing `ev_day_dates`
  mechanism (§4.10) — round-1 review confirmed this is not a deferrable question but the same
  contamination class as #82 (EV-day centroid contamination), with no exclusion today.
- A model-artifact rollback fallback for the three new feature columns, mirroring the existing
  `physics_kwh`/`heating_buffer_temp` pattern (§4.11).
- Config validation and a runtime sanity/warning mechanism for the unvalidated placeholder
  constants (§4.6, §6).

**Explicitly out of scope (v1), so a reviewer can hold a PR to this line:**
- `physics.py`'s baseline predictor (`_space_heating_kwh`, `UA_eff` calibration) — a separate
  module with its own calibration path (§3 of the physics-ml-hybrid design). It has no cooling
  analog and none is proposed here; physics-baseline forecasts stay heating-only even with
  `cooling_mode_enabled: true`. A cooling-capable physics baseline is a distinct, larger follow-up.
- A dedicated cooling thermal-response curve. The prediction-time projection in §4.1a reuses the
  existing tau-based Euler decay constant (calibrated for passive *heating* decay) as an accepted
  v1 approximation for cooling recovery too — a real cooling-specific time constant needs real
  data to fit and is not attempted here.
- A latent-load/dehumidification feature for cooling (the humidity-driven counterpart to
  `defrost_risk`). Flagged with equal weight to the EER placeholder in §7, not silently dropped —
  see §4.8.
- Distinguishing `dry`/`fan_only` HVAC states from genuine cooling when a binary
  `cooling_system_active_entity` is used (see §4.3) — the `hvac_mode_entity` alternative (§3)
  avoids this ambiguity where the source entity supports it, but the binary-entity path accepts
  the false-positive risk as a documented limitation.
- Real dual-mode heating+cooling blending for simultaneous-active hours — §4.2's conflict rule is
  a documented simplification, not a blend.
- A dedicated cooling regime cluster (separate from heating-season `regime_kwh` centroids) — §4.10
  only *excludes* cooling days from centroid fitting; building a cooling-specific regime model is
  future work (§9).

## 3. Config

```yaml
cooling_mode_enabled: false            # master switch; false = zero new code paths execute
# Option A — single mode-string entity (preferred where available; e.g. a reversible heat pump's
# climate.hvac_mode). Mutually exclusive by construction, so §4.2's conflict rule never applies.
hvac_mode_entity: null                 # e.g. climate.living_room; state ∈ {heating, cooling, off, ...}
# Option B — two independent binary entities (e.g. radiators + a separate window/split AC unit).
cooling_system_active_entity: null     # mirrors heating_system_active_entity
cooling_setpoint_on: 24.0              # °C, target while actively cooling
cooling_setpoint_off: 28.0             # °C, ceiling when cooling is off (mirrors heating's floor —
                                        # note the direction is inverted: cooling's "_on" value is
                                        # numerically LOWER than "_off", opposite of heating's
                                        # 20.0/12.0. This is intentional, not a transposition bug.)
cooling_eer_slope: -0.05               # placeholder — see §4.6, §7. Exposed as config specifically
cooling_eer_intercept: 4.0             # so a real cooling deployment can recalibrate without a code change.
cooling_sanity_bound: 20.0             # plausible-range ceiling for thermal_pressure_cop (a per-
                                        # hour ratio, ~0.5-4 by construction). NOT reused for
                                        # cooling_load_sum_168h — a 168h rolling accumulation is a
                                        # different natural magnitude; that check uses its own
                                        # bound, cooling_load_sanity_bound (default 50.0), so one
                                        # threshold can't be spuriously tight for one quantity and
                                        # permanently silent for the other.
cooling_load_sanity_bound: 50.0        # plausible-range ceiling for cooling_load_sum_168h; see above.
```

Config precedence: if `hvac_mode_entity` is set, it is the sole source of `heating_active`/
`cooling_active` (mode string mapped directly, no conflict possible) and
`cooling_system_active_entity` is ignored if also set (warn once if both are configured).
Otherwise `cooling_system_active_entity` drives `cooling_active` and §4.2's conflict rule applies
against the separately-configured `heating_system_active_entity`.

**Naming note:** Discussion #20 requested `cooling_temp_on`/`cooling_temp_off`. This spec uses
`cooling_setpoint_on`/`cooling_setpoint_off` instead, for consistency with the existing
`heating_setpoint_on`/`heating_setpoint_off` config keys (`energy_forecast.py:307-308`) rather than
the discussion's literal wording.

**Validation** (extends `_validate_config()`, `energy_forecast.py:437-475`, which already checks
this exact shape for `mqtt_discovery`/`ev_charging_threshold_kwh`):
- Warn (not raise, to preserve opt-in safety) when `cooling_mode_enabled: true` and neither
  `hvac_mode_entity` nor `cooling_system_active_entity` is set.
- Raise when `cooling_setpoint_on >= cooling_setpoint_off` (inverted from the sane default would
  silently flip §4.3's sign logic with no other symptom).

When `cooling_mode_enabled` is `false` (default) or no active entity is configured, every change
in §4 is skipped and the existing heating-only formulas run unchanged.

## 4. Feature Design

### 4.1 `cooling_active` (new, mirrors `heating_active`, `model.py:3129`)

Binary per-timestamp series. Sourced from `hvac_mode_entity` (mode string mapped to
heating/cooling/neither) or `cooling_system_active_entity` per §3's precedence. Defaults to 0
(never cooling) when unconfigured. Fetch wrapped in the same `try/except Exception` +
warning-fallback pattern `_build_heating_active_projection` already uses
(`energy_forecast.py:1940-1946`) — an entity that's configured but unavailable/nonexistent in HA
falls back to all-0 `cooling_active`, not an unhandled exception.

### 4.1a Threading through the call chain (new — closes round-1's largest gap)

Heating's `setpoint_on`/`setpoint_off`/`heating_active_series` are explicit parameters threaded
through `train()` → `_engineer_features()` (2 call sites), `_prepare_prediction_X()`, `predict()`,
`predict_intervals()`, `shap_summary()`, and `_project_indoor_temps()`
(`model.py:1000-1585, 2871-2942`), built and passed in from `energy_forecast.py` around
`_build_heating_active_projection` (lines ~1936-2070, 2319-2323). A cooling-mode feature needs the
identical treatment, not a local edit inside `_engineer_features()`:

- Add `cooling_setpoint_on`/`cooling_setpoint_off`/`cooling_active_series` as parallel optional
  parameters at every one of the signatures above.
- **Training-time** (`_engineer_features()`'s `climate_dfs` loop, `model.py:3168`): the
  household-level `cooling_active` series is not currently reachable inside the per-entity loop —
  `heating_active` is merged onto `df` *before* the loop, but the loop iterates `c_df`'s own
  timestamp index. Add a `cooling_active.reindex(c["timestamp"], method="nearest")` step per
  entity inside the loop (mirroring the existing reindex pattern at `model.py:2917/2928`) before
  computing `delta`.
- **Prediction-time** (`_project_indoor_temps()`, `model.py:2871-2942`): this is the 48h-ahead
  Euler-forward projection that resolves `c["setpoint"]` for every future hour *before*
  `_engineer_features()` ever runs (`setpoint_arr = np.where(ha_arr, setpoint_on, setpoint_off)`,
  lines 2927-2931). Without extending this function, `cooling_mode_enabled: true` fixes historical
  training features but future-hour *forecasts* stay heating-only regardless. Add a parallel
  `cooling_active_series`/`cooling_setpoint_on`/`cooling_setpoint_off` branch that selects the
  cooling setpoint when `cooling_active` is projected true for a given future hour (via the same
  `hvac_mode_entity`/`cooling_system_active_entity` projected flat, matching how
  `heating_active_series` is projected today). The Euler-forward decay itself keeps using the
  single existing `tau` constant for both directions — see §2's non-goal on a dedicated cooling
  time constant.

This inventory also covers §4.9's τ-calibration guard: `train()` calls
`self._calibrate_tau(climate_dfs, heating_active_df, weather_df)` (`model.py:431-432`), currently
passing only `heating_active_df`. Add a parallel `cooling_active_df` parameter at that call site
and into `_calibrate_tau()`'s own signature — this is threading through the same chain, not a
separate untracked cost, and §8's effort line for §4.9 already assumes it.

This is a materially larger change than a formula edit — see §8's revised effort estimate.

### 4.2 Conflict resolution: `heating_active` and `cooling_active` both 1

Only reachable via the two-independent-entities config path (§3 Option B) — a single
`hvac_mode_entity` (Option A) makes this state unreachable by construction, which is the
preferred configuration where available. When it is reachable (e.g. radiators + a separate window
AC unit, or a brief transitional overlap), **heating takes precedence** (the existing, tested
path). The first time both are seen active in the same hour, log **WARNING once, INFO on every
subsequent occurrence** — this is `_warn_once`'s actual contract (`ha_data.py:117-129`; it
downgrades repeats to INFO, it does not suppress them). `_warn_once` is currently `ha_data.py`-
private and stateless-function-incompatible with `_engineer_features()` (a module-level function
in `model.py` with no `self`); promote it to a shared utility (`const.py`) rather than importing a
private name cross-module, and thread a new `self._cooling_conflict_warned: set` on
`EnergyForecastModel.__init__` the same way `self._excluded_range_warned` is owned today.

**Precedence is enforced once, at the source, not per-formula:** when a conflict is detected,
`cooling_active` is set to 0 for that hour (`heating_active` is left at 1) *before* either series
is passed downstream. Every consumer in §4.3–4.7 branches solely on `cooling_active_at_ts` with no
additional `and not heating_active_at_ts` term needed — they see an already-resolved signal, not a
raw conflict they each have to re-adjudicate.

Optional, cheap addition: a rolling `heating_cooling_conflict_hours_7d` count, so the ambiguity is
visible to downstream analysis (dashboard/SHAP), not just op logs. Not covered by §6's test list —
it's optional and untested by design, unlike every required §4.x addition.

### 4.3 `thermal_pressure` — mode-aware sign

Current (`model.py:3177`):
```python
delta = (c["setpoint"] - c["current_temp"]).clip(lower=0.0)
```
Proposed (after §4.1a's per-entity `cooling_active` reindex is available in scope):
```python
if cooling_mode_enabled and cooling_active_at_ts:
    delta = (c["current_temp"] - cooling_setpoint_resolved).clip(lower=0.0)  # positive once above setpoint
else:
    delta = (c["setpoint"] - c["current_temp"]).clip(lower=0.0)              # unchanged heating path
```
`cooling_setpoint_resolved` is the per-row value produced by §4.1a's threading (analogous to how
`c["setpoint"]` is already pre-resolved for heating), **not** a bare config scalar — `§3`'s
`cooling_setpoint_on`/`_off` are hysteresis bounds, not a single value usable directly here.

`thermal_pressure_max` / `thermal_pressure_std` need no separate change — they're derived from
the same per-entity `delta_df`, so the sign fix propagates automatically.

**Dry/fan-only mode:** if the source entity is `hvac_mode_entity` and its state distinguishes
`dry`/`fan_only` from `cool`, exclude those states from `cooling_active` (they run the compressor
for humidity control without pursuing `cooling_setpoint`, so treating them as full cooling would
inject a spurious pressure value). If the source is a plain binary
`cooling_system_active_entity`, this distinction isn't available — accepted as a documented
false-positive risk (§2's non-goals).

### 4.4 `thermal_pressure_net` — solar compensation sign flips

Heating: solar gain *reduces* debt → subtract (`model.py:3275`, unchanged).
Cooling: solar gain *increases* cooling load → add.
```python
if cooling_mode_enabled and cooling_active_at_ts:
    thermal_pressure_net = thermal_pressure + 0.01 * weighted_solar_gain
else:
    thermal_pressure_net = np.maximum(0.0, thermal_pressure - 0.01 * weighted_solar_gain)  # unchanged
```
**Known approximation, not fixed here:** this reuses the existing 09:00–17:00 half-cosine window
built for heating's instantaneous solar-gain offset. Real cooling load driven by solar gain lags
irradiance by hours (thermal mass absorbs energy during the day, releases it into the space in the
evening), so peak AC demand often occurs after 17:00 when this term has already dropped to 0 —
understating `thermal_pressure_net` exactly when cooling load peaks. A lagged/decayed solar term
for the cooling branch would fix this properly but needs real data to tune; accepted as a v1
limitation (§7), not attempted here.

### 4.5 `infiltration_pressure` — no change needed

Already an unsigned wind × gradient-magnitude proxy (`model.py:3280`); once `thermal_pressure`
itself is mode-aware (§4.3) this feature is correct with zero additional code.

### 4.6 `thermal_pressure_cop` — cooling EER proxy, config-exposed and flagged as a placeholder

The existing linear COP model (`0.11 × T_out + 3.0`, `model.py:3254`) is calibrated for an
air-to-water heat pump in heating mode. Cooling-mode EER typically moves the *opposite* direction
— efficiency drops as outdoor temp rises. No real EER data exists in this deployment to calibrate
against, so the slope/intercept are exposed as config (`cooling_eer_slope`/`cooling_eer_intercept`,
§3) rather than hardcoded — a real cooling deployment can recalibrate without a code change, unlike
rev. 1's hardcoded version, which had no recalibration path at all:
```python
eer_proxy = (cooling_eer_slope * temp_c + cooling_eer_intercept).clip(lower=0.5)
divisor = eer_proxy if (cooling_mode_enabled and cooling_active_at_ts) else cop_proxy
thermal_pressure_cop = thermal_pressure / divisor
```
Documented at the same tier as the module's existing magic-constant rationale
(`model.py:3257-3271`): units (°C → dimensionless EER proxy), explicit **"placeholder, unvalidated
— see §7"** flag, and the default slope's rough rationale (chosen to keep `eer_proxy` in the same
0.5–4 magnitude band as `cop_proxy`, for consistent LightGBM split-gain scaling — not derived from
real EER curves).

**Runtime sanity mechanism (new, closes a round-1 High finding):** the first time
`cooling_active` is observed `true` in production, log a WARNING once naming the unvalidated
constants and pointing at §7 (reuses the `_warn_once` promotion from §4.2). Additionally, at
training time, log a WARNING if `thermal_pressure_cop` exceeds `cooling_sanity_bound` or
`cooling_load_sum_168h` (§4.7) exceeds `cooling_load_sanity_bound` (§3) — this surfaces a bad
calibration in logs rather than only as silently-degraded MAE.

### 4.7 New: `cooling_load_sum_24h` / `cooling_load_sum_168h` (mirrors #50)

**Renamed from rev. 1's `cooling_deg_sum_24h/168h`** — the codebase already has an unrelated,
pre-existing `cooling_degree` feature (`model.py:3087`, `max(0, temp_c - 22.0)`, a plain weather
proxy always non-zero in summer regardless of `cooling_mode_enabled`). This is, in fact, the exact
formula Discussion #20 proposed under the name "cooling degree" — it already exists and needs no
new work; §10 covers this. `cooling_deg_sum` was close enough to `cooling_degree` to be a real
interpretability trap in SHAP output (two similarly-named, semantically different signals), hence
the rename to `cooling_load_sum_24h/168h` for the AC-active-gated rolling sum proposed here. Same
rolling-sum pattern as `heating_deg_sum_24h/168h` (`model.py:3036-3037`): accumulates
cooling-direction `thermal_pressure` only (0 during heating-mode or cooling-inactive hours).

### 4.8 `defrost_risk` — forced to 0 while `cooling_active`

The Gaussian icing curve (`model.py:3285-3286`) models heating-mode outdoor-coil frost — correctly
inapplicable to cooling. But it is also the pipeline's *only* humidity-weighted feature; forcing it
to 0 for cooling hours means the model has **zero latent-load/dehumidification signal**, a
first-order driver of AC energy consumption in humid climates (often 20-40%+ of total cooling
load), not a secondary gap. Weighted equally with the EER placeholder in §7 (rev. 1 undersold this
relative to the EER item). No dedicated `humidity_load_proxy` is attempted in v1 — flagged as a
concrete, well-scoped follow-up rather than silently absorbed into "known limitations" generically.

### 4.9 `_calibrate_tau()` passive-window guard (new — closes a round-1 High finding)

`_calibrate_tau()` (`model.py:1811-1907`) classifies "passive decay" windows solely via
`heating_active == 0` (`combined["off"] = (combined["heating_active"] == 0)`, line 1888) and fits
`ln(T_in − T_out) = ln(ΔT₀) − t/τ` assuming no active conditioning during those windows. Once a
cooling system exists, every summer hour has `heating_active == 0`, so `cooling_active == 1` hours
— where the compressor is actively forcing `T_in` down, violating the passive-decay physics the
fit depends on — would be misclassified as passive candidates and corrupt τ. Fix: when
`cooling_mode_enabled`, gate the passive-window mask on `heating_active == 0 AND cooling_active ==
0`, not `heating_active == 0` alone. This is in scope (unlike `physics.py`'s `UA_eff`, §2) because
`_calibrate_tau` lives in `model.py`, the module this spec otherwise modifies.

### 4.10 `DailyProfileClusterer` cooling-day exclusion (new — closes a round-1 High finding)

`DailyProfileClusterer.fit()` (`clustering.py:45-138`) pivots and KMeans-fits directly on raw
hourly `gross_kwh` shape, with an existing `ev_day_dates` exclusion parameter
(`fit_days = [d for d in valid_days if d not in ev_day_dates]`, mirrored at
`clustering.py:88, 231-272`) added specifically to fix historical EV-day centroid contamination
(#82). A cooling deployment with no equivalent exclusion would let AC-driven midday consumption
spikes distort "Workday"/"Weekend" regime centroids for every day, the same failure mode #82 fixed
for EV — round-1 review flagged deferring this as unsafe, not merely open. Fix: add a
`cooling_day_dates` parameter to `fit()`, populated from `cooling_active` the same way
`ev_day_dates` is populated from EV detection, and excluded from centroid fitting whenever
`cooling_mode_enabled`. This must land with v1, not be deferred — §9 keeps a *different*,
genuinely-deferrable question (a dedicated cooling regime cluster) separate from this
contamination fix.

### 4.11 Model-artifact rollback fallback (new — closes a round-1 High finding)

The existing "Model-artifact portability fallback" pattern (`model.py:1126-1146`) fills
`physics_kwh`/`heating_buffer_temp` with `0.0` and logs a warning when a saved model's
`feature_cols` references a column `_engineer_features()` doesn't currently produce (e.g. sensor
outage, config change). This is safe for `cooling_active`/`cooling_load_sum_*` in the *forward*
direction (old model + new code = column just unused) but not for a **code-only rollback**: if a
model is retrained after this feature ships (feature_cols now includes the 3 cooling columns) and
the code is then rolled back to a pre-cooling-feature commit, `_engineer_features()` no longer
emits those columns and `feat_df[self.feature_cols]` raises `KeyError` with no fallback. Fix: add
the identical fill-with-`0.0`-and-warn guard for `cooling_active`, `cooling_load_sum_24h`,
`cooling_load_sum_168h` alongside the existing two blocks.

## 5. `_FEATURES_BASE` and SHAP labels (model.py, energy_forecast.py)

Add `cooling_active`, `cooling_load_sum_24h`, `cooling_load_sum_168h` unconditionally to
`_FEATURES_BASE`. **Correction from rev. 1:** these are not "the same pattern as `heating_active`"
— `heating_active` defaults to 1 only for the *unconfigured* subset and is a genuinely variable
signal for deployments that configure it; the three new columns will be constant-zero for
effectively the *entire* current install base (heating-only Central European users), by design,
indefinitely. In practice this costs nothing at training time — LightGBM's split-gain search is
free on a truly constant column and TreeSHAP attributes it exactly 0 — but the spec should say so
plainly rather than lean on an analogy that doesn't hold.

`_SHAP_FEATURE_LABELS` (`energy_forecast.py:63-146`) gets three new entries
(`cooling_active`, `cooling_load_sum_24h`, `cooling_load_sum_168h` → human labels mirroring the
existing heating ones). Additionally, the existing `thermal_pressure*` labels are heating-flavored
("heat debt (area-weighted)", "heat debt electrical cost") — once cooling is active the same keys
represent cooling load/EER cost, so the per-prediction "why today?" narrative (single-hour SHAP
attribution) should condition these labels on that hour's `cooling_active`. For any *aggregate*
SHAP summary spanning mixed heating/cooling hours (`shap_summary()`, multi-day views), no single
`cooling_active` value applies — fall back to the existing static heating-flavored label rather
than attempt a per-row conditional label in an aggregate context; this is a known, accepted
imprecision for the aggregate view specifically, not a gap in the single-prediction narrative
this feature primarily targets.

**Claim scope:** §5's LightGBM-constant-column cost analysis (below) covers training-time
split-gain and TreeSHAP attribution mechanics generally, based on how both are documented to
behave on zero-variance columns — §6's regression test verifies holdout MAE and `X.columns`
composition, not SHAP output specifically, so this claim is not itself test-verified by this spec.

## 6. Testing Plan

No synthetic "all scenarios" dataset — see the process note at the top. **Revised from rev. 1**,
whose "absent vs. explicitly-false `cooling_mode_enabled`" pairing was near-tautological (both
collapse to the same Python `False` before any downstream code runs via the existing
`self.args.get(key, False)` pattern, so it tested nothing beyond running the check once):

- **No-regression guarantee, tested against the actual risk:** a test comparing the *full*
  `X.columns` composition and holdout MAE against pre-change `main` output, using existing real
  heating-season fixtures, with `cooling_mode_enabled` both absent and explicitly `false` — this
  is what actually verifies the three unconditionally-added columns (§5) don't perturb trained
  tree structure or accuracy for this deployment, not just that three named Series match.
- **The "enabled but no active entity configured" branch** (§3's validation-warns-but-doesn't-
  block case) — untested in rev. 1 — gets its own test confirming it also reproduces heating-only
  output.
- **Formula correctness (hand-built edge cases, not full scenario coverage):** ~6-8 synthetic
  single-room `climate_dfs` rows checking the sign inversion and clip boundary directly — e.g.
  `current_temp` exactly at `cooling_setpoint_resolved` (→ 0), 3°C above it with
  `cooling_active=1` (→ 3.0), the same row with `cooling_active=0` (→ 0, heating path).
- **§4.9 τ-guard:** a test with synthetic AC-active hours mixed into a passive-window candidate
  set, confirming they're excluded from the regression when `cooling_mode_enabled`.
- **§4.2 conflict resolution:** a test asserting `cooling_active` is actually zeroed for an hour
  where both source signals read active, and that the log contract fires as specified (WARNING on
  first occurrence, INFO on repeats — matching `_warn_once`'s real behavior, not "logged once").
- **§4.10 clustering exclusion:** a test mirroring the existing `ev_day_dates` test, confirming
  `cooling_day_dates` are excluded from centroid fitting.
- **§4.11 rollback fallback:** a test asserting `feat_df[self.feature_cols]` doesn't raise when
  the three cooling columns are absent from `feat_df` but present in `self.feature_cols`.
- **§3 config validation:** tests for both new `_validate_config()` checks (missing active entity
  when enabled; inverted `cooling_setpoint_on >= cooling_setpoint_off`).
- **No forecast-accuracy simulation ships with this spec.** There is no real cooling-climate
  holdout to run it against. A contributor with an actual AC deployment would need to validate
  accuracy impact separately, ideally with their own pulled history via the same
  `pull_ha_data.py`-style workflow, once this lands.

## 7. Known Limitations

- §4.6's cooling EER proxy is an unvalidated placeholder slope, now config-exposed for
  recalibration (§3) and guarded by a first-observed production warning (§4.6) — but still needs
  real EER curve data from a cooling deployment before it should be trusted as more than "some
  signal beats none."
- §4.8: **zero latent-load/dehumidification signal for cooling** — weighted equal to the EER
  placeholder, not a lesser concern; a first-order driver of AC energy use in humid climates.
- §4.4: the cooling solar-compensation term reuses heating's same-day 09:00-17:00 window
  unadjusted for thermal-mass lag; likely understates evening-peak cooling load.
- §4.3: a binary `cooling_system_active_entity` (vs. `hvac_mode_entity`) cannot distinguish
  `dry`/`fan_only` from genuine cooling — accepted false-positive risk.
- §4.2: simultaneous heating+cooling hours (reachable only via the two-independent-entities config
  path) resolve to heating, not a true blend.
- `physics.py` baseline predictor has no cooling analog (§2) — physics-baseline forecasts stay
  heating-only regardless of this flag.
- No dedicated cooling thermal time constant — §4.1a's prediction-time projection reuses the
  existing heating-calibrated `tau` for cooling recovery too.
- All calibration constants here (`0.11`/`3.0` existing heating COP, `cooling_eer_slope`/
  `cooling_eer_intercept` defaults) are heating-climate-tuned or unvalidated guesses; a real
  cooling deployment's data should recalibrate them via the new config keys rather than trust the
  defaults long-term.
- §4.7: **`cooling_load_sum_24h/168h` has no historical seed at prediction time.** Unlike
  `heating_deg_sum_24h/168h` (computed from an `_extended` frame that concatenates
  `self._weather_tail`, a ~400-row historical tail, onto the forecast) and `rolling_mean_24h`/
  `rolling_std_24h` (seeded from `recent_actuals`), `cooling_load_sum` is computed straight from
  `df`/`thermal_pressure`, which at prediction time is always the exact 48-row `future_df` built by
  `_prepare_prediction_X()` with no tail extension. `cooling_load_sum_168h` can therefore never
  accumulate more than ~48 hours of signal at serve time, vs. a true 168-hour accumulation at
  training time — a systematic, permanent scale mismatch on every prediction cycle (not a
  transient cold-start effect), and a genuine gap rather than a pattern shared with its siblings.
  Currently inert (these columns are not yet in `_FEATURES_BASE`; §5 defers that registration to a
  later task), but this gap must be resolved — e.g. a cooling-load equivalent of
  `self._weather_tail` — before registering the columns as model features, or the model will train
  on a 168h-scaled signal while being served one capped at 48h.

## 8. Effort Estimate

**Substantially revised from rev. 1's ~6h**, which undercounted the call-chain threading fan-out
(§4.1a) and omitted the τ-calibration, clustering-exclusion, and rollback-fallback fixes entirely
— round-1 review's Software Engineer flagged the original config-plumbing estimate as a "severe
undercount" given ~15 function signatures involved.

| Item | Effort |
|---|---|
| §4.1, §4.1a `cooling_active` + full call-chain threading (train/predict/projection, ~15 signatures) | ~6h |
| §4.2 conflict handling + `hvac_mode_entity`/binary-entity precedence + tests | ~1.5h |
| §4.3–4.5 thermal_pressure family sign inversion + tests | ~2h |
| §4.6 COP/EER placeholder, config-exposed + sanity/warning mechanism | ~1.5h |
| §4.7 cooling-degree features (renamed) + tests | ~1h |
| §4.9 τ-calibration guard + tests | ~1.5h |
| §4.10 clustering cooling-day exclusion + tests | ~1.5h |
| §4.11 model-rollback fallback + test | ~0.5h |
| §3 config validation + tests | ~1h |
| §5 `_FEATURES_BASE`/SHAP label updates | ~1h |
| Docs | ~1h |
| **Total (ML-feature layer, as scoped here)** | **~19h (≈2.5 days)** |

This is well beyond the roadmap's original "unscoped, likely 1 day+" and reinforces that #96
remains correctly deprioritized behind higher-impact items — this spec exists so a future
community contributor has a concrete, correctness-checked starting point, not because the
maintainer intends to build it soon. Still excludes `physics.py` baseline cooling support (§2) —
separate, larger follow-up, not estimated.

## 9. Open Question for Next Review

Should a **dedicated cooling-season regime cluster** (distinct centroids for AC-driven daily
shapes, rather than just excluding cooling days from the existing heating-season centroids per
§4.10) be built? This is a genuinely open, deferrable design question — unlike §4.10's
contamination fix, which is a correctness requirement — and isn't attempted here given no real
cooling-season data exists to validate cluster shapes against.

## 10. Alignment Check Against Discussion #20

Re-read against the original post (@gabrieldelboniz, 2026-08-17) point by point:

1. **"New Entity: `cooling_system_active_entity` flag or generalized `hvac_mode_entity`"** — both
   proposed in §3, with `hvac_mode_entity` preferred where available (it also cleanly resolves the
   conflict-topology ambiguity round-1 review raised, since a single mode-string entity makes
   simultaneous heating+cooling state unreachable by construction).
2. **"Inverted Calculation: `(Current Temp - Setpoint)` for positive cooling pressure"** — §4.3,
   matches exactly.
3. **"Cooling Degrees: `max(0, temp_c - 22)`"** — this formula **already exists** in the codebase
   as `cooling_degree` (`model.py:3087`), unrelated to AC-active state, added independently of
   this spec. §4.7 cross-references it explicitly and renames this spec's new AC-gated rolling-sum
   feature (`cooling_load_sum_24h/168h`) to avoid colliding with it — a contributor implementing
   this spec should not re-add the discussion's literal formula; it's a no-op, already shipped.
4. **"Parameter Duplication: `cooling_temp_on/off`"** — §3 provides
   `cooling_setpoint_on`/`cooling_setpoint_off` (see the naming note in §3 for why "setpoint" was
   chosen over the discussion's "temp" wording — consistency with the existing
   `heating_setpoint_on/off` keys).
5. **"Fairly common for ACs to be sold with Cooling Mode only (no H)"** — this framing supports
   §3's dual config path: `hvac_mode_entity` alone doesn't cover a cooling-only unit reporting
   simply on/off, which is exactly what `cooling_system_active_entity` (Option B) is for.

No gaps found between the original ask and this spec's scope — round-1 review's findings were
entirely about implementation correctness and safety (the call-chain plumbing, contamination
risks, unvalidated constants), not about missing or misunderstanding what was requested.
