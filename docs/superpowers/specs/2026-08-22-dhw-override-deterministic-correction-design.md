# DHW Override Deterministic Correction — Design Spec

**Roadmap items affected:** supersedes hef `ROADMAP.md` #84 ("Legionella / DHW Boost Hour Feature") entirely, and supersedes the **hef-side** portion of the approved-but-unimplemented `2026-08-15-dhw-comfort-boost-commit-design.md` (#93) — specifically its "hef-side changes" §1 (`commit_dhw_schedule` merge semantics) and §2 (`_dhw_override_for_hour` kind→target-key lookup, `T_dhw_upper` reuse). #93's **EM-side** candidate-selection/`ScenarioScorer` gating chain is unaffected and stays as spec'd (see "Relationship to #93 and #84" below).

**Date:** 2026-08-22

**Repos touched:** `ha-energy-forecast` (`apps/energy_forecast/physics.py`, `apps/energy_forecast/model.py`, `apps/energy_forecast/energy_forecast.py`), `ha-energy-manager` (`apps/energy_manager/loads/heat_pump.py`)

## Context

Live evidence (2026-08-04, see `memory/project_physics_kwh_low_importance.md`) showed a correctly-committed legionella DHW override barely move the published forecast: the physics layer applied a clean +3.64 kWh spike to `physics_kwh`, but the trained LightGBM model's `predicted_kwh` moved by only ~6% of that — `physics_kwh` isn't even a top-5 SHAP feature. Root cause is structural, not just "missing feature": in Phase 1, `physics_kwh` (including any DHW override spike) is one plain input feature among ~19; the tree model is free to learn low weight for it. Compounding this, `_dhw_override_for_hour` reconstructs training history from a single `committed_override` slot that only ever reflects the *most recently* committed value — so most historical override days replay in training data as if no override happened at all, actively teaching the model that `physics_kwh`'s override signal doesn't correlate with the target.

Adding a per-scenario ML feature (`is_legionella_hour`, #84's original design) does not fix this: it's the same mechanism (a plain LightGBM input the tree is free to discount) that already failed once, and it would suffer the identical reconstruction-fidelity bug on top.

Separately, the approved #93 spec (for routine DHW comfort-boost commits, not yet implemented) deliberately approximates the comfort-boost target temperature as the static `T_dhw_upper` schedule constant, rather than hem's actual dynamically-computed `_dhw_boost_target_c` (which varies per boost — booking hours, headroom, drawdown percentile, and can be revised upward mid-boost). That's flagged in #93 itself as "accepted, not-yet-empirically-verified." Since comfort-boost fires far more often than legionella (near-daily vs. weekly), this approximation matters more than it did for legionella's fixed 60°C.

## Goals

1. Any committed DHW override (legionella today; comfort-boost once EM-side wiring lands) moves the published forecast by its true expected kWh amount, **regardless of what the trained ML model has learned to weigh** — a deterministic guarantee, not a hope that a feature ranks highly enough.
2. Fix training-data reconstruction fidelity: retraining must replay what was actually committed on every historical day, not just the latest value.
3. Use the real committed target temperature for comfort-boost, not the `T_dhw_upper` approximation.

## Non-goals

- A generic, subsystem-agnostic override-commit API. Only DHW commits to hef today (`heat_pump.py` is the sole `set_dhw_schedule` caller; `appliance_scheduler.py` only *reads* forecasts via `get_scenario`, it never commits). Building a generic registry for hypothetical future EV/appliance callers is speculative — the data model below (kind/date/hour/target_c) is already subsystem-agnostic *in shape*, so extending it later is cheap without over-building now.
- Phase 2 residual mode (`use_physics_residual`) acceleration or its winter/`UA_eff` gating logic — untouched, unrelated. This design is a narrowly-scoped, always-on analogue of Phase 2's residual pattern applied *only* to the DHW-override delta, independent of the winter cold-start gate.
- #93's EM-side comfort-boost candidate ranking, `ScenarioScorer` gating chain, coverage thresholds, or clear-on-cancellation semantics — all kept exactly as spec'd. That machinery decides *which hour* to commit; this design only changes how hef *consumes and applies* whatever hour gets committed.
- ROADMAP #92 (temperature-based `UA_eff` calibration window) — unrelated.

## Relationship to #93 and #84

- **#84 is superseded outright.** Mark it superseded/closed in ROADMAP.md once this ships; no `is_legionella_hour` feature is needed.
- **#93 is partially superseded.** Its EM-side design (§"EM-side changes": `_arm_dhw_schedule`, the `ScenarioScorer` candidate walk, `dhw_comfort_boost_min_coverage` gating, fail-open/exhaustion behavior, remote-clear-on-cancellation) is unaffected and should still be implemented as spec'd — with one addition: `_commit_dhw_schedule` must also pass the real `_dhw_boost_target_c` (see "EM-side changes" below). Its hef-side design (merge-semantics `commit_dhw_schedule`, kind→target-key `_dhw_override_for_hour`) is replaced by this spec's data model and correction layer. Whoever implements #93 next should read *this* spec's hef-side sections instead of #93's.

## Architecture

Split what's currently a single `physics_kwh` feature into two independently-computed series:

- **`physics_kwh`** (ML feature, same name/role as today) — always computed **override-blind**. `_dhw_kwh_series` (and its two callers, `predict_training_series`/`predict_series`) stop passing any override into the ML-facing DHW computation — it always models the tank as if no override ever happened, both when reconstructing training history and at live prediction time. The tree model never sees an override spike, so it structurally cannot learn to discount or cancel it — there's nothing override-shaped left in this feature to discount.
- **`override_delta_series`** (new) — for any hour with a committed override (historical or current), `dhw_kwh(target_temp=committed) − dhw_kwh(target_temp=baseline)`, i.e. the *marginal* extra electrical kWh the override actually causes, computed from `_dhw_kwh_series`'s existing tank-temperature/COP model, just called twice (once with the override applied, once without) and differenced. Zero for every hour with no committed override.

## Data model (`physics_schedule.json`)

Two structures, serving two different purposes — a current-state view (what's active right now, for serving) and an append-only log (what was ever committed, for training reconstruction):

```json
{
  "committed_override": {
    "legionella": ["2026-08-04", 12],
    "comfort_boost": ["2026-08-05", 14]
  },
  "override_history": [
    {"kind": "legionella", "date": "2026-08-04", "hour": 12, "target_c": 60.0, "committed_at": "2026-08-04T06:00:01+02:00"},
    {"kind": "comfort_boost", "date": "2026-08-05", "hour": 14, "target_c": 57.5, "committed_at": "2026-08-05T05:40:12+02:00"}
  ]
}
```

- `committed_override` keeps #93's merge-semantics shape (keyed by kind, `(date_str, hour)` pairs, staleness-expiry as #93 spec'd) — used only to answer "is an override active for this hour right now" at serving time.
- `override_history` is append-only and unbounded (same accumulate-forever pattern already used by `energy_history.csv` and `climate_<entity>.csv` — no pruning logic needed; retraining only ever looks up entries by exact `(kind, date, hour)`, so growth doesn't slow lookups meaningfully at this data volume). Every `commit_dhw_schedule` call appends one entry per kind present in the incoming override (skip appending for a `None` value, since that's a *clear*, not a commit). `target_c` is recorded at commit time — for `comfort_boost`, this is hem's real `_dhw_boost_target_c` (see EM-side changes); for `legionella`, it's `T_legionella` as today.
- `_dhw_override_for_hour` gains a second lookup mode alongside its existing one: given `history: list[dict]` (the `override_history` list), it scans for an entry whose `(kind, date, hour)` matches `ts` and returns that entry's `target_c` — this is the training-reconstruction path. Its existing mode — checking `committed_override` (the current-state view) directly — is unchanged and remains what the live/serving path uses, since "what's committed right now" is exactly what serving needs; it never needs to scan history.

## Data flow

**Training (`predict_training_series` / retrain target correction):**
1. Compute `physics_kwh` override-blind, as always (no change in call shape needed here beyond no longer threading override state through).
2. Compute `override_delta_series` for the full training window by replaying `override_history` — for each entry, the marginal delta at its `(date, hour)`; zero elsewhere.
3. Subtract `override_delta_series` from the training target (`gross_kwh_actual − override_delta`), mirroring the existing `_subtract_sub_sensors` pattern exactly (same shape: a correction series, subtracted from the target column, clipped at zero). The model trains on "consumption as if no DHW override ever happened," consistent with the now-override-blind `physics_kwh` feature it's given.

**Serving (`_update_sensors`):**
1. ML predicts the override-blind baseline, unchanged from today's Phase 1 path.
2. For any hour where `committed_override` has an active entry, compute that hour's `override_delta` (using the real committed `target_c`) and add it **unconditionally** to the final published forecast for that hour — never routed through the model, so double-counting is structurally impossible (the ML-facing feature never carried the override signal to begin with).

## EM-side changes (`heat_pump.py`)

`_commit_dhw_schedule` (already being generalized by #93 to accept `kind`/`date_iso`/`hour`) additionally passes the real target temperature: `dhw_schedule={"comfort_boost": (date_iso, hour, target_c)}` where `target_c = self._dhw_boost_target_c` at commit time. Legionella's call sites continue passing a 2-element `(date_iso, hour)` — hef treats a missing `target_c` as "use `T_legionella`," so legionella's payload shape is unchanged and no EM-side legionella call site needs editing.

## Error handling / edge cases

- No committed override ever (fresh install, or between overrides): `committed_override` is `None`/empty and `override_history` is empty — `override_delta_series` is all-zero everywhere, both paths degrade to exactly today's Phase 1 behavior. No regression risk for installs that never use DHW overrides.
- Missing/stale `committed_override` entries: governed by #93's existing staleness-expiry mechanism, unchanged.
- `override_history` lookup miss for a historical row (e.g. pre-migration data with no history recorded yet): treat as zero delta, log at INFO once per encountered gap (not WARNING — an expected, self-healing condition as history accumulates), never raise.
- Malformed `override_history` entry (bad kind, missing `target_c`, unparseable date): skip with a WARNING, matching this file's existing never-raises style for schedule/config handling (same as #93's malformed-entry handling).

## Testing

- **Unit (`tests/test_physics.py`):**
  - `override_delta_series` correctness for `legionella` (fixed `T_legionella`) and `comfort_boost` (variable `target_c`) kinds — delta reflects the real committed temp, not a fixed constant, for comfort_boost.
  - Target-subtraction / serve-time-re-addition round-trips to the original value when nothing else about a forecast changes (mirrors the symmetry check pattern already used for `_subtract_sub_sensors`).
  - `override_history` reconstruction returns the entry actually committed for a given historical `(kind, date, hour)`, not just the latest value across all history — this is the direct regression test for the reconstruction-fidelity bug.
  - `physics_kwh` (ML feature) shows no override-shaped spike on a day with a committed override — confirms the feature is genuinely override-blind, not just override-underweighted.
  - Zero-override degradation: with empty `committed_override`/`override_history`, both training and serving paths produce output identical to current Phase 1 behavior.
- **Integration (`tests/test_energy_forecast.py`):** a synthetic override day shows the *full* expected delta in the final published forecast sensor, independent of what the trained model happens to weigh — the direct regression test for the original live bug (2026-08-04).
- **Cross-repo (`ha-energy-manager` `tests/test_heat_pump.py`):** `_commit_dhw_schedule` call for comfort-boost carries the real `_dhw_boost_target_c`, not a hardcoded/approximated value.

## Migration

No backfill of `override_history` for overrides committed before this ships — training reconstruction simply treats pre-migration history as zero-delta (see Error handling above), which is a strict improvement over today's "always wrong except the most recent" behavior, not a regression.
