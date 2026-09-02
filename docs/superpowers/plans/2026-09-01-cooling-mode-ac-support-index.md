# Cooling Mode / AC Support — Plan Index

Spec: `docs/superpowers/specs/2026-09-01-cooling-mode-ac-support-design.md` (rev. 3)
Base branch for all four plans: `dev`

## Why four plans instead of one

The spec is one cohesive layered feature — not independent subsystems — but its 16 implementation tasks fall into four groups with different risk profiles and activation timing:

1. **Plumbing** — config, historical fetch, `cooling_active` threading through the full train/predict call chain, conflict resolution. Behaviorally **inert by construction**: nothing reads `cooling_active` for any formula yet, so this lands with zero output change even with `cooling_mode_enabled: true`.
2. **Feature semantics** — the five formula changes that make cooling actually affect predictions (`thermal_pressure` sign, `_net`, `_cop`, `cooling_load_sum_*`, `defrost_risk`).
3. **Safety & correctness** — τ-calibration guard, regime-clustering exclusion, model-artifact rollback fallback. The spec is explicit these are **not deferrable**: round-1 review classed skipping them as the same contamination-bug class as #82 (EV-day centroid contamination), not an open design question.
4. **Finalization** — `_FEATURES_BASE`/SHAP registration, the no-regression test against pre-change `dev` output, docs.

Each produces working, independently testable software and can be reviewed and merged on its own schedule. **Plan 3 must land before Plan 2's changes are ever tagged for release or deployed with `cooling_mode_enabled: true`** — see the gotcha below.

## Execution order

```
Plan A (plumbing)
   │
   ▼
Plan B (feature semantics)
   │
   ▼
Plan C (safety & correctness)
   │
   ▼
Plan D (finalization)
```

Strictly linear, no parallelism — this differs from the physics-ml-hybrid plan set's diamond shape. Each plan depends on the previous one's exact interfaces:

- **Plan B depends on Plan A** — needs `cooling_active_df`/`cooling_active_series` threaded through `_engineer_features()`/`train()`/`predict()`/etc., and the conflict-resolved `df["cooling_active"]` column.
- **Plan C depends on Plan B** — specifically, Plan C's model-artifact rollback fallback (its Task 3) references the `cooling_load_sum_24h` column name Plan B's Task 4 (§4.7) creates (`cooling_load_sum_168h` was also created by Plan B's Task 4 at the time but was removed post-Plan-C, pre-Plan-D — see `2026-09-02-cooling-load-sum-drop-168h.md`). Plan C's τ-guard and clustering-exclusion tasks only need Plan A's `cooling_active_df`, but the whole plan is sequenced after B for a single coherent dependency chain.
- **Plan D depends on Plan C** — registers `cooling_active`/`cooling_load_sum_24h` as trained features (`_FEATURES_BASE`) and runs the no-regression test against the complete feature set.

## Plan documents

| Plan | File | Produces |
|---|---|---|
| A | `2026-09-01-cooling-mode-ac-support-a-plumbing.md` | `warn_once` promoted to `const.py`; cooling config keys + `_validate_config()`; `ha_data.fetch_hvac_mode_history()`/`hvac_mode_to_active()`; `cooling_active_df`/`cooling_active_series` threaded through `train()`/`_engineer_features()`/`_prepare_prediction_X()`/`predict()`/`predict_intervals()`/`shap_summary()`/`_project_indoor_temps()`; §4.2 conflict resolution |
| B | `2026-09-01-cooling-mode-ac-support-b-feature-semantics.md` | Mode-aware `thermal_pressure`/`thermal_pressure_net`/`thermal_pressure_cop` (config-exposed EER proxy); `cooling_load_sum_24h` (originally also `_168h`, removed — see `2026-09-02-cooling-load-sum-drop-168h.md`); `defrost_risk` cooling exclusion |
| C | `2026-09-01-cooling-mode-ac-support-c-safety-correctness.md` | `_calibrate_tau()` passive-window guard; `DailyProfileClusterer`/`find_optimal_k()` cooling-day exclusion; model-artifact rollback fallback for the (now 2) new columns |
| — | `2026-09-02-cooling-load-sum-drop-168h.md` | Removes `cooling_load_sum_168h` (permanent train/serve window mismatch, spec §7) before Plan D registers trained features; retargets `cooling_load_sanity_bound`'s check onto `cooling_load_sum_24h` |
| D | `2026-09-01-cooling-mode-ac-support-d-finalization.md` | `_FEATURES_BASE`/`_SHAP_FEATURE_LABELS` registration; no-regression test (`X.columns` + holdout MAE vs. pre-change `dev`); CHANGELOG/ROADMAP/MEMORY updates |

## Gotchas

- **Do not tag or deploy a release with Plan B merged but Plan C not yet merged.** A contributor enabling `cooling_mode_enabled: true` against that state would silently corrupt τ-calibration (summer AC-active hours misclassified as passive decay) and regime-cluster centroids (AC-driven midday spikes contaminating "Workday"/"Weekend" shapes) — the exact bug class #82 already fixed for EV days. Plan A and Plan B alone are safe to sit on `dev` briefly (mirroring how physics-ml-hybrid's Plan A sat on `dev` with no caller) since `cooling_mode_enabled` defaults to `false` and this deployment never sets it true, but don't cut a `main` release or advise a contributor to enable cooling mode until Plan C has also landed.
- **`cooling_active`/`cooling_load_sum_24h` are computed but inert (never selected as trained features) between Plan B landing and Plan D landing.** This is harmless — wasted computation, not a bug — since they aren't added to `_FEATURES_BASE` until Plan D's Task 1. Only `thermal_pressure`/`thermal_pressure_net`/`thermal_pressure_cop` (already pre-existing `_FEATURES_BASE` members) take effect immediately when Plan B lands. (`cooling_load_sum_168h` was also inert during this window before being removed entirely — see `2026-09-02-cooling-load-sum-drop-168h.md`.)
- Each plan's final task is a full-suite regression run (`pytest tests/ -v`) plus a `git diff dev --stat` scope check, mirroring physics-ml-hybrid's closing-task convention — not just its own new tests.

## Implementation Decisions (shared across all four plans)

These are concrete engineering calls made to keep the plans buildable without placeholders, deviating from the spec's literal pseudocode in places. Each is a considered choice — flag any disagreement during Plan A's review before implementation starts, since later plans build on these.

1. **No `cooling_mode_enabled: bool` parameter threaded through `model.py`.** The spec's §4.3/§4.4/§4.6 pseudocode shows `if cooling_mode_enabled and cooling_active_at_ts:`. Instead, `energy_forecast.py` (Plan A) gates the *entire* cooling fetch/projection pipeline on `self._cooling_mode_enabled` — when disabled, `cooling_active_df`/`cooling_active_series` are never built and stay `None`/empty. `model.py` functions then gate purely on presence/value of `cooling_active`, mirroring how `heating_active_df` already works with no separate "heating enabled" flag. Behaviorally identical, satisfies §3's "zero new code paths execute" more literally, avoids widening ~15 signatures with a redundant bool.
2. **`c["setpoint"]` needs no separate `cooling_setpoint_resolved` variable.** At training time `c["setpoint"]` is the real HA-reported thermostat setpoint. At prediction time, Plan A's extended `_project_indoor_temps()` already resolves `c["setpoint"]` per-row using whichever of heating/cooling is projected active. So §4.3's `cooling_setpoint_resolved` *is* `c["setpoint"]` in both contexts by construction.
3. **Cooling's prediction-time projection is flat, not a temp-hysteresis simulation.** Heating's `_build_heating_active_projection()` simulates future on/off transitions using outdoor-temp thresholds not present in this spec's config for cooling. The spec's §4.1a wording — "the same `hvac_mode_entity`/`cooling_system_active_entity` projected **flat**" — is read literally: `_build_cooling_active_projection()` (Plan A) holds the current live on/off reading constant across all 48 future hours. Documented as a v1 approximation alongside §7's other limitations.
4. **`hvac_mode_entity`, when set, is the sole source for `heating_active` too**, not just `cooling_active` — per spec §3's precedence rule. Plan A's fetch layer modifies the *existing* heating-active fetch call sites, not just adds new cooling-only ones.
5. **A new `ha_data.fetch_hvac_mode_history()` is required** (Plan A). The existing `fetch_generic_sensor_history()`/`_fetch_history()` machinery only handles binary/numeric sensors — it cannot carry a mode string like `"cooling"`/`"heating"`/`"dry"`/`"fan_only"`.
6. **`_cooling_conflict_warned: set` lives on `EnergyForecastModel.__init__`** (model.py, Plan A), per the spec's explicit instruction — not on the AppDaemon app class. (`self._excluded_range_warned`, the pattern the spec cites, actually lives on the app class in `energy_forecast.py:371`, not on `EnergyForecastModel` — the spec's analogy is imprecise on this point, but its instruction for where the new attribute goes is unambiguous and is followed as written.)
7. **§5's per-row conditional SHAP label swap is not implemented** (Plan D). The only consumer of `_SHAP_FEATURE_LABELS`, `_build_shap_narrative()`, is fed exclusively by the *aggregate* `shap_summary()` path — which the spec itself says should keep the static heating-flavored label. No separate single-hour narrative pathway exists in the codebase to attach conditional logic to.
8. **§4.2's optional `heating_cooling_conflict_hours_7d` rolling counter is not implemented.** The spec itself marks it optional and untested by design. Skipped per YAGNI.

## Testing

Each plan's tasks end with `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v` (full suite, not just new tests) per project CLAUDE.md. Test files touched across the series: `tests/test_const.py`, `tests/test_ha_data.py`, `tests/test_energy_forecast.py`, `tests/test_model.py`, `tests/test_physics_features.py`, `tests/test_clustering.py`, and a new `tests/test_cooling_regression.py` (Plan D).
