# Physics-ML Hybrid — Plan Index

Spec: `docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md`
Base branch for all four plans: `dev` (currently `v0.11.7`; the spec's header says `v0.11.4-alpha-2` but that's stale — `dev` and `main` are in sync as of 2026-07-02 except for unrelated doc-only commits on `main`).

## Why four plans instead of one

The spec covers four subsystems with different risk profiles and different activation timing:

1. A pure-physics calculation engine with no dependency on the ML pipeline.
2. A de-risking integration into the existing LightGBM pipeline, active immediately.
3. A structural change to the training target, gated behind a cold-start condition that won't be satisfiable until winter 2026/27 — built and merged now, but dormant in production for months.
4. An operator-facing API extension consumed by `ha-energy-manager`.

Each produces working, independently testable software and can be reviewed, merged, and (for 2 and 4) deployed on its own schedule. Plan 3 is built now per user decision (2026-07-02) but stays behind `use_physics_residual: false` until the cold-start gate clears — see Plan C.

## Execution order

```
Plan A (physics core)
   │
   ▼
Plan B (Phase 1 integration) ──┐
   │                           │
   ▼                           ▼
Plan C (Phase 2, dormant)   Plan D (Scenario API)
```

- **Plan B depends on Plan A** — needs `ThermalPhysicsModel.predict_series()` / `predict_training_series()`.
- **Plan C depends on Plan B** — extends the same `train()`/`predict()` call sites Plan B modifies, and reuses the cold-start gate check Plan B wires into the calibration trigger.
- **Plan D depends on Plan A only** (it calls `predict_series(dhw_schedule_override=...)` directly) — it does **not** depend on Plan B or C and can be built in parallel with either, but the plan is sequenced last here because §5.4's cache-invalidation logic references sensors/config that Plan B introduces (`self._physics_model`, `physics:` config block).

Recommended merge order: A → B → D → C, or A → B → C → D. Do not merge C before B (Phase 2 modifies the same `train()`/`predict()` regions Phase 1 touches — merging out of order guarantees a conflicting rebase).

## Plan documents

| Plan | File | Produces |
|---|---|---|
| A | `2026-07-02-physics-ml-hybrid-a-core-engine.md` | `apps/energy_forecast/physics.py` (`ThermalPhysicsModel`), `_find_passive_windows()`, `models/physics_calibration.json` + `models/physics_schedule.json` read/write, `tests/test_physics.py` (unit + calibration + fallback sections) |
| B | `2026-07-02-physics-ml-hybrid-b-phase1-integration.md` | `physics_kwh` feature wired into `model.py` `_engineer_features()`/`train()`/`predict()`, `physics:` config ingest + `recalibrate_physics` service in `energy_forecast.py`, model-artifact portability fallback |
| C | `2026-07-02-physics-ml-hybrid-c-phase2-residual.md` | Residual-target training path, `physics_base_today`/`ml_adjustment_today` sensors, `model_phase` attribute — all gated behind `use_physics_residual` |
| D | `2026-07-02-physics-ml-hybrid-d-scenario-api.md` | `get_scenario(dhw_schedule=...)`, `set_dhw_schedule` service, cache invalidation, legionella instability-guard bypass |

## Shared interface contract

All four plans reference this contract. If any plan's implementation deviates from it, the deviation must be reflected back into this table before the dependent plan is executed.

```python
# apps/energy_forecast/physics.py

class ThermalPhysicsModel:
    def __init__(self, model_dir: Path, config: dict) -> None: ...

    def predict_series(
        self,
        forecast_df: pd.DataFrame,          # cols: timestamp, temp_c, direct_radiation_wm2 (naive ts, 48 rows typical)
        climate_recent: dict[str, pd.DataFrame] | None = None,   # {climate_entity: df[timestamp, current_temp, setpoint]}
        dhw_recent: pd.DataFrame | None = None,                  # df[timestamp, buffer_temp]
        room_areas: dict[str, float] | None = None,              # {climate_entity: m2}
        heating_active_series: pd.Series | None = None,
        setpoint_on: float | None = None,
        setpoint_off: float | None = None,
        dhw_schedule_override: dict | None = None,               # {"legionella": ("2026-06-25", 10), "boost": "13:00"}
    ) -> pd.Series: ...          # indexed by forecast_df["timestamp"], hourly physics_kwh

    def predict_training_series(
        self,
        energy_df: pd.DataFrame,            # cols: timestamp, gross_kwh — used only for the timestamp index
        weather_df: pd.DataFrame,           # cols: timestamp, temp_c, direct_radiation_wm2
        climate_dfs: dict[str, pd.DataFrame] | None = None,      # actual per-room readings, not projected
        dhw_df: pd.DataFrame | None = None,
        room_areas: dict[str, float] | None = None,
    ) -> pd.Series: ...          # historical physics_kwh_series aligned to energy_df["timestamp"]

    def calibrate(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        climate_dfs: dict[str, pd.DataFrame] | None,
        dhw_df: pd.DataFrame | None,
        holdout_cutoff: pd.Timestamp,       # calibration windows must satisfy timestamp < holdout_cutoff
    ) -> None: ...                # writes physics_calibration.json + physics_schedule.json atomically

    @property
    def is_cold_start_gated(self) -> bool: ...   # True if n_calibration_windows_ua_eff < 30
    @property
    def calibration_stale(self) -> bool: ...     # True if calibrated_at missing/unparseable/>30 days old

# module-level, shared by τ calibration (model.py) and UA_eff calibration (physics.py)
def _find_passive_windows(
    df: pd.DataFrame,             # cols: timestamp, T_outdoor, T_indoor, hp_running (bool), dhw_tank_temp (float|NaN)
    *,
    min_delta_t: float = 8.0,
    min_hp_off_hours: int = 2,
) -> pd.Index: ...
```

`ThermalPhysicsModel(model_dir, config)` — `model_dir` is the same `Path(__file__).parent / "models"` directory `EnergyForecastModel` already uses (`energy_forecast.py:288`); `config` is the parsed `physics:` block dict (defaults applied) built in Plan B.

## Config block (introduced in Plan B, read in full by Plan A)

```yaml
physics:
  cop_sensor: sensor.kermi_cop
  dhw_tank_temp_sensor: sensor.kermi_dhw_buffer_temp
  heating_buffer_temp_sensor: sensor.kermi_heating_buffer
  heating_curve_sensor: sensor.kermi_parallel_shift
  cop_formula: {a: 2.5, b: 0.07}
  dhw_tank_volume_l: 200
  dhw_power_w: 4000
  internal_gains_fraction: 0.8
  heating_curve_points: [[-20, 55.5], [-5, 46.0], [5, 39.5], [20, 25.0]]
  room_thermostats:
    - climate_entity: climate.living_room
      temp_sensor: sensor.netatmo_living_room_temp
      area_m2: 35
  use_physics_residual: false
```

**Design note on `room_thermostats` vs. existing `climate_entities`/`climate_room_areas`:** the codebase already has `self._climate_entities` + `self._climate_room_areas` (`energy_forecast.py:249,252`), fed by `ha_data.fetch_climate_history()` which reads a climate entity's own `current_temperature` attribute. The physics spec's `room_thermostats` block is deliberately separate — `temp_sensor` points at a dedicated sensor (e.g. Netatmo) that is more accurate than the heat pump's built-in probe, per spec §3.2. Plan A/B fetch `temp_sensor` history via `ha_data.fetch_generic_sensor_history()` (existing function, already used for `dhw_df`/`heating_active_df`) for the *actual* T_indoor used in training and UA_eff calibration, while continuing to use `climate_entity` + the existing `_project_indoor_temps()` for the *projected* T_indoor used at prediction time. Do not conflate the two config blocks or try to unify them — they serve different accuracy/purpose needs and unifying is out of scope.

## Testing

Each plan's tasks end with `python -m pytest tests/ -v` (full suite, not just new tests) per project CLAUDE.md. `tests/test_physics.py` is created in Plan A and extended in Plans B/C/D.
