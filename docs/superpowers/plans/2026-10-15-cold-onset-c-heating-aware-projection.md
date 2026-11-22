# Heating-Aware Indoor Projection — Implementation Plan (C)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Projected room temperatures and thermal pressure over the 48 h forecast stay realistic while heating runs, which removes the +6 kWh/day cold-day over-forecast.

**Architecture:** `_project_indoor_temps()` (`model.py:2895`) has two defects. (1) It is a pure cool-down ODE with no heating term: rooms decay toward outdoor temperature (tau 11.6 h → ~11.5 °C at 24–35 h lead), so projected pressure reaches 4.6 °C against 0.02 °C real; the training maximum is 0.72 °C. (2) During heating-on hours it compares every room to one shared `setpoint_on` (the mean of the live setpoints, ~19.4 °C) instead of each room's own setpoint (18–21 °C), which adds up to ~1.4 °C of fake deficit per room. Fix: while heating is on, each room tracks its own live setpoint (if heating was already on at hour 0) and a thermostat floor holds it at its current deficit. Training is untouched, because training rows already use measured deficits, so serving becomes train-consistent and no retrain is needed for correctness. `physics.py:_t_indoor_for_window` calls the same function, so `physics_kwh` and the DHW-override tank ambient are fixed with it.

**Tech Stack:** Python 3.13, numpy, pandas, pytest.

**Spec / evidence:** index row C; `memory/project_cold_onset_overforecast_2026_10_15.md`; user hypothesis 2026-10-15 ("if it doesn't project heating of those rooms, projected thermal pressure gets too high").
Live-model replay (`scripts/replay_indoor_projection.py`, live `data/models` trained 2026-10-14, origins 10-08..10-14 00:00):

| variant | MAE 0–23 h | MAE 24–47 h | bias 24–47 h (kWh/day) |
|---|---|---|---|
| current projection | 0.404 | 0.485 | +6.88 |
| heating floor, shared mean setpoint | 0.346 | 0.465 | +6.18 |
| **heating floor + per-room setpoint (this plan)** | **0.207** | **0.255** | **−2.15** |

Caveats:
- These days are in-sample for the live model (trained through 10-14). The projection only changes serve-time features, so the relative comparison stands (reproduced independently by review).
- The replay calls `predict()` **without `physics_model`**, so the live model's `physics_kwh` feature is 0-filled in every variant. Part of the −2.15 kWh/day residual may be that artefact; the residual is **unvalidated**. Physics' own heating term is inactive live anyway (`UA_eff` null), so the projection fix's effect through physics is DHW-ambient only.
- The replay used a heating series that is ON for all 48 h, so it validates the "heating on at hour 0" branch only. The off→on branch below is design, not evidence.

**Base branch:** `dev` → `fix/heating-aware-indoor-projection`. Independent of A/B. Ship it **first**: it is the dominant error.

## Global Constraints
- Tests via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`.
- Return contract unchanged: `{entity_id: DataFrame[timestamp, current_temp, setpoint, deficit]}`.
- Heating-off hours keep today's free-float cool-down exactly.
- **Heating on at hour 0** (validated branch): every on-hour uses the room's own latest setpoint, and the floor holds the room's current deficit d0 = max(0, room_setpoint − T_in[0]). This is identical to the replayed candidate (`scripts/indoor_projection_candidate.py`, verified equal on 300 random cases).
- **Heating off at hour 0, on later** (season onset): on-hours use the shared `setpoint_on` as today, with **d0 = 0**, so floor = `setpoint_on`. The reference stays consistent; mixing a room-setpoint d0 with a shared-setpoint floor is a bug found in review. Known limitation: the room snaps to `setpoint_on` and warm-up demand isn't shown as pressure (index → cross-plan interactions).
- No `heating_active_series` (heating entity not configured) → treat as thermostat-controlled: on for all hours, room's own setpoint.
- **Behaviour change for users without `heating_system_active_entity`:** rooms are now treated as thermostat-held at their own setpoint for the whole window (previously they free-floated). This is closer to how training data looks (measured rooms sit near setpoint); call it out in the CHANGELOG. `test_no_series_is_thermostat_controlled` pins it.
- The floor only ever *raises* a projected temperature (`max(free, floor)`), so rooms above setpoint (sun, cooling-mode setups) behave as before.
- The thermostat floor snaps instantly when heating switches on (no ramp model). This is accepted; heat-up energy is carried by the temperature features.

## Review Focus
1. Room colder than its setpoint at hour 0 (live deficit 0.5 °C, heating on) → that 0.5 °C deficit is held constant, neither zeroed nor grown.
2. Room warmer than its setpoint (sun, 22 °C vs 18 °C) → cools freely toward its setpoint, then the floor holds it there (deficit stays 0).
3. Room in frost/away mode (setpoint 7 °C, seen live) with heating on → its own 7 °C is used, so no fake deficit against 19–21 °C.
4. Stale observation (> 2 h) → start = setpoint (unchanged), so d0 = 0.
5. Physics consumer (`_t_indoor_for_window`) → projected indoor temperature stays near setpoint, not near outdoor; `tests/test_physics.py` around L740 must still pass or be updated with a reason.

---

### Task 1: Heating-aware `_project_indoor_temps`

**Files:**
- Modify: `apps/energy_forecast/model.py:2895-2993` (`_project_indoor_temps`). Rationale: the single function behind both the ML thermal-pressure features (model.py:1050) and physics `_t_indoor_for_window` (physics.py:516), so one fix covers both consumers.
- Test: `tests/test_setpoint_projection.py` (new class `TestHeatingAwareProjection`; update the 6 tests listed in Step 1). Rationale: the existing setpoint/blend tests live here and encode the old semantics.
- Test: `tests/test_model.py` (`TestProjectIndoorTemps::test_basic_convergence`, `::test_default_tau_applied_when_none`). Rationale: they test the cool-down physics, which now needs an explicit heating-off series.

**Interfaces:**
- Signature unchanged: `_project_indoor_temps(climate_recent, future_timestamps, outdoor_temp_series, tau_hours=None, heating_active_series=None, setpoint_on=None, setpoint_off=None) -> dict[str, pd.DataFrame]`

- [ ] **Step 1: Write the failing tests** (`tests/test_setpoint_projection.py`, reusing the file's `_future_ts`, `_outdoor_series`, `_climate_recent`, `_heating_active_series` helpers):

```python
class TestHeatingAwareProjection:
    """Rooms are thermostat-controlled: while heating is on they don't cool toward outdoor (2026-10-15)."""

    def _proj(self, cr, ts, states, setpoint_on=19.4, setpoint_off=12.0, t_out=5.0):
        return _project_indoor_temps(cr, ts, _outdoor_series(ts, t_out), tau_hours=11.64,
                                     heating_active_series=_heating_active_series(ts, states),
                                     setpoint_on=setpoint_on, setpoint_off=setpoint_off)["climate.room"]

    def test_heating_on_holds_room_at_setpoint(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=21.0, current=21.0), ts, [1] * 48)
        assert p["current_temp"].min() == pytest.approx(21.0)
        assert p["deficit"].max() == pytest.approx(0.0)

    def test_heating_on_holds_initial_deficit(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=21.0, current=20.5), ts, [1] * 48)
        assert p["deficit"].values == pytest.approx([0.5] * 48)

    def test_per_room_setpoint_used_when_on_at_hour_zero(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=18.0, current=18.0), ts, [1] * 48, setpoint_on=19.4)
        assert np.all(p["setpoint"].values == pytest.approx(18.0))
        assert p["deficit"].max() == pytest.approx(0.0)  # no fake 1.4 °C deficit vs the shared mean

    def test_frost_mode_room_stays_at_own_setpoint(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=7.0, current=15.0), ts, [1] * 48)
        assert p["deficit"].max() == pytest.approx(0.0)

    def test_warm_room_cools_to_setpoint_then_holds(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=18.0, current=22.0), ts, [1] * 48)
        t = p["current_temp"].values
        assert t[1] < t[0]
        assert t.min() == pytest.approx(18.0)

    def test_heating_off_still_free_floats(self):
        ts = _future_ts()
        p = self._proj(_climate_recent(ts, setpoint=21.0, current=21.0), ts, [0] * 48)
        t = p["current_temp"].values
        assert t[-1] < 10.0  # tau 11.6 h, T_out 5 → well below setpoint after 48 h

    def test_off_then_on_uses_shared_setpoint_on_with_zero_d0(self):
        ts = _future_ts()
        # room below its own setpoint at hour 0 (19 vs 20) — that d0 must NOT leak into the later setpoint_on floor
        p = self._proj(_climate_recent(ts, setpoint=20.0, current=19.0), ts, [0] * 10 + [1] * 38, setpoint_on=21.0)
        assert np.all(p["setpoint"].values[10:] == pytest.approx(21.0))
        assert p["current_temp"].values[10:].min() == pytest.approx(21.0)   # floor = setpoint_on − 0
        assert p["current_temp"].values[1:10].max() < 19.0                   # off hours still free-float

    def test_no_series_is_thermostat_controlled(self):
        ts = _future_ts()
        p = _project_indoor_temps(_climate_recent(ts, setpoint=20.0, current=20.0), ts, _outdoor_series(ts, 0.0),
                                  tau_hours=11.64)["climate.room"]
        assert p["current_temp"].min() == pytest.approx(20.0)
```

**Existing tests this breaks (8, all verified by review on a patched copy). Update exactly these:**
1. `tests/test_setpoint_projection.py::TestSetpointProjection::test_heating_on_uses_setpoint_on`: heating is on at hour 0, so the room's own setpoint (12.0) now wins. Assert `np.all(sp == pytest.approx(12.0))`; docstring: "heating already ON at hour 0 → room's own live setpoint (production passes setpoint_on = live mean in this state anyway)".
2–5. `TestDeficitBlending::test_near_term_uses_live_setpoint_even_when_hysteresis_projects_on`, `::test_interpolates_between_full_trust_and_blend_hours`, `::test_alpha_reaches_one_exactly_at_blend_hours`, `::test_setpoint_column_unaffected_by_blending`.
6. `TestThermalPressureWithHysteresis::test_pressure_positive_when_on`.
   - For 2–6: they use "live setpoint 12, current 19, heating on at hour 0, setpoint_on 21". Under the new rule, hour-0-on means the room setpoint (12) for all hours, so their intended live-vs-hysteresis divergence disappears.
   - Remedy (verified): change the heating series `[1]*8` / `[1]*6` to `[0]*8` / `[0]*6`, and swap to `setpoint_on=12.0, setpoint_off=21.0`. The off branch has no floor, so live and hysteresis deficits diverge exactly as the tests intend. Update each docstring's wording ("hysteresis projects ON" → "hysteresis setpoint differs from live").
7. `tests/test_model.py::TestProjectIndoorTemps::test_basic_convergence`: tests cool-down, so pass `heating_active_series=pd.Series(0, index=ts), setpoint_on=20.0, setpoint_off=12.0`.
8. `tests/test_model.py::TestProjectIndoorTemps::test_default_tau_applied_when_none` (asserts `t_in[1] == 10 + (5 − 10)/DEFAULT_TAU`): same remedy as 7.

With these edits, 30/30 tests in those classes pass (review). `tests/test_physics.py` (incl. ~L740) passes unchanged.

- [ ] **Step 2: Run to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_setpoint_projection.py tests/test_model.py::TestProjectIndoorTemps -v`
Expected: new tests FAIL (rooms cool; per-room test sees setpoint 19.4); the updated `test_heating_on_uses_setpoint_on` fails (sees 21.0).

- [ ] **Step 3: Implement.** Replace the block from `# Vectorised Euler forward over 48 steps` through the `deficit_arr = …` line with:

```python
        n = len(future_timestamps)
        room_setpoint = float(latest["setpoint"])

        # Heating state per step. No series → thermostat-controlled for the whole window.
        if heating_active_series is not None and setpoint_on is not None and setpoint_off is not None:
            on_arr = heating_active_series.reindex(future_timestamps, method="nearest").fillna(0).values.astype(bool)
            # Heating already on at hour 0 → this room's thermostat setpoint is the truth for every
            # on-hour. Comparing all rooms to the shared mean (setpoint_on) invented ~1.4 °C of
            # deficit for 18 °C rooms (2026-10-15). Turning on later → shared setpoint_on, as before.
            on_setpoint = room_setpoint if on_arr[0] else setpoint_on
            setpoint_arr = np.where(on_arr, on_setpoint, setpoint_off)
        else:
            on_arr = np.ones(n, dtype=bool)
            setpoint_arr = np.full(n, room_setpoint)

        # RC cool-down with a thermostat floor: while heating is on the room is held at its
        # current deficit below the active setpoint instead of decaying toward outdoor. A pure
        # cool-down projected ~4.6 °C of pressure at 24–35 h lead vs 0.02 °C real (2026-10-15).
        # d0 is measured against the same setpoint the floor uses: the room's own setpoint when heating
        # is already on at hour 0; when heating only switches on later the floor is the shared
        # setpoint_on with d0 = 0 (season onset; room assumed brought to target).
        initial_deficit = max(0.0, room_setpoint - t_in_start) if on_arr[0] else 0.0
        t_in_arr = np.empty(n)
        t_in_arr[0] = t_in_start
        for i in range(1, n):
            free = t_in_arr[i - 1] + (t_out_arr[i - 1] - t_in_arr[i - 1]) / tau
            t_in_arr[i] = max(free, setpoint_arr[i] - initial_deficit) if on_arr[i] else free

        # ── Deficit blend: live setpoint (near-term) → hysteresis-projected
        # setpoint (far-term) ────────────────────────────────────────────
        # <re-insert here, verbatim, the existing explanatory comment lines that sit between this
        #  header and `live_deficit_arr = …` in today's code (they're inside the replaced range)>
        live_deficit_arr = np.maximum(0.0, room_setpoint - t_in_arr)
        hyst_deficit_arr = np.maximum(0.0, setpoint_arr - t_in_arr)
        hours_ahead = np.arange(n, dtype=float)
        blend_span = max(SENSOR_BLEND_HOURS - SENSOR_FULL_TRUST_HOURS, 1)
        alpha = np.clip((hours_ahead - SENSOR_FULL_TRUST_HOURS) / blend_span, 0.0, 1.0)
        deficit_arr = live_deficit_arr * (1.0 - alpha) + hyst_deficit_arr * alpha
```
The replaced range already covers the old `setpoint_val = …` line and the old `if heating_active_series … setpoint_arr = …` block. `t_out_arr` stays computed above the new block. Update the docstring formula line to: `T_in[t+1] = max(T_in[t] + (T_out[t] − T_in[t]) / tau, setpoint[t] − d0)` while heating is on, else the free term; `d0` = deficit at hour 0.

- [ ] **Step 4: Run the projection tests, then the full suite**

Run: `…/bin/python -m pytest tests/test_setpoint_projection.py tests/test_model.py::TestProjectIndoorTemps -v`, then `…/bin/python -m pytest tests/ -q`.
Expected: all PASS. Any failure **beyond the 8 listed** (the review couldn't finish the full suite: `test_physics_features`, `test_energy_forecast_physics_config`, `test_heating_season_app`, `test_scenario_service`, `TestZeroFillEliminated` were unconfirmed): read it. If it asserted the old cool-down's magnitude, give it a heating-off series and name it in the commit body. If it reveals a real consumer problem, stop and report.

- [ ] **Step 5: Live-model replay with the real code** (local, gitignored script). In `scripts/replay_indoor_projection.py`, the `orig` variant now runs the fixed function, so:

Run: `D=$(mktemp -d) && cp -r data/models "$D/livemodel" && /home/jovyan/my_envs/ha-energy-forecast/bin/python -W ignore scripts/replay_indoor_projection.py "$D/livemodel"` (or use the session scratchpad dir)
Expected: `projection` column ≈ `clamp_roomsp` (MAE ≈ 0.207 / 0.255). If not, the implementation diverges from the validated candidate (`scripts/indoor_projection_candidate.py`); diff them before going on.

- [ ] **Step 6: ruff + commit**

```bash
/home/jovyan/my_envs/ha-energy-forecast/bin/ruff check apps tests && /home/jovyan/my_envs/ha-energy-forecast/bin/ruff format apps tests
git add apps/energy_forecast/model.py tests/test_setpoint_projection.py tests/test_model.py
git commit -m "fix: heating-aware indoor projection — thermostat floor + per-room setpoint"
```

---

### Task 2: Docs + finalize

- [ ] `@changelog-writer`: "Fixed: forecast over-predicted heating-season consumption by ~6 kWh/day. The 48 h indoor-temperature projection let rooms cool toward outdoor temperature while heating was on and compared all rooms to one shared setpoint, inflating projected thermal pressure 6× beyond anything seen in training. Rooms now hold their own setpoint while heating is on." README: if it describes the RC projection ("rooms cool toward outdoor"), update that sentence.
- [ ] deploy-agent Workflow A → next `v0.12.0-alpha-N`. No apps.yaml change, no forced retrain needed (training is unaffected).
- [ ] Immediately after deploy: `sensor…energy_forecast_tomorrow` should fall from ~19–20 to roughly 10–13 kWh on a cold day and sit inside its p10–p90 band. `sensor.ha_energy_forecast_thermal_pressure_net` should be unchanged (it reads hour 0).

### Task 3: Live verification (≥ 3 cold days after deploy)

- [ ] `pull_ha_data.py`, then recompute the cold-day daily table (pred vs actual, from the investigation) for post-deploy days only. Targets: cold-day bias within ±1.5 kWh/day; `mae_7d` < 0.30 and falling.
- [ ] Criterion per the index's verification section: cold-day 24–47 h bias moved from ≈ +6.9 into **[−3, +1] kWh/day** (−2 is the *expected* outcome of the validated candidate, so it's not a failure). If it lands below −1.5 and stays there for ≥ 14 cold days: first check Task 4's result (`heat_loss_deg`). The replay's residual is unvalidated (physics_kwh was 0-filled), so re-measure live first. Other suspect: the per-step floor removes *all* night-setback recovery energy, since the floor snaps instantly. Candidate: model the recovery ramp, or use the schedule's daytime setpoint instead of the hour-0 value when the origin falls in a setback. Second suspect: the fallback below.
- [ ] **Fallback, only if the bias stays > +1.5 kWh/day:** the earlier hypothesis that pre-2026-03-31 training rows carry `thermal_pressure = 0`. Revive the `thermal_pressure_missing: zero|nan|drop` design (training-side NaN mask exempt from `dropna` at `model.py:691`, LightGBM only) as its own plan.
- [ ] Update memory + MEMORY.md with the post-deploy numbers.

---

### Task 4: Candidate feature `heat_loss_deg` (steady-state heat-loss gradient) — gated

**Expectation management (review):** before 2026-03-31 every training row uses the 20 °C default, so there `heat_loss_deg = 20 − T_out` is nearly collinear with the existing `heating_degree`/`heating_deg_sum_*`. Only post-March rows carry new information, and the gain may be small. At serve the held T_in comes from the origin hour (possibly a night setback) for 48 h, while training sees measured, varying T_in. This is a mild skew; the gate decides.

**Why:** Thermal pressure only measures catch-up demand (room below its setpoint). The *maintenance* load (≈ UA·(T_in − T_out)/COP) is carried today only by outdoor-only features: `heating_deg_sum_*` uses a fixed 18 °C base, plus `temp_*`, lags and `regime_kwh`. Those don't know the real room temperatures. `physics_kwh` would carry UA·(T_in − T_out), but `UA_eff` is still `null`. `heat_loss_deg` gives the model the real gradient: measured T_in in training, the held projection from Task 1 at serve. It's the prime suspect for Task 3's residual −2 kWh/day, and it's independent of UA calibration. **Run after Task 1 is merged** (serve must use the held projection).

**Files:**
- Modify: `apps/energy_forecast/const.py` — `HEAT_LOSS_DEFAULT_INDOOR_C = 20.0`. Rationale: shared constant, project rule.
- Modify: `apps/energy_forecast/model.py` — new block in `_engineer_features` after the `infiltration_pressure` line (~L3332); `train()` signature + `base_features` (~L676). Rationale: features are computed in `_engineer_features` for both train and serve (one code path = no train/serve skew); `base_features` is where opt-in features are added.
- Modify: `apps/energy_forecast/energy_forecast.py` — arg `heat_loss_feature` (bool, default False) passed to every `self._ml_model.train(` call. Rationale: opt-in config so the gate, not the merge, decides.
- Create (local, gitignored): `scripts/backtest_heat_loss_feature.py`
- Test: `tests/test_model.py` (new class `TestHeatLossDeg`)

**Interfaces:**
- Produces: column `heat_loss_deg` always computed by `_engineer_features`; it is in `feature_cols` only when `train(..., heat_loss_feature=True)`; persisted via `feature_cols` (serve needs no flag).
- Produces: apps.yaml `heat_loss_feature: true|false` → `self._heat_loss_feature`.

- [ ] **Step 1: Failing tests** (use `from tests._frames import train_frames`; create `tests/_frames.py` exactly as specified in Plan B Task 2 Step 1 if it doesn't exist yet)

```python
class TestHeatLossDeg:
    def _clim(self, ts, t_in):
        return {"climate.a": pd.DataFrame({"timestamp": ts, "current_temp": t_in, "setpoint": 20.0})}

    def test_uses_area_weighted_indoor_minus_outdoor(self):
        from energy_forecast.model import _heat_loss_deg
        ts = pd.date_range("2026-01-10", periods=3, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "temp_c": [5.0, 25.0, 0.0]})
        clim = {"climate.a": pd.DataFrame({"timestamp": ts, "current_temp": 21.0, "setpoint": 21.0}),
                "climate.b": pd.DataFrame({"timestamp": ts, "current_temp": 18.0, "setpoint": 18.0})}
        out = _heat_loss_deg(df, clim, room_areas={"climate.a": 30.0, "climate.b": 10.0})
        assert list(out.round(3)) == [15.25, 0.0, 20.25]  # T_in = 20.25 area-weighted; clipped at 0

    def test_falls_back_to_default_indoor_without_climate(self):
        from energy_forecast.const import HEAT_LOSS_DEFAULT_INDOOR_C
        from energy_forecast.model import _heat_loss_deg
        ts = pd.date_range("2026-01-10", periods=2, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "temp_c": [5.0, 30.0]})
        assert list(_heat_loss_deg(df, None, None)) == [HEAT_LOSS_DEFAULT_INDOOR_C - 5.0, 0.0]

    def test_partial_room_coverage_uses_present_rooms(self):
        from energy_forecast.model import _heat_loss_deg
        ts = pd.date_range("2026-01-10", periods=4, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "temp_c": 10.0})
        clim = {"climate.a": pd.DataFrame({"timestamp": ts, "current_temp": 22.0, "setpoint": 22.0}),
                "climate.b": pd.DataFrame({"timestamp": ts[2:], "current_temp": 18.0, "setpoint": 18.0})}
        # rows 0–1: only room a present → 22 − 10; rows 2–3: (22 + 18)/2 − 10 (equal default areas)
        assert list(_heat_loss_deg(df, clim, None)) == [12.0, 12.0, 10.0, 10.0]

    def test_rows_before_climate_coverage_use_default(self):
        from energy_forecast.model import _heat_loss_deg
        ts = pd.date_range("2026-01-10", periods=4, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "temp_c": 10.0})
        clim = {"climate.a": pd.DataFrame({"timestamp": ts[2:], "current_temp": 22.0, "setpoint": 22.0})}
        assert list(_heat_loss_deg(df, clim, None)) == [10.0, 10.0, 12.0, 12.0]

    @pytest.mark.parametrize("flag", [False, True])
    def test_flag_controls_feature_cols(self, tmp_path, flag):
        energy_df, weather_df = train_frames()
        m = EnergyForecastModel(tmp_path, timezone="Europe/Zurich")
        m.train(energy_df, weather_df, outdoor_df=None, weight_halflife_days=0, heat_loss_feature=flag)
        assert ("heat_loss_deg" in m.feature_cols) is flag
```
Plus in `tests/test_energy_forecast.py`: `heat_loss_feature: true` in app args reaches `train(..., heat_loss_feature=True)`. Mirror the existing test that asserts a config arg is forwarded to `train` (grep `weight_halflife_days=` in that file's train-call assertions).

- [ ] **Step 2: Run** → FAIL (`ImportError: _heat_loss_deg`, `TypeError: heat_loss_feature`).

- [ ] **Step 3: Implement**

`const.py`:
```python
# Assumed indoor temperature for heat-loss degree-hours on rows without climate data
# (pre-climate-cache history or no climate_entities configured).
HEAT_LOSS_DEFAULT_INDOOR_C = 20.0
```
`model.py` module-level helper (next to `_mask`/`_compute_likely_ev_hours` helpers), importing `HEAT_LOSS_DEFAULT_INDOOR_C` and reusing `DEFAULT_ROOM_AREA_M2`:
```python
def _heat_loss_deg(df: Any, climate_dfs: dict | None, room_areas: dict | None) -> Any:
    """Steady-state heat-loss gradient max(0, T_in − T_out) per row (°C).

    T_in is the area-weighted indoor temperature: measured in training, the held
    projection at serve (_project_indoor_temps). Rows without climate data use
    HEAT_LOSS_DEFAULT_INDOOR_C, so it degrades to a 20 °C-base degree-hour.
    Unlike thermal_pressure (deficit vs setpoint = catch-up demand) this carries the
    maintenance load that keeps rooms warm.
    """
    import numpy as np
    import pandas as pd

    ts = pd.to_datetime(df["timestamp"]).dt.floor("1h")
    t_in = pd.Series(np.nan, index=df.index)
    parts, weights = [], []
    for eid, c in (climate_dfs or {}).items():
        if c.empty or "current_temp" not in c.columns:
            continue
        s = c.assign(_ts=pd.to_datetime(c["timestamp"]).dt.floor("1h")).groupby("_ts")["current_temp"].last()
        parts.append(ts.map(s))
        weights.append((room_areas or {}).get(eid, DEFAULT_ROOM_AREA_M2))
    if parts:
        vals = pd.concat(parts, axis=1)
        w = pd.DataFrame(np.tile(weights, (len(vals), 1)), index=vals.index).where(vals.notna().values)
        t_in = np.nansum(vals.values * w.fillna(0).values, axis=1) / w.sum(axis=1).replace(0, np.nan).values
        t_in = pd.Series(t_in, index=df.index)
    t_in = t_in.fillna(HEAT_LOSS_DEFAULT_INDOOR_C)
    return np.maximum(0.0, t_in - df["temp_c"]).astype(float)
```
In `_engineer_features`, directly after the `infiltration_pressure` line:
```python
    # ── Steady-state heat-loss gradient (2026-10-15) ─────────────────────────
    df["heat_loss_deg"] = _heat_loss_deg(df, climate_dfs, room_areas)
```
(Confirm `_engineer_features` receives `room_areas`; it's used for the area-weighted thermal pressure above. Use the same variable name.)
`train()` signature: add `heat_loss_feature: bool = False,`. After `base_features = [...]` (~L676), before `sensor_features = ...`:
```python
        if heat_loss_feature:
            base_features = [*base_features, "heat_loss_deg"]
```
`energy_forecast.py`: `self._heat_loss_feature: bool = bool(self.args.get("heat_loss_feature", False))`; pass `heat_loss_feature=self._heat_loss_feature` to every `self._ml_model.train(`; add it to the startup config log line.

- [ ] **Step 4: Full suite + ruff → PASS. Commit** `git commit -m "feat: opt-in heat_loss_deg feature (indoor−outdoor gradient)"`

- [ ] **Step 5: Backtest gate** (local, gitignored `scripts/backtest_heat_loss_feature.py`). Train full-feature models at each fold start, predict each day from origins D−1 00:00 (hours 24–47 = what `pred_history` scores) and D 00:00, and compare `heat_loss_feature` False vs True:

```python
#!/usr/bin/env python3
"""Plan 2026-10-15 C Task 4: full-feature 24h-lead backtest of heat_loss_deg (needs Task 1 merged)."""
from __future__ import annotations

import logging, pathlib, sys, tempfile, types
from datetime import date
from unittest.mock import MagicMock

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "apps")); sys.path.insert(0, str(REPO / "scripts"))
if "hassapi" not in sys.modules:
    _s = types.ModuleType("hassapi"); _s.Hass = MagicMock; sys.modules["hassapi"] = _s

import pandas as pd, yaml  # noqa: E402
import backtest_heating_feature as bh  # noqa: E402
from energy_forecast import ha_data  # noqa: E402
from energy_forecast import heating_season as hs  # noqa: E402
from energy_forecast.physics import ThermalPhysicsModel  # noqa: E402

sys.path.insert(0, str(REPO))
from tests.test_physics import DEFAULT_CONFIG as PHYSICS_CONFIG  # noqa: E402  (physics defaults, local script only)
from energy_forecast.model import EnergyForecastModel  # noqa: E402
from energy_forecast.weather import fetch_historical_weather  # noqa: E402

logging.disable(logging.WARNING)
DATA, TZ = REPO / "data", "Europe/Zurich"
FOLDS = [("cold", "2026-10-08", "2026-10-14"), ("sep", "2026-09-10", "2026-09-24")]


def main() -> None:
    cfg = yaml.safe_load((DATA / "apps.yaml").read_text())["energy_forecast"]
    areas = cfg.get("climate_room_areas") or {}
    target = bh.load_target()
    evk = bh._csv("ev_se_one_ev_charger_192_168_1_128_total_energy.csv", "kwh")
    base, ev_df = ha_data.split_ev_charging_hybrid(target, evk, 7.0, charger_kw=9.0)
    switch = bh._csv(bh.SWITCH, "heating_active")
    dhw = bh._csv("dhw_em_kermi_bridge_kermi_hot_water_temperature.csv", "buffer_temp")
    hbuf = bh._csv("physics_heating_buffer_em_kermi_bridge_kermi_buffer_temperature.csv", "heating_buffer_temp")
    clim = {}
    for eid in cfg["climate_entities"]:
        d = pd.read_csv(DATA / f"climate_{eid.split('.', 1)[1]}.csv")
        d["timestamp"] = pd.to_datetime(d["timestamp"], format="mixed"); clim[eid] = d.groupby("timestamp", as_index=False).last()
    subs = {f.stem: bh._csv(f.name, "kwh") for f in DATA.glob("sub_*.csv")}
    weather = fetch_historical_weather(cfg["latitude"], cfg["longitude"], target["timestamp"].min().date(),
                                       min(date.today(), date(2026, 10, 15)), timezone=TZ)  # archive rejects future dates
    meter = bh._csv(bh.HEAT_METER, "kwh")
    day_means = hs.daily_mean_temp(weather)
    cut = lambda d, t: d[d["timestamp"] < t]  # noqa: E731
    rows = []
    for fold, start, end in FOLDS:
        t0 = pd.Timestamp(start) - pd.Timedelta(days=1)
        for flag in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                bh.freeze_now(t0)
                m = EnergyForecastModel(pathlib.Path(tmp), timezone=TZ)
                lab, _ = hs.daily_heating_label(cut(meter, t0), None)
                th = hs.learn_thresholds(lab, day_means[day_means.index < t0], "meter")
                phys = ThermalPhysicsModel(pathlib.Path(tmp) / "phys", PHYSICS_CONFIG)
                m.train(cut(base, t0), cut(weather, t0), outdoor_df=None, weight_halflife_days=cfg.get("weight_halflife_days", 60),
                        country="CH", ev_df=cut(ev_df, t0), heating_active_df=cut(switch, t0),
                        heating_feature_df=hs.hourly_label_df(lab), physics_model=phys,
                        heating_sub_meter_df=cut(meter, t0),
                        sub_sensors_dict={k: cut(v, t0) for k, v in subs.items()},
                        climate_dfs={k: cut(d, t0) for k, d in clim.items()}, room_areas=areas,
                        dhw_df=cut(dhw, t0), heating_buffer_temp_df=cut(hbuf, t0),
                        enable_regimes=cfg.get("enable_regimes", False), regime_count=cfg.get("regime_count", 5),
                        heat_loss_feature=flag)
                for day in pd.date_range(start, end, freq="D"):
                    for lead, origin in (("24h", day - pd.Timedelta(days=1)), ("0h", day)):
                        bh.freeze_now(origin)
                        fc = weather[(weather["timestamp"] >= origin) & (weather["timestamp"] < origin + pd.Timedelta(hours=48))]
                        recent = {k: cut(d, origin).tail(48) for k, d in clim.items()}
                        sp = float(pd.Series([d["setpoint"].iloc[-1] for d in recent.values() if not d.empty]).mean())
                        lab_o, _ = hs.daily_heating_label(cut(meter, origin), None)
                        prior = int(lab_o.iloc[-1]) if not lab_o.empty else 0
                        hseries = hs.project_heating_active(origin, fc, prior, th)   # as production projects it
                        pred = m.predict(fc, live_temp=None,
                                         recent_actuals=base[(base["timestamp"] < origin) & (base["timestamp"] >= origin - pd.Timedelta(days=30))],
                                         sub_sensors_recent={k: cut(v, origin) for k, v in subs.items()},
                                         climate_recent=recent, room_areas=areas,
                                         heating_active_series=hseries,
                                         setpoint_on=sp if prior else 20.0, setpoint_off=12.0 if prior else sp,
                                         physics_model=phys,
                                         dhw_recent=cut(dhw, origin).tail(48), heating_buffer_temp_recent=cut(hbuf, origin).tail(48))
                        p = pred.set_index("timestamp")["predicted_kwh"]
                        p = p[(p.index >= day) & (p.index < day + pd.Timedelta(days=1))]
                        a = base.set_index("timestamp")["gross_kwh"].reindex(p.index); ok = a.notna()
                        rows.append(dict(fold=fold, flag=flag, lead=lead, day=day, pred=p[ok].sum(), act=a[ok].sum(),
                                         ae=(p[ok] - a[ok]).abs().sum(), n=int(ok.sum())))
            print(f"{fold} heat_loss_feature={flag} done", flush=True)
    pd.Timestamp.now = bh._REAL_NOW
    r = pd.DataFrame(rows)
    g = r.groupby(["lead", "fold", "flag"]).agg(ae=("ae", "sum"), n=("n", "sum"), pred=("pred", "sum"), act=("act", "sum"), days=("day", "nunique"))
    g["mae"] = g.ae / g.n; g["bias_kwh_day"] = (g.pred - g.act) / g.days
    print(g[["mae", "bias_kwh_day"]].round(4))
    mae = g.loc["24h", "mae"]
    d = r[(r["lead"] == "24h") & (r["fold"] == "cold")].pivot(index="day", columns="flag", values="ae")
    wins = int((d[True] < d[False]).sum())
    print(f"paired sign test: heat_loss_deg better on {wins}/{len(d)} cold days")
    go = (mae[("cold", True)] <= 0.95 * mae[("cold", False)] and mae[("sep", True)] <= 1.02 * mae[("sep", False)]
          and wins >= 5)
    print("\nGATE:", "GO → set heat_loss_feature: true" if go else "STOP — no ≥5 % cold gain or >2 % Sept regression")


if __name__ == "__main__":
    main()
```
Pre-check: the `heat_loss_feature=False` row's 24 h cold bias is the new baseline. Don't expect it to equal the Task 1 replay (−2.15): that replay 0-filled `physics_kwh`, and this harness retrains with physics, the meter label and the production-style heating projection. It must, however, be far below +5 kWh/day; if it isn't, Task 1 isn't in effect in the harness. Still not included (accepted): `program_histories`, `away_df`/`presence_df`, `canton`. Also report the paired daily sign test (variant better on ≥ 5 of 7 cold days, index → Gate rigor) next to the MAE gate. If the run is slow, `train_frames(480)` in the unit tests keeps the flag tests fast.

- [ ] **Step 6:** GO → `@changelog-writer` (Added: opt-in `heat_loss_feature`), README config row, deploy-agent Workflow A, then set `heat_loss_feature: true` in live apps.yaml (backup first), restart, confirm in the startup config log line, and re-run Task 3's live check. STOP → keep the code (default off), record the table in memory, and move on to Task 3's other suspects.
- [ ] **Optional variant in the same run** (one extra loop value, no extra code path): `infiltration_pressure` currently uses `thermal_pressure` as its temperature-gradient proxy (`model.py` §3 comment, "We use thermal_pressure as a proxy for the temperature gradient"). After Task 1 that's ~0 while heating is on, so wind-driven loss vanishes from the features. If the gate is GO, test `0.01 * wind_kmh * heat_loss_deg` as a follow-up; don't bundle it.
