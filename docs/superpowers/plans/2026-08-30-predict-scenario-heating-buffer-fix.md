# predict_scenario() heating_buffer_temp_recent Forwarding Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Base branch:** `dev` (v0.12.0-alpha-16). `main` does not have `heating_buffer_temp_recent`/`predict_scenario` physics wiring at all (confirmed: `git grep heating_buffer_temp_recent origin/main` returns nothing) — this bug and its fix are `dev`-only.

**Goal:** `EnergyForecastModel.predict_scenario()` never forwards `heating_buffer_temp_recent` to either of its internal `self.predict()` calls, so every `energy_forecast/get_scenario` call (legionella candidate scoring, appliance what-ifs) silently predicts against a phantom `0.0` buffer temp instead of the real live sensor value. Fix: add the parameter to `predict_scenario()`, forward it to both internal `predict()` calls, and wire `_get_scenario_cb` to pass the app's cached `self._physics_heating_buffer_df`.

**Architecture:** Two isolated, sequential wiring fixes, each following the existing pattern already used for `room_areas` in the same two functions. No new logic, no schema/behavior change beyond correctly threading an existing, already-supported parameter.

**Tech Stack:** Python 3.13, pandas, pytest. Test env: `/home/jovyan/my_envs/ha-energy-forecast/bin/python`.

**Spec:** No separate spec doc — root cause fully diagnosed and fix scoped in `memory/project_legionella_forecast_warnings_2026_08_19.md` (Fix proposal 1). This plan implements that proposal exactly; read that memory file for the original diagnosis if anything here is unclear.

## Global Constraints

- Always use `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest`, never bare `python`/`pytest` (base conda env is missing `lightgbm`/`holidays` and produces false failures — see `memory/feedback_testing.md`).
- Run the **full** test suite after each task, not just the new test.
- Branch off `dev`, never commit directly to `dev`/`main`.
- Commit messages: concise single-line subject, co-author line required (per global CLAUDE.md).
- Fix proposal 2 from the same memory file (the `lag_24h` NaN-warning threshold) is explicitly **out of scope** for this plan — it's a separate, lower-priority, likely-benign finding per that memory file; do not fix it here.

---

### Task 1: `predict_scenario()` accepts and forwards `heating_buffer_temp_recent`

**Files:**
- Modify: `apps/energy_forecast/model.py:1397-1470` (`predict_scenario()`)
- Test: `tests/test_model.py` (new test in the same class/area as `test_predict_scenario_accepts_room_areas`, ~line 3990)

**Interfaces:**
- Consumes: `EnergyForecastModel.predict()`'s existing `heating_buffer_temp_recent: pd.DataFrame | None = None` parameter (already defined, cols: `timestamp, heating_buffer_temp`) — no changes to `predict()` itself.
- Produces: `predict_scenario(..., heating_buffer_temp_recent: pd.DataFrame | None = None)` — later tasks (Task 2) rely on this exact parameter name.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_model.py`, directly after `test_predict_scenario_accepts_room_areas` (~line 4001):

```python
    def test_predict_scenario_forwards_heating_buffer_temp_recent(self, tmp_path):
        """predict_scenario() must forward heating_buffer_temp_recent to both internal
        predict() calls — regression test for the 2026-08-19 phantom-0.0-buffer-temp bug
        (predict_scenario() previously had no such parameter at all, so every
        get_scenario call silently predicted with heating_buffer_temp=0.0)."""
        from unittest.mock import MagicMock

        m, forecast = _make_trained_model(tmp_path)
        m._appliance_signatures = {
            "sub_dw": {"hourly_profile": [0.1, 0.2], "total_kwh": 0.3, "peak_hour": 1, "n_cycles": 3}
        }
        buffer_df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=3, freq="1h"), "heating_buffer_temp": [45.0, 46.0, 47.0]}
        )
        real_predict = m.predict
        wrapped = MagicMock(side_effect=real_predict)
        m.predict = wrapped

        m.predict_scenario(
            forecast,
            live_temp=None,
            schedule={"sub_dw": "12:00"},
            heating_buffer_temp_recent=buffer_df,
        )

        assert wrapped.call_count >= 1
        for call in wrapped.call_args_list:
            assert call.kwargs.get("heating_buffer_temp_recent") is buffer_df
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_model.py::TestCompositeForecast::test_predict_scenario_forwards_heating_buffer_temp_recent -v`
Expected: FAIL with `TypeError: predict_scenario() got an unexpected keyword argument 'heating_buffer_temp_recent'`.

- [ ] **Step 3: Add the parameter and forward it in both internal `predict()` calls**

In `apps/energy_forecast/model.py`, change the `predict_scenario()` signature (currently lines 1397-1414):

```python
    def predict_scenario(
        self,
        forecast_df: pd.DataFrame,
        live_temp: float | None,
        schedule: dict[str, str | None],
        recent_actuals: pd.DataFrame | None = None,
        sub_sensors_recent: dict | None = None,
        away_series: pd.Series | None = None,
        people_home_series: pd.Series | None = None,
        climate_recent: dict[str, pd.DataFrame] | None = None,
        dhw_recent: pd.DataFrame | None = None,
        heating_active_series: pd.Series | None = None,
        setpoint_on: float | None = None,
        setpoint_off: float | None = None,
        room_areas: dict[str, float] | None = None,
        physics_model: ThermalPhysicsModel | None = None,
        heating_buffer_temp_recent: pd.DataFrame | None = None,  # cols: timestamp, heating_buffer_temp
        dhw_schedule_override: dict | None = None,
    ) -> pd.DataFrame:
```

Then add `heating_buffer_temp_recent=heating_buffer_temp_recent,` to the `natural_baseline_df = self.predict(...)` call (currently lines 1429-1450 — add the new line right after `physics_model=physics_model,` at line 1442, before the existing `dhw_schedule_override` comment block):

```python
        natural_baseline_df = self.predict(
            forecast_df,
            live_temp,
            recent_actuals,
            sub_sensors_recent=sub_sensors_recent,
            away_series=away_series,
            people_home_series=people_home_series,
            climate_recent=climate_recent,
            dhw_recent=dhw_recent,
            heating_active_series=heating_active_series,
            setpoint_on=setpoint_on,
            setpoint_off=setpoint_off,
            room_areas=room_areas,
            physics_model=physics_model,
            heating_buffer_temp_recent=heating_buffer_temp_recent,
            # predict_series's silent committed_override fallback was removed (Phase A Task 6) —
            # this call site must ask for the live committed override explicitly so the "natural
            # baseline" still reflects whatever's actually committed right now (e.g. an already-
            # armed legionella boost) when scoring a *different* candidate dhw_schedule_override.
            dhw_schedule_override=(
                physics_model.schedule.get("committed_override") if physics_model is not None else None
            ),
        )
```

And add the same line to the `scenario_baseline_df = self.predict(...)` call (currently lines 1453-1468 — add after `physics_model=physics_model,` at line 1466):

```python
        if dhw_schedule_override is not None and physics_model is not None:
            scenario_baseline_df = self.predict(
                forecast_df,
                live_temp,
                recent_actuals,
                sub_sensors_recent=sub_sensors_recent,
                away_series=away_series,
                people_home_series=people_home_series,
                climate_recent=climate_recent,
                dhw_recent=dhw_recent,
                heating_active_series=heating_active_series,
                setpoint_on=setpoint_on,
                setpoint_off=setpoint_off,
                room_areas=room_areas,
                physics_model=physics_model,
                heating_buffer_temp_recent=heating_buffer_temp_recent,
                dhw_schedule_override=dhw_schedule_override,
            )
        else:
            scenario_baseline_df = natural_baseline_df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_model.py::TestScenario::test_predict_scenario_forwards_heating_buffer_temp_recent -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: all pass (no regressions — this only adds a parameter with a `None` default, so every existing call site without it is unaffected).

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "fix: forward heating_buffer_temp_recent through predict_scenario()"
```

---

### Task 2: Wire `_get_scenario_cb` to pass the live buffer-temp cache

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:1051-1120` (`_get_scenario_cb`, specifically the `predict_scenario(...)` call ~line 1098)
- Test: `tests/test_scenario_service.py` (new test alongside `test_room_areas_forwarded_to_predict_scenario`, ~line 141)

**Interfaces:**
- Consumes: `predict_scenario(..., heating_buffer_temp_recent=...)` from Task 1; `self._physics_heating_buffer_df` (existing instance attribute, set in `initialize()` and refreshed hourly by `_update_sensors()` per `memory/project_physics_sensor_hourly_refresh_fix.md` — already exists, no change needed here).
- Produces: nothing consumed by further tasks — this is the last task in this plan.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_scenario_service.py`, directly after `test_room_areas_forwarded_to_predict_scenario` (~line 162):

```python
    def test_heating_buffer_temp_forwarded_to_predict_scenario(self):
        """_get_scenario_cb must pass self._physics_heating_buffer_df to predict_scenario
        as heating_buffer_temp_recent — regression test for the 2026-08-19 bug where every
        get_scenario call silently predicted with a phantom heating_buffer_temp=0.0."""
        from energy_forecast.energy_forecast import EnergyForecast

        cached_df = _make_baseline_df()
        app = _make_app(cached_df=cached_df)
        buffer_df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-01", periods=3, freq="1h"), "heating_buffer_temp": [45.0, 46.0, 47.0]}
        )
        app._physics_heating_buffer_df = buffer_df

        scenario_result = cached_df.copy()
        scenario_result["delta_kwh"] = 0.0
        app._ml_model.predict_scenario.return_value = scenario_result

        EnergyForecast._get_scenario_cb(
            app,
            "homeassistant",
            "energy_forecast",
            "get_scenario",
            {"schedule": {}, "publish": False},
        )

        call_kwargs = app._ml_model.predict_scenario.call_args[1]
        assert call_kwargs.get("heating_buffer_temp_recent") is buffer_df
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_scenario_service.py::TestGetScenarioCb::test_heating_buffer_temp_forwarded_to_predict_scenario -v`
Expected: FAIL with `AssertionError` (`call_kwargs.get("heating_buffer_temp_recent")` is `None`, not `buffer_df` — `app._ml_model` is a `MagicMock`, so the call doesn't raise `TypeError`; it just silently accepts whatever kwargs `_get_scenario_cb` currently passes, which don't include this one).

- [ ] **Step 3: Wire the call site**

In `apps/energy_forecast/energy_forecast.py`, inside `_get_scenario_cb`, add `heating_buffer_temp_recent=self._physics_heating_buffer_df,` to the `predict_scenario(...)` call (currently ~lines 1098-1111):

```python
            result_df = self._ml_model.predict_scenario(
                self._cached_forecast_df,
                self._cached_live_temp,
                schedule,
                recent_actuals=self._cached_recent_actuals,
                sub_sensors_recent=self._cached_sub_sensors,
                away_series=self._cached_away_series,
                people_home_series=self._cached_people_home,
                climate_recent=self._cached_climate_recent,
                dhw_recent=self._cached_dhw_recent,
                room_areas=self._climate_room_areas or None,
                physics_model=self._physics_model,
                heating_buffer_temp_recent=self._physics_heating_buffer_df,
                dhw_schedule_override=dhw_schedule,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_scenario_service.py::TestGetScenarioCb::test_heating_buffer_temp_forwarded_to_predict_scenario -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_scenario_service.py
git commit -m "fix: wire _get_scenario_cb to pass live heating_buffer_temp_recent"
```

---

## Self-Review

**Spec coverage:** Fix proposal 1 from `memory/project_legionella_forecast_warnings_2026_08_19.md` has two parts — (a) add the parameter to `predict_scenario()` and forward to both internal `predict()` calls (Task 1), (b) update `_get_scenario_cb` to pass `self._physics_heating_buffer_df` (Task 2) — both covered. "Add a regression test asserting the forward happens" is covered at both layers (model-level in Task 1, callback-level in Task 2), matching the existing two-layer test pattern already used for `room_areas` in this codebase.

**Placeholder scan:** No TBD/"handle edge cases"/deferred-detail steps — every step has literal code.

**Type consistency:** `heating_buffer_temp_recent: pd.DataFrame | None = None` matches the existing type in `predict()`'s signature exactly (same name, same type, same "cols: timestamp, heating_buffer_temp" comment convention).

## Not in scope (confirm before a follow-up plan)

- Fix proposal 2 (`lag_24h` NaN-warning threshold) — separate, lower-priority, likely-benign finding per the same memory file.
- `heating_active_series`/`setpoint_on`/`setpoint_off` are **not** added to `_get_scenario_cb`'s call — those come from per-cycle local variables in `_update_sensors()`, not a persisted instance attribute like `_physics_heating_buffer_df`, so there's nothing for `_get_scenario_cb` to forward without new caching work. Confirmed this is consistent with the original root-cause diagnosis, which only flagged `heating_buffer_temp_recent`.
- CHANGELOG.md / MEMORY.md updates and the merge-to-dev/deploy sequence are handled by `@changelog-writer` / `@deploy-agent` per the standard finishing-a-development-branch workflow, not as tasks in this plan.
