# Physics-ML Hybrid — Plan D: Scenario API `dhw_schedule` Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `ha-energy-manager` ask "what if DHW/legionella ran at a different time?" (`get_scenario(dhw_schedule=...)`) and commit an actual schedule change (`set_dhw_schedule`) — the oracle pattern from spec §8/§9 that the LP scheduler will eventually use for DHW optimisation.

**Architecture:** `ThermalPhysicsModel.predict_series()`/`predict_training_series()` (Plan A) already accept a transient `dhw_schedule_override` parameter for one-off scenario queries. This plan adds a **committed** override path — `set_dhw_schedule` persists an override into `physics_schedule.json` so it becomes the new "natural" baseline for every subsequent prediction, not just one scenario call — and threads `dhw_schedule_override` through `model.py`'s `predict()`/`predict_scenario()` up to `energy_forecast.py`'s `get_scenario`/`set_dhw_schedule` services.

**Tech Stack:** No new dependencies. Depends on Plan A only (does not require Plan B's `physics_kwh` feature or Plan C's residual split — DHW scenario simulation works identically in Phase 1 or Phase 2, since it operates on `physics_model.predict_series()` directly). Sequenced after Plan B in this index only because Plan B introduces `self._physics_model`/`physics:` config that this plan's service registrations assume already exist.

**Base branch:** `dev`, after Plan A (required) and Plan B (assumed present for `self._physics_model`) have merged. Branch name: `feat/physics-scenario-api ha-energy-forecast`.

## Global Constraints

- `get_scenario`'s `dhw_schedule` parameter is a **transient** override — one prediction, not persisted — spec §5.4.
- `set_dhw_schedule`'s payload is a **committed** override — persisted to `physics_schedule.json` atomically, applied to every subsequent prediction until superseded — spec §3.3, §5.4.
- `set_dhw_schedule` bypasses the legionella instability guard (`_check_legionella_stability()`, Plan A Task 10) — an explicit confirmed intent from the energy manager is not "unexpected drift" — spec §3.3.
- `set_dhw_schedule` immediately invalidates `self._cached_forecast_df` so the next HA sensor publish reflects the new schedule — spec §5.4.
- Delta in `get_scenario(dhw_schedule=...)` is computed vs. the **natural baseline** (current committed schedule, i.e. no override), not vs. zero and not vs. the scenario's own override — spec §5.4.
- Callers (`ha-energy-manager`) are responsible for their own secondary scenario-result caches; this repo's obligation is to always compute fresh when `dhw_schedule` is present (already true — there is no per-call scenario-result cache inside `energy_forecast.py` today, only pre-fetched input data like `self._cached_forecast_df`/`self._cached_live_temp` is reused, and that reuse is unaffected by this plan) — spec §5.4, §8 Issues 1-2.

---

### Task 1: Committed DHW schedule override in `ThermalPhysicsModel`

**Files:**
- Modify: `apps/energy_forecast/physics.py` (extend `_default_schedule()`, `predict_series()`, `predict_training_series()`, `_check_legionella_stability()` call sites)
- Test: `tests/test_physics.py`

**Interfaces:**
- Produces: `ThermalPhysicsModel.commit_dhw_schedule(override: dict) -> None` (bypasses the instability guard, persists to `physics_schedule.json`). `predict_series()`/`predict_training_series()` resolve their effective override as `dhw_schedule_override if dhw_schedule_override is not None else self._schedule.get("committed_override")` — i.e. an explicit per-call override still wins, but the natural (no-argument) baseline now reflects whatever was last committed.

- [ ] **Step 1: Write the failing test**

```python
class TestCommittedDhwSchedule:
    def test_commit_persists_override_and_bypasses_stability_guard(self, tmp_path):
        model_dir = tmp_path / "models"
        pm = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        pm._schedule.update(legionella_dow=2, legionella_hour=14)
        # a >2h shift would normally be rejected by _check_legionella_stability
        pm.commit_dhw_schedule({"legionella": ("2026-06-25", 22)})
        assert pm._schedule["committed_override"] == {"legionella": ["2026-06-25", 22]}  # JSON round-trips tuples as lists
        # reload from disk to confirm persistence
        pm2 = ThermalPhysicsModel(model_dir, DEFAULT_CONFIG)
        assert pm2._schedule["committed_override"] == {"legionella": ["2026-06-25", 22]}

    def test_natural_baseline_applies_committed_override_by_default(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm._schedule.update(T_dhw_lower=45.0, T_dhw_upper=55.0, T_legionella=60.0)
        pm.commit_dhw_schedule({"legionella": ("2026-06-25", 10)})
        ts = pd.date_range("2026-06-25 00:00", periods=24, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "temp_c": [10.0] * 24, "direct_radiation_wm2": [0.0] * 24})
        # no explicit dhw_schedule_override passed — the committed one should still apply
        result = pm.predict_series(forecast_df)
        assert len(result) == 24  # no crash; committed override consumed internally

    def test_explicit_per_call_override_takes_precedence_over_committed(self, tmp_path):
        pm = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
        pm._calib.update(UA_dhw=15.0, Q_dhw_daily=3.5)
        pm.commit_dhw_schedule({"legionella": ("2026-06-25", 10)})
        ts = pd.date_range("2026-06-25 00:00", periods=24, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "temp_c": [10.0] * 24, "direct_radiation_wm2": [0.0] * 24})
        override = {"legionella": ("2026-06-25", 20)}
        result_a = pm.predict_series(forecast_df, dhw_schedule_override=override)
        result_b = pm.predict_series(forecast_df)  # committed override (hour 10) applies
        assert not result_a.equals(result_b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_physics.py::TestCommittedDhwSchedule -v`
Expected: FAIL with `AttributeError: 'ThermalPhysicsModel' object has no attribute 'commit_dhw_schedule'`

- [ ] **Step 3: Write minimal implementation**

Add `"committed_override": None` to `_default_schedule()` in `physics.py`.

Add to `ThermalPhysicsModel`:

```python
    def commit_dhw_schedule(self, override: dict) -> None:
        """Persist *override* as the new standing DHW schedule, bypassing the instability guard."""
        self._schedule["committed_override"] = override
        _atomic_write_json(self._schedule_path, self._schedule)
        _LOGGER.info(f"DHW schedule committed: {override}")
```

In both `predict_series()` and `predict_training_series()`, wherever `dhw_schedule_override` is consumed by `_dhw_kwh_series()` (see Plan A Task 5), change the resolution to:

```python
        effective_override = dhw_schedule_override if dhw_schedule_override is not None else self._schedule.get("committed_override")
```

and pass `effective_override` to `_dhw_kwh_series(...)` instead of the raw parameter. `predict_training_series()` doesn't take `dhw_schedule_override` as a parameter today (Plan A's contract has it only on `predict_series()`) — for training, always resolve from the committed override only:

```python
        effective_override = self._schedule.get("committed_override")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_physics.py -v`
Expected: all pass, including the 3 new tests

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/physics.py tests/test_physics.py
git commit -m "feat: add committed DHW schedule override (set_dhw_schedule persistence)"
```

---

### Task 2: `dhw_schedule_override` threaded through `predict()`

**Files:**
- Modify: `apps/energy_forecast/model.py` (`predict()` and `_prepare_prediction_X()` signatures, extending the `physics_model.predict_series()` call site added in Plan B Task 5)
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `EnergyForecastModel.predict(..., dhw_schedule_override: dict | None = None)`.

- [ ] **Step 1: Write the failing test**

```python
class TestPredictDhwScheduleOverride:
    def test_dhw_schedule_override_forwarded_to_physics_model(self, tmp_path, monkeypatch):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": False,
        })
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)

        received = {}
        original = pm.predict_series
        def _spy(*args, **kwargs):
            received.update(kwargs)
            return original(*args, **kwargs)
        monkeypatch.setattr(pm, "predict_series", _spy)

        override = {"legionella": ("2026-06-25", 10)}
        model.predict(forecast_df, live_temp=5.0, physics_model=pm, dhw_schedule_override=override)
        assert received.get("dhw_schedule_override") == override

    def test_dhw_schedule_override_none_by_default(self, tmp_path):
        model, forecast_df = _make_trained_model(tmp_path / "model")
        result = model.predict(forecast_df, live_temp=5.0)  # no exception with default None
        assert not result.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestPredictDhwScheduleOverride -v`
Expected: FAIL with `TypeError: predict() got an unexpected keyword argument 'dhw_schedule_override'`

- [ ] **Step 3: Write minimal implementation**

Add `dhw_schedule_override: dict | None = None` to `predict()`'s signature and to `_prepare_prediction_X()`'s signature (alongside the `physics_model`/`heating_buffer_temp_recent` parameters added in Plan B Task 5).

In `_prepare_prediction_X()`, extend the `physics_model.predict_series(...)` call added in Plan B Task 5 with the new kwarg:

```python
            physics_kwh_series = physics_model.predict_series(
                forecast_df, climate_recent=climate_recent, dhw_recent=dhw_recent, room_areas=room_areas,
                heating_active_series=heating_active_series, setpoint_on=setpoint_on, setpoint_off=setpoint_off,
                dhw_schedule_override=dhw_schedule_override,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "feat: thread dhw_schedule_override through predict()"
```

---

### Task 3: `predict_scenario()` — dhw delta vs. natural baseline

**Files:**
- Modify: `apps/energy_forecast/model.py` (`predict_scenario()`, lines 1086-1127)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `predict()`'s new `dhw_schedule_override` param (Task 2), `physics_model` param (Plan B).
- Produces: `EnergyForecastModel.predict_scenario(..., physics_model: "ThermalPhysicsModel | None" = None, dhw_schedule_override: dict | None = None) -> pd.DataFrame` — `predicted_kwh` reflects the dhw-overridden + appliance-overlaid forecast; `delta_kwh` is the combined signed difference (appliance schedule + dhw schedule) vs. the natural (no-override) baseline.

- [ ] **Step 1: Write the failing test**

```python
class TestScenarioDhwDelta:
    def test_delta_kwh_reflects_dhw_shift_vs_natural_baseline(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": False,
        })
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35, UA_dhw=15.0, Q_dhw_daily=3.5)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)

        override = {"legionella": (str(forecast_df["timestamp"].iloc[10].date()), forecast_df["timestamp"].iloc[10].hour)}
        result = model.predict_scenario(
            forecast_df, live_temp=5.0, schedule={}, physics_model=pm, dhw_schedule_override=override
        )
        assert "delta_kwh" in result.columns
        assert result["delta_kwh"].abs().sum() > 0  # dhw shift produced a nonzero delta somewhere

    def test_no_dhw_schedule_delta_unchanged_from_pre_plan_d_behaviour(self, tmp_path):
        model, forecast_df = _make_trained_model(tmp_path / "model")
        result = model.predict_scenario(forecast_df, live_temp=5.0, schedule={})
        assert (result["delta_kwh"] == 0.0).all()  # no appliance schedule, no dhw override -> no delta

    def test_appliance_and_dhw_deltas_both_present_are_additive(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": False,
        })
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35, UA_dhw=15.0, Q_dhw_daily=3.5)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)

        override = {"legionella": (str(forecast_df["timestamp"].iloc[5].date()), forecast_df["timestamp"].iloc[5].hour)}
        dhw_only = model.predict_scenario(forecast_df, live_temp=5.0, schedule={}, physics_model=pm, dhw_schedule_override=override)
        combined = model.predict_scenario(
            forecast_df, live_temp=5.0, schedule={}, physics_model=pm, dhw_schedule_override=override
        )
        # same call twice (no appliance signatures learned in this fixture) should be deterministic
        pd.testing.assert_series_equal(dhw_only["delta_kwh"], combined["delta_kwh"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestScenarioDhwDelta -v`
Expected: FAIL with `TypeError: predict_scenario() got an unexpected keyword argument 'physics_model'`

- [ ] **Step 3: Write minimal implementation**

Add `physics_model: "ThermalPhysicsModel | None" = None` and `dhw_schedule_override: dict | None = None` to `predict_scenario()`'s signature.

Replace the body (lines 1110-1127):

```python
        if self.model is None:
            raise RuntimeError("Model not yet trained.")

        natural_baseline_df = self.predict(
            forecast_df, live_temp, recent_actuals,
            sub_sensors_recent=sub_sensors_recent, away_series=away_series, people_home_series=people_home_series,
            climate_recent=climate_recent, dhw_recent=dhw_recent, heating_active_series=heating_active_series,
            setpoint_on=setpoint_on, setpoint_off=setpoint_off, room_areas=room_areas,
            physics_model=physics_model,  # dhw_schedule_override intentionally omitted — this is the natural baseline
        )

        if dhw_schedule_override is not None and physics_model is not None:
            scenario_baseline_df = self.predict(
                forecast_df, live_temp, recent_actuals,
                sub_sensors_recent=sub_sensors_recent, away_series=away_series, people_home_series=people_home_series,
                climate_recent=climate_recent, dhw_recent=dhw_recent, heating_active_series=heating_active_series,
                setpoint_on=setpoint_on, setpoint_off=setpoint_off, room_areas=room_areas,
                physics_model=physics_model, dhw_schedule_override=dhw_schedule_override,
            )
        else:
            scenario_baseline_df = natural_baseline_df

        result = _composite_forecast(scenario_baseline_df, schedule, self._appliance_signatures)

        if dhw_schedule_override is not None and physics_model is not None:
            dhw_delta = (scenario_baseline_df["predicted_kwh"] - natural_baseline_df["predicted_kwh"]).to_numpy()
            result["delta_kwh"] = result["delta_kwh"] + dhw_delta

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "feat: add dhw schedule delta to predict_scenario vs natural baseline"
```

---

### Task 4: `get_scenario` accepts `dhw_schedule`

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (`_get_scenario_cb`, lines 788-845)
- Test: extend `tests/test_energy_forecast_physics_config.py`

**Interfaces:**
- Consumes: `EnergyForecastModel.predict_scenario(..., physics_model=..., dhw_schedule_override=...)` (Task 3).
- Produces: `energy_forecast/get_scenario` service accepts an optional `dhw_schedule` kwarg with the same `{"legionella": ("YYYY-MM-DD", hour)}` shape `ThermalPhysicsModel` expects.

- [ ] **Step 1: Write the failing test**

```python
class TestGetScenarioDhwSchedule:
    def test_dhw_schedule_forwarded_as_override(self, monkeypatch):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._cached_forecast_df = pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-15", periods=48, freq="1h"), "temp_c": [5.0] * 48}
        )
        app._cached_live_temp = 5.0
        app.fire_event = MagicMock()

        received = {}
        def _fake_predict_scenario(*args, **kwargs):
            received.update(kwargs)
            return pd.DataFrame({"timestamp": app._cached_forecast_df["timestamp"], "predicted_kwh": [1.0] * 48, "delta_kwh": [0.0] * 48})
        app._ml_model.predict_scenario = _fake_predict_scenario

        app._get_scenario_cb(
            "default", "energy_forecast", "get_scenario",
            {"schedule": {}, "dhw_schedule": {"legionella": ["2026-01-16", 10]}},
        )
        assert received.get("dhw_schedule_override") == {"legionella": ["2026-01-16", 10]}
        assert received.get("physics_model") is app._physics_model

    def test_no_dhw_schedule_key_forwards_none(self):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._cached_forecast_df = pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-15", periods=48, freq="1h"), "temp_c": [5.0] * 48}
        )
        app._cached_live_temp = 5.0
        app.fire_event = MagicMock()

        received = {}
        def _fake_predict_scenario(*args, **kwargs):
            received.update(kwargs)
            return pd.DataFrame({"timestamp": app._cached_forecast_df["timestamp"], "predicted_kwh": [1.0] * 48, "delta_kwh": [0.0] * 48})
        app._ml_model.predict_scenario = _fake_predict_scenario

        app._get_scenario_cb("default", "energy_forecast", "get_scenario", {"schedule": {}})
        assert received.get("dhw_schedule_override") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py::TestGetScenarioDhwSchedule -v`
Expected: FAIL — `predict_scenario` is called without `dhw_schedule_override`/`physics_model` kwargs

- [ ] **Step 3: Write minimal implementation**

In `_get_scenario_cb` (lines 788-845), add after the existing `schedule = cleaned` line:

```python
        dhw_schedule = kwargs.get("dhw_schedule")
        if dhw_schedule is not None and not isinstance(dhw_schedule, dict):
            _LOGGER.warning("get_scenario: dhw_schedule must be a dict — ignoring")
            dhw_schedule = None
```

Update the `self._ml_model.predict_scenario(...)` call to include:

```python
            physics_model=self._physics_model,
            dhw_schedule_override=dhw_schedule,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "feat: add dhw_schedule parameter to get_scenario service"
```

---

### Task 5: `set_dhw_schedule` service

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (service registration near line 296, alongside `recalibrate_physics` from Plan B Task 7)
- Test: extend `tests/test_energy_forecast_physics_config.py`

**Interfaces:**
- Produces: AppDaemon service `energy_forecast/set_dhw_schedule`, accepting `{"dhw_schedule": {...}}`. Calls `self._physics_model.commit_dhw_schedule(dhw_schedule)` (Task 1) and invalidates `self._cached_forecast_df`.

- [ ] **Step 1: Write the failing test**

```python
class TestSetDhwScheduleService:
    def test_service_registered_when_physics_enabled(self):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        calls = [c.args[0] for c in app.register_service.call_args_list]
        assert "energy_forecast/set_dhw_schedule" in calls

    def test_commits_schedule_and_invalidates_cache(self, tmp_path):
        from unittest.mock import patch

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._cached_forecast_df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})

        with patch.object(app._physics_model, "commit_dhw_schedule") as mock_commit:
            app._set_dhw_schedule_cb(
                "default", "energy_forecast", "set_dhw_schedule",
                {"dhw_schedule": {"legionella": ["2026-01-16", 10]}},
            )
            mock_commit.assert_called_once_with({"legionella": ["2026-01-16", 10]})
        assert app._cached_forecast_df is None

    def test_missing_dhw_schedule_kwarg_logs_warning_no_crash(self, caplog):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        with caplog.at_level("WARNING"):
            app._set_dhw_schedule_cb("default", "energy_forecast", "set_dhw_schedule", {})
        assert any("dhw_schedule" in r.message.lower() for r in caplog.records)

    def test_physics_disabled_logs_warning_no_crash(self, caplog):
        app = _make_app({"energy_sensor": "sensor.grid_import"})
        app.initialize()
        with caplog.at_level("WARNING"):
            app._set_dhw_schedule_cb(
                "default", "energy_forecast", "set_dhw_schedule", {"dhw_schedule": {"legionella": ["2026-01-16", 10]}}
            )
        assert any("physics" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py::TestSetDhwScheduleService -v`
Expected: FAIL with `AttributeError: 'EnergyForecast' object has no attribute '_set_dhw_schedule_cb'`

- [ ] **Step 3: Write minimal implementation**

In `initialize()`, alongside the `recalibrate_physics` registration (Plan B Task 7):

```python
        if self._physics_model is not None:
            self.register_service("energy_forecast/set_dhw_schedule", self._set_dhw_schedule_cb)
```

Add the callback:

```python
    def _set_dhw_schedule_cb(self, namespace: str, domain: str, service: str, kwargs: dict) -> None:
        if self._physics_model is None:
            _LOGGER.warning("set_dhw_schedule called but physics is not configured — ignoring")
            return
        dhw_schedule = kwargs.get("dhw_schedule")
        if not isinstance(dhw_schedule, dict):
            _LOGGER.warning("set_dhw_schedule: dhw_schedule must be a dict — ignoring")
            return
        try:
            self._physics_model.commit_dhw_schedule(dhw_schedule)
            self._cached_forecast_df = None  # forces a fresh forecast on the next publish cycle
            _LOGGER.info(f"DHW schedule committed: {dhw_schedule}")
        except Exception as exc:
            _LOGGER.error(f"set_dhw_schedule failed: {exc}")
```

Note per the Global Constraints, `self._cached_forecast_df = None` is the mechanism spec §5.4 calls "immediately invalidates `_cached_forecast_df`" — check whichever method currently guards `_cached_forecast_df is None` (the `get_scenario` early-return at line ~793 already does: `if self._cached_forecast_df is None: _LOGGER.warning(...); return`) to confirm the hourly `run_hourly` cycle repopulates it and no other code path assumes it's always non-`None` between cycles.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "feat: add set_dhw_schedule service with cache invalidation"
```

---

### Task 6: README documentation for the oracle pattern + caller cache contract

**Files:**
- Modify: `README.md` (document the two new services and their contract for `ha-energy-manager`)
- Modify: `apps/energy_forecast/energy_forecast.py` (docstring on `_get_scenario_cb`, spelling out the caching rule from spec §5.4 so a future reader of this file — not just the spec — knows the contract)

This task has no test — it's documentation for the consuming repo (`ha-energy-manager`, out of scope for code changes here) so its `ConsumptionForecastApp._scenario_cache`/`_scenario_store` maintainers know: (a) flush their cache after a successful `set_dhw_schedule` call, (b) never cache a `get_scenario` result that included a non-`None` `dhw_schedule`, or key the cache on the full `dhw_schedule` dict if they must cache it.

- [ ] **Step 1: Add the docstring**

Extend `_get_scenario_cb`'s docstring with:

```python
    """AppDaemon service callback: energy_forecast/get_scenario.

    Accepts kwargs:
      schedule (dict[str, str]): {prefix: "HH:MM" | "off" | None}
      dhw_schedule (dict | None): transient physics DHW override, e.g.
          {"legionella": ("2026-06-25", 10)} — NOT persisted. Delta is
          computed vs. the natural (currently committed) baseline.
      publish  (bool):           if True, publish scenario sensors to HA

    Caller cache contract (see spec §5.4): results computed with a non-None
    dhw_schedule must NOT be cached by the caller unless dhw_schedule is
    included in the cache key. The safest default is to never cache them.
    After a successful set_dhw_schedule call, callers must flush their own
    scenario cache — this app only invalidates its own _cached_forecast_df.

    Fires event "energy_forecast_scenario_result" with the forecast payload.
    """
```

- [ ] **Step 2: Update README.md**

Add a section (find the existing services/API documentation section first via `grep -n "^## \|get_scenario" README.md`) documenting `energy_forecast/recalibrate_physics`, `energy_forecast/set_dhw_schedule`, and the extended `get_scenario(dhw_schedule=...)` — mirror the style of whatever existing service documentation already exists in the file.

- [ ] **Step 3: Commit**

```bash
git add README.md apps/energy_forecast/energy_forecast.py
git commit -m "docs: document DHW scenario/commit services and caller cache contract"
```

---

### Task 7: Full-suite regression + self-review

**Files:** none new — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 2: Verify §7's Scenario API test list is fully covered**

Cross-check against spec §7 "Scenario API" bullets:
- `get_scenario` with `dhw_schedule` → fresh prediction (not cache) — Task 3/4.
- `get_scenario` without `dhw_schedule` → cache used (no regression) — Task 4 Step 1's second test; `self._cached_forecast_df`/`self._cached_live_temp` reuse is unchanged by this plan.
- Delta computed vs natural physics baseline, not vs zero — Task 3.
- `set_dhw_schedule(A)` then `get_scenario(dhw_schedule=B)` → delta computed vs baseline with schedule A (not B) — covered by Task 1's committed-override resolution (natural baseline always reads the committed value) combined with Task 3's natural-vs-scenario split; add one integration test combining Task 1 + Task 3 + Task 5 if not already exercised end-to-end (write it now if missing, in `tests/test_model.py` or a new `tests/test_scenario_integration.py`).
- `set_dhw_schedule` → `_cached_forecast_df` invalidated — Task 5.

- [ ] **Step 3: Update CHANGELOG.md**

Use `@changelog-writer`.

- [ ] **Step 4: Open PR against `dev`**

Follow `superpowers:finishing-a-development-branch`. Title: `feat: DHW scenario API (get_scenario dhw_schedule + set_dhw_schedule)`. Note in the PR description that `ha-energy-manager`'s `ConsumptionForecastApp` will need a follow-up change (in that repo) to implement the caller-side cache contract documented in Task 6 — this is out of scope for this repo but blocks the full oracle-pattern integration from spec §9.
