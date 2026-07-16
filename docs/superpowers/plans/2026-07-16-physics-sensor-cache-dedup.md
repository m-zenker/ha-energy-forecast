# Physics Sensor History Duplication (#89) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Base branch:** `dev` — create via `/feat fix/physics-sensor-history-dedup ha-energy-forecast`.

**Goal:** Eliminate the physics-ML hybrid's redundant HA history fetches and on-disk CSV caches for DHW tank temp and room thermostat temp — data the ML pipeline already fetches for the same entities — without changing any forecast output.

**Architecture:** `_fetch_physics_sensor_histories()` gains two optional parameters, `climate_dfs` and `dhw_df`, holding the data the caller (`_retrain()` / `_update_sensors()`) already fetched this cycle for the ML pipeline. When the physics-configured entity is the same HA entity the ML side already pulled, the physics function reuses that data instead of independently re-fetching + re-caching it. The room-thermostat `temp_sensor` fetch (confirmed dead — its output is never read by anything) is deleted outright rather than rewired.

**Tech Stack:** Python 3.13, AppDaemon 4.x, pandas, pytest. No new dependencies.

## Global Constraints

- Never touch `hef`'s dedicated mamba env at `/home/jovyan/my_envs/ha-energy-forecast` — always run tests via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest`.
- No behavior change to forecast output: this is a pure I/O-path dedup. Every existing test not explicitly listed below as "update" must still pass unmodified.
- Reuse must be conditional on entity-equality, not assumed — other installs may configure `physics.dhw_tank_temp_sensor` / `room_thermostats[].climate_entity` to different entities than the ML pipeline's `dhw_buffer_sensor` / `climate_entities`. Never remove the independent-fetch fallback path.
- `rt["temp_sensor"]` config key itself is NOT removed — it is still consumed by `_warn_on_sensor_mismatch()` (`energy_forecast.py:518`) for config validation, which is out of scope for this fix.

---

## Confirmed Scope (investigation already done — do not re-derive)

Verified directly against the live HA instance (`homeassistant` Samba share, `addon_configs/a0d7b954_appdaemon/apps/energy_forecast/`) on 2026-07-16:

**1. DHW tank temp — genuine duplicate fetch (both sides consumed):**
- `physics.dhw_tank_temp_sensor` and top-level `dhw_buffer_sensor` are both configured to `sensor.em_kermi_bridge_kermi_hot_water_temperature` (confirmed in fresh `apps.yaml`, lines 104 and 125).
- ML pipeline fetches it into `dhw_em_kermi_bridge_kermi_hot_water_temperature.csv` (66,742 bytes — long history) via `_retrain()` (`energy_forecast.py:1617-1627`, local var `dhw_df`) and `_update_sensors()` (`energy_forecast.py:1789-1799`, local var `dhw_recent`), both with `column_name="buffer_temp"`.
- Physics side independently re-fetches the *same entity* into `physics_dhw_tank_em_kermi_bridge_kermi_hot_water_temperature.csv` (only 10,077 bytes — started later, since physics config was added 2026-07-14) via `_fetch_physics_sensor_histories()` (`energy_forecast.py:1293-1300`), also `column_name="buffer_temp"` — identical schema, same entity, same resample logic (`fetch_generic_sensor_history`/`fetch_recent_generic_sensor`).
- Both `self._physics_dhw_tank_df` (written here) and `dhw_df`/`self._cached_dhw_recent` (the ML side) ARE separately consumed downstream (`physics_model.calibrate()`/`predict_series()` vs. `model.train()`/`predict()`) — this is a real redundant-fetch bug, not dead code.

**2. Room thermostat temp — confirmed fully dead code (stronger finding than the original ticket assumed):**
- All 10 `physics.room_thermostats[].climate_entity` entries match 1:1 with the top-level `climate_entities` list (confirmed in fresh `apps.yaml`, lines 74-84 vs. 129-157) — comment in the file literally says "matched 1:1 to climate_entities".
- `physics.room_thermostats[].temp_sensor` (10 distinct Netatmo entities) is independently fetched into 10 CSVs (`physics_temp_0_*.csv` .. `physics_temp_9_*.csv`, ~9.5-9.7 KB each) and stored in `self._room_thermostat_temp_dfs[climate_entity]` (`energy_forecast.py:1274`, `1325-1337`).
- **Repo-wide grep confirms `self._room_thermostat_temp_dfs` is never read anywhere except tests asserting its presence/emptiness** (`tests/test_energy_forecast_physics_config.py:241,248,385,630`). `physics.py`'s `calibrate()`/`predict_series()`/`_area_weighted_t_indoor()` only ever consume `climate_dfs`/`climate_recent` (i.e. `self._physics_climate_dfs`), which already carries a `current_temp` column sourced from the *same* underlying Netatmo-bridge-republished `climate.*` entity (`netatmo_bridge.py:275-287` in `ha-energy-manager` republishes `climate.*.current_temperature` verbatim as the standalone `temp_sensor`).
- Conclusion: the entire `temp_sensor` fetch loop + `_room_thermostat_temp_dfs` attribute can be **deleted outright** — there is nothing downstream to migrate it to.

**3. Bonus: `climate_entity` is *also* independently re-fetched by the physics function** (`energy_forecast.py:1339-1346`, `self._climate_cache_path(climate_entity)` — the *same* cache path/file the main `climate_dfs`/`climate_recent` loop already populates at `energy_forecast.py:1608-1616`/`1780-1787`). No divergent cache file here (same file, same schema `timestamp, current_temp, setpoint`), but it is a second full HA history API round-trip per room per cycle for data already fetched moments earlier in the same call stack.

**Net removal once fixed:** 11 orphaned CSV cache files on HA (1 `physics_dhw_tank_*.csv` + 10 `physics_temp_*.csv`, ~107 KB total) stop being written; 11 redundant HA Recorder history API calls removed per retrain/hourly cycle (1 DHW + 10 room-temp), plus 10 redundant climate history API calls (same file, called twice) collapsed to reuse.

---

## Task 1: Delete the dead room-thermostat `temp_sensor` fetch

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:352` (init), `:1274` (reset), `:1325-1337` (fetch loop)
- Modify: `tests/test_energy_forecast_physics_config.py:220-222,238-248,278-292,382-385,614-630`

**Interfaces:**
- Removes: `self._room_thermostat_temp_dfs` attribute entirely.
- No change to `self._physics_climate_dfs`, `self._physics_dhw_tank_df`, `self._physics_heating_buffer_df`, `self._physics_cop_df`.

- [ ] **Step 1: Read current state of the two touched files to confirm line numbers still match**

```bash
grep -n "_room_thermostat_temp_dfs" apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
```

Expected: matches at `energy_forecast.py:352,1274,1333-1334` and `test_energy_forecast_physics_config.py:241,248,385,630`. If line numbers drifted, re-locate by the attribute name (already unique in the codebase) before editing.

- [ ] **Step 2: Remove the attribute from `energy_forecast.py`'s two init sites**

At `energy_forecast.py:352` (inside `__init__`, alongside the other physics attrs), remove the line:
```python
self._room_thermostat_temp_dfs: dict[str, Any] = {}
```

At `energy_forecast.py:1274` (inside `_fetch_physics_sensor_histories`, the `if not recent_only:` reset block), remove the same line — leaving:
```python
        if not recent_only:
            self._physics_dhw_tank_df: Any = None
            self._physics_heating_buffer_df: Any = None
            self._physics_cop_df: Any = None
            self._physics_climate_dfs: dict[str, Any] = {}
```

- [ ] **Step 3: Delete the dead fetch, keep the climate fetch, drop `enumerate`/`temp_path`**

Replace the current room-thermostat loop (`energy_forecast.py:1325-1346`):

```python
        for i, rt in enumerate(self._room_thermostats):
            temp_entity = rt["temp_sensor"]
            climate_entity = rt["climate_entity"]
            temp_path = self._generic_sensor_cache_path(temp_entity, prefix=f"physics_temp_{i}")
            try:
                temp_df = generic_fetch(
                    self, temp_entity, temp_path, column_name="current_temp", timezone=self._timezone
                )
                self._room_thermostat_temp_dfs[climate_entity] = _keep_or_replace(
                    self._room_thermostat_temp_dfs.get(climate_entity), temp_df
                )
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Physics room temp %s %s fetch failed: %s", temp_entity, verb, exc)

            climate_path = self._climate_cache_path(climate_entity)
            try:
                climate_df = climate_fetch(self, climate_entity, climate_path, timezone=self._timezone)
                self._physics_climate_dfs[climate_entity] = _keep_or_replace(
                    self._physics_climate_dfs.get(climate_entity), climate_df
                )
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Physics room climate %s %s fetch failed: %s", climate_entity, verb, exc)
```

with:

```python
        for rt in self._room_thermostats:
            climate_entity = rt["climate_entity"]
            climate_path = self._climate_cache_path(climate_entity)
            try:
                climate_df = climate_fetch(self, climate_entity, climate_path, timezone=self._timezone)
                self._physics_climate_dfs[climate_entity] = _keep_or_replace(
                    self._physics_climate_dfs.get(climate_entity), climate_df
                )
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Physics room climate %s %s fetch failed: %s", climate_entity, verb, exc)
```

Note: `rt["temp_sensor"]` itself is untouched elsewhere (config parsing at `energy_forecast.py:325-333` and the config-validation `temperature_roles` list at `energy_forecast.py:518` both still reference it — those stay as-is, out of scope).

- [ ] **Step 4: Update the docstring's "room-thermostat" wording**

In the `_fetch_physics_sensor_histories` docstring (`energy_forecast.py:1252-1271`), the phrase `dhw_buffer_sensor`/`climate_entities` hourly-refresh pattern already reads correctly — no change needed there. No other docstring line names `_room_thermostat_temp_dfs`.

- [ ] **Step 5: Update `tests/test_energy_forecast_physics_config.py` — remove now-invalid assertions**

In `test_recent_only_uses_recent_variants_not_full_history` (around line 213-223), change:
```python
        assert fetch_recent_generic.call_count == 4  # dhw_tank, heating_buffer, cop, room temp_sensor
```
to:
```python
        assert fetch_recent_generic.call_count == 3  # dhw_tank, heating_buffer, cop
```

In `test_recent_only_populates_attrs_without_any_retrain_having_run` (around line 225-248), delete these two lines:
```python
        assert app._room_thermostat_temp_dfs == {}
```
(the one before the fetch call, ~line 241) and:
```python
        assert not app._room_thermostat_temp_dfs["climate.living_room"].empty
```
(the one after, ~line 248).

In `test_recent_only_quiets_only_cop_sensor_no_data_warning` (around line 278-292), delete:
```python
        assert calls_by_entity["sensor.netatmo_living_room_temp"].kwargs.get("quiet_if_empty") is not True
```
Update the docstring above it (currently says "The other three physics sensors (heating_buffer_temp, dhw_tank, room-thermostat temp) have no such seasonal excuse") to:
```python
        """A heat pump's live COP sensor only reports while actively heating — an empty
        hourly recent-fetch is expected (not an error) for long idle stretches (e.g. all
        summer), so only its recent_only fetch should pass quiet_if_empty=True. The other
        two physics sensors (heating_buffer_temp, dhw_tank) have no such seasonal excuse
        and must keep the default (noisy-on-empty) behaviour."""
```

In `test_retrain_skips_physics_fetch_when_no_physics_model` (around line 366-385), delete:
```python
        assert app._room_thermostat_temp_dfs == {}
```

In `test_physics_history_attrs_defensively_initialized_before_first_retrain` (around line 614-631), delete:
```python
        assert app._room_thermostat_temp_dfs == {}
```

- [ ] **Step 6: Add a regression test proving the dead attribute is gone and no `physics_temp_*` cache file is written**

Add to `TestFetchPhysicsSensorHistories` (or the nearest fitting class) in `tests/test_energy_forecast_physics_config.py`:

```python
    def test_room_thermostat_temp_sensor_is_not_independently_fetched(self, monkeypatch, tmp_path):
        """#89: room_thermostats[].temp_sensor duplicates data already available from
        climate_entity (same Netatmo bridge poll, republished under two entity IDs) and
        was never consumed by physics.py — only climate_dfs/climate_recent is. Confirms
        the dead fetch + its on-disk cache file are both gone."""
        app, fetch_recent_generic, fetch_recent_climate, _, _ = self._configured_app(monkeypatch)
        app._cache_path = tmp_path / "energy_history.csv"  # cache paths derive from this

        app._fetch_physics_sensor_histories(recent_only=True)

        assert not hasattr(app, "_room_thermostat_temp_dfs")
        fetched_entities = {call.args[1] for call in fetch_recent_generic.call_args_list}
        assert "sensor.netatmo_living_room_temp" not in fetched_entities
        assert not (tmp_path / "physics_temp_0_sensor.netatmo_living_room_temp.csv").exists()
```

- [ ] **Step 7: Run the full physics test module**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: PASS, no `AttributeError: _room_thermostat_temp_dfs`.

- [ ] **Step 8: Run the full suite to check for unrelated regressions**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: PASS (same count as before minus removed assertions, plus the one new test).

- [ ] **Step 9: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "fix: remove dead room-thermostat temp_sensor fetch (#89)"
```

---

## Task 2: Reuse already-fetched `climate_dfs` for room thermostats (dedup the climate re-fetch)

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:1251` (signature), `1325-1338` (loop, post-Task-1 shape)
- Modify: `tests/test_energy_forecast_physics_config.py` (new tests only — existing tests call the function without the new kwarg, so they must keep exercising the fallback path unchanged)

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `_fetch_physics_sensor_histories(self, recent_only: bool = False, climate_dfs: dict[str, Any] | None = None, dhw_df: Any = None) -> None` — the final signature Task 3 and Task 4 build on. (Task 2 adds `climate_dfs` only; Task 3 adds `dhw_df` to the same signature.)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_energy_forecast_physics_config.py`:

```python
    def test_climate_dfs_kwarg_reuses_already_fetched_data_for_room_thermostat(self, monkeypatch):
        """#89: when the caller (_retrain()/_update_sensors()) already fetched the room's
        climate_entity this cycle, physics must reuse that DataFrame instead of an
        independent second HA history fetch of the same entity."""
        import pandas as pd

        app, _, fetch_recent_climate, _, _ = self._configured_app(monkeypatch)
        already_fetched = pd.DataFrame(
            {"timestamp": ["2026-07-16 08:00:00"], "current_temp": [19.5], "setpoint": [21.0]}
        )

        app._fetch_physics_sensor_histories(recent_only=True, climate_dfs={"climate.living_room": already_fetched})

        fetch_recent_climate.assert_not_called()
        assert app._physics_climate_dfs["climate.living_room"]["current_temp"].iloc[0] == 19.5

    def test_climate_dfs_kwarg_falls_back_to_independent_fetch_when_entity_not_present(self, monkeypatch):
        """A room_thermostats[].climate_entity that ISN'T among the caller's already-fetched
        climate_dfs keys (e.g. genuinely physics-only room in some other install) must still
        get its own independent fetch — the reuse path is opportunistic, not a hard cutover."""
        app, _, fetch_recent_climate, _, _ = self._configured_app(monkeypatch)

        app._fetch_physics_sensor_histories(recent_only=True, climate_dfs={"climate.some_other_room": None})

        fetch_recent_climate.assert_called_once()

    def test_climate_dfs_kwarg_omitted_keeps_old_independent_fetch_behavior(self, monkeypatch):
        """Callers that don't pass climate_dfs (e.g. _recalibrate_physics_cb's on-demand
        service, or this test) must be unaffected — full backward compatibility."""
        app, _, fetch_recent_climate, _, _ = self._configured_app(monkeypatch)

        app._fetch_physics_sensor_histories(recent_only=True)

        fetch_recent_climate.assert_called_once()
```

- [ ] **Step 2: Run tests to verify the first one fails**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py -k climate_dfs_kwarg -v`
Expected: `test_climate_dfs_kwarg_reuses_already_fetched_data_for_room_thermostat` FAILS with `TypeError: _fetch_physics_sensor_histories() got an unexpected keyword argument 'climate_dfs'`. The other two pass already (they describe current behavior).

- [ ] **Step 3: Add the `climate_dfs` parameter and reuse branch**

In `energy_forecast.py`, change the signature (currently, post-Task-1, still `def _fetch_physics_sensor_histories(self, recent_only: bool = False) -> None:`):

```python
    def _fetch_physics_sensor_histories(
        self, recent_only: bool = False, climate_dfs: dict[str, Any] | None = None, dhw_df: Any = None
    ) -> None:
```

Add one line to the docstring (after the existing `recent_only=True is called every hourly...` paragraph):

```python

        `climate_dfs`/`dhw_df`: the caller's already-fetched data for this cycle
        (`_retrain()`'s `climate_dfs`/`dhw_df` locals, or `_update_sensors()`'s
        `climate_recent`/`dhw_recent`). When the physics-configured entity is the same
        HA entity already fetched there, reuse it instead of a second independent HA
        history fetch + on-disk cache file (#89). Omit (or pass an entity not present)
        to fall back to the original independent-fetch behavior — used by
        `_recalibrate_physics_cb`'s on-demand service call, which has no such
        already-fetched data available.
```

Replace the Task-1-shaped loop:

```python
        for rt in self._room_thermostats:
            climate_entity = rt["climate_entity"]
            climate_path = self._climate_cache_path(climate_entity)
            try:
                climate_df = climate_fetch(self, climate_entity, climate_path, timezone=self._timezone)
                self._physics_climate_dfs[climate_entity] = _keep_or_replace(
                    self._physics_climate_dfs.get(climate_entity), climate_df
                )
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Physics room climate %s %s fetch failed: %s", climate_entity, verb, exc)
```

with:

```python
        for rt in self._room_thermostats:
            climate_entity = rt["climate_entity"]
            if climate_dfs is not None and climate_entity in climate_dfs:
                self._physics_climate_dfs[climate_entity] = _keep_or_replace(
                    self._physics_climate_dfs.get(climate_entity), climate_dfs[climate_entity]
                )
                continue
            climate_path = self._climate_cache_path(climate_entity)
            try:
                climate_df = climate_fetch(self, climate_entity, climate_path, timezone=self._timezone)
                self._physics_climate_dfs[climate_entity] = _keep_or_replace(
                    self._physics_climate_dfs.get(climate_entity), climate_df
                )
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Physics room climate %s %s fetch failed: %s", climate_entity, verb, exc)
```

- [ ] **Step 4: Run tests to verify they all pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "fix: reuse already-fetched climate_dfs for physics room thermostats (#89)"
```

---

## Task 3: Reuse already-fetched `dhw_df` for the physics DHW tank sensor

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:1293-1300` (DHW block, post-Task-2 signature)
- Modify: `tests/test_energy_forecast_physics_config.py` (new tests)

**Interfaces:**
- Consumes: the `climate_dfs`/`dhw_df` signature landed in Task 2.
- Produces: final signature `_fetch_physics_sensor_histories(self, recent_only: bool = False, climate_dfs: dict[str, Any] | None = None, dhw_df: Any = None) -> None` fully wired (both params now do something) — Task 4 calls this from `_retrain()`/`_update_sensors()`.

- [ ] **Step 1: Write the failing tests**

```python
    def test_dhw_df_kwarg_reuses_already_fetched_data_when_entity_matches(self, monkeypatch):
        """#89: when physics.dhw_tank_temp_sensor == the top-level dhw_buffer_sensor (the
        same entity, e.g. sensor.em_kermi_bridge_kermi_hot_water_temperature on the live
        instance), physics must reuse the ML pipeline's already-fetched DHW DataFrame
        instead of an independent second HA history fetch of the same entity."""
        import pandas as pd

        app, fetch_recent_generic, _, _, _ = self._configured_app(monkeypatch)
        app._dhw_buffer_sensor = "sensor.kermi_dhw_buffer_temp"  # matches physics.dhw_tank_temp_sensor in fixture
        already_fetched = pd.DataFrame({"timestamp": ["2026-07-16 08:00:00"], "buffer_temp": [52.3]})

        app._fetch_physics_sensor_histories(recent_only=True, dhw_df=already_fetched)

        dhw_calls = [c for c in fetch_recent_generic.call_args_list if c.args[1] == "sensor.kermi_dhw_buffer_temp"]
        assert dhw_calls == []
        assert app._physics_dhw_tank_df["buffer_temp"].iloc[0] == 52.3

    def test_dhw_df_kwarg_falls_back_when_entity_differs(self, monkeypatch):
        """physics.dhw_tank_temp_sensor pointed at a genuinely different entity than
        dhw_buffer_sensor must keep its own independent fetch — reuse is opportunistic,
        gated on entity equality, never assumed."""
        app, fetch_recent_generic, _, _, _ = self._configured_app(monkeypatch)
        app._dhw_buffer_sensor = "sensor.some_other_dhw_sensor"  # does NOT match fixture's dhw_tank_temp_sensor

        app._fetch_physics_sensor_histories(recent_only=True, dhw_df=None)

        dhw_calls = [c for c in fetch_recent_generic.call_args_list if c.args[1] == "sensor.kermi_dhw_buffer_temp"]
        assert len(dhw_calls) == 1

    def test_dhw_df_kwarg_omitted_keeps_old_independent_fetch_behavior(self, monkeypatch):
        app, fetch_recent_generic, _, _, _ = self._configured_app(monkeypatch)

        app._fetch_physics_sensor_histories(recent_only=True)

        dhw_calls = [c for c in fetch_recent_generic.call_args_list if c.args[1] == "sensor.kermi_dhw_buffer_temp"]
        assert len(dhw_calls) == 1
```

- [ ] **Step 2: Run tests to verify the first one fails**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py -k dhw_df_kwarg -v`
Expected: `test_dhw_df_kwarg_reuses_already_fetched_data_when_entity_matches` FAILS — `dhw_calls` is non-empty (the independent fetch still ran) because reuse isn't wired yet.

- [ ] **Step 3: Add the reuse branch to the DHW block**

Replace (`energy_forecast.py`, the `if cfg.get("dhw_tank_temp_sensor"):` block):

```python
        if cfg.get("dhw_tank_temp_sensor"):
            entity_id = cfg["dhw_tank_temp_sensor"]
            path = self._generic_sensor_cache_path(entity_id, prefix="physics_dhw_tank")
            try:
                df = generic_fetch(self, entity_id, path, column_name="buffer_temp", timezone=self._timezone)
                self._physics_dhw_tank_df = _keep_or_replace(self._physics_dhw_tank_df, df)
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Physics DHW tank %s %s fetch failed: %s", entity_id, verb, exc)
```

with:

```python
        if cfg.get("dhw_tank_temp_sensor"):
            entity_id = cfg["dhw_tank_temp_sensor"]
            if dhw_df is not None and entity_id == self._dhw_buffer_sensor:
                self._physics_dhw_tank_df = _keep_or_replace(self._physics_dhw_tank_df, dhw_df)
            else:
                path = self._generic_sensor_cache_path(entity_id, prefix="physics_dhw_tank")
                try:
                    df = generic_fetch(self, entity_id, path, column_name="buffer_temp", timezone=self._timezone)
                    self._physics_dhw_tank_df = _keep_or_replace(self._physics_dhw_tank_df, df)
                except (OSError, KeyError, ValueError) as exc:
                    _LOGGER.warning("Physics DHW tank %s %s fetch failed: %s", entity_id, verb, exc)
```

- [ ] **Step 4: Run tests to verify they all pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "fix: reuse already-fetched dhw_df for physics DHW tank sensor when entities match (#89)"
```

---

## Task 4: Wire `_retrain()` and `_update_sensors()` to pass their already-fetched data in

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:1653` (`_retrain()` call site), `:1816` (`_update_sensors()` call site)
- Test: `tests/test_energy_forecast_physics_config.py` (extend existing `TestRetrainCallsPhysicsFetch`; add an analogous class for `_update_sensors`)

**Interfaces:**
- Consumes: the fully-wired `_fetch_physics_sensor_histories(recent_only, climate_dfs, dhw_df)` signature from Tasks 2-3.
- Produces: nothing further downstream — this is the last task that touches `energy_forecast.py` production code.

- [ ] **Step 1: Write the failing tests**

Add to `TestRetrainCallsPhysicsFetch` in `tests/test_energy_forecast_physics_config.py`:

```python
    def test_retrain_passes_its_climate_dfs_and_dhw_df_to_physics_fetch(self, monkeypatch):
        """#89: _retrain() already fetches climate_dfs/dhw_df for the ML pipeline
        (energy_forecast.py ~1607-1627) — it must hand that data to
        _fetch_physics_sensor_histories() rather than let physics re-fetch it."""
        from energy_forecast.energy_forecast import EnergyForecast

        self._patch_retrain_deps(monkeypatch)

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        assert app._physics_model is not None

        app._ml_model = MagicMock()
        app._fetch_physics_sensor_histories = MagicMock()
        app._retrain = EnergyForecast._retrain.__get__(app, type(app))

        app._retrain()

        call_kwargs = app._fetch_physics_sensor_histories.call_args.kwargs
        assert "climate_dfs" in call_kwargs
        assert "dhw_df" in call_kwargs
```

Add a new class:

```python
class TestUpdateSensorsPassesRecentDataToPhysicsFetch:
    """#89: _update_sensors() already fetches climate_recent/dhw_recent for the ML
    pipeline (energy_forecast.py ~1780-1799) before calling
    _fetch_physics_sensor_histories(recent_only=True) — it must hand that data in too."""

    def test_update_sensors_passes_climate_recent_and_dhw_recent(self):
        import inspect

        from energy_forecast.energy_forecast import EnergyForecast

        source = inspect.getsource(EnergyForecast._update_sensors)
        assert "climate_dfs=climate_recent" in source
        assert "dhw_df=dhw_recent" in source
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py -k "passes_its_climate_dfs_and_dhw_df or passes_climate_recent_and_dhw_recent" -v`
Expected: both FAIL — `climate_dfs`/`dhw_df` not in `_retrain()`'s call kwargs yet; the literal strings not in `_update_sensors()`'s source yet.

- [ ] **Step 3: Wire `_retrain()`'s call site**

At `energy_forecast.py:1653`, replace:
```python
        self._fetch_physics_sensor_histories()
```
with:
```python
        self._fetch_physics_sensor_histories(climate_dfs=climate_dfs, dhw_df=dhw_df)
```
(`climate_dfs` and `dhw_df` are the local variables already built at `energy_forecast.py:1607-1627`, in scope at this call site.)

- [ ] **Step 4: Wire `_update_sensors()`'s call site**

At `energy_forecast.py:1816`, replace:
```python
        self._fetch_physics_sensor_histories(recent_only=True)
```
with:
```python
        self._fetch_physics_sensor_histories(recent_only=True, climate_dfs=climate_recent, dhw_df=dhw_recent)
```
(`climate_recent` and `dhw_recent` are the local variables already built at `energy_forecast.py:1779-1799`, in scope at this call site.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: PASS, full count.

- [ ] **Step 7: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "fix: wire _retrain()/_update_sensors() to share climate/dhw fetches with physics (#89)"
```

---

## Task 5: Docs — CHANGELOG, ROADMAP, MEMORY

**Files:**
- Modify: `CHANGELOG.md` (via `@changelog-writer` agent)
- Modify: `ROADMAP.md` — move `#89` from Backlog/Pending Summary to Done/Completed Items
- Modify: `MEMORY.md` / `memory/*.md` if this investigation revealed anything worth remembering long-term (e.g. the room-thermostat dead-code finding, if it suggests a broader pattern worth watching for)

- [ ] **Step 1: Run the changelog-writer agent**

Dispatch `@changelog-writer` with: "Document the #89 fix — physics sensor history duplication removed: dead room-thermostat temp_sensor fetch deleted, DHW tank + room climate fetches now reuse the ML pipeline's already-fetched data instead of independently re-fetching the same HA entities."

- [ ] **Step 2: Update `ROADMAP.md`**

Move the `#89` entry from `## Backlog` to a new row in `### Completed Items` (Done section), and remove its row from the `## Pending Summary` table. Use the version number that lands this fix (check `apps/energy_forecast/__init__.py`'s `__version__` after bumping, per the standard release workflow).

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md ROADMAP.md memory/ MEMORY.md 2>/dev/null; git add CHANGELOG.md ROADMAP.md
git commit -m "docs: close out #89 (physics sensor history dedup)"
```

(Note: `MEMORY.md` is gitignored — never `git add MEMORY.md`, per global instructions. Only stage `memory/*.md` files if new ones were created.)

---

## Task 6 (manual, optional, post-deploy): Delete the now-orphaned CSV caches on HA

**Status (2026-07-16): done.** Tasks 1-5 merged to `dev` (commit `fc78113`, pushed) and deployed to the live HA instance the same day (bundled with enabling the solar target-correction config, `scripts/deploy.py` + one AppDaemon restart). Step 1 confirmed clean via `scripts/check_ha_logs.py`: post-restart `energy_forecast:` log lines were limited to a pre-existing, unrelated HA-vs-hef timezone warning and one benign `lag_24h` NaN-fill (that fill warning itself confirms an hourly cycle — 15:01 — ran without error). Steps 2-3 executed: all 11 orphaned files deleted via SMB and re-listing confirmed zero remain.

- [x] **Step 1: Confirm the fix has been deployed and at least one retrain/hourly cycle has run clean** (check `scripts/check_ha_logs.py` output for no new errors from `energy_forecast.py`).

- [x] **Step 2: Delete the 11 orphaned files via SMB** (same connection pattern as `scripts/pull_ha_data.py`):

```python
import os
from smb.SMBConnection import SMBConnection

HA_HOST = "homeassistant"
SMB_USER = os.getenv("EM_SMB_USER", "martin")
SMB_PASSWORD = os.getenv("EM_SMB_PASSWORD")
SMB_SHARE = "addon_configs"
FORECAST_REMOTE = "a0d7b954_appdaemon/apps/energy_forecast"

ORPHANED = [
    "physics_dhw_tank_em_kermi_bridge_kermi_hot_water_temperature.csv",
] + [
    f"physics_temp_{i}_em_netatmo_bridge_netatmo_{room}_temperature_2.csv"
    for i, room in enumerate([
        "badezimmer", "buro_andrea", "buro_martin", "esszimmer", "gang",
        "kinderzimmer", "kuche", "schlafzimmer", "wc_ug", "wohnzimmer",
    ])
]

conn = SMBConnection(SMB_USER, SMB_PASSWORD, "cleanup", HA_HOST, use_ntlm_v2=True)
conn.connect(HA_HOST, 445)
for fname in ORPHANED:
    conn.deleteFiles(SMB_SHARE, f"{FORECAST_REMOTE}/{fname}")
    print(f"deleted {fname}")
```

- [x] **Step 3: Re-list the directory to confirm** (`_list_dir` pattern from `scripts/pull_ha_data.py`) — no `physics_dhw_tank_*` or `physics_temp_*` files remain.

---

## Self-Review Notes

- **Spec coverage:** DHW dedup (Task 3+4), room-thermostat dead-fetch removal (Task 1), room-thermostat climate re-fetch dedup (Task 2+4), orphaned cache cleanup (Task 6), docs (Task 5) — all three numbered items from the ROADMAP `#89` entry are covered, plus the stronger dead-code finding this investigation surfaced.
- **Backward compatibility:** every new kwarg defaults to `None` and falls back to the original independent-fetch behavior — `_recalibrate_physics_cb` (the on-demand service call) is intentionally left uncalled-with-new-kwargs, since it has no already-fetched data available at that call site.
- **Type consistency:** `_fetch_physics_sensor_histories(self, recent_only: bool = False, climate_dfs: dict[str, Any] | None = None, dhw_df: Any = None) -> None` is the same signature introduced incrementally in Tasks 2 and 3, and is exactly what Task 4's call sites use.
