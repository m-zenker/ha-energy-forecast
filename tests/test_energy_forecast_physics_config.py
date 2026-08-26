"""Tests for physics: config ingest in energy_forecast.py initialize()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _restore_module_loggers():
    """Prevent EnergyForecast.initialize() from leaking a MagicMock _LOGGER.

    initialize() reassigns the module-level `_LOGGER` global in
    energy_forecast, ha_data, model, and weather to `self.logger` (intentional
    production wiring for AppDaemon's per-app logger). Since these tests bind
    and call the real initialize() against a MagicMock app, that permanently
    overwrites those globals with a MagicMock for the rest of the pytest
    process, breaking caplog-based assertions in other test modules. Snapshot
    and restore the real loggers around every test in this file.
    """
    from energy_forecast import energy_forecast as ef_module
    from energy_forecast import ha_data, model, weather

    modules = (ef_module, ha_data, model, weather)
    original_loggers = [m._LOGGER for m in modules]
    try:
        yield
    finally:
        for m, logger in zip(modules, original_loggers):
            m._LOGGER = logger


def _make_app(args: dict):
    """Build a minimal fake EnergyForecast instance for initialize() testing."""
    from energy_forecast.energy_forecast import EnergyForecast

    # Provide minimal required args that initialize() expects
    full_args = {
        "energy_sensor": "sensor.grid_import",
        "latitude": 47.3,
        "longitude": 8.5,
        "altitude_m": 700,
        "climate_entities": [],
        "use_holidays": False,
        "use_sun_position": False,
        "use_oob_detection": False,
    }
    full_args.update(args)

    # Use a MagicMock without spec to allow any method call
    app = MagicMock()
    app.args = full_args
    app.logger = MagicMock()
    app.register_service = MagicMock()
    app.listen_event = MagicMock()
    app.run_hourly = MagicMock()
    app.run_every = MagicMock()
    app.get_timezone = MagicMock(return_value="Europe/Zurich")

    # Bind the real initialize method to the mock
    app.initialize = EnergyForecast.initialize.__get__(app, type(app))
    app._fetch_physics_sensor_histories = EnergyForecast._fetch_physics_sensor_histories.__get__(app, type(app))
    app._model_phase_attr = EnergyForecast._model_phase_attr.__get__(app, type(app))
    return app


class TestPhysicsConfigIngest:
    def test_absent_physics_block_disables_model(self, tmp_path, monkeypatch):
        from energy_forecast import energy_forecast as ef_module

        monkeypatch.setattr(ef_module.Path, "__truediv__", ef_module.Path.__truediv__)
        app = _make_app({})
        app.initialize()
        assert app._physics_model is None

    def test_physics_block_creates_model(self, tmp_path, monkeypatch):
        app = _make_app(
            {
                "physics": {
                    "cop_sensor": "sensor.kermi_cop",
                    "dhw_tank_temp_sensor": "sensor.kermi_dhw_buffer_temp",
                    "room_thermostats": [
                        {
                            "climate_entity": "climate.living_room",
                            "temp_sensor": "sensor.netatmo_living_room_temp",
                            "area_m2": 35,
                        }
                    ],
                    "use_physics_residual": False,
                },
            }
        )
        app.initialize()
        assert app._physics_model is not None
        assert app._room_thermostats == [
            {"climate_entity": "climate.living_room", "temp_sensor": "sensor.netatmo_living_room_temp", "area_m2": 35.0}
        ]

    def test_room_thermostats_missing_required_key_skipped_with_warning(self):
        app = _make_app(
            {
                "physics": {"room_thermostats": [{"climate_entity": "climate.living_room"}]},  # missing temp_sensor
            }
        )
        app.initialize()
        assert app._room_thermostats == []

    def test_defaults_applied_when_physics_block_partial(self):
        app = _make_app({"physics": {}})
        app.initialize()
        assert app._physics_model is not None
        assert app._physics_config["dhw_tank_volume_l"] == 200
        assert app._physics_config["internal_gains_fraction"] == 0.8
        assert app._physics_config["use_physics_residual"] is False


class TestPhysicsSensorFetch:
    def test_no_physics_model_skips_all_physics_fetches(self, monkeypatch):
        from energy_forecast import ha_data as hd

        fetch_generic = MagicMock(return_value=None)
        fetch_climate = MagicMock(return_value=None)
        monkeypatch.setattr(hd, "fetch_generic_sensor_history", fetch_generic)
        monkeypatch.setattr(hd, "fetch_climate_history", fetch_climate)

        app = _make_app({"energy_sensor": "sensor.grid_import"})  # no physics: block
        app.initialize()
        app._fetch_physics_sensor_histories()
        fetch_generic.assert_not_called()
        fetch_climate.assert_not_called()

    def test_physics_model_present_fetches_configured_sensors(self, monkeypatch):
        import pandas as pd
        from energy_forecast import ha_data as hd

        empty_df = pd.DataFrame(columns=["timestamp", "value"])
        fetch_generic = MagicMock(return_value=empty_df)
        fetch_climate = MagicMock(return_value=pd.DataFrame(columns=["timestamp", "current_temp", "setpoint"]))
        monkeypatch.setattr(hd, "fetch_generic_sensor_history", fetch_generic)
        monkeypatch.setattr(hd, "fetch_climate_history", fetch_climate)

        app = _make_app(
            {
                "energy_sensor": "sensor.grid_import",
                "physics": {
                    "dhw_tank_temp_sensor": "sensor.kermi_dhw_buffer_temp",
                    "heating_buffer_temp_sensor": "sensor.kermi_heating_buffer",
                    "cop_sensor": "sensor.kermi_cop",
                    "room_thermostats": [
                        {
                            "climate_entity": "climate.living_room",
                            "temp_sensor": "sensor.netatmo_living_room_temp",
                            "area_m2": 35,
                        }
                    ],
                },
            }
        )
        app.initialize()
        app._fetch_physics_sensor_histories()
        assert fetch_generic.call_count == 3  # dhw_tank, heating_buffer, cop
        fetch_climate.assert_called_once()  # the room_thermostat's climate_entity, for setpoint projection

        # Regression guard: physics.py's ThermalPhysicsModel hard-codes "buffer_temp"
        # as the column name for the DHW tank dataframe (see _calibrate_ua_dhw() and
        # _infer_dhw_schedule()). If the fetch call here uses any other column_name,
        # calibrate() silently no-ops the DHW branch via its blanket except-Exception
        # handler, with no test failure to catch it — so assert the literal string.
        dhw_call = next(call for call in fetch_generic.call_args_list if call.args[1] == "sensor.kermi_dhw_buffer_temp")
        assert dhw_call.kwargs["column_name"] == "buffer_temp"


class TestPhysicsSensorRecentFetch:
    """recent_only=True: the hourly refresh path that keeps physics sensor data fresh
    between retrains (weekly, or adaptive-retrain-gated), instead of it staying
    None/empty — from initialize()'s defensive reset — until the next training cycle,
    which can be up to a week away after any AppDaemon restart."""

    def _configured_app(self, monkeypatch):
        from energy_forecast import ha_data as hd

        fetch_recent_generic = MagicMock(return_value=pd.DataFrame(columns=["timestamp", "value"]))
        fetch_recent_climate = MagicMock(return_value=pd.DataFrame(columns=["timestamp", "current_temp", "setpoint"]))
        fetch_generic = MagicMock(return_value=pd.DataFrame(columns=["timestamp", "value"]))
        fetch_climate = MagicMock(return_value=pd.DataFrame(columns=["timestamp", "current_temp", "setpoint"]))
        monkeypatch.setattr(hd, "fetch_recent_generic_sensor", fetch_recent_generic)
        monkeypatch.setattr(hd, "fetch_recent_climate", fetch_recent_climate)
        monkeypatch.setattr(hd, "fetch_generic_sensor_history", fetch_generic)
        monkeypatch.setattr(hd, "fetch_climate_history", fetch_climate)

        app = _make_app(
            {
                "energy_sensor": "sensor.grid_import",
                "physics": {
                    "dhw_tank_temp_sensor": "sensor.kermi_dhw_buffer_temp",
                    "heating_buffer_temp_sensor": "sensor.kermi_heating_buffer",
                    "cop_sensor": "sensor.kermi_cop",
                    "room_thermostats": [
                        {
                            "climate_entity": "climate.living_room",
                            "temp_sensor": "sensor.netatmo_living_room_temp",
                            "area_m2": 35,
                        }
                    ],
                },
            }
        )
        app.initialize()
        return app, fetch_recent_generic, fetch_recent_climate, fetch_generic, fetch_climate

    def test_recent_only_uses_recent_variants_not_full_history(self, monkeypatch):
        app, fetch_recent_generic, fetch_recent_climate, fetch_generic, fetch_climate = self._configured_app(
            monkeypatch
        )

        app._fetch_physics_sensor_histories(recent_only=True)

        assert fetch_recent_generic.call_count == 3  # dhw_tank, heating_buffer, cop
        fetch_recent_climate.assert_called_once()
        fetch_generic.assert_not_called()
        fetch_climate.assert_not_called()

    def test_recent_only_populates_attrs_without_any_retrain_having_run(self, monkeypatch):
        """Reproduces the bug: right after initialize() (simulating a fresh AppDaemon
        restart), every physics history attribute is None/empty — _retrain() has never
        run. A single recent_only=True refresh must populate them from HA history
        directly, without waiting for the next (possibly week-away) retrain."""
        import pandas as pd

        app, fetch_recent_generic, fetch_recent_climate, _, _ = self._configured_app(monkeypatch)
        fetch_recent_generic.return_value = pd.DataFrame({"timestamp": ["2026-07-13 10:00:00"], "value": [27.3]})
        fetch_recent_climate.return_value = pd.DataFrame(
            {"timestamp": ["2026-07-13 10:00:00"], "current_temp": [21.0], "setpoint": [20.0]}
        )

        assert app._physics_heating_buffer_df is None
        assert app._physics_dhw_tank_df is None
        assert app._physics_cop_df is None

        app._fetch_physics_sensor_histories(recent_only=True)

        assert app._physics_heating_buffer_df is not None and not app._physics_heating_buffer_df.empty
        assert app._physics_dhw_tank_df is not None and not app._physics_dhw_tank_df.empty
        assert app._physics_cop_df is not None and not app._physics_cop_df.empty

    def test_recent_only_empty_fetch_preserves_existing_value(self, monkeypatch):
        """A transient failure (empty result) during the hourly refresh must not
        regress an already-good value — e.g. one just populated by a retrain — back
        to empty/None."""
        import pandas as pd

        app, fetch_recent_generic, _, _, _ = self._configured_app(monkeypatch)
        good_df = pd.DataFrame({"timestamp": ["2026-07-13 09:00:00"], "heating_buffer_temp": [28.7]})
        app._physics_heating_buffer_df = good_df
        fetch_recent_generic.return_value = pd.DataFrame(columns=["timestamp", "value"])  # empty this cycle

        app._fetch_physics_sensor_histories(recent_only=True)

        assert app._physics_heating_buffer_df is good_df  # unchanged, not clobbered by the empty result

    def test_full_history_fetch_still_resets_and_replaces_on_empty(self, monkeypatch):
        """recent_only=False (retrain path) keeps its original v0.11.7 behaviour: always
        reset first, and always assign the fetch result even when empty."""
        import pandas as pd

        app, _, _, fetch_generic, _ = self._configured_app(monkeypatch)
        app._physics_heating_buffer_df = pd.DataFrame({"timestamp": ["2026-07-06 09:00:00"], "value": [26.0]})
        fetch_generic.return_value = pd.DataFrame(columns=["timestamp", "value"])  # empty this retrain

        app._fetch_physics_sensor_histories()  # recent_only defaults to False

        assert app._physics_heating_buffer_df.empty

    def test_recent_only_quiets_only_cop_sensor_no_data_warning(self, monkeypatch):
        """A heat pump's live COP sensor only reports while actively heating — an empty
        hourly recent-fetch is expected (not an error) for long idle stretches (e.g. all
        summer), so only its recent_only fetch should pass quiet_if_empty=True. The other
        two physics sensors (heating_buffer_temp, dhw_tank) have no such seasonal excuse
        and must keep the default (noisy-on-empty) behaviour."""
        app, fetch_recent_generic, _, _, _ = self._configured_app(monkeypatch)

        app._fetch_physics_sensor_histories(recent_only=True)

        calls_by_entity = {call.args[1]: call for call in fetch_recent_generic.call_args_list}
        assert calls_by_entity["sensor.kermi_cop"].kwargs.get("quiet_if_empty") is True
        assert calls_by_entity["sensor.kermi_dhw_buffer_temp"].kwargs.get("quiet_if_empty") is not True
        assert calls_by_entity["sensor.kermi_heating_buffer"].kwargs.get("quiet_if_empty") is not True

    def test_full_history_fetch_never_quiets_cop_sensor(self, monkeypatch):
        """recent_only=False (retrain path): cop_sensor's weekly full-history fetch keeps
        its normal WARNING — only the hourly recent fetch is quieted, since the weekly
        fetch isn't the one that got noisy from this fix."""
        app, _, _, fetch_generic, _ = self._configured_app(monkeypatch)

        app._fetch_physics_sensor_histories()  # recent_only defaults to False

        cop_call = next(call for call in fetch_generic.call_args_list if call.args[1] == "sensor.kermi_cop")
        assert "quiet_if_empty" not in cop_call.kwargs

    def test_room_thermostat_temp_sensor_is_not_independently_fetched(self, monkeypatch, tmp_path):
        """#89: room_thermostats[].temp_sensor duplicates data already available from
        climate_entity (same Netatmo bridge poll, republished under two entity IDs) and
        was never consumed by physics.py — only climate_dfs/climate_recent is. Confirms
        the dead fetch + its on-disk cache file are both gone."""
        app, fetch_recent_generic, fetch_recent_climate, _, _ = self._configured_app(monkeypatch)
        app._cache_path = tmp_path / "energy_history.csv"  # cache paths derive from this

        app._fetch_physics_sensor_histories(recent_only=True)

        # Verify the dead attribute was never assigned to the instance (distinguishes real
        # attribute assignments from MagicMock's auto-vivified attributes; see comment at
        # test_physics_history_attrs_defensively_initialized_before_first_retrain)
        assert "_room_thermostat_temp_dfs" not in app.__dict__
        # Verify room temp sensor was not fetched (climate_entity provides the data instead)
        fetched_entities = {call.args[1] for call in fetch_recent_generic.call_args_list}
        assert "sensor.netatmo_living_room_temp" not in fetched_entities
        assert not (tmp_path / "physics_temp_0_sensor.netatmo_living_room_temp.csv").exists()

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

    def test_climate_dfs_kwarg_reuse_failure_does_not_abort_the_cycle(self, monkeypatch):
        """The reuse branch must match the function's stated resilience contract
        (docstring: 'a failed... fetch never regresses an already-good value') —
        a malformed climate_dfs entry must not raise and abort the whole call."""
        app, _, fetch_recent_climate, _, _ = self._configured_app(monkeypatch)

        app._fetch_physics_sensor_histories(recent_only=True, climate_dfs={"climate.living_room": None})  # malformed

        # must not raise; must not have fallen through to the independent fetch either
        fetch_recent_climate.assert_not_called()

    def test_dhw_df_kwarg_reuses_already_fetched_data_when_entity_matches(self, monkeypatch):
        """#89: when physics.dhw_tank_temp_sensor == the top-level dhw_buffer_sensor (the
        same entity, e.g. sensor.em_kermi_bridge_kermi_hot_water_temperature on the live
        instance), physics must reuse the ML pipeline's already-fetched DHW DataFrame
        instead of an independent second HA history fetch of the same entity."""
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

    def test_dhw_df_kwarg_reuse_failure_does_not_abort_the_cycle(self, monkeypatch):
        """The reuse branch must match the function's stated resilience contract —
        a malformed dhw_df must not raise and abort the whole call."""
        app, fetch_recent_generic, _, _, _ = self._configured_app(monkeypatch)
        app._dhw_buffer_sensor = "sensor.kermi_dhw_buffer_temp"  # matches physics.dhw_tank_temp_sensor in fixture

        app._fetch_physics_sensor_histories(recent_only=True, dhw_df={"not": "a dataframe"})  # malformed

        # must not raise; must not have fallen through to the independent fetch either
        dhw_calls = [c for c in fetch_recent_generic.call_args_list if c.args[1] == "sensor.kermi_dhw_buffer_temp"]
        assert dhw_calls == []


class TestUpdateSensorsCallsPhysicsRecentFetch:
    """_update_sensors() must call _fetch_physics_sensor_histories(recent_only=True)
    every hourly cycle — the same wiring gap class as the EV cache-path bug
    (TestEvChargingCachePathAlwaysInvoked in test_energy_forecast.py): source-level
    verification, since exercising the full _update_sensors() pipeline requires
    mocking weather/HA history fetches unrelated to this fix."""

    def test_update_sensors_calls_recent_only_refresh(self):
        import inspect

        from energy_forecast.energy_forecast import EnergyForecast

        source = inspect.getsource(EnergyForecast._update_sensors)
        assert "recent_only=True" in source
        assert "self._fetch_physics_sensor_histories" in source


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


class TestRetrainCallsPhysicsFetch:
    """Task 2 gap fix: _retrain() must call _fetch_physics_sensor_histories() once per cycle."""

    def _patch_retrain_deps(self, monkeypatch):
        import pandas as pd
        from energy_forecast import ha_data as ha_data_mod
        from energy_forecast import weather as weather_mod

        energy_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2000, freq="1h"),
                "gross_kwh": [1.0] * 2000,
            }
        )
        empty_df = pd.DataFrame(columns=["timestamp"])

        monkeypatch.setattr(ha_data_mod, "fetch_energy_history", lambda *a, **kw: energy_df)
        monkeypatch.setattr(
            ha_data_mod,
            "split_ev_charging",
            lambda df, *a, **kw: (df, pd.DataFrame(columns=["timestamp", "gross_kwh"])),
        )
        monkeypatch.setattr(weather_mod, "fetch_historical_weather", lambda *a, **kw: empty_df)
        monkeypatch.setattr(weather_mod, "fetch_open_meteo", lambda *a, **kw: empty_df)
        monkeypatch.setattr(ha_data_mod, "fetch_boolean_entity_history", lambda *a, **kw: empty_df)
        monkeypatch.setattr(ha_data_mod, "fetch_presence_history", lambda *a, **kw: empty_df)

    def test_retrain_calls_fetch_physics_sensor_histories_when_physics_enabled(self, monkeypatch):
        from energy_forecast.energy_forecast import EnergyForecast

        self._patch_retrain_deps(monkeypatch)

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        assert app._physics_model is not None

        app._ml_model = MagicMock()  # avoid a real training cycle
        app._fetch_physics_sensor_histories = MagicMock()
        app._retrain = EnergyForecast._retrain.__get__(app, type(app))

        app._retrain()

        app._fetch_physics_sensor_histories.assert_called_once()

    def test_retrain_skips_physics_fetch_when_no_physics_model(self, monkeypatch):
        """Sanity check: the real (unmocked) _fetch_physics_sensor_histories() is a no-op
        when physics is disabled, so calling it unconditionally from _retrain() is safe."""
        from energy_forecast.energy_forecast import EnergyForecast

        self._patch_retrain_deps(monkeypatch)

        app = _make_app({"energy_sensor": "sensor.grid_import"})  # no physics: block
        app.initialize()
        assert app._physics_model is None

        app._ml_model = MagicMock()
        app._retrain = EnergyForecast._retrain.__get__(app, type(app))

        app._retrain()  # must not raise

        assert app._physics_dhw_tank_df is None
        assert app._physics_heating_buffer_df is None
        assert app._physics_cop_df is None

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


class TestRetrainWiresUsePhysicsResidual:
    """Final-review fix: `_retrain()` must pass `_effective_use_physics_residual()` through to
    `EnergyForecastModel.train()` as `use_physics_residual=`. Without this wiring, the
    cold-start-gate design (config intent AND-ed with calibration-window count) is computed but
    never actually reaches training, so Phase 2 residual learning can never activate."""

    def _patch_retrain_deps(self, monkeypatch):
        import pandas as pd
        from energy_forecast import ha_data as ha_data_mod
        from energy_forecast import weather as weather_mod

        energy_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2000, freq="1h"),
                "gross_kwh": [1.0] * 2000,
            }
        )
        empty_df = pd.DataFrame(columns=["timestamp"])

        monkeypatch.setattr(ha_data_mod, "fetch_energy_history", lambda *a, **kw: energy_df)
        monkeypatch.setattr(
            ha_data_mod,
            "split_ev_charging",
            lambda df, *a, **kw: (df, pd.DataFrame(columns=["timestamp", "gross_kwh"])),
        )
        monkeypatch.setattr(weather_mod, "fetch_historical_weather", lambda *a, **kw: empty_df)
        monkeypatch.setattr(weather_mod, "fetch_open_meteo", lambda *a, **kw: empty_df)
        monkeypatch.setattr(ha_data_mod, "fetch_boolean_entity_history", lambda *a, **kw: empty_df)
        monkeypatch.setattr(ha_data_mod, "fetch_presence_history", lambda *a, **kw: empty_df)

    def _make_retrain_ready_app(self, monkeypatch, *, use_physics_residual: bool):
        from energy_forecast.energy_forecast import EnergyForecast

        self._patch_retrain_deps(monkeypatch)

        app = _make_app(
            {"energy_sensor": "sensor.grid_import", "physics": {"use_physics_residual": use_physics_residual}}
        )
        app.initialize()
        assert app._physics_model is not None

        app._ml_model = MagicMock()  # avoid a real training cycle
        app._fetch_physics_sensor_histories = MagicMock()
        app._effective_use_physics_residual = EnergyForecast._effective_use_physics_residual.__get__(app, type(app))
        app._retrain = EnergyForecast._retrain.__get__(app, type(app))
        return app

    def test_retrain_passes_use_physics_residual_true_when_gate_clear(self, monkeypatch):
        app = self._make_retrain_ready_app(monkeypatch, use_physics_residual=True)

        # Clear the cold-start gate so _effective_use_physics_residual() resolves to True.
        app._physics_model._calib["n_calibration_windows_ua_eff"] = 30
        assert app._physics_model.is_cold_start_gated is False

        app._retrain()

        app._ml_model.train.assert_called_once()
        assert app._ml_model.train.call_args.kwargs["use_physics_residual"] is True

    def test_retrain_passes_use_physics_residual_false_when_gate_not_clear(self, monkeypatch):
        app = self._make_retrain_ready_app(monkeypatch, use_physics_residual=True)

        # Fresh model, no calibration windows accumulated yet — gate stays closed.
        assert app._physics_model.is_cold_start_gated is True

        app._retrain()

        app._ml_model.train.assert_called_once()
        assert app._ml_model.train.call_args.kwargs["use_physics_residual"] is False


class TestRecalibratePhysicsService:
    def test_service_registered_when_physics_enabled(self):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        calls = [c.args[0] for c in app.register_service.call_args_list]
        assert "energy_forecast/recalibrate_physics" in calls

    def test_service_not_registered_when_physics_disabled(self):
        app = _make_app({"energy_sensor": "sensor.grid_import"})
        app.initialize()
        calls = [c.args[0] for c in app.register_service.call_args_list]
        assert "energy_forecast/recalibrate_physics" not in calls

    def test_recalibrate_service_calls_physics_calibrate_and_updates_calibrated_at(self, tmp_path):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._recalibrate_physics_cb = EnergyForecast._recalibrate_physics_cb.__get__(app, type(app))
        app._fetch_physics_sensor_histories = MagicMock()
        app._physics_climate_dfs = {}
        app._physics_dhw_tank_df = None
        # minimal energy/weather fetch stand-ins
        app._cached_energy_df = pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-01", periods=10, freq="1h"), "gross_kwh": [1.0] * 10}
        )
        app._cached_weather_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=10, freq="1h"),
                "temp_c": [5.0] * 10,
                "direct_radiation_wm2": [0.0] * 10,
            }
        )
        with patch.object(app._physics_model, "calibrate") as mock_calibrate:
            app._recalibrate_physics_cb("default", "energy_forecast", "recalibrate_physics", {})
            mock_calibrate.assert_called_once()

    def test_recalibrate_service_before_first_retrain_logs_warning_and_does_not_crash(self, caplog):
        """No _cached_energy_df/_cached_weather_df yet (never retrained) — must not raise."""
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.logger = logging.getLogger("energy_forecast")  # real logger so caplog sees post-initialize() warnings
        app.initialize()
        app._recalibrate_physics_cb = EnergyForecast._recalibrate_physics_cb.__get__(app, type(app))
        assert app._cached_energy_df is None
        assert app._cached_weather_df is None

        with patch.object(app._physics_model, "calibrate") as mock_calibrate:
            with caplog.at_level(logging.WARNING, logger="energy_forecast"):
                app._recalibrate_physics_cb("default", "energy_forecast", "recalibrate_physics", {})
            mock_calibrate.assert_not_called()
        assert any("recalibrate_physics" in r.message.lower() for r in caplog.records)

    def test_cold_start_gate_logs_warning_when_residual_requested_but_gated(self, caplog):
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {"use_physics_residual": True}})
        app.logger = logging.getLogger("energy_forecast")  # real logger so caplog sees post-initialize() warnings
        app.initialize()
        app._effective_use_physics_residual = EnergyForecast._effective_use_physics_residual.__get__(app, type(app))
        assert app._physics_model.is_cold_start_gated is True  # fresh model, no calibration windows yet
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            app._effective_use_physics_residual()
        assert any("cold-start" in r.message.lower() or "cold start" in r.message.lower() for r in caplog.records)

    def test_phase2_activation_logs_experimental_warning_when_gate_clear(self, caplog):
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {"use_physics_residual": True}})
        app.logger = logging.getLogger("energy_forecast")  # real logger so caplog sees post-initialize() warnings
        app.initialize()
        app._effective_use_physics_residual = EnergyForecast._effective_use_physics_residual.__get__(app, type(app))
        app._physics_model._calib["n_calibration_windows_ua_eff"] = 30
        assert app._physics_model.is_cold_start_gated is False

        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = app._effective_use_physics_residual()

        assert result is True
        assert any("experimental" in r.message.lower() and "phase 2" in r.message.lower() for r in caplog.records)


class TestPhysicsSensors:
    def test_physics_sensors_published_when_physics_configured(self, monkeypatch):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._publish_physics_sensors = EnergyForecast._publish_physics_sensors.__get__(app, type(app))
        app.set_state = MagicMock()
        ts = pd.date_range("2026-01-15", periods=48, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "predicted_kwh": [1.0] * 48})
        monkeypatch.setattr(app._physics_model, "predict_series", lambda *a, **kw: pd.Series([0.6] * 48, index=ts))
        app._publish_physics_sensors(forecast_df)
        published_entities = [c.args[0] for c in app.set_state.call_args_list]
        assert "sensor.energy_forecast_physics_base_today" in published_entities
        assert "sensor.energy_forecast_ml_adjustment_today" in published_entities

    def test_physics_sensors_not_published_when_physics_disabled(self):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import"})
        app.initialize()
        app._publish_physics_sensors = EnergyForecast._publish_physics_sensors.__get__(app, type(app))
        app.set_state = MagicMock()
        app._publish_physics_sensors(pd.DataFrame({"timestamp": [], "predicted_kwh": []}))
        app.set_state.assert_not_called()

    def test_ml_adjustment_can_be_negative(self, monkeypatch):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._publish_physics_sensors = EnergyForecast._publish_physics_sensors.__get__(app, type(app))
        app.set_state = MagicMock()
        ts = pd.date_range("2026-01-15", periods=48, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "predicted_kwh": [0.5] * 48})  # physics baseline exceeds total
        monkeypatch.setattr(app._physics_model, "predict_series", lambda *a, **kw: pd.Series([0.8] * 48, index=ts))
        app._publish_physics_sensors(forecast_df)
        adj_call = next(
            c for c in app.set_state.call_args_list if c.args[0] == "sensor.energy_forecast_ml_adjustment_today"
        )
        assert float(adj_call.kwargs["state"]) < 0

    def test_physics_sensors_use_mqtt_in_mqtt_mode(self, monkeypatch):
        """In MQTT mode, both sensors must go through _mqtt_set_sensor, not set_state."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._publish_physics_sensors = EnergyForecast._publish_physics_sensors.__get__(app, type(app))
        app._mqtt_discovery = True
        app.set_state = MagicMock()
        app._mqtt_set_sensor = MagicMock()
        app._mqtt_publish_sensor_attributes = MagicMock()
        ts = pd.date_range("2026-01-15", periods=48, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "predicted_kwh": [1.0] * 48})
        monkeypatch.setattr(app._physics_model, "predict_series", lambda *a, **kw: pd.Series([0.6] * 48, index=ts))
        app._publish_physics_sensors(forecast_df)

        mqtt_ids = [c.args[0] for c in app._mqtt_set_sensor.call_args_list]
        assert "energy_forecast_physics_base_today" in mqtt_ids
        assert "energy_forecast_ml_adjustment_today" in mqtt_ids
        attr_ids = [c.args[0] for c in app._mqtt_publish_sensor_attributes.call_args_list]
        assert "energy_forecast_physics_base_today" in attr_ids
        assert "energy_forecast_ml_adjustment_today" in attr_ids
        app.set_state.assert_not_called()

    def test_physics_history_attrs_defensively_initialized_before_first_retrain(self):
        """initialize() must set all _fetch_physics_sensor_histories() output attributes to
        their safe defaults immediately — not just _physics_heating_buffer_df — so
        _publish_physics_sensors() (called every hourly update cycle) never raises
        AttributeError if it fires before the first _retrain() has ever completed (e.g. right
        after physics: is newly enabled and the weekly retrain hasn't run yet)."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        # Real (non-mocked) attribute assignments from initialize() are distinguishable from
        # MagicMock's auto-vivified attributes: an unset attribute would return a fresh MagicMock
        # on access, not `{}`/`None`, so these assertions genuinely fail without the fix.
        assert app._physics_climate_dfs == {}
        assert app._physics_dhw_tank_df is None
        assert app._physics_cop_df is None
        assert app._physics_heating_buffer_df is None

        # Real predict_series (not monkeypatched away) must run without raising even though
        # _fetch_physics_sensor_histories() was never called.
        app._publish_physics_sensors = EnergyForecast._publish_physics_sensors.__get__(app, type(app))
        app.set_state = MagicMock()
        ts = pd.date_range("2026-01-15", periods=48, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "predicted_kwh": [1.0] * 48})
        app._cached_forecast_df = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 48})
        app._publish_physics_sensors(forecast_df)  # must not raise AttributeError
        published_entities = [c.args[0] for c in app.set_state.call_args_list]
        assert "sensor.energy_forecast_physics_base_today" in published_entities


class TestPhysicsSensorsOverrideDeltaAttribution:
    """Task 11 review fix: _publish_physics_sensors' `ml_adjustment_vals =
    predicted_kwh - physics_vals` must not misattribute the Task 11 DHW override
    correction (already folded into `predicted_kwh` by the caller) to "ML
    Adjustment" — `physics_vals` must absorb the same `override_delta` first."""

    def test_override_delta_folds_into_physics_base_not_ml_adjustment(self, monkeypatch):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._publish_physics_sensors = EnergyForecast._publish_physics_sensors.__get__(app, type(app))
        app.set_state = MagicMock()
        ts = pd.date_range("2026-01-15", periods=48, freq="1h")
        # predicted_kwh already carries the Task 11 correction, as _update_sensors
        # leaves it: a flat 1.0 kWh/h ML forecast plus a +2.0 kWh override bump at
        # hour 5. The override-blind physics baseline is a flat 0.6 kWh/h.
        forecast_df = pd.DataFrame({"timestamp": ts, "predicted_kwh": [1.0] * 48})
        forecast_df.loc[5, "predicted_kwh"] = 3.0
        monkeypatch.setattr(app._physics_model, "predict_series", lambda *a, **kw: pd.Series([0.6] * 48, index=ts))

        override_delta = pd.Series(0.0, index=ts)
        override_delta.iloc[5] = 2.0

        app._publish_physics_sensors(forecast_df, override_delta=override_delta)

        physics_call = next(
            c for c in app.set_state.call_args_list if c.args[0] == "sensor.energy_forecast_physics_base_today"
        )
        adj_call = next(
            c for c in app.set_state.call_args_list if c.args[0] == "sensor.energy_forecast_ml_adjustment_today"
        )
        physics_hourly = physics_call.kwargs["attributes"]["hourly_kwh"]
        adj_hourly = adj_call.kwargs["attributes"]["hourly_kwh"]

        # The override delta lands in the physics baseline...
        assert physics_hourly[5] == pytest.approx(0.6 + 2.0)
        assert physics_hourly[4] == pytest.approx(0.6)  # untouched hour unaffected
        # ...not in ML adjustment: with the fix, hour 5's adjustment is
        # predicted_kwh - (physics + delta) = 3.0 - 2.6 = 0.4, identical to every
        # other hour's 1.0 - 0.6 = 0.4 — the override contributes nothing to "ML
        # Adjustment". Pre-fix, hour 5 would read 3.0 - 0.6 = 2.4, wrongly
        # attributing the whole override bump to "ML".
        assert adj_hourly[5] == pytest.approx(0.4)
        assert adj_hourly[4] == pytest.approx(0.4)

    def test_no_override_delta_argument_matches_pre_fix_behavior(self, monkeypatch):
        """Backward compatibility: callers that omit override_delta (or pass None —
        e.g. no physics model committed override this cycle) must see identical
        behavior to before this fix."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._publish_physics_sensors = EnergyForecast._publish_physics_sensors.__get__(app, type(app))
        app.set_state = MagicMock()
        ts = pd.date_range("2026-01-15", periods=48, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "predicted_kwh": [1.0] * 48})
        monkeypatch.setattr(app._physics_model, "predict_series", lambda *a, **kw: pd.Series([0.6] * 48, index=ts))

        app._publish_physics_sensors(forecast_df)  # no override_delta kwarg at all

        adj_call = next(
            c for c in app.set_state.call_args_list if c.args[0] == "sensor.energy_forecast_ml_adjustment_today"
        )
        assert adj_call.kwargs["attributes"]["hourly_kwh"][0] == pytest.approx(1.0 - 0.6)


class TestPhysicsMqttDiscoveryRegistration:
    """initialize() must register MQTT Discovery config for the physics sensors (follow-up to d12cfda).

    _mqtt_set_sensor()/_mqtt_publish_sensor_attributes() publish sensor *state*, but HA's MQTT
    integration only creates the entity once a retained discovery *config* payload has been
    published for it — mirrors the existing energy_forecast_thermal_pressure_net registration.
    """

    def _make_mqtt_app(self, physics_args: dict | None):
        from energy_forecast.energy_forecast import EnergyForecast

        args: dict = {"energy_sensor": "sensor.grid_import", "mqtt_discovery": True}
        if physics_args is not None:
            args["physics"] = physics_args
        app = _make_app(args)
        # Bind the real _mqtt_publish_all_discovery so initialize()'s MQTT-mode branch actually
        # runs its body (it stays a plain auto-mocked attribute otherwise, like _mqtt_publish_discovery).
        app._mqtt_publish_all_discovery = EnergyForecast._mqtt_publish_all_discovery.__get__(app, type(app))
        return app

    def test_physics_sensors_registered_when_physics_configured(self):
        app = self._make_mqtt_app({})
        app.initialize()
        registered_ids = [c.args[0] for c in app._mqtt_publish_discovery.call_args_list]
        assert "energy_forecast_physics_base_today" in registered_ids
        assert "energy_forecast_ml_adjustment_today" in registered_ids

    def test_physics_sensors_not_registered_when_physics_disabled(self):
        app = self._make_mqtt_app(None)
        app.initialize()
        registered_ids = [c.args[0] for c in app._mqtt_publish_discovery.call_args_list]
        assert "energy_forecast_physics_base_today" not in registered_ids
        assert "energy_forecast_ml_adjustment_today" not in registered_ids


class TestModelPhaseAttribute:
    def test_model_phase_absent_when_physics_disabled(self):
        app = _make_app({"energy_sensor": "sensor.grid_import"})
        app.initialize()
        assert app._model_phase_attr() is None

    def test_model_phase_is_phase1_by_default(self, tmp_path):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._ml_model._use_physics_residual = False
        assert app._model_phase_attr() == "phase1"

    def test_model_phase_is_phase2_when_flag_set_on_model(self, tmp_path):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._ml_model._use_physics_residual = True
        assert app._model_phase_attr() == "phase2"


class TestGetScenarioDhwSchedule:
    def test_dhw_schedule_forwarded_as_override(self, monkeypatch):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._get_scenario_cb = EnergyForecast._get_scenario_cb.__get__(app, type(app))
        app._cached_forecast_df = pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-15", periods=48, freq="1h"), "temp_c": [5.0] * 48}
        )
        app._cached_live_temp = 5.0
        app.fire_event = MagicMock()

        received = {}

        def _fake_predict_scenario(*args, **kwargs):
            received.update(kwargs)
            return pd.DataFrame(
                {
                    "timestamp": app._cached_forecast_df["timestamp"],
                    "predicted_kwh": [1.0] * 48,
                    "delta_kwh": [0.0] * 48,
                }
            )

        app._ml_model.predict_scenario = _fake_predict_scenario

        app._get_scenario_cb(
            "default",
            "energy_forecast",
            "get_scenario",
            {"schedule": {}, "dhw_schedule": {"legionella": ["2026-01-16", 10]}},
        )
        assert received.get("dhw_schedule_override") == {"legionella": ["2026-01-16", 10]}
        assert received.get("physics_model") is app._physics_model

    def test_no_dhw_schedule_key_forwards_none(self):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._get_scenario_cb = EnergyForecast._get_scenario_cb.__get__(app, type(app))
        app._cached_forecast_df = pd.DataFrame(
            {"timestamp": pd.date_range("2026-01-15", periods=48, freq="1h"), "temp_c": [5.0] * 48}
        )
        app._cached_live_temp = 5.0
        app.fire_event = MagicMock()

        received = {}

        def _fake_predict_scenario(*args, **kwargs):
            received.update(kwargs)
            return pd.DataFrame(
                {
                    "timestamp": app._cached_forecast_df["timestamp"],
                    "predicted_kwh": [1.0] * 48,
                    "delta_kwh": [0.0] * 48,
                }
            )

        app._ml_model.predict_scenario = _fake_predict_scenario

        app._get_scenario_cb("default", "energy_forecast", "get_scenario", {"schedule": {}})
        assert received.get("dhw_schedule_override") is None


class TestSetDhwScheduleService:
    def test_service_registered_when_physics_enabled(self):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        calls = [c.args[0] for c in app.register_service.call_args_list]
        assert "energy_forecast/set_dhw_schedule" in calls

    def test_commits_schedule_and_invalidates_cache(self, tmp_path):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
        app._cached_forecast_df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})

        with patch.object(app._physics_model, "commit_dhw_schedule") as mock_commit:
            app._set_dhw_schedule_cb(
                "default",
                "energy_forecast",
                "set_dhw_schedule",
                {"dhw_schedule": {"legionella": ["2026-01-16", 10]}},
            )
            mock_commit.assert_called_once_with({"legionella": ["2026-01-16", 10]})
        assert app._cached_forecast_df is None

    def test_missing_dhw_schedule_kwarg_logs_warning_no_crash(self, caplog):
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.logger = logging.getLogger("energy_forecast")  # real logger so caplog sees post-initialize() warnings
        app.initialize()
        app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            app._set_dhw_schedule_cb("default", "energy_forecast", "set_dhw_schedule", {})
        assert any("dhw_schedule" in r.message.lower() for r in caplog.records)

    def test_physics_disabled_logs_warning_no_crash(self, caplog):
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import"})
        app.logger = logging.getLogger("energy_forecast")  # real logger so caplog sees post-initialize() warnings
        app.initialize()
        app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            app._set_dhw_schedule_cb(
                "default", "energy_forecast", "set_dhw_schedule", {"dhw_schedule": {"legionella": ["2026-01-16", 10]}}
            )
        assert any("physics" in r.message.lower() for r in caplog.records)

    def test_comfort_boost_3elem_payload_forwards_unmodified(self, tmp_path):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
        app._cached_forecast_df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})

        app._set_dhw_schedule_cb(
            "default",
            "energy_forecast",
            "set_dhw_schedule",
            {"dhw_schedule": {"comfort_boost": ["2026-08-05", 14, 57.5]}},
        )
        assert app._physics_model._schedule["committed_override"] == {"comfort_boost": ["2026-08-05", 14, 57.5]}
        assert len(app._physics_model._schedule["override_history"]) == 1
        assert app._cached_forecast_df is None  # cache invalidated, same as legionella today

    def test_legionella_and_comfort_boost_same_day_neither_clobbers_other(self, tmp_path):
        """Cross-repo scenario from the spec's Testing section: same-day legionella-then-
        comfort_boost commit sequence through the real call_service→commit_dhw_schedule
        path, verifying override_history accumulates both."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
        app._cached_forecast_df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})

        app._set_dhw_schedule_cb(
            "default",
            "energy_forecast",
            "set_dhw_schedule",
            {"dhw_schedule": {"legionella": ["2026-08-04", 12]}},
        )
        app._set_dhw_schedule_cb(
            "default",
            "energy_forecast",
            "set_dhw_schedule",
            {"dhw_schedule": {"comfort_boost": ["2026-08-05", 14, 57.5]}},
        )
        committed = app._physics_model._schedule["committed_override"]
        assert committed == {"legionella": ["2026-08-04", 12], "comfort_boost": ["2026-08-05", 14, 57.5]}
        assert len(app._physics_model._schedule["override_history"]) == 2


class TestServingOverrideCorrection:
    """Task 11: EnergyForecast._update_sensors() must add
    compute_serving_override_delta() to the published forecast immediately after
    self._ml_model.predict() returns — direct regression test for the 2026-08-04
    live bug, where a committed DHW override never moved the published forecast
    because nothing outside the ML model's own (possibly zero-weighted) feature
    blending ever added the deterministic correction back in.

    Builds a real ThermalPhysicsModel (not app.initialize()'s physics wiring,
    which hardcodes apps/energy_forecast/models/ as the model dir — committing a
    real override there would write into the repo; see TestSetDhwScheduleService,
    which mocks commit_dhw_schedule for the same reason) and drives
    EnergyForecast._update_sensors() end-to-end via test_energy_forecast.py's
    _FakeUpdateSensors/_FakeMLModelForUpdateSensors harness, which already stubs
    every optional-feature attribute at its skip-guard default.
    """

    def _physics_config(self):
        return {
            "cop_sensor": None,
            "dhw_tank_temp_sensor": None,
            "heating_buffer_temp_sensor": None,
            "heating_curve_sensor": None,
            "cop_formula": {"a": 2.5, "b": 0.07},
            "dhw_tank_volume_l": 200,
            "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8,
            "heating_curve_points": [[-20, 55.5], [-5, 46.0], [5, 39.5], [20, 25.0]],
            "room_thermostats": [],
            "use_physics_residual": False,
        }

    def _run_update_sensors(self, tmp_path, monkeypatch, physics_model, forecast_timestamps):
        """Drives EnergyForecast._update_sensors() end-to-end and returns
        (predictions, stub): `predictions` is the exact DataFrame it hands to
        _aggregate() — i.e. the fully-corrected, about-to-be-published forecast —
        captured via a spy wrapped around stub._aggregate (which already
        delegates to the real implementation); `stub` is the driving instance,
        whose `.set_state_calls` list (populated by _FakeUpdateSensors.set_state)
        callers can inspect for the physics/ml-adjustment diagnostic sensors
        _publish_physics_sensors() also publishes this cycle."""
        from energy_forecast import energy_forecast as ef_mod
        from energy_forecast import ha_data as ha_data_mod
        from energy_forecast import weather as weather_mod
        from energy_forecast.energy_forecast import EnergyForecast

        from tests.test_energy_forecast import _FakeUpdateSensors

        forecast_df = pd.DataFrame({"timestamp": forecast_timestamps, "temp_c": [5.0] * len(forecast_timestamps)})

        monkeypatch.setattr(weather_mod, "fetch_forecast", lambda *a, **kw: forecast_df.copy())
        monkeypatch.setattr(
            ha_data_mod, "fetch_recent_energy", lambda *a, **kw: pd.DataFrame(columns=["timestamp", "gross_kwh"])
        )
        monkeypatch.setattr(
            ha_data_mod,
            "split_ev_charging",
            lambda df, *a, **kw: (df, pd.DataFrame(columns=["timestamp", "gross_kwh"])),
        )
        monkeypatch.setattr(ha_data_mod, "fetch_recent_energy_15m", lambda *a, **kw: None)
        monkeypatch.setattr(ef_mod, "PRED_HISTORY_PATH", tmp_path / "pred_history.json")

        stub = _FakeUpdateSensors(tmp_path / "energy_history.csv")
        stub._physics_model = physics_model
        stub._physics_config = physics_model._config
        stub._room_thermostats = []
        stub._physics_climate_dfs = {}
        stub._physics_dhw_tank_df = None
        stub._physics_cop_df = None
        stub._ml_model.predict = lambda *a, **kw: pd.DataFrame(
            {"timestamp": forecast_timestamps, "predicted_kwh": [1.0] * len(forecast_timestamps)}
        )

        captured: dict = {}
        real_aggregate = stub._aggregate

        def _capture_aggregate(predictions_df, *a, **kw):
            captured["predictions"] = predictions_df.copy()
            return real_aggregate(predictions_df, *a, **kw)

        stub._aggregate = _capture_aggregate

        EnergyForecast._update_sensors(stub)
        return captured["predictions"], stub

    def test_committed_override_moves_published_forecast_by_full_delta(self, tmp_path, monkeypatch):
        from energy_forecast.physics import ThermalPhysicsModel

        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        forecast_timestamps = pd.date_range(start=now, periods=48, freq="1h")
        override_ts = forecast_timestamps[5]

        physics_with_override = ThermalPhysicsModel(tmp_path / "physics_models_override", self._physics_config())
        physics_with_override.commit_dhw_schedule({"legionella": [override_ts.strftime("%Y-%m-%d"), override_ts.hour]})
        physics_no_override = ThermalPhysicsModel(tmp_path / "physics_models_baseline", self._physics_config())

        # The expected delta is computed via the exact same public entry point
        # _update_sensors() is supposed to call, with the exact same inputs it
        # is supposed to build (t_ambient from forecast_df["temp_c"], initial
        # tank state from _initial_t_tank_for_window with no dhw_recent data —
        # this stub leaves _dhw_buffer_sensor unset).
        initial_t_tank = physics_with_override._initial_t_tank_for_window(None, forecast_timestamps)
        expected_delta = physics_with_override.compute_serving_override_delta(
            forecast_timestamps, pd.Series(5.0, index=forecast_timestamps), initial_t_tank
        )
        assert (expected_delta != 0.0).any(), "test setup sanity: override must actually diverge the DHW trajectory"

        predictions_with_override, _ = self._run_update_sensors(
            tmp_path, monkeypatch, physics_with_override, forecast_timestamps
        )
        predictions_baseline, _ = self._run_update_sensors(
            tmp_path, monkeypatch, physics_no_override, forecast_timestamps
        )

        actual_delta = (
            predictions_with_override.set_index("timestamp")["predicted_kwh"]
            - predictions_baseline.set_index("timestamp")["predicted_kwh"]
        )
        pd.testing.assert_series_equal(
            actual_delta, expected_delta.rename("predicted_kwh"), check_names=False, check_freq=False, atol=1e-9
        )
        # Explicit multi-hour-tail check at the override hour and beyond, not just
        # a single point — the whole point of the fix is the carryover tail moves too.
        assert actual_delta.loc[override_ts] != pytest.approx(0.0)

    def test_no_committed_override_leaves_published_forecast_unchanged(self, tmp_path, monkeypatch):
        """Zero-override-degradation: with nothing committed,
        compute_serving_override_delta() returns an all-zero series (Task 7/13),
        so this addition must be a complete no-op end-to-end."""
        from energy_forecast.physics import ThermalPhysicsModel

        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        forecast_timestamps = pd.date_range(start=now, periods=48, freq="1h")

        physics_model = ThermalPhysicsModel(tmp_path / "physics_models_noop", self._physics_config())
        assert physics_model.schedule.get("committed_override") is None

        predictions, _ = self._run_update_sensors(tmp_path, monkeypatch, physics_model, forecast_timestamps)

        assert (predictions.set_index("timestamp")["predicted_kwh"] == 1.0).all()

    def test_committed_override_not_misattributed_to_ml_adjustment_sensor(self, tmp_path, monkeypatch):
        """Review-fix regression, end-to-end through _update_sensors(): the
        published sensor.energy_forecast_ml_adjustment_today diagnostic must not
        silently absorb the DHW override delta and attribute it to "ML
        Adjustment" instead of "Physics Base". Companion to the direct unit tests
        in TestPhysicsSensorsOverrideDeltaAttribution — this one instead proves
        _update_sensors() actually threads _override_delta through to
        _publish_physics_sensors(), not just that the method behaves correctly
        in isolation."""
        from energy_forecast.physics import ThermalPhysicsModel

        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        forecast_timestamps = pd.date_range(start=now, periods=48, freq="1h")
        override_ts = forecast_timestamps[5]

        physics_with_override = ThermalPhysicsModel(tmp_path / "physics_models_attr_override", self._physics_config())
        physics_with_override.commit_dhw_schedule({"legionella": [override_ts.strftime("%Y-%m-%d"), override_ts.hour]})
        physics_no_override = ThermalPhysicsModel(tmp_path / "physics_models_attr_baseline", self._physics_config())

        _, stub_with_override = self._run_update_sensors(
            tmp_path, monkeypatch, physics_with_override, forecast_timestamps
        )
        _, stub_baseline = self._run_update_sensors(tmp_path, monkeypatch, physics_no_override, forecast_timestamps)

        def _hourly_kwh(stub, entity_id):
            call = next(c for c in stub.set_state_calls if c["entity_id"] == entity_id)
            return call["attributes"]["hourly_kwh"]

        physics_with = _hourly_kwh(stub_with_override, "sensor.energy_forecast_physics_base_today")
        physics_base = _hourly_kwh(stub_baseline, "sensor.energy_forecast_physics_base_today")
        ml_adj_with = _hourly_kwh(stub_with_override, "sensor.energy_forecast_ml_adjustment_today")
        ml_adj_base = _hourly_kwh(stub_baseline, "sensor.energy_forecast_ml_adjustment_today")

        # Sanity: the two physics_model instances (same fresh config, no
        # calibration difference) must actually differ at the override hour —
        # otherwise this test can't distinguish "fixed" from "coincidentally
        # equal".
        assert physics_with[5] != pytest.approx(physics_base[5])

        # The fake ML model always predicts a flat 1.0 kWh/h regardless of the
        # committed override (Task 11 adds the correction after predict()
        # returns) — so "ML Adjustment" must read identically whether or not an
        # override is committed. Before this fix, ml_adj_with[5] would be
        # inflated by the override delta relative to ml_adj_base[5].
        assert ml_adj_with == pytest.approx(ml_adj_base, abs=1e-9)
