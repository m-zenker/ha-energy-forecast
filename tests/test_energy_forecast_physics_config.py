"""Tests for physics: config ingest in energy_forecast.py initialize()."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
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
    app._sub_sensor_prefix = EnergyForecast._sub_sensor_prefix.__get__(app, type(app))
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
                    "dhw_tank_temp_sensor": "sensor.kermi_dhw_buffer_temp",
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
        assert "internal_gains_fraction" not in app._physics_config
        assert "use_physics_residual" not in app._physics_config
        assert "cop_sensor" not in app._physics_config


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
        assert fetch_generic.call_count == 2  # dhw_tank, heating_buffer
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

        assert fetch_recent_generic.call_count == 2  # dhw_tank, heating_buffer
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

        app._fetch_physics_sensor_histories(recent_only=True)

        assert app._physics_heating_buffer_df is not None and not app._physics_heating_buffer_df.empty
        assert app._physics_dhw_tank_df is not None and not app._physics_dhw_tank_df.empty

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

    def test_sub_sensor_prefix_key_matches_heating_sub_meter_lookup(self):
        app = _make_app(
            {
                "energy_sensor": "sensor.grid_import",
                "physics": {"heating_sub_meter_sensor": "sensor.em_kermi_bridge_kermi_electricity_heating"},
            }
        )
        app.initialize()
        prefix = app._sub_sensor_prefix(app._physics_config["heating_sub_meter_sensor"])
        assert prefix == "sub_em_kermi_bridge_kermi_electricity_heating"

    def test_calibrate_via_recalibrate_physics_matches_scheduled_retrain(self, tmp_path):
        """The on-demand recalibrate_physics path must forward the same
        heating_active/ev/away data _retrain() cached for it —
        not silently resolve a weaker tier than the scheduled path would."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._recalibrate_physics_cb = EnergyForecast._recalibrate_physics_cb.__get__(app, type(app))
        app._fetch_physics_sensor_histories = MagicMock()
        app._physics_climate_dfs = {}
        app._physics_dhw_tank_df = None
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
        heating_active_sentinel = pd.DataFrame({"timestamp": [], "heating_active": []})
        ev_sentinel = pd.DataFrame({"timestamp": []})
        away_sentinel = pd.DataFrame({"timestamp": [], "is_away": []})
        app._cached_heating_active_df = heating_active_sentinel
        app._cached_ev_df = ev_sentinel
        app._cached_away_df = away_sentinel

        with patch.object(app._physics_model, "calibrate") as mock_calibrate:
            app._recalibrate_physics_cb("default", "energy_forecast", "recalibrate_physics", {})
            mock_calibrate.assert_called_once()
            kwargs = mock_calibrate.call_args.kwargs
            assert kwargs["heating_active_df"] is heating_active_sentinel
            assert kwargs["ev_df"] is ev_sentinel
            assert kwargs["away_df"] is away_sentinel

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


class TestGetScenarioDhwSchedule:
    def test_dhw_schedule_no_longer_forwarded_to_the_model(self, monkeypatch):
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
        assert "dhw_schedule_override" not in received
        assert "physics_model" not in received

    def test_no_dhw_schedule_key_adds_no_override_kwarg(self):
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
        assert "dhw_schedule_override" not in received


class TestSetDhwScheduleService:
    def test_service_registered_when_physics_enabled(self):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        calls = [c.args[0] for c in app.register_service.call_args_list]
        assert "energy_forecast/set_dhw_schedule" in calls

    def test_commit_keeps_scenario_cache(self, tmp_path):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
        df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})
        app._cached_forecast_df = df

        with patch.object(app._physics_model, "commit_dhw_schedule") as mock_commit:
            app._set_dhw_schedule_cb(
                "default",
                "energy_forecast",
                "set_dhw_schedule",
                {"dhw_schedule": {"legionella": ["2026-01-16", 10]}},
            )
            mock_commit.assert_called_once_with({"legionella": ["2026-01-16", 10]})
        assert app._cached_forecast_df is df

    def test_clear_commit_keeps_scenario_cache(self, tmp_path):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        self._isolate_physics_model_dir(app, tmp_path)
        app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
        df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})
        app._cached_forecast_df = df

        app._set_dhw_schedule_cb(
            "default", "energy_forecast", "set_dhw_schedule", {"dhw_schedule": {"legionella": None}}
        )
        assert app._cached_forecast_df is df

    def test_failed_commit_keeps_scenario_cache(self, tmp_path):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
        df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})
        app._cached_forecast_df = df

        with patch.object(app._physics_model, "commit_dhw_schedule", side_effect=RuntimeError("disk full")):
            app._set_dhw_schedule_cb(
                "default",
                "energy_forecast",
                "set_dhw_schedule",
                {"dhw_schedule": {"legionella": ["2026-01-16", 10]}},
            )
        assert app._cached_forecast_df is df

    def test_get_scenario_answers_after_commit(self, tmp_path):
        """Live 2026-11-07: a 05:04 commit left get_scenario silent until the 06:01 cycle."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        self._isolate_physics_model_dir(app, tmp_path)
        app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
        app._get_scenario_cb = EnergyForecast._get_scenario_cb.__get__(app, type(app))
        app._cached_forecast_df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})
        app._ml_model = MagicMock()
        app._ml_model.predict_scenario.return_value = pd.DataFrame(
            {"timestamp": [1], "predicted_kwh": [1.0], "delta_kwh": [0.5]}
        )
        app.fire_event = MagicMock()

        app._set_dhw_schedule_cb(
            "default",
            "energy_forecast",
            "set_dhw_schedule",
            {"dhw_schedule": {"comfort_boost": ["2026-11-08", 12, 54.0]}},
        )
        app._get_scenario_cb(
            "default",
            "energy_forecast",
            "get_scenario",
            {"dhw_schedule": {"legionella": ["2026-11-07", 12]}, "request_id": "r1"},
        )

        app.fire_event.assert_called_once()
        assert app.fire_event.call_args.args[0] == "energy_forecast_scenario_result"
        assert app.fire_event.call_args.kwargs["request_id"] == "r1"

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

    @staticmethod
    def _isolate_physics_model_dir(app, tmp_path):
        """initialize() hardcodes the physics model dir to apps/energy_forecast/models/ —
        the REAL directory on disk. Tests that drive the unmocked commit_dhw_schedule
        write path must not persist a committed override there: it leaves stale state
        that the next run of the very same test reads back, making the suite
        non-idempotent (final-review #6). Swap in a tmp_path-backed model, mirroring
        TestServingOverrideCorrection's approach of building a real ThermalPhysicsModel
        against tmp_path rather than relying on initialize()'s wiring."""
        from energy_forecast.physics import ThermalPhysicsModel

        app._physics_model = ThermalPhysicsModel(tmp_path / "physics_models", app._physics_config)
        return app

    def test_comfort_boost_3elem_payload_forwards_unmodified(self, tmp_path):
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        self._isolate_physics_model_dir(app, tmp_path)
        app._set_dhw_schedule_cb = EnergyForecast._set_dhw_schedule_cb.__get__(app, type(app))
        df = pd.DataFrame({"timestamp": [1], "temp_c": [5.0]})
        app._cached_forecast_df = df

        app._set_dhw_schedule_cb(
            "default",
            "energy_forecast",
            "set_dhw_schedule",
            {"dhw_schedule": {"comfort_boost": ["2026-08-05", 14, 57.5]}},
        )
        assert app._physics_model._schedule["committed_override"] == {"comfort_boost": ["2026-08-05", 14, 57.5]}
        assert len(app._physics_model._schedule["override_history"]) == 1
        assert app._cached_forecast_df is df

    def test_legionella_and_comfort_boost_same_day_neither_clobbers_other(self, tmp_path):
        """Cross-repo scenario from the spec's Testing section: same-day legionella-then-
        comfort_boost commit sequence through the real call_service→commit_dhw_schedule
        path, verifying override_history accumulates both."""
        from energy_forecast.energy_forecast import EnergyForecast

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        self._isolate_physics_model_dir(app, tmp_path)
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
            "dhw_tank_temp_sensor": None,
            "heating_buffer_temp_sensor": None,
            "cop_formula": {"a": 2.5, "b": 0.07},
            "dhw_tank_volume_l": 200,
            "dhw_power_w": 4000,
            "heating_curve_points": [[-20, 55.5], [-5, 46.0], [5, 39.5], [20, 25.0]],
            "room_thermostats": [],
        }

    def _run_update_sensors(
        self,
        tmp_path,
        monkeypatch,
        physics_model,
        forecast_timestamps,
        outdoor_temp_c=5.0,
        climate_recent=None,
        base_kwh=1.0,
        prediction_timestamps=None,
        dhw_recent=None,
    ):
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

        forecast_df = pd.DataFrame(
            {"timestamp": forecast_timestamps, "temp_c": [outdoor_temp_c] * len(forecast_timestamps)}
        )

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
        if climate_recent:
            # Drive _update_sensors()'s real climate-fetch loop so it builds the same
            # `climate_recent` dict production does, which the override correction then
            # projects into an indoor-temperature series (final-review #2).
            stub._climate_entities = list(climate_recent)
            stub._climate_cache_path = lambda eid, _tmp=tmp_path: _tmp / f"climate_{eid}.csv"
            monkeypatch.setattr(
                ha_data_mod, "fetch_recent_climate", lambda _self, eid, *a, **kw: climate_recent[eid].copy()
            )
        if dhw_recent is not None:
            stub._dhw_buffer_sensor = "sensor.tank"
            # _FakeUpdateSensors has no cache-path helper; _update_sensors() calls it outside its try block.
            stub._generic_sensor_cache_path = lambda eid, prefix="x", _t=tmp_path: _t / f"{prefix}.csv"
            monkeypatch.setattr(ha_data_mod, "fetch_recent_generic_sensor", lambda *a, **kw: dhw_recent.copy())
        stub._physics_model = physics_model
        stub._physics_config = physics_model._config
        stub._room_thermostats = []
        stub._physics_climate_dfs = {}
        stub._physics_dhw_tank_df = None
        # Real predict() always returns 48 h from now, even when forecast_df carries a past tail.
        predicted_ts = forecast_timestamps if prediction_timestamps is None else prediction_timestamps
        stub._ml_model.predict = lambda *a, **kw: pd.DataFrame(
            {"timestamp": predicted_ts, "predicted_kwh": [base_kwh] * len(predicted_ts)}
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
        # is supposed to build: t_ambient from _t_indoor_for_window (INDOOR air —
        # final-review #2; with no climate_recent in this stub that is the
        # `setpoint_on or DEFAULT_AMBIENT_C` fallback, NOT forecast_df["temp_c"]),
        # and initial tank state from _initial_t_tank_for_window with no
        # dhw_recent data — this stub leaves _dhw_buffer_sensor unset.
        t_ambient = physics_with_override._t_indoor_for_window(
            None, forecast_timestamps, pd.Series(5.0, index=forecast_timestamps)
        )
        initial_t_tank = physics_with_override._initial_t_tank_for_window(None, forecast_timestamps)
        expected_delta = physics_with_override.compute_serving_override_delta(
            forecast_timestamps, t_ambient, initial_t_tank
        )
        assert (expected_delta != 0.0).any(), "test setup sanity: override must actually diverge the DHW trajectory"

        # base_kwh=5.0, not the harness default 1.0: the correction is now floored at
        # zero (final-review #4), and a single skipped reheat cycle is a ~1.03 kWh
        # NEGATIVE delta — larger than a 1.0 kWh/h baseline, so the floor would clip it
        # and this "moves by the FULL delta" assertion would be measuring the floor
        # instead. 5.0 keeps every corrected hour comfortably positive.
        predictions_with_override, _ = self._run_update_sensors(
            tmp_path, monkeypatch, physics_with_override, forecast_timestamps, base_kwh=5.0
        )
        predictions_baseline, _ = self._run_update_sensors(
            tmp_path, monkeypatch, physics_no_override, forecast_timestamps, base_kwh=5.0
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

    def test_serving_delta_uses_indoor_temperature_not_outdoor(self, tmp_path, monkeypatch):
        """Final-review #2: the DHW ODE's `t_ambient` is the air around the TANK, i.e.
        indoor temperature — exactly what predict_series feeds physics_kwh. The serving
        correction used to pass raw outdoor `forecast_df["temp_c"]` instead, so on a cold
        day it modelled a tank standing outside: inflated standing losses, wrong delta.

        Cold outdoor (-5 degC) + warm indoor (22 degC via climate_recent) makes the two
        derivations diverge sharply, so the published delta can only match one of them.
        """
        from energy_forecast.energy_forecast import _strip_tz
        from energy_forecast.physics import ThermalPhysicsModel

        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        forecast_timestamps = pd.date_range(start=now, periods=48, freq="1h")
        override_ts = forecast_timestamps[5]

        climate_df = pd.DataFrame(
            {
                "timestamp": pd.date_range(end=now, periods=6, freq="1h"),
                "current_temp": [22.0] * 6,
                "setpoint": [22.0] * 6,
            }
        )
        climate_recent = {"climate.living": climate_df}

        pm = ThermalPhysicsModel(tmp_path / "physics_models_indoor", self._physics_config())
        pm.commit_dhw_schedule({"legionella": [override_ts.strftime("%Y-%m-%d"), override_ts.hour]})
        pm_baseline = ThermalPhysicsModel(tmp_path / "physics_models_indoor_base", self._physics_config())

        t_outdoor = pd.Series(-5.0, index=forecast_timestamps)
        stripped = {eid: _strip_tz(df.copy(), "Europe/Zurich") for eid, df in climate_recent.items()}
        t_indoor = pm._t_indoor_for_window(stripped, forecast_timestamps, t_outdoor, room_areas=None)
        assert t_indoor.iloc[0] == pytest.approx(22.0), "sanity: projection must start from the warm indoor reading"

        initial_t_tank = pm._initial_t_tank_for_window(None, forecast_timestamps)
        expected_indoor_delta = pm.compute_serving_override_delta(forecast_timestamps, t_indoor, initial_t_tank)
        outdoor_delta = pm.compute_serving_override_delta(forecast_timestamps, t_outdoor, initial_t_tank)
        assert (expected_indoor_delta != 0.0).any(), "sanity: the override must move the forecast at all"
        assert not np.allclose(expected_indoor_delta.to_numpy(), outdoor_delta.to_numpy()), (
            "sanity: indoor- and outdoor-ambient deltas must differ for this test to discriminate"
        )

        predictions_with, _ = self._run_update_sensors(
            tmp_path,
            monkeypatch,
            pm,
            forecast_timestamps,
            outdoor_temp_c=-5.0,
            climate_recent=climate_recent,
            base_kwh=5.0,  # keep the final-review #4 zero-floor from clipping a negative correction hour
        )
        predictions_base, _ = self._run_update_sensors(
            tmp_path,
            monkeypatch,
            pm_baseline,
            forecast_timestamps,
            outdoor_temp_c=-5.0,
            climate_recent=climate_recent,
            base_kwh=5.0,  # keep the final-review #4 zero-floor from clipping a negative correction hour
        )
        actual_delta = (
            predictions_with.set_index("timestamp")["predicted_kwh"]
            - predictions_base.set_index("timestamp")["predicted_kwh"]
        )

        pd.testing.assert_series_equal(
            actual_delta,
            expected_indoor_delta.rename("predicted_kwh"),
            check_names=False,
            check_freq=False,
            atol=1e-9,
        )

    def test_large_negative_delta_never_publishes_a_negative_forecast(self, tmp_path, monkeypatch):
        """Final-review #4: the override correction is a SIGNED delta added to an
        already-final forecast, with nothing between it and the published sensor. A
        large enough negative delta would publish a visibly negative hourly forecast.
        Floor it at zero, matching model.py's existing `np.maximum(0, preds)` precedent.
        """
        from energy_forecast.physics import ThermalPhysicsModel

        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        forecast_timestamps = pd.date_range(start=now, periods=48, freq="1h")

        pm = ThermalPhysicsModel(tmp_path / "physics_models_negative", self._physics_config())
        # Synthetic, deliberately unphysical delta: the point is that the publish path
        # itself must be safe, whatever the correction hands it.
        monkeypatch.setattr(
            pm,
            "compute_serving_override_delta",
            lambda ts, *a, **kw: pd.Series(-50.0, index=ts),
        )

        predictions, _ = self._run_update_sensors(tmp_path, monkeypatch, pm, forecast_timestamps)

        # The fake ML model predicts a flat 1.0 kWh/h, so an unfloored -50.0 delta would
        # publish -49.0 on every hour.
        assert (predictions["predicted_kwh"] >= 0.0).all(), (
            f"negative forecast published: min={predictions['predicted_kwh'].min()}"
        )
        assert (predictions["predicted_kwh"] == 0.0).all()

    # ── forecast_df carries a past tail (fetch_open_meteo past_days=3) ──────────────────────

    def _tail_and_future(self, now):
        tail = pd.date_range(now - pd.Timedelta(hours=72), now + pd.Timedelta(hours=47), freq="1h")
        return tail, tail[tail >= now]

    def _tail_physics_model(self, tmp_path, name):
        """Live tank size and DHW calibration: the leak reproduces with them on any start hour."""
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / name, {**self._physics_config(), "dhw_tank_volume_l": 400})
        pm._calib.update(Q_dhw_daily=2.79, UA_dhw=3.18)
        return pm

    def _diverging_past_override(self, pm, tail, future, now):
        """A past hour whose override leaks into `future` when simulated over the whole tail.

        Searched, not fixed: whether a given offset leaks depends on the hour-of-day draw
        profile, and the harness runs on the real clock.
        """
        t_ambient = pm._t_indoor_for_window(None, tail, pd.Series(5.0, index=tail))
        initial_t_tank = pm._initial_t_tank_for_window(None, tail)
        try:
            for hours_back in range(1, 72):
                candidate = now - pd.Timedelta(hours=hours_back)
                pm._schedule["committed_override"] = {"legionella": [candidate.strftime("%Y-%m-%d"), candidate.hour]}
                delta = pm.compute_serving_override_delta(tail, t_ambient, initial_t_tank)
                if delta.reindex(future).abs().max() > 0.01:
                    return candidate
        finally:
            pm._schedule["committed_override"] = None
        pytest.fail("no past override leaks into the future window: the test no longer exercises the bug")

    def _spy_on_serving_delta(self, pm, monkeypatch):
        spy: dict = {}
        real = pm.compute_serving_override_delta

        def wrapped(timestamps, t_ambient, initial_t_tank):
            spy.update(timestamps=timestamps, t_ambient=t_ambient, initial_t_tank=initial_t_tank)
            return real(timestamps, t_ambient, initial_t_tank)

        monkeypatch.setattr(pm, "compute_serving_override_delta", wrapped)
        return spy

    @staticmethod
    def _climate_recent(now):
        return {
            "climate.living": pd.DataFrame(
                {
                    "timestamp": pd.date_range(end=now, periods=6, freq="1h"),
                    "current_temp": [22.0] * 6,
                    "setpoint": [22.0] * 6,
                }
            )
        }

    @staticmethod
    def _commit(pm, ts):
        pm.commit_dhw_schedule({"legionella": [ts.strftime("%Y-%m-%d"), ts.hour]})

    def test_expired_override_in_past_tail_does_not_move_forecast(self, tmp_path, monkeypatch):
        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        tail, future = self._tail_and_future(now)
        pm = self._tail_physics_model(tmp_path, "physics_models_expired")
        self._commit(pm, self._diverging_past_override(pm, tail, future, now))

        predictions, _ = self._run_update_sensors(
            tmp_path, monkeypatch, pm, tail, prediction_timestamps=future, base_kwh=5.0
        )

        assert (predictions.set_index("timestamp")["predicted_kwh"] == 5.0).all()

    def test_future_override_with_past_tail_moves_forecast_by_future_window_delta(self, tmp_path, monkeypatch):
        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        tail, future = self._tail_and_future(now)
        pm = self._tail_physics_model(tmp_path, "physics_models_future")
        self._commit(pm, future[5])
        expected = pm.compute_serving_override_delta(
            future,
            pm._t_indoor_for_window(None, future, pd.Series(5.0, index=future)),
            pm._initial_t_tank_for_window(None, future),
        )
        assert (expected != 0.0).any(), "test setup sanity: the override must diverge the DHW trajectory"

        predictions, _ = self._run_update_sensors(
            tmp_path, monkeypatch, pm, tail, prediction_timestamps=future, base_kwh=5.0
        )

        actual = predictions.set_index("timestamp")["predicted_kwh"] - 5.0
        pd.testing.assert_series_equal(
            actual, expected.rename("predicted_kwh"), check_names=False, check_freq=False, atol=1e-9
        )

    def test_override_delta_window_starts_at_current_hour(self, tmp_path, monkeypatch):
        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        tail, future = self._tail_and_future(now)
        pm = self._tail_physics_model(tmp_path, "physics_models_window")
        self._commit(pm, future[5])
        spy = self._spy_on_serving_delta(pm, monkeypatch)

        self._run_update_sensors(tmp_path, monkeypatch, pm, tail, prediction_timestamps=future, base_kwh=5.0)

        assert spy["timestamps"][0] == now

    def test_override_delta_seeds_from_latest_tank_reading(self, tmp_path, monkeypatch):
        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        tail, future = self._tail_and_future(now)
        past = tail[tail < now]
        dhw_recent = pd.DataFrame({"timestamp": past, "buffer_temp": [60.0] * (len(past) - 1) + [47.0]})
        pm = self._tail_physics_model(tmp_path, "physics_models_seed")
        self._commit(pm, future[5])
        spy = self._spy_on_serving_delta(pm, monkeypatch)

        self._run_update_sensors(
            tmp_path, monkeypatch, pm, tail, prediction_timestamps=future, base_kwh=5.0, dhw_recent=dhw_recent
        )

        assert spy["initial_t_tank"] == pytest.approx(47.0)

    def test_override_delta_ambient_matches_future_window(self, tmp_path, monkeypatch):
        from energy_forecast.energy_forecast import _strip_tz

        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        tail, future = self._tail_and_future(now)
        climate_recent = self._climate_recent(now)
        pm = self._tail_physics_model(tmp_path, "physics_models_ambient")
        self._commit(pm, future[5])
        spy = self._spy_on_serving_delta(pm, monkeypatch)
        stripped = {eid: _strip_tz(df.copy(), "Europe/Zurich") for eid, df in climate_recent.items()}
        expected = pm._t_indoor_for_window(stripped, future, pd.Series(5.0, index=future), room_areas=None)

        self._run_update_sensors(
            tmp_path,
            monkeypatch,
            pm,
            tail,
            climate_recent=climate_recent,
            prediction_timestamps=future,
            base_kwh=5.0,
        )

        pd.testing.assert_series_equal(spy["t_ambient"], expected, check_names=False, check_freq=False)

    def test_override_at_current_hour_still_applies(self, tmp_path, monkeypatch):
        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        tail, future = self._tail_and_future(now)
        pm = self._tail_physics_model(tmp_path, "physics_models_current")
        self._commit(pm, now)

        predictions, _ = self._run_update_sensors(
            tmp_path, monkeypatch, pm, tail, prediction_timestamps=future, base_kwh=5.0
        )

        assert predictions.set_index("timestamp")["predicted_kwh"].loc[now] != pytest.approx(5.0)

    def test_forecast_without_future_rows_leaves_forecast_unchanged(self, tmp_path, monkeypatch):
        """climate_recent is required: without any rows, the indoor projection is what raises."""
        now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        tail, future = self._tail_and_future(now)
        pm = self._tail_physics_model(tmp_path, "physics_models_no_future")
        self._commit(pm, now - pd.Timedelta(hours=5))

        predictions, _ = self._run_update_sensors(
            tmp_path,
            monkeypatch,
            pm,
            tail[tail < now],
            climate_recent=self._climate_recent(now),
            prediction_timestamps=future,
            base_kwh=5.0,
        )

        assert (predictions.set_index("timestamp")["predicted_kwh"] == 5.0).all()

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
