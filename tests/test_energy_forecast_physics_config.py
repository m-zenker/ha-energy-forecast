"""Tests for physics: config ingest in energy_forecast.py initialize()."""

from __future__ import annotations

from unittest.mock import MagicMock

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
