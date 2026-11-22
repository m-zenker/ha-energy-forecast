"""Physics heating removal, task 1: model.py serving API, legacy physics_kwh models, backtest kwarg stripping."""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd
import pytest
from energy_forecast import model as model_mod
from energy_forecast.model import EnergyForecastModel

from backtest.capture_frame import CaptureFrame
from backtest.env import strip_legacy_train_kwargs
from tests.test_model import _make_trained_model

SERVE = ("predict", "predict_intervals", "_prepare_prediction_X", "predict_scenario", "shap_summary")


def _physics_config() -> dict:
    return {
        "heating_system_active_entity": None,
        "indoor_temp_entities": {},
        "room_areas": {},
    }


def test_serving_api_has_no_physics_args():
    for name in SERVE:
        params = inspect.signature(getattr(EnergyForecastModel, name)).parameters
        assert "physics_model" not in params and "dhw_schedule_override" not in params, name


def test_train_keeps_physics_model_only():
    params = inspect.signature(EnergyForecastModel.train).parameters
    assert "physics_model" in params
    assert "use_physics_residual" not in params and "heating_sub_meter_df" not in params


def test_trained_model_has_no_physics_kwh(tmp_path):
    from energy_forecast.physics import ThermalPhysicsModel

    pm = ThermalPhysicsModel(tmp_path / "physics_models", _physics_config())
    model, _ = _make_trained_model(tmp_path / "model", physics_model=pm)
    assert "physics_kwh" not in model.feature_cols
    assert model._log_transform is True
    assert not hasattr(model, "physics_phase1_diagnostic") and not hasattr(model, "_use_physics_residual")


def test_legacy_physics_kwh_model_predicts_without_warning(tmp_path, monkeypatch, caplog):
    real_engineer = model_mod._engineer_features

    def _with_const(*args, **kwargs):
        df = real_engineer(*args, **kwargs)
        df["physics_kwh"] = 0.144
        return df

    with monkeypatch.context() as mp:
        mp.setattr(model_mod, "_FEATURES_BASE", [*model_mod._FEATURES_BASE, "physics_kwh"])
        mp.setattr(model_mod, "_engineer_features", _with_const)
        model, forecast = _make_trained_model(tmp_path / "model")
    assert "physics_kwh" in model.feature_cols

    with caplog.at_level(logging.DEBUG):
        _, X = model._prepare_prediction_X(forecast, live_temp=5.0, recent_actuals=None)
        preds = model.predict(forecast, live_temp=5.0)
    assert (X["physics_kwh"] == 0.0).all()
    X2 = X.copy()
    X2["physics_kwh"] = 0.144
    raw0 = model.model.predict(X[model.feature_cols])
    raw1 = model.model.predict(X2[model.feature_cols])
    np.testing.assert_allclose(raw0, raw1)
    assert len(preds) > 0
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING and "physics_kwh" in r.getMessage()]


def test_strip_legacy_train_kwargs():
    src = {"use_physics_residual": False, "heating_sub_meter_df": None, "a": 1}
    out = strip_legacy_train_kwargs(src)
    assert out == {"a": 1}
    assert "use_physics_residual" in src  # input untouched


def test_capture_frame_strips_legacy_train_kwargs():
    from tests.test_backtest_capture_frame import _payload

    payload = _payload()
    payload["train_kwargs"]["use_physics_residual"] = False
    payload["train_kwargs"]["heating_sub_meter_df"] = None
    frame = CaptureFrame(payload)
    kw = frame.train_kwargs(pd.Timestamp("2026-03-01 18:00"))
    assert "use_physics_residual" not in kw and "heating_sub_meter_df" not in kw


# ── Task 2: physics.py heating removal ────────────────────────────────────────

GONE = (
    "predict_series",
    "predict_training_series",
    "_space_heating_kwh",
    "_cop_series",
    "_calibrate_ua_eff",
    "_calibrate_solar_gain_area",
    "_resolve_heating_eligibility",
    "is_cold_start_gated",
    "check_zone_boundary",
)
_PHYSICS_SENSOR_IDS = ("physics_base_today", "ml_adjustment_today")


def test_physics_has_no_heating_api():
    from energy_forecast.physics import ThermalPhysicsModel

    for name in GONE:
        assert not hasattr(ThermalPhysicsModel, name), name


def test_calibrate_has_no_heating_inputs():
    from energy_forecast.physics import ThermalPhysicsModel

    params = inspect.signature(ThermalPhysicsModel.calibrate).parameters
    assert "climate_dfs" not in params and "heating_sub_meter_df" not in params


def test_default_calibration_has_no_heating_keys():
    from energy_forecast.physics import _default_calibration

    keys = set(_default_calibration())
    assert keys.isdisjoint(
        {
            "UA_eff",
            "solar_gain_area",
            "n_calibration_windows_ua_eff",
            "q_base_el_calibrated",
            "room_thermostats_at_calibration",
        }
    )
    assert {"Q_base_el", "Q_dhw_daily", "UA_dhw"} <= keys


@pytest.fixture
def _restore_module_loggers():
    """app.initialize() on a MagicMock app replaces module _LOGGERs with mocks; restore them (same as
    tests/test_energy_forecast_physics_config.py's autouse fixture) so later caplog tests are unaffected."""
    from energy_forecast import energy_forecast as ef_module
    from energy_forecast import ha_data, weather

    modules = (ef_module, ha_data, model_mod, weather)
    originals = [m._LOGGER for m in modules]
    try:
        yield
    finally:
        for m, lg in zip(modules, originals):
            m._LOGGER = lg


@pytest.mark.usefixtures("_restore_module_loggers")
def test_physics_sensors_never_published(tmp_path, monkeypatch):
    from energy_forecast.physics import ThermalPhysicsModel

    from tests.test_energy_forecast_physics_config import (
        TestPhysicsMqttDiscoveryRegistration,
        TestServingOverrideCorrection,
    )

    helper = TestServingOverrideCorrection()
    pm = ThermalPhysicsModel(tmp_path / "pm", helper._physics_config())
    now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
    ts = pd.date_range(start=now, periods=48, freq="1h")
    _, stub = helper._run_update_sensors(tmp_path, monkeypatch, pm, ts)
    published = [str(c["entity_id"]) for c in stub.set_state_calls]
    for pid in _PHYSICS_SENSOR_IDS:
        assert not any(pid in p for p in published), (pid, published)
    from energy_forecast.energy_forecast import EnergyForecast

    assert not hasattr(EnergyForecast, "_publish_physics_sensors")

    # MQTT discovery registers neither sensor either.
    app = TestPhysicsMqttDiscoveryRegistration()._make_mqtt_app({})
    app.initialize()
    registered = [c.args[0] for c in app._mqtt_publish_discovery.call_args_list]
    for pid in _PHYSICS_SENSOR_IDS:
        assert not any(pid in r for r in registered), (pid, registered)


def test_legacy_calibration_file_loads_and_dhw_delta_works(tmp_path):
    import json

    from energy_forecast.physics import ThermalPhysicsModel

    from tests.test_energy_forecast_physics_config import TestServingOverrideCorrection

    mdir = tmp_path / "legacy"
    mdir.mkdir()
    (mdir / "physics_calibration.json").write_text(
        json.dumps(
            {
                "calibrated_at": pd.Timestamp.now().isoformat(),
                "UA_eff": 123.0,
                "solar_gain_area": 2.0,
                "n_calibration_windows_ua_eff": 40,
                "q_base_el_calibrated": True,
                "room_thermostats_at_calibration": ["climate.a"],
                "Q_base_el": 0.3,
                "Q_dhw_daily": 3.0,
                "UA_dhw": 4.0,
                "cop_formula": None,
            }
        )
    )
    pm = ThermalPhysicsModel(mdir, TestServingOverrideCorrection()._physics_config())
    now = pd.Timestamp.now().floor("1h")
    ts = pd.date_range(start=now, periods=48, freq="1h")
    target = ts[10]
    pm.commit_dhw_schedule({"legionella": [target.strftime("%Y-%m-%d"), target.hour]})
    t_amb = pd.Series(20.0, index=ts)
    delta = pm.compute_serving_override_delta(ts, t_amb, 45.0)
    assert isinstance(delta, pd.Series)
    assert delta.loc[target] > 0


# ── Task 3: app surface ──────────────────────────────────────────────────────


def _scenario_app(monkeypatch=None):
    from unittest.mock import MagicMock

    from energy_forecast.energy_forecast import EnergyForecast

    from tests.test_energy_forecast_physics_config import _make_app

    app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
    app.initialize()
    app._get_scenario_cb = EnergyForecast._get_scenario_cb.__get__(app, type(app))
    app._cached_forecast_df = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-15", periods=48, freq="1h"), "temp_c": [5.0] * 48}
    )
    app._cached_live_temp = 5.0
    app.fire_event = MagicMock()
    app._ml_model.predict_scenario = lambda *a, **kw: pd.DataFrame(
        {
            "timestamp": app._cached_forecast_df["timestamp"],
            "predicted_kwh": [1.0] * 48,
            "delta_kwh": [0.0] * 48,
        }
    )
    return app


@pytest.mark.usefixtures("_restore_module_loggers")
def test_get_scenario_ignores_dhw_schedule():
    app = _scenario_app()
    today = pd.Timestamp("2026-01-16").date().isoformat()
    app._get_scenario_cb("default", "energy_forecast", "get_scenario", {"schedule": {}})
    first = app.fire_event.call_args_list[-1]
    app.fire_event.reset_mock()
    app._get_scenario_cb(
        "default", "energy_forecast", "get_scenario", {"schedule": {}, "dhw_schedule": {"legionella": [today, 12]}}
    )
    second = app.fire_event.call_args_list[-1]
    assert first == second


@pytest.mark.usefixtures("_restore_module_loggers")
def test_retired_physics_sensors_cleared_at_startup():
    from unittest.mock import MagicMock

    from energy_forecast.energy_forecast import EnergyForecast

    from tests.test_energy_forecast_physics_config import TestPhysicsMqttDiscoveryRegistration

    app = TestPhysicsMqttDiscoveryRegistration()._make_mqtt_app({})
    app._retire_physics_sensors = EnergyForecast._retire_physics_sensors.__get__(app, type(app))
    app.call_service = MagicMock()
    app.initialize()
    pubs = [c.kwargs for c in app.call_service.call_args_list if c.args and c.args[0] == "mqtt/publish"]
    for uid in ("energy_forecast_physics_base_today", "energy_forecast_ml_adjustment_today"):
        topic = f"homeassistant/sensor/{uid}/config"
        assert any(p["topic"] == topic and p["payload"] == "" and p["retain"] is True for p in pubs), uid
    for p in pubs:
        if "physics_base_today" in p["topic"] or "ml_adjustment_today" in p["topic"]:
            assert p["payload"] == ""
    for p in app._mqtt_publish_discovery.call_args_list:
        assert not any(x in str(p) for x in _PHYSICS_SENSOR_IDS)


@pytest.mark.usefixtures("_restore_module_loggers")
def test_retire_physics_sensors_failure_only_warns():
    from unittest.mock import MagicMock

    from energy_forecast.energy_forecast import EnergyForecast

    app = MagicMock()
    app._mqtt_discovery_prefix = "homeassistant"
    app._mqtt_namespace = "mqtt"
    app.call_service.side_effect = RuntimeError("broker down")
    EnergyForecast._retire_physics_sensors(app)  # must not raise
    assert app.call_service.call_count == 2


@pytest.mark.usefixtures("_restore_module_loggers")
def test_today_attributes_have_no_model_phase(tmp_path, monkeypatch):
    from energy_forecast.physics import ThermalPhysicsModel

    from tests.test_energy_forecast_physics_config import TestServingOverrideCorrection

    helper = TestServingOverrideCorrection()
    pm = ThermalPhysicsModel(tmp_path / "pm", helper._physics_config())
    now = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
    ts = pd.date_range(start=now, periods=48, freq="1h")
    _, stub = helper._run_update_sensors(tmp_path, monkeypatch, pm, ts)
    assert stub.set_state_calls
    assert "model_phase" not in str(stub.set_state_calls)
    assert not hasattr(type(stub), "_model_phase_attr")


@pytest.mark.usefixtures("_restore_module_loggers")
def test_removed_physics_keys_are_ignored(monkeypatch):
    from unittest.mock import MagicMock

    from energy_forecast import ha_data as hd

    from tests.test_energy_forecast_physics_config import _make_app

    fetch_recent = MagicMock(return_value=pd.DataFrame(columns=["timestamp", "value"]))
    fetch_hist = MagicMock(return_value=pd.DataFrame(columns=["timestamp", "value"]))
    monkeypatch.setattr(hd, "fetch_recent_generic_sensor", fetch_recent)
    monkeypatch.setattr(hd, "fetch_generic_sensor_history", fetch_hist)
    monkeypatch.setattr(hd, "fetch_recent_climate", MagicMock())
    monkeypatch.setattr(hd, "fetch_climate_history", MagicMock())
    app = _make_app(
        {
            "energy_sensor": "sensor.grid_import",
            "physics": {
                "use_physics_residual": True,
                "cop_sensor": "sensor.cop",
                "internal_gains_fraction": 0.8,
                "dhw_tank_temp_sensor": "sensor.tank",
            },
        }
    )
    app.initialize()
    app._cache_path = None
    app._generic_sensor_cache_path = lambda eid, prefix="x": f"{prefix}.csv"
    app._fetch_physics_sensor_histories(recent_only=True)
    app._fetch_physics_sensor_histories()
    fetched = {c.args[1] for c in [*fetch_recent.call_args_list, *fetch_hist.call_args_list]}
    assert "sensor.tank" in fetched
    assert "sensor.cop" not in fetched
    assert not hasattr(app, "_physics_cop_df") or isinstance(app._physics_cop_df, MagicMock)
