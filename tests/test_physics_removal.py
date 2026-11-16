"""Physics heating removal, task 1: model.py serving API, legacy physics_kwh models, backtest kwarg stripping."""

from __future__ import annotations

import inspect
import logging

import numpy as np
import pandas as pd
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
