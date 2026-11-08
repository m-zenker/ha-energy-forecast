"""Live input capture: flag wiring and the retrain hook (Task 2)."""

import copy
from unittest.mock import MagicMock

import pandas as pd
import pytest
from energy_forecast import capture, weather
from energy_forecast import energy_forecast as ef_mod
from energy_forecast.physics import ThermalPhysicsModel

from tests.test_energy_forecast import _empty_weather, _FakeRetrain, _make_energy_df
from tests.test_physics import DEFAULT_CONFIG


def _patch_deps(monkeypatch):
    import energy_forecast.ha_data as ha_data_mod
    import energy_forecast.weather as weather_mod

    energy_df = _make_energy_df(100)
    empty_df = pd.DataFrame()
    monkeypatch.setattr(ha_data_mod, "fetch_energy_history", lambda *a, **kw: energy_df)
    monkeypatch.setattr(ha_data_mod, "split_ev_charging", lambda df, *a, **kw: (df, empty_df))
    monkeypatch.setattr(weather_mod, "fetch_historical_weather", lambda *a, **kw: _empty_weather())
    monkeypatch.setattr(weather_mod, "fetch_open_meteo", lambda *a, **kw: _empty_weather())
    monkeypatch.setattr(ha_data_mod, "fetch_boolean_entity_history", lambda *a, **kw: empty_df)
    monkeypatch.setattr(ha_data_mod, "fetch_presence_history", lambda *a, **kw: empty_df)
    monkeypatch.setattr(ha_data_mod, "fetch_energy_history_15m", lambda *a, **kw: None)


def _stub(tmp_path, *, capture_on, train=None, physics=None, args=None):
    stub = _FakeRetrain(tmp_path / "energy_history.csv")
    stub._capture_inputs = capture_on
    stub._retrain_running = False
    stub._model_archive_count = 3
    stub.args = args or {}
    stub._physics_model = physics
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.txt").write_text("m")
    (model_dir / "archive").mkdir(exist_ok=True)
    stub._ml_model._model_dir = model_dir
    if train is not None:
        stub._ml_model.train.side_effect = train
    return stub


def _run(stub):
    ef_mod.EnergyForecast._retrain(stub)


def test_retrain_capture_off_by_default(tmp_path, monkeypatch):
    _patch_deps(monkeypatch)
    stub = _stub(tmp_path, capture_on=False)
    _run(stub)
    assert not ef_mod.CAPTURE_DIR.exists()
    stub._ml_model.train.assert_called_once()


def test_retrain_capture_records_exact_train_kwargs(tmp_path, monkeypatch):
    _patch_deps(monkeypatch)
    stub = _stub(tmp_path, capture_on=True)
    _run(stub)
    (cap,) = capture.list_captures(ef_mod.CAPTURE_DIR, "retrain")
    payload, meta = capture.load_capture(ef_mod.CAPTURE_DIR, "retrain", cap["capture_id"])
    got = stub._ml_model.train.call_args.kwargs
    assert set(payload["train_kwargs"]) == set(got)
    for k, v in got.items():
        pv = payload["train_kwargs"][k]
        if isinstance(v, pd.DataFrame):
            pd.testing.assert_frame_equal(pv, v)
        elif isinstance(v, dict) and v and all(isinstance(x, pd.DataFrame) for x in v.values()):
            assert set(pv) == set(v)
        else:
            assert pv == v
    assert payload["model_init"] == {"model_archive_count": 3, "timezone": "Europe/Zurich"}
    assert payload["now_local"].tzinfo is None
    assert meta["physics"] is False


def test_retrain_capture_physics_copied_before_train(tmp_path, monkeypatch):
    _patch_deps(monkeypatch)
    physics = ThermalPhysicsModel(tmp_path / "pmodels", DEFAULT_CONFIG)
    physics._tau_hours = 5.0

    def _train(**kw):
        kw["physics_model"]._tau_hours = 99.0

    stub = _stub(tmp_path, capture_on=True, train=_train, physics=physics)
    _run(stub)
    (cap,) = capture.list_captures(ef_mod.CAPTURE_DIR, "retrain")
    payload, meta = capture.load_capture(ef_mod.CAPTURE_DIR, "retrain", cap["capture_id"])
    assert payload["physics_model"]._tau_hours == 5.0
    assert meta["physics"] is True


def test_physics_model_deepcopy_works(tmp_path):
    pm = ThermalPhysicsModel(tmp_path / "pmodels", DEFAULT_CONFIG)
    assert copy.deepcopy(pm)._tau_hours == pm._tau_hours


def test_retrain_capture_drops_secrets(tmp_path, monkeypatch):
    _patch_deps(monkeypatch)
    stub = _stub(
        tmp_path,
        capture_on=True,
        args={"srg_client_secret": "x", "api_token": "y", "db_password": "z", "energy_sensor": "sensor.e"},
    )
    _run(stub)
    (cap,) = capture.list_captures(ef_mod.CAPTURE_DIR, "retrain")
    payload, _ = capture.load_capture(ef_mod.CAPTURE_DIR, "retrain", cap["capture_id"])
    assert payload["args"] == {"energy_sensor": "sensor.e"}


def test_retrain_capture_attaches_reference(tmp_path, monkeypatch):
    _patch_deps(monkeypatch)
    stub = _stub(tmp_path, capture_on=True)
    stub._ml_model.last_trained = pd.Timestamp("2026-01-01")
    _run(stub)
    (cap,) = capture.list_captures(ef_mod.CAPTURE_DIR, "retrain")
    ref = ef_mod.CAPTURE_DIR / "retrain" / f"{cap['capture_id']}.ref"
    assert (ref / "model.txt").read_text() == "m"
    assert not (ref / "archive").exists()
    assert cap["last_trained_after"] == "2026-01-01 00:00:00"


def test_retrain_capture_failure_does_not_break_retrain(tmp_path, monkeypatch):
    _patch_deps(monkeypatch)
    monkeypatch.setattr(capture, "write_capture", MagicMock(side_effect=RuntimeError("x")))
    stub = _stub(tmp_path, capture_on=True)
    _run(stub)
    stub._ml_model.train.assert_called_once()


def test_retrain_running_flag_reset_on_exception(tmp_path, monkeypatch):
    _patch_deps(monkeypatch)
    seen = []

    def _train(**kw):
        seen.append(stub._retrain_running)
        raise RuntimeError("boom")

    stub = _stub(tmp_path, capture_on=False, train=_train)
    with pytest.raises(RuntimeError):
        _run(stub)
    assert seen == [True]
    assert stub._retrain_running is False


def test_fetch_forecast_tags_source(monkeypatch):
    om = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-01")], "temp_c": [1.0]})
    monkeypatch.setattr(weather, "fetch_open_meteo", lambda *a, **kw: om.copy())
    assert weather.fetch_forecast("8000", 47.0, 8.5).attrs["weather_source"] == "open_meteo"

    import requests

    class _Resp:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            if getattr(self, "_url", "").endswith("accesstoken?grant_type=client_credentials"):
                return {"access_token": "t"}
            return {"hours": [{"date_time": "2026-01-01T12:00:00+01:00", "TTT_C": 1}]}

    monkeypatch.setattr(weather, "_srg_token", "t")
    monkeypatch.setattr(weather, "_srg_token_expires_at", 1e18)
    monkeypatch.setitem(weather._srg_geo_id, (47.0, 8.5), "g")
    monkeypatch.setattr(requests, "get", lambda *a, **kw: _Resp())
    out = weather.fetch_forecast("8000", 47.0, 8.5, client_id="a", client_secret="b")
    assert out.attrs["weather_source"] == "srg"

    def _fail(*a, **kw):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "get", _fail)
    out = weather.fetch_forecast("8000", 47.0, 8.5, client_id="a", client_secret="b")
    assert out.attrs["weather_source"] == "open_meteo"
