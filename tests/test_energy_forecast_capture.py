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


# ── Task 3: predict hook ─────────────────────────────────────────────────────

_PREDICT_KWARG_KEYS = {
    "forecast_df",
    "live_temp",
    "recent_actuals",
    "sub_sensors_recent",
    "away_series",
    "people_home_series",
    "climate_recent",
    "dhw_recent",
    "room_areas",
    "heating_active_series",
    "setpoint_on",
    "setpoint_off",
    "heating_buffer_temp_recent",
}


def _predict_stub(tmp_path, monkeypatch, *, capture_on, after_retrain=False, run=True, retrain_running=False):
    import json  # noqa: F401

    from tests.test_energy_forecast import TestUpdateSensorsExcludedRanges, _FakeUpdateSensors

    TestUpdateSensorsExcludedRanges()._patch_update_sensors_deps(monkeypatch, _make_energy_df(24), tmp_path)
    stub = _FakeUpdateSensors(tmp_path / "energy_history.csv")
    stub._capture_inputs = capture_on
    stub._retrain_running = retrain_running
    if run:
        if after_retrain:
            ef_mod.EnergyForecast._update_sensors(stub, after_retrain=True)
        else:
            ef_mod.EnergyForecast._update_sensors(stub)
    return stub


def _load_predict():
    (cap,) = capture.list_captures(ef_mod.CAPTURE_DIR, "predict")
    payload, meta = capture.load_capture(ef_mod.CAPTURE_DIR, "predict", cap["capture_id"])
    return payload, meta


def test_predict_capture_off_by_default(tmp_path, monkeypatch):
    _predict_stub(tmp_path, monkeypatch, capture_on=False)
    assert not ef_mod.CAPTURE_DIR.exists()


def test_published_output_identical_with_flag_on(tmp_path, monkeypatch):
    import json

    def _once(on, sub):
        d = tmp_path / sub
        d.mkdir()
        monkeypatch.setattr(ef_mod, "REGIME_LOG_PATH", d / "regime_log.jsonl")
        stub = _predict_stub(d, monkeypatch, capture_on=on)
        rec = json.loads((d / "regime_log.jsonl").read_text().splitlines()[0])
        rec.pop("ts")
        return stub.set_state_calls, rec, (d / "pred_history.json").read_text()

    off = _once(False, "off")
    assert not ef_mod.CAPTURE_DIR.exists()
    on = _once(True, "on")
    assert len(capture.list_captures(ef_mod.CAPTURE_DIR, "predict")) == 1
    assert off[0], "expected published set_state calls"
    assert on[0] == off[0]
    assert on[1] == off[1]
    assert on[2] == off[2]


def test_predict_capture_contents(tmp_path, monkeypatch):
    stub = _predict_stub(tmp_path, monkeypatch, capture_on=True)
    payload, meta = _load_predict()
    assert set(payload["predict_kwargs"]) == _PREDICT_KWARG_KEYS
    expected = stub._ml_model.predict()
    pd.testing.assert_frame_equal(payload["model_predictions"].reset_index(drop=True), expected)
    pd.testing.assert_frame_equal(payload["published_predictions"].reset_index(drop=True), expected)
    assert isinstance(payload["tomorrow_kwh"], float)
    assert payload["now_local"].tzinfo is None
    assert payload["physics_model"] is None
    assert set(meta) >= {"last_trained", "weather_source", "retrain_in_progress", "physics", "after_retrain"}
    assert meta["physics"] is False
    assert meta["after_retrain"] is False
    assert meta["retrain_in_progress"] is False
    assert meta["weather_source"] is None  # patched fetch_forecast returns an untagged frame


def test_predict_capture_records_weather_source(tmp_path, monkeypatch):
    import energy_forecast.weather as weather_mod

    def _fc(*a, **kw):
        df = pd.DataFrame(columns=["timestamp"])
        df.attrs["weather_source"] = "srg"
        return df

    stub = _predict_stub(tmp_path, monkeypatch, capture_on=True, run=False)
    monkeypatch.setattr(weather_mod, "fetch_forecast", _fc)
    ef_mod.EnergyForecast._update_sensors(stub)
    _, meta = _load_predict()
    assert meta["weather_source"] == "srg"


class _PM:
    def _t_indoor_for_window(self, *a, **kw):
        return pd.Series(20.0, index=a[1])

    def _initial_t_tank_for_window(self, *a, **kw):
        return 45.0

    def compute_serving_override_delta(self, idx, *a, **kw):
        return pd.Series(0.5, index=idx)


def test_predict_capture_model_predictions_precede_override(tmp_path, monkeypatch):
    stub = _predict_stub(tmp_path, monkeypatch, capture_on=True, run=False)
    base = stub._ml_model.predict()

    stub._physics_model = _PM()
    # physics hooks on the fake are real-delegating; neutralise the ones that need a real model
    stub._fetch_physics_sensor_histories = lambda *a, **kw: None
    stub._publish_physics_sensors = lambda *a, **kw: None
    import energy_forecast.weather as weather_mod

    fc = pd.DataFrame(
        {"timestamp": pd.date_range(base["timestamp"].iloc[0] - pd.Timedelta(hours=2), periods=60, freq="1h")}
    )
    fc["temp_c"] = 5.0
    monkeypatch.setattr(weather_mod, "fetch_forecast", lambda *a, **kw: fc.copy())
    ef_mod.EnergyForecast._update_sensors(stub)
    payload, meta = _load_predict()
    assert meta["physics"] is True
    assert (payload["model_predictions"]["predicted_kwh"] == 1.0).all()
    assert (payload["published_predictions"]["predicted_kwh"] == 1.5).all()
    assert payload["physics_model"] is not None


def test_capture_failure_does_not_break_update(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setattr(capture, "write_capture", MagicMock(side_effect=RuntimeError("x")))
    with caplog.at_level(logging.WARNING, logger="energy_forecast"):
        _predict_stub(tmp_path, monkeypatch, capture_on=True)
    assert ef_mod.REGIME_LOG_PATH.read_text().strip()
    assert len([r for r in caplog.records if "capture: predict snapshot failed" in r.getMessage()]) == 1


def test_after_retrain_marked(tmp_path, monkeypatch):
    _predict_stub(tmp_path, monkeypatch, capture_on=True, after_retrain=True, retrain_running=True)
    _, meta = _load_predict()
    assert meta["after_retrain"] is True
    assert meta["retrain_in_progress"] is False


def test_retrain_in_progress_marked(tmp_path, monkeypatch):
    _predict_stub(tmp_path, monkeypatch, capture_on=True, retrain_running=True)
    _, meta = _load_predict()
    assert meta["retrain_in_progress"] is True


def test_predict_capture_without_physics(tmp_path, monkeypatch):
    _predict_stub(tmp_path, monkeypatch, capture_on=True)
    payload, _ = _load_predict()
    assert payload["physics_model"] is None


def test_retrain_cb_passes_after_retrain(tmp_path):
    stub = MagicMock()
    stub._lock = MagicMock()
    stub._lock.acquire.return_value = True
    stub._ml_model.model = object()
    ef_mod.EnergyForecast._retrain_cb(stub)
    stub._update_sensors.assert_called_once_with(after_retrain=True)
