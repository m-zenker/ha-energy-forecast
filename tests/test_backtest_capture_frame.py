"""backtest.capture_frame / backtest.walkforward: leak guards on a synthetic capture payload, plus one real run."""

import numpy as np
import pandas as pd
import pytest

from backtest.capture_frame import SERVE_PAST, CaptureFrame, predict, train
from backtest.walkforward import ARMS, WfArm, retrain_days, run_arm
from tests._backtest_synth import synthetic_energy, synthetic_weather

DAYS = 80
CUTOFF = pd.Timestamp("2026-03-01 18:00")


def _payload(days: int = DAYS) -> dict:
    weather = synthetic_weather(days)
    energy = synthetic_energy(weather)
    ts = energy["timestamp"]
    hourly = lambda col, values: pd.DataFrame({"timestamp": ts, col: values})  # noqa: E731
    climate = pd.DataFrame({"timestamp": ts, "current_temp": 20.5, "setpoint": 21.0, "heating_power_request": 0.0})
    return {
        "train_kwargs": {
            "energy_df": energy,
            "weather_df": weather,
            "outdoor_df": None,
            "weight_halflife_days": 60.0,
            "canton": "SO",
            "country": "CH",
            "ev_df": None,
            "enable_regimes": False,
            "sub_sensors_dict": {"sub_x": hourly("kwh", 0.1)},
            "climate_dfs": {"climate.a": climate},
            "room_areas": {"climate.a": 20.0},
            "dhw_df": hourly("buffer_temp", 50.0),
            "away_df": hourly("is_away", 0),
            "heating_active_df": hourly("heating_active", 1),
            "heating_feature_df": hourly("heating_active", 1),
            "heating_buffer_temp_df": hourly("heating_buffer_temp", 33.0),
        },
        "physics_model": None,
        "model_init": {"timezone": "Europe/Zurich"},
    }


@pytest.fixture(scope="module")
def frame() -> CaptureFrame:
    return CaptureFrame(_payload())


def _max_ts(value) -> pd.Timestamp:
    return pd.to_datetime(value["timestamp"]).max()


def test_train_kwargs_hold_nothing_at_or_after_the_cutoff(frame):
    kw = frame.train_kwargs(CUTOFF)
    for key in ("energy_df", "weather_df", "dhw_df", "heating_active_df", "heating_buffer_temp_df", "away_df"):
        assert _max_ts(kw[key]) < CUTOFF, key
    for key in ("sub_sensors_dict", "climate_dfs"):
        assert all(_max_ts(df) < CUTOFF for df in kw[key].values()), key
    assert kw["weight_halflife_days"] == 60.0 and kw["canton"] == "SO"


def test_heating_label_runs_to_the_end_of_the_cutoff_day_only(frame):
    label = frame.train_kwargs(CUTOFF)["heating_feature_df"]
    assert _max_ts(label) == CUTOFF.normalize() + pd.Timedelta(hours=23)


def test_away_history_is_limited_to_ten_days(frame):
    away = frame.train_kwargs(CUTOFF)["away_df"]
    assert pd.to_datetime(away["timestamp"]).min() >= CUTOFF - pd.Timedelta(days=10)


def test_frames_empty_before_their_first_row_become_none():
    payload = _payload()
    late = payload["train_kwargs"]["dhw_df"]
    payload["train_kwargs"]["dhw_df"] = late[late["timestamp"] >= "2026-03-10"]
    payload["train_kwargs"]["climate_dfs"] = {
        k: v[v["timestamp"] >= "2026-03-10"] for k, v in payload["train_kwargs"]["climate_dfs"].items()
    }
    kw = CaptureFrame(payload).train_kwargs(CUTOFF)
    assert kw["dhw_df"] is None and kw["climate_dfs"] is None and kw["room_areas"] is None


def test_predict_kwargs_use_only_history_before_the_origin_hour(frame):
    origin = pd.Timestamp("2026-03-02 06:20")
    kw = frame.predict_kwargs(origin)
    hour = origin.floor("h")
    assert _max_ts(kw["recent_actuals"]) == hour - pd.Timedelta(hours=1)
    assert len(kw["recent_actuals"]) == 401
    for key in ("dhw_recent", "heating_buffer_temp_recent"):
        assert _max_ts(kw[key]) < hour, key
    assert all(_max_ts(df) < hour for df in kw["sub_sensors_recent"].values())
    assert all(_max_ts(df) < hour for df in kw["climate_recent"].values())
    assert kw["away_series"].index[0] == hour and len(kw["away_series"]) == 48
    assert kw["heating_active_series"].index[0] == hour and kw["setpoint_on"] == 21.0


def test_forecast_frame_spans(frame):
    origin = pd.Timestamp("2026-03-02 06:20")
    wide = frame.predict_kwargs(origin)["forecast_df"]["timestamp"]
    live = frame.forecast_frame(origin, SERVE_PAST)["timestamp"]
    day = origin.normalize()
    assert wide.min() == day - pd.Timedelta(days=16) and live.min() == day - pd.Timedelta(days=3)
    assert wide.max() == live.max() == day + pd.Timedelta(days=4, hours=1)
    assert len(live) == 170  # the shape live passes to predict()


def test_actuals_are_the_training_target(frame):
    a = frame.actuals()
    assert a.index.is_monotonic_increasing and len(a) == DAYS * 24
    assert a.iloc[0] == pytest.approx(frame.tk["energy_df"]["gross_kwh"].iloc[0])


def test_retrain_days_cadence():
    daily = retrain_days(pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-05"), 1)
    weekly = retrain_days(pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-20"), 7)
    assert daily[0] == pd.Timestamp("2026-03-01") and len(daily) == 5
    assert [d.day for d in weekly] == [1, 8, 15]


def test_arm_registry_has_a_jitter_control():
    assert ARMS["jit"].cutoff_shift_h == -1 and ARMS["base"] == WfArm("base")
    assert "thermal_pressure" in ARMS["droptp"].drop


def test_train_and_predict_run_the_real_model(frame, tmp_path):
    model, work = train(frame, CUTOFF, tmp_path)
    assert model.model is not None and work.parent == tmp_path
    origin = pd.Timestamp("2026-03-02 00:00")
    out = predict(frame, model, work, origin)
    assert list(out["lead"]) == list(range(48)) and out["ts"].iloc[0] == origin
    assert np.isfinite(out["p50"]).all() and (out["p50"] >= 0).all()
    assert {"lo", "hi"} <= set(out.columns) and (out["lo"] <= out["hi"]).all()


def test_run_arm_writes_four_origins_a_day_and_cleans_up(frame, tmp_path):
    rows = run_arm(ARMS["base"], frame, pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-02"), tmp_path)
    # 18:00 on the day before through 00:00 after the target day: the last origin that can still be scored
    assert rows["origin"].nunique() == 6 and rows["origin"].max() == pd.Timestamp("2026-03-03 00:00")
    assert set(rows["origin"].dt.hour) == {0, 6, 12, 18}
    assert (rows["cutoff"] <= rows["origin"]).all()
    assert rows["arm"].eq("base").all() and not any(tmp_path.iterdir())


def test_jitter_arm_trains_one_hour_earlier(frame, tmp_path):
    rows = run_arm(ARMS["jit"], frame, pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-02"), tmp_path)
    assert set(rows["cutoff"].dt.hour) == {17}
