"""Round-trip checks for backtest.capture_replay against captures produced by the real capture IO."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from energy_forecast import capture
from energy_forecast.model import EnergyForecastModel
from energy_forecast.physics import ThermalPhysicsModel

from backtest import capture_replay
from backtest.env import REPO, frozen_now
from tests._backtest_synth import synthetic_energy, synthetic_weather
from tests.test_physics import DEFAULT_CONFIG

NOW = pd.Timestamp("2026-03-01 06:00")
TZ = "Europe/Zurich"


def _make_captures(tmp_path: Path, with_physics: bool, retrain_in_progress: bool = False) -> Path:
    """Simulate one live retrain + the update right after it, writing captures exactly as the app does."""
    root = tmp_path / "captures"
    live_dir = tmp_path / "live_models"
    weather = synthetic_weather(60, start="2026-01-01")
    energy = synthetic_energy(weather)
    hist = energy[energy["timestamp"] < NOW]
    physics = ThermalPhysicsModel(tmp_path / "pmodels", DEFAULT_CONFIG) if with_physics else None
    init = {"model_archive_count": 3, "timezone": TZ}
    train_kwargs = {
        "energy_df": hist,
        "weather_df": weather[weather["timestamp"] < NOW],
        "outdoor_df": None,
        "weight_halflife_days": 0,
        "physics_model": physics,
    }
    import copy

    rid = capture.write_capture(
        root,
        "retrain",
        {
            "train_kwargs": {**train_kwargs, "physics_model": None},
            "physics_model": copy.deepcopy(physics),
            "now_local": NOW,
            "model_init": init,
        },
        {"last_trained_before": None, "physics": with_physics},
        NOW.tz_localize(TZ).tz_convert("UTC"),
    )
    with frozen_now(NOW, TZ):
        model = EnergyForecastModel(live_dir, **init)
        model.train(**train_kwargs)
    capture.attach_reference(root, rid, live_dir)
    capture.update_sidecar(root, "retrain", rid, last_trained_after=str(model.last_trained))

    forecast = weather[(weather["timestamp"] >= NOW - pd.Timedelta(days=3))].head(72 + 48).copy()
    predict_kwargs = {
        "forecast_df": forecast,
        "live_temp": None,
        "recent_actuals": hist.tail(24 * 8),
    }
    snapshot = {
        "predict_kwargs": copy.deepcopy(predict_kwargs),
        "physics_model": copy.deepcopy(physics),
        "now_local": NOW,
    }
    with frozen_now(NOW, TZ):
        prepared = model._prepare_prediction_X(**predict_kwargs, physics_model=physics)
        preds = model.predict(**predict_kwargs, physics_model=physics, _prepared=prepared)
    capture.write_capture(
        root,
        "predict",
        {**snapshot, "model_predictions": preds.copy(), "published_predictions": preds.copy(), "tomorrow_kwh": 0.0},
        {
            "last_trained": str(model.last_trained),
            "weather_source": "open_meteo",
            "retrain_in_progress": retrain_in_progress,
            "after_retrain": True,
            "physics": with_physics,
        },
        (NOW + pd.Timedelta(minutes=1)).tz_localize(TZ).tz_convert("UTC"),
    )
    return root


@pytest.mark.parametrize("with_physics", [False, True])
def test_replay_round_trips(tmp_path, with_physics):
    root = _make_captures(tmp_path, with_physics)
    rows = capture_replay.check(root)
    by_kind = {r["kind"]: r for r in rows}
    assert set(by_kind) == {"predict", "train"}
    assert by_kind["predict"]["max_abs_diff"] <= capture_replay.PREDICT_TOL_KWH
    assert abs(by_kind["train"]["next_day_total_pct"]) <= capture_replay.TRAIN_TOL_PCT
    assert all(capture_replay.passed(r) for r in rows)


def test_replay_does_not_write_live_physics_paths(tmp_path):
    root = _make_captures(tmp_path, with_physics=True)
    before = sorted(p.name for p in (tmp_path / "pmodels").iterdir())
    capture_replay.check(root)
    assert sorted(p.name for p in (tmp_path / "pmodels").iterdir()) == before


def test_check_skips_retrain_in_progress(tmp_path):
    root = _make_captures(tmp_path, with_physics=False, retrain_in_progress=True)
    assert all(r["kind"] != "predict" for r in capture_replay.check(root))


def test_replay_detects_a_changed_input(tmp_path):
    root = _make_captures(tmp_path, with_physics=False)
    sc = capture.list_captures(root, "predict")[0]
    payload, _ = capture.load_capture(root, "predict", sc["capture_id"])
    payload["model_predictions"]["predicted_kwh"] += 0.5
    capture.write_capture(
        root,
        "predict",
        payload,
        {k: v for k, v in sc.items() if k not in {"capture_id", "kind", "ts_utc"}},
        pd.Timestamp(sc["ts_utc"]),
    )
    rows = [r for r in capture_replay.check(root) if r["kind"] == "predict"]
    assert rows and not capture_replay.passed(rows[0])


def test_cli_prints_pass(tmp_path):
    root = _make_captures(tmp_path, with_physics=False)
    out = subprocess.run(
        [sys.executable, "-m", "backtest.capture_replay", str(root)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert "PASS" in out.stdout
