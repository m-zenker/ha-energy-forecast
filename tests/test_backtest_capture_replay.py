"""Round-trip checks for backtest.capture_replay against captures produced by the real capture IO."""

from __future__ import annotations

import shutil
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


def _make_captures(
    tmp_path: Path,
    with_physics: bool,
    retrain_in_progress: bool = False,
    after_retrain: bool = True,
    pre_state: str = "empty",
) -> Path:
    """Simulate one live retrain + the update right after it, writing captures exactly as the app does.

    ``pre_state``: ``"empty"`` attaches the (empty) model dir like a fresh install, ``"before"`` first trains an
    earlier model so the attached pre-train state is non-empty, ``"none"`` writes a capture that predates ``.pre/``.
    """
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
    if pre_state == "before":
        prior_now = NOW - pd.Timedelta(days=1)
        with frozen_now(prior_now, TZ):
            prior = EnergyForecastModel(live_dir, **init)
            prior.train(
                **{
                    **train_kwargs,
                    "energy_df": hist[hist["timestamp"] < prior_now],
                    "weather_df": weather[weather["timestamp"] < prior_now],
                    "physics_model": None,
                }
            )
    if pre_state != "none":
        capture.attach_pre_state(root, rid, live_dir)
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
            "after_retrain": after_retrain,
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


def test_adaptive_retrain_pairs_with_first_predict_on_new_model(tmp_path):
    """Adaptive retrains never produce an after_retrain capture; criterion 2 must still run."""
    root = _make_captures(tmp_path, with_physics=False, after_retrain=False)
    kinds = [r["kind"] for r in capture_replay.check(root)]
    assert "train" in kinds


def test_adaptive_pairing_skips_in_progress_captures(tmp_path):
    root = _make_captures(tmp_path, with_physics=False, after_retrain=False, retrain_in_progress=True)
    assert all(r["kind"] != "train" for r in capture_replay.check(root))


def test_next_day_total_nan_when_tomorrow_missing():
    import math

    preds = pd.DataFrame({"timestamp": pd.date_range("2026-03-01 06:00", periods=5, freq="1h"), "predicted_kwh": 1.0})
    assert math.isnan(capture_replay._next_day_total(preds, NOW))
    row = {"kind": "train", "max_abs_diff": 0.0, "next_day_total_pct": capture_replay._pct(preds, preds, NOW)}
    assert not capture_replay.passed(row)


def test_check_loads_each_payload_once(tmp_path, monkeypatch):
    root = _make_captures(tmp_path, with_physics=False)
    real = capture.load_capture
    calls: list[tuple[str, str]] = []

    def counting(r, kind, cid):
        calls.append((kind, cid))
        return real(r, kind, cid)

    monkeypatch.setattr(capture, "load_capture", counting)
    capture_replay.check(root)
    predict_loads = [c for c in calls if c[0] == "predict"]
    assert len(predict_loads) == len(set(predict_loads)) == 1
    retrain_loads = [c for c in calls if c[0] == "retrain"]
    # replay_train loads the retrain payload once; model_init lookups are cached (one more load at most)
    assert len(retrain_loads) <= 2


def test_app_written_predict_capture_replays(tmp_path, monkeypatch):
    """Capture through the real app hooks, then replay: catches key drift between app and replay."""
    from energy_forecast import energy_forecast as ef_mod

    root = _make_captures(tmp_path, with_physics=False)
    for p in (root / "predict").iterdir():  # drop the hand-built predict capture; keep retrain + .ref
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    monkeypatch.setattr(ef_mod, "CAPTURE_DIR", root)
    retrain_sc = capture.list_captures(root, "retrain")[0]
    rpayload, _ = capture.load_capture(root, "retrain", retrain_sc["capture_id"])
    weather = synthetic_weather(60, start="2026-01-01")
    energy = synthetic_energy(weather)
    hist = energy[energy["timestamp"] < NOW]
    forecast = weather[(weather["timestamp"] >= NOW - pd.Timedelta(days=3))].head(72 + 48).copy()
    predict_kwargs = {"forecast_df": forecast, "live_temp": None, "recent_actuals": hist.tail(24 * 8)}

    class _Model:
        last_trained = pd.Timestamp(retrain_sc["last_trained_after"])

    class _App:
        _capture_inputs = True
        _retrain_running = False
        _physics_model = None
        _ml_model = _Model()

    app = _App()
    ref = capture_replay.reference_dir_for(root, {"last_trained": retrain_sc["last_trained_after"]})
    # Produce the "live" model output with the real model from the retrain reference.
    with frozen_now(NOW, TZ):
        model = EnergyForecastModel(ref, **rpayload["model_init"])
        prepared = model._prepare_prediction_X(**predict_kwargs, physics_model=None)
        preds = model.predict(**predict_kwargs, physics_model=None, _prepared=prepared)
    app._ml_model.last_trained = model.last_trained
    snap = ef_mod._snapshot_predict_inputs(app, predict_kwargs, NOW)
    assert snap is not None
    with frozen_now(NOW + pd.Timedelta(minutes=1), TZ):  # keeps prune() from aging out the 2026-03 retrain
        ef_mod._write_predict_capture(
            app, snap, preds.copy(), preds.copy(), {"tomorrow": 0.0}, weather_source=None, after_retrain=False
        )
    (sc,) = capture.list_captures(root, "predict")
    replayed = capture_replay.replay_predict(root, sc["capture_id"], ref)
    assert capture_replay._max_abs_diff(replayed, preds.assign(timestamp=pd.to_datetime(preds["timestamp"]))) <= (
        capture_replay.PREDICT_TOL_KWH
    )


def _fake_retrain(root: Path, ts: str, tmp_path: Path, *, pre: str | None = None, ref: str | None = None) -> str:
    """A retrain capture with marker files in .pre/ and/or .ref/ (payload is never loaded by seed_dir_for)."""
    cid = capture.write_capture(root, "retrain", {}, {"last_trained_after": ts}, pd.Timestamp(ts, tz="UTC"))
    for suffix, marker in ((".pre", pre), (".ref", ref)):
        if marker is None:
            continue
        md = tmp_path / f"{cid}{suffix}.src"
        md.mkdir()
        (md / "marker.txt").write_text(marker)
        (capture.attach_pre_state if suffix == ".pre" else capture.attach_reference)(root, cid, md)
    return cid


def test_first_retrain_without_seed_is_skipped_not_failed(tmp_path, capsys):
    root = _make_captures(tmp_path, with_physics=False, pre_state="none")
    rows = capture_replay.check(root)
    train = next(r for r in rows if r["kind"] == "train")
    assert train["skipped"] == "no seed" and capture_replay.skipped(train)
    assert capture_replay.passed(next(r for r in rows if r["kind"] == "predict"))
    assert capture_replay.main([str(root)]) == 0
    out = capsys.readouterr().out
    assert "SKIP" in out and "1 skipped" in out and "FAIL" not in out


def test_round_trip_passes_with_pre_state(tmp_path):
    root = _make_captures(tmp_path, with_physics=False, pre_state="before")
    (rc,) = capture.list_captures(root, "retrain")
    assert any((root / "retrain" / f"{rc['capture_id']}.pre").iterdir())
    train = next(r for r in capture_replay.check(root) if r["kind"] == "train")
    assert not capture_replay.skipped(train)
    assert capture_replay.passed(train) and abs(train["next_day_total_pct"]) <= capture_replay.TRAIN_TOL_PCT


def test_pre_state_seed_is_preferred_over_previous_ref(tmp_path):
    root = tmp_path / "caps"
    _fake_retrain(root, "2026-03-01", tmp_path, ref="previous-ref")
    second = _fake_retrain(root, "2026-03-02", tmp_path, pre="own-pre", ref="own-ref")
    seed = capture_replay.seed_dir_for(root, second)
    assert seed == root / "retrain" / f"{second}.pre"
    assert (seed / "marker.txt").read_text() == "own-pre"


def test_falls_back_to_previous_ref_without_pre_state(tmp_path):
    root = tmp_path / "caps"
    first = _fake_retrain(root, "2026-03-01", tmp_path, ref="previous-ref")
    second = _fake_retrain(root, "2026-03-02", tmp_path, ref="own-ref")
    assert capture_replay.seed_dir_for(root, second) == root / "retrain" / f"{first}.ref"
    assert capture_replay.seed_dir_for(root, first) is None


def test_empty_pre_dir_is_a_valid_seed(tmp_path):
    root = _make_captures(tmp_path, with_physics=False, pre_state="empty")
    (rc,) = capture.list_captures(root, "retrain")
    seed = capture_replay.seed_dir_for(root, rc["capture_id"])
    assert seed is not None and list(seed.iterdir()) == []
    capture_replay.replay_train(root, rc["capture_id"], tmp_path / "scratch")  # no NoSeedError


def test_replay_train_raises_no_seed_error_unless_allowed(tmp_path):
    root = _make_captures(tmp_path, with_physics=False, pre_state="none")
    (rc,) = capture.list_captures(root, "retrain")
    with pytest.raises(capture_replay.NoSeedError):
        capture_replay.replay_train(root, rc["capture_id"], tmp_path / "scratch")
    out = capture_replay.replay_train(root, rc["capture_id"], tmp_path / "scratch2", allow_unseeded=True)
    assert (out / "energy_model.pkl").exists()


def test_seeded_replay_still_fails_when_an_input_changes(tmp_path):
    root = _make_captures(tmp_path, with_physics=False, pre_state="before")
    (rc,) = capture.list_captures(root, "retrain")
    payload, _ = capture.load_capture(root, "retrain", rc["capture_id"])
    df = payload["train_kwargs"]["energy_df"].copy()
    num = [c for c in df.columns if c != "timestamp" and pd.api.types.is_numeric_dtype(df[c])]
    df[num] = df[num] * 1.5
    payload["train_kwargs"]["energy_df"] = df
    capture.write_capture(
        root,
        "retrain",
        payload,
        {k: v for k, v in rc.items() if k not in {"capture_id", "kind", "ts_utc", "versions", "bytes"}},
        pd.Timestamp(rc["ts_utc"]),
    )
    train = next(r for r in capture_replay.check(root) if r["kind"] == "train")
    assert not capture_replay.skipped(train) and not capture_replay.passed(train)
