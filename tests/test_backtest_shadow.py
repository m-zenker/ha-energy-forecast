"""backtest.shadow: idempotent capture-based shadow runner."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from energy_forecast import capture, clustering

from backtest import capture_replay, shadow
from backtest import regime_arms as ra
from tests.test_backtest_capture_replay import NOW, TZ, _make_captures

LATER = NOW + pd.Timedelta(hours=10)
SINCE = NOW.normalize()


@pytest.fixture(scope="module")
def captures(tmp_path_factory):
    return _make_captures(tmp_path_factory.mktemp("caps"), with_physics=True, pre_state="before")


@pytest.fixture
def root(captures, tmp_path):
    return shutil.copytree(captures, tmp_path / "captures")


def _run(root, tmp_path, **kw):
    kw = {"arms": ("base",), "now": LATER, "since": SINCE} | kw
    return shadow.run(root, tmp_path / "out", **kw)


def _rows(tmp_path):
    return pd.concat([pd.read_pickle(p) for p in sorted((tmp_path / "out" / "rows").glob("*.pkl"))])


def _sidecar(hour, minute=1, **kw):
    ts = pd.Timestamp(f"2026-11-26 {hour:02d}:{minute:02d}", tz=TZ).tz_convert("UTC")
    return {"capture_id": f"{hour:02d}{minute:02d}", "ts_utc": ts.isoformat(), "retrain_in_progress": False} | kw


def test_select_origins_first_clean_capture_per_slot():
    sc = [
        _sidecar(0, 1, retrain_in_progress=True),
        _sidecar(0, 20),
        _sidecar(0, 40),
        _sidecar(5),
        _sidecar(6),
        _sidecar(12),
        _sidecar(18),
        _sidecar(23),
    ]
    assert [s["capture_id"] for s in shadow.select_origins(sc)] == ["0020", "0601", "1201", "1801"]


def test_base_matches_captured_predictions(root, tmp_path):
    status = _run(root, tmp_path)
    (sc,) = capture.list_captures(root, "predict")
    payload, _ = capture.load_capture(root, "predict", sc["capture_id"])
    rows = _rows(tmp_path)
    assert rows["p50"].to_numpy() == pytest.approx(payload["model_predictions"]["predicted_kwh"].to_numpy(), abs=1e-6)
    assert status["parity"] == {sc["capture_id"]: "PASS"} and status["processed"] == [sc["capture_id"]]
    assert {
        "arm",
        "origin",
        "cutoff",
        "ts",
        "lead",
        "p50",
        "lo",
        "hi",
        "logged_at",
        "blind",
        "code",
        "app_code",
    } <= set(rows)
    assert rows["origin"].eq(NOW).all() and rows["cutoff"].eq(NOW).all()
    assert rows["lead"].tolist() == [(t - NOW) / pd.Timedelta(hours=1) for t in rows["ts"]]


def test_second_run_is_a_noop(root, tmp_path):
    _run(root, tmp_path)
    files = lambda: {p: p.stat().st_mtime_ns for d in ("rows", "models") for p in (tmp_path / "out" / d).rglob("*")}  # noqa: E731
    before = files()
    status = _run(root, tmp_path, now=LATER + pd.Timedelta(days=1))
    assert status["processed"] == [] and files() == before


def test_tmp_file_is_redone(root, tmp_path):
    (sc,) = capture.list_captures(root, "predict")
    (tmp_path / "out" / "rows").mkdir(parents=True)
    (tmp_path / "out" / "rows" / f"{sc['capture_id']}.pkl.tmp").write_bytes(b"partial")
    half = tmp_path / "out" / "models" / "base" / "x.tmp"
    half.mkdir(parents=True)
    assert _run(root, tmp_path)["processed"] == [sc["capture_id"]]
    assert len(_rows(tmp_path)) == 48


def test_missing_retrain_is_skipped_and_reported(root, tmp_path):
    (rc,) = capture.list_captures(root, "retrain")
    (root / "retrain" / f"{rc['capture_id']}.json").unlink()
    status = _run(root, tmp_path)
    (reason,) = status["skipped"].values()
    assert "retrain" in reason and status["processed"] == []
    assert not list((tmp_path / "out" / "rows").glob("*.pkl"))


def test_unseeded_first_retrain_is_skipped_and_reported(root, tmp_path):
    (rc,) = capture.list_captures(root, "retrain")
    shutil.rmtree(root / "retrain" / f"{rc['capture_id']}.pre")
    status = _run(root, tmp_path, arms=("base", "droptp"))
    assert "seed" in next(iter(status["skipped"].values())) and status["processed"] == []


def test_blind_flag_per_row(root, tmp_path):
    _run(root, tmp_path)
    rows = _rows(tmp_path)
    assert rows["logged_at"].eq(LATER).all()
    assert (rows["blind"] == (rows["ts"] > LATER)).all() and rows["blind"].any() and not rows["blind"].all()


def test_code_hash_ignores_docs_changes(tmp_path):
    for name in shadow.ARM_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("a")
    first = shadow.code_hash(tmp_path)
    (tmp_path / "backtest" / "README.md").write_text("docs")
    assert shadow.code_hash(tmp_path) == first
    (tmp_path / shadow.ARM_FILES[0]).write_text("b")
    assert shadow.code_hash(tmp_path) != first


def test_parity_mismatch_exits_2_and_logs(root, tmp_path):
    (sc,) = capture.list_captures(root, "predict")
    payload, _ = capture.load_capture(root, "predict", sc["capture_id"])
    payload["model_predictions"]["predicted_kwh"] += 0.01
    import gzip
    import pickle

    with gzip.open(root / "predict" / f"{sc['capture_id']}.pkl.gz", "wb") as fh:
        pickle.dump(payload, fh)
    status = _run(root, tmp_path)
    assert status["parity"] == {sc["capture_id"]: "FAIL"} and shadow.exit_code(status) == 2
    assert len(_rows(tmp_path)) == 48
    assert json.loads((tmp_path / "out" / "status.json").read_text())["parity"] == status["parity"]


def test_one_broken_capture_does_not_stop_the_run(root, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(shadow, "_forecast", boom)
    status = _run(root, tmp_path)
    assert "boom" in next(iter(status["errors"].values())) and shadow.exit_code(status) == 1


def test_replay_train_defaults_unchanged(root, tmp_path):
    (rc,) = capture.list_captures(root, "retrain")
    (pc,) = capture.list_captures(root, "predict")
    a = capture_replay.replay_train(root, rc["capture_id"], tmp_path / "a")
    b = capture_replay.replay_train(
        root, rc["capture_id"], tmp_path / "b", seed_dir=capture_replay.seed_dir_for(root, rc["capture_id"])
    )
    pa, pb = (capture_replay.replay_predict(root, pc["capture_id"], d)["predicted_kwh"] for d in (a, b))
    np.testing.assert_array_equal(pa, pb)


def test_replay_train_patch_not_active_after(root, tmp_path):
    (rc,) = capture.list_captures(root, "retrain")
    fit, seen = clustering.DailyProfileClusterer.fit, []

    def transform(kwargs):
        seen.append(clustering.DailyProfileClusterer.fit is fit)
        return kwargs["energy_df"]

    capture_replay.replay_train(root, rc["capture_id"], tmp_path / "a", patch=ra.shape_only, energy_transform=transform)
    assert clustering.DailyProfileClusterer.fit is fit and len(seen) == 1


def test_replay_train_applies_the_energy_transform(root, tmp_path):
    (rc,) = capture.list_captures(root, "retrain")
    (pc,) = capture.list_captures(root, "predict")
    plain = capture_replay.replay_train(root, rc["capture_id"], tmp_path / "a")
    double = capture_replay.replay_train(
        root,
        rc["capture_id"],
        tmp_path / "b",
        energy_transform=lambda kw: kw["energy_df"].assign(gross_kwh=kw["energy_df"]["gross_kwh"] * 2),
    )
    p, d = (capture_replay.replay_predict(root, pc["capture_id"], m)["predicted_kwh"].sum() for m in (plain, double))
    assert d > 1.3 * p
