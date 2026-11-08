import json
import logging
import os
import time

import pandas as pd
import pytest
from energy_forecast import capture


def _payload():
    return {
        "df": pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]}, index=pd.date_range("2026-01-01", periods=2, freq="h")),
        "s": pd.Series([1.0, 2.0], name="x"),
        "none": None,
        "nested": {"k": {"z": [1, 2, 3]}},
    }


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_write_then_load_roundtrip(tmp_path):
    p = _payload()
    cid = capture.write_capture(tmp_path, "retrain", p, {"foo": "bar"}, _ts("2026-06-01 12:00"))
    assert cid == "20260601T120000000000Z"
    out, sc = capture.load_capture(tmp_path, "retrain", cid)
    pd.testing.assert_frame_equal(out["df"], p["df"])
    pd.testing.assert_series_equal(out["s"], p["s"])
    assert out["none"] is None
    assert out["nested"] == p["nested"]
    assert sc["kind"] == "retrain" and sc["foo"] == "bar"
    assert sc["ts_utc"] and sc["versions"]["pandas"] and sc["bytes"] > 0
    assert sc["capture_id"] == cid


def test_capture_ids_unique_across_dst_fallback(tmp_path):
    a = capture.write_capture(tmp_path, "predict", {}, {}, _ts("2026-10-25 00:30"))
    b = capture.write_capture(tmp_path, "predict", {}, {}, _ts("2026-10-25 01:30"))
    assert a != b
    assert len(capture.list_captures(tmp_path, "predict")) == 2


def test_update_sidecar_merges_fields(tmp_path):
    cid = capture.write_capture(tmp_path, "retrain", {}, {"foo": 1}, _ts("2026-06-01"))
    capture.update_sidecar(tmp_path, "retrain", cid, last_trained_after="x")
    sc = capture.list_captures(tmp_path, "retrain")[0]
    assert sc["foo"] == 1 and sc["last_trained_after"] == "x"


def test_write_never_raises(tmp_path, caplog):
    f = tmp_path / "file"
    f.write_text("x")
    with caplog.at_level(logging.WARNING):
        assert capture.write_capture(f, "retrain", {}, {}, _ts("2026-06-01")) is None
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_partial_files_ignored_and_pruned(tmp_path):
    now = _ts("2026-06-01")
    capture.write_capture(tmp_path, "predict", {}, {}, now)
    d = tmp_path / "predict"
    tmp = d / "x.pkl.gz.tmp"
    tmp.write_text("partial")
    old = time.time() - 7200
    os.utime(tmp, (old, old))
    (d / "orphan.json").write_text(json.dumps({"ts_utc": "2026-06-01T00:00:00+00:00"}))
    assert len(capture.list_captures(tmp_path, "predict")) == 1
    capture.prune(tmp_path, pd.Timestamp.now(tz="UTC"), retention_days=10_000)
    assert not tmp.exists()


def test_prune_retention(tmp_path):
    now = _ts("2026-06-01")
    capture.write_capture(tmp_path, "predict", {}, {}, now - pd.Timedelta(days=46))
    capture.write_capture(tmp_path, "predict", {}, {}, now - pd.Timedelta(days=44))
    capture.prune(tmp_path, now)
    caps = capture.list_captures(tmp_path, "predict")
    assert len(caps) == 1
    assert caps[0]["ts_utc"].startswith("2026-04-18")


def test_prune_size_cap(tmp_path):
    now = _ts("2026-06-01")
    ids = []
    for i in range(3):
        payload = {"x": os.urandom(5000)}
        ids.append(capture.write_capture(tmp_path, "predict", payload, {}, now - pd.Timedelta(days=3 - i)))
    size = capture.list_captures(tmp_path, "predict")[0]["bytes"]
    removed = capture.prune(tmp_path, now, max_bytes=int(size * 1.5))
    assert removed == 2
    left = capture.list_captures(tmp_path, "predict")
    assert [c["capture_id"] for c in left] == [ids[2]]


def test_attach_reference_copies_model_dir_without_archive(tmp_path):
    md = tmp_path / "model"
    (md / "archive").mkdir(parents=True)
    (md / "energy_model.pkl").write_bytes(b"m")
    (md / "physics_schedule.json").write_text("{}")
    (md / "archive" / "old.pkl").write_bytes(b"o")
    root = tmp_path / "caps"
    cid = capture.write_capture(root, "retrain", {}, {}, _ts("2026-06-01"))
    before = capture.list_captures(root, "retrain")[0]["bytes"]
    capture.attach_reference(root, cid, md)
    ref = root / "retrain" / f"{cid}.ref"
    assert sorted(p.name for p in ref.iterdir()) == ["energy_model.pkl", "physics_schedule.json"]
    sc = capture.list_captures(root, "retrain")[0]
    assert sc["reference"] is True and sc["bytes"] > before


def test_load_warns_on_version_mismatch(tmp_path):
    cid = capture.write_capture(tmp_path, "predict", {}, {}, _ts("2026-06-01"))
    sp = tmp_path / "predict" / f"{cid}.json"
    sc = json.loads(sp.read_text())
    sc["versions"]["pandas"] = "0.0"
    sp.write_text(json.dumps(sc))
    with pytest.warns(UserWarning, match="pandas"):
        capture.load_capture(tmp_path, "predict", cid)


def test_prune_removes_orphan_pickle_and_ref_dir(tmp_path):
    now = _ts("2026-06-01")
    d = tmp_path / "retrain"
    d.mkdir(parents=True)
    old = time.time() - 7200
    pkl, ref, fresh = d / "a.pkl.gz", d / "a.ref", d / "b.pkl.gz"
    pkl.write_bytes(b"x")
    ref.mkdir()
    (ref / "m.pkl").write_bytes(b"x")
    fresh.write_bytes(b"x")
    for p in (pkl, ref):
        os.utime(p, (old, old))
    capture.prune(tmp_path, now)
    assert not pkl.exists() and not ref.exists()
    assert fresh.exists()


def test_prune_survives_corrupt_sidecar(tmp_path):
    now = _ts("2026-06-01")
    capture.write_capture(tmp_path, "predict", {}, {}, now - pd.Timedelta(days=46))
    d = tmp_path / "predict"
    (d / "bad.json").write_text("{not json")
    (d / "bad.pkl.gz").write_bytes(b"x")
    (d / "naive.json").write_text(json.dumps({"ts_utc": "2026-05-30T00:00:00"}))
    (d / "naive.pkl.gz").write_bytes(b"x")
    assert capture.prune(tmp_path, now) == 3
    assert not (d / "bad.pkl.gz").exists()


def test_write_does_not_prune_new_capture(tmp_path, monkeypatch):
    orig = capture.prune
    monkeypatch.setattr(capture, "prune", lambda *a, **k: orig(*a, **{**k, "max_bytes": 0}))
    cid = capture.write_capture(tmp_path, "predict", {}, {}, _ts("2020-01-01"))
    assert cid is not None
    assert (tmp_path / "predict" / f"{cid}.json").exists()
