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


# ── chains, scoring, CLI ─────────────────────────────────────────────────────


def test_each_arm_keeps_its_own_seed_chain(tmp_path, monkeypatch):
    calls = []

    def fake(root, rid, scratch, *, seed_dir=None, **kw):
        calls.append((rid, seed_dir and f"{seed_dir.parent.name}/{seed_dir.name}"))
        scratch.mkdir(parents=True)

    monkeypatch.setattr(capture_replay, "replay_train", fake)
    chain = [{"capture_id": c} for c in "abc"]
    for name in ("base", "s1", "droptp"):
        shadow._ensure_model(tmp_path, tmp_path, name, chain, chain[2])
    assert calls == [
        ("c", None),
        ("a", None), ("b", "s1/a"), ("c", "s1/b"),
        ("a", None), ("b", "droptp/a"), ("c", "droptp/b"),
    ]  # fmt: skip
    calls.clear()
    shadow._ensure_model(tmp_path, tmp_path, "s1", chain + [{"capture_id": "d"}], {"capture_id": "d"})
    assert calls == [("d", "s1/c")]


def test_keeps_newest_three_models(tmp_path):
    chain = [{"capture_id": c} for c in "abcde"]
    for c in "abcde":
        (tmp_path / "models" / "s1" / c).mkdir(parents=True)
    shadow._prune_models(tmp_path, "s1", chain)
    assert sorted(p.name for p in (tmp_path / "models" / "s1").iterdir()) == ["c", "d", "e"]


DAYS = pd.date_range("2026-11-26", periods=20, freq="D")


def _actuals(last="2026-12-15 23:00"):
    idx = pd.date_range("2026-11-10", last, freq="h")
    return pd.Series(1.0, index=idx)


def _write_rows(out, code="c1", app="a1", offset=None, days=DAYS):
    """Day-ahead forecasts: base 0.2 kWh/h too high, every other arm 0.1 too high; four origins a day."""
    (out / "rows").mkdir(parents=True, exist_ok=True)
    for d in days:
        for h in (0, 6, 12, 18):
            origin = d - pd.Timedelta(days=1) + pd.Timedelta(hours=h)
            ts = pd.date_range(origin, periods=48, freq="h")
            parts = []
            for arm in shadow.ARM_NAMES:
                bias = 0.2 if arm in ("base", "s1") else 0.1
                parts.append(pd.DataFrame({"arm": arm, "ts": ts, "p50": 1.0 + bias, "lead": range(48)}))
            df = pd.concat(parts).assign(
                origin=origin,
                cutoff=origin,
                logged_at=origin,
                blind=True,
                code=code,
                app_code=app,
                parity="PASS",
                model_app_code=app,
                chain_skips=0,
            )
            df.to_pickle(out / "rows" / f"{origin:%Y%m%dT%H}.pkl", compression=None)


def _temp(value=5.0):
    return pd.Series(value, index=pd.date_range("2026-11-10", "2026-12-31", freq="D"))


def test_score_returns_both_verdict_kinds(tmp_path):
    _write_rows(tmp_path)
    temp = _temp()
    temp[temp.index >= "2026-12-06"] = 8.0
    res = shadow.score_rows(shadow.load_rows(tmp_path), _actuals(), temp, pd.Timestamp("2026-11-26"))
    assert set(res["shadow"]) == {"droptp", "droptp_nb"} and set(res["candidate"]) == {
        "shapecl",
        "clrecent",
        "rescale",
        "s1",
    }
    assert res["candidate"]["rescale"]["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert res["candidate"]["rescale"]["bias6_pct"] == pytest.approx(-50)
    assert res["days"] == 20


def test_score_drops_days_without_full_actuals(tmp_path):
    _write_rows(tmp_path)
    res = shadow.score_rows(
        shadow.load_rows(tmp_path), _actuals("2026-12-10 11:00"), _temp(), pd.Timestamp("2026-11-26")
    )
    assert res["days"] == 14  # 11-26..12-09


def test_score_refuses_mixed_code(tmp_path):
    _write_rows(tmp_path, code="c1", days=DAYS[:10])
    _write_rows(tmp_path, code="c2", days=DAYS[10:])
    res = shadow.score_rows(shadow.load_rows(tmp_path), _actuals(), _temp(), pd.Timestamp("2026-11-26"))
    assert "c1" in res["reason"] and "c2" in res["reason"]
    assert {v["verdict"] for k in ("shadow", "candidate") for v in res[k].values()} == {"INCONCLUSIVE"}


def test_score_lists_a_changed_app_code_but_scores(tmp_path):
    _write_rows(tmp_path, app="a1", days=DAYS[:10])
    _write_rows(tmp_path, app="a2", days=DAYS[10:])
    res = shadow.score_rows(shadow.load_rows(tmp_path), _actuals(), _temp(), pd.Timestamp("2026-11-26"))
    assert res["app_codes"] == ["a1", "a2"] and res["reason"] is None and res["days"] == 20


def test_score_filters_before_swing(tmp_path):
    _write_rows(tmp_path)
    rows = shadow.load_rows(tmp_path)
    # a wild burn-in day must not reach the swing number
    early = rows["ts"] < "2026-11-26"
    jump = rows["origin"].dt.hour.eq(18)
    rows.loc[early & jump & rows["arm"].eq("rescale"), "p50"] += 50
    res = shadow.score_rows(rows, _actuals(), _temp(), pd.Timestamp("2026-11-26"))
    assert res["swing"].loc["rescale", "range"] == pytest.approx(0.0, abs=1e-9)


def test_score_without_rows_is_inconclusive(tmp_path):
    res = shadow.score_rows(shadow.load_rows(tmp_path), _actuals(), _temp(), pd.Timestamp("2026-11-26"))
    assert res["days"] == 0 and "no rows" in res["reason"]


def test_status_lists_last_run_and_gaps(tmp_path):
    _write_rows(tmp_path, days=DAYS[:3].append(DAYS[8:10]))
    (tmp_path / "status.json").write_text(
        json.dumps({"last_run": "2026-12-06 04:00", "errors": {}, "parity": {}, "skipped": {}})
    )
    text = shadow.status_text(tmp_path, pd.Timestamp("2026-12-06 10:00"))
    assert "2026-12-06 04:00" in text and "gap" in text and "2026-11-27" in text and "2026-12-03" in text
    assert "no run" in shadow.status_text(tmp_path / "none", pd.Timestamp("2026-12-06"))


def test_cli_run_and_status(root, tmp_path, capsys):
    out = tmp_path / "out"
    rc = shadow.main(["run", str(root), "--out", str(out), "--arms", "base", "--since", "2026-03-01"])
    assert rc == 0 and len(list((out / "rows").glob("*.pkl"))) == 1
    assert shadow.main(["status", "--out", str(out)]) == 0
    assert "last run" in capsys.readouterr().out


# ── review findings ──────────────────────────────────────────────────────────


def _break_parity(root):
    import gzip
    import pickle

    (sc,) = capture.list_captures(root, "predict")
    payload, _ = capture.load_capture(root, "predict", sc["capture_id"])
    payload["model_predictions"]["predicted_kwh"] += 0.01
    with gzip.open(root / "predict" / f"{sc['capture_id']}.pkl.gz", "wb") as fh:
        pickle.dump(payload, fh)
    return sc["capture_id"]


def test_parity_failure_survives_the_next_run(root, tmp_path):
    cid = _break_parity(root)
    _run(root, tmp_path)
    second = _run(root, tmp_path)
    assert second["processed"] == [] and shadow.exit_code(second) == 0  # not a new failure
    assert f"parity FAIL: {cid}" in shadow.status_text(tmp_path / "out", LATER)


def test_score_excludes_parity_failures(tmp_path):
    _write_rows(tmp_path)
    rows = shadow.load_rows(tmp_path)
    bad = rows["origin"] == pd.Timestamp("2026-11-28 06:00")
    rows.loc[bad, "parity"] = "FAIL"
    rows.loc[bad & rows["arm"].eq("base"), "p50"] += 500
    res = shadow.score_rows(rows, _actuals(), _temp(), pd.Timestamp("2026-11-26"))
    assert res["parity_failed"] == [pd.Timestamp("2026-11-28 06:00")]
    assert res["table"].xs("base", level="arm")["mae"].max() == pytest.approx(0.2)


def test_subset_run_is_completed_later(root, tmp_path):
    _run(root, tmp_path, arms=("base",))
    first = _rows(tmp_path)
    status = _run(root, tmp_path, arms=("base", "s1"), now=LATER + pd.Timedelta(days=3))
    rows = _rows(tmp_path)
    assert len(status["processed"]) == 1 and set(rows["arm"]) == {"base", "s1"}
    kept = rows[rows["arm"] == "base"].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, first.reset_index(drop=True)[kept.columns])  # old rows untouched
    assert (rows.loc[rows["arm"] == "s1", "logged_at"] == LATER + pd.Timedelta(days=3)).all()
    assert _run(root, tmp_path, arms=("base",))["processed"] == []


def test_base_is_always_included(root, tmp_path):
    _run(root, tmp_path, arms=("s1",))
    assert set(_rows(tmp_path)["arm"]) == {"base", "s1"}


def test_score_keeps_only_origins_all_arms_have(tmp_path):
    _write_rows(tmp_path)
    rows = shadow.load_rows(tmp_path)
    rows = rows[~(rows["arm"].eq("droptp") & rows["origin"].dt.hour.eq(18))]
    res = shadow.score_rows(rows, _actuals(), _temp(), pd.Timestamp("2026-11-26"))
    assert res["incomplete_origins"] > 0
    assert res["swing"].loc["droptp", "range"] == pytest.approx(res["swing"].loc["base", "range"])
    counts = res["origins"].groupby("arm")["origin"].nunique()
    assert counts.nunique() == 1


def test_one_logged_origin_per_slot(root, tmp_path):
    _run(root, tmp_path)
    (sc,) = capture.list_captures(root, "predict")
    d = root / "predict"
    later = {**sc, "capture_id": "late", "ts_utc": (pd.Timestamp(sc["ts_utc"]) - pd.Timedelta(seconds=30)).isoformat()}
    shutil.copy2(d / f"{sc['capture_id']}.pkl.gz", d / "late.pkl.gz")
    (d / "late.json").write_text(json.dumps(later))
    assert _run(root, tmp_path)["processed"] == []
    assert len(list((tmp_path / "out" / "rows").glob("*.pkl"))) == 1


def test_chain_ignores_retrains_that_never_finished(root, tmp_path):
    (rc,) = capture.list_captures(root, "retrain")
    d = root / "retrain"
    sc = {**rc, "capture_id": "unfinished", "ts_utc": (pd.Timestamp(rc["ts_utc"]) - pd.Timedelta(days=1)).isoformat()}
    sc.pop("last_trained_after", None)
    (d / "unfinished.json").write_text(json.dumps(sc))
    (d / "unfinished.pkl.gz").write_bytes(b"broken")
    status = _run(root, tmp_path, arms=("base", "s1"))
    assert not status["errors"] and len(status["processed"]) == 1


def _stub_train(monkeypatch, broken=()):
    calls = []

    def fake(root, rid, scratch, *, seed_dir=None, **kw):
        calls.append((rid, seed_dir and seed_dir.name))
        if rid in broken:
            raise ValueError("bad capture")
        scratch.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(capture_replay, "replay_train", fake)
    return calls


def test_a_broken_link_is_skipped_and_counted(tmp_path, monkeypatch):
    calls = _stub_train(monkeypatch, broken={"b"})
    chain = [{"capture_id": c} for c in "abcd"]
    path, meta = shadow._ensure_model(tmp_path, tmp_path, "s1", chain, chain[2])
    assert path.name == "c" and calls == [("a", None), ("b", "a"), ("c", "a")] and meta["chain_skips"] == 1
    path, meta = shadow._ensure_model(tmp_path, tmp_path, "s1", chain, chain[3])
    assert path.name == "d" and meta["chain_skips"] == 1  # carried forward
    with pytest.raises(ValueError):
        shadow._ensure_model(tmp_path, tmp_path, "s1", chain, chain[1])  # the broken one itself
    with pytest.raises(ValueError):
        shadow._ensure_model(tmp_path, tmp_path, "base", chain, chain[1])


def test_models_are_rebuilt_when_arm_code_or_chain_start_changes(tmp_path, monkeypatch):
    calls = _stub_train(monkeypatch)
    chain = [{"capture_id": c} for c in "abc"]
    shadow._ensure_model(tmp_path, tmp_path, "s1", chain, chain[2], code="1", app="x")
    n = len(calls)
    shadow._ensure_model(tmp_path, tmp_path, "s1", chain, chain[2], code="1", app="y")
    assert len(calls) == n  # live continues across an app change, so does the chain
    _, meta = shadow._ensure_model(tmp_path, tmp_path, "s1", chain, chain[2], code="1", app="y")
    assert meta["app"] == "x"
    shadow._ensure_model(tmp_path, tmp_path, "s1", chain, chain[2], code="2", app="y")
    assert len(calls) == n + 3
    shadow._ensure_model(tmp_path, tmp_path, "s1", chain[1:], chain[2], code="2", app="y")
    assert len(calls) == n + 5


def test_prune_never_leaves_a_half_deleted_model(tmp_path, monkeypatch):
    chain = [{"capture_id": c} for c in "abcd"]
    for c in "abcd":
        (tmp_path / "models" / "s1" / c).mkdir(parents=True)

    def killed(path, *a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(shadow.shutil, "rmtree", killed)
    with pytest.raises(KeyboardInterrupt):
        shadow._prune_models(tmp_path, "s1", chain)
    assert not (tmp_path / "models" / "s1" / "a").exists()


def test_score_shadow_verdict_needs_fourteen_days(tmp_path):
    _write_rows(tmp_path, days=DAYS[:13])
    res = shadow.score_rows(
        shadow.load_rows(tmp_path), _actuals("2026-12-08 23:00"), _temp(), pd.Timestamp("2026-11-26")
    )
    assert {v["verdict"] for v in res["shadow"].values()} == {"INCONCLUSIVE"} and res["days"] == 13


def test_hash_covers_the_seed_and_clock_helpers():
    assert "backtest/env.py" in shadow.ARM_FILES


def test_no_captures_is_an_error(tmp_path):
    status = _run(tmp_path / "empty", tmp_path)
    assert shadow.exit_code(status) == 1 and "captures" in status["errors"]


def test_status_reports_a_failed_pull(root, tmp_path):
    _run(root, tmp_path, pull_exit=1)
    assert "pull failed" in shadow.status_text(tmp_path / "out", LATER)
    _run(root, tmp_path, pull_exit=0)
    assert "pull failed" not in shadow.status_text(tmp_path / "out", LATER)


def test_cli_run_is_locked(root, tmp_path, capsys):
    import fcntl

    out = tmp_path / "out"
    out.mkdir()
    with open(out / ".run.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        assert shadow.main(["run", str(root), "--out", str(out), "--arms", "base", "--since", "2026-03-01"]) == 0
    assert "active" in capsys.readouterr().out and not (out / "rows").exists()


def test_replay_train_rejects_a_skipped_train(root, tmp_path, monkeypatch):
    from energy_forecast.model import EnergyForecastModel

    (rc,) = capture.list_captures(root, "retrain")
    monkeypatch.setattr(EnergyForecastModel, "train", lambda self, **kw: None)
    with pytest.raises(RuntimeError, match="skipped"):
        capture_replay.replay_train(root, rc["capture_id"], tmp_path / "a")
