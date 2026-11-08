"""Replay live input captures and check the two round-trip criteria (report-only).

Criterion 1 (predict round trip): the model that served a predict capture, re-run on the captured raw inputs
under the captured clock, reproduces the captured pre-override vector to <= 1e-6 kWh/h.
Criterion 2 (train round trip): train() re-run on a retrain capture, then predicting on the update made
straight after that retrain, reproduces the live next-day total to <= 0.1 %.

``python -m backtest.capture_replay <captures_dir>`` prints one row per check and a PASS/FAIL line. It always
exits 0: a FAIL is a finding to report, not something this tool gates on (same stance as ``backtest/repro.py``).

Caveat: ``backtest.env.frozen_now`` patches only ``pd.Timestamp.now``, not ``datetime.now``. The latter is used
solely for ``last_trained`` / archive stamps, which do not influence predictions, so the round trips are unaffected.
"""

from __future__ import annotations

import argparse
import copy
import math
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from . import env  # noqa: F401  (hassapi stub + apps/ on sys.path before the app imports)
from .env import TZ, frozen_now

PREDICT_TOL_KWH = 1e-6
TRAIN_TOL_PCT = 0.1


def _capture_mod():
    from energy_forecast import capture

    return capture


def _retarget_physics(physics, model_dir: Path):
    """Point a captured physics model's persistence at *model_dir* so a replay never writes the live paths."""
    if physics is None:
        return None
    physics = copy.deepcopy(physics)
    physics._model_dir = Path(model_dir)
    physics._calibration_path = Path(model_dir) / "physics_calibration.json"
    physics._schedule_path = Path(model_dir) / "physics_schedule.json"
    return physics


def _model_init_for(root: Path, last_trained: str | None, cache: dict | None = None) -> dict:
    """Constructor args of the retrain capture that produced *last_trained*; defaults when none is captured.

    Results are memoised in *cache* (keyed by ``last_trained``) so each retrain payload is unpickled at most once.
    """
    capture = _capture_mod()
    if cache is not None and last_trained in cache:
        return dict(cache[last_trained])
    init: dict = {"timezone": TZ}
    for sc in capture.list_captures(root, "retrain"):
        if last_trained is not None and sc.get("last_trained_after") == last_trained:
            payload, _ = capture.load_capture(root, "retrain", sc["capture_id"])
            init = dict(payload.get("model_init") or {})
            break
    if cache is not None:
        cache[last_trained] = init
    return dict(init)


def reference_dir_for(root: Path, predict_sidecar: dict) -> Path | None:
    """The ``.ref/`` of the retrain capture whose ``last_trained_after`` matches the predict capture's model."""
    capture = _capture_mod()
    want = predict_sidecar.get("last_trained")
    for sc in capture.list_captures(root, "retrain"):
        if want is not None and sc.get("last_trained_after") == want and sc.get("reference"):
            ref = Path(root) / "retrain" / f"{sc['capture_id']}.ref"
            if ref.is_dir():
                return ref
    return None


def replay_predict(
    root: Path,
    predict_id: str,
    model_dir: Path,
    *,
    loaded: tuple[dict, dict] | None = None,
    init_cache: dict | None = None,
) -> pd.DataFrame:
    """Re-run feature building + predict() on a predict capture against the model files in *model_dir*.

    *loaded* is an already-loaded ``(payload, sidecar)`` and *init_cache* a ``_model_init_for`` memo; both are
    optional and only avoid repeated unpickling.
    """
    from energy_forecast.model import EnergyForecastModel

    capture = _capture_mod()
    payload, sidecar = loaded if loaded is not None else capture.load_capture(root, "predict", predict_id)
    init = _model_init_for(root, sidecar.get("last_trained"), init_cache)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for p in Path(model_dir).iterdir():  # never let the replay touch the reference copy
            if p.is_file():
                shutil.copy2(p, work / p.name)
        with frozen_now(payload["now_local"], init.get("timezone", TZ)):
            model = EnergyForecastModel(work, **init)
            physics = _retarget_physics(payload["physics_model"], work)
            kwargs = copy.deepcopy(payload["predict_kwargs"])
            prepared = model._prepare_prediction_X(**kwargs, physics_model=physics)
            preds = model.predict(**kwargs, physics_model=physics, _prepared=prepared)
    preds = preds.copy()
    preds["timestamp"] = pd.to_datetime(preds["timestamp"]).dt.tz_localize(None)
    return preds


def _previous_reference(root: Path, retrain_id: str) -> Path | None:
    capture = _capture_mod()
    prev = None
    for sc in capture.list_captures(root, "retrain"):
        if sc["capture_id"] == retrain_id:
            break
        ref = Path(root) / "retrain" / f"{sc['capture_id']}.ref"
        if sc.get("reference") and ref.is_dir():
            prev = ref
    return prev


def replay_train(root: Path, retrain_id: str, scratch: Path) -> Path:
    """Re-run train() on a retrain capture into *scratch* (seeded from the previous retrain's output)."""
    from energy_forecast.model import EnergyForecastModel

    capture = _capture_mod()
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    prev = _previous_reference(root, retrain_id)
    if prev is not None:
        for p in prev.iterdir():
            if p.is_file():
                shutil.copy2(p, scratch / p.name)
    payload, _ = capture.load_capture(root, "retrain", retrain_id)
    init = dict(payload.get("model_init") or {})
    with frozen_now(payload["now_local"], init.get("timezone", TZ)):
        model = EnergyForecastModel(scratch, **init)
        kwargs = dict(payload["train_kwargs"])
        kwargs["physics_model"] = _retarget_physics(payload["physics_model"], scratch)
        model.train(**kwargs)
    return scratch


def _next_day_total(preds: pd.DataFrame, now_local: pd.Timestamp) -> float:
    ts = pd.to_datetime(preds["timestamp"])
    tomorrow = (pd.Timestamp(now_local).normalize() + pd.Timedelta(days=1)).date()
    mask = ts.dt.date == tomorrow
    if not mask.any():
        return float("nan")  # no tomorrow rows: summing the whole vector would hide the gap
    return float(preds.loc[mask, "predicted_kwh"].sum())


def _max_abs_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    a = a.assign(timestamp=pd.to_datetime(a["timestamp"]).dt.tz_localize(None)).set_index("timestamp")
    b = b.assign(timestamp=pd.to_datetime(b["timestamp"]).dt.tz_localize(None)).set_index("timestamp")
    if not a.index.equals(b.index):
        return float("inf")
    return float((a["predicted_kwh"] - b["predicted_kwh"]).abs().max())


def check(root: Path) -> list[dict]:
    """Run criterion 1 on every usable predict capture and criterion 2 on every retrain/after-retrain pair."""
    capture = _capture_mod()
    root = Path(root)
    rows: list[dict] = []
    predicts = capture.list_captures(root, "predict")
    init_cache: dict = {}
    loaded: dict[str, tuple[dict, dict]] = {}

    def _load(cid: str) -> tuple[dict, dict]:
        if cid not in loaded:
            loaded[cid] = capture.load_capture(root, "predict", cid)
        return loaded[cid]

    def _row(kind: str, cid: str, replayed: pd.DataFrame) -> dict:
        payload, _ = _load(cid)
        live = payload["model_predictions"]
        return {
            "predict_id": cid,
            "kind": kind,
            "max_abs_diff": _max_abs_diff(replayed, live),
            "next_day_total_pct": _pct(replayed, live, payload["now_local"]),
        }

    for sc in predicts:
        if sc.get("retrain_in_progress"):
            continue
        ref = reference_dir_for(root, sc)
        if ref is None:
            continue
        replayed = replay_predict(root, sc["capture_id"], ref, loaded=_load(sc["capture_id"]), init_cache=init_cache)
        rows.append(_row("predict", sc["capture_id"], replayed))
    for rc in capture.list_captures(root, "retrain"):
        after = rc.get("last_trained_after")
        if after is None:
            continue
        # Prefer the explicit after-retrain update; adaptive retrains never produce one, so fall back to the first
        # clean predict capture served by the freshly trained model.
        pair = next((p for p in predicts if p.get("after_retrain") and p.get("last_trained") == after), None)
        if pair is None:
            pair = next(
                (p for p in predicts if not p.get("retrain_in_progress") and p.get("last_trained") == after), None
            )
        if pair is None:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            scratch = replay_train(root, rc["capture_id"], Path(tmp) / "model")
            replayed = replay_predict(
                root, pair["capture_id"], scratch, loaded=_load(pair["capture_id"]), init_cache=init_cache
            )
        rows.append(_row("train", pair["capture_id"], replayed))
    return rows


def _pct(replayed: pd.DataFrame, live: pd.DataFrame, now_local) -> float:
    lv = _next_day_total(live, now_local)
    rv = _next_day_total(replayed, now_local)
    if math.isnan(lv) or math.isnan(rv):
        return float("nan")
    return 0.0 if lv == rv else (float("inf") if lv == 0 else 100.0 * (rv - lv) / lv)


def passed(row: dict) -> bool:
    if row["kind"] == "predict":
        return row["max_abs_diff"] <= PREDICT_TOL_KWH
    pct = row["next_day_total_pct"]
    return not math.isnan(pct) and abs(pct) <= TRAIN_TOL_PCT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backtest.capture_replay", description=__doc__.splitlines()[0])
    ap.add_argument("captures_dir", type=Path)
    args = ap.parse_args(argv)
    rows = check(args.captures_dir)
    if not rows:
        print("no checkable captures (need a predict capture with a model .ref, or a retrain + after-retrain pair)")
        return 0
    print(f"{'kind':<8} {'predict_id':<24} {'max_abs_diff':>14} {'next_day_%':>11}  result")
    for r in rows:
        print(
            f"{r['kind']:<8} {r['predict_id']:<24} {r['max_abs_diff']:>14.3g} {r['next_day_total_pct']:>11.4f}  "
            f"{'PASS' if passed(r) else 'FAIL'}"
        )
    n_fail = sum(not passed(r) for r in rows)
    print(f"{'PASS' if n_fail == 0 else 'FAIL'}: {len(rows) - n_fail}/{len(rows)} checks within tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
