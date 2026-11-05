"""#110 regime-label backtest. Subcommands: ``gate`` (C2, reproduce the live model) -- ``backtest`` (C3).

Imports go through ``energy_forecast.*`` (``apps/`` on ``sys.path``); see ``regime_harness.py``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_harness as rh  # noqa: E402  (also puts apps/ on sys.path and stubs hassapi)

REPO = rh.REPO
GATE_MIN_AGREEMENT = 0.90
GATE_MAX_PRED_DIFF = 0.03


# ── Gate maths (pure) ─────────────────────────────────────────────────────────


def day_profiles(energy_df: pd.DataFrame) -> pd.DataFrame:
    """Date x 24h profile frame, built exactly as ``DailyProfileClusterer.fit`` does."""
    daily = energy_df.copy()
    daily["date"] = daily["timestamp"].dt.date
    daily["hour"] = daily["timestamp"].dt.hour
    counts = daily.groupby("date")["gross_kwh"].count()
    valid = counts[counts >= 18].index
    pivoted = daily[daily["date"].isin(valid)].pivot(index="date", columns="hour", values="gross_kwh")
    return pivoted.reindex(columns=range(24)).interpolate(axis=1, limit=6).ffill(axis=1).bfill(axis=1)


def nearest_centroid_labels(energy_df: pd.DataFrame, centroids: np.ndarray) -> pd.Series:
    """Argmin Euclidean distance of each valid day's profile to *centroids* (= ``KMeans.predict``)."""
    piv = day_profiles(energy_df)
    dist = ((piv.to_numpy()[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return pd.Series(dist.argmin(axis=1), index=piv.index)


def _rank_by_daysum(centroids: np.ndarray) -> dict[int, int]:
    order = np.argsort(centroids.sum(axis=1), kind="stable")
    return {int(old): rank for rank, old in enumerate(order)}


def label_agreement(
    live_centroids: np.ndarray,
    harness_centroids: np.ndarray,
    live_labels: pd.Series,
    harness_labels: pd.Series,
) -> float:
    """Share of common days with the same label after ranking each ID set by centroid day-sum."""
    common = live_labels.index.intersection(harness_labels.index)
    if len(common) == 0:
        return 0.0
    live_r = live_labels.loc[common].map(_rank_by_daysum(live_centroids))
    harn_r = harness_labels.loc[common].map(_rank_by_daysum(harness_centroids))
    return float((live_r == harn_r).mean())


def gate(results: dict) -> tuple[bool, list[str]]:
    """Reproduction gate: same k, label agreement >= 0.90, max relative daily-prediction gap <= 3 %."""
    reasons: list[str] = []
    if results["k_live"] != results["k_harness"]:
        reasons.append(f"k differs: live {results['k_live']} vs harness {results['k_harness']}")
    if results["agreement"] < GATE_MIN_AGREEMENT:
        reasons.append(f"label agreement {results['agreement']:.3f} < {GATE_MIN_AGREEMENT}")
    if results["max_rel_pred_diff"] > GATE_MAX_PRED_DIFF:
        reasons.append(f"max daily prediction gap {results['max_rel_pred_diff']:.4f} > {GATE_MAX_PRED_DIFF}")
    return (not reasons), reasons


# ── Gate run ──────────────────────────────────────────────────────────────────


def _versions() -> dict:
    out = {"python": sys.version.split()[0]}
    for name in ("sklearn", "lightgbm", "pandas", "numpy"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _next_day_total(pred: pd.DataFrame, origin: pd.Timestamp) -> float:
    ts = pd.to_datetime(pred["timestamp"])
    day = (origin.floor("1D") + pd.Timedelta(days=1)).date()
    sel = pred.loc[ts.dt.date == day, "predicted_kwh"]
    if len(sel) != 24:
        raise ValueError(f"expected 24 predicted hours on {day}, got {len(sel)}")
    return float(sel.sum())


def _predict_totals(model, origins: list[pd.Timestamp], inputs_by_origin: dict) -> tuple[list[float], list[dict]]:
    """Next-calendar-day totals per origin, plus the regime label/probabilities the model used for that day."""
    totals, regimes = [], []
    for o in origins:
        kw = dict(inputs_by_origin[o])
        with rh.frozen_clock(o):  # the model decides which hours are "future" from pd.Timestamp.now()
            prepared = model._prepare_prediction_X(**kw)
            pred = model.predict(**kw, _prepared=prepared)
        totals.append(_next_day_total(pred, o))
        info = (getattr(model, "_last_regime_info", None) or {}).get((o.floor("1D") + pd.Timedelta(days=1)).date())
        regimes.append(info or {})
    return totals, regimes


def run_gate(data_dir: Path, root: Path, n_origins: int = 7) -> dict:
    from energy_forecast.model import EnergyForecastModel, _strip_partial_last_day

    root.mkdir(parents=True, exist_ok=True)
    live_dir = data_dir / "models"
    meta = pickle.loads((live_dir / "meta.pkl").read_bytes())
    cutoff = pd.Timestamp(meta["last_trained"])  # naive LOCAL, no conversion (see ROADMAP #110 notes)
    cfg = rh.load_config(data_dir)
    archive = rh.load_weather_archive(
        root / "archive", float(cfg["latitude"]), float(cfg["longitude"]), (cutoff - pd.Timedelta(days=800)).date()
    )

    captured = rh.load_live_equivalent_training_frame(
        cutoff, data_dir, root / "train", archive=archive, seed_physics_state=True
    )
    args = captured["_args"]
    kwargs = {k: v for k, v in captured.items() if not k.startswith("_")}

    hdir = root / "harness_model"
    shutil.rmtree(hdir, ignore_errors=True)
    harness = EnergyForecastModel(hdir, timezone=cfg.get("timezone") or rh.TZ)
    harness.train(*args, **kwargs)

    ldir = root / "live_model"
    shutil.rmtree(ldir, ignore_errors=True)
    shutil.copytree(live_dir, ldir)
    live = EnergyForecastModel(ldir, timezone=cfg.get("timezone") or rh.TZ)

    energy = _strip_partial_last_day(args[0])
    lc, hc = live._clusterer.centroids, harness._clusterer.centroids
    live_labels = nearest_centroid_labels(energy, lc)
    harness_labels = nearest_centroid_labels(energy, hc)
    agreement = label_agreement(lc, hc, live_labels, harness_labels)

    origins = [cutoff.floor("1D") - pd.Timedelta(days=i) + pd.Timedelta(hours=17) for i in range(n_origins + 1)]
    origins = sorted(o for o in origins if o <= cutoff)[-n_origins:]
    inputs = {
        o: rh.build_prediction_inputs(
            o, data_dir, root / f"origin_{o:%Y%m%d}", archive=archive, seed_physics_state=True
        )
        for o in origins
    }
    live_tot, live_reg = _predict_totals(live, origins, inputs)
    harn_tot, harn_reg = _predict_totals(harness, origins, inputs)
    rel = [abs(h - lt) / lt for h, lt in zip(harn_tot, live_tot)]

    results = {
        "cutoff": str(cutoff),
        "commit": _git_commit(),
        "versions_local": _versions(),
        "versions_live": "NOT RECORDED -- read from the AppDaemon log / check_ha_logs.py",
        "k_live": int(lc.shape[0]),
        "k_harness": int(hc.shape[0]),
        "centroid_daysums_live": sorted(round(float(x), 2) for x in lc.sum(axis=1)),
        "centroid_daysums_harness": sorted(round(float(x), 2) for x in hc.sum(axis=1)),
        "agreement": agreement,
        "origins": [str(o) for o in origins],
        "next_day_total_live": live_tot,
        "next_day_total_harness": harn_tot,
        "rel_diff": rel,
        "regime_live": live_reg,
        "regime_harness": harn_reg,
        "max_rel_pred_diff": max(rel),
        "caveats": [
            "away/presence HA history absent in the harness (get_history -> []); live had 30 days of both",
            "30-day HA energy resync absent (cache used as pulled)",
            "perfect weather forecast at every origin; post-predict DHW override excluded",
        ],
    }
    results["gate_pass"], results["gate_reasons"] = gate(results)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="reproduce the live clusterer/model before any backtest counts")
    g.add_argument("--data-dir", type=Path, default=REPO / "data")
    g.add_argument("--root", type=Path, required=True, help="scratch directory (not shared across runs)")
    g.add_argument("--out", type=Path, default=REPO / "data" / "backtest_regime_gate.json")
    args = ap.parse_args(argv)
    if args.cmd == "gate":
        res = run_gate(args.data_dir, args.root)
        args.out.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: v for k, v in res.items() if k != "caveats"}, indent=2, default=str))
        return 0 if res["gate_pass"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
