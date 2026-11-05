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


# ── Decision rule (§6.4) + jitter control ─────────────────────────────────────

POOLED_TOL = 1.03  # candidate pooled daily MAE <= 1.03 x H
PRIMARY_TOL = 1.05  # candidate primary-fold MAE <= 1.05 x H
N_WINS_POOLED = 0.03  # N beats T only if >= 3 % better pooled ...
N_WINS_JAN = 0.05  # ... or >= 5 % better in the January fold


def _noise(r: dict) -> dict:
    """Relative spread between H and its jitter control J (H retrained at cutoff - 1 h); 0 without a control."""
    j, h = r.get("J"), r["H"]
    if not j:
        return {"pooled_mae": 0.0, "max_delta": 0.0}
    return {
        "pooled_mae": abs(j["pooled_mae"] - h["pooled_mae"]) / h["pooled_mae"],
        "max_delta": abs(j["max_delta"] - h["max_delta"]) / h["max_delta"] if h["max_delta"] else 0.0,
    }


def _passes(c: dict, h: dict, noise: dict) -> tuple[bool, list[str]]:
    why = []
    if not c["max_delta"] < h["max_delta"]:
        why.append("max |delta| not strictly lower than H")
    elif (h["max_delta"] - c["max_delta"]) / h["max_delta"] <= noise["max_delta"]:
        why.append(f"max |delta| gain within retrain jitter ({noise['max_delta']:.1%})")
    if c["n_over_2"] > h["n_over_2"]:
        why.append("more days with |delta| > 2 kWh than H")
    if c["pooled_mae"] > POOLED_TOL * h["pooled_mae"]:
        why.append("pooled MAE > 1.03 x H")
    if c["primary_mae"] > PRIMARY_TOL * h["primary_mae"]:
        why.append("primary-fold MAE > 1.05 x H")
    return not why, why


def decide_with_reasons(r: dict) -> tuple[str, list[str]]:
    """T / N / H per the plan's rule. ``r[arm]`` = pooled_mae, primary_mae, jan_mae, max_delta, n_over_2.

    A stability gain smaller than the retrain jitter (H vs ``r["J"]``) does not count, and N must beat T by more
    than max(3 % pooled / 5 % January, jitter) to be preferred.
    """
    noise = _noise(r)
    h = r["H"]
    ok: dict[str, bool] = {}
    notes: list[str] = []
    for arm in ("T", "N"):
        if arm not in r:
            notes.append(f"{arm}: not run")
            ok[arm] = False
            continue
        ok[arm], why = _passes(r[arm], h, noise)
        notes += [f"{arm}: {w}" for w in why]
    if ok["T"] and ok["N"]:
        t, n = r["T"], r["N"]
        pooled_gain = (t["pooled_mae"] - n["pooled_mae"]) / t["pooled_mae"]
        jan_gain = (t["jan_mae"] - n["jan_mae"]) / t["jan_mae"]
        if pooled_gain >= max(N_WINS_POOLED, noise["pooled_mae"]) or jan_gain >= max(N_WINS_JAN, noise["pooled_mae"]):
            return "N", notes + [f"N beats T (pooled {pooled_gain:.1%}, Jan {jan_gain:.1%})"]
        return "T", notes + ["both pass; N not clearly better"]
    if ok["T"]:
        return "T", notes
    if ok["N"]:
        return "N", notes
    return "H", notes + ["no candidate passes -> keep H"]


def decide(r: dict) -> str:
    return decide_with_reasons(r)[0]


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


# ── Backtest (C3) ─────────────────────────────────────────────────────────────

# Target-day ranges (inclusive). Origins are the day before each target, at 17:00 local.
FOLDS = {"primary": ("2026-10-15", None), "jan": ("2026-01-05", "2026-01-31")}
SUN_FACTORS = (0.66, 1.5)
CUTOFF_HOUR = 6  # weekly retrain at 06:00 local, like the live early-morning retrain
ARMS = ("H", "T", "N", "J")  # J = H retrained one hour earlier: the retrain-jitter control

# Arm -> (cutoff offset in hours, train() overrides). Variants of T (#110): TJ = T's own jitter control,
# TB = blocked out-of-fold folds.
ARM_SPECS: dict[str, tuple[int, dict]] = {
    "H": (0, {"regime_mode": "hard"}),
    "T": (0, {"regime_mode": "soft_oof", "regime_oof_cv": "shuffled"}),
    "N": (0, {"enable_regimes": False}),
    "J": (-1, {"regime_mode": "hard"}),
    "TJ": (-1, {"regime_mode": "soft_oof", "regime_oof_cv": "shuffled"}),
    "TB": (0, {"regime_mode": "soft_oof", "regime_oof_cv": "blocked"}),
}


def weekly_cutoffs(first_origin_day: pd.Timestamp, last_origin_day: pd.Timestamp) -> list[pd.Timestamp]:
    out, d = [], first_origin_day.normalize()
    while d <= last_origin_day:
        out.append(d + pd.Timedelta(hours=CUTOFF_HOUR))
        d += pd.Timedelta(days=7)
    return out


def origins_for(cutoff: pd.Timestamp, last_origin_day: pd.Timestamp) -> list[pd.Timestamp]:
    days = pd.date_range(cutoff.normalize(), min(cutoff.normalize() + pd.Timedelta(days=6), last_origin_day))
    return [d + pd.Timedelta(hours=17) for d in days]


def arm_cutoff(arm: str, cutoff: pd.Timestamp) -> pd.Timestamp:
    return cutoff + pd.Timedelta(hours=ARM_SPECS[arm][0])


def scale_sunshine(inputs: dict, origin: pd.Timestamp, factor: float) -> dict:
    out = dict(inputs)
    fc = inputs["forecast_df"].copy()
    future = pd.to_datetime(fc["timestamp"]) >= origin.floor("1h")
    fc.loc[future, "sunshine_min"] = fc.loc[future, "sunshine_min"] * factor
    out["forecast_df"] = fc
    return out


def fold_metrics(df: pd.DataFrame) -> dict:
    """Metrics for one arm. ``df`` rows: fold, base, lo, hi, actual (NaN when the day cannot be scored)."""
    sc = df[df["actual"].notna()]
    err = (sc["base"] - sc["actual"]).abs()
    delta = pd.concat([(sc["lo"] - sc["base"]).abs(), (sc["hi"] - sc["base"]).abs()], axis=1).max(axis=1)
    return {
        "pooled_mae": float(err.mean()),
        "primary_mae": float(err[sc["fold"] == "primary"].mean()),
        "jan_mae": float(err[sc["fold"] == "jan"].mean()),
        "max_delta": float(delta.max()),
        "n_over_2": int((delta > 2).sum()),
        "n_scored": int(len(sc)),
        "mean_bias": float((sc["base"] - sc["actual"]).mean()),
    }


def run_backtest(
    data_dir: Path, root: Path, arms: tuple[str, ...] | None = None, folds: tuple[str, ...] | None = None
) -> dict:
    import copy
    import inspect

    from energy_forecast.model import EnergyForecastModel

    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    cfg = rh.load_config(data_dir)
    tz = cfg.get("timezone") or rh.TZ
    live_cut = pd.Timestamp(pickle.loads((data_dir / "models" / "meta.pkl").read_bytes())["last_trained"])
    archive = rh.load_weather_archive(
        root / "archive", float(cfg["latitude"]), float(cfg["longitude"]), (live_cut - pd.Timedelta(days=800)).date()
    )
    train_params = set(inspect.signature(EnergyForecastModel.train).parameters)
    # Arms whose overrides this train() does not support (e.g. running without Part B) are skipped.
    arms = [a for a in (arms or ARMS) if set(ARM_SPECS[a][1]) - {"enable_regimes"} <= train_params]
    folds = tuple(folds or FOLDS)
    print("arms:", arms, "folds:", folds, flush=True)

    # Scoring target: corrected training-target frame at the live cutoff (hours with EV removed are absent).
    ref = rh.load_live_equivalent_training_frame(live_cut, data_dir, root / "ref", archive=archive)["_args"][0]
    actual_h = ref.set_index(pd.to_datetime(ref["timestamp"]))["gross_kwh"]
    last_complete = (actual_h.index.max() - pd.Timedelta(hours=23)).normalize()  # last day with a full 24 h
    if actual_h.index.max().hour != 23:
        last_complete = actual_h.index.max().normalize() - pd.Timedelta(days=1)

    def day_actual(day: pd.Timestamp) -> float:
        h = actual_h.reindex(pd.date_range(day, periods=24, freq="1h"))
        return float(h.sum()) if h.notna().all() else float("nan")

    rows: dict[str, list[dict]] = {a: [] for a in arms}
    inputs_cache: dict[pd.Timestamp, dict] = {}
    for fold, (start, end) in ((f, FOLDS[f]) for f in folds):
        t0 = pd.Timestamp(start)
        t1 = pd.Timestamp(end) if end else last_complete
        assert (t1 - t0).days + 1 >= (21 if fold == "primary" else 20), f"fold {fold} too short"
        first_o, last_o = t0 - pd.Timedelta(days=1), t1 - pd.Timedelta(days=1)
        for cutoff in weekly_cutoffs(first_o, last_o):
            origins = origins_for(cutoff, last_o)
            for o in origins:
                if o not in inputs_cache:
                    inputs_cache[o] = rh.build_prediction_inputs(
                        o, data_dir, root / f"origin_{o:%Y%m%d}", archive=archive
                    )
            for arm in arms:
                cf = cache / f"{fold}_{cutoff:%Y%m%d}_{arm}.pkl"
                if cf.exists():
                    rows[arm] += pickle.loads(cf.read_bytes())
                    continue
                cap = rh.load_live_equivalent_training_frame(
                    arm_cutoff(arm, cutoff), data_dir, root / f"tr_{fold}_{cutoff:%Y%m%d}_{arm}", archive=archive
                )
                kw = {k: v for k, v in copy.copy(cap).items() if not k.startswith("_")}
                kw.update(ARM_SPECS[arm][1])
                mdir = root / f"model_{fold}_{cutoff:%Y%m%d}_{arm}"
                shutil.rmtree(mdir, ignore_errors=True)
                model = EnergyForecastModel(mdir, timezone=tz)
                model.train(*cap["_args"], **kw)
                cell = []
                for o in origins:
                    totals = {}
                    for name, factor in (("base", 1.0), ("lo", SUN_FACTORS[0]), ("hi", SUN_FACTORS[1])):
                        # Predict with the arm's own trained physics model, as the live app does.
                        inp = {**inputs_cache[o], "physics_model": cap["physics_model"]}
                        inp = inp if factor == 1.0 else scale_sunshine(inp, o, factor)
                        with rh.frozen_clock(o):
                            pred = model.predict(**inp, _prepared=model._prepare_prediction_X(**inp))
                        totals[name] = _next_day_total(pred, o)
                    target = o.normalize() + pd.Timedelta(days=1)
                    actual = day_actual(target) if target <= last_complete else float("nan")
                    cell.append(
                        {
                            "fold": fold,
                            "cutoff": str(cutoff),
                            "origin": str(o),
                            **totals,
                            "actual": actual,
                            "oof_acc": getattr(model, "_regime_oof_acc", None),
                            "oof_majority": getattr(model, "_regime_oof_majority", None),
                            "k": int(model._clusterer.centroids.shape[0])
                            if getattr(model, "_clusterer", None)
                            else None,
                        }
                    )
                cf.write_bytes(pickle.dumps(cell))
                rows[arm] += cell
                print(f"[{fold} {cutoff:%m-%d} {arm}] done", flush=True)

    frames = {a: pd.DataFrame(r) for a, r in rows.items()}
    metrics = {a: fold_metrics(f) for a, f in frames.items()}
    full = {"H", "T", "N"} <= set(arms) and set(folds) == set(FOLDS)
    verdict, reasons = decide_with_reasons(metrics) if full else (None, ["variant run: no decision"])
    return {
        "commit": _git_commit(),
        "versions_local": _versions(),
        "live_cutoff": str(live_cut),
        "last_complete_target_day": str(last_complete.date()),
        "folds": {k: list(v) for k, v in FOLDS.items()},
        "metrics": metrics,
        "verdict": verdict,
        "reasons": reasons,
        "per_origin": {a: f.to_dict("records") for a, f in frames.items()},
        "caveats": [
            "gate: clusterer reproduced exactly, but daily totals differ from live by up to 14 % (retrain jitter ~9 %)",
            "scored against the corrected training-target frame; EV-masked / partial days are not scored",
            "away/presence history and 30-day resync absent; perfect weather forecast; physics state unseeded",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate", help="reproduce the live clusterer/model before any backtest counts")
    g.add_argument("--data-dir", type=Path, default=REPO / "data")
    g.add_argument("--root", type=Path, required=True, help="scratch directory (not shared across runs)")
    g.add_argument("--out", type=Path, default=REPO / "data" / "backtest_regime_gate.json")
    b = sub.add_parser("backtest", help="H / T / N (+ jitter control) backtest and decision")
    b.add_argument("--data-dir", type=Path, default=REPO / "data")
    b.add_argument("--root", type=Path, required=True, help="scratch directory (resumable: per-cell cache)")
    b.add_argument("--out", type=Path, default=REPO / "data" / "backtest_regime_results.json")
    b.add_argument("--arms", nargs="+", choices=sorted(ARM_SPECS), help="default: H T N J")
    b.add_argument("--folds", nargs="+", choices=sorted(FOLDS), help="default: all")
    args = ap.parse_args(argv)
    if args.cmd == "backtest":
        res = run_backtest(args.data_dir, args.root, tuple(args.arms or ()) or None, tuple(args.folds or ()) or None)
        args.out.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: v for k, v in res.items() if k not in ("per_origin", "caveats")}, indent=2, default=str))
        return 0
    if args.cmd == "gate":
        res = run_gate(args.data_dir, args.root)
        args.out.write_text(json.dumps(res, indent=2, default=str))
        print(json.dumps({k: v for k, v in res.items() if k != "caveats"}, indent=2, default=str))
        return 0 if res["gate_pass"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
