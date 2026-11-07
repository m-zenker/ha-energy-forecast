"""CLI: ``python -m backtest <experiment> [options]`` — see ``--help`` and backtest/README.md.

Run from the repo root with the project env, e.g.
``/home/jovyan/my_envs/ha-energy-forecast/bin/python -m backtest temp_ablation --repro --seeds 1 2``.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd

from .arms import Arm, noise_floor_arms
from .data import DEFAULT_FOLDS, load_config, load_inputs, load_weather, train_defaults
from .env import DATA_DIR
from .repro import gate_folds, load_live_pred, replay_series, repro_report
from .runner import run_experiment
from .score import CAVEATS, format_summary, summarize


def _parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m backtest", description=__doc__)
    ap.add_argument("experiment", help="module name under backtest/experiments/")
    ap.add_argument("--arms", nargs="*", help="run only these arms (the reference arm is always kept)")
    ap.add_argument("--folds", nargs="*", help="run only these folds")
    ap.add_argument("--seeds", nargs="*", type=int, default=[], help="add <base>@s<N> seed noise-floor arms")
    ap.add_argument("--smoke", action="store_true", help="last fold, first 2 days")
    ap.add_argument("--repro", action="store_true", help="compare a lead-25 base replay with live pred_history")
    ap.add_argument("--out", type=Path, help="result pickle (default data/backtest/<experiment>_<stamp>.pkl)")
    ap.add_argument("--no-fetch", action="store_true", help="don't fetch the recent Open-Meteo weather tail")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    try:
        exp = importlib.import_module(f"backtest.experiments.{args.experiment}")
    except ModuleNotFoundError:
        print(f"unknown experiment {args.experiment!r} (no backtest/experiments/{args.experiment}.py)", file=sys.stderr)
        return 2
    arms: list[Arm] = list(exp.ARMS)
    if not arms or arms[0].name != "base":
        print(f"{args.experiment}: ARMS[0] must be the reference arm named 'base'", file=sys.stderr)
        return 2
    if args.arms:
        arms = [a for i, a in enumerate(arms) if i == 0 or a.name in args.arms]
    arms += noise_floor_arms(arms[0], args.seeds)
    folds = list(getattr(exp, "FOLDS", DEFAULT_FOLDS))
    if args.folds:
        folds = [f for f in folds if f.name in args.folds]
    days_limit = None
    if args.smoke:
        folds, days_limit = folds[-1:], 2

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)  # app logs one line per predict (lag_24h NaN fill, regime fallback) — noise here
    inputs = load_inputs()
    weather = load_weather(fetch_tail=not args.no_fetch)
    defaults = train_defaults(load_config())
    header: list[str] = []
    if args.repro:
        live_pred, actuals = load_live_pred()
        rows = run_experiment([Arm("base")], gate_folds(live_pred), inputs, weather, defaults, origins="lead25")
        header += repro_report(replay_series(rows), live_pred, actuals).lines()

    out = args.out or DATA_DIR / "backtest" / f"{args.experiment}_{pd.Timestamp.now():%Y%m%d-%H%M}.pkl"
    origins = getattr(exp, "ORIGINS", (6, 18))
    df = run_experiment(arms, folds, inputs, weather, defaults, origins=origins, out=out, days_limit=days_limit)
    if df.empty:
        print("\n".join([*CAVEATS, *header, "no scored days in the selected folds"]))
        return 1
    print(format_summary(summarize(df), header))
    print(f"\nrows: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
