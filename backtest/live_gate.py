"""Relative cold-day gate (ROADMAP #118) on the live ``pred_history.json``.

Reads the forecasts and actuals the live app stored, scores them with ``backtest.wf_score`` and prints
PASS / FAIL / INCONCLUSIVE: pooled cold-day bias within -12 % .. +6 % of consumption over >= 14 cold days.

Usage:
  PYTHONPATH=apps:. /home/jovyan/my_envs/ha-energy-forecast/bin/python -m backtest.live_gate data \
      --since "2026-11-05 17:28"

``pred_history`` keeps the first forecast made within 25 h of a target hour, so hours stored before the change
under test keep their old value: with ``--since`` only target hours >= anchor + 24 h (rounded up to the hour) count.
Actuals must be on the training-target basis (marker ``"actuals_basis": "target"``, written by #117); without the
marker the verdict is INCONCLUSIVE unless ``--allow-mixed-actuals`` is given.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from . import wf_score
from .env import DATA_DIR

ARM = "live"
PRED_HISTORY = "pred_history.json"
TARGET_BASIS = "target"
_METRICS = ("cold_bias", "cold_bias_pct", "mae", "mae3", "daily_abs", "bias")


def _to_series(values: dict) -> pd.Series:
    s = pd.Series(values, dtype=float)
    s.index = pd.DatetimeIndex(pd.to_datetime(list(values)))
    return s.sort_index()


def load_pred_history(path: Path) -> tuple[pd.Series, pd.Series, str | None]:
    """``(pred, actual, actuals_basis)`` from a ``pred_history.json``; both Series hourly, naive local, sorted."""
    raw = json.loads(Path(path).read_text())
    return _to_series(raw.get("pred", {})), _to_series(raw.get("actuals", {})), raw.get("actuals_basis")


def first_scored_hour(since: pd.Timestamp) -> pd.Timestamp:
    """First target hour whose stored forecast was made after *since* (see module docstring)."""
    return (pd.Timestamp(since) + pd.Timedelta(hours=24)).ceil("1h")


def live_rows(
    pred: pd.Series,
    actual: pd.Series,
    daily_temp: pd.Series,
    since: pd.Timestamp | None = None,
    cold_below: float = wf_score.COLD_C,
) -> pd.DataFrame:
    """One row per forecast hour: ``arm, ts, p50, y, naive, day, cold`` (``cold`` False without a temperature)."""
    rows = pd.DataFrame({"arm": ARM, "ts": pred.index, "p50": pred.to_numpy()})
    if since is not None:
        rows = rows[rows["ts"] >= first_scored_hour(since)].reset_index(drop=True)
    if len(actual):
        rows = wf_score.attach_actuals(rows, actual, daily_temp)
    else:  # no actuals stored yet (fresh install, or pred_history deleted after a rollback): nothing to score
        rows = rows.assign(y=np.nan, naive=np.nan, day=rows["ts"].dt.normalize())
    rows["cold"] = (rows["day"].map(daily_temp) <= cold_below).fillna(False).astype(bool)
    return rows


def missing_temperature_days(rows: pd.DataFrame, daily_temp: pd.Series) -> pd.DatetimeIndex:
    """Forecast days without a daily temperature: they can never be cold days, so the caller should say so."""
    days = pd.DatetimeIndex(rows["day"].unique())
    return days[~days.isin(daily_temp.dropna().index)].sort_values()


def _summary(rows: pd.DataFrame) -> pd.DataFrame:
    return wf_score.summarize(wf_score.day_table(wf_score.add_naive(rows, ref=ARM)))


def verdict(
    rows: pd.DataFrame,
    actuals_basis: str | None,
    low_pct: float = -12.0,
    high_pct: float = 6.0,
    min_days: int = 14,
    allow_mixed_actuals: bool = False,
) -> dict:
    """Gate verdict plus the numbers behind it. ``reason`` is empty on PASS."""
    summary = _summary(rows)
    out = {"gate": "INCONCLUSIVE", "reason": "", "cold_days": 0, "days": 0}
    out.update(dict.fromkeys((*_METRICS, "naive_mae", "naive_mae3"), np.nan))
    row = None
    if ARM in summary.index:
        row = summary.loc[ARM]
        out["days"] = int(row["days"])
        out["cold_days"] = 0 if pd.isna(row["cold_days"]) else int(row["cold_days"])
        out.update({k: float(row[k]) for k in _METRICS})
    if "naive" in summary.index:
        out["naive_mae"] = float(summary.loc["naive", "mae"])
        out["naive_mae3"] = float(summary.loc["naive", "mae3"])

    if actuals_basis != TARGET_BASIS and not allow_mixed_actuals:
        out["reason"] = "actuals not on the target basis (#117 not live or migration pending)"
        return out
    gate = wf_score.cold_gate(row, low_pct, high_pct, min_days) if row is not None else "INCONCLUSIVE"
    out["gate"] = gate
    if gate == "INCONCLUSIVE":
        out["reason"] = f"only {out['cold_days']} cold day(s), need {min_days}"
    elif gate == "FAIL":
        out["reason"] = f"cold-day bias {out['cold_bias_pct']:+.1f} % outside [{low_pct:g}, {high_pct:+g}] %"
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backtest.live_gate", description=__doc__.splitlines()[0])
    ap.add_argument("data_dir", nargs="?", type=Path, default=DATA_DIR, help="directory with pred_history.json")
    ap.add_argument("--since", help="local timestamp of the deploy or retrain under test")
    ap.add_argument("--cold-below", type=float, default=wf_score.COLD_C, help="cold day: daily mean temp <= this")
    ap.add_argument("--min-days", type=int, default=14, help="cold days needed for PASS / FAIL")
    ap.add_argument("--allow-mixed-actuals", action="store_true", help="score without the actuals_basis marker")
    ap.add_argument("--no-fetch", action="store_true", help="cached weather only, no Open-Meteo tail")
    args = ap.parse_args(argv)

    from .data import load_weather

    pred, actual, basis = load_pred_history(args.data_dir / PRED_HISTORY)
    weather = load_weather(args.data_dir, fetch_tail=not args.no_fetch)
    daily_temp = weather.set_index("timestamp")["temp_c"].resample("D").mean()
    since = pd.Timestamp(args.since) if args.since else None
    rows = live_rows(pred, actual, daily_temp, since=since, cold_below=args.cold_below)
    no_temp = missing_temperature_days(rows, daily_temp)

    if since is not None:
        print(f"anchor {since}  ->  first scored hour {first_scored_hour(since)}")
    print(f"actuals_basis: {basis!r}   cold day: daily mean <= {args.cold_below:g} °C")
    table = wf_score.day_table(wf_score.add_naive(rows, ref=ARM))
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.3f}".format):
        if ARM in table.index.get_level_values("arm"):
            print(table.loc[ARM][["n", "mae", "mae3", "bias", "y", "cold"]].to_string())
        else:
            print("no day with enough scored hours")
        summary = wf_score.summarize(table)
        if len(summary):
            print(summary.reindex([a for a in (ARM, "naive") if a in summary.index]).to_string())
    if len(no_temp):
        print(
            f"WARNING: {len(no_temp)} forecast day(s) from {no_temp.min():%Y-%m-%d} to {no_temp.max():%Y-%m-%d} have "
            f"no temperature (weather ends {daily_temp.dropna().index.max():%Y-%m-%d}) and are NOT counted as cold"
        )
    v = verdict(rows, basis, min_days=args.min_days, allow_mixed_actuals=args.allow_mixed_actuals)
    print(f"GATE: {v['gate']}" + (f" — {v['reason']}" if v["reason"] else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
