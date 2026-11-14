"""Interval checks on walk-forward rows: what the published bands cover, and what recalibration would give.

The app publishes block and daily bands as sums of hourly P10 / P90 (``energy_forecast.py`` ``_isum``). Quantiles
do not add, so those bands are wider than the hourly ones are narrow; ``summed_band_coverage`` measures it.
``recalibrate_hourly`` replaces the in-sample conformal correction with a trailing one on served errors.

Input: day-ahead rows from ``wf_score.day_ahead`` with ``y`` attached, one arm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = 0.8


def _coverage(y: pd.Series, lo: pd.Series, hi: pd.Series) -> dict:
    return dict(
        n=len(y),
        cover=float(((y >= lo) & (y <= hi)).mean()),
        below=float((y < lo).mean()),
        above=float((y > hi).mean()),
        width=float((hi - lo).mean()),
    )


def summed_band_coverage(rows: pd.DataFrame, hours: int = 3) -> dict:
    """Coverage of the published construction for complete *hours*-long blocks (24 = whole days)."""
    x = rows.dropna(subset=["y", "lo", "hi"])
    key = x["ts"].dt.floor(f"{hours}h") if hours < 24 else x["ts"].dt.normalize()
    g = x.groupby(key).agg(n=("y", "size"), y=("y", "sum"), lo=("lo", "sum"), hi=("hi", "sum"))
    g = g[g["n"] >= (hours if hours < 24 else 22)]
    return _coverage(g["y"], g["lo"], g["hi"])


def recalibrate_hourly(rows: pd.DataFrame, window_days: int = 21, min_days: int = 10) -> pd.DataFrame:
    """Add ``lo_c`` / ``hi_c``: bounds widened by a conformal correction learned on the previous days' served errors.

    Same form as the app's correction (symmetric, in log1p space, 80 % level), but the scores come from forecasts
    as they were served, not from training rows. Days without *min_days* of history get NaN.
    """
    x = rows.dropna(subset=["y", "lo", "hi"]).sort_values("ts").copy()
    x["day"] = x["ts"].dt.normalize()
    x["lo_c"], x["hi_c"] = np.nan, np.nan
    score = np.maximum(np.log1p(x["lo"]) - np.log1p(x["y"]), np.log1p(x["y"]) - np.log1p(x["hi"]))
    for day in x["day"].unique():
        hist = (x["day"] < day) & (x["day"] >= day - pd.Timedelta(days=window_days))
        if x.loc[hist, "day"].nunique() < min_days:
            continue
        q = float(np.quantile(score[hist], TARGET))
        cur = x["day"] == day
        x.loc[cur, "lo_c"] = np.expm1(np.log1p(x.loc[cur, "lo"]) - q).clip(lower=0)
        x.loc[cur, "hi_c"] = np.expm1(np.log1p(x.loc[cur, "hi"]) + q)
    return x


def report(rows: pd.DataFrame, window_days: int = 21) -> pd.DataFrame:
    """One table: hourly as shipped, hourly recalibrated (same hours), summed 3 h blocks, summed days."""
    r = recalibrate_hourly(rows, window_days).dropna(subset=["lo_c"])
    out = {
        "hourly, as shipped": _coverage(r["y"], r["lo"], r["hi"]),
        "hourly, recalibrated on served errors": _coverage(r["y"], r["lo_c"], r["hi_c"]),
        "3 h blocks, summed hourly quantiles": summed_band_coverage(rows, 3),
        "days, summed hourly quantiles": summed_band_coverage(rows, 24),
    }
    return pd.DataFrame(out).T
