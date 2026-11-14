"""Block, daily and naive-guardrail metrics for the live accuracy sensors.

The per-hour definitions (error = pred - actual, clock-aligned complete 3 h blocks, days with >= 22 hours, naive =
mean of the same hour on days -2..-8) mirror ``backtest/wf_score.py``. Tests pin ``naive_same_hour_mean`` and the
block/hourly numbers on complete days against it. Aggregation differs: live pools ALL complete blocks in the window,
including blocks on days that are not scored (e.g. EV days); ``wf_score.day_table`` scores blocks only on days with
>= 22 hours and averages per-day means; ``backtest/runner._row`` averages per (day, origin) with an 18 h gate. All
agree when every day is complete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BLOCK_HOURS = 3
MIN_DAY_HOURS = 22
NAIVE_DAYS = range(2, 9)

_COUNT_KEYS = ("n_hours", "n_blocks", "n_days")
_NAN_KEYS = (
    "mae",
    "naive_mae",
    "skill_pct",
    "block_mae",
    "naive_block_mae",
    "block_skill_pct",
    "daily_abs_error",
    "daily_bias",
    "daily_bias_pct",
)


def naive_same_hour_mean(actual: pd.Series) -> pd.Series:
    """Mean of the same hour on days -2..-8 (NaN-skipping), on *actual*'s own index."""
    if actual.empty:
        return pd.Series(dtype=float)
    grid = actual.reindex(pd.date_range(actual.index.min(), actual.index.max(), freq="h"))
    stacked = pd.concat([grid.shift(24 * k) for k in NAIVE_DAYS], axis=1)
    return stacked.mean(axis=1).reindex(actual.index)


def _hourly_series(history: dict) -> pd.Series:
    """Dict of timestamp-like keys to values as a float Series on a floored hourly index (last value wins)."""
    if not history:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    idx = pd.DatetimeIndex([pd.Timestamp(k) for k in history]).floor("h")
    s = pd.Series(list(history.values()), index=idx, dtype=float)
    return s[~s.index.duplicated(keep="last")].sort_index()


def paired_errors(pred_history: dict, actuals_history: dict, since: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per hour in both histories (``ts >= since``); columns pred, actual, naive.

    ``naive`` is computed from all of *actuals_history*, before the *since* filter.
    """
    pred = _hourly_series(pred_history)
    actual = _hourly_series(actuals_history)
    naive = naive_same_hour_mean(actual)
    hours = pred.index.intersection(actual.index)
    if since is not None:
        hours = hours[hours >= pd.Timestamp(since)]
    hours = hours.sort_values()
    return pd.DataFrame(
        {"pred": pred.reindex(hours), "actual": actual.reindex(hours), "naive": naive.reindex(hours)},
        index=hours,
    )


def _skill(err: pd.Series, naive_err: pd.Series) -> float:
    """100 * (1 - mean|err| / mean|naive_err|) on rows where naive is present; NaN if none or naive MAE is 0."""
    ok = naive_err.notna()
    if not ok.any():
        return float("nan")
    denom = naive_err[ok].abs().mean()
    if denom == 0:
        return float("nan")
    return float(100.0 * (1.0 - err[ok].abs().mean() / denom))


def summarize(pairs: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {k: 0 for k in _COUNT_KEYS}
    out.update({k: float("nan") for k in _NAN_KEYS})
    if pairs.empty:
        return out

    e = pairs["pred"] - pairs["actual"]
    en = pairs["naive"] - pairs["actual"]
    out["n_hours"] = int(len(pairs))
    out["mae"] = float(e.abs().mean())
    if en.notna().any():
        out["naive_mae"] = float(en.abs().mean())
    out["skill_pct"] = _skill(e, en)

    frame = pd.DataFrame({"e": e, "en": en, "actual": pairs["actual"]})
    blocks = frame.groupby(frame.index.floor(f"{BLOCK_HOURS}h"))
    complete = blocks["e"].count() == BLOCK_HOURS
    be = blocks["e"].sum()[complete]
    out["n_blocks"] = int(complete.sum())
    if len(be):
        out["block_mae"] = float(be.abs().mean())
        naive_ok = complete & (blocks["en"].count() == BLOCK_HOURS)
        ben = blocks["en"].sum()[naive_ok]
        if len(ben):
            out["naive_block_mae"] = float(ben.abs().mean())
            out["block_skill_pct"] = _skill(be[naive_ok], ben)

    days = frame.groupby(frame.index.normalize())
    scored = days["e"].count() >= MIN_DAY_HOURS
    out["n_days"] = int(scored.sum())
    if scored.any():
        de = days["e"].sum()[scored]
        out["daily_bias"] = float(de.mean())
        out["daily_abs_error"] = float(de.abs().mean())
        total_actual = float(days["actual"].sum()[scored].sum())
        if total_actual != 0 and not np.isnan(total_actual):
            out["daily_bias_pct"] = float(100.0 * de.sum() / total_actual)
    return out
