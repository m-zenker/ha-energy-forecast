"""Score walk-forward rows on the metrics the consumer cares about: 3 h blocks, daily totals, intervals, swing.

Hourly MAE is reported but is not the primary number: it rewards under-forecasting (scaling a forecast by 0.8
improved hourly MAE by 6 % while pushing daily bias to -2.5 kWh/day in the 2026-11-13 audit), and
ha-energy-manager consumes 3 h block energy with block P10/P90.

Typical use::

    rows = pd.concat(pd.read_pickle(p) for p in Path("data/backtest/wf").glob("wf_*.pkl"))
    rows = attach_actuals(rows, frame.actuals(), daily_temp)
    table = day_table(add_naive(day_ahead(rows)))
    print(summarize(table)); print(paired(table, "droptp"))

Count an effect only when its CI excludes 0 *and* it is larger than the ``jit`` arm's distance from ``base``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLD_C = 10.0
MIN_DAY_HOURS = 22
NAIVE_DAYS = range(2, 9)  # same hour on days -2..-8: causal for a forecast issued a day ahead


def naive_same_hour_mean(actual: pd.Series) -> pd.Series:
    """Mean of the same hour on days -2..-8 (NaN-skipping), on *actual*'s own index."""
    grid = actual.reindex(pd.date_range(actual.index.min(), actual.index.max(), freq="h"))
    stacked = pd.concat([grid.shift(24 * k) for k in NAIVE_DAYS], axis=1)
    return stacked.mean(axis=1).reindex(actual.index)


def attach_actuals(rows: pd.DataFrame, actual: pd.Series, daily_temp: pd.Series | None = None) -> pd.DataFrame:
    """Add ``y``, ``naive``, ``day`` and, with *daily_temp* (indexed by day), ``cold``."""
    out = rows.copy()
    out["y"] = actual.reindex(out["ts"]).to_numpy()
    out["naive"] = naive_same_hour_mean(actual).reindex(out["ts"]).to_numpy()
    out["day"] = out["ts"].dt.normalize()
    out["cold"] = out["day"].map(daily_temp) <= COLD_C if daily_temp is not None else False
    return out


def day_ahead(rows: pd.DataFrame, min_lead: int = 24) -> pd.DataFrame:
    """Per (arm, target hour) the newest forecast with at least *min_lead* hours of lead."""
    x = rows[rows["lead"] >= min_lead].sort_values("lead")
    return x.drop_duplicates(["arm", "ts"], keep="first")


def add_naive(rows: pd.DataFrame, ref: str = "base") -> pd.DataFrame:
    """Append a ``naive`` pseudo-arm on the reference arm's target hours."""
    n = rows[rows["arm"] == ref].copy()
    n["p50"], n["arm"] = n["naive"], "naive"
    for col in ("lo", "hi"):
        if col in n:
            n[col] = np.nan
    return pd.concat([rows, n], ignore_index=True)


def _block_mae(x: pd.DataFrame, hours: int) -> pd.Series:
    block = x["ts"].dt.floor(f"{hours}h")
    g = x.groupby(["arm", "day", block])["e"].agg(["sum", "size"])
    return g[g["size"] == hours]["sum"].abs().groupby(level=[0, 1]).mean()


def day_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, day) with at least ``MIN_DAY_HOURS`` scored hours.

    ``mae`` kWh/h, ``mae3`` / ``mae6`` mean abs error of complete 3 h / 6 h block totals (kWh), ``bias`` signed
    daily error (kWh), ``cover`` share of hours inside lo..hi, ``width`` mean hi - lo.
    """
    x = rows.dropna(subset=["y", "p50"]).copy()
    x["e"] = x["p50"] - x["y"]
    g = x.groupby(["arm", "day"])
    out = g.agg(n=("e", "size"), mae=("e", lambda s: s.abs().mean()), bias=("e", "sum"), y=("y", "sum"))
    out["cold"] = g["cold"].first() if "cold" in x else False
    out["mae3"] = _block_mae(x, 3)
    out["mae6"] = _block_mae(x, 6)
    if "lo" in x and x["lo"].notna().any():
        x["inside"] = (x["y"] >= x["lo"]) & (x["y"] <= x["hi"])
        has = x["lo"].notna()
        out["cover"] = x[has].groupby(["arm", "day"])["inside"].mean()
        out["width"] = (x["hi"] - x["lo"])[has].groupby([x["arm"], x["day"]]).mean()
    return out[out["n"] >= MIN_DAY_HOURS]


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Per arm: days, hourly / 3 h / 6 h MAE, mean abs daily error, bias, cold-day bias (kWh and % of load)."""
    g = table.groupby(level="arm")
    out = g.agg(
        days=("mae", "size"),
        mae=("mae", "mean"),
        mae3=("mae3", "mean"),
        mae6=("mae6", "mean"),
        daily_abs=("bias", lambda v: v.abs().mean()),
        bias=("bias", "mean"),
    )
    for col in ("cover", "width"):
        if col in table:
            out[col] = g[col].mean()
    cold = table[table["cold"].astype(bool)]
    cg = cold.groupby(level="arm")
    out["cold_days"] = cg["bias"].size()
    out["cold_bias"] = cg["bias"].mean()
    out["cold_bias_pct"] = 100 * cg["bias"].sum() / cg["y"].sum()
    return out


def cold_gate(summary_row: pd.Series, low_pct: float = -12.0, high_pct: float = 6.0, min_days: int = 14) -> str:
    """ROADMAP #118: PASS / FAIL on the pooled cold-day bias in % of load; INCONCLUSIVE below *min_days*."""
    days = summary_row.get("cold_days")
    if pd.isna(days) or days < min_days:
        return "INCONCLUSIVE"
    return "PASS" if low_pct <= summary_row["cold_bias_pct"] <= high_pct else "FAIL"


def paired(table: pd.DataFrame, arm: str, ref: str = "base", metric: str = "mae", n_boot: int = 2000, seed: int = 0):
    """Relative change of *metric* for *arm* against *ref* in %, with an ISO-week block-bootstrap 95 % CI.

    ``metric`` may be any ``day_table`` column or ``abs_bias``. Negative is better. Returns None without overlap.
    """

    def values(name):
        t = table.xs(name, level="arm")
        return t["bias"].abs() if metric == "abs_bias" else t[metric]

    j = pd.concat([values(arm).rename("a"), values(ref).rename("b")], axis=1).dropna()
    if j.empty:
        return None
    diff = j["a"] - j["b"]
    week = [f"{d.isocalendar()[0]}-{d.isocalendar()[1]}" for d in j.index]
    d_sum, b_sum = diff.groupby(week).sum().to_numpy(), j["b"].groupby(week).sum().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(d_sum), (n_boot, len(d_sum)))
    rel = 100 * d_sum[idx].sum(axis=1) / b_sum[idx].sum(axis=1)
    return dict(
        arm=arm,
        metric=metric,
        days=len(diff),
        pct=float(100 * diff.sum() / j["b"].sum()),
        lo=float(np.percentile(rel, 2.5)),
        hi=float(np.percentile(rel, 97.5)),
        worse_days=int((diff > 0).sum()),
    )


def next_day_swing(rows: pd.DataFrame) -> pd.DataFrame:
    """Per arm: how much the next-day total moves across the previous day's origins (kWh).

    ``step`` = mean abs change between consecutive origins, ``range`` = mean max - min, ``range_p90`` its 90th
    percentile. Only origins that cover all 24 hours of the next day count.
    """
    x = rows[rows["ts"].dt.normalize() == rows["origin"].dt.normalize() + pd.Timedelta(days=1)]
    tot = x.groupby(["arm", x["ts"].dt.normalize().rename("day"), "origin"])["p50"].agg(["sum", "size"])
    tot = tot[tot["size"] == 24]["sum"]
    per_day = tot.groupby(level=["arm", "day"])
    step = per_day.apply(lambda s: s.diff().abs().mean())
    rng = per_day.apply(lambda s: s.max() - s.min())
    return pd.DataFrame(
        {
            "step": step.groupby(level="arm").mean(),
            "range": rng.groupby(level="arm").mean(),
            "range_p90": rng.groupby(level="arm").quantile(0.9),
        }
    )


def coverage_by_lead(rows: pd.DataFrame, arm: str = "base", edges=(-1, 0, 2, 5, 11, 23, 35, 47)) -> pd.DataFrame:
    """Hourly MAE, bias and lo..hi coverage of one arm by lead bucket."""
    x = rows[(rows["arm"] == arm)].dropna(subset=["y", "lo", "hi"]).copy()
    x["e"] = x["p50"] - x["y"]
    x["inside"], x["below"], x["above"] = (x["y"] >= x["lo"]) & (x["y"] <= x["hi"]), x["y"] < x["lo"], x["y"] > x["hi"]
    bucket = pd.cut(x["lead"], list(edges))
    return x.groupby(bucket, observed=True).agg(
        n=("e", "size"),
        mae=("e", lambda s: s.abs().mean()),
        bias=("e", "mean"),
        cover=("inside", "mean"),
        below=("below", "mean"),
        above=("above", "mean"),
    )
