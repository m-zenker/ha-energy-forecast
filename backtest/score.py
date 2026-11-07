"""Paired comparison of arms against a reference arm, with a (fold, ISO-week) block bootstrap.

Weeks are resampled instead of days: consecutive days share weather regimes and each fold shares one trained
model, so day-level resampling gives CIs that are too narrow. The summary always prints the two caveats that
apply to every lean-frame, archive-weather run.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

KEY = ["fold", "day", "origin"]
CAVEATS = (
    "weather: archive (perfect) — weather-feature gains are upper bounds; noise-robustness transforms are scored"
    " unfairly",
    "frame: lean — not valid for features/kwargs overlapping physics, climate, DHW or sub-meters"
    " (temperature group, heating degrees, buffer, regime_kwh)",
)
SEGMENTS = ("ALL", "COLD", "WARM")


def _values(df: pd.DataFrame, arm: str, col: str) -> pd.Series:
    sub = df[df["arm"] == arm].set_index(KEY)
    return sub["bias"].abs() if col == "abs_bias" else sub[col]


def paired_bootstrap(
    df: pd.DataFrame, arm: str, ref: str = "base", col: str = "mae", n_boot: int = 4000, seed: int = 0
) -> dict:
    """Mean paired difference arm − ref of *col* (``abs_bias`` = |bias|), with a week-block bootstrap CI."""
    r, x = _values(df, ref, col), _values(df, arm, col)
    diff = (x - r).dropna()
    ref_mean = float(r.reindex(diff.index).mean()) if len(diff) else np.nan
    days = diff.groupby(level=["fold", "day"]).mean()
    nan = dict(d=np.nan, lo=np.nan, hi=np.nan, pct=np.nan, lo_pct=np.nan, hi_pct=np.nan)
    if days.empty:
        return {**nan, "worse_days": 0, "n_days": 0, "fold_sign": "0/0"}
    d = float(days.mean())
    folds = days.groupby(level="fold").mean()
    out = dict(
        d=d,
        lo=np.nan,
        hi=np.nan,
        worse_days=int((days > 0).sum()),
        n_days=len(days),
        fold_sign=f"{int((np.sign(folds) == np.sign(d)).sum())}/{len(folds)}",
    )
    frame = days.rename("v").reset_index()
    frame["week"] = pd.to_datetime(frame["day"]).dt.isocalendar().week.astype(int)
    blocks = frame.groupby(["fold", "week"])["v"].agg(["sum", "count"])
    if len(blocks) >= 3:
        idx = np.random.default_rng(seed).integers(0, len(blocks), (n_boot, len(blocks)))
        stat = blocks["sum"].to_numpy()[idx].sum(1) / blocks["count"].to_numpy()[idx].sum(1)
        out["lo"], out["hi"] = float(np.quantile(stat, 0.025)), float(np.quantile(stat, 0.975))
    for k in ("d", "lo", "hi"):
        out[("pct" if k == "d" else f"{k}_pct")] = 100 * out[k] / ref_mean if ref_mean else np.nan
    return out


def _segment_table(df: pd.DataFrame, ref: str) -> pd.DataFrame:
    rows = {}
    for arm in dict.fromkeys(df["arm"]):
        sub = df[df["arm"] == arm]
        row = dict(
            n_days=sub[["fold", "day"]].drop_duplicates().shape[0],
            mae=sub["mae"].mean(),
            bias=sub["bias"].mean(),
            abs_bias=sub["bias"].abs().mean(),
        )
        if arm == ref:
            row.update(pct=np.nan, lo_pct=np.nan, hi_pct=np.nan, worse_days=np.nan, fold_sign="", abs_bias_pct=np.nan)
        else:
            b = paired_bootstrap(df, arm, ref)
            row.update({k: b[k] for k in ("pct", "lo_pct", "hi_pct", "worse_days", "fold_sign")})
            row["abs_bias_pct"] = paired_bootstrap(df, arm, ref, col="abs_bias")["pct"]
        rows[arm] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def summarize(df: pd.DataFrame, ref: str = "base") -> dict[str, pd.DataFrame]:
    """Per-segment tables (ALL / COLD / WARM), one row per arm in first-appearance order."""
    if df.empty or ref not in set(df["arm"]):
        raise ValueError(f"reference arm {ref!r} has no rows")
    parts = {"ALL": df, "COLD": df[df["cold"]], "WARM": df[~df["cold"]]}
    return {seg: (_segment_table(p, ref) if len(p) else pd.DataFrame()) for seg, p in parts.items()}


def format_summary(tables: dict[str, pd.DataFrame], header: Sequence[str] = (), ref: str = "base") -> str:
    """Fixed-width text: caveats, *header* lines, seed noise floor (if ``<ref>@s*`` arms exist), then each segment."""
    lines = list(CAVEATS) + list(header)
    all_t = tables.get("ALL", pd.DataFrame())
    seed_arms = [a for a in all_t.index if str(a).startswith(f"{ref}@s")]
    floor = float(all_t.loc[seed_arms, "pct"].abs().max()) if seed_arms else None
    if floor is not None:
        lines.append(f"seed noise floor: ±{floor:.1f} % (arms marked ~ are inside it)")
    for seg in SEGMENTS:
        t = tables.get(seg, pd.DataFrame())
        if t.empty:
            lines.append(f"\n{seg}: no scored days")
            continue
        lines.append(f"\n{seg}")
        lines.append(
            f"  {'arm':<14}{'days':>5}{'mae':>8}{'bias':>8}{'|bias|':>8}{'dMAE%':>8}"
            f"{'95% CI':>18}{'worse':>7}{'folds':>7}{'d|bias|%':>10}"
        )
        for arm, r in t.iterrows():
            mark = "~" if floor is not None and arm != ref and arm not in seed_arms and abs(r["pct"]) <= floor else " "
            ci = "" if np.isnan(r["lo_pct"]) else f"[{r['lo_pct']:+.1f},{r['hi_pct']:+.1f}]"
            pct = "" if np.isnan(r["pct"]) else f"{r['pct']:+.1f}"
            worse = "" if np.isnan(r["worse_days"]) else f"{int(r['worse_days'])}"
            dab = "" if np.isnan(r["abs_bias_pct"]) else f"{r['abs_bias_pct']:+.1f}"
            lines.append(
                f"{mark} {arm:<14}{int(r['n_days']):>5}{r['mae']:>8.3f}{r['bias']:>8.2f}{r['abs_bias']:>8.2f}"
                f"{pct:>8}{ci:>18}{worse:>7}{r['fold_sign']:>7}{dab:>10}"
            )
    return "\n".join(lines)
