"""Post-deploy check for #110 (soft out-of-fold regime labels).

Compares a pre-deploy and a post-deploy window on two criteria:

1. Share of consecutive same-model forecast pairs (``regime_log.jsonl``) whose next-day forecast moved by more than
   ``--threshold`` kWh. PASS when ``post_share <= pre_share``; INCONCLUSIVE when either window has < 50 pairs.
2. Daily MAE of the day-ahead forecast from ``pred_history.json`` (complete 24 h days only). PASS when
   ``post <= 1.10 * pre``.

Usage (after ``scripts/pull_ha_data.py``):
    python scripts/check_regime_stability.py --deploy 2026-11-06T09:00 --rebaseline 2026-11-05T00:00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

MIN_PAIRS = 50
MAE_TOL = 1.10
WINDOW_DAYS = 14


def load_lines(path: Path) -> list[dict]:
    out = []
    for raw in Path(path).read_text().splitlines():
        raw = raw.strip()
        if raw:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue  # a torn last line must not abort the check
    return out


def delta_pairs(lines: list[dict]) -> list[float]:
    """|Δ tomorrow_kwh| between consecutive lines with the same target_date and the same last_trained."""
    ordered = sorted(
        lines,
        key=lambda r: pd.Timestamp(r["ts"]).tz_convert("UTC")
        if pd.Timestamp(r["ts"]).tzinfo
        else pd.Timestamp(r["ts"]).tz_localize("UTC"),
    )
    deltas: list[float] = []
    prev = None
    for cur in ordered:
        if (
            prev is not None
            and prev.get("target_date") == cur.get("target_date")
            and prev.get("last_trained") == cur.get("last_trained")
            and prev.get("tomorrow_kwh") is not None
            and cur.get("tomorrow_kwh") is not None
        ):
            deltas.append(abs(float(cur["tomorrow_kwh"]) - float(prev["tomorrow_kwh"])))
        prev = cur
    return deltas


def delta_share(lines: list[dict], threshold: float = 2.0) -> float:
    deltas = delta_pairs(lines)
    if not deltas:
        return float("nan")
    return sum(d > threshold for d in deltas) / len(deltas)


def verdict_share(pre: tuple[float, int], post: tuple[float, int]) -> str:
    """pre/post = (share, n_pairs)."""
    if pre[1] < MIN_PAIRS or post[1] < MIN_PAIRS:
        return "INCONCLUSIVE"
    return "PASS" if post[0] <= pre[0] else "FAIL"


def daily_mae(pred_history: dict, start: str, end: str) -> tuple[float, int]:
    """Mean |sum(pred) - sum(actual)| over days in [start, end) with 24 predicted and 24 actual hours."""
    pred = pd.Series(pred_history.get("pred", {}), dtype=float)
    act = pd.Series(pred_history.get("actuals", {}), dtype=float)
    if pred.empty or act.empty:
        return float("nan"), 0
    pred.index, act.index = pd.to_datetime(pred.index), pd.to_datetime(act.index)
    p, a = pred.groupby(pred.index.date), act.groupby(act.index.date)
    lo, hi = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    errs = []
    for day, pg in p:
        if not (lo <= day < hi) or len(pg) != 24:
            continue
        try:
            ag = a.get_group(day)
        except KeyError:
            continue
        if len(ag) == 24:
            errs.append(abs(float(pg.sum()) - float(ag.sum())))
    return (sum(errs) / len(errs) if errs else float("nan")), len(errs)


def _window(lines: list[dict], lo: pd.Timestamp, hi: pd.Timestamp) -> list[dict]:
    return [r for r in lines if lo <= pd.Timestamp(r["ts"]).tz_localize(None) < hi]


def _naive(ts: str) -> pd.Timestamp:
    """Local wall time: a tz-aware timestamp keeps its wall clock, a naive one is taken as local."""
    t = pd.Timestamp(ts)
    return t.tz_localize(None) if t.tzinfo else t


def post_start(lines: list[dict], deploy: pd.Timestamp) -> pd.Timestamp:
    """Timestamp of the first log line written by a model trained after the deploy (the retrain that applies the
    change), or ``deploy`` when none is found yet. Forecasts before it still come from the old model."""
    before = [r for r in lines if _naive(r["ts"]) < deploy and r.get("last_trained")]
    old = max(before, key=lambda r: _naive(r["ts"]))["last_trained"] if before else None
    after = sorted((r for r in lines if _naive(r["ts"]) >= deploy), key=lambda r: _naive(r["ts"]))
    for r in after:
        if r.get("last_trained") and r["last_trained"] != old:
            return _naive(r["ts"])
    return deploy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", required=True, help="ISO local time of the deploy")
    ap.add_argument("--rebaseline", required=True, help="ISO local time the A+B re-baseline started")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--threshold", type=float, default=2.0)
    args = ap.parse_args(argv)

    deploy, rebase = _naive(args.deploy), _naive(args.rebaseline)
    pre_lo = max(rebase, deploy - pd.Timedelta(days=WINDOW_DAYS))
    post_hi = deploy + pd.Timedelta(days=WINDOW_DAYS)
    data = Path(args.data_dir)

    lines = [r for r in load_lines(data / "regime_log.jsonl") if "ts" in r]
    post_lo = post_start(lines, deploy)
    if post_lo == deploy:
        print("note: no post-deploy retrain found in regime_log.jsonl yet; post window starts at the deploy time")
    res = {}
    for name, lo, hi in (("pre", pre_lo, deploy), ("post", post_lo, post_hi)):
        w = _window(lines, lo, hi)
        d = delta_pairs(w)
        share = (sum(x > args.threshold for x in d) / len(d)) if d else float("nan")
        res[name] = (share, len(d))
        print(
            f"{name:>4}: {lo:%Y-%m-%d %H:%M} .. {hi:%Y-%m-%d %H:%M} ({(hi - lo).days} d)  share>|{args.threshold}| = "
            f"{share:.3f}  pairs = {len(d)}"
        )
    v1 = verdict_share(res["pre"], res["post"])
    print(f"delta-share criterion: {v1}")

    ph_path = data / "pred_history.json"
    v2 = "INCONCLUSIVE"
    if ph_path.exists():
        ph = json.loads(ph_path.read_text())
        pre_mae, n_pre = daily_mae(ph, str(pre_lo.date()), str(deploy.date()))
        # pred_history keeps predictions made <= 25 h ahead: the first fully new-model day starts after that
        mae_lo = post_lo if post_lo == deploy else (post_lo + pd.Timedelta(hours=25)).ceil("D")
        post_mae, n_post = daily_mae(ph, str(mae_lo.date()), str(post_hi.date()))
        print(f" MAE: pre {pre_mae:.3f} kWh/day (n={n_pre} d)  post {post_mae:.3f} kWh/day (n={n_post} d)")
        if n_pre and n_post:
            v2 = "PASS" if post_mae <= MAE_TOL * pre_mae else "FAIL"
    print(f"daily-MAE criterion (post <= {MAE_TOL} x pre): {v2}")
    return 0 if (v1 == "PASS" and v2 == "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
