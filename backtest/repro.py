"""Report-only comparison of a lead-25 replay against what live actually issued (``pred_history.json``).

Live stores each target hour keep-first when it enters the ≤ 25 h horizon (energy_forecast.py:3571), so the replay
predicts each hour from origin ``ts − 25 h``. No pass/fail: the lean frame is not expected to match live; the
numbers show how far off it is. Thresholds come with the live-equivalent frame.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import Fold
from .env import DATA_DIR


@dataclass(frozen=True)
class ReproReport:
    n_hours: int
    mae_replay: float
    mae_live: float
    ratio: float  # mae_replay / mae_live
    mean_abs_gap: float  # mean |replay − live|, kWh/h
    err_corr: float  # corr(replay − actual, live − actual)
    note: str

    def lines(self) -> list[str]:
        if np.isnan(self.ratio):
            return [f"live comparison: {self.n_hours} h — {self.note}"]
        return [
            f"live comparison ({self.n_hours} h, lead 25 h): replay MAE {self.mae_replay:.3f} vs live "
            f"{self.mae_live:.3f} (ratio {self.ratio:.2f}), mean |replay−live| {self.mean_abs_gap:.3f} kWh/h, "
            f"error corr {self.err_corr:.2f} — {self.note}"
        ]


def _series(d: dict) -> pd.Series:
    return pd.Series({pd.Timestamp(k): float(v) for k, v in d.items()}, dtype=float).sort_index()


def load_live_pred(data_dir: Path = DATA_DIR) -> tuple[pd.Series, pd.Series]:
    """``(pred, actuals)`` hourly series from the pulled ``pred_history.json``."""
    ph = json.loads((data_dir / "pred_history.json").read_text())
    return _series(ph.get("pred", {})), _series(ph.get("actuals", {}))


def gate_folds(live_pred: pd.Series, days: int = 28) -> list[Fold]:
    """Last *days* complete days of *live_pred* as consecutive ≤ 7-day folds (weekly retrain, like live)."""
    end = (live_pred.index.max() + pd.Timedelta(hours=1)).normalize() - pd.Timedelta(days=1)
    start = max(end - pd.Timedelta(days=days - 1), live_pred.index.min().normalize())
    folds, s = [], start
    while s <= end:
        e = min(s + pd.Timedelta(days=6), end)
        folds.append(Fold(f"repro{len(folds) + 1}", s, e))
        s = e + pd.Timedelta(days=1)
    return folds


def replay_series(df: pd.DataFrame, arm: str = "base") -> pd.Series:
    """Flatten the ``lead25`` rows of *arm* into one hourly series."""
    if df.empty:
        return pd.Series(dtype=float)
    sub = df[(df["arm"] == arm) & (df["origin"] == "lead25")]
    parts = [pd.Series(r.pred, index=pd.date_range(r.ts0, periods=24, freq="h")) for r in sub.itertuples()]
    return pd.concat(parts).dropna().sort_index() if parts else pd.Series(dtype=float)


def repro_report(replay: pd.Series, live: pd.Series, actuals: pd.Series, min_hours: int = 72) -> ReproReport:
    """Compare replay and live on the hours all three series share."""
    common = replay.dropna().index.intersection(live.dropna().index).intersection(actuals.dropna().index)
    n = len(common)
    if n < min_hours:
        return ReproReport(n, np.nan, np.nan, np.nan, np.nan, np.nan, "too few overlapping hours — no comparison")
    r, lv, a = replay[common], live[common], actuals[common]
    mae_r, mae_l = float((r - a).abs().mean()), float((lv - a).abs().mean())
    return ReproReport(
        n_hours=n,
        mae_replay=mae_r,
        mae_live=mae_l,
        ratio=mae_r / mae_l if mae_l else np.nan,
        mean_abs_gap=float((r - lv).abs().mean()),
        err_corr=float(np.corrcoef(r - a, lv - a)[0, 1]),
        note="lean frame — not expected to match live",
    )
