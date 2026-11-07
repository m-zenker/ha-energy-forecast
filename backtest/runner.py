"""Drive the real EnergyForecastModel.train()/predict() per (arm, fold) and score each target day.

Leak guards: training sees only rows before ``fold.start``; each prediction runs with the clock frozen at its
origin and ``recent_actuals`` strictly before it. The forecast window starts ``WINDOW_HISTORY`` before the
origin because ``_prepare_prediction_X`` prepends the 400-row weather tail frozen at train time
(model.py:671,1108-1115): without the history, row-based temperature lags/rolls straddle a gap.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .arms import Arm, feature_drop
from .data import Fold, Inputs
from .env import TZ, frozen_now, lgbm_seed

Origins = tuple[int, ...] | Literal["lead25"]
WINDOW_HISTORY = pd.Timedelta(days=15)
HORIZON = pd.Timedelta(hours=48)
RECENT = pd.Timedelta(days=30)
MAX_NAN_HOURS = 4  # a target day with more missing actual hours is skipped
MIN_SCORED_HOURS = 18  # a (day, origin) row needs at least this many scored hours
COLD_C = 10.0


def _scorable_days(actual: pd.Series, fold: Fold, days_limit: int | None) -> list[tuple[pd.Timestamp, pd.Series]]:
    out = []
    for day in pd.date_range(fold.start.normalize(), fold.end.normalize(), freq="D"):
        a = actual.reindex(pd.date_range(day, periods=24, freq="h"))
        if a.isna().sum() <= MAX_NAN_HOURS:
            out.append((day, a))
    return out[:days_limit] if days_limit else out


def _train(arm: Arm, fold: Fold, inputs: Inputs, weather: pd.DataFrame, defaults: dict):
    from energy_forecast.model import EnergyForecastModel

    energy = inputs.energy
    ev = inputs.ev_df
    with feature_drop(arm.drop), frozen_now(fold.start), lgbm_seed(arm.seed):
        model = EnergyForecastModel(Path(tempfile.mkdtemp()), timezone=TZ)
        model.train(
            energy[energy["timestamp"] < fold.start],
            weather[weather["timestamp"] < fold.start],
            ev_df=None if ev is None else ev[ev["timestamp"] < fold.start],
            **{**defaults, **arm.train_kwargs},
        )
    if model.model is None:
        raise RuntimeError(f"train() skipped for arm {arm.name!r}, fold {fold.name!r} (too little history?)")
    readded = sorted(set(arm.drop) & set(model.feature_cols))
    if readded:
        raise ValueError(f"arm {arm.name!r}: train() re-added dropped features {readded}; this drop has no effect")
    return model


def _predict(model, weather: pd.DataFrame, energy: pd.DataFrame, t_o: pd.Timestamp) -> pd.Series:
    ts = weather["timestamp"]
    fc = weather[(ts >= t_o - WINDOW_HISTORY) & (ts < t_o + HORIZON)].copy()
    e_ts = energy["timestamp"]
    recent = energy[(e_ts >= t_o - RECENT) & (e_ts < t_o)]
    with frozen_now(t_o):
        out = model.predict(fc, live_temp=None, recent_actuals=recent)
    return out.set_index("timestamp")["predicted_kwh"]


def _row(arm, fold, day, origin, cold, a: pd.Series, p: pd.Series) -> dict | None:
    m = a.notna() & p.notna()
    if m.sum() < MIN_SCORED_HOURS:
        return None
    err = p[m] - a[m]
    return dict(
        arm=arm.name,
        fold=fold.name,
        day=day.date(),
        origin=origin,
        cold=cold,
        mae=float(err.abs().mean()),
        bias=float(err.sum() * 24 / m.sum()),
        pred=p.where(m).to_numpy(dtype=float),
        ts0=day,
    )


def run_arm(
    arm: Arm,
    fold: Fold,
    inputs: Inputs,
    weather: pd.DataFrame,
    defaults: dict,
    origins: Origins = (6, 18),
    days_limit: int | None = None,
) -> list[dict]:
    """One train on data before ``fold.start``; one row per scorable target day × origin."""
    energy = inputs.energy
    actual = energy.drop_duplicates("timestamp", keep="last").set_index("timestamp")["gross_kwh"]
    days = _scorable_days(actual, fold, days_limit)
    if not days:
        return []
    wa = arm.weather(weather) if arm.weather else weather
    model = _train(arm, fold, inputs, wa, defaults)
    daymean = weather.groupby(weather["timestamp"].dt.normalize())["temp_c"].mean()
    rows = []
    for day, a in days:
        cold = bool(daymean.get(day, np.inf) <= COLD_C)
        if origins == "lead25":
            p = pd.Series(np.nan, index=a.index)
            for ts in a.index[a.notna()]:
                p[ts] = _predict(model, wa, energy, ts - pd.Timedelta(hours=25)).get(ts, np.nan)
            row = _row(arm, fold, day, "lead25", cold, a, p)
            rows += [row] if row else []
            continue
        for h in origins:
            t_o = day - pd.Timedelta(days=1) + pd.Timedelta(hours=int(h))
            p = _predict(model, wa, energy, t_o).reindex(a.index)
            row = _row(arm, fold, day, int(h), cold, a, p)
            rows += [row] if row else []
    return rows


def run_experiment(
    arms: Sequence[Arm],
    folds: Sequence[Fold],
    inputs: Inputs,
    weather: pd.DataFrame,
    defaults: dict,
    origins: Origins = (6, 18),
    out: Path | None = None,
    days_limit: int | None = None,
) -> pd.DataFrame:
    """Every arm × fold; the result pickle (if *out*) is rewritten after each arm so long runs can be inspected."""
    rows: list[dict] = []
    for arm in arms:
        for fold in folds:
            rows += run_arm(arm, fold, inputs, weather, defaults, origins, days_limit)
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_pickle(out)
    return pd.DataFrame(rows)
