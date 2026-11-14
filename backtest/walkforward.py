"""Walk the real train()/predict() forward over a retrain capture's history, one arm at a time.

    PYTHONPATH=apps:. /home/jovyan/my_envs/ha-energy-forecast-live/bin/python -m backtest.walkforward \\
        data/captures --start 2026-10-01 --end 2026-11-12 --arms base jit --out data/backtest/wf

Each arm retrains every ``every_days`` days at ``retrain_hour`` (default daily at 18:00, the current live rhythm),
seeded from its previous model dir like live, and forecasts 48 h from four origins a day (18:00, 00:00, 06:00,
12:00) with the newest model. Rows are written per arm to ``<out>/wf_<arm>_<start>_<end>.pkl`` with columns
``arm, origin, cutoff, ts, lead, p50, lo, hi``; score them with ``backtest.wf_score``.

Always run a ``jit`` arm next to ``base``: it is the same model retrained one hour earlier, and its distance from
``base`` is the noise floor an effect has to clear.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .arms import feature_drop
from .capture_frame import SERVE_PAST, CaptureFrame, predict, train

ORIGIN_HOURS = (18, 24, 30, 36)  # hours after the retrain day's midnight
THERMAL_PRESSURE = (
    "thermal_pressure",
    "thermal_pressure_max",
    "thermal_pressure_std",
    "thermal_pressure_cop",
    "thermal_pressure_net",
    "infiltration_pressure",
)


@dataclass(frozen=True)
class WfArm:
    name: str
    drop: tuple[str, ...] = ()  # static _FEATURES_BASE names removed at train and serve
    train_kwargs: Mapping[str, Any] = field(default_factory=dict)
    cutoff_shift_h: int = 0  # retrain this many hours off the nominal cutoff (jitter control: -1)
    every_days: int = 1  # retrain cadence
    live_frame: bool = False  # serve with live's 3-day weather tail instead of a contiguous one
    physics: bool = False  # run the physics baseline at serve time (needed for Phase 2 models)


ARMS: dict[str, WfArm] = {
    a.name: a
    for a in (
        WfArm("base"),
        WfArm("jit", cutoff_shift_h=-1),
        WfArm("droptp", drop=THERMAL_PRESSURE),
        WfArm("noreg", train_kwargs={"enable_regimes": False}),
        WfArm("weekly", every_days=7),
        WfArm("weekly_gap", every_days=7, live_frame=True),
    )
}


@contextlib.contextmanager
def single_thread_lgbm(n_jobs: int | None):
    """Pin ``n_jobs`` on every ``LGBMRegressor.fit``; the sklearn wrapper ignores ``OMP_NUM_THREADS``."""
    if n_jobs is None:
        yield
        return
    import lightgbm as lgb

    orig = lgb.LGBMRegressor.__dict__["fit"]

    def _fit(self, *args, **kwargs):
        self.n_jobs = n_jobs
        return orig(self, *args, **kwargs)

    lgb.LGBMRegressor.fit = _fit
    try:
        yield
    finally:
        lgb.LGBMRegressor.fit = orig


def retrain_days(start: pd.Timestamp, end: pd.Timestamp, every_days: int) -> list[pd.Timestamp]:
    """Days on which the arm retrains: the day before *start*, then every *every_days* days up to *end*."""
    first = pd.Timestamp(start).normalize() - pd.Timedelta(days=1)
    days = pd.date_range(first, pd.Timestamp(end).normalize(), freq="D")
    return [d for i, d in enumerate(days) if i % every_days == 0]


def run_arm(
    arm: WfArm,
    frame: CaptureFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    work_root: Path,
    retrain_hour: int = 18,
    keep_models: bool = False,
) -> pd.DataFrame:
    """All forecasts of one arm between *start* and *end* (inclusive target days)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    last_origin = min(end + pd.Timedelta(days=1), frame.end)
    retrain_on = set(retrain_days(start, end, arm.every_days))
    patch = (lambda: feature_drop(arm.drop)) if arm.drop else None
    rows: list[pd.DataFrame] = []
    model = work = cutoff = None
    day = start.normalize() - pd.Timedelta(days=1)
    while day <= end.normalize():
        if day in retrain_on:
            cutoff = day + pd.Timedelta(hours=retrain_hour + arm.cutoff_shift_h)
            previous = work
            model, work = train(frame, cutoff, work_root, seed_dir=previous, patch=patch, train_kwargs=arm.train_kwargs)
            if model.model is None:
                raise RuntimeError(f"arm {arm.name!r}: train() skipped at {cutoff} (too little history?)")
            if previous is not None and not keep_models:
                shutil.rmtree(previous, ignore_errors=True)
        for h in ORIGIN_HOURS:
            origin = day + pd.Timedelta(hours=h)
            if origin > last_origin:
                break
            kwargs = frame.predict_kwargs(origin)
            if arm.live_frame:
                kwargs["forecast_df"] = frame.forecast_frame(origin, SERVE_PAST)
            with patch() if patch else contextlib.nullcontext():
                out = predict(frame, model, work, origin, kwargs=kwargs, physics=arm.physics)
            out["arm"], out["origin"], out["cutoff"] = arm.name, origin, cutoff
            rows.append(out)
        day += pd.Timedelta(days=1)
    if work is not None and not keep_models:
        shutil.rmtree(work, ignore_errors=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backtest.walkforward", description=__doc__.splitlines()[0])
    ap.add_argument("captures_dir", type=Path)
    ap.add_argument("--capture-id", help="retrain capture to use (default: the latest)")
    ap.add_argument("--start", required=True, help="first target day, e.g. 2026-10-01")
    ap.add_argument("--end", required=True, help="last target day")
    ap.add_argument("--arms", nargs="+", default=["base", "jit"], choices=sorted(ARMS))
    ap.add_argument("--out", type=Path, required=True, help="directory for the result pickles")
    ap.add_argument("--work", type=Path, help="directory for model dirs (default: a temp dir, removed at exit)")
    ap.add_argument("--n-jobs", type=int, default=1, help="LightGBM threads per fit (default 1)")
    args = ap.parse_args(argv)

    frame = CaptureFrame.from_capture(args.captures_dir, args.capture_id)
    args.out.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    with contextlib.ExitStack() as stack:
        work_root = args.work or Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="wf_")))
        work_root.mkdir(parents=True, exist_ok=True)
        stack.enter_context(single_thread_lgbm(args.n_jobs))
        for name in args.arms:
            rows = run_arm(ARMS[name], frame, start, end, work_root)
            path = args.out / f"wf_{name}_{start:%m%d}_{end:%m%d}.pkl"
            rows.to_pickle(path)
            print(f"{name}: {rows['origin'].nunique() if len(rows) else 0} origins -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
