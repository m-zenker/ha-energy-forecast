"""Train/serve feature audit: for the same target hours, compare what was served with what was later trained on.

    PYTHONPATH=apps:. /home/jovyan/my_envs/ha-energy-forecast-live/bin/python -m backtest.audit_features data/captures

For every predict capture it rebuilds the 48-row feature matrix live fed the model. For the newest retrain capture
it re-runs ``train()`` and keeps the final training matrix. Rows are then paired by target hour, per feature and
lead range. A feature with high gain and a low served-vs-trained correlation, or a large shift, is being used
outside what the model learned. In the 2026-11-13 audit this found ``rolling_std_24h`` served as exactly 0, the
thermal-pressure family at about zero correlation, and lags imputed with means in one place and medians in another.

Needs the env that matches the HA host, like every capture replay.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .capture_frame import CaptureFrame, load_model, train
from .capture_replay import _retarget_physics, reference_dir_for
from .env import TZ, frozen_now

LEADS = ((1, 5), (24, 47))


@contextlib.contextmanager
def _record_training_matrix(store: dict):
    """Keep the largest X passed to a point-model fit (the final fit on all rows) and the engineered frame."""
    import energy_forecast.model as mm
    import lightgbm as lgb

    orig_fit = lgb.LGBMRegressor.__dict__["fit"]
    orig_ef = mm._engineer_features

    def fit(self, X, y, *args, **kwargs):
        if getattr(self, "objective", None) is None and len(X) > len(store.get("X", ())):
            store["X"] = X.copy()
        return orig_fit(self, X, y, *args, **kwargs)

    def engineer(df, *args, **kwargs):
        out = orig_ef(df, *args, **kwargs)
        if len(out) > len(store.get("frame", ())):
            store["frame"] = out
        return out

    lgb.LGBMRegressor.fit, mm._engineer_features = fit, engineer
    try:
        yield
    finally:
        lgb.LGBMRegressor.fit, mm._engineer_features = orig_fit, orig_ef


def trained_features(frame: CaptureFrame, work_root: Path, seed_dir: Path | None = None):
    """Re-run train() at the capture's end; returns ``(model, X indexed by timestamp)``."""
    store: dict = {}
    cutoff = frame.end + pd.Timedelta(hours=1, minutes=30)
    model, _ = train(frame, cutoff, work_root, seed_dir=seed_dir, patch=lambda: _record_training_matrix(store))
    X = store["X"]
    X.index = pd.DatetimeIndex(store["frame"].loc[X.index, "timestamp"])
    return model, X


def served_features(root: Path, work_root: Path) -> pd.DataFrame:
    """Feature rows as served: one block of 48 per predict capture, with ``ts`` and ``lead``."""
    from energy_forecast import capture

    blocks, models = [], {}
    for sc in capture.list_captures(root, "predict"):
        ref = reference_dir_for(Path(root), sc)
        if sc.get("retrain_in_progress") or ref is None:
            continue
        payload, _ = capture.load_capture(root, "predict", sc["capture_id"])
        if ref not in models:
            models[ref] = load_model(ref, work_root)
        model, work = models[ref]
        physics = _retarget_physics(payload["physics_model"], work)  # never the live paths
        with frozen_now(payload["now_local"], TZ):
            hours, X = model._prepare_prediction_X(**copy.deepcopy(payload["predict_kwargs"]), physics_model=physics)
        X = X.copy()
        X["ts"], X["lead"] = pd.DatetimeIndex(hours), np.arange(len(X))
        blocks.append(X)
    return pd.concat(blocks, ignore_index=True) if blocks else pd.DataFrame()


def compare(served: pd.DataFrame, trained: pd.DataFrame, gain: pd.Series | None = None) -> pd.DataFrame:
    """Per feature and lead range: means, shift and mean abs difference in training SDs, and correlation."""
    rows = []
    for lo, hi in LEADS:
        s = served[served["lead"].between(lo, hi) & served["ts"].isin(trained.index)]
        for col in trained.columns:
            if col not in s:
                continue
            sv = s[col].to_numpy(dtype=float)
            tv = trained[col].reindex(s["ts"]).to_numpy(dtype=float)
            sd = float(trained[col].std())
            scale = sd if sd > 0 else np.nan
            corr = np.corrcoef(sv, tv)[0, 1] if len(sv) > 2 and sv.std() > 0 and tv.std() > 0 else np.nan
            rows.append(
                dict(
                    feature=col,
                    leads=f"{lo}-{hi}",
                    n=len(sv),
                    gain_pct=float(gain[col]) if gain is not None and col in gain else np.nan,
                    served_mean=sv.mean() if len(sv) else np.nan,
                    trained_mean=tv.mean() if len(tv) else np.nan,
                    shift_sd=(sv.mean() - tv.mean()) / scale if len(sv) else np.nan,
                    abs_diff_sd=np.abs(sv - tv).mean() / scale if len(sv) else np.nan,
                    corr=corr,
                )
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backtest.audit_features", description=__doc__.splitlines()[0])
    ap.add_argument("captures_dir", type=Path)
    ap.add_argument("--min-gain", type=float, default=0.6, help="only print features with at least this gain %%")
    args = ap.parse_args(argv)
    frame = CaptureFrame.from_capture(args.captures_dir)
    with tempfile.TemporaryDirectory(prefix="wf_") as tmp:
        model, trained = trained_features(frame, Path(tmp))
        served = served_features(args.captures_dir, Path(tmp))
    booster = model.model.booster_
    gain = pd.Series(booster.feature_importance("gain"), index=model.feature_cols)
    table = compare(served, trained, 100 * gain / gain.sum())
    pd.set_option("display.width", 200)
    for leads, part in table[table["gain_pct"] >= args.min_gain].groupby("leads"):
        print(f"\nserved at lead {leads} h vs later-trained value, same target hours")
        print(part.drop(columns="leads").sort_values("gain_pct", ascending=False).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
