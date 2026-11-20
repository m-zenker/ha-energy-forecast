"""Longer ``weight_halflife_days`` and temperature-similarity sample weights (2026-11-20; live half-life 60, apps.yaml).

Audit F7: at 60 d, rows <= 3 degC carry about 2 % of the training weight. Last winter is the only cold history
(data starts 2025-10-10). Caveat from ROADMAP #88: this autumn's load is lower than last winter's at the same
temperature (18.9 vs 24.5 kWh/day at 3-6 degC), so up-weighting old cold rows can over-forecast.

``ts<sigma>`` arms multiply the sample weight by a Gaussian kernel on ``temp_c`` around the 7 days before the fold
start (kernel renormalised to keep the mean weight). Lean frame, perfect archive weather: ranks arms, does not
transfer 1:1 to live. ``run_arm`` trains once at ``fold.start``, so folds are kept short (<= 10 days).
"""

import contextlib

import numpy as np

from backtest.arms import Arm
from backtest.data import DEFAULT_FOLDS, _fold


def temp_similarity(sigma: float):
    @contextlib.contextmanager
    def patch(fold, weather):
        import lightgbm as lgb

        w = weather[(weather["timestamp"] < fold.start) & (weather["timestamp"] >= fold.start - np.timedelta64(7, "D"))]
        t_ref = float(w["temp_c"].mean())
        orig = lgb.LGBMRegressor.__dict__["fit"]

        def _fit(self, X, y, *args, **kwargs):
            sw = kwargs.get("sample_weight")
            if sw is not None and hasattr(X, "columns") and "temp_c" in X.columns:
                k = np.exp(-0.5 * ((X["temp_c"].to_numpy(dtype=float) - t_ref) / sigma) ** 2)
                k = k / k.mean()
                kwargs["sample_weight"] = np.asarray(sw, dtype=float) * k
                _fit.calls += 1
            return orig(self, X, y, *args, **kwargs)

        _fit.calls = 0
        lgb.LGBMRegressor.fit = _fit
        try:
            yield
        finally:
            lgb.LGBMRegressor.fit = orig
            print(f"  [ts{sigma:g}] t_ref={t_ref:.1f} degC, weighted fits={_fit.calls}")

    return patch


ARMS = [
    Arm("base"),  # live: 60 d
    Arm("hl30", train_kwargs={"weight_halflife_days": 30.0}),
    Arm("hl90", train_kwargs={"weight_halflife_days": 90.0}),
    Arm("hl120", train_kwargs={"weight_halflife_days": 120.0}),
    Arm("hl180", train_kwargs={"weight_halflife_days": 180.0}),
    Arm("hl0", train_kwargs={"weight_halflife_days": 0.0}),  # no recency weighting
    Arm("ts5", patch=temp_similarity(5.0)),  # 60 d x Gaussian(5 degC)
    Arm("ts8", patch=temp_similarity(8.0)),
    Arm("hl120_ts5", train_kwargs={"weight_halflife_days": 120.0}, patch=temp_similarity(5.0)),
]
_by = {f.name: f for f in DEFAULT_FOLDS}
FOLDS = [
    _by["Mar26"],
    _by["Oct26"],
    _by["Nov26"],  # 10-26..11-04, trained at 10-26
    _fold("Nov26b", "2026-11-05", "2026-11-14"),  # own training cut at 11-05: the recent cold days
]
