"""Tests for backtest/env.py — frozen clock and LightGBM seed override."""

import numpy as np
import pandas as pd
import pytest

from backtest.env import frozen_now, lgbm_seed


def test_frozen_now_naive_and_aware_then_restored():
    orig = pd.Timestamp.__dict__["now"]
    ts = pd.Timestamp("2026-01-10 06:00")
    with frozen_now(ts):
        assert pd.Timestamp.now() == ts
        assert pd.Timestamp.now(tz="Europe/Zurich") == ts.tz_localize("Europe/Zurich")
    assert pd.Timestamp.__dict__["now"] is orig


def test_frozen_now_restores_on_exception():
    orig = pd.Timestamp.__dict__["now"]
    with pytest.raises(RuntimeError), frozen_now(pd.Timestamp("2026-01-10")):
        raise RuntimeError
    assert pd.Timestamp.__dict__["now"] is orig


def test_lgbm_seed_sets_random_state_and_restores():
    lgb = pytest.importorskip("lightgbm")
    orig = lgb.LGBMRegressor.__dict__["fit"]
    X, y = np.arange(40).reshape(-1, 1), np.arange(40.0)
    with lgbm_seed(7):
        assert lgb.LGBMRegressor(n_estimators=2, verbose=-1, random_state=42).fit(X, y).random_state == 7
    assert lgb.LGBMRegressor.__dict__["fit"] is orig
    with lgbm_seed(None):  # None = leave the model's own random_state alone
        assert lgb.LGBMRegressor(n_estimators=2, verbose=-1, random_state=42).fit(X, y).random_state == 42


def test_import_makes_energy_forecast_importable():
    import energy_forecast.model  # noqa: F401  (hassapi stubbed, apps/ on path)

    import backtest  # noqa: F401
