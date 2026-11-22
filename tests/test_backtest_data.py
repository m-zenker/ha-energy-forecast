"""Tests for backtest/data.py — live-equivalent loaders and train defaults."""

import sys

import pandas as pd
import pytest

from backtest.data import load_inputs, load_weather, train_defaults
from backtest.env import REPO
from tests._backtest_synth import synthetic_weather


def test_missing_energy_history_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="energy_history.csv"):
        load_inputs(tmp_path)


def test_train_defaults_mirror_retrain():
    d = train_defaults({"weight_halflife_days": 60, "holiday_canton": "SO", "enable_regimes": True, "regime_count": 0})
    assert d == dict(
        outdoor_df=None, weight_halflife_days=60.0, canton="SO", country="CH", enable_regimes=True, regime_count=0
    )
    assert train_defaults({})["regime_count"] == 5 and train_defaults({})["enable_regimes"] is False


def test_load_weather_offline_is_sorted_naive_unique(tmp_path):
    w = synthetic_weather(days=5)
    pd.concat([w.iloc[::-1], w.iloc[:3]]).to_pickle(tmp_path / "regime_backtest_weather.pkl")  # reversed + dupes
    out = load_weather(tmp_path, fetch_tail=False)
    assert out.timestamp.dt.tz is None and out.timestamp.is_monotonic_increasing and out.timestamp.is_unique


@pytest.mark.skipif(
    not (REPO / "data/energy_history.csv").exists() or not (REPO / "scripts/backtest_regime_label.py").exists(),
    reason="needs pulled data + scripts",
)
def test_parity_with_backtest_regime_label():
    sys.path.insert(0, str(REPO / "scripts"))
    import backtest_regime_label as brl

    ref, got = brl.load_live_inputs(), load_inputs()
    pd.testing.assert_frame_equal(got.energy, ref["energy"])
    pd.testing.assert_frame_equal(got.ev_df, ref["ev_df"])
