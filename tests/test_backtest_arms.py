"""Tests for backtest/arms.py — declarative arms, feature drops, weather transforms."""

import energy_forecast.model as mm
import pytest

from backtest.arms import Arm, climatology, feature_drop, noise_floor_arms
from tests._backtest_synth import synthetic_weather


def test_feature_drop_removes_and_restores():
    orig = list(mm._FEATURES_BASE)
    with feature_drop(["temp_c", "regime_kwh"]):
        assert "temp_c" not in mm._FEATURES_BASE and "regime_kwh" not in mm._FEATURES_BASE
    assert mm._FEATURES_BASE == orig


def test_feature_drop_unknown_name_raises():
    orig = list(mm._FEATURES_BASE)
    with pytest.raises(ValueError, match="physics_kwh"):
        with feature_drop(["physics_kwh"]):
            pass
    assert mm._FEATURES_BASE == orig


def test_feature_drop_restores_on_exception():
    orig = list(mm._FEATURES_BASE)
    with pytest.raises(KeyError), feature_drop(["temp_c"]):
        raise KeyError
    assert mm._FEATURES_BASE == orig


def test_climatology_keeps_shape_and_kills_daily_variance():
    w = synthetic_weather(days=60)
    c = climatology(w)
    assert len(c) == len(w) and (c.timestamp.values == w.timestamp.values).all()
    assert c.drop(columns="temp_c").equals(w.drop(columns="temp_c"))

    def day_to_day(x):
        return x.groupby(x.timestamp.dt.date).temp_c.mean().diff().abs().mean()

    assert day_to_day(c) < 0.25 * day_to_day(w)


def test_noise_floor_arms():
    a = noise_floor_arms(Arm("base", train_kwargs={"regime_count": 3}))
    assert [x.name for x in a] == ["base@s1", "base@s2"]
    assert a[0].train_kwargs == {"regime_count": 3} and a[1].seed == 2
