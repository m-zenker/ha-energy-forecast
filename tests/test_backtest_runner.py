"""Tests for backtest/runner.py — real train()/predict() on synthetic data, leak and window guards."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from energy_forecast.model import EnergyForecastModel

from backtest.arms import Arm
from backtest.data import Fold, Inputs, train_defaults
from backtest.runner import run_arm, run_experiment
from tests._backtest_synth import synthetic_energy, synthetic_weather

DEFAULTS = train_defaults({})  # regimes off — keeps each train ~1 s


@dataclass
class Synth:
    inputs: Inputs
    weather: pd.DataFrame
    fold: Fold
    end: pd.Timestamp  # last energy timestamp
    gap_day: pd.Timestamp | None = None


def _make_synth(gap: bool = False) -> Synth:
    weather = synthetic_weather(days=44)
    energy = synthetic_energy(weather).iloc[: 40 * 24]
    fold = Fold("f", pd.Timestamp("2026-02-07"), pd.Timestamp("2026-02-09"))  # days 37-39
    gap_day = None
    if gap:
        gap_day = pd.Timestamp("2026-02-08")
        energy = energy[~energy.timestamp.between(gap_day + pd.Timedelta(hours=3), gap_day + pd.Timedelta(hours=8))]
    return Synth(Inputs(energy.reset_index(drop=True), None), weather, fold, energy.timestamp.max(), gap_day)


@pytest.fixture(scope="module")
def synth() -> Synth:
    return _make_synth()


@pytest.fixture(scope="module")
def base_rows(synth):
    return run_arm(Arm("base"), synth.fold, synth.inputs, synth.weather, DEFAULTS)


@dataclass
class _Calls:
    train: list = field(default_factory=list)
    predict: list = field(default_factory=list)


def spy(monkeypatch) -> _Calls:
    """Record every train/predict call with the (frozen) clock value seen *inside* the call."""
    calls = _Calls()
    real_train, real_predict = EnergyForecastModel.train, EnergyForecastModel.predict

    def _train(self, *args, **kwargs):
        calls.train.append(SimpleNamespace(args=args, kwargs=kwargs, now=pd.Timestamp.now()))
        return real_train(self, *args, **kwargs)

    def _predict(self, *args, **kwargs):
        calls.predict.append(SimpleNamespace(args=args, kwargs=kwargs, now=pd.Timestamp.now()))
        return real_predict(self, *args, **kwargs)

    monkeypatch.setattr(EnergyForecastModel, "train", _train)
    monkeypatch.setattr(EnergyForecastModel, "predict", _predict)
    return calls


def test_schema_and_count(base_rows):
    assert len(base_rows) == 3 * 2 and {r["origin"] for r in base_rows} == {6, 18}
    assert all(r["pred"].shape == (24,) and np.isfinite(r["mae"]) for r in base_rows)
    assert {"arm", "fold", "day", "cold", "bias", "ts0"} <= set(base_rows[0])


def test_no_future_leakage(synth, monkeypatch):
    calls = spy(monkeypatch)
    run_arm(Arm("base"), synth.fold, synth.inputs, synth.weather, DEFAULTS, days_limit=1)
    assert calls.train[0].args[0].timestamp.max() < synth.fold.start
    assert calls.train[0].args[1].timestamp.max() < synth.fold.start
    for c in calls.predict:
        assert c.kwargs["recent_actuals"].timestamp.max() < c.now


def test_predict_window_has_contiguous_history(synth, monkeypatch):
    calls = spy(monkeypatch)
    run_arm(Arm("base"), synth.fold, synth.inputs, synth.weather, DEFAULTS)
    c = calls.predict[-1]  # last origin: ~2.75 days after fold start
    ts = c.args[0].timestamp
    assert ts.min() <= c.now - pd.Timedelta(days=14)
    assert (ts.diff().dropna() == pd.Timedelta(hours=1)).all()


def test_lead25_one_row_per_day(synth):
    rows = run_arm(Arm("base"), synth.fold, synth.inputs, synth.weather, DEFAULTS, origins="lead25", days_limit=1)
    assert len(rows) == 1 and rows[0]["origin"] == "lead25" and np.isfinite(rows[0]["pred"]).sum() == 24


def test_dropping_a_rebuilt_lag_raises(synth):
    with pytest.raises(ValueError, match="lag_48h"):
        run_arm(Arm("-lag48", drop=("lag_48h",)), synth.fold, synth.inputs, synth.weather, DEFAULTS, days_limit=1)


def test_day_with_gaps_skipped():
    s = _make_synth(gap=True)
    days = {r["day"] for r in run_arm(Arm("base"), s.fold, s.inputs, s.weather, DEFAULTS)}
    assert s.gap_day.date() not in days and len(days) == 2


def test_fold_beyond_data_returns_empty_without_training(synth, monkeypatch):
    calls = spy(monkeypatch)
    late = Fold("late", synth.end + pd.Timedelta(days=5), synth.end + pd.Timedelta(days=7))
    assert run_arm(Arm("base"), late, synth.inputs, synth.weather, DEFAULTS) == [] and not calls.train


def test_run_experiment_writes_pickle(synth, tmp_path):
    arms = [Arm("base"), Arm("-T", drop=("temp_c",))]
    df = run_experiment(arms, [synth.fold], synth.inputs, synth.weather, DEFAULTS, out=tmp_path / "r.pkl", days_limit=1)
    assert set(pd.read_pickle(tmp_path / "r.pkl").arm) == {"base", "-T"} and len(df) == 4
