"""Tests for the backtest CLI and the first experiment."""

import types

import pandas as pd

from backtest.__main__ import main
from backtest.arms import Arm, feature_drop
from backtest.experiments import temp_ablation
from backtest.repro import replay_series


def test_unknown_experiment_exits_2(capsys):
    assert main(["nope"]) == 2 and "nope" in capsys.readouterr().err


def test_reference_arm_must_be_base(monkeypatch, capsys):
    fake = types.ModuleType("backtest.experiments.fake")
    fake.ARMS = [Arm("x")]
    monkeypatch.setitem(__import__("sys").modules, "backtest.experiments.fake", fake)
    assert main(["fake"]) == 2 and "base" in capsys.readouterr().err


def test_temp_ablation_arms_are_valid_static_drops():
    for arm in temp_ablation.ARMS:
        with feature_drop(arm.drop):
            pass
    assert temp_ablation.ARMS[0].name == "base"
    assert [a.name for a in temp_ablation.ARMS] == ["base", "-T", "-T-regime", "-regime", "clim", "-sun"]
    assert all(a.train_kwargs == {"regime_count": 3} for a in temp_ablation.ARMS)


def test_replay_series_empty_frame():  # gate window with no scorable days → run_experiment returns DataFrame()
    assert replay_series(pd.DataFrame()).empty


def test_repro_failure_becomes_header_line(tmp_path):
    from backtest.__main__ import _repro_lines

    lines = _repro_lines(None, None, {}, data_dir=tmp_path)  # no pred_history.json
    assert len(lines) == 1 and lines[0].startswith("live comparison: unavailable")


def test_defrost_risk_counts_as_temperature():
    assert "defrost_risk" in temp_ablation.TEMP_GROUP and "defrost_risk" not in temp_ablation.SUN_GROUP
