"""Tests for backtest/repro.py — report-only comparison of a lead-25 replay with live pred_history."""

import json

import numpy as np
import pandas as pd
import pytest

from backtest.repro import gate_folds, load_live_pred, replay_series, repro_report

IDX = pd.date_range("2026-10-08 00:00", "2026-11-07 12:00", freq="h")  # ends mid-day → 11-07 incomplete
_rng = np.random.default_rng(5)
A = pd.Series(0.5 + 0.3 * np.sin(np.arange(len(IDX)) * 2 * np.pi / 24) + _rng.normal(0, 0.1, len(IDX)), index=IDX)
L = A + _rng.normal(0, 0.15, len(IDX))


def test_identical_series():
    r = repro_report(L, L, A)
    assert r.ratio == pytest.approx(1) and r.mean_abs_gap == 0 and r.err_corr == pytest.approx(1)
    assert r.n_hours == len(IDX) and "lean frame" in r.note


def test_too_few_hours_gives_no_numbers():
    r = repro_report(L[:48], L, A)
    assert r.n_hours == 48 and np.isnan(r.ratio) and np.isnan(r.mae_replay) and "too few" in r.note


def test_only_common_hours_used():
    assert repro_report(L.iloc[::2], L, A.iloc[:-10]).n_hours == len(L.iloc[::2].index.intersection(A.index[:-10]))


def test_gate_folds_are_weekly_and_cover_last_complete_days():
    f = gate_folds(L, days=28)
    assert len(f) == 4 and [x.name for x in f] == ["repro1", "repro2", "repro3", "repro4"]
    assert all((x.end - x.start).days <= 6 for x in f)
    assert f[-1].end == pd.Timestamp("2026-11-06") and f[0].start == pd.Timestamp("2026-10-10")
    assert all(b.start == a.end + pd.Timedelta(days=1) for a, b in zip(f, f[1:]))


def test_replay_series_flattens_lead25_rows():
    day = pd.Timestamp("2026-11-01")
    rows = pd.DataFrame(
        [
            dict(arm="base", origin="lead25", ts0=day, pred=np.arange(24.0)),
            dict(arm="base", origin=6, ts0=day, pred=np.full(24, 9.0)),  # not lead25 → ignored
            dict(arm="x", origin="lead25", ts0=day, pred=np.full(24, 7.0)),  # other arm → ignored
        ]
    )
    s = replay_series(rows)
    assert len(s) == 24 and s[day + pd.Timedelta(hours=5)] == 5.0


def test_load_live_pred_roundtrip(tmp_path):
    (tmp_path / "pred_history.json").write_text(
        json.dumps({"pred": {"2026-11-01T00:00:00": 0.5}, "actuals": {"2026-11-01T00:00:00": 0.4}})
    )
    p, a = load_live_pred(tmp_path)
    assert p.iloc[0] == 0.5 and a.iloc[0] == 0.4 and isinstance(p.index, pd.DatetimeIndex)
