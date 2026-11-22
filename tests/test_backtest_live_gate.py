import json

import pandas as pd
import pytest

from backtest import live_gate

START = pd.Timestamp("2026-11-01")


def _series(days: int, value: float) -> pd.Series:
    return pd.Series(value, index=pd.date_range(START, periods=24 * days, freq="1h"))


def _temp(days: int, value: float = 5.0) -> pd.Series:
    return pd.Series(value, index=pd.date_range(START, periods=days, freq="D"))


def _gate(pred_value, days=20, temp=5.0, basis="target", **kw):
    rows = live_gate.live_rows(_series(days, pred_value), _series(days, 1.0), _temp(days, temp))
    return live_gate.verdict(rows, basis, **kw)


def test_pass_inside_band():
    v = _gate(1.02)
    assert v["gate"] == "PASS" and v["reason"] == ""
    assert v["cold_days"] == 20 and v["cold_bias_pct"] == pytest.approx(2.0)
    assert v["cold_bias"] == pytest.approx(0.48)  # 0.02 kWh/h x 24


def test_fail_above_and_below_band():
    assert _gate(1.10)["gate"] == "FAIL"
    low = _gate(0.85)
    assert low["gate"] == "FAIL" and "outside" in low["reason"]


def test_just_inside_and_outside_the_band_edges():
    # not exactly 1.06 / 0.88: float rounding puts those a hair outside the inclusive edge
    assert _gate(1.059)["gate"] == "PASS" and _gate(1.061)["gate"] == "FAIL"
    assert _gate(0.881)["gate"] == "PASS" and _gate(0.879)["gate"] == "FAIL"


def test_inconclusive_below_min_cold_days_even_with_large_bias():
    v = _gate(1.50, days=10)
    assert v["gate"] == "INCONCLUSIVE" and "10 cold day" in v["reason"]


def test_warm_days_do_not_count():
    v = _gate(1.02, temp=15.0)
    assert v["gate"] == "INCONCLUSIVE" and v["cold_days"] == 0


def test_unmarked_actuals_are_inconclusive_unless_allowed():
    assert _gate(1.02, basis=None)["gate"] == "INCONCLUSIVE"
    assert "target basis" in _gate(1.02, basis="gross")["reason"]
    assert _gate(1.02, basis=None, allow_mixed_actuals=True)["gate"] == "PASS"


def test_since_starts_24h_after_the_anchor_rounded_up():
    rows = live_gate.live_rows(_series(20, 1.0), _series(20, 1.0), _temp(20), since=pd.Timestamp("2026-11-05 17:28"))
    assert rows["ts"].min() == pd.Timestamp("2026-11-06 18:00")


def test_days_without_temperature_are_not_cold():
    rows = live_gate.live_rows(_series(20, 1.02), _series(20, 1.0), _temp(15))
    assert live_gate.verdict(rows, "target")["cold_days"] == 15


def test_cold_below_override():
    rows = live_gate.live_rows(_series(20, 1.02), _series(20, 1.0), _temp(20, 12.0), cold_below=13.5)
    assert live_gate.verdict(rows, "target")["cold_days"] == 20


def test_load_pred_history_round_trip(tmp_path):
    p = tmp_path / "pred_history.json"
    p.write_text(
        json.dumps(
            {
                "pred": {"2026-11-02T01:00:00": 1.5, "2026-11-02T00:00:00": 1.0},
                "actuals": {"2026-11-02T00:00:00": 0.9},
                "actuals_basis": "target",
            }
        )
    )
    pred, actual, basis = live_gate.load_pred_history(p)
    assert basis == "target" and list(pred) == [1.0, 1.5] and actual.index[0] == pd.Timestamp("2026-11-02")
    p.write_text(json.dumps({"pred": {}, "actuals": {}}))
    assert live_gate.load_pred_history(p)[2] is None


def test_naive_guardrail_is_reported():
    v = _gate(1.02)
    assert v["naive_mae3"] == pytest.approx(0.0)  # constant load: the naive mean is exact
    assert v["mae3"] == pytest.approx(0.06)


def test_no_actuals_is_inconclusive_not_a_crash():
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    v = live_gate.verdict(live_gate.live_rows(_series(20, 1.0), empty, _temp(20)), "target")
    assert v["gate"] == "INCONCLUSIVE" and v["cold_days"] == 0 and v["days"] == 0
    assert live_gate.verdict(live_gate.live_rows(empty, empty, _temp(20)), "target")["gate"] == "INCONCLUSIVE"


def test_missing_temperature_days_are_listed():
    rows = live_gate.live_rows(_series(20, 1.02), _series(20, 1.0), _temp(15))
    missing = live_gate.missing_temperature_days(rows, _temp(15))
    assert len(missing) == 5 and missing.min() == pd.Timestamp("2026-11-16")
    assert len(live_gate.missing_temperature_days(rows, _temp(20))) == 0
