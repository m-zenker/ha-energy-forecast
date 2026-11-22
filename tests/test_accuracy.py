import math

import numpy as np
import pandas as pd
import pytest
from energy_forecast import accuracy

from backtest import wf_score

START = pd.Timestamp("2026-11-01")


def _hist(values, start=START):
    return dict(zip(pd.date_range(start, periods=len(values), freq="1h"), values))


def test_constant_offset_two_days():
    actual = _hist([2.0] * 240)
    pred = {ts: 2.5 for ts in list(actual)[-48:]}
    s = accuracy.summarize(accuracy.paired_errors(pred, actual))
    assert s["n_hours"] == 48 and s["n_blocks"] == 16 and s["n_days"] == 2
    assert s["mae"] == pytest.approx(0.5)
    assert s["block_mae"] == pytest.approx(1.5)
    assert s["daily_bias"] == pytest.approx(12.0) and s["daily_abs_error"] == pytest.approx(12.0)
    assert s["daily_bias_pct"] == pytest.approx(25.0)
    assert s["naive_mae"] == pytest.approx(0.0)
    assert math.isnan(s["skill_pct"]) and math.isnan(s["block_skill_pct"])  # naive error is 0


def test_errors_cancel_inside_a_block():
    actual = _hist([2.0] * 240)
    last = list(actual)[-24:]
    pred = {ts: 2.0 + (1.0, -1.0, 0.0)[i % 3] for i, ts in enumerate(last)}
    s = accuracy.summarize(accuracy.paired_errors(pred, actual))
    assert s["mae"] == pytest.approx(2 / 3)
    assert s["block_mae"] == pytest.approx(0.0)
    assert s["daily_bias"] == pytest.approx(0.0)


def test_incomplete_day_and_block_are_not_scored():
    actual = _hist([2.0] * 240)
    last = list(actual)[-24:-3]  # 21 hours of the final day
    s = accuracy.summarize(accuracy.paired_errors({ts: 2.5 for ts in last}, actual))
    assert s["n_days"] == 0 and math.isnan(s["daily_bias"]) and math.isnan(s["daily_bias_pct"])
    assert s["n_blocks"] == 7 and s["mae"] == pytest.approx(0.5)


def test_mid_block_hole_block_is_not_scored():
    actual = _hist([2.0] * 240)
    last = list(actual)[-24:]
    pred = {ts: 2.5 for ts in last}
    full = accuracy.summarize(accuracy.paired_errors(pred, actual))
    hole = dict(actual)
    del hole[last[4]]  # 04:00 -> block 03-06 has only 2 paired hours
    s = accuracy.summarize(accuracy.paired_errors(pred, hole))
    assert full["n_blocks"] == 8 and s["n_blocks"] == 7 and s["n_hours"] == 23
    assert s["block_mae"] == pytest.approx(1.5)  # only complete blocks: 3 * 0.5 each
    assert s["mae"] == pytest.approx(0.5) and s["n_days"] == 1  # 23 h still counts as a day


def test_block_mae_matches_wf_score_on_complete_days():
    rng = np.random.default_rng(1)
    actual = pd.Series(rng.uniform(0, 3, 24 * 12), index=pd.date_range(START, periods=24 * 12, freq="1h"))
    pred = actual.iloc[-72:] + rng.normal(0, 0.5, 72)
    s = accuracy.summarize(accuracy.paired_errors(pred.to_dict(), actual.to_dict()))
    rows = pd.DataFrame({"arm": "a", "ts": pred.index, "y": actual.loc[pred.index].values, "p50": pred.values})
    rows["day"] = rows["ts"].dt.normalize()
    table = wf_score.day_table(rows)
    assert len(table) == 3
    assert s["block_mae"] == pytest.approx(table["mae3"].mean())
    assert s["mae"] == pytest.approx(table["mae"].mean())


def test_skill_against_naive():
    actual = _hist([1.0] * 216 + [3.0] * 24)  # last day jumps to 3.0; naive says 1.0
    last = list(actual)[-24:]
    s = accuracy.summarize(accuracy.paired_errors({ts: 3.0 for ts in last}, actual))
    assert s["naive_mae"] == pytest.approx(2.0) and s["skill_pct"] == pytest.approx(100.0)
    assert s["naive_block_mae"] == pytest.approx(6.0) and s["block_skill_pct"] == pytest.approx(100.0)
    half = accuracy.summarize(accuracy.paired_errors({ts: 2.0 for ts in last}, actual))
    assert half["skill_pct"] == pytest.approx(50.0)


def test_naive_matches_backtest_wf_score():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.uniform(0, 3, 24 * 20), index=pd.date_range(START, periods=24 * 20, freq="1h"))
    s = s.drop(s.index[[30, 31, 200, 333]])  # gaps (EV-excluded hours)
    pd.testing.assert_series_equal(accuracy.naive_same_hour_mean(s), wf_score.naive_same_hour_mean(s))


def test_since_filter_keeps_naive_history_and_floors_keys():
    actual = _hist([1.0] * 240)
    keys = list(actual)
    pred = {str(ts + pd.Timedelta(minutes=5)): 1.0 for ts in keys[-72:]}  # unfloored string keys
    pairs = accuracy.paired_errors(pred, actual, since=keys[-24])
    assert len(pairs) == 24 and pairs.index[0] == keys[-24]
    assert pairs["naive"].notna().all()  # computed from the full history, not the filtered window


def test_empty_and_young_history_give_nan_not_errors():
    s = accuracy.summarize(accuracy.paired_errors({}, {}))
    assert s["n_hours"] == 0 and s["n_blocks"] == 0 and s["n_days"] == 0
    assert all(math.isnan(v) for k, v in s.items() if not k.startswith("n_"))
    young = _hist([1.0] * 30)
    s2 = accuracy.summarize(accuracy.paired_errors({ts: 1.2 for ts in young}, young))
    assert s2["mae"] == pytest.approx(0.2) and math.isnan(s2["naive_mae"]) and math.isnan(s2["skill_pct"])


def test_summarize_windows_scores_only_complete_calendar_days():
    # Day d (Nov 1 = 0) has a constant per-hour error of 0.1 * (d + 1) -> daily error sum 2.4 * (d + 1).
    now = pd.Timestamp("2026-11-10 22:00")  # today has 22 paired hours -> would pass MIN_DAY_HOURS
    hours = pd.date_range(START, now - pd.Timedelta(hours=1), freq="1h")
    actual = {ts: 2.0 for ts in hours}
    pred = {ts: 2.0 + 0.1 * ((ts.normalize() - START).days + 1) for ts in hours}
    today = now.normalize()
    s = accuracy.summarize_windows(
        pred, actual, since=now - pd.Timedelta(days=7), day_start=today - pd.Timedelta(days=7), day_end=today
    )
    assert s["n_days"] == 7  # Nov 3..9 in full: oldest day not truncated by the rolling cutoff, today excluded
    assert s["daily_bias"] == pytest.approx(2.4 * np.mean(range(3, 10)))
    assert s["daily_abs_error"] == pytest.approx(s["daily_bias"])
    assert s["daily_bias_pct"] == pytest.approx(100 * sum(2.4 * d for d in range(3, 10)) / (7 * 48.0))
    assert s["n_hours"] == 7 * 24  # hourly / block metrics keep the rolling window


def test_summarize_windows_without_day_bounds_matches_summarize():
    actual = _hist([2.0] * 240)
    pred = {ts: 2.5 for ts in list(actual)[-48:]}
    got = accuracy.summarize_windows(pred, actual)
    want = accuracy.summarize(accuracy.paired_errors(pred, actual))
    assert got.keys() == want.keys()
    assert all(got[k] == want[k] or (math.isnan(got[k]) and math.isnan(want[k])) for k in want)
