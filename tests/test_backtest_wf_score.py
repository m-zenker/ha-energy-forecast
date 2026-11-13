"""backtest.wf_score / backtest.audit_intervals on hand-built forecast rows."""

import numpy as np
import pandas as pd
import pytest

from backtest import audit_intervals as ai
from backtest import wf_score as ws

HOURS = pd.date_range("2026-01-01", periods=30 * 24, freq="h")


def _actual() -> pd.Series:
    return pd.Series(1.0 + 0.5 * (HOURS.hour == 12), index=HOURS)


def _rows(arm: str, offset: float, origin_hours=(0,), lead0: int = 24, days=range(10, 30)) -> pd.DataFrame:
    """Forecast = actual + offset for every hour of each target day, issued the day before."""
    actual = _actual()
    out = []
    for d in days:
        day = HOURS[0] + pd.Timedelta(days=d)
        for oh in origin_hours:
            origin = day - pd.Timedelta(days=1) + pd.Timedelta(hours=oh)
            ts = pd.date_range(day, periods=24, freq="h")
            lead = ((ts - origin) / pd.Timedelta(hours=1)).astype(int)
            p = actual.reindex(ts).to_numpy() + offset
            out.append(
                pd.DataFrame(
                    {"arm": arm, "origin": origin, "ts": ts, "lead": lead, "p50": p, "lo": p - 0.2, "hi": p + 0.2}
                )
            )
    return pd.concat(out, ignore_index=True)


def _scored(*parts) -> pd.DataFrame:
    temp = pd.Series(5.0, index=pd.date_range("2026-01-01", periods=30, freq="D"))
    return ws.attach_actuals(pd.concat(parts, ignore_index=True), _actual(), temp)


def test_naive_is_the_same_hour_mean_of_days_minus_2_to_8():
    actual = _actual().copy()
    actual.loc["2026-01-20 12:00"] = 9.0
    naive = ws.naive_same_hour_mean(actual)
    assert naive.loc["2026-01-21 12:00"] == pytest.approx(1.5)  # day -1 is not used
    assert naive.loc["2026-01-22 12:00"] == pytest.approx((9.0 + 6 * 1.5) / 7)
    assert np.isnan(naive.iloc[0])


def test_day_table_block_and_daily_metrics():
    t = ws.day_table(_scored(_rows("base", 0.1)))
    row = t.xs("base", level="arm").iloc[0]
    assert row["mae"] == pytest.approx(0.1) and row["mae3"] == pytest.approx(0.3) and row["mae6"] == pytest.approx(0.6)
    assert row["bias"] == pytest.approx(2.4) and row["cover"] == 1.0 and row["width"] == pytest.approx(0.4)


def test_block_errors_cancel_inside_a_block():
    rows = _rows("base", 0.0)
    rows.loc[rows["ts"].dt.hour % 2 == 0, "p50"] += 0.2
    rows.loc[rows["ts"].dt.hour % 2 == 1, "p50"] -= 0.2
    t = ws.day_table(_scored(rows))
    assert t["mae"].iloc[0] == pytest.approx(0.2)
    assert t["mae6"].iloc[0] == pytest.approx(0.0, abs=1e-9)  # timing error only: totals are right
    assert t["bias"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_day_ahead_keeps_the_newest_forecast_with_enough_lead():
    rows = _scored(_rows("base", 0.1, origin_hours=(0,)), _rows("base", 0.3, origin_hours=(-12,)))
    da = ws.day_ahead(rows)
    assert len(da) == 20 * 24 and da["p50"].sub(da["y"]).round(6).eq(0.1).all()
    assert ws.day_ahead(_scored(_rows("base", 0.1, origin_hours=(6,)))).groupby("ts").size().max() == 1


def test_incomplete_days_are_dropped():
    rows = _scored(_rows("base", 0.1))
    short = rows[~((rows["day"] == rows["day"].min()) & (rows["ts"].dt.hour >= 10))]
    assert len(ws.day_table(short)) == 19


def test_summarize_cold_bias_in_percent_and_gate():
    s = ws.summarize(ws.day_table(ws.add_naive(_scored(_rows("base", 0.1)))))
    base = s.loc["base"]
    assert base["cold_days"] == 20 and base["cold_bias"] == pytest.approx(2.4)
    assert base["cold_bias_pct"] == pytest.approx(100 * 2.4 / 24.5)
    assert ws.cold_gate(base) == "FAIL"  # +9.8 % is above +6 %
    assert ws.cold_gate(base, min_days=30) == "INCONCLUSIVE"
    assert ws.cold_gate(ws.summarize(ws.day_table(_scored(_rows("base", -0.05)))).loc["base"]) == "PASS"
    assert s.loc["naive", "mae"] == pytest.approx(0.0, abs=1e-9)  # the synthetic load repeats daily


def test_paired_relative_change_and_ci():
    t = ws.day_table(_scored(_rows("base", 0.1), _rows("x", 0.2)))
    r = ws.paired(t, "x")
    assert r["pct"] == pytest.approx(100.0) and r["lo"] == pytest.approx(100.0) and r["worse_days"] == 20
    assert ws.paired(t, "x", metric="abs_bias")["pct"] == pytest.approx(100.0)
    assert ws.paired(t, "x", seed=3) == ws.paired(t, "x", seed=3)


def test_next_day_swing_measures_revisions_between_origins():
    a, b = _rows("base", 0.0, origin_hours=(6,)), _rows("base", 0.1, origin_hours=(18,))
    swing = ws.next_day_swing(_scored(a, b))
    assert swing.loc["base", "range"] == pytest.approx(2.4) and swing.loc["base", "step"] == pytest.approx(2.4)


def test_coverage_by_lead_counts_misses_on_each_side():
    rows = _scored(_rows("base", 0.0))
    rows.loc[rows["ts"].dt.hour == 3, "y"] += 1.0
    rows.loc[rows["ts"].dt.hour == 4, "y"] -= 1.0
    c = ws.coverage_by_lead(rows)
    assert c["n"].sum() == 20 * 24
    assert (c["n"] * c["above"]).sum() == pytest.approx(20) and (c["n"] * c["below"]).sum() == pytest.approx(20)


def test_summed_band_is_wider_than_the_hourly_one():
    rows = _scored(_rows("base", 0.0))
    rows.loc[rows["ts"].dt.hour % 3 == 0, "y"] += 0.3  # one miss per 3 h block
    assert ai._coverage(rows["y"], rows["lo"], rows["hi"])["cover"] == pytest.approx(2 / 3)
    block = ai.summed_band_coverage(rows, 3)
    assert block["cover"] == 1.0 and block["width"] == pytest.approx(1.2) and block["n"] == 20 * 8
    assert ai.summed_band_coverage(rows, 24)["n"] == 20


def test_recalibration_uses_only_earlier_days():
    rows = _scored(_rows("base", 0.0))
    rows.loc[rows["ts"].dt.hour % 3 == 0, "y"] += 0.3
    r = ai.recalibrate_hourly(rows, window_days=21, min_days=10)
    first_days = r[r["day"] < r["day"].min() + pd.Timedelta(days=10)]
    assert first_days["lo_c"].isna().all()
    late = r.dropna(subset=["lo_c"])
    assert late["day"].nunique() == 10 and ((late["y"] >= late["lo_c"]) & (late["y"] <= late["hi_c"])).mean() > 0.9
    report = ai.report(rows)
    assert report.loc["hourly, recalibrated on served errors", "cover"] > report.loc["hourly, as shipped", "cover"]
