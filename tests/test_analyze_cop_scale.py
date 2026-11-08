"""Tests for scripts/analyze_cop_scale.py pure functions (no HA access)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "analyze_cop_scale.py"
_spec = importlib.util.spec_from_file_location("analyze_cop_scale", _SCRIPT)
acs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acs)


def _rows(start_utc: str, values: list[float], key: str) -> list[dict]:
    t0 = pd.Timestamp(start_utc)
    return [{"start": int((t0 + pd.Timedelta(hours=i)).timestamp() * 1000), key: v} for i, v in enumerate(values)]


class TestHourlyDelta:
    def test_diff_of_cumulative_sum(self):
        s = acs.rows_to_hourly_delta(_rows("2026-11-01T00:00Z", [10.0, 11.5, 13.0, 13.0], "sum"), 5.0)
        assert s.tolist()[1:] == [1.5, 1.5, 0.0]
        assert np.isnan(s.iloc[0])

    def test_drops_negative_and_implausible_steps(self):
        s = acs.rows_to_hourly_delta(_rows("2026-11-01T00:00Z", [10.0, 9.0, 30.0, 31.0], "sum"), 5.0)
        assert np.isnan(s.iloc[1]) and np.isnan(s.iloc[2])
        assert s.iloc[3] == 1.0

    def test_converts_to_local_naive_time(self):
        s = acs.rows_to_hourly_delta(_rows("2026-11-01T00:00Z", [0.0, 1.0], "sum"), 5.0)
        assert s.index[0] == pd.Timestamp("2026-11-01 01:00")  # CET = UTC+1

    def test_dst_fallback_duplicate_local_hour_kept_once(self):
        # 2026-10-25 00:00Z and 01:00Z both map to local 02:00 (fall-back)
        s = acs.rows_to_hourly_delta(_rows("2026-10-24T23:00Z", [0.0, 1.0, 2.0, 3.0], "sum"), 5.0)
        assert s.index.is_unique

    def test_empty_rows_give_datetime_index(self):
        s = acs.rows_to_hourly_delta([{"start": 0, "sum": None}], 5.0)
        assert isinstance(s.index, pd.DatetimeIndex) and s.empty


def _hourly_day(day: str, heat: float, el_heat: float, el_dhw: float, t_out: float, cop_live=None, hours=24):
    idx = pd.date_range(day, periods=hours, freq="h")
    return pd.DataFrame(
        {
            "heat": heat / hours,
            "el_total": (el_heat + el_dhw) / hours,
            "el_heat": el_heat / hours,
            "el_dhw": el_dhw / hours,
            "pv": 0.0,
            "t_out": t_out,
            "t_in": 21.0,
            "cop_live": cop_live,
        },
        index=idx,
    )


class TestDailyFrame:
    def test_aggregates_complete_day(self):
        d = acs.daily_frame(_hourly_day("2026-11-01", 48.0, 10.0, 2.0, 8.0, cop_live=4.0))
        row = d.loc[pd.Timestamp("2026-11-01")]
        assert row["heat"] == pytest.approx(48.0)
        assert row["el_heat"] == pytest.approx(10.0)
        assert row["cop_live_w"] == pytest.approx(4.0)

    def test_incomplete_day_dropped(self):
        d = acs.daily_frame(_hourly_day("2026-11-01", 48.0, 10.0, 2.0, 8.0, hours=20))
        assert d.empty

    def test_live_cop_is_el_weighted(self):
        h = _hourly_day("2026-11-01", 48.0, 10.0, 2.0, 8.0, cop_live=4.0)
        h.loc[h.index[:12], "el_heat"] = 0.0  # first half no heating -> its COP must not count
        h.loc[h.index[:12], "cop_live"] = 99.0
        d = acs.daily_frame(h)
        assert d["cop_live_w"].iloc[0] == pytest.approx(4.0)


def _days(n: int, cop_h: float, cop_d: float, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    el_h = rng.uniform(2, 15, n)
    el_d = rng.uniform(0.5, 3, n)
    t_out = rng.uniform(0, 15, n)
    return pd.DataFrame(
        {
            "el_heat": el_h,
            "el_dhw": el_d,
            "heat": cop_h * el_h + cop_d * el_d,
            "t_out": t_out,
            "t_in": 21.0,
            "pv": 0.0,
            "cop_live_w": cop_h,
        },
        index=pd.date_range("2026-10-01", periods=n, freq="D"),
    )


class TestFits:
    def test_split_cop_recovers_exact_coefficients(self):
        cop_h, cop_d = acs.fit_split_cop(_days(30, 4.5, 3.2))
        assert cop_h == pytest.approx(4.5) and cop_d == pytest.approx(3.2)

    def test_heating_cop_given_dhw(self):
        assert acs.heating_cop_given_dhw(_days(10, 4.5, 3.2), 3.2) == pytest.approx(4.5)

    def test_bootstrap_interval_brackets_noiseless_value(self):
        lo, hi = acs.bootstrap(_days(20, 4.5, 3.2), lambda x: acs.fit_split_cop(x)[0], n=50)
        assert lo == pytest.approx(4.5) and hi == pytest.approx(4.5)

    def test_fit_ua_recovers_ua(self):
        d = _days(20, 4.5, 3.2)
        q = 150.0 * (d["t_in"] - d["t_out"]) * 24 / 1000
        ua, r2 = acs.fit_ua(d, q, with_pv=False)
        assert ua == pytest.approx(150.0) and r2 == pytest.approx(1.0)

    def test_bin_table_uses_production_formula_shape(self):
        d = _days(40, 4.5, 3.2)
        tbl = acs.bin_table(d, 3.2, lambda t: np.full(len(np.atleast_1d(t)), 3.0))
        assert not tbl.empty
        assert np.allclose(tbl["COP_h counters"], 4.5)
        assert np.allclose(tbl["COP formula"], 3.0)


def test_formula_cop_fn_matches_physics_model():
    cfg = {
        "cop_formula": {"a": 2.5, "b": 0.07},
        "heating_curve_points": [[-20, 55.5], [-5, 46.0], [5, 39.5], [20, 25.0]],
    }
    f = acs.formula_cop_fn(cfg)
    assert f([10.0])[0] == pytest.approx(3.2)  # linear term binds at 10 C
    assert f([-10.0])[0] == pytest.approx(1.8)
