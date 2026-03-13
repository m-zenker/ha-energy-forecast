"""Tests for energy_forecast.py module-level helpers.

Covers:
  - _blend_today_totals: actuals substituted for elapsed hours, predictions for future
  - _compute_live_mae: correct MAE over matched pairs, nan on no overlap
  - EnergyForecast._retrain_cb / _update_cb: callable from both timer and event contexts
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.energy_forecast import _blend_today_totals, _compute_live_mae


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_predictions(today: pd.Timestamp, kwh: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """48 hourly predictions starting at today midnight, all equal to kwh."""
    ts = pd.date_range(today, periods=48, freq="1h")
    p_times = ts.values.astype("datetime64[ns]")
    p_vals  = np.full(48, kwh)
    return p_times, p_vals


def _make_actuals(start: pd.Timestamp, n: int, kwh: float = 3.0) -> pd.DataFrame:
    """n hourly actuals starting at start, all equal to kwh."""
    ts = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"timestamp": ts, "gross_kwh": [kwh] * n})


# ── _blend_today_totals ───────────────────────────────────────────────────────

class TestBlendTodayTotals:

    def _nts(self, ts: pd.Timestamp) -> np.datetime64:
        return np.datetime64(ts, "ns")

    def test_no_actuals_equals_prediction_sum(self):
        """With full_actuals=None the result must equal the prediction sum for today."""
        today   = pd.Timestamp("2026-03-12 00:00")
        now     = pd.Timestamp("2026-03-12 00:00")   # start of day — all future
        tmrw    = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=2.0)
        total, blocks = _blend_today_totals(
            p_times, p_vals, None,
            self._nts(today), self._nts(tmrw), self._nts(now),
        )
        # predictions cover [00:00, 48h); only today's 24 hours sum to 24 × 2.0 = 48.0
        assert abs(total - 48.0) < 1e-6

    def test_blended_total_uses_actuals_for_elapsed_hours(self):
        """Elapsed-hour actuals (3.0 kWh) replace predictions (1.0 kWh) in today's total."""
        today   = pd.Timestamp("2026-03-12 00:00")
        now     = pd.Timestamp("2026-03-12 12:00")   # noon — 12h elapsed
        tmrw    = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=1.0)
        actuals = _make_actuals(today, n=12, kwh=3.0)  # 12 elapsed hours at 3.0

        total, _ = _blend_today_totals(
            p_times, p_vals, actuals,
            self._nts(today), self._nts(tmrw), self._nts(now),
        )
        # 12 elapsed hours × 3.0 (actuals) + 12 future hours × 1.0 (predictions) = 48.0
        assert abs(total - (12 * 3.0 + 12 * 1.0)) < 1e-6

    def test_fully_elapsed_block_uses_actuals(self):
        """A 3h block entirely in the past must sum actuals, not predictions."""
        today   = pd.Timestamp("2026-03-12 00:00")
        now     = pd.Timestamp("2026-03-12 12:00")   # block 00-03 is fully elapsed
        tmrw    = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=1.0)
        actuals = _make_actuals(today, n=12, kwh=5.0)  # 12 elapsed hours at 5.0

        _, blocks = _blend_today_totals(
            p_times, p_vals, actuals,
            self._nts(today), self._nts(tmrw), self._nts(now),
        )
        # Block 00_03: 3 actual hours × 5.0 = 15.0
        assert abs(blocks["00_03"] - 15.0) < 1e-6

    def test_fully_future_block_uses_predictions(self):
        """A 3h block entirely in the future must sum predictions only."""
        today   = pd.Timestamp("2026-03-12 00:00")
        now     = pd.Timestamp("2026-03-12 00:00")   # nothing elapsed yet
        tmrw    = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=2.5)
        actuals = _make_actuals(today, n=0, kwh=0.0)  # empty — nothing elapsed

        _, blocks = _blend_today_totals(
            p_times, p_vals, actuals,
            self._nts(today), self._nts(tmrw), self._nts(now),
        )
        # Block 21_24: 3 future hours × 2.5 = 7.5
        assert abs(blocks["21_24"] - 7.5) < 1e-6


# ── _compute_live_mae ─────────────────────────────────────────────────────────

class TestComputeLiveMae:

    def test_matched_pairs_return_correct_mae(self):
        """When predictions and actuals overlap perfectly, MAE equals mean absolute error."""
        ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        pred_history = {t: 2.0 for t in ts}   # predicted 2.0 kWh every hour
        actuals = pd.DataFrame({
            "timestamp": ts,
            "gross_kwh": [3.0] * 24,           # actual was 3.0 → error = 1.0 each
        })
        mae, n = _compute_live_mae(pred_history, actuals)
        assert n == 24
        assert abs(mae - 1.0) < 1e-6

    def test_no_overlap_returns_nan_and_zero(self):
        """Pred history and actuals in non-overlapping windows → (nan, 0)."""
        pred_ts = pd.date_range("2026-03-10 00:00", periods=24, freq="1h")
        actual_ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        pred_history = {t: 1.0 for t in pred_ts}
        actuals = pd.DataFrame({"timestamp": actual_ts, "gross_kwh": [2.0] * 24})
        mae, n = _compute_live_mae(pred_history, actuals)
        assert n == 0
        assert math.isnan(mae)

    def test_partial_overlap_uses_matched_pairs_only(self):
        """Only overlapping timestamps contribute to MAE."""
        ts_all    = pd.date_range("2026-03-12 00:00", periods=48, freq="1h")
        ts_first  = ts_all[:24]   # predictions cover first 24h
        ts_second = ts_all[24:]   # actuals cover last 24h — no overlap

        pred_history = {t: 1.0 for t in ts_first}
        actuals = pd.DataFrame({
            "timestamp": pd.concat([
                pd.Series(ts_first[:12]),   # 12h overlap
                pd.Series(ts_second),       # 24h no overlap
            ]),
            "gross_kwh": [3.0] * 36,
        })
        mae, n = _compute_live_mae(pred_history, actuals)
        assert n == 12
        assert abs(mae - 2.0) < 1e-6   # |3.0 - 1.0| = 2.0 for each of the 12 matched


# ── Interval blending (#13) ───────────────────────────────────────────────────

class TestIntervalBlend:

    def _nts(self, ts: pd.Timestamp) -> np.datetime64:
        return np.datetime64(ts, "ns")

    def test_low_total_less_than_high_total(self):
        """Blending q10 vals gives today_low < today_high when q10 < q90 for future hours."""
        today  = pd.Timestamp("2026-03-12 00:00")
        now    = pd.Timestamp("2026-03-12 00:00")  # nothing elapsed
        tmrw   = today + pd.Timedelta(days=1)

        ts = pd.date_range(today, periods=48, freq="1h")
        p_times = ts.values.astype("datetime64[ns]")

        low_vals  = np.full(48, 1.0)   # q10 = 1.0 kWh/h
        high_vals = np.full(48, 3.0)   # q90 = 3.0 kWh/h

        today_low,  _ = _blend_today_totals(p_times, low_vals,  None, self._nts(today), self._nts(tmrw), self._nts(now))
        today_high, _ = _blend_today_totals(p_times, high_vals, None, self._nts(today), self._nts(tmrw), self._nts(now))

        assert today_low < today_high

    def test_fully_elapsed_window_interval_collapses_to_actuals(self):
        """When all hours are elapsed, both low and high equal the actuals sum."""
        today  = pd.Timestamp("2026-03-12 00:00")
        now    = pd.Timestamp("2026-03-12 06:00")  # 6h elapsed
        tmrw   = today + pd.Timedelta(days=1)

        p_times   = pd.date_range(now, periods=48, freq="1h").values.astype("datetime64[ns]")
        low_vals  = np.full(48, 1.0)
        high_vals = np.full(48, 5.0)
        actuals   = pd.DataFrame({
            "timestamp": pd.date_range(today, periods=6, freq="1h"),
            "gross_kwh": [2.0] * 6,
        })
        # Block 00_03 is fully elapsed → both bounds must equal 3×2.0 = 6.0
        _, blocks_low  = _blend_today_totals(p_times, low_vals,  actuals, self._nts(today), self._nts(tmrw), self._nts(now))
        _, blocks_high = _blend_today_totals(p_times, high_vals, actuals, self._nts(today), self._nts(tmrw), self._nts(now))

        assert abs(blocks_low["00_03"]  - 6.0) < 1e-6
        assert abs(blocks_high["00_03"] - 6.0) < 1e-6


# ── _aggregate ────────────────────────────────────────────────────────────────

def _fake_self_for_aggregate(ev_threshold: float = 4.5, ev_charger_kw: float = 9.0) -> "_FakeSelf":
    app = _FakeSelf()
    app._ev_threshold  = ev_threshold
    app._ev_charger_kw = ev_charger_kw
    return app


def _pred_df(start: pd.Timestamp, n: int = 48, kwh: float = 1.0) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"timestamp": ts, "predicted_kwh": np.full(n, kwh)})


class TestAggregate:
    """_aggregate wires predictions + actuals into the sensor value dict."""

    def _run(self, today, now, predictions, actuals=None, intervals=None):
        from energy_forecast.energy_forecast import EnergyForecast
        return EnergyForecast._aggregate(
            _fake_self_for_aggregate(), predictions, actuals, live_temp=None, intervals=intervals
        )

    def test_next_3h_sums_only_three_future_hours(self):
        """next_3h must equal exactly 3 × per-hour prediction, not more."""
        today = pd.Timestamp.now().normalize()
        now   = pd.Timestamp.now().floor("1h")
        preds = _pred_df(now, n=48, kwh=2.0)
        result = self._run(today, now, preds)
        assert abs(result["next_3h"] - 6.0) < 1e-3

    def test_tomorrow_uses_predictions_only(self):
        """tomorrow must equal 24 × per-hour prediction for the following calendar day."""
        today = pd.Timestamp.now().normalize()
        tmrw  = today + pd.Timedelta(days=1)
        preds = _pred_df(today, n=48, kwh=3.0)
        result = self._run(today, today, preds)
        assert abs(result["tomorrow"] - 72.0) < 1e-3   # 24 × 3.0

    def test_ev_sensors_zero_when_no_actuals(self):
        """ev_today and ev_yesterday must be 0.0 when full_actuals is None."""
        today = pd.Timestamp.now().normalize()
        preds = _pred_df(today, n=48, kwh=1.0)
        result = self._run(today, today, preds, actuals=None)
        assert result["ev_today"]     == 0.0
        assert result["ev_yesterday"] == 0.0

    def test_interval_keys_present_when_intervals_supplied(self):
        """When intervals DataFrame is provided, *_low/*_high keys appear in result."""
        today = pd.Timestamp.now().normalize()
        now   = today
        preds = _pred_df(today, n=48, kwh=1.0)
        ts    = pd.date_range(today, periods=48, freq="1h")
        ivs   = pd.DataFrame({
            "timestamp": ts,
            "low_kwh":   np.full(48, 0.5),
            "high_kwh":  np.full(48, 2.0),
        })
        result = self._run(today, now, preds, intervals=ivs)
        for key in ("next_3h_low", "next_3h_high", "today_low", "today_high",
                    "tomorrow_low", "tomorrow_high"):
            assert key in result, f"missing key: {key}"

    def test_interval_keys_absent_when_intervals_is_none(self):
        """Without quantile models, no *_low/*_high keys should appear."""
        today = pd.Timestamp.now().normalize()
        preds = _pred_df(today, n=48, kwh=1.0)
        result = self._run(today, today, preds, intervals=None)
        for key in ("next_3h_low", "next_3h_high", "today_low", "today_high"):
            assert key not in result, f"unexpected key: {key}"

    def test_blocks_today_has_eight_slots(self):
        """blocks_today must contain exactly 8 slots (00_03 … 21_24)."""
        today = pd.Timestamp.now().normalize()
        preds = _pred_df(today, n=48, kwh=1.0)
        result = self._run(today, today, preds)
        assert len(result["blocks_today"]) == 8
        assert "00_03" in result["blocks_today"]
        assert "21_24" in result["blocks_today"]


# ── EV kWh sensor calculation ────────────────────────────────────────────────

class TestEvKwhSensorCalc:
    """EV kWh values published to HA sensors use charger_kw, not threshold."""

    def _nts(self, ts: pd.Timestamp) -> np.datetime64:
        return np.datetime64(ts, "ns")

    def _run_aggregate(self, ev_charger_kw: float, ev_threshold: float) -> dict:
        """Run _aggregate with synthetic actuals containing one EV hour."""
        from energy_forecast.energy_forecast import EnergyForecast

        today = pd.Timestamp.now().normalize()
        tmrw  = today + pd.Timedelta(days=1)
        now   = today  # start of day — nothing elapsed yet

        # Predictions: 48h at 1.0 kWh/h
        ts     = pd.date_range(today, periods=48, freq="1h")
        p_df   = pd.DataFrame({"timestamp": ts, "predicted_kwh": np.ones(48)})

        # Actuals: one EV-level hour today (gross=12 kWh, clearly above any threshold)
        actuals = pd.DataFrame({
            "timestamp": [today + pd.Timedelta(hours=2)],
            "gross_kwh": [12.0],
        })

        app = _FakeSelf()
        app._ev_threshold  = ev_threshold
        app._ev_charger_kw = ev_charger_kw

        return EnergyForecast._aggregate(app, p_df, actuals, live_temp=None)

    def test_ev_today_uses_charger_kw_not_threshold(self):
        """ev_today must equal gross − charger_kw, not gross − threshold."""
        result = self._run_aggregate(ev_charger_kw=7.4, ev_threshold=4.5)
        # 12.0 − 7.4 = 4.6 (not 12.0 − 4.5 = 7.5)
        assert abs(result["ev_today"] - 4.6) < 1e-6

    def test_ev_today_default_charger_kw(self):
        """Default charger_kw=9.0: ev_today = gross − 9.0."""
        result = self._run_aggregate(ev_charger_kw=9.0, ev_threshold=4.5)
        assert abs(result["ev_today"] - 3.0) < 1e-6


# ── Callback signature compatibility ─────────────────────────────────────────

class _FakeSelf:
    """Minimal stand-in for EnergyForecast used in callback / aggregate tests.

    EnergyForecast inherits from MagicMock (via the hassapi stub) so
    object.__new__ leaves mock internals uninitialized.  A plain class with
    the attributes touched by the tested methods is sufficient.
    """
    def __init__(self):
        self._ml_model = MagicMock()
        self._ml_model.model = None  # prevents _update_sensors execution
        self._lock = MagicMock()
        self._lock.acquire.return_value = False  # always "busy" → immediate return

    def log(self, msg, level="INFO"):  # AppDaemon log stub
        pass


class TestCallbackSignature:
    """_retrain_cb and _update_cb must accept both timer and event calling conventions."""

    def test_retrain_cb_timer_style(self):
        """Timer callbacks pass a single positional dict — must not raise TypeError."""
        from energy_forecast.energy_forecast import EnergyForecast
        EnergyForecast._retrain_cb(_FakeSelf(), {})

    def test_retrain_cb_event_style(self):
        """listen_event callbacks pass (event_name, data, kwargs) — must not raise TypeError."""
        from energy_forecast.energy_forecast import EnergyForecast
        EnergyForecast._retrain_cb(_FakeSelf(), "RELOAD_ENERGY_MODEL", {}, {})

    def test_update_cb_timer_style(self):
        from energy_forecast.energy_forecast import EnergyForecast
        EnergyForecast._update_cb(_FakeSelf(), {})

    def test_update_cb_event_style(self):
        from energy_forecast.energy_forecast import EnergyForecast
        EnergyForecast._update_cb(_FakeSelf(), "some_event", {}, {})
