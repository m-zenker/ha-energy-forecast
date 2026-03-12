"""Tests for energy_forecast.py module-level helpers.

Covers:
  - _blend_today_totals: actuals substituted for elapsed hours, predictions for future
  - _compute_live_mae: correct MAE over matched pairs, nan on no overlap
"""
from __future__ import annotations

import math

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
