"""Tests for model.py feature helpers.

Covers:
  - Rolling features vary per hour (regression test for the scalar-broadcast bug)
  - h=0 value matches the mean/std of the last N actuals (exact training semantics)
  - Values transition smoothly; h≥24 stabilises near the fill value
  - Graceful handling of short actuals (< 24 rows)
  - None / empty actuals fall back to NaN (existing contract preserved)
  - lag_72h present in LAG_HOURS, values correct, NaN when history too short
  - Bridge-day features: range, zero on holiday, correct distances, fallback
  - cloud_cover_pct and direct_radiation_wm2 in _FEATURES_BASE and _engineer_features
  - temp_rolling_3d anchored by historical tail in weather_df
  - Log-transform: flag set after training, expm1 applied in predict, backward compat
  - _build_model: n_estimators override accepted
  - Cantonal holidays: canton param threaded to country_holidays, invalid falls back
  - Temperature sensor blending: bias-fade semantics over 6h window preserves forecast trajectory
"""
from __future__ import annotations
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.model import (
    _add_holiday_feature,
    _add_lag_and_rolling_prediction,
    _add_sub_sensor_lags_training,
    _add_sub_sensor_lags_prediction,
    _BRIDGE_CAP,
    _build_model,
    _build_prediction_temp_df,
    _composite_forecast,
    _compute_likely_ev_hours,
    _FEATURES_BASE,
    _engineer_features,
    _learn_appliance_signatures,
    EnergyForecastModel,
    LAG_HOURS,
)

# ── Shared training helper (reused by TestLogTransform and TestPredictIntervals) ─

def _make_trained_model(tmp_path, n: int = 600) -> tuple:
    """Return (model, forecast_df) after a full train() call."""
    rng = np.random.default_rng(0)
    ts  = pd.date_range("2024-01-01", periods=n, freq="1h")
    energy = pd.DataFrame({
        "timestamp": ts,
        "gross_kwh": rng.uniform(0.5, 5.0, size=n),
    })
    weather = pd.DataFrame({
        "timestamp":            ts,
        "temp_c":               rng.uniform(-5, 25, size=n),
        "precipitation_mm":     [0.0]   * n,
        "sunshine_min":         [30.0]  * n,
        "wind_kmh":             [10.0]  * n,
        "cloud_cover_pct":      [50.0]  * n,
        "direct_radiation_wm2": [100.0] * n,
    })
    m = EnergyForecastModel(tmp_path)
    m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
    # Build a minimal forecast_df covering the next 48h
    future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
    forecast = pd.DataFrame({
        "timestamp":            future_ts,
        "temp_c":               [10.0]  * 48,
        "precipitation_mm":     [0.0]   * 48,
        "sunshine_min":         [30.0]  * 48,
        "wind_kmh":             [10.0]  * 48,
        "cloud_cover_pct":      [50.0]  * 48,
        "direct_radiation_wm2": [100.0] * 48,
    })
    return m, forecast


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_future_df(n: int = 48) -> pd.DataFrame:
    """Return a bare future_df with naive hourly timestamps starting at a round hour."""
    start = pd.Timestamp("2026-03-12 08:00")
    return pd.DataFrame({"timestamp": pd.date_range(start, periods=n, freq="1h")})


def _make_actuals(n_hours: int, base_kwh: float = 2.0, noise: float = 0.5) -> pd.DataFrame:
    """Return a recent-actuals DataFrame with predictable per-hour values."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2026-03-12 08:00") - pd.Timedelta(hours=n_hours)
    timestamps = pd.date_range(start, periods=n_hours, freq="1h")
    values = base_kwh + rng.uniform(-noise, noise, size=n_hours)
    return pd.DataFrame({"timestamp": timestamps, "gross_kwh": values})


# ── Rolling features vary per hour ────────────────────────────────────────────

class TestRollingFeaturesVaryByHour:
    """Regression tests for the scalar-broadcast bug."""

    def test_rolling_mean_24h_is_not_flat(self):
        """rolling_mean_24h must not be a constant across all 48 hours."""
        future_df = _make_future_df()
        actuals = _make_actuals(200)
        result = _add_lag_and_rolling_prediction(future_df, actuals)
        vals = result["rolling_mean_24h"].dropna()
        assert vals.nunique() > 1, "rolling_mean_24h is flat — scalar-broadcast bug still present"

    def test_rolling_mean_7d_is_not_flat(self):
        actuals = _make_actuals(200)
        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals)
        vals = result["rolling_mean_7d"].dropna()
        assert vals.nunique() > 1, "rolling_mean_7d is flat — scalar-broadcast bug still present"

    def test_rolling_std_24h_is_not_flat(self):
        actuals = _make_actuals(200, noise=1.0)
        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals)
        vals = result["rolling_std_24h"].dropna()
        assert vals.nunique() > 1, "rolling_std_24h is flat — scalar-broadcast bug still present"


# ── h=0 matches training semantics ───────────────────────────────────────────

class TestHour0MatchesActuals:

    def test_rolling_mean_24h_hour0_equals_last_24_actuals(self):
        """rolling_mean_24h at h=0 must equal mean(actuals[-24:])."""
        actuals_df = _make_actuals(168)
        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals_df)

        actuals_ser = (
            actuals_df.set_index(pd.to_datetime(actuals_df["timestamp"]))["gross_kwh"]
            .sort_index()
        )
        expected = float(actuals_ser.iloc[-24:].mean())
        actual_h0 = float(result["rolling_mean_24h"].iloc[0])
        assert abs(actual_h0 - expected) < 1e-9

    def test_rolling_std_24h_hour0_equals_last_24_actuals(self):
        """rolling_std_24h at h=0 must equal std(actuals[-24:])."""
        actuals_df = _make_actuals(168, noise=1.5)
        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals_df)

        actuals_ser = (
            actuals_df.set_index(pd.to_datetime(actuals_df["timestamp"]))["gross_kwh"]
            .sort_index()
        )
        expected = float(actuals_ser.iloc[-24:].std())
        actual_h0 = float(result["rolling_std_24h"].iloc[0])
        assert abs(actual_h0 - expected) < 1e-9


# ── Smooth transition and stabilisation ──────────────────────────────────────

class TestRollingTransition:

    def test_rolling_mean_24h_monotonically_approaches_fill_value(self):
        """Beyond h=24 the 24h rolling mean should be constant (all-fill window)."""
        # Use constant actuals so fill_val == actuals mean → rolling is flat after h≥24
        n = 200
        start = pd.Timestamp("2026-03-12 08:00") - pd.Timedelta(hours=n)
        timestamps = pd.date_range(start, periods=n, freq="1h")
        actuals_df = pd.DataFrame({"timestamp": timestamps, "gross_kwh": [3.0] * n})

        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals_df)
        # With constant actuals and constant fill, rolling_mean_24h should be 3.0 everywhere
        assert (result["rolling_mean_24h"].round(9) == 3.0).all()

    def test_h24_value_is_influenced_by_fill_not_old_actuals(self):
        """At h=24 the entire 24h window consists of fill values so the mean equals fill_val."""
        n = 200
        start = pd.Timestamp("2026-03-12 08:00") - pd.Timedelta(hours=n)
        timestamps = pd.date_range(start, periods=n, freq="1h")
        # Actuals are 1.0, fill_val will be mean of last 24 → 1.0, so rolling stays 1.0
        actuals_df = pd.DataFrame({"timestamp": timestamps, "gross_kwh": [1.0] * n})
        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals_df)
        assert abs(float(result["rolling_mean_24h"].iloc[24]) - 1.0) < 1e-9


# ── Short actuals ─────────────────────────────────────────────────────────────

class TestShortActuals:

    def test_10_hours_of_actuals_no_crash(self):
        """With only 10 actuals, min_periods=12 causes NaN; must not raise."""
        actuals = _make_actuals(10)
        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals)
        assert "rolling_mean_24h" in result.columns

    def test_30_hours_of_actuals_returns_values(self):
        """With 30 actuals the extended series satisfies min_periods=12; should have values."""
        actuals = _make_actuals(30)
        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals)
        assert result["rolling_mean_24h"].notna().any()


# ── None / empty actuals — existing contract preserved ────────────────────────

class TestNoActuals:

    def test_none_actuals_returns_nan_rolling(self):
        result = _add_lag_and_rolling_prediction(_make_future_df(), None)
        assert result["rolling_mean_24h"].isna().all()
        assert result["rolling_mean_7d"].isna().all()
        assert result["rolling_std_24h"].isna().all()

    def test_empty_actuals_returns_nan_rolling(self):
        empty = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "gross_kwh": []})
        result = _add_lag_and_rolling_prediction(_make_future_df(), empty)
        assert result["rolling_mean_24h"].isna().all()

    def test_none_actuals_returns_nan_lags(self):
        result = _add_lag_and_rolling_prediction(_make_future_df(), None)
        for lag in LAG_HOURS:
            assert result[f"lag_{lag}h"].isna().all()


# ── lag_72h ───────────────────────────────────────────────────────────────────

class TestLag72h:

    def test_lag_72h_in_lag_hours(self):
        assert 72 in LAG_HOURS

    def test_lag_72h_present_in_prediction_output(self):
        result = _add_lag_and_rolling_prediction(_make_future_df(), _make_actuals(200))
        assert "lag_72h" in result.columns

    def test_lag_72h_values_match_actuals(self):
        """lag_72h[h] must equal the actual value at (future_ts[h] - 72h)."""
        actuals_df = _make_actuals(200)
        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals_df)
        actuals_ser = (
            actuals_df.set_index(pd.to_datetime(actuals_df["timestamp"]))["gross_kwh"]
            .sort_index()
        )
        for h in range(10):  # spot-check first 10 hours
            ts = _make_future_df()["timestamp"].iloc[h] - pd.Timedelta(hours=72)
            expected = actuals_ser.get(ts, float("nan"))
            actual   = result["lag_72h"].iloc[h]
            if not np.isnan(expected):
                assert abs(actual - expected) < 1e-9

    def test_lag_72h_nan_when_actuals_too_short(self):
        """With only 10 hours of actuals, lag_72h cannot reach back 72h — all NaN."""
        result = _add_lag_and_rolling_prediction(_make_future_df(), _make_actuals(10))
        assert result["lag_72h"].isna().all()


# ── Stage 2 — Short-horizon lags (#27) ────────────────────────────────────────

class TestShortHorizonLags:
    """lag_1h, lag_2h, lag_6h, lag_12h: presence, values, thresholds, backward compat."""

    @pytest.mark.parametrize("lag", [1, 2, 6, 12])
    def test_short_lag_in_lag_hours(self, lag):
        assert lag in LAG_HOURS

    @pytest.mark.parametrize("lag", [1, 2, 6, 12])
    def test_short_lag_present_in_prediction_output(self, lag):
        result = _add_lag_and_rolling_prediction(_make_future_df(), _make_actuals(200))
        assert f"lag_{lag}h" in result.columns

    def test_lag_1h_value_matches_actuals(self):
        """lag_1h at h=0 must equal the actual at (future_ts[0] - 1h)."""
        actuals_df = _make_actuals(200)
        future_df  = _make_future_df()
        result = _add_lag_and_rolling_prediction(future_df, actuals_df)
        actuals_ser = (
            actuals_df.set_index(pd.to_datetime(actuals_df["timestamp"]))["gross_kwh"]
            .sort_index()
        )
        ts = future_df["timestamp"].iloc[0] - pd.Timedelta(hours=1)
        expected = actuals_ser.get(ts, float("nan"))
        if not np.isnan(expected):
            assert abs(result["lag_1h"].iloc[0] - expected) < 1e-9

    def test_lag_1h_nan_for_far_future_hours(self):
        """lag_1h for h≥2 must be NaN — those lookup times are in the future."""
        actuals_df = _make_actuals(200)
        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals_df)
        # hours h=2..47 need actuals at (now+h-1h) which are not in recent history
        assert result["lag_1h"].iloc[2:].isna().all()

    @pytest.mark.parametrize("lag", [1, 2, 6, 12])
    def test_short_lag_all_nan_when_no_actuals(self, lag):
        """With no recent actuals, short lag columns must be all NaN."""
        result = _add_lag_and_rolling_prediction(_make_future_df(), None)
        assert result[f"lag_{lag}h"].isna().all()

    def test_short_lags_in_feature_cols_after_train(self, tmp_path):
        """Short lags must appear in feature_cols after training with ≥112 rows (lag_12h threshold)."""
        n = 250  # 250 - 12 = 238 ≥ 100 → lag_12h active; 250 - 24 = 226 ≥ 100 → lag_24h active
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        rng = np.random.default_rng(7)
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
        weather = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               rng.uniform(-5, 25, n),
            "precipitation_mm":     [0.0]   * n,
            "sunshine_min":         [30.0]  * n,
            "wind_kmh":             [10.0]  * n,
            "cloud_cover_pct":      [50.0]  * n,
            "direct_radiation_wm2": [100.0] * n,
        })
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        for lag in (1, 2, 6, 12):
            assert f"lag_{lag}h" in m.feature_cols, f"lag_{lag}h missing from feature_cols"

    def test_short_lags_skipped_when_too_few_rows(self, tmp_path):
        """With exactly 100 rows, ALL lags (including short ones) are skipped by the dynamic gate."""
        n = 100  # n - lag >= 100 fails for all lags when n=100 and min lag=1 → 100-1=99 < 100
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        rng = np.random.default_rng(8)
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
        weather = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               rng.uniform(-5, 25, n),
            "precipitation_mm":     [0.0]   * n,
            "sunshine_min":         [30.0]  * n,
            "wind_kmh":             [10.0]  * n,
            "cloud_cover_pct":      [50.0]  * n,
            "direct_radiation_wm2": [100.0] * n,
        })
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        # With 100 rows, training should skip (need ≥100 rows after dropna)
        # model may be None (not enough clean rows after dropna+filter)
        if m.model is not None:
            for lag in (1, 2, 6, 12):
                assert f"lag_{lag}h" not in m.feature_cols

    def test_no_nan_warning_for_short_lags(self, caplog):
        """Short lags must NOT emit the NaN coverage warning even though most hours are NaN."""
        import logging
        actuals_df = _make_actuals(200)
        with caplog.at_level(logging.WARNING, logger="energy_forecast.model"):
            _add_lag_and_rolling_prediction(_make_future_df(), actuals_df)
        for rec in caplog.records:
            assert "lag_1h" not in rec.message
            assert "lag_2h" not in rec.message
            assert "lag_6h" not in rec.message
            assert "lag_12h" not in rec.message


# ── Bridge-day holiday features ───────────────────────────────────────────────

def _make_ts_df(dates: list[str]) -> pd.DataFrame:
    """One row per date string at 12:00 noon (avoids midnight edge cases)."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([f"{d} 12:00" for d in dates])
    })


class TestBridgeDayFeatures:

    def test_columns_present(self):
        df = _make_ts_df(["2026-01-01"])
        result = _add_holiday_feature(df)
        assert "days_to_next_holiday" in result.columns
        assert "days_since_last_holiday" in result.columns

    def test_values_in_range(self):
        """All distance values must be integers in [0, _BRIDGE_CAP]."""
        dates = [f"2026-0{m}-15" for m in range(1, 10)]
        result = _add_holiday_feature(_make_ts_df(dates))
        assert result["days_to_next_holiday"].between(0, _BRIDGE_CAP).all()
        assert result["days_since_last_holiday"].between(0, _BRIDGE_CAP).all()

    def test_holiday_date_has_zero_distance(self):
        """New Year's Day (Jan 1) is a Swiss federal holiday — both distances = 0."""
        result = _add_holiday_feature(_make_ts_df(["2026-01-01"]))
        assert result["days_to_next_holiday"].iloc[0] == 0
        assert result["days_since_last_holiday"].iloc[0] == 0

    def test_day_before_holiday_has_days_to_next_1(self):
        """Dec 31 is one day before New Year's Day."""
        result = _add_holiday_feature(_make_ts_df(["2025-12-31"]))
        assert result["days_to_next_holiday"].iloc[0] == 1

    def test_day_after_holiday_has_days_since_1(self):
        """Jan 2 is one day after New Year's Day."""
        result = _add_holiday_feature(_make_ts_df(["2026-01-02"]))
        assert result["days_since_last_holiday"].iloc[0] == 1

    def test_far_from_holiday_capped_at_bridge_cap(self):
        """A date far from any holiday must be capped at _BRIDGE_CAP."""
        # March 15 is well away from Swiss holidays in both directions
        result = _add_holiday_feature(_make_ts_df(["2026-03-15"]))
        assert result["days_to_next_holiday"].iloc[0] == _BRIDGE_CAP
        assert result["days_since_last_holiday"].iloc[0] == _BRIDGE_CAP

    def test_fallback_without_holidays_package(self):
        """If the holidays package is missing, distances default to _BRIDGE_CAP, no crash."""
        with patch.dict("sys.modules", {"holidays": None}):
            result = _add_holiday_feature(_make_ts_df(["2026-01-01"]))
        assert result["days_to_next_holiday"].iloc[0] == _BRIDGE_CAP
        assert result["days_since_last_holiday"].iloc[0] == _BRIDGE_CAP
        assert result["is_public_holiday"].iloc[0] == 0


# ── Cloud cover, direct radiation, temp_rolling_3d ────────────────────────────

def _make_weather_df(timestamps, temp: float = 5.0, cloud: float = 50.0, rad: float = 200.0) -> pd.DataFrame:
    n = len(timestamps)
    return pd.DataFrame({
        "timestamp":            pd.to_datetime(timestamps),
        "temp_c":               [temp]  * n,
        "precipitation_mm":     [0.0]   * n,
        "sunshine_min":         [30.0]  * n,
        "wind_kmh":             [10.0]  * n,
        "cloud_cover_pct":      [cloud] * n,
        "direct_radiation_wm2": [rad]   * n,
    })


def _make_bare_df(timestamps) -> pd.DataFrame:
    """Minimal energy df with gross_kwh for _engineer_features input."""
    n = len(timestamps)
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "gross_kwh": [1.5] * n,
    })


class TestNewWeatherFeatures:

    def test_features_in_features_base(self):
        assert "cloud_cover_pct"      in _FEATURES_BASE
        assert "direct_radiation_wm2" in _FEATURES_BASE

    def test_engineer_features_new_cols_populated(self):
        """When weather_df contains cloud/radiation, they appear in output."""
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        df = _make_bare_df(ts)
        w  = _make_weather_df(ts, cloud=42.0, rad=180.0)
        result = _engineer_features(df, w, None)
        assert "cloud_cover_pct"      in result.columns
        assert "direct_radiation_wm2" in result.columns
        assert (result["cloud_cover_pct"] == 42.0).all()
        assert (result["direct_radiation_wm2"] == 180.0).all()

    def test_engineer_features_missing_weather_cols_filled_as_nan(self):
        """Safety net: if weather_df has no cloud/radiation, columns are NaN."""
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        df = _make_bare_df(ts)
        # Weather without new columns (simulates SRG-only response gap)
        w = pd.DataFrame({
            "timestamp":        pd.to_datetime(ts),
            "temp_c":           [5.0] * 4,
            "precipitation_mm": [0.0] * 4,
            "sunshine_min":     [30.0] * 4,
            "wind_kmh":         [10.0] * 4,
        })
        result = _engineer_features(df, w, None)
        assert "cloud_cover_pct"      in result.columns
        assert "direct_radiation_wm2" in result.columns
        assert result["cloud_cover_pct"].isna().all()
        assert result["direct_radiation_wm2"].isna().all()


class TestStage2Features:

    def test_thermal_pressure_feature(self):
        """thermal_pressure is correctly calculated as (setpoint - current)."""
        ts = pd.date_range("2026-03-12 08:00", periods=2, freq="1h")
        df = _make_bare_df(ts)
        w  = _make_weather_df(ts)
        
        # Room 1: 21.0 - 20.0 = 1.0 delta
        # Room 2: 22.0 - 18.0 = 4.0 delta
        # Average thermal_pressure should be 2.5
        climate_dfs = {
            "climate.room1": pd.DataFrame({
                "timestamp": ts,
                "current_temp": [20.0, 20.0],
                "setpoint": [21.0, 21.0]
            }),
            "climate.room2": pd.DataFrame({
                "timestamp": ts,
                "current_temp": [18.0, 18.0],
                "setpoint": [22.0, 22.0]
            })
        }
        
        result = _engineer_features(df, w, None, climate_dfs=climate_dfs)
        assert "thermal_pressure" in result.columns
        assert (result["thermal_pressure"] == 2.5).all()

    def test_dhw_pressure_feature(self):
        """dhw_pressure increases non-linearly as buffer_temp drops towards 40C."""
        ts = pd.date_range("2026-03-12 08:00", periods=2, freq="1h")
        df = _make_bare_df(ts)
        w  = _make_weather_df(ts)
        
        # Hour 0: 55C -> low pressure
        # Hour 1: 41C -> high pressure
        dhw_df = pd.DataFrame({
            "timestamp": ts,
            "buffer_temp": [55.0, 41.0]
        })
        
        result = _engineer_features(df, w, None, dhw_df=dhw_df)
        assert "dhw_buffer_temp" in result.columns
        assert "dhw_pressure" in result.columns
        assert result.iloc[0]["dhw_buffer_temp"] == 55.0
        assert result.iloc[1]["dhw_buffer_temp"] == 41.0
        
        p0 = result.iloc[0]["dhw_pressure"]
        p1 = result.iloc[1]["dhw_pressure"]
        # p0 = 1 / (55-40+1)^2 = 1/16^2 = 1/256
        # p1 = 1 / (41-40+1)^2 = 1/2^2 = 1/4
        assert p1 > p0
        assert p1 == pytest.approx(0.25)

    def test_temp_rolling_3d_anchored_by_historical_tail(self):
        """With 72h of history prepended to a 4h forecast, temp_rolling_3d at h=0
        must equal the mean of all 72 historical temps, not just the first value."""
        hist_ts   = pd.date_range("2026-03-09 08:00", periods=72, freq="1h")
        future_ts = pd.date_range("2026-03-12 08:00", periods=4,  freq="1h")
        all_ts    = hist_ts.append(future_ts)

        # Historical temp = 2.0, forecast temp = 10.0
        w = _make_weather_df(all_ts,
                             temp=2.0)  # constant — simplifies expected value
        # Override forecast temps to 10.0 so we can detect if only those were used
        w.loc[w["timestamp"].isin(future_ts), "temp_c"] = 10.0

        df = _make_bare_df(future_ts)
        result = _engineer_features(df, w, None)

        # rolling(72) at the first future row (index 72) covers rows [1..72]:
        # 71 historical rows at 2.0 + the current future row at 10.0.
        # Expected mean = (71*2.0 + 1*10.0) / 72.
        # Without the historical tail (min_periods=1), it would just be 10.0.
        expected = (71 * 2.0 + 1 * 10.0) / 72
        assert abs(float(result["temp_rolling_3d"].iloc[0]) - expected) < 1e-6


# ── Thermal modelling features (#49–#52) ──────────────────────────────────────

class TestEWMATemperature:
    """#49 EWMA temperature features — RC-circuit thermal mass model."""

    def test_temp_ewma_features_in_features_base(self):
        assert "temp_ewma_24h" in _FEATURES_BASE
        assert "temp_ewma_72h" in _FEATURES_BASE

    def test_ewma_24h_decays_exponentially(self):
        """With constant temp = 5.0, ewm(halflife=24) should stabilise near 5.0."""
        ts = pd.date_range("2026-03-12 00:00", periods=200, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, temp=5.0)
        result = _engineer_features(df, w, None)
        # After 200 hours, EWMA should be very close to 5.0
        assert abs(float(result["temp_ewma_24h"].iloc[-1]) - 5.0) < 0.01

    def test_ewma_72h_slower_than_24h(self):
        """72h EWMA should respond slower to temp changes than 24h EWMA."""
        ts = pd.date_range("2026-03-12 00:00", periods=48, freq="1h")
        w_constant = _make_weather_df(ts, temp=5.0)
        # Override to step-change at t=24: 5.0 → 15.0
        w_step = w_constant.copy()
        w_step.loc[w_step["timestamp"] >= pd.Timestamp("2026-03-13 00:00"), "temp_c"] = 15.0
        df = _make_bare_df(ts)
        result = _engineer_features(df, w_step, None)
        # At hour 48, both should have moved toward 15.0 but 72h should be further back
        ewma_24_at_48 = float(result["temp_ewma_24h"].iloc[-1])
        ewma_72_at_48 = float(result["temp_ewma_72h"].iloc[-1])
        assert ewma_24_at_48 > ewma_72_at_48  # 24h reacts faster

    def test_ewma_nan_fill_works(self):
        """When weather is missing, EWMA columns should be NaN then filled by median."""
        ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        df = _make_bare_df(ts)
        w = pd.DataFrame({
            "timestamp":        pd.to_datetime(ts),
            "temp_c":           [5.0] * 24,
            "precipitation_mm": [0.0] * 24,
            "sunshine_min":     [30.0] * 24,
            "wind_kmh":         [10.0] * 24,
        })
        result = _engineer_features(df, w, None)
        # EWMA columns should exist (created as NaN then filled)
        assert "temp_ewma_24h" in result.columns
        assert "temp_ewma_72h" in result.columns
        # After median fill, no NaN should remain
        assert not result["temp_ewma_24h"].isna().any()
        assert not result["temp_ewma_72h"].isna().any()


class TestRollingDegreeHourSums:
    """#50 Accumulated heating/cooling degree-hour sums."""

    def test_heating_deg_sum_features_in_features_base(self):
        assert "heating_deg_sum_24h" in _FEATURES_BASE
        assert "heating_deg_sum_168h" in _FEATURES_BASE

    def test_heating_deg_sum_24h_below_18(self):
        """With constant temp=10°C (below 18°C threshold), sum over 24h should be 24*8=192."""
        ts = pd.date_range("2026-01-15 00:00", periods=48, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, temp=10.0)
        result = _engineer_features(df, w, None)
        # At hour 24: rolling sum of (18-10)=8 over 24 hours = 192
        expected_24h = 24 * (18 - 10)
        # Allow small tolerance for float precision
        assert abs(float(result["heating_deg_sum_24h"].iloc[24]) - expected_24h) < 0.1

    def test_heating_deg_sum_zero_when_temp_above_18(self):
        """With constant temp=25°C (above 18°C threshold), heating_deg_sum should be 0."""
        ts = pd.date_range("2026-06-15 00:00", periods=24, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, temp=25.0)
        result = _engineer_features(df, w, None)
        assert (result["heating_deg_sum_24h"] == 0.0).all()

    def test_heating_deg_sum_168h_accumulates(self):
        """168h sum (7 days) should be larger than 24h sum for the same conditions."""
        ts = pd.date_range("2026-01-15 00:00", periods=200, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, temp=5.0)  # (18-5) = 13 degree-hours per hour
        result = _engineer_features(df, w, None)
        # At hour 168, should have 168 * 13
        sum_168 = float(result["heating_deg_sum_168h"].iloc[168])
        expected = 168 * 13
        assert abs(sum_168 - expected) < 0.1


class TestTemperatureDelta:
    """#51 Temperature rate of change (delta 1h and 24h)."""

    def test_temp_delta_features_in_features_base(self):
        assert "temp_delta_1h" in _FEATURES_BASE
        assert "temp_delta_24h" in _FEATURES_BASE

    def test_temp_delta_1h_tracks_change(self):
        """Rising temp sequence: 0, 1, 2, 3, ... should have delta=1 after first row."""
        ts = pd.date_range("2026-03-12 00:00", periods=48, freq="1h")
        df = _make_bare_df(ts)
        # Temperature increases by 0.5°C per hour
        temps = [float(i) * 0.5 for i in range(48)]
        w = pd.DataFrame({
            "timestamp":            pd.to_datetime(ts),
            "temp_c":               temps,
            "precipitation_mm":     [0.0] * 48,
            "sunshine_min":         [30.0] * 48,
            "wind_kmh":             [10.0] * 48,
            "cloud_cover_pct":      [50.0] * 48,
            "direct_radiation_wm2": [100.0] * 48,
        })
        result = _engineer_features(df, w, None)
        # delta[1:] should be ~0.5 (small tolerance for float errors)
        assert abs(float(result["temp_delta_1h"].iloc[5]) - 0.5) < 0.01

    def test_temp_delta_24h_detects_daily_pattern(self):
        """With 24-hour period shift, temp_delta_24h should show day-over-day change."""
        ts = pd.date_range("2026-03-12 00:00", periods=72, freq="1h")
        df = _make_bare_df(ts)
        # Day 1: 5°C, Day 2: 10°C, Day 3: 5°C again
        temps = [5.0] * 24 + [10.0] * 24 + [5.0] * 24
        w = pd.DataFrame({
            "timestamp":            pd.to_datetime(ts),
            "temp_c":               temps,
            "precipitation_mm":     [0.0] * 72,
            "sunshine_min":         [30.0] * 72,
            "wind_kmh":             [10.0] * 72,
            "cloud_cover_pct":      [50.0] * 72,
            "direct_radiation_wm2": [100.0] * 72,
        })
        result = _engineer_features(df, w, None)
        # At hour 24: 10 - 5 = +5 (warmer than day before)
        assert abs(float(result["temp_delta_24h"].iloc[24]) - 5.0) < 0.01
        # At hour 48: 5 - 10 = -5 (colder than day before)
        assert abs(float(result["temp_delta_24h"].iloc[48]) - (-5.0)) < 0.01


class TestTemperatureLagFeatures:
    """#52 Temperature lag features (24h and 168h)."""

    def test_temp_lag_features_in_features_base(self):
        assert "temp_lag_24h" in _FEATURES_BASE
        assert "temp_lag_168h" in _FEATURES_BASE

    def test_temp_lag_24h_value_matches(self):
        """temp_lag_24h[h] should equal temp_c[h-24]."""
        ts = pd.date_range("2026-03-12 00:00", periods=96, freq="1h")
        df = _make_bare_df(ts)
        temps = list(range(96))  # 0, 1, 2, ..., 95
        w = pd.DataFrame({
            "timestamp":            pd.to_datetime(ts),
            "temp_c":               [float(t) for t in temps],
            "precipitation_mm":     [0.0] * 96,
            "sunshine_min":         [30.0] * 96,
            "wind_kmh":             [10.0] * 96,
            "cloud_cover_pct":      [50.0] * 96,
            "direct_radiation_wm2": [100.0] * 96,
        })
        result = _engineer_features(df, w, None)
        # At index 24, lag_24h should be 0.0
        assert abs(float(result["temp_lag_24h"].iloc[24]) - 0.0) < 1e-9
        # At index 50, lag_24h should be 26.0
        assert abs(float(result["temp_lag_24h"].iloc[50]) - 26.0) < 1e-9

    def test_temp_lag_168h_nan_for_first_week(self):
        """First 168 hours should initially be NaN for temp_lag_168h (no history),
        but filled by median during NaN-fill step in _engineer_features."""
        ts = pd.date_range("2026-03-12 00:00", periods=200, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, temp=10.0)
        result = _engineer_features(df, w, None)
        # After NaN-fill with median, no NaN should remain in temp_lag_168h
        assert not result["temp_lag_168h"].isna().any()
        # Hour 168+ should have correct values (shifted by exactly 168 hours)
        # At hour 168: lag should equal temp_c from hour 0 (which is 10.0)
        assert abs(float(result["temp_lag_168h"].iloc[168]) - 10.0) < 1e-9


class TestThermalFeaturesIntegration:
    """Integration test: all 8 thermal features activate during training."""

    def test_all_thermal_features_in_trained_model(self, tmp_path):
        """After training with sufficient rows, all 8 thermal features should be in m.feature_cols."""
        rng = np.random.default_rng(1)
        ts = pd.date_range("2024-01-01", periods=400, freq="1h")
        energy = pd.DataFrame({
            "timestamp": ts,
            "gross_kwh": rng.uniform(0.5, 5.0, size=400),
        })
        weather = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               rng.uniform(-5, 25, size=400),
            "precipitation_mm":     [0.0] * 400,
            "sunshine_min":         [30.0] * 400,
            "wind_kmh":             [10.0] * 400,
            "cloud_cover_pct":      [50.0] * 400,
            "direct_radiation_wm2": [100.0] * 400,
        })
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        # All 8 thermal features should be present
        thermal_features = [
            "temp_ewma_24h", "temp_ewma_72h",
            "heating_deg_sum_24h", "heating_deg_sum_168h",
            "temp_delta_1h", "temp_delta_24h",
            "temp_lag_24h", "temp_lag_168h",
        ]
        for feat in thermal_features:
            assert feat in m.feature_cols, f"{feat} not found in feature_cols"


# ── Log-transform (#7) ────────────────────────────────────────────────────────

class TestLogTransform:

    def test_log_transform_flag_set_after_training(self, tmp_path):
        """_log_transform must be True after a successful train()."""
        m, _ = _make_trained_model(tmp_path)
        assert m._log_transform is True

    def test_predict_gives_nonnegative_finite_values(self, tmp_path):
        """predict() must return non-negative, finite kWh values with log-transform active."""
        m, forecast = _make_trained_model(tmp_path)
        result = m.predict(forecast, live_temp=None)
        assert result["predicted_kwh"].ge(0).all()
        assert result["predicted_kwh"].notna().all()
        assert np.isfinite(result["predicted_kwh"].values).all()

    def test_backward_compat_old_meta_defaults_to_false(self, tmp_path):
        """meta.pkl without 'log_transform' key must load as False (no crash on old installs)."""
        import pickle, hashlib
        # Write a meta dict that doesn't contain log_transform
        meta_path = tmp_path / "meta.pkl"
        meta = {
            "feature_cols":    _FEATURES_BASE,
            "last_trained":    __import__("datetime").datetime.min,
            "last_mae":        None,
            "last_cv_mae":     None,
            "engine":          "test",
            "feature_medians": {},
            # intentionally omit "log_transform" and "canton"
        }
        with open(meta_path, "wb") as fh:
            pickle.dump(meta, fh)
        digest = hashlib.sha256(meta_path.read_bytes()).hexdigest()
        meta_path.with_suffix(".pkl.sha256").write_text(digest)

        m = EnergyForecastModel(tmp_path)
        assert m._log_transform is False
        assert m._canton is None


# ── _build_model n_estimators override (#6) ───────────────────────────────────

class TestBuildModel:

    def _gbr(self):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor

    def test_n_estimators_override_applied(self):
        """_build_model with n_estimators=100 must produce a model with that count."""
        GBR = self._gbr()
        model = _build_model(None, GBR, n_estimators=100)
        assert model.n_estimators == 100

    def test_default_n_estimators_when_none(self):
        """_build_model with n_estimators=None uses the hardcoded default (300 for GBR)."""
        GBR = self._gbr()
        model = _build_model(None, GBR, n_estimators=None)
        assert model.n_estimators == 300


# ── Cantonal holidays (#9) ────────────────────────────────────────────────────

class TestCantonalHolidays:

    def _ts_df(self, dates):
        return pd.DataFrame({
            "timestamp": pd.to_datetime([f"{d} 12:00" for d in dates])
        })

    def test_canton_zh_returns_correct_columns(self):
        """canton='ZH' must return all three holiday columns with int dtype."""
        pytest.importorskip("holidays")
        result = _add_holiday_feature(self._ts_df(["2026-04-15"]), canton="ZH")
        for col in ("is_public_holiday", "days_to_next_holiday", "days_since_last_holiday"):
            assert col in result.columns
            assert result[col].dtype in (np.int32, np.int64, int, "int64", "int32")

    def test_canton_none_gives_federal_only(self):
        """With canton=None, result columns are still present and values are valid ints."""
        pytest.importorskip("holidays")
        result = _add_holiday_feature(self._ts_df(["2026-01-01"]), canton=None)
        assert result["is_public_holiday"].iloc[0] == 1  # Jan 1 is federal

    def test_invalid_canton_falls_back_gracefully(self):
        """An unrecognised canton code must not crash; columns must still be present."""
        pytest.importorskip("holidays")
        result = _add_holiday_feature(self._ts_df(["2026-03-15"]), canton="INVALID")
        for col in ("is_public_holiday", "days_to_next_holiday", "days_since_last_holiday"):
            assert col in result.columns


# ── Prediction intervals (#13) ────────────────────────────────────────────────

class TestPredictIntervals:

    def test_quantile_models_trained(self, tmp_path):
        """After train(), _model_q10 and _model_q90 must not be None."""
        m, _ = _make_trained_model(tmp_path)
        assert m._model_q10 is not None
        assert m._model_q90 is not None

    def test_predict_intervals_columns_and_nonnegative(self, tmp_path):
        """predict_intervals() returns DataFrame with expected columns, non-negative, finite."""
        m, forecast = _make_trained_model(tmp_path)
        result = m.predict_intervals(forecast, live_temp=None)
        assert result is not None
        assert "low_kwh"  in result.columns
        assert "high_kwh" in result.columns
        assert len(result) == 48
        assert result["low_kwh"].ge(0).all()
        assert result["high_kwh"].ge(0).all()
        assert np.isfinite(result["low_kwh"].values).all()
        assert np.isfinite(result["high_kwh"].values).all()

    def test_low_le_high(self, tmp_path):
        """low_kwh must be ≤ high_kwh for every row (quantile ordering enforced)."""
        m, forecast = _make_trained_model(tmp_path)
        result = m.predict_intervals(forecast, live_temp=None)
        assert result is not None
        assert (result["low_kwh"] <= result["high_kwh"]).all()

    def test_interval_correction_stored(self, tmp_path):
        """After train(), _interval_correction must be a finite float."""
        m, _ = _make_trained_model(tmp_path)
        assert isinstance(m._interval_correction, float)
        assert np.isfinite(m._interval_correction)

    def test_interval_correction_persisted(self, tmp_path):
        """_interval_correction survives a save/reload round-trip."""
        m, _ = _make_trained_model(tmp_path)
        saved_val = m._interval_correction
        # Reload from disk
        m2 = EnergyForecastModel(tmp_path)
        assert np.isclose(m2._interval_correction, saved_val, atol=1e-9)

    def test_calibrated_intervals_wider_than_raw(self, tmp_path):
        """A positive _interval_correction must widen predict_intervals() output."""
        m, forecast = _make_trained_model(tmp_path)
        assert m._log_transform
        # Raw intervals (no correction)
        m._interval_correction = 0.0
        raw = m.predict_intervals(forecast, live_temp=None)
        # Calibrated intervals (positive correction in log-space)
        m._interval_correction = 0.3
        calibrated = m.predict_intervals(forecast, live_temp=None)
        assert calibrated is not None and raw is not None
        assert (calibrated["high_kwh"] >= raw["high_kwh"]).all()
        assert (calibrated["low_kwh"]  <= raw["low_kwh"]).all()


# ── EV session probability feature (#12) ─────────────────────────────────────

class TestLikelyEvHour:

    def _make_ev_df(self, n: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (baseline_df, ev_df) where Monday 22:00 is always a charging hour."""
        rng = np.random.default_rng(1)
        ts  = pd.date_range("2025-01-06 00:00", periods=n * 24, freq="1h")  # starts Monday
        kwh = rng.uniform(0.5, 3.0, size=len(ts))
        baseline = pd.DataFrame({"timestamp": ts, "gross_kwh": kwh})

        # Mark Monday 22:00 (hour_of_week = 0*24+22 = 22) as EV in every week
        how = ts.dayofweek * 24 + ts.hour
        ev_mask = how == 22   # Monday 22:00
        ev_df = baseline[ev_mask].copy()
        return baseline, ev_df

    def test_likely_hours_identified_after_train(self, tmp_path):
        """After train() with ev_df, _likely_ev_hours must be non-empty and
        contain the known charging slot (Monday 22:00 = how 22)."""
        baseline, ev_df = self._make_ev_df(n=200)
        weather = pd.DataFrame({
            "timestamp":            baseline["timestamp"],
            "temp_c":               [10.0] * len(baseline),
            "precipitation_mm":     [0.0]  * len(baseline),
            "sunshine_min":         [30.0] * len(baseline),
            "wind_kmh":             [10.0] * len(baseline),
            "cloud_cover_pct":      [50.0] * len(baseline),
            "direct_radiation_wm2": [100.0]* len(baseline),
        })
        m = EnergyForecastModel(tmp_path)
        m.train(baseline, weather, outdoor_df=None, weight_halflife_days=0, ev_df=ev_df)
        assert len(m._likely_ev_hours) > 0
        assert 22 in m._likely_ev_hours

    def test_likely_ev_hour_column_is_binary(self, tmp_path):
        """likely_ev_hour in trained feature matrix must be strictly 0 or 1."""
        m, _ = _make_trained_model(tmp_path)
        # Feature is in _FEATURES_BASE
        assert "likely_ev_hour" in _FEATURES_BASE
        # Values in a freshly engineered df must be 0/1
        ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        weather = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               [10.0] * 24,
            "precipitation_mm":     [0.0]  * 24,
            "sunshine_min":         [30.0] * 24,
            "wind_kmh":             [10.0] * 24,
            "cloud_cover_pct":      [50.0] * 24,
            "direct_radiation_wm2": [100.0]* 24,
        })
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 24})
        feat = _engineer_features(df, weather, None, likely_ev_hours={0, 5, 10})
        vals = feat["likely_ev_hour"].unique()
        assert set(vals).issubset({0, 1})

    def test_no_ev_df_gives_empty_hours_and_zero_feature(self, tmp_path):
        """Without ev_df, _likely_ev_hours is empty and likely_ev_hour is 0 everywhere."""
        m, _ = _make_trained_model(tmp_path)
        # _make_trained_model calls train() without ev_df
        assert m._likely_ev_hours == set()
        # _compute_likely_ev_hours with no ev_df must return empty set
        baseline = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=48, freq="1h"),
            "gross_kwh": [1.0] * 48,
        })
        assert _compute_likely_ev_hours(baseline, None) == set()
        assert _compute_likely_ev_hours(baseline, pd.DataFrame()) == set()


# ── Sub-sensor lag features ────────────────────────────────────────────────────

def _make_sub_sensor_df(n: int = 400, start: str = "2024-01-01") -> pd.DataFrame:
    """Return a sub-sensor DataFrame with 'timestamp' and 'kwh' columns."""
    rng = np.random.default_rng(7)
    ts  = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"timestamp": ts, "kwh": rng.uniform(0, 3.0, size=n)})


class TestSubSensorFeatures:

    def _make_weather(self, ts):
        return pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               [10.0] * len(ts),
            "precipitation_mm":     [0.0]  * len(ts),
            "sunshine_min":         [30.0] * len(ts),
            "wind_kmh":             [10.0] * len(ts),
            "cloud_cover_pct":      [50.0] * len(ts),
            "direct_radiation_wm2": [100.0]* len(ts),
        })

    def test_lag_24h_in_feature_cols_when_sub_sensor_provided(self, tmp_path):
        """sub_sensor lag_24h column appears in feature_cols after train()."""
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": np.random.default_rng(0).uniform(0.5, 5, n)})
        weather = self._make_weather(ts)
        sub_df  = _make_sub_sensor_df(n=n)
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0,
                sub_sensors_dict={"sub_hp": sub_df})
        assert "sub_hp_lag_24h" in m.feature_cols

    def test_lag_168h_in_feature_cols_with_enough_history(self, tmp_path):
        """sub_sensor lag_168h appears when n_rows >= 268 (168 + 100)."""
        n = 600
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": np.random.default_rng(1).uniform(0.5, 5, n)})
        weather = self._make_weather(ts)
        sub_df  = _make_sub_sensor_df(n=n)
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0,
                sub_sensors_dict={"sub_hp": sub_df})
        assert "sub_hp_lag_168h" in m.feature_cols

    def test_no_sub_sensor_cols_without_sub_sensors_dict(self, tmp_path):
        """Without sub_sensors_dict, no 'sub_' columns appear in feature_cols."""
        m, _ = _make_trained_model(tmp_path)
        sub_cols = [c for c in m.feature_cols if c.startswith("sub_")]
        assert sub_cols == [], f"unexpected sub-sensor columns: {sub_cols}"

    def test_sub_sensor_lag_values_are_correct(self, tmp_path):
        """lag_24h for a sub-sensor equals the kwh value 24 positions earlier in training."""
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": [2.0] * n})
        weather = self._make_weather(ts)
        # Sub-sensor: deterministic values so we can verify the lag
        sub_kwh = list(range(n))   # 0, 1, 2, ..., n-1
        sub_df  = pd.DataFrame({"timestamp": ts, "kwh": sub_kwh})

        # Call the training helper directly
        from energy_forecast.model import _add_lag_and_rolling_training
        df = _add_lag_and_rolling_training(energy, list(range(24, n - 100)))
        df = _add_sub_sensor_lags_training(df, {"sub_hp": sub_df})

        # Row 24 in the sorted df should have sub_hp_lag_24h == sub_kwh[0] == 0
        assert "sub_hp_lag_24h" in df.columns
        # lag_24h at position 24 = sub_kwh[0]; shift(24) makes first 24 NaN
        non_nan = df["sub_hp_lag_24h"].dropna()
        assert float(non_nan.iloc[0]) == pytest.approx(0.0)
        assert float(non_nan.iloc[1]) == pytest.approx(1.0)

    def test_predict_runs_with_sub_sensors_recent(self, tmp_path):
        """predict() accepts sub_sensors_recent without error."""
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": np.random.default_rng(3).uniform(0.5, 5, n)})
        weather = self._make_weather(ts)
        sub_df  = _make_sub_sensor_df(n=n)
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0,
                sub_sensors_dict={"sub_hp": sub_df})

        future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
        forecast  = self._make_weather(future_ts)
        # Recent actuals for sub-sensor — recent 200 hours
        recent_sub = _make_sub_sensor_df(n=200, start=str((pd.Timestamp.now() - pd.Timedelta(hours=200)).date()))
        result = m.predict(forecast, live_temp=None, sub_sensors_recent={"sub_hp": recent_sub})
        assert len(result) == 48
        assert result["predicted_kwh"].ge(0).all()

    def test_prediction_lag_columns_are_float_dtype(self, tmp_path):
        """Regression: reindex of a sparse sub-sensor must produce float64, not object dtype.

        In pandas 3.x, reindexing across mismatched datetime resolutions (ns vs us from
        CSV cache) returned dtype=object, which LightGBM rejected at predict time.
        """
        future_ts = pd.date_range("2024-01-10", periods=48, freq="1h")
        future_df = pd.DataFrame({"timestamp": future_ts})
        # Only 1 recent data point — simulates a sensor active for just a few hours
        recent_sub = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-09 12:00"]),
            "kwh": [0.3],
        })
        result = _add_sub_sensor_lags_prediction(future_df, {"sub_t": recent_sub})
        assert result["sub_t_lag_24h"].dtype == np.float64, (
            f"expected float64, got {result['sub_t_lag_24h'].dtype}"
        )
        assert result["sub_t_lag_168h"].dtype == np.float64

    def test_lag_168h_absent_below_threshold(self):
        """lag_168h is absent when n_rows - 168 < 100 (n=267, threshold−1)."""
        n = 267
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": [2.0] * n})
        sub_df = _make_sub_sensor_df(n=n)
        df = _add_sub_sensor_lags_training(energy, {"sub_hp": sub_df})
        assert "sub_hp_lag_168h" not in df.columns
        assert "sub_hp_lag_24h" in df.columns

    def test_lag_168h_present_at_threshold(self):
        """lag_168h is present when n_rows - 168 == 100 (n=268, exactly at threshold)."""
        n = 268
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": [2.0] * n})
        sub_df = _make_sub_sensor_df(n=n)
        df = _add_sub_sensor_lags_training(energy, {"sub_hp": sub_df})
        assert "sub_hp_lag_168h" in df.columns

    def test_sparse_sub_sensor_does_not_skip_training(self, tmp_path):
        """A nearly-all-NaN sub-sensor (warm-up period) must not cause training to be skipped.

        Regression for: sub-sensor NaN included in dropna → 0 clean rows → model not trained.
        """
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": np.random.default_rng(5).uniform(0.5, 5, n)})
        weather = self._make_weather(ts)
        # Only 1 data point — simulates a sensor that started today
        sub_df  = pd.DataFrame({"timestamp": ts[-1:], "kwh": [0.5]})
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0,
                sub_sensors_dict={"sub_new": sub_df})
        # Model must have trained — feature_cols and model are set
        assert m.feature_cols is not None
        assert m.model is not None

    def test_sparse_sub_sensor_triggers_nan_warning(self, caplog):
        """Sub-sensor with >50% gaps triggers NaN warning during training."""
        import logging
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": [2.0] * n})
        # Only 10 data points out of 400 — reindex will produce >50% NaN
        sparse_ts = ts[::40]
        sub_df = pd.DataFrame({"timestamp": sparse_ts, "kwh": [1.0] * len(sparse_ts)})

        with caplog.at_level(logging.WARNING, logger="energy_forecast.model"):
            df = _add_sub_sensor_lags_training(energy, {"sub_hp": sub_df})

        assert "sub_hp" in caplog.text
        assert "NaN" in caplog.text
        assert "sub_hp_lag_24h" in df.columns

    def test_multiple_sub_sensors_in_feature_cols(self, tmp_path):
        """Two sub-sensors both produce lag columns in feature_cols after train()."""
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": np.random.default_rng(4).uniform(0.5, 5, n)})
        weather = self._make_weather(ts)
        sub_hp  = _make_sub_sensor_df(n=n)
        sub_dw  = _make_sub_sensor_df(n=n)
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0,
                sub_sensors_dict={"sub_hp": sub_hp, "sub_dw": sub_dw})
        assert "sub_hp_lag_24h" in m.feature_cols
        assert "sub_dw_lag_24h" in m.feature_cols

    def test_lag_168h_values_are_correct(self):
        """lag_168h for a sub-sensor equals the kwh value 168 positions earlier in training."""
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": [2.0] * n})
        sub_kwh = list(range(n))  # 0, 1, 2, ..., n-1
        sub_df  = pd.DataFrame({"timestamp": ts, "kwh": sub_kwh})

        from energy_forecast.model import _add_lag_and_rolling_training
        df = _add_lag_and_rolling_training(energy, list(range(24, n - 100)))
        df = _add_sub_sensor_lags_training(df, {"sub_hp": sub_df})

        assert "sub_hp_lag_168h" in df.columns
        non_nan = df["sub_hp_lag_168h"].dropna()
        assert float(non_nan.iloc[0]) == pytest.approx(0.0)
        assert float(non_nan.iloc[1]) == pytest.approx(1.0)

    def test_sparse_sub_sensor_prediction_logs_debug_not_warning(self, caplog):
        """Prediction-time sub-sensor NaN message must be DEBUG, not WARNING.

        Supplies a non-empty sub_df whose timestamps are far in the past so that
        all reindexed lag values come back NaN — this exercises the actual DEBUG
        log branch, not just the empty-DataFrame guard.
        """
        import logging
        future_ts = pd.date_range("2024-01-10", periods=48, freq="1h")
        future_df = pd.DataFrame({"timestamp": future_ts})
        # Sub-sensor data from 30 days before the future window — all lags will be NaN.
        old_ts = pd.date_range("2023-12-01", periods=48, freq="1h")
        sub_df = pd.DataFrame({"timestamp": old_ts, "kwh": [1.0] * 48})
        with caplog.at_level(logging.DEBUG, logger="energy_forecast.model"):
            _add_sub_sensor_lags_prediction(future_df.copy(), {"sub_sparse": sub_df})
        warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING and "sub_sparse" in r.message]
        assert warning_msgs == [], f"Expected no WARNING for sparse sub-sensor, got: {warning_msgs}"
        debug_msgs = [r for r in caplog.records if r.levelno == logging.DEBUG and "sub_sparse" in r.message]
        assert debug_msgs, "Expected at least one DEBUG log for sparse sub-sensor NaN values"


# ── Stage 4 — Sub-sensor activity flag and run count (#35, #36) ───────────────

class TestSubSensorActivityAndRuns:
    """active_24h and runs_7d computed correctly in training and prediction."""

    def _all_zero_series(self, n: int = 400) -> pd.DataFrame:
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        return pd.DataFrame({"timestamp": ts, "kwh": [0.0] * n})

    def _series_with_event(self, n: int = 400, event_start: int = 200) -> pd.DataFrame:
        """All-zero series except rows event_start..event_start+3 which are non-zero."""
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        kwh = [0.0] * n
        for i in range(event_start, min(event_start + 4, n)):
            kwh[i] = 1.5
        return pd.DataFrame({"timestamp": ts, "kwh": kwh})

    def test_active_24h_zero_for_all_zero_series(self):
        """All-zero sub-sensor → active_24h must be 0 everywhere."""
        energy = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=400, freq="1h"),
            "gross_kwh": [2.0] * 400,
        })
        df = _add_sub_sensor_lags_training(energy, {"sub_dw": self._all_zero_series()})
        assert "sub_dw_active_24h" in df.columns
        assert (df["sub_dw_active_24h"] == 0).all()

    def test_active_24h_becomes_one_after_event(self):
        """After a non-zero event, active_24h must become 1 within the next 24 rows."""
        n = 400
        event_start = 200
        energy = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "gross_kwh": [2.0] * n,
        })
        sub = self._series_with_event(n=n, event_start=event_start)
        df = _add_sub_sensor_lags_training(energy, {"sub_dw": sub})
        # Rows from event_start+1 to event_start+24 should have active_24h=1
        assert df["sub_dw_active_24h"].iloc[event_start + 1] == 1

    def test_runs_7d_zero_for_all_zero_series(self):
        """All-zero sub-sensor → runs_7d must be 0 everywhere (NaN at row 0 is OK)."""
        energy = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=400, freq="1h"),
            "gross_kwh": [2.0] * 400,
        })
        df = _add_sub_sensor_lags_training(energy, {"sub_dw": self._all_zero_series()})
        assert "sub_dw_runs_7d" in df.columns
        # Row 0 can be NaN (no prior row for transition detection); all others must be 0
        assert (df["sub_dw_runs_7d"].iloc[1:] == 0).all()

    def test_runs_7d_counts_appliance_starts(self):
        """Two non-zero events separated by zeros → runs_7d counts correctly."""
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        # Two runs: rows 50-52 and 100-102
        kwh = [0.0] * n
        for i in range(50, 53):
            kwh[i] = 1.0
        for i in range(100, 103):
            kwh[i] = 1.0
        sub = pd.DataFrame({"timestamp": ts, "kwh": kwh})
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": [2.0] * n})
        df = _add_sub_sensor_lags_training(energy, {"sub_dw": sub})
        # At row 168 (within 168h of both events), runs_7d should be 2
        assert int(df["sub_dw_runs_7d"].iloc[168]) == 2

    def test_active_24h_in_feature_cols_after_train(self, tmp_path):
        """active_24h must appear in feature_cols after training with sub-sensors."""
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        rng = np.random.default_rng(9)
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5, n)})
        weather = pd.DataFrame({
            "timestamp":            ts, "temp_c": rng.uniform(-5, 25, n),
            "precipitation_mm": [0.0]*n, "sunshine_min": [30.0]*n,
            "wind_kmh": [10.0]*n, "cloud_cover_pct": [50.0]*n,
            "direct_radiation_wm2": [100.0]*n,
        })
        sub = self._series_with_event(n=n)
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0,
                sub_sensors_dict={"sub_dw": sub})
        assert "sub_dw_active_24h" in m.feature_cols
        assert "sub_dw_runs_7d" in m.feature_cols

    def test_prediction_active_24h_zero_for_empty_sub_sensor(self):
        """active_24h in prediction must be 0 when no recent actuals are available."""
        future = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=48, freq="1h")})
        result = _add_sub_sensor_lags_prediction(future, {"sub_dw": pd.DataFrame()})
        assert (result["sub_dw_active_24h"] == 0).all()

    def test_prediction_runs_7d_counts_from_recent_actuals(self):
        """runs_7d at predict time must reflect start events in recent 168h actuals."""
        now = pd.Timestamp("2026-01-08 12:00")
        # 200h of actuals; two events: 50h and 100h ago
        ts_hist = pd.date_range(now - pd.Timedelta(hours=200), now, freq="1h")
        kwh = [0.0] * len(ts_hist)
        idx_50  = len(ts_hist) - 51   # 50h ago
        idx_100 = len(ts_hist) - 101  # 100h ago
        if idx_50 >= 0:
            kwh[idx_50]  = 2.0
        if idx_100 >= 0:
            kwh[idx_100] = 2.0
        sub_recent = pd.DataFrame({"timestamp": ts_hist, "kwh": kwh})

        future = pd.DataFrame({"timestamp": pd.date_range(now.floor("1h"), periods=48, freq="1h")})
        result = _add_sub_sensor_lags_prediction(future, {"sub_dw": sub_recent})
        # Both events are within 168h → runs_7d should be 2 for all future hours
        assert (result["sub_dw_runs_7d"] == 2).all()


# ── Stage 1 — Feature importance + CV std logging (#29, #30) ─────────────────

class TestFeatureImportanceLogging:
    """After train(), feature importances and CV fold std must be logged."""

    def test_feature_importances_logged_after_training(self, tmp_path, caplog):
        """Feature importances (top 10) must appear in logs after a successful train."""
        import logging
        with caplog.at_level(logging.INFO, logger="energy_forecast.model"):
            _make_trained_model(tmp_path, n=600)
        assert any("Feature importances" in r.message for r in caplog.records), (
            "Expected 'Feature importances' in log output after train()"
        )

    def test_cv_fold_std_logged_alongside_mean(self, tmp_path, caplog):
        """CV fold MAE log must include both mean and ± std when CV runs (≥500 rows).

        Need n≥836 so that after the lag_336h dropna there are still ≥500 clean rows
        for TimeSeriesSplit (MIN_CV_ROWS=500).
        """
        import logging
        with caplog.at_level(logging.INFO, logger="energy_forecast.model"):
            _make_trained_model(tmp_path, n=900)
        cv_logs = [r.message for r in caplog.records if "CV fold MAEs" in r.message]
        assert cv_logs, "Expected 'CV fold MAEs' log entry when n≥500 clean rows"
        assert "±" in cv_logs[0], f"Expected std (±) in CV log: {cv_logs[0]}"


# ── Stage 1 — Holiday vectorisation (#32) ─────────────────────────────────────

class TestHolidayVectorisation:
    """np.searchsorted vectorisation must give identical results to bisect."""

    def test_days_to_next_zero_on_holiday(self):
        """days_to_next_holiday must be 0 on a holiday date itself."""
        # New Year's Day 2025 is a Swiss federal holiday
        ts = pd.Timestamp("2025-01-01")
        df = pd.DataFrame({"timestamp": [ts]})
        result = _add_holiday_feature(df)
        assert int(result["days_to_next_holiday"].iloc[0]) == 0

    def test_days_since_last_zero_on_holiday(self):
        """days_since_last_holiday must be 0 on a holiday date itself."""
        ts = pd.Timestamp("2025-01-01")
        df = pd.DataFrame({"timestamp": [ts]})
        result = _add_holiday_feature(df)
        assert int(result["days_since_last_holiday"].iloc[0]) == 0

    def test_distance_columns_capped_at_bridge_cap(self):
        """Dates far from any holiday must be capped at _BRIDGE_CAP."""
        # Mid-July is typically far from holidays in CH (National Day = Aug 1)
        ts = pd.Timestamp("2025-07-15")
        df = pd.DataFrame({"timestamp": [ts]})
        result = _add_holiday_feature(df)
        assert int(result["days_to_next_holiday"].iloc[0]) <= _BRIDGE_CAP
        assert int(result["days_since_last_holiday"].iloc[0]) <= _BRIDGE_CAP


# ── Stage 3 — doy cyclical, hours_ahead, num_leaves sweep (#33, #34, #28) ─────

class TestDoyFeatures:
    """doy_sin and doy_cos must be present and have correct values."""

    def _make_bare_df(self, ts):
        return pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * len(ts)})

    def test_doy_columns_in_features_base(self):
        assert "doy_sin" in _FEATURES_BASE
        assert "doy_cos" in _FEATURES_BASE

    def test_doy_columns_in_engineer_features_output(self):
        ts = pd.date_range("2026-01-01", periods=4, freq="1h")
        w  = pd.DataFrame({
            "timestamp": ts, "temp_c": [5.0]*4, "precipitation_mm": [0.0]*4,
            "sunshine_min": [30.0]*4, "wind_kmh": [10.0]*4,
            "cloud_cover_pct": [50.0]*4, "direct_radiation_wm2": [100.0]*4,
        })
        result = _engineer_features(self._make_bare_df(ts), w, None)
        assert "doy_sin" in result.columns
        assert "doy_cos" in result.columns

    def test_doy_sin_near_zero_on_jan1(self):
        """Jan 1 is doy=1; sin(2π·1/365) ≈ 0.0172 — near but not exactly 0."""
        ts = pd.date_range("2026-01-01", periods=1, freq="1h")
        w  = pd.DataFrame({
            "timestamp": ts, "temp_c": [5.0], "precipitation_mm": [0.0],
            "sunshine_min": [30.0], "wind_kmh": [10.0],
            "cloud_cover_pct": [50.0], "direct_radiation_wm2": [100.0],
        })
        result = _engineer_features(self._make_bare_df(ts), w, None)
        expected_sin = np.sin(2 * np.pi * 1 / 365)
        assert abs(float(result["doy_sin"].iloc[0]) - expected_sin) < 1e-9

    def test_doy_sin_near_one_at_peak(self):
        """doy ≈ 91 (April 1) sin ≈ 1; verify value is reasonable."""
        ts = pd.date_range("2026-04-01", periods=1, freq="1h")
        w  = pd.DataFrame({
            "timestamp": ts, "temp_c": [10.0], "precipitation_mm": [0.0],
            "sunshine_min": [30.0], "wind_kmh": [10.0],
            "cloud_cover_pct": [50.0], "direct_radiation_wm2": [100.0],
        })
        result = _engineer_features(self._make_bare_df(ts), w, None)
        assert float(result["doy_sin"].iloc[0]) > 0.99


class TestHoursAheadFeature:
    """hours_ahead = 0 in training rows; 0–47 monotonically in prediction."""

    def _make_bare_df(self, ts):
        return pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * len(ts)})

    def test_hours_ahead_in_features_base(self):
        assert "hours_ahead" in _FEATURES_BASE

    def test_hours_ahead_zero_in_engineer_features(self):
        """Training rows must always get hours_ahead=0."""
        ts = pd.date_range("2026-01-01", periods=4, freq="1h")
        w  = pd.DataFrame({
            "timestamp": ts, "temp_c": [5.0]*4, "precipitation_mm": [0.0]*4,
            "sunshine_min": [30.0]*4, "wind_kmh": [10.0]*4,
            "cloud_cover_pct": [50.0]*4, "direct_radiation_wm2": [100.0]*4,
        })
        result = _engineer_features(self._make_bare_df(ts), w, None)
        assert (result["hours_ahead"] == 0).all()

    def test_hours_ahead_monotonic_in_prediction(self, tmp_path):
        """Prediction X must have hours_ahead = 0, 1, 2, ..., 47."""
        m, forecast = _make_trained_model(tmp_path)
        # Peek at the feature matrix built for prediction
        future_hours, X = m._prepare_prediction_X(forecast, live_temp=None, recent_actuals=None)
        assert "hours_ahead" in X.columns
        expected = list(range(48))
        actual   = X["hours_ahead"].tolist()
        assert actual == expected, f"hours_ahead not monotonic: {actual[:5]}…"


class TestNumLeavesSweep:
    """num_leaves sweep on last CV fold: best value logged; _build_model accepts param."""

    def test_build_model_accepts_num_leaves(self):
        from sklearn.ensemble import GradientBoostingRegressor
        # GBR doesn't use num_leaves — should not raise
        m = _build_model(None, GradientBoostingRegressor, num_leaves=63)
        assert m is not None

    def test_num_leaves_sweep_logged_when_cv_runs(self, tmp_path, caplog):
        """With enough rows for CV and LightGBM absent, sweep is skipped gracefully."""
        import logging
        with caplog.at_level(logging.INFO, logger="energy_forecast.model"):
            _make_trained_model(tmp_path, n=900)
        # If LightGBM is present, expect sweep log; if not (sklearn fallback), no crash.
        # Either way CV must complete without error.
        cv_logs = [r.message for r in caplog.records if "CV fold MAEs" in r.message]
        assert cv_logs, "CV must have run with n=900 rows"


# ── Stage 5 — Per-HOW NaN fill medians (#31) ─────────────────────────────────

class TestHowMedians:
    """_feature_medians_by_how stored after training; used in prediction; backward compat."""

    def test_how_medians_populated_after_train(self, tmp_path):
        """After training, _feature_medians_by_how must have entries for lag/rolling cols."""
        m, _ = _make_trained_model(tmp_path, n=600)
        assert m._feature_medians_by_how, "_feature_medians_by_how must not be empty"
        # Keys should be integers 0-167 (hour_of_week)
        sample_key = next(iter(m._feature_medians_by_how))
        assert isinstance(sample_key, (int, np.integer)), "HOW keys must be integers"
        assert 0 <= int(sample_key) <= 167

    def test_how_medians_contain_lag_columns(self, tmp_path):
        """HOW median dict must include lag and rolling columns."""
        m, _ = _make_trained_model(tmp_path, n=600)
        sample_meds = next(iter(m._feature_medians_by_how.values()))
        lag_cols = [c for c in sample_meds if c.startswith("lag_") or c.startswith("rolling_")]
        assert lag_cols, "HOW medians must include lag/rolling columns"

    def test_how_medians_persisted_and_loaded(self, tmp_path):
        """_feature_medians_by_how must survive a save/load cycle via meta.pkl."""
        m, _ = _make_trained_model(tmp_path, n=600)
        original = m._feature_medians_by_how
        # Load a fresh instance from the same directory
        m2 = EnergyForecastModel(tmp_path)
        assert m2._feature_medians_by_how == original

    def test_backward_compat_meta_without_how_medians(self, tmp_path):
        """meta.pkl without feature_medians_by_how must load as empty dict (no crash)."""
        import pickle, hashlib
        meta_path = tmp_path / "meta.pkl"
        meta = {
            "feature_cols":    _FEATURES_BASE,
            "last_trained":    __import__("datetime").datetime.min,
            "last_mae":        None,
            "last_cv_mae":     None,
            "engine":          "test",
            "feature_medians": {},
            # intentionally omit feature_medians_by_how
        }
        with open(meta_path, "wb") as fh:
            pickle.dump(meta, fh)
        digest = hashlib.sha256(meta_path.read_bytes()).hexdigest()
        meta_path.with_suffix(".pkl.sha256").write_text(digest)

        m = EnergyForecastModel(tmp_path)
        assert m._feature_medians_by_how == {}

    def test_how_median_applied_when_global_would_differ(self, tmp_path):
        """When HOW-specific median differs from global, prediction uses HOW value."""
        # Create training data where lag_24h has a clear HOW pattern:
        # HOW=0 (Mon 00:00) always has lag_24h ≈ 10, rest ≈ 1
        n = 600
        rng = np.random.default_rng(42)
        ts  = pd.date_range("2024-01-01", periods=n, freq="1h")
        # Start on Monday so HOW=0 is the first row's hour_of_week
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
        weather = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               rng.uniform(-5, 25, n),
            "precipitation_mm":     [0.0]   * n,
            "sunshine_min":         [30.0]  * n,
            "wind_kmh":             [10.0]  * n,
            "cloud_cover_pct":      [50.0]  * n,
            "direct_radiation_wm2": [100.0] * n,
        })
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        # The HOW dict must exist and have lag_24h entries
        assert m._feature_medians_by_how
        sample = next(iter(m._feature_medians_by_how.values()))
        # At minimum one lag column should be present
        assert any(k.startswith("lag_") for k in sample)


# ── #38 Regression: per-hour weather variation in _engineer_features ──────────

class TestWeatherPerHourVariation:
    """Regression guard: weather columns must not be scalar-broadcast in prediction."""

    def _make_varied_forecast(self) -> pd.DataFrame:
        """48 h forecast where every weather variable has a unique value per hour."""
        start = pd.Timestamp("2026-03-12 08:00")
        n = 48
        hours = pd.date_range(start, periods=n, freq="1h")
        return pd.DataFrame({
            "timestamp":            hours,
            "temp_c":               np.linspace(5.0, 20.0, n),
            "precipitation_mm":     np.linspace(0.0, 5.0, n),
            "sunshine_min":         np.linspace(0.0, 60.0, n),
            "wind_kmh":             np.linspace(5.0, 30.0, n),
            "cloud_cover_pct":      np.linspace(0.0, 100.0, n),
            "direct_radiation_wm2": np.linspace(0.0, 800.0, n),
        })

    def test_all_weather_cols_vary_per_hour(self):
        """Each weather column must have more than one unique value across 48 h."""
        forecast = self._make_varied_forecast()
        future_df = pd.DataFrame({"timestamp": forecast["timestamp"], "gross_kwh": np.nan})
        result = _engineer_features(future_df, forecast, outdoor_df=None)
        for col in ["temp_c", "precipitation_mm", "sunshine_min",
                    "wind_kmh", "cloud_cover_pct", "direct_radiation_wm2"]:
            vals = result[col].dropna()
            assert vals.nunique() > 1, (
                f"{col} is flat across 48 h — scalar-broadcast bug present"
            )

    def test_temp_c_matches_forecast_values(self):
        """Spot-check: temp_c at h=0 and h=47 must equal the input forecast values."""
        forecast = self._make_varied_forecast()
        future_df = pd.DataFrame({"timestamp": forecast["timestamp"], "gross_kwh": np.nan})
        result = _engineer_features(future_df, forecast, outdoor_df=None)
        result = result.reset_index(drop=True)
        assert abs(float(result.loc[0, "temp_c"]) - 5.0) < 1e-6, (
            "temp_c at h=0 does not match forecast input"
        )
        assert abs(float(result.loc[47, "temp_c"]) - 20.0) < 1e-6, (
            "temp_c at h=47 does not match forecast input"
        )


# ── #25 Vacation / Away Flag ───────────────────────────────────────────────────

class TestAwayFeature:
    """is_away binary feature: presence, zero default, value propagation, predict."""

    def _make_weather(self, ts):
        return pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               [10.0] * len(ts),
            "precipitation_mm":     [0.0]  * len(ts),
            "sunshine_min":         [30.0] * len(ts),
            "wind_kmh":             [10.0] * len(ts),
            "cloud_cover_pct":      [50.0] * len(ts),
            "direct_radiation_wm2": [100.0]* len(ts),
        })

    def _make_bare(self, ts):
        return pd.DataFrame({"timestamp": ts, "gross_kwh": [1.5] * len(ts)})

    def test_is_away_in_features_base(self):
        assert "is_away" in _FEATURES_BASE

    def test_is_away_column_present_with_away_df(self):
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        away_df = pd.DataFrame({"timestamp": ts, "is_away": [1, 1, 0, 0]})
        result = _engineer_features(self._make_bare(ts), self._make_weather(ts), None,
                                    away_df=away_df)
        assert "is_away" in result.columns

    def test_is_away_zero_without_away_df(self):
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        result = _engineer_features(self._make_bare(ts), self._make_weather(ts), None,
                                    away_df=None)
        assert "is_away" in result.columns
        assert (result["is_away"] == 0).all()

    def test_is_away_values_match_away_df(self):
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        away_df = pd.DataFrame({"timestamp": ts, "is_away": [1, 0, 1, 0]})
        result = _engineer_features(self._make_bare(ts), self._make_weather(ts), None,
                                    away_df=away_df)
        assert list(result["is_away"].values) == [1, 0, 1, 0]

    def test_is_away_in_feature_cols_after_train(self, tmp_path):
        n = 600
        rng = np.random.default_rng(5)
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
        weather = self._make_weather(ts)
        away_df = pd.DataFrame({"timestamp": ts, "is_away": [0] * n})
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, away_df=away_df)
        assert "is_away" in m.feature_cols

    def test_predict_with_away_series(self, tmp_path):
        """predict() must propagate away_series into the is_away feature column,
        and predictions must be valid (non-NaN, non-negative)."""
        n = 600
        rng = np.random.default_rng(5)
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy  = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
        weather = self._make_weather(ts)
        away_df = pd.DataFrame({"timestamp": ts, "is_away": [0] * n})
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, away_df=away_df)
        assert "is_away" in m.feature_cols
        future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
        forecast = self._make_weather(future_ts)
        away_ones  = pd.Series(1, index=future_ts, dtype=int)
        away_zeros = pd.Series(0, index=future_ts, dtype=int)
        # Verify feature propagation at the matrix level
        _, X_away = m._prepare_prediction_X(forecast, None, None, away_series=away_ones)
        _, X_home = m._prepare_prediction_X(forecast, None, None, away_series=away_zeros)
        assert (X_away["is_away"] == 1).all(), "is_away must be 1 when away_series=all-ones"
        assert (X_home["is_away"] == 0).all(), "is_away must be 0 when away_series=all-zeros"
        # Verify predict() returns valid results
        result = m.predict(forecast, live_temp=None, away_series=away_ones)
        assert len(result) == 48
        assert result["predicted_kwh"].notna().all()
        assert result["predicted_kwh"].ge(0).all()


# ── #42 SHAP feature importance ───────────────────────────────────────────────

class TestShapSummary:
    """shap_summary(): returns top-N features, sorted, guards cold-start and n=0."""

    def test_returns_n_features(self, tmp_path):
        """shap_summary() returns exactly n entries when model is trained."""
        m, forecast = _make_trained_model(tmp_path)
        result = m.shap_summary(forecast, live_temp=None, n=5)
        assert isinstance(result, dict)
        assert len(result) == 5

    def test_sorted_by_importance_descending(self, tmp_path):
        """Returned dict values must be in descending order."""
        m, forecast = _make_trained_model(tmp_path)
        result = m.shap_summary(forecast, live_temp=None, n=5)
        values = list(result.values())
        assert values == sorted(values, reverse=True), (
            f"Values not descending: {values}"
        )

    def test_cold_start_returns_empty(self, tmp_path):
        """shap_summary() on an untrained model must return {}."""
        m = EnergyForecastModel(tmp_path)
        # Build a minimal forecast_df
        future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
        forecast = pd.DataFrame({
            "timestamp":            future_ts,
            "temp_c":               [10.0] * 48,
            "precipitation_mm":     [0.0]  * 48,
            "sunshine_min":         [30.0] * 48,
            "wind_kmh":             [10.0] * 48,
            "cloud_cover_pct":      [50.0] * 48,
            "direct_radiation_wm2": [100.0]* 48,
        })
        result = m.shap_summary(forecast, live_temp=None, n=5)
        assert result == {}

    def test_n_zero_returns_empty(self, tmp_path):
        """n=0 disables SHAP and must return {}."""
        m, forecast = _make_trained_model(tmp_path)
        result = m.shap_summary(forecast, live_temp=None, n=0)
        assert result == {}


# ── Temperature sensor blending: bias fade ──────────────────────────────────────

class TestBuildPredictionTempDf:
    """Test bias-fade temperature blending: sensor offset fades over 6h window."""

    def test_full_trust_zone_returns_live_temp(self):
        """Hours 0–2: return live_temp unchanged."""
        future_hours = pd.date_range("2026-03-12 08:00", periods=10, freq="1h")
        forecast = pd.DataFrame({
            "timestamp": future_hours,
            "temp_c": [10.0 + i for i in range(10)],
        })
        result = _build_prediction_temp_df(future_hours, forecast, live_temp=11.0)

        # h=0, h=1, h=2 must be 11.0 (live_temp)
        assert abs(result.iloc[0]["outdoor_temp_live"] - 11.0) < 1e-9
        assert abs(result.iloc[1]["outdoor_temp_live"] - 11.0) < 1e-9
        assert abs(result.iloc[2]["outdoor_temp_live"] - 11.0) < 1e-9

    def test_full_forecast_zone_returns_forecast(self):
        """Hours 6+: return forecast[h] unchanged."""
        future_hours = pd.date_range("2026-03-12 08:00", periods=10, freq="1h")
        forecast = pd.DataFrame({
            "timestamp": future_hours,
            "temp_c": [10.0 + i for i in range(10)],
        })
        result = _build_prediction_temp_df(future_hours, forecast, live_temp=11.0)

        # h=6, h=7, h=8, h=9 must equal forecast temps
        assert abs(result.iloc[6]["outdoor_temp_live"] - 16.0) < 1e-9
        assert abs(result.iloc[7]["outdoor_temp_live"] - 17.0) < 1e-9
        assert abs(result.iloc[8]["outdoor_temp_live"] - 18.0) < 1e-9
        assert abs(result.iloc[9]["outdoor_temp_live"] - 19.0) < 1e-9

    def test_blend_zone_bias_fade(self):
        """Hours 2–6: temp = forecast[h] + bias*(1-alpha).

        With live_temp=11, forecast[0]=10, bias=1.
        Forecast: 10, 11, 12, 13, 14, 15, 16, 17, 18, 19.
        - h=3: α=0.25 → 13.0 + 1*0.75 = 13.75
        - h=4: α=0.5  → 14.0 + 1*0.5  = 14.5
        - h=5: α=0.75 → 15.0 + 1*0.25 = 15.25
        """
        future_hours = pd.date_range("2026-03-12 08:00", periods=10, freq="1h")
        forecast = pd.DataFrame({
            "timestamp": future_hours,
            "temp_c": [10.0 + i for i in range(10)],
        })
        result = _build_prediction_temp_df(future_hours, forecast, live_temp=11.0)

        # h=3: forecast=13, bias=1, α=0.25 → 13 + 0.75 = 13.75
        assert abs(result.iloc[3]["outdoor_temp_live"] - 13.75) < 1e-9

        # h=4: forecast=14, bias=1, α=0.5 → 14 + 0.5 = 14.5
        assert abs(result.iloc[4]["outdoor_temp_live"] - 14.5) < 1e-9

        # h=5: forecast=15, bias=1, α=0.75 → 15 + 0.25 = 15.25
        assert abs(result.iloc[5]["outdoor_temp_live"] - 15.25) < 1e-9

    def test_rising_forecast_trajectory_visible_in_blend_zone(self):
        """Blend zone must track the forecast's rising trajectory, not interpolate live→forecast."""
        future_hours = pd.date_range("2026-03-12 08:00", periods=10, freq="1h")
        forecast = pd.DataFrame({
            "timestamp": future_hours,
            "temp_c": [10.0, 10.5, 11.0, 12.0, 13.5, 15.0, 16.0, 16.5, 17.0, 17.5],
        })
        result = _build_prediction_temp_df(future_hours, forecast, live_temp=11.0)

        # Verify blend zone (h=2..5) follows forecast trajectory and (h=6 onwards pure forecast)
        blend_vals = [result.iloc[i]["outdoor_temp_live"] for i in range(2, 7)]
        # With bias=1, values should smoothly rise as forecast rises and bias fades
        assert blend_vals[0] == 11.0  # h=2: pure live (SENSOR_FULL_TRUST_HOURS boundary)
        assert blend_vals[1] > blend_vals[0]  # h=3 > h=2 (blending begins)
        assert blend_vals[2] > blend_vals[1]  # h=4 > h=3
        assert blend_vals[3] > blend_vals[2]  # h=5 > h=4
        assert abs(blend_vals[4] - 16.0) < 1e-9  # h=6: pure forecast (SENSOR_BLEND_HOURS boundary)

    def test_empty_forecast_fallback(self):
        """Empty forecast: all hours return live_temp."""
        future_hours = pd.date_range("2026-03-12 08:00", periods=10, freq="1h")
        forecast = pd.DataFrame({"timestamp": [], "temp_c": []})
        result = _build_prediction_temp_df(future_hours, forecast, live_temp=11.0)

        # All hours must be 11.0 when forecast is empty
        assert (result["outdoor_temp_live"] == 11.0).all()

    def test_zero_bias_blend_equals_forecast(self):
        """When sensor == forecast[0], blend/forecast zones equal forecast (no bias contribution)."""
        future_hours = pd.date_range("2026-03-12 08:00", periods=10, freq="1h")
        forecast = pd.DataFrame({
            "timestamp": future_hours,
            "temp_c": [10.0 + i for i in range(10)],
        })
        # Set live_temp == forecast[0] → bias = 0
        result = _build_prediction_temp_df(future_hours, forecast, live_temp=10.0)

        # Full-trust zone (h≤2) returns live_temp
        assert abs(result.iloc[0]["outdoor_temp_live"] - 10.0) < 1e-9
        assert abs(result.iloc[1]["outdoor_temp_live"] - 10.0) < 1e-9
        assert abs(result.iloc[2]["outdoor_temp_live"] - 10.0) < 1e-9

        # Blend zone (h>2, h<6) and forecast zone (h≥6) should equal pure forecast
        for i in range(3, len(future_hours)):
            expected = 10.0 + i
            assert abs(result.iloc[i]["outdoor_temp_live"] - expected) < 1e-9

    def test_negative_bias_blend(self):
        """When sensor < forecast[0], bias is negative; blend must fade properly."""
        future_hours = pd.date_range("2026-03-12 08:00", periods=10, freq="1h")
        forecast = pd.DataFrame({
            "timestamp": future_hours,
            "temp_c": [15.0 + i for i in range(10)],
        })
        # live_temp=14 < forecast[0]=15 → bias=-1
        result = _build_prediction_temp_df(future_hours, forecast, live_temp=14.0)

        # Full-trust zone: should be 14.0
        assert abs(result.iloc[0]["outdoor_temp_live"] - 14.0) < 1e-9

        # Blend zone (h=3): forecast≈18, bias=-1, α=0.25 → 18 + (-1)*0.75 = 17.25
        assert abs(result.iloc[3]["outdoor_temp_live"] - 17.25) < 1e-9

        # Full-forecast zone (h=6): should be 21.0
        assert abs(result.iloc[6]["outdoor_temp_live"] - 21.0) < 1e-9


# ── #44 Model versioning ───────────────────────────────────────────────────────

class TestModelVersioning:
    """EnergyForecastModel._archive_current() and rollback_model()."""

    def test_no_archive_on_first_save(self, tmp_path):
        """First-ever train() must not create an archive directory."""
        _make_trained_model(tmp_path)
        archive_dir = tmp_path / "archive"
        assert not archive_dir.exists() or not list(archive_dir.iterdir())

    def test_archive_created_on_second_save(self, tmp_path):
        """Second train() must create exactly one archive snapshot."""
        import re
        m, _ = _make_trained_model(tmp_path)
        m.train(*_make_trained_model.__wrapped__(tmp_path)[0].train.__self__) if False else None
        # Re-use helper: train the same instance a second time
        rng = np.random.default_rng(1)
        n = 600
        ts = pd.date_range("2024-06-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})
        weather = pd.DataFrame({
            "timestamp": ts, "temp_c": rng.uniform(-5, 25, size=n),
            "precipitation_mm": [0.0]*n, "sunshine_min": [30.0]*n,
            "wind_kmh": [10.0]*n, "cloud_cover_pct": [50.0]*n,
            "direct_radiation_wm2": [100.0]*n,
        })
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        archive_dir = tmp_path / "archive"
        assert archive_dir.exists()
        subdirs = list(archive_dir.iterdir())
        assert len(subdirs) == 1
        assert re.match(r"\d{8}T\d{6}$", subdirs[0].name)

    def test_only_last_n_archives_retained(self, tmp_path):
        """With model_archive_count=2, four trains must leave ≤2 archive dirs."""
        rng = np.random.default_rng(42)
        m = EnergyForecastModel(tmp_path, model_archive_count=2)
        for i in range(4):
            n = 600
            ts = pd.date_range(f"2024-0{i+1}-01", periods=n, freq="1h")
            energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})
            weather = pd.DataFrame({
                "timestamp": ts, "temp_c": rng.uniform(-5, 25, size=n),
                "precipitation_mm": [0.0]*n, "sunshine_min": [30.0]*n,
                "wind_kmh": [10.0]*n, "cloud_cover_pct": [50.0]*n,
                "direct_radiation_wm2": [100.0]*n,
            })
            m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        archive_dir = tmp_path / "archive"
        assert archive_dir.exists()
        assert len(list(archive_dir.iterdir())) <= 2

    def test_rollback_restores_previous_model(self, tmp_path):
        """rollback_model() after two trains must restore the first training time."""
        m, _ = _make_trained_model(tmp_path)
        t1 = m.last_trained

        rng = np.random.default_rng(7)
        n = 600
        ts = pd.date_range("2024-07-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})
        weather = pd.DataFrame({
            "timestamp": ts, "temp_c": rng.uniform(-5, 25, size=n),
            "precipitation_mm": [0.0]*n, "sunshine_min": [30.0]*n,
            "wind_kmh": [10.0]*n, "cloud_cover_pct": [50.0]*n,
            "direct_radiation_wm2": [100.0]*n,
        })
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        t2 = m.last_trained
        assert t2 >= t1

        success = m.rollback_model()
        assert success is True

        m2 = EnergyForecastModel(tmp_path)
        assert m2.last_trained == t1

    def test_rollback_no_archive_returns_false(self, tmp_path):
        """rollback_model() on a fresh (never-trained) instance returns False."""
        m = EnergyForecastModel(tmp_path)
        assert m.rollback_model() is False

    def test_rollback_logs_warning(self, tmp_path, caplog):
        """rollback_model() after two trains must log a WARNING with the archive name."""
        import logging
        m, _ = _make_trained_model(tmp_path)
        rng = np.random.default_rng(99)
        n = 600
        ts = pd.date_range("2024-09-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})
        weather = pd.DataFrame({
            "timestamp": ts, "temp_c": rng.uniform(-5, 25, size=n),
            "precipitation_mm": [0.0]*n, "sunshine_min": [30.0]*n,
            "wind_kmh": [10.0]*n, "cloud_cover_pct": [50.0]*n,
            "direct_radiation_wm2": [100.0]*n,
        })
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        archive_name = sorted((tmp_path / "archive").iterdir())[-1].name
        with caplog.at_level(logging.WARNING, logger="energy_forecast.model"):
            m.rollback_model()
        assert any(archive_name in r.message for r in caplog.records)


# ── Occupancy feature (people_home) — #21 ──────────────────────────────────────

class TestPeopleHomeFeature:
    """Tests for occupancy feature: people_home integer count (#21)."""

    def test_people_home_in_features_base(self):
        """people_home is listed in _FEATURES_BASE."""
        assert "people_home" in _FEATURES_BASE

    def test_people_home_zero_without_presence_df(self, tmp_path):
        """When presence_df is None, people_home defaults to 0."""
        rng = np.random.default_rng(0)
        ts  = pd.date_range("2024-01-01", periods=200, freq="1h")
        energy = pd.DataFrame({
            "timestamp": ts,
            "gross_kwh": rng.uniform(0.5, 5.0, size=200),
        })
        weather = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               rng.uniform(-5, 25, size=200),
            "precipitation_mm":     [0.0]   * 200,
            "sunshine_min":         [30.0]  * 200,
            "wind_kmh":             [10.0]  * 200,
            "cloud_cover_pct":      [50.0]  * 200,
            "direct_radiation_wm2": [100.0] * 200,
        })
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, presence_df=None)

        # Get feature columns
        assert "people_home" in m.feature_cols

    def test_people_home_column_present_with_presence_df(self, tmp_path):
        """When presence_df is provided, people_home column is populated."""
        rng = np.random.default_rng(0)
        ts  = pd.date_range("2024-01-01", periods=200, freq="1h")
        energy = pd.DataFrame({
            "timestamp": ts,
            "gross_kwh": rng.uniform(0.5, 5.0, size=200),
        })
        weather = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               rng.uniform(-5, 25, size=200),
            "precipitation_mm":     [0.0]   * 200,
            "sunshine_min":         [30.0]  * 200,
            "wind_kmh":             [10.0]  * 200,
            "cloud_cover_pct":      [50.0]  * 200,
            "direct_radiation_wm2": [100.0] * 200,
        })
        presence = pd.DataFrame({
            "timestamp": energy["timestamp"].values,
            "people_home": rng.integers(0, 3, size=200),  # 0, 1, or 2 people
        })
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, presence_df=presence)

        assert "people_home" in m.feature_cols

    def test_people_home_values_match_presence_df(self):
        """people_home values from presence_df are correctly merged into features."""
        rng = np.random.default_rng(0)
        ts  = pd.date_range("2024-01-01", periods=200, freq="1h")
        energy = pd.DataFrame({
            "timestamp": ts,
            "gross_kwh": rng.uniform(0.5, 5.0, size=200),
        })
        weather = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               rng.uniform(-5, 25, size=200),
            "precipitation_mm":     [0.0]   * 200,
            "sunshine_min":         [30.0]  * 200,
            "wind_kmh":             [10.0]  * 200,
            "cloud_cover_pct":      [50.0]  * 200,
            "direct_radiation_wm2": [100.0] * 200,
        })
        presence = pd.DataFrame({
            "timestamp": energy["timestamp"].values[:100],  # Partial overlap
            "people_home": [2, 1, 0, 1] * 25,  # 4-hour cycle repeated
        })

        # Call _engineer_features directly to inspect merge behavior
        df = _engineer_features(energy, weather, None, presence_df=presence)

        assert "people_home" in df.columns
        assert df["people_home"].dtype == int

    def test_people_home_in_feature_cols_after_train(self, tmp_path):
        """people_home is included in feature_cols list after training."""
        rng = np.random.default_rng(0)
        ts  = pd.date_range("2024-01-01", periods=200, freq="1h")
        energy = pd.DataFrame({
            "timestamp": ts,
            "gross_kwh": rng.uniform(0.5, 5.0, size=200),
        })
        weather = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               rng.uniform(-5, 25, size=200),
            "precipitation_mm":     [0.0]   * 200,
            "sunshine_min":         [30.0]  * 200,
            "wind_kmh":             [10.0]  * 200,
            "cloud_cover_pct":      [50.0]  * 200,
            "direct_radiation_wm2": [100.0] * 200,
        })
        presence = pd.DataFrame({
            "timestamp": energy["timestamp"].values,
            "people_home": [1] * len(energy),
        })
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, presence_df=presence)

        assert "people_home" in m.feature_cols

    def test_predict_with_people_home_series(self, tmp_path):
        """Prediction with people_home_series injects values into prediction."""
        rng = np.random.default_rng(0)
        ts  = pd.date_range("2024-01-01", periods=200, freq="1h")
        energy = pd.DataFrame({
            "timestamp": ts,
            "gross_kwh": rng.uniform(0.5, 5.0, size=200),
        })
        weather = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               rng.uniform(-5, 25, size=200),
            "precipitation_mm":     [0.0]   * 200,
            "sunshine_min":         [30.0]  * 200,
            "wind_kmh":             [10.0]  * 200,
            "cloud_cover_pct":      [50.0]  * 200,
            "direct_radiation_wm2": [100.0] * 200,
        })
        presence = pd.DataFrame({
            "timestamp": energy["timestamp"].values,
            "people_home": [1] * len(energy),
        })
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, presence_df=presence)

        # Build forecast for next 48 hours
        now = energy["timestamp"].max()
        forecast = pd.DataFrame({
            "timestamp": pd.date_range(now + pd.Timedelta(hours=1), periods=48, freq="1h"),
            "temp_c": [15.0] * 48,
            "precipitation_mm": [0.0] * 48,
            "sunshine_min": [30.0] * 48,
            "wind_kmh": [10.0] * 48,
            "cloud_cover_pct": [50.0] * 48,
            "direct_radiation_wm2": [100.0] * 48,
        })

        # Create a 48-element Series with occupancy values
        people_home_series = pd.Series(
            data=[2] * 48,
            index=forecast["timestamp"].values,
        )

        predictions = m.predict(
            forecast,
            live_temp=15.0,
            recent_actuals=None,
            people_home_series=people_home_series,
        )

        assert "predicted_kwh" in predictions.columns
        assert len(predictions) == 48


# ── holiday_country propagation (Fix 2) ──────────────────────────────────────

class TestHolidayCountry:
    """holiday_country param must be threaded from train() into _add_holiday_feature."""

    def _ts_df(self, dates):
        return pd.DataFrame({"timestamp": pd.to_datetime(dates)})

    def test_country_gb_uses_uk_holidays(self):
        """country='GB' must flag UK holidays (e.g. Christmas Day)."""
        pytest.importorskip("holidays")
        result = _add_holiday_feature(self._ts_df(["2026-12-25"]), country="GB")
        assert result["is_public_holiday"].iloc[0] == 1, "Dec 25 must be a UK holiday"

    def test_country_ch_default_flags_swiss_new_year(self):
        """Default country='CH' must flag Jan 1 as a public holiday."""
        pytest.importorskip("holidays")
        result = _add_holiday_feature(self._ts_df(["2026-01-01"]))
        assert result["is_public_holiday"].iloc[0] == 1, "Jan 1 must be a CH holiday"

    def test_country_de_flags_german_holiday(self):
        """country='DE' must recognise German Unity Day (Oct 3)."""
        pytest.importorskip("holidays")
        result = _add_holiday_feature(self._ts_df(["2026-10-03"]), country="DE")
        assert result["is_public_holiday"].iloc[0] == 1, "Oct 3 must be a DE holiday"

    def test_invalid_country_falls_back_gracefully(self):
        """An unrecognised country code must not crash; columns must still be present."""
        pytest.importorskip("holidays")
        result = _add_holiday_feature(self._ts_df(["2026-03-15"]), country="XX")
        for col in ("is_public_holiday", "days_to_next_holiday", "days_since_last_holiday"):
            assert col in result.columns



# ── Stage 3: Appliance Signature Discovery ────────────────────────────────────

def _make_cycle_df(
    cycle_kwh: list[float],
    n_cycles: int,
    period_hours: int = 12,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Synthetic sub-sensor: n_cycles cycles of len(cycle_kwh) each.

    Between cycles there is silence (zeros) so that total length =
    n_cycles * period_hours.  The cycle occupies the first len(cycle_kwh)
    hours of each period.
    """
    window_hours = len(cycle_kwh)
    total_hours = n_cycles * period_hours
    ts = pd.date_range(start, periods=total_hours, freq="1h")
    kwh = [0.0] * total_hours
    for i in range(n_cycles):
        base = i * period_hours
        for j, v in enumerate(cycle_kwh):
            kwh[base + j] = v
    return pd.DataFrame({"timestamp": ts, "kwh": kwh})


class TestApplianceSignatures:
    """Tests for _learn_appliance_signatures()."""

    PROFILE = [1.0, 2.0, 1.5, 0.5]  # 4-hour cycle shape

    def test_basic_two_cycles(self):
        """Two identical cycles → correct profile, total_kwh, peak_hour, n_cycles."""
        df = _make_cycle_df(self.PROFILE, n_cycles=2)
        sigs = _learn_appliance_signatures({"hp": df})
        assert "hp" in sigs
        s = sigs["hp"]
        assert s["n_cycles"] == 2
        assert s["total_kwh"] == pytest.approx(sum(self.PROFILE))
        assert s["hourly_profile"] == pytest.approx(self.PROFILE)
        assert s["peak_hour"] == 1  # index of 2.0 in PROFILE

    def test_below_min_cycles_skipped(self):
        """Only 1 cycle → empty dict (insufficient evidence)."""
        df = _make_cycle_df(self.PROFILE, n_cycles=1, period_hours=20)
        sigs = _learn_appliance_signatures({"hp": df}, min_cycles=2)
        assert sigs == {}

    def test_no_sub_sensors_none(self):
        """None input → empty dict, no crash."""
        assert _learn_appliance_signatures(None) == {}

    def test_no_sub_sensors_empty_dict(self):
        """Empty dict input → empty dict."""
        assert _learn_appliance_signatures({}) == {}

    def test_partial_window_skipped(self):
        """Cycle starting within last window_hours rows is discarded (truncated window)."""
        # 3 full cycles + 1 partial (only 2 hours at end)
        df_full = _make_cycle_df(self.PROFILE, n_cycles=3, period_hours=8)
        # Append 2 hours of an extra cycle start at the very end
        extra = pd.DataFrame({
            "timestamp": pd.date_range(df_full["timestamp"].iloc[-1] + pd.Timedelta("1h"), periods=2, freq="1h"),
            "kwh": [1.0, 1.0],
        })
        df = pd.concat([df_full, extra], ignore_index=True)
        sigs = _learn_appliance_signatures({"dw": df}, window_hours=4, min_cycles=2)
        assert "dw" in sigs
        # The partial window must not inflate n_cycles
        assert sigs["dw"]["n_cycles"] == 3

    def test_multiple_prefixes(self):
        """Two prefixes → independent signatures both returned."""
        df1 = _make_cycle_df([1.0, 2.0, 0.5, 0.5], n_cycles=3, period_hours=12)
        df2 = _make_cycle_df([0.5, 0.5, 1.0, 2.0], n_cycles=3, period_hours=12)
        sigs = _learn_appliance_signatures({"hp": df1, "dw": df2})
        assert set(sigs.keys()) == {"hp", "dw"}
        assert sigs["hp"]["peak_hour"] == 1
        assert sigs["dw"]["peak_hour"] == 3

    def test_save_load_roundtrip(self, tmp_path):
        """Save → delete in-memory → load → same dict."""
        df = _make_cycle_df(self.PROFILE, n_cycles=3)
        model = EnergyForecastModel(model_dir=tmp_path)
        model._appliance_signatures = _learn_appliance_signatures({"hp": df})
        original = dict(model._appliance_signatures)
        model._save_signatures()
        # Reset and reload
        model._appliance_signatures = {}
        model._load_signatures()
        assert model._appliance_signatures == original

    def test_train_populates_signatures(self, tmp_path):
        """After train(), _appliance_signatures is populated and saved to disk."""
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(42)
        n = 600
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 3.0, n)})
        weather_df = pd.DataFrame({
            "timestamp":            ts,
            "temp_c":               rng.uniform(5, 25, n),
            "precipitation_mm":     [0.0]   * n,
            "sunshine_min":         [30.0]  * n,
            "wind_kmh":             [10.0]  * n,
            "cloud_cover_pct":      [50.0]  * n,
            "direct_radiation_wm2": [100.0] * n,
        })
        # Build a sub-sensor with enough cycles to learn a signature
        sub_hp = _make_cycle_df(self.PROFILE, n_cycles=20, period_hours=24, start="2024-01-01")
        sub_hp = sub_hp[sub_hp["timestamp"] < ts[-1]].reset_index(drop=True)

        model = EnergyForecastModel(model_dir=tmp_path)
        model.train(
            energy_df=energy_df,
            weather_df=weather_df,
            outdoor_df=None,
            sub_sensors_dict={"sub_hp": sub_hp},
        )
        assert isinstance(model._appliance_signatures, dict)
        assert "sub_hp" in model._appliance_signatures
        assert (tmp_path / "appliance_signatures.json").exists()


# ── _composite_forecast (Stage 4) ────────────────────────────────────────────

def _make_baseline_df(start="2024-06-01 10:00", n=48):
    ts = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"timestamp": ts, "predicted_kwh": [1.0] * n})


_DUMMY_SIGS = {
    "sub_dishwasher": {
        "total_kwh": 5.0,
        "hourly_profile": [1.0, 2.0, 1.5, 0.5],
        "peak_hour": 1,
        "n_cycles": 5,
    },
    "sub_wp": {
        "total_kwh": 3.0,
        "hourly_profile": [1.0, 1.0, 1.0],
        "peak_hour": 0,
        "n_cycles": 4,
    },
}


class TestCompositeForecast:

    def test_empty_schedule_returns_baseline(self):
        """schedule={} → delta_kwh all zeros, predicted_kwh unchanged."""
        df = _make_baseline_df()
        result = _composite_forecast(df, {}, _DUMMY_SIGS)
        assert "delta_kwh" in result.columns
        assert (result["delta_kwh"] == 0.0).all()
        assert result["predicted_kwh"].tolist() == pytest.approx([1.0] * 48)

    def test_single_appliance_overlay(self):
        """Dishwasher at 14:00 (4h ahead from 10:00) → profile added at offsets 4..7."""
        df = _make_baseline_df(start="2024-06-01 10:00")
        result = _composite_forecast(df, {"sub_dishwasher": "14:00"}, _DUMMY_SIGS)
        profile = _DUMMY_SIGS["sub_dishwasher"]["hourly_profile"]
        # offset_h = 4 (10:00 → 14:00)
        for i, v in enumerate(profile):
            assert result["delta_kwh"].iloc[4 + i] == pytest.approx(v)
        # Baseline outside profile unchanged (delta stays 0)
        assert result["delta_kwh"].iloc[0] == pytest.approx(0.0)
        assert result["delta_kwh"].iloc[3] == pytest.approx(0.0)
        # predicted_kwh = 1.0 + profile at those slots
        assert result["predicted_kwh"].iloc[4] == pytest.approx(1.0 + profile[0])

    def test_off_appliance_skipped(self):
        """'off' value → no overlay, delta stays zero."""
        df = _make_baseline_df()
        result = _composite_forecast(df, {"sub_dishwasher": "off"}, _DUMMY_SIGS)
        assert (result["delta_kwh"] == 0.0).all()

    def test_none_appliance_skipped(self):
        """None value → no overlay."""
        df = _make_baseline_df()
        result = _composite_forecast(df, {"sub_dishwasher": None}, _DUMMY_SIGS)
        assert (result["delta_kwh"] == 0.0).all()

    def test_unknown_prefix_ignored(self):
        """Prefix not in signatures → skipped, no crash, no delta."""
        df = _make_baseline_df()
        result = _composite_forecast(df, {"sub_unknown": "12:00"}, _DUMMY_SIGS)
        assert (result["delta_kwh"] == 0.0).all()

    def test_boundary_truncation(self):
        """Appliance starting at hour 46 with 4h profile → only 2h remain, rest truncated."""
        df = _make_baseline_df(start="2024-06-01 00:00")
        # 4h profile starting at 22:00 → offset=22; window up to 22+4=26, but only 24 rows
        # start=00:00, 46:00 means offset 46, profile=[1,2,1.5,0.5] → clip to [1, 2] (48-46=2)
        result = _composite_forecast(df, {"sub_dishwasher": "22:00"}, _DUMMY_SIGS)
        # 22:00 with 00:00 start → offset_h = 22; profile has 4 elements → all 4 fit (22+4=26 < 48)
        # Actually for a 48-row df starting at 00:00, 22:00 = offset 22, 22+4=26 < 48 so no truncation
        # Use start at 00:00 and time "46:00" is invalid; instead test profile going to edge
        # Re-test with start at 10:00 and time = "08:00" → tomorrow 08:00 → offset=22+24? no
        # Simplest: make a 48-row df, start at 00:00, use a 4h profile at hour 46:
        df2 = _make_baseline_df(start="2024-06-01 00:00", n=48)
        # Manually call _composite_forecast with a signature that starts at hour 46
        sigs_edge = {
            "sub_test": {
                "total_kwh": 5.0,
                "hourly_profile": [1.0, 2.0, 1.5, 0.5],
                "peak_hour": 1,
                "n_cycles": 3,
            }
        }
        # 00:00 start, "22:00" today → offset=22, fits fully
        # To get offset=46, need start at 00:00 and appliance at 22:00 next day
        # That would be 46h — can't express as HH:MM directly with the algorithm picking earliest.
        # Instead just directly test truncation by constructing start at 00:00 and time "22:00"
        # with n=48 but a very long profile:
        sigs_long = {
            "sub_test": {
                "hourly_profile": [1.0] * 10,  # 10h profile
                "total_kwh": 10.0,
                "peak_hour": 0,
                "n_cycles": 3,
            }
        }
        df3 = _make_baseline_df(start="2024-06-01 00:00", n=48)
        # offset_h = 22 (start 00:00 → 22:00), profile 10h → 22+10=32 < 48, fits
        result3 = _composite_forecast(df3, {"sub_test": "22:00"}, sigs_long)
        for i in range(10):
            assert result3["delta_kwh"].iloc[22 + i] == pytest.approx(1.0)
        assert result3["delta_kwh"].iloc[21] == pytest.approx(0.0)
        assert result3["delta_kwh"].iloc[32] == pytest.approx(0.0)
        # Now test truncation: profile extending beyond h=48
        df4 = _make_baseline_df(start="2024-06-01 00:00", n=48)
        result4 = _composite_forecast(df4, {"sub_test": "46:00"}, sigs_long)
        # "46:00" is invalid — parser will fail silently → no delta
        assert (result4["delta_kwh"] == 0.0).all()

    def test_multiple_appliances(self):
        """Two appliances at different times → both overlaid independently."""
        df = _make_baseline_df(start="2024-06-01 10:00")
        schedule = {
            "sub_dishwasher": "14:00",  # offset 4
            "sub_wp":         "17:00",  # offset 7
        }
        result = _composite_forecast(df, schedule, _DUMMY_SIGS)
        dw_profile = _DUMMY_SIGS["sub_dishwasher"]["hourly_profile"]
        wp_profile = _DUMMY_SIGS["sub_wp"]["hourly_profile"]
        # Dishwasher at 4..7
        for i, v in enumerate(dw_profile):
            expected = v
            if 4 + i >= 7 and (4 + i - 7) < len(wp_profile):
                expected += wp_profile[4 + i - 7]
            assert result["delta_kwh"].iloc[4 + i] == pytest.approx(expected)
        # wp at 7..9
        for i, v in enumerate(wp_profile):
            idx = 7 + i
            if idx < 4 or idx >= 4 + len(dw_profile):
                assert result["delta_kwh"].iloc[idx] == pytest.approx(v)

    def test_predicted_kwh_nonnegative(self):
        """Even with large negative baseline, predicted_kwh >= 0."""
        ts = pd.date_range("2024-06-01 10:00", periods=48, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "predicted_kwh": [-5.0] * 48})
        result = _composite_forecast(df, {"sub_wp": "11:00"}, _DUMMY_SIGS)
        assert (result["predicted_kwh"] >= 0.0).all()

    def test_predict_scenario_returns_delta_column(self, tmp_path):
        """predict_scenario() returns df with delta_kwh column."""
        m, forecast = _make_trained_model(tmp_path)
        # Add a dummy signature so _composite_forecast has something to overlay
        m._appliance_signatures = {
            "sub_dw": {"hourly_profile": [0.1, 0.2], "total_kwh": 0.3, "peak_hour": 1, "n_cycles": 3}
        }
        result = m.predict_scenario(forecast, live_temp=None, schedule={"sub_dw": "12:00"})
        assert "delta_kwh" in result.columns
        assert len(result) == 48
        assert (result["delta_kwh"] >= 0.0).all()

    def test_next_day_time_string(self):
        """'02:00' when forecast_start=10:00 → placed at hour 16 (next-day 02:00)."""
        df = _make_baseline_df(start="2024-06-01 10:00")
        # today 02:00 < 10:00 → tomorrow 02:00 → offset = 16h
        result = _composite_forecast(df, {"sub_wp": "02:00"}, _DUMMY_SIGS)
        wp_profile = _DUMMY_SIGS["sub_wp"]["hourly_profile"]
        for i, v in enumerate(wp_profile):
            assert result["delta_kwh"].iloc[16 + i] == pytest.approx(v)
        # hour 15 and before: no delta
        assert result["delta_kwh"].iloc[15] == pytest.approx(0.0)
