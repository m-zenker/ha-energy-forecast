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
"""
from __future__ import annotations
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.model import (
    _add_holiday_feature,
    _add_lag_and_rolling_prediction,
    _BRIDGE_CAP,
    _build_model,
    _FEATURES_BASE,
    _engineer_features,
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
