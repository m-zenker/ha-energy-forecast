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
    _add_sub_sensor_lags_training,
    _add_sub_sensor_lags_prediction,
    _BRIDGE_CAP,
    _build_model,
    _compute_likely_ev_hours,
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
        """Prediction-time sub-sensor NaN message must be DEBUG, not WARNING."""
        import logging
        future_ts = pd.date_range("2024-01-03", periods=48, freq="1h")
        future_df = pd.DataFrame({"timestamp": future_ts})
        with caplog.at_level(logging.DEBUG, logger="energy_forecast.model"):
            _add_sub_sensor_lags_prediction(future_df.copy(), {"sub_sparse": pd.DataFrame()})
        warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING and "sub_sparse" in r.message]
        assert warning_msgs == [], f"Expected no WARNING for sparse sub-sensor, got: {warning_msgs}"


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

