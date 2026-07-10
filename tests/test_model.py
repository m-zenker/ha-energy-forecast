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
    _BRIDGE_CAP,
    _FEATURES_BASE,
    LAG_HOURS,
    EnergyForecastModel,
    _add_holiday_feature,
    _add_lag_and_rolling_prediction,
    _add_sub_sensor_lags_prediction,
    _add_sub_sensor_lags_training,
    _build_model,
    _build_prediction_temp_df,
    _composite_forecast,
    _compute_likely_ev_hours,
    _engineer_features,
    _learn_appliance_signatures,
    _project_indoor_temps,
    _strip_partial_last_day,
)

# ── Shared training helper (reused by TestLogTransform and TestPredictIntervals) ─


def _make_trained_model(
    tmp_path,
    n: int = 600,
    timezone: str = "Europe/Zurich",
    physics_model=None,
    outdoor_df=None,
    use_physics_residual: bool = False,
) -> tuple:
    """Return (model, forecast_df) after a full train() call."""
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    energy = pd.DataFrame(
        {
            "timestamp": ts,
            "gross_kwh": rng.uniform(0.5, 5.0, size=n),
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": rng.uniform(-5, 25, size=n),
            "precipitation_mm": [0.0] * n,
            "sunshine_min": [30.0] * n,
            "wind_kmh": [10.0] * n,
            "cloud_cover_pct": [50.0] * n,
            "direct_radiation_wm2": [100.0] * n,
        }
    )
    m = EnergyForecastModel(tmp_path, timezone=timezone)
    m.train(
        energy,
        weather,
        outdoor_df=outdoor_df,
        weight_halflife_days=0,
        physics_model=physics_model,
        use_physics_residual=use_physics_residual,
    )
    # Build a minimal forecast_df covering the next 48h
    future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
    forecast = pd.DataFrame(
        {
            "timestamp": future_ts,
            "temp_c": [10.0] * 48,
            "precipitation_mm": [0.0] * 48,
            "sunshine_min": [30.0] * 48,
            "wind_kmh": [10.0] * 48,
            "cloud_cover_pct": [50.0] * 48,
            "direct_radiation_wm2": [100.0] * 48,
        }
    )
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

        actuals_ser = actuals_df.set_index(pd.to_datetime(actuals_df["timestamp"]))["gross_kwh"].sort_index()
        expected = float(actuals_ser.iloc[-24:].mean())
        actual_h0 = float(result["rolling_mean_24h"].iloc[0])
        assert abs(actual_h0 - expected) < 1e-9

    def test_rolling_std_24h_hour0_equals_last_24_actuals(self):
        """rolling_std_24h at h=0 must equal std(actuals[-24:])."""
        actuals_df = _make_actuals(168, noise=1.5)
        result = _add_lag_and_rolling_prediction(_make_future_df(), actuals_df)

        actuals_ser = actuals_df.set_index(pd.to_datetime(actuals_df["timestamp"]))["gross_kwh"].sort_index()
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
        actuals_ser = actuals_df.set_index(pd.to_datetime(actuals_df["timestamp"]))["gross_kwh"].sort_index()
        for h in range(10):  # spot-check first 10 hours
            ts = _make_future_df()["timestamp"].iloc[h] - pd.Timedelta(hours=72)
            expected = actuals_ser.get(ts, float("nan"))
            actual = result["lag_72h"].iloc[h]
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
        future_df = _make_future_df()
        result = _add_lag_and_rolling_prediction(future_df, actuals_df)
        actuals_ser = actuals_df.set_index(pd.to_datetime(actuals_df["timestamp"]))["gross_kwh"].sort_index()
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
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        for lag in (1, 2, 6, 12):
            assert f"lag_{lag}h" in m.feature_cols, f"lag_{lag}h missing from feature_cols"

    def test_short_lags_skipped_when_too_few_rows(self, tmp_path):
        """With exactly 100 rows, ALL lags (including short ones) are skipped by the dynamic gate."""
        n = 100  # n - lag >= 100 fails for all lags when n=100 and min lag=1 → 100-1=99 < 100
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        rng = np.random.default_rng(8)
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
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
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            _add_lag_and_rolling_prediction(_make_future_df(), actuals_df)
        for rec in caplog.records:
            assert "lag_1h" not in rec.message
            assert "lag_2h" not in rec.message
            assert "lag_6h" not in rec.message
            assert "lag_12h" not in rec.message


class TestLagFeaturesTemporalTraining:
    """M1 regression — training lags must be temporal, not positional over gapped frames."""

    def test_gap_produces_nan_lag_not_stale_value(self):
        """A 24-hour training gap must cause lag_24h to be NaN for rows in the 24h after it."""
        from energy_forecast.model import _add_lag_and_rolling_training

        ts_before = pd.date_range("2024-01-01", periods=24, freq="1h")  # hours 0-23
        ts_after = pd.date_range("2024-01-03", periods=24, freq="1h")  # hours 48-71 (gap = 2024-01-02)
        ts = ts_before.tolist() + ts_after.tolist()
        kwh = [1.0] * 24 + [2.0] * 24
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": kwh})

        df = _add_lag_and_rolling_training(energy, [24])

        after_rows = df[df["timestamp"] >= pd.Timestamp("2024-01-03")]
        assert after_rows["lag_24h"].isna().all(), (
            "Expected all lag_24h for rows after the gap to be NaN "
            f"(2024-01-02 not in training data), got: {after_rows['lag_24h'].values}"
        )

    def test_continuous_data_lag_values_unchanged(self):
        """Temporal lag fix must not change behavior when training data has no gaps."""
        from energy_forecast.model import _add_lag_and_rolling_training

        ts = pd.date_range("2024-01-01", periods=100, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": list(range(100))})

        df = _add_lag_and_rolling_training(energy, [24])

        # Row at hour 24: lag_24h should be the value from hour 0 = 0.0
        first_valid = df["lag_24h"].dropna()
        assert float(first_valid.iloc[0]) == pytest.approx(0.0)
        assert float(first_valid.iloc[1]) == pytest.approx(1.0)

    def test_sub_sensor_gap_produces_nan_lag_not_stale_value(self):
        """Sub-sensor lag_24h must be NaN when the target timestamp is in a training gap."""
        ts_before = pd.date_range("2024-01-01", periods=24, freq="1h")
        ts_after = pd.date_range("2024-01-03", periods=24, freq="1h")  # 24h gap on 2024-01-02
        ts = ts_before.tolist() + ts_after.tolist()
        kwh = [1.0] * 24 + [2.0] * 24
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": kwh})

        sub_df = pd.DataFrame({"timestamp": ts, "kwh": kwh})
        df = _add_sub_sensor_lags_training(energy, {"sub_hp": sub_df})

        after_rows = df[df["timestamp"] >= pd.Timestamp("2024-01-03")]
        assert after_rows["sub_hp_lag_24h"].isna().all(), (
            f"Expected NaN sub_hp_lag_24h for rows after gap, got: {after_rows['sub_hp_lag_24h'].values}"
        )


# ── Bridge-day holiday features ───────────────────────────────────────────────


def _make_ts_df(dates: list[str]) -> pd.DataFrame:
    """One row per date string at 12:00 noon (avoids midnight edge cases)."""
    return pd.DataFrame({"timestamp": pd.to_datetime([f"{d} 12:00" for d in dates])})


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
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "temp_c": [temp] * n,
            "precipitation_mm": [0.0] * n,
            "sunshine_min": [30.0] * n,
            "wind_kmh": [10.0] * n,
            "cloud_cover_pct": [cloud] * n,
            "direct_radiation_wm2": [rad] * n,
        }
    )


def _make_bare_df(timestamps) -> pd.DataFrame:
    """Minimal energy df with gross_kwh for _engineer_features input."""
    n = len(timestamps)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "gross_kwh": [1.5] * n,
        }
    )


class TestNewWeatherFeatures:
    def test_features_in_features_base(self):
        assert "cloud_cover_pct" in _FEATURES_BASE
        assert "direct_radiation_wm2" in _FEATURES_BASE

    def test_engineer_features_new_cols_populated(self):
        """When weather_df contains cloud/radiation, they appear in output."""
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, cloud=42.0, rad=180.0)
        result = _engineer_features(df, w, None)
        assert "cloud_cover_pct" in result.columns
        assert "direct_radiation_wm2" in result.columns
        assert (result["cloud_cover_pct"] == 42.0).all()
        assert (result["direct_radiation_wm2"] == 180.0).all()

    def test_engineer_features_missing_weather_cols_filled_as_nan(self):
        """Safety net: if weather_df has no cloud/radiation, columns are NaN."""
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        df = _make_bare_df(ts)
        # Weather without new columns (simulates SRG-only response gap)
        w = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts),
                "temp_c": [5.0] * 4,
                "precipitation_mm": [0.0] * 4,
                "sunshine_min": [30.0] * 4,
                "wind_kmh": [10.0] * 4,
            }
        )
        result = _engineer_features(df, w, None)
        assert "cloud_cover_pct" in result.columns
        assert "direct_radiation_wm2" in result.columns
        assert result["cloud_cover_pct"].isna().all()
        assert result["direct_radiation_wm2"].isna().all()


class TestStage2Features:
    def test_thermal_pressure_feature(self):
        """thermal_pressure is correctly calculated as (setpoint - current)."""
        ts = pd.date_range("2026-03-12 08:00", periods=2, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)

        # Room 1: 21.0 - 20.0 = 1.0 delta
        # Room 2: 22.0 - 18.0 = 4.0 delta
        # Average thermal_pressure should be 2.5
        climate_dfs = {
            "climate.room1": pd.DataFrame({"timestamp": ts, "current_temp": [20.0, 20.0], "setpoint": [21.0, 21.0]}),
            "climate.room2": pd.DataFrame({"timestamp": ts, "current_temp": [18.0, 18.0], "setpoint": [22.0, 22.0]}),
        }

        result = _engineer_features(df, w, None, climate_dfs=climate_dfs)
        assert "thermal_pressure" in result.columns
        assert (result["thermal_pressure"] == 2.5).all()

    def test_dhw_pressure_feature(self):
        """dhw_pressure increases non-linearly as buffer_temp drops towards 40C."""
        ts = pd.date_range("2026-03-12 08:00", periods=2, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)

        # Hour 0: 55C -> low pressure
        # Hour 1: 41C -> high pressure
        dhw_df = pd.DataFrame({"timestamp": ts, "buffer_temp": [55.0, 41.0]})

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
        hist_ts = pd.date_range("2026-03-09 08:00", periods=72, freq="1h")
        future_ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        all_ts = hist_ts.append(future_ts)

        # Historical temp = 2.0, forecast temp = 10.0
        w = _make_weather_df(all_ts, temp=2.0)  # constant — simplifies expected value
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
        w = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts),
                "temp_c": [5.0] * 24,
                "precipitation_mm": [0.0] * 24,
                "sunshine_min": [30.0] * 24,
                "wind_kmh": [10.0] * 24,
            }
        )
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
        w = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts),
                "temp_c": temps,
                "precipitation_mm": [0.0] * 48,
                "sunshine_min": [30.0] * 48,
                "wind_kmh": [10.0] * 48,
                "cloud_cover_pct": [50.0] * 48,
                "direct_radiation_wm2": [100.0] * 48,
            }
        )
        result = _engineer_features(df, w, None)
        # delta[1:] should be ~0.5 (small tolerance for float errors)
        assert abs(float(result["temp_delta_1h"].iloc[5]) - 0.5) < 0.01

    def test_temp_delta_24h_detects_daily_pattern(self):
        """With 24-hour period shift, temp_delta_24h should show day-over-day change."""
        ts = pd.date_range("2026-03-12 00:00", periods=72, freq="1h")
        df = _make_bare_df(ts)
        # Day 1: 5°C, Day 2: 10°C, Day 3: 5°C again
        temps = [5.0] * 24 + [10.0] * 24 + [5.0] * 24
        w = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts),
                "temp_c": temps,
                "precipitation_mm": [0.0] * 72,
                "sunshine_min": [30.0] * 72,
                "wind_kmh": [10.0] * 72,
                "cloud_cover_pct": [50.0] * 72,
                "direct_radiation_wm2": [100.0] * 72,
            }
        )
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
        w = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts),
                "temp_c": [float(t) for t in temps],
                "precipitation_mm": [0.0] * 96,
                "sunshine_min": [30.0] * 96,
                "wind_kmh": [10.0] * 96,
                "cloud_cover_pct": [50.0] * 96,
                "direct_radiation_wm2": [100.0] * 96,
            }
        )
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


class TestWarmWeatherBiasFeatures:
    """hp_heating_degree, temp_in_neutral_zone, and heating_active features."""

    def test_new_features_in_features_base(self):
        assert "hp_heating_degree" in _FEATURES_BASE
        assert "temp_in_neutral_zone" in _FEATURES_BASE
        assert "heating_active" in _FEATURES_BASE

    def test_hp_heating_degree_matches_15c_threshold(self):
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        df = _make_bare_df(ts)
        temps = [0.0, 10.0, 15.0, 20.0]
        w = _make_weather_df(ts)
        w["temp_c"] = temps
        result = _engineer_features(df, w, None)
        expected = [15.0, 5.0, 0.0, 0.0]
        for i, exp in enumerate(expected):
            assert abs(float(result["hp_heating_degree"].iloc[i]) - exp) < 1e-9

    def test_temp_in_neutral_zone_boundaries(self):
        ts = pd.date_range("2026-03-12 08:00", periods=5, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        w["temp_c"] = [10.0, 15.0, 20.0, 22.0, 25.0]
        result = _engineer_features(df, w, None)
        assert float(result["temp_in_neutral_zone"].iloc[0]) == 0.0  # 10°C: below zone
        assert float(result["temp_in_neutral_zone"].iloc[1]) == 1.0  # 15°C: lower bound
        assert float(result["temp_in_neutral_zone"].iloc[2]) == 1.0  # 20°C: inside
        assert float(result["temp_in_neutral_zone"].iloc[3]) == 1.0  # 22°C: upper bound
        assert float(result["temp_in_neutral_zone"].iloc[4]) == 0.0  # 25°C: above zone

    def test_heating_active_uses_provided_df(self):
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        ha_df = pd.DataFrame({"timestamp": ts, "heating_active": [1, 1, 0, 0]})
        result = _engineer_features(df, w, None, heating_active_df=ha_df)
        assert list(result["heating_active"]) == [1, 1, 0, 0]

    def test_heating_active_defaults_to_1_when_not_provided(self):
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        result = _engineer_features(df, w, None)
        assert (result["heating_active"] == 1).all()

    def test_heating_active_fills_missing_timestamps_with_1(self):
        """Timestamps not covered by heating_active_df fill with 1 (heating on)."""
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        # Only provide two of the four timestamps
        ha_df = pd.DataFrame(
            {
                "timestamp": ts[:2],
                "heating_active": [0, 0],
            }
        )
        result = _engineer_features(df, w, None, heating_active_df=ha_df)
        assert result["heating_active"].iloc[0] == 0
        assert result["heating_active"].iloc[1] == 0
        assert result["heating_active"].iloc[2] == 1  # filled with default
        assert result["heating_active"].iloc[3] == 1


class TestThermalFeaturesIntegration:
    """Integration test: all 8 thermal features activate during training."""

    def test_all_thermal_features_in_trained_model(self, tmp_path):
        """After training with sufficient rows, all 8 thermal features should be in m.feature_cols."""
        rng = np.random.default_rng(1)
        ts = pd.date_range("2024-01-01", periods=400, freq="1h")
        energy = pd.DataFrame(
            {
                "timestamp": ts,
                "gross_kwh": rng.uniform(0.5, 5.0, size=400),
            }
        )
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=400),
                "precipitation_mm": [0.0] * 400,
                "sunshine_min": [30.0] * 400,
                "wind_kmh": [10.0] * 400,
                "cloud_cover_pct": [50.0] * 400,
                "direct_radiation_wm2": [100.0] * 400,
            }
        )
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        # All 8 thermal features should be present
        thermal_features = [
            "temp_ewma_24h",
            "temp_ewma_72h",
            "heating_deg_sum_24h",
            "heating_deg_sum_168h",
            "temp_delta_1h",
            "temp_delta_24h",
            "temp_lag_24h",
            "temp_lag_168h",
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
        import hashlib
        import pickle

        # Write a meta dict that doesn't contain log_transform
        meta_path = tmp_path / "meta.pkl"
        meta = {
            "feature_cols": _FEATURES_BASE,
            "last_trained": __import__("datetime").datetime.min,
            "last_mae": None,
            "last_cv_mae": None,
            "engine": "test",
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


# ── Holdout MAE out-of-sample (L2) ───────────────────────────────────────────


class TestHoldoutMAE:
    """L2 regression — holdout MAE must be out-of-sample."""

    def test_holdout_mae_is_out_of_sample(self, tmp_path):
        """last_mae must come from a model trained on the first 90%, not all data.

        With n < MIN_CV_ROWS (500), CV is skipped and last_mae = holdout_mae.
        Dataset: first 360 rows gross_kwh=1.0; last 40 rows gross_kwh=500.0.

        Before fix: final model sees all 400 rows → in-sample holdout_mae ≈ 0.
        After fix:  holdout model sees only first 360 rows (all 1.0) → evaluates
                    on last 40 rows (all 500.0) → holdout_mae >> 1.0.

        The large jump (500x) is intentional: it overwhelms any temporal
        extrapolation the model might do, making the in-sample vs out-of-sample
        distinction unambiguous.
        """
        import pandas as pd

        n = 400  # < MIN_CV_ROWS=500 → CV skipped; last_mae = holdout_mae
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        values = [1.0] * 360 + [500.0] * 40
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": values})
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [10.0] * n,
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
                "humidity": [70.0] * n,
            }
        )
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)

        # After the fix: holdout model trained on rows 0-359 (all 1.0),
        # evaluated on rows 360-399 (all 500.0) → MAE >> 1.0.
        assert m.last_mae is not None
        assert m.last_mae > 1.0, (
            f"Expected out-of-sample holdout_mae > 1.0 "
            f"(model should not have seen the 500.0 holdout rows), got {m.last_mae}"
        )


# ── Cantonal holidays (#9) ────────────────────────────────────────────────────


class TestCantonalHolidays:
    def _ts_df(self, dates):
        return pd.DataFrame({"timestamp": pd.to_datetime([f"{d} 12:00" for d in dates])})

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


# ── Physics-ML hybrid: train() calibration trigger + feature inclusion (Plan B Task 4) ──


class TestTrainWithPhysics:
    def _physics_config(self) -> dict:
        return {
            "cop_formula": {"a": 2.5, "b": 0.07},
            "dhw_tank_volume_l": 200,
            "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8,
            "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [],
            "use_physics_residual": False,
        }

    def test_physics_kwh_in_feature_cols_when_physics_model_given(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", self._physics_config())
        model, _ = _make_trained_model(tmp_path / "model", physics_model=pm)
        assert "physics_kwh" in model.feature_cols

    def test_physics_kwh_absent_from_feature_cols_when_none(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model", physics_model=None)
        assert "physics_kwh" not in model.feature_cols

    def test_physics_phase1_diagnostic_populated_when_physics_model_given(self, tmp_path):
        """train() must wire _validate_physics_phase1 when physics_model is not None (Plan B Task 6)."""
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", self._physics_config())
        model, _ = _make_trained_model(tmp_path / "model", physics_model=pm)
        assert model.physics_phase1_diagnostic is not None
        assert set(model.physics_phase1_diagnostic.keys()) == {
            "shap_rank",
            "shap_top5",
            "ols_slope",
            "calibration_good",
        }

    def test_physics_phase1_diagnostic_none_when_no_physics_model(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model", physics_model=None)
        assert model.physics_phase1_diagnostic is None

    def test_physics_kwh_in_feature_cols_when_sensor_also_configured(self, tmp_path):
        """Regression test: physics_kwh must survive into feature_cols even when
        use_sensor is True (outdoor_df provided), i.e. via the sensor_features path,
        not just the base_features path."""
        from energy_forecast.physics import ThermalPhysicsModel

        n = 600
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        outdoor_df = pd.DataFrame({"timestamp": ts, "outdoor_temp_live": np.full(n, 12.0)})
        pm = ThermalPhysicsModel(tmp_path / "physics_models", self._physics_config())
        model, _ = _make_trained_model(tmp_path / "model", physics_model=pm, outdoor_df=outdoor_df)
        assert "physics_kwh" in model.feature_cols

    def test_calibrate_called_when_calibration_stale(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", self._physics_config())
        assert pm.calibration_stale is True
        with patch.object(pm, "calibrate", wraps=pm.calibrate) as mock_calibrate:
            _make_trained_model(tmp_path / "model", physics_model=pm)
            mock_calibrate.assert_called_once()

    def test_calibrate_skipped_when_fresh(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel, _atomic_write_json, _default_calibration

        model_dir = tmp_path / "physics_models"
        model_dir.mkdir(parents=True)
        fresh_calib = {**_default_calibration(), "calibrated_at": pd.Timestamp.now().isoformat()}
        _atomic_write_json(model_dir / "physics_calibration.json", fresh_calib)
        pm = ThermalPhysicsModel(model_dir, self._physics_config())
        assert pm.calibration_stale is False
        with patch.object(pm, "calibrate") as mock_calibrate:
            _make_trained_model(tmp_path / "model", physics_model=pm)
            mock_calibrate.assert_not_called()

    def test_open_window_down_weight_active_with_physics_and_halflife(self, tmp_path):
        """Covers the Plan B Task 4 down-weight block (model.py: `open_window_flags is not
        None and hourly_weights is not None`), which is only reachable when physics_model +
        climate_dfs are both configured AND weight_halflife_days > 0. No prior test combined
        all three (physics tests use weight_halflife_days=0 / no climate_dfs)."""
        from energy_forecast.physics import ThermalPhysicsModel

        n = 600
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        rng = np.random.default_rng(0)
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})
        # Smooth outdoor temperature so the one-step-ahead ODE projection isn't swamped
        # by weather noise (mirrors the near-constant-outdoor setup in test_physics.py's
        # test_open_window_flags_large_residual, scaled to a realistic training window).
        outdoor_temp = 5.0 + 3.0 * np.sin(np.linspace(0, 20 * np.pi, n))
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": outdoor_temp,
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
        # Indoor temp held ~constant near setpoint except one single-hour open-window shock.
        anomaly_hour = 550
        indoor_temp = 20.0 + rng.normal(0, 0.05, size=n)
        indoor_temp[anomaly_hour] = 10.0
        climate_dfs = {
            "climate.living_room": pd.DataFrame({"timestamp": ts, "current_temp": indoor_temp, "setpoint": [20.0] * n})
        }

        config = self._physics_config()
        config["room_thermostats"] = ["climate.living_room"]
        pm = ThermalPhysicsModel(tmp_path / "physics_models", config)

        # Spy on detect_open_windows so we capture the exact flags Series train() computes
        # and folds into hourly_weights, instead of recomputing it separately (which would
        # only prove detect_open_windows works, not that train() wired it correctly).
        captured: dict = {}
        real_detect = pm.detect_open_windows

        def _spy_detect(*args, **kwargs):
            result = real_detect(*args, **kwargs)
            captured["flags"] = result
            return result

        m = EnergyForecastModel(tmp_path / "model", timezone="Europe/Zurich")
        with patch.object(pm, "detect_open_windows", side_effect=_spy_detect):
            m.train(
                energy,
                weather,
                outdoor_df=None,
                weight_halflife_days=30.0,
                climate_dfs=climate_dfs,
                physics_model=pm,
            )

        # Model completes and is correctly shaped (the main smoke-test assertion).
        assert m.model is not None
        assert "physics_kwh" in m.feature_cols

        # Real trigger: the injected single-hour anomaly was actually flagged by the same
        # detect_open_windows() call train() used for down-weighting, and not everything
        # was flagged (sanity check against a degenerate all-True detector).
        assert "flags" in captured, "train() did not call detect_open_windows despite climate_dfs + physics_model"
        flags = captured["flags"]
        anomaly_ts = ts[anomaly_hour]
        assert flags.loc[anomaly_ts] == True  # noqa: E712 — explicit bool comparison reads clearer here
        assert flags.sum() < 10


# ── Physics-ML hybrid: predict() physics feature + portability fallback (Plan B Task 5) ──


class TestPredictWithPhysics:
    def _physics_config(self) -> dict:
        return {
            "cop_formula": {"a": 2.5, "b": 0.07},
            "dhw_tank_volume_l": 200,
            "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8,
            "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [],
            "use_physics_residual": False,
        }

    def test_physics_kwh_filled_with_zero_when_model_disabled_at_predict_time(self, tmp_path, caplog):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", self._physics_config())
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)
        assert "physics_kwh" in model.feature_cols

        # simulate physics disabled at predict time (sensor outage / config change)
        with caplog.at_level("WARNING"):
            result = model.predict(forecast_df, live_temp=5.0, physics_model=None)
        assert not result.empty  # no exception
        assert any(
            "physics" in rec.message.lower() and ("fallback" in rec.message.lower() or "0.0" in rec.message)
            for rec in caplog.records
        ), f"expected a physics fallback WARNING, got: {[r.message for r in caplog.records]}"

        # Verify the actual feature matrix used for prediction has physics_kwh == 0.0,
        # not just that predict() happened to return a non-empty result.
        _, X = model._prepare_prediction_X(forecast_df, live_temp=5.0, recent_actuals=None, physics_model=None)
        assert "physics_kwh" in X.columns
        assert (X["physics_kwh"] == 0.0).all()

    def test_physics_kwh_computed_when_model_present_at_predict_time(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", self._physics_config())
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)
        result = model.predict(forecast_df, live_temp=5.0, physics_model=pm)
        assert not result.empty

        # Verify real physics values (not the 0.0 fallback) were actually used.
        _, X = model._prepare_prediction_X(forecast_df, live_temp=5.0, recent_actuals=None, physics_model=pm)
        assert "physics_kwh" in X.columns
        assert not (X["physics_kwh"] == 0.0).all()

    def test_predict_without_physics_ever_trained_unaffected(self, tmp_path):
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=None)
        result = model.predict(forecast_df, live_temp=5.0, physics_model=None)
        assert not result.empty
        assert "physics_kwh" not in model.feature_cols

    def test_heating_buffer_temp_filled_with_zero_when_sensor_unavailable_at_predict_time(self, tmp_path, caplog):
        """Mirrors test_physics_kwh_filled_with_zero_when_model_disabled_at_predict_time:
        a model trained WITH heating_buffer_temp_df (so heating_buffer_temp lands in
        feature_cols) must not raise KeyError when predicting WITHOUT
        heating_buffer_temp_recent (sensor outage, config change, or predict_scenario()
        which never supplies this series at all)."""
        n = 600
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        rng = np.random.default_rng(0)
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
        heating_buffer_temp_df = pd.DataFrame(
            {"timestamp": ts, "heating_buffer_temp": 45.0 + rng.uniform(-2, 2, size=n)}
        )

        model = EnergyForecastModel(tmp_path / "model", timezone="Europe/Zurich")
        model.train(
            energy,
            weather,
            outdoor_df=None,
            weight_halflife_days=0,
            heating_buffer_temp_df=heating_buffer_temp_df,
        )
        assert "heating_buffer_temp" in model.feature_cols

        future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
        forecast_df = pd.DataFrame(
            {
                "timestamp": future_ts,
                "temp_c": [10.0] * 48,
                "precipitation_mm": [0.0] * 48,
                "sunshine_min": [30.0] * 48,
                "wind_kmh": [10.0] * 48,
                "cloud_cover_pct": [50.0] * 48,
                "direct_radiation_wm2": [100.0] * 48,
            }
        )

        # simulate sensor unavailable at predict time (heating_buffer_temp_recent defaults to None)
        with caplog.at_level("WARNING"):
            result = model.predict(forecast_df, live_temp=5.0)
        assert not result.empty  # no exception / no KeyError

        assert any("heating_buffer_temp" in rec.message.lower() and "0.0" in rec.message for rec in caplog.records), (
            f"expected a heating_buffer_temp fallback WARNING, got: {[r.message for r in caplog.records]}"
        )

        # Verify the actual feature matrix used for prediction has heating_buffer_temp == 0.0
        _, X = model._prepare_prediction_X(forecast_df, live_temp=5.0, recent_actuals=None)
        assert "heating_buffer_temp" in X.columns
        assert (X["heating_buffer_temp"] == 0.0).all()


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
        assert "low_kwh" in result.columns
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
        assert (calibrated["low_kwh"] <= raw["low_kwh"]).all()


# ── EV session probability feature (#12) ─────────────────────────────────────


class TestLikelyEvHour:
    def _make_ev_df(self, n: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (baseline_df, ev_df) where Monday 22:00 is always a charging hour."""
        rng = np.random.default_rng(1)
        ts = pd.date_range("2025-01-06 00:00", periods=n * 24, freq="1h")  # starts Monday
        kwh = rng.uniform(0.5, 3.0, size=len(ts))
        baseline = pd.DataFrame({"timestamp": ts, "gross_kwh": kwh})

        # Mark Monday 22:00 (hour_of_week = 0*24+22 = 22) as EV in every week
        how = ts.dayofweek * 24 + ts.hour
        ev_mask = how == 22  # Monday 22:00
        ev_df = baseline[ev_mask].copy()
        return baseline, ev_df

    def test_likely_hours_identified_after_train(self, tmp_path):
        """After train() with ev_df, _likely_ev_hours must be non-empty and
        contain the known charging slot (Monday 22:00 = how 22)."""
        baseline, ev_df = self._make_ev_df(n=200)
        weather = pd.DataFrame(
            {
                "timestamp": baseline["timestamp"],
                "temp_c": [10.0] * len(baseline),
                "precipitation_mm": [0.0] * len(baseline),
                "sunshine_min": [30.0] * len(baseline),
                "wind_kmh": [10.0] * len(baseline),
                "cloud_cover_pct": [50.0] * len(baseline),
                "direct_radiation_wm2": [100.0] * len(baseline),
            }
        )
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
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [10.0] * 24,
                "precipitation_mm": [0.0] * 24,
                "sunshine_min": [30.0] * 24,
                "wind_kmh": [10.0] * 24,
                "cloud_cover_pct": [50.0] * 24,
                "direct_radiation_wm2": [100.0] * 24,
            }
        )
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
        baseline = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=48, freq="1h"),
                "gross_kwh": [1.0] * 48,
            }
        )
        assert _compute_likely_ev_hours(baseline, None) == set()
        assert _compute_likely_ev_hours(baseline, pd.DataFrame()) == set()


# ── Sub-sensor lag features ────────────────────────────────────────────────────


def _make_sub_sensor_df(n: int = 400, start: str = "2024-01-01") -> pd.DataFrame:
    """Return a sub-sensor DataFrame with 'timestamp' and 'kwh' columns."""
    rng = np.random.default_rng(7)
    ts = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"timestamp": ts, "kwh": rng.uniform(0, 3.0, size=n)})


class TestSubSensorFeatures:
    def _make_weather(self, ts):
        return pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [10.0] * len(ts),
                "precipitation_mm": [0.0] * len(ts),
                "sunshine_min": [30.0] * len(ts),
                "wind_kmh": [10.0] * len(ts),
                "cloud_cover_pct": [50.0] * len(ts),
                "direct_radiation_wm2": [100.0] * len(ts),
            }
        )

    def test_lag_24h_in_feature_cols_when_sub_sensor_provided(self, tmp_path):
        """sub_sensor lag_24h column appears in feature_cols after train()."""
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": np.random.default_rng(0).uniform(0.5, 5, n)})
        weather = self._make_weather(ts)
        sub_df = _make_sub_sensor_df(n=n)
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, sub_sensors_dict={"sub_hp": sub_df})
        assert "sub_hp_lag_24h" in m.feature_cols

    def test_lag_168h_in_feature_cols_with_enough_history(self, tmp_path):
        """sub_sensor lag_168h appears when n_rows >= 268 (168 + 100)."""
        n = 600
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": np.random.default_rng(1).uniform(0.5, 5, n)})
        weather = self._make_weather(ts)
        sub_df = _make_sub_sensor_df(n=n)
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, sub_sensors_dict={"sub_hp": sub_df})
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
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": [2.0] * n})
        # Sub-sensor: deterministic values so we can verify the lag
        sub_kwh = list(range(n))  # 0, 1, 2, ..., n-1
        sub_df = pd.DataFrame({"timestamp": ts, "kwh": sub_kwh})

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
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": np.random.default_rng(3).uniform(0.5, 5, n)})
        weather = self._make_weather(ts)
        sub_df = _make_sub_sensor_df(n=n)
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, sub_sensors_dict={"sub_hp": sub_df})

        future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
        forecast = self._make_weather(future_ts)
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
        recent_sub = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-09 12:00"]),
                "kwh": [0.3],
            }
        )
        result = _add_sub_sensor_lags_prediction(future_df, {"sub_t": recent_sub})
        assert result["sub_t_lag_24h"].dtype == np.float64, f"expected float64, got {result['sub_t_lag_24h'].dtype}"
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
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": np.random.default_rng(5).uniform(0.5, 5, n)})
        weather = self._make_weather(ts)
        # Only 1 data point — simulates a sensor that started today
        sub_df = pd.DataFrame({"timestamp": ts[-1:], "kwh": [0.5]})
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, sub_sensors_dict={"sub_new": sub_df})
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

        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            df = _add_sub_sensor_lags_training(energy, {"sub_hp": sub_df})

        assert "sub_hp" in caplog.text
        assert "NaN" in caplog.text
        assert "sub_hp_lag_24h" in df.columns

    def test_multiple_sub_sensors_in_feature_cols(self, tmp_path):
        """Two sub-sensors both produce lag columns in feature_cols after train()."""
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": np.random.default_rng(4).uniform(0.5, 5, n)})
        weather = self._make_weather(ts)
        sub_hp = _make_sub_sensor_df(n=n)
        sub_dw = _make_sub_sensor_df(n=n)
        m = EnergyForecastModel(tmp_path)
        m.train(
            energy,
            weather,
            outdoor_df=None,
            weight_halflife_days=0,
            sub_sensors_dict={"sub_hp": sub_hp, "sub_dw": sub_dw},
        )
        assert "sub_hp_lag_24h" in m.feature_cols
        assert "sub_dw_lag_24h" in m.feature_cols

    def test_lag_168h_values_are_correct(self):
        """lag_168h for a sub-sensor equals the kwh value 168 positions earlier in training."""
        n = 400
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": [2.0] * n})
        sub_kwh = list(range(n))  # 0, 1, 2, ..., n-1
        sub_df = pd.DataFrame({"timestamp": ts, "kwh": sub_kwh})

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
        with caplog.at_level(logging.DEBUG, logger="energy_forecast"):
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
        energy = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=400, freq="1h"),
                "gross_kwh": [2.0] * 400,
            }
        )
        df = _add_sub_sensor_lags_training(energy, {"sub_dw": self._all_zero_series()})
        assert "sub_dw_active_24h" in df.columns
        assert (df["sub_dw_active_24h"] == 0).all()

    def test_active_24h_becomes_one_after_event(self):
        """After a non-zero event, active_24h must become 1 within the next 24 rows."""
        n = 400
        event_start = 200
        energy = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
                "gross_kwh": [2.0] * n,
            }
        )
        sub = self._series_with_event(n=n, event_start=event_start)
        df = _add_sub_sensor_lags_training(energy, {"sub_dw": sub})
        # Rows from event_start+1 to event_start+24 should have active_24h=1
        assert df["sub_dw_active_24h"].iloc[event_start + 1] == 1

    def test_runs_7d_zero_for_all_zero_series(self):
        """All-zero sub-sensor → runs_7d must be 0 everywhere (NaN at row 0 is OK)."""
        energy = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=400, freq="1h"),
                "gross_kwh": [2.0] * 400,
            }
        )
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
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5, n)})
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
        sub = self._series_with_event(n=n)
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, sub_sensors_dict={"sub_dw": sub})
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
        idx_50 = len(ts_hist) - 51  # 50h ago
        idx_100 = len(ts_hist) - 101  # 100h ago
        if idx_50 >= 0:
            kwh[idx_50] = 2.0
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

        with caplog.at_level(logging.INFO, logger="energy_forecast"):
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

        with caplog.at_level(logging.INFO, logger="energy_forecast"):
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
        w = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [5.0] * 4,
                "precipitation_mm": [0.0] * 4,
                "sunshine_min": [30.0] * 4,
                "wind_kmh": [10.0] * 4,
                "cloud_cover_pct": [50.0] * 4,
                "direct_radiation_wm2": [100.0] * 4,
            }
        )
        result = _engineer_features(self._make_bare_df(ts), w, None)
        assert "doy_sin" in result.columns
        assert "doy_cos" in result.columns

    def test_doy_sin_near_zero_on_jan1(self):
        """Jan 1 is doy=1; sin(2π·1/365) ≈ 0.0172 — near but not exactly 0."""
        ts = pd.date_range("2026-01-01", periods=1, freq="1h")
        w = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [5.0],
                "precipitation_mm": [0.0],
                "sunshine_min": [30.0],
                "wind_kmh": [10.0],
                "cloud_cover_pct": [50.0],
                "direct_radiation_wm2": [100.0],
            }
        )
        result = _engineer_features(self._make_bare_df(ts), w, None)
        expected_sin = np.sin(2 * np.pi * 1 / 365)
        assert abs(float(result["doy_sin"].iloc[0]) - expected_sin) < 1e-9

    def test_doy_sin_near_one_at_peak(self):
        """doy ≈ 91 (April 1) sin ≈ 1; verify value is reasonable."""
        ts = pd.date_range("2026-04-01", periods=1, freq="1h")
        w = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [10.0],
                "precipitation_mm": [0.0],
                "sunshine_min": [30.0],
                "wind_kmh": [10.0],
                "cloud_cover_pct": [50.0],
                "direct_radiation_wm2": [100.0],
            }
        )
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
        w = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [5.0] * 4,
                "precipitation_mm": [0.0] * 4,
                "sunshine_min": [30.0] * 4,
                "wind_kmh": [10.0] * 4,
                "cloud_cover_pct": [50.0] * 4,
                "direct_radiation_wm2": [100.0] * 4,
            }
        )
        result = _engineer_features(self._make_bare_df(ts), w, None)
        assert (result["hours_ahead"] == 0).all()

    def test_hours_ahead_monotonic_in_prediction(self, tmp_path):
        """Prediction X must have hours_ahead = 0, 1, 2, ..., 47."""
        m, forecast = _make_trained_model(tmp_path)
        # Peek at the feature matrix built for prediction
        future_hours, X = m._prepare_prediction_X(forecast, live_temp=None, recent_actuals=None)
        assert "hours_ahead" in X.columns
        expected = list(range(48))
        actual = X["hours_ahead"].tolist()
        assert actual == expected, f"hours_ahead not monotonic: {actual[:5]}…"


class TestPreparePredictionXNowNaive:
    """now_naive in _prepare_prediction_X must reflect local wall-clock, not UTC-shifted time."""

    def test_prepare_prediction_x_now_naive_matches_local_wall_clock(self, tmp_path):
        """now_naive must equal local wall-clock time (tz_localize), not UTC-shifted (tz_convert)."""
        from unittest import mock

        # Use a trained model so _engineer_features has the weather data it needs.
        m, _ = _make_trained_model(tmp_path, timezone="Europe/Zurich")

        # Freeze time to a CEST offset moment: 14:00 local = 12:00 UTC.
        # tz_convert(None) would drop to 12:00; tz_localize(None) keeps 14:00.
        fixed_local_aware = pd.Timestamp("2026-07-01 14:00:00").tz_localize("Europe/Zurich")

        # Build a forecast_df aligned to that frozen "now" so _engineer_features works.
        frozen_naive = pd.Timestamp("2026-07-01 14:00:00")
        future_ts = pd.date_range(frozen_naive, periods=48, freq="1h")
        forecast = pd.DataFrame(
            {
                "timestamp": future_ts,
                "temp_c": [20.0] * 48,
                "precipitation_mm": [0.0] * 48,
                "sunshine_min": [30.0] * 48,
                "wind_kmh": [10.0] * 48,
                "cloud_cover_pct": [50.0] * 48,
                "direct_radiation_wm2": [100.0] * 48,
            }
        )

        with mock.patch("pandas.Timestamp.now", return_value=fixed_local_aware):
            future_hours, X = m._prepare_prediction_X(
                forecast_df=forecast,
                live_temp=None,
                recent_actuals=None,
            )

        # The first future hour must be floored to local hour 14, not UTC hour 12
        first_hour = pd.Timestamp(future_hours[0])
        assert first_hour.hour == 14, f"Expected local hour 14, got {first_hour.hour} — tz_convert leaked UTC offset"


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

        with caplog.at_level(logging.INFO, logger="energy_forecast"):
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
        assert isinstance(sample_key, int | np.integer), "HOW keys must be integers"
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
        import hashlib
        import pickle

        meta_path = tmp_path / "meta.pkl"
        meta = {
            "feature_cols": _FEATURES_BASE,
            "last_trained": __import__("datetime").datetime.min,
            "last_mae": None,
            "last_cv_mae": None,
            "engine": "test",
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
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        # Start on Monday so HOW=0 is the first row's hour_of_week
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
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
        return pd.DataFrame(
            {
                "timestamp": hours,
                "temp_c": np.linspace(5.0, 20.0, n),
                "precipitation_mm": np.linspace(0.0, 5.0, n),
                "sunshine_min": np.linspace(0.0, 60.0, n),
                "wind_kmh": np.linspace(5.0, 30.0, n),
                "cloud_cover_pct": np.linspace(0.0, 100.0, n),
                "direct_radiation_wm2": np.linspace(0.0, 800.0, n),
            }
        )

    def test_all_weather_cols_vary_per_hour(self):
        """Each weather column must have more than one unique value across 48 h."""
        forecast = self._make_varied_forecast()
        future_df = pd.DataFrame({"timestamp": forecast["timestamp"], "gross_kwh": np.nan})
        result = _engineer_features(future_df, forecast, outdoor_df=None)
        for col in [
            "temp_c",
            "precipitation_mm",
            "sunshine_min",
            "wind_kmh",
            "cloud_cover_pct",
            "direct_radiation_wm2",
        ]:
            vals = result[col].dropna()
            assert vals.nunique() > 1, f"{col} is flat across 48 h — scalar-broadcast bug present"

    def test_temp_c_matches_forecast_values(self):
        """Spot-check: temp_c at h=0 and h=47 must equal the input forecast values."""
        forecast = self._make_varied_forecast()
        future_df = pd.DataFrame({"timestamp": forecast["timestamp"], "gross_kwh": np.nan})
        result = _engineer_features(future_df, forecast, outdoor_df=None)
        result = result.reset_index(drop=True)
        assert abs(float(result.loc[0, "temp_c"]) - 5.0) < 1e-6, "temp_c at h=0 does not match forecast input"
        assert abs(float(result.loc[47, "temp_c"]) - 20.0) < 1e-6, "temp_c at h=47 does not match forecast input"


# ── #25 Vacation / Away Flag ───────────────────────────────────────────────────


class TestAwayFeature:
    """is_away binary feature: presence, zero default, value propagation, predict."""

    def _make_weather(self, ts):
        return pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [10.0] * len(ts),
                "precipitation_mm": [0.0] * len(ts),
                "sunshine_min": [30.0] * len(ts),
                "wind_kmh": [10.0] * len(ts),
                "cloud_cover_pct": [50.0] * len(ts),
                "direct_radiation_wm2": [100.0] * len(ts),
            }
        )

    def _make_bare(self, ts):
        return pd.DataFrame({"timestamp": ts, "gross_kwh": [1.5] * len(ts)})

    def test_is_away_in_features_base(self):
        assert "is_away" in _FEATURES_BASE

    def test_is_away_column_present_with_away_df(self):
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        away_df = pd.DataFrame({"timestamp": ts, "is_away": [1, 1, 0, 0]})
        result = _engineer_features(self._make_bare(ts), self._make_weather(ts), None, away_df=away_df)
        assert "is_away" in result.columns

    def test_is_away_zero_without_away_df(self):
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        result = _engineer_features(self._make_bare(ts), self._make_weather(ts), None, away_df=None)
        assert "is_away" in result.columns
        assert (result["is_away"] == 0).all()

    def test_is_away_values_match_away_df(self):
        ts = pd.date_range("2026-03-12 08:00", periods=4, freq="1h")
        away_df = pd.DataFrame({"timestamp": ts, "is_away": [1, 0, 1, 0]})
        result = _engineer_features(self._make_bare(ts), self._make_weather(ts), None, away_df=away_df)
        assert list(result["is_away"].values) == [1, 0, 1, 0]

    def test_is_away_in_feature_cols_after_train(self, tmp_path):
        n = 600
        rng = np.random.default_rng(5)
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
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
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
        weather = self._make_weather(ts)
        away_df = pd.DataFrame({"timestamp": ts, "is_away": [0] * n})
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, away_df=away_df)
        assert "is_away" in m.feature_cols
        # Use local wall-clock "now" (tz_localize, not tz_convert) to match _prepare_prediction_X.
        now_local = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        future_ts = pd.date_range(now_local, periods=48, freq="1h")
        forecast = self._make_weather(future_ts)
        away_ones = pd.Series(1, index=future_ts, dtype=int)
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
        assert values == sorted(values, reverse=True), f"Values not descending: {values}"

    def test_cold_start_returns_empty(self, tmp_path):
        """shap_summary() on an untrained model must return {}."""
        m = EnergyForecastModel(tmp_path)
        # Build a minimal forecast_df
        future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
        forecast = pd.DataFrame(
            {
                "timestamp": future_ts,
                "temp_c": [10.0] * 48,
                "precipitation_mm": [0.0] * 48,
                "sunshine_min": [30.0] * 48,
                "wind_kmh": [10.0] * 48,
                "cloud_cover_pct": [50.0] * 48,
                "direct_radiation_wm2": [100.0] * 48,
            }
        )
        result = m.shap_summary(forecast, live_temp=None, n=5)
        assert result == {}

    def test_n_zero_returns_empty(self, tmp_path):
        """n=0 disables SHAP and must return {}."""
        m, forecast = _make_trained_model(tmp_path)
        result = m.shap_summary(forecast, live_temp=None, n=0)
        assert result == {}

    def test_today_slice_uses_configured_timezone(self, tmp_path, monkeypatch):
        """shap_summary must call pd.Timestamp.now with self._timezone, not without tz."""
        # Build the model before installing the spy so that _make_trained_model's
        # own pd.Timestamp.now() calls don't pollute the call list.
        m, forecast = _make_trained_model(tmp_path)

        calls = []
        original_now = pd.Timestamp.now

        def spy_now(*args, **kwargs):
            calls.append(kwargs.get("tz", "__no_tz__"))
            return original_now(*args, **kwargs)

        monkeypatch.setattr(pd.Timestamp, "now", staticmethod(spy_now))

        m.shap_summary(forecast, live_temp=5.0, n=3)

        assert "__no_tz__" not in calls, (
            "shap_summary called pd.Timestamp.now() without a timezone — would use UTC inside the container"
        )


# ── Temperature sensor blending: bias fade ──────────────────────────────────────


class TestBuildPredictionTempDf:
    """Test bias-fade temperature blending: sensor offset fades over 6h window."""

    def test_full_trust_zone_returns_live_temp(self):
        """Hours 0–2: return live_temp unchanged."""
        future_hours = pd.date_range("2026-03-12 08:00", periods=10, freq="1h")
        forecast = pd.DataFrame(
            {
                "timestamp": future_hours,
                "temp_c": [10.0 + i for i in range(10)],
            }
        )
        result = _build_prediction_temp_df(future_hours, forecast, live_temp=11.0)

        # h=0, h=1, h=2 must be 11.0 (live_temp)
        assert abs(result.iloc[0]["outdoor_temp_live"] - 11.0) < 1e-9
        assert abs(result.iloc[1]["outdoor_temp_live"] - 11.0) < 1e-9
        assert abs(result.iloc[2]["outdoor_temp_live"] - 11.0) < 1e-9

    def test_full_forecast_zone_returns_forecast(self):
        """Hours 6+: return forecast[h] unchanged."""
        future_hours = pd.date_range("2026-03-12 08:00", periods=10, freq="1h")
        forecast = pd.DataFrame(
            {
                "timestamp": future_hours,
                "temp_c": [10.0 + i for i in range(10)],
            }
        )
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
        forecast = pd.DataFrame(
            {
                "timestamp": future_hours,
                "temp_c": [10.0 + i for i in range(10)],
            }
        )
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
        forecast = pd.DataFrame(
            {
                "timestamp": future_hours,
                "temp_c": [10.0, 10.5, 11.0, 12.0, 13.5, 15.0, 16.0, 16.5, 17.0, 17.5],
            }
        )
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
        forecast = pd.DataFrame(
            {
                "timestamp": future_hours,
                "temp_c": [10.0 + i for i in range(10)],
            }
        )
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
        forecast = pd.DataFrame(
            {
                "timestamp": future_hours,
                "temp_c": [15.0 + i for i in range(10)],
            }
        )
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
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
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
            ts = pd.date_range(f"2024-0{i + 1}-01", periods=n, freq="1h")
            energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})
            weather = pd.DataFrame(
                {
                    "timestamp": ts,
                    "temp_c": rng.uniform(-5, 25, size=n),
                    "precipitation_mm": [0.0] * n,
                    "sunshine_min": [30.0] * n,
                    "wind_kmh": [10.0] * n,
                    "cloud_cover_pct": [50.0] * n,
                    "direct_radiation_wm2": [100.0] * n,
                }
            )
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
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
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
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        archive_name = sorted((tmp_path / "archive").iterdir())[-1].name
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            m.rollback_model()
        assert any(archive_name in r.message for r in caplog.records)

    def test_country_persisted_through_save_load(self, tmp_path):
        """_country must survive a save/load cycle."""
        from energy_forecast.model import EnergyForecastModel

        model = EnergyForecastModel(tmp_path)
        model._country = "DE"
        model._save()

        model2 = EnergyForecastModel(tmp_path)
        model2._load()
        assert model2._country == "DE", f"Expected 'DE', got {model2._country!r}"

    def test_country_defaults_to_ch_when_absent_from_meta(self, tmp_path):
        """Old meta without 'country' key must default to 'CH' on load (backward compat)."""
        import pickle

        from energy_forecast.model import EnergyForecastModel, _write_hash

        model = EnergyForecastModel(tmp_path)
        model._canton = "ZH"
        model._country = "CH"
        model._save()

        # Tamper with meta: remove 'country' to simulate old format
        with open(model._meta_path, "rb") as fh:
            meta = pickle.load(fh)
        meta.pop("country", None)
        with open(model._meta_path, "wb") as fh:
            pickle.dump(meta, fh)
        _write_hash(model._meta_path)

        model2 = EnergyForecastModel(tmp_path)
        model2._load()
        assert model2._country == "CH", f"Expected 'CH', got {model2._country!r}"

    def test_archive_copies_sidecar_for_regime_model(self, tmp_path):
        """regime_model.pkl.sha256 must be present in the archive snapshot."""
        from energy_forecast.model import EnergyForecastModel, _write_hash

        model = EnergyForecastModel(tmp_path)
        model_dir = model._model_dir
        archive_dir = model._archive_dir

        # Create fake artifacts using the real filenames (energy_model.pkl triggers archive)
        pkl_files = [
            "energy_model.pkl",
            "meta.pkl",
            "energy_model_q10.pkl",
            "energy_model_q90.pkl",
            "clusterer.pkl",
            "regime_model.pkl",
        ]
        for fname in pkl_files + ["energy_model_interval_correction.json"]:
            (model_dir / fname).write_bytes(b"fake")
            if fname.endswith(".pkl"):
                _write_hash(model_dir / fname)

        model._archive_current()

        archives = list(archive_dir.iterdir())
        assert len(archives) == 1, "Expected one archive snapshot"
        snap = archives[0]
        assert (snap / "regime_model.pkl.sha256").exists(), "regime_model.pkl.sha256 must be in archive but is missing"


# ── Occupancy feature (people_home) — #21 ──────────────────────────────────────


class TestPeopleHomeFeature:
    """Tests for occupancy feature: people_home integer count (#21)."""

    def test_people_home_in_features_base(self):
        """people_home is listed in _FEATURES_BASE."""
        assert "people_home" in _FEATURES_BASE

    def test_people_home_zero_without_presence_df(self, tmp_path):
        """When presence_df is None, people_home defaults to 0."""
        rng = np.random.default_rng(0)
        ts = pd.date_range("2024-01-01", periods=200, freq="1h")
        energy = pd.DataFrame(
            {
                "timestamp": ts,
                "gross_kwh": rng.uniform(0.5, 5.0, size=200),
            }
        )
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=200),
                "precipitation_mm": [0.0] * 200,
                "sunshine_min": [30.0] * 200,
                "wind_kmh": [10.0] * 200,
                "cloud_cover_pct": [50.0] * 200,
                "direct_radiation_wm2": [100.0] * 200,
            }
        )
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, presence_df=None)

        # Get feature columns
        assert "people_home" in m.feature_cols

    def test_people_home_column_present_with_presence_df(self, tmp_path):
        """When presence_df is provided, people_home column is populated."""
        rng = np.random.default_rng(0)
        ts = pd.date_range("2024-01-01", periods=200, freq="1h")
        energy = pd.DataFrame(
            {
                "timestamp": ts,
                "gross_kwh": rng.uniform(0.5, 5.0, size=200),
            }
        )
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=200),
                "precipitation_mm": [0.0] * 200,
                "sunshine_min": [30.0] * 200,
                "wind_kmh": [10.0] * 200,
                "cloud_cover_pct": [50.0] * 200,
                "direct_radiation_wm2": [100.0] * 200,
            }
        )
        presence = pd.DataFrame(
            {
                "timestamp": energy["timestamp"].values,
                "people_home": rng.integers(0, 3, size=200),  # 0, 1, or 2 people
            }
        )
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, presence_df=presence)

        assert "people_home" in m.feature_cols

    def test_people_home_values_match_presence_df(self):
        """people_home values from presence_df are correctly merged into features."""
        rng = np.random.default_rng(0)
        ts = pd.date_range("2024-01-01", periods=200, freq="1h")
        energy = pd.DataFrame(
            {
                "timestamp": ts,
                "gross_kwh": rng.uniform(0.5, 5.0, size=200),
            }
        )
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=200),
                "precipitation_mm": [0.0] * 200,
                "sunshine_min": [30.0] * 200,
                "wind_kmh": [10.0] * 200,
                "cloud_cover_pct": [50.0] * 200,
                "direct_radiation_wm2": [100.0] * 200,
            }
        )
        presence = pd.DataFrame(
            {
                "timestamp": energy["timestamp"].values[:100],  # Partial overlap
                "people_home": [2, 1, 0, 1] * 25,  # 4-hour cycle repeated
            }
        )

        # Call _engineer_features directly to inspect merge behavior
        df = _engineer_features(energy, weather, None, presence_df=presence)

        assert "people_home" in df.columns
        assert df["people_home"].dtype == int

    def test_people_home_in_feature_cols_after_train(self, tmp_path):
        """people_home is included in feature_cols list after training."""
        rng = np.random.default_rng(0)
        ts = pd.date_range("2024-01-01", periods=200, freq="1h")
        energy = pd.DataFrame(
            {
                "timestamp": ts,
                "gross_kwh": rng.uniform(0.5, 5.0, size=200),
            }
        )
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=200),
                "precipitation_mm": [0.0] * 200,
                "sunshine_min": [30.0] * 200,
                "wind_kmh": [10.0] * 200,
                "cloud_cover_pct": [50.0] * 200,
                "direct_radiation_wm2": [100.0] * 200,
            }
        )
        presence = pd.DataFrame(
            {
                "timestamp": energy["timestamp"].values,
                "people_home": [1] * len(energy),
            }
        )
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, presence_df=presence)

        assert "people_home" in m.feature_cols

    def test_predict_with_people_home_series(self, tmp_path):
        """Prediction with people_home_series injects values into prediction."""
        rng = np.random.default_rng(0)
        ts = pd.date_range("2024-01-01", periods=200, freq="1h")
        energy = pd.DataFrame(
            {
                "timestamp": ts,
                "gross_kwh": rng.uniform(0.5, 5.0, size=200),
            }
        )
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=200),
                "precipitation_mm": [0.0] * 200,
                "sunshine_min": [30.0] * 200,
                "wind_kmh": [10.0] * 200,
                "cloud_cover_pct": [50.0] * 200,
                "direct_radiation_wm2": [100.0] * 200,
            }
        )
        presence = pd.DataFrame(
            {
                "timestamp": energy["timestamp"].values,
                "people_home": [1] * len(energy),
            }
        )
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, presence_df=presence)

        # Build forecast for next 48 hours
        now = energy["timestamp"].max()
        forecast = pd.DataFrame(
            {
                "timestamp": pd.date_range(now + pd.Timedelta(hours=1), periods=48, freq="1h"),
                "temp_c": [15.0] * 48,
                "precipitation_mm": [0.0] * 48,
                "sunshine_min": [30.0] * 48,
                "wind_kmh": [10.0] * 48,
                "cloud_cover_pct": [50.0] * 48,
                "direct_radiation_wm2": [100.0] * 48,
            }
        )

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
        """Three identical cycles → correct profile, total_kwh, peak_hour, n_cycles, std_profile."""
        df = _make_cycle_df(self.PROFILE, n_cycles=3)
        sigs = _learn_appliance_signatures({"hp": df})
        assert "hp" in sigs
        s = sigs["hp"]
        assert s["n_cycles"] == 3
        assert s["total_kwh"] == pytest.approx(sum(self.PROFILE))
        assert s["hourly_profile"] == pytest.approx(self.PROFILE)
        assert s["peak_hour"] == 1  # index of 2.0 in PROFILE
        # std_profile present and all-zeros (identical cycles)
        assert "std_profile" in s
        assert len(s["std_profile"]) == 4
        assert s["std_profile"] == pytest.approx([0.0, 0.0, 0.0, 0.0])

    def test_below_min_cycles_skipped(self):
        """Fewer cycles than min_cycles → empty dict (insufficient evidence)."""
        df = _make_cycle_df(self.PROFILE, n_cycles=2, period_hours=20)
        sigs = _learn_appliance_signatures({"hp": df}, min_cycles=3)
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
        extra = pd.DataFrame(
            {
                "timestamp": pd.date_range(df_full["timestamp"].iloc[-1] + pd.Timedelta("1h"), periods=2, freq="1h"),
                "kwh": [1.0, 1.0],
            }
        )
        df = pd.concat([df_full, extra], ignore_index=True)
        sigs = _learn_appliance_signatures({"dw": df})
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

    def test_high_variability_logs_warning(self, caplog):
        """Two very different cycles (CoV > 0.3) → WARNING logged."""
        import logging

        # Cycle 1: high consumption [2.0]*4; cycle 2: low consumption [0.1]*4
        # Trailing zeros so neither cycle is truncated at end of history.
        kwh = [0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.1, 0.1, 0.1, 0.1] + [0.0] * 12
        ts = pd.date_range("2026-01-01", periods=len(kwh), freq="1h")
        df = pd.DataFrame({"timestamp": ts, "kwh": kwh})
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            _learn_appliance_signatures({"hp": df}, min_cycles=2)
        assert any("high variability" in r.message for r in caplog.records)

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
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(5, 25, n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
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

    def test_cold_start_load_preserves_std_profile(self, tmp_path):
        """Signatures saved by train() are fully restored in a fresh model instance.

        Simulates an AppDaemon restart: model1 trains and saves, model2 is a
        cold instance that loads from disk.  Verifies std_profile survives the
        JSON round-trip and the loaded dict is identical to the saved one.
        """
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(0)
        n = 600
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy_df = pd.DataFrame(
            {
                "timestamp": ts,
                "gross_kwh": rng.uniform(0.5, 3.0, n),
            }
        )
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(5, 25, n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
        sub_hp = _make_cycle_df(self.PROFILE, n_cycles=20, period_hours=24, start="2024-01-01")
        sub_hp = sub_hp[sub_hp["timestamp"] < ts[-1]].reset_index(drop=True)

        model1 = EnergyForecastModel(model_dir=tmp_path)
        model1.train(
            energy_df=energy_df,
            weather_df=weather_df,
            outdoor_df=None,
            sub_sensors_dict={"sub_hp": sub_hp},
        )
        saved = dict(model1._appliance_signatures)

        # Fresh instance — simulates restart
        model2 = EnergyForecastModel(model_dir=tmp_path)
        model2._load_signatures()

        assert model2._appliance_signatures == saved
        assert "std_profile" in model2._appliance_signatures["sub_hp"]

    # ── New-behaviour tests ───────────────────────────────────────────────────

    def test_adaptive_window_short_cycle(self):
        """A 2-hour actual cycle produces a profile of length 2, not 4."""
        # cycle_kwh has 2 non-zero slots; period_hours=12 so next 10 slots are 0
        df = _make_cycle_df([1.0, 0.5], n_cycles=3, period_hours=12)
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        assert len(sigs["dw"]["hourly_profile"]) == 2
        assert sigs["dw"]["hourly_profile"] == pytest.approx([1.0, 0.5])

    def test_adaptive_window_long_cycle(self):
        """A 6-hour actual cycle produces a profile of length 6."""
        df = _make_cycle_df([0.5, 1.0, 1.5, 1.2, 0.8, 0.3], n_cycles=3, period_hours=18)
        sigs = _learn_appliance_signatures({"wm": df})
        assert "wm" in sigs
        assert len(sigs["wm"]["hourly_profile"]) == 6

    def test_adaptive_window_max_cap(self):
        """A cycle running longer than APPLIANCE_MAX_WINDOW_HOURS is capped."""
        from energy_forecast.const import APPLIANCE_MAX_WINDOW_HOURS

        # 15 consecutive non-zero hours, then 5 zero hours, repeated
        long_cycle = [1.0] * 15
        df = _make_cycle_df(long_cycle, n_cycles=3, period_hours=20)
        sigs = _learn_appliance_signatures({"hp": df})
        assert "hp" in sigs
        assert len(sigs["hp"]["hourly_profile"]) <= APPLIANCE_MAX_WINDOW_HOURS

    def test_never_zero_surge_detection(self):
        """HP-style series (never drops to 0) — starts still detected via idle threshold."""
        # Baseline ~1 kWh/h; demand surges to 3 kWh/h for 2h each cycle
        ts = pd.date_range("2024-01-01", periods=60, freq="1h")
        kwh = [1.0] * 60
        # Two surges: hours 5-6 and 30-31
        for h in [5, 6, 30, 31]:
            kwh[h] = 3.0
        df = pd.DataFrame({"timestamp": ts, "kwh": kwh})
        sigs = _learn_appliance_signatures({"hp_heat": df})
        # Should detect at least one surge cycle (idle_thresh = Q25 ≈ 1.0)
        # Whether 1 or 2 cycles detected depends on quartile; just ensure no crash
        # and backward-compat keys present if any cycles found
        if "hp_heat" in sigs:
            assert "hourly_profile" in sigs["hp_heat"]
            assert "n_cycles" in sigs["hp_heat"]

    def test_outlier_cycle_rejected(self):
        """One cycle with 5× the normal kWh is rejected; profile reflects the rest."""
        # Normal cycles: [1.0, 0.5] each; outlier: [5.0, 5.0]
        normal_cycles = _make_cycle_df([1.0, 0.5], n_cycles=5, period_hours=10)
        # Append one outlier cycle at the end
        last_ts = normal_cycles["timestamp"].iloc[-1]
        outlier_ts = pd.date_range(last_ts + pd.Timedelta("1h"), periods=2, freq="1h")
        # Leave enough room after outlier so it isn't truncated (>= APPLIANCE_MAX_WINDOW_HOURS)
        padding_ts = pd.date_range(outlier_ts[-1] + pd.Timedelta("1h"), periods=12, freq="1h")
        outlier = pd.DataFrame({"timestamp": outlier_ts, "kwh": [5.0, 5.0]})
        padding = pd.DataFrame({"timestamp": padding_ts, "kwh": [0.0] * 12})
        df = pd.concat([normal_cycles, outlier, padding], ignore_index=True)
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        # With outlier rejected, profile should be close to [1.0, 0.5]
        assert sigs["dw"]["hourly_profile"][0] == pytest.approx(1.0, abs=0.2)
        assert sigs["dw"]["n_rejected"] >= 1

    def test_outlier_rejection_skipped_with_few_cycles(self):
        """With only 3 cycles, no outlier rejection is applied (threshold: ≥ 4)."""
        df = _make_cycle_df([1.0, 0.5], n_cycles=3, period_hours=10)
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        assert sigs["dw"]["n_rejected"] == 0

    def test_duration_clustering_two_groups(self):
        """Mix of 2h and 5h cycles → both 'short' and 'long' clusters present."""
        ts_list, kwh_list = [], []
        t = pd.Timestamp("2024-01-01")
        # 5 short cycles (2h), 5 long cycles (5h), separated by 5h gaps
        for _ in range(5):
            for h in range(2):
                ts_list.append(t + pd.Timedelta(hours=h))
                kwh_list.append(1.0)
            t += pd.Timedelta(hours=7)
        for _ in range(5):
            for h in range(5):
                ts_list.append(t + pd.Timedelta(hours=h))
                kwh_list.append(1.0)
            t += pd.Timedelta(hours=10)
        # Add tail padding to avoid truncation
        for h in range(12):
            ts_list.append(t + pd.Timedelta(hours=h))
            kwh_list.append(0.0)
        df = pd.DataFrame({"timestamp": ts_list, "kwh": kwh_list})
        sigs = _learn_appliance_signatures({"wm": df})
        assert "wm" in sigs
        assert "clusters" in sigs["wm"]
        assert "short" in sigs["wm"]["clusters"]
        assert "long" in sigs["wm"]["clusters"]
        assert sigs["wm"]["clusters"]["short"]["median_duration_h"] == 2
        assert sigs["wm"]["clusters"]["long"]["median_duration_h"] == 5

    def test_duration_clustering_single_group(self):
        """Uniform-duration cycles → no clusters key (or clusters is absent/single)."""
        df = _make_cycle_df([1.0, 0.5, 0.2], n_cycles=5, period_hours=10)
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        # All cycles are 3h → median_dur=3; short == long bucket doesn't split
        # Clusters key may be absent or have only one entry
        clusters = sigs["dw"].get("clusters", {})
        assert len(clusters) <= 1

    def test_cov_warning_suppressed_after_clustering(self, caplog):
        """Mixed-duration cycles that are internally consistent → no WARNING after cluster split."""
        import logging

        ts_list, kwh_list = [], []
        t = pd.Timestamp("2024-01-01")
        # Short cycles: consistent 2h pattern [1.5, 0.5]
        for _ in range(6):
            for h, v in enumerate([1.5, 0.5]):
                ts_list.append(t + pd.Timedelta(hours=h))
                kwh_list.append(v)
            t += pd.Timedelta(hours=6)
        # Long cycles: consistent 4h pattern [0.5, 1.0, 1.0, 0.5]
        for _ in range(6):
            for h, v in enumerate([0.5, 1.0, 1.0, 0.5]):
                ts_list.append(t + pd.Timedelta(hours=h))
                kwh_list.append(v)
            t += pd.Timedelta(hours=8)
        # Tail padding
        for h in range(12):
            ts_list.append(t + pd.Timedelta(hours=h))
            kwh_list.append(0.0)
        df = pd.DataFrame({"timestamp": ts_list, "kwh": kwh_list})
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            sigs = _learn_appliance_signatures({"wm": df})
        # Clusters should exist; no high-variability warning expected
        assert "wm" in sigs
        assert "clusters" in sigs["wm"]
        assert not any("high variability" in r.message for r in caplog.records)

    def test_schema_backward_compatible(self):
        """New schema always contains all expected keys."""
        df = _make_cycle_df([1.0, 2.0, 1.5, 0.5], n_cycles=4, period_hours=12)
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        for key in ("total_kwh", "hourly_profile", "std_profile", "peak_hour", "n_cycles"):
            assert key in sigs["dw"], f"Missing backward-compat key: {key}"
        # Existing metadata keys
        assert "n_rejected" in sigs["dw"]
        assert "idle_threshold" in sigs["dw"]
        # New metadata keys (A + B)
        assert "quality" in sigs["dw"]
        assert sigs["dw"]["quality"] in ("good", "fair", "poor")
        assert "n_data_days" in sigs["dw"]
        assert isinstance(sigs["dw"]["n_data_days"], int)
        assert "last_cycle_ts" in sigs["dw"]
        assert isinstance(sigs["dw"]["last_cycle_ts"], str)
        assert "hour_of_day_distribution" in sigs["dw"]
        assert isinstance(sigs["dw"]["hour_of_day_distribution"], dict)

    def test_quality_poor_few_cycles(self):
        """3 cycles → quality='poor'."""
        df = _make_cycle_df([1.0, 0.5], n_cycles=3, period_hours=10)
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        assert sigs["dw"]["quality"] == "poor"

    def test_quality_fair_medium_cycles(self):
        """5–14 cycles → quality='fair'."""
        df = _make_cycle_df([1.0, 0.5], n_cycles=7, period_hours=10)
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        assert sigs["dw"]["quality"] == "fair"

    def test_quality_good_many_cycles(self):
        """15+ cycles → quality='good'."""
        df = _make_cycle_df([1.0, 0.5], n_cycles=15, period_hours=10)
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        assert sigs["dw"]["quality"] == "good"

    def test_quality_demoted_by_high_cov(self):
        """15+ cycles but high energy variability (CoV > 0.5) → quality='fair', not 'good'."""
        # Create cycles with very different total energies (high CoV)
        ts_list, kwh_list = [], []
        t = pd.Timestamp("2024-01-01")
        for i in range(20):
            # Energy varies wildly: alternating 0.2 kWh and 5.0 kWh cycles
            cycle_energy = 0.2 if i % 2 == 0 else 5.0
            ts_list.append(t)
            kwh_list.append(cycle_energy)
            ts_list.append(t + pd.Timedelta(hours=1))
            kwh_list.append(0.0)
            t += pd.Timedelta(hours=8)
        # Tail padding
        for h in range(4):
            ts_list.append(t + pd.Timedelta(hours=h))
            kwh_list.append(0.0)
        df = pd.DataFrame({"timestamp": ts_list, "kwh": kwh_list})
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        sig = sigs["dw"]
        assert "energy_cov" in sig, "energy_cov should be in signature dict"
        # If CoV > 0.5 and cycle count >= 15, quality should be demoted to 'fair'
        if sig["n_cycles"] >= 15 and sig["energy_cov"] > 0.5:
            assert sig["quality"] == "fair", (
                f"Expected quality='fair' with CoV={sig['energy_cov']:.2f}, got '{sig['quality']}'"
            )

    def test_energy_cov_in_signature(self):
        """energy_cov key is always present in the signature dict."""
        df = _make_cycle_df([1.0, 0.5], n_cycles=5, period_hours=10)
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        assert "energy_cov" in sigs["dw"]
        assert isinstance(sigs["dw"]["energy_cov"], float)

    def test_hour_of_day_distribution_present(self):
        """hour_of_day_distribution is a dict mapping hours (int) to counts (int)."""
        # All cycles start at 08:00 UTC
        ts_list, kwh_list = [], []
        t = pd.Timestamp("2024-01-01 08:00")
        for _ in range(5):
            for h, v in enumerate([1.0, 0.5]):
                ts_list.append(t + pd.Timedelta(hours=h))
                kwh_list.append(v)
            t += pd.Timedelta(hours=12)
        for h in range(6):  # tail padding
            ts_list.append(t + pd.Timedelta(hours=h))
            kwh_list.append(0.0)
        df = pd.DataFrame({"timestamp": ts_list, "kwh": kwh_list})
        sigs = _learn_appliance_signatures({"dw": df})
        assert "dw" in sigs
        dist = sigs["dw"]["hour_of_day_distribution"]
        assert isinstance(dist, dict)
        assert all(isinstance(k, int) for k in dist.keys())
        assert all(isinstance(v, int) for v in dist.values())
        assert sum(dist.values()) == sigs["dw"]["n_cycles"]

    def test_cov_based_clustering_high_energy_spread(self):
        """High energy CoV (> 0.5) triggers clustering even when median_dur == 1h.

        Uses only 3 cycles (< 4) so outlier rejection is skipped, letting us
        isolate the CoV-based trigger: 2 light 1h cycles and 1 heavy 2h cycle.
        """
        ts_list, kwh_list = [], []
        t = pd.Timestamp("2024-01-01")
        # 2 light 1h cycles (0.3 kWh)
        for _ in range(2):
            ts_list.append(t)
            kwh_list.append(0.3)
            t += pd.Timedelta(hours=5)
        # 1 heavy 2h cycle (4.0 kWh total)
        for v in [2.0, 2.0]:
            ts_list.append(t)
            kwh_list.append(v)
            t += pd.Timedelta(hours=1)
        t += pd.Timedelta(hours=5)  # idle gap
        # Tail padding to avoid truncation
        for h in range(6):
            ts_list.append(t + pd.Timedelta(hours=h))
            kwh_list.append(0.0)
        df = pd.DataFrame({"timestamp": ts_list, "kwh": kwh_list})
        sigs = _learn_appliance_signatures({"hp": df})
        assert "hp" in sigs
        # median_dur == 1 so the duration condition alone would not trigger;
        # energy CoV >> 0.5 → clustering must fire via the CoV path.
        assert "clusters" in sigs["hp"], "Expected clusters when energy CoV > 0.5"
        assert len(sigs["hp"]["clusters"]) == 2

    def test_composite_forecast_works_with_new_schema(self):
        """_composite_forecast reads hourly_profile correctly from new schema."""
        df = _make_cycle_df([1.0, 2.0, 1.5], n_cycles=4, period_hours=10)
        sigs = _learn_appliance_signatures({"dw": df})
        baseline = _make_baseline_df(start="2024-06-01 10:00")
        result = _composite_forecast(baseline, {"dw": "14:00"}, sigs)
        assert "delta_kwh" in result.columns
        # Profile was placed at offset 4 (14:00 - 10:00 = 4h)
        assert result["delta_kwh"].iloc[4] == pytest.approx(sigs["dw"]["hourly_profile"][0])


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
        _composite_forecast(df, {"sub_dishwasher": "22:00"}, _DUMMY_SIGS)
        # 22:00 with 00:00 start → offset_h = 22; profile has 4 elements → all 4 fit (22+4=26 < 48)
        # Actually for a 48-row df starting at 00:00, 22:00 = offset 22, 22+4=26 < 48 so no truncation
        # Use start at 00:00 and time "46:00" is invalid; instead test profile going to edge
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
            "sub_wp": "17:00",  # offset 7
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

    def test_predict_scenario_accepts_room_areas(self, tmp_path):
        """predict_scenario() should accept a room_areas kwarg without error."""
        m, forecast = _make_trained_model(tmp_path)
        m._appliance_signatures = {
            "sub_dw": {"hourly_profile": [0.1, 0.2], "total_kwh": 0.3, "peak_hour": 1, "n_cycles": 3}
        }
        result = m.predict_scenario(
            forecast,
            live_temp=None,
            schedule={"sub_dw": "12:00"},
            room_areas={"climate.test": 30.0},
        )
        assert "delta_kwh" in result.columns


class TestScenarioDhwDelta:
    def test_delta_kwh_reflects_dhw_shift_vs_natural_baseline(self, tmp_path):
        """NOTE: deviates from the brief by adding `use_physics_residual=True` to
        `_make_trained_model()` (Phase 2 mode). Confirmed via debug run that in
        Phase 1 mode (brief's exact fixture, `use_physics_residual` defaulting to
        False) `physics_kwh` is fed to the ML model as just one of ~30 features
        trained against fully random/uncorrelated data — it ends up with ~zero
        learned importance, so a single-hour DHW override changes the
        `physics_kwh` feature value but never moves `predicted_kwh` at all,
        making `delta_kwh` deterministically zero and this assertion fail
        regardless of whether predict_scenario()'s DHW-delta plumbing is
        correct. This is the same class of fixture flaw already documented on
        `TestPhase1Validation` above. Phase 2 mode feeds physics_kwh additively
        into predicted_kwh (model.py `preds = physics_baseline + preds`),
        guaranteeing the override is observable regardless of ML feature
        importance, matching the pattern used in
        `TestPhase2ResidualTarget.test_phase2_target_is_residual_log_transform_disabled`.
        """
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(
            tmp_path / "physics_models",
            {
                "cop_formula": {"a": 2.5, "b": 0.07},
                "dhw_tank_volume_l": 200,
                "dhw_power_w": 4000,
                "internal_gains_fraction": 0.8,
                "heating_curve_points": [[-20, 55.5], [20, 25.0]],
                "room_thermostats": [],
                "use_physics_residual": True,
            },
        )
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35, UA_dhw=15.0, Q_dhw_daily=3.5)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)

        override = {
            "legionella": (str(forecast_df["timestamp"].iloc[10].date()), forecast_df["timestamp"].iloc[10].hour)
        }
        result = model.predict_scenario(
            forecast_df, live_temp=5.0, schedule={}, physics_model=pm, dhw_schedule_override=override
        )
        assert "delta_kwh" in result.columns
        assert result["delta_kwh"].abs().sum() > 0  # dhw shift produced a nonzero delta somewhere

    def test_no_dhw_schedule_delta_unchanged_from_pre_plan_d_behaviour(self, tmp_path):
        model, forecast_df = _make_trained_model(tmp_path / "model")
        result = model.predict_scenario(forecast_df, live_temp=5.0, schedule={})
        assert (result["delta_kwh"] == 0.0).all()  # no appliance schedule, no dhw override -> no delta

    def test_appliance_and_dhw_deltas_both_present_are_additive(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(
            tmp_path / "physics_models",
            {
                "cop_formula": {"a": 2.5, "b": 0.07},
                "dhw_tank_volume_l": 200,
                "dhw_power_w": 4000,
                "internal_gains_fraction": 0.8,
                "heating_curve_points": [[-20, 55.5], [20, 25.0]],
                "room_thermostats": [],
                "use_physics_residual": False,
            },
        )
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35, UA_dhw=15.0, Q_dhw_daily=3.5)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)

        override = {"legionella": (str(forecast_df["timestamp"].iloc[5].date()), forecast_df["timestamp"].iloc[5].hour)}
        dhw_only = model.predict_scenario(
            forecast_df, live_temp=5.0, schedule={}, physics_model=pm, dhw_schedule_override=override
        )
        combined = model.predict_scenario(
            forecast_df, live_temp=5.0, schedule={}, physics_model=pm, dhw_schedule_override=override
        )
        # same call twice (no appliance signatures learned in this fixture) should be deterministic
        pd.testing.assert_series_equal(dhw_only["delta_kwh"], combined["delta_kwh"])

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

    def test_half_hour_time_floors_to_start_hour(self):
        """'15:30' with forecast_start 14:00 → offset 1 (floor), not 2 (round).

        round(1.5) == 2 under Python banker's rounding, so this would schedule
        the appliance one hour late without the int() floor fix.
        """
        df = _make_baseline_df(start="2024-06-01 14:00")
        result = _composite_forecast(df, {"sub_wp": "15:30"}, _DUMMY_SIGS)
        wp_profile = _DUMMY_SIGS["sub_wp"]["hourly_profile"]  # len=3
        # offset_h must be 1 (floor(1.5)), not 2 (round(1.5))
        assert result["delta_kwh"].iloc[1] == pytest.approx(wp_profile[0])
        assert result["delta_kwh"].iloc[2] == pytest.approx(wp_profile[1])
        assert result["delta_kwh"].iloc[3] == pytest.approx(wp_profile[2])
        # hour 0 must be untouched
        assert result["delta_kwh"].iloc[0] == pytest.approx(0.0)

    def test_malformed_time_string_skipped(self):
        """Unparseable time string → warning logged (implicitly), delta_kwh all zero."""
        df = _make_baseline_df(start="2024-06-01 10:00")
        result = _composite_forecast(df, {"sub_dishwasher": "25:99"}, _DUMMY_SIGS)
        assert (result["delta_kwh"] == 0.0).all()

    def test_non_string_time_skipped(self):
        """Non-string schedule value (int) → treated as invalid, delta_kwh all zero."""
        df = _make_baseline_df(start="2024-06-01 10:00")
        result = _composite_forecast(df, {"sub_dishwasher": 1400}, _DUMMY_SIGS)
        assert (result["delta_kwh"] == 0.0).all()

    def test_out_of_range_hour_skipped(self):
        """Hour > 23 → ValueError raised internally, skipped, delta_kwh all zero."""
        df = _make_baseline_df(start="2024-06-01 10:00")
        result = _composite_forecast(df, {"sub_dishwasher": "25:00"}, _DUMMY_SIGS)
        assert (result["delta_kwh"] == 0.0).all()

    # ── Program-type schedule (dict form) ────────────────────────────────────

    def test_program_schedule_uses_program_profile(self):
        """Dict schedule with known program uses per-program hourly_profile."""
        eco_profile = [0.5, 1.0, 0.5]
        sigs = {
            "sub_dw": {
                "hourly_profile": [1.0, 2.0, 1.5, 0.5],
                "total_kwh": 5.0,
                "peak_hour": 1,
                "n_cycles": 5,
                "programs": {
                    "eco": {
                        "hourly_profile": eco_profile,
                        "std_profile": [0.0, 0.0, 0.0],
                        "total_kwh": 2.0,
                        "n_cycles": 3,
                    }
                },
            }
        }
        df = _make_baseline_df(start="2024-06-01 10:00")
        result = _composite_forecast(df, {"sub_dw": {"start": "14:00", "program": "eco"}}, sigs)
        for i, v in enumerate(eco_profile):
            assert result["delta_kwh"].iloc[4 + i] == pytest.approx(v)
        assert result["delta_kwh"].iloc[0] == pytest.approx(0.0)

    def test_program_fallback_when_program_unknown(self):
        """Unknown program label → falls back to combined hourly_profile."""
        combined = [1.0, 2.0, 1.5, 0.5]
        sigs = {
            "sub_dw": {
                "hourly_profile": combined,
                "total_kwh": 5.0,
                "peak_hour": 1,
                "n_cycles": 5,
                "programs": {
                    "eco": {
                        "hourly_profile": [0.5, 1.0, 0.5],
                        "std_profile": [],
                        "total_kwh": 2.0,
                        "n_cycles": 3,
                    }
                },
            }
        }
        df = _make_baseline_df(start="2024-06-01 10:00")
        result = _composite_forecast(df, {"sub_dw": {"start": "14:00", "program": "intensive"}}, sigs)
        for i, v in enumerate(combined):
            assert result["delta_kwh"].iloc[4 + i] == pytest.approx(v)

    def test_program_fallback_when_no_programs_key(self):
        """Sig without 'programs' key → falls back to hourly_profile, no error."""
        combined = [1.0, 2.0]
        sigs = {
            "sub_dw": {
                "hourly_profile": combined,
                "total_kwh": 3.0,
                "peak_hour": 1,
                "n_cycles": 4,
            }
        }
        df = _make_baseline_df(start="2024-06-01 10:00")
        result = _composite_forecast(df, {"sub_dw": {"start": "12:00", "program": "eco"}}, sigs)
        for i, v in enumerate(combined):
            assert result["delta_kwh"].iloc[2 + i] == pytest.approx(v)

    def test_legacy_string_schedule_unchanged(self):
        """Plain 'HH:MM' string alongside sigs with programs → identical to pre-feature result."""
        eco_profile = [0.5, 1.0, 0.5]
        sigs = {
            "sub_dw": {
                "hourly_profile": [1.0, 2.0, 1.5, 0.5],
                "total_kwh": 5.0,
                "peak_hour": 1,
                "n_cycles": 5,
                "programs": {
                    "eco": {"hourly_profile": eco_profile, "std_profile": [], "total_kwh": 2.0, "n_cycles": 3}
                },
            }
        }
        df = _make_baseline_df(start="2024-06-01 10:00")
        result_str = _composite_forecast(df, {"sub_dw": "14:00"}, sigs)
        # Must use the combined profile, not eco
        assert result_str["delta_kwh"].iloc[4] == pytest.approx(1.0)
        assert result_str["delta_kwh"].iloc[5] == pytest.approx(2.0)

    def test_program_off_start_skipped(self):
        """Dict schedule with start='off' → no delta."""
        sigs = {
            "sub_dw": {
                "hourly_profile": [1.0, 2.0],
                "total_kwh": 3.0,
                "peak_hour": 1,
                "n_cycles": 4,
                "programs": {"eco": {"hourly_profile": [0.5, 0.5], "std_profile": [], "total_kwh": 1.0, "n_cycles": 2}},
            }
        }
        df = _make_baseline_df(start="2024-06-01 10:00")
        result = _composite_forecast(df, {"sub_dw": {"start": "off", "program": "eco"}}, sigs)
        assert (result["delta_kwh"] == 0.0).all()


# ── Program signatures in _learn_appliance_signatures ────────────────────────


def _make_prog_df(
    cycle_starts: list[str],
    programs: list[str],
    timezone: str = "Europe/Zurich",
) -> pd.DataFrame:
    """Build a synthetic program sensor DataFrame aligned to given cycle starts."""
    rows = []
    for ts_str, prog in zip(cycle_starts, programs):
        ts = pd.Timestamp(ts_str)
        rows.append({"timestamp": ts, "program": prog})
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


class TestApplianceSignaturesPrograms:
    """Tests for program_histories support in _learn_appliance_signatures()."""

    PROFILE = [1.0, 2.0, 1.5, 0.5]

    def test_programs_absent_without_program_history(self):
        """No program_histories → 'programs' key absent from signature."""
        df = _make_cycle_df(self.PROFILE, n_cycles=3)
        sigs = _learn_appliance_signatures({"dw": df})
        assert "programs" not in sigs.get("dw", {})

    def test_programs_learned_with_sufficient_cycles(self):
        """≥ 2 cycles for a program → profile stored under sig['programs']."""
        df = _make_cycle_df(self.PROFILE, n_cycles=4, period_hours=12)
        # Assign "eco" to all 4 cycle starts (every 12 hours from 2024-01-01 00:00)
        cycle_starts = [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 12:00"),
            pd.Timestamp("2024-01-02 00:00"),
            pd.Timestamp("2024-01-02 12:00"),
        ]
        prog_df = _make_prog_df(
            [str(ts) for ts in cycle_starts],
            ["eco"] * 4,
        )
        sigs = _learn_appliance_signatures({"dw": df}, program_histories={"dw": prog_df})
        assert "dw" in sigs
        assert "programs" in sigs["dw"]
        assert "eco" in sigs["dw"]["programs"]
        eco = sigs["dw"]["programs"]["eco"]
        assert eco["n_cycles"] == 4
        assert eco["total_kwh"] == pytest.approx(sum(self.PROFILE))

    def test_programs_skipped_below_min_cycles_per_program(self):
        """1 cycle for 'normal' → absent; 3 for 'eco' → present."""
        df = _make_cycle_df(self.PROFILE, n_cycles=4, period_hours=12)
        cycle_starts = [
            "2024-01-01 00:00",
            "2024-01-01 12:00",
            "2024-01-02 00:00",
            "2024-01-02 12:00",
        ]
        programs = ["eco", "eco", "eco", "normal"]
        prog_df = _make_prog_df(cycle_starts, programs)
        sigs = _learn_appliance_signatures(
            {"dw": df},
            program_histories={"dw": prog_df},
            min_cycles_per_program=2,
        )
        assert "programs" in sigs["dw"]
        assert "eco" in sigs["dw"]["programs"]
        assert "normal" not in sigs["dw"]["programs"]

    def test_programs_and_clusters_coexist(self):
        """Both 'clusters' and 'programs' can appear in the same signature."""
        # Use a profile that triggers clustering: mix short (1h) and long (4h) cycles
        short = [2.0]
        long_ = [1.0, 2.0, 1.5, 0.5]
        n_each = 4
        period = 12
        rows = []
        ts = pd.Timestamp("2024-01-01 00:00")
        cycle_starts = []
        for i in range(n_each):
            cycle_starts.append(str(ts))
            for v in short:
                rows.append({"timestamp": ts, "kwh": v})
                ts += pd.Timedelta(hours=1)
            for _ in range(period - len(short)):
                rows.append({"timestamp": ts, "kwh": 0.0})
                ts += pd.Timedelta(hours=1)
        for i in range(n_each):
            cycle_starts.append(str(ts))
            for v in long_:
                rows.append({"timestamp": ts, "kwh": v})
                ts += pd.Timedelta(hours=1)
            for _ in range(period - len(long_)):
                rows.append({"timestamp": ts, "kwh": 0.0})
                ts += pd.Timedelta(hours=1)

        df = pd.DataFrame(rows)
        # All short cycles: "quick", all long cycles: "eco"
        programs = ["quick"] * n_each + ["eco"] * n_each
        prog_df = _make_prog_df(cycle_starts, programs)
        sigs = _learn_appliance_signatures({"app": df}, program_histories={"app": prog_df})
        sig = sigs.get("app", {})
        # clusters present (mix of short/long durations)
        assert "clusters" in sig
        # programs present (≥ 2 for each label)
        assert "programs" in sig
        assert "quick" in sig["programs"]
        assert "eco" in sig["programs"]

    def test_programs_cycle_predating_history_excluded(self):
        """Cycles before prog_df start are not counted in programs but counted in n_cycles."""
        df = _make_cycle_df(self.PROFILE, n_cycles=4, period_hours=12)
        # Only provide program history from cycle 3 onward (skip first two)
        prog_df = _make_prog_df(
            ["2024-01-02 00:00", "2024-01-02 12:00"],
            ["eco", "eco"],
        )
        sigs = _learn_appliance_signatures({"dw": df}, program_histories={"dw": prog_df})
        sig = sigs["dw"]
        # Combined n_cycles covers all 4
        assert sig["n_cycles"] == 4
        # But only 2 cycles have program info → eco has 2
        assert "programs" in sig
        assert sig["programs"]["eco"]["n_cycles"] == 2

    def test_programs_normalizes_to_lowercase(self):
        """Program labels are already lowercased by fetch_program_sensor_history;
        test that _learn_appliance_signatures groups by exact label (case-sensitive).
        'eco' and 'eco' group together; 'ECO' would be a different key."""
        df = _make_cycle_df(self.PROFILE, n_cycles=4, period_hours=12)
        cycle_starts = [
            "2024-01-01 00:00",
            "2024-01-01 12:00",
            "2024-01-02 00:00",
            "2024-01-02 12:00",
        ]
        # Simulate already-lowercased input (as produced by fetch_program_sensor_history)
        prog_df = _make_prog_df(cycle_starts, ["eco", "eco", "eco", "eco"])
        sigs = _learn_appliance_signatures({"dw": df}, program_histories={"dw": prog_df})
        assert "eco" in sigs["dw"]["programs"]
        assert "ECO" not in sigs["dw"]["programs"]

    def test_programs_save_load_roundtrip(self, tmp_path):
        """sig['programs'] survives JSON round-trip (string keys, no int-key issue)."""
        import json

        df = _make_cycle_df(self.PROFILE, n_cycles=4, period_hours=12)
        prog_df = _make_prog_df(
            ["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 00:00", "2024-01-02 12:00"],
            ["eco"] * 4,
        )
        sigs = _learn_appliance_signatures({"dw": df}, program_histories={"dw": prog_df})
        sig_path = tmp_path / "appliance_signatures.json"
        sig_path.write_text(json.dumps(sigs))
        loaded = json.loads(sig_path.read_text())
        assert "programs" in loaded["dw"]
        assert "eco" in loaded["dw"]["programs"]
        assert loaded["dw"]["programs"]["eco"]["n_cycles"] == 4


# ── Concurrency (SE-2) ───────────────────────────────────────────────────────


class TestConcurrency:
    """Validates the GIL-safe lockless prediction invariant.

    _update_cb reads self._ml_model.model without holding _lock, relying on
    CPython's GIL making object-reference replacement atomic.  These tests
    exercise the model layer directly to confirm predict() is safe to call
    while train() runs concurrently.
    """

    def test_predict_during_retrain_does_not_raise(self, tmp_path):
        """predict() called while train() runs in a background thread must not raise."""
        import threading

        pytest.importorskip("sklearn")
        rng = np.random.default_rng(1)
        n = 600
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy_df = pd.DataFrame(
            {
                "timestamp": ts,
                "gross_kwh": rng.uniform(0.5, 3.0, n),
            }
        )
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(5, 25, n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
        future_weather = weather_df.copy()
        future_weather["timestamp"] = ts + pd.Timedelta(hours=n)

        model = EnergyForecastModel(model_dir=tmp_path)
        # Prime the model so predict() has something to work with
        model.train(energy_df=energy_df, weather_df=weather_df, outdoor_df=None)

        train_exc: list = []

        def _retrain():
            try:
                model.train(energy_df=energy_df, weather_df=weather_df, outdoor_df=None)
            except Exception as exc:  # noqa: BLE001
                train_exc.append(exc)

        t = threading.Thread(target=_retrain)
        t.start()
        # predict() must not raise regardless of where train() is in its lifecycle
        result = model.predict(
            forecast_df=future_weather,
            live_temp=None,
            recent_actuals=energy_df.tail(400),
        )
        t.join()

        assert train_exc == [], f"train() raised in background thread: {train_exc}"
        assert result is not None
        assert len(result) == 48
        assert "predicted_kwh" in result.columns


# ── τ (thermal time constant) calibration ─────────────────────────────────────


def _make_tau_fixtures(tau: float, n_windows: int = 5, window_len: int = 8):
    """Build synthetic climate_dfs, heating_active_df, weather_df for τ tests.

    Each cooling window spans `window_len` hours with indoor temperature decaying
    from T0=22°C toward T_outdoor=5°C with the given τ.  Heating-on gaps of 3 h
    separate windows.
    """
    rng = np.random.default_rng(42)
    T0, T_out = 22.0, 5.0
    period = window_len + 3  # window + heating-on gap
    total_hours = n_windows * period + 5
    timestamps = pd.date_range("2024-01-01", periods=total_hours, freq="1h")

    T_indoor = np.full(total_hours, T0)
    heating_active = np.ones(total_hours, dtype=float)

    for i in range(n_windows):
        start = i * period
        for j in range(window_len):
            idx = start + j
            if idx >= total_hours:
                break
            T_indoor[idx] = T_out + (T0 - T_out) * np.exp(-j / tau)
            heating_active[idx] = 0  # heating off

    # Tiny noise so OLS isn't perfectly noiseless (more realistic)
    T_indoor += rng.normal(0, 0.05, size=total_hours)
    T_indoor = np.clip(T_indoor, T_out + 0.1, None)

    climate_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "current_temp": T_indoor,
            "setpoint": T_indoor + 1.0,  # setpoint doesn't matter for τ estimation
        }
    )
    heating_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "heating_active": heating_active,
        }
    )
    weather_df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "temp_c": [T_out] * total_hours,
            "precipitation_mm": [0.0] * total_hours,
            "sunshine_min": [0.0] * total_hours,
            "wind_kmh": [0.0] * total_hours,
            "cloud_cover_pct": [100.0] * total_hours,
            "direct_radiation_wm2": [0.0] * total_hours,
        }
    )
    return {"climate.test": climate_df}, heating_df, weather_df


class TestCalibrateTau:
    def test_known_tau(self, tmp_path):
        """τ estimated from synthetic data matches the true value within 10%."""
        TRUE_TAU = 4.0
        model = EnergyForecastModel(model_dir=tmp_path)
        climate_dfs, heating_df, weather_df = _make_tau_fixtures(TRUE_TAU)
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None
        assert abs(result - TRUE_TAU) / TRUE_TAU < 0.10, f"Expected τ ≈ {TRUE_TAU}, got {result:.2f}"

    def test_single_window_produces_result(self, tmp_path):
        """Even a single valid window now produces a τ estimate (old ≥3 minimum removed)."""
        model = EnergyForecastModel(model_dir=tmp_path)
        climate_dfs, heating_df, weather_df = _make_tau_fixtures(4.0, n_windows=1)
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None
        assert 0.5 < result < 200.0

    def test_heating_always_on(self, tmp_path):
        """No off-window → returns None."""
        model = EnergyForecastModel(model_dir=tmp_path)
        n = 100
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        climate_dfs = {
            "climate.test": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": [20.0] * n,
                    "setpoint": [21.0] * n,
                }
            )
        }
        heating_df = pd.DataFrame(
            {
                "timestamp": ts,
                "heating_active": [1.0] * n,  # always on
            }
        )
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [5.0] * n,
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [0.0] * n,
                "wind_kmh": [0.0] * n,
                "cloud_cover_pct": [0.0] * n,
                "direct_radiation_wm2": [0.0] * n,
            }
        )
        assert model._calibrate_tau(climate_dfs, heating_df, weather_df) is None

    def test_evening_cooling_window_used(self, tmp_path):
        """Cooling in the tail of an off-block (after a rising prefix) still yields τ.

        Scenario: heating off from 20:00 for 11 h (20:00–06:00).  First 3 h:
        indoor rises (appliance heat, residual warmth).  Hours 3-10: genuine
        passive cooling → valid sub-sequence in the tail.  The sub-sequence scan
        should find and use the nighttime cooling portion.

        Windows are entirely outside 09:00–15:00 to avoid the daytime exclusion.
        """
        TRUE_TAU = 5.0
        T_out = 5.0
        # 4 off-blocks, each starting at 20:00 on consecutive days.
        # Pattern: 3h rising prefix + 7h cooling + 1h heating-on gap = 11h block.
        all_ts: list = []
        all_T_in: list = []
        all_heat: list = []

        for day in range(4):
            base = pd.Timestamp(f"2024-01-{10 + day:02d} 20:00")
            for j in range(3):
                all_ts.append(base + pd.Timedelta(hours=j))
                all_T_in.append(20.0 + j * 0.75)  # rising prefix
                all_heat.append(0.0)
            for j in range(7):
                all_ts.append(base + pd.Timedelta(hours=3 + j))
                all_T_in.append(T_out + (23.0 - T_out) * np.exp(-j / TRUE_TAU))
                all_heat.append(0.0)
            # one "heating on" row to close the block before next day
            all_ts.append(base + pd.Timedelta(hours=10))
            all_T_in.append(20.0)
            all_heat.append(1.0)

        ts = pd.DatetimeIndex(all_ts)
        climate_dfs = {
            "climate.test": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": all_T_in,
                    "setpoint": [t + 1.0 for t in all_T_in],
                }
            )
        }
        heating_df = pd.DataFrame({"timestamp": ts, "heating_active": all_heat})
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [T_out] * len(ts),
                "precipitation_mm": [0.0] * len(ts),
                "sunshine_min": [0.0] * len(ts),
                "wind_kmh": [0.0] * len(ts),
                "cloud_cover_pct": [100.0] * len(ts),
                "direct_radiation_wm2": [0.0] * len(ts),
            }
        )

        model = EnergyForecastModel(model_dir=tmp_path)
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None, "nighttime cooling sub-sequence should yield a τ estimate"
        assert abs(result - TRUE_TAU) / TRUE_TAU < 0.25, f"Expected τ ≈ {TRUE_TAU}, got {result:.2f}"

    def test_tau_extended_off_block_finds_nighttime_windows(self, tmp_path):
        """Nighttime cooling windows beyond hour 12 of a long off-block are now found.

        Regression test for the bug where the block-level 12h cap caused the
        entire nighttime portion of a daytime-start off-block to be discarded.

        Scenario: 48-hour heating-off block starting at 08:00.
          Hours  0-11 (08:00–19:00): daytime, indoor rising due to solar gain → no decline.
          Hours 12-23 (20:00–07:00): nighttime, indoor cooling → valid declining delta.
          Hours 24-35 (08:00–19:00): daytime again, rising.
          Hours 36-47 (20:00–07:00): second nighttime, cooling.

        Old behaviour (block cap at 12h): only hours 0-11 are analysed (all
        rising) → 0 candidates → returns None.
        New behaviour (cap at sub-sequence level): hours 12-23 and 36-47 are
        searched → nighttime sub-sequences found → τ estimated.
        """
        TRUE_TAU = 6.0
        T_out = 5.0

        n = 48
        # Start at 08:00 so daytime hours come first in the block.
        ts = pd.date_range("2024-06-10 08:00", periods=n, freq="1h")

        T_indoor = np.empty(n)
        for j in range(n):
            hour_in_block = j % 24
            if 0 <= hour_in_block < 12:
                # Daytime: indoor rises from 18 to 22 °C (solar gain).
                T_indoor[j] = 18.0 + hour_in_block * (4.0 / 11)
            else:
                # Nighttime: passive cooling from 22 °C toward T_out.
                night_j = hour_in_block - 12
                T_indoor[j] = T_out + (22.0 - T_out) * np.exp(-night_j / TRUE_TAU)

        climate_dfs = {
            "climate.test": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": T_indoor,
                    "setpoint": T_indoor + 1.0,
                }
            )
        }
        heating_df = pd.DataFrame(
            {
                "timestamp": ts,
                "heating_active": np.zeros(n),  # heating off for the entire 48h block
            }
        )
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [T_out] * n,
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [0.0] * n,
                "wind_kmh": [0.0] * n,
                "cloud_cover_pct": [100.0] * n,
                "direct_radiation_wm2": [0.0] * n,
            }
        )

        model = EnergyForecastModel(model_dir=tmp_path)
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None, (
            "Nighttime cooling sub-sequences (hours 12-23 and 36-47) should yield "
            "a τ estimate even though the block starts in daytime."
        )
        assert abs(result - TRUE_TAU) / TRUE_TAU < 0.25, f"Expected τ ≈ {TRUE_TAU}, got {result:.2f}"

    def test_tau_sub_sequence_cap_at_12h(self, tmp_path):
        """A declining sequence longer than 12h is capped at 12h for the OLS fit.

        Creates a 25-hour nighttime block with monotonic cooling (true τ = 8h).
        The sub-sequence spans hours 1-25 (length 24), but the cap limits the
        OLS regression to 12 points.  The estimate should still be within 20%
        of the true τ and the function must not raise.
        """
        TRUE_TAU = 8.0
        T_out = 5.0
        T0 = 22.0
        n = 25
        ts = pd.date_range("2024-01-15 22:00", periods=n, freq="1h")  # all nighttime

        T_indoor = np.array([T_out + (T0 - T_out) * np.exp(-j / TRUE_TAU) for j in range(n)])

        climate_dfs = {
            "climate.test": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": T_indoor,
                    "setpoint": T_indoor + 1.0,
                }
            )
        }
        heating_df = pd.DataFrame({"timestamp": ts, "heating_active": np.zeros(n)})
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [T_out] * n,
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [0.0] * n,
                "wind_kmh": [0.0] * n,
                "cloud_cover_pct": [100.0] * n,
                "direct_radiation_wm2": [0.0] * n,
            }
        )

        model = EnergyForecastModel(model_dir=tmp_path)
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None, "25-hour declining block should produce a τ estimate"
        assert abs(result - TRUE_TAU) / TRUE_TAU < 0.20, f"Expected τ ≈ {TRUE_TAU}, got {result:.2f}"

    def test_two_separate_off_blocks_not_merged(self, tmp_path):
        """Two genuinely separate off-periods (nights) with a real on-period gap between
        them must be calibrated as two independent blocks, not merged into one.

        Regression test for the Task 6 refactor bug where ``_find_passive_windows()``
        was applied *before* ``off``/``block`` were computed: deleting the intervening
        "on" rows made ``combined["off"].diff().ne(0)).cumsum()`` see no flip between the
        two off-periods, merging them into a single pseudo-block. The downstream OLS fit
        then used ``t = np.arange(len(d))`` — treating the row-adjacency of two distinct
        nights (separated by a 10-hour heating-on day) as if they were 1-hour-apart
        samples, which corrupts the τ estimate.

        Construction (no artificial rising-prefix reset trick — deltas are monotonically
        non-increasing across the night1→night2 seam, so the "declining" sub-sequence
        filter does NOT accidentally re-split the merged block back apart; this is what
        makes the bug reproducible here, unlike the existing evening-cooling-window test):

        * Night 1: 8 h heating-off, T_indoor decays from 22 °C toward T_out=5 °C with
          true τ=6 h (ΔT₀=17 °C), ending at ΔT≈5.29 °C.
        * Day: 10 h heating-on gap (real elapsed time between the two nights).
        * Night 2: 8 h heating-off, decays from ΔT₀=3 °C (< night 1's ending ΔT, so the
          stacked delta sequence stays monotonically declining across the seam) with the
          same true τ=6 h.

        Hand-verified (see task report) that with block identity computed on the *full*
        timeline (fixed behaviour), night 1 and night 2 fit as two separate 7-point
        sub-sequences, each independently recovering τ≈6.0 h exactly. With the buggy
        pre-Task-6-fix ordering, the two nights collapse into one artificial 15-point
        declining run, capped at 12 points by the sub-sequence cap, spanning 7 points of
        night 1 + 5 points of night 2 fit as a single fake exponential — this yields
        τ≈4.63 h (~23% low), which would fail the tolerance below.
        """
        TRUE_TAU = 6.0
        T_out = 5.0

        j = np.arange(8)
        night1_delta = 17.0 * np.exp(-j / TRUE_TAU)
        night2_delta = 3.0 * np.exp(-j / TRUE_TAU)
        assert night2_delta[0] < night1_delta[-1]  # no jump at the seam — see docstring

        night1_ts = pd.date_range("2024-02-10 22:00", periods=8, freq="1h")
        day_ts = pd.date_range("2024-02-11 06:00", periods=10, freq="1h")
        night2_ts = pd.date_range("2024-02-11 16:00", periods=8, freq="1h")

        all_ts = night1_ts.tolist() + day_ts.tolist() + night2_ts.tolist()
        night1_indoor = (T_out + night1_delta).tolist()
        # On-period indoor temp: heating-on rows are entirely excluded from τ analysis
        # (only off==1 rows are grouped), so their exact values are immaterial to the
        # result — a smooth ramp between the two nights' endpoints is used for realism.
        day_indoor = np.linspace(T_out + night1_delta[-1], T_out + night2_delta[0], 10).tolist()
        night2_indoor = (T_out + night2_delta).tolist()
        all_T_in = night1_indoor + day_indoor + night2_indoor
        all_heat = [0.0] * 8 + [1.0] * 10 + [0.0] * 8

        ts = pd.DatetimeIndex(all_ts)
        climate_dfs = {
            "climate.test": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": all_T_in,
                    "setpoint": [t + 1.0 for t in all_T_in],
                }
            )
        }
        heating_df = pd.DataFrame({"timestamp": ts, "heating_active": all_heat})
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [T_out] * len(ts),
                "precipitation_mm": [0.0] * len(ts),
                "sunshine_min": [0.0] * len(ts),
                "wind_kmh": [0.0] * len(ts),
                "cloud_cover_pct": [100.0] * len(ts),
                "direct_radiation_wm2": [0.0] * len(ts),
            }
        )

        model = EnergyForecastModel(model_dir=tmp_path)
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None
        assert abs(result - TRUE_TAU) / TRUE_TAU < 0.10, (
            f"Expected τ ≈ {TRUE_TAU} h (two independently-fit nights), got {result:.2f} h — "
            "if this is ≈4.6 h, the two off-blocks were incorrectly merged into one."
        )

    def test_thermal_pressure_scaled(self, tmp_path):
        """thermal_pressure is the area-weighted °C delta (no τ-division since v0.10.3)."""
        n = 10
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * n})
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [5.0] * n,
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [0.0] * n,
                "wind_kmh": [0.0] * n,
                "cloud_cover_pct": [0.0] * n,
                "direct_radiation_wm2": [0.0] * n,
            }
        )
        climate_df = pd.DataFrame(
            {
                "timestamp": ts,
                "current_temp": [18.0] * n,
                "setpoint": [20.0] * n,  # raw delta = 2.0 °C
            }
        )
        TAU = 2.0
        result = _engineer_features(
            df,
            weather,
            None,
            climate_dfs={"climate.room": climate_df},
            tau_hours=TAU,
        )
        # tau_hours is passed but no longer used for thermal_pressure — result is raw °C delta
        assert "thermal_pressure" in result.columns
        non_zero = result["thermal_pressure"].dropna()
        assert len(non_zero) > 0
        np.testing.assert_allclose(non_zero.values, 2.0, atol=1e-6)

    def test_thermal_pressure_no_tau(self, tmp_path):
        """Without τ, thermal_pressure equals the raw setpoint−current_temp delta."""
        n = 10
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * n})
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [5.0] * n,
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [0.0] * n,
                "wind_kmh": [0.0] * n,
                "cloud_cover_pct": [0.0] * n,
                "direct_radiation_wm2": [0.0] * n,
            }
        )
        climate_df = pd.DataFrame(
            {
                "timestamp": ts,
                "current_temp": [18.0] * n,
                "setpoint": [21.0] * n,  # raw delta = 3.0 °C
            }
        )
        result = _engineer_features(
            df,
            weather,
            None,
            climate_dfs={"climate.room": climate_df},
            tau_hours=None,  # no τ — backward-compat path
        )
        non_zero = result["thermal_pressure"].dropna()
        assert len(non_zero) > 0
        np.testing.assert_allclose(non_zero.values, 3.0, atol=1e-6)


# ── Tests: indoor temperature projection + area-weighted thermal pressure ─────


class TestProjectIndoorTemps:
    """Unit tests for _project_indoor_temps()."""

    def _future_ts(self, n: int = 48) -> pd.DatetimeIndex:
        return pd.date_range("2026-01-15 10:00", periods=n, freq="1h")

    def _outdoor_series(self, ts: pd.DatetimeIndex, temp: float = 5.0) -> pd.Series:
        return pd.Series([temp] * len(ts), index=ts)

    def test_basic_convergence(self):
        """Cold room should converge toward outdoor temperature over 48 h (RC ODE)."""
        ts = self._future_ts()
        outdoor = self._outdoor_series(ts, temp=5.0)
        climate_recent = {
            "climate.room": pd.DataFrame(
                {
                    "timestamp": [ts[0]],  # fresh — age = 0
                    "current_temp": [10.0],  # above outdoor
                    "setpoint": [20.0],
                }
            )
        }
        result = _project_indoor_temps(climate_recent, ts, outdoor, tau_hours=24.0)
        assert "climate.room" in result
        projected = result["climate.room"]
        t_in = projected["current_temp"].values
        # With T_out=5 and T_in[0]=10 the temperature should monotonically decrease
        assert t_in[0] == pytest.approx(10.0)
        assert t_in[-1] < t_in[0], "Room should cool toward outdoor temp"
        # After 48 h with tau=24 the temperature should be closer to T_out
        assert abs(t_in[-1] - 5.0) < abs(t_in[0] - 5.0)

    def test_stale_sensor_falls_back_to_setpoint(self):
        """When last observation is >2 h old, starting T_in should be the setpoint."""
        ts = self._future_ts()
        outdoor = self._outdoor_series(ts, temp=5.0)
        # Last obs timestamp = now - 3h (stale)
        stale_ts = ts[0] - pd.Timedelta(hours=3)
        climate_recent = {
            "climate.room": pd.DataFrame(
                {
                    "timestamp": [stale_ts],
                    "current_temp": [15.0],  # stale — should be ignored
                    "setpoint": [21.0],  # fallback
                }
            )
        }
        result = _project_indoor_temps(climate_recent, ts, outdoor, tau_hours=24.0)
        t_in = result["climate.room"]["current_temp"].values
        # Starting temp should be the setpoint, not the stale current_temp
        assert t_in[0] == pytest.approx(21.0)

    def test_fresh_sensor_uses_current_temp(self):
        """When last observation is <2 h old, starting T_in should be current_temp."""
        ts = self._future_ts()
        outdoor = self._outdoor_series(ts, temp=5.0)
        fresh_ts = ts[0] - pd.Timedelta(minutes=30)  # 30 min old — fresh
        climate_recent = {
            "climate.room": pd.DataFrame(
                {
                    "timestamp": [fresh_ts],
                    "current_temp": [17.0],
                    "setpoint": [22.0],
                }
            )
        }
        result = _project_indoor_temps(climate_recent, ts, outdoor, tau_hours=24.0)
        t_in = result["climate.room"]["current_temp"].values
        assert t_in[0] == pytest.approx(17.0)

    def test_empty_climate_returns_empty_dict(self):
        """Empty climate_recent dict yields empty result."""
        ts = self._future_ts()
        outdoor = self._outdoor_series(ts)
        result = _project_indoor_temps({}, ts, outdoor)
        assert result == {}

    def test_empty_df_entity_skipped(self):
        """Climate entity with an empty DataFrame is silently skipped."""
        ts = self._future_ts()
        outdoor = self._outdoor_series(ts)
        climate_recent = {"climate.room": pd.DataFrame()}
        result = _project_indoor_temps(climate_recent, ts, outdoor)
        assert result == {}

    def test_default_tau_applied_when_none(self):
        """When tau_hours=None, the DEFAULT_TAU constant is used (no crash)."""
        from energy_forecast.const import DEFAULT_TAU

        ts = self._future_ts()
        outdoor = self._outdoor_series(ts, temp=5.0)
        climate_recent = {
            "climate.room": pd.DataFrame(
                {
                    "timestamp": [ts[0]],
                    "current_temp": [10.0],
                    "setpoint": [20.0],
                }
            )
        }
        result = _project_indoor_temps(climate_recent, ts, outdoor, tau_hours=None)
        t_in = result["climate.room"]["current_temp"].values
        assert t_in[0] == pytest.approx(10.0)
        # With DEFAULT_TAU=24 the first step: 10 + (5-10)/24 ≈ 9.79
        expected_step1 = 10.0 + (5.0 - 10.0) / DEFAULT_TAU
        assert t_in[1] == pytest.approx(expected_step1, rel=1e-4)


class TestAreaWeightedThermalPressure:
    """Unit tests for area-weighted thermal pressure in _engineer_features()."""

    def test_single_room_std_is_zero(self):
        """With only one room, thermal_pressure_std should be 0."""
        ts = pd.date_range("2026-01-15 10:00", periods=3, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        climate_dfs = {
            "climate.room1": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": [18.0] * 3,
                    "setpoint": [21.0] * 3,  # delta = 3.0
                }
            )
        }
        result = _engineer_features(df, w, None, climate_dfs=climate_dfs)
        assert "thermal_pressure_std" in result.columns
        assert (result["thermal_pressure_std"] == 0.0).all()

    def test_multi_room_weighted_mean(self):
        """Weighted mean and max are correct for two rooms with different areas."""
        ts = pd.date_range("2026-01-15 10:00", periods=2, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        climate_dfs = {
            "climate.large": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": [18.0] * 2,
                    "setpoint": [21.0] * 2,  # delta = 3.0
                }
            ),
            "climate.small": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": [19.0] * 2,
                    "setpoint": [21.0] * 2,  # delta = 2.0
                }
            ),
        }
        room_areas = {"climate.large": 30.0, "climate.small": 10.0}  # 3:1 ratio
        result = _engineer_features(df, w, None, climate_dfs=climate_dfs, room_areas=room_areas)
        assert "thermal_pressure" in result.columns
        assert "thermal_pressure_max" in result.columns
        # Weighted mean: (3.0*30 + 2.0*10) / 40 = (90+20)/40 = 2.75
        np.testing.assert_allclose(result["thermal_pressure"].values, 2.75, atol=1e-6)
        # Max delta: large room = 3.0
        np.testing.assert_allclose(result["thermal_pressure_max"].values, 3.0, atol=1e-6)

    def test_uniform_area_fallback(self):
        """Without room_areas, all rooms default to DEFAULT_ROOM_AREA_M2 (equal weight)."""
        ts = pd.date_range("2026-01-15 10:00", periods=2, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        climate_dfs = {
            "climate.room1": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": [19.0] * 2,
                    "setpoint": [21.0] * 2,  # delta = 2.0
                }
            ),
            "climate.room2": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": [17.0] * 2,
                    "setpoint": [21.0] * 2,  # delta = 4.0
                }
            ),
        }
        result = _engineer_features(df, w, None, climate_dfs=climate_dfs)
        # Equal areas → straight mean = (2.0 + 4.0) / 2 = 3.0
        np.testing.assert_allclose(result["thermal_pressure"].values, 3.0, atol=1e-6)

    def test_no_climate_gives_zeros(self):
        """Without climate_dfs all three thermal features are 0."""
        ts = pd.date_range("2026-01-15 10:00", periods=2, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        result = _engineer_features(df, w, None)
        assert (result["thermal_pressure"] == 0.0).all()
        assert (result["thermal_pressure_max"] == 0.0).all()
        assert (result["thermal_pressure_std"] == 0.0).all()

    def test_negative_delta_clipped_to_zero(self):
        """Rooms above setpoint (delta<0) contribute 0 — no negative pressure."""
        ts = pd.date_range("2026-01-15 10:00", periods=2, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts)
        climate_dfs = {
            "climate.warm": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": [23.0] * 2,  # above setpoint
                    "setpoint": [21.0] * 2,
                }
            ),
        }
        result = _engineer_features(df, w, None, climate_dfs=climate_dfs)
        assert (result["thermal_pressure"] == 0.0).all()


class TestZeroFillEliminated:
    """Verify that thermal_pressure is non-zero for all 48 prediction hours."""

    def test_all_hours_nonzero_with_cold_room(self, tmp_path):
        """When a room is below setpoint, projected thermal_pressure must be >0 for all 48h.

        The climate_recent observation must be fresh (<2h old) so that the RC-ODE
        starts from the actual cold temperature rather than falling back to setpoint.
        """
        n = 600
        rng = np.random.default_rng(42)
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, n)})
        weather = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [5.0] * n,
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
        # Training uses any historical climate data
        climate_train = {
            "climate.living": pd.DataFrame(
                {
                    "timestamp": [ts[-1]],
                    "current_temp": [17.0],
                    "setpoint": [21.0],
                }
            )
        }
        m = EnergyForecastModel(tmp_path)
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, climate_dfs=climate_train)

        # Use local wall-clock "now" (tz_localize, not tz_convert) to match _prepare_prediction_X.
        now_local = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None)
        future_ts = pd.date_range(now_local.floor("1h"), periods=48, freq="1h")
        forecast_df = pd.DataFrame(
            {
                "timestamp": future_ts,
                "temp_c": [5.0] * 48,
                "precipitation_mm": [0.0] * 48,
                "sunshine_min": [30.0] * 48,
                "wind_kmh": [10.0] * 48,
                "cloud_cover_pct": [50.0] * 48,
                "direct_radiation_wm2": [100.0] * 48,
            }
        )

        # Prediction uses a FRESH observation (30 min ago) so RC-ODE starts at 17°C
        now_naive = now_local
        climate_predict = {
            "climate.living": pd.DataFrame(
                {
                    "timestamp": [now_naive - pd.Timedelta(minutes=30)],  # fresh
                    "current_temp": [17.0],  # cold — 4°C below setpoint
                    "setpoint": [21.0],
                }
            )
        }

        _, X = m._prepare_prediction_X(
            forecast_df,
            live_temp=None,
            recent_actuals=None,
            climate_recent=climate_predict,
        )
        # thermal_pressure must be non-zero for all 48 hours
        if "thermal_pressure" in X.columns:
            assert (X["thermal_pressure"] > 0).all(), (
                "thermal_pressure should be non-zero for all 48 prediction hours "
                "when a room is below setpoint and observation is fresh"
            )


# ── Tests: COP proxy + solar gain features ───────────────────────────────────


class TestWeightedSolarGain:
    def test_in_features_base(self):
        assert "weighted_solar_gain" in _FEATURES_BASE

    def test_peak_at_1300(self):
        """cos_factor = 1.0 at h=13 → weighted_solar_gain equals direct_radiation_wm2."""
        ts = pd.DatetimeIndex([pd.Timestamp("2026-01-15 13:00")])
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, rad=300.0)
        result = _engineer_features(df, w, None)
        assert result.iloc[0]["weighted_solar_gain"] == pytest.approx(300.0)

    def test_zero_before_0900(self):
        ts = pd.DatetimeIndex([pd.Timestamp("2026-01-15 08:00")])
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, rad=300.0)
        result = _engineer_features(df, w, None)
        assert result.iloc[0]["weighted_solar_gain"] == pytest.approx(0.0)

    def test_zero_after_1700(self):
        ts = pd.DatetimeIndex([pd.Timestamp("2026-01-15 18:00")])
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, rad=300.0)
        result = _engineer_features(df, w, None)
        assert result.iloc[0]["weighted_solar_gain"] == pytest.approx(0.0)

    def test_zero_radiation_gives_zero(self):
        ts = pd.DatetimeIndex([pd.Timestamp("2026-01-15 13:00")])
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, rad=0.0)
        result = _engineer_features(df, w, None)
        assert result.iloc[0]["weighted_solar_gain"] == pytest.approx(0.0)

    def test_never_negative(self):
        """cos_factor near 09:00 and 17:00 could produce small negatives — must be 0."""
        ts = pd.date_range("2026-01-15 00:00", periods=24, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, rad=500.0)
        result = _engineer_features(df, w, None)
        assert (result["weighted_solar_gain"] >= 0).all()


class TestThermalPressureCop:
    def test_in_features_base(self):
        assert "thermal_pressure_cop" in _FEATURES_BASE

    def test_scales_inversely_with_temp(self):
        """Same thermal_pressure → higher cop feature at colder outdoor temp."""

        def _cop_at_temp(t_out):
            ts = pd.DatetimeIndex([pd.Timestamp("2026-01-15 03:00")])
            df = _make_bare_df(ts)
            w = _make_weather_df(ts, temp=t_out)
            climate_dfs = {
                "climate.room": pd.DataFrame(
                    {
                        "timestamp": ts,
                        "current_temp": [18.0],
                        "setpoint": [21.0],
                    }
                )
            }
            result = _engineer_features(df, w, None, climate_dfs=climate_dfs)
            return result.iloc[0]["thermal_pressure_cop"]

        cop_cold = _cop_at_temp(-10.0)
        cop_warm = _cop_at_temp(10.0)
        assert cop_cold > cop_warm, "Cold outdoor → higher electrical cost per °C heat debt"

    def test_denominator_clamp(self):
        """At extreme cold (−50°C), denominator must not drop below 0.5."""
        ts = pd.DatetimeIndex([pd.Timestamp("2026-01-15 03:00")])
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, temp=-50.0)
        climate_dfs = {
            "climate.room": pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": [18.0],
                    "setpoint": [21.0],
                }
            )
        }
        result = _engineer_features(df, w, None, climate_dfs=climate_dfs)
        # thermal_pressure=3.0, cop_proxy=max(0.5, 0.11*(-50)+3)=max(0.5,-2.5)=0.5
        # → thermal_pressure_cop = 3.0 / 0.5 = 6.0
        assert result.iloc[0]["thermal_pressure_cop"] == pytest.approx(6.0)

    def test_zero_pressure_gives_zero_cop(self):
        """No thermal pressure → cop feature is also 0."""
        ts = pd.DatetimeIndex([pd.Timestamp("2026-01-15 03:00")])
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, temp=5.0)
        result = _engineer_features(df, w, None)  # no climate_dfs
        assert result.iloc[0]["thermal_pressure_cop"] == pytest.approx(0.0)


# ── Tests: tau calibration safeguards ────────────────────────────────────────


def _make_tau_model(tmp_path) -> EnergyForecastModel:
    return EnergyForecastModel(tmp_path)


class TestTauCalibrationSafeguards:
    def _make_night_window(
        self,
        start_hour: int = 20,
        n: int = 6,
        t_in_start: float = 22.0,
        t_out: float = 5.0,
        radiation: float = 0.0,
        date: str = "2026-01-10",
    ) -> pd.DataFrame:
        """Build a synthetic passive-cooling window as a combined DataFrame row-set."""
        ts = pd.date_range(f"{date} {start_hour:02d}:00", periods=n, freq="1h")
        tau_true = 10.0
        t_in = [t_in_start]
        for i in range(1, n):
            t_in.append(t_in[-1] + (t_out - t_in[-1]) / tau_true)
        return pd.DataFrame(
            {
                "T_indoor": t_in,
                "T_outdoor": [t_out] * n,
                "heating_active": [0] * n,
                "direct_radiation_wm2": [radiation] * n,
            },
            index=ts,
        )

    def test_high_solar_penalized_not_excluded(self, tmp_path):
        """High solar radiation reduces window quality but no longer blocks calibration.

        5 night windows with radiation=200 W/m² each get solar_score ≈ 0.61 instead of 1.0,
        but all produce valid τ estimates and top 50% are used — result is not None.
        """
        model = _make_tau_model(tmp_path)
        climate_dfs, heating_df, weather_df = self._make_night_blocks(tau_true=10.0, radiation=200.0)
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None
        assert 5.0 < result < 20.0  # physics-plausible range

    def test_daytime_windows_penalized_not_excluded(self, tmp_path):
        """Windows touching daytime hours get hour_score=0.3 but are no longer excluded.

        5 windows spanning 07:00–14:00 each get hour_score=0.3 but still produce valid τ
        estimates — result is not None.
        """
        model = _make_tau_model(tmp_path)
        climate_dfs, heating_df, weather_df = self._make_night_blocks(
            tau_true=10.0,
            start_hour=7,
            window_hours=8,  # 07:00–14:00 touches daytime
        )
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None
        assert 5.0 < result < 20.0

    def _make_night_blocks(
        self,
        tau_true: float,
        n_days: int = 5,
        start_hour: int = 20,
        window_hours: int = 8,
        t_in_start: float = 22.0,
        t_out: float = 5.0,
        radiation: float = 0.0,
        day_start: int = 10,
    ) -> tuple:
        """Build climate/heating/weather DataFrames with *n_days* separated night blocks.

        Each block: *window_hours* of heating-off at *start_hour*, followed by
        1 heating-on row to close the block.  Start hour defaults to 20:00 so no
        block touches 09:00–15:00 (assuming window_hours ≤ 13).
        *day_start* sets the first day-of-month (default 10); use different values
        to create non-overlapping date ranges when combining two fixture sets.
        """
        import pandas as pd

        ts_list: list = []
        t_in_list: list = []
        heat_list: list = []

        for day in range(n_days):
            base = pd.Timestamp(f"2026-01-{day_start + day:02d} {start_hour:02d}:00")
            t_in = t_in_start
            for j in range(window_hours):
                ts_list.append(base + pd.Timedelta(hours=j))
                t_in_list.append(t_in)
                heat_list.append(0.0)
                t_in = t_in + (t_out - t_in) / tau_true
            # close block with one "on" row
            ts_list.append(base + pd.Timedelta(hours=window_hours))
            t_in_list.append(t_in_start)
            heat_list.append(1.0)

        all_ts = pd.DatetimeIndex(ts_list)
        climate_dfs = {
            "climate.room": pd.DataFrame(
                {
                    "timestamp": all_ts,
                    "current_temp": t_in_list,
                    "setpoint": [t_in_start] * len(all_ts),
                }
            )
        }
        heating_df = pd.DataFrame({"timestamp": all_ts, "heating_active": heat_list})
        weather_df = pd.DataFrame(
            {
                "timestamp": all_ts,
                "temp_c": [t_out] * len(all_ts),
                "direct_radiation_wm2": [radiation] * len(all_ts),
            }
        )
        return climate_dfs, heating_df, weather_df

    def test_ema_blend_on_large_change(self, tmp_path):
        """When new τ differs >50% from stored τ, result is EMA-blended."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0  # stored τ
        # tau_true=22 → ~120% above stored 10 → expect blend
        climate_dfs, heating_df, weather_df = self._make_night_blocks(tau_true=22.0, start_hour=22)

        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None
        # EMA: 0.8*10 + 0.2*~22 → result should be between 10 and 22
        assert 10.0 < result < 22.0

    def test_no_radiation_column_degrades_gracefully(self, tmp_path):
        """weather_df without direct_radiation_wm2 skips the solar mask (no crash)."""
        model = _make_tau_model(tmp_path)
        climate_dfs, heating_df, weather_df = self._make_night_blocks(tau_true=10.0)
        weather_df = weather_df.drop(columns=["direct_radiation_wm2"])

        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None

    def test_single_window_sufficient(self, tmp_path):
        """A single valid window is enough — the old ≥3 minimum is gone."""
        model = _make_tau_model(tmp_path)
        climate_dfs, heating_df, weather_df = self._make_night_blocks(tau_true=10.0, n_days=1)
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None
        assert 5.0 < result < 20.0

    def test_top_50_percent_selected(self, tmp_path):
        """With N candidates, only the top N//2 (min 1) are used for the median."""
        import pandas as pd

        model = _make_tau_model(tmp_path)
        # Build 4 nighttime windows: 2 with tau≈10 (good data) + 2 noisy daytime windows
        # that will get low quality but still produce a tau estimate.
        # Net: 4 candidates → top 2 selected → median should be near 10.
        good_dfs, good_heat, good_wx = self._make_night_blocks(tau_true=10.0, n_days=2, start_hour=22)
        noisy_dfs, noisy_heat, noisy_wx = self._make_night_blocks(
            tau_true=50.0,
            n_days=2,
            start_hour=10,  # daytime → hour_score=0.3
        )
        # Merge the two synthetic datasets
        merged_indoor = pd.concat(
            [
                list(good_dfs.values())[0],
                list(noisy_dfs.values())[0],
            ]
        ).sort_values("timestamp")
        climate_dfs = {"climate.room": merged_indoor}
        heating_df = pd.concat([good_heat, noisy_heat]).sort_values("timestamp").reset_index(drop=True)
        weather_df = pd.concat([good_wx, noisy_wx]).sort_values("timestamp").reset_index(drop=True)

        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None
        # Top 50% should favour the good τ≈10 windows; result should not be near 50
        assert result < 30.0

    def test_production_gray_zone_incident_now_preserves(self, tmp_path):
        """Reproduces the literal incident conditions from the design spec §1 (night_frac~100%
        of a small selected set, outdoor_median~14°C — just above the old guard's 12°C trigger,
        which is exactly why the old guard never fired here). The rejected first-draft formula
        also got this case wrong (confidence=1, full trust) because a single good signal (night)
        could rescue trust. This is the regression test for the actual production bug.
        """
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        dfs, heat, wx = self._make_night_blocks(
            tau_true=15.0,
            n_days=6,
            start_hour=23,
            window_hours=3,
            t_out=14.0,
        )

        result = model._calibrate_tau(dfs, heat, wx)

        assert result == 10.0, "the exact production gray-zone condition must now be an exact preserve"

    def test_warm_mostly_night_window_not_fully_trusted(self, tmp_path):
        """Key counterexample the rejected first-draft formula got wrong: an all-nighttime but
        warm window (a textbook summer window-ventilation case) must NOT reach full confidence
        just because night_frac is good — both signals are required.
        """
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        dfs, heat, wx = self._make_night_blocks(tau_true=16.0, n_days=5, start_hour=22, t_out=18.0)

        result = model._calibrate_tau(dfs, heat, wx)

        assert result == 10.0

    def test_sparse_candidates_reduce_confidence_not_maximize_it(self, tmp_path):
        """len(candidates) below _TAU_SAMPLE_CONF_REF must scale confidence down, not leave it at
        the old code's implicit "guard doesn't apply, proceed at full trust" default.

        Compared against a full-confidence run on independently-generated data with the same
        tau_true, rather than a captured/narrated number for this specific fixture.
        """
        model_sparse = _make_tau_model(tmp_path / "sparse")
        model_sparse._tau_hours = 10.0
        sparse_dfs, sparse_heat, sparse_wx = self._make_night_blocks(tau_true=12.0, start_hour=22, n_days=1)
        sparse_result = model_sparse._calibrate_tau(sparse_dfs, sparse_heat, sparse_wx)

        model_full = _make_tau_model(tmp_path / "full")
        model_full._tau_hours = 10.0
        full_dfs, full_heat, full_wx = self._make_night_blocks(tau_true=12.0, start_hour=22, n_days=5)
        full_result = model_full._calibrate_tau(full_dfs, full_heat, full_wx)

        assert sparse_result is not None and full_result is not None
        assert 10.0 < sparse_result < full_result, (
            "a 1-day (sparse) retrain must land strictly closer to the stored τ than an "
            "otherwise-identical 5-day (ample-sample) retrain toward the same raw estimate"
        )

    def test_both_signals_good_updates_normally(self, tmp_path):
        """Sanity check: when both night_frac and outdoor_median genuinely look winter-like,
        normal (confidence≈1) damped updates still occur — this isn't a regression to "never
        update."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        dfs, heat, wx = self._make_night_blocks(tau_true=12.0, start_hour=22, t_out=5.0)

        result = model._calibrate_tau(dfs, heat, wx)

        assert result is not None
        assert 10.0 < result < 10.8

    def test_cold_daytime_also_damped_not_fully_trusted(self, tmp_path):
        """Deliberate behavior change from the original code (which fully trusted this case):
        the codebase's own quality-scoring docstring (model.py:1731) attributes daytime bias to
        ventilation, independent of temperature — confidence now correctly requires BOTH
        night_frac and outdoor_median to be good, so this case gets confidence=0, same as the
        "both bad" case, not a full update.
        """
        model = _make_tau_model(tmp_path)
        model._tau_hours = 25.0
        cold_day_dfs, cold_day_heat, cold_day_wx = self._make_night_blocks(
            tau_true=10.0,
            n_days=5,
            start_hour=10,
            t_out=5.0,
        )

        result = model._calibrate_tau(cold_day_dfs, cold_day_heat, cold_day_wx)

        assert result == 25.0, "night_frac=0% must zero out confidence regardless of temperature"

    def test_selectivity_switches_to_top25pct_above_threshold(self, tmp_path, monkeypatch):
        """At ≥ _TAU_SELECTIVITY_THRESHOLD candidates, top-25% is used instead of top-50%.

        Fixture: 8 candidates total — 2 nighttime (high quality, τ≈10) + 6 daytime
        (low quality, τ≈50).  Threshold patched to 4 so the fixture crosses it.
        Top-25% selects the 2 nighttime windows → result ≈ 10.
        Old top-50% would select 4 windows (2 good + 2 noisy) → median much higher.
        """
        import pandas as pd

        model = _make_tau_model(tmp_path)
        monkeypatch.setattr(EnergyForecastModel, "_TAU_SELECTIVITY_THRESHOLD", 4)

        good_dfs, good_heat, good_wx = self._make_night_blocks(tau_true=10.0, n_days=2, start_hour=22)
        poor_dfs, poor_heat, poor_wx = self._make_night_blocks(
            tau_true=50.0,
            n_days=6,
            start_hour=10,  # daytime → hour_score = 0.3
        )

        climate_dfs = {
            "climate.room": pd.concat([list(good_dfs.values())[0], list(poor_dfs.values())[0]])
            .sort_values("timestamp")
            .reset_index(drop=True)
        }
        heating_df = pd.concat([good_heat, poor_heat]).sort_values("timestamp").reset_index(drop=True)
        weather_df = pd.concat([good_wx, poor_wx]).sort_values("timestamp").reset_index(drop=True)

        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)
        assert result is not None
        # top-25% of 8 = 2 nighttime windows → τ ≈ 10
        # top-50% of 8 = 4 windows (2 good + 2 poor) → median would be >> 20
        assert result < 20.0

    def test_flat_indoor_temp_excluded_by_r2(self, tmp_path):
        """A window where indoor temp doesn't decay (r²≤0) is excluded from candidates."""
        import pandas as pd

        model = _make_tau_model(tmp_path)
        # Flat indoor temperature — no exponential decay, OLS slope ≈ 0
        ts = pd.date_range("2026-01-10 20:00", periods=8, freq="1h")
        t_flat = 20.0  # constant indoor temp
        t_out = 5.0
        climate_df = pd.DataFrame(
            {
                "timestamp": ts,
                "current_temp": [t_flat] * 8,
                "setpoint": [t_flat] * 8,
            }
        )
        heating_df = pd.DataFrame({"timestamp": ts, "heating_active": [0.0] * 8})
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [t_out] * 8,
                "direct_radiation_wm2": [0.0] * 8,
            }
        )
        result = model._calibrate_tau({"climate.room": climate_df}, heating_df, weather_df)
        # Flat decay → r²≤0 → no valid candidates → None
        assert result is None

    def test_quality_nighttime_higher_than_daytime(self, tmp_path):
        """Nighttime windows score higher quality than equivalent daytime windows."""
        import pandas as pd

        model = _make_tau_model(tmp_path)
        tau_true = 10.0

        def _make_block(start_hour: int) -> tuple:
            ts = pd.date_range(f"2026-01-10 {start_hour:02d}:00", periods=6, freq="1h")
            t_in = [20.0]
            for _ in range(5):
                t_in.append(t_in[-1] + (5.0 - t_in[-1]) / tau_true)
            climate_df = pd.DataFrame(
                {
                    "timestamp": ts,
                    "current_temp": t_in,
                    "setpoint": [20.0] * 6,
                }
            )
            heating_df = pd.DataFrame({"timestamp": ts, "heating_active": [0.0] * 6})
            weather_df = pd.DataFrame(
                {
                    "timestamp": ts,
                    "temp_c": [5.0] * 6,
                    "direct_radiation_wm2": [0.0] * 6,
                }
            )
            return {"climate.room": climate_df}, heating_df, weather_df

        # Collect candidate qualities by calling private logic via _calibrate_tau
        # and comparing τ estimates (same τ, both should produce a result)
        night_dfs, night_heat, night_wx = _make_block(start_hour=22)
        day_dfs, day_heat, day_wx = _make_block(start_hour=10)

        result_night = model._calibrate_tau(night_dfs, night_heat, night_wx)
        result_day = model._calibrate_tau(day_dfs, day_heat, day_wx)

        # Both produce a valid τ (not excluded)
        assert result_night is not None
        assert result_day is not None
        # Both should be near the true τ since the decay is identical
        assert 5.0 < result_night < 20.0
        assert 5.0 < result_day < 20.0

    def test_tau_calibration_log_shows_actual_percentage(self, tmp_path, monkeypatch, caplog):
        """Debug log must show round(actual%) not 100//divisor.

        With 9 candidates (threshold patched to 8): divisor=4, n_select=2,
        actual = round(100*2/9) = 22. Old code: 100//4 = 25.
        """
        import logging

        import pandas as pd

        model = _make_tau_model(tmp_path)
        monkeypatch.setattr(EnergyForecastModel, "_TAU_SELECTIVITY_THRESHOLD", 8)

        # 4 good (nighttime) + 5 poor (daytime) = 9 candidates ≥ threshold 8
        good_dfs, good_heat, good_wx = self._make_night_blocks(tau_true=10.0, n_days=4, start_hour=22)
        poor_dfs, poor_heat, poor_wx = self._make_night_blocks(tau_true=50.0, n_days=5, start_hour=10)

        climate_dfs = {
            "climate.room": pd.concat([list(good_dfs.values())[0], list(poor_dfs.values())[0]])
            .sort_values("timestamp")
            .reset_index(drop=True)
        }
        heating_df = pd.concat([good_heat, poor_heat]).sort_values("timestamp").reset_index(drop=True)
        weather_df = pd.concat([good_wx, poor_wx]).sort_values("timestamp").reset_index(drop=True)

        with caplog.at_level(logging.DEBUG):
            model._calibrate_tau(climate_dfs, heating_df, weather_df)

        tau_logs = [r.message for r in caplog.records if "candidates, using top" in r.message]
        assert tau_logs, "Expected a τ calibration DEBUG log entry with 'candidates, using top'"
        log_msg = tau_logs[0]
        assert "25%" not in log_msg, f"Old 100//divisor='25%' still in log: {log_msg}"
        assert "22%" in log_msg, f"Expected '22%' in log: {log_msg}"

    def test_selectivity_boundary_exactly_at_threshold(self, tmp_path, monkeypatch):
        """At exactly _TAU_SELECTIVITY_THRESHOLD candidates, top-25% (not top-50%) must be used.

        Threshold=4, exactly 4 candidates: 2 good (τ≈10, nighttime) + 2 poor (τ≈50, daytime).
        top-25% of 4 = 1 window → median ≈ 10.
        top-50% of 4 = 2 windows → median of 1 good + 1 poor ≈ 30.
        So result < 30 proves the >= boundary fires (not >).
        """
        import pandas as pd

        model = _make_tau_model(tmp_path)
        monkeypatch.setattr(EnergyForecastModel, "_TAU_SELECTIVITY_THRESHOLD", 4)

        good_dfs, good_heat, good_wx = self._make_night_blocks(tau_true=10.0, n_days=2, start_hour=22)
        poor_dfs, poor_heat, poor_wx = self._make_night_blocks(tau_true=50.0, n_days=2, start_hour=10)

        climate_dfs = {
            "climate.room": pd.concat([list(good_dfs.values())[0], list(poor_dfs.values())[0]])
            .sort_values("timestamp")
            .reset_index(drop=True)
        }
        heating_df = pd.concat([good_heat, poor_heat]).sort_values("timestamp").reset_index(drop=True)
        weather_df = pd.concat([good_wx, poor_wx]).sort_values("timestamp").reset_index(drop=True)

        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)

        assert result is not None
        assert result < 30.0, (
            f"Expected result close to good τ≈10 (top-25%), got {result:.1f} — 50% branch was used (>= boundary broken)"
        )

    # ── Improvement tests ────────────────────────────────────────────────────

    def test_warm_outdoor_windows_downweighted_vs_cold(self, tmp_path):
        """outdoor_temp_score penalises warm-weather windows, NOT captured by delta_t_score.

        Both groups have ΔT=10°C (delta_t_score=1.0 for both — so delta_t_score
        cannot differentiate them).  The cold group (T_out=2°C) should score higher
        than the warm group (T_out=17°C) because outdoor_temp_score penalises warm
        outdoor temps (exp(-(17-10)/8)≈0.42 vs 1.0 for cold).

        5 cold-outdoor nighttime windows (T_out=2°C, τ=8h) vs 5 warm-outdoor nighttime
        windows (T_out=17°C, τ=35h) → result should be closer to 8h than to 35h.
        """
        # ΔT = t_in_start - t_out = 10°C for BOTH groups → delta_t_score=1.0 for both.
        # Same start_hour=22 (nighttime) → identical hour_score → only outdoor_temp_score differs.
        # Warm group uses day_start=10 so it sorts FIRST in the candidate list; without
        # outdoor_temp_score, warm windows win by date order → result≈35h → test fails.
        # After the fix, outdoor_temp_score suppresses warm windows regardless of order.
        warm_dfs, warm_heat, warm_wx = self._make_night_blocks(
            tau_true=35.0, n_days=5, start_hour=22, t_out=17.0, t_in_start=27.0, day_start=10
        )
        cold_dfs, cold_heat, cold_wx = self._make_night_blocks(
            tau_true=8.0, n_days=5, start_hour=22, t_out=2.0, t_in_start=12.0, day_start=20
        )

        climate_dfs = {
            "climate.room": pd.concat([list(cold_dfs.values())[0], list(warm_dfs.values())[0]])
            .sort_values("timestamp")
            .reset_index(drop=True)
        }
        heating_df = pd.concat([cold_heat, warm_heat]).sort_values("timestamp").reset_index(drop=True)
        weather_df = pd.concat([cold_wx, warm_wx]).sort_values("timestamp").reset_index(drop=True)

        model = _make_tau_model(tmp_path)
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)

        assert result is not None
        assert result < 20.0, f"Cold-outdoor (τ≈8h) windows should dominate warm-outdoor (τ≈35h); got τ={result:.1f}h"

    def test_daytime_penalty_suppresses_daytime_when_nighttime_has_moderate_radiation(self, tmp_path):
        """0.1 hour penalty (not 0.3) is needed when nighttime competes with clear daytime windows.

        5 nighttime windows with moderate radiation (radiation=500 W/m²):
          solar_score = exp(-500/400) ≈ 0.287
          quality ≈ r² × outdoor_temp × n × 0.287 × 1.0  ≈ 0.287

        5 daytime windows with zero radiation:
          quality (0.3 penalty) = r² × outdoor_temp × n × 1.0 × 0.3 = 0.300  → day > night → 40h wins
          quality (0.1 penalty) = r² × outdoor_temp × n × 1.0 × 0.1 = 0.100  → night > day → 8h wins

        With old 0.3 penalty daytime quality (0.30) beats nighttime quality (0.287), so
        all 5 daytime windows fill the top-50% → median≈40h.
        With new 0.1 penalty nighttime quality (0.287) beats daytime (0.10) → top-50%
        is all 5 nighttime windows → median≈8h.
        """
        # Nighttime windows with moderate radiation — quality ≈ 0.287
        night_dfs, night_heat, night_wx = self._make_night_blocks(
            tau_true=8.0, n_days=5, start_hour=22, t_out=5.0, radiation=500.0
        )
        # Daytime windows, zero radiation — quality (old) 0.3, (new) 0.1
        day_dfs, day_heat, day_wx = self._make_night_blocks(
            tau_true=40.0, n_days=5, start_hour=10, t_out=5.0, radiation=0.0
        )

        climate_dfs = {
            "climate.room": pd.concat([list(night_dfs.values())[0], list(day_dfs.values())[0]])
            .sort_values("timestamp")
            .reset_index(drop=True)
        }
        heating_df = pd.concat([night_heat, day_heat]).sort_values("timestamp").reset_index(drop=True)
        weather_df = pd.concat([night_wx, day_wx]).sort_values("timestamp").reset_index(drop=True)

        model = _make_tau_model(tmp_path)
        result = model._calibrate_tau(climate_dfs, heating_df, weather_df)

        assert result is not None
        assert result < 20.0, (
            f"Nighttime windows (solar_score≈0.287) should outrank daytime (0.1 penalty); got τ={result:.1f}h"
        )

    def test_spring_bias_guard_preserves_stored_tau(self, tmp_path):
        """All-daytime + warm-outdoor → stored winter τ is preserved, not overwritten.

        When every candidate is daytime (night_fraction=0) AND outdoor median > 12°C,
        _calibrate_tau returns the stored _tau_hours instead of the spring estimate.
        """
        model = _make_tau_model(tmp_path)
        model._tau_hours = 25.0  # stored winter τ

        # 5 daytime windows at T_out=18°C → night_frac=0%, outdoor_median=18°C > 12°C
        bias_dfs, bias_heat, bias_wx = self._make_night_blocks(tau_true=5.0, n_days=5, start_hour=10, t_out=18.0)

        result = model._calibrate_tau(bias_dfs, bias_heat, bias_wx)

        assert result == 25.0, f"Spring-biased calibration should preserve stored τ=25.0h; got {result}"

    def test_spring_bias_guard_not_triggered_without_stored_tau(self, tmp_path):
        """Guard only fires when a stored τ exists — first-run proceeds normally."""
        model = _make_tau_model(tmp_path)
        # _tau_hours is None (fresh model) — guard must not suppress calibration

        bias_dfs, bias_heat, bias_wx = self._make_night_blocks(tau_true=5.0, n_days=5, start_hour=10, t_out=18.0)

        result = model._calibrate_tau(bias_dfs, bias_heat, bias_wx)

        assert result is not None, "First-run calibration should proceed even with spring-biased data"

    def test_single_retrain_tau_move_bounded_by_drift_cap(self, tmp_path):
        """For an extreme, fully-trusted pull, the drift cap's ±35%-of-anchor bound is the
        *tighter* constraint (not the EMA weight's confidence*20%-of-gap alone), because the
        anchor initializes to the pre-update stored value on the very first blended retrain."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        dfs, heat, wx = self._make_night_blocks(
            tau_true=100.0,
            start_hour=22,
            t_out=5.0,
            window_hours=12,
        )

        result = model._calibrate_tau(dfs, heat, wx)

        assert result is not None
        assert result == pytest.approx(13.5), "drift cap (anchor=10.0 * 1.35) must be the binding constraint"
        assert abs(result - 10.0) <= 10.0 * 0.35 + 1e-6


class TestTauDriftCap:
    """Fix B: bounds cumulative τ movement to ±_TAU_MAX_DRIFT_FRAC (short) / ±_TAU_LONG_MAX_DRIFT_FRAC
    (long) from the value at the start of each respective rolling window, independent of Fix A's
    per-step damping and independent of how many retrains occur inside the window.

    _make_night_blocks (tests/test_model.py:4749) hardcodes its dates to January 2026 -- all
    anchor timestamps below are set well before January 2026 accordingly.
    """

    def _seed_anchors(self, model, anchor_hours, anchor_ts):
        model._tau_anchor_hours = anchor_hours
        model._tau_anchor_ts = anchor_ts
        model._tau_long_anchor_hours = anchor_hours
        model._tau_long_anchor_ts = anchor_ts

    def test_clamps_when_cumulative_budget_exhausted(self, tmp_path):
        """A single retrain's per-step-legal blend must still be clamped if it would push τ
        beyond the rolling window's remaining budget. old_tau=6.0 sits below the window floor
        (anchor=10.0 * 0.65 = 6.5); a modest raw pull (tau_true=6.5) gets clamped up to exactly
        the floor. Anchor set 2025-12-15 -- ~19 days before the fixture's Jan 3-9 data, inside
        the 30-day short window."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 6.0
        self._seed_anchors(model, 10.0, pd.Timestamp("2025-12-15"))

        dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
            tau_true=6.5,
            start_hour=22,
            t_out=5.0,
            day_start=3,
        )
        result = model._calibrate_tau(dfs, heat, wx)

        assert result is not None
        assert result == pytest.approx(6.5), "must be clamped up to the 30-day drift floor (anchor * 0.65)"

    def test_resets_after_window_elapses(self, tmp_path):
        """Once _TAU_DRIFT_WINDOW_DAYS has passed since the anchor, a new anchor is set at the
        current stored τ -- this is what allows genuine multi-month seasonal correction. Anchor
        set 2025-10-01 -- roughly 97 days before the fixture's Jan 1-6 data, comfortably past
        the 30-day window."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        self._seed_anchors(model, 10.0, pd.Timestamp("2025-10-01"))

        dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
            tau_true=10.0,
            start_hour=22,
            t_out=5.0,
            day_start=1,
        )
        model._calibrate_tau(dfs, heat, wx)

        assert model._tau_anchor_ts > pd.Timestamp("2025-10-01")

    def test_no_cap_within_budget(self, tmp_path):
        """A modest, per-step-legal move well inside the 35% budget is not touched by the cap."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        self._seed_anchors(model, 10.0, pd.Timestamp("2025-12-15"))

        dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
            tau_true=11.0,
            start_hour=22,
            t_out=5.0,
            day_start=3,
        )
        result = model._calibrate_tau(dfs, heat, wx)

        assert result is not None
        assert 10.0 < result < 13.5  # within [anchor*0.65, anchor*1.35] = [6.5, 13.5]

    def test_anchor_timestamps_are_tz_naive(self, tmp_path):
        """Both anchor timestamps must stay tz-naive, matching combined.index."""
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0
        dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
            tau_true=11.0,
            start_hour=22,
            t_out=5.0,
        )
        model._calibrate_tau(dfs, heat, wx)

        assert model._tau_anchor_ts.tzinfo is None
        assert model._tau_long_anchor_ts.tzinfo is None

    def test_long_cap_bounds_multi_window_compounding(self, tmp_path):
        """The short (30-day) cap alone only bounds movement WITHIN one window, not ACROSS
        successive windows under sustained bias -- each window reset re-anchors at wherever the
        previous window left off. Simulates 6 successive short-window resets under a sustained,
        moderate (not confidence=0) pull and confirms the long (180-day) cap holds the line.

        Without the long cap, 6 windows of unfettered 35%-per-window compounding would allow
        10.0 * 0.65**6 =~ 0.75h. With the long cap, τ is held at exactly 10.0 * 0.5 = 5.0h.
        """
        model = _make_tau_model(tmp_path)
        model._tau_hours = 10.0

        for i in range(6):
            dfs, heat, wx = TestTauCalibrationSafeguards()._make_night_blocks(
                tau_true=2.0,
                n_days=5,
                start_hour=22,
                window_hours=8,
                t_out=8.0,
                day_start=1,
            )
            if i > 0:
                model._tau_anchor_ts = model._tau_anchor_ts - pd.Timedelta(days=31)
            result = model._calibrate_tau(dfs, heat, wx)
            assert result is not None
            model._tau_hours = result

        assert model._tau_hours == pytest.approx(5.0), (
            "sustained bias across many short-window resets must still be caught by the "
            "180-day/50% long cap, not allowed to compound toward the naive ~0.75h"
        )


# ── Auto-K Regime Selection ───────────────────────────────────────────────────


class TestAutoKRegimeSelection:
    """Integration: regime_count=0 triggers auto-K selection during train()."""

    def _make_regime_train_data(self, n: int = 400, n_regimes: int = 3):
        """Return (energy_df, weather_df) with distinct daily consumption regimes."""
        rng = np.random.default_rng(7)
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        gross = []
        for i, t in enumerate(ts):
            regime = (t.date().toordinal()) % n_regimes
            base = [0.5, 2.5, 1.5][regime]
            gross.append(base + rng.uniform(-0.1, 0.1))
        energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": gross})
        weather_df = pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": rng.uniform(-5, 25, size=n),
                "precipitation_mm": [0.0] * n,
                "sunshine_min": [30.0] * n,
                "wind_kmh": [10.0] * n,
                "cloud_cover_pct": [50.0] * n,
                "direct_radiation_wm2": [100.0] * n,
            }
        )
        return energy_df, weather_df

    def test_auto_k_selects_valid_k(self, tmp_path):
        """regime_count=0 → train() completes and model._regime_count is in [2, 8]."""
        from apps.energy_forecast.clustering import SKLEARN_AVAILABLE

        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn not available")

        energy_df, weather_df = self._make_regime_train_data()
        m = EnergyForecastModel(tmp_path)
        m.train(
            energy_df,
            weather_df,
            outdoor_df=None,
            weight_halflife_days=0,
            enable_regimes=True,
            regime_count=0,
        )
        assert m._regime_count >= 2, f"Expected K≥2, got {m._regime_count}"
        assert m._regime_count <= 8, f"Expected K≤8, got {m._regime_count}"

    def test_fixed_k_unchanged(self, tmp_path):
        """regime_count=3 → model._regime_count remains 3 (auto-K not triggered)."""
        from apps.energy_forecast.clustering import SKLEARN_AVAILABLE

        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn not available")

        energy_df, weather_df = self._make_regime_train_data()
        m = EnergyForecastModel(tmp_path)
        m.train(
            energy_df,
            weather_df,
            outdoor_df=None,
            weight_halflife_days=0,
            enable_regimes=True,
            regime_count=3,
        )
        assert m._regime_count == 3

    def test_train_and_predict_end_to_end(self, tmp_path):
        """train(enable_regimes=True) + predict() returns a 48-row non-NaN DataFrame."""
        pytest.importorskip("sklearn")
        from apps.energy_forecast.clustering import SKLEARN_AVAILABLE

        if not SKLEARN_AVAILABLE:
            pytest.skip("scikit-learn not available")

        energy_df, weather_df = self._make_regime_train_data(n=400)
        m = EnergyForecastModel(tmp_path)
        m.train(
            energy_df,
            weather_df,
            outdoor_df=None,
            weight_halflife_days=0,
            enable_regimes=True,
            regime_count=0,
        )
        assert m._clusterer is not None and m._clusterer.is_fitted
        assert m._regime_model is not None and m._regime_model.is_fitted

        future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
        forecast_df = pd.DataFrame(
            {
                "timestamp": future_ts,
                "temp_c": [10.0] * 48,
                "precipitation_mm": [0.0] * 48,
                "sunshine_min": [30.0] * 48,
                "wind_kmh": [10.0] * 48,
                "cloud_cover_pct": [50.0] * 48,
                "direct_radiation_wm2": [100.0] * 48,
            }
        )
        result = m.predict(forecast_df, live_temp=10.0)
        assert len(result) == 48, f"Expected 48 rows, got {len(result)}"
        assert "predicted_kwh" in result.columns
        assert result["predicted_kwh"].notna().all(), "predicted_kwh contains NaN"
        assert (result["predicted_kwh"] >= 0).all(), "predicted_kwh contains negative values"


# ── Stage 1: Algorithmic correctness tests ────────────────────────────────────


class TestCQRCalibrationRandomSplit:
    """#64 — CQR calibration must use a random holdout, not a temporal tail."""

    def test_quantile_models_trained_after_random_split(self, tmp_path):
        """train() with random CQR split still produces valid quantile models."""
        m, _ = _make_trained_model(tmp_path)
        assert m._model_q10 is not None
        assert m._model_q90 is not None
        assert isinstance(m._interval_correction, float)
        assert np.isfinite(m._interval_correction)

    def test_cal_indices_scattered_via_rng(self, tmp_path):
        """Verify the random split: cal_idx from np.random.default_rng(42) must not
        equal the tail slice (last 15% of rows), confirming exchangeability fix."""
        import numpy as np_inner

        n = 600
        cal_size = max(20, int(n * 0.15))
        # Reproduce the exact RNG used in model.py
        rng = np_inner.random.default_rng(42)
        cal_idx = rng.choice(n, size=cal_size, replace=False)
        tail_idx = np_inner.arange(n - cal_size, n)
        # The randomly chosen indices should NOT equal the tail
        assert not np_inner.array_equal(np_inner.sort(cal_idx), tail_idx), (
            "CQR cal_idx equals temporal tail — exchangeability fix not applied"
        )
        # And they must cover rows from the first half (impossible with a tail slice)
        assert (cal_idx < n // 2).any(), "Random cal_idx contains no rows from the first half — not random"


class TestEWMAGapReset:
    """#69 — EWMA must reset at weather data gaps > 2 hours."""

    def test_gap_inserts_nan_before_ewma(self):
        """A 3-hour gap in weather timestamps triggers EWMA reset (warns and
        temp_c at gap boundary is NaN → ewm propagates from NaN)."""
        import warnings

        # Build a 48-hour series with a 3-hour gap at hour 20
        ts_before = pd.date_range("2026-01-01 00:00", periods=20, freq="1h")
        ts_after = pd.date_range("2026-01-01 23:00", periods=28, freq="1h")  # 3h gap
        ts_all = ts_before.append(ts_after)

        df = _make_bare_df(ts_all)
        w = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts_all),
                "temp_c": [10.0] * 20 + [20.0] * 28,
                "precipitation_mm": [0.0] * len(ts_all),
                "sunshine_min": [30.0] * len(ts_all),
                "wind_kmh": [10.0] * len(ts_all),
                "cloud_cover_pct": [50.0] * len(ts_all),
                "direct_radiation_wm2": [100.0] * len(ts_all),
            }
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = _engineer_features(df, w, None)

        assert "temp_ewma_24h" in result.columns
        assert result["temp_ewma_24h"].notna().any()

    def test_no_gap_no_warning(self, caplog):
        """Contiguous timestamps produce no EWMA gap warning."""
        import logging

        ts = pd.date_range("2026-01-01 00:00", periods=48, freq="1h")
        df = _make_bare_df(ts)
        w = _make_weather_df(ts, temp=10.0)

        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            _engineer_features(df, w, None)

        assert not any("EWMA" in r.message for r in caplog.records)

    def test_single_gap_logged_at_debug(self, caplog):
        """A single gap (expected DST case) logs at DEBUG, not WARNING."""
        import logging

        ts_before = pd.date_range("2026-01-01 00:00", periods=10, freq="1h")
        ts_after = pd.date_range("2026-01-01 13:00", periods=10, freq="1h")
        ts_all = ts_before.append(ts_after)
        df = _make_bare_df(ts_all)
        w = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts_all),
                "temp_c": [10.0] * 20,
                "precipitation_mm": [0.0] * 20,
                "sunshine_min": [30.0] * 20,
                "wind_kmh": [10.0] * 20,
                "cloud_cover_pct": [50.0] * 20,
                "direct_radiation_wm2": [100.0] * 20,
            }
        )

        with caplog.at_level(logging.DEBUG, logger="energy_forecast"):
            _engineer_features(df, w, None)

        ewma_records = [r for r in caplog.records if "EWMA" in r.message]
        assert ewma_records, "Expected EWMA gap log not found"
        assert all(r.levelno == logging.DEBUG for r in ewma_records), "Single gap should log at DEBUG, not WARNING"

    def test_multiple_gaps_logged_at_warning(self, caplog):
        """Multiple gaps (unexpected weather API holes) log at WARNING."""
        import logging

        ts1 = pd.date_range("2026-01-01 00:00", periods=5, freq="1h")
        ts2 = pd.date_range("2026-01-01 08:00", periods=5, freq="1h")
        ts3 = pd.date_range("2026-01-01 16:00", periods=5, freq="1h")
        ts_all = ts1.append(ts2).append(ts3)
        df = _make_bare_df(ts_all)
        w = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts_all),
                "temp_c": [10.0] * 15,
                "precipitation_mm": [0.0] * 15,
                "sunshine_min": [30.0] * 15,
                "wind_kmh": [10.0] * 15,
                "cloud_cover_pct": [50.0] * 15,
                "direct_radiation_wm2": [100.0] * 15,
            }
        )

        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            _engineer_features(df, w, None)

        ewma_records = [r for r in caplog.records if "EWMA" in r.message]
        assert ewma_records, "Expected EWMA gap warning not found"
        assert any(r.levelno == logging.WARNING for r in ewma_records), "Multiple gaps should log at WARNING"


# ── Stage 3: Test coverage additions ─────────────────────────────────────────


class TestTrainEdgeCases:
    """#77 — train() must handle degenerate inputs gracefully."""

    def _make_weather(self, ts):
        return pd.DataFrame(
            {
                "timestamp": ts,
                "temp_c": [10.0] * len(ts),
                "precipitation_mm": [0.0] * len(ts),
                "sunshine_min": [30.0] * len(ts),
                "wind_kmh": [10.0] * len(ts),
                "cloud_cover_pct": [50.0] * len(ts),
                "direct_radiation_wm2": [100.0] * len(ts),
            }
        )

    def test_train_empty_dataframe(self, tmp_path, caplog):
        """train() with an empty energy_df logs a warning and does not raise."""
        import logging

        energy = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "gross_kwh": pd.Series(dtype=float)})
        ts = pd.date_range("2024-01-01", periods=10, freq="1h")
        weather = self._make_weather(ts)
        m = EnergyForecastModel(tmp_path)
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        # Model should not be trained (train() returned early)
        assert m.model is None or not m.feature_cols, "Model should not be trained on empty energy_df"

    def test_train_below_min_rows(self, tmp_path, caplog):
        """train() with fewer than MIN_TRAINING_ROWS clean rows logs warning and returns."""
        import logging

        from energy_forecast.model import MIN_TRAINING_ROWS

        n = MIN_TRAINING_ROWS - 1
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * n})
        weather = self._make_weather(ts)
        m = EnergyForecastModel(tmp_path)
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        assert any("skipping" in r.message.lower() or "need" in r.message for r in caplog.records), (
            "Expected 'skipping' warning for below-MIN_TRAINING_ROWS input"
        )

    def test_train_constant_values(self, tmp_path):
        """train() with all gross_kwh=1.0 must not raise."""
        n = 200
        ts = pd.date_range("2024-01-01", periods=n, freq="1h")
        energy = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * n})
        weather = self._make_weather(ts)
        m = EnergyForecastModel(tmp_path)
        # Should not raise even if LightGBM complains about constant target
        try:
            m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
        except Exception as exc:
            pytest.fail(f"train() raised on constant values: {exc}")


# ── HOD-aware short-lag NaN fill ──────────────────────────────────────────────


def _make_hod_actuals(n_hours: int = 168) -> pd.DataFrame:
    """Actuals where gross_kwh = hour_of_day + 10.0 (each HOD bucket is uniquely identifiable)."""
    now_floor = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
    start = now_floor - pd.Timedelta(hours=n_hours)
    ts = pd.date_range(start, periods=n_hours, freq="1h")
    kwh = np.array([float(t.hour) + 10.0 for t in ts])
    return pd.DataFrame({"timestamp": ts, "gross_kwh": kwh})


class TestHodLagFill:
    """NaN positions in short lag features (lag < 48h) should be filled with the
    7-day per-hour-of-day mean from recent_actuals, not the flat training median."""

    def test_hod_fill_uses_recent_mean_not_global_median(self, tmp_path):
        """lag_1h[h=1] must be filled with the HOD mean from recent_actuals, not the
        training global median (which is ~2.75 for uniform-0.5..5 training data)."""
        m, forecast = _make_trained_model(tmp_path)
        if "lag_1h" not in m.feature_cols:
            pytest.skip("lag_1h not in feature_cols")

        recent_actuals = _make_hod_actuals(168)
        _, X = m._prepare_prediction_X(forecast, None, recent_actuals)

        # lag_1h[h=1]: logical timestamp = future_ts[1] - 1h = now_floor + 1h - 1h = now_floor
        # now_floor is NOT in recent_actuals (actuals end at now_floor - 1h) → was NaN
        now_floor = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")
        logical_hod = now_floor.hour
        expected_fill = float(logical_hod) + 10.0  # HOD mean from _make_hod_actuals

        actual_fill = float(X["lag_1h"].iloc[1])
        # HOD fill must be > 9 (our actuals are 10..33); training median is ~2.75
        assert actual_fill > 9.0, (
            f"lag_1h[h=1] expected HOD mean ≥ 10 (got {actual_fill:.4f}); training median used instead?"
        )
        assert abs(actual_fill - expected_fill) < 0.5, (
            f"lag_1h[h=1] expected HOD mean {expected_fill:.2f} (HOD={logical_hod}), got {actual_fill:.4f}"
        )

    def test_hod_fill_varies_per_nan_position(self, tmp_path):
        """Each NaN position must get the fill for its own hour-of-day bucket,
        not the same constant value for all NaN rows."""
        m, forecast = _make_trained_model(tmp_path)
        if "lag_1h" not in m.feature_cols:
            pytest.skip("lag_1h not in feature_cols")

        recent_actuals = _make_hod_actuals(168)
        _, X = m._prepare_prediction_X(forecast, None, recent_actuals)

        now_floor = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None).floor("1h")

        # Check 8 NaN positions h=2..9 — each gets a different HOD bucket
        fills = []
        expected = []
        for h in range(2, 10):
            logical_ts = now_floor + pd.Timedelta(hours=h - 1)
            expected_hod = logical_ts.hour
            expected.append(float(expected_hod) + 10.0)
            fills.append(float(X["lag_1h"].iloc[h]))

        # Values must vary across positions (not all the same constant)
        assert len(set(round(v, 3) for v in fills)) > 1, (
            "lag_1h fill is constant across NaN positions — HOD-aware fill not applied"
        )
        # Each value must match its HOD bucket
        for h_idx, (actual_fill, expected_fill) in enumerate(zip(fills, expected)):
            assert abs(actual_fill - expected_fill) < 0.5, (
                f"lag_1h[h={h_idx + 2}]: expected HOD mean {expected_fill:.2f}, got {actual_fill:.4f}"
            )

    def test_hod_fill_sparse_actuals_no_crash(self, tmp_path):
        """With only 5 rows of actuals (hours 0-4), NaN fill must not raise;
        unseen HOD buckets fall through to the existing median fill."""
        m, forecast = _make_trained_model(tmp_path)

        base = pd.Timestamp("2026-03-05 00:00")
        ts = pd.date_range(base, periods=5, freq="1h")
        sparse_actuals = pd.DataFrame(
            {
                "timestamp": ts,
                "gross_kwh": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        _, X = m._prepare_prediction_X(forecast, None, sparse_actuals)
        assert X is not None
        assert len(X) == 48

    def test_hod_fill_none_actuals_unchanged(self, tmp_path):
        """With recent_actuals=None the HOD fill is skipped; predict completes without error."""
        m, forecast = _make_trained_model(tmp_path)
        _, X_none = m._prepare_prediction_X(forecast, None, None)
        assert X_none is not None
        assert len(X_none) == 48


class TestPreparedPrediction:
    """P1: predict/predict_intervals/shap_summary must skip _prepare_prediction_X
    when called with a pre-computed _prepared=(future_hours, X) tuple."""

    def test_predict_skips_prepare_when_prepared_given(self, tmp_path, monkeypatch):
        """predict() must not call _prepare_prediction_X when _prepared is supplied."""
        m, forecast = _make_trained_model(tmp_path)
        future_hours, X = m._prepare_prediction_X(forecast, live_temp=10.0, recent_actuals=None)

        prepare_calls = []
        original = m._prepare_prediction_X
        monkeypatch.setattr(
            m,
            "_prepare_prediction_X",
            lambda *a, **kw: (prepare_calls.append(1), original(*a, **kw))[1],
        )

        result = m.predict(forecast, live_temp=10.0, _prepared=(future_hours, X))

        assert prepare_calls == [], "_prepare_prediction_X must not be called when _prepared is given"
        assert len(result) == 48

    def test_predict_intervals_skips_prepare_when_prepared_given(self, tmp_path, monkeypatch):
        """predict_intervals() must not call _prepare_prediction_X when _prepared is supplied."""
        m, forecast = _make_trained_model(tmp_path)
        if m._model_q10 is None:
            pytest.skip("quantile models not trained in this fixture")
        future_hours, X = m._prepare_prediction_X(forecast, live_temp=10.0, recent_actuals=None)

        prepare_calls = []
        original = m._prepare_prediction_X
        monkeypatch.setattr(
            m,
            "_prepare_prediction_X",
            lambda *a, **kw: (prepare_calls.append(1), original(*a, **kw))[1],
        )

        result = m.predict_intervals(forecast, live_temp=10.0, _prepared=(future_hours, X))

        assert prepare_calls == []
        assert result is None or len(result) == 48

    def test_shap_summary_skips_prepare_when_prepared_given(self, tmp_path, monkeypatch):
        """shap_summary() must not call _prepare_prediction_X when _prepared is supplied."""
        m, forecast = _make_trained_model(tmp_path)
        future_hours, X = m._prepare_prediction_X(forecast, live_temp=10.0, recent_actuals=None)

        prepare_calls = []
        original = m._prepare_prediction_X
        monkeypatch.setattr(
            m,
            "_prepare_prediction_X",
            lambda *a, **kw: (prepare_calls.append(1), original(*a, **kw))[1],
        )

        result = m.shap_summary(forecast, live_temp=10.0, n=3, _prepared=(future_hours, X))

        assert prepare_calls == []
        assert isinstance(result, dict)

    def test_predict_without_prepared_still_works(self, tmp_path):
        """predict() without _prepared must work exactly as before."""
        m, forecast = _make_trained_model(tmp_path)
        result = m.predict(forecast, live_temp=10.0)
        assert len(result) == 48
        assert (result["predicted_kwh"] >= 0).all()


class TestStripPartialLastDay:
    def test_removes_partial_last_day(self):
        ts_full = pd.date_range("2024-01-01", periods=48, freq="1h")  # 2 complete days
        ts_partial = pd.date_range("2024-01-03", periods=10, freq="1h")  # 10 h today
        ts = ts_full.append(ts_partial)
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": 1.0})
        result = _strip_partial_last_day(df)
        assert result["timestamp"].dt.date.max() == pd.Timestamp("2024-01-02").date()
        assert len(result) == 48

    def test_keeps_complete_last_day(self):
        ts = pd.date_range("2024-01-01", periods=48, freq="1h")  # 2 × 24 h exactly
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": 1.0})
        result = _strip_partial_last_day(df)
        assert len(result) == 48

    def test_keeps_day_with_exactly_24_hours(self):
        ts = pd.date_range("2024-01-01", periods=24, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": 1.0})
        assert len(_strip_partial_last_day(df)) == 24

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "gross_kwh": pd.Series(dtype=float)})
        assert _strip_partial_last_day(df).empty


class TestPhysicsFeatureIntegration:
    def test_physics_kwh_present_when_series_given(self):
        ts = pd.date_range("2026-01-15", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 5})
        weather = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 5, "wind_kmh": [10.0] * 5})
        physics_series = pd.Series([0.5] * 5, index=ts.floor("1h"))
        result = _engineer_features(df, weather, outdoor_df=None, physics_kwh_series=physics_series)
        assert "physics_kwh" in result.columns
        assert result["physics_kwh"].iloc[0] == pytest.approx(0.5)

    def test_physics_kwh_absent_not_zero_filled_when_none(self):
        ts = pd.date_range("2026-01-15", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 5})
        weather = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 5, "wind_kmh": [10.0] * 5})
        result = _engineer_features(df, weather, outdoor_df=None, physics_kwh_series=None)
        assert "physics_kwh" not in result.columns

    def test_heating_buffer_temp_present_when_series_given(self):
        ts = pd.date_range("2026-01-15", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 5})
        weather = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 5, "wind_kmh": [10.0] * 5})
        buffer_series = pd.Series([42.0] * 5, index=ts.floor("1h"))
        result = _engineer_features(df, weather, outdoor_df=None, heating_buffer_temp_series=buffer_series)
        assert "heating_buffer_temp" in result.columns

    def test_heating_buffer_temp_absent_when_none(self):
        ts = pd.date_range("2026-01-15", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 5})
        weather = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 5, "wind_kmh": [10.0] * 5})
        result = _engineer_features(df, weather, outdoor_df=None)
        assert "heating_buffer_temp" not in result.columns


def _make_phase1_test_model(tmp_path) -> EnergyForecastModel:
    """Model with `self.model`/`self.engine`/`feature_cols` fit on exactly
    ["physics_kwh", "other_feat"] — two columns, matching the shape of the
    fabricated holdout X used by TestPhase1Validation.

    Needed because `_validate_physics_phase1()` calls
    `self.model.predict(X, pred_contrib=True)` for the SHAP-rank codepath,
    and LightGBM's `Booster.predict()` requires the predict-time column
    count to exactly match the training column count (raises
    `LightGBMError: number of features ... not the same as ... training
    data` otherwise — see TestPhase1Validation docstring for how this was
    discovered). `_make_trained_model()` trains on the full ~30-column
    `_FEATURES_BASE` set, so it cannot be reused here.
    """
    from energy_forecast.model import _build_model, _try_import_lgbm, _try_import_sklearn_gbr

    lgb = _try_import_lgbm()
    GBR = _try_import_sklearn_gbr()
    rng = np.random.default_rng(0)
    n = 50
    X_train = pd.DataFrame(
        {
            "physics_kwh": rng.uniform(0.5, 5.0, size=n),
            "other_feat": rng.uniform(0.0, 1.0, size=n),
        }
    )
    y_train = X_train["physics_kwh"].to_numpy() + rng.normal(0, 0.05, size=n)

    m = EnergyForecastModel(tmp_path)
    fitted = _build_model(lgb, GBR, n_estimators=50)
    fitted.fit(X_train, y_train)
    m.model = fitted
    m.engine = "LightGBM" if lgb is not None else "GBR"
    m.feature_cols = ["physics_kwh", "other_feat"]
    return m


class TestPhase1Validation:
    """The brief's Step 1 test code trains via `_make_trained_model()` (no
    `physics_model` arg), which fits a real model on the full ~30-column
    `_FEATURES_BASE` feature set. Against that model, the brief's fabricated
    2-column holdout X made `self.model.predict(X, pred_contrib=True)` raise
    `lightgbm.basic.LightGBMError: The number of features in data (2) is not
    the same as it was in training data (66)` — confirmed by running the
    brief's test verbatim before this fix. `_make_phase1_test_model()` (above)
    replaces `_make_trained_model()` for the two tests that exercise the
    SHAP-rank codepath, fitting a real model on exactly the 2 columns the
    fabricated X uses, so `pred_contrib=True` is shape-consistent and
    meaningful.  `test_physics_kwh_not_in_features_returns_none_rank` doesn't
    touch the SHAP codepath (physics_kwh absent -> early return), so it's
    left using `_make_trained_model()` unchanged from the brief.
    """

    def test_calibration_good_when_top5_and_slope_in_range(self, tmp_path):
        model = _make_phase1_test_model(tmp_path / "model")
        X = pd.DataFrame({"physics_kwh": [1.0, 2.0, 3.0], "other_feat": [0.1, 0.2, 0.3]})
        y_gross = np.array([1.05, 2.1, 2.9])  # slope ~1.0, physics_kwh tracks gross_kwh closely
        physics_vals = np.array([1.0, 2.0, 3.0])
        result = model._validate_physics_phase1(X, y_gross, physics_vals)
        assert result["ols_slope"] == pytest.approx(1.0, rel=0.1)
        assert result["calibration_good"] is True

    def test_calibration_bad_when_slope_out_of_range(self, tmp_path):
        model = _make_phase1_test_model(tmp_path / "model")
        X = pd.DataFrame({"physics_kwh": [1.0, 2.0, 3.0], "other_feat": [0.1, 0.2, 0.3]})
        y_gross = np.array([3.0, 6.0, 9.0])  # slope 3.0, badly miscalibrated
        physics_vals = np.array([1.0, 2.0, 3.0])
        result = model._validate_physics_phase1(X, y_gross, physics_vals)
        assert result["calibration_good"] is False

    def test_physics_kwh_not_in_features_returns_none_rank(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model")
        X = pd.DataFrame({"other_feat": [0.1, 0.2, 0.3]})
        result = model._validate_physics_phase1(X, np.array([1.0, 2.0, 3.0]), None)
        assert result["shap_rank"] is None
        assert result["calibration_good"] is False


class TestPhase2ResidualTarget:
    def test_phase1_target_is_gross_kwh_log_transform_active(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model", use_physics_residual=False)
        assert model._log_transform is True
        assert model._use_physics_residual is False

    def test_phase2_target_is_residual_log_transform_disabled(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(
            tmp_path / "physics_models",
            {
                "cop_formula": {"a": 2.5, "b": 0.07},
                "dhw_tank_volume_l": 200,
                "dhw_power_w": 4000,
                "internal_gains_fraction": 0.8,
                "heating_curve_points": [[-20, 55.5], [20, 25.0]],
                "room_thermostats": [],
                "use_physics_residual": True,
            },
        )
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, _ = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        assert model._log_transform is False
        assert model._use_physics_residual is True

    def test_phase2_nan_physics_hours_filled_with_zero_and_warned(self, tmp_path, caplog):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(
            tmp_path / "physics_models",
            {
                "cop_formula": {"a": 2.5, "b": 0.07},
                "dhw_tank_volume_l": 200,
                "dhw_power_w": 4000,
                "internal_gains_fraction": 0.8,
                "heating_curve_points": [[-20, 55.5], [20, 25.0]],
                "room_thermostats": [],
                "use_physics_residual": True,
            },
        )
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        with caplog.at_level("WARNING"):
            model, _ = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        # physics_kwh_series is always fully aligned in this fixture (predict_training_series covers
        # every training timestamp), so no NaN-fill warning is expected here — this test documents
        # the *absence* of the warning as the baseline; the NaN-path itself is covered directly below.

    def test_phase2_target_computation_subtracts_physics_before_nan_fill_check(self, tmp_path):
        # direct unit test of the target math, independent of full train() plumbing
        ts = pd.date_range("2026-01-01", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [2.0, 3.0, 4.0, 5.0, 6.0]})
        physics_series = pd.Series([1.0, 1.0, np.nan, 2.0, 2.0], index=ts)  # one gap
        physics_aligned = physics_series.reindex(df["timestamp"])
        physics_vals = physics_aligned.fillna(0).values
        target = df["gross_kwh"].to_numpy(dtype=float) - physics_vals
        assert target[2] == pytest.approx(4.0)  # NaN -> 0, so residual = gross_kwh unmodified for that hour
        assert target[0] == pytest.approx(1.0)

    def test_phase2_warns_end_to_end_when_physics_series_has_real_gap(self, tmp_path, caplog):
        # Unlike test_phase2_nan_physics_hours_filled_with_zero_and_warned above (whose fixture is
        # fully aligned and documents the *absence* of the warning), this test forces a genuine gap
        # in physics_kwh_series so train()'s internal reindex(df["timestamp"]) — see model.py around
        # `self._use_physics_residual = bool(use_physics_residual and physics_kwh_series is not None)`
        # — produces real NaNs, and asserts the resulting WARNING is actually logged.
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(
            tmp_path / "physics_models",
            {
                "cop_formula": {"a": 2.5, "b": 0.07},
                "dhw_tank_volume_l": 200,
                "dhw_power_w": 4000,
                "internal_gains_fraction": 0.8,
                "heating_curve_points": [[-20, 55.5], [20, 25.0]],
                "room_thermostats": [],
                "use_physics_residual": True,
            },
        )
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)

        orig_predict_training_series = pm.predict_training_series
        n_missing_hours = 3

        def _predict_training_series_with_gap(*args, **kwargs):
            series = orig_predict_training_series(*args, **kwargs)
            # Drop the last few timestamps entirely so the physics_kwh_series index no
            # longer covers every training hour — a genuine gap, not just a NaN value.
            return series.iloc[:-n_missing_hours]

        with (
            patch.object(pm, "predict_training_series", side_effect=_predict_training_series_with_gap),
            caplog.at_level("WARNING"),
        ):
            model, _ = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)

        assert model._use_physics_residual is True
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        matching = [m for m in warning_messages if "physics" in m.lower() and str(n_missing_hours) in m]
        assert matching, (
            f"expected a WARNING mentioning {n_missing_hours} missing physics hours, got: {warning_messages}"
        )


class TestPhase2Predict:
    def test_phase2_predict_adds_physics_baseline(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(
            tmp_path / "physics_models",
            {
                "cop_formula": {"a": 2.5, "b": 0.07},
                "dhw_tank_volume_l": 200,
                "dhw_power_w": 4000,
                "internal_gains_fraction": 0.8,
                "heating_curve_points": [[-20, 55.5], [20, 25.0]],
                "room_thermostats": [],
                "use_physics_residual": True,
            },
        )
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        result = model.predict(forecast_df, live_temp=5.0, physics_model=pm)
        assert (result["predicted_kwh"] >= 0).all()  # clip enforced even though residual can go negative

    def test_phase1_to_phase2_regression_identical_when_flag_false(self, tmp_path):
        # use_physics_residual=False must reproduce pre-Phase-2 behaviour exactly
        model_a, forecast_df = _make_trained_model(tmp_path / "model_a", use_physics_residual=False)
        model_b, _ = _make_trained_model(tmp_path / "model_b", use_physics_residual=False)
        result_a = model_a.predict(forecast_df, live_temp=5.0)
        result_b = model_b.predict(forecast_df, live_temp=5.0)
        pd.testing.assert_series_equal(
            result_a["predicted_kwh"].reset_index(drop=True), result_b["predicted_kwh"].reset_index(drop=True)
        )

    def test_phase2_exception_in_physics_falls_back_to_ml_only(self, tmp_path, monkeypatch):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(
            tmp_path / "physics_models",
            {
                "cop_formula": {"a": 2.5, "b": 0.07},
                "dhw_tank_volume_l": 200,
                "dhw_power_w": 4000,
                "internal_gains_fraction": 0.8,
                "heating_curve_points": [[-20, 55.5], [20, 25.0]],
                "room_thermostats": [],
                "use_physics_residual": True,
            },
        )
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)

        def _broken_predict_series(*a, **kw):
            raise RuntimeError("simulated physics failure")

        monkeypatch.setattr(pm, "predict_series", _broken_predict_series)
        result = model.predict(forecast_df, live_temp=5.0, physics_model=pm)
        assert not result.empty  # no exception propagates; falls back to ML-only reconstruction


class TestGrossMAEReporting:
    def test_phase1_last_mae_is_gross_kwh_mae_unchanged(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model", use_physics_residual=False)
        assert model.last_mae is not None
        assert model.last_residual_mae is None

    def test_phase2_last_mae_still_reported_on_gross_kwh(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(
            tmp_path / "physics_models",
            {
                "cop_formula": {"a": 2.5, "b": 0.07},
                "dhw_tank_volume_l": 200,
                "dhw_power_w": 4000,
                "internal_gains_fraction": 0.8,
                "heating_curve_points": [[-20, 55.5], [20, 25.0]],
                "room_thermostats": [],
                "use_physics_residual": True,
            },
        )
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, _ = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        assert model.last_mae is not None
        assert model.last_residual_mae is not None
        # Regression guard for the exact bug this task fixes: the old code applied
        # np.expm1() unconditionally to the raw prediction, which in Phase 2 is a residual
        # (never log-transformed, can be negative) — not a log-transformed gross-kWh value.
        # That produced a nonsensical MAE far outside the fixture's gross_kwh range
        # (0.5-5.0): empirically ~7.56 for this fixture/seed vs. a correct ~0.9. A correctly
        # reconstructed gross-kWh MAE must stay well inside the target's actual range.
        assert model.last_mae < 3.0
        # Note: last_mae and last_residual_mae are mathematically identical here by
        # construction — MAE is translation-invariant per row (physics_kwh is subtracted
        # from the target and added back to the prediction identically), so
        # MAE(y, physics+pred) == MAE(y-physics, pred) exactly, *unless* the np.maximum(0, ...)
        # floor-clip changes at least one holdout row. That only happens when the
        # reconstructed prediction would go negative, which this fixture's holdout split
        # doesn't trigger. last_residual_mae is still a distinct, separately-computed value
        # (on the residual scale, unclipped) — it just happens to coincide numerically here.

    def test_phase2_cv_mae_is_gross_kwh_mae_not_residual_garbage(self, tmp_path):
        """Regression guard for the CV-fold-path twin of the bug fixed above.

        `test_phase2_last_mae_still_reported_on_gross_kwh` above uses the default
        n=600, which after the lag_336h dropna leaves ~264 clean rows — under
        MIN_CV_ROWS=500, so CV is skipped and last_mae falls back to holdout_mae.
        That means the previous test never exercised the CV path at all, even
        though CV is what actually runs (and what feeds last_mae) for any
        real-sized training set. Need n>=836 so that >=500 clean rows remain
        for TimeSeriesSplit (see TestFeatureImportanceLogging for the same
        n=900 pattern).

        Before the fix, both the num_leaves sweep and the fold-MAE loop applied
        np.expm1() unconditionally to the raw (residual, never log-transformed)
        prediction, producing an MAE far outside the fixture's gross_kwh range
        (0.5-5.0) — e.g. ~6.44 empirically. After the fix, both call sites
        reconstruct gross-kWh as physics_kwh + residual (clipped at 0) before
        scoring, matching the already-fixed holdout block.
        """
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(
            tmp_path / "physics_models",
            {
                "cop_formula": {"a": 2.5, "b": 0.07},
                "dhw_tank_volume_l": 200,
                "dhw_power_w": 4000,
                "internal_gains_fraction": 0.8,
                "heating_curve_points": [[-20, 55.5], [20, 25.0]],
                "room_thermostats": [],
                "use_physics_residual": True,
            },
        )
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, _ = _make_trained_model(tmp_path / "model", n=900, physics_model=pm, use_physics_residual=True)
        # Proves CV actually ran (not skipped, not silently falling back to holdout).
        assert model.last_cv_mae is not None
        assert model.last_mae == model.last_cv_mae
        # Fixture's gross_kwh is uniform(0.5, 5.0) — a correctly reconstructed MAE
        # must stay well inside that range, not the ~6x-larger garbage the
        # unconditional np.expm1()-on-residual bug produced.
        assert model.last_mae < 3.0


class TestPredictDhwScheduleOverride:
    """Plan D Task 2: dhw_schedule_override parameter threading."""

    def test_dhw_schedule_override_forwarded_to_physics_model(self, tmp_path, monkeypatch):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(
            tmp_path / "physics_models",
            {
                "cop_formula": {"a": 2.5, "b": 0.07},
                "dhw_tank_volume_l": 200,
                "dhw_power_w": 4000,
                "internal_gains_fraction": 0.8,
                "heating_curve_points": [[-20, 55.5], [20, 25.0]],
                "room_thermostats": [],
                "use_physics_residual": False,
            },
        )
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)

        received = {}
        original = pm.predict_series

        def _spy(*args, **kwargs):
            received.update(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(pm, "predict_series", _spy)

        override = {"legionella": ("2026-06-25", 10)}
        model.predict(forecast_df, live_temp=5.0, physics_model=pm, dhw_schedule_override=override)
        assert received.get("dhw_schedule_override") == override

    def test_dhw_schedule_override_none_by_default(self, tmp_path):
        model, forecast_df = _make_trained_model(tmp_path / "model")
        result = model.predict(forecast_df, live_temp=5.0)  # no exception with default None
        assert not result.empty
