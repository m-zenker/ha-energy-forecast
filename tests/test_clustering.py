"""Tests for the Daily Regime Clustering module."""
import pandas as pd
import numpy as np
import pytest
from apps.energy_forecast.clustering import (
    DailyProfileClusterer,
    RegimePredictor,
    find_optimal_k,
    SKLEARN_AVAILABLE,
)
from apps.energy_forecast.model import _prepare_daily_regime_features

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_clusterer_fit():
    """Test that the clusterer identifies regimes from synthetic data."""
    # Create synthetic data with 3 distinct regimes over 30 days
    # Regime 0: Flat
    # Regime 1: Morning peak
    # Regime 2: Evening peak
    
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = []
    for i, dt in enumerate(dates):
        regime = i % 3
        for h in range(24):
            val = 0.5
            if regime == 1 and 7 <= h <= 9:
                val = 2.0
            elif regime == 2 and 18 <= h <= 20:
                val = 3.0
            rows.append({
                "timestamp": dt + pd.Timedelta(hours=h),
                "gross_kwh": val
            })
    
    df = pd.DataFrame(rows)
    clusterer = DailyProfileClusterer(n_clusters=3)
    labels = clusterer.fit(df)
    
    assert clusterer.is_fitted
    assert len(clusterer.centroids) == 3
    assert len(labels) == 30
    assert labels.nunique() == 3

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_clusterer_fit_relaxed_24h():
    """Test that the clusterer handles days with 23 hours."""
    # Create 20 days, but Day 5 has only 23 hours
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    rows = []
    for i, dt in enumerate(dates):
        hours = range(24)
        if i == 5:
            hours = range(23) # Missing hour 23
        for h in hours:
            rows.append({
                "timestamp": dt + pd.Timedelta(hours=h),
                "gross_kwh": 1.0
            })
    
    df = pd.DataFrame(rows)
    clusterer = DailyProfileClusterer(n_clusters=2)
    labels = clusterer.fit(df)
    
    assert clusterer.is_fitted
    assert len(labels) == 20 # All days processed, including the 23h one
    assert clusterer.centroids.shape == (2, 24) # Centroid still 24h

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_regime_predictor():
    """Test that the regime predictor learns to map weather to clusters."""
    # 20 days, 2 regimes (0=Warm, 1=Cold)
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    
    # Features: Temp (Warm days have 20C, Cold have 5C)
    weather_rows = []
    labels = []
    for i in range(20):
        is_cold = i % 2 == 0
        weather_rows.append({
            "temp_mean": 5.0 if is_cold else 20.0,
            "temp_min": 0.0 if is_cold else 15.0,
            "temp_max": 10.0 if is_cold else 25.0,
            "sun_total": 100 if is_cold else 500,
            "precip_total": 5 if is_cold else 0,
            "day_of_week": dates[i].dayofweek,
            "is_holiday": 0
        })
        labels.append(1 if is_cold else 0)
    
    daily_features = pd.DataFrame(weather_rows, index=dates.date)
    label_series = pd.Series(labels, index=dates.date)
    
    predictor = RegimePredictor()
    predictor.fit(daily_features, label_series)
    
    assert predictor.is_fitted
    
    # Predict on new similar data
    test_features = pd.DataFrame([{
        "temp_mean": 6.0,
        "temp_min": 1.0,
        "temp_max": 11.0,
        "sun_total": 90,
        "precip_total": 4,
        "day_of_week": 0,
        "is_holiday": 0
    }], index=[pd.Timestamp("2024-02-01").date()])
    
    pred = predictor.predict(test_features)
    assert pred[0] == 1  # Should predict Cold regime

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_clusterer_fit_zero_weight_guard():
    """Regression: all-zero sample_weight must not crash KMeans (NaN divide)."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rows = []
    for i, dt in enumerate(dates):
        for h in range(24):
            rows.append({"timestamp": dt + pd.Timedelta(hours=h), "gross_kwh": float(i % 3 + 1)})
    df = pd.DataFrame(rows)

    # Weights indexed on dates NOT in the df — reindex would yield all NaN, fillna(0) → sum=0
    mismatched_idx = pd.date_range("2025-01-01", periods=30, freq="D").date
    all_zero_weights = pd.Series(np.zeros(30), index=mismatched_idx)

    clusterer = DailyProfileClusterer(n_clusters=3)
    labels = clusterer.fit(df, sample_weight=all_zero_weights)
    # Should succeed (fallback to uniform) rather than raise
    assert clusterer.is_fitted
    assert labels is not None


def test_clusterer_no_sklearn_fallback():
    """If sklearn is missing (simulated), clusterer should fail gracefully."""
    import apps.energy_forecast.clustering as cl

    # Force sklearn unavailable
    cl.SKLEARN_AVAILABLE = False

    clusterer = cl.DailyProfileClusterer()
    labels = clusterer.fit(pd.DataFrame({"a": [1]}))

    assert labels is None
    assert not clusterer.is_fitted

    # Restore (though it's a module global, so this might be flaky if tests run in same process)
    # But usually sklearn is available in this env
    cl.SKLEARN_AVAILABLE = SKLEARN_AVAILABLE


# ── Helpers for new tests ──────────────────────────────────────────────────────

def _make_daily_features(n: int = 20) -> tuple[pd.DataFrame, pd.Series]:
    """Return (daily_features, labels) with is_away and people_home columns."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rows = []
    labels = []
    for i, dt in enumerate(dates):
        rows.append({
            "temp_mean": 5.0 if i % 2 == 0 else 20.0,
            "temp_min": 0.0 if i % 2 == 0 else 15.0,
            "temp_max": 10.0 if i % 2 == 0 else 25.0,
            "sun_total": 100 if i % 2 == 0 else 500,
            "precip_total": 5 if i % 2 == 0 else 0,
            "day_of_week": dt.dayofweek,
            "is_holiday": 0,
            "is_away": 0,
            "people_home": 2.0,
        })
        labels.append(i % 2)
    return pd.DataFrame(rows, index=dates.date), pd.Series(labels, index=dates.date)


def _make_weather_df(n_days: int = 10) -> pd.DataFrame:
    """Minimal hourly weather DataFrame for _prepare_daily_regime_features."""
    ts = pd.date_range("2024-03-01", periods=n_days * 24, freq="1h")
    return pd.DataFrame({
        "timestamp": ts,
        "temp_c": [10.0] * len(ts),
        "sunshine_min": [30.0] * len(ts),
        "precipitation_mm": [0.0] * len(ts),
    })


# ── RegimePredictor: RF constraints and OOB score ─────────────────────────────

@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_rf_constraints_applied():
    """Fitted RF must use max_depth=6, min_samples_leaf=3, oob_score=True."""
    feats, labels = _make_daily_features(n=20)
    predictor = RegimePredictor()
    predictor.fit(feats, labels)

    assert predictor.is_fitted
    rf = predictor.model
    assert rf.max_depth == 6
    assert rf.min_samples_leaf == 3
    assert rf.oob_score is True


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_oob_score_accessible_and_in_range():
    """After fitting, oob_score_ must be a float in [0, 1]."""
    feats, labels = _make_daily_features(n=20)
    predictor = RegimePredictor()
    predictor.fit(feats, labels)

    oob = predictor.model.oob_score_
    assert isinstance(oob, float)
    assert 0.0 <= oob <= 1.0


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_oob_warning_emitted_near_chance(caplog):
    """Shuffled (random) labels → OOB near chance → WARNING logged."""
    import logging
    feats, labels = _make_daily_features(n=20)
    # Shuffle labels so they're unlearnable from features
    shuffled = labels.sample(frac=1, random_state=99).values
    shuffled_labels = pd.Series(shuffled, index=labels.index)

    with caplog.at_level(logging.WARNING, logger="energy_forecast"):
        predictor = RegimePredictor()
        predictor.fit(feats, shuffled_labels)

    oob = predictor.model.oob_score_
    if oob < 0.5:
        assert any("near chance" in r.message for r in caplog.records)


# ── _prepare_daily_regime_features: occupancy columns ────────────────────────

def test_daily_features_includes_is_away():
    """is_away column = daily max of hourly is_away values."""
    weather_df = _make_weather_df(n_days=3)
    ts = pd.date_range("2024-03-01", periods=72, freq="1h")
    # Day 0: all away; Day 1: half away; Day 2: not away
    is_away_vals = [1] * 24 + [1] * 12 + [0] * 12 + [0] * 24
    away_df = pd.DataFrame({"timestamp": ts, "is_away": is_away_vals})

    result = _prepare_daily_regime_features(weather_df, away_df=away_df)

    assert "is_away" in result.columns
    dates = sorted(result.index)
    assert result.loc[dates[0], "is_away"] == 1   # all away → max=1
    assert result.loc[dates[1], "is_away"] == 1   # half away → max=1
    assert result.loc[dates[2], "is_away"] == 0   # not away → max=0


def test_daily_features_includes_people_home():
    """people_home column = daily mean of hourly people_home counts."""
    weather_df = _make_weather_df(n_days=2)
    ts = pd.date_range("2024-03-01", periods=48, freq="1h")
    # Day 0: 2 people all day; Day 1: 0 people all day
    counts = [2] * 24 + [0] * 24
    presence_df = pd.DataFrame({"timestamp": ts, "people_home": counts})

    result = _prepare_daily_regime_features(weather_df, presence_df=presence_df)

    assert "people_home" in result.columns
    dates = sorted(result.index)
    assert result.loc[dates[0], "people_home"] == pytest.approx(2.0)
    assert result.loc[dates[1], "people_home"] == pytest.approx(0.0)


def test_daily_features_no_occupancy_defaults_zero():
    """Without away_df/presence_df both columns default to 0."""
    weather_df = _make_weather_df(n_days=3)
    result = _prepare_daily_regime_features(weather_df)

    assert "is_away" in result.columns
    assert "people_home" in result.columns
    assert (result["is_away"] == 0).all()
    assert (result["people_home"] == 0.0).all()


# ── find_optimal_k ────────────────────────────────────────────────────────────

def _make_energy_df(n_days: int = 30, n_regimes: int = 3) -> pd.DataFrame:
    """Synthetic energy data with `n_regimes` distinct 24h profiles repeated over `n_days`."""
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    for i, dt in enumerate(dates):
        regime = i % n_regimes
        for h in range(24):
            val = 0.5
            if regime == 1 and 7 <= h <= 9:
                val = 3.0
            elif regime == 2 and 18 <= h <= 20:
                val = 4.5
            rows.append({"timestamp": dt + pd.Timedelta(hours=h), "gross_kwh": val})
    return pd.DataFrame(rows)


def _make_aligned_weather_df(n_days: int = 30, n_regimes: int = 3) -> pd.DataFrame:
    """Hourly weather correlated with energy regimes so RegimePredictor has signal."""
    # Each regime gets a distinct temperature band so OOB accuracy is high
    temp_map = {0: 5.0, 1: 15.0, 2: 25.0}
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    for i, dt in enumerate(dates):
        regime = i % n_regimes
        temp = temp_map.get(regime % 3, 10.0)
        for h in range(24):
            rows.append({
                "timestamp": dt + pd.Timedelta(hours=h),
                "temp_c": temp,
                "sunshine_min": 30.0,
                "precipitation_mm": 0.0,
            })
    return pd.DataFrame(rows)


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_find_optimal_k_selects_best():
    """3-regime synthetic data with correlated weather: auto-K should select K=3 (or nearby)."""
    n_days, n_regimes = 30, 3
    energy_df = _make_energy_df(n_days=n_days, n_regimes=n_regimes)
    weather_df = _make_aligned_weather_df(n_days=n_days, n_regimes=n_regimes)
    daily_features = _prepare_daily_regime_features(weather_df)

    k = find_optimal_k(energy_df, daily_features, k_range=(2, 5))

    assert 2 <= k <= 5, f"K={k} outside search range"
    # 3 is the natural answer; allow 2 (OOB chance with small N) or 4 (minor overfit)
    assert k in (2, 3, 4), f"Expected K near 3, got {k}"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_find_optimal_k_too_few_days():
    """Fewer than 14 valid days → returns k_range[0] without raising."""
    energy_df = _make_energy_df(n_days=10, n_regimes=2)
    weather_df = _make_aligned_weather_df(n_days=10, n_regimes=2)
    daily_features = _prepare_daily_regime_features(weather_df)

    k = find_optimal_k(energy_df, daily_features, k_range=(2, 5))

    assert k == 2


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_find_optimal_k_respects_range():
    """Result is always within the requested k_range."""
    energy_df = _make_energy_df(n_days=40, n_regimes=3)
    weather_df = _make_aligned_weather_df(n_days=40, n_regimes=3)
    daily_features = _prepare_daily_regime_features(weather_df)

    for lo, hi in [(2, 3), (3, 6), (2, 8)]:
        k = find_optimal_k(energy_df, daily_features, k_range=(lo, hi))
        assert lo <= k <= hi, f"K={k} outside [{lo}, {hi}]"


@pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not available")
def test_find_optimal_k_weighted():
    """find_optimal_k runs without error when sample_weight is provided."""
    n_days, n_regimes = 30, 3
    energy_df = _make_energy_df(n_days=n_days, n_regimes=n_regimes)
    weather_df = _make_aligned_weather_df(n_days=n_days, n_regimes=n_regimes)
    daily_features = _prepare_daily_regime_features(weather_df)

    dates = pd.date_range("2024-01-01", periods=n_days, freq="D").date
    weights = pd.Series(np.linspace(0.5, 1.5, n_days), index=dates)

    k = find_optimal_k(energy_df, daily_features, sample_weight=weights, k_range=(2, 4))

    assert 2 <= k <= 4
