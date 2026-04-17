"""Tests for the Daily Regime Clustering module."""
import pandas as pd
import numpy as np
import pytest
from apps.energy_forecast.clustering import DailyProfileClusterer, RegimePredictor, SKLEARN_AVAILABLE

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
