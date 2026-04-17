"""Optional Daily Regime Clustering and Prediction.

This module clusters historical 24-hour energy consumption profiles into $K$
typical 'regimes' (e.g., Workday, Weekend, High-Heating) and provides a
secondary model to predict tomorrow's regime from weather and calendar.

Designed as an optional dependency: if scikit-learn is missing, it falls back
gracefully to a no-op implementation.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

_LOGGER = logging.getLogger("energy_forecast")

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    _LOGGER.warning("scikit-learn not found. Daily Regime Clustering will be disabled.")


class DailyProfileClusterer:
    """Clusters historical 24h consumption profiles into typical regimes."""

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.centroids: np.ndarray | None = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> pd.Series | None:
        """Find clusters in hourly energy data.

        Args:
            df: DataFrame with 'timestamp' and 'gross_kwh'.

        Returns:
            A Series of cluster labels indexed by date, or None if failed.
        """
        if not SKLEARN_AVAILABLE or df.empty:
            return None

        try:
            # 1. Reshape to daily profiles
            daily = df.copy()
            daily["date"] = daily["timestamp"].dt.date
            daily["hour"] = daily["timestamp"].dt.hour
            
            # Keep only days with at least 22 hours
            counts = daily.groupby("date")["hour"].count()
            valid_days = counts[counts >= 22].index
            if len(valid_days) < 14:  # Minimum 2 weeks for meaningful clustering
                _LOGGER.info(f"Not enough history for clustering ({len(valid_days)}/14 days).")
                return None

            pivoted = daily[daily["date"].isin(valid_days)].pivot(
                index="date", columns="hour", values="gross_kwh"
            )
            
            # Interpolate to fill missing hours (up to 2 per day)
            pivoted = pivoted.reindex(columns=range(24)).interpolate(axis=1).ffill(axis=1).bfill(axis=1)
            
            # 2. Fit KMeans
            # Use raw values to capture both shape and magnitude (e.g. high-heating vs low-heating)
            n = min(self.n_clusters, len(pivoted))
            km = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = km.fit_predict(pivoted)
            
            self.centroids = km.cluster_centers_
            self.is_fitted = True
            
            _LOGGER.info(f"Regime Clustering: Identified {n} profiles from {len(pivoted)} days.")
            return pd.Series(labels, index=pivoted.index)
            
        except Exception as e:
            _LOGGER.error(f"Regime Clustering failed: {e}")
            return None

    def get_centroid(self, cluster_id: int) -> np.ndarray:
        """Return the 24h profile for a given cluster."""
        if self.centroids is not None and 0 <= cluster_id < len(self.centroids):
            return self.centroids[cluster_id]
        return np.zeros(24)


class RegimePredictor:
    """Predicts which cluster tomorrow belongs to using daily weather/calendar."""

    def __init__(self):
        self.model: Any = None
        self.is_fitted = False

    def fit(self, daily_features: pd.DataFrame, labels: pd.Series):
        """Train the regime classifier.

        Args:
            daily_features: DataFrame indexed by date with weather/calendar features.
            labels: Series of cluster IDs indexed by date.
        """
        if not SKLEARN_AVAILABLE:
            return

        try:
            # Ensure indices match
            common_idx = daily_features.index.intersection(labels.index)
            X = daily_features.loc[common_idx]
            y = labels.loc[common_idx]

            if len(X) < 14:
                return

            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            self.is_fitted = True
            
            acc = self.model.score(X, y)
            _LOGGER.info(f"Regime Predictor trained. Training accuracy: {acc:.2f}")
            
        except Exception as e:
            _LOGGER.error(f"Regime Predictor training failed: {e}")

    def predict(self, daily_features: pd.DataFrame) -> np.ndarray:
        """Predict cluster IDs for the given daily features."""
        if not self.is_fitted or self.model is None:
            return np.full(len(daily_features), -1, dtype=int)
        
        try:
            return self.model.predict(daily_features)
        except Exception as e:
            _LOGGER.error(f"Regime Prediction failed: {e}")
            return np.full(len(daily_features), -1, dtype=int)
