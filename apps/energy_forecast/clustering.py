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
    from sklearn.metrics import silhouette_score as _silhouette_score
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

    def fit(self, df: pd.DataFrame, sample_weight: pd.Series | None = None) -> pd.Series | None:
        """Find clusters in hourly energy data.

        Args:
            df: DataFrame with 'timestamp' and 'gross_kwh'.
            sample_weight: Optional Series of daily weights indexed by date.

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
            
            # Prepare sample weights for valid days
            fit_kwargs = {}
            if sample_weight is not None:
                # Align weights to pivoted index; fill missing with the mean weight so
                # unmatched days get a neutral contribution rather than zero weight.
                # A zero-weight day causes sample_weight.sum()==0 → KMeans NaN divide.
                mean_w = sample_weight.mean() if len(sample_weight) > 0 else 1.0
                w = sample_weight.reindex(pivoted.index).fillna(mean_w).values
                # Final safety guard: if all weights are zero or NaN, drop them entirely
                if not np.isfinite(w).all() or w.sum() == 0:
                    _LOGGER.warning(
                        "Regime Clustering: sample_weight is invalid (all-zero or NaN) — "
                        "falling back to uniform weights."
                    )
                else:
                    fit_kwargs["sample_weight"] = w

            # 2. Fit KMeans
            # Use raw values to capture both shape and magnitude (e.g. high-heating vs low-heating)
            n = min(self.n_clusters, len(pivoted))
            km = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = km.fit_predict(pivoted, **fit_kwargs)
            
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

    def fit(self, daily_features: pd.DataFrame, labels: pd.Series, sample_weight: pd.Series | None = None):
        """Train the regime classifier.

        Args:
            daily_features: DataFrame indexed by date with weather/calendar features.
            labels: Series of cluster IDs indexed by date.
            sample_weight: Optional Series of daily weights indexed by date.
        """
        if not SKLEARN_AVAILABLE:
            return

        try:
            # Ensure indices match
            common_idx = daily_features.index.intersection(labels.index)
            X = daily_features.loc[common_idx]
            y = labels.loc[common_idx]
            
            fit_kwargs = {}
            if sample_weight is not None:
                mean_w = sample_weight.mean() if len(sample_weight) > 0 else 1.0
                w = sample_weight.reindex(common_idx).fillna(mean_w).values
                if np.isfinite(w).all() and w.sum() > 0:
                    fit_kwargs["sample_weight"] = w

            if len(X) < 14:
                return

            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                min_samples_leaf=3,
                oob_score=True,
            )
            self.model.fit(X, y, **fit_kwargs)
            self.is_fitted = True

            train_acc = self.model.score(X, y)
            oob_acc = self.model.oob_score_
            _LOGGER.info(
                "Regime Predictor trained — train acc: %.2f, OOB acc: %.2f (n=%d, k=%d)",
                train_acc, oob_acc, len(X), len(y.unique()),
            )
            if oob_acc < 0.5:
                _LOGGER.warning(
                    "Regime Predictor OOB accuracy %.2f is near chance — regime signal may be "
                    "unreliable. More training history needed.", oob_acc
                )
            
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


def find_optimal_k(
    energy_df: pd.DataFrame,
    daily_features: pd.DataFrame,
    sample_weight: "pd.Series | None" = None,
    k_range: tuple = (2, 8),
) -> int:
    """Return K ∈ k_range that maximises silhouette_score × OOB_accuracy.

    Activated when regime_count=0 in apps.yaml. Falls back to k_range[0] on
    insufficient data or sklearn unavailability.
    """
    if not SKLEARN_AVAILABLE or energy_df.empty:
        return k_range[0]

    try:
        # Pivot energy_df to daily 24h profiles (mirrors DailyProfileClusterer.fit logic)
        daily = energy_df.copy()
        daily["date"] = daily["timestamp"].dt.date
        daily["hour"] = daily["timestamp"].dt.hour
        counts = daily.groupby("date")["hour"].count()
        valid_days = counts[counts >= 22].index
        if len(valid_days) < 14:
            _LOGGER.info(
                "Auto-K: not enough history (%d/14 days) — using K=%d.",
                len(valid_days), k_range[0],
            )
            return k_range[0]

        pivoted = daily[daily["date"].isin(valid_days)].pivot(
            index="date", columns="hour", values="gross_kwh"
        )
        pivoted = pivoted.reindex(columns=range(24)).interpolate(axis=1).ffill(axis=1).bfill(axis=1)

        fit_kwargs: dict = {}
        if sample_weight is not None:
            mean_w = sample_weight.mean() if len(sample_weight) > 0 else 1.0
            w = sample_weight.reindex(pivoted.index).fillna(mean_w).values
            if np.isfinite(w).all() and w.sum() > 0:
                fit_kwargs["sample_weight"] = w

        best_k = k_range[0]
        best_score = -1.0
        k_lo, k_hi = k_range

        for k in range(k_lo, min(k_hi, len(pivoted)) + 1):
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_arr = km.fit_predict(pivoted, **fit_kwargs)
                labels_series = pd.Series(labels_arr, index=pivoted.index)

                unique = np.unique(labels_arr)
                sil = float(_silhouette_score(pivoted, labels_arr)) if len(unique) >= 2 else 0.0

                predictor = RegimePredictor()
                predictor.fit(daily_features, labels_series, sample_weight=sample_weight)
                oob = predictor.model.oob_score_ if predictor.is_fitted else 0.0

                score = sil * oob
                _LOGGER.debug(
                    "Auto-K: K=%d  silhouette=%.3f  oob=%.3f  score=%.4f",
                    k, sil, oob, score,
                )
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                _LOGGER.debug("Auto-K: K=%d evaluation failed: %s", k, e)
                continue

        _LOGGER.info("Auto-K selected K=%d (score=%.4f).", best_k, best_score)
        return best_k

    except Exception as e:
        _LOGGER.error("Auto-K selection failed: %s — using K=%d.", e, k_range[0])
        return k_range[0]
