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
    """Return K ∈ k_range selected by the inertia elbow (second derivative).

    Fits KMeans at each K, computes within-cluster inertia, then picks the K
    where the marginal inertia drop accelerates most — the "knee" of the elbow
    curve. Unlike silhouette, this metric is unbiased toward small K and
    naturally selects K=4–6 for real energy profiles.

    After selecting K, fits a RegimePredictor and logs OOB accuracy as INFO
    (informational only — no gating on OOB).

    Falls back to k_range[0] on insufficient data or sklearn unavailability.
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

        k_lo, k_hi = k_range
        k_max = min(k_hi, len(pivoted))
        k_values = list(range(k_lo, k_max + 1))

        # Edge case: only one candidate → return it directly
        if len(k_values) == 1:
            return k_values[0]

        # First pass: collect inertias for all K
        inertias: dict[int, float] = {}
        for k in k_values:
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(pivoted, **fit_kwargs)
                inertias[k] = float(km.inertia_)
                _LOGGER.debug("Auto-K: K=%d  inertia=%.1f", k, inertias[k])
            except Exception as e:
                _LOGGER.debug("Auto-K: K=%d fit failed: %s", k, e)

        if not inertias:
            return k_lo

        available_k = sorted(inertias)

        # Edge case: only 2 candidates → return the higher one
        if len(available_k) == 2:
            selected_k = available_k[1]
        else:
            # Elbow = argmax of second differences (interior K values only)
            # d2[k] = inertia[k-1] - 2*inertia[k] + inertia[k+1]
            best_k = available_k[1]   # default: second candidate
            best_d2 = -np.inf
            for i in range(1, len(available_k) - 1):
                k_prev, k_cur, k_next = available_k[i - 1], available_k[i], available_k[i + 1]
                d2 = inertias[k_prev] - 2 * inertias[k_cur] + inertias[k_next]
                _LOGGER.debug("Auto-K: K=%d  d2=%.2f", k_cur, d2)
                if d2 > best_d2:
                    best_d2 = d2
                    best_k = k_cur
            selected_k = best_k

        # Second pass (informational): log OOB for selected K
        try:
            km_sel = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
            labels_arr = km_sel.fit_predict(pivoted, **fit_kwargs)
            labels_series = pd.Series(labels_arr, index=pivoted.index)
            predictor = RegimePredictor()
            predictor.fit(daily_features, labels_series, sample_weight=sample_weight)
            oob = predictor.model.oob_score_ if predictor.is_fitted else float("nan")
            _LOGGER.info(
                "Auto-K selected K=%d (inertia=%.1f, predictor OOB=%.2f).",
                selected_k, inertias[selected_k], oob,
            )
        except Exception:
            _LOGGER.info("Auto-K selected K=%d (inertia=%.1f).", selected_k, inertias[selected_k])

        return selected_k

    except Exception as e:
        _LOGGER.error("Auto-K selection failed: %s — using K=%d.", e, k_range[0])
        return k_range[0]
