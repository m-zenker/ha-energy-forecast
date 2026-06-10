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

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger("energy_forecast")

try:
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score

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

    def fit(
        self,
        df: pd.DataFrame,
        sample_weight: pd.Series | None = None,
        ev_day_dates: set | None = None,
    ) -> pd.Series | None:
        """Find clusters in hourly energy data.

        Args:
            df: DataFrame with 'timestamp' and 'gross_kwh'.
            sample_weight: Accepted but ignored — KMeans runs unweighted to avoid
                exponential decay distorting cluster geometry. Kept for API compatibility.
            ev_day_dates: Optional set of dates (datetime.date) that contain EV charging
                sessions. These days are excluded from centroid fitting so that the
                evening-peaked EV residual shape does not distort thermal/behavioural
                clusters. EV days are still assigned to their nearest centroid afterward
                so that every day gets a valid regime label.

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

            # Filter days with too much missing data
            daily_counts = daily.groupby("date")["gross_kwh"].count()
            valid_days = daily_counts[daily_counts >= 18].index
            if len(valid_days) < 14:  # Minimum 2 weeks for meaningful clustering
                _LOGGER.info(f"Not enough history for clustering ({len(valid_days)}/14 days).")
                return None

            pivoted = daily[daily["date"].isin(valid_days)].pivot(index="date", columns="hour", values="gross_kwh")

            # Interpolate to fill missing hours (up to 6 per day)
            pivoted = pivoted.reindex(columns=range(24)).interpolate(axis=1, limit=6).ffill(axis=1).bfill(axis=1)

            # 2. Determine which days to fit on (exclude EV days from centroid learning)
            ev_set = ev_day_dates or set()
            fit_index = [d for d in pivoted.index if d not in ev_set]
            ev_excluded = len(pivoted) - len(fit_index)

            if ev_excluded > 0 and len(fit_index) < 14:
                _LOGGER.info(
                    "Clustering: only %d non-EV days (< 14); including all %d days for fitting.",
                    len(fit_index),
                    len(pivoted),
                )
                fit_index = list(pivoted.index)
                ev_excluded = 0

            if ev_excluded > 0:
                _LOGGER.info("Clustering: excluding %d EV days from centroid fitting.", ev_excluded)

            piv_fit = pivoted.loc[fit_index]

            # 3. Fit KMeans on non-EV days only (unweighted — exponential weights distort geometry)
            n = min(self.n_clusters, len(piv_fit))
            km = KMeans(n_clusters=n, random_state=42, n_init=10)
            km.fit(piv_fit)

            self.centroids = km.cluster_centers_
            self.is_fitted = True

            # 4. Assign ALL valid days (including EV days) to their nearest centroid
            all_labels = km.predict(pivoted)

            _LOGGER.info(f"Regime Clustering: Identified {n} profiles from {len(piv_fit)} days.")
            return pd.Series(all_labels, index=pivoted.index)

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

    def fit(
        self,
        daily_features: pd.DataFrame,
        labels: pd.Series,
        sample_weight: pd.Series | None = None,
        verbose: bool = True,
    ):
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

            # TimeSeriesSplit CV measures forward-generalization (OOB alone
            # uses bootstrap which ignores temporal ordering).
            n_splits = min(3, max(2, len(X) // 30))
            tscv = TimeSeriesSplit(n_splits=n_splits)
            clf_proto = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6, min_samples_leaf=3)
            tscv_scores = cross_val_score(clf_proto, X, y, cv=tscv, scoring="accuracy")
            if verbose:
                dist = y.value_counts(normalize=True).sort_index()
                dist_str = ", ".join(f"{k}: {v:.0%}" for k, v in dist.items())
                _LOGGER.info(
                    "Regime Predictor trained — train acc: %.2f, OOB=%.2f, "
                    "TimeSeriesCV=%.2f±%.2f (n=%d, k=%d) | class dist: %s",
                    train_acc,
                    oob_acc,
                    tscv_scores.mean(),
                    tscv_scores.std(),
                    len(X),
                    len(y.unique()),
                    dist_str,
                )
            if tscv_scores.mean() < 0.5:
                _LOGGER.warning(
                    "Regime Predictor TimeSeriesCV accuracy %.2f is near chance — regime signal "
                    "may be unreliable. More training history needed.",
                    tscv_scores.mean(),
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
    sample_weight: pd.Series | None = None,
    k_range: tuple = (2, 8),
    ev_day_dates: set | None = None,
) -> int:
    """Return K ∈ k_range selected by the inertia elbow (second derivative).

    Algorithm
    ---------
    1. Fit KMeans at each K in k_range; record within-cluster inertia.
    2. Normalize inertias to [0, 1] by (i - min) / (max - min).
       Bail-out: if max - min < 1e-6 (all K values nearly identical, data is
       homogeneous), return k_range[0] immediately with a WARNING.
    3. Apply 3-point smoothing (rolling average on interior points) to reduce
       noise before computing the discrete second derivative.
    4. Compute d²inertia/dK² at each interior K. Gather all K whose d2 is
       within 10 % of the global max — the "tolerance band". Among those
       candidates, prefer the lowest K (parsimony tie-breaker).
    5. After selecting K, fit a RegimePredictor and log OOB *and* TimeSeriesCV
       accuracy as INFO — purely informational; no selection gate on OOB.

    Falls back to k_range[0] on insufficient data (< 14 valid days) or if
    scikit-learn is unavailable.
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
                len(valid_days),
                k_range[0],
            )
            return k_range[0]

        # Exclude EV days from inertia computation so EV session timing does not
        # inflate the apparent number of meaningful clusters.
        if ev_day_dates:
            fit_days = [d for d in valid_days if d not in ev_day_dates]
            if len(fit_days) >= 14:
                valid_days = fit_days
            else:
                _LOGGER.info(
                    "Auto-K: only %d non-EV days (< 14); including all days for K selection.",
                    len(fit_days),
                )

        pivoted = daily[daily["date"].isin(valid_days)].pivot(index="date", columns="hour", values="gross_kwh")
        pivoted = pivoted.reindex(columns=range(24)).interpolate(axis=1).ffill(axis=1).bfill(axis=1)

        k_lo, k_hi = k_range
        k_max = min(k_hi, len(pivoted))
        k_values = list(range(k_lo, k_max + 1))

        # Edge case: only one candidate → return it directly
        if len(k_values) == 1:
            return k_values[0]

        # First pass: collect inertias for all K (unweighted — weights distort geometry)
        raw_inertias: dict[int, float] = {}
        for k in k_values:
            try:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(pivoted)
                raw_inertias[k] = float(km.inertia_)
            except Exception as e:
                _LOGGER.debug("Auto-K: K=%d fit failed: %s", k, e)

        if not raw_inertias:
            return k_lo

        # Normalize inertias to 0-1 range; bail out if data is homogeneous
        # (range < 1e-6 means all K values look identical — elbow is arbitrary)
        min_i, max_i = min(raw_inertias.values()), max(raw_inertias.values())
        if max_i - min_i < 1e-6:
            _LOGGER.warning(
                "Auto-K: inertia range %.2e — data too homogeneous, falling back to K=%d",
                max_i - min_i,
                k_lo,
            )
            return k_lo
        inertias = {k: (i - min_i) / (max_i - min_i) for k, i in raw_inertias.items()}

        # Smooth inertias slightly (rolling average)
        sorted_k = sorted(inertias.keys())
        smoothed = {k: inertias[k] for k in sorted_k}
        if len(sorted_k) > 3:
            for i in range(1, len(sorted_k) - 1):
                smoothed[sorted_k[i]] = (
                    inertias[sorted_k[i - 1]] + inertias[sorted_k[i]] + inertias[sorted_k[i + 1]]
                ) / 3

        # Second pass: compute d2 and track potential K candidates
        candidates = []
        best_d2 = -np.inf

        # Compute d2
        for i in range(1, len(sorted_k) - 1):
            k_prev, k_cur, k_next = sorted_k[i - 1], sorted_k[i], sorted_k[i + 1]
            d2 = smoothed[k_prev] - 2 * smoothed[k_cur] + smoothed[k_next]
            if d2 > best_d2:
                best_d2 = d2
            candidates.append((k_cur, d2))

        # Select candidates within 10% of best_d2; take the first (lowest K) on ties
        best_candidates = [k for k, d2 in candidates if d2 >= best_d2 * 0.9]
        selected_k = best_candidates[0]

        # Informational pass: log OOB for selected K (does not affect selection)
        try:
            km_sel = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
            labels_arr = km_sel.fit_predict(pivoted)
            labels_series = pd.Series(labels_arr, index=pivoted.index)
            predictor = RegimePredictor()
            predictor.fit(daily_features, labels_series, sample_weight=sample_weight, verbose=False)
            oob = predictor.model.oob_score_ if predictor.is_fitted else float("nan")
            _LOGGER.info(
                "Auto-K: inertia elbow selected K=%d. RegimePredictor OOB=%.2f (informational only).",
                selected_k,
                oob,
            )
        except Exception:
            _LOGGER.info("Auto-K: inertia elbow selected K=%d.", selected_k)

        return selected_k

    except Exception as e:
        _LOGGER.error("Auto-K selection failed: %s — using K=%d.", e, k_range[0])
        return k_range[0]
