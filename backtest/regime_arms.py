"""Cold-bias candidate arms: regime patches around ``train()`` and the last-winter target rescale.

Background: on the cold days of November 2026 the model over-forecast because history from before the house was
insulated carries a higher load at the same temperature. It reaches the model through the regime centroids (KMeans is
unweighted) and through the LightGBM rows. These arms test three ways out; see "Cold-bias candidates" in README.md.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import pandas as pd

from . import env  # noqa: F401  (apps/ on sys.path before the app import below)

# isort: split
from energy_forecast import clustering

MIN_DAILY_MEAN = 0.1  # kWh/h: floor for the shape-only division


@contextlib.contextmanager
def _rebind(fit, find_optimal_k) -> Iterator[None]:
    """Replace ``DailyProfileClusterer.fit`` and ``find_optimal_k``; restore both on exit."""
    orig_fit, orig_k = clustering.DailyProfileClusterer.fit, clustering.find_optimal_k
    clustering.DailyProfileClusterer.fit, clustering.find_optimal_k = fit, find_optimal_k
    try:
        yield
    finally:
        clustering.DailyProfileClusterer.fit, clustering.find_optimal_k = orig_fit, orig_k


def _per_day_shape(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mean = out["gross_kwh"].groupby(out["timestamp"].dt.date).transform("mean").clip(lower=MIN_DAILY_MEAN)
    out["gross_kwh"] = out["gross_kwh"] / mean
    return out


def shape_only() -> contextlib.AbstractContextManager:
    """Cluster daily profiles divided by their own mean, so regimes and ``regime_kwh`` carry shape, not level."""
    orig_fit, orig_k = clustering.DailyProfileClusterer.fit, clustering.find_optimal_k

    def fit(self, df, *args, **kwargs):
        return orig_fit(self, _per_day_shape(df), *args, **kwargs)

    def find_optimal_k(energy_df, *args, **kwargs):
        return orig_k(_per_day_shape(energy_df), *args, **kwargs)

    return _rebind(fit, find_optimal_k)


def centroids_since(since: pd.Timestamp) -> contextlib.AbstractContextManager:
    """Fit centroids (and choose k) on days from *since* on; older days are still assigned to a centroid."""
    orig_fit, orig_k = clustering.DailyProfileClusterer.fit, clustering.find_optimal_k
    since = pd.Timestamp(since)

    def fit(self, df, sample_weight=None, ev_day_dates=None):
        days = pd.to_datetime(df["timestamp"]).dt.date
        old = set(days[days < since.date()])
        return orig_fit(self, df, sample_weight, set(ev_day_dates or ()) | old)

    def find_optimal_k(energy_df, *args, **kwargs):
        return orig_k(energy_df[pd.to_datetime(energy_df["timestamp"]) >= since], *args, **kwargs)

    return _rebind(fit, find_optimal_k)
