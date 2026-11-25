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


HEAT_LIMIT_C = 15.0  # days warmer than this carry no heating load
RATIO_BOUNDS = (0.5, 1.0)
MIN_DAY_HOURS = 22


def _daily_frame(train_kwargs: dict) -> pd.DataFrame:
    """Per day: mean-rate kWh, number of hours, mean outdoor temperature and whether an EV charged."""
    energy, weather, ev = (train_kwargs.get(k) for k in ("energy_df", "weather_df", "ev_df"))
    by_day = energy["gross_kwh"].groupby(pd.to_datetime(energy["timestamp"]).dt.normalize())
    days = pd.DataFrame({"kwh": by_day.mean() * 24, "hours": by_day.count()})
    days["temp"] = weather["temp_c"].groupby(pd.to_datetime(weather["timestamp"]).dt.normalize()).mean()
    ev_days = [] if ev is None else pd.to_datetime(ev.loc[ev["gross_kwh"] > 0, "timestamp"]).dt.normalize()
    days["ev"] = days.index.isin(ev_days)
    return days


def rescale_target(train_kwargs: dict, before: pd.Timestamp, min_days: int = 14) -> tuple[pd.DataFrame, float]:
    """``energy_df`` with the heating load of days before *before* scaled to the newer load, and the ratio used.

    Heat excess is the daily kWh above the warm-day median. The ratio compares excess per heating degree after and
    before *before*, over days in the 10th..90th percentile of the newer heating days' temperatures, clipped to
    ``RATIO_BOUNDS``. With fewer than *min_days* fit days on either side, or no warm day, nothing changes (1.0).
    Only the data in *train_kwargs* is used, so a frame cut at a cutoff cannot leak.
    """
    energy = train_kwargs["energy_df"]
    before = pd.Timestamp(before)
    days = _daily_frame(train_kwargs)
    fit = days[(days["hours"] >= MIN_DAY_HOURS) & ~days["ev"]].dropna(subset=["temp"])
    warm = fit.loc[fit["temp"] >= HEAT_LIMIT_C, "kwh"]
    heating = fit[fit["temp"] < HEAT_LIMIT_C]
    new = heating[heating.index >= before]
    if warm.empty or new.empty:
        return energy, 1.0
    base = float(warm.median())
    heating = heating[heating["temp"].between(new["temp"].quantile(0.1), new["temp"].quantile(0.9))]
    new, old = heating[heating.index >= before], heating[heating.index < before]
    if len(new) < min_days or len(old) < min_days:
        return energy, 1.0

    def excess_per_degree(d: pd.DataFrame) -> float:
        return (d["kwh"] - base).clip(lower=0).sum() / (HEAT_LIMIT_C - d["temp"]).sum()

    old_rate = excess_per_degree(old)
    if old_rate <= 0:
        return energy, 1.0
    ratio = float(min(max(excess_per_degree(new) / old_rate, RATIO_BOUNDS[0]), RATIO_BOUNDS[1]))
    excess = (days["kwh"] - base).clip(lower=0)
    scaled = (days.index < before) & (days["temp"] < HEAT_LIMIT_C) & (excess > 0)
    factor = ((base + ratio * excess) / days["kwh"]).where(scaled, 1.0)
    if ratio == 1.0:
        return energy, ratio
    out = energy.copy()
    out["gross_kwh"] = out["gross_kwh"] * pd.to_datetime(out["timestamp"]).dt.normalize().map(factor).to_numpy()
    return out, ratio
