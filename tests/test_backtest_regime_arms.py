"""backtest.regime_arms: regime patches and the last-winter target rescale."""

import numpy as np
import pandas as pd
import pytest
from energy_forecast import clustering as cl

from backtest import regime_arms as ra

SHAPE = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(24) / 24)


def _profile_days(start: str, daily_kwh: list[float], shape=SHAPE) -> pd.DataFrame:
    unit = np.asarray(shape) / np.sum(shape)
    ts = pd.date_range(start, periods=24 * len(daily_kwh), freq="h")
    return pd.DataFrame({"timestamp": ts, "gross_kwh": np.concatenate([k * unit for k in daily_kwh])})


def test_shape_only_centroids_have_unit_mean():
    peaked = SHAPE**3
    df = pd.concat(
        [_profile_days("2026-01-01", [10.0, 40.0] * 10), _profile_days("2026-01-21", [10.0, 40.0] * 10, shape=peaked)],
        ignore_index=True,
    )
    c = cl.DailyProfileClusterer(n_clusters=2)
    with ra.shape_only():
        labels = c.fit(df)
    assert labels is not None and len(labels) == 40
    assert c.centroids.mean(axis=1) == pytest.approx(1.0, abs=1e-6)


def test_shape_only_k_search_sees_normalised_profiles(monkeypatch):
    seen = {}
    monkeypatch.setattr(cl, "find_optimal_k", lambda df, *a, **k: seen.setdefault("df", df) is None or 3)
    with ra.shape_only():
        cl.find_optimal_k(_profile_days("2026-01-01", [10.0, 40.0]), None)
    daily = seen["df"].groupby(seen["df"]["timestamp"].dt.date)["gross_kwh"].mean()
    assert daily.to_numpy() == pytest.approx(1.0)


def _old_and_new() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    old = _profile_days("2026-01-01", list(rng.uniform(30, 40, 60)))
    new = _profile_days("2026-03-02", list(rng.uniform(8, 20, 60)), shape=SHAPE[::-1])
    return old, new


def test_centroids_since_labels_all_days():
    old, new = _old_and_new()
    c = cl.DailyProfileClusterer(n_clusters=2)
    with ra.centroids_since(pd.Timestamp("2026-03-02")):
        labels = c.fit(pd.concat([old, new], ignore_index=True))
    assert len(labels) == 120
    ref = cl.DailyProfileClusterer(n_clusters=2)
    ref.fit(new)
    assert c.centroids == pytest.approx(ref.centroids)


def test_centroids_since_keeps_ev_days_excluded():
    old, new = _old_and_new()
    ev_day = pd.Timestamp("2026-03-10").date()
    new.loc[new["timestamp"].dt.date == ev_day, "gross_kwh"] += 50.0
    c = cl.DailyProfileClusterer(n_clusters=2)
    with ra.centroids_since(pd.Timestamp("2026-03-02")):
        c.fit(pd.concat([old, new], ignore_index=True), ev_day_dates={ev_day})
    assert c.centroids.max() < 5.0


def test_centroids_since_k_uses_recent_days(monkeypatch):
    seen = {}

    def fake(df, *args, **kwargs):
        seen["min"] = df["timestamp"].min()
        return 3

    monkeypatch.setattr(cl, "find_optimal_k", fake)
    with ra.centroids_since(pd.Timestamp("2026-03-02")):
        assert cl.find_optimal_k(pd.concat(_old_and_new(), ignore_index=True), None) == 3
    assert seen["min"] == pd.Timestamp("2026-03-02")


@pytest.mark.parametrize("ctx", [ra.shape_only, lambda: ra.centroids_since(pd.Timestamp("2026-03-02"))])
def test_patches_restore_on_exit_and_on_error(ctx):
    fit, k = cl.DailyProfileClusterer.fit, cl.find_optimal_k
    with ctx():
        assert cl.DailyProfileClusterer.fit is not fit and cl.find_optimal_k is not k
    assert cl.DailyProfileClusterer.fit is fit and cl.find_optimal_k is k
    with pytest.raises(RuntimeError), ctx():
        raise RuntimeError
    assert cl.DailyProfileClusterer.fit is fit and cl.find_optimal_k is k
