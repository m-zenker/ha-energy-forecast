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


BEFORE = pd.Timestamp("2026-10-01")
BASE_KWH = 8.0


def _rescale_kwargs(factor: float, n_old: int = 40, n_new: int = 40, ev_kwh: float = 0.0) -> dict:
    """Old heating days + 20 warm days (8 kWh, 20 degC) + new heating days whose excess is *factor* x the old one."""
    temps = lambda n: np.linspace(2.0, 12.0, n)  # noqa: E731
    t_old, t_new = temps(n_old), temps(n_new)
    blocks = [
        ("2026-01-01", t_old, BASE_KWH + 2.0 * (15 - t_old)),
        ("2026-07-01", np.full(20, 20.0), np.full(20, BASE_KWH)),
        ("2026-10-01", t_new, BASE_KWH + factor * 2.0 * (15 - t_new)),
    ]
    energy = pd.concat([_profile_days(s, list(y)) for s, _, y in blocks], ignore_index=True)
    weather = pd.DataFrame(
        {"timestamp": energy["timestamp"], "temp_c": np.concatenate([np.repeat(t, 24) for _, t, _ in blocks])}
    )
    ev_df = None
    if ev_kwh:
        day = pd.date_range("2026-10-05", periods=24, freq="h")
        energy.loc[energy["timestamp"].isin(day), "gross_kwh"] += ev_kwh / 24
        ev_df = pd.DataFrame({"timestamp": day, "gross_kwh": ev_kwh / 24, "ev_source": "sensor"})
    return {"energy_df": energy, "weather_df": weather, "ev_df": ev_df}


def _daily(df: pd.DataFrame) -> pd.Series:
    return df["gross_kwh"].groupby(df["timestamp"].dt.normalize()).sum()


def test_ratio_recovers_known_factor():
    kw = _rescale_kwargs(0.6)
    out, r = ra.rescale_target(kw, BEFORE)
    assert r == pytest.approx(0.6, abs=0.02)
    before, after = _daily(kw["energy_df"]), _daily(out)
    old = before.index < "2026-03-01"
    assert after[old].to_numpy() == pytest.approx(BASE_KWH + r * (before[old] - BASE_KWH).to_numpy(), abs=1e-6)
    assert after[~old].to_numpy() == pytest.approx(before[~old].to_numpy())  # warm days and new days unchanged
    d = out["timestamp"].dt.normalize() == "2026-01-01"
    assert (out.loc[d, "gross_kwh"] / out.loc[d, "gross_kwh"].sum()).to_numpy() == pytest.approx(SHAPE / SHAPE.sum())


@pytest.mark.parametrize("factor, expected", [(1.4, 1.0), (0.2, 0.5)])
def test_ratio_clips(factor, expected):
    kw = _rescale_kwargs(factor)
    out, r = ra.rescale_target(kw, BEFORE)
    assert r == expected
    if expected == 1.0:
        pd.testing.assert_frame_equal(out, kw["energy_df"])


@pytest.mark.parametrize("n_old, n_new", [(40, 13), (13, 40)])
def test_too_few_days_is_noop(n_old, n_new):
    kw = _rescale_kwargs(0.6, n_old=n_old, n_new=n_new)
    out, r = ra.rescale_target(kw, BEFORE)
    assert r == 1.0
    pd.testing.assert_frame_equal(out, kw["energy_df"])


def test_no_warm_days_is_noop():
    kw = _rescale_kwargs(0.6)
    kw["weather_df"] = kw["weather_df"].assign(temp_c=kw["weather_df"]["temp_c"].clip(upper=12.0))
    assert ra.rescale_target(kw, BEFORE)[1] == 1.0


def test_ev_days_excluded_from_fit():
    plain, with_ev = ra.rescale_target(_rescale_kwargs(0.6), BEFORE)[1], _rescale_kwargs(0.6, ev_kwh=30.0)
    assert ra.rescale_target(with_ev, BEFORE)[1] == pytest.approx(plain, abs=0.01)


def test_input_not_mutated():
    kw = _rescale_kwargs(0.6)
    copy = kw["energy_df"].copy()
    ra.rescale_target(kw, BEFORE)
    pd.testing.assert_frame_equal(kw["energy_df"], copy)


def _messy_kwargs() -> dict:
    """Old (-4..12 degC) and new (4..12 degC) days differ in range; the ratio inside the new days' band is 0.6.

    Outside the band the old rate is four times higher, the warm days scatter (median 8, mean higher), one old
    in-band day is an EV day, one old in-band day has only 10 (very high) hours, one warm old day sits above base.
    """
    t_old = np.r_[np.linspace(-4.0, 12.0, 40), np.full(10, 6.0)]
    t_new = np.linspace(4.0, 12.0, 30)
    lo, hi = np.quantile(t_new, 0.1), np.quantile(t_new, 0.9)
    rate_old = np.where((t_old >= lo) & (t_old <= hi), 2.0, 8.0)
    rate_new = np.where((t_new >= lo) & (t_new <= hi), 1.2, 0.1)
    warm = np.array([6.0, 8.0, 8.0, 8.0, 30.0])
    blocks = [
        ("2026-01-01", t_old, BASE_KWH + rate_old * (15 - t_old)),
        ("2026-07-01", np.full(5, 20.0), warm),
        ("2026-10-01", t_new, BASE_KWH + rate_new * (15 - t_new)),
    ]
    energy = pd.concat([_profile_days(s, list(y)) for s, _, y in blocks], ignore_index=True)
    temp = np.concatenate([np.repeat(t, 24) for _, t, _ in blocks])
    weather = pd.DataFrame({"timestamp": energy["timestamp"], "temp_c": temp})
    ev_day = pd.Timestamp("2026-02-10")  # old, 6.0 degC
    on_ev = energy["timestamp"].dt.normalize() == ev_day
    energy.loc[on_ev, "gross_kwh"] += 30.0 / 24
    ev_df = pd.DataFrame({"timestamp": energy.loc[on_ev, "timestamp"], "gross_kwh": 30.0 / 24, "ev_source": "sensor"})
    short = energy["timestamp"].dt.normalize() == pd.Timestamp("2026-02-12")  # old, 6.0 degC
    energy = energy[~(short & (energy["timestamp"].dt.hour >= 10))].copy()
    energy.loc[energy["timestamp"].dt.normalize() == pd.Timestamp("2026-02-12"), "gross_kwh"] = 5.0
    return {"energy_df": energy.reset_index(drop=True), "weather_df": weather, "ev_df": ev_df}


def test_ratio_uses_only_clean_days_in_the_band_per_heating_degree():
    out, r = ra.rescale_target(_messy_kwargs(), BEFORE)
    assert r == pytest.approx(0.6, abs=1e-9)


def test_warm_day_above_base_is_not_scaled():
    kw = _messy_kwargs()
    out, _ = ra.rescale_target(kw, BEFORE)
    assert _daily(out)["2026-07-05"] == pytest.approx(30.0) == pytest.approx(_daily(kw["energy_df"])["2026-07-05"])


def test_old_ev_day_is_scaled_although_not_fitted():
    kw = _messy_kwargs()
    out, r = ra.rescale_target(kw, BEFORE)
    before = _daily(kw["energy_df"])["2026-02-10"]
    assert _daily(out)["2026-02-10"] == pytest.approx(BASE_KWH + r * (before - BASE_KWH)) != pytest.approx(before)


@pytest.mark.parametrize("n_old, n_new, acts", [(14, 14, True), (14, 13, False), (13, 14, False)])
def test_fourteen_fit_days_on_each_side(n_old, n_new, acts):
    kw = _rescale_kwargs(0.6, n_old=n_old, n_new=n_new)
    cold = kw["weather_df"]["temp_c"] < 15
    kw["weather_df"] = kw["weather_df"].assign(temp_c=kw["weather_df"]["temp_c"].mask(cold, 8.0))  # all in band
    assert (ra.rescale_target(kw, BEFORE)[1] < 1.0) is acts
