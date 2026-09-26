"""Tests for heating_season: daily heating label, learned daily-mean thresholds, 48h projection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from energy_forecast.heating_season import (
    DEFAULT_OFF_ABOVE,
    DEFAULT_ON_BELOW,
    HeatingThresholds,
    daily_heating_label,
    daily_mean_temp,
    hourly_label_df,
    learn_thresholds,
    load_thresholds,
    meter_prior_and_today,
    project_heating_active,
    resolve_thresholds,
    save_thresholds,
    simulate_rule,
)


def _hourly(start: str, days: int, value_fn) -> pd.DataFrame:
    ts = pd.date_range(start, periods=days * 24, freq="1h")
    return pd.DataFrame({"timestamp": ts, "v": [value_fn(t) for t in ts]})


def _weather(start: str, days: int, daily_means: list[float]) -> pd.DataFrame:
    ts = pd.date_range(start, periods=days * 24, freq="1h")
    return pd.DataFrame({"timestamp": ts, "temp_c": [daily_means[i // 24] for i in range(len(ts))]})


def _seasonal_means(n: int = 120) -> pd.Series:
    """Alternating cold/warm spells (10-day blocks) around the true thresholds 11/15."""
    vals = [8.0 if (i // 10) % 2 == 0 else 18.0 for i in range(n)]
    return pd.Series(vals, index=pd.date_range("2026-03-01", periods=n, freq="D"))


class TestDailyMeanTemp:
    def test_means_per_day(self):
        w = _weather("2026-01-01", 2, [5.0, 10.0])
        m = daily_mean_temp(w)
        assert list(m.values) == [5.0, 10.0]
        assert m.index[0] == pd.Timestamp("2026-01-01")

    def test_missing_temp_column_yields_empty(self):
        """A weather frame without temp_c (e.g. an empty fetch) must not break retrain — defaults apply."""
        assert daily_mean_temp(pd.DataFrame(columns=["timestamp"])).empty

    def test_drops_short_days(self):
        w = _weather("2026-01-01", 2, [5.0, 10.0]).iloc[:30]  # day 2 has only 6 hours
        assert len(daily_mean_temp(w)) == 1


class TestDailyHeatingLabel:
    def test_meter_label_threshold(self):
        # day 1: 0.1 kWh/h * 24 = 2.4 kWh -> 1 ; day 2: 0.01 * 24 = 0.24 kWh -> 0
        m = _hourly("2026-01-01", 2, lambda t: 0.1 if t.day == 1 else 0.01).rename(columns={"v": "kwh"})
        label, src = daily_heating_label(m, None)
        assert src == "meter"
        assert list(label.values) == [1, 0]

    def test_label_falls_back_to_entity_when_meter_empty(self):
        e = _hourly("2026-01-01", 2, lambda t: 1.0 if t.day == 1 else 0.0).rename(columns={"v": "heating_active"})
        label, src = daily_heating_label(pd.DataFrame(columns=["timestamp", "kwh"]), e)
        assert src == "entity"
        assert list(label.values) == [1, 0]

    def test_entity_label_majority_of_day(self):
        # ON only 05:00-09:00 (dawn flap) -> 4/24 of the day -> label 0
        e = _hourly("2026-01-01", 1, lambda t: 1.0 if 5 <= t.hour < 9 else 0.0).rename(columns={"v": "heating_active"})
        label, _ = daily_heating_label(None, e)
        assert list(label.values) == [0]

    def test_no_sources(self):
        label, src = daily_heating_label(None, None)
        assert src == "none" and label.empty


class TestSimulateRule:
    def test_hysteresis_holds_in_dead_band(self):
        means = pd.Series([10.0, 14.0, 17.0, 14.0], index=pd.date_range("2026-01-01", periods=4, freq="D"))
        assert list(simulate_rule(means, 0, 12.0, 16.0).values) == [1, 1, 0, 0]


class TestLearnThresholds:
    def test_recovers_rule(self):
        means = _seasonal_means()
        label = simulate_rule(means, 0, 11.0, 15.0)
        t = learn_thresholds(label, means, "meter")
        assert t.source == "learned"
        assert t.mismatch_rate == 0.0
        # any threshold pair that separates 8 from 18 reproduces the label; it must do so
        assert t.on_below > 8.0 and t.off_above < 18.0 and t.on_below <= t.off_above

    def test_learn_falls_back_without_transitions(self):
        means = pd.Series(20.0, index=pd.date_range("2026-06-01", periods=90, freq="D"))
        label = pd.Series(0, index=means.index)
        t = learn_thresholds(label, means, "entity")
        assert t.source == "default"
        assert (t.on_below, t.off_above) == (DEFAULT_ON_BELOW, DEFAULT_OFF_ABOVE)
        assert t.label_source == "entity"

    def test_learn_falls_back_with_too_few_days(self):
        means = _seasonal_means(40)
        t = learn_thresholds(simulate_rule(means, 0, 11.0, 15.0), means, "meter")
        assert t.source == "default"


class TestResolveThresholds:
    def test_config_overrides_learned(self):
        learned = HeatingThresholds(11.0, 15.0, "learned", "meter", 100, 0.05)
        t = resolve_thresholds(13.0, 17.0, learned)
        assert (t.on_below, t.off_above, t.source, t.label_source) == (13.0, 17.0, "config", "meter")

    def test_no_config_keeps_learned(self):
        learned = HeatingThresholds(11.0, 15.0, "learned", "meter", 100, 0.05)
        assert resolve_thresholds(None, None, learned) is learned


class TestHourlyLabelDf:
    def test_broadcasts_day_to_hours(self):
        label = pd.Series([1, 0], index=pd.date_range("2026-01-01", periods=2, freq="D"))
        df = hourly_label_df(label)
        assert len(df) == 48
        assert df["heating_active"].iloc[0] == 1 and df["heating_active"].iloc[47] == 0

    def test_gap_days_are_absent_not_nan(self):
        label = pd.Series([1, 0], index=pd.DatetimeIndex(["2026-01-01", "2026-01-03"]))
        df = hourly_label_df(label)
        assert len(df) == 48  # 2026-01-02 omitted
        assert df["heating_active"].dtype.kind == "i"


class TestMeterPriorAndToday:
    def test_prior_from_yesterday_and_today_heated(self):
        m = _hourly("2026-01-01", 2, lambda t: 0.1).rename(columns={"v": "kwh"})
        now = pd.Timestamp("2026-01-02 12:00")
        prior, today = meter_prior_and_today(m.iloc[: 24 + 12], now)
        assert prior == 1 and today == 1  # 12 h * 0.1 = 1.2 kWh > 0.5

    def test_today_none_when_not_yet_heated(self):
        m = _hourly("2026-01-01", 2, lambda t: 0.0).rename(columns={"v": "kwh"})
        prior, today = meter_prior_and_today(m.iloc[: 24 + 3], pd.Timestamp("2026-01-02 03:00"))
        assert prior == 0 and today is None

    def test_meter_prior_insufficient_rows(self):
        m = _hourly("2026-01-01", 1, lambda t: 0.1).rename(columns={"v": "kwh"}).iloc[18:]  # 6 rows yesterday
        prior, today = meter_prior_and_today(m, pd.Timestamp("2026-01-02 01:00"))
        assert prior is None

    def test_none_df(self):
        assert meter_prior_and_today(None, pd.Timestamp("2026-01-02 01:00")) == (None, None)


class TestProjectHeatingActive:
    TH = HeatingThresholds(12.0, 16.0, "learned", "entity")

    def _fc(self, now: pd.Timestamp, fn) -> pd.DataFrame:
        ts = pd.date_range(now.floor("1h"), periods=72, freq="1h")
        return pd.DataFrame({"timestamp": ts, "temp_c": [fn(t) for t in ts]})

    def test_cold_nights_warm_days_stay_off(self):
        """The Sept-2026 failure: nights 8 °C, days 22 °C -> daily mean ~16.75 -> OFF all 48 h."""
        now = pd.Timestamp("2026-09-24 00:00")
        fc = self._fc(now, lambda t: 8.0 if t.hour < 9 else 22.0)
        s = project_heating_active(now, fc, prior_state=0, thresholds=self.TH, today_state=0)
        assert len(s) == 48 and s.sum() == 0

    def test_cold_days_switch_on_from_tomorrow(self):
        now = pd.Timestamp("2026-10-10 18:00")
        fc = self._fc(now, lambda t: 5.0)
        s = project_heating_active(now, fc, prior_state=0, thresholds=self.TH, today_state=0)
        assert s.loc[:"2026-10-10 23:00"].sum() == 0
        assert (s.loc["2026-10-11 00:00":] == 1).all()

    def test_partial_day_carries_state(self):
        """Evening: 2 forecast hours left today and a partial 3rd day -> both carry, never decide on a partial mean."""
        now = pd.Timestamp("2026-10-10 22:00")
        fc = self._fc(now, lambda t: 5.0 if t.day == 10 else 20.0)
        s = project_heating_active(now, fc, prior_state=0, thresholds=self.TH, today_state=None)
        assert s.loc[:"2026-10-10 23:00"].sum() == 0  # today: only 2 h of forecast -> carry prior 0
        assert s.loc["2026-10-11"].sum() == 0  # tomorrow warm -> 0

    def test_today_state_overrides_rule_for_today(self):
        now = pd.Timestamp("2026-10-10 08:00")
        fc = self._fc(now, lambda t: 20.0)
        s = project_heating_active(now, fc, prior_state=1, thresholds=self.TH, today_state=1)
        assert (s.loc[:"2026-10-10 23:00"] == 1).all()
        assert s.loc["2026-10-11"].sum() == 0

    def test_index_is_48_naive_hours(self):
        now = pd.Timestamp("2026-10-10 08:30")
        s = project_heating_active(now, self._fc(now, lambda t: 10.0), 0, self.TH)
        assert s.index[0] == pd.Timestamp("2026-10-10 08:00") and len(s) == 48
        assert s.index.tz is None and s.dtype == np.int64


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        t = HeatingThresholds(11.5, 14.5, "learned", "meter", 180, 0.056)
        p = tmp_path / "heating_thresholds.json"
        save_thresholds(t, p)
        assert load_thresholds(p) == t

    def test_load_missing_or_corrupt(self, tmp_path):
        assert load_thresholds(tmp_path / "nope.json") is None
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        assert load_thresholds(bad) is None

    def test_rule_matches_default_constants(self):
        assert DEFAULT_ON_BELOW == pytest.approx(12.0) and DEFAULT_OFF_ABOVE == pytest.approx(16.0)
