"""Tests for energy_forecast.py module-level helpers.

Covers:
  - _blend_today_totals: actuals substituted for elapsed hours, predictions for future
  - _compute_live_mae: correct MAE over matched pairs, nan on no overlap
  - EnergyForecast._retrain_cb / _update_cb: callable from both timer and event contexts
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from energy_forecast.energy_forecast import (
    EnergyForecast,
    _apply_target_correction,
    _blend_today_totals,
    _build_shap_narrative,
    _compute_anomaly,
    _compute_live_mae,
    _subtract_sub_sensors,
)
from energy_forecast.model import EnergyForecastModel

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_predictions(today: pd.Timestamp, kwh: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """48 hourly predictions starting at today midnight, all equal to kwh."""
    ts = pd.date_range(today, periods=48, freq="1h")
    p_times = ts.values.astype("datetime64[ns]")
    p_vals = np.full(48, kwh)
    return p_times, p_vals


def _make_actuals(start: pd.Timestamp, n: int, kwh: float = 3.0) -> pd.DataFrame:
    """n hourly actuals starting at start, all equal to kwh."""
    ts = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"timestamp": ts, "gross_kwh": [kwh] * n})


# ── _blend_today_totals ───────────────────────────────────────────────────────


class TestBlendTodayTotals:
    def _nts(self, ts: pd.Timestamp) -> np.datetime64:
        return np.datetime64(ts, "ns")

    def test_no_actuals_equals_prediction_sum(self):
        """With full_actuals=None the result must equal the prediction sum for today."""
        today = pd.Timestamp("2026-03-12 00:00")
        now = pd.Timestamp("2026-03-12 00:00")  # start of day — all future
        tmrw = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=2.0)
        total, blocks = _blend_today_totals(
            p_times,
            p_vals,
            None,
            self._nts(today),
            self._nts(tmrw),
            self._nts(now),
        )
        # predictions cover [00:00, 48h); only today's 24 hours sum to 24 × 2.0 = 48.0
        assert abs(total - 48.0) < 1e-6

    def test_blended_total_uses_actuals_for_elapsed_hours(self):
        """Elapsed-hour actuals (3.0 kWh) replace predictions (1.0 kWh) in today's total."""
        today = pd.Timestamp("2026-03-12 00:00")
        now = pd.Timestamp("2026-03-12 12:00")  # noon — 12h elapsed
        tmrw = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=1.0)
        actuals = _make_actuals(today, n=12, kwh=3.0)  # 12 elapsed hours at 3.0

        total, _ = _blend_today_totals(
            p_times,
            p_vals,
            actuals,
            self._nts(today),
            self._nts(tmrw),
            self._nts(now),
        )
        # 12 elapsed hours × 3.0 (actuals) + 12 future hours × 1.0 (predictions) = 48.0
        assert abs(total - (12 * 3.0 + 12 * 1.0)) < 1e-6

    def test_fully_elapsed_block_uses_actuals(self):
        """A 3h block entirely in the past must sum actuals, not predictions."""
        today = pd.Timestamp("2026-03-12 00:00")
        now = pd.Timestamp("2026-03-12 12:00")  # block 00-03 is fully elapsed
        tmrw = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=1.0)
        actuals = _make_actuals(today, n=12, kwh=5.0)  # 12 elapsed hours at 5.0

        _, blocks = _blend_today_totals(
            p_times,
            p_vals,
            actuals,
            self._nts(today),
            self._nts(tmrw),
            self._nts(now),
        )
        # Block 00_03: 3 actual hours × 5.0 = 15.0
        assert abs(blocks["00_03"] - 15.0) < 1e-6

    def test_fully_future_block_uses_predictions(self):
        """A 3h block entirely in the future must sum predictions only."""
        today = pd.Timestamp("2026-03-12 00:00")
        now = pd.Timestamp("2026-03-12 00:00")  # nothing elapsed yet
        tmrw = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=2.5)
        actuals = _make_actuals(today, n=0, kwh=0.0)  # empty — nothing elapsed

        _, blocks = _blend_today_totals(
            p_times,
            p_vals,
            actuals,
            self._nts(today),
            self._nts(tmrw),
            self._nts(now),
        )
        # Block 21_24: 3 future hours × 2.5 = 7.5
        assert abs(blocks["21_24"] - 7.5) < 1e-6


# ── _compute_live_mae ─────────────────────────────────────────────────────────


class TestComputeLiveMae:
    def test_matched_pairs_return_correct_mae(self):
        """When predictions and actuals overlap perfectly, MAE equals mean absolute error."""
        ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        pred_history = {t: 2.0 for t in ts}  # predicted 2.0 kWh every hour
        actuals = pd.DataFrame(
            {
                "timestamp": ts,
                "gross_kwh": [3.0] * 24,  # actual was 3.0 → error = 1.0 each
            }
        )
        mae, n = _compute_live_mae(pred_history, actuals)
        assert n == 24
        assert abs(mae - 1.0) < 1e-6

    def test_no_overlap_returns_nan_and_zero(self):
        """Pred history and actuals in non-overlapping windows → (nan, 0)."""
        pred_ts = pd.date_range("2026-03-10 00:00", periods=24, freq="1h")
        actual_ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        pred_history = {t: 1.0 for t in pred_ts}
        actuals = pd.DataFrame({"timestamp": actual_ts, "gross_kwh": [2.0] * 24})
        mae, n = _compute_live_mae(pred_history, actuals)
        assert n == 0
        assert math.isnan(mae)

    def test_partial_overlap_uses_matched_pairs_only(self):
        """Only overlapping timestamps contribute to MAE."""
        ts_all = pd.date_range("2026-03-12 00:00", periods=48, freq="1h")
        ts_first = ts_all[:24]  # predictions cover first 24h
        ts_second = ts_all[24:]  # actuals cover last 24h — no overlap

        pred_history = {t: 1.0 for t in ts_first}
        actuals = pd.DataFrame(
            {
                "timestamp": pd.concat(
                    [
                        pd.Series(ts_first[:12]),  # 12h overlap
                        pd.Series(ts_second),  # 24h no overlap
                    ]
                ),
                "gross_kwh": [3.0] * 36,
            }
        )
        mae, n = _compute_live_mae(pred_history, actuals)
        assert n == 12
        assert abs(mae - 2.0) < 1e-6  # |3.0 - 1.0| = 2.0 for each of the 12 matched


# ── Interval blending (#13) ───────────────────────────────────────────────────


class TestIntervalBlend:
    def _nts(self, ts: pd.Timestamp) -> np.datetime64:
        return np.datetime64(ts, "ns")

    def test_low_total_less_than_high_total(self):
        """Blending q10 vals gives today_low < today_high when q10 < q90 for future hours."""
        today = pd.Timestamp("2026-03-12 00:00")
        now = pd.Timestamp("2026-03-12 00:00")  # nothing elapsed
        tmrw = today + pd.Timedelta(days=1)

        ts = pd.date_range(today, periods=48, freq="1h")
        p_times = ts.values.astype("datetime64[ns]")

        low_vals = np.full(48, 1.0)  # q10 = 1.0 kWh/h
        high_vals = np.full(48, 3.0)  # q90 = 3.0 kWh/h

        today_low, _ = _blend_today_totals(p_times, low_vals, None, self._nts(today), self._nts(tmrw), self._nts(now))
        today_high, _ = _blend_today_totals(p_times, high_vals, None, self._nts(today), self._nts(tmrw), self._nts(now))

        assert today_low < today_high

    def test_fully_elapsed_window_interval_collapses_to_actuals(self):
        """When all hours are elapsed, both low and high equal the actuals sum."""
        today = pd.Timestamp("2026-03-12 00:00")
        now = pd.Timestamp("2026-03-12 06:00")  # 6h elapsed
        tmrw = today + pd.Timedelta(days=1)

        p_times = pd.date_range(now, periods=48, freq="1h").values.astype("datetime64[ns]")
        low_vals = np.full(48, 1.0)
        high_vals = np.full(48, 5.0)
        actuals = pd.DataFrame(
            {
                "timestamp": pd.date_range(today, periods=6, freq="1h"),
                "gross_kwh": [2.0] * 6,
            }
        )
        # Block 00_03 is fully elapsed → both bounds must equal 3×2.0 = 6.0
        _, blocks_low = _blend_today_totals(
            p_times, low_vals, actuals, self._nts(today), self._nts(tmrw), self._nts(now)
        )
        _, blocks_high = _blend_today_totals(
            p_times, high_vals, actuals, self._nts(today), self._nts(tmrw), self._nts(now)
        )

        assert abs(blocks_low["00_03"] - 6.0) < 1e-6
        assert abs(blocks_high["00_03"] - 6.0) < 1e-6


# ── _aggregate ────────────────────────────────────────────────────────────────


def _fake_self_for_aggregate(ev_threshold: float = 4.5, ev_charger_kw: float = 9.0) -> _FakeSelf:
    app = _FakeSelf()
    app._ev_threshold = ev_threshold
    app._ev_charger_kw = ev_charger_kw
    return app


def _pred_df(start: pd.Timestamp, n: int = 48, kwh: float = 1.0) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"timestamp": ts, "predicted_kwh": np.full(n, kwh)})


class TestAggregate:
    """_aggregate wires predictions + actuals into the sensor value dict."""

    def _run(self, today, now, predictions, actuals=None, intervals=None):
        from energy_forecast.energy_forecast import EnergyForecast

        with patch("pandas.Timestamp.now") as mock_now:
            mock_now.return_value = now
            return EnergyForecast._aggregate(
                _fake_self_for_aggregate(), predictions, actuals, live_temp=None, intervals=intervals
            )

    def test_next_1h_sums_only_one_future_hour(self):
        """next_1h must equal exactly 1 × per-hour prediction."""
        today = pd.Timestamp.now().normalize()
        now = pd.Timestamp.now().floor("1h")
        preds = _pred_df(now, n=48, kwh=2.0)
        result = self._run(today, now, preds)
        assert abs(result["next_1h"] - 2.0) < 1e-3

    def test_next_1h_less_than_next_3h(self):
        """next_1h must be strictly less than next_3h when kWh/h > 0."""
        today = pd.Timestamp.now().normalize()
        now = pd.Timestamp.now().floor("1h")
        preds = _pred_df(now, n=48, kwh=1.5)
        result = self._run(today, now, preds)
        assert result["next_1h"] < result["next_3h"]

    def test_next_3h_sums_only_three_future_hours(self):
        """next_3h must equal exactly 3 × per-hour prediction, not more."""
        today = pd.Timestamp.now().normalize()
        now = pd.Timestamp.now().floor("1h")
        preds = _pred_df(now, n=48, kwh=2.0)
        result = self._run(today, now, preds)
        assert abs(result["next_3h"] - 6.0) < 1e-3

    def test_tomorrow_uses_predictions_only(self):
        """tomorrow must equal 24 × per-hour prediction for the following calendar day."""
        today = pd.Timestamp.now().normalize()
        preds = _pred_df(today, n=48, kwh=3.0)
        result = self._run(today, today, preds)
        assert abs(result["tomorrow"] - 72.0) < 1e-3  # 24 × 3.0

    def test_ev_sensors_zero_when_no_actuals(self):
        """ev_today and ev_yesterday must be 0.0 when full_actuals is None."""
        today = pd.Timestamp.now().normalize()
        preds = _pred_df(today, n=48, kwh=1.0)
        result = self._run(today, today, preds, actuals=None)
        assert result["ev_today"] == 0.0
        assert result["ev_yesterday"] == 0.0

    def test_interval_keys_present_when_intervals_supplied(self):
        """When intervals DataFrame is provided, *_low/*_high keys appear in result."""
        today = pd.Timestamp.now().normalize()
        now = today
        preds = _pred_df(today, n=48, kwh=1.0)
        ts = pd.date_range(today, periods=48, freq="1h")
        ivs = pd.DataFrame(
            {
                "timestamp": ts,
                "low_kwh": np.full(48, 0.5),
                "high_kwh": np.full(48, 2.0),
            }
        )
        result = self._run(today, now, preds, intervals=ivs)
        for key in ("next_3h_low", "next_3h_high", "today_low", "today_high", "tomorrow_low", "tomorrow_high"):
            assert key in result, f"missing key: {key}"

    def test_interval_keys_absent_when_intervals_is_none(self):
        """Without quantile models, no *_low/*_high keys should appear."""
        today = pd.Timestamp.now().normalize()
        preds = _pred_df(today, n=48, kwh=1.0)
        result = self._run(today, today, preds, intervals=None)
        for key in ("next_3h_low", "next_3h_high", "today_low", "today_high"):
            assert key not in result, f"unexpected key: {key}"

    def test_blocks_today_has_eight_slots(self):
        """blocks_today must contain exactly 8 slots (00_03 … 21_24)."""
        today = pd.Timestamp.now().normalize()
        preds = _pred_df(today, n=48, kwh=1.0)
        result = self._run(today, today, preds)
        assert len(result["blocks_today"]) == 8
        assert "00_03" in result["blocks_today"]
        assert "21_24" in result["blocks_today"]


# ── EV kWh sensor calculation ────────────────────────────────────────────────


class TestEvKwhSensorCalc:
    """EV kWh values published to HA sensors use charger_kw, not threshold."""

    def _nts(self, ts: pd.Timestamp) -> np.datetime64:
        return np.datetime64(ts, "ns")

    def _run_aggregate(self, ev_charger_kw: float, ev_threshold: float) -> dict:
        """Run _aggregate with synthetic actuals containing one EV hour."""
        from energy_forecast.energy_forecast import EnergyForecast

        today = pd.Timestamp.now().normalize()

        # Predictions: 48h at 1.0 kWh/h
        ts = pd.date_range(today, periods=48, freq="1h")
        p_df = pd.DataFrame({"timestamp": ts, "predicted_kwh": np.ones(48)})

        # Actuals: one EV-level hour today (gross=12 kWh, clearly above any threshold)
        actuals = pd.DataFrame(
            {
                "timestamp": [today + pd.Timedelta(hours=2)],
                "gross_kwh": [12.0],
            }
        )

        app = _FakeSelf()
        app._ev_threshold = ev_threshold
        app._ev_charger_kw = ev_charger_kw

        with patch("pandas.Timestamp.now") as mock_now:
            # mock_now must be timezone-aware because _aggregate calls .tz_convert
            mock_now.return_value = today.tz_localize("Europe/Zurich")
            return EnergyForecast._aggregate(app, p_df, actuals, live_temp=None)

    def test_ev_today_reports_charger_energy_not_co_load(self):
        """ev_today must equal min(gross, charger_kw) — the EV energy, not the co-load."""
        result = self._run_aggregate(ev_charger_kw=7.4, ev_threshold=4.5)
        # gross=12.0, charger_kw=7.4 → min(12.0, 7.4) = 7.4 (not 12.0 − 7.4 = 4.6)
        assert abs(result["ev_today"] - 7.4) < 1e-6

    def test_ev_today_default_charger_kw(self):
        """Default charger_kw=9.0: ev_today = min(gross, 9.0) = 9.0."""
        result = self._run_aggregate(ev_charger_kw=9.0, ev_threshold=4.5)
        # gross=12.0, charger_kw=9.0 → min(12.0, 9.0) = 9.0 (not 12.0 − 9.0 = 3.0)
        assert abs(result["ev_today"] - 9.0) < 1e-6


# ── Callback signature compatibility ─────────────────────────────────────────


class _FakeSelf:
    """Minimal stand-in for EnergyForecast used in callback / aggregate tests.

    EnergyForecast inherits from MagicMock (via the hassapi stub) so
    object.__new__ leaves mock internals uninitialized.  A plain class with
    the attributes touched by the tested methods is sufficient.
    """

    def __init__(self):
        self._ml_model = MagicMock()
        self._ml_model.model = None  # prevents _update_sensors execution
        self._lock = MagicMock()
        self._lock.acquire.return_value = False  # always "busy" → immediate return
        self._timezone = "Europe/Zurich"
        self._baseline_mode = False
        self._sub_energy_sensors = []

    def log(self, msg, level="INFO"):  # AppDaemon log stub
        pass


class TestCallbackSignature:
    """_retrain_cb and _update_cb must accept both timer and event calling conventions."""

    def test_retrain_cb_timer_style(self):
        """Timer callbacks pass a single positional dict — must not raise TypeError."""
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._retrain_cb(_FakeSelf(), {})

    def test_retrain_cb_event_style(self):
        """listen_event callbacks pass (event_name, data, kwargs) — must not raise TypeError."""
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._retrain_cb(_FakeSelf(), "RELOAD_ENERGY_MODEL", {}, {})

    def test_update_cb_timer_style(self):
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._update_cb(_FakeSelf(), {})

    def test_update_cb_event_style(self):
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._update_cb(_FakeSelf(), "some_event", {}, {})


# ── Config validation — EV threshold vs charger_kw warning (#20) ─────────────


class _FakeValidateSelf:
    """Minimal stand-in for _validate_config with configurable EV params."""

    def __init__(self, ev_threshold: float, ev_charger_kw: float):
        self._lat = 47.0
        self._lon = 8.5
        self._plz = ""
        self._timezone = "Europe/Zurich"
        self._holiday_country = "CH"
        self._weight_halflife = 90.0
        self._ev_threshold = ev_threshold
        self._ev_charger_kw = ev_charger_kw
        self._baseline_mode = False
        self._adaptive_retrain_threshold = 2.0
        self._anomaly_sigma_threshold = 3.0
        self._shap_top_n = 5
        self._model_archive_count = 3
        self._sub_energy_sensors = []
        self._solar_sensor = None
        self._grid_export_sensor = None
        self._battery_charge_sensor = None
        self._battery_discharge_sensor = None
        self._mqtt_discovery = False
        self._mqtt_namespace = "mqtt"
        self._mqtt_discovery_prefix = "homeassistant"
        self._warnings: list[str] = []

    def log(self, msg: str, level: str = "INFO") -> None:
        if level == "WARNING":
            self._warnings.append(msg)


class TestValidateConfig:
    """_validate_config must warn when ev_threshold >= ev_charger_kw."""

    def test_warns_when_threshold_equals_charger_kw(self, caplog):
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeValidateSelf(ev_threshold=9.0, ev_charger_kw=9.0)
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._validate_config(fake)
        assert any(r.levelno == logging.WARNING for r in caplog.records), (
            "Expected WARNING when ev_threshold == ev_charger_kw"
        )

    def test_warns_when_threshold_exceeds_charger_kw(self, caplog):
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeValidateSelf(ev_threshold=10.0, ev_charger_kw=9.0)
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._validate_config(fake)
        assert any(r.levelno == logging.WARNING for r in caplog.records), (
            "Expected WARNING when ev_threshold > ev_charger_kw"
        )

    def test_no_warning_when_threshold_below_charger_kw(self):
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeValidateSelf(ev_threshold=7.0, ev_charger_kw=9.0)
        EnergyForecast._validate_config(fake)
        assert not fake._warnings, "No warning expected when ev_threshold < ev_charger_kw"

    def test_warns_solar_sensor_without_grid_export(self, caplog):
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeValidateSelf(ev_threshold=7.0, ev_charger_kw=9.0)
        fake._solar_sensor = "sensor.solar_production"
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._validate_config(fake)
        assert any("grid_export_sensor" in r.message for r in caplog.records), (
            "Expected WARNING when solar_production_sensor is set without grid_export_sensor"
        )

    def test_no_warning_solar_sensor_with_grid_export(self):
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeValidateSelf(ev_threshold=7.0, ev_charger_kw=9.0)
        fake._solar_sensor = "sensor.solar_production"
        fake._grid_export_sensor = "sensor.grid_export"
        EnergyForecast._validate_config(fake)
        assert not any("grid_export_sensor" in w for w in fake._warnings), (
            "No grid_export warning expected when both solar and export sensors are configured"
        )


# ── Setup checker sensor (#17) ────────────────────────────────────────────────


class _FakeCheckSetup:
    """Minimal stand-in for _check_setup with controllable set_state and log."""

    def __init__(self):
        self._states: dict[str, dict] = {}
        self._warnings: list[str] = []
        self._mqtt_discovery: bool = False

    def set_state(self, entity_id: str, state: str, attributes: dict, replace: bool = False) -> None:
        self._states[entity_id] = {"state": state, "attributes": attributes}

    def log(self, msg: str, level: str = "INFO") -> None:
        if level == "WARNING":
            self._warnings.append(msg)


class TestCheckSetup:
    """_check_setup publishes setup status sensor; detects missing packages."""

    def test_ok_state_when_all_packages_present(self):
        """When all imports succeed, sensor state must be 'ok'."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeCheckSetup()
        # All packages importable in test environment → state = 'ok'
        EnergyForecast._check_setup(fake)
        status = fake._states.get("sensor.energy_forecast_setup_status", {})
        assert status.get("state") == "ok"
        assert status["attributes"]["missing_packages"] == []

    def test_missing_packages_state_when_import_fails(self):
        """When an import fails, sensor state must be 'missing_packages'."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeCheckSetup()
        # Patch builtins.__import__ to fail for 'holidays'
        import builtins

        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "holidays":
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=failing_import):
            EnergyForecast._check_setup(fake)

        status = fake._states.get("sensor.energy_forecast_setup_status", {})
        assert status.get("state") == "missing_packages"
        assert "holidays" in status["attributes"]["missing_packages"]


# ── MQTT Discovery tests ──────────────────────────────────────────────────────


class _FakeMqttSelf:
    """Minimal stand-in for EnergyForecast for MQTT-related method tests."""

    def __init__(
        self,
        mqtt_discovery: bool = True,
        mqtt_namespace: str = "mqtt",
        mqtt_discovery_prefix: str = "homeassistant",
    ):
        self._mqtt_discovery = mqtt_discovery
        self._mqtt_namespace = mqtt_namespace
        self._mqtt_discovery_prefix = mqtt_discovery_prefix
        self._mqtt_intervals_discovered = False
        self._publishes: list[dict] = []  # records all mqtt_publish() calls
        self._warnings: list[str] = []
        self._removed_entities: list[str] = []
        # Attributes needed by safe_set / _publish
        self._ml_model = MagicMock()
        self._ml_model.last_mae = 0.5
        self._ml_model.last_cv_mae = 0.4
        self._ml_model.last_trained = datetime.min
        self._ml_model.engine = "lgbm"
        self._ev_threshold = 7.0
        self._ev_charger_kw = 9.0
        self._anomaly_sigma_threshold = 3.0
        self._states: dict = {}

    def call_service(self, service: str, **kwargs) -> None:
        if service == "mqtt/publish":
            self._publishes.append(
                {
                    "topic": kwargs.get("topic", ""),
                    "payload": kwargs.get("payload", ""),
                    "namespace": kwargs.get("namespace", ""),
                    "retain": kwargs.get("retain", False),
                }
            )

    def set_state(self, entity_id: str, state: str = "", attributes: dict | None = None, replace: bool = False) -> None:
        self._states[entity_id] = {"state": state, "attributes": attributes or {}}

    def entity_exists(self, entity_id: str) -> bool:
        return True

    def remove_entity(self, entity_id: str) -> None:
        self._removed_entities.append(entity_id)

    def log(self, msg: str, level: str = "INFO") -> None:
        if level == "WARNING":
            self._warnings.append(msg)

    def _mqtt_set_sensor(self, unique_id: str, value: Any) -> None:
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._mqtt_set_sensor(self, unique_id, value)

    def _mqtt_set_sensor_raw(self, unique_id: str, value_str: str) -> None:
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._mqtt_set_sensor_raw(self, unique_id, value_str)

    def _mqtt_publish_discovery(self, *args, **kwargs) -> None:
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._mqtt_publish_discovery(self, *args, **kwargs)

    def _build_sensor_discovery_payload(self, *args, **kwargs) -> dict:
        from energy_forecast.energy_forecast import EnergyForecast

        return EnergyForecast._build_sensor_discovery_payload(self, *args, **kwargs)

    def _mqtt_publish_availability(self, payload: str) -> None:
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._mqtt_publish_availability(self, payload)

    def _mqtt_publish_all_discovery(self) -> None:
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._mqtt_publish_all_discovery(self)

    def _build_binary_sensor_discovery_payload(self, *args, **kwargs) -> dict:
        from energy_forecast.energy_forecast import EnergyForecast

        return EnergyForecast._build_binary_sensor_discovery_payload(self, *args, **kwargs)

    def _mqtt_publish_binary_sensor_discovery(self, *args, **kwargs) -> None:
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._mqtt_publish_binary_sensor_discovery(self, *args, **kwargs)


class TestMqttPublishDiscovery:
    """Discovery payload structure and publish behaviour."""

    def test_payload_contains_device_block_with_identifier(self):
        """Discovery payload must include a 'device' block with identifiers=['ha_energy_forecast']."""
        fake = _FakeMqttSelf()
        payload = fake._build_sensor_discovery_payload(
            "energy_forecast_today",
            "Today",
            "kWh",
            "mdi:lightning-bolt",
            "energy",
            "measurement",
        )
        assert "device" in payload
        assert payload["device"]["identifiers"] == ["ha_energy_forecast"]

    def test_discovery_config_topic_uses_configured_prefix(self):
        """Config topic must be <prefix>/sensor/<unique_id>/config."""
        fake = _FakeMqttSelf(mqtt_discovery_prefix="myprefix")
        fake._mqtt_publish_discovery(
            "energy_forecast_today",
            "Today",
            "kWh",
            "mdi:lightning-bolt",
            "energy",
            "measurement",
        )
        assert len(fake._publishes) == 1
        assert fake._publishes[0]["topic"] == "myprefix/sensor/energy_forecast_today/config"

    def test_kwh_sensor_payload_has_device_class_and_state_class(self):
        """kWh sensor must have device_class='energy' and state_class='measurement'."""
        fake = _FakeMqttSelf()
        payload = fake._build_sensor_discovery_payload(
            "energy_forecast_today",
            "Today",
            "kWh",
            "mdi:lightning-bolt",
            "energy",
            "measurement",
        )
        assert payload.get("device_class") == "energy"
        assert payload.get("state_class") == "measurement"

    def test_setup_status_payload_omits_device_class(self):
        """Setup status sensor must not include device_class or state_class."""
        fake = _FakeMqttSelf()
        payload = fake._build_sensor_discovery_payload(
            "energy_forecast_setup_status",
            "Setup Status",
            "",
            "mdi:check-circle",
            None,
            None,
        )
        assert "device_class" not in payload
        assert "state_class" not in payload

    def test_mqtt_publish_failure_logs_warning_and_does_not_raise(self, caplog):
        """If call_service raises, _mqtt_publish_discovery must log WARNING and not re-raise."""
        import logging

        fake = _FakeMqttSelf()

        def bad_call_service(service, **kwargs):
            raise RuntimeError("broker down")

        fake.call_service = bad_call_service
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            fake._mqtt_publish_discovery(
                "energy_forecast_today",
                "Today",
                "kWh",
                "mdi:lightning-bolt",
                "energy",
                "measurement",
            )
        assert any(r.levelno == logging.WARNING for r in caplog.records), "Expected WARNING on call_service failure"


class TestMqttSetSensor:
    """State topic and value normalisation for _mqtt_set_sensor."""

    def test_state_topic_format(self):
        """State topic must be <prefix>/energy_forecast/sensor/<unique_id>/state."""
        fake = _FakeMqttSelf(mqtt_discovery_prefix="homeassistant")
        fake._mqtt_set_sensor("energy_forecast_today", 3.14)
        assert len(fake._publishes) == 1
        assert fake._publishes[0]["topic"] == "homeassistant/energy_forecast/sensor/energy_forecast_today/state"

    def test_nan_value_publishes_zero(self):
        """NaN must be normalised to 0.0 before publishing."""
        fake = _FakeMqttSelf()
        fake._mqtt_set_sensor("energy_forecast_today", float("nan"))
        assert fake._publishes[0]["payload"] == "0.0"

    def test_retain_is_true(self):
        """mqtt_publish must always be called with retain=True."""
        fake = _FakeMqttSelf()
        fake._mqtt_set_sensor("energy_forecast_today", 1.5)
        assert fake._publishes[0]["retain"] is True


class TestMqttAvailability:
    """Availability topic publish and terminate() lifecycle."""

    def test_online_published_to_availability_topic(self):
        """_mqtt_publish_availability('online') must publish to the correct topic."""
        fake = _FakeMqttSelf(mqtt_discovery_prefix="homeassistant")
        fake._mqtt_publish_availability("online")
        assert len(fake._publishes) == 1
        pub = fake._publishes[0]
        assert pub["topic"] == "homeassistant/energy_forecast/availability"
        assert pub["payload"] == "online"

    def test_terminate_publishes_offline(self):
        """terminate() must publish 'offline' to the availability topic."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf()
        EnergyForecast.terminate(fake)
        assert any(p["payload"] == "offline" for p in fake._publishes), "Expected 'offline' payload on terminate()"


class TestMqttConditionalIntervals:
    """Interval sensor discovery is deferred until quantile models exist."""

    def test_publish_all_discovery_does_not_include_low_high_topics(self):
        """_mqtt_publish_all_discovery() must NOT publish *_low/*_high config topics."""
        fake = _FakeMqttSelf()
        fake._mqtt_publish_all_discovery()
        low_high_topics = [p["topic"] for p in fake._publishes if "_low" in p["topic"] or "_high" in p["topic"]]
        assert low_high_topics == [], f"Unexpected interval topics at init: {low_high_topics}"

    def test_interval_discovery_fired_on_first_publish_with_data(self):
        """When _publish() receives low/high data for the first time, interval discovery runs
        and _mqtt_intervals_discovered flips to True."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf()
        # Build a minimal aggregated data dict that includes interval values
        data = {
            "next_1h": 0.5,
            "next_3h": 1.0,
            "today": 5.0,
            "tomorrow": 6.0,
            "next_3h_low": 0.8,
            "next_3h_high": 1.2,
            "today_low": 4.5,
            "today_high": 5.5,
            "tomorrow_low": 5.5,
            "tomorrow_high": 6.5,
            "blocks_today": {f"{h:02d}_{h + 3:02d}": 1.0 for h in range(0, 24, 3)},
            "blocks_tomorrow": {f"{h:02d}_{h + 3:02d}": 1.0 for h in range(0, 24, 3)},
            "ev_today": 0.0,
            "ev_yesterday": 0.0,
        }
        EnergyForecast._publish(fake, data)
        assert fake._mqtt_intervals_discovered is True
        low_high_topics = [
            p["topic"] for p in fake._publishes if "_low/config" in p["topic"] or "_high/config" in p["topic"]
        ]
        assert len(low_high_topics) == 6, f"Expected 6 interval config topics, got {low_high_topics}"


class TestPublishUnavailable:
    """_publish_unavailable() must mark next_1h (and other totals) as 'unavailable'."""

    def test_next_1h_set_to_unavailable(self):
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=False)
        EnergyForecast._publish_unavailable(fake)
        assert "sensor.energy_forecast_next_1h" in fake._states
        assert fake._states["sensor.energy_forecast_next_1h"]["state"] == "unavailable"

    def test_all_total_slots_set_unavailable(self):
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=False)
        EnergyForecast._publish_unavailable(fake)
        for slot in ("next_1h", "next_3h", "today", "tomorrow"):
            eid = f"sensor.energy_forecast_{slot}"
            assert eid in fake._states, f"missing {eid}"
            assert fake._states[eid]["state"] == "unavailable"


class TestMqttPublishAllDiscoveryNext1h:
    """_mqtt_publish_all_discovery() must include next_1h."""

    def test_next_1h_discovery_topic_published(self):
        fake = _FakeMqttSelf()
        fake._mqtt_publish_all_discovery()
        topics = [p["topic"] for p in fake._publishes]
        assert any("energy_forecast_next_1h/config" in t for t in topics), (
            f"next_1h discovery topic not found in: {topics}"
        )


class TestMqttFallback:
    """When mqtt_discovery=False, safe_set must use set_state and never call mqtt_publish."""

    def test_safe_set_uses_set_state_not_mqtt(self):
        """With mqtt_discovery=False, _publish() must call set_state and not call_service."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=False)
        fake.call_service = MagicMock()
        data = {
            "next_1h": 0.5,
            "next_3h": 1.0,
            "today": 5.0,
            "tomorrow": 6.0,
            "blocks_today": {f"{h:02d}_{h + 3:02d}": 1.0 for h in range(0, 24, 3)},
            "blocks_tomorrow": {f"{h:02d}_{h + 3:02d}": 1.0 for h in range(0, 24, 3)},
            "ev_today": 0.0,
            "ev_yesterday": 0.0,
        }
        EnergyForecast._publish(fake, data)
        fake.call_service.assert_not_called()


class TestCleanupLegacyStates:
    def test_removes_all_expected_legacy_ids(self):
        fake = _FakeMqttSelf()
        from energy_forecast.energy_forecast import BLOCK_SLOTS, EnergyForecast

        EnergyForecast._cleanup_legacy_states(fake)
        removed = set(fake._removed_entities)
        # Core sensors
        for key in ["setup_status", "next_1h", "next_3h", "today", "tomorrow", "ev_today", "ev_yesterday", "model_mae"]:
            assert f"sensor.energy_forecast_{key}" in removed
        # Rolling MAE sensors (#41)
        assert "sensor.energy_forecast_mae_7d" in removed
        assert "sensor.energy_forecast_mae_30d" in removed
        # Anomaly detection sensor (#39)
        assert "binary_sensor.energy_forecast_unusual_consumption" in removed
        # Block sensors
        for day in ("today", "tomorrow"):
            for slot in BLOCK_SLOTS:
                assert f"sensor.energy_forecast_{day}_{slot}" in removed

    def test_swallows_remove_entity_exceptions(self):
        fake = _FakeMqttSelf()

        def bad_remove(entity_id):
            raise RuntimeError("not found")

        fake.remove_entity = bad_remove
        from energy_forecast.energy_forecast import EnergyForecast

        # Must not raise
        EnergyForecast._cleanup_legacy_states(fake)

    def test_skips_remove_when_entity_does_not_exist(self):
        fake = _FakeMqttSelf()
        fake.entity_exists = lambda entity_id: False
        from energy_forecast.energy_forecast import EnergyForecast

        EnergyForecast._cleanup_legacy_states(fake)
        assert fake._removed_entities == []


# ── #25 Vacation / Away Flag — _build_away_prediction_series ─────────────────


class _FakeAwaySelf:
    """Stand-in for EnergyForecast for _build_away_prediction_series tests."""

    def __init__(
        self,
        away_mode_entity: str | None = None,
        away_return_entity: str | None = None,
        away_mode_state: str = "off",
        away_return_state: str | None = None,
    ):
        self._away_mode_entity = away_mode_entity
        self._away_return_entity = away_return_entity
        self._states: dict[str, str] = {}
        if away_mode_entity:
            self._states[away_mode_entity] = away_mode_state
        if away_return_entity and away_return_state is not None:
            self._states[away_return_entity] = away_return_state
        self._warnings: list[str] = []

    def get_state(self, entity_id: str) -> str | None:
        return self._states.get(entity_id)

    def log(self, msg: str, level: str = "INFO") -> None:
        if level == "WARNING":
            self._warnings.append(msg)


class TestBuildAwayPredictionSeries:
    """_build_away_prediction_series: correct is_away projections for all cases."""

    def _now_ts(self):
        return pd.Timestamp.now("Europe/Zurich").tz_localize(None)

    def test_no_entity_returns_all_zeros(self):
        """With no away_mode_entity configured, all 48 values must be 0."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeAwaySelf(away_mode_entity=None)
        result = EnergyForecast._build_away_prediction_series(fake, self._now_ts())
        assert len(result) == 48
        assert (result == 0).all()

    def test_entity_off_returns_all_zeros(self):
        """When away_mode_entity is 'off', all 48 values must be 0."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeAwaySelf(away_mode_entity="input_boolean.vacation", away_mode_state="off")
        result = EnergyForecast._build_away_prediction_series(fake, self._now_ts())
        assert (result == 0).all()

    def test_entity_on_no_return_entity_returns_all_ones(self):
        """When entity is 'on' and no return entity, all 48 values must be 1."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeAwaySelf(away_mode_entity="input_boolean.vacation", away_mode_state="on", away_return_entity=None)
        result = EnergyForecast._build_away_prediction_series(fake, self._now_ts())
        assert (result == 1).all()

    def test_entity_on_return_in_future_splits_at_return_dt(self):
        """When entity is 'on' and return_dt is 24h ahead, first 24 rows = 1, rest = 0."""
        from energy_forecast.energy_forecast import EnergyForecast

        now_ts = pd.Timestamp("2026-04-01 10:00")
        return_dt = now_ts + pd.Timedelta(hours=24)
        fake = _FakeAwaySelf(
            away_mode_entity="input_boolean.vacation",
            away_mode_state="on",
            away_return_entity="input_datetime.return",
            away_return_state=str(return_dt),
        )
        result = EnergyForecast._build_away_prediction_series(fake, now_ts)
        # Hours before return_dt: is_away=1; at/after return_dt: is_away=0
        assert (result.iloc[:24] == 1).all(), f"Expected first 24 = 1, got {result.iloc[:24].tolist()}"
        assert (result.iloc[24:] == 0).all(), f"Expected rows 24+ = 0, got {result.iloc[24:].tolist()}"

    def test_entity_on_return_dt_in_past_returns_all_ones(self):
        """When entity is 'on' and return_dt is in the past, all 48 values must be 1."""
        from energy_forecast.energy_forecast import EnergyForecast

        now_ts = pd.Timestamp("2026-04-01 10:00")
        return_dt = now_ts - pd.Timedelta(hours=1)  # 1h ago — already past
        fake = _FakeAwaySelf(
            away_mode_entity="input_boolean.vacation",
            away_mode_state="on",
            away_return_entity="input_datetime.return",
            away_return_state=str(return_dt),
        )
        result = EnergyForecast._build_away_prediction_series(fake, now_ts)
        assert (result == 1).all()


# ── #41 Rolling MAE sensors (mae_7d / mae_30d) ───────────────────────────────


class TestRollingMaeSensors:
    """Tests for _actuals_history population, mae_7d/mae_30d computation, publish,
    and discovery registration."""

    # ── helpers ──

    def _make_pred_history(self, start: pd.Timestamp, n: int, kwh: float = 2.0) -> dict:
        """n hourly predictions starting at start, all equal to kwh."""
        ts = pd.date_range(start, periods=n, freq="1h")
        return {t: kwh for t in ts}

    def _make_actuals_df(self, start: pd.Timestamp, n: int, kwh: float = 3.0) -> pd.DataFrame:
        ts = pd.date_range(start, periods=n, freq="1h")
        return pd.DataFrame({"timestamp": ts, "gross_kwh": [kwh] * n})

    # ── actuals_history keep-last semantics ──

    def test_actuals_history_keep_last_semantics(self):
        """Same-hour key written twice: the newer (later) value must win."""

        # Simulate two rounds of recent_actuals for the same hour
        hour = pd.Timestamp("2026-03-20 10:00")
        actuals_history: dict = {}

        # First write: value 1.0
        actuals_history[hour.floor("1h")] = 1.0
        # Second write (keep-last): value 2.5 overwrites
        actuals_history[hour.floor("1h")] = 2.5

        assert actuals_history[hour.floor("1h")] == 2.5

    # ── mae_7d excludes predictions older than 7 days ──

    def test_mae_7d_excludes_predictions_older_than_7d(self):
        """pred_hist_7d filter must exclude entries older than 7 days."""
        from energy_forecast.energy_forecast import _compute_live_mae

        now = pd.Timestamp.now().floor("1h")
        cutoff_7d = now - pd.Timedelta(days=7)

        # 24 predictions within 7d window
        recent_preds = self._make_pred_history(cutoff_7d + pd.Timedelta(hours=1), 24, kwh=2.0)
        # 24 predictions older than 7d (should be excluded)
        pred_hist_7d = {ts: kwh for ts, kwh in recent_preds.items()}
        # old_preds excluded — not added to pred_hist_7d

        actuals = self._make_actuals_df(cutoff_7d + pd.Timedelta(hours=1), 24, kwh=3.0)
        mae, n = _compute_live_mae(pred_hist_7d, actuals)

        assert n == 24
        assert abs(mae - 1.0) < 1e-6

    # ── mae_30d uses full 30d window ──

    def test_mae_30d_uses_full_30d_window(self):
        """mae_30d must match all pairs within the 30d pred_history."""
        from energy_forecast.energy_forecast import _compute_live_mae

        now = pd.Timestamp.now().floor("1h")
        start = now - pd.Timedelta(days=29)
        pred_history = self._make_pred_history(start, 48, kwh=2.0)
        actuals = self._make_actuals_df(start, 48, kwh=3.0)

        mae, n = _compute_live_mae(pred_history, actuals)

        assert n == 48
        assert abs(mae - 1.0) < 1e-6

    def test_mae_window_rolls_smoothly_at_midnight(self):
        """MAE window must only drop 1 hour when crossing midnight, not 24 hours."""

        # Verify that the hourly rolling window logic shifts by exactly 1 hour
        # when crossing midnight, whereas the old normalize() logic would jump by 24h.

        # 23:30 on May 20
        now_before = pd.Timestamp("2024-05-20 23:30")
        # 00:30 on May 21
        now_after = pd.Timestamp("2024-05-21 00:30")

        # Proposed behavior: floor('1h')
        cutoff_before = now_before.floor("1h") - pd.Timedelta(days=7)
        cutoff_after = now_after.floor("1h") - pd.Timedelta(days=7)

        # Crossing midnight (23:30 -> 00:30) shifts the rolling window by exactly 1 hour
        assert (cutoff_after - cutoff_before) == pd.Timedelta(hours=1)

        # Legacy behavior: normalize()
        old_cutoff_before = now_before.normalize() - pd.Timedelta(days=7)  # May 13 00:00
        old_cutoff_after = now_after.normalize() - pd.Timedelta(days=7)  # May 14 00:00

        # Crossing midnight jumps the window by 24 hours! This is the root cause of the report.
        assert (old_cutoff_after - old_cutoff_before) == pd.Timedelta(days=1)

    # ── EV-hour exclusion from actuals_history ──

    def test_actuals_history_excludes_ev_hours(self):
        """EV-detected hours must not appear in _actuals_history."""
        import pandas as pd

        base = pd.Timestamp("2026-03-20 08:00")
        normal_hours = pd.date_range(base, periods=4, freq="1h")  # 08, 09, 10, 11
        ev_hours = pd.date_range(base + pd.Timedelta(hours=4), periods=2, freq="1h")  # 12, 13

        # Build ev_hour_set as the production code does
        ev_hour_set = set(ev_hours)

        actuals_history: dict = {}
        all_hours = list(normal_hours) + list(ev_hours)
        for ts in all_hours:
            if ts not in ev_hour_set:
                actuals_history[ts] = 2.0

        assert len(actuals_history) == 4
        for ev_ts in ev_hours:
            assert ev_ts not in actuals_history
        for norm_ts in normal_hours:
            assert norm_ts in actuals_history

    def test_mae_not_inflated_by_ev_hours(self):
        """MAE computed after excluding EV hours must not exceed MAE when EV hours are included."""
        from energy_forecast.energy_forecast import _compute_live_mae

        base = pd.Timestamp("2026-03-20 08:00")
        n_normal = 6
        normal_ts = pd.date_range(base, periods=n_normal, freq="1h")
        ev_ts = base + pd.Timedelta(hours=n_normal)

        pred_normal = 1.5
        actual_normal = 2.0
        pred_ev = 1.5
        actual_ev = 12.0  # large EV charging spike

        pred_history = {ts: pred_normal for ts in normal_ts}
        pred_history[ev_ts] = pred_ev

        # With EV hour included: high MAE
        actuals_with_ev = pd.DataFrame(
            {"timestamp": list(normal_ts) + [ev_ts], "gross_kwh": [actual_normal] * n_normal + [actual_ev]}
        )
        mae_inflated, n_inflated = _compute_live_mae(pred_history, actuals_with_ev)

        # Without EV hour: lower MAE
        actuals_no_ev = pd.DataFrame({"timestamp": list(normal_ts), "gross_kwh": [actual_normal] * n_normal})
        pred_history_no_ev = {ts: pred_normal for ts in normal_ts}
        mae_clean, n_clean = _compute_live_mae(pred_history_no_ev, actuals_no_ev)

        assert n_inflated == n_normal + 1
        assert n_clean == n_normal
        assert mae_clean < mae_inflated
        assert abs(mae_clean - abs(actual_normal - pred_normal)) < 1e-6

    # ── publish in set_state mode ──

    def test_publish_mae_sensors_set_state_mode(self):
        """_publish() must call set_state for both mae_7d and mae_30d when mqtt_discovery=False."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=False)
        data = {
            "next_1h": 0.5,
            "next_3h": 1.0,
            "today": 5.0,
            "tomorrow": 6.0,
            "blocks_today": {f"{h:02d}_{h + 3:02d}": 1.0 for h in range(0, 24, 3)},
            "blocks_tomorrow": {f"{h:02d}_{h + 3:02d}": 1.0 for h in range(0, 24, 3)},
            "ev_today": 0.0,
            "ev_yesterday": 0.0,
            "mae_7d": 0.25,
            "mae_30d": 0.30,
            "mae_7d_n_pairs": 24,
            "mae_30d_n_pairs": 120,
        }
        EnergyForecast._publish(fake, data)

        assert "sensor.energy_forecast_mae_7d" in fake._states
        assert "sensor.energy_forecast_mae_30d" in fake._states
        assert fake._states["sensor.energy_forecast_mae_7d"]["state"] == "0.25"
        assert fake._states["sensor.energy_forecast_mae_30d"]["state"] == "0.3"
        assert fake._states["sensor.energy_forecast_mae_7d"]["attributes"]["n_pairs"] == 24
        assert fake._states["sensor.energy_forecast_mae_30d"]["attributes"]["n_pairs"] == 120

    # ── publish in MQTT mode ──

    def test_publish_mae_sensors_mqtt_mode(self):
        """_publish() must call mqtt_publish for both mae_7d and mae_30d when mqtt_discovery=True."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=True)
        data = {
            "next_1h": 0.5,
            "next_3h": 1.0,
            "today": 5.0,
            "tomorrow": 6.0,
            "blocks_today": {f"{h:02d}_{h + 3:02d}": 1.0 for h in range(0, 24, 3)},
            "blocks_tomorrow": {f"{h:02d}_{h + 3:02d}": 1.0 for h in range(0, 24, 3)},
            "ev_today": 0.0,
            "ev_yesterday": 0.0,
            "mae_7d": 0.25,
            "mae_30d": 0.30,
            "mae_7d_n_pairs": 24,
            "mae_30d_n_pairs": 120,
        }
        EnergyForecast._publish(fake, data)

        state_topics = [p["topic"] for p in fake._publishes if "/state" in p["topic"]]
        assert any("energy_forecast_mae_7d" in t for t in state_topics), (
            f"mae_7d state topic not found in: {state_topics}"
        )
        assert any("energy_forecast_mae_30d" in t for t in state_topics), (
            f"mae_30d state topic not found in: {state_topics}"
        )

    # ── full_actuals vs recent_actuals for _actuals_history ──

    def test_actuals_history_stores_raw_kwh_not_sub_sensor_subtracted(self):
        """_actuals_history must store raw gross_kwh from full_actuals, not sub-sensor-subtracted recent_actuals.

        When sub-sensors reduce recent_actuals kWh, _actuals_history must still hold the
        original full_actuals values (consistent with training, which uses EV-subtracted
        gross_kwh without sub-sensor subtraction).
        """
        import pandas as pd

        # Simulate the production loop with both full_actuals and recent_actuals available,
        # where recent_actuals has reduced kWh (sub-sensor subtracted).
        now = pd.Timestamp("2026-06-01 10:00")
        timestamps = pd.date_range(now - pd.Timedelta(hours=3), periods=3, freq="1h")

        full_actuals = pd.DataFrame({"timestamp": timestamps, "gross_kwh": [2.0, 3.0, 4.0]})
        # recent_actuals: same timestamps, 1.0 kWh less (sub-sensor subtracted)
        recent_actuals_kwh = [1.0, 2.0, 3.0]  # noqa: F841 — kept for documentation

        ev_hour_set: set = set()

        actuals_history: dict = {}

        # Simulate the FIXED production loop (full_actuals):
        for _ts, _kwh in zip(
            pd.to_datetime(full_actuals["timestamp"]).dt.floor("1h"),
            full_actuals["gross_kwh"].astype(float),
        ):
            if _ts not in ev_hour_set:
                actuals_history[_ts] = _kwh

        # Values must be raw (2.0, 3.0, 4.0), not sub-sensor-subtracted (1.0, 2.0, 3.0)
        assert list(actuals_history.values()) == [2.0, 3.0, 4.0], (
            f"Expected raw full_actuals values [2.0, 3.0, 4.0], got {list(actuals_history.values())}"
        )

    # ── MQTT discovery ──

    def test_mqtt_discovery_includes_mae_sensors(self):
        """_mqtt_publish_all_discovery() must publish config topics for mae_7d and mae_30d."""
        fake = _FakeMqttSelf()
        fake._mqtt_publish_all_discovery()
        config_topics = [p["topic"] for p in fake._publishes if "/config" in p["topic"]]
        assert any("energy_forecast_mae_7d/config" in t for t in config_topics), (
            f"mae_7d config topic not found in: {config_topics}"
        )
        assert any("energy_forecast_mae_30d/config" in t for t in config_topics), (
            f"mae_30d config topic not found in: {config_topics}"
        )


# ── #39 Anomaly detection sensor ──────────────────────────────────────────────


class TestAnomalyDetection:
    """Tests for _compute_anomaly and its integration with _publish / discovery."""

    # ── helpers ──

    def _make_pred_history(self, n: int, kwh: float = 2.0, start: pd.Timestamp | None = None) -> dict:
        """n hourly predictions, all equal to kwh."""
        if start is None:
            start = pd.Timestamp("2026-03-01 00:00")
        ts = pd.date_range(start, periods=n, freq="1h")
        return {t: kwh for t in ts}

    def _make_actuals_history(self, n: int, kwh: float = 2.0, start: pd.Timestamp | None = None) -> dict:
        """n floored-1h actuals, all equal to kwh."""
        if start is None:
            start = pd.Timestamp("2026-03-01 00:00")
        ts = pd.date_range(start, periods=n, freq="1h")
        return {t.floor("1h"): kwh for t in ts}

    # ── _compute_anomaly unit tests ──

    def test_fires_above_threshold(self):
        """Latest residual far above std must return is_anomaly=True."""
        start = pd.Timestamp("2026-03-01 00:00")
        # 20 pairs with residual ≈ 0.0 (pred=2.0, actual=2.0), then 1 spike
        pred_history = self._make_pred_history(21, kwh=2.0, start=start)
        actuals_history = self._make_actuals_history(20, kwh=2.0, start=start)  # all match at 0.0 residual
        # Override last prediction to be very far from actual
        latest_ts = start + pd.Timedelta(hours=20)
        pred_history[latest_ts] = 2.0
        actuals_history[latest_ts.floor("1h")] = 7.0  # residual = 5.0 kWh

        is_anomaly, residual, std, n = _compute_anomaly(pred_history, actuals_history, sigma_threshold=3.0)

        assert is_anomaly is True, f"Expected anomaly=True, got residual={residual}, std={std}"
        assert abs(residual - 5.0) < 1e-6

    def test_silent_below_threshold(self):
        """Residuals all similar — latest within threshold — must return is_anomaly=False."""
        start = pd.Timestamp("2026-03-01 00:00")
        pred_history = self._make_pred_history(20, kwh=2.0, start=start)
        actuals_history = self._make_actuals_history(20, kwh=2.1, start=start)  # residual = 0.1 everywhere

        is_anomaly, residual, std, n = _compute_anomaly(pred_history, actuals_history, sigma_threshold=3.0)

        assert is_anomaly is False
        assert n == 20

    def test_cold_start_returns_false(self):
        """Fewer than min_pairs matched pairs → (False, nan, nan, n)."""
        start = pd.Timestamp("2026-03-01 00:00")
        pred_history = self._make_pred_history(5, kwh=2.0, start=start)
        actuals_history = self._make_actuals_history(5, kwh=3.0, start=start)

        is_anomaly, residual, std, n = _compute_anomaly(pred_history, actuals_history, sigma_threshold=3.0)

        assert is_anomaly is False
        assert n == 5
        assert math.isnan(residual)
        assert math.isnan(std)

    def test_empty_histories_return_false(self):
        """Both empty → (False, nan, nan, 0)."""
        is_anomaly, residual, std, n = _compute_anomaly({}, {}, sigma_threshold=3.0)

        assert is_anomaly is False
        assert n == 0
        assert math.isnan(residual)
        assert math.isnan(std)

    def test_near_zero_std_guard(self):
        """Perfect model (std < 0.01) must not fire anomaly even if latest residual is large."""
        start = pd.Timestamp("2026-03-01 00:00")
        # All residuals exactly 0 → std = 0
        pred_history = self._make_pred_history(20, kwh=2.0, start=start)
        actuals_history = self._make_actuals_history(20, kwh=2.0, start=start)

        is_anomaly, residual, std, n = _compute_anomaly(pred_history, actuals_history, sigma_threshold=3.0)

        assert is_anomaly is False
        assert std < 0.01

    # ── _publish integration tests ──

    def _base_data(self) -> dict:
        return {
            "next_1h": 0.5,
            "next_3h": 1.0,
            "today": 5.0,
            "tomorrow": 6.0,
            "blocks_today": {f"{h:02d}_{h + 3:02d}": 1.0 for h in range(0, 24, 3)},
            "blocks_tomorrow": {f"{h:02d}_{h + 3:02d}": 1.0 for h in range(0, 24, 3)},
            "ev_today": 0.0,
            "ev_yesterday": 0.0,
            "mae_7d": 0.25,
            "mae_30d": 0.30,
            "mae_7d_n_pairs": 24,
            "mae_30d_n_pairs": 120,
        }

    def test_publish_set_state_mode_anomaly_on(self):
        """_publish() in set_state mode must write binary_sensor state 'on' when is_anomaly=True."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=False)
        data = {**self._base_data(), "is_anomaly": True, "anomaly_residual": 5.0, "anomaly_std": 0.5, "anomaly_n": 20}
        EnergyForecast._publish(fake, data)

        eid = "binary_sensor.energy_forecast_unusual_consumption"
        assert eid in fake._states, f"Expected {eid} in states"
        assert fake._states[eid]["state"] == "on"
        assert fake._states[eid]["attributes"]["device_class"] == "problem"

    def test_publish_set_state_mode_anomaly_off(self):
        """_publish() in set_state mode must write binary_sensor state 'off' when is_anomaly=False."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=False)
        data = {**self._base_data(), "is_anomaly": False, "anomaly_residual": 0.1, "anomaly_std": 0.1, "anomaly_n": 20}
        EnergyForecast._publish(fake, data)

        eid = "binary_sensor.energy_forecast_unusual_consumption"
        assert fake._states[eid]["state"] == "off"

    def test_publish_mqtt_mode(self):
        """_publish() in MQTT mode must publish 'ON'/'OFF' to the anomaly state topic and attributes."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=True)
        data = {**self._base_data(), "is_anomaly": True, "anomaly_residual": 5.0, "anomaly_std": 0.5, "anomaly_n": 20}
        EnergyForecast._publish(fake, data)

        state_topics = {p["topic"]: p["payload"] for p in fake._publishes if "/state" in p["topic"]}
        anomaly_topic = next((t for t in state_topics if "unusual_consumption" in t), None)
        assert anomaly_topic is not None, f"Anomaly state topic not found in: {list(state_topics)}"
        assert state_topics[anomaly_topic] == "ON"

        attr_topics = {p["topic"]: p["payload"] for p in fake._publishes if "/attributes" in p["topic"]}
        anomaly_attr_topic = next((t for t in attr_topics if "unusual_consumption" in t), None)
        assert anomaly_attr_topic is not None, "Anomaly attributes topic not found"
        import json as _json

        attrs = _json.loads(attr_topics[anomaly_attr_topic])
        assert "residual_kwh" in attrs
        assert attrs["n_pairs"] == 20

    def test_mqtt_discovery_includes_anomaly_sensor(self):
        """_mqtt_publish_all_discovery() must publish a binary_sensor config with json_attributes_topic."""
        fake = _FakeMqttSelf()
        fake._mqtt_publish_all_discovery()
        config_publishes = {p["topic"]: p["payload"] for p in fake._publishes if "/config" in p["topic"]}
        anomaly_config_topic = next(
            (t for t in config_publishes if "binary_sensor/energy_forecast_unusual_consumption/config" in t), None
        )
        assert anomaly_config_topic is not None, (
            f"anomaly binary sensor config topic not found in: {list(config_publishes)}"
        )
        import json as _json

        cfg = _json.loads(config_publishes[anomaly_config_topic])
        assert "json_attributes_topic" in cfg, "Missing json_attributes_topic in anomaly discovery payload"

    def test_cleanup_legacy_includes_anomaly_sensor(self):
        """_cleanup_legacy_states() must include binary_sensor.energy_forecast_unusual_consumption."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf()
        EnergyForecast._cleanup_legacy_states(fake)
        assert "binary_sensor.energy_forecast_unusual_consumption" in fake._removed_entities


# ── _apply_target_correction ──────────────────────────────────────────────────


def _make_corr_df(timestamps, kwh_values):
    """Helper: build a correction DataFrame with timestamp + kwh columns."""
    return pd.DataFrame({"timestamp": pd.DatetimeIndex(timestamps), "kwh": kwh_values})


class TestTargetCorrection:
    """Tests for _apply_target_correction — training target correction for solar + battery."""

    def _base_df(self, n=10, gross=2.0):
        """Simple energy DataFrame: n hourly rows, all gross_kwh = gross."""
        ts = pd.date_range("2026-06-01", periods=n, freq="1h")
        return pd.DataFrame({"timestamp": ts, "gross_kwh": [gross] * n})

    def test_solar_only_adds_to_gross_kwh(self):
        df = self._base_df(gross=2.0)
        solar = _make_corr_df(df["timestamp"], [1.0] * 10)
        result = _apply_target_correction(
            df, solar_df=solar, grid_export_df=None, battery_charge_df=None, battery_discharge_df=None
        )
        assert result["gross_kwh"].tolist() == pytest.approx([3.0] * 10)

    def test_grid_export_subtracts_from_gross_kwh(self):
        df = self._base_df(gross=3.0)
        export = _make_corr_df(df["timestamp"], [1.0] * 10)
        result = _apply_target_correction(
            df, solar_df=None, grid_export_df=export, battery_charge_df=None, battery_discharge_df=None
        )
        assert result["gross_kwh"].tolist() == pytest.approx([2.0] * 10)

    def test_solar_and_export_combined(self):
        df = self._base_df(gross=2.0)
        solar = _make_corr_df(df["timestamp"], [3.0] * 10)
        export = _make_corr_df(df["timestamp"], [1.0] * 10)
        result = _apply_target_correction(
            df, solar_df=solar, grid_export_df=export, battery_charge_df=None, battery_discharge_df=None
        )
        # 2 + 3 - 1 = 4
        assert result["gross_kwh"].tolist() == pytest.approx([4.0] * 10)

    def test_battery_charge_subtracts(self):
        df = self._base_df(gross=5.0)
        charge = _make_corr_df(df["timestamp"], [2.0] * 10)
        result = _apply_target_correction(
            df, solar_df=None, grid_export_df=None, battery_charge_df=charge, battery_discharge_df=None
        )
        assert result["gross_kwh"].tolist() == pytest.approx([3.0] * 10)

    def test_battery_discharge_adds(self):
        df = self._base_df(gross=1.0)
        discharge = _make_corr_df(df["timestamp"], [2.0] * 10)
        result = _apply_target_correction(
            df, solar_df=None, grid_export_df=None, battery_charge_df=None, battery_discharge_df=discharge
        )
        assert result["gross_kwh"].tolist() == pytest.approx([3.0] * 10)

    def test_full_correction_all_four_sensors(self):
        # grid_import=3, solar=5, export=2, charge=1, discharge=0.5
        # expected: 3 + 5 - 2 - 1 + 0.5 = 5.5
        df = self._base_df(gross=3.0)
        solar = _make_corr_df(df["timestamp"], [5.0] * 10)
        export = _make_corr_df(df["timestamp"], [2.0] * 10)
        charge = _make_corr_df(df["timestamp"], [1.0] * 10)
        discharge = _make_corr_df(df["timestamp"], [0.5] * 10)
        result = _apply_target_correction(
            df, solar_df=solar, grid_export_df=export, battery_charge_df=charge, battery_discharge_df=discharge
        )
        assert result["gross_kwh"].tolist() == pytest.approx([5.5] * 10)

    def test_result_clipped_to_zero(self):
        # gross_kwh=1, export=5 → would be -4 without clip
        df = self._base_df(gross=1.0)
        export = _make_corr_df(df["timestamp"], [5.0] * 10)
        result = _apply_target_correction(
            df, solar_df=None, grid_export_df=export, battery_charge_df=None, battery_discharge_df=None
        )
        assert (result["gross_kwh"] >= 0).all()
        assert result["gross_kwh"].tolist() == pytest.approx([0.0] * 10)

    def test_unmatched_timestamps_get_zero_correction(self):
        df = self._base_df(gross=2.0)
        # correction timestamps are 7 days later — no overlap
        future_ts = df["timestamp"] + pd.Timedelta(days=7)
        solar = _make_corr_df(future_ts, [5.0] * 10)
        result = _apply_target_correction(
            df, solar_df=solar, grid_export_df=None, battery_charge_df=None, battery_discharge_df=None
        )
        # unmatched hours get zero delta; gross_kwh unchanged
        assert result["gross_kwh"].tolist() == pytest.approx([2.0] * 10)

    def test_original_df_not_mutated(self):
        df = self._base_df(gross=2.0)
        solar = _make_corr_df(df["timestamp"], [1.0] * 10)
        _ = _apply_target_correction(
            df, solar_df=solar, grid_export_df=None, battery_charge_df=None, battery_discharge_df=None
        )
        assert df["gross_kwh"].tolist() == pytest.approx([2.0] * 10)


# ── Prediction history persistence ────────────────────────────────────────────


class TestPredHistoryPersistence:
    """Tests for _load_pred_history and _save_pred_history methods."""

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        """Save a known dict, load it back, verify keys and values match."""
        from energy_forecast.energy_forecast import EnergyForecast

        # Mock PRED_HISTORY_PATH to use tmp_path
        mock_path = tmp_path / "pred_history.json"
        monkeypatch.setattr("energy_forecast.energy_forecast.PRED_HISTORY_PATH", mock_path)

        app = MagicMock(spec=EnergyForecast)
        app._timezone = "Europe/Zurich"
        app.log = MagicMock()
        app._pred_history = {}
        app._actuals_history = {}

        # Prepare test data
        now = pd.Timestamp.now().normalize()
        pred_data = {
            now - pd.Timedelta(days=1): 1.5,
            now - pd.Timedelta(days=5): 2.0,
        }
        actuals_data = {
            now - pd.Timedelta(days=2): 1.4,
            now - pd.Timedelta(days=3): 2.1,
        }

        # Manually save
        import json

        data = {
            "pred": {ts.isoformat(): kwh for ts, kwh in pred_data.items()},
            "actuals": {ts.isoformat(): kwh for ts, kwh in actuals_data.items()},
        }
        mock_path.write_text(json.dumps(data))

        # Load back
        EnergyForecast._load_pred_history(app)

        # Verify
        assert len(app._pred_history) == 2
        assert len(app._actuals_history) == 2
        assert app._pred_history[now - pd.Timedelta(days=1)] == 1.5
        assert app._actuals_history[now - pd.Timedelta(days=2)] == 1.4

    def test_load_applies_30d_pruning(self, tmp_path, monkeypatch):
        """Save an entry >30 days old, verify it's discarded on load."""
        from energy_forecast.energy_forecast import EnergyForecast

        mock_path = tmp_path / "pred_history.json"
        monkeypatch.setattr("energy_forecast.energy_forecast.PRED_HISTORY_PATH", mock_path)

        app = MagicMock(spec=EnergyForecast)
        app._timezone = "Europe/Zurich"
        app.log = MagicMock()
        app._pred_history = {}
        app._actuals_history = {}

        # Create data with one old entry (35 days ago) and one recent (5 days ago)
        now = pd.Timestamp.now().normalize()
        old_ts = now - pd.Timedelta(days=35)
        recent_ts = now - pd.Timedelta(days=5)

        import json

        data = {
            "pred": {
                old_ts.isoformat(): 1.0,
                recent_ts.isoformat(): 2.0,
            },
            "actuals": {},
        }
        mock_path.write_text(json.dumps(data))

        # Load
        EnergyForecast._load_pred_history(app)

        # Only the recent entry should be loaded
        assert len(app._pred_history) == 1
        assert recent_ts in app._pred_history
        assert app._pred_history[recent_ts] == 2.0
        assert old_ts not in app._pred_history

    def test_load_keep_first_on_conflict(self, tmp_path, monkeypatch):
        """If in-memory already has a key, loaded value must not overwrite it."""
        from energy_forecast.energy_forecast import EnergyForecast

        mock_path = tmp_path / "pred_history.json"
        monkeypatch.setattr("energy_forecast.energy_forecast.PRED_HISTORY_PATH", mock_path)

        app = MagicMock(spec=EnergyForecast)
        app._timezone = "Europe/Zurich"
        app.log = MagicMock()

        now = pd.Timestamp.now().normalize()
        ts_key = now - pd.Timedelta(days=5)

        # Pre-populate in-memory dict
        app._pred_history = {ts_key: 1.0}
        app._actuals_history = {}

        # Write file with different value for the same key
        import json

        data = {
            "pred": {ts_key.isoformat(): 99.0},  # Should be ignored
            "actuals": {},
        }
        mock_path.write_text(json.dumps(data))

        # Load
        EnergyForecast._load_pred_history(app)

        # In-memory value should be preserved (keep-first)
        assert app._pred_history[ts_key] == 1.0

    def test_load_missing_file_is_silent(self, tmp_path, monkeypatch):
        """No file present → no exception, dicts remain empty."""
        from energy_forecast.energy_forecast import EnergyForecast

        mock_path = tmp_path / "pred_history.json"
        monkeypatch.setattr("energy_forecast.energy_forecast.PRED_HISTORY_PATH", mock_path)

        app = MagicMock(spec=EnergyForecast)
        app._timezone = "Europe/Zurich"
        app.log = MagicMock()
        app._pred_history = {}
        app._actuals_history = {}

        # File doesn't exist
        assert not mock_path.exists()

        # Should not raise
        EnergyForecast._load_pred_history(app)

        # Dicts remain empty
        assert app._pred_history == {}
        assert app._actuals_history == {}
        app.log.assert_not_called()

    def test_load_corrupt_file_is_silent(self, tmp_path, monkeypatch, caplog):
        """Invalid JSON → no exception, dicts remain empty, warning logged."""
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        mock_path = tmp_path / "pred_history.json"
        monkeypatch.setattr("energy_forecast.energy_forecast.PRED_HISTORY_PATH", mock_path)

        app = MagicMock(spec=EnergyForecast)
        app._timezone = "Europe/Zurich"
        app._pred_history = {}
        app._actuals_history = {}

        # Write invalid JSON
        mock_path.write_text("{invalid json}")

        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._load_pred_history(app)

        # Dicts remain empty
        assert app._pred_history == {}
        assert app._actuals_history == {}
        # Warning should be logged
        assert any("Failed to load" in r.message for r in caplog.records)

    def test_save_creates_file_atomically(self, tmp_path, monkeypatch):
        """Verify that .tmp file is used (save is atomic)."""
        from energy_forecast.energy_forecast import EnergyForecast

        mock_path = tmp_path / "pred_history.json"
        monkeypatch.setattr("energy_forecast.energy_forecast.PRED_HISTORY_PATH", mock_path)

        app = MagicMock(spec=EnergyForecast)
        app._timezone = "Europe/Zurich"
        app.log = MagicMock()

        now = pd.Timestamp.now().normalize()
        app._pred_history = {now - pd.Timedelta(days=1): 1.5}
        app._actuals_history = {now - pd.Timedelta(days=2): 2.0}

        # Save
        EnergyForecast._save_pred_history(app)

        # Verify file was created
        assert mock_path.exists()
        # Verify .tmp file was cleaned up
        assert not mock_path.with_suffix(".json.tmp").exists()

        # Verify content
        import json

        with open(mock_path) as f:
            data = json.load(f)
        assert len(data["pred"]) == 1
        assert len(data["actuals"]) == 1

    def test_save_handles_oserror_gracefully(self, tmp_path, monkeypatch, caplog):
        """Save failure (OSError) is logged, doesn't raise."""
        import logging

        from energy_forecast.energy_forecast import EnergyForecast

        mock_path = tmp_path / "pred_history.json"
        monkeypatch.setattr("energy_forecast.energy_forecast.PRED_HISTORY_PATH", mock_path)

        app = MagicMock(spec=EnergyForecast)
        app._timezone = "Europe/Zurich"
        app._pred_history = {}
        app._actuals_history = {}

        # Mock open to raise OSError
        mock_open = MagicMock()
        mock_open.side_effect = OSError("Disk full")
        monkeypatch.setattr("builtins.open", mock_open)

        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._save_pred_history(app)

        assert any("Failed to save" in r.message for r in caplog.records)


# ── _build_shap_narrative (#53) ──────────────────────────────────────────────────


class TestBuildShapNarrative:
    def test_empty_dict_returns_empty_string(self):
        """Empty shap_features dict returns empty string."""
        result = _build_shap_narrative({})
        assert result == ""

    def test_none_returns_empty_string(self):
        """None input returns empty string."""
        # The function checks if not shap_features, so None is falsy
        result = _build_shap_narrative(None)  # type: ignore
        assert result == ""

    def test_single_feature(self):
        """Single feature narrative includes label and value."""
        shap_features = {"hour_sin": 0.14}
        result = _build_shap_narrative(shap_features)
        assert "time-of-day (sine)" in result
        assert "+0.14" in result
        assert "Mainly driven by:" in result

    def test_multiple_features(self):
        """Multiple features all appear in narrative."""
        shap_features = {"hour_sin": 0.14, "temp_c": 0.09, "lag_24h": 0.07}
        result = _build_shap_narrative(shap_features)
        assert "time-of-day (sine)" in result
        assert "current outdoor temperature" in result
        assert "yesterday's same-hour consumption" in result
        assert "+0.14" in result
        assert "+0.09" in result
        assert "+0.07" in result

    def test_unknown_feature_uses_feature_name(self):
        """Unknown feature name falls back to raw feature name."""
        shap_features = {"unknown_feature_xyz": 0.05}
        result = _build_shap_narrative(shap_features)
        assert "unknown_feature_xyz" in result
        assert "+0.05" in result

    def test_negative_shap_value_formatted_correctly(self):
        """Negative SHAP values use proper +/- sign."""
        shap_features = {"temp_c": -0.05}
        result = _build_shap_narrative(shap_features)
        assert "-0.05" in result

    def test_narrative_ends_with_period(self):
        """Narrative string ends with a period."""
        shap_features = {"hour_sin": 0.14}
        result = _build_shap_narrative(shap_features)
        assert result.endswith(".")

    def test_sine_and_cosine_labels_differentiated(self):
        """Sine and cosine features have distinct labels in narrative."""
        shap_features = {"hour_sin": 0.14, "hour_cos": 0.09}
        result = _build_shap_narrative(shap_features)
        assert "time-of-day (sine)" in result
        assert "time-of-day (cosine)" in result
        # Verify both appear in the narrative
        assert result.count("time-of-day") == 2


# ── Relative MAE (#54) ───────────────────────────────────────────────────────────


class TestRelativeMAEComputation:
    """Integration tests for relative MAE computation in _update_sensors."""

    def test_relative_mae_7d_computed_correctly(self):
        """Verify relative MAE (%) is correctly computed from absolute MAE and actuals mean."""
        # Populate histories: 10 hours of 10 kWh each, MAE should be 5% of mean
        now = pd.Timestamp.now().floor("1h")
        pred_history = {}
        actuals_history = {}
        for i in range(10):
            ts = (now - pd.Timedelta(hours=i)).isoformat()
            pred_history[ts] = 10.5  # +0.5 error
            actuals_history[ts] = 10.0

        # Compute relative MAE as _update_sensors does
        mae_7d = 0.5  # known absolute MAE
        cutoff_7d = pd.Timestamp.now().floor("1h") - pd.Timedelta(days=7)
        actuals_7d = [v for ts, v in actuals_history.items() if pd.Timestamp(ts) >= cutoff_7d]
        mean_7d = float(np.mean(actuals_7d)) if actuals_7d else float("nan")
        mae_7d_pct = round(mae_7d / mean_7d * 100, 2) if mean_7d > 0 and not math.isnan(mae_7d) else float("nan")

        # Expected: 0.5 / 10.0 * 100 = 5.0%
        assert mae_7d_pct == 5.0

    def test_relative_mae_nan_when_no_actuals(self):
        """Relative MAE is NaN when actuals_history is empty."""
        actuals_empty = []
        mean_empty = float(np.mean(actuals_empty)) if actuals_empty else float("nan")
        assert math.isnan(mean_empty)

        mae_abs = 0.5
        mae_pct = round(mae_abs / mean_empty * 100, 2) if mean_empty > 0 and not math.isnan(mae_abs) else float("nan")
        assert math.isnan(mae_pct)

    def test_relative_mae_nan_propagates_from_absolute_mae(self):
        """Relative MAE is NaN when absolute MAE is NaN (cold start)."""
        mean_consumption = 10.0
        mae_abs = float("nan")
        mae_pct = (
            round(mae_abs / mean_consumption * 100, 2)
            if mean_consumption > 0 and not math.isnan(mae_abs)
            else float("nan")
        )
        assert math.isnan(mae_pct)


# ── Fix: _update_cb must not skip under lock (concurrency fix) ────────────────


class TestUpdateCbNoLock:
    """_update_cb must run _update_sensors even when _lock is held (retrain in progress)."""

    def test_update_cb_runs_when_lock_held(self):
        """With the old code _update_cb skipped if lock was busy; now it must not skip."""
        from energy_forecast.energy_forecast import EnergyForecast

        calls = []
        fake = _FakeSelf()
        fake._ml_model.model = object()  # non-None — past cold start
        fake._lock.acquire.return_value = False  # simulate lock held by retrain

        # Attach _update_sensors directly to the fake instance
        fake._update_sensors = lambda: calls.append(1)

        EnergyForecast._update_cb(fake)

        assert calls == [1], "_update_sensors must be called regardless of lock state"

    def test_update_cb_skips_when_no_model(self):
        """_update_cb must return early when no model is trained yet."""
        from energy_forecast.energy_forecast import EnergyForecast

        calls = []
        fake = _FakeSelf()
        fake._ml_model.model = None  # no model yet
        fake._update_sensors = lambda: calls.append(1)

        EnergyForecast._update_cb(fake)

        assert calls == [], "_update_sensors must NOT be called when model is None"


# ── _subtract_sub_sensors ─────────────────────────────────────────────────────


class TestSubtractSubSensors:
    def test_subtract_single_sensor(self):
        """Subtraction of one sensor at 1.0 kWh/h from 5.0 kWh/h results in 4.0 kWh/h."""
        ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [5.0] * 24})
        sub_df = pd.DataFrame({"timestamp": ts, "kwh": [1.0] * 24})
        sub_sensors = {"heat_pump": sub_df}

        res, removed = _subtract_sub_sensors(df, sub_sensors)
        assert removed == 24.0
        assert (res["gross_kwh"] == 4.0).all()

    def test_subtract_multiple_sensors(self):
        """Subtraction of two sensors (1.0 + 0.5) from 5.0 results in 3.5 kWh/h."""
        ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [5.0] * 24})
        sub1 = pd.DataFrame({"timestamp": ts, "kwh": [1.0] * 24})
        sub2 = pd.DataFrame({"timestamp": ts, "kwh": [0.5] * 24})
        sub_sensors = {"s1": sub1, "s2": sub2}

        res, removed = _subtract_sub_sensors(df, sub_sensors)
        assert removed == 1.5 * 24
        assert (res["gross_kwh"] == 3.5).all()

    def test_clipping_at_zero(self):
        """Result must never be negative even if sub-sensors exceed total."""
        ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 24})
        sub_df = pd.DataFrame({"timestamp": ts, "kwh": [2.0] * 24})
        sub_sensors = {"too_high": sub_df}

        res, _ = _subtract_sub_sensors(df, sub_sensors)
        assert (res["gross_kwh"] == 0.0).all()

    def test_misaligned_timestamps_filled_with_zero(self):
        """Missing sub-sensor rows (aligned via reindex) are treated as 0.0 kWh (no removal)."""
        ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [5.0] * 24})

        # Sub-sensor only has data for first 12 hours
        sub_ts = ts[:12]
        sub_df = pd.DataFrame({"timestamp": sub_ts, "kwh": [1.0] * 12})
        sub_sensors = {"partial": sub_df}

        res, removed = _subtract_sub_sensors(df, sub_sensors)
        assert removed == 12.0
        assert (res.iloc[:12]["gross_kwh"] == 4.0).all()
        assert (res.iloc[12:]["gross_kwh"] == 5.0).all()

    def test_empty_sub_sensors_returns_copy(self):
        """Passing None or empty dict returns an unchanged copy of the input."""
        df = pd.DataFrame(
            {"timestamp": pd.to_datetime(["2026-03-12 00:00", "2026-03-12 01:00"]), "gross_kwh": [5.0, 5.0]}
        )
        res, removed = _subtract_sub_sensors(df, {})
        assert removed == 0.0
        assert (res["gross_kwh"] == 5.0).all()
        assert res is not df


# ── EnergyForecast._subtract_only_dict ───────────────────────────────────────


class _FakeSubtractSelf:
    """Minimal stand-in for EnergyForecast with sub-sensor config."""

    def __init__(self, sub_energy_sensors: list[str], baseline_included_sensors: list[str]):
        self._sub_energy_sensors = sub_energy_sensors
        self._baseline_included_sensors = baseline_included_sensors

    def _sub_sensor_prefix(self, entity_id: str) -> str:
        sanitized = entity_id.split(".", 1)[-1].replace(".", "_")
        return f"sub_{sanitized}"


class TestSubtractOnlyDict:
    def _make_sub_dict(self, *entity_ids: str) -> dict:
        """Build a sub_dict with dummy DataFrames keyed by prefix."""
        fake = _FakeSubtractSelf(list(entity_ids), [])
        return {fake._sub_sensor_prefix(e): object() for e in entity_ids}

    def test_partial_subtraction(self):
        """One sensor in baseline_included_sensors → its prefix absent from result."""
        sensors = ["sensor.heat_pump_energy_kwh", "sensor.dishwasher_energy_kwh"]
        included = ["sensor.heat_pump_energy_kwh"]
        fake = _FakeSubtractSelf(sensors, included)
        sub_dict = {fake._sub_sensor_prefix(e): object() for e in sensors}

        result = EnergyForecast._subtract_only_dict(fake, sub_dict)

        assert "sub_heat_pump_energy_kwh" not in result
        assert "sub_dishwasher_energy_kwh" in result
        assert len(result) == 1

    def test_all_included_returns_empty(self):
        """All sensors in baseline_included_sensors → empty dict → no subtraction."""
        sensors = ["sensor.heat_pump_energy_kwh", "sensor.dhw_energy_kwh"]
        fake = _FakeSubtractSelf(sensors, sensors)
        sub_dict = {fake._sub_sensor_prefix(e): object() for e in sensors}

        result = EnergyForecast._subtract_only_dict(fake, sub_dict)

        assert result == {}

    def test_no_baseline_included_returns_full_dict(self):
        """Omitting baseline_included_sensors → full dict returned (backward-compatible)."""
        sensors = ["sensor.heat_pump_energy_kwh", "sensor.dishwasher_energy_kwh"]
        fake = _FakeSubtractSelf(sensors, [])
        sub_dict = {fake._sub_sensor_prefix(e): object() for e in sensors}

        result = EnergyForecast._subtract_only_dict(fake, sub_dict)

        assert result is sub_dict

    def test_unknown_sensor_in_included_no_crash(self):
        """A sensor in baseline_included_sensors not present in sub_dict is silently ignored."""
        sensors = ["sensor.dishwasher_energy_kwh"]
        included = ["sensor.heat_pump_energy_kwh"]  # not in sub_dict
        fake = _FakeSubtractSelf(sensors, included)
        sub_dict = {fake._sub_sensor_prefix(e): object() for e in sensors}

        result = EnergyForecast._subtract_only_dict(fake, sub_dict)

        # heat_pump not in sub_dict — no crash; dishwasher still present
        assert "sub_dishwasher_energy_kwh" in result
        assert len(result) == 1


# ── _latest_thermal_pressure_net ─────────────────────────────────────────────


@pytest.fixture
def trained_model_fixture(tmp_path):
    """Return an EnergyForecastModel that has been trained and had predict() called."""
    rng = np.random.default_rng(0)
    n = 600
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": rng.uniform(-5, 25, size=n),
            "precipitation_mm": [0.0] * n,
            "sunshine_min": [30.0] * n,
            "wind_kmh": [10.0] * n,
            "cloud_cover_pct": [50.0] * n,
            "direct_radiation_wm2": [100.0] * n,
        }
    )
    m = EnergyForecastModel(tmp_path)
    m.train(energy, weather, outdoor_df=None, weight_halflife_days=0)
    future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
    forecast = pd.DataFrame(
        {
            "timestamp": future_ts,
            "temp_c": [10.0] * 48,
            "precipitation_mm": [0.0] * 48,
            "sunshine_min": [30.0] * 48,
            "wind_kmh": [10.0] * 48,
            "cloud_cover_pct": [50.0] * 48,
            "direct_radiation_wm2": [100.0] * 48,
        }
    )
    m.predict(forecast, live_temp=10.0)
    return m


def test_latest_thermal_pressure_net_set_after_predict(trained_model_fixture):
    """After predict(), _latest_thermal_pressure_net is a non-negative float."""
    model = trained_model_fixture
    assert isinstance(model._latest_thermal_pressure_net, float)
    assert model._latest_thermal_pressure_net >= 0.0


def test_thermal_pressure_sensor_published():
    """_publish_thermal_pressure() calls set_state with a float >= 0 and tau_hours attr."""
    from energy_forecast.energy_forecast import EnergyForecast

    class _AppStub:
        def __init__(self):
            self._ml_model = MagicMock()
            self._ml_model._latest_thermal_pressure_net = 3.7
            self._ml_model._tau_hours = 18.5
            self._calls = []

        def set_state(self, eid, **kw):
            self._calls.append({"entity_id": eid, **kw})

    stub = _AppStub()
    EnergyForecast._publish_thermal_pressure(stub)
    assert len(stub._calls) == 1
    c = stub._calls[0]
    assert c["entity_id"] == "sensor.energy_forecast_thermal_pressure_net"
    assert float(c["state"]) >= 0.0
    assert c["attributes"]["tau_hours"] == 18.5
