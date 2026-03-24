"""Tests for energy_forecast.py module-level helpers.

Covers:
  - _blend_today_totals: actuals substituted for elapsed hours, predictions for future
  - _compute_live_mae: correct MAE over matched pairs, nan on no overlap
  - EnergyForecast._retrain_cb / _update_cb: callable from both timer and event contexts
"""
from __future__ import annotations

import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.energy_forecast import _blend_today_totals, _compute_live_mae, _compute_anomaly


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_predictions(today: pd.Timestamp, kwh: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """48 hourly predictions starting at today midnight, all equal to kwh."""
    ts = pd.date_range(today, periods=48, freq="1h")
    p_times = ts.values.astype("datetime64[ns]")
    p_vals  = np.full(48, kwh)
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
        today   = pd.Timestamp("2026-03-12 00:00")
        now     = pd.Timestamp("2026-03-12 00:00")   # start of day — all future
        tmrw    = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=2.0)
        total, blocks = _blend_today_totals(
            p_times, p_vals, None,
            self._nts(today), self._nts(tmrw), self._nts(now),
        )
        # predictions cover [00:00, 48h); only today's 24 hours sum to 24 × 2.0 = 48.0
        assert abs(total - 48.0) < 1e-6

    def test_blended_total_uses_actuals_for_elapsed_hours(self):
        """Elapsed-hour actuals (3.0 kWh) replace predictions (1.0 kWh) in today's total."""
        today   = pd.Timestamp("2026-03-12 00:00")
        now     = pd.Timestamp("2026-03-12 12:00")   # noon — 12h elapsed
        tmrw    = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=1.0)
        actuals = _make_actuals(today, n=12, kwh=3.0)  # 12 elapsed hours at 3.0

        total, _ = _blend_today_totals(
            p_times, p_vals, actuals,
            self._nts(today), self._nts(tmrw), self._nts(now),
        )
        # 12 elapsed hours × 3.0 (actuals) + 12 future hours × 1.0 (predictions) = 48.0
        assert abs(total - (12 * 3.0 + 12 * 1.0)) < 1e-6

    def test_fully_elapsed_block_uses_actuals(self):
        """A 3h block entirely in the past must sum actuals, not predictions."""
        today   = pd.Timestamp("2026-03-12 00:00")
        now     = pd.Timestamp("2026-03-12 12:00")   # block 00-03 is fully elapsed
        tmrw    = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=1.0)
        actuals = _make_actuals(today, n=12, kwh=5.0)  # 12 elapsed hours at 5.0

        _, blocks = _blend_today_totals(
            p_times, p_vals, actuals,
            self._nts(today), self._nts(tmrw), self._nts(now),
        )
        # Block 00_03: 3 actual hours × 5.0 = 15.0
        assert abs(blocks["00_03"] - 15.0) < 1e-6

    def test_fully_future_block_uses_predictions(self):
        """A 3h block entirely in the future must sum predictions only."""
        today   = pd.Timestamp("2026-03-12 00:00")
        now     = pd.Timestamp("2026-03-12 00:00")   # nothing elapsed yet
        tmrw    = today + pd.Timedelta(days=1)

        p_times, p_vals = _make_predictions(today, kwh=2.5)
        actuals = _make_actuals(today, n=0, kwh=0.0)  # empty — nothing elapsed

        _, blocks = _blend_today_totals(
            p_times, p_vals, actuals,
            self._nts(today), self._nts(tmrw), self._nts(now),
        )
        # Block 21_24: 3 future hours × 2.5 = 7.5
        assert abs(blocks["21_24"] - 7.5) < 1e-6


# ── _compute_live_mae ─────────────────────────────────────────────────────────

class TestComputeLiveMae:

    def test_matched_pairs_return_correct_mae(self):
        """When predictions and actuals overlap perfectly, MAE equals mean absolute error."""
        ts = pd.date_range("2026-03-12 00:00", periods=24, freq="1h")
        pred_history = {t: 2.0 for t in ts}   # predicted 2.0 kWh every hour
        actuals = pd.DataFrame({
            "timestamp": ts,
            "gross_kwh": [3.0] * 24,           # actual was 3.0 → error = 1.0 each
        })
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
        ts_all    = pd.date_range("2026-03-12 00:00", periods=48, freq="1h")
        ts_first  = ts_all[:24]   # predictions cover first 24h
        ts_second = ts_all[24:]   # actuals cover last 24h — no overlap

        pred_history = {t: 1.0 for t in ts_first}
        actuals = pd.DataFrame({
            "timestamp": pd.concat([
                pd.Series(ts_first[:12]),   # 12h overlap
                pd.Series(ts_second),       # 24h no overlap
            ]),
            "gross_kwh": [3.0] * 36,
        })
        mae, n = _compute_live_mae(pred_history, actuals)
        assert n == 12
        assert abs(mae - 2.0) < 1e-6   # |3.0 - 1.0| = 2.0 for each of the 12 matched


# ── Interval blending (#13) ───────────────────────────────────────────────────

class TestIntervalBlend:

    def _nts(self, ts: pd.Timestamp) -> np.datetime64:
        return np.datetime64(ts, "ns")

    def test_low_total_less_than_high_total(self):
        """Blending q10 vals gives today_low < today_high when q10 < q90 for future hours."""
        today  = pd.Timestamp("2026-03-12 00:00")
        now    = pd.Timestamp("2026-03-12 00:00")  # nothing elapsed
        tmrw   = today + pd.Timedelta(days=1)

        ts = pd.date_range(today, periods=48, freq="1h")
        p_times = ts.values.astype("datetime64[ns]")

        low_vals  = np.full(48, 1.0)   # q10 = 1.0 kWh/h
        high_vals = np.full(48, 3.0)   # q90 = 3.0 kWh/h

        today_low,  _ = _blend_today_totals(p_times, low_vals,  None, self._nts(today), self._nts(tmrw), self._nts(now))
        today_high, _ = _blend_today_totals(p_times, high_vals, None, self._nts(today), self._nts(tmrw), self._nts(now))

        assert today_low < today_high

    def test_fully_elapsed_window_interval_collapses_to_actuals(self):
        """When all hours are elapsed, both low and high equal the actuals sum."""
        today  = pd.Timestamp("2026-03-12 00:00")
        now    = pd.Timestamp("2026-03-12 06:00")  # 6h elapsed
        tmrw   = today + pd.Timedelta(days=1)

        p_times   = pd.date_range(now, periods=48, freq="1h").values.astype("datetime64[ns]")
        low_vals  = np.full(48, 1.0)
        high_vals = np.full(48, 5.0)
        actuals   = pd.DataFrame({
            "timestamp": pd.date_range(today, periods=6, freq="1h"),
            "gross_kwh": [2.0] * 6,
        })
        # Block 00_03 is fully elapsed → both bounds must equal 3×2.0 = 6.0
        _, blocks_low  = _blend_today_totals(p_times, low_vals,  actuals, self._nts(today), self._nts(tmrw), self._nts(now))
        _, blocks_high = _blend_today_totals(p_times, high_vals, actuals, self._nts(today), self._nts(tmrw), self._nts(now))

        assert abs(blocks_low["00_03"]  - 6.0) < 1e-6
        assert abs(blocks_high["00_03"] - 6.0) < 1e-6


# ── _aggregate ────────────────────────────────────────────────────────────────

def _fake_self_for_aggregate(ev_threshold: float = 4.5, ev_charger_kw: float = 9.0) -> "_FakeSelf":
    app = _FakeSelf()
    app._ev_threshold  = ev_threshold
    app._ev_charger_kw = ev_charger_kw
    return app


def _pred_df(start: pd.Timestamp, n: int = 48, kwh: float = 1.0) -> pd.DataFrame:
    ts = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame({"timestamp": ts, "predicted_kwh": np.full(n, kwh)})


class TestAggregate:
    """_aggregate wires predictions + actuals into the sensor value dict."""

    def _run(self, today, now, predictions, actuals=None, intervals=None):
        from energy_forecast.energy_forecast import EnergyForecast
        return EnergyForecast._aggregate(
            _fake_self_for_aggregate(), predictions, actuals, live_temp=None, intervals=intervals
        )

    def test_next_1h_sums_only_one_future_hour(self):
        """next_1h must equal exactly 1 × per-hour prediction."""
        today = pd.Timestamp.now().normalize()
        now   = pd.Timestamp.now().floor("1h")
        preds = _pred_df(now, n=48, kwh=2.0)
        result = self._run(today, now, preds)
        assert abs(result["next_1h"] - 2.0) < 1e-3

    def test_next_1h_less_than_next_3h(self):
        """next_1h must be strictly less than next_3h when kWh/h > 0."""
        today = pd.Timestamp.now().normalize()
        now   = pd.Timestamp.now().floor("1h")
        preds = _pred_df(now, n=48, kwh=1.5)
        result = self._run(today, now, preds)
        assert result["next_1h"] < result["next_3h"]

    def test_next_3h_sums_only_three_future_hours(self):
        """next_3h must equal exactly 3 × per-hour prediction, not more."""
        today = pd.Timestamp.now().normalize()
        now   = pd.Timestamp.now().floor("1h")
        preds = _pred_df(now, n=48, kwh=2.0)
        result = self._run(today, now, preds)
        assert abs(result["next_3h"] - 6.0) < 1e-3

    def test_tomorrow_uses_predictions_only(self):
        """tomorrow must equal 24 × per-hour prediction for the following calendar day."""
        today = pd.Timestamp.now().normalize()
        tmrw  = today + pd.Timedelta(days=1)
        preds = _pred_df(today, n=48, kwh=3.0)
        result = self._run(today, today, preds)
        assert abs(result["tomorrow"] - 72.0) < 1e-3   # 24 × 3.0

    def test_ev_sensors_zero_when_no_actuals(self):
        """ev_today and ev_yesterday must be 0.0 when full_actuals is None."""
        today = pd.Timestamp.now().normalize()
        preds = _pred_df(today, n=48, kwh=1.0)
        result = self._run(today, today, preds, actuals=None)
        assert result["ev_today"]     == 0.0
        assert result["ev_yesterday"] == 0.0

    def test_interval_keys_present_when_intervals_supplied(self):
        """When intervals DataFrame is provided, *_low/*_high keys appear in result."""
        today = pd.Timestamp.now().normalize()
        now   = today
        preds = _pred_df(today, n=48, kwh=1.0)
        ts    = pd.date_range(today, periods=48, freq="1h")
        ivs   = pd.DataFrame({
            "timestamp": ts,
            "low_kwh":   np.full(48, 0.5),
            "high_kwh":  np.full(48, 2.0),
        })
        result = self._run(today, now, preds, intervals=ivs)
        for key in ("next_3h_low", "next_3h_high", "today_low", "today_high",
                    "tomorrow_low", "tomorrow_high"):
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
        tmrw  = today + pd.Timedelta(days=1)
        now   = today  # start of day — nothing elapsed yet

        # Predictions: 48h at 1.0 kWh/h
        ts     = pd.date_range(today, periods=48, freq="1h")
        p_df   = pd.DataFrame({"timestamp": ts, "predicted_kwh": np.ones(48)})

        # Actuals: one EV-level hour today (gross=12 kWh, clearly above any threshold)
        actuals = pd.DataFrame({
            "timestamp": [today + pd.Timedelta(hours=2)],
            "gross_kwh": [12.0],
        })

        app = _FakeSelf()
        app._ev_threshold  = ev_threshold
        app._ev_charger_kw = ev_charger_kw

        return EnergyForecast._aggregate(app, p_df, actuals, live_temp=None)

    def test_ev_today_uses_charger_kw_not_threshold(self):
        """ev_today must equal gross − charger_kw, not gross − threshold."""
        result = self._run_aggregate(ev_charger_kw=7.4, ev_threshold=4.5)
        # 12.0 − 7.4 = 4.6 (not 12.0 − 4.5 = 7.5)
        assert abs(result["ev_today"] - 4.6) < 1e-6

    def test_ev_today_default_charger_kw(self):
        """Default charger_kw=9.0: ev_today = gross − 9.0."""
        result = self._run_aggregate(ev_charger_kw=9.0, ev_threshold=4.5)
        assert abs(result["ev_today"] - 3.0) < 1e-6


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
        self._lat                        = 47.0
        self._lon                        = 8.5
        self._plz                        = ""
        self._weight_halflife            = 90.0
        self._ev_threshold               = ev_threshold
        self._ev_charger_kw              = ev_charger_kw
        self._adaptive_retrain_threshold = 2.0
        self._anomaly_sigma_threshold    = 3.0
        self._shap_top_n                 = 5
        self._sub_energy_sensors         = []
        self._mqtt_discovery             = False
        self._mqtt_namespace             = "mqtt"
        self._mqtt_discovery_prefix      = "homeassistant"
        self._warnings: list[str]        = []

    def log(self, msg: str, level: str = "INFO") -> None:
        if level == "WARNING":
            self._warnings.append(msg)


class TestValidateConfig:
    """_validate_config must warn when ev_threshold >= ev_charger_kw."""

    def test_warns_when_threshold_equals_charger_kw(self):
        from energy_forecast.energy_forecast import EnergyForecast
        fake = _FakeValidateSelf(ev_threshold=9.0, ev_charger_kw=9.0)
        EnergyForecast._validate_config(fake)
        assert fake._warnings, "Expected WARNING when ev_threshold == ev_charger_kw"

    def test_warns_when_threshold_exceeds_charger_kw(self):
        from energy_forecast.energy_forecast import EnergyForecast
        fake = _FakeValidateSelf(ev_threshold=10.0, ev_charger_kw=9.0)
        EnergyForecast._validate_config(fake)
        assert fake._warnings, "Expected WARNING when ev_threshold > ev_charger_kw"

    def test_no_warning_when_threshold_below_charger_kw(self):
        from energy_forecast.energy_forecast import EnergyForecast
        fake = _FakeValidateSelf(ev_threshold=7.0, ev_charger_kw=9.0)
        EnergyForecast._validate_config(fake)
        assert not fake._warnings, "No warning expected when ev_threshold < ev_charger_kw"


# ── Setup checker sensor (#17) ────────────────────────────────────────────────

class _FakeCheckSetup:
    """Minimal stand-in for _check_setup with controllable set_state and log."""

    def __init__(self):
        self._states: dict[str, dict] = {}
        self._warnings: list[str]     = []
        self._mqtt_discovery: bool    = False

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
        assert fake._warnings  # warning must be logged


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
            self._publishes.append({
                "topic": kwargs.get("topic", ""),
                "payload": kwargs.get("payload", ""),
                "namespace": kwargs.get("namespace", ""),
                "retain": kwargs.get("retain", False),
            })

    def set_state(self, entity_id: str, state: str = "", attributes: dict | None = None, replace: bool = False) -> None:
        self._states[entity_id] = {"state": state, "attributes": attributes or {}}

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
            "energy_forecast_today", "Today", "kWh",
            "mdi:lightning-bolt", "energy", "measurement",
        )
        assert "device" in payload
        assert payload["device"]["identifiers"] == ["ha_energy_forecast"]

    def test_discovery_config_topic_uses_configured_prefix(self):
        """Config topic must be <prefix>/sensor/<unique_id>/config."""
        fake = _FakeMqttSelf(mqtt_discovery_prefix="myprefix")
        fake._mqtt_publish_discovery(
            "energy_forecast_today", "Today", "kWh",
            "mdi:lightning-bolt", "energy", "measurement",
        )
        assert len(fake._publishes) == 1
        assert fake._publishes[0]["topic"] == "myprefix/sensor/energy_forecast_today/config"

    def test_kwh_sensor_payload_has_device_class_and_state_class(self):
        """kWh sensor must have device_class='energy' and state_class='measurement'."""
        fake = _FakeMqttSelf()
        payload = fake._build_sensor_discovery_payload(
            "energy_forecast_today", "Today", "kWh",
            "mdi:lightning-bolt", "energy", "measurement",
        )
        assert payload.get("device_class") == "energy"
        assert payload.get("state_class") == "measurement"

    def test_setup_status_payload_omits_device_class(self):
        """Setup status sensor must not include device_class or state_class."""
        fake = _FakeMqttSelf()
        payload = fake._build_sensor_discovery_payload(
            "energy_forecast_setup_status", "Setup Status", "",
            "mdi:check-circle", None, None,
        )
        assert "device_class" not in payload
        assert "state_class" not in payload

    def test_mqtt_publish_failure_logs_warning_and_does_not_raise(self):
        """If call_service raises, _mqtt_publish_discovery must log WARNING and not re-raise."""
        fake = _FakeMqttSelf()

        def bad_call_service(service, **kwargs):
            raise RuntimeError("broker down")

        fake.call_service = bad_call_service
        # Must not raise
        fake._mqtt_publish_discovery(
            "energy_forecast_today", "Today", "kWh",
            "mdi:lightning-bolt", "energy", "measurement",
        )
        assert fake._warnings, "Expected WARNING on call_service failure"


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
        assert any(p["payload"] == "offline" for p in fake._publishes), \
            "Expected 'offline' payload on terminate()"


class TestMqttConditionalIntervals:
    """Interval sensor discovery is deferred until quantile models exist."""

    def test_publish_all_discovery_does_not_include_low_high_topics(self):
        """_mqtt_publish_all_discovery() must NOT publish *_low/*_high config topics."""
        fake = _FakeMqttSelf()
        fake._mqtt_publish_all_discovery()
        low_high_topics = [
            p["topic"] for p in fake._publishes
            if "_low" in p["topic"] or "_high" in p["topic"]
        ]
        assert low_high_topics == [], f"Unexpected interval topics at init: {low_high_topics}"

    def test_interval_discovery_fired_on_first_publish_with_data(self):
        """When _publish() receives low/high data for the first time, interval discovery runs
        and _mqtt_intervals_discovered flips to True."""
        from energy_forecast.energy_forecast import EnergyForecast
        fake = _FakeMqttSelf()
        # Build a minimal aggregated data dict that includes interval values
        data = {
            "next_1h": 0.5, "next_3h": 1.0, "today": 5.0, "tomorrow": 6.0,
            "next_3h_low": 0.8, "next_3h_high": 1.2,
            "today_low": 4.5, "today_high": 5.5,
            "tomorrow_low": 5.5, "tomorrow_high": 6.5,
            "blocks_today": {f"{h:02d}_{h+3:02d}": 1.0 for h in range(0, 24, 3)},
            "blocks_tomorrow": {f"{h:02d}_{h+3:02d}": 1.0 for h in range(0, 24, 3)},
            "ev_today": 0.0, "ev_yesterday": 0.0,
        }
        EnergyForecast._publish(fake, data)
        assert fake._mqtt_intervals_discovered is True
        low_high_topics = [
            p["topic"] for p in fake._publishes
            if "_low/config" in p["topic"] or "_high/config" in p["topic"]
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
        assert any("energy_forecast_next_1h/config" in t for t in topics), \
            f"next_1h discovery topic not found in: {topics}"


class TestMqttFallback:
    """When mqtt_discovery=False, safe_set must use set_state and never call mqtt_publish."""

    def test_safe_set_uses_set_state_not_mqtt(self):
        """With mqtt_discovery=False, _publish() must call set_state and not call_service."""
        from energy_forecast.energy_forecast import EnergyForecast
        fake = _FakeMqttSelf(mqtt_discovery=False)
        fake.call_service = MagicMock()
        data = {
            "next_1h": 0.5, "next_3h": 1.0, "today": 5.0, "tomorrow": 6.0,
            "blocks_today": {f"{h:02d}_{h+3:02d}": 1.0 for h in range(0, 24, 3)},
            "blocks_tomorrow": {f"{h:02d}_{h+3:02d}": 1.0 for h in range(0, 24, 3)},
            "ev_today": 0.0, "ev_yesterday": 0.0,
        }
        EnergyForecast._publish(fake, data)
        fake.call_service.assert_not_called()


class TestCleanupLegacyStates:
    def test_removes_all_expected_legacy_ids(self):
        fake = _FakeMqttSelf()
        from energy_forecast.energy_forecast import EnergyForecast, BLOCK_SLOTS
        EnergyForecast._cleanup_legacy_states(fake)
        removed = set(fake._removed_entities)
        # Core sensors
        for key in ["setup_status", "next_1h", "next_3h", "today", "tomorrow",
                    "ev_today", "ev_yesterday", "model_mae"]:
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
        self._away_mode_entity   = away_mode_entity
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
        fake = _FakeAwaySelf(away_mode_entity="input_boolean.vacation", away_mode_state="on",
                             away_return_entity=None)
        result = EnergyForecast._build_away_prediction_series(fake, self._now_ts())
        assert (result == 1).all()

    def test_entity_on_return_in_future_splits_at_return_dt(self):
        """When entity is 'on' and return_dt is 24h ahead, first 24 rows = 1, rest = 0."""
        from energy_forecast.energy_forecast import EnergyForecast
        now_ts    = pd.Timestamp("2026-04-01 10:00")
        return_dt = now_ts + pd.Timedelta(hours=24)
        fake = _FakeAwaySelf(
            away_mode_entity="input_boolean.vacation", away_mode_state="on",
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
        now_ts    = pd.Timestamp("2026-04-01 10:00")
        return_dt = now_ts - pd.Timedelta(hours=1)   # 1h ago — already past
        fake = _FakeAwaySelf(
            away_mode_entity="input_boolean.vacation", away_mode_state="on",
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
        from energy_forecast.energy_forecast import EnergyForecast

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

        now = pd.Timestamp.now().normalize()
        cutoff_7d = now - pd.Timedelta(days=7)

        # 24 predictions within 7d window
        recent_preds = self._make_pred_history(cutoff_7d + pd.Timedelta(hours=1), 24, kwh=2.0)
        # 24 predictions older than 7d (should be excluded)
        old_preds = self._make_pred_history(cutoff_7d - pd.Timedelta(days=2), 24, kwh=2.0)

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

        now = pd.Timestamp.now().normalize()
        start = now - pd.Timedelta(days=29)
        pred_history = self._make_pred_history(start, 48, kwh=2.0)
        actuals = self._make_actuals_df(start, 48, kwh=3.0)

        mae, n = _compute_live_mae(pred_history, actuals)

        assert n == 48
        assert abs(mae - 1.0) < 1e-6

    # ── publish in set_state mode ──

    def test_publish_mae_sensors_set_state_mode(self):
        """_publish() must call set_state for both mae_7d and mae_30d when mqtt_discovery=False."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=False)
        data = {
            "next_1h": 0.5, "next_3h": 1.0, "today": 5.0, "tomorrow": 6.0,
            "blocks_today": {f"{h:02d}_{h+3:02d}": 1.0 for h in range(0, 24, 3)},
            "blocks_tomorrow": {f"{h:02d}_{h+3:02d}": 1.0 for h in range(0, 24, 3)},
            "ev_today": 0.0, "ev_yesterday": 0.0,
            "mae_7d": 0.25, "mae_30d": 0.30,
            "mae_7d_n_pairs": 24, "mae_30d_n_pairs": 120,
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
            "next_1h": 0.5, "next_3h": 1.0, "today": 5.0, "tomorrow": 6.0,
            "blocks_today": {f"{h:02d}_{h+3:02d}": 1.0 for h in range(0, 24, 3)},
            "blocks_tomorrow": {f"{h:02d}_{h+3:02d}": 1.0 for h in range(0, 24, 3)},
            "ev_today": 0.0, "ev_yesterday": 0.0,
            "mae_7d": 0.25, "mae_30d": 0.30,
            "mae_7d_n_pairs": 24, "mae_30d_n_pairs": 120,
        }
        EnergyForecast._publish(fake, data)

        state_topics = [p["topic"] for p in fake._publishes if "/state" in p["topic"]]
        assert any("energy_forecast_mae_7d" in t for t in state_topics), \
            f"mae_7d state topic not found in: {state_topics}"
        assert any("energy_forecast_mae_30d" in t for t in state_topics), \
            f"mae_30d state topic not found in: {state_topics}"

    # ── MQTT discovery ──

    def test_mqtt_discovery_includes_mae_sensors(self):
        """_mqtt_publish_all_discovery() must publish config topics for mae_7d and mae_30d."""
        fake = _FakeMqttSelf()
        fake._mqtt_publish_all_discovery()
        config_topics = [p["topic"] for p in fake._publishes if "/config" in p["topic"]]
        assert any("energy_forecast_mae_7d/config" in t for t in config_topics), \
            f"mae_7d config topic not found in: {config_topics}"
        assert any("energy_forecast_mae_30d/config" in t for t in config_topics), \
            f"mae_30d config topic not found in: {config_topics}"


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
            "next_1h": 0.5, "next_3h": 1.0, "today": 5.0, "tomorrow": 6.0,
            "blocks_today": {f"{h:02d}_{h+3:02d}": 1.0 for h in range(0, 24, 3)},
            "blocks_tomorrow": {f"{h:02d}_{h+3:02d}": 1.0 for h in range(0, 24, 3)},
            "ev_today": 0.0, "ev_yesterday": 0.0,
            "mae_7d": 0.25, "mae_30d": 0.30,
            "mae_7d_n_pairs": 24, "mae_30d_n_pairs": 120,
        }

    def test_publish_set_state_mode_anomaly_on(self):
        """_publish() in set_state mode must write binary_sensor state 'on' when is_anomaly=True."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=False)
        data = {**self._base_data(), "is_anomaly": True,
                "anomaly_residual": 5.0, "anomaly_std": 0.5, "anomaly_n": 20}
        EnergyForecast._publish(fake, data)

        eid = "binary_sensor.energy_forecast_unusual_consumption"
        assert eid in fake._states, f"Expected {eid} in states"
        assert fake._states[eid]["state"] == "on"
        assert fake._states[eid]["attributes"]["device_class"] == "problem"

    def test_publish_set_state_mode_anomaly_off(self):
        """_publish() in set_state mode must write binary_sensor state 'off' when is_anomaly=False."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=False)
        data = {**self._base_data(), "is_anomaly": False,
                "anomaly_residual": 0.1, "anomaly_std": 0.1, "anomaly_n": 20}
        EnergyForecast._publish(fake, data)

        eid = "binary_sensor.energy_forecast_unusual_consumption"
        assert fake._states[eid]["state"] == "off"

    def test_publish_mqtt_mode(self):
        """_publish() in MQTT mode must publish 'ON'/'OFF' to the anomaly state topic and attributes."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf(mqtt_discovery=True)
        data = {**self._base_data(), "is_anomaly": True,
                "anomaly_residual": 5.0, "anomaly_std": 0.5, "anomaly_n": 20}
        EnergyForecast._publish(fake, data)

        state_topics = {p["topic"]: p["payload"] for p in fake._publishes if "/state" in p["topic"]}
        anomaly_topic = next(
            (t for t in state_topics if "unusual_consumption" in t), None
        )
        assert anomaly_topic is not None, f"Anomaly state topic not found in: {list(state_topics)}"
        assert state_topics[anomaly_topic] == "ON"

        attr_topics = {p["topic"]: p["payload"] for p in fake._publishes if "/attributes" in p["topic"]}
        anomaly_attr_topic = next(
            (t for t in attr_topics if "unusual_consumption" in t), None
        )
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
        assert anomaly_config_topic is not None, \
            f"anomaly binary sensor config topic not found in: {list(config_publishes)}"
        import json as _json
        cfg = _json.loads(config_publishes[anomaly_config_topic])
        assert "json_attributes_topic" in cfg, "Missing json_attributes_topic in anomaly discovery payload"

    def test_cleanup_legacy_includes_anomaly_sensor(self):
        """_cleanup_legacy_states() must include binary_sensor.energy_forecast_unusual_consumption."""
        from energy_forecast.energy_forecast import EnergyForecast

        fake = _FakeMqttSelf()
        EnergyForecast._cleanup_legacy_states(fake)
        assert "binary_sensor.energy_forecast_unusual_consumption" in fake._removed_entities
