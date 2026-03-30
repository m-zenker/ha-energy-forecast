"""
HA Energy Forecast — AppDaemon app.

EV charging handling:
  Hours where gross grid import exceeds ev_charging_threshold_kwh (default
  7 kWh/h) are classified as EV charging.  The fixed charger load (9 kWh/h)
  is subtracted, leaving the concurrent household baseline intact.  The model
  trains on this cleaned signal, so EV sessions don't distort forecasts.

  Two sensors are published from measured actuals:
    sensor.energy_forecast_ev_today      — EV kWh detected today
    sensor.energy_forecast_ev_yesterday  — EV kWh detected yesterday

  The threshold and charger power are configurable in apps.yaml:
    ev_charging_threshold_kwh: 7     # default
    ev_charger_kw: 9.0               # default
"""
from __future__ import annotations

import json
import math
import threading
from datetime import datetime, time
from pathlib import Path
from typing import Any

import hassapi as hass

from . import ha_data, weather
from .const import CACHE_PATH, EV_CHARGING_THRESHOLD_KWH
from .model import EnergyForecastModel

# ── Operational constants l ─────────────────────────────────────────────────────
RETRAIN_INTERVAL_S = 168 * 3600   # weekly
MIN_HISTORY_HOURS  = 48
BLOCK_SLOTS        = [f"{h:02d}_{h+3:02d}" for h in range(0, 24, 3)]
ATTRIBUTION        = "HA Energy Forecast — LightGBM + MeteoSwiss/Open-Meteo"


class EnergyForecast(hass.Hass):
    """AppDaemon app that forecasts household energy consumption."""

    def initialize(self) -> None:
        self.log("HA Energy Forecast initialising…")

        self._energy_sensor: str         = self.args["energy_sensor"]
        self._outdoor_sensor: str | None = self.args.get("outdoor_temp_sensor")
        self._plz: str                   = str(self.args.get("plz", ""))
        self._lat: float                 = float(self.args["latitude"])
        self._lon: float                 = float(self.args["longitude"])
        self._weight_halflife: float     = float(self.args.get("weight_halflife_days", 90))
        self._ev_threshold: float        = float(
            self.args.get("ev_charging_threshold_kwh", EV_CHARGING_THRESHOLD_KWH)
        )
        # Fixed charger power in kW — subtracted from charging hours so the
        # concurrent household baseline is preserved in training data.
        self._ev_charger_kw: float       = float(self.args.get("ev_charger_kw", 9.0))
        self._cache_path: Path           = Path(self.args.get("cache_path", str(CACHE_PATH)))
        self._holiday_canton: str | None = self.args.get("holiday_canton") or None
        self._adaptive_retrain_threshold: float = float(
            self.args.get("adaptive_retrain_threshold", 2.0)
        )
        # Optional sub-sensors: cumulative kWh meters (heat pump, dishwasher, etc.)
        # whose consumption is tracked as lag features to improve forecast accuracy.
        self._sub_energy_sensors: list[str] = list(self.args.get("sub_energy_sensors") or [])
        # Optional away / vacation mode entities:
        # away_mode_entity    — input_boolean whose "on" state marks a vacation period
        # away_return_entity  — input_datetime holding the expected return (for prediction only)
        self._away_mode_entity: str | None   = self.args.get("away_mode_entity") or None
        self._away_return_entity: str | None = self.args.get("away_return_entity") or None
        # Optional solar PV + battery target correction sensors.
        # When configured, gross_kwh (grid import) is corrected to true household
        # consumption before training:
        #   total_consumption = grid_import − grid_export + solar_production
        #                       − battery_charge + battery_discharge
        self._solar_sensor: str | None             = self.args.get("solar_production_sensor") or None
        self._grid_export_sensor: str | None       = self.args.get("grid_export_sensor") or None
        self._battery_charge_sensor: str | None    = self.args.get("battery_charge_sensor") or None
        self._battery_discharge_sensor: str | None = self.args.get("battery_discharge_sensor") or None
        # Anomaly detection: fire binary sensor when residual > sigma_threshold × std(residuals)
        self._anomaly_sigma_threshold: float = float(
            self.args.get("anomaly_sigma_threshold", 3.0)
        )
        # SHAP feature importance: top-N features exposed as sensor attribute (0 = off)
        self._shap_top_n: int = int(self.args.get("shap_top_n", 5))
        # Model versioning: number of archived model snapshots to retain
        self._model_archive_count: int = int(self.args.get("model_archive_count", 3))
        # Prediction history for adaptive retrain: {target_timestamp: predicted_kwh}.
        # Keep-first semantics so we track h≈24+ ahead predictions, not h=1.
        self._pred_history: dict[Any, float]    = {}
        self._actuals_history: dict[Any, float] = {}  # key: pd.Timestamp (floored 1h), rolling 30d actuals
        self._last_adaptive_retrain: datetime = datetime.min

        # MQTT Discovery (opt-in)
        self._mqtt_discovery: bool       = bool(self.args.get("mqtt_discovery", False))
        self._mqtt_namespace: str        = str(self.args.get("mqtt_namespace", "mqtt"))
        self._mqtt_discovery_prefix: str = str(self.args.get("mqtt_discovery_prefix", "homeassistant"))
        self._mqtt_intervals_discovered: bool = False

        self._validate_config()

        if self._mqtt_discovery:
            self._cleanup_legacy_states()   # remove ghost set_state entities
            self._mqtt_publish_all_discovery()
            self._mqtt_publish_availability("online")

        model_dir = Path(__file__).parent / "models"
        self._ml_model = EnergyForecastModel(model_dir, model_archive_count=self._model_archive_count)
        self._lock = threading.Lock()

        self.listen_event(self._retrain_cb, "RELOAD_ENERGY_MODEL")
        self.listen_event(self._rollback_model_cb, "energy_forecast_rollback_model")

        self._check_setup()
        self._publish_unavailable()
        self.run_in(self._retrain_cb, 10)
        self.run_every(self._retrain_cb, f"now+{RETRAIN_INTERVAL_S + 10}", RETRAIN_INTERVAL_S)
        self.run_in(self._update_cb, 130)
        self.run_hourly(self._update_cb, time(0, 1, 0))

        self.log(
            f"HA Energy Forecast ready. "
            f"EV threshold: {self._ev_threshold} kWh/h, "
            f"charger: {self._ev_charger_kw} kW"
        )

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Validate configuration values at startup; raises ValueError on bad input."""
        if not (-90 <= self._lat <= 90):
            raise ValueError(f"latitude must be between -90 and 90, got {self._lat}")
        if not (-180 <= self._lon <= 180):
            raise ValueError(f"longitude must be between -180 and 180, got {self._lon}")
        if self._weight_halflife <= 0:
            raise ValueError(
                f"weight_halflife_days must be positive, got {self._weight_halflife}"
            )
        if self._ev_threshold <= 0:
            raise ValueError(
                f"ev_charging_threshold_kwh must be positive, got {self._ev_threshold}"
            )
        if self._ev_charger_kw <= 0:
            raise ValueError(f"ev_charger_kw must be positive, got {self._ev_charger_kw}")
        if self._adaptive_retrain_threshold < 0:
            raise ValueError(
                f"adaptive_retrain_threshold must be ≥ 0, got {self._adaptive_retrain_threshold}"
            )
        if self._anomaly_sigma_threshold <= 0:
            raise ValueError(
                f"anomaly_sigma_threshold must be > 0, got {self._anomaly_sigma_threshold}"
            )
        if self._shap_top_n < 0:
            raise ValueError(f"shap_top_n must be >= 0, got {self._shap_top_n}")
        if self._model_archive_count < 0:
            raise ValueError(
                f"model_archive_count must be >= 0, got {self._model_archive_count}"
            )
        if self._ev_threshold >= self._ev_charger_kw:
            self.log(
                f"ev_charging_threshold_kwh ({self._ev_threshold}) is ≥ ev_charger_kw "
                f"({self._ev_charger_kw}). EV sessions may not be detected correctly — "
                "lower the threshold or raise ev_charger_kw.",
                level="WARNING",
            )
        if self._solar_sensor and not self._grid_export_sensor:
            self.log(
                "solar_production_sensor is set but grid_export_sensor is not — "
                "surplus solar exported to the grid will inflate the training target. "
                "Add grid_export_sensor for accurate consumption correction.",
                level="WARNING",
            )
        if self._mqtt_discovery:
            if not self._mqtt_namespace:
                raise ValueError("mqtt_namespace must be a non-empty string when mqtt_discovery is True")
            if not self._mqtt_discovery_prefix:
                raise ValueError("mqtt_discovery_prefix must be a non-empty string when mqtt_discovery is True")
        self.log(
            f"Config validated — lat={self._lat}, lon={self._lon}, plz={self._plz}, "
            f"weight_halflife={self._weight_halflife}d, "
            f"ev_threshold={self._ev_threshold} kWh/h, ev_charger={self._ev_charger_kw} kW, "
            f"sub_energy_sensors={len(self._sub_energy_sensors)}, "
            f"anomaly_sigma_threshold={self._anomaly_sigma_threshold}, "
            f"shap_top_n={self._shap_top_n}, "
            f"mqtt_discovery={self._mqtt_discovery}"
        )

    # ── Setup checker ─────────────────────────────────────────────────────────

    def _check_setup(self) -> None:
        """Publish sensor.energy_forecast_setup_status with import diagnostics.

        State is "ok" when all required packages are importable.  If a package
        is missing the state is "missing_packages" and the attributes list which
        ones failed, so users can diagnose install issues from HA dev tools
        without reading AppDaemon logs.
        """
        _REQUIRED = [
            ("pandas",    "pandas"),
            ("numpy",     "numpy"),
            ("sklearn",   "scikit-learn"),
            ("requests",  "requests"),
            ("holidays",  "holidays"),
        ]
        missing: list[str] = []
        for module, pip_name in _REQUIRED:
            try:
                __import__(module)
            except ImportError:
                missing.append(pip_name)

        if missing:
            state = "missing_packages"
            self.log(
                f"Setup check: missing packages — {missing}. "
                "Install them via AppDaemon add-on configuration.",
                level="WARNING",
            )
        else:
            state = "ok"

        try:
            if self._mqtt_discovery:
                self._mqtt_set_sensor_raw("energy_forecast_setup_status", state)
            else:
                self.set_state(
                    "sensor.energy_forecast_setup_status",
                    state=state,
                    attributes={
                        "friendly_name": "Energy Forecast Setup Status",
                        "unique_id": "energy_forecast_setup_status",
                        "missing_packages": missing,
                        "icon": "mdi:check-circle" if state == "ok" else "mdi:alert-circle",
                    },
                    replace=True,
                )
        except (AttributeError, TypeError, RuntimeError) as exc:
            self.log(f"Could not publish setup status sensor: {exc}", level="WARNING")

    # ── Sub-sensor helpers ────────────────────────────────────────────────────

    def _sub_sensor_prefix(self, entity_id: str) -> str:
        """Return the feature-column prefix for a sub-energy sensor entity_id."""
        sanitized = entity_id.split(".", 1)[-1].replace(".", "_")
        return f"sub_{sanitized}"

    def _sub_sensor_cache_path(self, entity_id: str) -> Path:
        """Return the CSV cache path for a sub-energy sensor."""
        sanitized = entity_id.split(".", 1)[-1].replace(".", "_")
        return self._cache_path.parent / f"sub_{sanitized}.csv"

    # ── MQTT Discovery ────────────────────────────────────────────────────────

    def _build_sensor_discovery_payload(
        self,
        unique_id: str,
        friendly_name: str,
        unit: str,
        icon: str,
        device_class: str | None,
        state_class: str | None,
        json_attributes_topic: str | None = None,
    ) -> dict:
        """Return the HA MQTT Discovery config dict for a single sensor."""
        payload: dict = {
            "name": friendly_name,
            "unique_id": unique_id,
            "state_topic": f"{self._mqtt_discovery_prefix}/energy_forecast/sensor/{unique_id}/state",
            "availability_topic": f"{self._mqtt_discovery_prefix}/energy_forecast/availability",
            "unit_of_measurement": unit,
            "icon": icon,
            "device": {
                "identifiers": ["ha_energy_forecast"],
                "name": "HA Energy Forecast",
                "model": "AppDaemon App",
                "sw_version": "0.6.0",
            },
        }
        if device_class is not None:
            payload["device_class"] = device_class
        if state_class is not None:
            payload["state_class"] = state_class
        if json_attributes_topic is not None:
            payload["json_attributes_topic"] = json_attributes_topic
        return payload

    def _build_binary_sensor_discovery_payload(
        self,
        unique_id: str,
        friendly_name: str,
        icon: str,
        device_class: str | None,
        json_attributes_topic: str | None = None,
    ) -> dict:
        """Return the HA MQTT Discovery config dict for a single binary sensor."""
        payload: dict = {
            "name": friendly_name,
            "unique_id": unique_id,
            "state_topic": f"{self._mqtt_discovery_prefix}/energy_forecast/binary_sensor/{unique_id}/state",
            "availability_topic": f"{self._mqtt_discovery_prefix}/energy_forecast/availability",
            "payload_on": "ON",
            "payload_off": "OFF",
            "icon": icon,
            "device": {
                "identifiers": ["ha_energy_forecast"],
                "name": "HA Energy Forecast",
                "model": "AppDaemon App",
                "sw_version": "0.6.0",
            },
        }
        if device_class is not None:
            payload["device_class"] = device_class
        if json_attributes_topic is not None:
            payload["json_attributes_topic"] = json_attributes_topic
        return payload

    def _mqtt_publish_binary_sensor_discovery(
        self,
        unique_id: str,
        friendly_name: str,
        icon: str,
        device_class: str | None,
        json_attributes_topic: str | None = None,
    ) -> None:
        """Publish a retained MQTT Discovery config payload for one binary sensor."""
        try:
            payload = self._build_binary_sensor_discovery_payload(
                unique_id, friendly_name, icon, device_class,
                json_attributes_topic=json_attributes_topic,
            )
            topic = f"{self._mqtt_discovery_prefix}/binary_sensor/{unique_id}/config"
            self.call_service(
                "mqtt/publish",
                topic=topic,
                payload=json.dumps(payload),
                retain=True,
                namespace=self._mqtt_namespace,
            )
        except Exception as exc:  # noqa: BLE001
            self.log(
                f"MQTT binary sensor discovery publish failed for {unique_id}: {exc}",
                level="WARNING",
            )

    def _mqtt_publish_discovery(
        self,
        unique_id: str,
        friendly_name: str,
        unit: str,
        icon: str,
        device_class: str | None,
        state_class: str | None,
        json_attributes_topic: str | None = None,
    ) -> None:
        """Publish a retained MQTT Discovery config payload for one sensor."""
        try:
            payload = self._build_sensor_discovery_payload(
                unique_id, friendly_name, unit, icon, device_class, state_class,
                json_attributes_topic=json_attributes_topic,
            )
            topic = f"{self._mqtt_discovery_prefix}/sensor/{unique_id}/config"
            self.call_service(
                "mqtt/publish",
                topic=topic,
                payload=json.dumps(payload),
                retain=True,
                namespace=self._mqtt_namespace,
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"MQTT discovery publish failed for {unique_id}: {exc}", level="WARNING")

    def _mqtt_set_sensor(self, unique_id: str, value: Any) -> None:
        """Publish a numeric sensor state (NaN/Inf → 0.0) to the MQTT state topic."""
        try:
            val = float(value)
            if math.isnan(val) or math.isinf(val):
                val = 0.0
            topic = f"{self._mqtt_discovery_prefix}/energy_forecast/sensor/{unique_id}/state"
            self.call_service(
                "mqtt/publish",
                topic=topic,
                payload=str(val),
                retain=True,
                namespace=self._mqtt_namespace,
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"MQTT state publish failed for {unique_id}: {exc}", level="WARNING")

    def _mqtt_set_sensor_raw(self, unique_id: str, value_str: str) -> None:
        """Publish a verbatim string payload to the MQTT state topic."""
        try:
            topic = f"{self._mqtt_discovery_prefix}/energy_forecast/sensor/{unique_id}/state"
            self.call_service(
                "mqtt/publish",
                topic=topic,
                payload=value_str,
                retain=True,
                namespace=self._mqtt_namespace,
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"MQTT raw state publish failed for {unique_id}: {exc}", level="WARNING")

    def _mqtt_publish_sensor_attributes(self, unique_id: str, attrs: dict,
                                        category: str = "sensor") -> None:
        """Publish a JSON attributes dict to the sensor's json_attributes_topic (retained)."""
        try:
            topic = f"{self._mqtt_discovery_prefix}/energy_forecast/{category}/{unique_id}/attributes"
            self.call_service(
                "mqtt/publish",
                topic=topic,
                payload=json.dumps(attrs),
                retain=True,
                namespace=self._mqtt_namespace,
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"MQTT attributes publish failed for {unique_id}: {exc}", level="WARNING")

    def _mqtt_publish_availability(self, payload: str) -> None:
        """Publish 'online' or 'offline' to the shared availability topic."""
        try:
            topic = f"{self._mqtt_discovery_prefix}/energy_forecast/availability"
            self.call_service(
                "mqtt/publish",
                topic=topic,
                payload=payload,
                retain=True,
                namespace=self._mqtt_namespace,
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"MQTT availability publish failed: {exc}", level="WARNING")

    def _mqtt_publish_all_discovery(self) -> None:
        """Publish discovery configs for all non-conditional sensors at init."""
        # Setup status
        self._mqtt_publish_discovery(
            "energy_forecast_setup_status",
            "Setup Status",
            "",
            "mdi:check-circle",
            None,
            None,
        )
        # Forecast totals
        for key, label in [("next_1h", "Next 1h"), ("next_3h", "Next 3h"), ("today", "Today"), ("tomorrow", "Tomorrow")]:
            attrs_topic = (
                f"{self._mqtt_discovery_prefix}/energy_forecast/sensor/energy_forecast_today/attributes"
                if key == "today" else None
            )
            self._mqtt_publish_discovery(
                f"energy_forecast_{key}",
                label,
                "kWh",
                "mdi:lightning-bolt",
                "energy",
                "measurement",
                json_attributes_topic=attrs_topic,
            )
        # 3h blocks — today and tomorrow, 8 slots each
        for day in ("today", "tomorrow"):
            for h in range(0, 24, 3):
                slot = f"{h:02d}_{h+3:02d}"
                h_start, h_end = f"{h:02d}", f"{h+3:02d}"
                self._mqtt_publish_discovery(
                    f"energy_forecast_{day}_{slot}",
                    f"{day.title()} {h_start}:00–{h_end}:00",
                    "kWh",
                    "mdi:calendar-clock",
                    "energy",
                    "measurement",
                )
        # EV actuals
        for day in ("today", "yesterday"):
            self._mqtt_publish_discovery(
                f"energy_forecast_ev_{day}",
                f"EV Charging Detected {day.title()}",
                "kWh",
                "mdi:car-electric",
                "energy",
                "measurement",
            )
        # Model MAE
        self._mqtt_publish_discovery(
            "energy_forecast_model_mae",
            "Model MAE",
            "kWh",
            "mdi:chart-bell-curve-cumulative",
            "energy",
            "measurement",
        )
        # Rolling MAE sensors (#41)
        for uid, name in [("energy_forecast_mae_7d", "Energy Forecast MAE 7d"),
                          ("energy_forecast_mae_30d", "Energy Forecast MAE 30d")]:
            self._mqtt_publish_discovery(uid, name, "kWh", "mdi:chart-bell-curve-cumulative", "energy", "measurement")
        # Anomaly detection sensor (#39)
        _anomaly_attrs_topic = (
            f"{self._mqtt_discovery_prefix}/energy_forecast"
            f"/binary_sensor/energy_forecast_unusual_consumption/attributes"
        )
        self._mqtt_publish_binary_sensor_discovery(
            "energy_forecast_unusual_consumption",
            "Unusual Consumption",
            "mdi:alert-circle-outline",
            "problem",
            json_attributes_topic=_anomaly_attrs_topic,
        )

    def terminate(self) -> None:
        """AppDaemon lifecycle hook — publish offline availability on shutdown."""
        if getattr(self, "_mqtt_discovery", False):
            self._mqtt_publish_availability("offline")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _retrain_cb(self, event_name=None, data=None, kwargs=None) -> None:
        # Accepts both timer callbacks (single positional arg) and
        # listen_event callbacks (event_name, data, kwargs).
        if not self._lock.acquire(blocking=False):
            self.log("Retrain skipped — another operation is running.", level="DEBUG")
            return
        try:
            self._retrain()
            if self._ml_model.model is not None:
                self._update_sensors()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Retraining failed: {exc}", level="ERROR")
        finally:
            self._lock.release()

    def _rollback_model_cb(self, event_name=None, data=None, kwargs=None) -> None:
        """Restore the previous model snapshot and refresh sensors."""
        if not self._lock.acquire(blocking=False):
            self.log("Rollback skipped — another operation is running.", level="DEBUG")
            return
        try:
            success = self._ml_model.rollback_model()
            if success and self._ml_model.model is not None:
                self._update_sensors()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Model rollback failed: {exc}", level="ERROR")
        finally:
            self._lock.release()

    def _update_cb(self, event_name=None, data=None, kwargs=None) -> None:
        if self._ml_model.model is None:
            return
        if not self._lock.acquire(blocking=False):
            self.log("Sensor update skipped — another operation is running.", level="DEBUG")
            return
        try:
            self._update_sensors()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Sensor update failed: {exc}", level="ERROR")
        finally:
            self._lock.release()

    # ── Core logic ────────────────────────────────────────────────────────────

    def _retrain(self) -> None:
        self.log("Starting model retraining…")
        energy_df = ha_data.fetch_energy_history(self, self._energy_sensor, cache_path=self._cache_path)

        if len(energy_df) < MIN_HISTORY_HOURS:
            self.log(f"Insufficient history ({len(energy_df)} h). Skipping.", level="WARNING")
            return

        energy_df = _strip_tz(energy_df)

        # ── Subtract EV charging from gross import ────────────────────────────
        baseline_df, ev_df = ha_data.split_ev_charging(
            energy_df, self._ev_threshold, charger_kw=self._ev_charger_kw
        )
        if len(ev_df):
            self.log(
                f"EV filter: {len(ev_df)} charging hours detected "
                f"({ev_df['gross_kwh'].sum():.1f} kWh gross). "
                f"Sessions on: {sorted(ev_df['timestamp'].dt.date.unique().tolist())}"
            )

        # ── Solar / grid-export / battery target correction ───────────────────
        # Corrects gross_kwh (grid import) to total household consumption.
        # Each sensor is optional; any combination is valid.
        _correction_specs = [
            (self._solar_sensor,             "solar_production.csv"),
            (self._grid_export_sensor,       "grid_export.csv"),
            (self._battery_charge_sensor,    "battery_charge.csv"),
            (self._battery_discharge_sensor, "battery_discharge.csv"),
        ]
        correction_dfs: dict[str, Any] = {}
        for sensor, cache_name in _correction_specs:
            if sensor:
                cache = self._cache_path.parent / cache_name
                try:
                    cdf = ha_data.fetch_sub_sensor_history(self, sensor, cache)
                    correction_dfs[cache_name] = _strip_tz(cdf)
                except (OSError, KeyError, ValueError) as exc:
                    self.log(
                        f"Target correction fetch failed ({cache_name}): {exc}",
                        level="WARNING",
                    )
                    correction_dfs[cache_name] = None
            else:
                correction_dfs[cache_name] = None

        if any(v is not None for v in correction_dfs.values()):
            baseline_df = _apply_target_correction(
                baseline_df,
                solar_df=correction_dfs["solar_production.csv"],
                grid_export_df=correction_dfs["grid_export.csv"],
                battery_charge_df=correction_dfs["battery_charge.csv"],
                battery_discharge_df=correction_dfs["battery_discharge.csv"],
            )
            self.log(
                "Target corrected: total_consumption = grid_import"
                + (" + solar" if correction_dfs["solar_production.csv"] is not None else "")
                + (" − grid_export" if correction_dfs["grid_export.csv"] is not None else "")
                + (" − battery_charge" if correction_dfs["battery_charge.csv"] is not None else "")
                + (" + battery_discharge" if correction_dfs["battery_discharge.csv"] is not None else "")
            )

        start_date = baseline_df["timestamp"].min().date()
        end_date   = baseline_df["timestamp"].max().date()

        try:
            weather_df = weather.fetch_historical_weather(self._lat, self._lon, start_date, end_date)
            weather_df = _strip_tz(weather_df)
        except (OSError, KeyError, ValueError) as exc:
            self.log(
                f"Historical weather fetch failed: {exc} — "
                "temp_c, heating_degree, cooling_degree and temp_rolling_3d will be "
                "imputed from training-set medians; forecast quality will be reduced.",
                level="WARNING",
            )
            weather_df = _empty_weather_df()

        sub_sensors_dict: dict = {}
        for entity_id in self._sub_energy_sensors:
            prefix = self._sub_sensor_prefix(entity_id)
            cache_path = self._sub_sensor_cache_path(entity_id)
            try:
                sub_df = ha_data.fetch_sub_sensor_history(self, entity_id, cache_path)
                sub_df = _strip_tz(sub_df)
                sub_sensors_dict[prefix] = sub_df
            except (OSError, KeyError, ValueError) as exc:
                self.log(f"Sub-sensor {entity_id} history fetch failed: {exc}", level="WARNING")

        away_df = ha_data.fetch_boolean_entity_history(
            self, self._away_mode_entity, days=30
        )
        if not away_df.empty:
            away_df = _strip_tz(away_df)

        self._ml_model.train(
            baseline_df,
            weather_df,
            outdoor_df=None,
            weight_halflife_days=self._weight_halflife,
            canton=self._holiday_canton,
            ev_df=ev_df,
            sub_sensors_dict=sub_sensors_dict or None,
            away_df=away_df if not away_df.empty else None,
        )
        self.log(f"Retrained. MAE: {self._ml_model.last_mae}")

    def _update_sensors(self) -> None:
        import pandas as pd

        # ── Fetch weather forecast ────────────────────────────────────────────
        forecast_df = weather.fetch_forecast(
            self._plz,
            self._lat,
            self._lon,
            self.args.get("srg_client_id"),
            self.args.get("srg_client_secret"),
        )
        forecast_df = _strip_tz(forecast_df)

        # ── Fetch recent actuals ──────────────────────────────────────────────
        # Uses the lightweight fetch (last 2 days only) to stay well within
        # AppDaemon's 10s callback limit. Full 30-day resync happens in _retrain().
        try:
            full_actuals = ha_data.fetch_recent_energy(self, self._energy_sensor, cache_path=self._cache_path)
            full_actuals = _strip_tz(full_actuals)
            # Subtract EV from actuals so lag_24h pointing at a charging hour
            # doesn't inflate tomorrow's baseline prediction.
            recent_actuals, _ = ha_data.split_ev_charging(
                full_actuals, self._ev_threshold, charger_kw=self._ev_charger_kw
            )
        except (OSError, ValueError, KeyError) as exc:
            self.log(f"Could not fetch recent actuals for lag features: {exc}", level="WARNING")
            recent_actuals = None
            full_actuals   = None

        sub_sensors_recent: dict = {}
        for entity_id in self._sub_energy_sensors:
            prefix = self._sub_sensor_prefix(entity_id)
            cache_path = self._sub_sensor_cache_path(entity_id)
            try:
                sub_df = ha_data.fetch_recent_sub_sensor(self, entity_id, cache_path)
                sub_df = _strip_tz(sub_df)
                sub_sensors_recent[prefix] = sub_df
            except (OSError, KeyError, ValueError) as exc:
                self.log(f"Sub-sensor {entity_id} recent fetch failed: {exc}", level="WARNING")

        live_temp  = self._read_live_temp()
        now_ts     = pd.Timestamp.now(tz="Europe/Zurich").tz_localize(None)
        away_series = self._build_away_prediction_series(now_ts)

        predictions = self._ml_model.predict(
            forecast_df, live_temp, recent_actuals,
            sub_sensors_recent=sub_sensors_recent or None,
            away_series=away_series,
        )
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"]).dt.tz_localize(None)

        intervals = self._ml_model.predict_intervals(
            forecast_df, live_temp, recent_actuals,
            sub_sensors_recent=sub_sensors_recent or None,
            away_series=away_series,
        )
        if intervals is not None:
            intervals["timestamp"] = pd.to_datetime(intervals["timestamp"]).dt.tz_localize(None)

        # ── Store predictions for adaptive retrain tracking ───────────────────
        # Keep-first: only store a prediction for each target hour the first time
        # we see it (~24h ahead), so MAE is measured on day-ahead forecasts.
        # Pruned to 30 days so mae_30d sensor has enough history (#41).
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
        self._pred_history = {
            ts: kwh for ts, kwh in self._pred_history.items()
            if pd.Timestamp(ts) >= cutoff
        }
        for _, row in predictions.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            if ts not in self._pred_history:
                self._pred_history[ts] = float(row["predicted_kwh"])

        # ── Populate rolling actuals history for mae_7d / mae_30d sensors (#41) ─
        # keep-last semantics: fresher actuals overwrite older ones for the same hour
        if recent_actuals is not None and not recent_actuals.empty:
            self._actuals_history.update(dict(zip(
                pd.to_datetime(recent_actuals["timestamp"]).dt.floor("1h"),
                recent_actuals["gross_kwh"].astype(float),
            )))
        actuals_cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
        self._actuals_history = {
            ts: kwh for ts, kwh in self._actuals_history.items()
            if pd.Timestamp(ts) >= actuals_cutoff
        }

        self._maybe_adaptive_retrain(recent_actuals)

        # ── Compute rolling MAE sensors (#41) ────────────────────────────────
        actuals_hist_df = None
        if self._actuals_history:
            actuals_hist_df = pd.DataFrame(
                [(ts, kwh) for ts, kwh in self._actuals_history.items()],
                columns=["timestamp", "gross_kwh"],
            )
        cutoff_7d = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
        pred_hist_7d = {
            ts: kwh for ts, kwh in self._pred_history.items()
            if pd.Timestamp(ts) >= cutoff_7d
        }
        mae_7d,  n_7d  = _compute_live_mae(pred_hist_7d, actuals_hist_df)
        mae_30d, n_30d = _compute_live_mae(self._pred_history, actuals_hist_df)

        # ── Anomaly detection (#39) ───────────────────────────────────────────
        is_anomaly, anomaly_residual, anomaly_std, anomaly_n = _compute_anomaly(
            self._pred_history,
            self._actuals_history,
            self._anomaly_sigma_threshold,
        )

        # ── SHAP feature importance (#42) ─────────────────────────────────────
        shap_data: dict = {}
        if self._shap_top_n > 0:
            try:
                shap_data = self._ml_model.shap_summary(
                    forecast_df, live_temp, recent_actuals,
                    sub_sensors_recent=sub_sensors_recent or None,
                    away_series=away_series,
                    n=self._shap_top_n,
                )
            except Exception as exc:  # noqa: BLE001
                self.log(f"SHAP summary failed: {exc}", level="WARNING")

        aggregated = self._aggregate(predictions, full_actuals, live_temp, intervals=intervals)
        aggregated["shap_top_features"] = shap_data
        aggregated["mae_7d"]          = mae_7d
        aggregated["mae_30d"]         = mae_30d
        aggregated["mae_7d_n_pairs"]  = n_7d
        aggregated["mae_30d_n_pairs"] = n_30d
        aggregated["is_anomaly"]        = is_anomaly
        aggregated["anomaly_residual"]  = anomaly_residual
        aggregated["anomaly_std"]       = anomaly_std
        aggregated["anomaly_n"]         = anomaly_n
        self._publish(aggregated)

    def _read_live_temp(self) -> float | None:
        if not self._outdoor_sensor:
            return None
        state = self.get_state(self._outdoor_sensor)
        if state in (None, "unavailable", "unknown"):
            return None
        try:
            return float(state)
        except (ValueError, TypeError):
            return None

    def _build_away_prediction_series(self, now_ts: Any) -> Any:
        """Return a 48-value pd.Series (indexed by naive prediction timestamps) of is_away flags.

        Logic:
        - Entity is "off" (or not configured)  → all zeros.
        - Entity is "on", no return entity or return datetime already past → all ones.
        - Entity is "on", return datetime in the future → 1 before return_dt, 0 at/after.
        """
        import pandas as pd

        future_hours = pd.date_range(
            start=pd.Timestamp(now_ts).floor("1h"), periods=48, freq="1h"
        )
        is_away = pd.Series(0, index=future_hours, dtype=int)

        if not self._away_mode_entity:
            return is_away

        state = self.get_state(self._away_mode_entity)
        if state not in ("on",):
            return is_away  # "off", "unavailable", "unknown", None → all zeros

        # Entity is "on" — determine when the away period ends
        return_dt: pd.Timestamp | None = None
        if self._away_return_entity:
            try:
                raw_return = self.get_state(self._away_return_entity)
                if raw_return not in (None, "unavailable", "unknown", ""):
                    return_dt = pd.Timestamp(raw_return)
                    # Strip tz if present, normalise to naive Europe/Zurich
                    if return_dt.tzinfo is not None:
                        return_dt = return_dt.tz_convert("Europe/Zurich").tz_localize(None)
            except (ValueError, TypeError) as exc:
                self.log(
                    f"Could not parse away_return_entity state as datetime: {exc}",
                    level="WARNING",
                )
                return_dt = None

        if return_dt is None or return_dt <= pd.Timestamp(now_ts):
            # No valid return time → away for the whole 48h window
            is_away[:] = 1
        else:
            is_away[future_hours < return_dt] = 1

        return is_away

    def _maybe_adaptive_retrain(self, actuals_df: Any) -> None:
        """Trigger an early retrain if live MAE exceeds threshold × CV MAE."""
        import pandas as pd

        cv_mae = self._ml_model.last_cv_mae
        if cv_mae is None:
            return
        # Use Europe/Zurich local time (tz-naive) consistent with pipeline timestamps.
        # datetime.now() would use system time, which is UTC in Docker/HA and
        # causes the cooldown to fire up to ±2h early/late and wrong during DST.
        _now = pd.Timestamp.now("Europe/Zurich").tz_localize(None)
        hours_since = (_now - self._last_adaptive_retrain).total_seconds() / 3600
        if hours_since < 24:
            return
        live_mae, n_pairs = _compute_live_mae(self._pred_history, actuals_df)
        if n_pairs < 24:
            return
        if live_mae > self._adaptive_retrain_threshold * cv_mae:
            self.log(
                f"Adaptive retrain triggered: live_MAE={live_mae:.4f} > "
                f"{self._adaptive_retrain_threshold}× cv_MAE={cv_mae:.4f} "
                f"(over {n_pairs} matched hours)",
                level="WARNING",
            )
            self._last_adaptive_retrain = pd.Timestamp.now("Europe/Zurich").tz_localize(None)
            self._retrain()

    # ── Sensor publishing ─────────────────────────────────────────────────────

    def _cleanup_legacy_states(self) -> None:
        """Remove AppDaemon-managed states for sensors now served via MQTT Discovery.

        Called at startup when mqtt_discovery=True to eliminate ghost entities left
        over from a previous run with mqtt_discovery=False.
        """
        legacy_ids: list[str] = [
            "sensor.energy_forecast_setup_status",
            "sensor.energy_forecast_next_1h",
            "sensor.energy_forecast_next_3h",
            "sensor.energy_forecast_today",
            "sensor.energy_forecast_tomorrow",
            "sensor.energy_forecast_ev_today",
            "sensor.energy_forecast_ev_yesterday",
            "sensor.energy_forecast_model_mae",
            # Rolling MAE sensors (#41)
            "sensor.energy_forecast_mae_7d",
            "sensor.energy_forecast_mae_30d",
            # Anomaly detection sensor (#39)
            "binary_sensor.energy_forecast_unusual_consumption",
            # Interval sensors
            "sensor.energy_forecast_next_3h_low",
            "sensor.energy_forecast_next_3h_high",
            "sensor.energy_forecast_today_low",
            "sensor.energy_forecast_today_high",
            "sensor.energy_forecast_tomorrow_low",
            "sensor.energy_forecast_tomorrow_high",
        ]
        # Block sensors
        for day in ("today", "tomorrow"):
            for slot in BLOCK_SLOTS:
                legacy_ids.append(f"sensor.energy_forecast_{day}_{slot}")

        for entity_id in legacy_ids:
            try:
                if self.entity_exists(entity_id):
                    self.remove_entity(entity_id)
            except Exception:  # noqa: BLE001
                pass

    def _publish_unavailable(self) -> None:
        if self._mqtt_discovery:
            return  # availability topic serves this purpose in MQTT mode
        for slot in ["next_1h", "next_3h", "today", "tomorrow"]:
            self.set_state(
                f"sensor.energy_forecast_{slot}",
                state="unavailable",
                attributes={
                    "unit_of_measurement": "kWh",
                    "friendly_name": f"Energy Forecast {slot.title().replace('_', ' ')}",
                    "unique_id": f"energy_forecast_{slot}",
                    "icon": "mdi:lightning-bolt",
                },
                replace=True,
            )
        for day in ("today", "yesterday"):
            self.set_state(
                f"sensor.energy_forecast_ev_{day}",
                state="unavailable",
                attributes={
                    "unit_of_measurement": "kWh",
                    "friendly_name": f"EV Charging Detected {day.title()}",
                    "unique_id": f"energy_forecast_ev_{day}",
                    "icon": "mdi:car-electric",
                },
                replace=True,
            )

    def _publish(self, data: dict) -> None:
        model       = self._ml_model
        trained_str = (
            model.last_trained.strftime("%Y-%m-%d %H:%M")
            if model.last_trained != datetime.min
            else "never"
        )
        base_attrs = {
            "unit_of_measurement": "kWh",
            "attribution": ATTRIBUTION,
            "model_engine": str(model.engine),
            "last_trained": trained_str,
        }

        def safe_set(entity_id: str, value: Any, friendly_name: str, extra_attrs: dict | None = None, icon: str | None = None) -> None:
            try:
                val = float(value)
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
            except (TypeError, ValueError):
                val = 0.0
            if self._mqtt_discovery:
                self._mqtt_set_sensor(entity_id.split(".", 1)[-1], val)
            else:
                attrs = {
                    **base_attrs,
                    "friendly_name": friendly_name,
                    "unique_id": entity_id.split(".", 1)[-1],
                }
                if icon:
                    attrs["icon"] = icon
                if extra_attrs:
                    attrs.update(extra_attrs)
                self.set_state(
                    entity_id,
                    state=str(round(val, 3)),
                    attributes=attrs,
                    replace=True,
                )

        # ── Forecast totals ───────────────────────────────────────────────────
        shap_features = data.get("shap_top_features") or {}
        for key, label in [("next_1h", "Next 1h"), ("next_3h", "Next 3h"), ("today", "Today"), ("tomorrow", "Tomorrow")]:
            extra = {"shap_top_features": shap_features} if key == "today" and shap_features else None
            safe_set(f"sensor.energy_forecast_{key}", data.get(key, 0), f"Energy Forecast {label}",
                     extra_attrs=extra, icon="mdi:lightning-bolt")
        # In MQTT mode, publish shap_top_features as json_attributes for energy_forecast_today
        if self._mqtt_discovery and shap_features:
            self._mqtt_publish_sensor_attributes(
                "energy_forecast_today",
                {"shap_top_features": shap_features},
            )

        # ── Prediction intervals (only published when quantile models trained) ─
        _any_intervals = any(
            data.get(f"{key}_low") is not None and data.get(f"{key}_high") is not None
            for key in ("next_3h", "today", "tomorrow")
        )
        if _any_intervals and self._mqtt_discovery and not self._mqtt_intervals_discovered:
            for key, label in [("next_3h", "Next 3h"), ("today", "Today"), ("tomorrow", "Tomorrow")]:
                self._mqtt_publish_discovery(
                    f"energy_forecast_{key}_low",
                    f"{label} Low (10th pct)",
                    "kWh",
                    "mdi:arrow-down-bold",
                    "energy",
                    "measurement",
                )
                self._mqtt_publish_discovery(
                    f"energy_forecast_{key}_high",
                    f"{label} High (90th pct)",
                    "kWh",
                    "mdi:arrow-up-bold",
                    "energy",
                    "measurement",
                )
            self._mqtt_intervals_discovered = True
        for key, label in [("next_3h", "Next 3h"), ("today", "Today"), ("tomorrow", "Tomorrow")]:
            low  = data.get(f"{key}_low")
            high = data.get(f"{key}_high")
            if low is not None and high is not None:
                safe_set(f"sensor.energy_forecast_{key}_low",  low,  f"Energy Forecast {label} Low (10th pct)",  icon="mdi:arrow-down-bold")
                safe_set(f"sensor.energy_forecast_{key}_high", high, f"Energy Forecast {label} High (90th pct)", icon="mdi:arrow-up-bold")

        # ── Forecast 3-hour blocks ────────────────────────────────────────────
        for day in ("today", "tomorrow"):
            blocks = data.get(f"blocks_{day}", {})
            for slot in BLOCK_SLOTS:
                h_start, h_end = slot.split("_")
                safe_set(
                    f"sensor.energy_forecast_{day}_{slot}",
                    blocks.get(slot, 0),
                    f"Energy Forecast {day.title()} {h_start}:00–{h_end}:00",
                    icon="mdi:calendar-clock",
                )

        # ── EV actuals sensors ────────────────────────────────────────────────
        ev_attrs = {
            "ev_threshold_kwh": self._ev_threshold,
            "ev_charger_kw":    self._ev_charger_kw,
        }
        safe_set(
            "sensor.energy_forecast_ev_today",
            data.get("ev_today", 0),
            "EV Charging Detected Today",
            extra_attrs=ev_attrs,
            icon="mdi:car-electric",
        )
        safe_set(
            "sensor.energy_forecast_ev_yesterday",
            data.get("ev_yesterday", 0),
            "EV Charging Detected Yesterday",
            extra_attrs=ev_attrs,
            icon="mdi:car-electric",
        )

        # ── Model MAE sensor ──────────────────────────────────────────────────
        mae_val = model.last_mae if model.last_mae is not None else 0
        if self._mqtt_discovery:
            self._mqtt_set_sensor("energy_forecast_model_mae", mae_val)
        else:
            self.set_state(
                "sensor.energy_forecast_model_mae",
                state=str(round(float(mae_val), 4)),
                attributes={
                    "unit_of_measurement": "kWh",
                    "friendly_name": "Energy Forecast Model MAE",
                    "unique_id": "energy_forecast_model_mae",
                    "icon": "mdi:chart-bell-curve-cumulative",
                    "attribution": ATTRIBUTION,
                    "cv_mae": str(model.last_cv_mae) if model.last_cv_mae is not None else "n/a",
                    "model_engine": str(model.engine),
                    "last_trained": trained_str,
                },
                replace=True,
            )

        # ── Rolling MAE sensors (#41) ─────────────────────────────────────────
        for key, label in [("mae_7d", "7-day MAE"), ("mae_30d", "30-day MAE")]:
            val = data.get(key, float("nan"))
            if self._mqtt_discovery:
                self._mqtt_set_sensor(f"energy_forecast_{key}", val)
            else:
                self.set_state(
                    f"sensor.energy_forecast_{key}",
                    state=str(round(float(val), 4)) if not math.isnan(float(val)) else "0.0",
                    attributes={
                        "unit_of_measurement": "kWh",
                        "friendly_name": f"Energy Forecast {label}",
                        "unique_id": f"energy_forecast_{key}",
                        "icon": "mdi:chart-bell-curve-cumulative",
                        "attribution": ATTRIBUTION,
                        "n_pairs": data.get(f"{key}_n_pairs", 0),
                        "model_engine": str(model.engine),
                    },
                    replace=True,
                )

        # ── Anomaly detection sensor (#39) ────────────────────────────────────
        is_anomaly = data.get("is_anomaly", False)
        anomaly_attrs = {
            "residual_kwh":      data.get("anomaly_residual", float("nan")),
            "residual_std_kwh":  data.get("anomaly_std",      float("nan")),
            "sigma_threshold":   self._anomaly_sigma_threshold,
            "n_pairs":           data.get("anomaly_n", 0),
        }
        if self._mqtt_discovery:
            _anomaly_uid = "energy_forecast_unusual_consumption"
            _state_topic = (
                f"{self._mqtt_discovery_prefix}/energy_forecast"
                f"/binary_sensor/{_anomaly_uid}/state"
            )
            try:
                self.call_service(
                    "mqtt/publish",
                    topic=_state_topic,
                    payload="ON" if is_anomaly else "OFF",
                    retain=True,
                    namespace=self._mqtt_namespace,
                )
            except Exception as exc:  # noqa: BLE001
                self.log(f"MQTT anomaly state publish failed: {exc}", level="WARNING")
            _attr_topic = (
                f"{self._mqtt_discovery_prefix}/energy_forecast"
                f"/binary_sensor/{_anomaly_uid}/attributes"
            )
            try:
                self.call_service(
                    "mqtt/publish",
                    topic=_attr_topic,
                    payload=json.dumps(anomaly_attrs),
                    retain=True,
                    namespace=self._mqtt_namespace,
                )
            except Exception as exc:  # noqa: BLE001
                self.log(f"MQTT anomaly attributes publish failed: {exc}", level="WARNING")
        else:
            self.set_state(
                "binary_sensor.energy_forecast_unusual_consumption",
                state="on" if is_anomaly else "off",
                attributes={
                    "friendly_name": "Unusual Consumption",
                    "unique_id": "energy_forecast_unusual_consumption",
                    "device_class": "problem",
                    "icon": "mdi:alert-circle-outline",
                    "attribution": ATTRIBUTION,
                    **anomaly_attrs,
                },
                replace=True,
            )

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, predictions: Any, full_actuals: Any, live_temp: float | None, intervals: Any = None) -> dict:
        import numpy as np
        import pandas as pd

        now_dt      = pd.Timestamp.now(tz="Europe/Zurich").replace(tzinfo=None)
        now_np      = np.datetime64(now_dt.floor("h"))
        today_np    = np.datetime64(now_dt.normalize())
        tomorrow_np = today_np + np.timedelta64(1, "D")
        yesterday_np = today_np - np.timedelta64(1, "D")

        p_times = pd.to_datetime(predictions["timestamp"]).values.astype("datetime64[ns]")
        p_vals  = predictions["predicted_kwh"].values.astype(float)

        def _sum(s, e):
            return round(float(np.sum(p_vals[(p_times >= s) & (p_times < e)])), 3)

        def _blocks(day_start):
            return {
                f"{h:02d}_{h+3:02d}": _sum(
                    day_start + np.timedelta64(h, "h"),
                    day_start + np.timedelta64(h + 3, "h"),
                )
                for h in range(0, 24, 3)
            }

        # Today: substitute actuals for elapsed hours so the sensor reflects
        # measured consumption for the past and forecast for the remainder.
        today_total, blocks_today = _blend_today_totals(
            p_times, p_vals, full_actuals, today_np, tomorrow_np, now_np
        )

        result = {
            "next_1h":         _sum(now_np, now_np + np.timedelta64(1, "h")),
            "next_3h":         _sum(now_np, now_np + np.timedelta64(3, "h")),
            "today":           today_total,
            "tomorrow":        _sum(tomorrow_np, tomorrow_np + np.timedelta64(1, "D")),
            "blocks_today":    blocks_today,
            "blocks_tomorrow": _blocks(tomorrow_np),
            "live_temp":       live_temp,
            "ev_today":        0.0,
            "ev_yesterday":    0.0,
        }

        # ── Prediction intervals ─────────────────────────────────────────────
        if intervals is not None:
            iv_times = pd.to_datetime(intervals["timestamp"]).values.astype("datetime64[ns]")
            iv_low   = intervals["low_kwh"].values.astype(float)
            iv_high  = intervals["high_kwh"].values.astype(float)

            def _isum(vals, s, e):
                return round(float(np.sum(vals[(iv_times >= s) & (iv_times < e)])), 3)

            today_low,  _ = _blend_today_totals(iv_times, iv_low,  full_actuals, today_np, tomorrow_np, now_np)
            today_high, _ = _blend_today_totals(iv_times, iv_high, full_actuals, today_np, tomorrow_np, now_np)

            result.update({
                "next_3h_low":   _isum(iv_low,  now_np,      now_np + np.timedelta64(3, "h")),
                "next_3h_high":  _isum(iv_high, now_np,      now_np + np.timedelta64(3, "h")),
                "today_low":     today_low,
                "today_high":    today_high,
                "tomorrow_low":  _isum(iv_low,  tomorrow_np, tomorrow_np + np.timedelta64(1, "D")),
                "tomorrow_high": _isum(iv_high, tomorrow_np, tomorrow_np + np.timedelta64(1, "D")),
            })

        # ── EV kWh from actuals: sum (gross - charger_kw) for charging hours ──
        # Subtracts the configured charger power to get the household co-load
        # contribution; the remainder is the estimated EV energy.  Clipped at 0
        # for hours where gross < charger_kw (partial sessions or EV not home).
        if full_actuals is not None and not full_actuals.empty:
            ev_mask  = full_actuals["gross_kwh"] > self._ev_threshold
            ev_rows  = full_actuals[ev_mask].copy()
            if not ev_rows.empty:
                ev_rows["ev_kwh"] = np.maximum(
                    0.0, ev_rows["gross_kwh"] - self._ev_charger_kw
                )
                ev_times = ev_rows["timestamp"].values.astype("datetime64[ns]")
                ev_vals  = ev_rows["ev_kwh"].values.astype(float)

                def _ev_sum(s, e):
                    return round(float(np.sum(ev_vals[(ev_times >= s) & (ev_times < e)])), 3)

                result["ev_today"]     = _ev_sum(today_np,     tomorrow_np)
                result["ev_yesterday"] = _ev_sum(yesterday_np, today_np)

        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_tz(df: Any) -> Any:
    """Convert timestamp column to naive Europe/Zurich local time."""
    import pandas as pd
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
        df = df.copy()
        df["timestamp"] = ts
    return df


def _empty_weather_df() -> Any:
    import pandas as pd
    return pd.DataFrame(
        columns=[
            "timestamp", "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh",
            "cloud_cover_pct", "direct_radiation_wm2",
        ]
    )


def _apply_target_correction(
    df: Any,
    solar_df: Any,
    grid_export_df: Any,
    battery_charge_df: Any,
    battery_discharge_df: Any,
) -> Any:
    """Correct gross_kwh (grid import) to total household consumption.

    total_consumption = grid_import
                        + solar_production
                        − grid_export
                        − battery_charge
                        + battery_discharge

    Each correction DataFrame has columns [timestamp, kwh] and may be None.
    Timestamps are matched exactly (hourly floor assumed). Unmatched hours
    receive zero correction — absence of data means no flow, not NaN.
    The result is clipped to 0 (consumption cannot be negative).
    Returns a copy of df with corrected gross_kwh.
    """
    import pandas as pd
    import numpy as np

    df = df.copy()
    ts_index = pd.DatetimeIndex(df["timestamp"])
    delta = pd.Series(np.zeros(len(df), dtype=float), index=ts_index)

    signed_dfs = [
        (solar_df,           +1.0),
        (grid_export_df,     -1.0),
        (battery_charge_df,  -1.0),
        (battery_discharge_df, +1.0),
    ]
    for correction_df, sign in signed_dfs:
        if correction_df is None or correction_df.empty:
            continue
        corr = (
            correction_df
            .set_index(pd.DatetimeIndex(correction_df["timestamp"]))["kwh"]
            .reindex(ts_index)
            .fillna(0.0)
        )
        delta = delta + sign * corr

    df["gross_kwh"] = (df["gross_kwh"] + delta.values).clip(lower=0.0)
    return df


def _blend_today_totals(
    p_times: Any,       # np.ndarray of datetime64[ns] — prediction timestamps
    p_vals: Any,        # np.ndarray of float — predicted kWh per hour
    full_actuals: Any,  # pd.DataFrame | None — cols: timestamp, gross_kwh
    today_np: Any,      # np.datetime64 — midnight today
    tomorrow_np: Any,   # np.datetime64 — midnight tomorrow
    now_np: Any,        # np.datetime64 — current hour (floored)
) -> tuple[float, dict]:
    """Compute today's blended total and 3h blocks.

    Elapsed hours (< now_np) use actuals from full_actuals where available;
    future hours (>= now_np) use model predictions.  Falls back to predictions
    only when full_actuals is None or empty.
    """
    import numpy as np

    fa_times = None
    fa_vals  = None
    if full_actuals is not None and not getattr(full_actuals, "empty", True):
        fa_times = full_actuals["timestamp"].values.astype("datetime64[ns]")
        fa_vals  = full_actuals["gross_kwh"].values.astype(float)

    def _pred_sum(s: Any, e: Any) -> float:
        return float(np.sum(p_vals[(p_times >= s) & (p_times < e)]))

    def _actual_sum(s: Any, e: Any) -> float:
        if fa_times is None:
            return 0.0
        return float(np.sum(fa_vals[(fa_times >= s) & (fa_times < e)]))

    def _blended(s: Any, e: Any) -> float:
        elapsed_end  = min(e, now_np)
        future_start = max(s, now_np)
        total = 0.0
        if elapsed_end > s:
            total += _actual_sum(s, elapsed_end)
        if future_start < e:
            total += _pred_sum(future_start, e)
        return round(total, 3)

    today_total = _blended(today_np, tomorrow_np)
    blocks = {
        f"{h:02d}_{h+3:02d}": _blended(
            today_np + np.timedelta64(h, "h"),
            today_np + np.timedelta64(h + 3, "h"),
        )
        for h in range(0, 24, 3)
    }
    return today_total, blocks


def _compute_live_mae(
    pred_history: dict,  # {timestamp-like: predicted_kwh}
    actuals_df: Any,     # pd.DataFrame | None — cols: timestamp, gross_kwh
) -> tuple[float, int]:
    """Compute MAE between stored predictions and actuals for matched timestamps.

    Returns (mae, n_pairs).  mae is float('nan') when n_pairs == 0.
    Only hours present in both pred_history and actuals_df are included.

    DST fall-back caveat: during the October clock-back, the naive 02:xx hour
    appears twice in the history.  Both occurrences map to the same floor("1h")
    key here, so the second occurrence overwrites the first in actuals_map and
    the wrong actual may be matched to the prediction.  This is an accepted
    edge case (one hour per year) and not worth the complexity of tz-aware storage.
    """
    import pandas as pd

    if actuals_df is None or getattr(actuals_df, "empty", True) or not pred_history:
        return float("nan"), 0

    actuals_map = {
        pd.Timestamp(ts).floor("1h"): float(kwh)
        for ts, kwh in zip(actuals_df["timestamp"], actuals_df["gross_kwh"])
    }

    errors = []
    for ts, pred in pred_history.items():
        key = pd.Timestamp(ts).floor("1h")
        if key in actuals_map:
            errors.append(abs(actuals_map[key] - pred))

    n = len(errors)
    if n == 0:
        return float("nan"), 0
    return round(sum(errors) / n, 4), n


def _compute_anomaly(
    pred_history: dict,      # {timestamp-like: predicted_kwh}  keep-first, raw ts keys
    actuals_history: dict,   # {pd.Timestamp: float}            keep-last, floored 1h keys
    sigma_threshold: float,
    min_pairs: int = 10,
) -> tuple[bool, float, float, int]:
    """Return (is_anomaly, latest_abs_residual, residual_std, n_pairs).

    Fires when |latest actual − latest prediction| > sigma_threshold × std(all residuals).
    Returns (False, nan, nan, 0) during cold start (< min_pairs matched pairs).

    DST fall-back caveat: the naive 02:xx hour appears twice in October; both map to
    the same floor("1h") key in pred_map, so the second overwrites the first — accepted
    edge case (one hour per year).
    """
    import numpy as np
    import pandas as pd

    if not pred_history or not actuals_history:
        return False, float("nan"), float("nan"), 0

    pred_map = {pd.Timestamp(ts).floor("1h"): kwh for ts, kwh in pred_history.items()}

    matched: dict = {}
    for ts, actual in actuals_history.items():
        key = pd.Timestamp(ts).floor("1h")
        if key in pred_map:
            matched[key] = actual - pred_map[key]

    n = len(matched)
    if n < min_pairs:
        return False, float("nan"), float("nan"), n

    latest_ts = max(matched.keys())
    latest_residual = matched[latest_ts]
    std = float(np.std(list(matched.values())))

    # Guard: near-perfect model → avoid spurious fires from floating-point noise
    if std < 0.01:
        return False, round(abs(latest_residual), 4), round(std, 4), n

    is_anomaly = abs(latest_residual) > sigma_threshold * std
    return is_anomaly, round(abs(latest_residual), 4), round(std, 4), n
