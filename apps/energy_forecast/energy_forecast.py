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
import logging
import math
import os
import threading
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hassapi as hass

if TYPE_CHECKING:
    import pandas as pd

from . import __version__, ha_data, weather
from .const import CACHE_PATH, EV_CHARGING_THRESHOLD_KWH, PRED_HISTORY_PATH, PRESENCE_STATE_HOME
from .const import strip_tz as _strip_tz_util
from .model import EnergyForecastModel

# Placeholder replaced in initialize() with AppDaemon's per-app logger.
# Using logging.getLogger() here would produce an "AppDaemon" category in HA logs
# because AppDaemon only routes its own app-hierarchy loggers (self.logger) correctly.
_LOGGER = logging.getLogger("energy_forecast")

# ── Operational constants l ─────────────────────────────────────────────────────
RETRAIN_INTERVAL_S = 168 * 3600  # weekly
MIN_HISTORY_HOURS = 48
BLOCK_SLOTS = [f"{h:02d}_{h + 3:02d}" for h in range(0, 24, 3)]
ATTRIBUTION = "HA Energy Forecast — LightGBM + MeteoSwiss/Open-Meteo"

# SHAP feature labels for narrative generation (#53)
_SHAP_FEATURE_LABELS: dict[str, str] = {
    # Calendar (raw)
    "hour": "hour of day",
    "day_of_week": "day of week",
    "month": "month",
    "season": "season",
    "hour_of_week": "hour of week",
    # Cyclical encodings
    "hour_sin": "time-of-day (sine)",
    "hour_cos": "time-of-day (cosine)",
    "month_sin": "month (sine)",
    "month_cos": "month (cosine)",
    "dow_sin": "day-of-week (sine)",
    "dow_cos": "day-of-week (cosine)",
    "how_sin": "hour-of-week (sine)",
    "how_cos": "hour-of-week (cosine)",
    "doy_sin": "day-of-year (sine)",
    "doy_cos": "day-of-year (cosine)",
    # Horizon
    "hours_ahead": "forecast horizon",
    # Weather
    "temp_c": "current outdoor temperature",
    "precipitation_mm": "rainfall",
    "sunshine_min": "sunshine duration",
    "wind_kmh": "wind speed",
    "humidity": "relative humidity",
    "cloud_cover_pct": "cloud cover",
    "direct_radiation_wm2": "solar irradiance",
    "heating_degree": "heating degree-hours",
    "cooling_degree": "cooling degree-hours",
    "temp_rolling_3d": "3-day average temperature",
    # Thermal modelling
    "temp_ewma_24h": "short-term thermal inertia",
    "temp_ewma_72h": "multi-day thermal inertia",
    "heating_deg_sum_24h": "accumulated heating demand (24h)",
    "heating_deg_sum_168h": "accumulated heating demand (7d)",
    "temp_delta_1h": "temperature rate of change",
    "temp_delta_24h": "24h temperature trend",
    "temp_lag_24h": "yesterday's temperature",
    "temp_lag_168h": "last week's temperature",
    # Autoregressive lags
    "lag_1h": "1h ago consumption",
    "lag_2h": "2h ago consumption",
    "lag_6h": "6h ago consumption",
    "lag_12h": "12h ago consumption",
    "lag_24h": "yesterday's same-hour consumption",
    "lag_48h": "2 days ago same-hour consumption",
    "lag_72h": "3 days ago same-hour consumption",
    "lag_168h": "last week's same-hour consumption",
    "lag_336h": "2 weeks ago same-hour consumption",
    # Rolling stats
    "rolling_mean_24h": "24h rolling average consumption",
    "rolling_mean_7d": "7-day rolling average consumption",
    "rolling_std_24h": "24h consumption variability",
    # Calendar extras
    "is_public_holiday": "public holiday",
    "days_to_next_holiday": "days until next holiday",
    "days_since_last_holiday": "days since last holiday",
    # EV
    "likely_ev_hour": "EV charging likely",
    # Occupancy / away
    "is_away": "vacation / away mode",
    "people_home": "number of people home",
    "regime_kwh": "typical daily pattern (regime)",
    # Heat pump / solar intent features
    "thermal_pressure": "heat debt (area-weighted)",
    "thermal_pressure_max": "coldest room heat deficit",
    "thermal_pressure_std": "room temperature imbalance",
    "thermal_pressure_cop": "heat debt electrical cost",
    "thermal_pressure_net": "solar-compensated heat debt",
    "infiltration_pressure": "wind-driven heat loss",
    "defrost_risk": "heat pump defrost risk",
    "weighted_solar_gain": "passive solar heat gain",
    "dhw_buffer_temp": "hot water buffer temperature",
    "dhw_pressure": "DHW heat demand",
    # Optional sensor features
    "outdoor_temp_live": "live outdoor temperature (sensor)",
    "temp_bias": "temperature sensor bias",
}


def _build_shap_narrative(shap_features: dict[str, float]) -> str:
    """Build a human-readable narrative from top SHAP features (#53)."""
    if not shap_features:
        return ""
    parts = []
    for feat, val in shap_features.items():
        label = _SHAP_FEATURE_LABELS.get(feat, feat)
        parts.append(f"{label} ({val:+.2f} kWh)")
    return "Mainly driven by: " + "; ".join(parts) + "."


class EnergyForecast(hass.Hass):
    """AppDaemon app that forecasts household energy consumption."""

    def initialize(self) -> None:
        # Wire AppDaemon's per-app logger into all sub-modules so every log entry
        # appears under "energy_forecast" in HA logs, not the generic "AppDaemon" category.
        global _LOGGER
        _LOGGER = self.logger
        from . import ha_data as _hd
        from . import model as _mdl
        from . import weather as _wth

        _hd._LOGGER = self.logger
        _mdl._LOGGER = self.logger
        _wth._LOGGER = self.logger

        _LOGGER.info("HA Energy Forecast initialising…")

        self._energy_sensor: str = self.args["energy_sensor"]
        self._outdoor_sensor: str | None = self.args.get("outdoor_temp_sensor")
        self._plz: str = str(self.args.get("plz", ""))
        self._lat: float = float(self.args["latitude"])
        self._lon: float = float(self.args["longitude"])
        self._weight_halflife: float = float(self.args.get("weight_halflife_days", 90))
        self._ev_threshold: float = float(self.args.get("ev_charging_threshold_kwh", EV_CHARGING_THRESHOLD_KWH))
        # Fixed charger power in kW — subtracted from charging hours so the
        # concurrent household baseline is preserved in training data.
        self._ev_charger_kw: float = float(self.args.get("ev_charger_kw", 9.0))
        self._baseline_mode: bool = bool(self.args.get("baseline_mode", False))
        self._cache_path: Path = Path(self.args.get("cache_path", str(CACHE_PATH)))
        self._timezone: str = str(self.args.get("timezone") or self.get_timezone() or "Europe/Zurich")
        self._holiday_canton: str | None = self.args.get("holiday_canton") or None
        self._holiday_country: str = str(self.args.get("holiday_country", "CH")).upper()
        self._adaptive_retrain_threshold: float = float(self.args.get("adaptive_retrain_threshold", 2.0))
        # Optional sub-sensors: cumulative kWh meters (heat pump, dishwasher, etc.)
        # whose consumption is tracked as lag features to improve forecast accuracy.
        # Each item may be a plain entity_id string or a dict with "entity_id" and
        # optionally "program_sensor" for per-program energy profile learning.
        _raw_sub = list(self.args.get("sub_energy_sensors") or [])
        self._sub_energy_sensors: list[str] = []
        self._program_sensors: dict[str, str] = {}  # energy_entity → program_entity
        for _item in _raw_sub:
            if isinstance(_item, str):
                self._sub_energy_sensors.append(_item)
            elif isinstance(_item, dict) and "entity_id" in _item:
                _eid = str(_item["entity_id"])
                self._sub_energy_sensors.append(_eid)
                if "program_sensor" in _item:
                    self._program_sensors[_eid] = str(_item["program_sensor"])
            else:
                _LOGGER.warning("sub_energy_sensors: unrecognized item %r — skipping", _item)
        self._baseline_included_sensors: list[str] = list(self.args.get("baseline_included_sensors") or [])
        # Optional away / vacation mode entities:
        # away_mode_entity    — input_boolean whose "on" state marks a vacation period
        # away_return_entity  — input_datetime holding the expected return (for prediction only)
        self._away_mode_entity: str | None = self.args.get("away_mode_entity") or None
        self._away_return_entity: str | None = self.args.get("away_return_entity") or None
        # Optional presence sensors for occupancy feature:
        # presence_sensors    — list of HA person entities (e.g. person.alice, person.bob)
        #                       to count how many are home at each hour
        self._presence_sensors: list[str] = list(self.args.get("presence_sensors") or [])
        # Optional solar PV + battery target correction sensors.
        # When configured, gross_kwh (grid import) is corrected to true household
        # consumption before training:
        #   total_consumption = grid_import − grid_export + solar_production
        #                       − battery_charge + battery_discharge
        self._solar_sensor: str | None = self.args.get("solar_production_sensor") or None
        self._grid_export_sensor: str | None = self.args.get("grid_export_sensor") or None
        self._battery_charge_sensor: str | None = self.args.get("battery_charge_sensor") or None
        self._battery_discharge_sensor: str | None = self.args.get("battery_discharge_sensor") or None
        # Anomaly detection: fire binary sensor when residual > sigma_threshold × std(residuals)
        self._anomaly_sigma_threshold: float = float(self.args.get("anomaly_sigma_threshold", 3.0))
        # SHAP feature importance: top-N features exposed as sensor attribute (0 = off)
        self._shap_top_n: int = int(self.args.get("shap_top_n", 5))
        # Model versioning: number of archived model snapshots to retain
        self._model_archive_count: int = int(self.args.get("model_archive_count", 3))

        # Daily Regime Clustering (Stage 4)
        self._enable_regimes: bool = bool(self.args.get("enable_regimes", False))
        self._regime_count: int = int(self.args.get("regime_count", 5))

        # Stage 2: Intent-Driven Thermal & DHW Modeling
        # climate_entities: list of HA climate entities (e.g. climate.living_room)
        # dhw_buffer_sensor: entity ID for DHW temperature sensor
        # heating_system_active_entity: binary sensor to verify "heating off" periods for Tau
        # climate_room_areas: optional {entity_id: m²} map for area-weighted thermal pressure
        self._climate_entities: list[str] = list(self.args.get("climate_entities") or [])
        self._dhw_buffer_sensor: str | None = self.args.get("dhw_buffer_sensor") or None
        self._heating_active_entity: str | None = self.args.get("heating_system_active_entity") or None
        self._climate_room_areas: dict[str, float] = self._parse_room_areas(self.args.get("climate_room_areas") or {})
        self._heating_temp_on: float = float(self.args.get("heating_temp_on", 14.0))
        self._heating_temp_off: float = float(self.args.get("heating_temp_off", 18.0))
        self._heating_setpoint_on: float = float(self.args.get("heating_setpoint_on", 20.0))
        self._heating_setpoint_off: float = float(self.args.get("heating_setpoint_off", 12.0))

        # Prediction history for adaptive retrain: {target_timestamp: predicted_kwh}.
        # Keep-first semantics so we track h≈24+ ahead predictions, not h=1.
        self._pred_history: dict[Any, float] = {}
        self._actuals_history: dict[Any, float] = {}  # key: pd.Timestamp (floored 1h), rolling 30d actuals
        self._last_adaptive_retrain: datetime = datetime.min

        # Stage 4: cached inputs for scenario/what-if API
        self._cached_forecast_df: Any = None
        self._cached_live_temp: Any = None
        self._cached_recent_actuals: Any = None
        self._cached_sub_sensors: Any = None
        self._cached_away_series: Any = None
        self._cached_people_home: Any = None
        self._cached_climate_recent: Any = None
        self._cached_dhw_recent: Any = None

        # MQTT Discovery (opt-in)
        self._mqtt_discovery: bool = bool(self.args.get("mqtt_discovery", False))
        self._mqtt_namespace: str = str(self.args.get("mqtt_namespace", "mqtt"))
        self._mqtt_discovery_prefix: str = str(self.args.get("mqtt_discovery_prefix", "homeassistant"))
        self._mqtt_intervals_discovered: bool = False

        self._validate_config()

        if self._mqtt_discovery:
            self._cleanup_legacy_states()  # remove ghost set_state entities
            self._mqtt_publish_all_discovery()
            self._mqtt_publish_availability("online")

        model_dir = Path(__file__).parent / "models"
        self._ml_model = EnergyForecastModel(
            model_dir, model_archive_count=self._model_archive_count, timezone=self._timezone
        )
        self._lock = threading.Lock()

        self.listen_event(self._retrain_cb, "RELOAD_ENERGY_MODEL")
        self.listen_event(self._rollback_model_cb, "energy_forecast_rollback_model")
        self.register_service("energy_forecast/get_scenario", self._get_scenario_cb)

        self._check_setup()
        self._publish_unavailable()
        self.run_in(self._retrain_cb, 10)
        self.run_every(self._retrain_cb, f"now+{RETRAIN_INTERVAL_S + 10}", RETRAIN_INTERVAL_S)
        self.run_in(self._update_cb, 130)
        self.run_hourly(self._update_cb, time(0, 1, 0))

        _LOGGER.info(
            "HA Energy Forecast ready. EV threshold: %s kWh/h, charger: %s kW",
            self._ev_threshold,
            self._ev_charger_kw,
        )

        self._load_pred_history()

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Validate configuration values at startup; raises ValueError on bad input."""
        if not (-90 <= self._lat <= 90):
            raise ValueError(f"latitude must be between -90 and 90, got {self._lat}")
        if not (-180 <= self._lon <= 180):
            raise ValueError(f"longitude must be between -180 and 180, got {self._lon}")
        if self._weight_halflife <= 0:
            raise ValueError(f"weight_halflife_days must be positive, got {self._weight_halflife}")
        if self._ev_threshold <= 0:
            raise ValueError(f"ev_charging_threshold_kwh must be positive, got {self._ev_threshold}")
        if self._ev_charger_kw <= 0:
            raise ValueError(f"ev_charger_kw must be positive, got {self._ev_charger_kw}")
        if self._adaptive_retrain_threshold < 0:
            raise ValueError(f"adaptive_retrain_threshold must be ≥ 0, got {self._adaptive_retrain_threshold}")
        if self._anomaly_sigma_threshold <= 0:
            raise ValueError(f"anomaly_sigma_threshold must be > 0, got {self._anomaly_sigma_threshold}")
        if self._shap_top_n < 0:
            raise ValueError(f"shap_top_n must be >= 0, got {self._shap_top_n}")
        if self._model_archive_count < 0:
            raise ValueError(f"model_archive_count must be >= 0, got {self._model_archive_count}")
        if self._ev_threshold >= self._ev_charger_kw:
            _LOGGER.warning(
                "ev_charging_threshold_kwh (%s) is ≥ ev_charger_kw (%s). "
                "EV sessions may not be detected correctly — "
                "lower the threshold or raise ev_charger_kw.",
                self._ev_threshold,
                self._ev_charger_kw,
            )
        if self._solar_sensor and not self._grid_export_sensor:
            _LOGGER.warning(
                "solar_production_sensor is set but grid_export_sensor is not — "
                "surplus solar exported to the grid will inflate the training target. "
                "Add grid_export_sensor for accurate consumption correction."
            )
        if self._mqtt_discovery:
            if not self._mqtt_namespace:
                raise ValueError("mqtt_namespace must be a non-empty string when mqtt_discovery is True")
            if not self._mqtt_discovery_prefix:
                raise ValueError("mqtt_discovery_prefix must be a non-empty string when mqtt_discovery is True")
        _LOGGER.info(
            "Config validated — lat=%s, lon=%s, plz=%s, timezone=%s, "
            "holiday_country=%s, weight_halflife=%sd, baseline_mode=%s, "
            "ev_threshold=%s kWh/h, ev_charger=%s kW, sub_energy_sensors=%s, "
            "anomaly_sigma_threshold=%s, shap_top_n=%s, mqtt_discovery=%s",
            self._lat,
            self._lon,
            self._plz,
            self._timezone,
            self._holiday_country,
            self._weight_halflife,
            self._baseline_mode,
            self._ev_threshold,
            self._ev_charger_kw,
            len(self._sub_energy_sensors),
            self._anomaly_sigma_threshold,
            self._shap_top_n,
            self._mqtt_discovery,
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
            ("pandas", "pandas"),
            ("numpy", "numpy"),
            ("sklearn", "scikit-learn"),
            ("requests", "requests"),
            ("holidays", "holidays"),
        ]
        missing: list[str] = []
        for module, pip_name in _REQUIRED:
            try:
                __import__(module)
            except ImportError:
                missing.append(pip_name)

        if missing:
            state = "missing_packages"
            _LOGGER.warning(
                "Setup check: missing packages — %s. Install them via AppDaemon add-on configuration.",
                missing,
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
            _LOGGER.warning("Could not publish setup status sensor: %s", exc)

    # ── Sub-sensor helpers ────────────────────────────────────────────────────

    def _sub_sensor_prefix(self, entity_id: str) -> str:
        """Return the feature-column prefix for a sub-energy sensor entity_id."""
        sanitized = entity_id.split(".", 1)[-1].replace(".", "_")
        return f"sub_{sanitized}"

    def _sub_sensor_cache_path(self, entity_id: str) -> Path:
        """Return the CSV cache path for a sub-energy sensor."""
        sanitized = entity_id.split(".", 1)[-1].replace(".", "_")
        return self._cache_path.parent / f"sub_{sanitized}.csv"

    def _subtract_only_dict(self, sub_dict: dict) -> dict:
        """Return sub_dict filtered to sensors not in baseline_included_sensors.

        Sensors listed in baseline_included_sensors stay in the training target
        (the main model predicts them via thermal/weather features). Only the
        remaining sensors are subtracted from the baseline.
        """
        if not self._baseline_included_sensors:
            return sub_dict
        included = {self._sub_sensor_prefix(e) for e in self._baseline_included_sensors}
        return {k: v for k, v in sub_dict.items() if k not in included}

    def _climate_cache_path(self, entity_id: str) -> Path:
        """Return the CSV cache path for a climate entity (setpoints/temp)."""
        sanitized = entity_id.split(".", 1)[-1].replace(".", "_")
        return self._cache_path.parent / f"climate_{sanitized}.csv"

    def _generic_sensor_cache_path(self, entity_id: str, prefix: str = "sensor") -> Path:
        """Return the CSV cache path for a generic absolute sensor."""
        sanitized = entity_id.split(".", 1)[-1].replace(".", "_")
        return self._cache_path.parent / f"{prefix}_{sanitized}.csv"

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
                "sw_version": __version__,
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
                "sw_version": __version__,
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
                unique_id,
                friendly_name,
                icon,
                device_class,
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
            _LOGGER.warning("MQTT binary sensor discovery publish failed for %s: %s", unique_id, exc)

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
                unique_id,
                friendly_name,
                unit,
                icon,
                device_class,
                state_class,
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
            _LOGGER.warning("MQTT discovery publish failed for %s: %s", unique_id, exc)

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
            _LOGGER.warning("MQTT state publish failed for %s: %s", unique_id, exc)

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
            _LOGGER.warning("MQTT raw state publish failed for %s: %s", unique_id, exc)

    def _mqtt_publish_sensor_attributes(self, unique_id: str, attrs: dict, category: str = "sensor") -> None:
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
            _LOGGER.warning("MQTT attributes publish failed for %s: %s", unique_id, exc)

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
            _LOGGER.warning("MQTT availability publish failed: %s", exc)

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
        for key, label in [
            ("next_1h", "Next 1h"),
            ("next_3h", "Next 3h"),
            ("today", "Today"),
            ("tomorrow", "Tomorrow"),
        ]:
            attrs_topic = (
                f"{self._mqtt_discovery_prefix}/energy_forecast/sensor/energy_forecast_today/attributes"
                if key == "today"
                else None
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
                slot = f"{h:02d}_{h + 3:02d}"
                h_start, h_end = f"{h:02d}", f"{h + 3:02d}"
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
        for uid, name in [
            ("energy_forecast_mae_7d", "Energy Forecast MAE 7d"),
            ("energy_forecast_mae_30d", "Energy Forecast MAE 30d"),
        ]:
            self._mqtt_publish_discovery(uid, name, "kWh", "mdi:chart-bell-curve-cumulative", "energy", "measurement")
        # Relative MAE sensors (#54)
        for uid, name in [
            ("energy_forecast_relative_mae_7d", "Energy Forecast Relative MAE 7d"),
            ("energy_forecast_relative_mae_30d", "Energy Forecast Relative MAE 30d"),
        ]:
            self._mqtt_publish_discovery(uid, name, "%", "mdi:percent", None, "measurement")
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
        # Thermal pressure (for HP optimiser)
        _tp_attrs_topic = (
            f"{self._mqtt_discovery_prefix}/energy_forecast/sensor/energy_forecast_thermal_pressure_net/attributes"
        )
        self._mqtt_publish_discovery(
            "energy_forecast_thermal_pressure_net",
            "Thermal Pressure (net)",
            "°C·m²",
            "mdi:thermometer",
            None,
            "measurement",
            json_attributes_topic=_tp_attrs_topic,
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
            _LOGGER.debug("Retrain skipped — another operation is running.")
            return
        try:
            self._retrain()
            if self._ml_model.model is not None:
                self._update_sensors()
        except Exception as exc:  # noqa: BLE001
            import traceback

            _LOGGER.error("Retraining failed: %s\n%s", exc, traceback.format_exc())
        finally:
            self._lock.release()

    def _rollback_model_cb(self, event_name=None, data=None, kwargs=None) -> None:
        """Restore the previous model snapshot and refresh sensors."""
        if not self._lock.acquire(blocking=False):
            _LOGGER.debug("Rollback skipped — another operation is running.")
            return
        try:
            success = self._ml_model.rollback_model()
            if success and self._ml_model.model is not None:
                self._update_sensors()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("Model rollback failed: %s", exc)
        finally:
            self._lock.release()

    def _get_scenario_cb(self, namespace: str, domain: str, service: str, kwargs: dict) -> None:
        """AppDaemon service callback: energy_forecast/get_scenario.

        Accepts kwargs:
          schedule (dict[str, str]): {prefix: "HH:MM" | "off" | None}
          publish  (bool):           if True, publish scenario sensors to HA
        Fires event "energy_forecast_scenario_result" with the forecast payload.
        """
        try:
            if self._cached_forecast_df is None:
                _LOGGER.warning("get_scenario called before first forecast cycle")
                return

            schedule = kwargs.get("schedule", {})
            if not isinstance(schedule, dict):
                _LOGGER.warning("get_scenario: 'schedule' must be a dict, got %s", type(schedule).__name__)
                return

            # Validate keys and time values; silently drop invalid entries
            valid_prefixes = set(self._ml_model._appliance_signatures.keys()) | {
                self._sub_sensor_prefix(e) for e in self._sub_energy_sensors
            }
            cleaned: dict = {}
            for key, val in schedule.items():
                if valid_prefixes and key not in valid_prefixes:
                    _LOGGER.warning("get_scenario: unknown prefix '%s' in schedule — ignored", key)
                    continue
                if val is not None and val != "off":
                    try:
                        datetime.strptime(str(val), "%H:%M")
                    except ValueError:
                        _LOGGER.warning("get_scenario: invalid time '%s' for prefix '%s' — ignored", val, key)
                        continue
                cleaned[key] = val
            schedule = cleaned

            result_df = self._ml_model.predict_scenario(
                self._cached_forecast_df,
                self._cached_live_temp,
                schedule,
                recent_actuals=self._cached_recent_actuals,
                sub_sensors_recent=self._cached_sub_sensors,
                away_series=self._cached_away_series,
                people_home_series=self._cached_people_home,
                climate_recent=self._cached_climate_recent,
                dhw_recent=self._cached_dhw_recent,
            )

            if kwargs.get("publish", False):
                self._publish_scenario_forecast(result_df)

            self.fire_event(
                "energy_forecast_scenario_result",
                forecast=result_df.to_dict("records"),
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("get_scenario failed: %s", exc)

    def _publish_scenario_forecast(self, result_df: pd.DataFrame) -> None:
        """Publish scenario forecast sensors to HA (called when publish=True).

        Writes the following HA sensor entities (all in kWh):
          - sensor.energy_forecast_scenario_today       — total today
          - sensor.energy_forecast_scenario_tomorrow    — total tomorrow
          - sensor.energy_forecast_scenario_delta_today — appliance-induced delta vs baseline (today)
          - sensor.energy_forecast_scenario_today_HH_HH — 8 three-hour block sensors for today
            (00_03, 03_06, 06_09, 09_12, 12_15, 15_18, 18_21, 21_24)

        All sensors carry ``unit_of_measurement: kWh`` and ``attribution`` attributes.
        Values are rounded to 3 decimal places; NaN/Inf are coerced to 0.
        """
        import math as _math

        import numpy as np
        import pandas as pd

        now_dt = pd.Timestamp.now(tz=self._timezone).tz_localize(None)
        today_np = np.datetime64(now_dt.normalize())
        tomorrow_np = today_np + np.timedelta64(1, "D")
        after_np = tomorrow_np + np.timedelta64(1, "D")

        p_times = pd.to_datetime(result_df["timestamp"]).values.astype("datetime64[ns]")
        p_vals = result_df["predicted_kwh"].values.astype(float)
        d_vals = result_df["delta_kwh"].values.astype(float)

        def _sum(arr, s, e):
            return round(float(np.sum(arr[(p_times >= s) & (p_times < e)])), 3)

        def _safe_set_scenario(entity_id: str, value: Any, friendly_name: str) -> None:
            try:
                val = float(value)
                if _math.isnan(val) or _math.isinf(val):
                    val = 0.0
            except (TypeError, ValueError):
                val = 0.0
            self.set_state(
                entity_id,
                state=str(round(val, 3)),
                attributes={
                    "unit_of_measurement": "kWh",
                    "friendly_name": friendly_name,
                    "unique_id": entity_id.split(".", 1)[-1],
                    "attribution": ATTRIBUTION,
                },
                replace=True,
            )

        scenario_today = _sum(p_vals, today_np, tomorrow_np)
        scenario_tomorrow = _sum(p_vals, tomorrow_np, after_np)
        delta_today = _sum(d_vals, today_np, tomorrow_np)

        _safe_set_scenario("sensor.energy_forecast_scenario_today", scenario_today, "Energy Forecast Scenario Today")
        _safe_set_scenario(
            "sensor.energy_forecast_scenario_tomorrow", scenario_tomorrow, "Energy Forecast Scenario Tomorrow"
        )
        _safe_set_scenario(
            "sensor.energy_forecast_scenario_delta_today", delta_today, "Energy Forecast Scenario Delta Today"
        )

        for h in range(0, 24, 3):
            slot = f"{h:02d}_{h + 3:02d}"
            slot_start = today_np + np.timedelta64(h, "h")
            slot_end = today_np + np.timedelta64(h + 3, "h")
            _safe_set_scenario(
                f"sensor.energy_forecast_scenario_today_{slot}",
                _sum(p_vals, slot_start, slot_end),
                f"Energy Forecast Scenario Today {h:02d}:{h + 3:02d}",
            )

    def _update_cb(self, event_name=None, data=None, kwargs=None) -> None:
        if self._ml_model.model is None:
            return
        # No lock: prediction reads self._ml_model.model via atomic Python attribute
        # access (GIL). _retrain_cb replaces it atomically at the end of train().
        # Concurrent predict + retrain is safe — worst case uses the last-good model.
        try:
            self._update_sensors()
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("Sensor update failed: %s", exc)

    # ── Core logic ────────────────────────────────────────────────────────────

    def _retrain(self) -> None:
        import pandas as pd

        _LOGGER.info("Starting model retraining…")
        energy_df = ha_data.fetch_energy_history(
            self, self._energy_sensor, cache_path=self._cache_path, timezone=self._timezone
        )

        if len(energy_df) < MIN_HISTORY_HOURS:
            _LOGGER.warning("Insufficient history (%d h). Skipping.", len(energy_df))
            return

        energy_df = _strip_tz(energy_df, self._timezone)

        # ── Subtract EV charging from gross import ────────────────────────────
        baseline_df, ev_df = ha_data.split_ev_charging(energy_df, self._ev_threshold, charger_kw=self._ev_charger_kw)
        if len(ev_df):
            _LOGGER.info(
                "EV filter: %d charging hours detected (%.1f kWh gross). Sessions on: %s",
                len(ev_df),
                ev_df["gross_kwh"].sum(),
                sorted(ev_df["timestamp"].dt.date.unique().tolist()),
            )
            # Drop ±1h adjacent hours (ramp-up/down) — split_ev_charging subtracts the
            # charger load from exact EV hours but can't cleanly correct adjacent hours
            # whose elevation comes from a tapering charger.  Dropping 1–2 rows per
            # session (≈15% of days) has negligible training impact; NaN lags are
            # filled by feature medians in meta.pkl.
            _ev_adj_ts: set = {
                ts + pd.Timedelta(hours=d) for ts in pd.to_datetime(ev_df["timestamp"]).dt.floor("1h") for d in (-1, 1)
            }
            baseline_df = baseline_df[~pd.to_datetime(baseline_df["timestamp"]).dt.floor("1h").isin(_ev_adj_ts)]

        # ── Solar / grid-export / battery target correction ───────────────────
        # Corrects gross_kwh (grid import) to total household consumption.
        # Each sensor is optional; any combination is valid.
        _correction_specs = [
            (self._solar_sensor, "solar_production.csv"),
            (self._grid_export_sensor, "grid_export.csv"),
            (self._battery_charge_sensor, "battery_charge.csv"),
            (self._battery_discharge_sensor, "battery_discharge.csv"),
        ]
        correction_dfs: dict[str, Any] = {}
        for sensor, cache_name in _correction_specs:
            if sensor:
                cache = self._cache_path.parent / cache_name
                try:
                    cdf = ha_data.fetch_sub_sensor_history(self, sensor, cache, timezone=self._timezone)
                    correction_dfs[cache_name] = _strip_tz(cdf, self._timezone)
                except (OSError, KeyError, ValueError) as exc:
                    _LOGGER.warning("Target correction fetch failed (%s): %s", cache_name, exc)
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
            _LOGGER.info(
                "Target corrected: total_consumption = grid_import%s%s%s%s",
                " + solar" if correction_dfs["solar_production.csv"] is not None else "",
                " − grid_export" if correction_dfs["grid_export.csv"] is not None else "",
                " − battery_charge" if correction_dfs["battery_charge.csv"] is not None else "",
                " + battery_discharge" if correction_dfs["battery_discharge.csv"] is not None else "",
            )

        sub_sensors_dict: dict = {}
        program_histories: dict = {}
        for entity_id in self._sub_energy_sensors:
            prefix = self._sub_sensor_prefix(entity_id)
            cache_path = self._sub_sensor_cache_path(entity_id)
            program_entity_id = self._program_sensors.get(entity_id)
            try:
                sub_df = ha_data.fetch_sub_sensor_history(
                    self,
                    entity_id,
                    cache_path,
                    timezone=self._timezone,
                    program_entity_id=program_entity_id,
                )
                sub_df = _strip_tz(sub_df, self._timezone)
                sub_sensors_dict[prefix] = sub_df
                # Build program_histories from the persisted CSV program column
                if "program" in sub_df.columns:
                    prog_df = sub_df[sub_df["program"].notna() & (sub_df["program"].astype(str) != "")][
                        ["timestamp", "program"]
                    ].copy()
                    if not prog_df.empty:
                        program_histories[prefix] = prog_df
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Sub-sensor %s history fetch failed: %s", entity_id, exc)

        if self._baseline_mode and sub_sensors_dict:
            subtract_dict = self._subtract_only_dict(sub_sensors_dict)
            baseline_df, removed_kwh = _subtract_sub_sensors(baseline_df, subtract_dict)
            included_prefixes = {self._sub_sensor_prefix(e) for e in self._baseline_included_sensors}
            _LOGGER.info(
                "Passive Baseline: removed %.1f kWh from (%s)%s.",
                removed_kwh,
                ", ".join(subtract_dict.keys()) or "none",
                f"; keeping {', '.join(sorted(included_prefixes))} in baseline" if included_prefixes else "",
            )

        start_date = baseline_df["timestamp"].min().date()
        end_date = min(baseline_df["timestamp"].max().date(), date.today() - timedelta(days=5))

        try:
            weather_df = weather.fetch_historical_weather(
                self._lat, self._lon, start_date, end_date, timezone=self._timezone
            )
            weather_df = _strip_tz(weather_df, self._timezone)
        except (OSError, KeyError, ValueError) as exc:
            _LOGGER.warning(
                "Historical weather fetch failed: %s — "
                "temp_c, heating_degree, cooling_degree and temp_rolling_3d will be "
                "imputed from training-set medians; forecast quality will be reduced.",
                exc,
            )
            weather_df = _empty_weather_df()

        # Cover the 5-day archive lag: Open-Meteo forecast API with past_days=7
        # includes measured (not forecast) data for recent hours.
        recent_weather = weather.fetch_open_meteo(self._lat, self._lon, timezone=self._timezone, past_days=7)
        recent_weather = _strip_tz(recent_weather, self._timezone)
        now_local = pd.Timestamp.now(tz=self._timezone).tz_localize(None)
        weather_df = _stitch_recent_weather(weather_df, recent_weather, now_local)

        away_df = ha_data.fetch_boolean_entity_history(self, self._away_mode_entity, days=30, timezone=self._timezone)
        if not away_df.empty:
            away_df = _strip_tz(away_df, self._timezone)

        presence_df = ha_data.fetch_presence_history(
            self, self._presence_sensors or None, days=30, timezone=self._timezone
        )
        if not presence_df.empty:
            presence_df = _strip_tz(presence_df, self._timezone)

        # ── Stage 2: Climate & DHW History ──────────────────────────────────
        climate_dfs: dict[str, Any] = {}
        for entity_id in self._climate_entities:
            c_path = self._climate_cache_path(entity_id)
            try:
                c_df = ha_data.fetch_climate_history(self, entity_id, c_path, timezone=self._timezone)
                if not c_df.empty:
                    climate_dfs[entity_id] = _strip_tz(c_df, self._timezone)
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Climate %s history fetch failed: %s", entity_id, exc)

        dhw_df = pd.DataFrame()
        if self._dhw_buffer_sensor:
            dhw_path = self._generic_sensor_cache_path(self._dhw_buffer_sensor, prefix="dhw")
            try:
                dhw_df = ha_data.fetch_generic_sensor_history(
                    self, self._dhw_buffer_sensor, dhw_path, column_name="buffer_temp", timezone=self._timezone
                )
                if not dhw_df.empty:
                    dhw_df = _strip_tz(dhw_df, self._timezone)
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("DHW %s history fetch failed: %s", self._dhw_buffer_sensor, exc)

        # ── Stage 2 / #55: Heating active sensor (τ calibration) ─────────────
        heating_active_df = pd.DataFrame()
        if self._heating_active_entity:
            ha_path = self._generic_sensor_cache_path(self._heating_active_entity, prefix="heating_active")
            try:
                heating_active_df = ha_data.fetch_generic_sensor_history(
                    self,
                    self._heating_active_entity,
                    ha_path,
                    column_name="heating_active",
                    timezone=self._timezone,
                )
                if not heating_active_df.empty:
                    heating_active_df = _strip_tz(heating_active_df, self._timezone)
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Heating active %s fetch failed: %s", self._heating_active_entity, exc)

        self._ml_model.train(
            baseline_df,
            weather_df,
            outdoor_df=None,
            weight_halflife_days=self._weight_halflife,
            canton=self._holiday_canton,
            country=self._holiday_country,
            ev_df=ev_df,
            sub_sensors_dict=sub_sensors_dict or None,
            away_df=away_df if not away_df.empty else None,
            presence_df=presence_df if not presence_df.empty else None,
            climate_dfs=climate_dfs or None,
            dhw_df=dhw_df if not dhw_df.empty else None,
            heating_active_df=heating_active_df if not heating_active_df.empty else None,
            program_histories=program_histories or None,
            room_areas=self._climate_room_areas or None,
            enable_regimes=self._enable_regimes,
            regime_count=self._regime_count,
        )
        _LOGGER.info("Retrained. MAE: %s", self._ml_model.last_mae)

    def _update_sensors(self) -> None:
        import numpy as np
        import pandas as pd

        # ── Fetch weather forecast ────────────────────────────────────────────
        forecast_df = weather.fetch_forecast(
            self._plz,
            self._lat,
            self._lon,
            self.args.get("srg_client_id"),
            self.args.get("srg_client_secret"),
            timezone=self._timezone,
        )
        forecast_df = _strip_tz(forecast_df, self._timezone)

        # ── Fetch recent actuals ──────────────────────────────────────────────
        # Uses the lightweight fetch (last 2 days only) to stay well within
        # AppDaemon's 10s callback limit. Full 30-day resync happens in _retrain().
        try:
            full_actuals = ha_data.fetch_recent_energy(
                self, self._energy_sensor, cache_path=self._cache_path, timezone=self._timezone
            )
            full_actuals = _strip_tz(full_actuals, self._timezone)
            # Subtract EV from actuals so lag_24h pointing at a charging hour
            # doesn't inflate tomorrow's baseline prediction.
            recent_actuals, ev_detected = ha_data.split_ev_charging(
                full_actuals, self._ev_threshold, charger_kw=self._ev_charger_kw
            )
        except (OSError, ValueError, KeyError) as exc:
            _LOGGER.warning("Could not fetch recent actuals for lag features: %s", exc)
            recent_actuals = None
            full_actuals = None
            ev_detected = None

        ev_hour_set: set = set()
        if ev_detected is not None and not ev_detected.empty:
            ev_hour_set = set(pd.to_datetime(ev_detected["timestamp"]).dt.floor("1h"))
        # Hours immediately before/after EV charging are also contaminated (ramp-up/down).
        # Excluded from _actuals_history so 7d/30d MAE sensors stay clean.
        _ev_adjacent: set = {ts + pd.Timedelta(hours=d) for ts in ev_hour_set for d in (-1, 1)}
        ev_mae_excluded: set = ev_hour_set | _ev_adjacent

        sub_sensors_recent: dict = {}
        for entity_id in self._sub_energy_sensors:
            prefix = self._sub_sensor_prefix(entity_id)
            cache_path = self._sub_sensor_cache_path(entity_id)
            program_entity_id = self._program_sensors.get(entity_id)
            try:
                sub_df = ha_data.fetch_recent_sub_sensor(
                    self,
                    entity_id,
                    cache_path,
                    timezone=self._timezone,
                    program_entity_id=program_entity_id,
                )
                sub_df = _strip_tz(sub_df, self._timezone)
                sub_sensors_recent[prefix] = sub_df
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Sub-sensor %s recent fetch failed: %s", entity_id, exc)

        if self._baseline_mode and sub_sensors_recent:
            if recent_actuals is not None:
                recent_actuals, _ = _subtract_sub_sensors(recent_actuals, self._subtract_only_dict(sub_sensors_recent))

        live_temp = self._read_live_temp()
        now_ts = pd.Timestamp.now(tz=self._timezone).tz_localize(None)
        away_series = self._build_away_prediction_series(now_ts)
        people_home_series = self._build_people_home_prediction_series(now_ts)

        # ── Stage 2: Climate & DHW Recent Fetch ──────────────────────────────
        climate_recent: dict[str, Any] = {}
        for entity_id in self._climate_entities:
            c_path = self._climate_cache_path(entity_id)
            try:
                c_df = ha_data.fetch_recent_climate(self, entity_id, c_path, timezone=self._timezone)
                if not c_df.empty:
                    climate_recent[entity_id] = _strip_tz(c_df, self._timezone)
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Climate %s recent fetch failed: %s", entity_id, exc)

        dhw_recent = pd.DataFrame()
        if self._dhw_buffer_sensor:
            dhw_path = self._generic_sensor_cache_path(self._dhw_buffer_sensor, prefix="dhw")
            try:
                dhw_recent = ha_data.fetch_recent_generic_sensor(
                    self, self._dhw_buffer_sensor, dhw_path, column_name="buffer_temp", timezone=self._timezone
                )
                if not dhw_recent.empty:
                    dhw_recent = _strip_tz(dhw_recent, self._timezone)
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("DHW %s recent fetch failed: %s", self._dhw_buffer_sensor, exc)

        if self._heating_active_entity:
            ha_path = self._generic_sensor_cache_path(self._heating_active_entity, prefix="heating_active")
            try:
                ha_data.fetch_recent_generic_sensor(
                    self,
                    self._heating_active_entity,
                    ha_path,
                    column_name="heating_active",
                    timezone=self._timezone,
                )
            except (OSError, KeyError, ValueError) as exc:
                _LOGGER.warning("Heating active %s recent fetch failed: %s", self._heating_active_entity, exc)

        # ── Heating active projection (setpoint hysteresis) ──────────────────
        heating_active_series = None
        heating_setpoint_on = None
        heating_setpoint_off = None
        if self._heating_active_entity and climate_recent:
            try:
                heating_active_series, heating_setpoint_on, heating_setpoint_off = (
                    self._build_heating_active_projection(forecast_df, climate_recent)
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("Heating active projection failed: %s", exc)

        # Cache inputs for scenario/what-if API (Stage 4)
        self._cached_forecast_df = forecast_df
        self._cached_live_temp = live_temp
        self._cached_recent_actuals = recent_actuals
        self._cached_sub_sensors = sub_sensors_recent or None
        self._cached_away_series = away_series
        self._cached_people_home = people_home_series
        self._cached_climate_recent = climate_recent or None
        self._cached_dhw_recent = dhw_recent if not dhw_recent.empty else None

        predictions = self._ml_model.predict(
            forecast_df,
            live_temp,
            recent_actuals,
            sub_sensors_recent=sub_sensors_recent or None,
            away_series=away_series,
            people_home_series=people_home_series,
            climate_recent=climate_recent or None,
            dhw_recent=dhw_recent if not dhw_recent.empty else None,
            room_areas=self._climate_room_areas or None,
            heating_active_series=heating_active_series,
            setpoint_on=heating_setpoint_on,
            setpoint_off=heating_setpoint_off,
        )
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"]).dt.tz_localize(None)
        self._publish_thermal_pressure()

        intervals = self._ml_model.predict_intervals(
            forecast_df,
            live_temp,
            recent_actuals,
            sub_sensors_recent=sub_sensors_recent or None,
            away_series=away_series,
            people_home_series=people_home_series,
            climate_recent=climate_recent or None,
            dhw_recent=dhw_recent if not dhw_recent.empty else None,
            room_areas=self._climate_room_areas or None,
            heating_active_series=heating_active_series,
            setpoint_on=heating_setpoint_on,
            setpoint_off=heating_setpoint_off,
        )
        if intervals is not None:
            intervals["timestamp"] = pd.to_datetime(intervals["timestamp"]).dt.tz_localize(None)

        # ── Store predictions for adaptive retrain tracking ───────────────────
        self._pred_history = _accumulate_pred_history(self._pred_history, predictions, now_ts)

        # ── Populate rolling actuals history for mae_7d / mae_30d sensors (#41) ─
        # keep-last semantics: fresher actuals overwrite older ones for the same hour.
        # Use full_actuals (pre-split, pre-sub-sensor-subtraction) so stored kWh
        # matches the training target (raw gross_kwh on non-EV hours).
        if full_actuals is not None and not full_actuals.empty:
            completed_cutoff = now_ts.floor("1h")  # skip the current (incomplete) hour
            # Retroactively evict EV hours + neighbors already in actuals_history.
            # Handles stale partials stored before the session was fully detected and
            # ramp-down hours stored before adjacent exclusion kicked in.
            for _ts in ev_mae_excluded:
                self._actuals_history.pop(_ts, None)
            for _ts, _kwh in zip(
                pd.to_datetime(full_actuals["timestamp"]).dt.floor("1h"),
                full_actuals["gross_kwh"].astype(float),
            ):
                if _ts >= completed_cutoff:  # skip current (partial) hour
                    continue
                if _ts not in ev_mae_excluded:
                    self._actuals_history[_ts] = _kwh
        actuals_cutoff = now_ts.floor("1h") - pd.Timedelta(days=30)
        self._actuals_history = {
            ts: kwh for ts, kwh in self._actuals_history.items() if pd.Timestamp(ts) >= actuals_cutoff
        }

        self._save_pred_history()

        _actuals_for_retrain = (
            recent_actuals[~pd.to_datetime(recent_actuals["timestamp"]).dt.floor("1h").isin(ev_mae_excluded)]
            if (recent_actuals is not None and not recent_actuals.empty and ev_mae_excluded)
            else recent_actuals
        )
        self._maybe_adaptive_retrain(_actuals_for_retrain)

        # ── Compute rolling MAE sensors (#41) ────────────────────────────────
        actuals_hist_df = None
        if self._actuals_history:
            actuals_hist_df = pd.DataFrame(
                [(ts, kwh) for ts, kwh in self._actuals_history.items()],
                columns=["timestamp", "gross_kwh"],
            )
        cutoff_7d = now_ts.floor("1h") - pd.Timedelta(days=7)
        pred_hist_7d = {ts: kwh for ts, kwh in self._pred_history.items() if pd.Timestamp(ts) >= cutoff_7d}
        mae_7d, n_7d = _compute_live_mae(pred_hist_7d, actuals_hist_df)
        mae_30d, n_30d = _compute_live_mae(self._pred_history, actuals_hist_df)

        # ── Relative MAE sensors (#54) ──────────────────────────────────────────
        actuals_7d = [v for ts, v in self._actuals_history.items() if pd.Timestamp(ts) >= cutoff_7d]
        actuals_30d = list(self._actuals_history.values())
        mean_7d = float(np.mean(actuals_7d)) if actuals_7d else float("nan")
        mean_30d = float(np.mean(actuals_30d)) if actuals_30d else float("nan")
        mae_7d_pct = round(mae_7d / mean_7d * 100, 2) if mean_7d > 0 and not math.isnan(mae_7d) else float("nan")
        mae_30d_pct = round(mae_30d / mean_30d * 100, 2) if mean_30d > 0 and not math.isnan(mae_30d) else float("nan")

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
                    forecast_df,
                    live_temp,
                    recent_actuals,
                    sub_sensors_recent=sub_sensors_recent or None,
                    away_series=away_series,
                    people_home_series=people_home_series,
                    climate_recent=climate_recent or None,
                    dhw_recent=dhw_recent if not dhw_recent.empty else None,
                    room_areas=self._climate_room_areas or None,
                    n=self._shap_top_n,
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("SHAP summary failed: %s", exc)

        # ── Aggregate and publish ─────────────────────────────────────────────
        aggregated = self._aggregate(
            predictions,
            full_actuals,
            live_temp,
            intervals=intervals,
            sub_sensors_recent=sub_sensors_recent or None,
        )
        aggregated["shap_top_features"] = shap_data
        aggregated["shap_narrative"] = _build_shap_narrative(shap_data)
        aggregated["mae_7d"] = mae_7d
        aggregated["mae_7d_pct"] = mae_7d_pct
        aggregated["mae_30d"] = mae_30d
        aggregated["mae_30d_pct"] = mae_30d_pct
        aggregated["mae_7d_n_pairs"] = n_7d
        aggregated["mae_30d_n_pairs"] = n_30d
        aggregated["is_anomaly"] = is_anomaly
        aggregated["anomaly_residual"] = anomaly_residual
        aggregated["anomaly_std"] = anomaly_std
        aggregated["anomaly_n"] = anomaly_n
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

    @staticmethod
    def _parse_room_areas(raw: Any) -> dict[str, float]:
        """Validate and coerce ``climate_room_areas`` from apps.yaml.

        Expects a dict ``{entity_id: float}``. Non-numeric values are silently
        dropped (invalid entries must not crash the app on startup).
        """
        if not isinstance(raw, dict):
            return {}
        result: dict[str, float] = {}
        for eid, area in raw.items():
            try:
                val = float(area)
                if val > 0:
                    result[str(eid)] = val
            except (TypeError, ValueError):
                pass
        return result

    def _build_away_prediction_series(self, now_ts: Any) -> Any:
        """Return a 48-value pd.Series (indexed by naive prediction timestamps) of is_away flags.

        Logic:
        - Entity is "off" (or not configured)  → all zeros.
        - Entity is "on", no return entity or return datetime already past → all ones.
        - Entity is "on", return datetime in the future → 1 before return_dt, 0 at/after.
        """
        import pandas as pd

        future_hours = pd.date_range(start=pd.Timestamp(now_ts).floor("1h"), periods=48, freq="1h")
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
                        return_dt = return_dt.tz_convert(self._timezone).tz_localize(None)
            except (ValueError, TypeError) as exc:
                _LOGGER.warning("Could not parse away_return_entity state as datetime: %s", exc)
                return_dt = None

        if return_dt is None or return_dt <= pd.Timestamp(now_ts):
            # No valid return time → away for the whole 48h window
            is_away[:] = 1
        else:
            is_away[future_hours < return_dt] = 1

        return is_away

    def _build_people_home_prediction_series(self, now_ts: Any) -> Any:
        """Return a 48-value pd.Series (indexed by naive prediction timestamps) of people_home counts.

        Counts how many person entities in _presence_sensors are currently in state "home",
        returns a constant Series with that count replicated across all 48 hours.
        If no sensors are configured, returns all zeros.
        """
        import pandas as pd

        future_hours = pd.date_range(start=pd.Timestamp(now_ts).floor("1h"), periods=48, freq="1h")

        if not self._presence_sensors:
            return pd.Series(0, index=future_hours, dtype=int)

        # Count how many person entities are currently home
        n_home = sum(1 for entity_id in self._presence_sensors if self.get_state(entity_id) == PRESENCE_STATE_HOME)

        return pd.Series(n_home, index=future_hours, dtype=int)

    def _build_heating_active_projection(
        self,
        forecast_df: Any,
        climate_recent: dict[str, Any],
    ) -> tuple[Any, float, float]:
        """Return (heating_active_series, setpoint_on, setpoint_off) for 48-hour window.

        heating_active_series: pd.Series[int] indexed by naive hourly timestamps (1=heating on).
        setpoint_on/off: live values derived from climate entities when available, else config defaults.
        Uses outdoor temp hysteresis to project future heating state.
        Only call when _heating_active_entity is configured.
        """
        import numpy as np
        import pandas as pd

        now_naive = pd.Timestamp.now(tz=self._timezone).tz_localize(None)
        future_hours = pd.date_range(start=now_naive.floor("1h"), periods=48, freq="1h")

        active = self.get_state(self._heating_active_entity) in ("on", "1", "true", "True")

        # Derive setpoint_on / setpoint_off from live climate entity data when available
        live_setpoints = []
        for eid, cdf in (climate_recent or {}).items():
            if cdf is not None and not cdf.empty and "setpoint" in cdf.columns:
                latest_sp = cdf.sort_values("timestamp").iloc[-1]["setpoint"]
                try:
                    live_setpoints.append(float(latest_sp))
                except (TypeError, ValueError):
                    pass

        live_setpoint = float(np.mean(live_setpoints)) if live_setpoints else None

        if active:
            s_on = live_setpoint if live_setpoint is not None else self._heating_setpoint_on
            s_off = self._heating_setpoint_off
        else:
            s_off = live_setpoint if live_setpoint is not None else self._heating_setpoint_off
            s_on = self._heating_setpoint_on

        temp_on = self._heating_temp_on
        temp_off = self._heating_temp_off

        # Align outdoor forecast temps to future hours
        forecast_indexed = forecast_df.set_index(pd.to_datetime(forecast_df["timestamp"]))["temp_c"]
        outdoor_temps = forecast_indexed.reindex(future_hours, method="nearest").ffill().bfill().values

        # Hysteresis projection
        state = int(active)
        states: list[int] = []
        for temp in outdoor_temps:
            if temp < temp_on:
                state = 1
            elif temp > temp_off:
                state = 0
            states.append(state)

        heating_active_series = pd.Series(states, index=future_hours, dtype=int)
        return heating_active_series, float(s_on), float(s_off)

    def _maybe_adaptive_retrain(self, actuals_df: Any) -> None:
        """Trigger an early retrain if live MAE exceeds threshold × CV MAE."""
        import pandas as pd

        cv_mae = self._ml_model.last_cv_mae
        if cv_mae is None:
            return
        # Use configured local time (tz-naive) consistent with pipeline timestamps.
        # datetime.now() would use system time, which is UTC in Docker/HA and
        # causes the cooldown to fire up to ±2h early/late and wrong during DST.
        _now = pd.Timestamp.now(self._timezone).tz_localize(None)
        hours_since = (_now - self._last_adaptive_retrain).total_seconds() / 3600
        if hours_since < 24:
            return
        live_mae, n_pairs = _compute_live_mae(self._pred_history, actuals_df)
        if n_pairs < 24:
            return
        if live_mae > self._adaptive_retrain_threshold * cv_mae:
            _LOGGER.warning(
                "Adaptive retrain triggered: live_MAE=%.4f > %.4f× cv_MAE=%.4f (over %d matched hours)",
                live_mae,
                self._adaptive_retrain_threshold,
                cv_mae,
                n_pairs,
            )
            self._last_adaptive_retrain = pd.Timestamp.now(self._timezone).tz_localize(None)
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
            # Relative MAE sensors (#54)
            "sensor.energy_forecast_relative_mae_7d",
            "sensor.energy_forecast_relative_mae_30d",
            # Anomaly detection sensor (#39)
            "binary_sensor.energy_forecast_unusual_consumption",
            # Interval sensors
            "sensor.energy_forecast_next_3h_low",
            "sensor.energy_forecast_next_3h_high",
            "sensor.energy_forecast_today_low",
            "sensor.energy_forecast_today_high",
            "sensor.energy_forecast_tomorrow_low",
            "sensor.energy_forecast_tomorrow_high",
            # Thermal pressure sensor (now served via MQTT)
            "sensor.energy_forecast_thermal_pressure_net",
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

    def _publish_thermal_pressure(self) -> None:
        """Publish current-hour thermal_pressure_net for the HP optimiser."""
        value = self._ml_model._latest_thermal_pressure_net or 0.0
        tau = self._ml_model._tau_hours  # pass None as-is; consumer distinguishes None vs 0.0
        attrs = {
            "tau_hours": tau,
            "unit_of_measurement": "°C·m²",
            "friendly_name": "Thermal Pressure (net)",
        }
        if self._mqtt_discovery:
            self._mqtt_set_sensor("energy_forecast_thermal_pressure_net", value)
            self._mqtt_publish_sensor_attributes("energy_forecast_thermal_pressure_net", attrs)
        else:
            self.set_state(
                "sensor.energy_forecast_thermal_pressure_net",
                state=str(round(float(value), 3)),
                attributes=attrs,
            )

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
        model = self._ml_model
        trained_str = model.last_trained.strftime("%Y-%m-%d %H:%M") if model.last_trained != datetime.min else "never"
        base_attrs = {
            "unit_of_measurement": "kWh",
            "attribution": ATTRIBUTION,
            "model_engine": str(model.engine),
            "last_trained": trained_str,
        }

        def safe_set(
            entity_id: str, value: Any, friendly_name: str, extra_attrs: dict | None = None, icon: str | None = None
        ) -> None:
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
        shap_narrative = data.get("shap_narrative") or ""
        for key, label in [
            ("next_1h", "Next 1h"),
            ("next_3h", "Next 3h"),
            ("today", "Today"),
            ("tomorrow", "Tomorrow"),
        ]:
            extra = None
            if key == "today":
                extra = {}
                if shap_features:
                    extra["shap_top_features"] = shap_features
                if shap_narrative:
                    extra["shap_narrative"] = shap_narrative
                if not extra:  # if no attributes, set to None
                    extra = None
            safe_set(
                f"sensor.energy_forecast_{key}",
                data.get(key, 0),
                f"Energy Forecast {label}",
                extra_attrs=extra,
                icon="mdi:lightning-bolt",
            )
        # In MQTT mode, publish shap_top_features and shap_narrative as json_attributes for energy_forecast_today
        if self._mqtt_discovery and (shap_features or shap_narrative):
            attrs = {}
            if shap_features:
                attrs["shap_top_features"] = shap_features
            if shap_narrative:
                attrs["shap_narrative"] = shap_narrative
            if attrs:
                self._mqtt_publish_sensor_attributes(
                    "energy_forecast_today",
                    attrs,
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
            low = data.get(f"{key}_low")
            high = data.get(f"{key}_high")
            if low is not None and high is not None:
                safe_set(
                    f"sensor.energy_forecast_{key}_low",
                    low,
                    f"Energy Forecast {label} Low (10th pct)",
                    icon="mdi:arrow-down-bold",
                )
                safe_set(
                    f"sensor.energy_forecast_{key}_high",
                    high,
                    f"Energy Forecast {label} High (90th pct)",
                    icon="mdi:arrow-up-bold",
                )

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
            "ev_charger_kw": self._ev_charger_kw,
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

        # ── Relative MAE sensors (#54) ──────────────────────────────────────────
        for key, label in [("mae_7d_pct", "7-day Rel. MAE"), ("mae_30d_pct", "30-day Rel. MAE")]:
            val = data.get(key, float("nan"))
            uid = key.removesuffix("_pct")  # convert "mae_7d_pct" → "mae_7d"
            if self._mqtt_discovery:
                if math.isnan(float(val)):
                    self._mqtt_set_sensor_raw(f"energy_forecast_relative_{uid}", "unavailable")
                else:
                    self._mqtt_set_sensor(f"energy_forecast_relative_{uid}", val)
            else:
                self.set_state(
                    f"sensor.energy_forecast_relative_{uid}",
                    state=str(val) if not math.isnan(float(val)) else "unavailable",
                    attributes={
                        "unit_of_measurement": "%",
                        "friendly_name": f"Energy Forecast {label}",
                        "unique_id": f"energy_forecast_relative_{uid}",
                        "icon": "mdi:percent",
                        "attribution": ATTRIBUTION,
                        "model_engine": str(model.engine),
                    },
                    replace=True,
                )

        # ── Anomaly detection sensor (#39) ────────────────────────────────────
        is_anomaly = data.get("is_anomaly", False)
        anomaly_attrs = {
            "residual_kwh": data.get("anomaly_residual", float("nan")),
            "residual_std_kwh": data.get("anomaly_std", float("nan")),
            "sigma_threshold": self._anomaly_sigma_threshold,
            "n_pairs": data.get("anomaly_n", 0),
        }
        if self._mqtt_discovery:
            _anomaly_uid = "energy_forecast_unusual_consumption"
            _state_topic = f"{self._mqtt_discovery_prefix}/energy_forecast/binary_sensor/{_anomaly_uid}/state"
            try:
                self.call_service(
                    "mqtt/publish",
                    topic=_state_topic,
                    payload="ON" if is_anomaly else "OFF",
                    retain=True,
                    namespace=self._mqtt_namespace,
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("MQTT anomaly state publish failed: %s", exc)
            _attr_topic = f"{self._mqtt_discovery_prefix}/energy_forecast/binary_sensor/{_anomaly_uid}/attributes"
            try:
                self.call_service(
                    "mqtt/publish",
                    topic=_attr_topic,
                    payload=json.dumps(anomaly_attrs),
                    retain=True,
                    namespace=self._mqtt_namespace,
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("MQTT anomaly attributes publish failed: %s", exc)
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

    def _aggregate(
        self,
        predictions: Any,
        full_actuals: Any,
        live_temp: float | None,
        intervals: Any = None,
        sub_sensors_recent: dict[str, Any] | None = None,
    ) -> dict:
        import numpy as np
        import pandas as pd

        now_dt = pd.Timestamp.now(tz=self._timezone).tz_localize(None)
        now_np = np.datetime64(now_dt.floor("h"))
        today_np = np.datetime64(now_dt.normalize())
        tomorrow_np = today_np + np.timedelta64(1, "D")
        yesterday_np = today_np - np.timedelta64(1, "D")

        p_times = pd.to_datetime(predictions["timestamp"]).values.astype("datetime64[ns]")
        p_vals = predictions["predicted_kwh"].values.astype(float)

        def _sum(s, e):
            return round(float(np.sum(p_vals[(p_times >= s) & (p_times < e)])), 3)

        def _blocks(day_start):
            return {
                f"{h:02d}_{h + 3:02d}": _sum(
                    day_start + np.timedelta64(h, "h"),
                    day_start + np.timedelta64(h + 3, "h"),
                )
                for h in range(0, 24, 3)
            }

        # For the "today" blended total, use actuals for elapsed hours.
        # Training is unconditionally EV-free → predictions are always EV-free baseline.
        # Strip EV in both modes so elapsed actuals and future predictions share the same
        # semantic.  Sub-sensors are only stripped in baseline_mode because in non-baseline
        # mode they are included in training and therefore in predictions too.
        blended_actuals = full_actuals
        if full_actuals is not None and not full_actuals.empty:
            blended_actuals, _ = ha_data.split_ev_charging(
                full_actuals, self._ev_threshold, charger_kw=self._ev_charger_kw
            )
            if self._baseline_mode and sub_sensors_recent:
                blended_actuals, _ = _subtract_sub_sensors(
                    blended_actuals, self._subtract_only_dict(sub_sensors_recent)
                )

        today_total, blocks_today = _blend_today_totals(p_times, p_vals, blended_actuals, today_np, tomorrow_np, now_np)

        result = {
            "next_1h": _sum(now_np, now_np + np.timedelta64(1, "h")),
            "next_3h": _sum(now_np, now_np + np.timedelta64(3, "h")),
            "today": today_total,
            "tomorrow": _sum(tomorrow_np, tomorrow_np + np.timedelta64(1, "D")),
            "blocks_today": blocks_today,
            "blocks_tomorrow": _blocks(tomorrow_np),
            "live_temp": live_temp,
            "ev_today": 0.0,
            "ev_yesterday": 0.0,
        }

        # ── Prediction intervals ─────────────────────────────────────────────
        if intervals is not None:
            iv_times = pd.to_datetime(intervals["timestamp"]).values.astype("datetime64[ns]")
            iv_low = intervals["low_kwh"].values.astype(float)
            iv_high = intervals["high_kwh"].values.astype(float)

            def _isum(vals, s, e):
                return round(float(np.sum(vals[(iv_times >= s) & (iv_times < e)])), 3)

            today_low, _ = _blend_today_totals(iv_times, iv_low, blended_actuals, today_np, tomorrow_np, now_np)
            today_high, _ = _blend_today_totals(iv_times, iv_high, blended_actuals, today_np, tomorrow_np, now_np)

            result.update(
                {
                    "next_3h_low": _isum(iv_low, now_np, now_np + np.timedelta64(3, "h")),
                    "next_3h_high": _isum(iv_high, now_np, now_np + np.timedelta64(3, "h")),
                    "today_low": today_low,
                    "today_high": today_high,
                    "tomorrow_low": _isum(iv_low, tomorrow_np, tomorrow_np + np.timedelta64(1, "D")),
                    "tomorrow_high": _isum(iv_high, tomorrow_np, tomorrow_np + np.timedelta64(1, "D")),
                }
            )

        # ── EV kWh from actuals: charger load per detected hour ──────────────
        # Always use original full_actuals for EV reporting, even in baseline_mode.
        # EV energy per detected hour = charger_kw (the model's assumed fixed load).
        # We clip by gross_kwh to handle partial hours near end-of-charge where
        # gross may be slightly below charger_kw.
        if full_actuals is not None and not full_actuals.empty:
            ev_mask = full_actuals["gross_kwh"] > self._ev_threshold
            ev_rows = full_actuals[ev_mask].copy()
            if not ev_rows.empty:
                ev_rows["ev_kwh"] = np.minimum(ev_rows["gross_kwh"], self._ev_charger_kw)
                ev_times = ev_rows["timestamp"].values.astype("datetime64[ns]")
                ev_vals = ev_rows["ev_kwh"].values.astype(float)

                def _ev_sum(s, e):
                    return round(float(np.sum(ev_vals[(ev_times >= s) & (ev_times < e)])), 3)

                result["ev_today"] = _ev_sum(today_np, tomorrow_np)
                result["ev_yesterday"] = _ev_sum(yesterday_np, today_np)

        return result

    def _load_pred_history(self) -> None:
        """Load prediction and actuals history from JSON file.

        Format: {"pred": {"<ISO ts>": float, ...}, "actuals": {"<ISO ts>": float, ...}}
        Applies 30-day pruning immediately and uses keep-first semantics (loaded entries
        do not overwrite anything already in memory). On startup, memory dicts are empty,
        so all loaded entries are accepted. Handles missing or corrupt files gracefully.
        """
        import pandas as pd

        if not PRED_HISTORY_PATH.exists():
            return  # No saved history yet, start fresh
        try:
            with open(PRED_HISTORY_PATH) as f:
                data = json.load(f)
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            _LOGGER.warning("Failed to load pred_history: %s", exc)
            return

        try:
            now = pd.Timestamp.now(tz=self._timezone).tz_localize(None)
            cutoff = now.floor("1h") - pd.Timedelta(days=30)

            # Load and prune prediction history
            for ts_str, kwh in data.get("pred", {}).items():
                ts = pd.Timestamp(ts_str)
                if ts >= cutoff and ts not in self._pred_history:
                    self._pred_history[ts] = float(kwh)

            # Load and prune actuals history
            for ts_str, kwh in data.get("actuals", {}).items():
                ts = pd.Timestamp(ts_str)
                if ts >= cutoff and ts not in self._actuals_history:
                    self._actuals_history[ts] = float(kwh)

            n_pred = len(self._pred_history)
            n_actuals = len(self._actuals_history)
            _LOGGER.info("Loaded pred_history: %d predictions, %d actuals", n_pred, n_actuals)
        except (ValueError, TypeError, IndexError) as exc:
            _LOGGER.warning("Failed to parse pred_history data: %s", exc)

    def _save_pred_history(self) -> None:
        """Serialize prediction and actuals history to JSON file.

        Writes atomically (to .tmp, then os.replace) to avoid corruption on crash.
        Catches OSError and logs warning — save failure should never break forecast cycle.
        """

        try:
            data = {
                "pred": {ts.isoformat(): float(kwh) for ts, kwh in self._pred_history.items()},
                "actuals": {ts.isoformat(): float(kwh) for ts, kwh in self._actuals_history.items()},
            }
            tmp_path = PRED_HISTORY_PATH.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                json.dump(data, f, default=str)
            os.replace(tmp_path, PRED_HISTORY_PATH)
        except OSError as exc:
            _LOGGER.warning("Failed to save pred_history: %s", exc)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _strip_tz(df: Any, timezone: str = "Europe/Zurich") -> Any:
    """Convert timestamp column to naive local time. Delegates to const.strip_tz."""
    return _strip_tz_util(df, timezone)


def _stitch_recent_weather(archive_df: Any, recent_df: Any, now_local: Any) -> Any:
    """Extend archive weather with the measured tail from Open-Meteo, archive wins on overlap.

    Trims recent_df to rows at or before now_local (strips forecast hours).
    Returns archive_df unchanged when recent_df is empty after trimming.
    """
    import pandas as pd

    tail = recent_df[recent_df["timestamp"] <= now_local]
    if tail.empty:
        return archive_df
    return (
        pd.concat([archive_df, tail])
        .drop_duplicates(subset="timestamp", keep="first")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _empty_weather_df() -> Any:
    import pandas as pd

    return pd.DataFrame(
        {
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "temp_c": pd.Series(dtype=float),
            "precipitation_mm": pd.Series(dtype=float),
            "sunshine_min": pd.Series(dtype=float),
            "wind_kmh": pd.Series(dtype=float),
            "cloud_cover_pct": pd.Series(dtype=float),
            "direct_radiation_wm2": pd.Series(dtype=float),
            "humidity": pd.Series(dtype=float),
        }
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
    import numpy as np
    import pandas as pd

    df = df.copy()
    ts_index = pd.DatetimeIndex(df["timestamp"])
    delta = pd.Series(np.zeros(len(df), dtype=float), index=ts_index)

    signed_dfs = [
        (solar_df, +1.0),
        (grid_export_df, -1.0),
        (battery_charge_df, -1.0),
        (battery_discharge_df, +1.0),
    ]
    for correction_df, sign in signed_dfs:
        if correction_df is None or correction_df.empty:
            continue
        corr = (
            correction_df.set_index(pd.DatetimeIndex(correction_df["timestamp"]))["kwh"].reindex(ts_index).fillna(0.0)
        )
        delta = delta + sign * corr

    df["gross_kwh"] = (df["gross_kwh"] + delta.values).clip(lower=0.0)
    return df


def _subtract_sub_sensors(
    df: Any,
    sub_sensors_dict: dict[str, Any] | None,
    column: str = "gross_kwh",
) -> tuple[Any, float]:
    """Subtract all sub-sensor consumption from the target column.

    Returns (df_corrected, total_removed_kwh).
    """
    import numpy as np
    import pandas as pd

    if not sub_sensors_dict:
        return df.copy(), 0.0

    df = df.copy()
    ts_index = pd.DatetimeIndex(df["timestamp"])
    sub_total = pd.Series(np.zeros(len(df), dtype=float), index=ts_index)

    for prefix, sub_df in sub_sensors_dict.items():
        if sub_df is None or sub_df.empty:
            continue
        # Sub-sensors use column "kwh"
        val = sub_df.set_index(pd.DatetimeIndex(sub_df["timestamp"]))["kwh"].reindex(ts_index).fillna(0.0)
        sub_total += val

    removed_kwh = float(sub_total.sum())
    df[column] = (df[column] - sub_total.values).clip(lower=0.0)
    return df, removed_kwh


def _blend_today_totals(
    p_times: Any,  # np.ndarray of datetime64[ns] — prediction timestamps
    p_vals: Any,  # np.ndarray of float — predicted kWh per hour
    full_actuals: Any,  # pd.DataFrame | None — cols: timestamp, gross_kwh
    today_np: Any,  # np.datetime64 — midnight today
    tomorrow_np: Any,  # np.datetime64 — midnight tomorrow
    now_np: Any,  # np.datetime64 — current hour (floored)
) -> tuple[float, dict]:
    """Compute today's blended total and 3h blocks.

    Elapsed hours (< now_np) use actuals from full_actuals where available;
    future hours (>= now_np) use model predictions.  Falls back to predictions
    only when full_actuals is None or empty.
    """
    import numpy as np

    fa_times = None
    fa_vals = None
    if full_actuals is not None and not getattr(full_actuals, "empty", True):
        fa_times = full_actuals["timestamp"].values.astype("datetime64[ns]")
        fa_vals = full_actuals["gross_kwh"].values.astype(float)

    def _pred_sum(s: Any, e: Any) -> float:
        return float(np.sum(p_vals[(p_times >= s) & (p_times < e)]))

    def _actual_sum(s: Any, e: Any) -> float:
        if fa_times is None:
            return 0.0
        return float(np.sum(fa_vals[(fa_times >= s) & (fa_times < e)]))

    def _blended(s: Any, e: Any) -> float:
        elapsed_end = min(e, now_np)
        future_start = max(s, now_np)
        total = 0.0
        if elapsed_end > s:
            total += _actual_sum(s, elapsed_end)
        if future_start < e:
            total += _pred_sum(future_start, e)
        return round(total, 3)

    today_total = _blended(today_np, tomorrow_np)
    blocks = {
        f"{h:02d}_{h + 3:02d}": _blended(
            today_np + np.timedelta64(h, "h"),
            today_np + np.timedelta64(h + 3, "h"),
        )
        for h in range(0, 24, 3)
    }
    return today_total, blocks


def _accumulate_pred_history(
    pred_history: dict,
    predictions: Any,
    now_ts: Any,
) -> dict:
    """Keep-first at ≤25h horizon; prune entries older than 30 days.

    Each target hour is stored the first time it enters the 24–25h window, so
    mae_7d/mae_30d measure genuine day-ahead forecast quality.  Targets beyond
    25h are deferred until the next hourly cycle when they enter the window.
    """
    import pandas as pd

    cutoff = now_ts.floor("1h") - pd.Timedelta(days=30)
    result = {ts: kwh for ts, kwh in pred_history.items() if pd.Timestamp(ts) >= cutoff}
    now_h = now_ts.floor("1h")
    for _, row in predictions.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        if ts not in result and (ts - now_h).total_seconds() / 3600 <= 25:
            result[ts] = float(row["predicted_kwh"])
    return result


def _compute_live_mae(
    pred_history: dict,  # {timestamp-like: predicted_kwh}
    actuals_df: Any,  # pd.DataFrame | None — cols: timestamp, gross_kwh
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
        pd.Timestamp(ts).floor("1h"): float(kwh) for ts, kwh in zip(actuals_df["timestamp"], actuals_df["gross_kwh"])
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
    pred_history: dict,  # {timestamp-like: predicted_kwh}  keep-first, raw ts keys
    actuals_history: dict,  # {pd.Timestamp: float}            keep-last, floored 1h keys
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
