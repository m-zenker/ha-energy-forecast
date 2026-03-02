"""
HA Energy Forecast — AppDaemon app.

Trains a LightGBM (or sklearn GBR fallback) model on your historical grid
import data, then publishes 48-hour hourly forecasts as HA sensor entities
every hour.

Configuration lives entirely in apps.yaml — no config flow, no manifest.json,
no HA package sandbox to fight.

Sensors created (all under the 'sensor' domain):
  sensor.energy_forecast_next_3h      — kWh for the next 3 hours
  sensor.energy_forecast_today        — kWh total for today
  sensor.energy_forecast_tomorrow     — kWh total for tomorrow
  sensor.energy_forecast_today_HH_HH  — 3-hour block sensors for today (×8)
  sensor.energy_forecast_tomorrow_HH_HH — 3-hour block sensors for tomorrow (×8)
  sensor.energy_forecast_model_mae    — last model MAE (diagnostic)
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import hassapi as hass

from . import ha_data, weather
from .model import EnergyForecastModel

# ── Operational constants ─────────────────────────────────────────────────────
RETRAIN_INTERVAL_S   = 168 * 3600   # Weekly retraining
UPDATE_INTERVAL_S    = 60  * 60     # Hourly prediction refresh
MIN_HISTORY_HOURS    = 720          # ~1 month before first train attempt
BLOCK_SLOTS          = [f"{h:02d}_{h+3:02d}" for h in range(0, 24, 3)]
ATTRIBUTION          = "HA Energy Forecast — LightGBM + MeteoSwiss/Open-Meteo"


class EnergyForecast(hass.Hass):
    """AppDaemon app that forecasts household energy consumption."""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Called by AppDaemon on startup. Reads config, wires up timers."""
        self.log("HA Energy Forecast initialising…")

        # ── Read config from apps.yaml ──────────────────────────────────────
        self._energy_sensor: str   = self.args["energy_sensor"]
        self._outdoor_sensor: str | None = self.args.get("outdoor_temp_sensor")
        self._plz: str             = str(self.args["plz"])
        self._lat: float           = float(self.args["latitude"])
        self._lon: float           = float(self.args["longitude"])
        self._timezone: str        = self.args.get("timezone", "Europe/Zurich")

        # ── Model storage ───────────────────────────────────────────────────
        # Store models next to the app files so they survive AD restarts.
        model_dir = Path(__file__).parent / "models"
        self._ml_model = EnergyForecastModel(model_dir)

        # Lock prevents a slow retrain and a prediction update from racing.
        self._lock = threading.Lock()

        # ── Publish initial sensor stubs so entities exist immediately ───────
        self._publish_unavailable()

        # ── Schedule retraining ─────────────────────────────────────────────
        # First run after 10 s (gives HA time to finish startup), then weekly.
        self.run_in(self._retrain_cb, 10)
        self.run_every(self._retrain_cb, f"now+{RETRAIN_INTERVAL_S + 10}", RETRAIN_INTERVAL_S)

        # ── Schedule prediction updates ──────────────────────────────────────
        # Start 2 min after first retrain attempt, then every hour.
        self.run_in(self._update_cb, 130)
        self.run_every(self._update_cb, f"now+{UPDATE_INTERVAL_S + 130}", UPDATE_INTERVAL_S)

        self.log("HA Energy Forecast ready — sensors will populate after first training.")

    # ── Timer callbacks ───────────────────────────────────────────────────────

    def _retrain_cb(self, kwargs: dict) -> None:
        """Scheduled retraining callback. Runs in AppDaemon's thread pool."""
        if not self._lock.acquire(blocking=False):
            self.log("Retrain skipped — previous run still in progress", level="WARNING")
            return
        try:
            self._retrain()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Retraining failed: {exc}", level="ERROR")
        finally:
            self._lock.release()

    def _update_cb(self, kwargs: dict) -> None:
        """Scheduled prediction update callback. Runs in AppDaemon's thread pool."""
        if self._ml_model.model is None:
            self.log(
                f"Model not yet trained (need {MIN_HISTORY_HOURS} h of history). "
                "Sensors remain unavailable.",
                level="INFO",
            )
            return
        if not self._lock.acquire(blocking=False):
            self.log("Prediction update skipped — retrain in progress", level="DEBUG")
            return
        try:
            self._update_sensors()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Sensor update failed: {exc}", level="ERROR")
        finally:
            self._lock.release()

    # ── Core logic ────────────────────────────────────────────────────────────

    def _retrain(self) -> None:
        """Pull history, build features, train model. Blocking — runs in thread."""
        import pandas as pd

        self.log("Starting model retraining…")

        energy_df = ha_data.fetch_energy_history(self, self._energy_sensor)

        if len(energy_df) < MIN_HISTORY_HOURS:
            self.log(
                f"Only {len(energy_df)} hourly records (need {MIN_HISTORY_HOURS}) "
                "— skipping training. Check your energy_sensor entity_id.",
                level="WARNING",
            )
            return

        start_date = energy_df["timestamp"].min().date()
        end_date   = energy_df["timestamp"].max().date()

        try:
            weather_df = weather.fetch_historical_weather(
                self._lat, self._lon, start_date, end_date
            )
        except Exception as exc:  # noqa: BLE001
            self.log(f"Historical weather fetch failed: {exc} — training without it", level="WARNING")
            weather_df = pd.DataFrame(
                columns=["timestamp", "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh"]
            )

        outdoor_df = None
        if self._outdoor_sensor:
            try:
                outdoor_df = ha_data.fetch_outdoor_temp_history(
                    self,
                    self._outdoor_sensor,
                    energy_df["timestamp"].min().to_pydatetime(),
                    energy_df["timestamp"].max().to_pydatetime(),
                )
            except Exception as exc:  # noqa: BLE001
                self.log(f"Outdoor temp history fetch failed: {exc}", level="WARNING")

        self._ml_model.train(energy_df, weather_df, outdoor_df)
        self.log(
            f"Retraining complete — engine: {self._ml_model.engine}, "
            f"MAE: {self._ml_model.last_mae} kWh/h"
        )

    def _update_sensors(self) -> None:
        """Run prediction and push all sensor states to HA. Blocking — runs in thread."""
        import pandas as pd

        forecast_df = weather.fetch_forecast(self._plz, self._lat, self._lon)
        live_temp   = self._read_live_temp()
        predictions = self._ml_model.predict(forecast_df, live_temp)
        aggregated  = self._aggregate(predictions, live_temp)
        self._publish(aggregated)

    # ── Sensor publishing ─────────────────────────────────────────────────────

    def _publish_unavailable(self) -> None:
        """Create sensor stubs so entities appear in HA immediately on startup."""
        base_attrs = {
            "unit_of_measurement": "kWh",
            "friendly_name_template": "",
            "attribution": ATTRIBUTION,
        }
        for slot in ["next_3h", "today", "tomorrow"]:
            self.set_state(
                f"sensor.energy_forecast_{slot}",
                state="unavailable",
                attributes={**base_attrs, "friendly_name": f"Energy Forecast {slot.replace('_', ' ').title()}"},
            )
        for day in ("today", "tomorrow"):
            for slot in BLOCK_SLOTS:
                h_start, h_end = int(slot[:2]), int(slot[3:])
                self.set_state(
                    f"sensor.energy_forecast_{day}_{slot}",
                    state="unavailable",
                    attributes={
                        **base_attrs,
                        "friendly_name": f"Energy Forecast {day.title()} {h_start:02d}:00–{h_end:02d}:00",
                    },
                )
        self.set_state(
            "sensor.energy_forecast_model_mae",
            state="unavailable",
            attributes={
                "unit_of_measurement": "kWh",
                "friendly_name": "Energy Forecast Model MAE",
                "attribution": ATTRIBUTION,
            },
        )

    def _publish(self, data: dict) -> None:
        """Push aggregated prediction results into HA sensor states."""
        model  = self._ml_model
        trained_str = (
            model.last_trained.strftime("%Y-%m-%d %H:%M")
            if model.last_trained != datetime.min
            else "never"
        )
        live_temp = data.get("live_temp")
        live_temp_str = f"{live_temp:.1f} °C" if live_temp is not None else "n/a"

        shared_attrs = {
            "unit_of_measurement": "kWh",
            "attribution": ATTRIBUTION,
            "model_engine": model.engine,
            "last_trained": trained_str,
            "live_outdoor_temp": live_temp_str,
        }

        # Summary sensors
        for key, label in [
            ("next_3h", "Next 3h"),
            ("today",   "Today"),
            ("tomorrow","Tomorrow"),
        ]:
            self.set_state(
                f"sensor.energy_forecast_{key}",
                state=data[key],
                attributes={**shared_attrs, "friendly_name": f"Energy Forecast {label}"},
            )

        # 3-hour block sensors
        for day in ("today", "tomorrow"):
            blocks = data[f"blocks_{day}"]
            for slot in BLOCK_SLOTS:
                h_start, h_end = int(slot[:2]), int(slot[3:])
                self.set_state(
                    f"sensor.energy_forecast_{day}_{slot}",
                    state=blocks.get(slot, 0),
                    attributes={
                        **shared_attrs,
                        "friendly_name": f"Energy Forecast {day.title()} {h_start:02d}:00–{h_end:02d}:00",
                    },
                )

        # MAE diagnostic sensor
        self.set_state(
            "sensor.energy_forecast_model_mae",
            state=model.last_mae if model.last_mae is not None else "unavailable",
            attributes={
                "unit_of_measurement": "kWh",
                "friendly_name": "Energy Forecast Model MAE",
                "attribution": ATTRIBUTION,
                "model_engine": model.engine,
                "last_trained": trained_str,
                "feature_count": len(model.feature_cols),
                "features": ", ".join(model.feature_cols),
                "hours_since_trained": (
                    round(model.hours_since_trained(), 1)
                    if model.hours_since_trained() != float("inf")
                    else "never"
                ),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

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
    def _aggregate(predictions: Any, live_temp: float | None) -> dict:
        import pandas as pd
        from datetime import timedelta

        now      = pd.Timestamp.now(tz="Europe/Zurich")
        today    = now.normalize()
        tomorrow = today + timedelta(days=1)
        p        = predictions

        def _sum(mask: Any) -> float:
            return round(float(p[mask]["predicted_kwh"].sum()), 3)

        def _blocks(day: Any) -> dict:
            result = {}
            for h in range(0, 24, 3):
                s = day + timedelta(hours=h)
                e = s + timedelta(hours=3)
                result[f"{h:02d}_{h+3:02d}"] = _sum(
                    (p["timestamp"] >= s) & (p["timestamp"] < e)
                )
            return result

        return {
            "next_3h":         _sum((p["timestamp"] >= now) & (p["timestamp"] < now + timedelta(hours=3))),
            "today":           _sum((p["timestamp"] >= today) & (p["timestamp"] < tomorrow)),
            "tomorrow":        _sum((p["timestamp"] >= tomorrow) & (p["timestamp"] < tomorrow + timedelta(days=1))),
            "blocks_today":    _blocks(today),
            "blocks_tomorrow": _blocks(tomorrow),
            "live_temp":       live_temp,
        }
