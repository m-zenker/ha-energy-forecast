"""DataUpdateCoordinator for HA Energy Forecast."""
from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed

from .const import (
    CONF_ENERGY_SENSOR,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PLZ,
    CONF_LATITUDE,
    CONF_LONGITUDE,
    DOMAIN,
    MIN_HISTORY_HOURS,
    RETRAIN_INTERVAL_HOURS,
    UPDATE_INTERVAL_MINUTES,
)
from . import ha_data, weather
from .model import EnergyForecastModel

_LOGGER = logging.getLogger(__name__)


class EnergyForecastCoordinator(DataUpdateCoordinator):
    """Coordinator — fetches weather, runs predictions, triggers retraining."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(minutes=UPDATE_INTERVAL_MINUTES),
        )
        self._entry = entry
        cfg = {**entry.data, **entry.options}

        self._energy_sensor: str = cfg[CONF_ENERGY_SENSOR]
        self._outdoor_sensor: str | None = cfg.get(CONF_OUTDOOR_TEMP_SENSOR)
        self._plz: str = cfg[CONF_PLZ]
        self._lat: float = float(cfg[CONF_LATITUDE])
        self._lon: float = float(cfg[CONF_LONGITUDE])

        model_dir = Path(hass.config.config_dir) / "custom_components" / DOMAIN / "models"
        self.ml_model = EnergyForecastModel(model_dir)
        self.predictions: Any = None

    async def _async_update_data(self) -> dict:
        """Called by HA every UPDATE_INTERVAL_MINUTES."""
        await self._maybe_retrain()

        if self.ml_model.model is None:
            raise UpdateFailed(
                f"Model not yet trained (need {MIN_HISTORY_HOURS} hourly records)"
            )

        forecast = await self._fetch_forecast()
        live_temp = self._read_live_temp()

        predictions = await self.hass.async_add_executor_job(
            self.ml_model.predict, forecast, live_temp
        )
        self.predictions = predictions
        return self._aggregate(predictions, live_temp)

    async def _maybe_retrain(self) -> None:
        if self.ml_model.hours_since_trained() < RETRAIN_INTERVAL_HOURS:
            return
        _LOGGER.info("Starting model retraining...")
        try:
            await self.hass.async_add_executor_job(self._retrain_sync)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("Retraining failed: %s", exc)

    def _retrain_sync(self) -> None:
        """Synchronous retraining — runs in executor thread."""
        import pandas as pd  # noqa: PLC0415 — lazy import, pandas may not be ready at module load

        token  = self._get_ha_token()
        ha_url = self._get_ha_url()

        energy_df = ha_data.fetch_energy_history(ha_url, token, self._energy_sensor)

        if len(energy_df) < MIN_HISTORY_HOURS:
            _LOGGER.warning(
                "Only %d hourly records (need %d) — skipping training",
                len(energy_df), MIN_HISTORY_HOURS,
            )
            return

        start_date = energy_df["timestamp"].min().date()
        end_date   = energy_df["timestamp"].max().date()

        try:
            weather_df = weather.fetch_historical_weather(
                self._lat, self._lon, start_date, end_date
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Historical weather fetch failed: %s", exc)
            weather_df = pd.DataFrame(
                columns=["timestamp", "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh"]
            )

        outdoor_df = None
        if self._outdoor_sensor:
            try:
                outdoor_df = ha_data.fetch_outdoor_temp_history(
                    ha_url, token, self._outdoor_sensor,
                    energy_df["timestamp"].min().to_pydatetime(),
                    energy_df["timestamp"].max().to_pydatetime(),
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("Outdoor temp history fetch failed: %s", exc)

        self.ml_model.train(energy_df, weather_df, outdoor_df)

    async def _fetch_forecast(self) -> Any:
        return await self.hass.async_add_executor_job(
            weather.fetch_forecast, self._plz, self._lat, self._lon
        )

    def _read_live_temp(self) -> float | None:
        if not self._outdoor_sensor:
            return None
        state = self.hass.states.get(self._outdoor_sensor)
        if state is None or state.state in ("unavailable", "unknown"):
            return None
        try:
            return float(state.state)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _aggregate(predictions: Any, live_temp: float | None) -> dict:
        import pandas as pd  # noqa: PLC0415
        from datetime import timedelta  # already imported above but explicit here

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

    def _get_ha_token(self) -> str:
        return (
            os.environ.get("SUPERVISOR_TOKEN")
            or os.environ.get("HASSIO_TOKEN")
            or ""
        )

    def _get_ha_url(self) -> str:
        return "http://homeassistant:8123"
