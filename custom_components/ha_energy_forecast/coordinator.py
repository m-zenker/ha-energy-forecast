"""
DataUpdateCoordinator for HA Energy Forecast.

Orchestrates:
  - Periodic retraining (executor thread)
  - Hourly forecast fetch + prediction (executor thread)
  - Exposing results to sensor entities via coordinator.data
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
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

        # Cache last predictions so sensors can read without triggering fetches
        self.predictions: pd.DataFrame | None = None

    # ── HA entry points ───────────────────────────────────────────────────────

    async def _async_update_data(self) -> dict:
        """Called by HA every UPDATE_INTERVAL_MINUTES."""
        await self._maybe_retrain()

        if self.ml_model.model is None:
            raise UpdateFailed(
                "Model not yet trained — waiting for sufficient history "
                f"(need {MIN_HISTORY_HOURS} hourly records)"
            )

        forecast = await self._fetch_forecast()
        live_temp = self._read_live_temp()

        predictions = await self.hass.async_add_executor_job(
            self.ml_model.predict, forecast, live_temp
        )
        self.predictions = predictions

        return self._aggregate(predictions, live_temp)

    # ── Retraining ────────────────────────────────────────────────────────────

    async def _maybe_retrain(self) -> None:
        if self.ml_model.hours_since_trained() < RETRAIN_INTERVAL_HOURS:
            return

        _LOGGER.info("Starting model retraining (executor thread)...")
        try:
            await self.hass.async_add_executor_job(self._retrain_sync)
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("Retraining failed: %s", exc)

    def _retrain_sync(self) -> None:
        """Synchronous retraining — runs in executor thread."""
        token = self._get_ha_token()
        ha_url = self._get_ha_url()

        energy_df = ha_data.fetch_energy_history(ha_url, token, self._energy_sensor)

        if len(energy_df) < MIN_HISTORY_HOURS:
            _LOGGER.warning(
                "Only %d hourly records available (need %d) — skipping training",
                len(energy_df),
                MIN_HISTORY_HOURS,
            )
            return

        start_date = energy_df["timestamp"].min().date()
        end_date   = energy_df["timestamp"].max().date()

        try:
            weather_df = weather.fetch_historical_weather(
                self._lat, self._lon, start_date, end_date
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Historical weather fetch failed: %s — using empty", exc)
            weather_df = pd.DataFrame(
                columns=["timestamp", "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh"]
            )

        outdoor_df: pd.DataFrame | None = None
        if self._outdoor_sensor:
            try:
                outdoor_df = ha_data.fetch_outdoor_temp_history(
                    ha_url,
                    token,
                    self._outdoor_sensor,
                    energy_df["timestamp"].min().to_pydatetime(),
                    energy_df["timestamp"].max().to_pydatetime(),
                )
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("Outdoor temp history fetch failed: %s", exc)

        self.ml_model.train(energy_df, weather_df, outdoor_df)

    # ── Forecast fetch ────────────────────────────────────────────────────────

    async def _fetch_forecast(self) -> pd.DataFrame:
        return await self.hass.async_add_executor_job(
            weather.fetch_forecast, self._plz, self._lat, self._lon
        )

    # ── Live sensor ───────────────────────────────────────────────────────────

    def _read_live_temp(self) -> float | None:
        if not self._outdoor_sensor:
            return None
        state = self.hass.states.get(self._outdoor_sensor)
        if state is None or state.state in ("unavailable", "unknown", None):
            _LOGGER.debug("Outdoor sensor %s unavailable", self._outdoor_sensor)
            return None
        try:
            return float(state.state)
        except (ValueError, TypeError):
            return None

    # ── Aggregation ───────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate(predictions: pd.DataFrame, live_temp: float | None) -> dict:
        """
        Collapse hourly predictions into the values sensors need:
          next_3h, today, tomorrow, blocks_today, blocks_tomorrow, mae, engine, live_temp
        """
        now      = pd.Timestamp.now(tz="Europe/Zurich")
        today    = now.normalize()
        tomorrow = today + timedelta(days=1)
        p = predictions

        def _sum(mask) -> float:
            return round(float(p[mask]["predicted_kwh"].sum()), 3)

        next_3h_mask  = (p["timestamp"] >= now)      & (p["timestamp"] < now + timedelta(hours=3))
        today_mask    = (p["timestamp"] >= today)    & (p["timestamp"] < tomorrow)
        tomorrow_mask = (p["timestamp"] >= tomorrow) & (p["timestamp"] < tomorrow + timedelta(days=1))

        # 3-hour blocks
        def _blocks(day: pd.Timestamp) -> dict[str, float]:
            result = {}
            for h in range(0, 24, 3):
                block_s = day + timedelta(hours=h)
                block_e = block_s + timedelta(hours=3)
                mask = (p["timestamp"] >= block_s) & (p["timestamp"] < block_e)
                key = f"{h:02d}_{h+3:02d}"
                result[key] = _sum(mask)
            return result

        return {
            "next_3h":        _sum(next_3h_mask),
            "today":          _sum(today_mask),
            "tomorrow":       _sum(tomorrow_mask),
            "blocks_today":   _blocks(today),
            "blocks_tomorrow": _blocks(tomorrow),
            "live_temp":      live_temp,
            "mae":            None,   # filled by sensor from ml_model directly
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_ha_token(self) -> str:
        token = (
            os.environ.get("SUPERVISOR_TOKEN")
            or os.environ.get("HASSIO_TOKEN")
            or ""
        )
        return token

    def _get_ha_url(self) -> str:
        # When running as an add-on or in the same container, use the internal URL
        return "http://homeassistant:8123"
