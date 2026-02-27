"""Sensor platform for HA Energy Forecast."""
from __future__ import annotations

import logging
from datetime import datetime

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import (
    ATTRIBUTION,
    BLOCK_SLOTS,
    DOMAIN,
)
from .coordinator import EnergyForecastCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    coordinator: EnergyForecastCoordinator = hass.data[DOMAIN][entry.entry_id]

    entities: list[SensorEntity] = [
        EnergyForecastSummary(coordinator, entry, "next_3h",   "Next 3h"),
        EnergyForecastSummary(coordinator, entry, "today",     "Today"),
        EnergyForecastSummary(coordinator, entry, "tomorrow",  "Tomorrow"),
        EnergyForecastMAE(coordinator, entry),
    ]

    for slot in BLOCK_SLOTS:
        h_start = int(slot[:2])
        h_end   = int(slot[3:])
        entities.append(EnergyForecastBlock(coordinator, entry, "today",    slot, h_start, h_end))
        entities.append(EnergyForecastBlock(coordinator, entry, "tomorrow", slot, h_start, h_end))

    async_add_entities(entities)


# ── Shared base ───────────────────────────────────────────────────────────────

class _EnergyForecastBase(CoordinatorEntity[EnergyForecastCoordinator]):
    """Base class shared by all HA Energy Forecast sensors."""

    _attr_attribution = ATTRIBUTION
    _attr_has_entity_name = True

    def __init__(
        self,
        coordinator: EnergyForecastCoordinator,
        entry: ConfigEntry,
        unique_suffix: str,
    ) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_{entry.entry_id}_{unique_suffix}"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name="HA Energy Forecast",
            manufacturer="HA Energy Forecast",
            model=coordinator.ml_model.engine,
            sw_version="1.0.0",
        )

    @property
    def extra_state_attributes(self) -> dict:
        model = self.coordinator.ml_model
        return {
            "last_trained": (
                model.last_trained.strftime("%Y-%m-%d %H:%M")
                if model.last_trained != datetime.min
                else "never"
            ),
            "model_engine": model.engine,
            "live_outdoor_temp": (
                f"{self.coordinator.data['live_temp']:.1f} °C"
                if self.coordinator.data and self.coordinator.data.get("live_temp") is not None
                else "n/a"
            ),
        }


# ── Summary sensors (next_3h / today / tomorrow) ──────────────────────────────

class EnergyForecastSummary(_EnergyForecastBase):
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class  = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "kWh"
    _attr_icon = "mdi:lightning-bolt"

    def __init__(
        self,
        coordinator: EnergyForecastCoordinator,
        entry: ConfigEntry,
        key: str,
        label: str,
    ) -> None:
        super().__init__(coordinator, entry, key)
        self._key   = key
        self._label = label
        self._attr_name = f"Gross Forecast {label}"

    @property
    def native_value(self) -> float | None:
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get(self._key)


# ── 3-hour block sensors ──────────────────────────────────────────────────────

class EnergyForecastBlock(_EnergyForecastBase):
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_state_class  = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "kWh"
    _attr_icon = "mdi:clock-time-three-outline"

    def __init__(
        self,
        coordinator: EnergyForecastCoordinator,
        entry: ConfigEntry,
        day_label: str,   # "today" | "tomorrow"
        slot: str,         # "00_03", "03_06", ...
        h_start: int,
        h_end: int,
    ) -> None:
        super().__init__(coordinator, entry, f"{day_label}_{slot}")
        self._day_label = day_label
        self._slot      = slot
        self._attr_name = (
            f"Gross Forecast {day_label.capitalize()} "
            f"{h_start:02d}:00–{h_end:02d}:00"
        )

    @property
    def native_value(self) -> float | None:
        if not self.coordinator.data:
            return None
        blocks_key = f"blocks_{self._day_label}"
        return self.coordinator.data.get(blocks_key, {}).get(self._slot)


# ── MAE diagnostics sensor ────────────────────────────────────────────────────

class EnergyForecastMAE(_EnergyForecastBase):
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = "kWh"
    _attr_icon = "mdi:chart-bell-curve"
    _attr_name = "Forecast Model MAE"
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    def __init__(
        self,
        coordinator: EnergyForecastCoordinator,
        entry: ConfigEntry,
    ) -> None:
        super().__init__(coordinator, entry, "mae")

    @property
    def native_value(self) -> float | None:
        return self.coordinator.ml_model.last_mae

    @property
    def extra_state_attributes(self) -> dict:
        model = self.coordinator.ml_model
        base = super().extra_state_attributes
        return {
            **base,
            "feature_count": len(model.feature_cols),
            "features": ", ".join(model.feature_cols),
            "hours_since_trained": (
                round(model.hours_since_trained(), 1)
                if model.hours_since_trained() != float("inf")
                else "never"
            ),
        }
