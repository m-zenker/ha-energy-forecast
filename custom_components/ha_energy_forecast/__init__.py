"""HA Energy Forecast — HACS custom integration."""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady

from .const import DOMAIN, PLATFORMS
from .coordinator import EnergyForecastCoordinator

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA Energy Forecast from a config entry."""
    coordinator = EnergyForecastCoordinator(hass, entry)

    # First refresh — may raise UpdateFailed if no model yet (that's OK,
    # HA will retry; the integration stays loaded and trains in background)
    try:
        await coordinator.async_config_entry_first_refresh()
    except ConfigEntryNotReady:
        # Not enough history yet — still register, will retry on schedule
        _LOGGER.info(
            "HA Energy Forecast: not enough history for initial prediction. "
            "Will retry in %d minutes.",
            coordinator.update_interval.seconds // 60,
        )

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    # Re-run coordinator when options change (e.g. sensor entity updated)
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the integration when options are changed via the UI."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload a config entry."""
    unload_ok = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unload_ok:
        hass.data[DOMAIN].pop(entry.entry_id, None)
    return unload_ok
