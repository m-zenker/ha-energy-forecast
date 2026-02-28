"""HA Energy Forecast — HACS custom integration."""
from __future__ import annotations

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, PLATFORMS

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA Energy Forecast from a config entry."""
    # Import coordinator here, not at module level, so that the config flow
    # can load cleanly even before manifest requirements are installed.
    from .coordinator import EnergyForecastCoordinator  # noqa: PLC0415

    coordinator = EnergyForecastCoordinator(hass, entry)

    try:
        await coordinator.async_refresh()
    except Exception as exc:  # noqa: BLE001
        _LOGGER.info(
            "HA Energy Forecast: initial refresh did not complete (%s). "
            "Sensors will be unavailable until the model is trained.", exc,
        )

    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
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
