"""HA Energy Forecast — HACS custom integration."""
from __future__ import annotations

import logging
import os

import homeassistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .const import DOMAIN, PLATFORMS

_LOGGER = logging.getLogger(__name__)

# HA's own constraints file — passed to pip to avoid version conflicts.
_CONSTRAINTS_FILE = os.path.join(
    os.path.dirname(homeassistant.__file__), "package_constraints.txt"
)


def _install_ml_packages() -> tuple[bool, str]:
    """Install ML packages using HA's own package installer (synchronous).

    Must be called from an executor thread (never the event loop).

    lightgbm and scikit-learn are NOT listed in manifest.json requirements
    because HA's startup requirements resolver runs before any integration
    code executes — a pip failure there throws RequirementsNotFound and
    prevents the integration from loading entirely, with no recovery path.
    By installing here instead we:
      - Use the same homeassistant.util.package.install_package() that HA
        itself uses, so the constraints file and dep target are respected.
      - Degrade gracefully: sensors show unavailable rather than crashing HA.
      - Survive platforms (e.g. armv7l) where lightgbm has no pre-built wheel
        by falling back to scikit-learn GBR automatically.

    Returns (sklearn_available, reason_if_not).
    """
    from homeassistant.util.package import install_package  # noqa: PLC0415

    constraints: str | None = (
        _CONSTRAINTS_FILE if os.path.isfile(_CONSTRAINTS_FILE) else None
    )

    # lightgbm — optional; fall back to sklearn GBR silently if unavailable.
    try:
        lgbm_ok = install_package(
            "lightgbm>=4.0.0",
            upgrade=False,
            constraints=constraints,
        )
        if not lgbm_ok:
            _LOGGER.info(
                "lightgbm could not be installed — will use scikit-learn GBR "
                "(predictions still work, just slightly slower to train)"
            )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.info(
            "lightgbm not available on this platform (%s) — "
            "will use scikit-learn GBR", exc,
        )

    # scikit-learn — required; integration runs but won't train without it.
    try:
        sklearn_ok = install_package(
            "scikit-learn>=1.3.0",
            upgrade=False,
            constraints=constraints,
        )
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error(
            "scikit-learn installation failed: %s. "
            "Sensors will remain unavailable until this is resolved.", exc,
        )
        return False, str(exc)

    if not sklearn_ok:
        msg = (
            "install_package returned False for scikit-learn. "
            "Check HA logs for pip errors."
        )
        _LOGGER.error(msg)
        return False, msg

    _LOGGER.debug("ML packages ready")
    return True, "ok"


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up HA Energy Forecast from a config entry."""
    # Install ML packages via HA's internal package installer.
    # We log but do NOT raise on failure so the integration still loads —
    # sensors will show unavailable and training will be skipped until
    # the packages become available (usually after a restart).
    await hass.async_add_executor_job(_install_ml_packages)

    # Coordinator import is deferred until here so the config flow can load
    # cleanly even if ML packages aren't yet installed.
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
