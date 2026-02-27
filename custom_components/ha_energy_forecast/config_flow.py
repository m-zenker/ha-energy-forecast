"""Config flow for HA Energy Forecast integration."""
from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.selector import (
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_ENERGY_SENSOR,
    CONF_LATITUDE,
    CONF_LONGITUDE,
    CONF_OUTDOOR_TEMP_SENSOR,
    CONF_PLZ,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


def _base_schema(
    defaults: dict[str, Any] | None = None,
    ha_lat: float = 47.3769,
    ha_lon: float = 8.5417,
) -> vol.Schema:
    """Build the shared schema for setup and options flows."""
    d = defaults or {}

    return vol.Schema(
        {
            vol.Required(
                CONF_ENERGY_SENSOR,
                default=d.get(CONF_ENERGY_SENSOR, ""),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain="sensor",
                    device_class="energy",
                    multiple=False,
                )
            ),
            vol.Optional(
                CONF_OUTDOOR_TEMP_SENSOR,
                default=d.get(CONF_OUTDOOR_TEMP_SENSOR, ""),
            ): EntitySelector(
                EntitySelectorConfig(
                    domain="sensor",
                    device_class="temperature",
                    multiple=False,
                )
            ),
            vol.Required(
                CONF_PLZ,
                default=d.get(CONF_PLZ, "8001"),
            ): TextSelector(
                TextSelectorConfig(type=TextSelectorType.TEXT)
            ),
            vol.Required(
                CONF_LATITUDE,
                default=d.get(CONF_LATITUDE, ha_lat),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=-90, max=90, step=0.0001, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_LONGITUDE,
                default=d.get(CONF_LONGITUDE, ha_lon),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=-180, max=180, step=0.0001, mode=NumberSelectorMode.BOX
                )
            ),
        }
    )


async def _validate_energy_sensor(hass: HomeAssistant, entity_id: str) -> str | None:
    """Return an error key if the energy sensor looks wrong, else None."""
    if not entity_id:
        return "sensor_not_found"
    state = hass.states.get(entity_id)
    if state is None:
        return "sensor_not_found"
    try:
        float(state.state)
    except (ValueError, TypeError):
        return "sensor_not_numeric"
    return None


class HAEnergyForecastConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle the initial setup flow."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        errors: dict[str, str] = {}

        if user_input is not None:
            error = await _validate_energy_sensor(
                self.hass, user_input.get(CONF_ENERGY_SENSOR, "")
            )
            if error:
                errors[CONF_ENERGY_SENSOR] = error
            else:
                # Strip empty outdoor sensor so it doesn't pollute config
                if not user_input.get(CONF_OUTDOOR_TEMP_SENSOR):
                    user_input.pop(CONF_OUTDOOR_TEMP_SENSOR, None)

                await self.async_set_unique_id(
                    f"{DOMAIN}_{user_input[CONF_ENERGY_SENSOR]}"
                )
                self._abort_if_unique_id_configured()

                return self.async_create_entry(
                    title=f"Energy Forecast ({user_input.get(CONF_PLZ, '')})",
                    data=user_input,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=_base_schema(
                ha_lat=self.hass.config.latitude,
                ha_lon=self.hass.config.longitude,
            ),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> HAEnergyForecastOptionsFlow:
        return HAEnergyForecastOptionsFlow(config_entry)


class HAEnergyForecastOptionsFlow(config_entries.OptionsFlow):
    """Handle reconfiguration (Settings → Integrations → Configure)."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self._config_entry = config_entry

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        errors: dict[str, str] = {}
        current = {**self._config_entry.data, **self._config_entry.options}

        if user_input is not None:
            error = await _validate_energy_sensor(
                self.hass, user_input.get(CONF_ENERGY_SENSOR, "")
            )
            if error:
                errors[CONF_ENERGY_SENSOR] = error
            else:
                if not user_input.get(CONF_OUTDOOR_TEMP_SENSOR):
                    user_input.pop(CONF_OUTDOOR_TEMP_SENSOR, None)
                return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=_base_schema(
                defaults=current,
                ha_lat=self.hass.config.latitude,
                ha_lon=self.hass.config.longitude,
            ),
            errors=errors,
        )
