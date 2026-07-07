"""Tests for shared const.py helpers."""

from __future__ import annotations

import logging


class TestResolveTimezone:
    """resolve_timezone(): explicit config wins, warns on mismatch, falls back to HA then Zurich."""

    def test_uses_configured_value_when_no_ha_timezone(self):
        from energy_forecast.const import resolve_timezone

        result = resolve_timezone("America/New_York", None, logging.getLogger("test_const"))
        assert result == "America/New_York"

    def test_falls_back_to_ha_timezone_when_not_configured(self):
        from energy_forecast.const import resolve_timezone

        result = resolve_timezone(None, "America/New_York", logging.getLogger("test_const"))
        assert result == "America/New_York"

    def test_falls_back_to_zurich_when_neither_set(self):
        from energy_forecast.const import resolve_timezone

        result = resolve_timezone(None, None, logging.getLogger("test_const"))
        assert result == "Europe/Zurich"

    def test_configured_matches_ha_timezone_no_warning(self, caplog):
        from energy_forecast.const import resolve_timezone

        with caplog.at_level(logging.WARNING, logger="test_const"):
            result = resolve_timezone("Europe/Zurich", "Europe/Zurich", logging.getLogger("test_const"))
        assert result == "Europe/Zurich"
        assert not caplog.records

    def test_configured_mismatch_warns_but_keeps_configured_value(self, caplog):
        from energy_forecast.const import resolve_timezone

        with caplog.at_level(logging.WARNING, logger="test_const"):
            result = resolve_timezone("Europe/Zurich", "America/New_York", logging.getLogger("test_const"))
        assert result == "Europe/Zurich"
        assert any("America/New_York" in r.message for r in caplog.records)
        assert any("Europe/Zurich" in r.message for r in caplog.records)
