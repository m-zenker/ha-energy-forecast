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

    def test_coerces_non_str_ha_timezone_when_not_configured(self):
        """AppDaemon's get_timezone() type hint claims str but the installed
        AppDaemonConfig.time_zone field is validated through pytz.timezone(...),
        so it actually returns a pytz tzinfo object. Passing that straight into
        urllib.parse.quote() (weather.py) raises
        'TypeError: quote_from_bytes() expected bytes' — reported live in
        GitHub Discussion #15. zoneinfo.ZoneInfo stands in for pytz's tzinfo
        here (stdlib, no extra test dependency); both are non-str objects
        whose str() recovers the IANA zone key.
        """
        from zoneinfo import ZoneInfo

        from energy_forecast.const import resolve_timezone

        result = resolve_timezone(None, ZoneInfo("America/New_York"), logging.getLogger("test_const"))
        assert result == "America/New_York"
        assert isinstance(result, str)

    def test_configured_matches_non_str_ha_timezone_no_spurious_warning(self, caplog):
        """Before coercion, `ha_timezone != configured` always compared a
        tzinfo object against a str and was always True (different types),
        even when both represent the same zone -- firing a false "timezone
        mismatch" warning on every startup. Coercing ha_timezone to str
        before the comparison fixes this as a side effect.
        """
        from zoneinfo import ZoneInfo

        from energy_forecast.const import resolve_timezone

        with caplog.at_level(logging.WARNING, logger="test_const"):
            result = resolve_timezone("Europe/Zurich", ZoneInfo("Europe/Zurich"), logging.getLogger("test_const"))
        assert result == "Europe/Zurich"
        assert not caplog.records


class TestWarnOnce:
    def test_first_occurrence_logs_warning(self, caplog):
        from energy_forecast.const import warn_once

        logger = logging.getLogger("energy_forecast.test_warn_once")
        warned: set = set()
        with caplog.at_level(logging.WARNING, logger="energy_forecast.test_warn_once"):
            warn_once(logger, warned, ("k1",), "first %s", "occurrence")
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert ("k1",) in warned

    def test_repeat_occurrence_logs_info_not_warning(self, caplog):
        from energy_forecast.const import warn_once

        logger = logging.getLogger("energy_forecast.test_warn_once")
        warned: set = {("k1",)}
        with caplog.at_level(logging.INFO, logger="energy_forecast.test_warn_once"):
            warn_once(logger, warned, ("k1",), "repeat %s", "occurrence")
        records = [r for r in caplog.records if r.message.startswith("repeat")]
        assert len(records) == 1
        assert records[0].levelno == logging.INFO

    def test_warned_none_disables_dedup_always_warns(self, caplog):
        from energy_forecast.const import warn_once

        logger = logging.getLogger("energy_forecast.test_warn_once")
        with caplog.at_level(logging.WARNING, logger="energy_forecast.test_warn_once"):
            warn_once(logger, None, ("k1",), "msg one")
            warn_once(logger, None, ("k1",), "msg two")
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2
