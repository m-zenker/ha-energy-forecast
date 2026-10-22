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


class TestDropDstDuplicates:
    """DST fall-back maps 02:00 CEST and 02:00 CET onto one naive 02:00 label."""

    def test_fallback_utc_frame_collapses_to_unique(self):
        import pandas as pd
        from energy_forecast.const import drop_dst_duplicates, strip_tz

        utc = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-10-24 22:00", periods=4, freq="h", tz="UTC"),
                "temp_c": [1.0, 2.0, 3.0, 4.0],
            }
        )
        out = drop_dst_duplicates(strip_tz(utc))
        assert out["timestamp"].is_unique
        assert list(out["timestamp"].dt.hour) == [0, 1, 2]
        assert out.loc[out["timestamp"].dt.hour == 2, "temp_c"].item() == 3.0  # first (CEST) kept
        assert list(out.index) == [0, 1, 2]

    def test_spring_forward_is_noop(self):
        import pandas as pd
        from energy_forecast.const import drop_dst_duplicates, strip_tz

        utc = pd.DataFrame({"timestamp": pd.date_range("2026-03-29 00:00", periods=4, freq="h", tz="UTC"), "v": 1.0})
        df = strip_tz(utc)
        pd.testing.assert_frame_equal(drop_dst_duplicates(df), df)

    def test_empty_and_missing_column_passthrough(self):
        import pandas as pd
        from energy_forecast.const import drop_dst_duplicates

        assert drop_dst_duplicates(pd.DataFrame()).empty
        df = pd.DataFrame({"x": [1, 1]})
        pd.testing.assert_frame_equal(drop_dst_duplicates(df), df)
