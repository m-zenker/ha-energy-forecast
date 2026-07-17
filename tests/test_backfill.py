"""Tests for energy_history_backfill unit conversion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd


class _BackfillStub:
    """Minimal stub with only what _backfill() accesses (self.args, self.get_timezone)."""

    def __init__(self, args: dict, ha_timezone: str | None = None):
        self.args = args
        self._ha_timezone = ha_timezone

    def get_timezone(self) -> str | None:
        return self._ha_timezone


def _run_backfill(
    energy_unit: str,
    cumsum_rows: list[tuple],
    tmp_path,
    timezone_arg: str | None = None,
    ha_timezone: str | None = None,
) -> pd.DataFrame:
    """Run EnergyHistoryBackfill._backfill() with a mocked SQLite DB.

    Args:
        energy_unit: Value for the energy_unit arg (e.g. "kWh", "MWh").
        cumsum_rows: List of (epoch_seconds, cumulative_sum) tuples as the
                     statistics table would return.
        tmp_path:    pytest tmp_path fixture for the CSV output.
        timezone_arg: Value for the optional `timezone` apps.yaml key, or
                      None to omit it.
        ha_timezone:  Value returned by self.get_timezone() (simulates HA's
                      own configured timezone), or None.

    Returns:
        DataFrame read from the written CSV.
    """
    from energy_forecast.energy_history_backfill import EnergyHistoryBackfill

    args = {
        "energy_sensor": "sensor.grid_import",
        "ha_db_path": str(tmp_path / "fake.db"),
        "energy_unit": energy_unit,
    }
    if timezone_arg is not None:
        args["timezone"] = timezone_arg
    stub = _BackfillStub(args=args, ha_timezone=ha_timezone)
    cache_path = tmp_path / "energy_history.csv"

    # Create the fake DB file so Path(db_path).exists() passes
    (tmp_path / "fake.db").touch()

    # Mock sqlite3.connect so no real DB access occurs
    mock_con = MagicMock()
    # First execute() call: PRAGMA table_info → row[1] must contain "start_ts"
    pragma_rows = [(0, "start_ts", "REAL", 0, None, 0)]
    # Second execute() call: statistics query → rows with (epoch, cumsum)
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = cumsum_rows
    mock_con.execute.side_effect = [pragma_rows, mock_cursor]

    with (
        patch("energy_forecast.energy_history_backfill.CACHE_PATH", cache_path),
        patch("sqlite3.connect", return_value=mock_con),
    ):
        EnergyHistoryBackfill._backfill(stub)

    return pd.read_csv(cache_path)


class TestBackfillUnitMultiplier:
    """Backfill applies energy_unit conversion before writing the CSV."""

    def test_kwh_sensor_unchanged(self, tmp_path):
        """Default kWh unit: diff values are stored as-is (no scaling)."""
        # Cumulative kWh: 0 → 1.5 → 3.0 over two hours
        base = 1705312800.0  # 2024-01-15 09:00 UTC
        rows = [(base, 0.0), (base + 3600, 1.5), (base + 7200, 3.0)]
        df = _run_backfill("kWh", rows, tmp_path)
        assert len(df) > 0
        assert all(df["gross_kwh"] <= 50.0)
        assert any(abs(v - 1.5) < 1e-6 for v in df["gross_kwh"])

    def test_mwh_sensor_scaled(self, tmp_path):
        """energy_unit=MWh: diff values multiplied by 1000 before storage."""
        # Cumulative MWh: 0 → 0.0015 → 0.003 over two hours
        base = 1705312800.0  # 2024-01-15 09:00 UTC
        rows = [(base, 0.0), (base + 3600, 0.0015), (base + 7200, 0.003)]
        df = _run_backfill("MWh", rows, tmp_path)
        assert len(df) > 0
        # 0.0015 MWh × 1000 = 1.5 kWh
        assert any(abs(v - 1.5) < 1e-6 for v in df["gross_kwh"])


class TestBackfillZeroConsumptionKept:
    """gross_kwh == 0 (e.g. solar covering 100% of load for an hour) is a real
    reading and must be kept as a row, not silently dropped like a bad reading."""

    def test_zero_diff_hour_kept_not_dropped(self, tmp_path):
        base = 1705312800.0  # 2024-01-15 09:00 UTC
        rows = [
            (base, 0.0),
            (base + 3600, 1.5),  # diff 1.5
            (base + 7200, 1.5),  # diff 0.0 — flat hour, e.g. solar covered the load
            (base + 10800, 3.0),  # diff 1.5
        ]
        df = _run_backfill("kWh", rows, tmp_path)
        # First row's diff is NaN (no prior reading) and is correctly dropped —
        # the other three diffs (1.5, 0.0, 1.5) must all be kept.
        assert len(df) == 3
        assert any(abs(v - 0.0) < 1e-9 for v in df["gross_kwh"])


class TestBackfillTimezone:
    """_backfill() must honour timezone/get_timezone() instead of hardcoding Europe/Zurich."""

    def test_default_falls_back_to_europe_zurich(self, tmp_path):
        # base = 2024-01-15 10:00 UTC; the first row's diff is NaN and is
        # dropped, so the surviving row is timestamped at base+3600
        # (2024-01-15 11:00 UTC) == 2024-01-15 12:00 CET.
        base = 1705312800.0
        rows = [(base, 0.0), (base + 3600, 1.5)]
        df = _run_backfill("kWh", rows, tmp_path, timezone_arg=None, ha_timezone=None)
        assert pd.Timestamp(df["timestamp"].iloc[0]) == pd.Timestamp("2024-01-15 12:00:00")

    def test_explicit_timezone_arg_used(self, tmp_path):
        # Surviving row at base+3600 == 2024-01-15 11:00 UTC.
        base = 1705312800.0
        rows = [(base, 0.0), (base + 3600, 1.5)]
        df = _run_backfill("kWh", rows, tmp_path, timezone_arg="America/New_York", ha_timezone=None)
        # January = EST = UTC-5
        assert pd.Timestamp(df["timestamp"].iloc[0]) == pd.Timestamp("2024-01-15 06:00:00")

    def test_falls_back_to_get_timezone_when_arg_absent(self, tmp_path):
        base = 1705312800.0
        rows = [(base, 0.0), (base + 3600, 1.5)]
        df = _run_backfill("kWh", rows, tmp_path, timezone_arg=None, ha_timezone="America/New_York")
        assert pd.Timestamp(df["timestamp"].iloc[0]) == pd.Timestamp("2024-01-15 06:00:00")

    def test_mismatch_between_arg_and_ha_timezone_logs_warning(self, tmp_path, caplog):
        import logging

        base = 1705312800.0
        rows = [(base, 0.0), (base + 3600, 1.5)]
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            _run_backfill("kWh", rows, tmp_path, timezone_arg="Europe/Zurich", ha_timezone="America/New_York")
        assert any("America/New_York" in r.message for r in caplog.records)
