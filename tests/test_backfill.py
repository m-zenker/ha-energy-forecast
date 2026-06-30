"""Tests for energy_history_backfill unit conversion."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd


class _BackfillStub:
    """Minimal stub with only what _backfill() accesses (self.args)."""

    def __init__(self, args: dict):
        self.args = args


def _run_backfill(energy_unit: str, cumsum_rows: list[tuple], tmp_path) -> pd.DataFrame:
    """Run EnergyHistoryBackfill._backfill() with a mocked SQLite DB.

    Args:
        energy_unit: Value for the energy_unit arg (e.g. "kWh", "MWh").
        cumsum_rows: List of (epoch_seconds, cumulative_sum) tuples as the
                     statistics table would return.
        tmp_path:    pytest tmp_path fixture for the CSV output.

    Returns:
        DataFrame read from the written CSV.
    """
    from energy_forecast.energy_history_backfill import EnergyHistoryBackfill

    stub = _BackfillStub(
        args={
            "energy_sensor": "sensor.grid_import",
            "ha_db_path": str(tmp_path / "fake.db"),
            "energy_unit": energy_unit,
        }
    )
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
