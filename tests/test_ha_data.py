"""Tests for ha_data merge logic and fetch functions.

Covers:
  - _merge_energy_frames: winner selection, empty inputs, NaN dropping, ordering
  - fetch_energy_history: HA-only, cache-only, conflict resolution, error cases
  - fetch_recent_energy:  same merge contract as fetch_energy_history

_fetch_history is patched throughout so tests run without AppDaemon or a live
Home Assistant instance.  Timestamps follow Europe/Zurich (UTC+1 in January,
UTC+2 in summer) — test data uses UTC inputs that map to predictable local hours.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_forecast import ha_data
from energy_forecast.ha_data import _merge_energy_frames


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_energy_df(timestamps: list[str], kwh_values: list[float]) -> pd.DataFrame:
    """Build a naive-timestamp energy DataFrame (as stored in the CSV cache)."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps),
        "gross_kwh": kwh_values,
    })


def make_ha_raw(timestamps_utc: list[str], cumulative_values: list[float]) -> pd.DataFrame:
    """Build a _fetch_history-style DataFrame.

    _fetch_history returns tz-aware Europe/Zurich timestamps with cumulative
    meter readings in the 'value' column.  fetch_energy_history then diffs and
    strips timezone to produce per-hour kWh values.

    January dates: UTC+1, so e.g. 08:00 UTC → 09:00 local, 09:00 UTC → 10:00 local.
    """
    return pd.DataFrame({
        "timestamp": (
            pd.to_datetime(timestamps_utc, utc=True)
            .tz_convert("Europe/Zurich")
        ),
        "value": cumulative_values,
    })


@pytest.fixture
def mock_app() -> MagicMock:
    app = MagicMock()
    app.log = MagicMock()
    return app


# ── _merge_energy_frames ─────────────────────────────────────────────────────

class TestMergeEnergyFrames:

    def test_winner_takes_conflict(self):
        """When winner and loser share a timestamp, winner's value is kept."""
        ts = "2024-01-01 10:00"
        winner = make_energy_df([ts], [2.0])
        loser  = make_energy_df([ts], [1.0])
        result = _merge_energy_frames(winner, loser)
        assert len(result) == 1
        assert result.iloc[0]["gross_kwh"] == pytest.approx(2.0)

    def test_no_conflict_all_rows_kept(self):
        """Non-overlapping timestamps from both frames are all present in output."""
        winner = make_energy_df(["2024-01-01 10:00"], [2.0])
        loser  = make_energy_df(["2024-01-01 09:00"], [1.0])
        result = _merge_energy_frames(winner, loser)
        assert len(result) == 2

    def test_empty_loser_returns_winner(self):
        """Empty loser: only winner rows appear in output."""
        winner = make_energy_df(["2024-01-01 10:00"], [2.0])
        loser  = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        result = _merge_energy_frames(winner, loser)
        assert len(result) == 1
        assert result.iloc[0]["gross_kwh"] == pytest.approx(2.0)

    def test_empty_winner_returns_loser(self):
        """Empty winner: only loser rows appear in output."""
        winner = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        loser  = make_energy_df(["2024-01-01 09:00"], [1.0])
        result = _merge_energy_frames(winner, loser)
        assert len(result) == 1
        assert result.iloc[0]["gross_kwh"] == pytest.approx(1.0)

    def test_nan_gross_kwh_rows_dropped(self):
        """Rows with NaN gross_kwh are dropped from the result."""
        winner = make_energy_df(
            ["2024-01-01 10:00", "2024-01-01 11:00"],
            [2.0, None],
        )
        loser = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        result = _merge_energy_frames(winner, loser)
        assert len(result) == 1
        assert result.iloc[0]["gross_kwh"] == pytest.approx(2.0)

    def test_result_sorted_by_timestamp(self):
        """Output rows are in ascending timestamp order regardless of input order."""
        winner = make_energy_df(["2024-01-01 12:00", "2024-01-01 10:00"], [3.0, 1.0])
        loser  = make_energy_df(["2024-01-01 11:00"], [2.0])
        result = _merge_energy_frames(winner, loser)
        assert list(result["gross_kwh"]) == pytest.approx([1.0, 2.0, 3.0])

    def test_multiple_conflicts_winner_always_wins(self):
        """Winner's value is selected for every conflicting timestamp."""
        timestamps = ["2024-01-01 09:00", "2024-01-01 10:00", "2024-01-01 11:00"]
        winner = make_energy_df(timestamps, [10.0, 20.0, 30.0])
        loser  = make_energy_df(timestamps, [1.0,  2.0,  3.0])
        result = _merge_energy_frames(winner, loser)
        assert len(result) == 3
        assert list(result["gross_kwh"]) == pytest.approx([10.0, 20.0, 30.0])


# ── fetch_energy_history ──────────────────────────────────────────────────────

class TestFetchEnergyHistory:

    def test_ha_only_no_cache(self, mock_app, tmp_path):
        """No cache file: HA data is processed and returned."""
        cache_path = tmp_path / "energy_history.csv"

        # 08:00 UTC → 09:00 local, 09:00 UTC → 10:00 local (January, UTC+1)
        # diff at 10:00 local = 101.0 - 100.0 = 1.0 kWh
        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 101.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        assert len(result) == 1
        assert result.iloc[0]["gross_kwh"] == pytest.approx(1.0)

    def test_cache_only_empty_ha(self, mock_app, tmp_path):
        """HA returns nothing: existing cache is returned."""
        cache_path = tmp_path / "energy_history.csv"
        make_energy_df(["2024-01-01 10:00"], [1.5]).to_csv(cache_path, index=False)

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            result = ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        assert len(result) == 1
        assert result.iloc[0]["gross_kwh"] == pytest.approx(1.5)

    def test_ha_wins_on_conflict(self, mock_app, tmp_path):
        """When cache and HA have the same timestamp, fresh HA data wins."""
        cache_path = tmp_path / "energy_history.csv"
        # Cache has 1.0 kWh at local 10:00
        make_energy_df(["2024-01-01 10:00"], [1.0]).to_csv(cache_path, index=False)

        # HA cumulative: diff at local 10:00 = 102.0 - 100.0 = 2.0 kWh
        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 102.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        row_10 = result[result["timestamp"] == pd.Timestamp("2024-01-01 10:00")]
        assert len(row_10) == 1
        assert row_10.iloc[0]["gross_kwh"] == pytest.approx(2.0)

    def test_cache_row_preserved_when_no_ha_overlap(self, mock_app, tmp_path):
        """Cache rows for timestamps not covered by HA fetch are preserved."""
        cache_path = tmp_path / "energy_history.csv"
        # Cache has a row from 3 days ago
        make_energy_df(["2023-12-29 10:00", "2024-01-01 10:00"], [0.5, 1.0]).to_csv(
            cache_path, index=False
        )

        # HA only covers 2024-01-01 — old cache row should survive
        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 101.5],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        old_row = result[result["timestamp"] == pd.Timestamp("2023-12-29 10:00")]
        assert len(old_row) == 1
        assert old_row.iloc[0]["gross_kwh"] == pytest.approx(0.5)

    def test_both_empty_raises_value_error(self, mock_app, tmp_path):
        """Both empty sources raise ValueError with a descriptive message."""
        cache_path = tmp_path / "energy_history.csv"

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="No history found"):
                ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

    def test_saves_result_to_cache(self, mock_app, tmp_path):
        """Merged result is written back to the cache CSV file."""
        cache_path = tmp_path / "energy_history.csv"

        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 101.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        assert cache_path.exists()
        saved = pd.read_csv(cache_path)
        assert len(saved) == 1

    def test_spikes_filtered_out(self, mock_app, tmp_path):
        """Hourly values >= MAX_HOURLY_KWH are filtered as meter resets/spikes."""
        cache_path = tmp_path / "energy_history.csv"

        # diff at 10:00 = 999.0 kWh — spike, should be filtered
        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 1099.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        assert len(result) == 0


# ── fetch_recent_energy ───────────────────────────────────────────────────────

class TestFetchRecentEnergy:

    def test_ha_wins_on_conflict(self, mock_app, tmp_path):
        """fetch_recent_energy applies the same merge contract: fresh HA wins."""
        cache_path = tmp_path / "energy_history.csv"
        make_energy_df(["2024-01-01 10:00"], [1.0]).to_csv(cache_path, index=False)

        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 102.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        row_10 = result[result["timestamp"] == pd.Timestamp("2024-01-01 10:00")]
        assert len(row_10) == 1
        assert row_10.iloc[0]["gross_kwh"] == pytest.approx(2.0)

    def test_both_empty_raises_value_error(self, mock_app, tmp_path):
        """Both empty sources raise ValueError."""
        cache_path = tmp_path / "energy_history.csv"

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="No history found"):
                ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

    def test_cache_returned_when_ha_empty(self, mock_app, tmp_path):
        """Full cache is returned when HA data is unavailable."""
        cache_path = tmp_path / "energy_history.csv"
        make_energy_df(
            ["2024-01-01 09:00", "2024-01-01 10:00"],
            [1.0, 1.5],
        ).to_csv(cache_path, index=False)

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            result = ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        assert len(result) == 2

    def test_saves_result_to_cache(self, mock_app, tmp_path):
        """Merged result is written to the cache CSV."""
        cache_path = tmp_path / "energy_history.csv"

        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 101.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        assert cache_path.exists()
