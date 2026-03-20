"""Tests for ha_data merge logic and fetch functions.

Covers:
  - _merge_energy_frames: winner selection, empty inputs, NaN dropping, ordering
  - fetch_energy_history: HA-only, cache-only, conflict resolution, error cases
  - fetch_recent_energy:  same merge contract as fetch_energy_history
  - _check_dst_duplicates: DST fall-back duplicate detection, spring-forward gap

_fetch_history is patched throughout so tests run without AppDaemon or a live
Home Assistant instance.  Timestamps follow Europe/Zurich (UTC+1 in January,
UTC+2 in summer) — test data uses UTC inputs that map to predictable local hours.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_forecast import ha_data
from energy_forecast.ha_data import _check_dst_duplicates, _merge_energy_frames


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


# ── _check_dst_duplicates ─────────────────────────────────────────────────────

class TestCheckDstDuplicates:
    """Fix 5.1 — DST fall-back produces duplicate naive timestamps.

    Europe/Zurich falls back on the last Sunday of October: at 03:00 CEST the
    clock jumps back to 02:00 CET.  After tz_localize(None) the naive timestamps
    02:00 and 02:59 appear twice — once in summer time, once in winter time.

    Spring-forward (last Sunday of March) creates a gap: 02:00–02:59 never
    exist.  The resample/ffill in fetch functions fills this gap silently; this
    is documented accepted behaviour and does NOT trigger a warning.
    """

    # ── fall-back (autumn DST): duplicate naive timestamps ────────────────────

    def test_no_duplicates_no_warning(self, caplog):
        """Clean data — no WARNING is emitted."""
        df = make_energy_df(
            ["2024-10-27 01:00", "2024-10-27 03:00", "2024-10-27 04:00"],
            [1.0, 1.0, 1.0],
        )
        with caplog.at_level(logging.WARNING, logger="energy_forecast.ha_data"):
            _check_dst_duplicates(df, _LOGGER)
        assert not any("DST" in r.message or "duplicate" in r.message.lower() for r in caplog.records)

    def test_duplicate_timestamps_emits_warning(self, caplog):
        """Duplicate naive 02:00 (fall-back) triggers a WARNING."""
        # Both rows have the naive timestamp 02:00; one was CEST, one CET
        df = make_energy_df(
            ["2024-10-27 02:00", "2024-10-27 02:00", "2024-10-27 03:00"],
            [1.0, 1.1, 1.0],
        )
        with caplog.at_level(logging.WARNING, logger="energy_forecast.ha_data"):
            _check_dst_duplicates(df, _LOGGER)
        assert any("duplicate" in r.message.lower() or "DST" in r.message for r in caplog.records)

    def test_duplicate_count_mentioned_in_warning(self, caplog):
        """Warning message includes the count of duplicated timestamps."""
        df = make_energy_df(
            ["2024-10-27 02:00", "2024-10-27 02:00"],
            [1.0, 1.1],
        )
        with caplog.at_level(logging.WARNING, logger="energy_forecast.ha_data"):
            _check_dst_duplicates(df, _LOGGER)
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_texts, "expected at least one WARNING"
        assert any("1" in t for t in warning_texts), "expected duplicate count in message"

    def test_empty_dataframe_no_warning(self, caplog):
        """Empty DataFrame does not raise and emits no warning."""
        df = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        with caplog.at_level(logging.WARNING, logger="energy_forecast.ha_data"):
            _check_dst_duplicates(df, _LOGGER)
        assert not caplog.records

    def test_single_row_no_warning(self, caplog):
        """Single-row DataFrame cannot have duplicates."""
        df = make_energy_df(["2024-10-27 02:00"], [1.0])
        with caplog.at_level(logging.WARNING, logger="energy_forecast.ha_data"):
            _check_dst_duplicates(df, _LOGGER)
        assert not caplog.records

    # ── spring-forward (gap): accepted, no warning ────────────────────────────

    def test_spring_forward_gap_no_warning(self, caplog):
        """Spring-forward gap (02:00–02:59 missing) is accepted — no WARNING."""
        # 2024 spring-forward: 31 March 02:00 CEST → 03:00; 02:xx never exist
        df = make_energy_df(
            ["2024-03-31 01:00", "2024-03-31 03:00", "2024-03-31 04:00"],
            [1.0, 1.0, 1.0],
        )
        with caplog.at_level(logging.WARNING, logger="energy_forecast.ha_data"):
            _check_dst_duplicates(df, _LOGGER)
        assert not caplog.records

    # ── integration: warning fires after fetch when fall-back data present ────

    def test_fetch_energy_history_warns_on_dst_duplicates(self, mock_app, tmp_path, caplog):
        """fetch_energy_history emits a DST WARNING when merged data has duplicates."""
        cache_path = tmp_path / "energy_history.csv"

        # Seed the cache with the first 02:00 occurrence (CEST naive)
        make_energy_df(["2024-10-27 02:00"], [1.0]).to_csv(cache_path, index=False)

        # HA returns the second 02:00 occurrence (CET naive) — different value
        # We inject it via df_new directly by making HA raw return something that
        # after diff/processing yields a row at naive 02:00 with value 1.1.
        # Simplest: patch _merge_energy_frames to return a frame with duplicates,
        # so we specifically test that fetch_energy_history calls _check_dst_duplicates.
        dup_df = make_energy_df(
            ["2024-10-27 01:00", "2024-10-27 02:00", "2024-10-27 02:00"],
            [1.0, 1.0, 1.1],
        )
        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            with patch.object(ha_data, "_merge_energy_frames", return_value=dup_df):
                with caplog.at_level(logging.WARNING, logger="energy_forecast.ha_data"):
                    ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        assert any("duplicate" in r.message.lower() or "DST" in r.message for r in caplog.records)


# Module-level logger used directly in DST tests (mirrors ha_data's own logger)
import logging as _logging
_LOGGER = _logging.getLogger("energy_forecast.ha_data")


# ── split_ev_charging ─────────────────────────────────────────────────────────

class TestSplitEvCharging:

    def _make_df(self) -> pd.DataFrame:
        """Four rows: two below threshold (3 kWh), two above (10 kWh)."""
        return pd.DataFrame({
            "timestamp": pd.date_range("2026-03-12 00:00", periods=4, freq="1h"),
            "gross_kwh": [3.0, 3.0, 10.0, 10.0],
        })

    def test_custom_charger_kw_subtracted(self):
        """charger_kw=7.4 is subtracted from charging hours, not the default 9.0."""
        df = self._make_df()
        baseline, ev = ha_data.split_ev_charging(df, threshold_kwh=4.5, charger_kw=7.4)
        # EV hours: 10.0 - 7.4 = 2.6
        assert abs(baseline.iloc[2]["gross_kwh"] - 2.6) < 1e-6
        assert abs(baseline.iloc[3]["gross_kwh"] - 2.6) < 1e-6

    def test_default_charger_kw_is_nine(self):
        """Default charger_kw=9.0 subtracts 9 from charging hours."""
        df = self._make_df()
        baseline, ev = ha_data.split_ev_charging(df, threshold_kwh=4.5)
        # EV hours: 10.0 - 9.0 = 1.0
        assert abs(baseline.iloc[2]["gross_kwh"] - 1.0) < 1e-6
        # Non-EV hours are unchanged
        assert abs(baseline.iloc[0]["gross_kwh"] - 3.0) < 1e-6
        assert len(ev) == 2


# ── fetch_sub_sensor_history / fetch_recent_sub_sensor ────────────────────────

class TestFetchSubSensorHistory:
    """Tests for fetch_sub_sensor_history and fetch_recent_sub_sensor.

    Both functions track cumulative kWh sub-sensors (heat pump, dishwasher, etc.)
    and differ from the main energy fetch in two ways:
    - Column name is 'kwh' (not 'gross_kwh')
    - Zero-kWh hours (appliance off) are kept so lag features return 0, not NaN
    """

    def test_returns_kwh_column(self, mock_app, tmp_path):
        """Result DataFrame has 'timestamp' and 'kwh' columns (not gross_kwh)."""
        cache_path = tmp_path / "sub_heat_pump.csv"
        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 101.5],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_sub_sensor_history(mock_app, "sensor.heat_pump_kwh", cache_path)

        assert "kwh" in result.columns
        assert "gross_kwh" not in result.columns
        assert "timestamp" in result.columns

    def test_falls_back_to_cache_when_ha_empty(self, mock_app, tmp_path):
        """When HA returns nothing, the existing cache is returned."""
        cache_path = tmp_path / "sub_heat_pump.csv"
        pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 10:00"]),
            "kwh": [2.5],
        }).to_csv(cache_path, index=False)

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            result = ha_data.fetch_sub_sensor_history(mock_app, "sensor.heat_pump_kwh", cache_path)

        assert len(result) == 1
        assert result.iloc[0]["kwh"] == pytest.approx(2.5)

    def test_spike_filter_applied(self, mock_app, tmp_path):
        """Hours with diff >= MAX_HOURLY_KWH are filtered as meter resets/spikes."""
        from energy_forecast.const import MAX_HOURLY_KWH
        cache_path = tmp_path / "sub_heat_pump.csv"
        # diff = 999 kWh — well above MAX_HOURLY_KWH, should be filtered
        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 100.0 + MAX_HOURLY_KWH + 10],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_sub_sensor_history(mock_app, "sensor.heat_pump_kwh", cache_path)

        assert len(result) == 0

    def test_zero_kwh_hours_kept(self, mock_app, tmp_path):
        """Zero-kWh diff hours (appliance off) are retained, unlike the main sensor."""
        cache_path = tmp_path / "sub_heat_pump.csv"
        # diff at 10:00 local = 100.0 - 100.0 = 0.0 kWh (appliance off)
        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 100.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_sub_sensor_history(mock_app, "sensor.heat_pump_kwh", cache_path)

        assert len(result) == 1
        assert result.iloc[0]["kwh"] == pytest.approx(0.0)

    def test_ha_wins_on_conflict(self, mock_app, tmp_path):
        """Fresh HA data overwrites cached value for the same timestamp."""
        cache_path = tmp_path / "sub_heat_pump.csv"
        pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 10:00"]),
            "kwh": [1.0],
        }).to_csv(cache_path, index=False)

        # HA cumulative: diff at local 10:00 = 102.0 - 100.0 = 2.0 kWh
        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 102.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_sub_sensor_history(mock_app, "sensor.heat_pump_kwh", cache_path)

        row_10 = result[result["timestamp"] == pd.Timestamp("2024-01-01 10:00")]
        assert len(row_10) == 1
        assert row_10.iloc[0]["kwh"] == pytest.approx(2.0)

    def test_recent_sub_sensor_saves_cache(self, mock_app, tmp_path):
        """fetch_recent_sub_sensor merges and saves result to cache."""
        cache_path = tmp_path / "sub_dishwasher.csv"
        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [50.0, 50.5],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_recent_sub_sensor(mock_app, "sensor.dishwasher_kwh", cache_path)

        assert cache_path.exists()
        assert "kwh" in result.columns
        assert len(result) == 1

    def test_both_empty_returns_empty_with_warning(self, mock_app, tmp_path):
        """When both HA and cache are empty, returns empty DataFrame and logs WARNING."""
        cache_path = tmp_path / "sub_heat_pump.csv"
        # No cache file, HA returns empty DataFrame
        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            result = ha_data.fetch_sub_sensor_history(mock_app, "sensor.heat_pump_kwh", cache_path)

        assert result.empty
        assert list(result.columns) == ["timestamp", "kwh"]
        mock_app.log.assert_called_once()
        args, kwargs = mock_app.log.call_args
        assert kwargs.get("level") == "WARNING"


# ── Stage 6 — CSV append-only writes (#19) ────────────────────────────────────

class TestCsvAppendOnlyWrites:
    """fetch_recent_energy must only append new timestamps, not rewrite the whole CSV."""

    def test_append_does_not_duplicate_existing_rows(self, mock_app, tmp_path):
        """Rows already in the CSV cache must not appear twice after fetch_recent_energy."""
        cache_path = tmp_path / "energy_history.csv"
        # Pre-populate cache with one row
        existing = make_energy_df(["2024-01-01 09:00"], [1.0])
        existing.to_csv(cache_path, index=False)

        # HA returns the same timestamp with slightly different value (edge-case)
        ha_raw = make_ha_raw(
            ["2024-01-01T07:00:00Z", "2024-01-01T08:00:00Z"],
            [100.0, 101.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        saved = pd.read_csv(cache_path)
        # No duplicate timestamps in the saved CSV
        assert saved["timestamp"].duplicated().sum() == 0

    def test_new_rows_appended_to_csv(self, mock_app, tmp_path):
        """Genuinely new rows from HA must appear in the CSV after fetch_recent_energy."""
        cache_path = tmp_path / "energy_history.csv"
        # Cache has one row at 09:00; HA brings a new row at 10:00
        make_energy_df(["2024-01-01 09:00"], [0.8]).to_csv(cache_path, index=False)

        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 101.5],    # diff at 10:00 local = 1.5 kWh
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        saved = pd.read_csv(cache_path)
        saved_ts = pd.to_datetime(saved["timestamp"])
        # New row at 10:00 local (09:00 UTC + 1h) must be in the CSV
        assert pd.Timestamp("2024-01-01 10:00") in saved_ts.values

    def test_csv_created_when_not_exists(self, mock_app, tmp_path):
        """When no cache file exists, fetch_recent_energy creates it on first write."""
        cache_path = tmp_path / "energy_history.csv"
        assert not cache_path.exists()

        ha_raw = make_ha_raw(
            ["2024-01-01T08:00:00Z", "2024-01-01T09:00:00Z"],
            [100.0, 101.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        assert cache_path.exists()
        saved = pd.read_csv(cache_path)
        assert len(saved) >= 1

    def test_fetch_recent_energy_mixed_timestamp_format(self, mock_app, tmp_path):
        """Regression: CSV with mixed-format timestamps (datetime + date-only) must parse cleanly.

        Reproduces the pandas 3.x failure where a date-only midnight entry
        ("2026-03-20") caused a ValueError because format was inferred as
        "%Y-%m-%d %H:%M:%S" from the first row.  With format="mixed" all rows
        parse successfully and fetch_recent_energy returns a non-empty result.
        """
        cache_path = tmp_path / "energy_history.csv"
        # Write a CSV with mixed formats: one datetime string, one date-only string
        cache_path.write_text(
            "timestamp,gross_kwh\n"
            "2026-03-20 01:00:00,1.2\n"
            "2026-03-20,0.9\n"
        )

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            result = ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        assert not result.empty, "Expected non-empty result with mixed-format timestamps"
        assert len(result) == 2

    def test_fetch_energy_history_compacts_and_deduplicates(self, mock_app, tmp_path):
        """fetch_energy_history must write a sorted, deduped CSV (compaction)."""
        cache_path = tmp_path / "energy_history.csv"
        # Pre-populate with out-of-order rows and a duplicate
        make_energy_df(
            ["2024-01-01 11:00", "2024-01-01 09:00", "2024-01-01 10:00", "2024-01-01 10:00"],
            [1.0, 0.5, 0.8, 0.8],
        ).to_csv(cache_path, index=False)

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        saved = pd.read_csv(cache_path)
        saved_ts = pd.to_datetime(saved["timestamp"])
        # Sorted ascending
        assert list(saved_ts) == sorted(saved_ts)
        # No duplicates
        assert saved_ts.duplicated().sum() == 0
