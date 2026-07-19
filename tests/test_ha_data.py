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
from energy_forecast.ha_data import _check_dst_duplicates, _fetch_history, _merge_energy_frames, _merge_frames

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_climate_ha_raw(timestamps_utc: list[str], current_temps: list[float], setpoints: list[float]) -> pd.DataFrame:
    """Build a climate-style raw HA history DataFrame with attributes."""
    rows = []
    for ts, ct, sp in zip(timestamps_utc, current_temps, setpoints):
        rows.append(
            {
                "timestamp": pd.to_datetime(ts, utc=True).tz_convert("Europe/Zurich"),
                "current_temperature": ct,
                "temperature": sp,
            }
        )
    return pd.DataFrame(rows)


def make_generic_ha_raw(timestamps_utc: list[str], values: list[float]) -> pd.DataFrame:
    """Build a generic absolute sensor raw HA history DataFrame."""
    return pd.DataFrame(
        {"timestamp": pd.to_datetime(timestamps_utc, utc=True).tz_convert("Europe/Zurich"), "value": values}
    )


def make_energy_df(timestamps: list[str], kwh_values: list[float]) -> pd.DataFrame:
    """Build a naive-timestamp energy DataFrame (as stored in the CSV cache)."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps),
            "gross_kwh": kwh_values,
        }
    )


def make_ha_raw(timestamps_utc: list[str], cumulative_values: list[float]) -> pd.DataFrame:
    """Build a _fetch_history-style DataFrame.

    _fetch_history returns tz-aware Europe/Zurich timestamps with cumulative
    meter readings in the 'value' column.  fetch_energy_history then diffs and
    strips timezone to produce per-hour kWh values.

    January dates: UTC+1, so e.g. 08:00 UTC → 09:00 local, 09:00 UTC → 10:00 local.
    """
    return pd.DataFrame(
        {
            "timestamp": (pd.to_datetime(timestamps_utc, utc=True).tz_convert("Europe/Zurich")),
            "value": cumulative_values,
        }
    )


@pytest.fixture
def mock_app() -> MagicMock:
    app = MagicMock()
    app.log = MagicMock()
    return app


# ── Stage 2: Climate & Generic ───────────────────────────────────────────────


class TestClimateAndGenericHistory:
    def test_merge_frames_generic(self):
        """_merge_frames works correctly with any value column."""
        ts = "2024-01-01 10:00"
        winner = pd.DataFrame({"timestamp": [pd.to_datetime(ts)], "val": [22.0]})
        loser = pd.DataFrame({"timestamp": [pd.to_datetime(ts)], "val": [21.0]})
        result = _merge_frames(winner, loser, "val")
        assert result.iloc[0]["val"] == 22.0

    @patch("energy_forecast.ha_data._fetch_history")
    def test_fetch_climate_history_basic(self, mock_fetch, mock_app, tmp_path):
        """fetch_climate_history correctly extracts and resamples attributes."""
        # 10:00 UTC -> 11:00 CET
        ha_raw = make_climate_ha_raw(["2024-01-01 10:00", "2024-01-01 10:30"], [20.0, 20.5], [21.0, 21.0])
        mock_fetch.return_value = ha_raw
        cache_file = tmp_path / "climate_test.csv"

        df = ha_data.fetch_climate_history(mock_app, "climate.living_room", cache_file)

        assert not df.empty
        assert "current_temp" in df.columns
        assert "setpoint" in df.columns
        # Should be resampled to 11:00 local (last state in the 10:xx UTC window)
        assert df.iloc[0]["timestamp"] == pd.to_datetime("2024-01-01 11:00")
        assert df.iloc[0]["current_temp"] == 20.5
        assert df.iloc[0]["setpoint"] == 21.0
        assert cache_file.exists()

    @patch("energy_forecast.ha_data._fetch_history")
    def test_fetch_generic_sensor_history(self, mock_fetch, mock_app, tmp_path):
        """fetch_generic_sensor_history handles absolute values correctly."""
        ha_raw = make_generic_ha_raw(["2024-01-01 10:00", "2024-01-01 10:30"], [55.0, 54.5])
        mock_fetch.return_value = ha_raw
        cache_file = tmp_path / "sensor_test.csv"

        df = ha_data.fetch_generic_sensor_history(mock_app, "sensor.dhw_temp", cache_file, column_name="buffer_temp")

        assert not df.empty
        assert "buffer_temp" in df.columns
        assert df.iloc[0]["buffer_temp"] == 54.5
        assert cache_file.exists()

    @patch("energy_forecast.ha_data._fetch_history")
    def test_fetch_recent_generic_sensor_warns_by_default_on_empty(self, mock_fetch, mock_app, tmp_path, caplog):
        """No cache, no HA data: defaults to a WARNING (unchanged prior behaviour)."""
        mock_fetch.return_value = pd.DataFrame(columns=["timestamp", "value"])
        cache_file = tmp_path / "empty_sensor.csv"

        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            df = ha_data.fetch_recent_generic_sensor(mock_app, "sensor.kermi_cop", cache_file, column_name="cop")

        assert df.empty
        assert any("No recent data for sensor sensor.kermi_cop" in r.message for r in caplog.records)
        assert all(r.levelno == logging.WARNING for r in caplog.records if "No recent data" in r.message)

    @patch("energy_forecast.ha_data._fetch_history")
    def test_fetch_recent_generic_sensor_quiet_if_empty_logs_debug_not_warning(
        self, mock_fetch, mock_app, tmp_path, caplog
    ):
        """quiet_if_empty=True: same empty result, but logged at DEBUG — for sensors with
        a known, expected idle period (e.g. a heat pump's live COP sensor, only reporting
        while actively heating) so an hourly empty result doesn't read as an error."""
        mock_fetch.return_value = pd.DataFrame(columns=["timestamp", "value"])
        cache_file = tmp_path / "empty_sensor.csv"

        with caplog.at_level(logging.DEBUG, logger="energy_forecast"):
            df = ha_data.fetch_recent_generic_sensor(
                mock_app, "sensor.kermi_cop", cache_file, column_name="cop", quiet_if_empty=True
            )

        assert df.empty
        no_data_records = [r for r in caplog.records if "No recent data for sensor sensor.kermi_cop" in r.message]
        assert no_data_records, "expected the message to still be logged, just at DEBUG"
        assert all(r.levelno == logging.DEBUG for r in no_data_records)
        assert not any(r.levelno == logging.WARNING for r in no_data_records)

    def test_fetch_history_boolean_states(self, mock_app):
        """_fetch_history maps 'on'→1.0 and 'off'→0.0 for input_boolean entities."""
        states = [
            {"last_updated": "2024-01-01T10:00:00+01:00", "state": "on"},
            {"last_updated": "2024-01-01T11:00:00+01:00", "state": "off"},
            {"last_updated": "2024-01-01T12:00:00+01:00", "state": "unavailable"},
        ]
        mock_app.get_history.return_value = [states]

        df = _fetch_history(mock_app, "input_boolean.heizung_wintermodus", days=30)

        assert len(df) == 2, "unavailable state must be skipped"
        assert df.iloc[0]["value"] == pytest.approx(1.0)
        assert df.iloc[1]["value"] == pytest.approx(0.0)

    def test_fetch_history_numeric_states_unchanged(self, mock_app):
        """_fetch_history still parses numeric states correctly."""
        states = [
            {"last_updated": "2024-01-01T10:00:00+01:00", "state": "55.3"},
            {"last_updated": "2024-01-01T11:00:00+01:00", "state": "54.1"},
        ]
        mock_app.get_history.return_value = [states]

        df = _fetch_history(mock_app, "sensor.dhw_temp", days=30)

        assert len(df) == 2
        assert df.iloc[0]["value"] == pytest.approx(55.3)
        assert df.iloc[1]["value"] == pytest.approx(54.1)


# ── _merge_energy_frames ─────────────────────────────────────────────────────


class TestMergeEnergyFrames:
    def test_winner_takes_conflict(self):
        """When winner and loser share a timestamp, winner's value is kept."""
        ts = "2024-01-01 10:00"
        winner = make_energy_df([ts], [2.0])
        loser = make_energy_df([ts], [1.0])
        result = _merge_energy_frames(winner, loser)
        assert len(result) == 1
        assert result.iloc[0]["gross_kwh"] == pytest.approx(2.0)

    def test_no_conflict_all_rows_kept(self):
        """Non-overlapping timestamps from both frames are all present in output."""
        winner = make_energy_df(["2024-01-01 10:00"], [2.0])
        loser = make_energy_df(["2024-01-01 09:00"], [1.0])
        result = _merge_energy_frames(winner, loser)
        assert len(result) == 2

    def test_empty_loser_returns_winner(self):
        """Empty loser: only winner rows appear in output."""
        winner = make_energy_df(["2024-01-01 10:00"], [2.0])
        loser = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        result = _merge_energy_frames(winner, loser)
        assert len(result) == 1
        assert result.iloc[0]["gross_kwh"] == pytest.approx(2.0)

    def test_empty_winner_returns_loser(self):
        """Empty winner: only loser rows appear in output."""
        winner = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        loser = make_energy_df(["2024-01-01 09:00"], [1.0])
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
        loser = make_energy_df(["2024-01-01 11:00"], [2.0])
        result = _merge_energy_frames(winner, loser)
        assert list(result["gross_kwh"]) == pytest.approx([1.0, 2.0, 3.0])

    def test_multiple_conflicts_winner_always_wins(self):
        """Winner's value is selected for every conflicting timestamp."""
        timestamps = ["2024-01-01 09:00", "2024-01-01 10:00", "2024-01-01 11:00"]
        winner = make_energy_df(timestamps, [10.0, 20.0, 30.0])
        loser = make_energy_df(timestamps, [1.0, 2.0, 3.0])
        result = _merge_energy_frames(winner, loser)
        assert len(result) == 3
        assert list(result["gross_kwh"]) == pytest.approx([10.0, 20.0, 30.0])

    def test_empty_winner_preserves_float_dtype(self):
        """Regression: pandas 3.x promotes concat(float64, empty_object) to object.

        When raw_ha is empty, df_new = pd.DataFrame(columns=[...]) has object dtype.
        After concat with the float64 cache, gross_kwh must still be float64, not object.
        Otherwise downstream lag features become object and LightGBM raises
        'pandas dtypes must be int, float or bool'.
        """
        import numpy as np

        winner = pd.DataFrame(columns=["timestamp", "gross_kwh"])  # object dtype (empty)
        loser = make_energy_df(["2024-01-01 09:00", "2024-01-01 10:00"], [0.5, 0.3])
        result = _merge_energy_frames(winner, loser)
        assert result["gross_kwh"].dtype == np.float64
        assert list(result["gross_kwh"]) == pytest.approx([0.5, 0.3])


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

        # NB: end_time now extends the reindex through "now", so the total row
        # count also includes trailing zero-kwh backfill hours between this
        # 2024 fixture and today — assert on the specific hour instead of len().
        row_10 = result[result["timestamp"] == pd.Timestamp("2024-01-01 10:00")]
        assert len(row_10) == 1
        assert row_10.iloc[0]["gross_kwh"] == pytest.approx(1.0)

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
        make_energy_df(["2023-12-29 10:00", "2024-01-01 10:00"], [0.5, 1.0]).to_csv(cache_path, index=False)

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
        saved["timestamp"] = pd.to_datetime(saved["timestamp"])
        # NB: end_time now extends the reindex through "now", so the saved CSV
        # also includes trailing zero-kwh backfill hours between this 2024
        # fixture and today — assert the real row was written, not len().
        row_10 = saved[saved["timestamp"] == pd.Timestamp("2024-01-01 10:00")]
        assert len(row_10) == 1
        assert row_10.iloc[0]["gross_kwh"] == pytest.approx(1.0)

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

        # NB: end_time now extends the reindex through "now", so trailing
        # zero-kwh backfill hours between this 2024 fixture and today are
        # legitimately present — assert the spike hour specifically is
        # filtered out, rather than asserting the whole result is empty.
        spike_row = result[result["timestamp"] == pd.Timestamp("2024-01-01 10:00")]
        assert spike_row.empty, "Spike hour must be filtered out, not backfilled with the spike value"

    def test_excludes_current_partial_hour(self, mock_app, tmp_path):
        """fetch_energy_history must not return the current (incomplete) hourly bucket.

        The most recent row in HA data represents a still-open hour — its kWh is only
        the accumulation so far.  Training on it with full exponential weight biases the
        model for that hour-of-week.
        """
        cache_path = tmp_path / "energy_history.csv"
        # Use Europe/Zurich (same timezone as production) so current_hour matches
        # the completed_cutoff computed inside the function.
        current_hour = pd.Timestamp.now(tz="Europe/Zurich").floor("1h").tz_localize(None)
        prev_hour = current_hour - pd.Timedelta(hours=1)

        make_energy_df(
            [prev_hour.isoformat(), current_hour.isoformat()],
            [1.5, 0.3],
        ).to_csv(cache_path, index=False)

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            result = ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        result_ts = set(result["timestamp"])
        assert current_hour not in result_ts, "Current (incomplete) hour must not be returned"
        assert prev_hour in result_ts, "Previous complete hour must be returned"

    def test_trailing_sensor_silence_backfilled_through_now(self, mock_app, tmp_path):
        """Same trailing-silence bug as fetch_recent_energy, on the weekly
        full-resync path used by _retrain()."""
        cache_path = tmp_path / "energy_history.csv"
        now_local = pd.Timestamp.now(tz="Europe/Zurich")
        last_real_hour = (now_local - pd.Timedelta(hours=3)).floor("1h")
        ha_raw = make_ha_raw(
            [
                (last_real_hour - pd.Timedelta(hours=1)).tz_convert("UTC").isoformat(),
                last_real_hour.tz_convert("UTC").isoformat(),
            ],
            [50.0, 51.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        result_ts = set(result["timestamp"])
        for h in (1, 2):
            ts = (last_real_hour + pd.Timedelta(hours=h)).tz_localize(None)
            assert ts in result_ts, f"hour {ts} missing — trailing silence wasn't backfilled"


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

    def test_excludes_current_partial_hour(self, mock_app, tmp_path):
        """fetch_recent_energy must not return the current (incomplete) hourly bucket."""
        cache_path = tmp_path / "energy_history.csv"
        current_hour = pd.Timestamp.now(tz="Europe/Zurich").floor("1h").tz_localize(None)
        prev_hour = current_hour - pd.Timedelta(hours=1)

        make_energy_df(
            [prev_hour.isoformat(), current_hour.isoformat()],
            [1.5, 0.3],
        ).to_csv(cache_path, index=False)

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            result = ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        result_ts = set(result["timestamp"])
        assert current_hour not in result_ts, "Current (incomplete) hour must not be returned"
        assert prev_hour in result_ts, "Previous complete hour must be returned"

    def test_trailing_sensor_silence_backfilled_through_now(self, mock_app, tmp_path):
        """Regression test for the recurring 'lag_24h has N/48 NaN' warning:
        a grid-import sensor that stops emitting states because solar covers
        100% of household load (e.g. sensor.gplugk_z_ei going quiet on
        2026-07-18) must not leave a growing gap between the last real HA
        state and 'now'. The silent hours are genuine 0.0-kWh readings and
        fetch_recent_energy must backfill them through the current hour."""
        cache_path = tmp_path / "energy_history.csv"
        now_local = pd.Timestamp.now(tz="Europe/Zurich")
        last_real_hour = (now_local - pd.Timedelta(hours=3)).floor("1h")
        ha_raw = make_ha_raw(
            [
                (last_real_hour - pd.Timedelta(hours=1)).tz_convert("UTC").isoformat(),
                last_real_hour.tz_convert("UTC").isoformat(),
            ],
            [50.0, 51.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        result_ts = set(result["timestamp"])
        for h in (1, 2):
            ts = (last_real_hour + pd.Timedelta(hours=h)).tz_localize(None)
            assert ts in result_ts, f"hour {ts} missing — trailing silence wasn't backfilled"


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
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            _check_dst_duplicates(df, _LOGGER)
        assert not any("DST" in r.message or "duplicate" in r.message.lower() for r in caplog.records)

    def test_duplicate_timestamps_emits_warning(self, caplog):
        """Duplicate naive 02:00 (fall-back) triggers a WARNING."""
        # Both rows have the naive timestamp 02:00; one was CEST, one CET
        df = make_energy_df(
            ["2024-10-27 02:00", "2024-10-27 02:00", "2024-10-27 03:00"],
            [1.0, 1.1, 1.0],
        )
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            _check_dst_duplicates(df, _LOGGER)
        assert any("duplicate" in r.message.lower() or "DST" in r.message for r in caplog.records)

    def test_duplicate_count_mentioned_in_warning(self, caplog):
        """Warning message includes the count of duplicated timestamps."""
        df = make_energy_df(
            ["2024-10-27 02:00", "2024-10-27 02:00"],
            [1.0, 1.1],
        )
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            _check_dst_duplicates(df, _LOGGER)
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_texts, "expected at least one WARNING"
        assert any("1" in t for t in warning_texts), "expected duplicate count in message"

    def test_empty_dataframe_no_warning(self, caplog):
        """Empty DataFrame does not raise and emits no warning."""
        df = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            _check_dst_duplicates(df, _LOGGER)
        assert not caplog.records

    def test_single_row_no_warning(self, caplog):
        """Single-row DataFrame cannot have duplicates."""
        df = make_energy_df(["2024-10-27 02:00"], [1.0])
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
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
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
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
                with caplog.at_level(logging.WARNING, logger="energy_forecast"):
                    ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)

        assert any("duplicate" in r.message.lower() or "DST" in r.message for r in caplog.records)


# Module-level logger used directly in DST tests (mirrors ha_data's own logger)
import logging as _logging  # noqa: E402

_LOGGER = _logging.getLogger("energy_forecast.ha_data")


# ── split_ev_charging ─────────────────────────────────────────────────────────


class TestSplitEvCharging:
    def _make_df(self) -> pd.DataFrame:
        """Four rows: two below threshold (3 kWh), two above (10 kWh)."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-12 00:00", periods=4, freq="1h"),
                "gross_kwh": [3.0, 3.0, 10.0, 10.0],
            }
        )

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


# ── split_ev_charging_from_sensor ────────────────────────────────────────────


class TestSplitEvChargingFromSensor:
    def _energy_df(self) -> pd.DataFrame:
        """Four hourly rows at 3 kWh each."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-12 00:00", periods=4, freq="1h"),
                "gross_kwh": [3.0, 3.0, 3.0, 3.0],
            }
        )

    def _ev_df(self, hours_kwh: dict) -> pd.DataFrame:
        """Build a wallbox kWh DataFrame. hours_kwh: {hour_int: kwh}."""
        base = pd.Timestamp("2026-03-12 00:00")
        rows = [{"timestamp": base + pd.Timedelta(hours=h), "kwh": v} for h, v in hours_kwh.items()]
        return pd.DataFrame(rows)

    def test_wallbox_kwh_subtracted_from_gross(self):
        """Wallbox kWh is subtracted from gross_kwh for matching hours."""
        baseline, ev = ha_data.split_ev_charging_from_sensor(self._energy_df(), self._ev_df({1: 2.5, 2: 4.0}))
        assert abs(baseline.iloc[1]["gross_kwh"] - 0.5) < 1e-6  # 3.0 - 2.5
        assert abs(baseline.iloc[2]["gross_kwh"] - 0.0) < 1e-6  # clipped at 0

    def test_subtraction_clipped_to_zero(self):
        """gross_kwh never goes negative even when wallbox > gross."""
        baseline, _ = ha_data.split_ev_charging_from_sensor(self._energy_df(), self._ev_df({0: 10.0}))
        assert baseline.iloc[0]["gross_kwh"] == 0.0

    def test_ev_df_holds_actual_wallbox_kwh(self):
        """ev_df.gross_kwh must equal actual wallbox kWh, not the original gross."""
        _, ev = ha_data.split_ev_charging_from_sensor(self._energy_df(), self._ev_df({2: 4.7}))
        assert len(ev) == 1
        assert abs(ev.iloc[0]["gross_kwh"] - 4.7) < 1e-6

    def test_zero_kwh_hours_not_marked_as_ev(self):
        """Hours where wallbox kwh == 0 must not appear in ev_df."""
        _, ev = ha_data.split_ev_charging_from_sensor(self._energy_df(), self._ev_df({0: 0.0, 1: 0.0}))
        assert len(ev) == 0

    def test_non_ev_hours_unchanged(self):
        """Hours with no wallbox energy must keep original gross_kwh."""
        baseline, _ = ha_data.split_ev_charging_from_sensor(self._energy_df(), self._ev_df({2: 1.0}))
        assert abs(baseline.iloc[0]["gross_kwh"] - 3.0) < 1e-6
        assert abs(baseline.iloc[1]["gross_kwh"] - 3.0) < 1e-6
        assert abs(baseline.iloc[3]["gross_kwh"] - 3.0) < 1e-6

    def test_timestamp_misalignment_floored_to_1h(self):
        """Wallbox timestamps not on the hour are matched after flooring to 1h."""
        ev_df = pd.DataFrame([{"timestamp": pd.Timestamp("2026-03-12 01:37"), "kwh": 3.0}])
        baseline, ev = ha_data.split_ev_charging_from_sensor(self._energy_df(), ev_df)
        # 01:37 → floor → 01:00 matches energy row at 01:00
        assert abs(baseline.iloc[1]["gross_kwh"] - 0.0) < 1e-6
        assert len(ev) == 1


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
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 10:00"]),
                "kwh": [2.5],
            }
        ).to_csv(cache_path, index=False)

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
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 10:00"]),
                "kwh": [1.0],
            }
        ).to_csv(cache_path, index=False)

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
            [100.0, 101.5],  # diff at 10:00 local = 1.5 kWh
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
        cache_path.write_text("timestamp,gross_kwh\n2026-03-20 01:00:00,1.2\n2026-03-20,0.9\n")

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


# ── unit multiplier ──────────────────────────────────────────────────────────


class TestUnitMultiplier:
    """Unit conversion via unit_multiplier in _raw_to_kwh_diff and fetch_energy_history."""

    def test_raw_to_kwh_diff_scales_mwh(self):
        """_raw_to_kwh_diff with unit_multiplier=1000 converts MWh diffs to kWh."""
        from energy_forecast.ha_data import _raw_to_kwh_diff

        # Cumulative meter in MWh: 0.000 → 0.001 → 0.003 MWh per hour
        raw = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2024-01-15 08:00", "2024-01-15 09:00", "2024-01-15 10:00"], utc=True
                ).tz_convert("Europe/Zurich"),
                "value": [0.000, 0.001, 0.003],
            }
        )
        result = _raw_to_kwh_diff(raw, "1h", max_kwh=50.0, unit_multiplier=1000.0)
        assert len(result) == 2
        # 0.001 MWh × 1000 = 1.0 kWh; 0.002 MWh × 1000 = 2.0 kWh
        assert abs(result.iloc[0]["gross_kwh"] - 1.0) < 1e-9
        assert abs(result.iloc[1]["gross_kwh"] - 2.0) < 1e-9

    def test_raw_to_kwh_diff_spike_filter_applied_after_scaling(self):
        """Spike filter (max_kwh) is applied to scaled values, not raw ones."""
        from energy_forecast.ha_data import _raw_to_kwh_diff

        # 0.060 MWh diff → 60 kWh after scaling — exceeds MAX_HOURLY_KWH=50 → dropped
        raw = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-15 08:00", "2024-01-15 09:00"], utc=True).tz_convert(
                    "Europe/Zurich"
                ),
                "value": [0.000, 0.060],
            }
        )
        result = _raw_to_kwh_diff(raw, "1h", max_kwh=50.0, unit_multiplier=1000.0)
        assert result.empty

    def test_fetch_energy_history_applies_unit_multiplier(self, mock_app, tmp_path):
        """fetch_energy_history with unit_multiplier=1000 converts MWh sensor to kWh."""
        cache_path = tmp_path / "energy_history.csv"
        # Cumulative MWh meter: 0.000 → 0.002 → 0.005 MWh
        raw = make_ha_raw(
            ["2024-01-15 08:00:00+00:00", "2024-01-15 09:00:00+00:00", "2024-01-15 10:00:00+00:00"],
            [0.000, 0.002, 0.005],
        )
        with patch.object(ha_data, "_fetch_history", return_value=raw):
            result = ha_data.fetch_energy_history(
                mock_app, "sensor.energy", cache_path=cache_path, unit_multiplier=1000.0
            )
        assert len(result) >= 1
        assert result["gross_kwh"].max() <= 50.0  # spike filter still works
        # 0.002 MWh diff × 1000 = 2.0 kWh
        assert any(abs(v - 2.0) < 1e-6 for v in result["gross_kwh"])


# ── #45 validate_energy_cache ─────────────────────────────────────────────────

from energy_forecast.ha_data import validate_energy_cache  # noqa: E402


class TestValidateEnergyCache:
    def test_clean_data_no_warnings(self, caplog):
        """5 rows, 1h apart, valid values → no WARNING logged."""
        ts = pd.date_range("2024-03-15 08:00", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0, 1.5, 2.0, 1.2, 0.8]})
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            validate_energy_cache(df, _LOGGER)
        assert not caplog.records

    def test_non_monotonic_timestamp_warns(self, caplog):
        """A row with an earlier timestamp than its predecessor triggers WARNING."""
        ts = pd.to_datetime(
            [
                "2024-03-15 08:00",
                "2024-03-15 09:00",
                "2024-03-15 08:30",  # goes backwards
                "2024-03-15 10:00",
            ]
        )
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0, 1.5, 0.8, 2.0]})
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            validate_energy_cache(df, _LOGGER)
        assert any("non-monotonic" in r.message for r in caplog.records)

    def test_gap_greater_than_2h_warns(self, caplog):
        """A 3.5h gap between consecutive rows triggers WARNING."""
        ts = pd.to_datetime(["2024-03-15 08:00", "2024-03-15 11:30"])
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0, 1.5]})
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            validate_energy_cache(df, _LOGGER)
        assert any("gap" in r.message.lower() for r in caplog.records)

    def test_gap_exactly_2h_not_flagged(self, caplog):
        """Exactly 2h gap is NOT flagged (threshold is strictly > 2h)."""
        ts = pd.to_datetime(["2024-03-15 08:00", "2024-03-15 10:00"])
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0, 1.5]})
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            validate_energy_cache(df, _LOGGER)
        assert not any("gap" in r.message.lower() for r in caplog.records)

    def test_dst_gap_warns_with_dst_note(self, caplog):
        """A 3h gap at the DST spring-forward hour warns and mentions DST."""
        # 2024-03-31: clocks spring forward; use 00:00→03:00 (3h gap)
        ts = pd.to_datetime(["2024-03-31 00:00", "2024-03-31 03:00"])
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0, 1.5]})
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            validate_energy_cache(df, _LOGGER)
        assert any("DST" in r.message for r in caplog.records)

    def test_out_of_range_value_warns(self, caplog):
        """gross_kwh outside [0, MAX_HOURLY_KWH] triggers WARNING."""
        ts = pd.date_range("2024-03-15 08:00", periods=2, freq="1h")
        df_over = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0, 60.0]})  # 60 > 50
        df_negative = pd.DataFrame({"timestamp": ts, "gross_kwh": [-1.0, 1.0]})  # negative
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            validate_energy_cache(df_over, _LOGGER)
        assert any("gross_kwh" in r.message for r in caplog.records)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            validate_energy_cache(df_negative, _LOGGER)
        assert any("gross_kwh" in r.message for r in caplog.records)

    def test_zero_value_does_not_warn(self, caplog):
        """gross_kwh == 0 is a real reading (e.g. solar covering 100% of load)
        and must NOT be flagged as out-of-range."""
        ts = pd.date_range("2024-03-15 08:00", periods=2, freq="1h")
        df_zero = pd.DataFrame({"timestamp": ts, "gross_kwh": [0.0, 1.0]})
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            validate_energy_cache(df_zero, _LOGGER)
        assert not any("gross_kwh" in r.message for r in caplog.records)

    def test_no_raise_on_missing_column(self, caplog):
        """DataFrame without gross_kwh column must not raise."""
        ts = pd.date_range("2024-03-15 08:00", periods=3, freq="1h")
        df = pd.DataFrame({"timestamp": ts})  # no gross_kwh
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            validate_energy_cache(df, _LOGGER)  # must not raise


class TestValidateCacheIntegration:
    """validate_energy_cache is called from fetch_energy_history."""

    def test_validate_called_in_fetch_energy_history(self, mock_app, tmp_path):
        """fetch_energy_history must invoke validate_energy_cache after merging."""
        cache_path = tmp_path / "energy_history.csv"
        # Pre-populate cache so the function doesn't hit the empty-history guard
        make_energy_df(
            ["2024-01-01 08:00", "2024-01-01 09:00"],
            [1.0, 1.5],
        ).to_csv(cache_path, index=False)
        with (
            patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()),
            patch.object(ha_data, "validate_energy_cache") as mock_validate,
        ):
            ha_data.fetch_energy_history(mock_app, "sensor.energy", cache_path=cache_path)
        mock_validate.assert_called_once()


# ── #new load_excluded_ranges / filter_excluded_ranges ────────────────────────

from energy_forecast.ha_data import load_excluded_ranges  # noqa: E402


class TestLoadExcludedRanges:
    def test_missing_file_returns_empty_no_log(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert not caplog.records

    def test_header_only_file_returns_empty_no_warning(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert not caplog.records

    def test_well_formed_multi_row_file(self, tmp_path):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text(
            "start,end,reason\n"
            "2026-07-19 14:00,2026-07-21 09:30,gplug fault\n"
            "2026-07-25 00:00,2026-07-25 12:00,second fault\n"
        )
        result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == [
            (pd.Timestamp("2026-07-19 14:00"), pd.Timestamp("2026-07-21 09:30"), "gplug fault"),
            (pd.Timestamp("2026-07-25 00:00"), pd.Timestamp("2026-07-25 12:00"), "second fault"),
        ]

    def test_reason_column_absent_defaults_to_empty_string(self, tmp_path):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end\n2026-07-19,2026-07-20\n")
        result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == [(pd.Timestamp("2026-07-19 00:00"), pd.Timestamp("2026-07-20 23:59:59"), "")]

    def test_extra_column_ignored(self, tmp_path):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason,ticket\n2026-07-19,2026-07-20,fault,JIRA-123\n")
        result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert len(result) == 1
        assert result[0][2] == "fault"

    def test_malformed_row_skipped_others_still_load(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\nnot-a-date,2026-07-20,bad row\n2026-07-25,2026-07-26,good row\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert len(result) == 1
        assert result[0][2] == "good row"
        assert any("row" in r.message.lower() for r in caplog.records)

    def test_end_before_start_skipped(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-07-20 10:00,2026-07-19 10:00,backwards\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert caplog.records

    def test_missing_required_columns_returns_empty_with_warning(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("begin,finish\n2026-07-19,2026-07-20\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert any("start" in r.message.lower() or "end" in r.message.lower() for r in caplog.records)

    def test_ambiguous_date_format_rejected(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n07/19/2026,07/20/2026,slash format\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []

    def test_timezone_offset_rejected(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-07-19 14:00+02:00,2026-07-20 14:00,tz offset\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []

    def test_bare_date_end_expands_to_end_of_day(self, tmp_path):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-07-19,2026-07-21,multi-day\n")
        result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result[0][1] == pd.Timestamp("2026-07-21 23:59:59")

    def test_explicit_time_end_used_exactly(self, tmp_path):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-07-19,2026-07-21 00:00,exact midnight\n")
        result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result[0][1] == pd.Timestamp("2026-07-21 00:00:00")

    def test_spring_forward_nonexistent_time_warns_distinctly(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-03-29 02:30,2026-03-29 04:00,gap\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert len(result) == 1  # still loaded, just warned
        assert any("nonexistent" in r.message.lower() for r in caplog.records)

    def test_fall_back_ambiguous_time_loads_without_special_warning(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-10-25 02:00,2026-10-25 03:00,fall-back\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert len(result) == 1
        assert not any("nonexistent" in r.message.lower() for r in caplog.records)

    def test_truncated_csv_returns_empty_with_warning(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_bytes(b"")  # zero-byte file, simulates a torn Samba write
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert caplog.records

    def test_encoding_corrupted_csv_returns_empty_with_warning(self, tmp_path, caplog):
        path = tmp_path / "excluded_ranges.csv"
        path.write_bytes(b"start,end,reason\n2026-07-19,2026-07-20,caf\xe9 fault\n")  # invalid UTF-8 byte
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            result = load_excluded_ranges(path, "Europe/Zurich", _LOGGER)
        assert result == []
        assert caplog.records

    def test_different_timezone_changes_spring_forward_detection(self, tmp_path, caplog):
        """US/Eastern's 2026 spring-forward gap (Mar 8) differs from Europe/Zurich's (Mar 29) —
        proves the timezone parameter is actually used, not hardcoded."""
        path = tmp_path / "excluded_ranges.csv"
        path.write_text("start,end,reason\n2026-03-08 02:30,2026-03-08 04:00,gap\n")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            load_excluded_ranges(path, "US/Eastern", _LOGGER)
        assert any("nonexistent" in r.message.lower() for r in caplog.records)


# ── fetch_presence_history ──────────────────────────────────────────────────────


class TestFetchPresenceHistory:
    """Tests for fetch_presence_history (occupancy feature #21)."""

    def test_returns_empty_when_no_sensors(self, mock_app):
        """Empty entity list returns empty DataFrame."""
        result = ha_data.fetch_presence_history(mock_app, None, days=30)
        assert len(result) == 0
        assert list(result.columns) == ["timestamp", "people_home"]

    def test_returns_empty_when_empty_list(self, mock_app):
        """Empty list returns empty DataFrame."""
        result = ha_data.fetch_presence_history(mock_app, [], days=30)
        assert len(result) == 0
        assert list(result.columns) == ["timestamp", "people_home"]

    def test_single_person_home_counts_correctly(self, mock_app):
        """Single person entity in home state produces count=1."""
        # HA history raw response: person.alice in "home" state
        # 10:00–11:00 local: home (state="home")
        # 11:00–12:00 local: not_home (state="not_home")
        raw_response = {
            "person.alice": [
                {"state": "home", "last_changed": "2024-01-01T09:00:00+01:00"},  # 10:00 local
                {"state": "not_home", "last_changed": "2024-01-01T10:00:00+01:00"},  # 11:00 local
            ]
        }
        mock_app.get_history.return_value = raw_response

        result = ha_data.fetch_presence_history(mock_app, ["person.alice"], days=30)

        assert len(result) == 2
        assert result.iloc[0]["people_home"] == 1  # home during hour 0
        assert result.iloc[1]["people_home"] == 0  # not_home during hour 1

    def test_two_persons_sum_correctly(self, mock_app):
        """Two person entities count independently and sum."""
        raw_alice = {
            "person.alice": [
                {"state": "home", "last_changed": "2024-01-01T09:00:00+01:00"},
                {"state": "not_home", "last_changed": "2024-01-01T10:00:00+01:00"},
            ]
        }
        raw_bob = {
            "person.bob": [
                {"state": "home", "last_changed": "2024-01-01T09:00:00+01:00"},
                {"state": "home", "last_changed": "2024-01-01T10:00:00+01:00"},
            ]
        }

        def side_effect_fn(*args, **kwargs):
            entity = kwargs.get("entity_id")
            if entity == "person.alice":
                return raw_alice
            elif entity == "person.bob":
                return raw_bob
            return {}

        mock_app.get_history.side_effect = side_effect_fn

        result = ha_data.fetch_presence_history(mock_app, ["person.alice", "person.bob"], days=30)

        assert len(result) == 2
        assert result.iloc[0]["people_home"] == 2  # both home at 10:00
        assert result.iloc[1]["people_home"] == 1  # only bob home at 11:00

    def test_person_not_home_is_zero(self, mock_app):
        """Person always in not_home state produces all zeros."""
        raw_response = {
            "person.alice": [
                {"state": "not_home", "last_changed": "2024-01-01T09:00:00+01:00"},
            ]
        }
        mock_app.get_history.return_value = raw_response

        result = ha_data.fetch_presence_history(mock_app, ["person.alice"], days=30)

        assert len(result) == 1
        assert result.iloc[0]["people_home"] == 0

    def test_returns_empty_on_fetch_error(self, mock_app):
        """Exception during get_history returns empty DataFrame."""
        mock_app.get_history.side_effect = Exception("API error")

        result = ha_data.fetch_presence_history(mock_app, ["person.alice"], days=30)

        assert len(result) == 0
        assert list(result.columns) == ["timestamp", "people_home"]

    def test_timestamps_are_naive_europe_zurich(self, mock_app):
        """Returned timestamps are naive (no timezone) in Europe/Zurich local time."""
        raw_response = {
            "person.alice": [
                {"state": "home", "last_changed": "2024-01-01T09:00:00+01:00"},  # 09:00 local (UTC+1)
            ]
        }
        mock_app.get_history.return_value = raw_response

        result = ha_data.fetch_presence_history(mock_app, ["person.alice"], days=30)

        assert len(result) == 1
        ts = result.iloc[0]["timestamp"]
        assert ts.tzinfo is None  # naive
        assert ts.hour == 9  # 09:00 local time (UTC+1)

    def test_ignores_invalid_state_values(self, mock_app):
        """States other than home/not_home are skipped."""
        raw_response = {
            "person.alice": [
                {"state": "home", "last_changed": "2024-01-01T09:00:00+01:00"},
                {"state": "unavailable", "last_changed": "2024-01-01T09:30:00+01:00"},  # skipped
                {"state": "not_home", "last_changed": "2024-01-01T10:00:00+01:00"},
            ]
        }
        mock_app.get_history.return_value = raw_response

        result = ha_data.fetch_presence_history(mock_app, ["person.alice"], days=30)

        # Only valid transitions: home→not_home
        assert len(result) >= 1
        assert result.iloc[0]["people_home"] == 1


# ── fetch_recent_energy — tail read (Fix 4) ───────────────────────────────────


class TestFetchRecentEnergyTailRead:
    """fetch_recent_energy must read only the last _FETCH_RECENT_TAIL_ROWS rows
    from a large CSV (memory efficiency), while still returning a correct result.
    """

    def _make_large_cache(self, path, n_rows: int, base: pd.Timestamp) -> list:
        """Write n_rows hourly rows to a CSV and return the expected timestamps."""
        ts = pd.date_range(base, periods=n_rows, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * n_rows})
        df.to_csv(path, index=False)
        return ts.tolist()

    def test_large_cache_returns_tail_rows(self, mock_app, tmp_path):
        """With 600 rows in the CSV and no new HA data, the returned DataFrame
        has at most _FETCH_RECENT_TAIL_ROWS rows (deque tail read)."""
        from energy_forecast.ha_data import _FETCH_RECENT_TAIL_ROWS

        cache_path = tmp_path / "energy_history.csv"
        base = pd.Timestamp("2024-01-01 00:00")
        self._make_large_cache(cache_path, n_rows=600, base=base)

        # Return empty HA data so only the cache contributes
        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            result = ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        assert len(result) <= _FETCH_RECENT_TAIL_ROWS

    def test_tail_read_preserves_latest_rows(self, mock_app, tmp_path):
        """The returned rows must be the *last* rows in the CSV, not the first."""
        from energy_forecast.ha_data import _FETCH_RECENT_TAIL_ROWS

        cache_path = tmp_path / "energy_history.csv"
        base = pd.Timestamp("2024-01-01 00:00")
        timestamps = self._make_large_cache(cache_path, n_rows=600, base=base)

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame()):
            result = ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        assert result["timestamp"].max() >= timestamps[600 - _FETCH_RECENT_TAIL_ROWS]

    def test_tail_read_merges_new_ha_rows(self, mock_app, tmp_path):
        """New HA rows are still merged in even when the cache is large."""
        cache_path = tmp_path / "energy_history.csv"
        base = pd.Timestamp("2024-01-01 00:00")
        self._make_large_cache(cache_path, n_rows=600, base=base)

        # HA provides 2 new rows just after the cache window. cache_end_ts is a
        # naive-local timestamp (same convention as `base` / the rest of this
        # class). Localize it to Europe/Zurich before converting to UTC for
        # make_ha_raw — mirroring test_trailing_sensor_silence_backfilled_through_now
        # above — rather than treating the naive value as if it were already UTC.
        cache_end_ts = base + pd.Timedelta(hours=599)
        cache_end_ts_local = cache_end_ts.tz_localize("Europe/Zurich")
        ha_raw = make_ha_raw(
            [
                (cache_end_ts_local + pd.Timedelta(hours=1)).tz_convert("UTC").isoformat(),
                (cache_end_ts_local + pd.Timedelta(hours=2)).tz_convert("UTC").isoformat(),
            ],
            [100.0, 101.5],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_recent_energy(mock_app, "sensor.energy", cache_path=cache_path)

        # The two raw HA cumulative readings diff to a genuine 1.5 kWh row at
        # cache_end_ts + 2h. (The +1h reading's diff is NaN — no prior reading
        # to diff against — so it's correctly dropped, same as the leading edge
        # of any diff-based series.) Assert the exact merged value via a
        # timestamp lookup so this test actually fails if new-HA-row merging
        # breaks, rather than only checking that the tail timestamp is "large"
        # (which passed trivially once end_time extended the reindex to ~now).
        by_ts = result.set_index("timestamp")["gross_kwh"]
        merged_ts = cache_end_ts + pd.Timedelta(hours=2)
        assert merged_ts in by_ts.index, f"{merged_ts} missing — new HA rows weren't merged"
        assert by_ts.loc[merged_ts] == pytest.approx(1.5)


# ── fetch_program_sensor_history ─────────────────────────────────────────────


class TestFetchProgramSensorHistory:
    """Tests for fetch_program_sensor_history (program-type sensor support)."""

    def _make_raw(self, states: list[dict]) -> dict:
        return {"sensor.dw_program": states}

    def test_returns_string_states(self, mock_app):
        """Normal states are returned with correct columns and values."""
        raw = self._make_raw(
            [
                {"state": "eco", "last_changed": "2024-01-01T10:00:00+01:00"},
                {"state": "normal", "last_changed": "2024-01-01T12:00:00+01:00"},
            ]
        )
        mock_app.get_history.return_value = raw
        result = ha_data.fetch_program_sensor_history(mock_app, "sensor.dw_program", days=30)
        assert list(result.columns) == ["timestamp", "program"]
        assert len(result) == 2
        assert result.iloc[0]["program"] == "eco"
        assert result.iloc[1]["program"] == "normal"

    def test_drops_unavailable_and_unknown(self, mock_app):
        """'unavailable', 'unknown', and '' are excluded from the result."""
        raw = self._make_raw(
            [
                {"state": "unavailable", "last_changed": "2024-01-01T08:00:00+01:00"},
                {"state": "unknown", "last_changed": "2024-01-01T09:00:00+01:00"},
                {"state": "", "last_changed": "2024-01-01T09:30:00+01:00"},
                {"state": "eco", "last_changed": "2024-01-01T10:00:00+01:00"},
            ]
        )
        mock_app.get_history.return_value = raw
        result = ha_data.fetch_program_sensor_history(mock_app, "sensor.dw_program", days=30)
        assert len(result) == 1
        assert result.iloc[0]["program"] == "eco"

    def test_lowercases_states(self, mock_app):
        """State values are lowercased — 'ECO', 'Eco' → 'eco'."""
        raw = self._make_raw(
            [
                {"state": "ECO", "last_changed": "2024-01-01T10:00:00+01:00"},
                {"state": "Normal", "last_changed": "2024-01-01T12:00:00+01:00"},
            ]
        )
        mock_app.get_history.return_value = raw
        result = ha_data.fetch_program_sensor_history(mock_app, "sensor.dw_program", days=30)
        assert result.iloc[0]["program"] == "eco"
        assert result.iloc[1]["program"] == "normal"

    def test_returns_empty_on_get_history_failure(self, mock_app):
        """Exception from get_history → empty DataFrame, no exception propagated."""
        mock_app.get_history.side_effect = RuntimeError("HA unavailable")
        result = ha_data.fetch_program_sensor_history(mock_app, "sensor.dw_program", days=30)
        assert result.empty
        assert list(result.columns) == ["timestamp", "program"]

    def test_returns_empty_on_empty_history(self, mock_app):
        """Empty state list → empty DataFrame."""
        mock_app.get_history.return_value = {"sensor.dw_program": []}
        result = ha_data.fetch_program_sensor_history(mock_app, "sensor.dw_program", days=30)
        assert result.empty
        assert list(result.columns) == ["timestamp", "program"]

    def test_timestamps_are_naive(self, mock_app):
        """Returned timestamps must be naive (no tzinfo)."""
        raw = self._make_raw(
            [
                {"state": "eco", "last_changed": "2024-01-01T10:00:00+01:00"},
            ]
        )
        mock_app.get_history.return_value = raw
        result = ha_data.fetch_program_sensor_history(mock_app, "sensor.dw_program", days=30)
        assert result.iloc[0]["timestamp"].tzinfo is None

    def test_sorted_by_timestamp(self, mock_app):
        """Result is sorted ascending by timestamp regardless of input order."""
        raw = self._make_raw(
            [
                {"state": "intensive", "last_changed": "2024-01-01T14:00:00+01:00"},
                {"state": "eco", "last_changed": "2024-01-01T10:00:00+01:00"},
            ]
        )
        mock_app.get_history.return_value = raw
        result = ha_data.fetch_program_sensor_history(mock_app, "sensor.dw_program", days=30)
        assert result.iloc[0]["program"] == "eco"
        assert result.iloc[1]["program"] == "intensive"


# ── _resolve_programs_for_series ──────────────────────────────────────────────


class TestResolveProgramsForSeries:
    """Tests for the _resolve_programs_for_series() last-value-carry-forward helper."""

    def _make_prog_df(self, rows):
        """Build a program event DataFrame from list of (timestamp_str, label) tuples."""
        return (
            pd.DataFrame([{"timestamp": pd.Timestamp(ts), "program": lbl} for ts, lbl in rows])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def test_basic_lvfc(self):
        """Each hourly timestamp gets the last program state before or at that time."""
        prog_df = self._make_prog_df(
            [
                ("2024-01-01 09:30", "eco"),
                ("2024-01-01 11:00", "intensive"),
            ]
        )
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2024-01-01 10:00",
                    "2024-01-01 11:00",
                    "2024-01-01 12:00",
                ]
            )
        )
        result = ha_data._resolve_programs_for_series(timestamps, prog_df)
        assert result.tolist() == ["eco", "intensive", "intensive"]

    def test_no_preceding_event_returns_empty_string(self):
        """Timestamps before the first program event get an empty string."""
        prog_df = self._make_prog_df([("2024-01-01 12:00", "cotton")])
        timestamps = pd.Series(pd.to_datetime(["2024-01-01 08:00", "2024-01-01 13:00"]))
        result = ha_data._resolve_programs_for_series(timestamps, prog_df)
        assert result.tolist() == ["", "cotton"]

    def test_empty_prog_df_returns_all_empty(self):
        """An empty program DataFrame yields all empty strings."""
        prog_df = pd.DataFrame(columns=["timestamp", "program"])
        timestamps = pd.Series(pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]))
        result = ha_data._resolve_programs_for_series(timestamps, prog_df)
        assert result.tolist() == ["", ""]

    def test_exact_timestamp_match_uses_that_event(self):
        """When the program event timestamp exactly matches the hourly row, it's used."""
        prog_df = self._make_prog_df([("2024-01-01 10:00", "quick")])
        timestamps = pd.Series(pd.to_datetime(["2024-01-01 10:00"]))
        result = ha_data._resolve_programs_for_series(timestamps, prog_df)
        assert result.tolist() == ["quick"]

    def test_preserves_original_index(self):
        """Returned Series has the same index as the input timestamps Series."""
        prog_df = self._make_prog_df([("2024-01-01 09:00", "eco")])
        timestamps = pd.Series(
            pd.to_datetime(["2024-01-01 10:00", "2024-01-01 11:00"]),
            index=[5, 7],
        )
        result = ha_data._resolve_programs_for_series(timestamps, prog_df)
        assert list(result.index) == [5, 7]

    # -- forward-lookup fallback tests -----------------------------------------

    def test_late_firing_sensor_gets_correct_label(self):
        """Program event 15 min after hour boundary is attributed to that hour.

        Reproduces: user starts machine at 12:05 → program sensor fires 'eco'
        at 12:05 → hour row stamped 12:00 backward-sees 'no_program' → should
        be corrected to 'eco' by the forward fallback.
        """
        prog_df = self._make_prog_df(
            [
                ("2024-01-01 11:30", "no_program"),  # previous cycle ended
                ("2024-01-01 12:05", "eco"),  # new cycle selected at 12:05
                ("2024-01-01 16:30", "no_program"),  # cycle finished
            ]
        )
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2024-01-01 11:00",
                    "2024-01-01 12:00",  # ← should get "eco", not "no_program"
                    "2024-01-01 13:00",
                    "2024-01-01 14:00",
                    "2024-01-01 17:00",
                ]
            )
        )
        result = ha_data._resolve_programs_for_series(timestamps, prog_df)
        # 11:00 precedes the first event (11:30) → no backward match → ""
        assert result.tolist() == ["", "eco", "eco", "eco", "no_program"]

    def test_forward_fallback_does_not_override_running_cycle(self):
        """A real backward label is never replaced by a forward match.

        If eco is still running at 13:00 and 'intensive' starts at 13:50,
        the 13:00 row must keep 'eco', not be overwritten by 'intensive'.
        """
        prog_df = self._make_prog_df(
            [
                ("2024-01-01 12:00", "eco"),
                ("2024-01-01 13:50", "intensive"),
            ]
        )
        timestamps = pd.Series(
            pd.to_datetime(
                [
                    "2024-01-01 13:00",  # ← 'eco' still running; 'intensive' within 1 h forward
                    "2024-01-01 14:00",
                ]
            )
        )
        result = ha_data._resolve_programs_for_series(timestamps, prog_df)
        assert result.tolist() == ["eco", "intensive"]

    def test_forward_fallback_catches_65min_gap(self):
        """Program event 65 min after hour boundary is attributed (was the failing case)."""
        prog_df = self._make_prog_df(
            [
                ("2024-01-01 20:30", "no_program"),
                ("2024-01-01 22:05", "power_wash"),  # 65 min after 21:00
            ]
        )
        timestamps = pd.Series(pd.to_datetime(["2024-01-01 21:00"]))
        result = ha_data._resolve_programs_for_series(timestamps, prog_df)
        assert result.tolist() == ["power_wash"]

    def test_forward_fallback_not_triggered_beyond_two_hours(self):
        """Program event more than 2 h ahead does not affect backward label."""
        prog_df = self._make_prog_df(
            [
                ("2024-01-01 11:30", "no_program"),
                ("2024-01-01 14:10", "eco"),  # 2 h 10 min after 12:00 → outside tolerance
            ]
        )
        timestamps = pd.Series(pd.to_datetime(["2024-01-01 12:00"]))
        result = ha_data._resolve_programs_for_series(timestamps, prog_df)
        assert result.tolist() == ["no_program"]

    def test_forward_fallback_idle_sentinel_not_substituted(self):
        """A forward 'no_program' event does not replace a backward '' result."""
        prog_df = self._make_prog_df([("2024-01-01 12:30", "no_program")])
        timestamps = pd.Series(pd.to_datetime(["2024-01-01 12:00"]))
        result = ha_data._resolve_programs_for_series(timestamps, prog_df)
        # Forward finds 'no_program' at 12:30 but it is also idle → stay ""
        assert result.tolist() == [""]


# ── fetch_sub_sensor_history with program_entity_id ───────────────────────────


class TestFetchSubSensorHistoryWithProgram:
    """Tests for fetch_sub_sensor_history / fetch_recent_sub_sensor program integration."""

    def _make_ha_raw(self, iso_times, values):
        """Build a minimal HA history DataFrame (as _fetch_history returns).

        Returns tz-aware Europe/Zurich timestamps, matching the real _fetch_history output.
        """
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(iso_times, utc=True).tz_convert("Europe/Zurich"),
                "value": values,
            }
        )

    def _make_prog_raw(self, rows):
        """Build a minimal program-sensor raw list (format expected by fetch_program_sensor_history)."""
        return [[{"state": lbl, "last_changed": ts} for ts, lbl in rows]]

    def test_program_column_written_to_csv(self, mock_app, tmp_path):
        """When program_entity_id is provided, CSV contains a 'program' column."""
        cache_path = tmp_path / "sub_dw.csv"
        ha_raw = self._make_ha_raw(
            ["2024-01-01T08:00:00+01:00", "2024-01-01T09:00:00+01:00"],
            [10.0, 11.5],
        )
        mock_app.get_history.return_value = self._make_prog_raw(
            [
                ("2024-01-01T07:00:00+01:00", "eco"),
            ]
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_sub_sensor_history(
                mock_app,
                "sensor.dw_kwh",
                cache_path,
                program_entity_id="sensor.dw_program",
            )

        assert "program" in result.columns
        assert cache_path.exists()
        saved = pd.read_csv(cache_path)
        assert "program" in saved.columns

    def test_program_labels_resolved_correctly(self, mock_app, tmp_path):
        """Each hourly row gets the program active at that time."""
        cache_path = tmp_path / "sub_dw.csv"
        # Local timestamps after conversion: 09:00, 10:00, 11:00
        ha_raw = self._make_ha_raw(
            ["2024-01-01T08:00:00+01:00", "2024-01-01T09:00:00+01:00", "2024-01-01T10:00:00+01:00"],
            [10.0, 11.0, 11.5],
        )
        # Program switches from eco → intensive at 10:00 local
        mock_app.get_history.return_value = self._make_prog_raw(
            [
                ("2024-01-01T08:30:00+01:00", "eco"),
                ("2024-01-01T10:00:00+01:00", "intensive"),
            ]
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_sub_sensor_history(
                mock_app,
                "sensor.dw_kwh",
                cache_path,
                program_entity_id="sensor.dw_program",
            )

        # 10:00 row (diff from 09→10) should be "eco"; 11:00 row should be "intensive"
        rows = result.sort_values("timestamp").reset_index(drop=True)
        assert rows.iloc[0]["program"] == "eco"
        assert rows.iloc[1]["program"] == "intensive"

    def test_no_program_entity_id_no_program_column(self, mock_app, tmp_path):
        """Without program_entity_id, result has no 'program' column (backward compat)."""
        cache_path = tmp_path / "sub_hp.csv"
        ha_raw = self._make_ha_raw(
            ["2024-01-01T08:00:00+01:00", "2024-01-01T09:00:00+01:00"],
            [5.0, 6.0],
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_sub_sensor_history(mock_app, "sensor.hp_kwh", cache_path)

        assert "program" not in result.columns

    def test_existing_cache_without_program_column_is_backward_compatible(self, mock_app, tmp_path):
        """A cached CSV without a 'program' column is loaded and extended correctly."""
        cache_path = tmp_path / "sub_dw.csv"
        # Old-format cache: no program column
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01 08:00"]),
                "kwh": [0.5],
            }
        ).to_csv(cache_path, index=False)

        ha_raw = self._make_ha_raw(
            ["2024-01-01T09:00:00+01:00", "2024-01-01T10:00:00+01:00"],
            [20.0, 21.0],
        )
        mock_app.get_history.return_value = self._make_prog_raw(
            [
                ("2024-01-01T09:30:00+01:00", "cotton"),
            ]
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_sub_sensor_history(
                mock_app,
                "sensor.dw_kwh",
                cache_path,
                program_entity_id="sensor.dw_program",
            )

        assert "program" in result.columns
        # The old cached row has no program — should be empty string, not raise
        old_row = result[result["timestamp"] == pd.Timestamp("2024-01-01 08:00")]
        assert len(old_row) == 1
        assert old_row.iloc[0]["program"] == ""

    def test_cached_program_labels_preserved_after_fresh_fetch(self, mock_app, tmp_path):
        """Existing non-empty program labels in cache survive a subsequent fetch that doesn't cover them."""
        cache_path = tmp_path / "sub_dw.csv"
        # Cache has a labelled row from > 30 days ago (outside fresh fetch window)
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-11-01 10:00"]),
                "kwh": [1.2],
                "program": ["eco"],
            }
        ).to_csv(cache_path, index=False)

        # Fresh HA fetch only covers a recent window — no overlap with cached row
        ha_raw = self._make_ha_raw(
            ["2024-01-01T08:00:00+01:00", "2024-01-01T09:00:00+01:00"],
            [5.0, 6.5],
        )
        mock_app.get_history.return_value = self._make_prog_raw(
            [
                ("2024-01-01T07:00:00+01:00", "intensive"),
            ]
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_sub_sensor_history(
                mock_app,
                "sensor.dw_kwh",
                cache_path,
                program_entity_id="sensor.dw_program",
            )

        old_row = result[result["timestamp"] == pd.Timestamp("2023-11-01 10:00")]
        assert len(old_row) == 1
        assert old_row.iloc[0]["program"] == "eco"

    def test_recent_sub_sensor_with_program(self, mock_app, tmp_path):
        """fetch_recent_sub_sensor also persists program labels."""
        cache_path = tmp_path / "sub_dw.csv"
        ha_raw = self._make_ha_raw(
            ["2024-01-01T08:00:00+01:00", "2024-01-01T09:00:00+01:00"],
            [10.0, 10.8],
        )
        mock_app.get_history.return_value = self._make_prog_raw(
            [
                ("2024-01-01T07:30:00+01:00", "quick"),
            ]
        )
        with patch.object(ha_data, "_fetch_history", return_value=ha_raw):
            result = ha_data.fetch_recent_sub_sensor(
                mock_app,
                "sensor.dw_kwh",
                cache_path,
                program_entity_id="sensor.dw_program",
            )

        assert "program" in result.columns
        assert result.iloc[0]["program"] == "quick"

    def test_empty_raw_ha_with_program_entity_and_cache(self, mock_app, tmp_path):
        """fetch_recent_sub_sensor must not raise when raw HA is empty but cache
        has data and program_entity_id is set.

        Regression: pandas 3.x merge_asof raises "Incompatible merge dtype,
        dtype('O') and dtype('<M8[us]')" when df_new was built via
        pd.DataFrame(columns=[...]) which gives object-typed timestamp column.
        """
        cache_path = tmp_path / "sub_wash.csv"
        # Pre-populate cache with one row
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01T10:00:00"]),
                "kwh": [0.5],
                "program": ["eco"],
            }
        ).to_csv(cache_path, index=False)

        mock_app.get_history.return_value = self._make_prog_raw(
            [
                ("2024-01-01T07:00:00+01:00", "eco"),
            ]
        )

        # raw_ha empty → df_new fallback path is exercised
        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame(columns=["timestamp", "value"])):
            result = ha_data.fetch_recent_sub_sensor(
                mock_app,
                "sensor.wash_kwh",
                cache_path,
                program_entity_id="sensor.wash_program",
            )

        assert len(result) == 1
        assert result.iloc[0]["kwh"] == pytest.approx(0.5)

    def test_empty_raw_ha_with_program_entity_and_cache_full_history(self, mock_app, tmp_path):
        """fetch_sub_sensor_history must not raise when raw HA is empty but
        cache has data and program_entity_id is set (same dtype regression)."""
        cache_path = tmp_path / "sub_dryer.csv"
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-02T10:00:00"]),
                "kwh": [1.2],
                "program": ["cotton"],
            }
        ).to_csv(cache_path, index=False)

        mock_app.get_history.return_value = self._make_prog_raw(
            [
                ("2024-01-02T09:00:00+01:00", "cotton"),
            ]
        )

        with patch.object(ha_data, "_fetch_history", return_value=pd.DataFrame(columns=["timestamp", "value"])):
            result = ha_data.fetch_sub_sensor_history(
                mock_app,
                "sensor.dryer_kwh",
                cache_path,
                program_entity_id="sensor.dryer_program",
            )

        assert len(result) == 1
        assert result.iloc[0]["kwh"] == pytest.approx(1.2)


from energy_forecast.ha_data import _raw_to_kwh_diff  # noqa: E402


class TestRawToKwhDiff:
    def _make_raw(self, timestamps, values):
        import pandas as pd

        ts = pd.to_datetime(timestamps).tz_localize("Europe/Zurich")
        return pd.DataFrame({"timestamp": ts, "value": values})

    def test_basic_hourly_diff(self):
        import pandas as pd

        # Use one reading per hour so resample("1h").last() picks the only value
        # in each bucket, giving clean 1.0 kWh diffs.
        raw = self._make_raw(
            [
                "2024-01-01 00:30",
                "2024-01-01 01:30",
                "2024-01-01 02:30",
            ],
            [100.0, 101.0, 102.0],
        )
        result = _raw_to_kwh_diff(raw, "1h", max_kwh=50.0)
        assert list(result.columns) == ["timestamp", "gross_kwh"]
        assert result["timestamp"].dt.tz is None  # naive
        kwh = result.set_index("timestamp")["gross_kwh"]
        assert abs(kwh.loc[pd.Timestamp("2024-01-01 01:00")] - 1.0) < 0.01
        assert abs(kwh.loc[pd.Timestamp("2024-01-01 02:00")] - 1.0) < 0.01

    def test_basic_15min_diff(self):
        raw = self._make_raw(
            ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:30"],
            [100.0, 100.25, 100.5],
        )
        result = _raw_to_kwh_diff(raw, "15min", max_kwh=12.5)
        assert len(result) >= 1
        assert result["gross_kwh"].max() <= 12.5

    def test_empty_returns_empty(self):
        import pandas as pd

        result = _raw_to_kwh_diff(pd.DataFrame(), "1h", max_kwh=50.0)
        assert list(result.columns) == ["timestamp", "gross_kwh"]
        assert result.empty

    def test_max_kwh_filter_applied(self):
        raw = self._make_raw(
            ["2024-01-01 00:00", "2024-01-01 01:00"],
            [0.0, 100.0],  # 100 kWh in one hour — above any reasonable limit
        )
        result = _raw_to_kwh_diff(raw, "1h", max_kwh=50.0)
        assert result.empty  # filtered out

    def test_negative_diff_dropped_not_fabricated_as_zero(self):
        """A meter reset (negative raw diff) must be dropped, not recorded as a
        fabricated gross_kwh=0.0 — true consumption during a reset is unknown,
        unlike a genuinely flat hour where the raw diff really is 0."""
        raw = self._make_raw(
            ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"],
            [100.0, 99.0, 101.0],  # meter reset between h0 and h1 (raw diff -1)
        )
        result = _raw_to_kwh_diff(raw, "1h", max_kwh=50.0)
        kwh = result.set_index("timestamp")["gross_kwh"]
        assert pd.Timestamp("2024-01-01 01:00") not in kwh.index
        # h2: diff = 2 → kept normally.
        assert abs(kwh.loc[pd.Timestamp("2024-01-01 02:00")] - 2.0) < 0.01

    def test_zero_diff_hour_kept_not_dropped(self):
        """A flat hour (e.g. solar fully covering household load) is a real
        gross_kwh=0.0 reading and must not be dropped like a bad/missing row."""
        raw = self._make_raw(
            ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 03:00"],
            [100.0, 101.0, 101.0, 102.5],  # h2: no change → diff 0.0
        )
        result = _raw_to_kwh_diff(raw, "1h", max_kwh=50.0)
        kwh = result.set_index("timestamp")["gross_kwh"]
        assert pd.Timestamp("2024-01-01 02:00") in kwh.index
        assert abs(kwh.loc[pd.Timestamp("2024-01-01 02:00")]) < 1e-9

    def test_end_time_extends_trailing_silence_as_zero(self):
        """A sensor that stops emitting entirely (e.g. a grid-import meter
        with solar fully covering household load — it has nothing new to
        report, so it never pushes an update) must still produce 0.0-kWh
        rows through end_time. resample() alone stops at the last raw
        state and silently drops the hours after it."""
        raw = self._make_raw(
            ["2024-01-01 00:00", "2024-01-01 01:00"],
            [100.0, 101.0],
        )
        end_time = pd.Timestamp("2024-01-01 04:00", tz="Europe/Zurich")
        result = _raw_to_kwh_diff(raw, "1h", max_kwh=50.0, end_time=end_time)
        kwh = result.set_index("timestamp")["gross_kwh"]
        for hour in ("02:00", "03:00", "04:00"):
            ts = pd.Timestamp(f"2024-01-01 {hour}")
            assert ts in kwh.index, f"{hour} missing — trailing silence wasn't extended"
            assert abs(kwh.loc[ts]) < 1e-9

    def test_end_time_none_preserves_old_behavior(self):
        """Without end_time (the default), trailing silence still produces
        no rows — end_time is opt-in so unmigrated callers are unaffected."""
        raw = self._make_raw(
            ["2024-01-01 00:00", "2024-01-01 01:00"],
            [100.0, 101.0],
        )
        result = _raw_to_kwh_diff(raw, "1h", max_kwh=50.0)
        kwh = result.set_index("timestamp")["gross_kwh"]
        assert pd.Timestamp("2024-01-01 02:00") not in kwh.index

    def test_end_time_before_last_raw_timestamp_is_noop(self):
        """If end_time is earlier than the last real raw state, real data
        must not be truncated or altered."""
        raw = self._make_raw(
            ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00"],
            [100.0, 101.0, 103.0],
        )
        end_time = pd.Timestamp("2024-01-01 00:30", tz="Europe/Zurich")
        result = _raw_to_kwh_diff(raw, "1h", max_kwh=50.0, end_time=end_time)
        kwh = result.set_index("timestamp")["gross_kwh"]
        assert abs(kwh.loc[pd.Timestamp("2024-01-01 01:00")] - 1.0) < 0.01
        assert abs(kwh.loc[pd.Timestamp("2024-01-01 02:00")] - 2.0) < 0.01

    def test_end_time_extension_respects_max_kwh_and_negative_diff_rules(self):
        """Extended trailing rows are genuine 0.0 diffs, not exempt from the
        existing max_kwh/negative-diff filtering that runs after ffill."""
        raw = self._make_raw(
            ["2024-01-01 00:00", "2024-01-01 01:00"],
            [100.0, 99.0],  # meter reset going into the silent stretch
        )
        end_time = pd.Timestamp("2024-01-01 03:00", tz="Europe/Zurich")
        result = _raw_to_kwh_diff(raw, "1h", max_kwh=50.0, end_time=end_time)
        kwh = result.set_index("timestamp")["gross_kwh"]
        assert pd.Timestamp("2024-01-01 01:00") not in kwh.index  # reset still dropped
        # 02:00 and 03:00 are flat relative to the post-reset value (99.0) → genuine zeros
        assert abs(kwh.loc[pd.Timestamp("2024-01-01 02:00")]) < 1e-9
        assert abs(kwh.loc[pd.Timestamp("2024-01-01 03:00")]) < 1e-9


class TestFetchEnergyHistory15m:
    @patch("energy_forecast.ha_data._fetch_history")
    def test_writes_15m_cache(self, mock_fetch, tmp_path):
        import pandas as pd

        cache = tmp_path / "energy_history_15m.csv"
        ts = pd.date_range("2024-01-01", periods=120, freq="15min").tz_localize("Europe/Zurich")
        readings = pd.DataFrame({"timestamp": ts, "value": range(len(ts))})
        mock_fetch.return_value = readings
        mock_app = MagicMock()

        ha_data.fetch_energy_history_15m(mock_app, "sensor.energy", cache_path=cache)

        assert cache.exists()
        saved = pd.read_csv(cache)
        assert list(saved.columns) == ["timestamp", "gross_kwh"]
        assert len(saved) > 0

    @patch("energy_forecast.ha_data._fetch_history")
    def test_ha_empty_and_no_cache_raises(self, mock_fetch, tmp_path):
        import pandas as pd

        mock_fetch.return_value = pd.DataFrame()
        mock_app = MagicMock()
        cache = tmp_path / "energy_history_15m.csv"

        with pytest.raises(ValueError, match="No history"):
            ha_data.fetch_energy_history_15m(mock_app, "sensor.energy", cache_path=cache)

    @patch("energy_forecast.ha_data._fetch_history")
    def test_strips_partial_current_slot(self, mock_fetch, tmp_path):
        """The returned DataFrame must not include the currently-open 15-min slot."""
        import pandas as pd

        ts = pd.date_range("2024-01-01", periods=120, freq="15min").tz_localize("Europe/Zurich")
        readings = pd.DataFrame({"timestamp": ts, "value": list(range(120))})
        mock_fetch.return_value = readings
        mock_app = MagicMock()
        cache = tmp_path / "energy_history_15m.csv"

        result = ha_data.fetch_energy_history_15m(mock_app, "sensor.energy", cache_path=cache)
        cutoff = pd.Timestamp.now(tz="Europe/Zurich").floor("15min").tz_localize(None)
        assert (result["timestamp"] < cutoff).all()

    @patch("energy_forecast.ha_data._fetch_history")
    def test_trailing_sensor_silence_backfilled_through_now(self, mock_fetch, tmp_path):
        """Same trailing-silence bug as the hourly path, on the 15-minute cache."""
        import pandas as pd

        cache = tmp_path / "energy_history_15m.csv"
        now_local = pd.Timestamp.now(tz="Europe/Zurich")
        last_real_slot = (now_local - pd.Timedelta(hours=1)).floor("15min")
        ts = pd.to_datetime(
            [
                (last_real_slot - pd.Timedelta(minutes=15)).tz_convert("UTC"),
                last_real_slot.tz_convert("UTC"),
            ]
        ).tz_convert("Europe/Zurich")
        mock_fetch.return_value = pd.DataFrame({"timestamp": ts, "value": [50.0, 50.5]})
        mock_app = MagicMock()

        result = ha_data.fetch_energy_history_15m(mock_app, "sensor.energy", cache_path=cache)

        result_ts = set(result["timestamp"])
        expected = (last_real_slot + pd.Timedelta(minutes=15)).tz_localize(None)
        assert expected in result_ts, f"slot {expected} missing — trailing silence wasn't backfilled"


class TestFetchRecentEnergy15m:
    @patch("energy_forecast.ha_data._fetch_history")
    def test_appends_new_rows_to_cache(self, mock_fetch, tmp_path):
        import pandas as pd

        cache = tmp_path / "energy_history_15m.csv"
        seed_ts = pd.date_range("2024-01-01 00:00", periods=8, freq="15min")
        pd.DataFrame({"timestamp": seed_ts, "gross_kwh": 0.5}).to_csv(cache, index=False)

        ts = pd.date_range("2024-01-01", periods=200, freq="15min").tz_localize("Europe/Zurich")
        readings = pd.DataFrame({"timestamp": ts, "value": list(range(200))})
        mock_fetch.return_value = readings
        mock_app = MagicMock()

        ha_data.fetch_recent_energy_15m(mock_app, "sensor.energy", cache_path=cache)

        saved = pd.read_csv(cache)
        assert len(saved) > 8

    @patch("energy_forecast.ha_data._fetch_history")
    def test_no_error_on_both_empty(self, mock_fetch, tmp_path):
        """Unlike fetch_energy_history_15m, the recent variant logs and returns None."""
        import pandas as pd

        mock_fetch.return_value = pd.DataFrame()
        mock_app = MagicMock()
        cache = tmp_path / "energy_history_15m.csv"

        ha_data.fetch_recent_energy_15m(mock_app, "sensor.energy", cache_path=cache)

    @patch("energy_forecast.ha_data._fetch_history")
    def test_trailing_sensor_silence_backfilled_through_now(self, mock_fetch, tmp_path):
        """Same trailing-silence bug as the hourly path, on the recent-fetch
        15-minute cache updater (fire-and-forget from _update_sensors())."""
        import pandas as pd

        cache = tmp_path / "energy_history_15m.csv"
        now_local = pd.Timestamp.now(tz="Europe/Zurich")
        last_real_slot = (now_local - pd.Timedelta(hours=1)).floor("15min")
        ts = pd.to_datetime(
            [
                (last_real_slot - pd.Timedelta(minutes=15)).tz_convert("UTC"),
                last_real_slot.tz_convert("UTC"),
            ]
        ).tz_convert("Europe/Zurich")
        mock_fetch.return_value = pd.DataFrame({"timestamp": ts, "value": [50.0, 50.5]})
        mock_app = MagicMock()

        ha_data.fetch_recent_energy_15m(mock_app, "sensor.energy", cache_path=cache)

        saved = pd.read_csv(cache)
        saved_ts = set(pd.to_datetime(saved["timestamp"]))
        expected = (last_real_slot + pd.Timedelta(minutes=15)).tz_localize(None)
        assert expected in saved_ts, f"slot {expected} missing — trailing silence wasn't backfilled"
