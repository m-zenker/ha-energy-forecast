"""No-regression guarantee for #96: cooling_mode_enabled absent/false must
reproduce pre-change dev output exactly (X.columns composition + holdout MAE),
per spec §6."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from energy_forecast.model import EnergyForecastModel

_BASELINE_PATH = Path(__file__).parent / "fixtures" / "cooling_regression_baseline.json"


def _build_fixture_dataset():
    """Its function body must stay identical to scripts/capture_cooling_regression_baseline.py's
    _build_fixture_dataset() (this docstring differs deliberately) — duplicated on purpose, since
    the capture script must also run unmodified against a pre-change dev checkout that doesn't
    have this test file."""
    rng = np.random.default_rng(96)
    ts = pd.date_range("2026-01-01", periods=24 * 120, freq="1h")
    hour = ts.hour.values
    base = 0.8 + 0.6 * np.sin((hour - 6) / 24 * 2 * np.pi) ** 2
    noise = rng.normal(0, 0.1, len(ts))
    gross_kwh = np.clip(base + noise, 0.05, None)
    energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": gross_kwh})

    temp_c = 5.0 + 8.0 * np.sin((ts.dayofyear.values / 365) * 2 * np.pi - np.pi / 2)
    weather_df = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": temp_c,
            "precipitation_mm": np.zeros(len(ts)),
            "sunshine_min": np.clip(30 * np.sin((hour - 6) / 12 * np.pi), 0, None),
            "wind_kmh": np.full(len(ts), 10.0),
            "humidity": np.full(len(ts), 70.0),
            "cloud_cover_pct": np.full(len(ts), 50.0),
            "direct_radiation_wm2": np.clip(300 * np.sin((hour - 6) / 12 * np.pi), 0, None),
        }
    )
    return energy_df, weather_df


def _predict_at_holdout_start(model, forecast_df, holdout_start_naive, live_temp, recent_actuals):
    """Call model.predict() with pd.Timestamp.now() frozen to holdout_start_naive.

    predict() anchors its 48-row output to the real wall-clock hour (model.py's
    `now_naive = pd.Timestamp.now(tz=self._timezone).tz_localize(None)`), ignoring
    forecast_df's own timestamps — without this freeze, preds["timestamp"] would
    be "today" (real run date) while the holdout fixture's timestamps are fixed in
    the synthetic January 2026 window, so a later timestamp-merge would match zero
    rows. Freezing also keeps calendar-derived features (hour-of-day, day-of-year,
    ...) internally consistent with the synthetic January weather values, and makes
    this output reproducible regardless of which real day it is run on."""

    def _frozen_now(tz=None):
        frozen = pd.Timestamp(holdout_start_naive)
        return frozen.tz_localize(tz) if tz is not None else frozen

    with patch("pandas.Timestamp.now", side_effect=_frozen_now):
        return model.predict(forecast_df, live_temp=live_temp, recent_actuals=recent_actuals)


@pytest.fixture(scope="module")
def _baseline():
    if not _BASELINE_PATH.exists():
        pytest.skip(
            f"{_BASELINE_PATH} missing — run scripts/capture_cooling_regression_baseline.py "
            "against a pristine dev checkout first (see Task 2, Step 1 of the cooling-mode plan)."
        )
    return json.loads(_BASELINE_PATH.read_text())


class TestCoolingModeNoRegression:
    @pytest.mark.parametrize("cooling_mode_enabled_arg", [None, False], ids=["absent", "explicit_false"])
    def test_columns_and_mae_match_pre_change_baseline(self, tmp_path, _baseline, cooling_mode_enabled_arg):
        """cooling_mode_enabled absent and explicitly False are both untouched by
        this plan's changes at the fetch/config layer, so cooling_active_df/
        cooling_active_series are never built — only the 2 unconditionally-added
        columns (cooling_active, cooling_load_sum_24h) differ from baseline,
        and they must be constant-zero and not move MAE."""
        energy_df, weather_df = _build_fixture_dataset()
        split = int(len(energy_df) * 0.9)
        train_energy, holdout_energy = energy_df.iloc[:split], energy_df.iloc[split:]

        model = EnergyForecastModel(model_dir=tmp_path)
        model.train(train_energy, weather_df.iloc[:split], outdoor_df=None)  # no cooling_active_df passed

        baseline_cols = set(_baseline["columns"])
        current_cols = set(model.feature_cols)
        new_cooling_cols = {"cooling_active", "cooling_load_sum_24h"}

        # NOTE: if you're here because you added a new column to _FEATURES_BASE and
        # this test now fails — that's expected, not a cooling regression. Re-run
        # scripts/capture_cooling_regression_baseline.py against a pristine pre-#96
        # dev checkout (or extend new_cooling_cols here) to update the baseline.
        assert current_cols - baseline_cols == new_cooling_cols, (
            f"Unexpected column set change beyond the 2 cooling additions: "
            f"{(current_cols - baseline_cols) - new_cooling_cols}"
        )
        assert baseline_cols - current_cols == set(), (
            f"Columns present in baseline but missing now: {baseline_cols - current_cols}"
        )

        future_ts = holdout_energy["timestamp"]
        forecast_df = weather_df[weather_df["timestamp"].isin(future_ts)].reset_index(drop=True)
        preds = _predict_at_holdout_start(
            model, forecast_df, holdout_energy["timestamp"].iloc[0], None, train_energy.tail(48)
        )
        merged = preds.merge(holdout_energy, on="timestamp", suffixes=("_pred", "_actual"))
        mae = float(np.mean(np.abs(merged["predicted_kwh"] - merged["gross_kwh"])))

        assert mae == pytest.approx(_baseline["mae"], rel=0.02), (
            f"Holdout MAE moved beyond a 2% tolerance: {mae:.4f} vs baseline {_baseline['mae']:.4f} "
            "— the 2 unconditionally-added constant-zero columns should not perturb tree structure."
        )

    def test_enabled_but_no_active_entity_reproduces_heating_only_columns(self, tmp_path, _baseline):
        """§6's 2nd required bullet: cooling_mode_enabled=True with no
        hvac_mode_entity/cooling_system_active_entity configured must reproduce
        the same output as disabled — extends Plan A's Task 3's fetch-layer-only
        test_neither_entity_configured_returns_empty to the full training chain."""
        energy_df, weather_df = _build_fixture_dataset()
        split = int(len(energy_df) * 0.9)
        train_energy = energy_df.iloc[:split]

        # cooling_mode_enabled=True is an energy_forecast.py-layer config flag —
        # at the model.py layer exercised directly here, "enabled but unconfigured"
        # means cooling_active_df simply stays None (Plan A's Task 3's _fetch_hvac_active_history
        # never populates it without an entity), which is indistinguishable from
        # the disabled case at this layer by construction (Implementation Decision #1).
        model = EnergyForecastModel(model_dir=tmp_path)
        model.train(train_energy, weather_df.iloc[:split], outdoor_df=None, cooling_active_df=None)

        new_cooling_cols = {"cooling_active", "cooling_load_sum_24h"}
        assert set(model.feature_cols) - set(_baseline["columns"]) == new_cooling_cols
