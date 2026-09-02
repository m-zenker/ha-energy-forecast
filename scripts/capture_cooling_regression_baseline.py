"""One-time baseline capture for the cooling-mode no-regression test (#96).

Run this script from a pristine pre-cooling-change checkout (the `dev` branch,
before Plan A's Task 1 of the cooling-mode-ac-support plan set
(docs/superpowers/plans/2026-09-01-cooling-mode-ac-support-index.md) lands) to
produce tests/fixtures/cooling_regression_baseline.json. The no-regression
test (tests/test_cooling_regression.py) then compares current code's output
against this frozen baseline — it does NOT re-run this script.

Usage (from the repo root, dedicated env active):
    python scripts/capture_cooling_regression_baseline.py
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

import numpy as np
import pandas as pd
from energy_forecast.model import EnergyForecastModel


def _build_fixture_dataset():
    """Deterministic synthetic heating-season dataset — no climate/cooling
    entities configured, matching the current install base's typical shape."""
    rng = np.random.default_rng(96)
    ts = pd.date_range("2026-01-01", periods=24 * 120, freq="1h")  # 120 days
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


def main():
    energy_df, weather_df = _build_fixture_dataset()
    split = int(len(energy_df) * 0.9)
    train_energy, holdout_energy = energy_df.iloc[:split], energy_df.iloc[split:]

    model = EnergyForecastModel(model_dir=Path("/tmp/cooling_baseline_model"))
    model.train(train_energy, weather_df.iloc[:split], outdoor_df=None)

    future_ts = holdout_energy["timestamp"]
    forecast_df = weather_df[weather_df["timestamp"].isin(future_ts)].reset_index(drop=True)
    preds = _predict_at_holdout_start(
        model, forecast_df, holdout_energy["timestamp"].iloc[0], None, train_energy.tail(48)
    )
    merged = preds.merge(holdout_energy, on="timestamp", suffixes=("_pred", "_actual"))
    mae = float(np.mean(np.abs(merged["predicted_kwh"] - merged["gross_kwh"])))

    baseline = {"columns": sorted(model.feature_cols), "mae": mae}
    out_path = Path(__file__).parent.parent / "tests" / "fixtures" / "cooling_regression_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2))
    print(f"Wrote baseline: {len(baseline['columns'])} columns, MAE={mae:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
