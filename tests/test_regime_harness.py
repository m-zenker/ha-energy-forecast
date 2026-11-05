"""Tests for scripts/regime_harness.py (#110 Part C): live-equivalent loader + prediction inputs."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rh():
    path = _REPO_ROOT / "scripts" / "regime_harness.py"
    if not path.exists():
        pytest.skip("scripts/regime_harness.py not present in this checkout")
    spec = importlib.util.spec_from_file_location("regime_harness", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["regime_harness"] = mod
    spec.loader.exec_module(mod)
    yield mod
    del sys.modules["regime_harness"]


def _weather(start: str, hours: int) -> pd.DataFrame:
    ts = pd.date_range(start, periods=hours, freq="1h")
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": 5 + rng.uniform(-3, 3, hours),
            "precipitation_mm": 0.0,
            "sunshine_min": 20.0,
            "wind_kmh": 8.0,
            "cloud_cover_pct": 50.0,
            "direct_radiation_wm2": 80.0,
            "humidity": 70.0,
        }
    )


@pytest.fixture
def data_dir(tmp_path):
    """Tiny stand-in for the pulled HA data/: 120 h of energy history + a minimal apps.yaml."""
    d = tmp_path / "data"
    d.mkdir()
    ts = pd.date_range("2026-01-01 00:00", periods=120, freq="1h")
    pd.DataFrame({"timestamp": ts, "gross_kwh": 0.5 + 0.1 * (ts.hour % 6)}).to_csv(
        d / "energy_history.csv", index=False
    )
    cfg = {
        "energy_forecast": {
            "energy_sensor": "sensor.grid_import",
            "latitude": 47.3,
            "longitude": 8.5,
            "timezone": "Europe/Zurich",
            "enable_regimes": True,
            "regime_count": 3,
        }
    }
    (d / "apps.yaml").write_text(yaml.safe_dump(cfg))
    return d


def test_trim_caches_drops_future_rows(rh, tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01 00:00:00", "2026-01-01 05:00:00", "2026-01-01 06:00:00"],
            "kwh": ["1.0", "2.0", "3.0"],
        }
    ).to_csv(src / "a.csv", index=False)
    # tz-aware strings are compared in local time: 04:00+00:00 is 05:00 Zurich (CET), kept at cutoff 05:00
    pd.DataFrame({"timestamp": ["2026-01-01 04:00:00+00:00", "2026-01-01 05:00:00+00:00"], "v": ["x", "y"]}).to_csv(
        src / "b.csv", index=False
    )
    rh.trim_caches(src, dst, pd.Timestamp("2026-01-01 05:00"))
    assert list(pd.read_csv(dst / "a.csv")["kwh"]) == [1.0, 2.0]
    assert list(pd.read_csv(dst / "b.csv")["v"]) == ["x"]


def test_trim_caches_copies_non_timestamp_files(rh, tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "cfg.json").write_text('{"a": 1}')
    (src / "excluded_ranges.csv").write_text("start,end\n2026-01-01,2026-01-02\n")
    (src / "model.pkl").write_bytes(b"\x00")
    (src / "sub").mkdir()
    rh.trim_caches(src, dst, pd.Timestamp("2020-01-01"))
    assert (dst / "cfg.json").read_text() == '{"a": 1}'
    assert (dst / "excluded_ranges.csv").read_text() == "start,end\n2026-01-01,2026-01-02\n"
    assert not (dst / "model.pkl").exists() and not (dst / "sub").exists()


def test_weather_stub_clips_to_cutoff(rh):
    cutoff = pd.Timestamp("2026-01-03 12:00")
    stub = rh.WeatherStub(_weather("2026-01-01 00:00", 24 * 5), cutoff)
    hist = stub.historical(47.0, 8.0, pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-04").date())
    assert hist["timestamp"].min() == pd.Timestamp("2026-01-02 00:00")
    assert hist["timestamp"].max() == cutoff  # end date is inclusive, but never past the cutoff
    recent = stub.open_meteo(47.0, 8.0, past_days=1)
    assert recent["timestamp"].min() == pd.Timestamp("2026-01-02 00:00")
    assert recent["timestamp"].max() == cutoff


def test_loader_captures_train_kwargs(rh, data_dir, tmp_path):
    cutoff = pd.Timestamp("2026-01-04 12:30")
    captured = rh.load_live_equivalent_training_frame(
        cutoff, data_dir, tmp_path / "scratch", archive=_weather("2025-12-28 00:00", 24 * 14)
    )
    energy_df, weather_df = captured["_args"][:2]
    assert captured["enable_regimes"] is True and captured["regime_count"] == 3
    assert energy_df["timestamp"].max() <= cutoff
    assert energy_df["timestamp"].max() < cutoff.floor("1h")  # the open hourly bucket is never trained on
    assert weather_df["timestamp"].max() <= cutoff


def test_build_prediction_inputs_has_no_future_actuals(rh, data_dir, tmp_path):
    origin = pd.Timestamp("2026-01-04 17:00")
    inputs = rh.build_prediction_inputs(
        origin, data_dir, tmp_path / "scratch_o", archive=_weather("2025-12-28 00:00", 24 * 14)
    )
    assert inputs["recent_actuals"]["timestamp"].max() < origin
    fc = inputs["forecast_df"]
    assert fc["timestamp"].max() == origin.floor("1h") + pd.Timedelta(hours=47)
    assert inputs["live_temp"] is not None
