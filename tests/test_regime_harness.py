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
            "physics": {},
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


# ── Gate maths (#110 Part C2) ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def br(rh):
    path = _REPO_ROOT / "scripts" / "backtest_regime.py"
    if not path.exists():
        pytest.skip("scripts/backtest_regime.py not present in this checkout")
    spec = importlib.util.spec_from_file_location("backtest_regime", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["backtest_regime"] = mod
    spec.loader.exec_module(mod)
    yield mod
    del sys.modules["backtest_regime"]


def _toy_energy(n_days: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    ts = pd.date_range("2026-01-01", periods=24 * n_days, freq="1h")
    base = np.array([[0.3] * 24, [0.3] * 7 + [2.0] * 3 + [0.4] * 14, [0.4] * 17 + [2.5] * 4 + [0.4] * 3])
    kwh = [base[(t.date().toordinal()) % 3][t.hour] + rng.uniform(-0.05, 0.05) for t in ts]
    return pd.DataFrame({"timestamp": ts, "gross_kwh": kwh})


def test_nearest_centroid_labels_matches_fit(br):
    from energy_forecast.clustering import SKLEARN_AVAILABLE, DailyProfileClusterer

    if not SKLEARN_AVAILABLE:
        pytest.skip("scikit-learn not available")
    df = _toy_energy()
    c = DailyProfileClusterer(n_clusters=3)
    fit_labels = c.fit(df)
    out = br.nearest_centroid_labels(df, c.centroids)
    pd.testing.assert_series_equal(out, fit_labels, check_names=False, check_dtype=False)


def test_label_agreement_permutation_invariant(br):
    live_c = np.array([[1.0] * 24, [5.0] * 24, [9.0] * 24])
    perm = [2, 0, 1]  # harness IDs are a relabelling of the same regimes
    harness_c = live_c[perm]
    days = pd.Index(
        [pd.Timestamp("2026-01-01").date(), pd.Timestamp("2026-01-02").date(), pd.Timestamp("2026-01-03").date()]
    )
    live_labels = pd.Series([0, 1, 2], index=days)
    harness_labels = pd.Series([perm.index(0), perm.index(1), perm.index(2)], index=days)
    assert br.label_agreement(live_c, harness_c, live_labels, harness_labels) == 1.0
    # one day genuinely differs -> 2/3
    harness_labels.iloc[0] = perm.index(1)
    assert br.label_agreement(live_c, harness_c, live_labels, harness_labels) == pytest.approx(2 / 3)


def test_gate_thresholds(br):
    ok = {"k_live": 4, "k_harness": 4, "agreement": 0.90, "max_rel_pred_diff": 0.03}
    assert br.gate(ok) == (True, [])
    assert br.gate({**ok, "agreement": 0.899})[0] is False
    assert br.gate({**ok, "max_rel_pred_diff": 0.0301})[0] is False
    passed, reasons = br.gate({**ok, "k_harness": 3})
    assert passed is False and any("k differs" in r for r in reasons)


def test_coerce_numeric_objects_restores_float_only_when_numeric(rh):
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2026-01-01", periods=3, freq="1h"), "t": pd.Series([1.0, None, 3.0], dtype=object)}
    )
    df["program"] = pd.Series(["eco", None, "fast"], dtype=object)
    out = rh.coerce_numeric_objects({"a": df})["a"]
    assert out["t"].dtype == "float64" and np.isnan(out["t"].iloc[1])
    assert out["program"].dtype == object and list(out["program"].dropna()) == ["eco", "fast"]


def test_physics_state_stays_out_of_repo_models_dir(rh, data_dir, tmp_path):
    """initialize() hard-codes apps/energy_forecast/models for the physics model; the harness must redirect it."""
    repo_models = _REPO_ROOT / "apps" / "energy_forecast" / "models"
    before = sorted(p.name for p in repo_models.iterdir()) if repo_models.exists() else []
    scratch = tmp_path / "scratch_p"
    captured = rh.load_live_equivalent_training_frame(
        pd.Timestamp("2026-01-04 12:30"), data_dir, scratch, archive=_weather("2025-12-28 00:00", 24 * 14)
    )
    physics = captured["physics_model"]
    assert physics is not None
    assert physics._calibration_path.parent == scratch / "models"
    after = sorted(p.name for p in repo_models.iterdir()) if repo_models.exists() else []
    assert after == before


def test_physics_state_seeded_only_on_request(rh, data_dir, tmp_path):
    (data_dir / "models").mkdir()
    (data_dir / "models" / "physics_calibration.json").write_text('{"Q_dhw_daily": 4.25}')
    arch = _weather("2025-12-28 00:00", 24 * 14)
    cut = pd.Timestamp("2026-01-04 12:30")
    plain = rh.load_live_equivalent_training_frame(cut, data_dir, tmp_path / "s1", archive=arch)
    seeded = rh.load_live_equivalent_training_frame(
        cut, data_dir, tmp_path / "s2", archive=arch, seed_physics_state=True
    )
    assert plain["physics_model"]._calib.get("Q_dhw_daily") != 4.25
    assert seeded["physics_model"]._calib.get("Q_dhw_daily") == 4.25


# ── Decision rule (#110 Part C3) ──────────────────────────────────────────────


def _arm(pooled=1.0, primary=1.0, jan=1.0, max_delta=5.0, n_over_2=4):
    return {"pooled_mae": pooled, "primary_mae": primary, "jan_mae": jan, "max_delta": max_delta, "n_over_2": n_over_2}


def _res(**arms):
    r = {"H": _arm(), **arms}
    return r


def test_decide_t_when_both_pass_and_n_not_3pct_better(br):
    r = _res(T=_arm(max_delta=2.0, n_over_2=1), N=_arm(pooled=0.98, jan=0.97, max_delta=2.0, n_over_2=1))
    assert br.decide(r) == "T"


def test_decide_n_when_3pct_pooled(br):
    r = _res(T=_arm(max_delta=2.0, n_over_2=1), N=_arm(pooled=0.96, max_delta=1.0, n_over_2=0))
    assert br.decide(r) == "N"


def test_decide_n_when_5pct_january(br):
    r = _res(T=_arm(max_delta=2.0, n_over_2=1), N=_arm(jan=0.94, max_delta=1.0, n_over_2=0))
    assert br.decide(r) == "N"


def test_decide_only_passing_candidate(br):
    r = _res(T=_arm(pooled=1.10, max_delta=2.0, n_over_2=1), N=_arm(max_delta=2.0, n_over_2=1))
    assert br.decide(r) == "N"
    r = _res(T=_arm(max_delta=2.0, n_over_2=1), N=_arm(max_delta=9.0))
    assert br.decide(r) == "T"


def test_decide_h_when_none_pass(br):
    r = _res(T=_arm(primary=1.20, max_delta=2.0), N=_arm(max_delta=6.0))
    assert br.decide(r) == "H"


def test_decide_without_t_arm(br):
    assert br.decide(_res(N=_arm(max_delta=2.0, n_over_2=1))) == "N"
    assert br.decide(_res()) == "H"


def test_sensitivity_requires_strictly_lower_max(br):
    assert br.decide(_res(T=_arm(max_delta=5.0, n_over_2=1))) == "H"  # equal max does not pass
    assert br.decide(_res(T=_arm(max_delta=4.99, n_over_2=1))) == "T"


def test_count_over_2_allows_equal(br):
    h = {"H": _arm(n_over_2=0), "T": _arm(max_delta=1.0, n_over_2=0)}
    assert br.decide(h) == "T"
    h["T"]["n_over_2"] = 1
    assert br.decide(h) == "H"


def test_decide_ignores_stability_gain_within_jitter(br):
    r = _res(T=_arm(max_delta=4.6, n_over_2=1))  # 8 % lower max delta
    assert br.decide(r) == "T"
    r["J"] = _arm(max_delta=5.5)  # H retrained 1 h earlier moves max delta by 10 %: the gain is noise
    assert br.decide(r) == "H"


def test_decide_n_must_beat_t_by_more_than_jitter(br):
    r = _res(T=_arm(max_delta=2.0, n_over_2=1), N=_arm(pooled=0.96, max_delta=1.0, n_over_2=0))
    assert br.decide(r) == "N"
    r["J"] = _arm(pooled=1.05, max_delta=5.0)  # 5 % pooled jitter swamps N's 4 % edge
    assert br.decide(r) == "T"


# ── Backtest helpers (#110 Part C3) ───────────────────────────────────────────


def test_weekly_cutoffs_and_origins_cover_every_origin_once(br):
    first, last = pd.Timestamp("2026-10-14"), pd.Timestamp("2026-11-03")
    cuts = br.weekly_cutoffs(first, last)
    assert [c.strftime("%m-%d %H") for c in cuts] == ["10-14 06", "10-21 06", "10-28 06"]
    origins = [o for c in cuts for o in br.origins_for(c, last)]
    assert len(origins) == len(set(origins)) == 21  # 21 target days, each served by exactly one model
    assert origins[0] == pd.Timestamp("2026-10-14 17:00") and origins[-1] == pd.Timestamp("2026-11-03 17:00")
    assert all(o >= c for c in cuts for o in br.origins_for(c, last))  # a model never serves a day before its cutoff


def test_arm_cutoff_jitter_control_is_one_hour_earlier(br):
    c = pd.Timestamp("2026-10-14 06:00")
    assert br.arm_cutoff("H", c) == c and br.arm_cutoff("T", c) == c
    assert br.arm_cutoff("J", c) == c - pd.Timedelta(hours=1)


def test_scale_sunshine_only_touches_future_hours(br):
    origin = pd.Timestamp("2026-01-04 17:00")
    fc = _weather("2026-01-01 17:00", 24 * 6)
    out = br.scale_sunshine({"forecast_df": fc, "x": 1}, origin, 0.5)
    ts = pd.to_datetime(out["forecast_df"]["timestamp"])
    assert (out["forecast_df"].loc[ts < origin, "sunshine_min"] == 20.0).all()
    assert (out["forecast_df"].loc[ts >= origin, "sunshine_min"] == 10.0).all()
    assert (fc["sunshine_min"] == 20.0).all()  # input frame untouched
    assert out["x"] == 1


def test_fold_metrics(br):
    df = pd.DataFrame(
        {
            "fold": ["primary", "primary", "jan", "primary"],
            "base": [10.0, 12.0, 8.0, 9.0],
            "lo": [9.0, 12.5, 5.0, 9.0],
            "hi": [11.0, 12.0, 8.5, 9.0],
            "actual": [11.0, 10.0, 9.0, np.nan],  # last row is not scoreable
        }
    )
    m = br.fold_metrics(df)
    assert m["n_scored"] == 3
    assert m["primary_mae"] == pytest.approx(1.5) and m["jan_mae"] == pytest.approx(1.0)
    assert m["pooled_mae"] == pytest.approx(4 / 3)
    assert m["max_delta"] == pytest.approx(3.0)  # jan row: |5.0 - 8.0|
    assert m["n_over_2"] == 1
    assert m["mean_bias"] == pytest.approx((-1 + 2 - 1) / 3)


def test_prune_unavailable_sensors_drops_sensors_without_rows_at_cutoff(rh, tmp_path):
    ts = ["2026-01-01 00:00:00", "2026-01-01 01:00:00"]
    pd.DataFrame({"timestamp": ts, "kwh": [1.0, 2.0]}).to_csv(tmp_path / "sub_washer.csv", index=False)
    pd.DataFrame({"timestamp": [], "kwh": []}).to_csv(tmp_path / "sub_dryer.csv", index=False)  # trimmed to empty
    pd.DataFrame({"timestamp": ts, "current_temp": [20.0, 20.0]}).to_csv(tmp_path / "climate_a.csv", index=False)
    cfg = {
        "energy_sensor": "sensor.x",
        "sub_energy_sensors": ["sensor.washer", {"entity_id": "sensor.dryer", "program_sensor": "sensor.p"}],
        "climate_entities": ["climate.a", "climate.b"],  # b has no cache file at all
    }
    out = rh.prune_unavailable_sensors(cfg, tmp_path)
    assert out["sub_energy_sensors"] == ["sensor.washer"]
    assert out["climate_entities"] == ["climate.a"]
    assert cfg["sub_energy_sensors"][1]["entity_id"] == "sensor.dryer"  # input config untouched
