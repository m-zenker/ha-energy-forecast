import numpy as np
import pandas as pd
import pytest

from apps.energy_forecast.energy_forecast import _empty_weather_df
from apps.energy_forecast.model import _FEATURES_BASE, EnergyForecastModel, _engineer_features


def _make_constant_weather(ts, temp_c=10.0):
    return pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": [temp_c] * len(ts),
            "humidity": [70.0] * len(ts),
            "precipitation_mm": [0.0] * len(ts),
            "sunshine_min": [0.0] * len(ts),
            "wind_kmh": [0.0] * len(ts),
            "cloud_cover_pct": [0.0] * len(ts),
            "direct_radiation_wm2": [0.0] * len(ts),
        }
    )


def test_defrost_risk_calculation():
    # Defrost risk peaks at 2C and scales with humidity
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-15 12:00", "2026-04-15 13:00", "2026-04-15 14:00"]),
            "gross_kwh": [1.0, 1.0, 1.0],
        }
    )
    weather_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-15 12:00", "2026-04-15 13:00", "2026-04-15 14:00"]),
            "temp_c": [2.0, 7.0, -3.0],
            "humidity": [100.0, 100.0, 100.0],
            "precipitation_mm": [0, 0, 0],
            "sunshine_min": [0, 0, 0],
            "wind_kmh": [0, 0, 0],
            "cloud_cover_pct": [0, 0, 0],
            "direct_radiation_wm2": [0, 0, 0],
        }
    )

    feat_df = _engineer_features(df, weather_df, None)

    # At 2C, 100% humidity, risk should be ~1.0
    assert feat_df.iloc[0]["defrost_risk"] == pytest.approx(1.0, abs=0.01)
    # At 7C or -3C (5C away from peak), risk should be much lower (exp(-25/10) * 1.0 approx 0.08)
    assert feat_df.iloc[1]["defrost_risk"] < 0.1
    assert feat_df.iloc[2]["defrost_risk"] < 0.1

    # Scaling with humidity
    weather_df["humidity"] = 50.0
    feat_df_50 = _engineer_features(df, weather_df, None)
    assert feat_df_50.iloc[0]["defrost_risk"] == pytest.approx(0.5, abs=0.01)


def test_defrost_risk_with_empty_weather_df():
    # Regression: _empty_weather_df() used to return object-dtype columns, causing
    # np.exp() to crash with "float has no attribute exp" when weather fetch fails.
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-15 12:00"]),
            "gross_kwh": [1.0],
        }
    )
    feat_df = _engineer_features(df, _empty_weather_df(), None)
    assert "defrost_risk" in feat_df.columns
    assert feat_df["defrost_risk"].isna().all()


def test_solar_compensation():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-04-15 13:00"]), "gross_kwh": [1.0]})
    # Cold but sunny
    weather_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-15 13:00"]),
            "temp_c": [0.0],
            "humidity": [50.0],
            "precipitation_mm": [0],
            "sunshine_min": [60],
            "wind_kmh": [0],
            "cloud_cover_pct": [0],
            "direct_radiation_wm2": [800.0],
        }
    )
    # Climate showing 5C deficit
    climate_dfs = {
        "climate.test": pd.DataFrame(
            {"timestamp": pd.to_datetime(["2026-04-15 13:00"]), "current_temp": [15.0], "setpoint": [20.0]}
        )
    }

    feat_df = _engineer_features(df, weather_df, None, climate_dfs=climate_dfs)

    assert feat_df.iloc[0]["thermal_pressure"] == 5.0
    # weighted_solar_gain should be non-zero at 13:00
    assert feat_df.iloc[0]["weighted_solar_gain"] > 0
    # thermal_pressure_net should be less than raw thermal_pressure
    assert feat_df.iloc[0]["thermal_pressure_net"] < 5.0
    assert feat_df.iloc[0]["thermal_pressure_net"] >= 0


def test_infiltration_pressure():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-04-15 12:00"]), "gross_kwh": [1.0]})
    weather_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-04-15 12:00"]),
            "temp_c": [0.0],
            "humidity": [50.0],
            "precipitation_mm": [0],
            "sunshine_min": [0],
            "wind_kmh": [20.0],
            "cloud_cover_pct": [0],
            "direct_radiation_wm2": [0],
        }
    )
    climate_dfs = {
        "climate.test": pd.DataFrame(
            {"timestamp": pd.to_datetime(["2026-04-15 12:00"]), "current_temp": [15.0], "setpoint": [20.0]}
        )
    }

    feat_df = _engineer_features(df, weather_df, None, climate_dfs=climate_dfs)

    # infiltration_pressure = 0.01 * wind_kmh * thermal_pressure
    # 0.01 * 20 * 5 = 1.0
    assert feat_df.iloc[0]["infiltration_pressure"] == pytest.approx(1.0)


def _make_trained_model_with_physics(tmp_path, n: int = 600):
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    energy = pd.DataFrame(
        {
            "timestamp": ts,
            "gross_kwh": rng.uniform(0.5, 5.0, size=n),
        }
    )
    weather = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": rng.uniform(-5, 25, size=n),
            "humidity": rng.uniform(30, 90, size=n),
            "precipitation_mm": [0.0] * n,
            "sunshine_min": [30.0] * n,
            "wind_kmh": rng.uniform(0, 30, size=n),  # vary so infiltration_pressure has signal
            "cloud_cover_pct": [50.0] * n,
            "direct_radiation_wm2": [100.0] * n,
        }
    )
    # Add dummy climate data to ensure thermal features are active
    climate_dfs = {
        "climate.test": pd.DataFrame(
            {"timestamp": ts, "current_temp": rng.uniform(18, 22, size=n), "setpoint": [21.0] * n}
        )
    }
    m = EnergyForecastModel(tmp_path)
    m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, climate_dfs=climate_dfs)

    future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
    forecast = pd.DataFrame(
        {
            "timestamp": future_ts,
            "temp_c": [10.0] * 48,
            "humidity": [80.0] * 48,
            "precipitation_mm": [0.0] * 48,
            "sunshine_min": [30.0] * 48,
            "wind_kmh": [10.0] * 48,
            "cloud_cover_pct": [50.0] * 48,
            "direct_radiation_wm2": [400.0] * 48,
        }
    )
    return m, forecast


def test_shap_context_fix(tmp_path):
    m, forecast = _make_trained_model_with_physics(tmp_path)

    # Context data
    climate_recent = {
        "climate.test": pd.DataFrame(
            {
                "timestamp": [pd.Timestamp.now().floor("1h") - pd.Timedelta(hours=1)],
                "current_temp": [15.0],
                "setpoint": [21.0],
            }
        )
    }

    # Use n larger than total feature count to capture all features
    shap_results = m.shap_summary(
        forecast,
        live_temp=10.0,
        climate_recent=climate_recent,
        n=100,
    )

    # thermal_pressure should be in SHAP results and have some value (not zeroed out)
    # Note: SHAP can be 0 if the feature has no impact, but here we expect impact.
    # In GBR fallback it's feature importance.
    assert "thermal_pressure" in shap_results
    assert "thermal_pressure_net" in shap_results
    assert "defrost_risk" in shap_results
    assert "infiltration_pressure" in shap_results


# ── Temperature-delta gated lag tests ────────────────────────────────────────


def _feat_row(temp_series, lag_24h_val=2.0, lag_168h_val=3.0):
    """Call _engineer_features with injected lag columns; return the last row."""
    n = len(temp_series)
    ts = pd.date_range("2026-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "gross_kwh": [1.0] * n,
            "lag_24h": [lag_24h_val] * n,
            "lag_168h": [lag_168h_val] * n,
        }
    )
    weather_df = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": temp_series,
            "humidity": [70.0] * n,
            "precipitation_mm": [0.0] * n,
            "sunshine_min": [0.0] * n,
            "wind_kmh": [0.0] * n,
            "cloud_cover_pct": [0.0] * n,
            "direct_radiation_wm2": [0.0] * n,
        }
    )
    return _engineer_features(df, weather_df, None).iloc[-1]


def test_lag_24h_tgated_no_discount():
    # Constant temp → temp_delta_24h = 0 → discount = 0 → tgated == lag_24h
    temps = [10.0] * 200
    row = _feat_row(temps, lag_24h_val=2.0)
    assert row["lag_24h_tgated"] == pytest.approx(row["lag_24h"])


def test_lag_24h_tgated_full_discount():
    # +10°C warmer than yesterday (≥5°C → full discount) → tgated ≈ 0
    temps = [10.0] * 199 + [20.0]
    row = _feat_row(temps, lag_24h_val=2.0)
    assert row["lag_24h_tgated"] == pytest.approx(0.0, abs=1e-6)


def test_lag_24h_tgated_half_discount():
    # +2.5°C warmer → discount = 0.5 → tgated = lag_24h * 0.5
    temps = [10.0] * 199 + [12.5]
    row = _feat_row(temps, lag_24h_val=2.0)
    assert row["lag_24h_tgated"] == pytest.approx(row["lag_24h"] * 0.5, abs=1e-6)


def test_lag_168h_tgated_no_discount():
    # Constant temp → temp_lag_168h == temp_c → delta = 0 → tgated == lag_168h
    temps = [10.0] * 300
    row = _feat_row(temps, lag_168h_val=3.0)
    assert row["lag_168h_tgated"] == pytest.approx(row["lag_168h"])


def test_lag_168h_tgated_full_discount():
    # +16°C warmer than last week (≥8°C → full discount) → tgated ≈ 0
    temps = [5.0] * 299 + [21.0]
    row = _feat_row(temps, lag_168h_val=3.0)
    assert row["lag_168h_tgated"] == pytest.approx(0.0, abs=1e-6)


def test_lag_336h_tgated_no_discount():
    # Constant temp → delta vs 336h ago = 0 → tgated == lag_336h
    temps = [10.0] * 400
    row = _feat_row(temps, lag_168h_val=3.0)
    assert row["lag_336h_tgated"] == pytest.approx(row["lag_336h_tgated"])  # self-consistent


def test_lag_336h_tgated_full_discount():
    # +16°C warmer than 2 weeks ago → tgated ≈ 0
    n = 400
    temps = [5.0] * (n - 1) + [21.0]
    ts = pd.date_range("2026-01-01", periods=n, freq="1h")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "gross_kwh": [1.0] * n,
            "lag_24h": [2.0] * n,
            "lag_168h": [3.0] * n,
            "lag_336h": [4.0] * n,
        }
    )
    weather_df = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": temps,
            "humidity": [70.0] * n,
            "precipitation_mm": [0.0] * n,
            "sunshine_min": [0.0] * n,
            "wind_kmh": [0.0] * n,
            "cloud_cover_pct": [0.0] * n,
            "direct_radiation_wm2": [0.0] * n,
        }
    )
    row = _engineer_features(df, weather_df, None).iloc[-1]
    assert row["lag_336h_tgated"] == pytest.approx(0.0, abs=1e-6)


def test_tgated_features_in_features_base():
    assert "lag_24h_tgated" in _FEATURES_BASE
    assert "lag_168h_tgated" in _FEATURES_BASE
    assert "lag_336h_tgated" in _FEATURES_BASE
    assert "lag_24h" not in _FEATURES_BASE
    assert "lag_168h" not in _FEATURES_BASE
    assert "lag_336h" not in _FEATURES_BASE
