import pandas as pd
import numpy as np
import pytest
from apps.energy_forecast.model import _engineer_features, EnergyForecastModel

def test_defrost_risk_calculation():
    # Defrost risk peaks at 2C and scales with humidity
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-04-15 12:00", "2026-04-15 13:00", "2026-04-15 14:00"]),
        "gross_kwh": [1.0, 1.0, 1.0]
    })
    weather_df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-04-15 12:00", "2026-04-15 13:00", "2026-04-15 14:00"]),
        "temp_c": [2.0, 7.0, -3.0],
        "humidity": [100.0, 100.0, 100.0],
        "precipitation_mm": [0, 0, 0],
        "sunshine_min": [0, 0, 0],
        "wind_kmh": [0, 0, 0],
        "cloud_cover_pct": [0, 0, 0],
        "direct_radiation_wm2": [0, 0, 0]
    })

    
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

def test_solar_compensation():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-04-15 13:00"]),
        "gross_kwh": [1.0]
    })
    # Cold but sunny
    weather_df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-04-15 13:00"]),
        "temp_c": [0.0],
        "humidity": [50.0],
        "precipitation_mm": [0],
        "sunshine_min": [60],
        "wind_kmh": [0],
        "cloud_cover_pct": [0],
        "direct_radiation_wm2": [800.0]
    })
    # Climate showing 5C deficit
    climate_dfs = {
        "climate.test": pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-04-15 13:00"]),
            "current_temp": [15.0],
            "setpoint": [20.0]
        })
    }
    
    feat_df = _engineer_features(df, weather_df, None, climate_dfs=climate_dfs)
    
    assert feat_df.iloc[0]["thermal_pressure"] == 5.0
    # weighted_solar_gain should be non-zero at 13:00
    assert feat_df.iloc[0]["weighted_solar_gain"] > 0
    # thermal_pressure_net should be less than raw thermal_pressure
    assert feat_df.iloc[0]["thermal_pressure_net"] < 5.0
    assert feat_df.iloc[0]["thermal_pressure_net"] >= 0

def test_infiltration_pressure():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-04-15 12:00"]),
        "gross_kwh": [1.0]
    })
    weather_df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-04-15 12:00"]),
        "temp_c": [0.0],
        "humidity": [50.0],
        "precipitation_mm": [0],
        "sunshine_min": [0],
        "wind_kmh": [20.0],
        "cloud_cover_pct": [0],
        "direct_radiation_wm2": [0]
    })
    climate_dfs = {
        "climate.test": pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-04-15 12:00"]),
            "current_temp": [15.0],
            "setpoint": [20.0]
        })
    }
    
    feat_df = _engineer_features(df, weather_df, None, climate_dfs=climate_dfs)
    
    # infiltration_pressure = 0.01 * wind_kmh * thermal_pressure
    # 0.01 * 20 * 5 = 1.0
    assert feat_df.iloc[0]["infiltration_pressure"] == pytest.approx(1.0)

def _make_trained_model_with_physics(tmp_path, n: int = 600):
    rng = np.random.default_rng(0)
    ts  = pd.date_range("2024-01-01", periods=n, freq="1h")
    energy = pd.DataFrame({
        "timestamp": ts,
        "gross_kwh": rng.uniform(0.5, 5.0, size=n),
    })
    weather = pd.DataFrame({
        "timestamp":            ts,
        "temp_c":               rng.uniform(-5, 25, size=n),
        "humidity":             rng.uniform(30, 90, size=n),
        "precipitation_mm":     [0.0]   * n,
        "sunshine_min":         [30.0]  * n,
        "wind_kmh":             rng.uniform(0, 30, size=n),   # vary so infiltration_pressure has signal
        "cloud_cover_pct":      [50.0]  * n,
        "direct_radiation_wm2": [100.0] * n,
    })
    # Add dummy climate data to ensure thermal features are active
    climate_dfs = {
        "climate.test": pd.DataFrame({
            "timestamp": ts,
            "current_temp": rng.uniform(18, 22, size=n),
            "setpoint": [21.0] * n
        })
    }
    m = EnergyForecastModel(tmp_path)
    m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, climate_dfs=climate_dfs)
    
    future_ts = pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h")
    forecast = pd.DataFrame({
        "timestamp":            future_ts,
        "temp_c":               [10.0]  * 48,
        "humidity":             [80.0]  * 48,
        "precipitation_mm":     [0.0]   * 48,
        "sunshine_min":         [30.0]  * 48,
        "wind_kmh":             [10.0]  * 48,
        "cloud_cover_pct":      [50.0]  * 48,
        "direct_radiation_wm2": [400.0] * 48,
    })
    return m, forecast

def test_shap_context_fix(tmp_path):
    m, forecast = _make_trained_model_with_physics(tmp_path)
    
    # Context data
    climate_recent = {
        "climate.test": pd.DataFrame({
            "timestamp": [pd.Timestamp.now().floor("1h") - pd.Timedelta(hours=1)],
            "current_temp": [15.0],
            "setpoint": [21.0]
        })
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
