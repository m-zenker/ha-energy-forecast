"""
ML model — feature engineering, training, prediction and disk persistence.

All CPU-bound work (training, batch prediction) is designed to be called
via hass.async_add_executor_job() so it never blocks the HA event loop.
"""
from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    _USE_LGBM = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[assignment]
    _USE_LGBM = False

from sklearn.metrics import mean_absolute_error

from .const import (
    MAX_HOURLY_KWH,
    SENSOR_BLEND_HOURS,
    SENSOR_FULL_TRUST_HOURS,
)

_LOGGER = logging.getLogger(__name__)

# ── Feature column sets ───────────────────────────────────────────────────────
_FEATURES_BASE = [
    "hour", "hour_block", "day_of_week", "month", "is_weekend", "season",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh",
    "heating_degree", "cooling_degree",
]
_FEATURES_WITH_SENSOR = _FEATURES_BASE + ["outdoor_temp_live", "temp_bias"]


class EnergyForecastModel:
    """Encapsulates training data, model weights and prediction logic."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = model_dir / "energy_model.pkl"
        self._meta_path = model_dir / "meta.pkl"

        self.model = None
        self.feature_cols: list[str] = _FEATURES_BASE
        self.last_trained: datetime = datetime.min
        self.last_mae: float | None = None
        self.engine: str = "LightGBM" if _USE_LGBM else "sklearn GBR"

        self._load()

    # ── Public API (all CPU-bound — call via executor_job) ────────────────────

    def train(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        outdoor_df: pd.DataFrame | None,
    ) -> None:
        """
        Train (or retrain) the model.

        energy_df  : columns [timestamp, gross_kwh]
        weather_df : columns [timestamp, temp_c, precipitation_mm, sunshine_min, wind_kmh]
        outdoor_df : columns [timestamp, outdoor_temp_live]  — may be None
        """
        df = _engineer_features(energy_df, weather_df, outdoor_df)

        use_sensor = (
            outdoor_df is not None
            and not outdoor_df.empty
            and "outdoor_temp_live" in df.columns
        )
        feature_cols = _FEATURES_WITH_SENSOR if use_sensor else _FEATURES_BASE

        df = df.dropna(subset=feature_cols + ["gross_kwh"])
        df = df[df["gross_kwh"] > 0]

        if len(df) < 100:
            _LOGGER.warning(
                "Only %d clean training rows — skipping (need ≥100)", len(df)
            )
            return

        X = df[feature_cols].values
        y = df["gross_kwh"].values

        # Hold out last 10 % for MAE estimation
        split = max(int(len(X) * 0.9), len(X) - 500)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        model = _build_model()
        model.fit(X_tr, y_tr)

        mae = float(mean_absolute_error(y_val, model.predict(X_val)))

        self.model = model
        self.feature_cols = feature_cols
        self.last_trained = datetime.now()
        self.last_mae = round(mae, 4)
        self._save()

        _LOGGER.info(
            "Model trained on %d rows | engine: %s | features: %d "
            "(%s sensor) | MAE (holdout): %.4f kWh/h",
            len(df),
            self.engine,
            len(feature_cols),
            "with" if use_sensor else "without",
            mae,
        )

    def predict(
        self,
        forecast_df: pd.DataFrame,
        live_temp: float | None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame with columns [timestamp, predicted_kwh] for the
        next 48 hours.

        forecast_df : columns [timestamp, temp_c, precipitation_mm, sunshine_min, wind_kmh]
        live_temp   : current outdoor sensor reading in °C, or None
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet")

        now = pd.Timestamp.now(tz="Europe/Zurich")
        future_hours = pd.date_range(
            start=now.floor("1h"), periods=48, freq="1h", tz="Europe/Zurich"
        )
        future_df = pd.DataFrame({"timestamp": future_hours, "gross_kwh": np.nan})

        outdoor_pred_df: pd.DataFrame | None = None
        if live_temp is not None:
            outdoor_pred_df = _build_prediction_temp_df(
                future_hours, forecast_df, live_temp
            )

        feat_df = _engineer_features(future_df, forecast_df, outdoor_pred_df)
        X = feat_df[self.feature_cols].fillna(0).values
        preds = np.maximum(0, self.model.predict(X))

        return pd.DataFrame({"timestamp": future_hours, "predicted_kwh": preds})

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        with open(self._model_path, "wb") as fh:
            pickle.dump(self.model, fh)
        meta = {
            "feature_cols": self.feature_cols,
            "last_trained": self.last_trained,
            "last_mae": self.last_mae,
        }
        with open(self._meta_path, "wb") as fh:
            pickle.dump(meta, fh)

    def _load(self) -> None:
        if self._model_path.exists():
            try:
                with open(self._model_path, "rb") as fh:
                    self.model = pickle.load(fh)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("Could not load saved model: %s", exc)

        if self._meta_path.exists():
            try:
                with open(self._meta_path, "rb") as fh:
                    meta = pickle.load(fh)
                self.feature_cols = meta.get("feature_cols", _FEATURES_BASE)
                self.last_trained = meta.get("last_trained", datetime.min)
                self.last_mae = meta.get("last_mae")
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("Could not load model metadata: %s", exc)

    def hours_since_trained(self) -> float:
        if self.last_trained == datetime.min:
            return float("inf")
        return (datetime.now() - self.last_trained).total_seconds() / 3600


# ── Feature engineering (module-level pure functions) ────────────────────────

def _engineer_features(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
    outdoor_df: pd.DataFrame | None,
) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Time
    df["hour"]        = df["timestamp"].dt.hour
    df["hour_block"]  = (df["timestamp"].dt.hour // 3) * 3
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["season"]      = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Weather
    w = weather_df.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"]).dt.floor("1h")
    df["_ts_floor"] = df["timestamp"].dt.floor("1h")
    df = df.merge(w, left_on="_ts_floor", right_on="timestamp",
                  how="left", suffixes=("", "_w"))
    df.drop(columns=["timestamp_w", "_ts_floor"], errors="ignore", inplace=True)

    for col in ["temp_c", "precipitation_mm", "sunshine_min", "wind_kmh"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df["heating_degree"] = np.maximum(0, 18.0 - df["temp_c"])
    df["cooling_degree"] = np.maximum(0, df["temp_c"] - 22.0)

    # Outdoor sensor
    if outdoor_df is not None and not outdoor_df.empty:
        o = outdoor_df.copy()
        o["timestamp"] = pd.to_datetime(o["timestamp"]).dt.floor("1h")
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(o, left_on="_ts_floor", right_on="timestamp",
                      how="left", suffixes=("", "_ot"))
        df.drop(columns=["timestamp_ot", "_ts_floor"], errors="ignore", inplace=True)
        df["outdoor_temp_live"] = df["outdoor_temp_live"].fillna(df["temp_c"])
        df["temp_bias"] = df["outdoor_temp_live"] - df["temp_c"]
    else:
        df["outdoor_temp_live"] = df["temp_c"]
        df["temp_bias"] = 0.0

    return df


def _build_prediction_temp_df(
    future_hours: pd.DatetimeIndex,
    forecast_df: pd.DataFrame,
    live_temp: float,
) -> pd.DataFrame:
    """
    Blend live sensor reading into the forecast horizon:
      0 – SENSOR_FULL_TRUST_HOURS  → fully trust live sensor
      SENSOR_FULL_TRUST_HOURS – SENSOR_BLEND_HOURS → linear blend live → forecast
      SENSOR_BLEND_HOURS+          → fully trust forecast
    """
    fc_indexed = (
        forecast_df.set_index("timestamp")["temp_c"]
        .reindex(future_hours, method="nearest")
    )

    temps = []
    blend_range = SENSOR_BLEND_HOURS - SENSOR_FULL_TRUST_HOURS
    for i, ts in enumerate(future_hours):
        fc = fc_indexed.get(ts, live_temp)
        if i < SENSOR_FULL_TRUST_HOURS:
            temps.append(live_temp)
        elif i < SENSOR_BLEND_HOURS:
            alpha = (i - SENSOR_FULL_TRUST_HOURS) / blend_range
            temps.append(live_temp * (1 - alpha) + fc * alpha)
        else:
            temps.append(fc)

    return pd.DataFrame({"timestamp": future_hours, "outdoor_temp_live": temps})


def _build_model():
    if _USE_LGBM:
        return lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.04,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
    return GradientBoostingRegressor(  # type: ignore[return-value]
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
