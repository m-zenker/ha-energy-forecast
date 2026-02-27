"""
ML model — feature engineering, training, prediction and disk persistence.

lightgbm and scikit-learn are installed lazily at first training time using
HA's own package installer (pkg_util), which respects the HAOS constraints
file and finds pre-built wheels correctly.

All CPU-bound work is designed to be called via hass.async_add_executor_job()
so it never blocks the HA event loop.
"""
from __future__ import annotations

import importlib
import logging
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any




from .const import (
    MAX_HOURLY_KWH,
    SENSOR_BLEND_HOURS,
    SENSOR_FULL_TRUST_HOURS,
)

_LOGGER = logging.getLogger(__name__)

# ── Package specs ─────────────────────────────────────────────────────────────
_SKLEARN_PKG = "scikit-learn"
_SKLEARN_MIN = "1.3.0"
_LGBM_PKG    = "lightgbm"
_LGBM_MIN    = "4.0.0"

# ── Feature column sets ───────────────────────────────────────────────────────
_FEATURES_BASE = [
    "hour", "hour_block", "day_of_week", "month", "is_weekend", "season",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh",
    "heating_degree", "cooling_degree",
]
_FEATURES_WITH_SENSOR = _FEATURES_BASE + ["outdoor_temp_live", "temp_bias"]


def _install_package(package: str, min_version: str) -> bool:
    """
    Install a package using pip if not already present.
    Uses --no-cache-dir and --prefer-binary to maximise chance of finding
    a pre-built wheel in the HAOS environment.
    Returns True on success.
    """
    try:
        # Check if already installed
        dist = importlib.metadata.version(package)
        from packaging.version import Version
        if Version(dist) >= Version(min_version):
            _LOGGER.debug("%s %s already installed", package, dist)
            return True
    except importlib.metadata.PackageNotFoundError:
        pass
    except Exception:  # noqa: BLE001
        pass

    _LOGGER.info("Installing %s>=%s ...", package, min_version)
    try:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable, "-m", "pip", "install",
                f"{package}>={min_version}",
                "--prefer-binary",
                "--no-cache-dir",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            _LOGGER.info("Successfully installed %s", package)
            return True
        _LOGGER.error(
            "Failed to install %s: %s", package, result.stderr[-500:]
        )
        return False
    except Exception as exc:  # noqa: BLE001
        _LOGGER.error("Exception installing %s: %s", package, exc)
        return False


def _try_import_lgbm() -> Any | None:
    """Try importing lightgbm; return the module or None."""
    try:
        import lightgbm as lgb  # noqa: PLC0415
        return lgb
    except ImportError:
        return None


def _try_import_sklearn_gbr() -> Any | None:
    """Try importing GradientBoostingRegressor; return the class or None."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415
        return GradientBoostingRegressor
    except ImportError:
        return None


def _try_import_mae() -> Any | None:
    """Try importing mean_absolute_error; return the function or None."""
    try:
        from sklearn.metrics import mean_absolute_error  # noqa: PLC0415
        return mean_absolute_error
    except ImportError:
        return None


def ensure_ml_packages() -> tuple[bool, str]:
    """
    Ensure scikit-learn and (optionally) lightgbm are available.
    Called from the executor thread before training.

    Returns (success, engine_name).
    """
    # Try lightgbm first
    lgb = _try_import_lgbm()
    if lgb is None:
        if _install_package(_LGBM_PKG, _LGBM_MIN):
            lgb = _try_import_lgbm()

    # sklearn is the hard requirement
    gbr = _try_import_sklearn_gbr()
    if gbr is None:
        if not _install_package(_SKLEARN_PKG, _SKLEARN_MIN):
            return False, "none"
        gbr = _try_import_sklearn_gbr()
        if gbr is None:
            return False, "none"

    engine = "LightGBM" if lgb is not None else "sklearn GBR"
    return True, engine


class EnergyForecastModel:
    """Encapsulates training data, model weights and prediction logic."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = model_dir / "energy_model.pkl"
        self._meta_path  = model_dir / "meta.pkl"

        self.model = None
        self.feature_cols: list[str] = _FEATURES_BASE
        self.last_trained: datetime = datetime.min
        self.last_mae: float | None = None
        self.engine: str = "not trained"

        self._load()

    # ── Public API (CPU-bound — call via executor_job) ────────────────────────

    def train(
        self,
        energy_df: Any,
        weather_df: Any,
        outdoor_df: Any | None,
    ) -> None:
        """Train (or retrain) the model. Installs ML packages if needed."""

        import pandas as pd  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        ok, engine_name = ensure_ml_packages()
        if not ok:
            raise RuntimeError(
                "Could not install scikit-learn. Check HA logs for pip errors."
            )

        lgb = _try_import_lgbm()
        GBR = _try_import_sklearn_gbr()
        mae_fn = _try_import_mae()

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

        # Hold out last 10% for MAE estimation
        split = max(int(len(X) * 0.9), len(X) - 500)
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        model = _build_model(lgb, GBR)
        model.fit(X_tr, y_tr)

        mae = None
        if mae_fn is not None:
            try:
                mae = round(float(mae_fn(y_val, model.predict(X_val))), 4)
            except Exception:  # noqa: BLE001
                pass

        self.model        = model
        self.feature_cols = feature_cols
        self.last_trained = datetime.now()
        self.last_mae     = mae
        self.engine       = engine_name
        self._save()

        _LOGGER.info(
            "Model trained on %d rows | engine: %s | features: %d "
            "(%s sensor)%s",
            len(df),
            engine_name,
            len(feature_cols),
            "with" if use_sensor else "without",
            f" | MAE: {mae:.4f} kWh/h" if mae is not None else "",
        )

    def predict(
        self,
        forecast_df: Any,
        live_temp: float | None,
    ) -> Any:
        """Return 48-hour DataFrame with columns [timestamp, predicted_kwh]."""
        import pandas as pd  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
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

    def hours_since_trained(self) -> float:
        if self.last_trained == datetime.min:
            return float("inf")
        return (datetime.now() - self.last_trained).total_seconds() / 3600

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        with open(self._model_path, "wb") as fh:
            pickle.dump(self.model, fh)
        meta = {
            "feature_cols": self.feature_cols,
            "last_trained": self.last_trained,
            "last_mae":     self.last_mae,
            "engine":       self.engine,
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
                self.last_mae     = meta.get("last_mae")
                self.engine       = meta.get("engine", "unknown")
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning("Could not load model metadata: %s", exc)


# ── Feature engineering ───────────────────────────────────────────────────────

def _engineer_features(
    df,  # pd.DataFrame
    weather_df,  # pd.DataFrame
    outdoor_df,  # pd.DataFrame | None
) -> Any:
    import pandas as pd  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df["hour"]        = df["timestamp"].dt.hour
    df["hour_block"]  = (df["timestamp"].dt.hour // 3) * 3
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["season"]      = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]        / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]        / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]       / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]       / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Merge weather
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

    # Merge outdoor sensor
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
    future_hours,  # pd.DatetimeIndex
    forecast_df,   # pd.DataFrame
    live_temp: float,
) -> Any:
    import pandas as pd  # noqa: PLC0415
    fc_indexed = (
        forecast_df.set_index("timestamp")["temp_c"]
        .reindex(future_hours, method="nearest")
    )
    blend_range = SENSOR_BLEND_HOURS - SENSOR_FULL_TRUST_HOURS
    temps = []
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


def _build_model(lgb: Any | None, GBR: Any | None) -> Any:
    if lgb is not None:
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
    return GBR(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        subsample=0.8,
        random_state=42,
    )
