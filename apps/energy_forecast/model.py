"""
ML model — feature engineering, training, prediction and disk persistence.

Feature set:
  - Calendar: hour, day_of_week, month, season, hour_of_week (0-167),
    cyclical encodings (sin/cos) for hour, month, day-of-week, hour-of-week
  - Weather: temp_c, precipitation_mm, sunshine_min, wind_kmh,
    cloud_cover_pct, direct_radiation_wm2, heating_degree, cooling_degree,
    temp_rolling_3d (3-day thermal mass proxy)
  - Autoregressive lags: lag_24h, lag_48h, lag_72h, lag_168h, lag_336h
    (dynamically selected based on available history)
  - Rolling activity: rolling_mean_24h, rolling_mean_7d, rolling_std_24h
    (per-hour sliding projection at predict time to match training semantics)
  - Calendar extras: is_public_holiday, days_to_next_holiday,
    days_since_last_holiday (Swiss federal + optional cantonal)
  - EV charging pattern: likely_ev_hour binary flag

Model:
  LightGBM preferred; scikit-learn GBR automatic fallback (e.g. armv7).
  Target is log1p(gross_kwh); predictions are expm1-inverted.
  Exponential sample weighting (weight_halflife_days, default 90 days).
  LightGBM early stopping against CV fold validation set.
  TimeSeriesSplit CV MAE (≥500 rows) or holdout MAE reported.
  Two quantile models (α=0.1, α=0.9) trained for prediction intervals.

Persistence:
  energy_model.pkl + meta.pkl, each with a SHA-256 sidecar.
  Integrity mismatch → warning + cold-start retrain.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from .const import (
    HOLDOUT_FRACTION,
    MAX_HOURLY_KWH,
    MIN_CV_ROWS,
    MIN_TRAINING_ROWS,
    SENSOR_BLEND_HOURS,
    SENSOR_FULL_TRUST_HOURS,
)

_LOGGER = logging.getLogger(__name__)

# ── Lag hours used as autoregressive features ─────────────────────────────────
# All are safe for a 48-hour forecast: target is always ≥1h ahead,
# so lag_24h for h=+1 points to now-23h — always in the past.
LAG_HOURS = [24, 48, 72, 168, 336]

# ── Feature column sets ───────────────────────────────────────────────────────
_FEATURES_BASE = [
    # Calendar
    "hour", "day_of_week", "month", "season",
    "hour_of_week",                                  # 0-167 — replaces is_weekend + hour_block
    # Cyclical encodings
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "how_sin", "how_cos",                            # hour-of-week cyclical
    # Weather
    "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh",
    "cloud_cover_pct", "direct_radiation_wm2",
    "heating_degree", "cooling_degree",
    "temp_rolling_3d",                               # thermal mass proxy
    # Autoregressive lags — always safe (see note above)
    "lag_24h", "lag_48h", "lag_72h", "lag_168h", "lag_336h",
    # Rolling activity stats
    "rolling_mean_24h", "rolling_mean_7d", "rolling_std_24h",
    # Calendar extras
    "is_public_holiday",
    "days_to_next_holiday",
    "days_since_last_holiday",
    # EV charging pattern
    "likely_ev_hour",
]
_FEATURES_WITH_SENSOR = _FEATURES_BASE + ["outdoor_temp_live", "temp_bias"]


def _write_hash(path: Path) -> None:
    """Write a SHA-256 sidecar file next to *path*."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest)


def _verify_hash(path: Path) -> bool:
    """Return True if the SHA-256 sidecar matches *path*, or if no sidecar exists.

    No sidecar means the file predates integrity checking — allowed to load so
    existing installations are not broken on upgrade.
    """
    hash_path = path.with_suffix(path.suffix + ".sha256")
    if not hash_path.exists():
        return True
    expected = hash_path.read_text().strip()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    return actual == expected


def _try_import_lgbm() -> Any | None:
    try:
        import lightgbm as lgb  # noqa: PLC0415
        return lgb
    except ImportError:
        return None


def _try_import_sklearn_gbr() -> Any | None:
    try:
        from sklearn.ensemble import GradientBoostingRegressor  # noqa: PLC0415
        return GradientBoostingRegressor
    except ImportError:
        return None


def _try_import_mae() -> Any | None:
    try:
        from sklearn.metrics import mean_absolute_error  # noqa: PLC0415
        return mean_absolute_error
    except ImportError:
        return None


def ensure_ml_packages() -> tuple[bool, str]:
    lgb = _try_import_lgbm()
    gbr = _try_import_sklearn_gbr()
    if gbr is None:
        _LOGGER.error("scikit-learn is not importable. Check AppDaemon requirements.txt.")
        return False, "none"
    engine = "LightGBM" if lgb is not None else "sklearn GBR"
    _LOGGER.info("ML engine: %s", engine)
    return True, engine


class EnergyForecastModel:
    """Encapsulates training data, model weights and prediction logic."""

    def __init__(self, model_dir: Path) -> None:
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_path  = model_dir / "energy_model.pkl"
        self._meta_path   = model_dir / "meta.pkl"

        self.model = None
        self.feature_cols: list[str]       = _FEATURES_BASE
        self.last_trained: datetime        = datetime.min
        self.last_mae: float | None        = None
        self.last_cv_mae: float | None     = None   # cross-validated MAE
        self.engine: str                   = "not trained"
        self._feature_medians: dict        = {}     # used to fill NaN at predict time
        self._log_transform: bool          = False  # log1p target; False = backward compat
        self._canton: str | None           = None   # cantonal holiday subdivision
        # hour_of_week slots (0-167) with ≥ EV_HOW_MIN_FRACTION EV occurrences
        self._likely_ev_hours: set[int]    = set()
        # Quantile models for prediction intervals (α=0.1, α=0.9)
        self._model_q10 = None
        self._model_q90 = None
        self._model_q10_path = model_dir / "energy_model_q10.pkl"
        self._model_q90_path = model_dir / "energy_model_q90.pkl"
        # Sub-energy sensor prefixes used in last training run
        self._sub_sensor_prefixes: list[str] = []

        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def train(
        self,
        energy_df: pd.DataFrame,          # naive timestamps, cols: timestamp, gross_kwh
        weather_df: pd.DataFrame,         # naive timestamps
        outdoor_df: pd.DataFrame | None,  # naive timestamps
        weight_halflife_days: float = 90.0,
        canton: str | None = None,
        ev_df: pd.DataFrame | None = None,  # EV charging rows (original gross_kwh)
        sub_sensors_dict: dict | None = None,  # {prefix: DataFrame[timestamp, kwh]}
    ) -> None:
        """Train/retrain the model on historical data."""
        import pandas as pd
        import numpy as np

        ok, engine_name = ensure_ml_packages()
        if not ok:
            raise RuntimeError("scikit-learn unavailable — check AppDaemon requirements.txt.")

        lgb    = _try_import_lgbm()
        GBR    = _try_import_sklearn_gbr()
        mae_fn = _try_import_mae()

        # ── Dynamic lag selection based on available history ────────────────
        # Only include lag_Nh if at least 100 rows remain after shift(N).
        # With short history (e.g. first 2 weeks), lag_336h would leave only
        # ~12 rows and dropna() would wipe the entire training set.
        n_rows = len(energy_df)
        active_lags = [lag for lag in LAG_HOURS if n_rows - lag >= 100]
        skipped_lags = [lag for lag in LAG_HOURS if lag not in active_lags]
        if skipped_lags:
            _LOGGER.info(
                "Dynamic lag selection: skipping %s (need %d+ rows, have %d). "
                "These will be added automatically as history grows.",
                [f"lag_{l}h" for l in skipped_lags],
                max(skipped_lags) + 100,
                n_rows,
            )

        # ── Lag & rolling features (must happen before _engineer_features) ──
        df = _add_lag_and_rolling_training(energy_df, active_lags)
        df = _add_sub_sensor_lags_training(df, sub_sensors_dict)

        # ── EV session probability: which hour_of_week slots charge regularly ─
        likely_ev_hours = _compute_likely_ev_hours(energy_df, ev_df)
        self._likely_ev_hours = likely_ev_hours

        # ── Weather / outdoor / calendar features ───────────────────────────
        df = _engineer_features(df, weather_df, outdoor_df, canton=canton,
                                likely_ev_hours=likely_ev_hours)

        # ── Build feature list from active lags ─────────────────────────────
        # Replace the static lag columns in _FEATURES_BASE/_WITH_SENSOR with
        # only the lags that were actually computed for this training run.
        all_lag_cols = {f"lag_{l}h" for l in LAG_HOURS}
        active_lag_cols = [f"lag_{l}h" for l in active_lags]

        # Sub-sensor lag columns: lag_24h always; lag_168h only with enough history
        sub_sensor_cols: list[str] = []
        if sub_sensors_dict:
            for prefix in sub_sensors_dict:
                sub_sensor_cols.append(f"{prefix}_lag_24h")
                if n_rows - 168 >= 100:
                    sub_sensor_cols.append(f"{prefix}_lag_168h")

        base_features = [c for c in _FEATURES_BASE if c not in all_lag_cols] + active_lag_cols + sub_sensor_cols
        sensor_features = base_features + ["outdoor_temp_live", "temp_bias"]

        use_sensor = (
            outdoor_df is not None
            and not outdoor_df.empty
            and "outdoor_temp_live" in df.columns
        )
        feature_cols = sensor_features if use_sensor else base_features

        df = df.dropna(subset=feature_cols + ["gross_kwh"])
        df = df[df["gross_kwh"] > 0]

        if len(df) < MIN_TRAINING_ROWS:
            _LOGGER.warning("Only %d clean rows — skipping (need ≥%d)", len(df), MIN_TRAINING_ROWS)
            return

        # ── Store medians for NaN-filling at predict time ───────────────────
        self._feature_medians = {
            col: float(df[col].median()) for col in feature_cols if col in df.columns
        }

        X = df[feature_cols]            # keep as DataFrame for LightGBM feature names
        y = df["gross_kwh"].values
        y_fit = np.log1p(y)             # log-transform reduces influence of rare high peaks

        # ── Exponential sample weighting ────────────────────────────────────
        sample_weight = None
        if weight_halflife_days > 0:
            end_ts = df["timestamp"].max()
            days_ago = (end_ts - df["timestamp"]).dt.total_seconds() / 86400
            sample_weight = np.exp(-days_ago.values * np.log(2) / weight_halflife_days)

        # ── TimeSeriesSplit cross-validation for MAE reporting ───────────────
        # Also used to determine optimal n_estimators via LightGBM early stopping.
        cv_mae = None
        best_n_est: int | None = None
        if mae_fn is not None and len(df) >= MIN_CV_ROWS:
            try:
                from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415
                tscv      = TimeSeriesSplit(n_splits=3)
                fold_maes = []
                for tr_idx, val_idx in tscv.split(X):
                    sw_fold = sample_weight[tr_idx] if sample_weight is not None else None
                    m = _build_model(lgb, GBR)
                    fit_kwargs: dict = {}
                    if sw_fold is not None:
                        fit_kwargs["sample_weight"] = sw_fold
                    if lgb is not None:
                        try:
                            m.fit(
                                X.iloc[tr_idx], y_fit[tr_idx],
                                eval_set=[(X.iloc[val_idx], y_fit[val_idx])],
                                callbacks=[
                                    lgb.early_stopping(50, verbose=False),
                                    lgb.log_evaluation(-1),
                                ],
                                **fit_kwargs,
                            )
                            best_n_est = getattr(m, "best_iteration_", best_n_est)
                        except Exception as _es_exc:  # noqa: BLE001
                            _LOGGER.debug("Early stopping failed for fold: %s — using fixed estimators", _es_exc)
                            m.fit(X.iloc[tr_idx], y_fit[tr_idx], **fit_kwargs)
                    else:
                        m.fit(X.iloc[tr_idx], y_fit[tr_idx], **fit_kwargs)
                    # MAE reported in original kWh space (expm1 undoes log1p)
                    fold_maes.append(float(mae_fn(y[val_idx], np.expm1(m.predict(X.iloc[val_idx])))))
                cv_mae = round(float(np.mean(fold_maes)), 4)
            except (ValueError, np.linalg.LinAlgError) as exc:
                _LOGGER.warning("CV MAE failed: %s", exc)

        if best_n_est is not None:
            _LOGGER.info("LightGBM early stopping: using %d estimators (from last CV fold)", best_n_est)

        # ── Final model on all data ─────────────────────────────────────────
        model = _build_model(lgb, GBR, n_estimators=best_n_est)
        if sample_weight is not None:
            model.fit(X, y_fit, sample_weight=sample_weight)
        else:
            model.fit(X, y_fit)

        # ── Hold-out MAE (last 10%) as a quick sanity check ─────────────────
        holdout_mae = None
        if mae_fn is not None:
            split = max(int(len(X) * HOLDOUT_FRACTION), len(X) - MIN_CV_ROWS)
            try:
                holdout_mae = round(float(mae_fn(y[split:], np.expm1(model.predict(X.iloc[split:])))), 4)
            except (ValueError, IndexError):
                pass

        self.model                 = model
        self.feature_cols          = feature_cols
        self.last_trained          = datetime.now()
        self.last_mae              = cv_mae if cv_mae is not None else holdout_mae
        self.last_cv_mae           = cv_mae
        self.engine                = engine_name
        self._log_transform        = True
        self._canton               = canton
        self._sub_sensor_prefixes  = list(sub_sensors_dict.keys()) if sub_sensors_dict else []
        self._save()

        mae_str = f"cv_MAE={cv_mae:.4f}" if cv_mae else (f"holdout_MAE={holdout_mae:.4f}" if holdout_mae else "MAE=n/a")
        _LOGGER.info(
            "Model trained | rows=%d | engine=%s | features=%d | %s",
            len(df), engine_name, len(feature_cols), mae_str,
        )

        # ── Quantile models for prediction intervals ─────────────────────────
        # Trained on same X/y_fit/sample_weight; n_estimators from early stopping.
        # Wrapped so a quantile failure never interrupts normal operation.
        try:
            q10 = _build_quantile_model(lgb, GBR, alpha=0.1, n_estimators=best_n_est)
            q90 = _build_quantile_model(lgb, GBR, alpha=0.9, n_estimators=best_n_est)
            if sample_weight is not None:
                q10.fit(X, y_fit, sample_weight=sample_weight)
                q90.fit(X, y_fit, sample_weight=sample_weight)
            else:
                q10.fit(X, y_fit)
                q90.fit(X, y_fit)
            self._model_q10 = q10
            self._model_q90 = q90
            self._save_quantile_models()
            _LOGGER.info("Quantile models trained (q10, q90) — prediction intervals available")
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Quantile model training failed: %s — prediction intervals unavailable", exc)
            self._model_q10 = None
            self._model_q90 = None


    def _prepare_prediction_X(
        self,
        forecast_df: pd.DataFrame,
        live_temp: float | None,
        recent_actuals: pd.DataFrame | None,
        sub_sensors_recent: dict | None = None,
    ):
        """Build the 48-hour feature matrix shared by predict() and predict_intervals().

        Returns (future_hours, X) where future_hours is a naive DatetimeIndex
        and X is the filled DataFrame ready for model.predict().
        """
        import pandas as pd
        import numpy as np

        now_naive = pd.Timestamp.now(tz="Europe/Zurich").tz_convert(None)
        future_hours = pd.date_range(
            start=now_naive.floor("1h"), periods=48, freq="1h"
        )

        future_df = pd.DataFrame({"timestamp": future_hours, "gross_kwh": np.nan})
        future_df = _add_lag_and_rolling_prediction(future_df, recent_actuals)
        future_df = _add_sub_sensor_lags_prediction(future_df, sub_sensors_recent or {})

        outdoor_pred_df: pd.DataFrame | None = None
        if live_temp is not None and "outdoor_temp_live" in self.feature_cols:
            outdoor_pred_df = _build_prediction_temp_df(future_hours, forecast_df, live_temp)

        feat_df = _engineer_features(future_df, forecast_df, outdoor_pred_df, canton=self._canton,
                                     likely_ev_hours=self._likely_ev_hours)

        for col in self.feature_cols:
            if col in feat_df.columns and feat_df[col].isna().any():
                feat_df[col] = feat_df[col].fillna(self._feature_medians.get(col, 0.0))

        X = feat_df[self.feature_cols].fillna(0)
        return future_hours, X

    def predict(
        self,
        forecast_df: pd.DataFrame,                    # naive timestamps
        live_temp: float | None,
        recent_actuals: pd.DataFrame | None = None,   # naive timestamps, cols: timestamp, gross_kwh
        sub_sensors_recent: dict | None = None,       # {prefix: DataFrame[timestamp, kwh]}
    ) -> pd.DataFrame:
        """Return 48-hour DataFrame [timestamp (naive), predicted_kwh]."""
        import pandas as pd
        import numpy as np

        if self.model is None:
            raise RuntimeError("Model not yet trained.")

        future_hours, X = self._prepare_prediction_X(forecast_df, live_temp, recent_actuals, sub_sensors_recent)
        preds = self.model.predict(X)
        if self._log_transform:
            preds = np.expm1(preds)
        preds = np.maximum(0, preds)
        return pd.DataFrame({"timestamp": future_hours, "predicted_kwh": preds})

    def predict_intervals(
        self,
        forecast_df: pd.DataFrame,
        live_temp: float | None,
        recent_actuals: pd.DataFrame | None = None,
        sub_sensors_recent: dict | None = None,
    ) -> pd.DataFrame | None:
        """Return 48-hour DataFrame [timestamp, low_kwh, high_kwh], or None.

        Returns None when quantile models are not yet trained.  low_kwh and
        high_kwh are guaranteed non-negative and ordered (low ≤ high).
        """
        import pandas as pd
        import numpy as np

        if self._model_q10 is None or self._model_q90 is None:
            return None

        future_hours, X = self._prepare_prediction_X(forecast_df, live_temp, recent_actuals, sub_sensors_recent)
        low  = self._model_q10.predict(X)
        high = self._model_q90.predict(X)
        if self._log_transform:
            low  = np.expm1(low)
            high = np.expm1(high)
        low  = np.maximum(0, low)
        high = np.maximum(0, high)
        # Enforce ordering in case of quantile crossing
        low, high = np.minimum(low, high), np.maximum(low, high)
        return pd.DataFrame({"timestamp": future_hours, "low_kwh": low, "high_kwh": high})

    def hours_since_trained(self) -> float:
        if self.last_trained == datetime.min:
            return float("inf")
        return (datetime.now() - self.last_trained).total_seconds() / 3600

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        with open(self._model_path, "wb") as fh:
            pickle.dump(self.model, fh)
        _write_hash(self._model_path)

        meta = {
            "feature_cols":          self.feature_cols,
            "last_trained":          self.last_trained,
            "last_mae":              self.last_mae,
            "last_cv_mae":           self.last_cv_mae,
            "engine":                self.engine,
            "feature_medians":       self._feature_medians,
            "log_transform":         self._log_transform,
            "canton":                self._canton,
            "likely_ev_hours":       self._likely_ev_hours,
            "sub_sensor_prefixes":   self._sub_sensor_prefixes,
        }
        with open(self._meta_path, "wb") as fh:
            pickle.dump(meta, fh)
        _write_hash(self._meta_path)

    def _save_quantile_models(self) -> None:
        for path, mdl in [
            (self._model_q10_path, self._model_q10),
            (self._model_q90_path, self._model_q90),
        ]:
            if mdl is not None:
                with open(path, "wb") as fh:
                    pickle.dump(mdl, fh)
                _write_hash(path)

    def _load_quantile_models(self) -> None:
        for path, attr in [
            (self._model_q10_path, "_model_q10"),
            (self._model_q90_path, "_model_q90"),
        ]:
            if not path.exists():
                continue
            if not _verify_hash(path):
                _LOGGER.warning(
                    "Quantile model integrity check failed (%s) — discarding.", path
                )
                continue
            try:
                with open(path, "rb") as fh:
                    setattr(self, attr, pickle.load(fh))
            except (pickle.UnpicklingError, EOFError, OSError) as exc:
                _LOGGER.warning("Could not load quantile model %s: %s", path.name, exc)

    def _load(self) -> None:
        if self._model_path.exists():
            if not _verify_hash(self._model_path):
                _LOGGER.warning(
                    "Model file integrity check failed (%s) — discarding, will retrain.",
                    self._model_path,
                )
            else:
                try:
                    with open(self._model_path, "rb") as fh:
                        self.model = pickle.load(fh)
                except (pickle.UnpicklingError, EOFError, OSError) as exc:
                    _LOGGER.warning("Could not load saved model: %s", exc)

        if self._meta_path.exists():
            if not _verify_hash(self._meta_path):
                _LOGGER.warning(
                    "Model metadata integrity check failed (%s) — discarding.",
                    self._meta_path,
                )
            else:
                try:
                    with open(self._meta_path, "rb") as fh:
                        meta = pickle.load(fh)
                    self.feature_cols          = meta.get("feature_cols",          _FEATURES_BASE)
                    self.last_trained          = meta.get("last_trained",          datetime.min)
                    self.last_mae              = meta.get("last_mae")
                    self.last_cv_mae           = meta.get("last_cv_mae")
                    self.engine                = meta.get("engine",                "unknown")
                    self._feature_medians      = meta.get("feature_medians",       {})
                    self._log_transform        = meta.get("log_transform",         False)
                    self._canton               = meta.get("canton",                None)
                    self._likely_ev_hours      = meta.get("likely_ev_hours",       set())
                    self._sub_sensor_prefixes  = meta.get("sub_sensor_prefixes",   [])
                except (pickle.UnpicklingError, EOFError, OSError) as exc:
                    _LOGGER.warning("Could not load model metadata: %s", exc)

        self._load_quantile_models()


# ── Lag & rolling feature helpers ─────────────────────────────────────────────

def _add_lag_and_rolling_training(energy_df: pd.DataFrame, active_lags: list[int]) -> pd.DataFrame:
    """Compute lag and rolling features during training using shift().

    Only computes lags in active_lags — lags that would leave fewer than
    100 rows after shift() are excluded by the caller to avoid dropna()
    wiping the training set when history is short.

    Uses shift(1) before rolling to prevent data leakage — the rolling
    stats represent "what has been happening" up to but not including
    the current hour.
    """
    import pandas as pd

    df = energy_df.sort_values("timestamp").reset_index(drop=True).copy()

    for lag in active_lags:
        df[f"lag_{lag}h"] = df["gross_kwh"].shift(lag)

    shifted = df["gross_kwh"].shift(1)
    df["rolling_mean_24h"] = shifted.rolling(24,    min_periods=12).mean()
    df["rolling_mean_7d"]  = shifted.rolling(7*24,  min_periods=48).mean()
    df["rolling_std_24h"]  = shifted.rolling(24,    min_periods=12).std().fillna(0)

    return df


def _add_lag_and_rolling_prediction(future_df: pd.DataFrame, recent_actuals: pd.DataFrame | None) -> pd.DataFrame:
    """Fill lag and rolling features for the 48-hour prediction horizon.

    For each future hour h:
      lag_24h[h]  = actual consumption at (h - 24h)  — always a real past value
      lag_168h[h] = actual consumption at (h - 168h) — always a real past value

    If recent_actuals is missing or sparse, falls back to NaN (the model
    will fill those with stored training medians).
    """
    import pandas as pd
    import numpy as np

    if recent_actuals is None or recent_actuals.empty:
        for lag in LAG_HOURS:
            future_df[f"lag_{lag}h"] = np.nan
        future_df["rolling_mean_24h"] = np.nan
        future_df["rolling_mean_7d"]  = np.nan
        future_df["rolling_std_24h"]  = np.nan
        return future_df

    # Index actuals by timestamp for O(1) lookup
    actuals = (
        recent_actuals
        .set_index(pd.to_datetime(recent_actuals["timestamp"]))["gross_kwh"]
        .sort_index()
    )

    for lag in LAG_HOURS:
        lag_td = pd.Timedelta(hours=lag)
        lag_times = future_df["timestamp"] - lag_td
        future_df[f"lag_{lag}h"] = actuals.reindex(lag_times).values
        nan_count = int(future_df[f"lag_{lag}h"].isna().sum())
        if nan_count > len(future_df) * 0.5:
            _LOGGER.warning(
                "lag_%dh has %d/%d NaN values — recent_actuals doesn't reach "
                "back %dh; these will be filled with training medians.",
                lag, nan_count, len(future_df), lag,
            )

    # ── Rolling stats — sliding window projection ────────────────────────────
    # During training, rolling_mean_24h[i] = mean(gross_kwh[i-24:i]) — it varies
    # per row as the window slides forward.  Broadcasting a single scalar to all
    # 48 future hours creates a systematic train/predict mismatch that worsens
    # beyond h=24.
    #
    # Fix: extend the actuals series with fill values at the 48 future timestamps,
    # then compute rolling stats on the combined series.  Unknown future values
    # are filled with the recent 24h mean — a stable proxy that causes the
    # rolling stats to decay smoothly toward the current household baseline as
    # h increases rather than jumping discontinuously.
    #
    # For h=0  the window is 100% known actuals  → exact match with training.
    # For h=12 the window is half-known, half-filled → gradual blend.
    # For h≥24 the window is mostly/entirely filled → stabilises at fill_val.
    future_index = pd.to_datetime(future_df["timestamp"])
    try:
        fill_val = float(actuals.iloc[-24:].mean())
    except (ValueError, TypeError, IndexError):
        fill_val = float(actuals.mean()) if len(actuals) > 0 else np.nan

    future_fill = pd.Series(fill_val, index=future_index)
    extended = pd.concat([actuals, future_fill]).sort_index()
    extended = extended[~extended.index.duplicated(keep="last")]

    # shift(1) mirrors the training computation (_add_lag_and_rolling_training
    # applies shift(1) before rolling so the window excludes the current row).
    # Without it, the window at h=0 would include the fill-value slot itself,
    # causing a mismatch with the training semantics at the boundary.
    extended_shifted = extended.shift(1)
    rm24 = extended_shifted.rolling(24,  min_periods=12).mean()
    rm7d = extended_shifted.rolling(168, min_periods=48).mean()
    rs24 = extended_shifted.rolling(24,  min_periods=12).std().fillna(0)

    future_df["rolling_mean_24h"] = rm24.reindex(future_index).values
    future_df["rolling_mean_7d"]  = rm7d.reindex(future_index).values
    future_df["rolling_std_24h"]  = rs24.reindex(future_index).values

    return future_df


# ── Sub-sensor lag helpers ────────────────────────────────────────────────────

def _add_sub_sensor_lags_training(
    df: pd.DataFrame,
    sub_sensors: dict | None,
) -> pd.DataFrame:
    """Add lag_24h (and optionally lag_168h) columns for each sub-sensor.

    For each prefix/DataFrame pair in sub_sensors:
    - Reindexes the sub-sensor kWh series onto the training timestamps
    - Computes lag_24h via shift(24), lag_168h via shift(168) when n_rows ≥ 268

    Returns df unchanged when sub_sensors is empty/None.
    """
    import pandas as pd

    if not sub_sensors:
        return df

    n_rows = len(df)
    ts_idx = pd.to_datetime(df["timestamp"])

    for prefix, sub_df in sub_sensors.items():
        if sub_df is None or sub_df.empty:
            continue
        sub_series = (
            sub_df.set_index(pd.to_datetime(sub_df["timestamp"]))["kwh"]
            .sort_index()
        )
        kwh_series = pd.Series(sub_series.reindex(ts_idx).values)
        df[f"{prefix}_lag_24h"] = kwh_series.shift(24).values
        if n_rows - 168 >= 100:
            df[f"{prefix}_lag_168h"] = kwh_series.shift(168).values

    return df


def _add_sub_sensor_lags_prediction(
    future_df: pd.DataFrame,
    sub_sensors_recent: dict,
) -> pd.DataFrame:
    """Fill sub-sensor lag features for the 48-hour prediction horizon.

    For each sub-sensor, reindexes the recent actuals series at
    (timestamp - 24h) and (timestamp - 168h).  Falls back to NaN when actuals
    don't reach far enough back (logged at WARNING).
    """
    import pandas as pd
    import numpy as np

    for prefix, sub_df in sub_sensors_recent.items():
        if sub_df is None or sub_df.empty:
            future_df[f"{prefix}_lag_24h"]  = np.nan
            future_df[f"{prefix}_lag_168h"] = np.nan
            continue

        sub_series = (
            sub_df.set_index(pd.to_datetime(sub_df["timestamp"]))["kwh"]
            .sort_index()
        )
        for lag, col in [(24, f"{prefix}_lag_24h"), (168, f"{prefix}_lag_168h")]:
            lag_td   = pd.Timedelta(hours=lag)
            lag_times = pd.to_datetime(future_df["timestamp"]) - lag_td
            future_df[col] = sub_series.reindex(lag_times).values
            nan_count = int(future_df[col].isna().sum())
            if nan_count > len(future_df) * 0.5:
                _LOGGER.warning(
                    "%s has %d/%d NaN values — sub-sensor actuals don't reach back %dh; "
                    "will be filled with training medians.",
                    col, nan_count, len(future_df), lag,
                )

    return future_df


# ── Core feature engineering ──────────────────────────────────────────────────

def _engineer_features(
    df: pd.DataFrame,                   # naive timestamps
    weather_df: pd.DataFrame,           # naive timestamps
    outdoor_df: pd.DataFrame | None,    # naive timestamps
    canton: str | None = None,
    likely_ev_hours: set | None = None,
) -> pd.DataFrame:
    import pandas as pd
    import numpy as np

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # stays naive

    # ── Calendar features ────────────────────────────────────────────────────
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["season"]       = df["month"].map(
        {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
         6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    )
    # hour_of_week gives the model a unique slot for every hour in the weekly
    # cycle (0 = Monday 00:00, 167 = Sunday 23:00). Replaces is_weekend + hour_block.
    df["hour_of_week"] = df["day_of_week"] * 24 + df["hour"]

    # ── Cyclical encodings ───────────────────────────────────────────────────
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]         / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]         / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"]        / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]        / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"]  / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"]  / 7)
    df["how_sin"]   = np.sin(2 * np.pi * df["hour_of_week"] / 168)
    df["how_cos"]   = np.cos(2 * np.pi * df["hour_of_week"] / 168)

    # ── EV charging pattern ───────────────────────────────────────────────────
    if likely_ev_hours:
        df["likely_ev_hour"] = df["hour_of_week"].isin(likely_ev_hours).astype(int)
    else:
        df["likely_ev_hour"] = 0

    # ── Public holidays ──────────────────────────────────────────────────────
    df = _add_holiday_feature(df, canton=canton)

    # ── Weather merge ────────────────────────────────────────────────────────
    w = weather_df.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"]).dt.floor("1h")

    # 3-day rolling temperature (captures thermal mass / multi-day cold spells)
    w = w.sort_values("timestamp").reset_index(drop=True)
    w["temp_rolling_3d"] = w["temp_c"].rolling(72, min_periods=1).mean()

    df["_ts_floor"] = df["timestamp"].dt.floor("1h")
    df = df.merge(w, left_on="_ts_floor", right_on="timestamp",
                  how="left", suffixes=("", "_w"))
    df.drop(columns=["timestamp_w", "_ts_floor"], errors="ignore", inplace=True)

    for col in ["temp_c", "precipitation_mm", "sunshine_min", "wind_kmh",
                "cloud_cover_pct", "direct_radiation_wm2", "temp_rolling_3d"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Safety net: ensure new weather columns always exist even when weather_df
    # doesn't include them (e.g. unexpected API response gaps). The model's
    # _feature_medians fill in predict() handles the resulting NaN values.
    for col in ["cloud_cover_pct", "direct_radiation_wm2"]:
        if col not in df.columns:
            df[col] = np.nan

    df["heating_degree"] = np.maximum(0, 18.0 - df["temp_c"])
    df["cooling_degree"] = np.maximum(0, df["temp_c"] - 22.0)

    # ── Outdoor sensor merge ─────────────────────────────────────────────────
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

_BRIDGE_CAP = 3   # days — bridge-day effect beyond this distance is negligible


def _add_holiday_feature(df: pd.DataFrame, canton: str | None = None) -> pd.DataFrame:
    """Add holiday proximity columns using the `holidays` package.

    Columns added:
      is_public_holiday       — 1 on a Swiss federal (or cantonal) holiday, else 0
      days_to_next_holiday    — calendar days until the next holiday, capped at _BRIDGE_CAP
      days_since_last_holiday — calendar days since the last holiday, capped at _BRIDGE_CAP

    Both distance columns are 0 on a holiday itself.  The cap keeps the feature
    range tight so the model focuses on the bridging-day window where consumption
    patterns actually differ.

    Pass canton (e.g. "ZH", "BE") to include cantonal holidays in addition to
    federal ones.  Falls back to federal-only if the canton code is unrecognised.

    Falls back gracefully (is_public_holiday=0, distances=_BRIDGE_CAP) if the
    `holidays` package is not installed.
    """
    try:
        import bisect  # noqa: PLC0415
        import holidays as hd  # noqa: PLC0415
        import pandas as pd

        # Extend by ±1 year so dates near year boundaries (e.g. Dec 31)
        # can see the next/previous holiday across the year boundary.
        years_in_data = df["timestamp"].dt.year.unique()
        years = sorted({y + d for y in years_in_data for d in (-1, 0, 1)})
        if canton:
            try:
                ch_holidays = hd.country_holidays("CH", years=years, subdiv=canton)
            except (NotImplementedError, KeyError):
                _LOGGER.warning(
                    "Unknown canton code %r — falling back to federal holidays.", canton
                )
                ch_holidays = hd.country_holidays("CH", years=years)
        else:
            ch_holidays = hd.country_holidays("CH", years=years)
        holiday_dates = sorted(ch_holidays.keys())

        dates = df["timestamp"].dt.date

        df["is_public_holiday"] = dates.isin(set(ch_holidays.keys())).astype(int)

        def _days_to_next(d):
            # bisect_left: if d is a holiday, idx points to d → distance 0
            idx = bisect.bisect_left(holiday_dates, d)
            if idx >= len(holiday_dates):
                return _BRIDGE_CAP
            return min(_BRIDGE_CAP, (holiday_dates[idx] - d).days)

        def _days_since_last(d):
            # bisect_right - 1: if d is a holiday, idx points to d → distance 0
            idx = bisect.bisect_right(holiday_dates, d) - 1
            if idx < 0:
                return _BRIDGE_CAP
            return min(_BRIDGE_CAP, (d - holiday_dates[idx]).days)

        df["days_to_next_holiday"]    = dates.map(_days_to_next).astype(int)
        df["days_since_last_holiday"] = dates.map(_days_since_last).astype(int)

    except ImportError:
        df["is_public_holiday"]       = 0
        df["days_to_next_holiday"]    = _BRIDGE_CAP
        df["days_since_last_holiday"] = _BRIDGE_CAP
    except Exception as exc:  # noqa: BLE001
        _LOGGER.warning("Holiday feature failed: %s — defaulting to 0 / %d", exc, _BRIDGE_CAP)
        df["is_public_holiday"]       = 0
        df["days_to_next_holiday"]    = _BRIDGE_CAP
        df["days_since_last_holiday"] = _BRIDGE_CAP
    return df


_EV_HOW_MIN_FRACTION = 0.15   # an hour_of_week slot is "likely EV" if it was a
                               # charging hour in ≥ 15% of its historical occurrences


def _compute_likely_ev_hours(
    baseline_df: Any,           # all training rows (EV hours have kwh already subtracted)
    ev_df: Any,                 # rows classified as EV charging (original kwh)
) -> set[int]:
    """Return the set of hour_of_week values (0-167) that regularly see EV charging.

    For each slot, we compare the number of EV occurrences against total
    training occurrences.  Slots at or above _EV_HOW_MIN_FRACTION are returned.
    Returns an empty set when ev_df is None, empty, or no slot clears the bar.
    """
    import pandas as pd

    if ev_df is None or (hasattr(ev_df, "empty") and ev_df.empty):
        return set()

    def _how(df: Any) -> Any:
        ts = pd.to_datetime(df["timestamp"])
        return ts.dt.dayofweek * 24 + ts.dt.hour

    total_counts = _how(baseline_df).value_counts()
    ev_counts    = _how(ev_df).value_counts()

    # Align on the same index; missing slots in ev_counts → 0
    fractions = ev_counts.reindex(total_counts.index, fill_value=0) / total_counts
    likely = fractions[fractions >= _EV_HOW_MIN_FRACTION].index.tolist()
    if likely:
        _LOGGER.info(
            "EV pattern: %d hour_of_week slots flagged as likely charging windows "
            "(fraction ≥ %.0f%%).",
            len(likely), _EV_HOW_MIN_FRACTION * 100,
        )
    return set(likely)


def _build_prediction_temp_df(
    future_hours,   # pd.DatetimeIndex — naive
    forecast_df,    # pd.DataFrame — naive timestamps
    live_temp: float,
) -> Any:
    import pandas as pd

    fc_indexed = (
        forecast_df.set_index(pd.to_datetime(forecast_df["timestamp"]))["temp_c"]
        .sort_index()
    )
    blend_range = SENSOR_BLEND_HOURS - SENSOR_FULL_TRUST_HOURS  # hours over which we interpolate

    rows = []
    for i, ts in enumerate(future_hours):
        hours_ahead = i  # future_hours[0] is current hour
        if hours_ahead <= SENSOR_FULL_TRUST_HOURS:
            temp = live_temp
        elif hours_ahead >= SENSOR_BLEND_HOURS:
            temp = float(fc_indexed.asof(ts)) if not fc_indexed.empty else live_temp
        else:
            # Linear blend: 0 at SENSOR_FULL_TRUST_HOURS → 1 at SENSOR_BLEND_HOURS
            alpha = (hours_ahead - SENSOR_FULL_TRUST_HOURS) / blend_range
            fc_temp = float(fc_indexed.asof(ts)) if not fc_indexed.empty else live_temp
            temp = (1 - alpha) * live_temp + alpha * fc_temp
        rows.append({"timestamp": ts, "outdoor_temp_live": temp})

    return pd.DataFrame(rows)


def _build_quantile_model(lgb: Any, GBR: Any, alpha: float, n_estimators: int | None = None) -> Any:
    """Instantiate a quantile regression model for prediction intervals."""
    if lgb is not None:
        return lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=n_estimators or 500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
    return GBR(
        loss="quantile",
        alpha=alpha,
        n_estimators=n_estimators or 300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )


def _build_model(lgb: Any, GBR: Any, n_estimators: int | None = None) -> Any:
    """Instantiate the best available model (LightGBM preferred, sklearn GBR fallback).

    n_estimators overrides the default when provided (e.g. from early stopping).
    """
    if lgb is not None:
        return lgb.LGBMRegressor(
            n_estimators=n_estimators or 500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
        )
    return GBR(
        n_estimators=n_estimators or 300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )

