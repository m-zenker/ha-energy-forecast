"""
ML model — feature engineering, training, prediction and disk persistence.

Feature set:
  - Calendar: hour, day_of_week, month, season, hour_of_week (0-167),
    cyclical encodings (sin/cos) for hour, month, day-of-week, hour-of-week,
    day-of-year; hours_ahead horizon feature
  - Weather: temp_c, precipitation_mm, sunshine_min, wind_kmh,
    cloud_cover_pct, direct_radiation_wm2, heating_degree, cooling_degree,
    temp_rolling_3d (3-day thermal mass proxy)
  - Autoregressive lags: lag_1h, lag_2h, lag_6h, lag_12h,
    lag_24h, lag_48h, lag_72h, lag_168h, lag_336h
    (dynamically selected based on available history)
  - Rolling activity: rolling_mean_24h, rolling_mean_7d, rolling_std_24h
    (per-hour sliding projection at predict time to match training semantics)
  - Calendar extras: is_public_holiday, days_to_next_holiday,
    days_since_last_holiday (Swiss federal + optional cantonal)
  - EV charging pattern: likely_ev_hour binary flag
  - Vacation / away: is_away binary flag (0/1, via away_mode_entity)

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
import json
import logging
import pickle
import shutil
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from .physics import ThermalPhysicsModel

from . import clustering
from .const import (
    APPLIANCE_MAX_WINDOW_HOURS,
    DEFAULT_ROOM_AREA_M2,
    DEFAULT_TAU,
    HOLDOUT_FRACTION,
    MIN_CV_ROWS,
    MIN_TRAINING_ROWS,
    SENSOR_BLEND_HOURS,
    SENSOR_FULL_TRUST_HOURS,
)

_LOGGER = logging.getLogger("energy_forecast")

# ── Lag hours used as autoregressive features ─────────────────────────────────
# All are safe for a 48-hour forecast: target is always ≥1h ahead,
# so lag_24h for h=+1 points to now-23h — always in the past.
#
# Short-horizon lags (1h, 2h, 6h, 12h): at predict time only the first L
# future hours can look up a real past value for lag_Lh; beyond that the
# model receives NaN → filled with the training median.  This is by design:
# the model learns that median-valued short lags carry no information and
# weights them heavily only in the near-term window.  Impact is concentrated
# on hours 1–12 ahead.
LAG_HOURS = [1, 2, 6, 12, 24, 48, 72, 168, 336]

# ── Feature column sets ───────────────────────────────────────────────────────
_FEATURES_BASE = [
    # Calendar
    "hour",
    "day_of_week",
    "month",
    "season",
    "hour_of_week",  # 0-167 — replaces is_weekend + hour_block
    # Cyclical encodings
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "how_sin",
    "how_cos",  # hour-of-week cyclical
    "doy_sin",
    "doy_cos",  # day-of-year cyclical (seasonal curve)
    # Horizon
    "hours_ahead",  # 0 at training (actuals), 0-47 at predict
    # Weather
    "temp_c",
    "precipitation_mm",
    "sunshine_min",
    "wind_kmh",
    "humidity",  # relative humidity %
    "cloud_cover_pct",
    "direct_radiation_wm2",
    "heating_degree",
    "cooling_degree",
    "hp_heating_degree",  # HP-calibrated cutoff: max(0, 15 − temp_c)
    "temp_in_neutral_zone",  # dead-band flag: 1 when 15 ≤ temp_c ≤ 22
    "heating_active",  # seasonal on/off from heating_system_active_entity
    "temp_rolling_3d",  # thermal mass proxy (rectangular 72h)
    # Thermal modelling features (#49–#52)
    "temp_ewma_24h",
    "temp_ewma_72h",  # #49 EWMA — RC-circuit thermal mass
    "heating_deg_sum_24h",
    "heating_deg_sum_168h",  # #50 accumulated heating debt
    "temp_delta_1h",
    "temp_delta_24h",  # #51 temperature rate of change
    "temp_lag_24h",
    "temp_lag_168h",  # #52 temperature lags
    # Autoregressive lags — always safe (see note above)
    "lag_1h",
    "lag_2h",
    "lag_6h",
    "lag_12h",  # short-horizon: NaN beyond h=lag at predict time
    "lag_48h",
    "lag_72h",
    "lag_24h_tgated",
    "lag_168h_tgated",
    "lag_336h_tgated",  # temp-delta gated: suppressed when warmer than lag period
    # Rolling activity stats
    "rolling_mean_24h",
    "rolling_mean_7d",
    "rolling_std_24h",
    # Calendar extras
    "is_public_holiday",
    "days_to_next_holiday",
    "days_since_last_holiday",
    # EV charging pattern
    "likely_ev_hour",
    # Vacation / away flag
    "is_away",
    # Occupancy
    "people_home",
    # Daily regime profile (optional module)
    "regime_kwh",
    # Stage 2: Intent-driven features
    "thermal_pressure",
    "thermal_pressure_max",
    "thermal_pressure_std",
    "thermal_pressure_cop",
    "thermal_pressure_net",  # #56 solar-compensated pressure
    "infiltration_pressure",  # #57 wind-driven heat loss
    "defrost_risk",  # #58 HP defrost proxy
    "weighted_solar_gain",
    "dhw_buffer_temp",
    "dhw_pressure",
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
    except ImportError as exc:
        _LOGGER.warning("sklearn import failed: %s", exc)
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
        _LOGGER.error(
            "scikit-learn is not importable. "
            "Install it via the AppDaemon add-on config (init_commands), NOT via pip from a terminal — "
            "terminal pip targets a different Python environment. "
            "See README → Installation → AppDaemon add-on configuration."
        )
        return False, "none"
    engine = "LightGBM" if lgb is not None else "sklearn GBR"
    _LOGGER.info("ML engine: %s", engine)
    return True, engine


def _strip_partial_last_day(energy_df: pd.DataFrame) -> pd.DataFrame:
    """Remove the last date if it has fewer than 24 complete hours.

    Prevents today's partial profile from distorting clustering centroids when
    a retrain fires late in the day (e.g. 23:xx → 23 complete hours pass the
    ≥18-hour filter but leave column 23 bfill-interpolated).
    """
    if energy_df.empty:
        return energy_df
    last_date = energy_df["timestamp"].dt.date.max()
    if (energy_df["timestamp"].dt.date == last_date).sum() < 24:
        return energy_df[energy_df["timestamp"].dt.date != last_date]
    return energy_df


class EnergyForecastModel:
    """Encapsulates training data, model weights and prediction logic."""

    _TAU_SELECTIVITY_THRESHOLD: int = 32  # ≥ this many candidates → top-25%; below → top-50%
    _SPRING_BIAS_OUTDOOR_TEMP: float = 12.0  # °C outdoor median at/above which temp confidence is zero
    _SPRING_BIAS_OUTDOOR_TEMP_FLOOR: float = 5.0  # °C outdoor median at/below which temp confidence is full
    _SPRING_BIAS_NIGHT_FRAC_FULL: float = 0.40  # night_frac at/above which night confidence is full
    _TAU_SAMPLE_CONF_REF: int = 3  # candidate count at/above which sample confidence is full
    _TAU_EMA_MAX_NEW_WEIGHT: float = 0.2  # weight given to the fresh τ estimate at full confidence
    _TAU_DRIFT_WINDOW_DAYS: int = 30  # short drift-cap window (days)
    _TAU_MAX_DRIFT_FRAC: float = 0.35  # short drift-cap max fractional move
    _TAU_LONG_DRIFT_WINDOW_DAYS: int = 180  # long drift-cap window (days)
    _TAU_LONG_MAX_DRIFT_FRAC: float = 0.50  # long drift-cap max fractional move

    assert _SPRING_BIAS_OUTDOOR_TEMP > _SPRING_BIAS_OUTDOOR_TEMP_FLOOR

    def __init__(self, model_dir: Path, model_archive_count: int = 3, timezone: str = "Europe/Zurich") -> None:
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._model_path = model_dir / "energy_model.pkl"
        self._meta_path = model_dir / "meta.pkl"

        self.model = None
        self.feature_cols: list[str] = _FEATURES_BASE
        self.last_trained: datetime = datetime.min
        self.last_mae: float | None = None
        self.last_residual_mae: float | None = None  # Phase 2 only — internal diagnostic (residual scale)
        self.last_cv_mae: float | None = None  # cross-validated MAE
        self.engine: str = "not trained"
        self._feature_medians: dict = {}  # global medians — used to fill NaN at predict time
        self._feature_medians_by_how: dict = {}  # {how: {col: median}} — finer HOW-specific fill
        self._log_transform: bool = False  # log1p target; False = backward compat
        self._use_physics_residual: bool = False  # Phase 2: train on gross_kwh - physics_kwh
        self._last_physics_kwh_series: pd.Series | None = None  # set by _prepare_prediction_X(), read by predict()
        self._canton: str | None = None  # cantonal holiday subdivision
        self._country: str = "CH"  # ISO 3166-1 alpha-2 country for holidays
        self._timezone: str = timezone  # local timezone for wall-clock "now"
        # hour_of_week slots (0-167) with ≥ EV_HOW_MIN_FRACTION EV occurrences
        self._likely_ev_hours: set[int] = set()
        # Quantile models for prediction intervals (α=0.1, α=0.9)
        self._model_q10 = None
        self._model_q90 = None
        self._model_q10_path = model_dir / "energy_model_q10.pkl"
        self._model_q90_path = model_dir / "energy_model_q90.pkl"
        self._signatures_path = model_dir / "appliance_signatures.json"
        self._appliance_signatures: dict = {}  # {prefix: {total_kwh, hourly_profile, peak_hour, n_cycles}}
        # CQR interval correction scalar (log-space additive offset)
        self._interval_correction: float = 0.0
        self._interval_correction_path = model_dir / "energy_model_interval_correction.json"
        # Sub-energy sensor prefixes used in last training run
        self._sub_sensor_prefixes: list[str] = []
        # Building thermal time constant (hours) — calibrated from passive-cooling windows
        self._tau_hours: float | None = None
        # Rolling drift-cap anchors for _tau_hours (short: 30-day/35%, long: 180-day/50%)
        self._tau_anchor_hours: float | None = None
        self._tau_anchor_ts: pd.Timestamp | None = None
        self._tau_long_anchor_hours: float | None = None
        self._tau_long_anchor_ts: pd.Timestamp | None = None
        # Solar-compensated thermal pressure from last predict() call
        self._latest_thermal_pressure_net: float | None = None

        # Daily Regime Clustering (optional)
        self._enable_regimes: bool = False
        self._regime_count: int = 5
        self._clusterer: clustering.DailyProfileClusterer | None = None
        self._regime_model: clustering.RegimePredictor | None = None

        # Recent weather tail — last 400h stored at training time so that
        # temp_lag_168h / temp_lag_336h are computable at prediction time
        # (the 48h forecast window alone cannot supply a 168-row shift).
        self._weather_tail: pd.DataFrame | None = None

        # Model versioning — archive dir + how many snapshots to keep
        self._archive_dir: Path = model_dir / "archive"
        self._model_archive_count: int = model_archive_count

        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def train(
        self,
        energy_df: pd.DataFrame,  # naive timestamps, cols: timestamp, gross_kwh
        weather_df: pd.DataFrame,  # naive timestamps
        outdoor_df: pd.DataFrame | None,  # naive timestamps
        weight_halflife_days: float = 90.0,
        canton: str | None = None,
        country: str = "CH",
        ev_df: pd.DataFrame | None = None,  # EV charging rows (original gross_kwh)
        sub_sensors_dict: dict | None = None,  # {prefix: DataFrame[timestamp, kwh]}
        away_df: pd.DataFrame | None = None,  # cols: timestamp, is_away (0/1)
        presence_df: pd.DataFrame | None = None,  # cols: timestamp, people_home (int)
        climate_dfs: dict[str, pd.DataFrame] | None = None,  # {entity_id: DataFrame[timestamp, current_temp, setpoint]}
        dhw_df: pd.DataFrame | None = None,  # cols: timestamp, buffer_temp
        heating_active_df: pd.DataFrame | None = None,  # cols: timestamp, heating_active (0/1)
        program_histories: dict | None = None,  # {prefix: DataFrame[timestamp, program]}
        room_areas: dict[str, float] | None = None,  # entity_id → m² for area-weighted thermal pressure
        enable_regimes: bool = False,
        regime_count: int = 5,
        physics_model: ThermalPhysicsModel | None = None,
        heating_buffer_temp_df: pd.DataFrame | None = None,  # cols: timestamp, heating_buffer_temp
        use_physics_residual: bool = False,
    ) -> None:
        """Train/retrain the model on historical data."""
        import numpy as np
        import pandas as pd

        ok, engine_name = ensure_ml_packages()
        if not ok:
            raise RuntimeError(
                "scikit-learn unavailable — install via AppDaemon add-on init_commands, not terminal pip. "
                "See README → Installation."
            )

        lgb = _try_import_lgbm()
        GBR = _try_import_sklearn_gbr()
        mae_fn = _try_import_mae()

        # ── Dynamic lag selection based on available history ────────────────
        # Only include lag_Nh if at least 100 rows remain after shift(N).
        # With short history (e.g. first 2 weeks), lag_336h would leave only
        # ~12 rows and dropna() would wipe the entire training set.
        n_rows = len(energy_df)
        active_lags = [lag for lag in LAG_HOURS if n_rows - lag >= 100]
        skipped_lags = [lag for lag in LAG_HOURS if lag not in active_lags]
        if not active_lags:
            _LOGGER.warning(
                "Dynamic lag selection: no lag features active (need ≥%d rows, have %d). "
                "The model will train without autoregressive features — accuracy will be "
                "significantly reduced until more history is collected.",
                min(LAG_HOURS) + 100,
                n_rows,
            )
        elif skipped_lags:
            _LOGGER.info(
                "Dynamic lag selection: skipping %s (need %d+ rows, have %d). "
                "These will be added automatically as history grows.",
                [f"lag_{lag}h" for lag in skipped_lags],
                max(skipped_lags) + 100,
                n_rows,
            )

        # _override_history is hoisted here (rather than declared only inside the
        # `if physics_model is not None:` block below) so it stays in scope for the
        # R1-#13 pre-migration down-weight block further down in this function even
        # when physics_model is None.
        _override_history = physics_model.schedule.get("override_history", []) if physics_model is not None else []

        # Goal 1/2 (R1-#11): correct gross_kwh for any committed DHW override ONCE, upstream
        # of both calibration and the ML training target — single source of truth, not two
        # independent subtraction points that could drift apart. Uses self._calib's current
        # (pre-this-cycle) values as a bootstrap approximation — calibrate() below refines
        # them further using the now-corrected series.
        if physics_model is not None:
            _timestamps_for_delta = pd.DatetimeIndex(pd.to_datetime(energy_df["timestamp"]))
            if _override_history:
                # t_ambient for the DHW ODE is INDOOR air (the room the tank stands in),
                # derived exactly the way predict_training_series derives it for the real
                # physics_kwh feature — not raw outdoor temperature (final-review #2).
                # Deliberately the simple _area_weighted_t_indoor form, without
                # _project_indoor_temps: this is a historical training window with real
                # readings, so there is nothing to project, and predict_training_series
                # doesn't project here either — the two must agree.
                from .physics import DEFAULT_AMBIENT_C

                _t_indoor_for_delta = physics_model._area_weighted_t_indoor(
                    climate_dfs or {}, _timestamps_for_delta, room_areas
                )
                if _t_indoor_for_delta is None:
                    _t_indoor_for_delta = pd.Series(DEFAULT_AMBIENT_C, index=_timestamps_for_delta)
                _initial_t_tank = physics_model._initial_t_tank_for_window(dhw_df, _timestamps_for_delta)
                _override_delta = physics_model.compute_training_override_delta(
                    _timestamps_for_delta, _t_indoor_for_delta, _initial_t_tank, _override_history
                )
                energy_df = energy_df.copy()
                energy_df["gross_kwh"] = energy_df["gross_kwh"].to_numpy() - _override_delta.to_numpy()

        # ── Lag & rolling features (must happen before _engineer_features) ──
        df = _add_lag_and_rolling_training(energy_df, active_lags)
        df = _add_sub_sensor_lags_training(df, sub_sensors_dict)

        # ── Appliance signature learning (Stage 3) ──────────────────────────
        self._appliance_signatures = _learn_appliance_signatures(
            sub_sensors_dict,
            program_histories=program_histories,
        )
        self._save_signatures()

        # ── Thermal time constant calibration (#55) ──────────────────────────
        if climate_dfs and heating_active_df is not None and not heating_active_df.empty:
            self._tau_hours = self._calibrate_tau(climate_dfs, heating_active_df, weather_df)
        else:
            if not climate_dfs:
                _LOGGER.debug("τ calibration skipped: no climate_entities configured.")
            elif heating_active_df is None or heating_active_df.empty:
                _LOGGER.debug("τ calibration skipped: heating_system_active_entity not configured.")

        # ── Physics-ML hybrid: calibration trigger + physics baseline series ──
        # (Plan B Task 4). Runs before the hourly_weights computation below so
        # open_window_flags is available to fold into it, and before the
        # _engineer_features() call further down so physics_kwh_series /
        # heating_buffer_temp_series can be threaded through.
        physics_kwh_series = None
        heating_buffer_temp_series = None
        open_window_flags = None
        if physics_model is not None:
            physics_model.check_zone_boundary(list(climate_dfs.keys()) if climate_dfs else [])
            if physics_model.calibration_stale:
                # Calendar-day span, not row count: a multi-day excluded date range
                # (or any other real gap) leaves len(energy_df)/24 undercounting the
                # true span, silently shrinking/misplacing the holdout window.
                _span_days = (energy_df["timestamp"].max() - energy_df["timestamp"].min()).days
                physics_model.calibrate(
                    energy_df,
                    weather_df,
                    climate_dfs,
                    dhw_df,
                    holdout_cutoff=energy_df["timestamp"].max() - pd.Timedelta(days=int(_span_days * 0.1) or 1),
                    heating_active_df=heating_active_df,
                    away_df=away_df,
                )
            physics_model._tau_hours = self._tau_hours
            physics_kwh_series = physics_model.predict_training_series(
                energy_df, weather_df, climate_dfs=climate_dfs, dhw_df=dhw_df, room_areas=room_areas
            )
            if climate_dfs:
                open_window_flags = physics_model.detect_open_windows(climate_dfs, weather_df, room_areas)

        if heating_buffer_temp_df is not None and not heating_buffer_temp_df.empty:
            heating_buffer_temp_series = heating_buffer_temp_df.set_index(
                pd.to_datetime(heating_buffer_temp_df["timestamp"])
            )["heating_buffer_temp"]

        for prefix, sig in self._appliance_signatures.items():
            _LOGGER.info(
                "Appliance signature | %s | cycles=%d | total=%.2f kWh | peak_hour=%d",
                prefix,
                sig["n_cycles"],
                sig["total_kwh"],
                sig["peak_hour"],
            )
        if not self._appliance_signatures and sub_sensors_dict:
            _LOGGER.info("Appliance signatures: no signatures learned (insufficient cycles in history)")

        # ── EV session probability: which hour_of_week slots charge regularly ─
        likely_ev_hours = _compute_likely_ev_hours(energy_df, ev_df)
        self._likely_ev_hours = likely_ev_hours

        # ── Sample weighting for decay (Stage 1.5) ──────────────────────────
        # Calculate hourly weights for all rows in energy_df
        hourly_weights = None
        daily_weights = None
        if weight_halflife_days > 0:
            end_ts = energy_df["timestamp"].max()
            days_ago = (end_ts - energy_df["timestamp"]).dt.total_seconds() / 86400
            h_weights = np.exp(-days_ago.values * np.log(2) / weight_halflife_days)
            hourly_weights = pd.Series(h_weights, index=energy_df["timestamp"])
            daily_weights = hourly_weights.groupby(energy_df["timestamp"].dt.date).mean()

        # ── Down-weight open-window hours (Plan B Task 4) ───────────────────
        if open_window_flags is not None and hourly_weights is not None:
            ow = open_window_flags.reindex(df["timestamp"].values if "timestamp" in df else hourly_weights.index)
            down_weight = pd.Series(np.where(ow.fillna(False), 0.5, 1.0), index=hourly_weights.index)
            hourly_weights = hourly_weights * down_weight

        # ── Down-weight pre-migration hours (R1-#13) ─────────────────────────
        # Bounds how long training rows from before the DHW override-correction
        # fix shipped (no override_history to reconstruct from, so any override-shaped
        # spikes in gross_kwh are noise the model has to explain away some other way)
        # stay at full training influence — roughly weight_halflife_days after ship,
        # via the existing recency-halflife weighting above.
        if _override_history and hourly_weights is not None:
            _ship_cutoff = pd.Timestamp(min(e["committed_at"] for e in _override_history))
            if _ship_cutoff.tzinfo is not None:
                # committed_at is written as an ISO-8601 UTC-aware string
                # (dt.datetime.now(dt.UTC).isoformat() in commit_dhw_schedule), while
                # df["timestamp"] is naive LOCAL time throughout this module (stripped
                # via _strip_tz(df, self._timezone) before train() ever sees it). A bare
                # tz_localize(None) would strip the tz label without converting, leaving
                # UTC wall-clock numbers mislabeled as naive — silently misclassifying
                # every row within the local UTC offset of the true cutoff. Must convert
                # to local time first, same pattern as energy_forecast.py:1013
                # (`.tz_convert(self._timezone).tz_localize(None)`), not the bare
                # tz_localize(None) used elsewhere in this class for values that were
                # already local when constructed (e.g. model.py's own
                # `pd.Timestamp.now(tz=self._timezone).tz_localize(None)` calls).
                _ship_cutoff = _ship_cutoff.tz_convert(self._timezone).tz_localize(None)
            _pre_migration = pd.to_datetime(df["timestamp"]) < _ship_cutoff
            _down_weight_migration = pd.Series(np.where(_pre_migration, 0.5, 1.0), index=hourly_weights.index)
            hourly_weights = hourly_weights * _down_weight_migration

        # ── Daily Regime Clustering (Stage 4) ───────────────────────────────
        self._enable_regimes = enable_regimes
        self._regime_count = regime_count
        regime_kwh_series = None

        if enable_regimes:
            _actual_k = regime_count
            _prebuilt_daily_features = None
            ev_day_dates: set = set(ev_df["timestamp"].dt.date) if ev_df is not None and len(ev_df) > 0 else set()
            if regime_count == 0:
                # Auto-K: build daily features once and reuse for both selection and predictor
                _prebuilt_daily_features = _prepare_daily_regime_features(
                    weather_df,
                    country=country,
                    canton=canton,
                    away_df=away_df,
                    presence_df=presence_df,
                )
                _actual_k = clustering.find_optimal_k(
                    _strip_partial_last_day(energy_df),
                    _prebuilt_daily_features,
                    sample_weight=daily_weights,
                    ev_day_dates=ev_day_dates,
                )
            self._clusterer = clustering.DailyProfileClusterer(n_clusters=_actual_k)
            self._regime_count = _actual_k
            labels = self._clusterer.fit(_strip_partial_last_day(energy_df), ev_day_dates=ev_day_dates)

            if labels is not None and not labels.empty:
                # ── Train Regime Predictor ──
                daily_features = (
                    _prebuilt_daily_features
                    if _prebuilt_daily_features is not None
                    else _prepare_daily_regime_features(
                        weather_df,
                        country=country,
                        canton=canton,
                        away_df=away_df,
                        presence_df=presence_df,
                    )
                )

                self._regime_model = clustering.RegimePredictor()
                self._regime_model.fit(daily_features, labels, sample_weight=daily_weights)

                # 3. Create hourly regime_kwh_series (vectorized)
                # Map daily labels to hourly timestamps
                ts_idx = pd.to_datetime(df["timestamp"])
                hourly_labels = ts_idx.dt.date
                # Reindex labels to all dates in df, ffill any gaps
                mapped_labels = labels.reindex(hourly_labels).ffill().fillna(-1).astype(int).values

                # Use centroids matrix to lookup values for each hour
                regime_rows = np.zeros(len(ts_idx))
                valid_mask = mapped_labels >= 0
                if valid_mask.any():
                    hours = ts_idx.dt.hour.values
                    n_clusters = self._clusterer.centroids.shape[0]
                    safe_labels = np.clip(mapped_labels[valid_mask], 0, n_clusters - 1)
                    regime_rows[valid_mask] = self._clusterer.centroids[safe_labels, hours[valid_mask]]

                regime_kwh_series = pd.Series(regime_rows, index=ts_idx)

        # ── Store recent weather tail for prediction-time lag gating ────────
        self._weather_tail = weather_df.sort_values("timestamp").tail(400).reset_index(drop=True).copy()

        # ── Weather / outdoor / calendar features ───────────────────────────
        df = _engineer_features(
            df,
            weather_df,
            outdoor_df,
            canton=canton,
            country=country,
            likely_ev_hours=likely_ev_hours,
            away_df=away_df,
            presence_df=presence_df,
            climate_dfs=climate_dfs,
            dhw_df=dhw_df,
            tau_hours=self._tau_hours,
            room_areas=room_areas,
            regime_kwh_series=regime_kwh_series,
            heating_active_df=heating_active_df,
            physics_kwh_series=physics_kwh_series,
            heating_buffer_temp_series=heating_buffer_temp_series,
        )

        # ── Build feature list from active lags ─────────────────────────────
        # Replace the static lag columns in _FEATURES_BASE/_WITH_SENSOR with
        # only the lags that were actually computed for this training run.
        # Lags for 24/168/336h are computed (needed for tgated gating) but
        # excluded here — the model sees only the temperature-gated versions.
        _GATED_LAG_HOURS = frozenset({24, 168, 336})
        all_lag_cols = {f"lag_{lag}h" for lag in LAG_HOURS}
        active_lag_cols = [f"lag_{lag}h" for lag in active_lags if lag not in _GATED_LAG_HOURS]

        # Sub-sensor lag columns: lag_24h always; lag_168h only with enough history
        sub_sensor_cols: list[str] = []
        if sub_sensors_dict:
            for prefix in sub_sensors_dict:
                sub_sensor_cols.append(f"{prefix}_lag_24h")
                if n_rows - 168 >= 100:
                    sub_sensor_cols.append(f"{prefix}_lag_168h")
                sub_sensor_cols.append(f"{prefix}_active_24h")
                sub_sensor_cols.append(f"{prefix}_runs_7d")

        base_features = [c for c in _FEATURES_BASE if c not in all_lag_cols] + active_lag_cols + sub_sensor_cols
        if physics_kwh_series is not None:
            base_features = [*base_features, "physics_kwh"]
        if heating_buffer_temp_series is not None:
            base_features = [*base_features, "heating_buffer_temp"]
        sensor_features = base_features + ["outdoor_temp_live", "temp_bias"]

        use_sensor = outdoor_df is not None and not outdoor_df.empty and "outdoor_temp_live" in df.columns
        feature_cols = sensor_features if use_sensor else base_features

        # Sub-sensor lag columns can be sparsely populated during warm-up (e.g. a
        # new sensor that started recording today).  Fill NaN with 0 before the
        # dropna so these rows are not discarded — 0 means "appliance was off / no
        # data", which is a safe neutral value that does not distort other features.
        for col in sub_sensor_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        df = df.dropna(subset=feature_cols + ["gross_kwh"])
        # gross_kwh == 0 is a real reading (e.g. solar fully covering household
        # load for that hour) and is kept as a training example, not treated as
        # a bad/corrupt row.
        df = df[df["gross_kwh"] >= 0]

        if len(df) < MIN_TRAINING_ROWS:
            _LOGGER.warning("Only %d clean rows — skipping (need ≥%d)", len(df), MIN_TRAINING_ROWS)
            return

        # ── Store medians for NaN-filling at predict time ───────────────────
        self._feature_medians = {col: float(df[col].median()) for col in feature_cols if col in df.columns}

        # Per-hour-of-week medians for lag and rolling columns (#31).
        # These columns vary significantly by HOW (e.g. lag_24h at Monday 08:00
        # differs from Saturday 08:00), so HOW-specific imputation is more accurate
        # than the global median when recent actuals are sparse.
        # Falls back to global median when a HOW bucket has no data.
        how_cols = [
            c
            for c in feature_cols
            if (
                c.startswith("lag_") or c.startswith("rolling_") or "_lag_" in c
            )  # sub-sensor lags: e.g. sub_hp_lag_24h
            and c in df.columns
        ]
        if how_cols and "hour_of_week" in df.columns:
            how_med = (
                df.groupby("hour_of_week")[how_cols].median().to_dict(orient="index")  # {how: {col: median}}
            )
            self._feature_medians_by_how = how_med
        else:
            self._feature_medians_by_how = {}

        X = df[feature_cols].astype(float)  # coerce int32/extension types → float64 (sklearn/lgb compat)
        y = df["gross_kwh"].to_numpy(dtype=float)
        self._use_physics_residual = bool(use_physics_residual and physics_kwh_series is not None)
        if use_physics_residual and not self._use_physics_residual:
            _LOGGER.warning(
                "use_physics_residual requested but no physics_kwh_series available — falling back to Phase 1."
            )

        if self._use_physics_residual:
            physics_aligned = physics_kwh_series.reindex(df["timestamp"])
            n_nans = int(physics_aligned.isna().sum())
            if n_nans > 0:
                _LOGGER.warning(
                    f"{n_nans} hours have no physics prediction — set to 0 in residual target (check weather data gaps)"
                )
            physics_vals = physics_aligned.fillna(0).values
            y_fit = y - physics_vals  # residual target — can be negative, no log transform
        else:
            y_fit = np.log1p(y)  # log-transform reduces influence of rare high peaks

        # ── Sample weighting for main model ─────────────────────────────────
        sample_weight = None
        if hourly_weights is not None:
            # Map hourly weights to the (potentially subsetted) training df
            sample_weight = hourly_weights.reindex(df["timestamp"]).fillna(0).values

        # ── TimeSeriesSplit cross-validation for MAE reporting ───────────────
        # Also used to determine optimal n_estimators via LightGBM early stopping.
        cv_mae = None
        best_n_est: int | None = None
        _NUM_LEAVES_SWEEP = [16, 31, 63]  # candidates; only used on last CV fold
        best_num_leaves: int = 31
        if mae_fn is not None and len(df) >= MIN_CV_ROWS:
            try:
                from sklearn.model_selection import TimeSeriesSplit  # noqa: PLC0415

                tscv = TimeSeriesSplit(n_splits=3)
                splits = list(tscv.split(X))
                fold_maes = []
                for fold_idx, (tr_idx, val_idx) in enumerate(splits):
                    sw_fold = sample_weight[tr_idx] if sample_weight is not None else None
                    fit_kwargs: dict = {}
                    if sw_fold is not None:
                        fit_kwargs["sample_weight"] = sw_fold

                    is_last_fold = fold_idx == len(splits) - 1

                    if is_last_fold and lgb is not None:
                        # ── num_leaves sweep on last fold (LightGBM only) ──────────
                        sweep_maes: dict[int, float] = {}
                        for nl in _NUM_LEAVES_SWEEP:
                            m_nl = _build_model(lgb, GBR, num_leaves=nl)
                            try:
                                m_nl.fit(
                                    X.iloc[tr_idx],
                                    y_fit[tr_idx],
                                    eval_X=X.iloc[val_idx],
                                    eval_y=y_fit[val_idx],
                                    callbacks=[
                                        lgb.early_stopping(50, verbose=False),
                                        lgb.log_evaluation(-1),
                                    ],
                                    **fit_kwargs,
                                )
                            except (ValueError, RuntimeError):
                                m_nl.fit(X.iloc[tr_idx], y_fit[tr_idx], **fit_kwargs)
                            nl_raw = m_nl.predict(X.iloc[val_idx])
                            if self._use_physics_residual:
                                X_val_nl = X.iloc[val_idx]
                                physics_val_nl = (
                                    X_val_nl["physics_kwh"].values if "physics_kwh" in X_val_nl.columns else 0.0
                                )
                                nl_pred = np.maximum(0, physics_val_nl + nl_raw)
                            else:
                                nl_pred = np.expm1(nl_raw)
                            sweep_maes[nl] = float(mae_fn(y[val_idx], nl_pred))
                        best_num_leaves = min(sweep_maes, key=sweep_maes.__getitem__)
                        _LOGGER.info(
                            "num_leaves sweep on last fold: %s → best=%d (MAE=%.4f)",
                            {nl: round(v, 4) for nl, v in sweep_maes.items()},
                            best_num_leaves,
                            sweep_maes[best_num_leaves],
                        )
                        # Refit last fold with the winning num_leaves for consistent n_est
                        m = _build_model(lgb, GBR, num_leaves=best_num_leaves)
                        try:
                            m.fit(
                                X.iloc[tr_idx],
                                y_fit[tr_idx],
                                eval_X=X.iloc[val_idx],
                                eval_y=y_fit[val_idx],
                                callbacks=[
                                    lgb.early_stopping(50, verbose=False),
                                    lgb.log_evaluation(-1),
                                ],
                                **fit_kwargs,
                            )
                            best_n_est = getattr(m, "best_iteration_", best_n_est)
                        except (ValueError, RuntimeError):
                            m.fit(X.iloc[tr_idx], y_fit[tr_idx], **fit_kwargs)
                        fold_maes.append(sweep_maes[best_num_leaves])
                    else:
                        m = _build_model(lgb, GBR)
                        if lgb is not None:
                            try:
                                m.fit(
                                    X.iloc[tr_idx],
                                    y_fit[tr_idx],
                                    eval_X=X.iloc[val_idx],
                                    eval_y=y_fit[val_idx],
                                    callbacks=[
                                        lgb.early_stopping(50, verbose=False),
                                        lgb.log_evaluation(-1),
                                    ],
                                    **fit_kwargs,
                                )
                                best_n_est = getattr(m, "best_iteration_", best_n_est)
                            except (ValueError, RuntimeError) as _es_exc:
                                _LOGGER.debug(
                                    "Early stopping failed for fold: %s — using fixed estimators",
                                    _es_exc,
                                )
                                m.fit(X.iloc[tr_idx], y_fit[tr_idx], **fit_kwargs)
                        else:
                            m.fit(X.iloc[tr_idx], y_fit[tr_idx], **fit_kwargs)
                        # MAE reported in original kWh space (expm1 undoes log1p, or
                        # reconstructed from physics + residual in Phase 2)
                        fold_raw = m.predict(X.iloc[val_idx])
                        if self._use_physics_residual:
                            X_val = X.iloc[val_idx]
                            physics_val = X_val["physics_kwh"].values if "physics_kwh" in X_val.columns else 0.0
                            fold_pred = np.maximum(0, physics_val + fold_raw)
                        else:
                            fold_pred = np.expm1(fold_raw)
                        fold_maes.append(float(mae_fn(y[val_idx], fold_pred)))
                cv_mae = round(float(np.mean(fold_maes)), 4)
                cv_std = round(float(np.std(fold_maes)), 4)
                _LOGGER.info(
                    "CV fold MAEs: %s → mean=%.4f ± %.4f",
                    [round(v, 4) for v in fold_maes],
                    cv_mae,
                    cv_std,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                _LOGGER.warning("CV MAE failed: %s", exc)

        if best_n_est is not None:
            _LOGGER.info("LightGBM early stopping: using %d estimators (from last CV fold)", best_n_est)

        # ── Final model on all data ─────────────────────────────────────────
        model = _build_model(lgb, GBR, n_estimators=best_n_est, num_leaves=best_num_leaves)
        if sample_weight is not None:
            model.fit(X, y_fit, sample_weight=sample_weight)
        else:
            model.fit(X, y_fit)

        # ── Feature importance (top 10, log-scale gains) ─────────────────────
        if hasattr(model, "feature_importances_"):
            top = sorted(
                zip(feature_cols, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
            _LOGGER.info(
                "Feature importances (top 10): %s",
                ", ".join(f"{n}={v:.4f}" for n, v in top),
            )

        # ── Hold-out MAE (last 10%) — trained on train-split only for honest estimate ─
        # The final `model` is still trained on all rows for best accuracy; this
        # temporary holdout model is only for the MAE fallback used when CV is skipped.
        holdout_mae = None
        self.last_residual_mae = None
        if mae_fn is not None:
            split = max(int(len(X) * HOLDOUT_FRACTION), len(X) - MIN_CV_ROWS)
            try:
                ho_model = _build_model(lgb, GBR, n_estimators=best_n_est, num_leaves=best_num_leaves)
                if sample_weight is not None:
                    ho_model.fit(X.iloc[:split], y_fit[:split], sample_weight=sample_weight[:split])
                else:
                    ho_model.fit(X.iloc[:split], y_fit[:split])

                X_ho = X.iloc[split:]
                ho_raw = ho_model.predict(X_ho)
                if self._use_physics_residual:
                    physics_ho = X_ho["physics_kwh"].values if "physics_kwh" in X_ho.columns else 0.0
                    gross_pred = np.maximum(0, physics_ho + ho_raw)
                    holdout_mae = round(float(mae_fn(y[split:], gross_pred)), 4)
                    self.last_residual_mae = round(float(mae_fn(y_fit[split:], ho_raw)), 4)
                else:
                    holdout_mae = round(float(mae_fn(y[split:], np.expm1(ho_raw))), 4)
            except (ValueError, IndexError):
                pass

        self.model = model
        self.feature_cols = feature_cols
        self.last_trained = datetime.now()
        self.last_mae = cv_mae if cv_mae is not None else holdout_mae
        self.last_cv_mae = cv_mae
        self.engine = engine_name
        self._log_transform = not self._use_physics_residual
        self._canton = canton
        self._country = country
        self._sub_sensor_prefixes = list(sub_sensors_dict.keys()) if sub_sensors_dict else []
        self._save()

        mae_str = f"cv_MAE={cv_mae:.4f}" if cv_mae else (f"holdout_MAE={holdout_mae:.4f}" if holdout_mae else "MAE=n/a")
        _LOGGER.info(
            "Model trained | rows=%d | engine=%s | features=%d | %s",
            len(df),
            engine_name,
            len(feature_cols),
            mae_str,
        )
        if self._use_physics_residual and self.last_residual_mae is not None:
            _LOGGER.info(
                f"Holdout MAE (gross kWh)={holdout_mae}, residual MAE (internal diagnostic)={self.last_residual_mae}"
            )

        # ── Phase 1 physics validation diagnostic (Plan B Task 6) ────────────
        # Read-only: reports whether physics_kwh is a top-5 SHAP feature and
        # well-calibrated (OLS slope ~1.0) against gross_kwh on the same
        # holdout split used for holdout_mae above. Never flips config —
        # `self.model`/`self.engine` are already assigned above so the SHAP
        # codepath in _validate_physics_phase1 sees the freshly trained model.
        self.physics_phase1_diagnostic: dict | None = None
        if physics_model is not None and "physics_kwh" in feature_cols and holdout_mae is not None:
            X_ho = X.iloc[split:]
            y_ho_gross = y[split:]  # gross_kwh, not log-transformed
            physics_ho = X_ho["physics_kwh"].values if "physics_kwh" in X_ho.columns else None
            self.physics_phase1_diagnostic = self._validate_physics_phase1(X_ho, y_ho_gross, physics_ho)

        # ── Quantile models for prediction intervals (CQR) ───────────────────
        # q10/q90 trained on random 85% of rows; random 15% (≥20 rows) used for
        # split-conformal calibration. Random split (not tail) is required so
        # CQR's exchangeability assumption holds, guaranteeing ≥80% marginal
        # coverage. Wrapped so a quantile failure never interrupts normal operation.
        try:
            rng = np.random.default_rng(42)
            cal_size = max(20, int(len(X) * 0.15))
            cal_idx = rng.choice(len(X), size=cal_size, replace=False)
            train_mask = np.ones(len(X), dtype=bool)
            train_mask[cal_idx] = False
            X_qtrain = X.iloc[train_mask]
            y_qtrain = y_fit[train_mask]
            X_cal = X.iloc[cal_idx]
            y_cal_log = y_fit[cal_idx]
            sw_qtrain = sample_weight[train_mask] if sample_weight is not None else None

            q10 = _build_quantile_model(lgb, GBR, alpha=0.1, n_estimators=best_n_est)
            q90 = _build_quantile_model(lgb, GBR, alpha=0.9, n_estimators=best_n_est)
            if sw_qtrain is not None:
                q10.fit(X_qtrain, y_qtrain, sample_weight=sw_qtrain)
                q90.fit(X_qtrain, y_qtrain, sample_weight=sw_qtrain)
            else:
                q10.fit(X_qtrain, y_qtrain)
                q90.fit(X_qtrain, y_qtrain)
            self._model_q10 = q10
            self._model_q90 = q90
            self._calibrate_intervals(X_cal, y_cal_log)
            self._save_quantile_models()
            self._save_interval_correction()
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
        away_series: pd.Series | None = None,  # 48-value Series indexed by timestamp
        people_home_series: pd.Series | None = None,  # 48-value Series indexed by timestamp
        climate_recent: dict[str, pd.DataFrame] | None = None,
        dhw_recent: pd.DataFrame | None = None,
        room_areas: dict[str, float] | None = None,
        heating_active_series: pd.Series | None = None,
        setpoint_on: float | None = None,
        setpoint_off: float | None = None,
        physics_model: ThermalPhysicsModel | None = None,
        heating_buffer_temp_recent: pd.DataFrame
        | None = None,  # cols: timestamp, heating_buffer_temp — most recent reading(s)
        dhw_schedule_override: dict | None = None,
    ):
        """Build the 48-hour feature matrix shared by predict() and predict_intervals().

        Returns (future_hours, X) where future_hours is a naive DatetimeIndex
        and X is the filled DataFrame ready for model.predict().
        """
        import numpy as np
        import pandas as pd

        now_naive = pd.Timestamp.now(tz=self._timezone).tz_localize(None)
        future_hours = pd.date_range(start=now_naive.floor("1h"), periods=48, freq="1h")

        future_df = pd.DataFrame({"timestamp": future_hours, "gross_kwh": np.nan})
        future_df = _add_lag_and_rolling_prediction(future_df, recent_actuals)

        # 7-day per-hour-of-day mean for smarter short-lag NaN fill.
        # Shape (24,) indexed by hour 0-23; NaN where hour bucket has no data.
        _hod_means: np.ndarray | None = None
        if recent_actuals is not None and not recent_actuals.empty:
            _ra = recent_actuals.tail(168)  # last 7 days (168 h)
            _hod_series = _ra.groupby(_ra["timestamp"].dt.hour)["gross_kwh"].mean().reindex(range(24))
            _hod_means = _hod_series.to_numpy()

        future_df = _add_sub_sensor_lags_prediction(future_df, sub_sensors_recent or {})

        outdoor_pred_df: pd.DataFrame | None = None
        if live_temp is not None and "outdoor_temp_live" in self.feature_cols:
            outdoor_pred_df = _build_prediction_temp_df(future_hours, forecast_df, live_temp)

        # ── Indoor temperature projection ────────────────────────────────────
        # Replace stale climate_recent with RC-ODE projected temperatures so
        # that thermal_pressure is non-zero for all 48 prediction hours.
        climate_dfs_for_features = climate_recent
        if climate_recent:
            outdoor_temp_series = (
                outdoor_pred_df.set_index("timestamp")["outdoor_temp_live"]
                if outdoor_pred_df is not None
                else forecast_df.set_index(pd.to_datetime(forecast_df["timestamp"]))["temp_c"]
            )
            climate_dfs_for_features = _project_indoor_temps(
                climate_recent,
                future_hours,
                outdoor_temp_series,
                tau_hours=self._tau_hours,
                heating_active_series=heating_active_series,
                setpoint_on=setpoint_on,
                setpoint_off=setpoint_off,
            )

        ha_df_for_features = None
        if heating_active_series is not None:
            ha_df_for_features = heating_active_series.rename_axis("timestamp").rename("heating_active").reset_index()

        # Extend the 48h forecast with recent historical weather so that
        # shift(168) and shift(336) in _engineer_features produce real values
        # instead of NaN — enabling the temperature-delta gating to fire.
        if self._weather_tail is not None and not self._weather_tail.empty:
            import pandas as _pd

            _extended = (
                _pd.concat([self._weather_tail, forecast_df], ignore_index=True)
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
        else:
            _extended = forecast_df

        # ── Physics-ML hybrid: physics baseline series at prediction time ────
        # (Plan B Task 5). Uses the raw climate_recent (not climate_dfs_for_features,
        # which is the RC-ODE-projected version built for _engineer_features()'s own
        # climate feature) — predict_series() does its own internal projection.
        physics_kwh_series = None
        if physics_model is not None:
            try:
                physics_kwh_series = physics_model.predict_series(
                    forecast_df,
                    climate_recent=climate_recent,
                    dhw_recent=dhw_recent,
                    room_areas=room_areas,
                    heating_active_series=heating_active_series,
                    setpoint_on=setpoint_on,
                    setpoint_off=setpoint_off,
                    dhw_schedule_override=dhw_schedule_override,
                )
            except Exception as e:
                _LOGGER.warning(f"Physics baseline prediction failed at predict time: {e} — falling back to ML-only")
                physics_kwh_series = None

        # Stashed for predict()'s Phase 2 reconstruction step. predict() may consume
        # a pre-built (future_hours, X) tuple via its `_prepared` param rather than
        # calling this method directly (see energy_forecast.py's threading of
        # _prepare_prediction_X()'s output into predict()/predict_intervals()/
        # shap_summary()); this attribute lets it recover physics_kwh_series without
        # widening that shared 2-tuple contract.
        self._last_physics_kwh_series = physics_kwh_series

        heating_buffer_temp_series = None
        if heating_buffer_temp_recent is not None and not heating_buffer_temp_recent.empty:
            # hold the most recent reading flat across the forecast horizon — same "recent" pattern
            # used for dhw_recent/climate_recent elsewhere in this method
            latest_val = float(heating_buffer_temp_recent.sort_values("timestamp")["heating_buffer_temp"].iloc[-1])
            heating_buffer_temp_series = pd.Series(latest_val, index=pd.DatetimeIndex(forecast_df["timestamp"]))

        feat_df = _engineer_features(
            future_df,
            _extended,
            outdoor_pred_df,
            canton=self._canton,
            likely_ev_hours=self._likely_ev_hours,
            climate_dfs=climate_dfs_for_features,
            dhw_df=dhw_recent,
            tau_hours=self._tau_hours,
            room_areas=room_areas,
            heating_active_df=ha_df_for_features,
            physics_kwh_series=physics_kwh_series,
            heating_buffer_temp_series=heating_buffer_temp_series,
        )

        # ── Model-artifact portability fallback ──────────────────────────────
        # A saved model may have physics_kwh in feature_cols (trained with physics
        # enabled) while physics_model is None at predict time (sensor outage,
        # config change disabling physics). Fill with 0.0 rather than raising
        # KeyError when feat_df[self.feature_cols] is sliced below.
        if "physics_kwh" in self.feature_cols and "physics_kwh" not in feat_df.columns:
            feat_df["physics_kwh"] = 0.0
            _LOGGER.warning(
                "physics_kwh in trained feature list but physics_model disabled at predict time — filling with 0.0"
            )

        # Same portability guarantee for heating_buffer_temp: a saved model may have
        # heating_buffer_temp in feature_cols (trained with a buffer-temp sensor available)
        # while heating_buffer_temp_recent is None/empty at predict time (sensor outage,
        # config change, or predict_scenario() which never supplies it). Fill with 0.0
        # rather than raising KeyError when feat_df[self.feature_cols] is sliced below.
        if "heating_buffer_temp" in self.feature_cols and "heating_buffer_temp" not in feat_df.columns:
            feat_df["heating_buffer_temp"] = 0.0
            _LOGGER.warning(
                "heating_buffer_temp in trained feature list but sensor unavailable at predict time — filling with 0.0"
            )

        # ── Daily Regime Profile Prediction ──────────────────────────────────
        if self._regime_model and self._clusterer and "regime_kwh" in self.feature_cols:
            try:
                # 1. Prepare daily features for Today and Tomorrow
                # Convert hourly away/occupancy series → DataFrames for regime predictor
                _away_df_r = None
                if away_series is not None and not away_series.empty:
                    _away_df_r = away_series.rename("is_away").reset_index().rename(columns={"index": "timestamp"})
                _presence_df_r = None
                if people_home_series is not None and not people_home_series.empty:
                    _presence_df_r = (
                        people_home_series.rename("people_home").reset_index().rename(columns={"index": "timestamp"})
                    )
                daily_features = _prepare_daily_regime_features(
                    forecast_df,
                    country=self._country,
                    canton=self._canton,
                    away_df=_away_df_r,
                    presence_df=_presence_df_r,
                )

                # 2. Predict regime IDs
                predicted_labels = self._regime_model.predict(daily_features)
                regime_map = dict(zip(daily_features.index, predicted_labels))

                # 3. Populate regime_kwh column (vectorized)
                # Mirror training ffill so partial/gap days get a filled label
                # rather than -1 (which would zero out regime_kwh).
                ts_idx = pd.to_datetime(feat_df["timestamp"])
                hourly_labels = ts_idx.dt.date
                date_series = pd.Series(hourly_labels)
                label_series = date_series.map(regime_map).ffill().fillna(-1).astype(int)
                mapped_labels = label_series.values

                regime_vals = np.zeros(len(ts_idx))
                valid_mask = mapped_labels >= 0
                if valid_mask.any():
                    hours = ts_idx.dt.hour.values
                    n_clusters = self._clusterer.centroids.shape[0]
                    safe_labels = np.clip(mapped_labels[valid_mask], 0, n_clusters - 1)
                    regime_vals[valid_mask] = self._clusterer.centroids[safe_labels, hours[valid_mask]]
                feat_df["regime_kwh"] = regime_vals
            except Exception as e:
                _LOGGER.warning(f"Regime prediction failed during forecast: {e}")
                feat_df["regime_kwh"] = 0.0
        else:
            feat_df["regime_kwh"] = 0.0

        # Temperature-gate regime_kwh — same 8°C/168h discount as tgated lags.
        # Centroids are trained on mostly heating-season data; suppress them
        # during warm-season transitions so they don't anchor predictions high.
        if "temp_c" in feat_df.columns and "temp_lag_168h" in feat_df.columns:
            _delta = (feat_df["temp_c"] - feat_df["temp_lag_168h"]).clip(lower=0)
            feat_df["regime_kwh"] *= 1.0 - (_delta / 8.0).clip(upper=1.0)

        # ── Away / vacation flag ─────────────────────────────────────────────
        # away_series is a 48-value Series indexed by naive prediction timestamps.
        # Merge by timestamp; fill unmatched rows with 0 (default: not away).
        if away_series is not None and not away_series.empty:
            away_map = away_series.reindex(pd.to_datetime(feat_df["timestamp"])).fillna(0)
            feat_df["is_away"] = away_map.values.astype(int)
        else:
            feat_df["is_away"] = 0

        # ── Occupancy / people_home ─────────────────────────────────────────
        # people_home_series is a 48-value Series indexed by naive prediction timestamps.
        # Merge by timestamp; fill unmatched rows with 0 (default: nobody home).
        if people_home_series is not None and not people_home_series.empty:
            ph_map = people_home_series.reindex(pd.to_datetime(feat_df["timestamp"])).fillna(0)
            feat_df["people_home"] = ph_map.values.astype(int)
        else:
            feat_df["people_home"] = 0

        # Overwrite hours_ahead with the actual prediction horizon (0–47).
        # _engineer_features sets it to 0 (training-row semantics); here we
        # populate the real horizon so the model can learn near-vs-far bias.
        feat_df["hours_ahead"] = np.arange(len(feat_df))

        # Build per-HOW fill lookups once (for lag/rolling cols with HOW-specific medians).
        # For each such column, map hour_of_week → stored HOW median.
        how_fill_lookup: dict[str, pd.Series] = {}
        if self._feature_medians_by_how and "hour_of_week" in feat_df.columns:
            sample_meds = next(iter(self._feature_medians_by_how.values()), {})
            for col in sample_meds:
                how_fill_lookup[col] = pd.Series(
                    {how: meds.get(col) for how, meds in self._feature_medians_by_how.items()}
                )

        # HOD-aware fill for short lags (< 48h) before the flat median fallback below.
        # Remaining NaN (bucket had no actuals) falls through to HOW/global median.
        if _hod_means is not None:
            _future_ts = pd.to_datetime(feat_df["timestamp"])
            for _lag in LAG_HOURS:
                if _lag >= 48:
                    continue
                _col = f"lag_{_lag}h"
                if _col not in feat_df.columns:
                    continue
                _nan_mask = feat_df[_col].isna()
                if not _nan_mask.any():
                    continue
                _logical_hod = (_future_ts[_nan_mask] - pd.Timedelta(hours=_lag)).dt.hour
                _fill_vals = _hod_means[_logical_hod.to_numpy()]
                _valid = ~np.isnan(_fill_vals)
                _idx = _nan_mask[_nan_mask].index[_valid]
                feat_df.loc[_idx, _col] = _fill_vals[_valid]

        for col in self.feature_cols:
            if col in feat_df.columns and feat_df[col].isna().any():
                global_med = self._feature_medians.get(col, 0.0)
                if col in how_fill_lookup and "hour_of_week" in feat_df.columns:
                    # HOW-specific fill; fall back to global median for empty HOW buckets
                    how_fill = feat_df["hour_of_week"].map(how_fill_lookup[col]).fillna(global_med)
                    feat_df[col] = feat_df[col].where(feat_df[col].notna(), how_fill)
                else:
                    feat_df[col] = feat_df[col].fillna(global_med)

        X = feat_df[self.feature_cols].fillna(0)
        return future_hours, X

    def predict(
        self,
        forecast_df: pd.DataFrame,  # naive timestamps
        live_temp: float | None,
        recent_actuals: pd.DataFrame | None = None,  # naive timestamps, cols: timestamp, gross_kwh
        sub_sensors_recent: dict | None = None,  # {prefix: DataFrame[timestamp, kwh]}
        away_series: pd.Series | None = None,  # 48-value Series indexed by timestamp
        people_home_series: pd.Series | None = None,  # 48-value Series indexed by timestamp
        climate_recent: dict[str, pd.DataFrame] | None = None,
        dhw_recent: pd.DataFrame | None = None,
        room_areas: dict[str, float] | None = None,
        heating_active_series: pd.Series | None = None,
        setpoint_on: float | None = None,
        setpoint_off: float | None = None,
        physics_model: ThermalPhysicsModel | None = None,
        heating_buffer_temp_recent: pd.DataFrame | None = None,  # cols: timestamp, heating_buffer_temp
        dhw_schedule_override: dict | None = None,
        _prepared: tuple | None = None,
    ) -> pd.DataFrame:
        """Return 48-hour DataFrame [timestamp (naive), predicted_kwh]."""
        import numpy as np
        import pandas as pd

        if self.model is None:
            raise RuntimeError("Model not yet trained.")

        if _prepared is not None:
            future_hours, X = _prepared
        else:
            future_hours, X = self._prepare_prediction_X(
                forecast_df,
                live_temp,
                recent_actuals,
                sub_sensors_recent,
                away_series,
                people_home_series,
                climate_recent=climate_recent,
                dhw_recent=dhw_recent,
                room_areas=room_areas,
                heating_active_series=heating_active_series,
                setpoint_on=setpoint_on,
                setpoint_off=setpoint_off,
                physics_model=physics_model,
                heating_buffer_temp_recent=heating_buffer_temp_recent,
                dhw_schedule_override=dhw_schedule_override,
            )
        if "thermal_pressure_net" in X.columns:
            self._latest_thermal_pressure_net = float(X["thermal_pressure_net"].iloc[0])
        else:
            # Defensive fallback: thermal_pressure_net is always in _FEATURES_BASE,
            # so this branch should never execute in practice.
            self._latest_thermal_pressure_net = 0.0
        preds = self.model.predict(X)
        if self._log_transform:
            preds = np.expm1(preds)

        if self._use_physics_residual and physics_model is not None:
            try:
                physics_kwh_series = self._last_physics_kwh_series
                if physics_kwh_series is None:
                    raise ValueError("physics baseline unavailable")
                physics_baseline = (
                    physics_kwh_series.reindex(pd.DatetimeIndex(forecast_df["timestamp"])).fillna(0).values
                )
                preds = physics_baseline + preds
            except Exception as e:
                _LOGGER.warning(f"Phase 2 physics reconstruction failed: {e} — falling back to ML-only output")

        preds = np.maximum(0, preds)
        return pd.DataFrame({"timestamp": future_hours, "predicted_kwh": preds})

    def predict_intervals(
        self,
        forecast_df: pd.DataFrame,
        live_temp: float | None,
        recent_actuals: pd.DataFrame | None = None,
        sub_sensors_recent: dict | None = None,
        away_series: pd.Series | None = None,  # 48-value Series indexed by timestamp
        people_home_series: pd.Series | None = None,  # 48-value Series indexed by timestamp
        climate_recent: dict[str, pd.DataFrame] | None = None,
        dhw_recent: pd.DataFrame | None = None,
        room_areas: dict[str, float] | None = None,
        heating_active_series: pd.Series | None = None,
        setpoint_on: float | None = None,
        setpoint_off: float | None = None,
        _prepared: tuple | None = None,
    ) -> pd.DataFrame | None:
        """Return 48-hour DataFrame [timestamp, low_kwh, high_kwh], or None.

        Returns None when quantile models are not yet trained.  low_kwh and
        high_kwh are guaranteed non-negative and ordered (low ≤ high).
        """
        import numpy as np
        import pandas as pd

        if self._model_q10 is None or self._model_q90 is None:
            return None

        if _prepared is not None:
            future_hours, X = _prepared
        else:
            future_hours, X = self._prepare_prediction_X(
                forecast_df,
                live_temp,
                recent_actuals,
                sub_sensors_recent,
                away_series,
                people_home_series,
                climate_recent=climate_recent,
                dhw_recent=dhw_recent,
                room_areas=room_areas,
                heating_active_series=heating_active_series,
                setpoint_on=setpoint_on,
                setpoint_off=setpoint_off,
            )
        low = self._model_q10.predict(X)
        high = self._model_q90.predict(X)
        if self._log_transform:
            correction = self._interval_correction  # 0.0 when uncalibrated
            low -= correction
            high += correction
            low = np.expm1(low)
            high = np.expm1(high)
        low = np.maximum(0, low)
        high = np.maximum(0, high)
        # Enforce ordering in case of quantile crossing
        low, high = np.minimum(low, high), np.maximum(low, high)
        return pd.DataFrame({"timestamp": future_hours, "low_kwh": low, "high_kwh": high})

    def predict_scenario(
        self,
        forecast_df: pd.DataFrame,
        live_temp: float | None,
        schedule: dict[str, str | None],
        recent_actuals: pd.DataFrame | None = None,
        sub_sensors_recent: dict | None = None,
        away_series: pd.Series | None = None,
        people_home_series: pd.Series | None = None,
        climate_recent: dict[str, pd.DataFrame] | None = None,
        dhw_recent: pd.DataFrame | None = None,
        heating_active_series: pd.Series | None = None,
        setpoint_on: float | None = None,
        setpoint_off: float | None = None,
        room_areas: dict[str, float] | None = None,
        physics_model: ThermalPhysicsModel | None = None,
        dhw_schedule_override: dict | None = None,
    ) -> pd.DataFrame:
        """Return composite 48h forecast [timestamp, predicted_kwh, delta_kwh].

        Runs the baseline predict() then overlays the appliance profiles from
        *schedule* using the learned appliance signatures. If *dhw_schedule_override*
        and *physics_model* are both given, also folds in the DHW-schedule delta
        (scenario vs. the natural/no-override baseline) so `delta_kwh` reflects
        the combined signed difference of both overlays vs. the natural baseline.

        Raises:
            RuntimeError: if the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model not yet trained.")

        natural_baseline_df = self.predict(
            forecast_df,
            live_temp,
            recent_actuals,
            sub_sensors_recent=sub_sensors_recent,
            away_series=away_series,
            people_home_series=people_home_series,
            climate_recent=climate_recent,
            dhw_recent=dhw_recent,
            heating_active_series=heating_active_series,
            setpoint_on=setpoint_on,
            setpoint_off=setpoint_off,
            room_areas=room_areas,
            physics_model=physics_model,
            # predict_series's silent committed_override fallback was removed (Phase A Task 6) —
            # this call site must ask for the live committed override explicitly so the "natural
            # baseline" still reflects whatever's actually committed right now (e.g. an already-
            # armed legionella boost) when scoring a *different* candidate dhw_schedule_override.
            dhw_schedule_override=(
                physics_model._schedule.get("committed_override") if physics_model is not None else None
            ),
        )

        if dhw_schedule_override is not None and physics_model is not None:
            scenario_baseline_df = self.predict(
                forecast_df,
                live_temp,
                recent_actuals,
                sub_sensors_recent=sub_sensors_recent,
                away_series=away_series,
                people_home_series=people_home_series,
                climate_recent=climate_recent,
                dhw_recent=dhw_recent,
                heating_active_series=heating_active_series,
                setpoint_on=setpoint_on,
                setpoint_off=setpoint_off,
                room_areas=room_areas,
                physics_model=physics_model,
                dhw_schedule_override=dhw_schedule_override,
            )
        else:
            scenario_baseline_df = natural_baseline_df

        result = _composite_forecast(scenario_baseline_df, schedule, self._appliance_signatures)

        if dhw_schedule_override is not None and physics_model is not None:
            dhw_delta = (scenario_baseline_df["predicted_kwh"] - natural_baseline_df["predicted_kwh"]).to_numpy()
            result["delta_kwh"] = result["delta_kwh"] + dhw_delta

        return result

    def _validate_physics_phase1(
        self, X_holdout: pd.DataFrame, y_holdout_gross: np.ndarray, physics_kwh_holdout: np.ndarray | None
    ) -> dict:
        """Phase 1 read-only diagnostic: is `physics_kwh` a top-5 SHAP feature, well-calibrated (slope ~1.0)?

        Reports whether the physics baseline is (a) among the model's top-5 most
        important features by mean absolute SHAP contribution, and (b) linearly
        calibrated against gross_kwh (OLS slope through the origin in [0.8, 1.2]).
        Purely diagnostic — never flips ``use_physics_residual``. That remains a
        manual operator decision (spec §2.1) once Phase 2 is planned.

        Never raises: any missing precondition (no physics_kwh column, no
        physics values, untrained model) yields the safe all-``None``/``False``
        default dict.
        """
        import numpy as np
        import pandas as pd

        result = {"shap_rank": None, "shap_top5": False, "ols_slope": None, "calibration_good": False}
        if "physics_kwh" not in X_holdout.columns or physics_kwh_holdout is None or self.model is None:
            return result

        if self.engine == "LightGBM":
            contrib = self.model.predict(X_holdout, pred_contrib=True)
            mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
            ranked = pd.Series(mean_abs, index=X_holdout.columns).sort_values(ascending=False)
            rank = int(ranked.index.get_loc("physics_kwh")) + 1  # 1-indexed
            result["shap_rank"] = rank
            result["shap_top5"] = rank <= 5

        x = np.asarray(physics_kwh_holdout, dtype=float)
        y = np.asarray(y_holdout_gross, dtype=float)
        if np.sum(x**2) > 1e-9:
            slope = float(np.sum(x * y) / np.sum(x**2))
            result["ols_slope"] = slope
            result["calibration_good"] = result["shap_top5"] and (0.8 <= slope <= 1.2)

        if result["calibration_good"]:
            _LOGGER.info(
                "Phase 1 physics validation: PASS (shap_rank=%s, ols_slope=%.2f) — "
                "physics signal is well-calibrated and predictive",
                result["shap_rank"],
                result["ols_slope"],
            )
        else:
            _LOGGER.info(
                "Phase 1 physics validation: not yet ready for Phase 2 (shap_rank=%s, ols_slope=%s)",
                result["shap_rank"],
                result["ols_slope"],
            )
        return result

    def shap_summary(
        self,
        forecast_df: pd.DataFrame,
        live_temp: float | None,
        recent_actuals: pd.DataFrame | None = None,
        sub_sensors_recent: dict | None = None,
        away_series: pd.Series | None = None,
        people_home_series: pd.Series | None = None,
        climate_recent: dict[str, pd.DataFrame] | None = None,
        dhw_recent: pd.DataFrame | None = None,
        room_areas: dict[str, float] | None = None,
        n: int = 5,
        heating_active_series: pd.Series | None = None,
        setpoint_on: float | None = None,
        setpoint_off: float | None = None,
        physics_model: ThermalPhysicsModel | None = None,
        heating_buffer_temp_recent: pd.DataFrame | None = None,  # cols: timestamp, heating_buffer_temp
        _prepared: tuple | None = None,
    ) -> dict[str, float]:
        """Return the top-N driving features for today's prediction slice.

        Uses LightGBM's native TreeSHAP (``pred_contrib=True``) when the engine
        is LightGBM; falls back to global ``feature_importances_`` for GBR.
        Values are mean absolute SHAP contributions over today's hours (or all
        48 hours if today's slice is empty).  When ``_log_transform=True`` the
        contributions are in log-space — still valid for ranking.

        Returns an empty dict when the model is not trained or ``n <= 0``.
        """
        import numpy as np
        import pandas as pd

        if self.model is None or n <= 0:
            return {}

        if _prepared is not None:
            future_hours, X = _prepared
        else:
            future_hours, X = self._prepare_prediction_X(
                forecast_df,
                live_temp,
                recent_actuals,
                sub_sensors_recent,
                away_series,
                people_home_series=people_home_series,
                climate_recent=climate_recent,
                dhw_recent=dhw_recent,
                room_areas=room_areas,
                heating_active_series=heating_active_series,
                setpoint_on=setpoint_on,
                setpoint_off=setpoint_off,
                physics_model=physics_model,
                heating_buffer_temp_recent=heating_buffer_temp_recent,
            )

        # Filter to today's local date; fall back to all rows if none match
        today = pd.Timestamp.now(tz=self._timezone).tz_localize(None).normalize()
        mask = (pd.Series(future_hours) >= today) & (pd.Series(future_hours) < today + pd.Timedelta(days=1))
        X_slice = X[mask.values] if mask.sum() >= 3 else X

        if self.engine == "LightGBM":
            # pred_contrib=True → shape (n_rows, n_features + 1); last column is bias
            contrib = self.model.predict(X_slice, pred_contrib=True)
            mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
        else:
            # GBR: global feature_importances_ (no per-prediction SHAP)
            _LOGGER.debug("shap_summary: GBR engine uses global feature_importances_ (not per-prediction SHAP)")
            mean_abs = self.model.feature_importances_.astype(float)

        # Pair with feature names and return top-N descending
        pairs = sorted(
            zip(self.feature_cols, mean_abs.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return {name: round(float(val), 6) for name, val in pairs[:n]}

    def hours_since_trained(self) -> float:
        if self.last_trained == datetime.min:
            return float("inf")
        return (datetime.now() - self.last_trained).total_seconds() / 3600

    # ── Persistence ───────────────────────────────────────────────────────────

    def _archive_current(self) -> None:
        """Copy the current model files to a timestamped archive subdirectory.

        Called before overwriting the active model files.  Skipped when no
        model exists yet (first-ever save).  Prunes the oldest archive
        directories beyond self._model_archive_count.
        """
        if not self._model_path.exists():
            return  # nothing to archive on first save
        try:
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
            snap_dir = self._archive_dir / stamp
            snap_dir.mkdir(parents=True, exist_ok=True)
            _artifacts = [
                self._model_path,
                self._meta_path,
                self._model_q10_path,
                self._model_q90_path,
                self._interval_correction_path,
                self._model_dir / "clusterer.pkl",
                self._model_dir / "regime_model.pkl",
            ]
            for src in _artifacts:
                if src.exists():
                    shutil.copy2(src, snap_dir / src.name)
            # Copy SHA-256 sidecars for pickle files
            for src in [a for a in _artifacts if a.suffix == ".pkl"]:  # only pickles have sidecars
                sidecar = src.with_suffix(src.suffix + ".sha256")
                if sidecar.exists():
                    shutil.copy2(sidecar, snap_dir / sidecar.name)
            # Prune oldest archives beyond the configured limit
            dirs = sorted(self._archive_dir.iterdir())
            while len(dirs) > self._model_archive_count:
                shutil.rmtree(dirs.pop(0))
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("Model archive failed (save will continue): %s", exc)

    def rollback_model(self) -> bool:
        """Restore model files from the most recent archive and reload in-memory state.

        Returns True on success, False when no archive exists (no-op).
        Logs a WARNING naming the archive snapshot being restored from.
        """
        try:
            if not self._archive_dir.exists():
                _LOGGER.warning("rollback_model: no archive available — nothing to restore.")
                return False
            dirs = sorted(d for d in self._archive_dir.iterdir() if d.is_dir())
            if not dirs:
                _LOGGER.warning("rollback_model: no archive available — nothing to restore.")
                return False
            latest = dirs[-1]
            _LOGGER.warning("rollback_model: restoring from archive %s", latest.name)
            for src in latest.iterdir():
                shutil.copy2(src, self._model_dir / src.name)
            self._load()
            return True
        except Exception as exc:  # noqa: BLE001
            _LOGGER.error("rollback_model failed: %s", exc)
            return False

    def _save(self) -> None:
        self._archive_current()
        with open(self._model_path, "wb") as fh:
            pickle.dump(self.model, fh)
        _write_hash(self._model_path)

        # Save regime components if present
        if self._clusterer:
            with open(self._model_dir / "clusterer.pkl", "wb") as fh:
                pickle.dump(self._clusterer, fh)
        if self._regime_model:
            with open(self._model_dir / "regime_model.pkl", "wb") as fh:
                pickle.dump(self._regime_model, fh)

        meta = {
            "feature_cols": self.feature_cols,
            "last_trained": self.last_trained,
            "last_mae": self.last_mae,
            "last_cv_mae": self.last_cv_mae,
            "engine": self.engine,
            "feature_medians": self._feature_medians,
            "feature_medians_by_how": self._feature_medians_by_how,
            "log_transform": self._log_transform,
            "use_physics_residual": self._use_physics_residual,
            "canton": self._canton,
            "country": self._country,
            "likely_ev_hours": self._likely_ev_hours,
            "sub_sensor_prefixes": self._sub_sensor_prefixes,
            "tau_hours": self._tau_hours,
            "tau_anchor_hours": self._tau_anchor_hours,
            "tau_anchor_ts": self._tau_anchor_ts,
            "tau_long_anchor_hours": self._tau_long_anchor_hours,
            "tau_long_anchor_ts": self._tau_long_anchor_ts,
            "enable_regimes": self._enable_regimes,
            "regime_count": self._regime_count,
            "weather_tail": self._weather_tail,
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
                _LOGGER.warning("Quantile model integrity check failed (%s) — discarding.", path)
                continue
            try:
                with open(path, "rb") as fh:
                    setattr(self, attr, pickle.load(fh))
            except (pickle.UnpicklingError, EOFError, OSError) as exc:
                _LOGGER.warning("Could not load quantile model %s: %s", path.name, exc)
        self._load_interval_correction()

    def _calibrate_intervals(self, X_cal: pd.DataFrame, y_cal_log: np.ndarray) -> None:
        """Compute CQR correction scalar from the calibration split and store it.

        Conformity score: s_i = max(q10(x_i) − y_i,  y_i − q90(x_i)) in log-space.
        q_hat is the ⌈(n+1)·0.8⌉/n empirical quantile of scores — the additive
        correction applied symmetrically before expm1 in predict_intervals().
        """
        import numpy as np

        n = len(X_cal)
        if n < 20:
            _LOGGER.info("CQR calibration skipped (cal_n=%d < 20)", n)
            self._interval_correction = 0.0
            return

        q10_pred = self._model_q10.predict(X_cal)
        q90_pred = self._model_q90.predict(X_cal)
        scores = np.maximum(q10_pred - y_cal_log, y_cal_log - q90_pred)
        level = min(np.ceil((n + 1) * 0.8) / n, 1.0)
        q_hat = float(np.quantile(scores, level))
        self._interval_correction = q_hat
        _LOGGER.info("CQR correction: q_hat=%.4f (cal_n=%d)", q_hat, n)

    def _save_interval_correction(self) -> None:
        try:
            with open(self._interval_correction_path, "w") as fh:
                json.dump({"correction": self._interval_correction}, fh)
        except OSError as exc:
            _LOGGER.warning("Could not save interval correction: %s", exc)

    def _load_interval_correction(self) -> None:
        if not self._interval_correction_path.exists():
            return
        self._interval_correction = 0.0
        try:
            with open(self._interval_correction_path) as fh:
                data = json.load(fh)
            self._interval_correction = float(data.get("correction", 0.0))
        except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
            _LOGGER.warning("Could not load interval correction: %s", exc)

    def _save_signatures(self) -> None:
        try:
            with open(self._signatures_path, "w") as fh:
                json.dump(self._appliance_signatures, fh)
        except OSError as exc:
            _LOGGER.warning("Could not save appliance signatures: %s", exc)

    def _load_signatures(self) -> None:
        if not self._signatures_path.exists():
            return
        try:
            with open(self._signatures_path) as fh:
                raw = json.load(fh)
            # JSON serialises dict keys as strings; convert hour_of_day_distribution
            # keys back to int so the in-memory representation is consistent.
            for sig in raw.values():
                if "hour_of_day_distribution" in sig:
                    sig["hour_of_day_distribution"] = {int(k): v for k, v in sig["hour_of_day_distribution"].items()}
            self._appliance_signatures = raw
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            _LOGGER.warning("Could not load appliance signatures: %s", exc)

    def _calibrate_tau(
        self,
        climate_dfs: dict[str, pd.DataFrame],
        heating_active_df: pd.DataFrame,  # cols: timestamp, heating_active (0/1)
        weather_df: pd.DataFrame,  # cols: timestamp, temp_c
    ) -> float | None:
        """Estimate building thermal time constant τ (hours) from passive-cooling windows.

        Fits log-linear OLS on ``ln(T_indoor − T_outdoor) = ln(ΔT₀) − t/τ`` for each
        contiguous sub-sequence where indoor is warmer than outdoor and cooling.

        Windows are scored rather than hard-filtered.  Each candidate receives a
        composite quality score (0–1):

        * ``r²``             — goodness of OLS log-linear fit (primary signal quality indicator;
                               candidates with r² ≤ 0 are dropped as physics failures)
        * ``outdoor_temp``   — ``exp(−max(T_out_mean − 10, 0) / 8)``; penalises warm-weather
                               windows where open-window ventilation shortens apparent τ
        * ``n score``        — length bonus, capped at 6 points
        * ``solar``          — ``exp(−max_radiation / 400)``; continuous penalty, no hard cut-off
        * ``hour``           — 1.0 nighttime (22–06), 0.7 shoulder (06–09, 16–22), 0.1 daytime

        The top 50 % of candidates by quality (minimum 1) are used to compute the median τ.

        A continuous confidence weight — requiring both a sufficient nighttime fraction *and*
        a cold-enough outdoor median among the selected top-N candidates, plus enough total
        candidates — scales how much a fresh estimate can move the stored τ in a single retrain
        (0–20 %). A two-tier rolling cap independently bounds total movement to ±35 % over any
        30 days and ±50 % over any 180 days, regardless of retrain frequency. See
        `docs/superpowers/specs/2026-07-09-tau-calibration-drift-fix-design.md` §2.1–2.2 for the
        full design rationale.
        """
        import numpy as np
        import pandas as pd

        indoor_parts = []
        for eid, c_df in climate_dfs.items():
            if c_df.empty:
                continue
            c = c_df[["timestamp", "current_temp"]].copy()
            c["timestamp"] = pd.to_datetime(c["timestamp"]).dt.floor("1h")
            indoor_parts.append(c.set_index("timestamp")["current_temp"])

        if not indoor_parts:
            return None

        T_indoor_series = pd.concat(indoor_parts, axis=1).mean(axis=1)

        outdoor = weather_df[["timestamp", "temp_c"]].copy()
        outdoor["timestamp"] = pd.to_datetime(outdoor["timestamp"]).dt.floor("1h")
        T_outdoor_series = outdoor.set_index("timestamp")["temp_c"]

        has_radiation = "direct_radiation_wm2" in weather_df.columns
        if has_radiation:
            rad = weather_df[["timestamp", "direct_radiation_wm2"]].copy()
            rad["timestamp"] = pd.to_datetime(rad["timestamp"]).dt.floor("1h")
            rad_series = rad.set_index("timestamp")["direct_radiation_wm2"]
        else:
            rad_series = None

        active = heating_active_df[["timestamp", "heating_active"]].copy()
        active["timestamp"] = pd.to_datetime(active["timestamp"]).dt.floor("1h")
        active_series = (active.set_index("timestamp")["heating_active"] > 0.5).astype(int)

        combined_data: dict[str, pd.Series] = {
            "T_indoor": T_indoor_series,
            "T_outdoor": T_outdoor_series,
            "heating_active": active_series,
        }
        if rad_series is not None:
            combined_data["direct_radiation_wm2"] = rad_series

        combined = pd.DataFrame(combined_data).dropna(subset=["T_indoor", "T_outdoor", "heating_active"]).sort_index()

        if combined.empty:
            return None

        combined["off"] = (combined["heating_active"] == 0).astype(int)
        combined["block"] = (combined["off"].diff().ne(0)).cumsum()

        combined_reset = combined.reset_index().rename(columns={"index": "timestamp"})
        passive_df = pd.DataFrame(
            {
                "timestamp": combined_reset["timestamp"],
                "T_outdoor": combined_reset["T_outdoor"],
                "T_indoor": combined_reset["T_indoor"],
                "hp_running": combined_reset["heating_active"].astype(bool),
                "dhw_tank_temp": np.nan,  # τ calibration has no DHW filter input today — unchanged behaviour
            }
        )
        passive_idx = _find_passive_windows(passive_df, min_delta_t=0.0, min_hp_off_hours=1)
        # block identity is computed above, on the full unfiltered timeline, so block boundaries
        # reflect true temporal contiguity of on/off transitions. Filtering afterward only removes
        # individual rows within already-correctly-bounded blocks (harmless with these loose
        # min_delta_t=0.0/min_hp_off_hours=1 parameters, which add no filtering beyond off==1 given
        # this input) — it does not merge distinct off-periods into one pseudo-block.
        combined = combined.iloc[passive_idx].copy() if len(passive_idx) else combined.iloc[0:0]

        candidates: list[tuple[float, float, int]] = []  # (tau, quality, hour_start)

        n_blocks = combined[combined["off"] == 1]["block"].nunique()
        _LOGGER.debug("τ calibration: %d heating-off blocks found in combined data", n_blocks)

        for _, group in combined[combined["off"] == 1].groupby("block"):
            if len(group) < 2:
                continue

            delta = group["T_indoor"].values - group["T_outdoor"].values

            valid = delta > 0
            declining = np.concatenate([[False], np.diff(delta) < 0])
            mask = valid & declining

            if not np.any(mask):
                continue

            edges = np.diff(mask.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0]
            ends = np.where(edges == -1)[0]

            for s, e in zip(starts, ends):
                # Cap each sub-sequence at 12h to avoid ambient drift in the OLS fit.
                # The cap is applied here (not at the block level) so that nighttime
                # cooling windows in extended off-periods (>12h) are still reachable.
                e_cap = min(e, s + 12)
                if e_cap - s < 2:
                    continue

                d = delta[s:e_cap]
                t = np.arange(len(d), dtype=float)
                log_d = np.log(d)
                slope, intercept = np.polyfit(t, log_d, 1)

                if slope >= 0:
                    continue
                tau = -1.0 / slope
                if not (0.5 <= tau <= 200.0):
                    continue

                # R² of the log-linear fit — windows where the fit explains no
                # variance (flat decay profile, noisy sensor) are discarded.
                y_hat = slope * t + intercept
                ss_res = float(np.sum((log_d - y_hat) ** 2))
                ss_tot = float(np.sum((log_d - log_d.mean()) ** 2))
                r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
                if r2 <= 0.0:
                    continue

                # Outdoor temperature score: penalises warm-weather windows where open
                # ventilation dominates (replaces delta_t_score, which had no empirical
                # correlation with τ accuracy in real-world data).
                # exp(-(T_out_mean - 10) / 8): score=1.0 at ≤10°C, ≈0.29 at 20°C, ≈0.08 at 30°C.
                sub_t_outdoor = float(group["T_outdoor"].iloc[s:e_cap].mean())
                outdoor_temp_score = float(np.exp(-max(sub_t_outdoor - 10.0, 0.0) / 8.0))

                # Length bonus — capped at 6 points
                n_score = min(len(d) / 6.0, 1.0)

                # Continuous solar penalty — no hard threshold
                if "direct_radiation_wm2" in group.columns:
                    sub_max_rad = float(group["direct_radiation_wm2"].iloc[s:e_cap].max())
                else:
                    sub_max_rad = 0.0
                solar_score = float(np.exp(-sub_max_rad / 400.0))

                # Hour-of-day penalty applied at sub-sequence level.
                # Daytime reduced to 0.1 (from 0.3) — empirically, daytime windows
                # produce τ estimates 5× shorter than nighttime ones due to ventilation.
                sub_hours = np.asarray(group.index[s:e_cap].hour)
                hour_scores = np.where(
                    (sub_hours >= 9) & (sub_hours < 16),
                    0.1,
                    np.where(
                        ((sub_hours >= 6) & (sub_hours < 9)) | ((sub_hours >= 16) & (sub_hours < 22)),
                        0.7,
                        1.0,
                    ),
                )
                hour_score = float(hour_scores.min())
                hour_start = int(group.index[s].hour)

                quality = r2 * outdoor_temp_score * n_score * solar_score * hour_score
                candidates.append((tau, quality, hour_start, sub_t_outdoor))

        if not candidates:
            _LOGGER.info("τ calibration: no valid passive-cooling windows found — skipping.")
            return None

        old_tau = self._tau_hours

        # With few candidates use top-50%; with many (≥ threshold) tighten to top-25%
        # to reduce bias from lower-quality windows in data-rich retrains.
        candidates.sort(key=lambda c: c[1], reverse=True)
        divisor = 4 if len(candidates) >= self._TAU_SELECTIVITY_THRESHOLD else 2
        n_select = max(1, len(candidates) // divisor)
        selected = candidates[:n_select]
        tau_estimates = [c[0] for c in selected]

        _LOGGER.debug(
            "τ calibration: %d candidates, using top %d (%d%%) (quality %.2f–%.2f, τ range %.1f–%.1f h)",
            len(candidates),
            n_select,
            round(100 * n_select / len(candidates)),
            selected[-1][1],
            selected[0][1],
            min(tau_estimates),
            max(tau_estimates),
        )

        # Confidence: requires BOTH a sufficient nighttime fraction AND a cold-enough outdoor
        # median among the SELECTED (quality-filtered) candidates -- computed over the same
        # population tau_median comes from, not the raw candidate pool. A single good signal
        # must not be able to rescue full trust for a window where the other signal is bad
        # (e.g. a warm, all-night window is a textbook summer ventilation case, not safe data).
        night_frac = sum(1 for c in selected if c[2] >= 22 or c[2] < 6) / len(selected)
        outdoor_median = float(np.median([c[3] for c in selected]))

        night_conf = min(1.0, night_frac / self._SPRING_BIAS_NIGHT_FRAC_FULL)
        temp_conf = min(
            1.0,
            max(
                0.0,
                (self._SPRING_BIAS_OUTDOOR_TEMP - outdoor_median)
                / (self._SPRING_BIAS_OUTDOOR_TEMP - self._SPRING_BIAS_OUTDOOR_TEMP_FLOOR),
            ),
        )
        # len(candidates) (pre-selection pool size) -- sparse retrains should be trusted less.
        sample_conf = min(1.0, len(candidates) / self._TAU_SAMPLE_CONF_REF)

        # Geometric mean, not a plain product: a plain 3-way product over-penalizes moderately
        # -good conditions (three factors of 0.7 would multiply to 0.34); the geometric mean
        # preserves the same AND-semantics at the boundaries (any single factor at exactly 0
        # still forces confidence to exactly 0) while being much gentler in the interior.
        confidence = (night_conf * temp_conf * sample_conf) ** (1.0 / 3.0)

        tau_median = float(np.median(tau_estimates))

        if old_tau is not None and old_tau > 0:
            new_weight = confidence * self._TAU_EMA_MAX_NEW_WEIGHT
            tau_result = (1.0 - new_weight) * old_tau + new_weight * tau_median
            if confidence < 0.05:
                _LOGGER.info(
                    "τ calibration: ~0%% confidence (night_frac=%.0f%%, T_out median=%.1f°C, "
                    "%d candidates) — preserving stored τ=%.1f h.",
                    night_frac * 100,
                    outdoor_median,
                    len(candidates),
                    old_tau,
                )
            else:
                _LOGGER.info(
                    "τ EMA blend: %.1f h → %.1f h (raw %.1f h, new_weight=%.0f%%, confidence=%.0f%% "
                    "[night=%.0f%%, temp=%.0f%%, sample=%.0f%%])",
                    old_tau,
                    tau_result,
                    tau_median,
                    new_weight * 100,
                    confidence * 100,
                    night_conf * 100,
                    temp_conf * 100,
                    sample_conf * 100,
                )

            latest_ts = combined.index.max()

            def _apply_drift_cap(anchor_h_attr, anchor_ts_attr, window_days, max_frac, result):
                anchor_h = getattr(self, anchor_h_attr, None)
                anchor_ts = getattr(self, anchor_ts_attr, None)
                if anchor_h is None or anchor_ts is None or (latest_ts - anchor_ts).days >= window_days:
                    anchor_h = old_tau
                    setattr(self, anchor_h_attr, anchor_h)
                    setattr(self, anchor_ts_attr, latest_ts)
                max_drift = anchor_h * max_frac
                lo, hi = anchor_h - max_drift, anchor_h + max_drift
                if not (lo <= result <= hi):
                    _LOGGER.warning(
                        "τ drift cap (%d-day): %.1f h clamped to [%.1f, %.1f] h (anchor=%.1f h @ %s)",
                        window_days,
                        result,
                        lo,
                        hi,
                        anchor_h,
                        anchor_ts,
                    )
                return min(max(result, lo), hi)

            tau_result = _apply_drift_cap(
                "_tau_anchor_hours", "_tau_anchor_ts", self._TAU_DRIFT_WINDOW_DAYS, self._TAU_MAX_DRIFT_FRAC, tau_result
            )
            tau_result = _apply_drift_cap(
                "_tau_long_anchor_hours",
                "_tau_long_anchor_ts",
                self._TAU_LONG_DRIFT_WINDOW_DAYS,
                self._TAU_LONG_MAX_DRIFT_FRAC,
                tau_result,
            )
        else:
            tau_result = tau_median

        _LOGGER.info(
            "Building τ estimated at %.1f h from %d passive-cooling windows (%d candidates scored).",
            tau_result,
            len(tau_estimates),
            len(candidates),
        )
        return tau_result

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
                    self.feature_cols = meta.get("feature_cols", _FEATURES_BASE)
                    self.last_trained = meta.get("last_trained", datetime.min)
                    self.last_mae = meta.get("last_mae")
                    self.last_cv_mae = meta.get("last_cv_mae")
                    self.engine = meta.get("engine", "unknown")
                    self._feature_medians = meta.get("feature_medians", {})
                    self._feature_medians_by_how = meta.get("feature_medians_by_how", {})
                    self._log_transform = meta.get("log_transform", False)
                    self._use_physics_residual = meta.get("use_physics_residual", False)
                    self._canton = meta.get("canton", None)
                    self._country = meta.get("country", "CH")
                    self._likely_ev_hours = meta.get("likely_ev_hours", set())
                    self._sub_sensor_prefixes = meta.get("sub_sensor_prefixes", [])
                    self._tau_hours = meta.get("tau_hours", None)
                    self._tau_anchor_hours = meta.get("tau_anchor_hours", None)
                    self._tau_anchor_ts = meta.get("tau_anchor_ts", None)
                    self._tau_long_anchor_hours = meta.get("tau_long_anchor_hours", None)
                    self._tau_long_anchor_ts = meta.get("tau_long_anchor_ts", None)
                    self._enable_regimes = meta.get("enable_regimes", False)
                    self._regime_count = meta.get("regime_count", 5)
                    self._weather_tail = meta.get("weather_tail", None)
                except (pickle.UnpicklingError, EOFError, OSError) as exc:
                    _LOGGER.warning("Could not load model metadata: %s", exc)

        # Load regime models if they exist
        c_path = self._model_dir / "clusterer.pkl"
        if c_path.exists():
            try:
                with open(c_path, "rb") as fh:
                    self._clusterer = pickle.load(fh)
            except Exception:
                self._clusterer = None
        r_path = self._model_dir / "regime_model.pkl"
        if r_path.exists():
            try:
                with open(r_path, "rb") as fh:
                    self._regime_model = pickle.load(fh)
            except Exception:
                self._regime_model = None

        self._load_quantile_models()
        self._load_signatures()


def _prepare_daily_regime_features(
    weather_df: pd.DataFrame,
    country: str = "CH",
    canton: str | None = None,
    away_df: pd.DataFrame | None = None,  # cols: timestamp, is_away
    presence_df: pd.DataFrame | None = None,  # cols: timestamp, people_home
) -> pd.DataFrame:
    """Aggregate hourly weather into daily features for regime prediction."""
    import pandas as pd

    # 1. Aggregate daily weather features
    w_daily = weather_df.copy()
    w_daily["date"] = pd.to_datetime(w_daily["timestamp"]).dt.date
    w_daily = w_daily.groupby("date").agg(
        {"temp_c": ["mean", "min", "max"], "sunshine_min": "sum", "precipitation_mm": "sum"}
    )
    w_daily.columns = ["temp_mean", "temp_min", "temp_max", "sun_total", "precip_total"]

    # 2. Add calendar features
    cal_df = pd.DataFrame(index=w_daily.index)
    cal_dates = pd.to_datetime(cal_df.index)
    cal_df["day_of_week"] = cal_dates.dayofweek

    # Reuse holiday logic
    try:
        import holidays as hd

        years = cal_dates.year.unique().tolist()
        ch_hols = hd.country_holidays(country, years=years)
        # Add cantonal holidays if provided
        if canton and country.upper() == "CH":
            try:
                ch_hols = hd.country_holidays(country, prov=canton, years=years)
            except Exception:
                pass
        cal_df["is_holiday"] = pd.Series(cal_df.index).map(lambda d: 1 if d in ch_hols else 0).values
    except Exception:
        cal_df["is_holiday"] = 0

    # 3. Occupancy features
    # is_away: 1 if any hour of the day was marked away
    if away_df is not None and not away_df.empty and "is_away" in away_df.columns:
        a = away_df.copy()
        a["date"] = pd.to_datetime(a["timestamp"]).dt.date
        daily_away = a.groupby("date")["is_away"].max()
        cal_df["is_away"] = daily_away.reindex(cal_df.index).fillna(0).astype(int)
    else:
        cal_df["is_away"] = 0

    # people_home: mean occupied count across the day
    if presence_df is not None and not presence_df.empty and "people_home" in presence_df.columns:
        p = presence_df.copy()
        p["date"] = pd.to_datetime(p["timestamp"]).dt.date
        daily_pres = p.groupby("date")["people_home"].mean()
        cal_df["people_home"] = daily_pres.reindex(cal_df.index).fillna(0)
    else:
        cal_df["people_home"] = 0

    return pd.concat([w_daily, cal_df], axis=1).dropna()


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
    ts_index = pd.to_datetime(df["timestamp"])

    if df.empty:
        # Empty frame — no grid to build; return df with all-NaN feature columns.
        new_cols: dict = {}
        for lag in active_lags:
            new_cols[f"lag_{lag}h"] = pd.Series(dtype=float)
        new_cols["rolling_mean_24h"] = pd.Series(dtype=float)
        new_cols["rolling_mean_7d"] = pd.Series(dtype=float)
        new_cols["rolling_std_24h"] = pd.Series(dtype=float)
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # Reindex to a continuous hourly grid so shift(N) means "N hours ago" even when
    # the training frame has holes (zero-import hours dropped, EV adjacent hours
    # dropped, outages). Without this, shift(24) on row i looks at row i-24 which
    # may be more than 24 hours earlier when rows are missing.
    hourly_grid = pd.date_range(ts_index.min(), ts_index.max(), freq="1h")
    dense_kwh = pd.Series(df["gross_kwh"].values, index=ts_index).reindex(hourly_grid)

    new_cols: dict = {}
    for lag in active_lags:
        new_cols[f"lag_{lag}h"] = dense_kwh.shift(lag).reindex(ts_index).values

    dense_shifted = dense_kwh.shift(1)
    new_cols["rolling_mean_24h"] = dense_shifted.rolling(24, min_periods=12).mean().reindex(ts_index).values
    new_cols["rolling_mean_7d"] = dense_shifted.rolling(7 * 24, min_periods=48).mean().reindex(ts_index).values
    new_cols["rolling_std_24h"] = dense_shifted.rolling(24, min_periods=12).std().fillna(0).reindex(ts_index).values

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def _add_lag_and_rolling_prediction(future_df: pd.DataFrame, recent_actuals: pd.DataFrame | None) -> pd.DataFrame:
    """Fill lag and rolling features for the 48-hour prediction horizon.

    For each future hour h:
      lag_24h[h]  = actual consumption at (h - 24h)  — always a real past value
      lag_168h[h] = actual consumption at (h - 168h) — always a real past value

    If recent_actuals is missing or sparse, falls back to NaN (the model
    will fill those with stored training medians).
    """
    import numpy as np
    import pandas as pd

    if recent_actuals is None or recent_actuals.empty:
        for lag in LAG_HOURS:
            future_df[f"lag_{lag}h"] = np.nan
        future_df["rolling_mean_24h"] = np.nan
        future_df["rolling_mean_7d"] = np.nan
        future_df["rolling_std_24h"] = np.nan
        return future_df

    # Index actuals by timestamp for O(1) lookup
    actuals = recent_actuals.set_index(pd.to_datetime(recent_actuals["timestamp"]))["gross_kwh"].sort_index()

    for lag in LAG_HOURS:
        lag_td = pd.Timedelta(hours=lag)
        lag_times = future_df["timestamp"] - lag_td
        future_df[f"lag_{lag}h"] = actuals.reindex(lag_times).values
        # Short-horizon lags (lag < 48h) are expected to be NaN for most future
        # hours — only the first `lag` hours can look up a real past value.
        # Only warn for long lags where NaN indicates an actuals coverage gap.
        if lag >= 24:
            nan_count = int(future_df[f"lag_{lag}h"].isna().sum())
            if nan_count > len(future_df) * 0.5:
                _LOGGER.warning(
                    "lag_%dh has %d/%d NaN values — recent_actuals doesn't reach "
                    "back %dh; these will be filled with training medians.",
                    lag,
                    nan_count,
                    len(future_df),
                    lag,
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
    rm24 = extended_shifted.rolling(24, min_periods=12).mean()
    rm7d = extended_shifted.rolling(168, min_periods=48).mean()
    rs24 = extended_shifted.rolling(24, min_periods=12).std().fillna(0)

    future_df["rolling_mean_24h"] = rm24.reindex(future_index).values
    future_df["rolling_mean_7d"] = rm7d.reindex(future_index).values
    future_df["rolling_std_24h"] = rs24.reindex(future_index).values

    return future_df


# ── Sub-sensor lag helpers ────────────────────────────────────────────────────


def _add_sub_sensor_lags_training(
    df: pd.DataFrame,
    sub_sensors: dict | None,
) -> pd.DataFrame:
    """Add lag_24h, lag_168h, active_24h, and runs_7d columns for each sub-sensor.

    For each prefix/DataFrame pair in sub_sensors:
    - Reindexes the sub-sensor kWh series onto the training timestamps
    - lag_24h via shift(24); lag_168h via shift(168) when n_rows ≥ 268
    - active_24h: 1 if any non-zero reading in the 24h window ending at the row
    - runs_7d: count of appliance start events (0→>0 transitions) in past 168h

    All rolling ops use shift(1) first to exclude the current row, matching
    the training semantics for the main lag features.
    Returns df unchanged when sub_sensors is empty/None.
    """
    import pandas as pd

    if not sub_sensors:
        return df

    n_rows = len(df)
    ts_idx = pd.to_datetime(df["timestamp"])
    hourly_grid = pd.date_range(ts_idx.min(), ts_idx.max(), freq="1h")
    new_cols: dict = {}

    for prefix, sub_df in sub_sensors.items():
        if sub_df is None or sub_df.empty:
            continue
        sub_series = sub_df.set_index(pd.to_datetime(sub_df["timestamp"]))["kwh"].sort_index()
        # Reindex to dense hourly grid for temporal shift accuracy (same fix as
        # _add_lag_and_rolling_training — positional shift over gapped ts_idx is wrong).
        dense_kwh = pd.to_numeric(sub_series.reindex(hourly_grid), errors="coerce")
        nan_count = int(dense_kwh.reindex(ts_idx).isna().sum())
        if nan_count > len(ts_idx) * 0.5:
            _LOGGER.warning(
                "%s reindex introduces %d/%d NaN values during training — "
                "sub-sensor data may have gaps or misaligned timestamps; "
                "will be filled with training medians.",
                prefix,
                nan_count,
                len(ts_idx),
            )
        new_cols[f"{prefix}_lag_24h"] = dense_kwh.shift(24).reindex(ts_idx).values
        if n_rows - 168 >= 100:  # lag_168h needs 168 lag rows + MIN_TRAINING_ROWS
            new_cols[f"{prefix}_lag_168h"] = dense_kwh.shift(168).reindex(ts_idx).values

        # active_24h: was the appliance active any time in the past 24h?
        # Use shift(1) so the current-row value is excluded (lag semantics).
        filled = dense_kwh.fillna(0)
        shifted_kwh = filled.shift(1)
        active = (shifted_kwh.rolling(24, min_periods=1).max() > 0).astype(int)
        new_cols[f"{prefix}_active_24h"] = active.reindex(ts_idx).values

        # runs_7d: how many times did the appliance start (0→>0 transition) in
        # the past 168h?  shift(1) applied before rolling to exclude current row.
        is_start = ((filled > 0) & (filled.shift(1).fillna(0) == 0)).astype(float)
        runs = is_start.shift(1).rolling(168, min_periods=1).sum()
        new_cols[f"{prefix}_runs_7d"] = runs.reindex(ts_idx).values

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _learn_appliance_signatures(
    sub_sensors: dict | None,
    min_cycles: int = 3,
    program_histories: dict | None = None,
    min_cycles_per_program: int = 2,
) -> dict:
    """Learn the energy shape (kWh per hour) for each sub-sensor appliance.

    Improvements over the original fixed-window approach:

    * **Adaptive window** — cycle end is detected when consumption drops back to
      the idle level rather than using a hard 4-hour cutoff. A cap of
      ``APPLIANCE_MAX_WINDOW_HOURS`` prevents runaway windows.
    * **Demand-surge detection** — appliances that never drop to zero (e.g. heat
      pumps in heating season) use the 25th-percentile consumption as the idle
      baseline; starts are detected as upward crossings of that threshold.
    * **Outlier cycle rejection** — cycles whose total kWh exceeds
      ``median + 2 × σ`` are discarded before averaging (requires ≥ 4 cycles).
    * **Duration clustering** — if detected cycles naturally split into short
      and long programs (median duration ≥ 2 h), separate cluster profiles are
      stored under ``"clusters"``.  The top-level ``hourly_profile`` is always
      present for backward compatibility with ``_composite_forecast()``.

    Returns a dict keyed by prefix; prefixes with fewer than ``min_cycles``
    valid cycles are omitted.
    """
    import numpy as np
    import pandas as pd

    if not sub_sensors:
        return {}

    def _avg_profile(
        wins: list[list[float]],
    ) -> tuple[list[float], list[float]]:
        """Pad variable-length windows to the same length and average column-wise."""
        max_len = max(len(w) for w in wins)
        padded = [w + [0.0] * (max_len - len(w)) for w in wins]
        arr = np.array(padded)
        return np.mean(arr, axis=0).tolist(), np.std(arr, axis=0).tolist()

    def _resolve_program(cycle_ts: pd.Timestamp, prog_df: pd.DataFrame) -> str | None:
        """Last-value-carry-forward lookup of program at cycle start."""
        earlier = prog_df[prog_df["timestamp"] <= cycle_ts]
        return str(earlier.iloc[-1]["program"]) if not earlier.empty else None

    def _cov_warning(prefix: str, label: str, profile: list[float], std: list[float]) -> None:
        mean_arr = np.array(profile)
        std_arr = np.array(std)
        nonzero = mean_arr > 0
        if nonzero.any():
            cov_max = float((std_arr[nonzero] / mean_arr[nonzero]).max())
            if cov_max > 0.3:
                _LOGGER.warning(
                    "Appliance '%s' (%s) signature has high variability (CoV=%.2f)"
                    " — scenario predictions may be inaccurate.",
                    prefix,
                    label,
                    cov_max,
                )

    result: dict = {}
    for prefix, sub_df in sub_sensors.items():
        if sub_df is None or sub_df.empty:
            continue

        # ── Align to hourly index ──────────────────────────────────────────────
        kwh_series = (
            sub_df.set_index(pd.to_datetime(sub_df["timestamp"]))["kwh"].sort_index().resample("h").sum().astype(float)
        )
        filled = kwh_series.fillna(0.0)
        n = len(filled)

        # ── Idle threshold (adaptive) ──────────────────────────────────────────
        # Appliances that never truly idle (e.g. HP heating in winter) have
        # <5 % zero-hours; use the bottom-quartile level as the "off" baseline.
        zero_frac = float((filled == 0).mean())
        if zero_frac >= 0.05:
            idle_thresh = 0.0
        else:
            idle_thresh = float(filled.quantile(0.25))

        # ── Cycle start detection ──────────────────────────────────────────────
        prev = filled.shift(1).fillna(idle_thresh)
        is_start = (filled > idle_thresh) & (prev <= idle_thresh)
        start_positions = [i for i, v in enumerate(is_start) if v]

        # ── Extract variable-length windows ────────────────────────────────────
        windows: list[list[float]] = []
        valid_starts: list[int] = []  # parallel to windows; tracks start index in filled
        for pos in start_positions:
            end = pos + 1
            while end < n and end - pos < APPLIANCE_MAX_WINDOW_HOURS and filled.iloc[end] > idle_thresh:
                end += 1
            # Skip if the cycle was still running when history ended (truncated).
            # A cycle that hit the MAX_WINDOW cap is kept — it's complete enough.
            if end >= n and filled.iloc[end - 1] > idle_thresh:
                continue
            windows.append(filled.iloc[pos:end].tolist())
            valid_starts.append(pos)

        if len(windows) < min_cycles:
            continue

        # ── Outlier cycle rejection (requires ≥ 4 cycles) ─────────────────────
        totals = np.array([sum(w) for w in windows])
        n_rejected = 0
        if len(totals) >= 4:
            median_t = float(np.median(totals))
            std_t = float(np.std(totals))
            keep_mask = totals <= median_t + 2.0 * std_t
            n_rejected = int((~keep_mask).sum())
            if n_rejected:
                windows = [w for w, keep in zip(windows, keep_mask) if keep]
                valid_starts = [s for s, keep in zip(valid_starts, keep_mask) if keep]
                _LOGGER.debug(
                    "Appliance '%s': rejected %d outlier cycle(s) (total kWh > median + 2σ)",
                    prefix,
                    n_rejected,
                )

        if len(windows) < min_cycles:
            continue

        # ── Per-program grouping ───────────────────────────────────────────────
        prog_df = (program_histories or {}).get(prefix)
        programs: dict = {}
        if prog_df is not None and not prog_df.empty:
            prog_windows: dict[str, list] = {}
            for win, pos in zip(windows, valid_starts):
                label = _resolve_program(kwh_series.index[pos], prog_df)
                if label is None:
                    continue
                prog_windows.setdefault(label, []).append(win)
            for label, pwins in prog_windows.items():
                if len(pwins) < min_cycles_per_program:
                    continue
                prof, std = _avg_profile(pwins)
                programs[label] = {
                    "hourly_profile": prof,
                    "std_profile": std,
                    "total_kwh": float(sum(prof)),
                    "n_cycles": len(pwins),
                }

        # ── Duration clustering ────────────────────────────────────────────────
        # Trigger on high energy CoV (> 0.5) in addition to long median duration,
        # so bimodal consumers like heat-pump heating (many 1h pulses + occasional
        # multi-hour runs) are also split into short/long clusters.
        durations = [len(w) for w in windows]
        median_dur = statistics.median(durations)
        cycle_totals = np.array([sum(w) for w in windows])
        mean_total = float(np.mean(cycle_totals))
        energy_cov = float(np.std(cycle_totals) / mean_total) if mean_total > 0 else 0.0
        clusters: dict = {}
        if median_dur >= 2 or energy_cov > 0.5:
            short_wins = [w for w, d in zip(windows, durations) if d <= median_dur]
            long_wins = [w for w, d in zip(windows, durations) if d > median_dur]
            for label, wins in [("short", short_wins), ("long", long_wins)]:
                if not wins:
                    continue
                prof, std = _avg_profile(wins)
                clusters[label] = {
                    "n_cycles": len(wins),
                    "median_duration_h": int(statistics.median(len(w) for w in wins)),
                    "hourly_profile": prof,
                    "std_profile": std,
                    "total_kwh": float(sum(prof)),
                }
                _cov_warning(prefix, label, prof, std)

        # ── Combined profile (backward-compatible top-level keys) ──────────────
        hourly_profile, std_profile = _avg_profile(windows)

        # CoV warning only if clustering did not split (or only one cluster exists)
        if len(clusters) <= 1:
            _cov_warning(prefix, "combined", hourly_profile, std_profile)

        # ── Hour-of-day distribution ───────────────────────────────────────────
        from collections import Counter as _Counter

        hour_dist = _Counter(int(kwh_series.index[pos].hour) for pos in valid_starts)

        # ── Signature quality metadata ─────────────────────────────────────────
        n_data_days = int((kwh_series.index[-1] - kwh_series.index[0]).total_seconds() / 86400)
        last_cycle_ts = str(kwh_series.index[valid_starts[-1]]) if valid_starts else ""
        quality_base = "good" if len(windows) >= 15 else "fair" if len(windows) >= 5 else "poor"
        # Demote to "fair" when energy variability is high despite adequate cycle count
        if energy_cov > 0.5 and quality_base == "good":
            quality_base = "fair"
        quality = quality_base

        sig: dict = {
            "total_kwh": float(sum(hourly_profile)),
            "hourly_profile": hourly_profile,
            "std_profile": std_profile,
            "peak_hour": int(np.argmax(hourly_profile)),
            "n_cycles": len(windows),
            "n_rejected": n_rejected,
            "idle_threshold": idle_thresh,
            "quality": quality,
            "energy_cov": float(energy_cov),
            "n_data_days": n_data_days,
            "last_cycle_ts": last_cycle_ts,
            "hour_of_day_distribution": dict(sorted(hour_dist.items())),
        }
        if clusters:
            sig["clusters"] = clusters
        if programs:
            sig["programs"] = programs

        result[prefix] = sig

    return result


def _composite_forecast(
    baseline_df: pd.DataFrame,
    schedule: dict[str, str | dict | None],
    signatures: dict,
) -> pd.DataFrame:
    """Overlay appliance run profiles onto a baseline forecast.

    Args:
        baseline_df: DataFrame with columns [timestamp, predicted_kwh] — 48 rows, naive tz.
        schedule:    {prefix: "HH:MM" | {"start": "HH:MM", "program": "eco"} | "off" | None}.
                     String form: plain start time, e.g. ``"22:30"``.
                     Dict form: ``{"start": "HH:MM", "program": "<label>"}`` — selects a
                     per-program profile from ``sig["programs"]`` if available, otherwise
                     falls back to ``sig["hourly_profile"]``.
                     ``"off"`` and ``None`` skip the appliance.
                     Invalid or out-of-range time strings (e.g. "25:00", "abc") are
                     logged as warnings and skipped; ``delta_kwh`` stays 0 for that
                     prefix.
        signatures:  appliance signatures dict from _learn_appliance_signatures().

    Returns:
        DataFrame with columns [timestamp, predicted_kwh, delta_kwh].
        ``delta_kwh`` is the signed kWh difference vs. the baseline introduced by
        the scheduled appliances; it is 0 when schedule is empty or a prefix has no
        matching signature.
    """
    import numpy as np
    import pandas as pd

    result = baseline_df.copy()
    result["delta_kwh"] = 0.0

    if not schedule or not signatures:
        return result

    forecast_start = pd.Timestamp(result["timestamp"].iloc[0])

    for prefix, sched_value in schedule.items():
        if sched_value is None or sched_value == "off":
            continue
        # Dict form: {"start": "HH:MM", "program": "eco"}
        if isinstance(sched_value, dict):
            time_str = sched_value.get("start")
            program = sched_value.get("program")
        else:
            time_str = sched_value
            program = None
        if time_str is None or time_str == "off":
            continue
        if prefix not in signatures:
            _LOGGER.warning("_composite_forecast: no signature for prefix '%s' — skipping", prefix)
            continue

        try:
            hour, minute = (int(x) for x in str(time_str).split(":"))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError(f"out of range: {hour}:{minute}")
        except (ValueError, AttributeError):
            _LOGGER.warning("_composite_forecast: invalid time_str '%s' for prefix '%s'", time_str, prefix)
            continue

        # Build candidate timestamps for today and tomorrow relative to forecast_start
        today_ts = forecast_start.replace(hour=hour, minute=minute, second=0, microsecond=0)
        tomorrow_ts = today_ts + pd.Timedelta(days=1)

        # Pick the earliest one >= forecast_start
        if today_ts >= forecast_start:
            target_ts = today_ts
        else:
            target_ts = tomorrow_ts

        offset_h = int((target_ts - forecast_start).total_seconds() / 3600)  # floor: "14:30" → offset 0, not 1

        if offset_h >= 48 or offset_h < 0:
            continue

        # Profile selection: prefer per-program profile; fall back to combined
        sig = signatures[prefix]
        profile: list | None = None
        if program is not None:
            prog_profiles = sig.get("programs", {})
            if program in prog_profiles:
                profile = prog_profiles[program]["hourly_profile"]
            else:
                _LOGGER.debug(
                    "_composite_forecast: no profile for program '%s' on '%s' — using combined",
                    program,
                    prefix,
                )
        if profile is None:
            profile = sig["hourly_profile"]
        # Clip profile to available window
        profile = profile[: 48 - offset_h]
        if not profile:
            continue

        n = len(profile)
        result.loc[result.index[offset_h : offset_h + n], "delta_kwh"] += profile

    result["predicted_kwh"] = np.maximum(0.0, result["predicted_kwh"].values + result["delta_kwh"].values)
    return result


def _add_sub_sensor_lags_prediction(
    future_df: pd.DataFrame,
    sub_sensors_recent: dict,
) -> pd.DataFrame:
    """Fill sub-sensor lag + derived features for the 48-hour prediction horizon.

    For each sub-sensor:
    - lag_24h / lag_168h: reindex recent actuals at (timestamp - Lh)
    - active_24h: 1 if any non-zero reading in the 24h window before each future
      hour (populated from recent actuals; becomes 0 for future hours > 24h ahead)
    - runs_7d: appliance start count in the past 168h (constant across all 48 future
      hours — we can only know the count from recent actuals)

    Falls back to NaN/0 when actuals don't reach far enough back.
    """
    import pandas as pd

    for prefix, sub_df in sub_sensors_recent.items():
        if sub_df is None or sub_df.empty:
            for col in (f"{prefix}_lag_24h", f"{prefix}_lag_168h"):
                future_df[col] = float("nan")
            future_df[f"{prefix}_active_24h"] = 0
            future_df[f"{prefix}_runs_7d"] = 0
            continue

        sub_series = sub_df.set_index(pd.to_datetime(sub_df["timestamp"]))["kwh"].sort_index().astype(float)
        future_ts = pd.DatetimeIndex(pd.to_datetime(future_df["timestamp"]))

        for lag, col in [(24, f"{prefix}_lag_24h"), (168, f"{prefix}_lag_168h")]:
            lag_td = pd.Timedelta(hours=lag)
            lag_times = future_ts - lag_td
            # Normalise index resolution so reindex finds matches regardless of
            # whether the CSV cache was written with ns or us precision (pandas 3.x).
            lag_times = lag_times.as_unit(sub_series.index.unit if hasattr(sub_series.index, "unit") else "ns")
            future_df[col] = sub_series.reindex(lag_times).to_numpy(dtype=float, na_value=float("nan"))
            nan_count = int(future_df[col].isna().sum())
            if lag >= 24 and nan_count > len(future_df) * 0.5:
                _LOGGER.debug(
                    "%s has %d/%d NaN values — sub-sensor actuals don't reach back %dh; "
                    "will be filled with training medians.",
                    col,
                    nan_count,
                    len(future_df),
                    lag,
                )

        # active_24h: check 24h window before each future hour from recent actuals.
        # For future hours beyond the 24h actuals horizon this will be 0.
        active_vals = []
        for fts in future_ts:
            window_start = fts - pd.Timedelta(hours=24)
            window = sub_series[(sub_series.index >= window_start) & (sub_series.index < fts)]
            active_vals.append(int((window > 0).any()))
        future_df[f"{prefix}_active_24h"] = active_vals

        # runs_7d: appliance start count from the most recent 168h of actuals.
        # Constant across all future hours — future starts are unknown.
        if len(sub_series) > 0:
            filled = sub_series.fillna(0)
            is_start = ((filled > 0) & (filled.shift(1).fillna(0) == 0)).astype(float)
            # Use actuals in the window [last_ts - 168h, last_ts]
            last_ts = sub_series.index.max()
            window_168 = is_start[is_start.index >= last_ts - pd.Timedelta(hours=168)]
            runs_count = int(window_168.sum())
        else:
            runs_count = 0
        future_df[f"{prefix}_runs_7d"] = runs_count

    return future_df


def _find_passive_windows(
    df: pd.DataFrame,
    *,
    min_delta_t: float = 8.0,
    min_hp_off_hours: int = 2,
) -> pd.Index:
    """Return the index of rows suitable for passive-cooling calibration (τ, UA_eff).

    A row qualifies when: the heat pump has been off for at least
    ``min_hp_off_hours`` consecutive hours ending at that row, ΔT = T_indoor −
    T_outdoor ≥ ``min_delta_t``, and ``dhw_tank_temp`` is not rising into that
    row (rising tank temp indicates an active DHW cycle, which would inflate
    UA_eff / shorten apparent τ if included).
    """
    d = df.sort_values("timestamp").reset_index(drop=True)

    off = (~d["hp_running"].astype(bool)).astype(int)
    off_run_length = off.groupby((off != off.shift()).cumsum()).cumcount() + 1
    off_run_length = off_run_length.where(off == 1, 0)
    enough_off = off_run_length >= min_hp_off_hours

    delta_t = d["T_indoor"] - d["T_outdoor"]
    enough_delta = delta_t >= min_delta_t

    dhw_rising = d["dhw_tank_temp"].diff() > 0
    not_dhw_active = ~dhw_rising.fillna(False)

    mask = enough_off & enough_delta & not_dhw_active
    return d.index[mask]


# ── Indoor temperature projection (RC-ODE forward simulation) ────────────────


def _project_indoor_temps(
    climate_recent: dict[str, pd.DataFrame],
    future_timestamps: pd.DatetimeIndex,
    outdoor_temp_series: pd.Series,
    tau_hours: float | None = None,
    heating_active_series: pd.Series | None = None,
    setpoint_on: float | None = None,
    setpoint_off: float | None = None,
) -> dict[str, pd.Series]:
    """Project indoor temperature for each climate entity over *future_timestamps*.

    Uses a first-order RC ODE (Euler forward):
        T_in[t+1] = T_in[t] + (T_out[t] - T_in[t]) / tau

    Starting T_in: most-recent ``current_temp`` if the observation is <2 h old;
    otherwise falls back to the most-recent ``setpoint``.

    Returns a dict ``{entity_id: pd.Series}`` with ``current_temp`` projections
    indexed by *future_timestamps*. These can be passed as *climate_dfs* to
    ``_engineer_features()`` so that prediction-time thermal pressure reflects
    projected — rather than stale — indoor temperatures.
    """
    import numpy as np
    import pandas as pd

    tau = tau_hours if (tau_hours and tau_hours > 0) else DEFAULT_TAU
    now = future_timestamps[0]  # first step ≈ current hour (naive)
    stale_threshold = pd.Timedelta(hours=2)

    result: dict[str, pd.Series] = {}

    for eid, cdf in climate_recent.items():
        if cdf.empty:
            continue

        cdf = cdf.copy()
        cdf["timestamp"] = pd.to_datetime(cdf["timestamp"])
        latest = cdf.sort_values("timestamp").iloc[-1]

        age = now - latest["timestamp"]
        if age <= stale_threshold:
            t_in_start = float(latest["current_temp"])
        else:
            t_in_start = float(latest["setpoint"])

        # Align outdoor temps to future_timestamps
        t_out_arr = outdoor_temp_series.reindex(future_timestamps, method="nearest").values.astype(float)

        # Vectorised Euler forward over 48 steps
        t_in_arr = np.empty(len(future_timestamps))
        t_in_arr[0] = t_in_start
        for i in range(1, len(future_timestamps)):
            t_in_arr[i] = t_in_arr[i - 1] + (t_out_arr[i - 1] - t_in_arr[i - 1]) / tau

        # Build a projected climate DataFrame matching what _engineer_features expects
        setpoint_val = float(latest["setpoint"])
        if heating_active_series is not None and setpoint_on is not None and setpoint_off is not None:
            ha_arr = heating_active_series.reindex(future_timestamps, method="nearest").fillna(0).values
            setpoint_arr = np.where(ha_arr.astype(bool), setpoint_on, setpoint_off)
        else:
            setpoint_arr = np.full(len(future_timestamps), setpoint_val)

        projected_df = pd.DataFrame(
            {
                "timestamp": future_timestamps,
                "current_temp": t_in_arr,
                "setpoint": setpoint_arr,
            }
        )
        result[eid] = projected_df

    return result


# ── Core feature engineering ──────────────────────────────────────────────────


def _engineer_features(
    df: pd.DataFrame,  # naive timestamps
    weather_df: pd.DataFrame,  # naive timestamps
    outdoor_df: pd.DataFrame | None,  # naive timestamps
    canton: str | None = None,
    country: str = "CH",
    likely_ev_hours: set | None = None,
    away_df: pd.DataFrame | None = None,  # cols: timestamp, is_away (0/1)
    presence_df: pd.DataFrame | None = None,  # cols: timestamp, people_home (int)
    climate_dfs: dict[str, pd.DataFrame] | None = None,
    dhw_df: pd.DataFrame | None = None,
    tau_hours: float | None = None,
    room_areas: dict[str, float] | None = None,  # entity_id → m² (defaults to DEFAULT_ROOM_AREA_M2)
    regime_kwh_series: pd.Series | None = None,  # hourly regime profile
    heating_active_df: pd.DataFrame | None = None,  # cols: timestamp, heating_active (0/1)
    physics_kwh_series: pd.Series | None = None,  # hourly physics baseline, absent when physics disabled
    heating_buffer_temp_series: pd.Series | None = None,  # direct sensor feature, absent when sensor not configured
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # stays naive

    # ── Calendar features ────────────────────────────────────────────────────
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["season"] = df["month"].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    # hour_of_week gives the model a unique slot for every hour in the weekly
    # cycle (0 = Monday 00:00, 167 = Sunday 23:00). Replaces is_weekend + hour_block.
    df["hour_of_week"] = df["day_of_week"] * 24 + df["hour"]

    # ── Cyclical encodings ───────────────────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["how_sin"] = np.sin(2 * np.pi * df["hour_of_week"] / 168)
    df["how_cos"] = np.cos(2 * np.pi * df["hour_of_week"] / 168)
    doy = df["timestamp"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365)

    # ── Horizon feature (0 for training rows — always actuals) ───────────────
    # At predict time _prepare_prediction_X overwrites this with 0..47 so the
    # model can learn horizon-specific bias without distributional leakage.
    df["hours_ahead"] = 0

    # ── EV charging pattern ───────────────────────────────────────────────────
    if likely_ev_hours:
        df["likely_ev_hour"] = df["hour_of_week"].isin(likely_ev_hours).astype(int)
    else:
        df["likely_ev_hour"] = 0

    # ── Public holidays ──────────────────────────────────────────────────────
    df = _add_holiday_feature(df, canton=canton, country=country)

    # ── Weather merge ────────────────────────────────────────────────────────
    w = weather_df.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"]).dt.floor("1h")

    # 3-day rolling temperature (captures thermal mass / multi-day cold spells)
    w = w.sort_values("timestamp").reset_index(drop=True)
    w["temp_rolling_3d"] = w["temp_c"].rolling(72, min_periods=1).mean()

    # ── Thermal modelling features (#49–#52) ──────────────────────────────────
    # #49 EWMA — RC-circuit thermal mass model (halflife in hours)
    # Insert NaN at gap boundaries > 2h so ewm() resets rather than bleeding
    # stale temperature across multi-hour weather API gaps.
    dt_diff = w["timestamp"].diff().dt.total_seconds() / 3600
    gap_starts = dt_diff > 2.0
    if gap_starts.any():
        n_gaps = int(gap_starts.sum())
        # 1 gap is the expected DST spring-forward hole — debug only.
        # Multiple gaps indicate weather API problems — keep as WARNING.
        (_LOGGER.warning if n_gaps > 1 else _LOGGER.debug)(
            "EWMA: %d weather gap(s) > 2 h detected — resetting EWMA at boundaries",
            n_gaps,
        )
        w.loc[gap_starts, "temp_c"] = np.nan
    w["temp_ewma_24h"] = w["temp_c"].ewm(halflife=24, min_periods=1).mean()
    w["temp_ewma_72h"] = w["temp_c"].ewm(halflife=72, min_periods=1).mean()

    # #50 Rolling degree-hour sums — accumulated thermal debt
    _hd = np.maximum(0, 18.0 - w["temp_c"])
    w["heating_deg_sum_24h"] = _hd.rolling(24, min_periods=1).sum()
    w["heating_deg_sum_168h"] = _hd.rolling(168, min_periods=1).sum()

    # #51 Temperature rate of change
    w["temp_delta_1h"] = w["temp_c"].diff(1)
    w["temp_delta_24h"] = w["temp_c"].diff(24)

    # #52 Temperature lags
    w["temp_lag_24h"] = w["temp_c"].shift(24)
    w["temp_lag_168h"] = w["temp_c"].shift(168)
    w["temp_lag_336h"] = w["temp_c"].shift(336)

    df["_ts_floor"] = df["timestamp"].dt.floor("1h")
    df = df.merge(w, left_on="_ts_floor", right_on="timestamp", how="left", suffixes=("", "_w"))
    df.drop(columns=["timestamp_w", "_ts_floor"], errors="ignore", inplace=True)

    for col in [
        "temp_c",
        "precipitation_mm",
        "sunshine_min",
        "wind_kmh",
        "humidity",
        "cloud_cover_pct",
        "direct_radiation_wm2",
        "temp_rolling_3d",
        # Thermal modelling features
        "temp_ewma_24h",
        "temp_ewma_72h",
        "heating_deg_sum_24h",
        "heating_deg_sum_168h",
        "temp_delta_1h",
        "temp_delta_24h",
        "temp_lag_24h",
        "temp_lag_168h",
        "temp_lag_336h",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Safety net: ensure new weather columns always exist even when weather_df
    # doesn't include them (e.g. unexpected API response gaps). The model's
    # _feature_medians fill in predict() handles the resulting NaN values.
    for col in ["cloud_cover_pct", "direct_radiation_wm2"]:
        if col not in df.columns:
            df[col] = np.nan
    # humidity: default to 70 % RH when missing — avoids all-NaN dropna in training.
    if "humidity" not in df.columns:
        df["humidity"] = 70.0

    df["heating_degree"] = np.maximum(0, 18.0 - df["temp_c"])
    df["hp_heating_degree"] = np.maximum(0, 15.0 - df["temp_c"])  # empirically: HP cutoff ~15°C
    df["cooling_degree"] = np.maximum(0, df["temp_c"] - 22.0)
    df["temp_in_neutral_zone"] = ((df["temp_c"] >= 15.0) & (df["temp_c"] <= 22.0)).astype(float)

    # ── Outdoor sensor merge ─────────────────────────────────────────────────
    if outdoor_df is not None and not outdoor_df.empty:
        o = outdoor_df.copy()
        o["timestamp"] = pd.to_datetime(o["timestamp"]).dt.floor("1h")
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(o, left_on="_ts_floor", right_on="timestamp", how="left", suffixes=("", "_ot"))
        df.drop(columns=["timestamp_ot", "_ts_floor"], errors="ignore", inplace=True)
        df["outdoor_temp_live"] = df["outdoor_temp_live"].fillna(df["temp_c"])
        df["temp_bias"] = df["outdoor_temp_live"] - df["temp_c"]
    else:
        df["outdoor_temp_live"] = df["temp_c"]
        df["temp_bias"] = 0.0

    # ── Away / vacation flag ─────────────────────────────────────────────────
    if away_df is not None and not away_df.empty:
        a = away_df[["timestamp", "is_away"]].copy()
        a["timestamp"] = pd.to_datetime(a["timestamp"]).dt.floor("1h")
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(a, left_on="_ts_floor", right_on="timestamp", how="left", suffixes=("", "_aw"))
        df.drop(columns=["timestamp_aw", "_ts_floor"], errors="ignore", inplace=True)
        df["is_away"] = df["is_away"].fillna(0).astype(int)
    else:
        df["is_away"] = 0

    # ── Occupancy / people_home feature ───────────────────────────────────────
    if presence_df is not None and not presence_df.empty:
        p = presence_df[["timestamp", "people_home"]].copy()
        p["timestamp"] = pd.to_datetime(p["timestamp"]).dt.floor("1h")
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(p, left_on="_ts_floor", right_on="timestamp", how="left", suffixes=("", "_ph"))
        df.drop(columns=["timestamp_ph", "_ts_floor"], errors="ignore", inplace=True)
        df["people_home"] = df["people_home"].fillna(0).astype(int)
    else:
        df["people_home"] = 0

    # ── Heating system active (seasonal on/off) ───────────────────────────────
    # Binary feature from heating_system_active_entity. When 0, the heat pump is
    # off for the season, letting trees suppress HP-related lag features.
    # Default 1 (heating on) when entity is not configured — conservative choice.
    if heating_active_df is not None and not heating_active_df.empty:
        h = heating_active_df[["timestamp", "heating_active"]].copy()
        h["timestamp"] = pd.to_datetime(h["timestamp"]).dt.floor("1h")
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(h, left_on="_ts_floor", right_on="timestamp", how="left", suffixes=("", "_ha"))
        df.drop(columns=["timestamp_ha", "_ts_floor"], errors="ignore", inplace=True)
        df["heating_active"] = df["heating_active"].fillna(1).astype(int)
    else:
        df["heating_active"] = 1

    # ── Temperature-delta gated lags ─────────────────────────────────────
    # Suppress lag features proportionally when today is warmer than the lag
    # period — prevents stale HP-consumption lags from anchoring predictions
    # during warm-season transitions.  Self-heals once the season stabilises
    # (delta → 0, lags fully restored).
    if "lag_24h" in df.columns and "temp_delta_24h" in df.columns:
        discount_24h = (df["temp_delta_24h"].clip(lower=0) / 5.0).clip(upper=1.0)
        df["lag_24h_tgated"] = df["lag_24h"] * (1.0 - discount_24h)
    else:
        df["lag_24h_tgated"] = df.get("lag_24h", 0.0)

    if "lag_168h" in df.columns and "temp_lag_168h" in df.columns:
        delta_168h = (df["temp_c"] - df["temp_lag_168h"]).clip(lower=0)
        discount_168h = (delta_168h / 8.0).clip(upper=1.0)
        df["lag_168h_tgated"] = df["lag_168h"] * (1.0 - discount_168h)
    else:
        df["lag_168h_tgated"] = df.get("lag_168h", 0.0)

    if "lag_336h" in df.columns and "temp_lag_336h" in df.columns:
        delta_336h = (df["temp_c"] - df["temp_lag_336h"]).clip(lower=0)
        discount_336h = (delta_336h / 8.0).clip(upper=1.0)
        df["lag_336h_tgated"] = df["lag_336h"] * (1.0 - discount_336h)
    else:
        df["lag_336h_tgated"] = df.get("lag_336h", 0.0)

    # ── Stage 2: Thermal Pressure (Climate) ──────────────────────────────
    # thermal_pressure      — area-weighted mean deficit (°C·h), primary signal
    # thermal_pressure_max  — largest per-room deficit (cold outlier room)
    # thermal_pressure_std  — spread across rooms (imbalance signal)
    if climate_dfs:
        delta_series: dict[str, pd.Series] = {}  # entity_id → per-timestamp delta Series
        ts_index = None
        for eid, c_df in climate_dfs.items():
            if c_df.empty:
                continue
            c = c_df[["timestamp", "current_temp", "setpoint"]].copy()
            c["timestamp"] = pd.to_datetime(c["timestamp"]).dt.floor("1h")
            c = c.sort_values("timestamp")
            delta = (c["setpoint"] - c["current_temp"]).clip(lower=0.0)
            delta.index = c["timestamp"]
            delta_series[eid] = delta
            if ts_index is None:
                ts_index = c["timestamp"]

        if delta_series:
            delta_df = pd.DataFrame(delta_series)  # rows=timestamps, cols=entities
            areas = {eid: (room_areas or {}).get(eid, DEFAULT_ROOM_AREA_M2) for eid in delta_series}
            total_area = sum(areas.values()) or 1.0
            area_weights = pd.Series(areas)

            # Weighted mean (°C·h) — primary feature
            weighted_sum = (delta_df * area_weights).sum(axis=1)
            pressure_series = weighted_sum / total_area

            # Per-row secondary stats
            max_series = delta_df.max(axis=1)
            std_series = delta_df.std(axis=1).fillna(0.0)

            summary = pd.DataFrame(
                {
                    "timestamp": delta_df.index,
                    "thermal_pressure": pressure_series.values,
                    "thermal_pressure_max": max_series.values,
                    "thermal_pressure_std": std_series.values,
                }
            )

            df["_ts_floor"] = df["timestamp"].dt.floor("1h")
            df = df.merge(summary, left_on="_ts_floor", right_on="timestamp", how="left", suffixes=("", "_tp"))
            df.drop(columns=["timestamp_tp", "_ts_floor"], errors="ignore", inplace=True)
            df["thermal_pressure"] = df["thermal_pressure"].fillna(0.0)
            df["thermal_pressure_max"] = df["thermal_pressure_max"].fillna(0.0)
            df["thermal_pressure_std"] = df["thermal_pressure_std"].fillna(0.0)
        else:
            df["thermal_pressure"] = 0.0
            df["thermal_pressure_max"] = 0.0
            df["thermal_pressure_std"] = 0.0
    else:
        df["thermal_pressure"] = 0.0
        df["thermal_pressure_max"] = 0.0
        df["thermal_pressure_std"] = 0.0

    # ── Stage 2: DHW Pressure ───────────────────────────────────────────
    if dhw_df is not None and not dhw_df.empty:
        d = dhw_df[["timestamp", "buffer_temp"]].copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"]).dt.floor("1h")
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(d, left_on="_ts_floor", right_on="timestamp", how="left", suffixes=("", "_dhw"))
        df.drop(columns=["timestamp_dhw", "_ts_floor"], errors="ignore", inplace=True)
        # Rename merged column to the feature name
        if "buffer_temp" in df.columns:
            df.rename(columns={"buffer_temp": "dhw_buffer_temp"}, inplace=True)

        df["dhw_buffer_temp"] = df["dhw_buffer_temp"].fillna(50.0)  # Assume neutral 50C
        # non-linear trigger: higher as temp drops towards 40C floor
        df["dhw_pressure"] = 1.0 / (np.maximum(0.5, df["dhw_buffer_temp"] - 40.0 + 1.0) ** 2)
    else:
        df["dhw_buffer_temp"] = 50.0
        df["dhw_pressure"] = 0.0

    # ── §3: Passive solar gain (time-of-day weighted radiation) ──────────────
    # Half-cosine window peaks at 13:00, reaches 0 at 09:00 and 17:00.
    # Captures south-facing passive gain that reduces actual heating demand.
    hour_arr = df["timestamp"].dt.hour.values.astype(float)
    cos_factor = np.where(
        (hour_arr >= 9) & (hour_arr <= 17),
        np.cos(np.pi * (hour_arr - 13.0) / 4.0),
        0.0,
    ).clip(0.0)  # cosine is negative near edges — floor at 0
    df["weighted_solar_gain"] = df["direct_radiation_wm2"] * cos_factor

    # ── §3: COP proxy — thermal pressure as electrical urgency ───────────────
    # Linear COP model: COP ≈ 0.11 × T_out + 3.0 (air-to-water HP, radiators).
    # Same °C heat debt costs ~2× more electricity at −10°C vs +10°C outdoors.
    # Denominator clamped at 0.5 to guard against extreme negative temps.
    cop_proxy = (0.11 * df["temp_c"] + 3.0).clip(lower=0.5)
    df["thermal_pressure_cop"] = df["thermal_pressure"] / cop_proxy

    # ── §3: Physics feature scaling rationale ─────────────────────────────────
    # Constants below are empirical, chosen to keep each feature in a ~0–5 range
    # so LightGBM split-gain is not dominated by one feature's raw magnitude:
    #
    #   0.01 (solar gain, infiltration): wind_kmh × thermal_pressure can reach
    #     ~500 raw (e.g. 50 km/h × 10 °C delta); 0.01 rescales to ~5. Solar gain
    #     similarly reaches ~500 Wm⁻² × peak cosine; 0.01 maps to ~5 kWh equiv.
    #
    #   10.0 (defrost Gaussian σ²): width ≈ ±3σ = ±√30 ≈ ±5.5 °C from the 2 °C
    #     peak. This covers the observed icing range of −3 °C to +7 °C at ≥ 5 %
    #     of peak intensity. Empirical: wider (σ²=20) loses the 0 °C/5 °C split;
    #     narrower (σ²=4) misses high-humidity defrost above 5 °C.
    #
    # Both 0.01 and 10.0 are candidates for learnable interaction terms if a
    # dedicated physics-calibration pipeline is added (see ROADMAP).
    # ── §3: Solar-compensated pressure (#56) ─────────────────────────────────
    # Primary signal: subtract passive solar gain from thermal pressure.
    # High impact on reducing over-forecast on sunny winter days.
    df["thermal_pressure_net"] = np.maximum(0.0, df["thermal_pressure"] - 0.01 * df["weighted_solar_gain"])

    # ── §3: Wind-driven infiltration (#57) ───────────────────────────────────
    # Feature interaction: infiltration loss ∝ WindSpeed * (T_in - T_out).
    # We use thermal_pressure as a proxy for the temperature gradient.
    df["infiltration_pressure"] = 0.01 * df["wind_kmh"] * df["thermal_pressure"]

    # ── §3: Humidity-aware defrost proxy (#58) ───────────────────────────────
    # Peaks at 2°C; Gaussian σ²=10 covers observed icing range −3°C to +7°C.
    # Scaled by relative humidity (dimensionless, 0–1).
    defrost_temp_curve = np.exp(-((df["temp_c"] - 2.0) ** 2) / 10.0)
    df["defrost_risk"] = (df["humidity"] / 100.0) * defrost_temp_curve

    # ── Daily Regime Profile (optional) ──────────────────────────────────────
    if regime_kwh_series is not None:
        # Merge hourly profile by timestamp
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(regime_kwh_series.to_frame("regime_kwh"), left_on="_ts_floor", right_index=True, how="left")
        df.drop(columns=["_ts_floor"], inplace=True, errors="ignore")
        df["regime_kwh"] = df["regime_kwh"].fillna(0.0)
    else:
        df["regime_kwh"] = 0.0

    # Suppress regime_kwh when warmer than last week — centroids trained on
    # heating-season data overestimate during warm-season transitions.
    if "temp_c" in df.columns and "temp_lag_168h" in df.columns:
        delta_168h = (df["temp_c"] - df["temp_lag_168h"]).clip(lower=0)
        df["regime_kwh"] *= 1.0 - (delta_168h / 8.0).clip(upper=1.0)

    # ── Physics baseline (optional) ──────────────────────────────────────────
    if physics_kwh_series is not None:
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(physics_kwh_series.to_frame("physics_kwh"), left_on="_ts_floor", right_index=True, how="left")
        df.drop(columns=["_ts_floor"], inplace=True, errors="ignore")
        df["physics_kwh"] = df["physics_kwh"].fillna(0.0)

    # ── Heating buffer temp (optional direct sensor feature) ──────────────────
    if heating_buffer_temp_series is not None:
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(
            heating_buffer_temp_series.to_frame("heating_buffer_temp"),
            left_on="_ts_floor",
            right_index=True,
            how="left",
        )
        df.drop(columns=["_ts_floor"], inplace=True, errors="ignore")
        df["heating_buffer_temp"] = df["heating_buffer_temp"].ffill().fillna(df["heating_buffer_temp"].median())

    return df


_BRIDGE_CAP = 3  # days — bridge-day effect beyond this distance is negligible


def _add_holiday_feature(df: pd.DataFrame, canton: str | None = None, country: str = "CH") -> pd.DataFrame:
    """Add holiday proximity columns using the `holidays` package.

    Columns added:
      is_public_holiday       — 1 on a public holiday for the configured country, else 0
      days_to_next_holiday    — calendar days until the next holiday, capped at _BRIDGE_CAP
      days_since_last_holiday — calendar days since the last holiday, capped at _BRIDGE_CAP

    Both distance columns are 0 on a holiday itself.  The cap keeps the feature
    range tight so the model focuses on the bridging-day window where consumption
    patterns actually differ.

    Pass canton (e.g. "ZH", "BE") to include cantonal holidays in addition to
    federal ones.  Falls back to federal-only if the canton code is unrecognised.
    Pass country (ISO 3166-1 alpha-2, e.g. "CH", "GB", "DE") to use the holidays
    of the user's country.  Defaults to "CH".

    Falls back gracefully (is_public_holiday=0, distances=_BRIDGE_CAP) if the
    `holidays` package is not installed.
    """
    try:
        import holidays as hd  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: F401

        # Extend by ±1 year so dates near year boundaries (e.g. Dec 31)
        # can see the next/previous holiday across the year boundary.
        years_in_data = df["timestamp"].dt.year.unique()
        years = sorted({y + d for y in years_in_data for d in (-1, 0, 1)})
        if canton:
            try:
                ch_holidays = hd.country_holidays(country, years=years, subdiv=canton)
            except (NotImplementedError, KeyError):
                _LOGGER.warning(
                    "Unknown canton/subdivision code %r for country %r — falling back to national holidays.",
                    canton,
                    country,
                )
                ch_holidays = hd.country_holidays(country, years=years)
        else:
            ch_holidays = hd.country_holidays(country, years=years)
        holiday_dates = sorted(ch_holidays.keys())

        dates = df["timestamp"].dt.date

        df["is_public_holiday"] = dates.isin(set(ch_holidays.keys())).astype(int)

        if holiday_dates:
            # Vectorised distance computation via ordinals + np.searchsorted.
            # bisect_left semantics: on a holiday, idx points to the holiday → distance 0.
            holiday_ords = np.array([d.toordinal() for d in holiday_dates])
            date_ords = np.array([d.toordinal() for d in dates])

            idx_next = np.searchsorted(holiday_ords, date_ords, side="left")
            safe_next = np.minimum(idx_next, len(holiday_ords) - 1)
            days_next = np.where(
                idx_next >= len(holiday_ords),
                _BRIDGE_CAP,
                holiday_ords[safe_next] - date_ords,
            )
            df["days_to_next_holiday"] = np.minimum(_BRIDGE_CAP, days_next).astype(int)

            idx_prev = np.searchsorted(holiday_ords, date_ords, side="right") - 1
            safe_prev = np.maximum(idx_prev, 0)
            days_since = np.where(
                idx_prev < 0,
                _BRIDGE_CAP,
                date_ords - holiday_ords[safe_prev],
            )
            df["days_since_last_holiday"] = np.minimum(_BRIDGE_CAP, days_since).astype(int)
        else:
            df["days_to_next_holiday"] = _BRIDGE_CAP
            df["days_since_last_holiday"] = _BRIDGE_CAP

    except ImportError:
        df["is_public_holiday"] = 0
        df["days_to_next_holiday"] = _BRIDGE_CAP
        df["days_since_last_holiday"] = _BRIDGE_CAP
    except (KeyError, ValueError, TypeError, AttributeError, NotImplementedError) as exc:
        _LOGGER.warning("Holiday feature failed: %s — defaulting to 0 / %d", exc, _BRIDGE_CAP)
        df["is_public_holiday"] = 0
        df["days_to_next_holiday"] = _BRIDGE_CAP
        df["days_since_last_holiday"] = _BRIDGE_CAP
    return df


_EV_HOW_MIN_FRACTION = 0.15  # an hour_of_week slot is "likely EV" if it was a
# charging hour in ≥ 15% of its historical occurrences


def _compute_likely_ev_hours(
    baseline_df: Any,  # all training rows (EV hours have kwh already subtracted)
    ev_df: Any,  # rows classified as EV charging (original kwh)
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
    ev_counts = _how(ev_df).value_counts()

    # Align on the same index; missing slots in ev_counts → 0
    fractions = ev_counts.reindex(total_counts.index, fill_value=0) / total_counts
    likely = fractions[fractions >= _EV_HOW_MIN_FRACTION].index.tolist()
    if likely:
        _LOGGER.info(
            "EV pattern: %d hour_of_week slots flagged as likely charging windows (fraction ≥ %.0f%%).",
            len(likely),
            _EV_HOW_MIN_FRACTION * 100,
        )
    return set(likely)


def _build_prediction_temp_df(
    future_hours,  # pd.DatetimeIndex — naive
    forecast_df,  # pd.DataFrame — naive timestamps
    live_temp: float,
) -> Any:
    import pandas as pd

    fc_indexed = forecast_df.set_index(pd.to_datetime(forecast_df["timestamp"]))["temp_c"].sort_index()
    blend_range = SENSOR_BLEND_HOURS - SENSOR_FULL_TRUST_HOURS  # hours over which we fade bias

    # Compute bias (sensor offset) relative to forecast at h=0
    fc_temp_0 = float(fc_indexed.iloc[0]) if not fc_indexed.empty else live_temp
    bias = live_temp - fc_temp_0

    rows = []
    for i, ts in enumerate(future_hours):
        hours_ahead = i  # future_hours[0] is current hour
        if hours_ahead <= SENSOR_FULL_TRUST_HOURS:
            temp = live_temp
        elif hours_ahead >= SENSOR_BLEND_HOURS:
            temp = float(fc_indexed.asof(ts)) if not fc_indexed.empty else live_temp
        else:
            # Bias fade: anchor to forecast trajectory, fade out sensor offset.
            # alpha ramps from 0 at SENSOR_FULL_TRUST_HOURS → 1 at SENSOR_BLEND_HOURS.
            # As alpha → 1, (1-alpha) → 0, so bias contribution shrinks to zero.
            alpha = (hours_ahead - SENSOR_FULL_TRUST_HOURS) / blend_range
            fc_temp = float(fc_indexed.asof(ts)) if not fc_indexed.empty else live_temp
            temp = fc_temp + bias * (1 - alpha)
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


def _build_model(lgb: Any, GBR: Any, n_estimators: int | None = None, num_leaves: int = 31) -> Any:
    """Instantiate the best available model (LightGBM preferred, sklearn GBR fallback).

    n_estimators overrides the default when provided (e.g. from early stopping).
    num_leaves is LightGBM-only; ignored for sklearn GBR.
    """
    if lgb is not None:
        return lgb.LGBMRegressor(
            n_estimators=n_estimators or 500,
            learning_rate=0.05,
            num_leaves=num_leaves,
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
