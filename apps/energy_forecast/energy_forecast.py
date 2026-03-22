"""
HA Energy Forecast — AppDaemon app.

EV charging handling:
  Hours where gross grid import exceeds ev_charging_threshold_kwh (default
  7 kWh/h) are classified as EV charging.  The fixed charger load (9 kWh/h)
  is subtracted, leaving the concurrent household baseline intact.  The model
  trains on this cleaned signal, so EV sessions don't distort forecasts.

  Two sensors are published from measured actuals:
    sensor.energy_forecast_ev_today      — EV kWh detected today
    sensor.energy_forecast_ev_yesterday  — EV kWh detected yesterday

  The threshold and charger power are configurable in apps.yaml:
    ev_charging_threshold_kwh: 7     # default
    ev_charger_kw: 9.0               # default
"""
from __future__ import annotations

import threading
from datetime import datetime, time
from pathlib import Path
from typing import Any

import hassapi as hass

from . import ha_data, weather
from .const import CACHE_PATH, EV_CHARGING_THRESHOLD_KWH
from .model import EnergyForecastModel

# ── Operational constants l ─────────────────────────────────────────────────────
RETRAIN_INTERVAL_S = 168 * 3600   # weekly
MIN_HISTORY_HOURS  = 48
BLOCK_SLOTS        = [f"{h:02d}_{h+3:02d}" for h in range(0, 24, 3)]
ATTRIBUTION        = "HA Energy Forecast — LightGBM + MeteoSwiss/Open-Meteo"


class EnergyForecast(hass.Hass):
    """AppDaemon app that forecasts household energy consumption."""

    def initialize(self) -> None:
        self.log("HA Energy Forecast initialising…")

        self._energy_sensor: str         = self.args["energy_sensor"]
        self._outdoor_sensor: str | None = self.args.get("outdoor_temp_sensor")
        self._plz: str                   = str(self.args.get("plz", ""))
        self._lat: float                 = float(self.args["latitude"])
        self._lon: float                 = float(self.args["longitude"])
        self._weight_halflife: float     = float(self.args.get("weight_halflife_days", 90))
        self._ev_threshold: float        = float(
            self.args.get("ev_charging_threshold_kwh", EV_CHARGING_THRESHOLD_KWH)
        )
        # Fixed charger power in kW — subtracted from charging hours so the
        # concurrent household baseline is preserved in training data.
        self._ev_charger_kw: float       = float(self.args.get("ev_charger_kw", 9.0))
        self._cache_path: Path           = Path(self.args.get("cache_path", str(CACHE_PATH)))
        self._holiday_canton: str | None = self.args.get("holiday_canton") or None
        self._adaptive_retrain_threshold: float = float(
            self.args.get("adaptive_retrain_threshold", 2.0)
        )
        # Optional sub-sensors: cumulative kWh meters (heat pump, dishwasher, etc.)
        # whose consumption is tracked as lag features to improve forecast accuracy.
        self._sub_energy_sensors: list[str] = list(self.args.get("sub_energy_sensors") or [])
        # Prediction history for adaptive retrain: {target_timestamp: predicted_kwh}.
        # Keep-first semantics so we track h≈24+ ahead predictions, not h=1.
        self._pred_history: dict        = {}
        self._last_adaptive_retrain: datetime = datetime.min

        self._validate_config()

        model_dir = Path(__file__).parent / "models"
        self._ml_model = EnergyForecastModel(model_dir)
        self._lock = threading.Lock()

        self.listen_event(self._retrain_cb, "RELOAD_ENERGY_MODEL")

        self._check_setup()
        self._publish_unavailable()
        self.run_in(self._retrain_cb, 10)
        self.run_every(self._retrain_cb, f"now+{RETRAIN_INTERVAL_S + 10}", RETRAIN_INTERVAL_S)
        self.run_in(self._update_cb, 130)
        self.run_hourly(self._update_cb, time(0, 1, 0))

        self.log(
            f"HA Energy Forecast ready. "
            f"EV threshold: {self._ev_threshold} kWh/h, "
            f"charger: {self._ev_charger_kw} kW"
        )

    # ── Config validation ─────────────────────────────────────────────────────

    def _validate_config(self) -> None:
        """Validate configuration values at startup; raises ValueError on bad input."""
        if not (-90 <= self._lat <= 90):
            raise ValueError(f"latitude must be between -90 and 90, got {self._lat}")
        if not (-180 <= self._lon <= 180):
            raise ValueError(f"longitude must be between -180 and 180, got {self._lon}")
        if self._weight_halflife <= 0:
            raise ValueError(
                f"weight_halflife_days must be positive, got {self._weight_halflife}"
            )
        if self._ev_threshold <= 0:
            raise ValueError(
                f"ev_charging_threshold_kwh must be positive, got {self._ev_threshold}"
            )
        if self._ev_charger_kw <= 0:
            raise ValueError(f"ev_charger_kw must be positive, got {self._ev_charger_kw}")
        if self._adaptive_retrain_threshold < 0:
            raise ValueError(
                f"adaptive_retrain_threshold must be ≥ 0, got {self._adaptive_retrain_threshold}"
            )
        if self._ev_threshold >= self._ev_charger_kw:
            self.log(
                f"ev_charging_threshold_kwh ({self._ev_threshold}) is ≥ ev_charger_kw "
                f"({self._ev_charger_kw}). EV sessions may not be detected correctly — "
                "lower the threshold or raise ev_charger_kw.",
                level="WARNING",
            )
        self.log(
            f"Config validated — lat={self._lat}, lon={self._lon}, plz={self._plz}, "
            f"weight_halflife={self._weight_halflife}d, "
            f"ev_threshold={self._ev_threshold} kWh/h, ev_charger={self._ev_charger_kw} kW, "
            f"sub_energy_sensors={len(self._sub_energy_sensors)}"
        )

    # ── Setup checker ─────────────────────────────────────────────────────────

    def _check_setup(self) -> None:
        """Publish sensor.energy_forecast_setup_status with import diagnostics.

        State is "ok" when all required packages are importable.  If a package
        is missing the state is "missing_packages" and the attributes list which
        ones failed, so users can diagnose install issues from HA dev tools
        without reading AppDaemon logs.
        """
        _REQUIRED = [
            ("pandas",    "pandas"),
            ("numpy",     "numpy"),
            ("sklearn",   "scikit-learn"),
            ("requests",  "requests"),
            ("holidays",  "holidays"),
        ]
        missing: list[str] = []
        for module, pip_name in _REQUIRED:
            try:
                __import__(module)
            except ImportError:
                missing.append(pip_name)

        if missing:
            state = "missing_packages"
            self.log(
                f"Setup check: missing packages — {missing}. "
                "Install them via AppDaemon add-on configuration.",
                level="WARNING",
            )
        else:
            state = "ok"

        try:
            self.set_state(
                "sensor.energy_forecast_setup_status",
                state=state,
                attributes={
                    "friendly_name": "Energy Forecast Setup Status",
                    "unique_id": "energy_forecast_setup_status",
                    "missing_packages": missing,
                    "icon": "mdi:check-circle" if state == "ok" else "mdi:alert-circle",
                },
                replace=True,
            )
        except (AttributeError, TypeError, RuntimeError) as exc:
            self.log(f"Could not publish setup status sensor: {exc}", level="WARNING")

    # ── Sub-sensor helpers ────────────────────────────────────────────────────

    def _sub_sensor_prefix(self, entity_id: str) -> str:
        """Return the feature-column prefix for a sub-energy sensor entity_id."""
        sanitized = entity_id.split(".", 1)[-1].replace(".", "_")
        return f"sub_{sanitized}"

    def _sub_sensor_cache_path(self, entity_id: str) -> Path:
        """Return the CSV cache path for a sub-energy sensor."""
        sanitized = entity_id.split(".", 1)[-1].replace(".", "_")
        return self._cache_path.parent / f"sub_{sanitized}.csv"

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _retrain_cb(self, event_name=None, data=None, kwargs=None) -> None:
        # Accepts both timer callbacks (single positional arg) and
        # listen_event callbacks (event_name, data, kwargs).
        if not self._lock.acquire(blocking=False):
            self.log("Retrain skipped — another operation is running.", level="DEBUG")
            return
        try:
            self._retrain()
            if self._ml_model.model is not None:
                self._update_sensors()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Retraining failed: {exc}", level="ERROR")
        finally:
            self._lock.release()

    def _update_cb(self, event_name=None, data=None, kwargs=None) -> None:
        if self._ml_model.model is None:
            return
        if not self._lock.acquire(blocking=False):
            self.log("Sensor update skipped — another operation is running.", level="DEBUG")
            return
        try:
            self._update_sensors()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Sensor update failed: {exc}", level="ERROR")
        finally:
            self._lock.release()

    # ── Core logic ────────────────────────────────────────────────────────────

    def _retrain(self) -> None:
        self.log("Starting model retraining…")
        energy_df = ha_data.fetch_energy_history(self, self._energy_sensor, cache_path=self._cache_path)

        if len(energy_df) < MIN_HISTORY_HOURS:
            self.log(f"Insufficient history ({len(energy_df)} h). Skipping.", level="WARNING")
            return

        energy_df = _strip_tz(energy_df)

        # ── Subtract EV charging from gross import ────────────────────────────
        baseline_df, ev_df = ha_data.split_ev_charging(
            energy_df, self._ev_threshold, charger_kw=self._ev_charger_kw
        )
        if len(ev_df):
            self.log(
                f"EV filter: {len(ev_df)} charging hours detected "
                f"({ev_df['gross_kwh'].sum():.1f} kWh gross). "
                f"Sessions on: {sorted(ev_df['timestamp'].dt.date.unique().tolist())}"
            )

        start_date = baseline_df["timestamp"].min().date()
        end_date   = baseline_df["timestamp"].max().date()

        try:
            weather_df = weather.fetch_historical_weather(self._lat, self._lon, start_date, end_date)
            weather_df = _strip_tz(weather_df)
        except (OSError, KeyError, ValueError) as exc:
            self.log(
                f"Historical weather fetch failed: {exc} — "
                "temp_c, heating_degree, cooling_degree and temp_rolling_3d will be "
                "imputed from training-set medians; forecast quality will be reduced.",
                level="WARNING",
            )
            weather_df = _empty_weather_df()

        sub_sensors_dict: dict = {}
        for entity_id in self._sub_energy_sensors:
            prefix = self._sub_sensor_prefix(entity_id)
            cache_path = self._sub_sensor_cache_path(entity_id)
            try:
                sub_df = ha_data.fetch_sub_sensor_history(self, entity_id, cache_path)
                sub_df = _strip_tz(sub_df)
                sub_sensors_dict[prefix] = sub_df
            except (OSError, KeyError, ValueError) as exc:
                self.log(f"Sub-sensor {entity_id} history fetch failed: {exc}", level="WARNING")

        self._ml_model.train(
            baseline_df,
            weather_df,
            outdoor_df=None,
            weight_halflife_days=self._weight_halflife,
            canton=self._holiday_canton,
            ev_df=ev_df,
            sub_sensors_dict=sub_sensors_dict or None,
        )
        self.log(f"Retrained. MAE: {self._ml_model.last_mae}")

    def _update_sensors(self) -> None:
        import pandas as pd

        # ── Fetch weather forecast ────────────────────────────────────────────
        forecast_df = weather.fetch_forecast(
            self._plz,
            self._lat,
            self._lon,
            self.args.get("srg_client_id"),
            self.args.get("srg_client_secret"),
        )
        forecast_df = _strip_tz(forecast_df)

        # ── Fetch recent actuals ──────────────────────────────────────────────
        # Uses the lightweight fetch (last 2 days only) to stay well within
        # AppDaemon's 10s callback limit. Full 30-day resync happens in _retrain().
        try:
            full_actuals = ha_data.fetch_recent_energy(self, self._energy_sensor, cache_path=self._cache_path)
            full_actuals = _strip_tz(full_actuals)
            # Subtract EV from actuals so lag_24h pointing at a charging hour
            # doesn't inflate tomorrow's baseline prediction.
            recent_actuals, _ = ha_data.split_ev_charging(
                full_actuals, self._ev_threshold, charger_kw=self._ev_charger_kw
            )
        except (OSError, ValueError, KeyError) as exc:
            self.log(f"Could not fetch recent actuals for lag features: {exc}", level="WARNING")
            recent_actuals = None
            full_actuals   = None

        sub_sensors_recent: dict = {}
        for entity_id in self._sub_energy_sensors:
            prefix = self._sub_sensor_prefix(entity_id)
            cache_path = self._sub_sensor_cache_path(entity_id)
            try:
                sub_df = ha_data.fetch_recent_sub_sensor(self, entity_id, cache_path)
                sub_df = _strip_tz(sub_df)
                sub_sensors_recent[prefix] = sub_df
            except (OSError, KeyError, ValueError) as exc:
                self.log(f"Sub-sensor {entity_id} recent fetch failed: {exc}", level="WARNING")

        live_temp   = self._read_live_temp()
        predictions = self._ml_model.predict(forecast_df, live_temp, recent_actuals,
                                             sub_sensors_recent=sub_sensors_recent or None)
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"]).dt.tz_localize(None)

        intervals = self._ml_model.predict_intervals(forecast_df, live_temp, recent_actuals,
                                                     sub_sensors_recent=sub_sensors_recent or None)
        if intervals is not None:
            intervals["timestamp"] = pd.to_datetime(intervals["timestamp"]).dt.tz_localize(None)

        # ── Store predictions for adaptive retrain tracking ───────────────────
        # Keep-first: only store a prediction for each target hour the first time
        # we see it (~24h ahead), so MAE is measured on day-ahead forecasts.
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=7)
        self._pred_history = {
            ts: kwh for ts, kwh in self._pred_history.items()
            if pd.Timestamp(ts) >= cutoff
        }
        for _, row in predictions.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            if ts not in self._pred_history:
                self._pred_history[ts] = float(row["predicted_kwh"])

        self._maybe_adaptive_retrain(recent_actuals)

        aggregated = self._aggregate(predictions, full_actuals, live_temp, intervals=intervals)
        self._publish(aggregated)

    def _read_live_temp(self) -> float | None:
        if not self._outdoor_sensor:
            return None
        state = self.get_state(self._outdoor_sensor)
        if state in (None, "unavailable", "unknown"):
            return None
        try:
            return float(state)
        except (ValueError, TypeError):
            return None

    def _maybe_adaptive_retrain(self, actuals_df: Any) -> None:
        """Trigger an early retrain if live MAE exceeds threshold × CV MAE."""
        import pandas as pd

        cv_mae = self._ml_model.last_cv_mae
        if cv_mae is None:
            return
        # Use Europe/Zurich local time (tz-naive) consistent with pipeline timestamps.
        # datetime.now() would use system time, which is UTC in Docker/HA and
        # causes the cooldown to fire up to ±2h early/late and wrong during DST.
        _now = pd.Timestamp.now("Europe/Zurich").tz_localize(None)
        hours_since = (_now - self._last_adaptive_retrain).total_seconds() / 3600
        if hours_since < 24:
            return
        live_mae, n_pairs = _compute_live_mae(self._pred_history, actuals_df)
        if n_pairs < 24:
            return
        if live_mae > self._adaptive_retrain_threshold * cv_mae:
            self.log(
                f"Adaptive retrain triggered: live_MAE={live_mae:.4f} > "
                f"{self._adaptive_retrain_threshold}× cv_MAE={cv_mae:.4f} "
                f"(over {n_pairs} matched hours)",
                level="WARNING",
            )
            self._last_adaptive_retrain = pd.Timestamp.now("Europe/Zurich").tz_localize(None)
            self._retrain()

    # ── Sensor publishing ─────────────────────────────────────────────────────

    def _publish_unavailable(self) -> None:
        for slot in ["next_3h", "today", "tomorrow"]:
            self.set_state(
                f"sensor.energy_forecast_{slot}",
                state="unavailable",
                attributes={
                    "unit_of_measurement": "kWh",
                    "friendly_name": f"Energy Forecast {slot.title().replace('_', ' ')}",
                    "unique_id": f"energy_forecast_{slot}",
                    "icon": "mdi:lightning-bolt",
                },
                replace=True,
            )
        for day in ("today", "yesterday"):
            self.set_state(
                f"sensor.energy_forecast_ev_{day}",
                state="unavailable",
                attributes={
                    "unit_of_measurement": "kWh",
                    "friendly_name": f"EV Charging Detected {day.title()}",
                    "unique_id": f"energy_forecast_ev_{day}",
                    "icon": "mdi:car-electric",
                },
                replace=True,
            )

    def _publish(self, data: dict) -> None:
        import math

        model       = self._ml_model
        trained_str = (
            model.last_trained.strftime("%Y-%m-%d %H:%M")
            if model.last_trained != datetime.min
            else "never"
        )
        base_attrs = {
            "unit_of_measurement": "kWh",
            "attribution": ATTRIBUTION,
            "model_engine": str(model.engine),
            "last_trained": trained_str,
        }

        def safe_set(entity_id: str, value: Any, friendly_name: str, extra_attrs: dict | None = None, icon: str | None = None) -> None:
            try:
                val = float(value)
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
            except (TypeError, ValueError):
                val = 0.0
            attrs = {
                **base_attrs,
                "friendly_name": friendly_name,
                "unique_id": entity_id.split(".", 1)[-1],
            }
            if icon:
                attrs["icon"] = icon
            if extra_attrs:
                attrs.update(extra_attrs)
            self.set_state(
                entity_id,
                state=str(round(val, 3)),
                attributes=attrs,
                replace=True,
            )

        # ── Forecast totals ───────────────────────────────────────────────────
        for key, label in [("next_3h", "Next 3h"), ("today", "Today"), ("tomorrow", "Tomorrow")]:
            safe_set(f"sensor.energy_forecast_{key}", data.get(key, 0), f"Energy Forecast {label}", icon="mdi:lightning-bolt")

        # ── Prediction intervals (only published when quantile models trained) ─
        for key, label in [("next_3h", "Next 3h"), ("today", "Today"), ("tomorrow", "Tomorrow")]:
            low  = data.get(f"{key}_low")
            high = data.get(f"{key}_high")
            if low is not None and high is not None:
                safe_set(f"sensor.energy_forecast_{key}_low",  low,  f"Energy Forecast {label} Low (10th pct)",  icon="mdi:arrow-down-bold")
                safe_set(f"sensor.energy_forecast_{key}_high", high, f"Energy Forecast {label} High (90th pct)", icon="mdi:arrow-up-bold")

        # ── Forecast 3-hour blocks ────────────────────────────────────────────
        for day in ("today", "tomorrow"):
            blocks = data.get(f"blocks_{day}", {})
            for slot in BLOCK_SLOTS:
                h_start, h_end = slot.split("_")
                safe_set(
                    f"sensor.energy_forecast_{day}_{slot}",
                    blocks.get(slot, 0),
                    f"Energy Forecast {day.title()} {h_start}:00–{h_end}:00",
                    icon="mdi:calendar-clock",
                )

        # ── EV actuals sensors ────────────────────────────────────────────────
        ev_attrs = {
            "ev_threshold_kwh": self._ev_threshold,
            "ev_charger_kw":    self._ev_charger_kw,
        }
        safe_set(
            "sensor.energy_forecast_ev_today",
            data.get("ev_today", 0),
            "EV Charging Detected Today",
            extra_attrs=ev_attrs,
            icon="mdi:car-electric",
        )
        safe_set(
            "sensor.energy_forecast_ev_yesterday",
            data.get("ev_yesterday", 0),
            "EV Charging Detected Yesterday",
            extra_attrs=ev_attrs,
            icon="mdi:car-electric",
        )

        # ── Model MAE sensor ──────────────────────────────────────────────────
        mae_val = model.last_mae if model.last_mae is not None else 0
        self.set_state(
            "sensor.energy_forecast_model_mae",
            state=str(round(float(mae_val), 4)),
            attributes={
                "unit_of_measurement": "kWh",
                "friendly_name": "Energy Forecast Model MAE",
                "unique_id": "energy_forecast_model_mae",
                "icon": "mdi:chart-bell-curve-cumulative",
                "attribution": ATTRIBUTION,
                "cv_mae": str(model.last_cv_mae) if model.last_cv_mae is not None else "n/a",
                "model_engine": str(model.engine),
                "last_trained": trained_str,
            },
            replace=True,
        )

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, predictions: Any, full_actuals: Any, live_temp: float | None, intervals: Any = None) -> dict:
        import numpy as np
        import pandas as pd

        now_dt      = pd.Timestamp.now(tz="Europe/Zurich").replace(tzinfo=None)
        now_np      = np.datetime64(now_dt.floor("h"))
        today_np    = np.datetime64(now_dt.normalize())
        tomorrow_np = today_np + np.timedelta64(1, "D")
        yesterday_np = today_np - np.timedelta64(1, "D")

        p_times = pd.to_datetime(predictions["timestamp"]).values.astype("datetime64[ns]")
        p_vals  = predictions["predicted_kwh"].values.astype(float)

        def _sum(s, e):
            return round(float(np.sum(p_vals[(p_times >= s) & (p_times < e)])), 3)

        def _blocks(day_start):
            return {
                f"{h:02d}_{h+3:02d}": _sum(
                    day_start + np.timedelta64(h, "h"),
                    day_start + np.timedelta64(h + 3, "h"),
                )
                for h in range(0, 24, 3)
            }

        # Today: substitute actuals for elapsed hours so the sensor reflects
        # measured consumption for the past and forecast for the remainder.
        today_total, blocks_today = _blend_today_totals(
            p_times, p_vals, full_actuals, today_np, tomorrow_np, now_np
        )

        result = {
            "next_3h":         _sum(now_np, now_np + np.timedelta64(3, "h")),
            "today":           today_total,
            "tomorrow":        _sum(tomorrow_np, tomorrow_np + np.timedelta64(1, "D")),
            "blocks_today":    blocks_today,
            "blocks_tomorrow": _blocks(tomorrow_np),
            "live_temp":       live_temp,
            "ev_today":        0.0,
            "ev_yesterday":    0.0,
        }

        # ── Prediction intervals ─────────────────────────────────────────────
        if intervals is not None:
            iv_times = pd.to_datetime(intervals["timestamp"]).values.astype("datetime64[ns]")
            iv_low   = intervals["low_kwh"].values.astype(float)
            iv_high  = intervals["high_kwh"].values.astype(float)

            def _isum(vals, s, e):
                return round(float(np.sum(vals[(iv_times >= s) & (iv_times < e)])), 3)

            today_low,  _ = _blend_today_totals(iv_times, iv_low,  full_actuals, today_np, tomorrow_np, now_np)
            today_high, _ = _blend_today_totals(iv_times, iv_high, full_actuals, today_np, tomorrow_np, now_np)

            result.update({
                "next_3h_low":   _isum(iv_low,  now_np,      now_np + np.timedelta64(3, "h")),
                "next_3h_high":  _isum(iv_high, now_np,      now_np + np.timedelta64(3, "h")),
                "today_low":     today_low,
                "today_high":    today_high,
                "tomorrow_low":  _isum(iv_low,  tomorrow_np, tomorrow_np + np.timedelta64(1, "D")),
                "tomorrow_high": _isum(iv_high, tomorrow_np, tomorrow_np + np.timedelta64(1, "D")),
            })

        # ── EV kWh from actuals: sum (gross - charger_kw) for charging hours ──
        # Subtracts the configured charger power to get the household co-load
        # contribution; the remainder is the estimated EV energy.  Clipped at 0
        # for hours where gross < charger_kw (partial sessions or EV not home).
        if full_actuals is not None and not full_actuals.empty:
            ev_mask  = full_actuals["gross_kwh"] > self._ev_threshold
            ev_rows  = full_actuals[ev_mask].copy()
            if not ev_rows.empty:
                ev_rows["ev_kwh"] = np.maximum(
                    0.0, ev_rows["gross_kwh"] - self._ev_charger_kw
                )
                ev_times = ev_rows["timestamp"].values.astype("datetime64[ns]")
                ev_vals  = ev_rows["ev_kwh"].values.astype(float)

                def _ev_sum(s, e):
                    return round(float(np.sum(ev_vals[(ev_times >= s) & (ev_times < e)])), 3)

                result["ev_today"]     = _ev_sum(today_np,     tomorrow_np)
                result["ev_yesterday"] = _ev_sum(yesterday_np, today_np)

        return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_tz(df: Any) -> Any:
    """Convert timestamp column to naive Europe/Zurich local time."""
    import pandas as pd
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
        df = df.copy()
        df["timestamp"] = ts
    return df


def _empty_weather_df() -> Any:
    import pandas as pd
    return pd.DataFrame(
        columns=[
            "timestamp", "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh",
            "cloud_cover_pct", "direct_radiation_wm2",
        ]
    )


def _blend_today_totals(
    p_times: Any,       # np.ndarray of datetime64[ns] — prediction timestamps
    p_vals: Any,        # np.ndarray of float — predicted kWh per hour
    full_actuals: Any,  # pd.DataFrame | None — cols: timestamp, gross_kwh
    today_np: Any,      # np.datetime64 — midnight today
    tomorrow_np: Any,   # np.datetime64 — midnight tomorrow
    now_np: Any,        # np.datetime64 — current hour (floored)
) -> tuple[float, dict]:
    """Compute today's blended total and 3h blocks.

    Elapsed hours (< now_np) use actuals from full_actuals where available;
    future hours (>= now_np) use model predictions.  Falls back to predictions
    only when full_actuals is None or empty.
    """
    import numpy as np

    fa_times = None
    fa_vals  = None
    if full_actuals is not None and not getattr(full_actuals, "empty", True):
        fa_times = full_actuals["timestamp"].values.astype("datetime64[ns]")
        fa_vals  = full_actuals["gross_kwh"].values.astype(float)

    def _pred_sum(s: Any, e: Any) -> float:
        return float(np.sum(p_vals[(p_times >= s) & (p_times < e)]))

    def _actual_sum(s: Any, e: Any) -> float:
        if fa_times is None:
            return 0.0
        return float(np.sum(fa_vals[(fa_times >= s) & (fa_times < e)]))

    def _blended(s: Any, e: Any) -> float:
        elapsed_end  = min(e, now_np)
        future_start = max(s, now_np)
        total = 0.0
        if elapsed_end > s:
            total += _actual_sum(s, elapsed_end)
        if future_start < e:
            total += _pred_sum(future_start, e)
        return round(total, 3)

    today_total = _blended(today_np, tomorrow_np)
    blocks = {
        f"{h:02d}_{h+3:02d}": _blended(
            today_np + np.timedelta64(h, "h"),
            today_np + np.timedelta64(h + 3, "h"),
        )
        for h in range(0, 24, 3)
    }
    return today_total, blocks


def _compute_live_mae(
    pred_history: dict,  # {timestamp-like: predicted_kwh}
    actuals_df: Any,     # pd.DataFrame | None — cols: timestamp, gross_kwh
) -> tuple[float, int]:
    """Compute MAE between stored predictions and actuals for matched timestamps.

    Returns (mae, n_pairs).  mae is float('nan') when n_pairs == 0.
    Only hours present in both pred_history and actuals_df are included.

    DST fall-back caveat: during the October clock-back, the naive 02:xx hour
    appears twice in the history.  Both occurrences map to the same floor("1h")
    key here, so the second occurrence overwrites the first in actuals_map and
    the wrong actual may be matched to the prediction.  This is an accepted
    edge case (one hour per year) and not worth the complexity of tz-aware storage.
    """
    import pandas as pd

    if actuals_df is None or getattr(actuals_df, "empty", True) or not pred_history:
        return float("nan"), 0

    actuals_map = {
        pd.Timestamp(ts).floor("1h"): float(kwh)
        for ts, kwh in zip(actuals_df["timestamp"], actuals_df["gross_kwh"])
    }

    errors = []
    for ts, pred in pred_history.items():
        key = pd.Timestamp(ts).floor("1h")
        if key in actuals_map:
            errors.append(abs(actuals_map[key] - pred))

    n = len(errors)
    if n == 0:
        return float("nan"), 0
    return round(sum(errors) / n, 4), n
