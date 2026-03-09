"""
HA Energy Forecast — AppDaemon app.

EV charging handling:
  Hours where gross grid import exceeds ev_charging_threshold_kwh (default
  4.5 kWh/h) are classified as EV charging.  The fixed charger load (9 kWh/h)
  is subtracted, leaving the concurrent household baseline intact.  The model
  trains on this cleaned signal, so EV sessions don't distort forecasts.

  Two sensors are published from measured actuals:
    sensor.energy_forecast_ev_today      — EV kWh detected today
    sensor.energy_forecast_ev_yesterday  — EV kWh detected yesterday

  The threshold and charger power are configurable in apps.yaml:
    ev_charging_threshold_kwh: 4.5   # default
    ev_charger_kw: 9.0               # default
"""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import hassapi as hass

from . import ha_data, weather
from .const import EV_CHARGING_THRESHOLD_KWH
from .model import EnergyForecastModel

# ── Operational constants l ─────────────────────────────────────────────────────
RETRAIN_INTERVAL_S = 168 * 3600   # weekly
UPDATE_INTERVAL_S  =       3600   # hourly
MIN_HISTORY_HOURS  = 48
BLOCK_SLOTS        = [f"{h:02d}_{h+3:02d}" for h in range(0, 24, 3)]
ATTRIBUTION        = "HA Energy Forecast — LightGBM + MeteoSwiss/Open-Meteo"


class EnergyForecast(hass.Hass):
    """AppDaemon app that forecasts household energy consumption."""

    def initialize(self) -> None:
        self.log("HA Energy Forecast initialising…")

        self._energy_sensor: str         = self.args["energy_sensor"]
        self._outdoor_sensor: str | None = self.args.get("outdoor_temp_sensor")
        self._plz: str                   = str(self.args["plz"])
        self._lat: float                 = float(self.args["latitude"])
        self._lon: float                 = float(self.args["longitude"])
        self._weight_halflife: float     = float(self.args.get("weight_halflife_days", 90))
        self._ev_threshold: float        = float(
            self.args.get("ev_charging_threshold_kwh", EV_CHARGING_THRESHOLD_KWH)
        )
        # Fixed charger power in kW — subtracted from charging hours so the
        # concurrent household baseline is preserved in training data.
        self._ev_charger_kw: float       = float(self.args.get("ev_charger_kw", 9.0))

        model_dir = Path(__file__).parent / "models"
        self._ml_model = EnergyForecastModel(model_dir)
        self._lock = threading.Lock()

        self.listen_event(self._retrain_cb, "RELOAD_ENERGY_MODEL")

        self._publish_unavailable()
        self.run_in(self._retrain_cb, 10)
        self.run_every(self._retrain_cb, f"now+{RETRAIN_INTERVAL_S + 10}", RETRAIN_INTERVAL_S)
        self.run_in(self._update_cb, 130)
        self.run_every(self._update_cb, f"now+{UPDATE_INTERVAL_S + 130}", UPDATE_INTERVAL_S)

        self.log(
            f"HA Energy Forecast ready. "
            f"EV threshold: {self._ev_threshold} kWh/h, "
            f"charger: {self._ev_charger_kw} kW"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _retrain_cb(self, kwargs: dict) -> None:
        if not self._lock.acquire(blocking=False):
            return
        try:
            self._retrain()
            if self._ml_model.model is not None:
                self._update_sensors()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Retraining failed: {exc}", level="ERROR")
        finally:
            self._lock.release()

    def _update_cb(self, kwargs: dict) -> None:
        if self._ml_model.model is None:
            return
        if not self._lock.acquire(blocking=False):
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
        energy_df = ha_data.fetch_energy_history(self, self._energy_sensor)

        if len(energy_df) < MIN_HISTORY_HOURS:
            self.log(f"Insufficient history ({len(energy_df)} h). Skipping.", level="WARNING")
            return

        energy_df = _strip_tz(energy_df)

        # ── Subtract EV charging from gross import ────────────────────────────
        baseline_df, ev_df = ha_data.split_ev_charging(energy_df, self._ev_threshold)
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
        except Exception as exc:  # noqa: BLE001
            self.log(f"Historical weather fetch failed: {exc}", level="WARNING")
            weather_df = _empty_weather_df()

        self._ml_model.train(
            baseline_df,
            weather_df,
            outdoor_df=None,
            weight_halflife_days=self._weight_halflife,
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
        forecast_df["timestamp"] = (
            pd.to_datetime(forecast_df["timestamp"], utc=True)
            .dt.tz_convert("Europe/Zurich")
            .dt.tz_localize(None)
        )

        # ── Fetch recent actuals ──────────────────────────────────────────────
        # Uses the lightweight fetch (last 2 days only) to stay well within
        # AppDaemon's 10s callback limit. Full 30-day resync happens in _retrain().
        try:
            full_actuals = ha_data.fetch_recent_energy(self, self._energy_sensor)
            full_actuals = _strip_tz(full_actuals)
            # Subtract EV from actuals so lag_24h pointing at a charging hour
            # doesn't inflate tomorrow's baseline prediction.
            recent_actuals, _ = ha_data.split_ev_charging(full_actuals, self._ev_threshold)
        except Exception as exc:  # noqa: BLE001
            self.log(f"Could not fetch recent actuals for lag features: {exc}", level="WARNING")
            recent_actuals = None
            full_actuals   = None

        live_temp   = self._read_live_temp()
        predictions = self._ml_model.predict(forecast_df, live_temp, recent_actuals)
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"]).dt.tz_localize(None)

        aggregated = self._aggregate(predictions, full_actuals, live_temp)
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

    # ── Sensor publishing ─────────────────────────────────────────────────────

    def _publish_unavailable(self) -> None:
        for slot in ["next_3h", "today", "tomorrow"]:
            self.set_state(
                f"sensor.energy_forecast_{slot}",
                state="unavailable",
                attributes={
                    "unit_of_measurement": "kWh",
                    "friendly_name": f"Energy Forecast {slot.title().replace('_', ' ')}",
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

        def safe_set(entity_id: str, value: Any, friendly_name: str, extra_attrs: dict | None = None) -> None:
            try:
                val = float(value)
                if math.isnan(val) or math.isinf(val):
                    val = 0.0
            except (TypeError, ValueError):
                val = 0.0
            attrs = {**base_attrs, "friendly_name": friendly_name}
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
            safe_set(f"sensor.energy_forecast_{key}", data.get(key, 0), f"Energy Forecast {label}")

        # ── Forecast 3-hour blocks ────────────────────────────────────────────
        for day in ("today", "tomorrow"):
            blocks = data.get(f"blocks_{day}", {})
            for slot in BLOCK_SLOTS:
                h_start, h_end = slot.split("_")
                safe_set(
                    f"sensor.energy_forecast_{day}_{slot}",
                    blocks.get(slot, 0),
                    f"Energy Forecast {day.title()} {h_start}:00–{h_end}:00",
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
        )
        safe_set(
            "sensor.energy_forecast_ev_yesterday",
            data.get("ev_yesterday", 0),
            "EV Charging Detected Yesterday",
            extra_attrs=ev_attrs,
        )

        # ── Model MAE sensor ──────────────────────────────────────────────────
        mae_val = model.last_mae if model.last_mae is not None else 0
        self.set_state(
            "sensor.energy_forecast_model_mae",
            state=str(round(float(mae_val), 4)),
            attributes={
                "unit_of_measurement": "kWh",
                "friendly_name": "Energy Forecast Model MAE",
                "attribution": ATTRIBUTION,
                "cv_mae": str(model.last_cv_mae) if model.last_cv_mae is not None else "n/a",
                "model_engine": str(model.engine),
                "last_trained": trained_str,
            },
            replace=True,
        )

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate(self, predictions: Any, full_actuals: Any, live_temp: float | None) -> dict:
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

        result = {
            "next_3h":         _sum(now_np, now_np + np.timedelta64(3, "h")),
            "today":           _sum(today_np, tomorrow_np),
            "tomorrow":        _sum(tomorrow_np, tomorrow_np + np.timedelta64(1, "D")),
            "blocks_today":    _blocks(today_np),
            "blocks_tomorrow": _blocks(tomorrow_np),
            "live_temp":       live_temp,
            "ev_today":        0.0,
            "ev_yesterday":    0.0,
        }

        # ── EV kWh from actuals: sum (gross - threshold) for charging hours ───
        # This gives net charger energy, excluding the household co-load.
        if full_actuals is not None and not full_actuals.empty:
            ev_mask  = full_actuals["gross_kwh"] > self._ev_threshold
            ev_rows  = full_actuals[ev_mask].copy()
            if not ev_rows.empty:
                ev_rows["ev_kwh"] = ev_rows["gross_kwh"] - self._ev_threshold
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
        columns=["timestamp", "temp_c", "precipitation_mm", "sunshine_min", "wind_kmh"]
    )
    