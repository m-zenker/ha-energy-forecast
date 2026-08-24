"""Physics-based hourly electricity predictor (space heating, DHW, base load).

See docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md for the design.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger("energy_forecast.physics")

_VALID_OVERRIDE_KINDS = ("legionella", "comfort_boost")

COP_MIN = 1.1
ETA_CARNOT = 0.45
WATER_SPECIFIC_HEAT_WH_PER_L_K = 1.163
DEFAULT_T_FLOW_C = 45.0
DEFAULT_AMBIENT_C = 20.0
COLD_START_MIN_WINDOWS = 30
STALE_AFTER_DAYS = 30

# Fixed 24h draw-timing shape (sums to 1.0) — see Plan A "Assumptions" section.
# Morning peak (06-08h) + evening peak (18-22h), flat baseline otherwise.
_DEFAULT_DRAW_PROFILE = np.array(
    [
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.02,
        0.08,
        0.10,
        0.06,
        0.03,
        0.02,
        0.02,
        0.03,
        0.02,
        0.02,
        0.02,
        0.03,
        0.05,
        0.09,
        0.11,
        0.10,
        0.07,
        0.04,
        0.02,
    ]
)
_DEFAULT_DRAW_PROFILE = _DEFAULT_DRAW_PROFILE / _DEFAULT_DRAW_PROFILE.sum()


def _default_calibration() -> dict[str, Any]:
    return {
        "calibrated_at": None,
        "n_calibration_windows_ua_eff": 0,
        "UA_eff": None,
        "solar_gain_area": 0.0,
        "Q_base_el": 0.35,
        "Q_dhw_daily": 3.5,
        "UA_dhw": 15.0,
        "cop_formula": None,  # None → caller falls back to config cop_formula
    }


def _default_schedule() -> dict[str, Any]:
    return {
        "T_dhw_upper": 55.0,
        "T_legionella": 60.0,
        "legionella_dow": 2,
        "legionella_hour": 14,
        "T_dhw_lower": 45.0,
        "dhw_tank_volume_l": 200,
        "committed_override": None,
        "override_history": [],
    }


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    os.replace(tmp_path, path)


def _read_json_or_default(path: Path, default: dict) -> dict:
    if not path.exists():
        return dict(default)
    try:
        with open(path) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("root is not an object")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as e:
        _LOGGER.warning(f"Failed to read {path.name}: {e} — using defaults")
        return dict(default)


class ThermalPhysicsModel:
    """Calibrated physics baseline for hourly household electricity consumption."""

    _OVERRIDE_TAIL_THRESHOLD_C = 0.1  # R2-#3: shared cutoff, both call sites use it identically

    def __init__(self, model_dir: Path, config: dict) -> None:
        self._model_dir = model_dir
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._calibration_path = model_dir / "physics_calibration.json"
        self._schedule_path = model_dir / "physics_schedule.json"
        self._config = config

        calib_defaults = _default_calibration()
        self._calib = {**calib_defaults, **_read_json_or_default(self._calibration_path, calib_defaults)}
        schedule_defaults = _default_schedule()
        self._schedule = {**schedule_defaults, **_read_json_or_default(self._schedule_path, schedule_defaults)}

        self._tau_hours: float | None = None  # set externally by Plan B from EnergyForecastModel._tau_hours

    @property
    def schedule(self) -> dict:
        return self._schedule

    def _initial_t_tank_for_window(self, dhw_df: pd.DataFrame | None, timestamps: pd.DatetimeIndex) -> float:
        """Shared initial-tank-state estimate for a training/forecast window — extracted
        from predict_training_series to avoid duplicating this logic at the new
        override-delta call site in model.py."""
        if dhw_df is not None and not dhw_df.empty:
            d = dhw_df.set_index(pd.to_datetime(dhw_df["timestamp"]))["buffer_temp"].reindex(
                timestamps, method="nearest"
            )
            return float(d.iloc[0]) if not d.empty and pd.notna(d.iloc[0]) else self._schedule["T_dhw_upper"]
        return (self._schedule["T_dhw_upper"] + self._schedule["T_dhw_lower"]) / 2

    @property
    def calibration_stale(self) -> bool:
        raw = self._calib.get("calibrated_at")
        if not raw:
            return True
        try:
            calibrated_at = pd.Timestamp(raw)
        except (ValueError, TypeError):
            return True
        age_days = (pd.Timestamp.now() - calibrated_at.tz_localize(None)).total_seconds() / 86400
        return age_days > STALE_AFTER_DAYS

    @property
    def is_cold_start_gated(self) -> bool:
        return self._calib.get("n_calibration_windows_ua_eff", 0) < COLD_START_MIN_WINDOWS

    def _t_flow_c(self, t_outdoor_c: float, live_shift_k: float | None) -> float:
        points = self._config.get("heating_curve_points") or []
        if not points:
            return DEFAULT_T_FLOW_C
        shift = live_shift_k if live_shift_k is not None else 0.0
        xs = [p[0] for p in points]
        ys = [p[1] + shift for p in points]
        return float(np.interp(t_outdoor_c, xs, ys))

    def _cop_formula_value(self, t_outdoor_c: float, live_shift_k: float | None) -> float:
        formula = self._calib.get("cop_formula") or self._config.get("cop_formula", {"a": 2.5, "b": 0.07})
        a, b = formula["a"], formula["b"]
        t_flow_k = self._t_flow_c(t_outdoor_c, live_shift_k) + 273.15
        t_outdoor_k = t_outdoor_c + 273.15
        denom = t_flow_k - t_outdoor_k
        carnot = ETA_CARNOT * t_flow_k / denom if denom > 0 else COP_MIN
        linear = a + b * t_outdoor_c
        return max(COP_MIN, min(carnot, linear))

    def _cop_series(
        self,
        timestamps: pd.DatetimeIndex,
        t_outdoor: pd.Series,
        cop_sensor_series: pd.Series | None,
        live_shift_series: pd.Series | None = None,
    ) -> pd.Series:
        formula_vals = np.array(
            [
                self._cop_formula_value(
                    t_o, None if live_shift_series is None else live_shift_series.reindex(timestamps).iloc[i]
                )
                for i, t_o in enumerate(t_outdoor.reindex(timestamps).values)
            ]
        )
        result = pd.Series(formula_vals, index=timestamps)
        if cop_sensor_series is not None:
            aligned = cop_sensor_series.reindex(timestamps)
            result = aligned.combine_first(result)
        return result.clip(lower=COP_MIN)

    def _space_heating_kwh(
        self,
        t_indoor: pd.Series,
        t_outdoor: pd.Series,
        ghi: pd.Series,
        cop: pd.Series,
    ) -> pd.Series:
        ua = self._calib.get("UA_eff")
        if ua is None:
            return pd.Series(0.0, index=t_indoor.index)

        solar_area = self._calib.get("solar_gain_area") or 0.0
        q_base_el = self._calib.get("Q_base_el") or 0.0
        gains_fraction = self._config["internal_gains_fraction"]

        q_loss = ua * (t_indoor - t_outdoor).clip(lower=0.0)
        q_solar = solar_area * ghi.fillna(0.0)
        q_gain_int = q_base_el * gains_fraction * 1000.0

        tau = self._tau_hours or 8.0
        c_building = self._config.get("c_building_wh_k") or (ua * tau)
        t_indoor_next = t_indoor.shift(-1).fillna(t_indoor)
        q_mass = c_building * (t_indoor_next - t_indoor)

        q_heat = (q_loss - q_solar - q_gain_int + q_mass).clip(lower=0.0)
        q_heat_el = q_heat / cop.clip(lower=COP_MIN) / 1000.0
        return q_heat_el

    def _dhw_override_for_hour(self, ts: pd.Timestamp, override: dict | None) -> float | None:
        """Return an override target temp for *ts* if a committed override (legionella
        or comfort_boost) applies, else None. Same-hour precedence: legionella wins —
        defense in depth for the data model; Goal 5 (EM-side) makes two genuinely
        concurrent same-hour events physically unreachable in practice."""
        if not override:
            return None
        legionella_val = override.get("legionella")
        if legionella_val is not None:
            date_str, hour = legionella_val
            if ts == pd.Timestamp(f"{date_str} {int(hour):02d}:00"):
                return self._schedule["T_legionella"]
        comfort_boost_val = override.get("comfort_boost")
        if comfort_boost_val is not None:
            date_str, hour, target_c = comfort_boost_val
            if ts == pd.Timestamp(f"{date_str} {int(hour):02d}:00"):
                return float(target_c)
        return None

    def _dhw_override_for_hour_from_history(self, ts: pd.Timestamp, history: list[dict]) -> float | None:
        """Training-reconstruction lookup mode (Goal 2): scans override_history for a
        non-cancelled entry matching (kind, date, hour) == ts. Same-hour precedence:
        legionella wins (mirrors _dhw_override_for_hour). Malformed entries are skipped
        with a WARNING, never raised — read-time validation mirrors commit_dhw_schedule's
        write-time contract (R1-#20)."""
        legionella_target: float | None = None
        comfort_boost_target: float | None = None
        for entry in history:
            try:
                if entry.get("cancelled_at") is not None:
                    continue
                kind = entry["kind"]
                date_str = entry["date"]
                hour = entry["hour"]
                target_c = entry["target_c"]
                if kind not in _VALID_OVERRIDE_KINDS or target_c is None:
                    _LOGGER.warning(f"override_history: malformed entry skipped: {entry}")
                    continue
                if ts != pd.Timestamp(f"{date_str} {int(hour):02d}:00"):
                    continue
                if kind == "legionella":
                    legionella_target = float(target_c)
                else:
                    comfort_boost_target = float(target_c)
            except Exception as e:  # noqa: BLE001 — read-time validation contract, R1-#20
                _LOGGER.warning(f"override_history: malformed entry skipped: {entry} ({e})")
                continue
        return legionella_target if legionella_target is not None else comfort_boost_target

    def _dhw_kwh_series(
        self,
        timestamps: pd.DatetimeIndex,
        t_ambient: pd.Series,
        initial_t_tank: float,
        override_lookup: Callable[[pd.Timestamp], float | None] | None = None,
    ) -> tuple[pd.Series, pd.Series, float]:
        """Returns (el_kwh_series, t_tank_series, final_t_tank). override_lookup is a
        per-hour callable returning an override target temp or None; None means
        override-blind (the mandatory mode for physics_kwh as an ML feature — Goal 1).
        Same ODE math as before Phase A — only the override-source abstraction changed,
        from a dict to a callable, so one loop can serve live-dict, history-based, and
        blind lookup modes without duplicating it (R2-#3's "one shared function")."""
        volume_l = self._config["dhw_tank_volume_l"]
        c_dhw = volume_l * WATER_SPECIFIC_HEAT_WH_PER_L_K
        if c_dhw <= 0:
            _LOGGER.warning("DHW tank volume is zero or invalid — skipping DHW component")
            zeros = pd.Series(0.0, index=timestamps)
            return zeros, pd.Series(float(initial_t_tank), index=timestamps), float(initial_t_tank)
        q_dhw_power = self._config["dhw_power_w"]
        heating_rise = q_dhw_power / c_dhw  # K/h, derived each call — not a stored constant

        t_lower = self._schedule["T_dhw_lower"]
        t_legionella = self._schedule["T_legionella"]

        q_dhw_daily = self._calib.get("Q_dhw_daily") or 0.0
        draw_rate = (q_dhw_daily * 1000.0 / c_dhw) if c_dhw > 0 else 0.0  # K-equivalent/day, scaled by shape below

        cop_dhw = max(COP_MIN, self._cop_formula_value(t_ambient.iloc[0] if len(t_ambient) else 10.0, None))

        t_tank = float(initial_t_tank)
        el_kwh = np.zeros(len(timestamps))
        t_tank_trajectory = np.zeros(len(timestamps))
        for i, ts in enumerate(timestamps):
            ua_dhw = self._calib.get("UA_dhw") or 15.0
            dT = -ua_dhw * (t_tank - float(t_ambient.iloc[i])) / c_dhw
            hour_of_day = ts.hour
            dT -= _DEFAULT_DRAW_PROFILE[hour_of_day] * draw_rate

            override_target = override_lookup(ts) if override_lookup is not None else None
            if override_target is not None:
                q_el_w = q_dhw_power / cop_dhw
                el_kwh[i] = q_el_w / 1000.0
                t_tank = override_target
                t_tank_trajectory[i] = t_tank
                continue

            if t_tank <= t_lower:
                q_el_w = q_dhw_power / cop_dhw
                el_kwh[i] = q_el_w / 1000.0
                t_tank = float(np.clip(t_tank + dT + heating_rise, t_lower, t_legionella))
            else:
                el_kwh[i] = 0.0
                t_tank = float(np.clip(t_tank + dT, t_lower, t_legionella))
            t_tank_trajectory[i] = t_tank

        return pd.Series(el_kwh, index=timestamps), pd.Series(t_tank_trajectory, index=timestamps), t_tank

    def _compute_override_delta_series(
        self,
        timestamps: pd.DatetimeIndex,
        t_ambient: pd.Series,
        initial_t_tank: float,
        override_lookup: Callable[[pd.Timestamp], float | None],
        baseline_kwh: pd.Series,
        baseline_t_tank: pd.Series,
    ) -> pd.Series:
        """The one shared joint trajectory-diff-and-cutoff function (R2-#3) — called
        identically by both compute_training_override_delta and
        compute_serving_override_delta so the two never resolve an equivalent override
        to different tail lengths. Computes ONE with-overrides trajectory (applying
        every override the given lookup can see, in chronological order — R2-#1, not a
        per-override sum), diffs it against the already-computed baseline, terminates
        the tail at the first hour the two trajectories reconverge within 0.1°C, and
        clips to the sanity bound (R1-#12/R2-#2)."""
        with_overrides_kwh, with_overrides_t_tank, _ = self._dhw_kwh_series(
            timestamps, t_ambient, initial_t_tank, override_lookup=override_lookup
        )
        raw_delta = with_overrides_kwh - baseline_kwh
        tank_diff = (with_overrides_t_tank - baseline_t_tank).abs()

        diverged = tank_diff >= self._OVERRIDE_TAIL_THRESHOLD_C
        if diverged.any():
            diverged_start = diverged.to_numpy().argmax()
            reconverged = tank_diff.iloc[diverged_start:] < self._OVERRIDE_TAIL_THRESHOLD_C
            if reconverged.any():
                tail_end = diverged_start + reconverged.to_numpy().argmax()
                raw_delta.iloc[tail_end:] = 0.0
        else:
            raw_delta[:] = 0.0  # no override ever diverged the trajectory in this window

        bound = pd.concat([with_overrides_kwh, baseline_kwh], axis=1).max(axis=1)
        violated = raw_delta.abs() > bound
        if violated.any():
            _LOGGER.warning(f"override_delta_series exceeded sanity bound on {int(violated.sum())} hour(s) — clipping")
        return raw_delta.clip(lower=-bound, upper=bound)

    def compute_training_override_delta(
        self,
        timestamps: pd.DatetimeIndex,
        t_ambient: pd.Series,
        initial_t_tank: float,
        override_history: list[dict],
    ) -> pd.Series:
        """Training-side entry point (Goal 2): replays override_history over the full
        training window, including each override's full multi-hour carryover tail."""
        baseline_kwh, baseline_t_tank, _ = self._dhw_kwh_series(
            timestamps, t_ambient, initial_t_tank, override_lookup=None
        )
        if not override_history:
            return pd.Series(0.0, index=timestamps)

        def lookup(ts: pd.Timestamp) -> float | None:
            return self._dhw_override_for_hour_from_history(ts, override_history)

        return self._compute_override_delta_series(
            timestamps, t_ambient, initial_t_tank, lookup, baseline_kwh, baseline_t_tank
        )

    def compute_serving_override_delta(
        self,
        timestamps: pd.DatetimeIndex,
        t_ambient: pd.Series,
        initial_t_tank: float,
    ) -> pd.Series:
        """Serving-side entry point (Goal 1): the deterministic post-model forecast
        correction. Callers add this UNCONDITIONALLY to the model's already-final
        published forecast — never fed back into physics_kwh or the trained model."""
        baseline_kwh, baseline_t_tank, _ = self._dhw_kwh_series(
            timestamps, t_ambient, initial_t_tank, override_lookup=None
        )
        committed = self._schedule.get("committed_override")
        if not committed:
            return pd.Series(0.0, index=timestamps)

        def lookup(ts: pd.Timestamp) -> float | None:
            return self._dhw_override_for_hour(ts, committed)

        return self._compute_override_delta_series(
            timestamps, t_ambient, initial_t_tank, lookup, baseline_kwh, baseline_t_tank
        )

    def _area_weighted_t_indoor(
        self, climate_data: dict[str, pd.DataFrame], timestamps: pd.DatetimeIndex, room_areas: dict[str, float] | None
    ) -> pd.Series | None:
        if not climate_data:
            return None
        parts, weights = [], []
        for eid, cdf in climate_data.items():
            if cdf.empty:
                continue
            c = cdf.set_index("timestamp")["current_temp"].reindex(timestamps, method="nearest")
            parts.append(c)
            weights.append((room_areas or {}).get(eid, 20.0))
        if not parts:
            return None
        stacked = pd.concat(parts, axis=1)
        w = np.array(weights)
        return pd.Series(np.average(stacked.values, axis=1, weights=w), index=timestamps)

    def predict_series(
        self,
        forecast_df: pd.DataFrame,
        climate_recent: dict[str, pd.DataFrame] | None = None,
        dhw_recent: pd.DataFrame | None = None,
        room_areas: dict[str, float] | None = None,
        heating_active_series: pd.Series | None = None,
        setpoint_on: float | None = None,
        setpoint_off: float | None = None,
        dhw_schedule_override: dict | None = None,
    ) -> pd.Series:
        from .model import _project_indoor_temps  # local import avoids a circular import at module load time

        timestamps = pd.DatetimeIndex(forecast_df["timestamp"])
        t_outdoor = pd.Series(forecast_df["temp_c"].values, index=timestamps)
        ghi = (
            pd.Series(forecast_df["direct_radiation_wm2"].values, index=timestamps)
            if "direct_radiation_wm2" in forecast_df.columns
            else pd.Series(0.0, index=timestamps)
        )

        try:
            if climate_recent:
                projected = _project_indoor_temps(
                    climate_recent,
                    timestamps,
                    t_outdoor,
                    tau_hours=self._tau_hours,
                    heating_active_series=heating_active_series,
                    setpoint_on=setpoint_on,
                    setpoint_off=setpoint_off,
                )
                t_indoor = self._area_weighted_t_indoor(projected, timestamps, room_areas)
            else:
                t_indoor = None

            if t_indoor is None:
                t_indoor = pd.Series(setpoint_on or DEFAULT_AMBIENT_C, index=timestamps)

            cop = self._cop_series(timestamps, t_outdoor, cop_sensor_series=None)
            q_heat_el = self._space_heating_kwh(t_indoor, t_outdoor, ghi, cop)

            if dhw_recent is not None and not dhw_recent.empty:
                latest = dhw_recent.sort_values("timestamp").iloc[-1]
                age = timestamps[0] - pd.Timestamp(latest["timestamp"])
                initial_t_tank = (
                    float(latest["buffer_temp"])
                    if age <= pd.Timedelta(hours=2)
                    else (self._schedule["T_dhw_upper"] + self._schedule["T_dhw_lower"]) / 2
                )
            else:
                initial_t_tank = (self._schedule["T_dhw_upper"] + self._schedule["T_dhw_lower"]) / 2

            override_lookup = (
                (lambda ts: self._dhw_override_for_hour(ts, dhw_schedule_override)) if dhw_schedule_override else None
            )
            q_dhw_el, _dhw_t_tank_series, _ = self._dhw_kwh_series(
                timestamps, t_indoor, initial_t_tank, override_lookup
            )

            q_base_el = self._calib.get("Q_base_el") or 0.35
            physics_kwh = q_heat_el + q_dhw_el + q_base_el
            return physics_kwh.clip(lower=0.0)
        except Exception as e:
            _LOGGER.warning(f"physics predict_series failed: {e} — returning zeros")
            return pd.Series(0.0, index=timestamps)

    def predict_training_series(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        climate_dfs: dict[str, pd.DataFrame] | None = None,
        dhw_df: pd.DataFrame | None = None,
        room_areas: dict[str, float] | None = None,
    ) -> pd.Series:
        timestamps = pd.DatetimeIndex(pd.to_datetime(energy_df["timestamp"]))
        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        t_outdoor = w["temp_c"].reindex(timestamps, method="nearest")
        ghi = (
            w["direct_radiation_wm2"].reindex(timestamps, method="nearest")
            if "direct_radiation_wm2" in w.columns
            else pd.Series(0.0, index=timestamps)
        )

        t_indoor = self._area_weighted_t_indoor(climate_dfs or {}, timestamps, room_areas)
        if t_indoor is None:
            t_indoor = pd.Series(DEFAULT_AMBIENT_C, index=timestamps)

        cop = self._cop_series(timestamps, t_outdoor, cop_sensor_series=None)
        q_heat_el = self._space_heating_kwh(t_indoor, t_outdoor, ghi, cop)

        initial_t_tank = self._initial_t_tank_for_window(dhw_df, timestamps)

        q_dhw_el, _dhw_t_tank_series, _ = self._dhw_kwh_series(
            timestamps, t_indoor, initial_t_tank, override_lookup=None
        )
        q_base_el = self._calib.get("Q_base_el") or 0.35
        return (q_heat_el + q_dhw_el + q_base_el).clip(lower=0.0)

    def _calibrate_base_load(
        self,
        energy_df: pd.DataFrame,
        away_df: pd.DataFrame | None,
        ev_df: pd.DataFrame | None,
        holdout_cutoff: pd.Timestamp,
    ) -> float | None:
        df = energy_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"] < holdout_cutoff]
        df = df[(df["timestamp"].dt.hour >= 1) & (df["timestamp"].dt.hour < 5)]
        df = df[df["timestamp"].dt.month.isin([6, 7, 8])]

        if away_df is not None and not away_df.empty:
            away_ts = set(pd.to_datetime(away_df.loc[away_df["is_away"] > 0, "timestamp"]).dt.floor("1h"))
            df = df[~df["timestamp"].dt.floor("1h").isin(away_ts)]
        if ev_df is not None and not ev_df.empty:
            ev_ts = set(pd.to_datetime(ev_df["timestamp"]).dt.floor("1h"))
            df = df[~df["timestamp"].dt.floor("1h").isin(ev_ts)]

        # Drop rows with NaN gross_kwh before counting usable data
        df = df.dropna(subset=["gross_kwh"])

        n_nights = df["timestamp"].dt.date.nunique()
        if n_nights < 14 or len(df) < 14:
            _LOGGER.warning(f"Q_base_el calibration: only {n_nights} summer nights available (need 14) — skipping")
            return None
        result = float(df["gross_kwh"].median())
        if not np.isfinite(result):
            _LOGGER.warning("Q_base_el calibration: median is NaN/inf — skipping")
            return None
        return result

    def _calibrate_dhw_daily(
        self, energy_df: pd.DataFrame, q_base_el: float, holdout_cutoff: pd.Timestamp
    ) -> float | None:
        df = energy_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df[df["timestamp"] < holdout_cutoff]
        df = df[df["timestamp"].dt.month.isin([6, 7, 8])]
        if df.empty:
            _LOGGER.warning("Q_dhw_daily calibration: no summer data available — skipping")
            return None

        daily = df.set_index("timestamp")["gross_kwh"].resample("1D").sum().dropna()
        if len(daily) < 14:
            _LOGGER.warning(f"Q_dhw_daily calibration: only {len(daily)} summer days available (need 14) — skipping")
            return None

        result = float(daily.mean()) - 24 * q_base_el
        return max(0.0, result)

    def _calibrate_ua_eff(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        climate_dfs: dict[str, pd.DataFrame] | None,
        dhw_df: pd.DataFrame | None,
        holdout_cutoff: pd.Timestamp,
    ) -> tuple[float | None, int]:
        from .model import _find_passive_windows

        if not climate_dfs:
            return None, 0

        e = energy_df.copy()
        e["timestamp"] = pd.to_datetime(e["timestamp"])
        e = e[e["timestamp"] < holdout_cutoff]
        e = e[(e["timestamp"].dt.hour >= 22) | (e["timestamp"].dt.hour < 6)]
        e = e[e["timestamp"].dt.month.isin([11, 12, 1, 2, 3])]
        if e.empty:
            return None, 0

        t_indoor = self._area_weighted_t_indoor(climate_dfs, pd.DatetimeIndex(e["timestamp"]), room_areas=None)
        if t_indoor is None:
            # climate_dfs was non-empty but every inner DataFrame was empty — no usable indoor
            # readings. Bail out here (rather than passing None through) so later `.iloc` calls
            # on t_indoor can't raise; matches the "never raise, always return None" contract.
            return None, 0
        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        t_outdoor = w["temp_c"].reindex(e["timestamp"], method="nearest").values

        has_dhw_sensor = dhw_df is not None and not dhw_df.empty
        if has_dhw_sensor:
            d = dhw_df.set_index(pd.to_datetime(dhw_df["timestamp"]))["buffer_temp"].reindex(
                e["timestamp"], method="nearest"
            )
            dhw_tank_temp = d.values
            min_delta_t = 8.0
        else:
            dhw_tank_temp = np.full(len(e), np.nan)
            min_delta_t = 12.0
            _LOGGER.warning("DHW tank sensor absent — UA_eff calibration may be inflated")

        passive_df = pd.DataFrame(
            {
                "timestamp": e["timestamp"].values,
                "T_outdoor": t_outdoor,
                "T_indoor": t_indoor.values if t_indoor is not None else np.nan,
                "hp_running": False,  # already filtered to heating-off... actually this is nighttime window,
                # not HP-off; UA_eff wants the observed heating demand itself, so hp_running=False here
                # would incorrectly zero out min_hp_off_hours=2 requirement. Set min_hp_off_hours=0 below.
                "dhw_tank_temp": dhw_tank_temp,
            }
        )
        passive_idx = _find_passive_windows(passive_df, min_delta_t=min_delta_t, min_hp_off_hours=0)
        n_windows = len(passive_idx)
        if n_windows < 30:
            return None, n_windows

        sub = e.iloc[passive_idx]
        sub_t_indoor = t_indoor.iloc[passive_idx].values
        sub_t_outdoor = t_outdoor[passive_idx]
        delta_t = sub_t_indoor - sub_t_outdoor

        cop = np.array([self._cop_formula_value(t, None) for t in sub_t_outdoor])
        # gross_kwh includes standing base load (Q_base_el) — subtract it so the OLS models
        # Q_heat_obs (space-heating only), matching spec §4.2's "Q_heat_obs ≈ UA_eff/1000 × ΔT/COP".
        # Without this, the constant base-load offset biases the through-origin fit and — for
        # tight-variance ΔT/COP windows (e.g. stable winter nights) — can drive R² negative,
        # discarding an otherwise-accurate calibration. Mirrors the same subtraction already
        # done in _calibrate_dhw_daily().
        q_base_el = self._calib.get("Q_base_el") or 0.0
        q_heat_obs = sub["gross_kwh"].values - q_base_el

        x = (delta_t / cop).reshape(-1, 1)
        y = q_heat_obs
        # OLS through origin (Q_heat_obs ≈ UA_eff/1000 × ΔT/COP): slope = sum(xy)/sum(xx)
        slope = float(np.sum(x.flatten() * y) / np.sum(x.flatten() ** 2))
        y_hat = slope * x.flatten()
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0

        if r2 < 0.5:
            _LOGGER.warning(f"UA_eff calibration: R²={r2:.2f} < 0.5 — discarding result, using config default")
            return None, n_windows

        ua_eff = slope * 1000.0
        return ua_eff, n_windows

    def _calibrate_solar_gain_area(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        climate_dfs: dict[str, pd.DataFrame] | None,
        ua_eff: float | None,
        holdout_cutoff: pd.Timestamp,
    ) -> float | None:
        if ua_eff is None or not climate_dfs:
            return None

        e = energy_df.copy()
        e["timestamp"] = pd.to_datetime(e["timestamp"])
        e = e[e["timestamp"] < holdout_cutoff]
        e = e[e["timestamp"].dt.month.isin([11, 12, 1, 2])]

        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        ghi = w["direct_radiation_wm2"].reindex(e["timestamp"], method="nearest")
        e = e[ghi.values > 50]
        if e.empty:
            return None

        n_days = e["timestamp"].dt.date.nunique()
        if n_days < 14:
            _LOGGER.warning(f"solar_gain_area calibration: only {n_days} sunny winter days (need 14) — skipping")
            return None

        t_indoor = self._area_weighted_t_indoor(climate_dfs, pd.DatetimeIndex(e["timestamp"]), room_areas=None)
        t_outdoor = w["temp_c"].reindex(e["timestamp"], method="nearest").values
        ghi_sub = w["direct_radiation_wm2"].reindex(e["timestamp"], method="nearest").values
        cop = np.array([self._cop_formula_value(t, None) for t in t_outdoor])

        q_loss_w = ua_eff * np.clip((t_indoor.values if t_indoor is not None else 20.0) - t_outdoor, 0, None)
        # gross_kwh includes standing base load (Q_base_el) — subtract it so q_heat_el_observed*cop*1000
        # isolates space-heating-delivered watts only, mirroring the identical subtraction in
        # _calibrate_ua_eff() and _calibrate_dhw_daily(). Verified numerically against this method's own
        # test fixture (test_recovers_known_ghi_offset, true_area=8.0, Q_base_el=0.35): the regression
        # slope comes out to 4.63 (42% low, outside the test's rel=0.3 tolerance) when gross_kwh is used
        # directly, vs 8.05 (0.6% error) once Q_base_el is subtracted — the same class of bias Task 8
        # found in _calibrate_ua_eff.
        q_base_el = self._calib.get("Q_base_el") or 0.0
        q_heat_el_observed = e["gross_kwh"].values - q_base_el
        # residual = predicted UA-loss minus the heat actually delivered (reconstructed from observed
        # electrical draw x COP). Physically, delivered heat = q_loss - q_solar (before any demand floor
        # clipping), so residual = q_loss_w - (q_loss - q_solar) = q_solar = solar_gain_area * ghi — i.e.
        # residual is POSITIVE and directly proportional to ghi with slope = solar_gain_area.
        #
        # NOTE: the brief's literal code set `y = -residual_w`, inverting this sign. That silently
        # flipped every recovered slope negative, which the trailing `max(0.0, slope)` then clamped to
        # 0.0 on every call — i.e. solar_gain_area calibration always returned exactly 0 regardless of
        # any real solar effect. Confirmed numerically: literal code (with the Q_base_el fix applied)
        # gives slope=-8.05 -> clamped to 0.0 (test asserts ~8.0, fails); removing the negation
        # (y = residual_w, as below) gives slope=+8.05, matching the fixture's true_area=8.0.
        residual_w = q_loss_w - q_heat_el_observed * cop * 1000.0
        x = ghi_sub
        y = residual_w
        if np.sum(x**2) < 1e-9:
            return None
        slope = float(np.sum(x * y) / np.sum(x**2))
        return max(0.0, slope)

    def _calibrate_ua_dhw(
        self,
        dhw_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        heating_active_df: pd.DataFrame,
        holdout_cutoff: pd.Timestamp,
    ) -> float | None:
        if dhw_df is None or dhw_df.empty:
            return None

        d = dhw_df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d[d["timestamp"] < holdout_cutoff].sort_values("timestamp").reset_index(drop=True)

        ha = heating_active_df.copy() if heating_active_df is not None else pd.DataFrame()
        if not ha.empty:
            ha["timestamp"] = pd.to_datetime(ha["timestamp"])
            hp_off = (
                ha.set_index("timestamp")["heating_active"].reindex(d["timestamp"], method="nearest").fillna(0) == 0
            )
        else:
            hp_off = pd.Series(True, index=d.index)

        d["hp_running"] = ~hp_off.values
        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        t_ambient = w["temp_c"].reindex(d["timestamp"], method="nearest").fillna(DEFAULT_AMBIENT_C).values

        # passive decay = tank cooling with no reheat and no rising trend (a draw would also make it fall,
        # so restrict to windows where the temp decline matches a smooth exponential, i.e. no big single-step drops)
        #
        # NOTE: the brief's literal bound was (-1.0, 0.0). That's tighter than this module's own default
        # UA_dhw=15.0 (see _default_calibration()) actually produces: at deltaT=35K (55C tank / 20C
        # ambient) and the default 200L tank, dT/dt = -15*35/(200*1.163) = -2.26 K/h, which falls outside
        # (-1.0, 0.0) and would be filtered out entirely — confirmed via this method's own test fixture
        # (test_passive_decay_regression_recovers_ua_dhw), where every one of the 11 available decay
        # samples (diffs ranging -1.16 to -2.26 K/h) was excluded, leaving 0 rows and forcing an early
        # None return. Widened to (-5.0, 0.0): comfortably covers realistic passive-decay rates (up to
        # ~5 K/h even for a hotter/larger tank or higher UA) while still well below typical single-hour
        # draw-event drops (multiple K from cold-water mixing), so it still discriminates decay from draws.
        d["is_falling_smooth"] = d["buffer_temp"].diff().between(-5.0, 0.0)
        mask = (~d["hp_running"]) & d["is_falling_smooth"].fillna(False)
        sub = d[mask]
        if len(sub) < 6:
            return None

        delta_t = sub["buffer_temp"].values - t_ambient[sub.index]
        dT_dt = -sub["buffer_temp"].diff().fillna(0).values  # K lost per hour, positive
        c_dhw = self._config["dhw_tank_volume_l"] * WATER_SPECIFIC_HEAT_WH_PER_L_K
        # dT/dt = -UA_dhw * deltaT / C_dhw  ->  UA_dhw = (dT/dt * C_dhw) / deltaT
        valid = delta_t > 0.5
        if valid.sum() < 6:
            return None
        ua_dhw_samples = (dT_dt[valid] * c_dhw) / delta_t[valid]
        ua_dhw_samples = ua_dhw_samples[(ua_dhw_samples > 0) & (ua_dhw_samples < 200)]
        if len(ua_dhw_samples) < 3:
            return None
        return float(np.median(ua_dhw_samples))

    def _infer_dhw_schedule(self, dhw_df: pd.DataFrame) -> dict | None:
        if dhw_df is None or dhw_df.empty or len(dhw_df) < 7 * 24:
            _LOGGER.warning("DHW schedule inference: insufficient history (need ≥7 days) — skipping")
            return None

        d = dhw_df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        d = d.sort_values("timestamp").reset_index(drop=True)

        # local peaks: value higher than both neighbours
        is_peak = (d["buffer_temp"] > d["buffer_temp"].shift(1)) & (d["buffer_temp"] > d["buffer_temp"].shift(-1))
        peaks = d[is_peak.fillna(False)]
        if peaks.empty:
            _LOGGER.warning("DHW schedule inference: no local peaks found in temperature history — skipping")
            return None

        t_dhw_upper = float(peaks["buffer_temp"].quantile(0.90))
        legionella_peaks = peaks[peaks["buffer_temp"] > t_dhw_upper + 3.0]
        if legionella_peaks.empty:
            _LOGGER.warning(
                "DHW schedule inference: no legionella boost peak found — no peak exceeds threshold — skipping"
            )
            return None

        t_legionella = float(legionella_peaks["buffer_temp"].max())
        legionella_dow = int(legionella_peaks["timestamp"].dt.dayofweek.mode().iloc[0])
        legionella_hour = int(legionella_peaks["timestamp"].dt.hour.mode().iloc[0])

        # cycle-start local minima -> T_dhw_lower
        is_trough = (d["buffer_temp"] < d["buffer_temp"].shift(1)) & (d["buffer_temp"] < d["buffer_temp"].shift(-1))
        troughs = d[is_trough.fillna(False) & (d["buffer_temp"] < t_dhw_upper)]
        t_dhw_lower = float(troughs["buffer_temp"].quantile(0.5)) if not troughs.empty else t_dhw_upper - 10.0

        return {
            "T_dhw_upper": t_dhw_upper,
            "T_legionella": t_legionella,
            "legionella_dow": legionella_dow,
            "legionella_hour": legionella_hour,
            "T_dhw_lower": t_dhw_lower,
            "dhw_tank_volume_l": self._schedule["dhw_tank_volume_l"],
        }

    def _check_legionella_stability(self, new_dow: int, new_hour: int) -> bool:
        old_dow = self._schedule.get("legionella_dow")
        old_hour = self._schedule.get("legionella_hour")
        if old_dow is None or old_hour is None:
            return True
        if new_dow != old_dow:
            _LOGGER.warning(
                f"Legionella timing day-of-week shifted ({old_dow} -> {new_dow}) — suspending autonomous learning"
            )
            return False
        if abs(new_hour - old_hour) > 2:
            _LOGGER.warning(
                f"Legionella timing shifted {abs(new_hour - old_hour)}h week-over-week — suspending autonomous learning"
            )
            return False
        return True

    def check_zone_boundary(self, current_thermostat_entities: list[str]) -> None:
        recorded = self._calib.get("room_thermostats_at_calibration")
        if not recorded:
            return
        if sorted(recorded) != sorted(current_thermostat_entities):
            _LOGGER.warning(
                "Zone boundary changed: room_thermostats at calibration time %s differs from current %s — "
                "UA_eff may no longer be consistent with Q_loss until recalibration",
                recorded,
                current_thermostat_entities,
            )

    def detect_open_windows(
        self,
        climate_dfs: dict[str, pd.DataFrame],
        weather_df: pd.DataFrame,
        room_areas: dict[str, float] | None,
    ) -> pd.Series:
        """Detect single-hour temperature anomalies consistent with instantaneous window openings.

        Uses a one-step-ahead ODE projection (re-anchored every hour to the previous actual reading,
        not open-loop) to compute each hour's expected indoor temperature, then flags hours where
        the residual (actual - expected) deviates significantly from the median by more than 2× MAD.

        LIMITATION: This is a single-hour-shock detector. It does NOT reliably detect gradual or
        slow multi-hour drift events (e.g., a window left slightly ajar for several hours). Each
        hour's residual is computed relative to the previous hour's *actual* reading, so a slow
        drift "forgives" itself each step — the residual never accumulates into a single-hour
        outlier large enough to clear the median/MAD threshold. Future consumers (e.g., Plan B's
        training-sample weighting via `open_window_hour` flags) should be aware of this distinction
        if they need to handle slower anomalous periods differently.

        Args:
            climate_dfs: Dict mapping room entity IDs to DataFrames with "timestamp" and "current_temp" columns.
            weather_df: DataFrame with "timestamp" and "temp_c" columns (outdoor temperature).
            room_areas: Optional dict mapping room entity IDs to area weights; defaults to 20.0 per room.

        Returns:
            Boolean Series indexed by timestamps, True for hours flagged as anomalous.
        """
        from .model import _find_passive_windows

        if not climate_dfs:
            return pd.Series(dtype=bool)

        timestamps = pd.DatetimeIndex(next(iter(climate_dfs.values()))["timestamp"])
        t_actual = self._area_weighted_t_indoor(climate_dfs, timestamps, room_areas)
        if t_actual is None:
            return pd.Series(False, index=timestamps)

        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        t_outdoor = w["temp_c"].reindex(timestamps, method="nearest")

        # One-step-ahead ODE projection: each hour's expected temp is derived from the
        # PREVIOUS hour's actual reading, not compounded forward from the first reading.
        #
        # NOTE: the brief's literal code anchored once at t_actual[0] and simulated the
        # whole window open-loop (t_ode[i] built from t_ode[i-1]). That diverges hard from
        # reality whenever the room is being actively heated (which this ODE has no notion
        # of) — e.g. in test_open_window_flags_large_residual, actual indoor holds near 20C
        # while outdoor sits at 0C; the open-loop projection with tau=8 free-decays toward
        # 0C (17.5 -> 15.3 -> 13.4 -> ... by hour 3), so residual = actual - projection GROWS
        # every hour regardless of any anomaly, hitting 7.8-9.1C by hours 4-5 while the actual
        # open-window drop at hour 3 (-1.4C residual) is comparatively small — the growing
        # baseline drift masks the real event (confirmed: literal code leaves flags.iloc[3]
        # False). The same drift-driven bias blows up over 48h in
        # test_open_window_threshold_from_passive_windows_only (residual pinned near a
        # persistent multi-degree offset with tiny variance around it), so a plain
        # `residual.abs() > 2*std` flags 42/48 hours — confirmed via direct run of the
        # literal code. Re-anchoring every step at the previous actual reading keeps the
        # residual stationary (bounded drift-free noise around a small constant offset)
        # instead of an ever-growing open-loop error, so a real single-hour anomaly stands
        # out from the rest.
        tau = self._tau_hours or 8.0
        t_actual_arr = t_actual.values
        t_out_arr = t_outdoor.values
        t_ode = np.empty(len(timestamps))
        t_ode[0] = t_actual_arr[0]
        for i in range(1, len(timestamps)):
            t_ode[i] = t_actual_arr[i - 1] + (t_out_arr[i - 1] - t_actual_arr[i - 1]) / tau
        t_ode_series = pd.Series(t_ode, index=timestamps)

        residual = t_actual - t_ode_series

        passive_df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "T_outdoor": t_out_arr,
                "T_indoor": t_actual_arr,
                "hp_running": False,
                "dhw_tank_temp": np.nan,
            }
        )
        passive_idx = _find_passive_windows(passive_df, min_delta_t=0.0, min_hp_off_hours=1)
        sample = residual.iloc[passive_idx] if len(passive_idx) >= 5 else residual

        # Robust (median/MAD) spread rather than mean/std: with a single genuine
        # open-window event, that hour's residual IS the outlier a mean/std estimate is
        # not robust to — it inflates the mean and std enough to mask itself, especially
        # in small samples (confirmed: mean/std centering on this same one-step residual
        # still leaves flags.iloc[3] False in test_open_window_flags_large_residual).
        median = float(sample.median()) if len(sample) else 0.0
        mad = float((sample - median).abs().median()) if len(sample) else 0.0
        sigma = 1.4826 * mad

        if sigma <= 1e-9:
            return pd.Series(False, index=timestamps)

        return (residual - median).abs() > 2 * sigma

    def calibrate(
        self,
        energy_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        climate_dfs: dict[str, pd.DataFrame] | None,
        dhw_df: pd.DataFrame | None,
        holdout_cutoff: pd.Timestamp,
        heating_active_df: pd.DataFrame | None = None,
        ev_df: pd.DataFrame | None = None,
        away_df: pd.DataFrame | None = None,
    ) -> None:
        try:
            q_base_el = self._calibrate_base_load(energy_df, away_df, ev_df, holdout_cutoff)
            if q_base_el is not None:
                self._calib["Q_base_el"] = q_base_el

            q_dhw_daily = self._calibrate_dhw_daily(energy_df, self._calib["Q_base_el"], holdout_cutoff)
            if q_dhw_daily is not None:
                self._calib["Q_dhw_daily"] = q_dhw_daily

            ua_eff, n_windows = self._calibrate_ua_eff(energy_df, weather_df, climate_dfs, dhw_df, holdout_cutoff)
            self._calib["n_calibration_windows_ua_eff"] = n_windows
            if ua_eff is not None:
                self._calib["UA_eff"] = ua_eff
            else:
                _LOGGER.warning("UA_eff calibration unavailable — heating component will be skipped")

            solar_area = self._calibrate_solar_gain_area(
                energy_df, weather_df, climate_dfs, self._calib["UA_eff"], holdout_cutoff
            )
            if solar_area is not None:
                self._calib["solar_gain_area"] = solar_area

            ua_dhw = self._calibrate_ua_dhw(dhw_df, weather_df, heating_active_df, holdout_cutoff)
            if ua_dhw is not None:
                self._calib["UA_dhw"] = ua_dhw

            if climate_dfs:
                self._calib["room_thermostats_at_calibration"] = sorted(climate_dfs.keys())

            self._calib["calibrated_at"] = pd.Timestamp.now().isoformat()
            _atomic_write_json(self._calibration_path, self._calib)

            if dhw_df is not None and not dhw_df.empty:
                inferred = self._infer_dhw_schedule(dhw_df)
                if inferred is not None:
                    if self._check_legionella_stability(inferred["legionella_dow"], inferred["legionella_hour"]):
                        self._schedule.update(inferred)
                    else:
                        self._schedule.update({k: v for k, v in inferred.items() if not k.startswith("legionella")})
            _atomic_write_json(self._schedule_path, self._schedule)
        except Exception as e:
            _LOGGER.warning(f"Physics calibration failed: {e} — retaining previous/default parameters")

    def commit_dhw_schedule(self, override: dict) -> None:
        """Merge semantics into committed_override + append-only override_history (Goal 4).
        This is net-new code — the previous implementation was a flat replace. Never
        raises: a malformed entry logs a WARNING and is skipped, never corrupting the
        rest of the merge (R1 write-time validation contract)."""
        if self._schedule.get("committed_override") is None:
            self._schedule["committed_override"] = {}
        committed = self._schedule["committed_override"]
        history = self._schedule.setdefault("override_history", [])
        now_iso = dt.datetime.now(dt.UTC).isoformat()

        for kind, value in override.items():
            if value is None:
                prior = committed.pop(kind, None)
                if prior is not None:
                    self._mark_history_cancelled(history, kind, prior[0], prior[1])
                continue
            if kind not in _VALID_OVERRIDE_KINDS:
                _LOGGER.warning(f"commit_dhw_schedule: unknown kind {kind!r} — skipping")
                continue
            if kind == "legionella":
                if not (isinstance(value, list | tuple) and len(value) == 2):
                    _LOGGER.warning(f"commit_dhw_schedule: malformed legionella value {value!r} — skipping")
                    continue
                date_str, hour = value
                target_c = self._schedule["T_legionella"]
            else:  # comfort_boost
                if not (isinstance(value, list | tuple) and len(value) == 3):
                    _LOGGER.warning(f"commit_dhw_schedule: malformed comfort_boost value {value!r} — skipping")
                    continue
                date_str, hour, target_c = value
                if not isinstance(target_c, int | float):
                    _LOGGER.warning(
                        f"commit_dhw_schedule: comfort_boost target_c {target_c!r} is not numeric — skipping"
                    )
                    continue

            committed[kind] = [date_str, hour] if kind == "legionella" else [date_str, hour, target_c]
            self._replace_or_append_history(history, kind, date_str, hour, float(target_c), now_iso)

        if not committed:
            self._schedule["committed_override"] = None
        _atomic_write_json(self._schedule_path, self._schedule)
        _LOGGER.info(f"DHW schedule committed: {override}")

    @staticmethod
    def _mark_history_cancelled(history: list[dict], kind: str, date_str: str, hour: int) -> None:
        """Genuine remote cancellation (not self-expiry): mark the specific still-pending
        (kind, date, hour) entry cancelled rather than deleting it — keeps history
        append-only/auditable and correctly zero-deltas it in reconstruction (R2 fix)."""
        now_iso = dt.datetime.now(dt.UTC).isoformat()
        for entry in history:
            if (
                entry.get("cancelled_at") is None
                and entry.get("kind") == kind
                and entry.get("date") == date_str
                and entry.get("hour") == hour
            ):
                entry["cancelled_at"] = now_iso
                return

    @staticmethod
    def _replace_or_append_history(
        history: list[dict], kind: str, date_str: str, hour: int, target_c: float, committed_at: str
    ) -> None:
        """Last-write-wins dedup on (kind, date, hour) among non-cancelled entries (R1-#21)."""
        for entry in history:
            if (
                entry.get("cancelled_at") is None
                and entry.get("kind") == kind
                and entry.get("date") == date_str
                and entry.get("hour") == hour
            ):
                entry["target_c"] = target_c
                entry["committed_at"] = committed_at
                return
        history.append(
            {
                "kind": kind,
                "date": date_str,
                "hour": hour,
                "target_c": target_c,
                "committed_at": committed_at,
                "cancelled_at": None,
            }
        )
