"""Live-equivalent backtest frame built from one retrain capture.

A retrain capture holds everything live ``train()`` saw: the full hourly history of energy, weather, sub-meters,
climate, tank and buffer readings. Truncating those frames at a cutoff gives the training inputs live would have
had then, and the serve inputs of any origin are the same frames cut at the origin hour. So one capture is enough
to walk the real ``train()`` / ``predict()`` / ``predict_intervals()`` forward over the whole captured history.

Checked against live on 2026-11-13 (see docs/analysis/2026-11-13-forecast-improvement-audit.md): serve inputs
built this way reproduce captured live forecasts to 0.007 kWh/h mean abs difference, and a retrain at the capture
time reproduces the live model exactly.

Limits:
- History is today's view. Rows that were revised after the fact (late sensor data, resyncs) are the revised ones.
  Near a capture this is at most the last hour; that one hour can still move a 48 h total by several percent,
  which is ordinary retrain jitter (see the jitter arm in ``walkforward``).
- Weather is the captured archive weather, i.e. perfect. Measured cost of real forecast weather: 1-2 % MAE.
- The physics model's state is the capture's state at every cutoff. With ``physics_kwh`` at zero gain this does not
  change forecasts; it would matter for Phase 2.
- ``away_df`` is limited to the 10 days before the cutoff, as live's recorder window does.

Loading real captures needs the env that matches the HA host (``/home/jovyan/my_envs/ha-energy-forecast-live``).
"""

from __future__ import annotations

import contextlib
import copy
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import env  # noqa: F401  (hassapi stub + apps/ on sys.path before the app imports)
from .env import TZ, frozen_now

HEATING_SUB_METER = "sub_em_kermi_bridge_kermi_electricity_heating"
RECENT_ROWS = 401  # rows of recent_actuals live passes to predict()
AWAY_WINDOW = pd.Timedelta(days=10)
SERVE_PAST = pd.Timedelta(days=3)  # live forecast frame: 3 past days ...
SERVE_AHEAD = pd.Timedelta(days=4, hours=1)  # ... to midnight + 4 d 1 h
CONTIGUOUS_PAST = pd.Timedelta(days=16)  # enough history that the frozen weather tail never leaves a gap
DEFAULT_SETPOINT_ON = 20.0
DEFAULT_SETPOINT_OFF = 12.0


def _before(df: pd.DataFrame, limit: pd.Timestamp) -> pd.DataFrame:
    return df[pd.to_datetime(df["timestamp"]) < limit].reset_index(drop=True)


def _is_frame_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(isinstance(v, pd.DataFrame) for v in value.values())


class CaptureFrame:
    """Training and serve inputs for any cutoff or origin inside one retrain capture's history."""

    def __init__(self, payload: dict, heating_thresholds: Any | None = None) -> None:
        self.tk: dict = payload["train_kwargs"]
        self.physics = payload.get("physics_model")
        self.model_init: dict = dict(payload.get("model_init") or {"timezone": TZ})
        self.weather = (
            self.tk["weather_df"].drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        )
        self.end = pd.Timestamp(self.tk["energy_df"]["timestamp"].max())
        self.heating_thresholds = heating_thresholds

    @classmethod
    def from_capture(cls, root: Path, capture_id: str | None = None, **kwargs) -> CaptureFrame:
        """Load a retrain capture (the latest when *capture_id* is None)."""
        from energy_forecast import capture

        if capture_id is None:
            sidecars = capture.list_captures(root, "retrain")
            if not sidecars:
                raise FileNotFoundError(f"no retrain captures under {root}")
            capture_id = sidecars[-1]["capture_id"]
        payload, _ = capture.load_capture(root, "retrain", capture_id)
        if "heating_thresholds" not in kwargs:
            # the thresholds live learned at that retrain, saved with the model files
            from energy_forecast import heating_season

            saved = Path(root) / "retrain" / f"{capture_id}.ref" / "heating_thresholds.json"
            kwargs["heating_thresholds"] = heating_season.load_thresholds(saved) if saved.exists() else None
        return cls(payload, **kwargs)

    def actuals(self) -> pd.Series:
        """The model's own target (EV and baseline-mode appliances removed), hourly, indexed by timestamp."""
        e = self.tk["energy_df"].drop_duplicates("timestamp", keep="last")
        return pd.Series(e["gross_kwh"].to_numpy(), index=pd.DatetimeIndex(e["timestamp"])).sort_index()

    # ── training inputs ──────────────────────────────────────────────────────
    def train_kwargs(self, cutoff: pd.Timestamp) -> dict:
        """``train()`` kwargs as of *cutoff*: every frame holds only rows strictly before it."""
        cutoff = pd.Timestamp(cutoff)
        out: dict = {}
        for key, value in self.tk.items():
            if isinstance(value, pd.DataFrame):
                # the daily heating label is carried forward to the end of the current day, as live does
                limit = cutoff.normalize() + pd.Timedelta(days=1) if key == "heating_feature_df" else cutoff
                cut = _before(value, limit)
                if key == "away_df":
                    cut = cut[pd.to_datetime(cut["timestamp"]) >= cutoff - AWAY_WINDOW].reset_index(drop=True)
                out[key] = None if cut.empty else cut
            elif _is_frame_dict(value):
                cut_d = {k: _before(v, cutoff) for k, v in value.items()}
                cut_d = {k: v for k, v in cut_d.items() if not v.empty}
                out[key] = cut_d or None
            else:
                out[key] = value
        if out.get("climate_dfs") is None:
            out["room_areas"] = None
        return out

    # ── serve inputs ─────────────────────────────────────────────────────────
    def forecast_frame(self, origin: pd.Timestamp, past: pd.Timedelta = CONTIGUOUS_PAST) -> pd.DataFrame:
        """Weather rows from ``origin.normalize() - past`` to the end of the live forecast window."""
        day = pd.Timestamp(origin).normalize()
        ts = self.weather["timestamp"]
        return self.weather[(ts >= day - past) & (ts <= day + SERVE_AHEAD)].reset_index(drop=True)

    def _heating_projection(self, origin, forecast_df, subs, climate):
        from energy_forecast import heating_season

        thresholds = self.heating_thresholds or heating_season.default_thresholds("meter")
        meter = (subs or {}).get(HEATING_SUB_METER)
        entity = self.tk.get("heating_active_df")
        entity = _before(entity, origin.floor("h")) if entity is not None else None
        meter_days = 0 if meter is None else pd.to_datetime(meter["timestamp"]).dt.normalize().nunique()
        has_entity = entity is not None and not entity.empty
        if meter_days < heating_season.MIN_LABEL_DAYS and not has_entity:
            return None, None, None
        entity_state = int(entity["heating_active"].iloc[-1]) if has_entity else None
        if meter_days >= heating_season.MIN_LABEL_DAYS:
            prior, today = heating_season.meter_prior_and_today(meter, origin)
            if prior is None:
                prior = entity_state if entity_state is not None else 0
        else:
            prior, today = (entity_state or 0), entity_state
        active = bool(today if today is not None else prior)
        setpoints = []
        for c_df in (climate or {}).values():
            valid = c_df.dropna(subset=["setpoint"])
            if not valid.empty:
                setpoints.append(float(valid["setpoint"].iloc[-1]))
        live_sp = float(np.mean(setpoints)) if setpoints else None
        if active:
            s_on, s_off = (live_sp if live_sp is not None else DEFAULT_SETPOINT_ON), DEFAULT_SETPOINT_OFF
        else:
            s_on, s_off = DEFAULT_SETPOINT_ON, (live_sp if live_sp is not None else DEFAULT_SETPOINT_OFF)
        series = heating_season.project_heating_active(origin, forecast_df, prior, thresholds, today_state=today)
        return series, s_on, s_off

    def predict_kwargs(self, origin: pd.Timestamp, forecast_df: pd.DataFrame | None = None) -> dict:
        """``predict()`` kwargs as of *origin*: history strictly before the origin hour, weather to +4 days.

        *forecast_df* overrides the weather frame, e.g. a captured live frame or ``forecast_frame(origin,
        SERVE_PAST)`` to reproduce live's 3-day tail.
        """
        origin = pd.Timestamp(origin)
        hour = origin.floor("h")
        tk = self.tk
        if forecast_df is None:
            forecast_df = self.forecast_frame(origin)
        future = pd.date_range(hour, periods=48, freq="h")

        def cut(key):
            df = tk.get(key)
            if df is None:
                return None
            df = _before(df, hour)
            return None if df.empty else df

        def cut_dict(key):
            d = tk.get(key)
            if not d:
                return None
            d = {k: _before(v, hour) for k, v in d.items()}
            return {k: v for k, v in d.items() if not v.empty} or None

        subs = cut_dict("sub_sensors_dict")
        climate = cut_dict("climate_dfs")
        heating_series, s_on, s_off = self._heating_projection(origin, forecast_df, subs, climate)
        energy = tk["energy_df"]
        return dict(
            forecast_df=forecast_df,
            live_temp=None,
            recent_actuals=energy[pd.to_datetime(energy["timestamp"]) < hour].tail(RECENT_ROWS).reset_index(drop=True),
            sub_sensors_recent=subs,
            away_series=pd.Series(0, index=future),
            people_home_series=pd.Series(0, index=future),
            climate_recent=climate,
            dhw_recent=cut("dhw_df"),
            room_areas=tk.get("room_areas") if climate else None,
            heating_active_series=heating_series,
            setpoint_on=s_on,
            setpoint_off=s_off,
            heating_buffer_temp_recent=cut("heating_buffer_temp_df"),
        )

    def physics_for(self, work: Path):
        """A copy of the captured physics model whose persistence points at *work* (never the live paths)."""
        if self.physics is None:
            return None
        physics = copy.deepcopy(self.physics)
        physics._model_dir = Path(work)
        physics._calibration_path = Path(work) / "physics_calibration.json"
        physics._schedule_path = Path(work) / "physics_schedule.json"
        return physics


def _copy_files(src: Path, dst: Path) -> None:
    for p in Path(src).iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)


def load_model(model_dir: Path, work_root: Path, model_init: dict | None = None):
    """Load saved model files from a copy under *work_root* (the source dir is never written)."""
    from energy_forecast.model import EnergyForecastModel

    work = Path(tempfile.mkdtemp(prefix="wf_", dir=work_root))
    _copy_files(model_dir, work)
    return EnergyForecastModel(work, **(model_init or {"timezone": TZ})), work


def train(
    frame: CaptureFrame,
    cutoff: pd.Timestamp,
    work_root: Path,
    seed_dir: Path | None = None,
    patch: Callable[[], contextlib.AbstractContextManager] | None = None,
    train_kwargs: dict | None = None,
):
    """Run the real ``train()`` as of *cutoff* in a fresh dir under *work_root*; returns ``(model, work_dir)``.

    *seed_dir* is the model dir the retrain starts from (live retrains start from the previous model's files).
    *patch* is a context-manager factory active during training; *train_kwargs* overrides the captured kwargs.
    """
    from energy_forecast.model import EnergyForecastModel

    cutoff = pd.Timestamp(cutoff)
    work = Path(tempfile.mkdtemp(prefix="wf_", dir=work_root))
    if seed_dir is not None:
        _copy_files(seed_dir, work)
    kwargs = {**frame.train_kwargs(cutoff), **(train_kwargs or {})}
    kwargs["physics_model"] = frame.physics_for(work)
    with frozen_now(cutoff, frame.model_init.get("timezone", TZ)), patch() if patch else contextlib.nullcontext():
        model = EnergyForecastModel(work, **frame.model_init)
        model.train(**kwargs)
    return model, work


def predict(
    frame: CaptureFrame,
    model,
    work: Path,
    origin: pd.Timestamp,
    kwargs: dict | None = None,
    physics: bool = False,
) -> pd.DataFrame:
    """48 h forecast as of *origin*: columns ``ts, lead, p50`` and, when quantile models exist, ``lo, hi``.

    *physics* False skips the physics baseline at serve time. That is exact while ``physics_kwh`` has no splits
    in the model and several times faster; pass True for Phase 2 models.
    """
    origin = pd.Timestamp(origin)
    kwargs = copy.deepcopy(kwargs) if kwargs is not None else frame.predict_kwargs(origin)
    physics_model = frame.physics_for(work) if physics else None
    with frozen_now(origin, frame.model_init.get("timezone", TZ)):
        prepared = model._prepare_prediction_X(**kwargs, physics_model=physics_model)
        point = model.predict(**kwargs, physics_model=physics_model, _prepared=prepared)
        interval_kwargs = {k: v for k, v in kwargs.items() if k != "heating_buffer_temp_recent"}
        intervals = model.predict_intervals(**interval_kwargs, _prepared=prepared)
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(point["timestamp"]).dt.tz_localize(None).to_numpy(),
            "lead": np.arange(len(point)),
            "p50": point["predicted_kwh"].to_numpy(dtype=float),
        }
    )
    if intervals is not None:
        out["lo"] = intervals["low_kwh"].to_numpy(dtype=float)
        out["hi"] = intervals["high_kwh"].to_numpy(dtype=float)
    return out
