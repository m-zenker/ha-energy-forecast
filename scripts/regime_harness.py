"""Live-equivalent training frame + prediction inputs for the #110 regime backtest.

Drives the real ``EnergyForecast.initialize()`` and ``_retrain()`` against caches trimmed to a
cutoff, with a recorder in place of the model, so the captured ``train()`` arguments are exactly
what the live app would have passed at that moment (minus the 30-day HA resync, which needs HA).

Imports go through ``energy_forecast.*`` (``apps/`` on ``sys.path``), never ``apps.energy_forecast``,
so monkeypatches land on the module objects ``_retrain`` actually runs in.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import logging
import shutil
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "apps"))
if "hassapi" not in sys.modules:  # AppDaemon is only present inside HA; the app only needs a base class
    _stub = ModuleType("hassapi")
    _stub.Hass = MagicMock  # type: ignore[attr-defined]
    sys.modules["hassapi"] = _stub

TZ = "Europe/Zurich"
_QUIET = logging.getLogger("regime_harness")
_QUIET.addHandler(logging.NullHandler())
_QUIET.propagate = False


# ── Cache trimming ────────────────────────────────────────────────────────────


def _naive_local(ts: pd.Series, tz: str) -> pd.Series:
    parsed = pd.to_datetime(ts, format="mixed", utc=False, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(tz).dt.tz_localize(None)
    return parsed


def trim_caches(src: Path, dst: Path, cutoff: pd.Timestamp, tz: str = TZ) -> None:
    """Copy every ``*.csv``/``*.json`` from *src* to *dst*; CSVs with a ``timestamp`` column keep rows <= cutoff.

    *cutoff* is naive local time. Row text is preserved (read/written as strings) so the app parses
    exactly what it would have parsed live.
    """
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if not f.is_file() or f.suffix not in (".csv", ".json"):
            continue
        out = dst / f.name
        if f.suffix == ".csv":
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
            if "timestamp" in df.columns:
                keep = _naive_local(df["timestamp"], tz) <= cutoff
                df[keep].to_csv(out, index=False)
                continue
        shutil.copy2(f, out)


# ── Weather stub ──────────────────────────────────────────────────────────────


class WeatherStub:
    """Serves archive weather clipped to <= cutoff, in the shape of the two real fetchers (naive local)."""

    def __init__(self, archive: pd.DataFrame, cutoff: pd.Timestamp):
        self._df = archive.assign(timestamp=pd.to_datetime(archive["timestamp"])).sort_values("timestamp")
        self._cutoff = cutoff

    def historical(self, lat, lon, start_date, end_date, timezone=TZ) -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1)  # end_date is inclusive, through 23:00
        ts = self._df["timestamp"]
        return self._df[(ts >= start) & (ts < end) & (ts <= self._cutoff)].reset_index(drop=True).copy()

    def open_meteo(self, lat, lon, timezone=TZ, past_days=3) -> pd.DataFrame:
        start = self._cutoff.floor("1D") - pd.Timedelta(days=past_days)
        ts = self._df["timestamp"]
        return self._df[(ts >= start) & (ts <= self._cutoff)].reset_index(drop=True).copy()


def load_weather_archive(scratch: Path, lat: float, lon: float, start: _dt.date, tz: str = TZ) -> pd.DataFrame:
    """Open-Meteo archive + recent measured tail, fetched once and kept in *scratch* (never shared across cutoffs)."""
    path = scratch / "weather_archive.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    from energy_forecast import weather as wx

    arch = wx.fetch_historical_weather(lat, lon, start, _dt.date.today(), timezone=tz)
    arch["timestamp"] = pd.to_datetime(arch["timestamp"])
    arch = arch.dropna(subset=["temp_c"])
    recent = wx.fetch_open_meteo(lat, lon, timezone=tz, past_days=10)
    recent["timestamp"] = pd.to_datetime(recent["timestamp"])
    if recent["timestamp"].dt.tz is not None:
        recent["timestamp"] = recent["timestamp"].dt.tz_convert(tz).dt.tz_localize(None)
    tail = recent[recent["timestamp"] > arch["timestamp"].max()]
    out = (
        pd.concat([arch, tail[arch.columns.intersection(tail.columns)]], ignore_index=True)
        .drop_duplicates("timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    scratch.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


# ── Frozen clock ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def frozen_clock(cutoff: pd.Timestamp):
    """Freeze ``pd.Timestamp.now`` and ``energy_forecast.energy_forecast.date`` at *cutoff* (naive local)."""
    from energy_forecast import energy_forecast as ef

    def _now(tz=None):
        return cutoff if tz is None else cutoff.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")

    class _FrozenDate(_dt.date):
        @classmethod
        def today(cls):
            return cutoff.date()

    with patch.object(pd.Timestamp, "now", staticmethod(_now)), patch.object(ef, "date", _FrozenDate):
        yield


# ── App construction ──────────────────────────────────────────────────────────


@contextlib.contextmanager
def _preserved_loggers():
    """initialize() rebinds the module-level ``_LOGGER`` of four modules to ``self.logger``; restore after."""
    from energy_forecast import energy_forecast as ef
    from energy_forecast import ha_data, model, weather

    mods = (ef, ha_data, model, weather)
    saved = [m._LOGGER for m in mods]
    try:
        yield
    finally:
        for m, lg in zip(mods, saved):
            m._LOGGER = lg


def load_config(data_dir: Path) -> dict:
    return dict(yaml.safe_load((data_dir / "apps.yaml").read_text())["energy_forecast"])


class TrainRecorder(MagicMock):
    """Stands in for ``EnergyForecastModel`` inside the app: stores ``train()`` arguments, never trains."""

    def __init__(self, *_a, **_k):  # EnergyForecastModel(model_dir, ...) args must not become a MagicMock spec
        super().__init__()
        self.captured: dict | None = None
        self.last_trained = _dt.datetime.min
        self.last_mae = None

        def _train(*args, **kwargs):
            self.captured = {"_args": args, **kwargs}

        self.train.side_effect = _train

    def _get_child_mock(self, **kw):
        return MagicMock(**kw)


def make_app(cfg: dict, scratch: Path):
    """Run the real ``initialize()`` on a MagicMock-based app whose caches live in *scratch*."""
    from energy_forecast import energy_forecast as ef

    args = dict(cfg)
    args["cache_path"] = str(scratch / "energy_history.csv")
    args["mqtt_discovery"] = False
    app = ef.EnergyForecast()
    app.args = args
    app.logger = _QUIET
    app.get_timezone = MagicMock(return_value=args.get("timezone") or TZ)
    app.get_history = MagicMock(return_value=[])
    for name in ("_validate_sensor_types", "_check_setup", "_publish_unavailable", "_load_pred_history"):
        setattr(app, name, MagicMock())
    with patch.object(ef, "EnergyForecastModel", TrainRecorder), _preserved_loggers():
        app.initialize()
    return app


# ── Live-equivalent training frame ────────────────────────────────────────────


def load_live_equivalent_training_frame(
    cutoff: pd.Timestamp, data_dir: Path, scratch: Path, archive: pd.DataFrame | None = None
) -> dict:
    """Return the exact ``(args, kwargs)`` the live ``_retrain()`` passed to ``model.train()`` at *cutoff*.

    Keyword arguments are returned as-is; the positional ones come back under ``"_args"``.
    """
    from energy_forecast import energy_forecast as ef
    from energy_forecast import ha_data, heating_season, weather

    cfg = load_config(data_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    trim_caches(data_dir, scratch, cutoff)
    if archive is None:
        archive = load_weather_archive(
            scratch, float(cfg["latitude"]), float(cfg["longitude"]), (cutoff - pd.Timedelta(days=800)).date()
        )
    stub = WeatherStub(archive, cutoff)

    with (
        frozen_clock(cutoff),
        patch.object(weather, "fetch_historical_weather", stub.historical),
        patch.object(weather, "fetch_open_meteo", stub.open_meteo),
        patch.object(heating_season, "save_thresholds", lambda *a, **k: None),
        patch.object(ha_data, "fetch_energy_history_15m", lambda *a, **k: None),
        _preserved_loggers(),
    ):
        app = make_app(cfg, scratch)
        app._retrain()
    captured = app._ml_model.captured
    if captured is None:
        raise RuntimeError("_retrain() returned without calling train() (insufficient history?)")
    captured["_app"] = app
    captured["_cfg"] = cfg
    assert ef is not None
    return captured


# ── Prediction inputs at an origin ────────────────────────────────────────────


class _Captured(Exception):
    """Raised from the stubbed ``_prepare_prediction_X`` to stop ``_update_sensors`` once inputs are captured."""


_PREPARE_POSITIONAL = ("forecast_df", "live_temp", "recent_actuals")


def build_prediction_inputs(
    origin: pd.Timestamp,
    data_dir: Path,
    scratch: Path,
    archive: pd.DataFrame | None = None,
    thresholds=None,
    horizon_h: int = 48,
    past_days: int = 3,
) -> dict:
    """Kwargs for ``EnergyForecastModel._prepare_prediction_X`` at *origin*, as ``_update_sensors`` builds them.

    Runs the real ``_update_sensors`` against caches trimmed to <= origin and aborts at the model call.
    The forecast is a perfect one: archive weather from origin-past_days to origin+horizon_h. The post-predict
    DHW override is not applied (only the model output is scored). Away/presence series are None (no HA state).
    *thresholds* defaults to the live ``models/heating_thresholds.json`` resolved as ``initialize()`` does.
    """
    from energy_forecast import heating_season, weather

    cfg = load_config(data_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    trim_caches(data_dir, scratch, origin)
    if archive is None:
        archive = load_weather_archive(
            scratch, float(cfg["latitude"]), float(cfg["longitude"]), (origin - pd.Timedelta(days=800)).date()
        )
    ts = pd.to_datetime(archive["timestamp"])
    hour0 = origin.floor("1h")
    window = archive[(ts >= hour0 - pd.Timedelta(days=past_days)) & (ts < hour0 + pd.Timedelta(hours=horizon_h))]
    forecast_df = window.reset_index(drop=True).copy()
    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
    at_origin = forecast_df.loc[forecast_df["timestamp"] == hour0, "temp_c"]
    live_temp = float(at_origin.iloc[0]) if len(at_origin) else None

    captured: dict = {}

    def _grab(*args, **kwargs):
        captured.update(dict(zip(_PREPARE_POSITIONAL, args)), **kwargs)
        raise _Captured

    with (
        frozen_clock(origin),
        patch.object(weather, "fetch_forecast", lambda *a, **k: forecast_df.copy()),
        patch.object(heating_season, "save_thresholds", lambda *a, **k: None),
        _preserved_loggers(),
    ):
        app = make_app(cfg, scratch)
        if thresholds is None:
            thresholds = heating_season.load_thresholds(data_dir / "models" / "heating_thresholds.json")
        if thresholds is not None:
            app._heating_thresholds = heating_season.resolve_thresholds(
                app._heating_temp_on_cfg, app._heating_temp_off_cfg, thresholds
            )
        app._read_live_temp = lambda: live_temp
        app._build_away_prediction_series = lambda now_ts: None
        app._build_people_home_prediction_series = lambda now_ts: None
        app._ml_model = MagicMock()
        app._ml_model._prepare_prediction_X.side_effect = _grab
        try:
            app._update_sensors()
        except _Captured:
            pass
    if not captured:
        raise RuntimeError("_update_sensors() ended without reaching _prepare_prediction_X")
    return captured
