"""Live-equivalent inputs for backtests, read from data/ (pulled from HA by scripts/pull_ha_data.py).

``load_inputs`` and ``load_weather`` are 1:1 ports of ``scripts/backtest_regime_label.py`` (load_live_inputs with its
defaults, load_weather), parameterised by ``data_dir``. A parity test pins them to the original.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from .env import DATA_DIR, TZ

EV_CSV = "ev_se_one_ev_charger_192_168_1_128_total_energy.csv"
APPLIANCE_CSVS = [
    "sub_spulmaschine_energy_consumption.csv",
    "sub_waschmaschine_energy_consumption.csv",
    "sub_tumbler_energy_consumption.csv",
]  # baseline_included_sensors (kermi dhw/heating) stay in the target
CORRECTIONS = {
    "solar_df": "solar_production.csv",
    "grid_export_df": "grid_export.csv",
    "battery_charge_df": "battery_charge.csv",
    "battery_discharge_df": "battery_discharge.csv",
}
WEATHER_CACHE = "regime_backtest_weather.pkl"


@dataclass(frozen=True)
class Inputs:
    energy: pd.DataFrame  # cols: timestamp, gross_kwh — baseline target (EV + appliances removed, corrected)
    ev_df: pd.DataFrame | None


@dataclass(frozen=True)
class Fold:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp  # last target day, inclusive


def _fold(name: str, start: str, end: str) -> Fold:
    return Fold(name, pd.Timestamp(start), pd.Timestamp(end))


DEFAULT_FOLDS: tuple[Fold, ...] = (
    _fold("Jan26", "2026-01-05", "2026-01-31"),
    _fold("Mar26", "2026-03-02", "2026-03-28"),
    _fold("Oct26", "2026-10-08", "2026-10-25"),
    _fold("Nov26", "2026-10-26", "2026-11-04"),
)


def _csv(data_dir: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(data_dir / name)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    return df.groupby("timestamp", as_index=False).last()


def load_config(data_dir: Path = DATA_DIR) -> dict:
    """The ``energy_forecast`` block of the pulled ``apps.yaml``."""
    return dict(yaml.safe_load((data_dir / "apps.yaml").read_text())["energy_forecast"])


def train_defaults(cfg: dict) -> dict:
    """``train()`` kwargs the way ``EnergyForecast._retrain()`` builds them from apps.yaml (same defaults)."""
    return dict(
        outdoor_df=None,
        weight_halflife_days=float(cfg.get("weight_halflife_days", 90)),
        canton=cfg.get("holiday_canton") or None,
        country=str(cfg.get("holiday_country", "CH")).upper(),
        enable_regimes=bool(cfg.get("enable_regimes", False)),
        regime_count=int(cfg.get("regime_count", 5)),
    )


def load_inputs(data_dir: Path = DATA_DIR) -> Inputs:
    """Rebuild live ``train()``'s energy_df / ev_df: hybrid EV split, appliances subtracted, target corrected."""
    from energy_forecast import ha_data
    from energy_forecast.const import drop_dst_duplicates
    from energy_forecast.energy_forecast import _apply_target_correction, _subtract_sub_sensors

    if not (data_dir / "energy_history.csv").exists():
        raise FileNotFoundError(f"{data_dir / 'energy_history.csv'} missing — run scripts/pull_ha_data.py")
    energy = drop_dst_duplicates(_csv(data_dir, "energy_history.csv")[["timestamp", "gross_kwh"]])
    ranges = pd.read_csv(data_dir / "excluded_ranges.csv")
    for a, b in zip(ranges["start"], ranges["end"], strict=True):
        energy = energy[~energy["timestamp"].between(pd.Timestamp(a), pd.Timestamp(b))]
    energy = energy.reset_index(drop=True)
    energy = _apply_target_correction(
        energy, **{k: _csv(data_dir, v)[["timestamp", "kwh"]] for k, v in CORRECTIONS.items()}
    )

    cfg = load_config(data_dir)
    thr = float(cfg.get("ev_charging_threshold_kwh", 7))
    kw = float(cfg.get("ev_charger_kw", 9))
    ev_hist = _csv(data_dir, EV_CSV)[["timestamp", "kwh"]]
    baseline, ev_df = ha_data.split_ev_charging_hybrid(energy, ev_hist, thr, charger_kw=kw)
    cov = ha_data.ev_sensor_coverage(ev_hist)
    if len(ev_df):
        floor = pd.to_datetime(ev_df["timestamp"]).dt.floor("1h")
        pad = floor if cov is None else floor[(floor < cov[0]) | (floor > cov[1])]
        adj = {t + pd.Timedelta(hours=d) for t in pad for d in (-1, 1)}
        baseline = baseline[~pd.to_datetime(baseline["timestamp"]).dt.floor("1h").isin(adj)]
    subs = {p: _csv(data_dir, p)[["timestamp", "kwh"]] for p in APPLIANCE_CSVS}
    baseline, _ = _subtract_sub_sensors(baseline, subs)
    return Inputs(energy=baseline.reset_index(drop=True), ev_df=ev_df)


def load_weather(data_dir: Path = DATA_DIR, fetch_tail: bool = True) -> pd.DataFrame:
    """Archive weather (cached pickle) + optionally the recent Open-Meteo measured tail (archive lags ~5 days)."""
    w = pd.read_pickle(data_dir / WEATHER_CACHE)
    if fetch_tail:
        from energy_forecast import weather

        cfg = load_config(data_dir)
        recent = weather.fetch_open_meteo(cfg["latitude"], cfg["longitude"], timezone=TZ, past_days=7)
        recent["timestamp"] = pd.to_datetime(recent["timestamp"]).dt.tz_localize(None)
        now = pd.Timestamp.now(tz=TZ).tz_localize(None)
        tail = recent[(recent["timestamp"] > w["timestamp"].max()) & (recent["timestamp"] <= now)]
        w = pd.concat([w, tail[w.columns.intersection(tail.columns)]], ignore_index=True)
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    return w.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
