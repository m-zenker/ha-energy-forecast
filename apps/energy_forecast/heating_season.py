"""heating_season.py — daily heating-season label, learned daily-mean thresholds, 48h projection.

Replaces the hourly outdoor-temp hysteresis that projected heating ON on every cold night
while the real heating system stayed off (live MAE regression, Sept 2026). The rule works on
daily mean temperature and is decided once per calendar day:

    state_d = 1 if mean_d < on_below, 0 if mean_d > off_above, else state_{d-1}

Thresholds are learned per installation from a daily heating label, in tier order:
  1. heating sub-meter (a day with > METER_DAY_KWH of space-heating energy is a heating day)
  2. heating_system_active_entity (ON for at least half the day)
  3. none -> defaults
Days with fewer than MIN_DAY_HOURS hourly values never decide; they carry the previous state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_LOGGER = logging.getLogger(__name__)

DEFAULT_ON_BELOW = 12.0
DEFAULT_OFF_ABOVE = 16.0
METER_DAY_KWH = 0.5
MIN_LABEL_DAYS = 60
MIN_TRANSITIONS = 4
MIN_DAY_HOURS = 12
_GRID = np.round(np.arange(5.0, 18.01, 0.5), 1)


@dataclass(frozen=True)
class HeatingThresholds:
    on_below: float
    off_above: float
    source: str  # "learned" | "config" | "default"
    label_source: str  # "meter" | "entity" | "none"
    n_days: int = 0
    mismatch_rate: float | None = None


def default_thresholds(label_source: str = "none") -> HeatingThresholds:
    return HeatingThresholds(DEFAULT_ON_BELOW, DEFAULT_OFF_ABOVE, "default", label_source)


def _daily(series: pd.Series, how: str) -> pd.Series:
    """Resample an hourly series to days, dropping days with fewer than MIN_DAY_HOURS values."""
    grouped = series.resample("D")
    agg = getattr(grouped, how)()
    counts = grouped.count()
    return agg[counts >= MIN_DAY_HOURS].dropna()


def daily_mean_temp(weather_df: pd.DataFrame) -> pd.Series:
    if "temp_c" not in weather_df.columns:
        return pd.Series(dtype=float)
    w = weather_df[["timestamp", "temp_c"]].dropna()
    s = w.set_index(pd.to_datetime(w["timestamp"]))["temp_c"].astype(float)
    return _daily(s, "mean")


def daily_heating_label(
    heating_sub_meter_df: pd.DataFrame | None,
    heating_active_df: pd.DataFrame | None,
    meter_day_kwh: float = METER_DAY_KWH,
) -> tuple[pd.Series, str]:
    if heating_sub_meter_df is not None and not heating_sub_meter_df.empty:
        m = heating_sub_meter_df.set_index(pd.to_datetime(heating_sub_meter_df["timestamp"]))["kwh"].astype(float)
        daily = _daily(m, "sum")
        if not daily.empty:
            return (daily > meter_day_kwh).astype(int), "meter"
    if heating_active_df is not None and not heating_active_df.empty:
        e = heating_active_df.set_index(pd.to_datetime(heating_active_df["timestamp"]))["heating_active"].astype(float)
        hourly = e.resample("1h").last().ffill()
        daily = _daily(hourly, "mean")
        if not daily.empty:
            return (daily >= 0.5).astype(int), "entity"
    return pd.Series(dtype=int), "none"


def simulate_rule(day_means: pd.Series, prior_state: int, on_below: float, off_above: float) -> pd.Series:
    state = int(prior_state)
    out = []
    for m in day_means.values:
        if m < on_below:
            state = 1
        elif m > off_above:
            state = 0
        out.append(state)
    return pd.Series(out, index=day_means.index, dtype=int)


def learn_thresholds(label: pd.Series, day_means: pd.Series, label_source: str) -> HeatingThresholds:
    joined = pd.concat({"label": label, "mean": day_means}, axis=1, join="inner").dropna()
    n_days = len(joined)
    transitions = int((joined["label"].diff().abs() == 1).sum())
    if n_days < MIN_LABEL_DAYS or transitions < MIN_TRANSITIONS:
        _LOGGER.info(
            "Heating season: %d labelled days / %d transitions (< %d / %d) — using default thresholds",
            n_days,
            transitions,
            MIN_LABEL_DAYS,
            MIN_TRANSITIONS,
        )
        return HeatingThresholds(DEFAULT_ON_BELOW, DEFAULT_OFF_ABOVE, "default", label_source, n_days)
    y = joined["label"].astype(int).values
    means = joined["mean"]
    prior = int(y[0])
    best: tuple[float, float, float] | None = None
    for on in _GRID:
        for off in _GRID[_GRID >= on]:
            err = float((simulate_rule(means, prior, on, off).values != y).mean())
            if best is None or err < best[0]:
                best = (err, float(on), float(off))
    err, on, off = best
    return HeatingThresholds(on, off, "learned", label_source, n_days, round(err, 4))


def resolve_thresholds(
    config_on: float | None, config_off: float | None, learned: HeatingThresholds
) -> HeatingThresholds:
    if config_on is None or config_off is None:
        return learned
    return HeatingThresholds(float(config_on), float(config_off), "config", learned.label_source, learned.n_days)


def hourly_label_df(label: pd.Series) -> pd.DataFrame:
    if label.empty:
        return pd.DataFrame(columns=["timestamp", "heating_active"])
    hours = pd.date_range(label.index.min(), label.index.max() + pd.Timedelta(hours=23), freq="1h")
    df = pd.DataFrame({"timestamp": hours, "heating_active": label.reindex(hours.normalize()).values})
    # Unlabelled days (short meter coverage) stay absent -> _engineer_features' fillna(1) default applies.
    df = df.dropna(subset=["heating_active"])
    df["heating_active"] = df["heating_active"].astype(int)
    return df.reset_index(drop=True)


def meter_prior_and_today(
    meter_df: pd.DataFrame | None, now: pd.Timestamp, meter_day_kwh: float = METER_DAY_KWH
) -> tuple[int | None, int | None]:
    if meter_df is None or meter_df.empty:
        return None, None
    m = meter_df.set_index(pd.to_datetime(meter_df["timestamp"]))["kwh"].astype(float)
    today = now.normalize()
    yesterday = m[(m.index >= today - pd.Timedelta(days=1)) & (m.index < today)]
    prior = int(yesterday.sum() > meter_day_kwh) if yesterday.count() >= 18 else None
    today_kwh = m[(m.index >= today) & (m.index <= now)].sum()
    return prior, (1 if today_kwh > meter_day_kwh else None)


def project_heating_active(
    now: pd.Timestamp,
    forecast_df: pd.DataFrame,
    prior_state: int,
    thresholds: HeatingThresholds,
    today_state: int | None = None,
    horizon_h: int = 48,
) -> pd.Series:
    future_hours = pd.date_range(now.floor("1h"), periods=horizon_h, freq="1h")
    temps = forecast_df.set_index(pd.to_datetime(forecast_df["timestamp"]))["temp_c"].astype(float)
    day_means = daily_mean_temp(pd.DataFrame({"timestamp": temps.index, "temp_c": temps.values}))
    state = int(prior_state)
    per_day: dict[pd.Timestamp, int] = {}
    for i, day in enumerate(future_hours.normalize().unique()):
        if i == 0 and today_state is not None:
            state = int(today_state)
        elif day in day_means.index:
            m = day_means.loc[day]
            if m < thresholds.on_below:
                state = 1
            elif m > thresholds.off_above:
                state = 0
        per_day[day] = state
    return pd.Series([per_day[d] for d in future_hours.normalize()], index=future_hours, dtype=np.int64)


def save_thresholds(t: HeatingThresholds, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(t)))
    except OSError as exc:
        _LOGGER.warning("Could not persist heating thresholds to %s: %s", path, exc)


def load_thresholds(path: Path) -> HeatingThresholds | None:
    try:
        return HeatingThresholds(**json.loads(path.read_text()))
    except (OSError, ValueError, TypeError):
        return None
