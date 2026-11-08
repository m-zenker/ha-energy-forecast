"""Declarative experiment arms: feature drops, weather transforms, train-kwarg overrides, LightGBM seed."""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import env  # noqa: F401  (apps/ on sys.path before the app import below)


@dataclass(frozen=True)
class Arm:
    name: str
    drop: tuple[str, ...] = ()
    weather: Callable[[pd.DataFrame], pd.DataFrame] | None = None
    train_kwargs: Mapping[str, Any] = field(default_factory=dict)
    seed: int | None = None


@contextlib.contextmanager
def feature_drop(drop: Iterable[str]) -> Iterator[None]:
    """Rebind ``energy_forecast.model._FEATURES_BASE`` without *drop* (train() reads it at call time); restore on exit.

    Only static ``_FEATURES_BASE`` names can be dropped. Names train() re-adds (lag_*h, physics_kwh, sub-sensor
    lags) are caught after training by the runner.
    """
    import energy_forecast.model as mm

    drop = list(drop)
    orig = mm._FEATURES_BASE
    unknown = [d for d in drop if d not in orig]
    if unknown:
        raise ValueError(f"cannot drop {unknown}: only static _FEATURES_BASE names can be dropped")
    mm._FEATURES_BASE = [c for c in orig if c not in drop]
    try:
        yield
    finally:
        mm._FEATURES_BASE = orig


def climatology(w: pd.DataFrame, days: int = 21) -> pd.DataFrame:
    """Replace ``temp_c`` with a centred *days*-day rolling mean: keeps the season, removes day-to-day weather."""
    out = w.copy()
    order = out["timestamp"].argsort(kind="stable")
    smoothed = out["temp_c"].iloc[order].rolling(days * 24, center=True, min_periods=24).mean()
    out["temp_c"] = smoothed.reindex(out.index)
    return out


def noise_floor_arms(base: Arm, seeds: Sequence[int] = (1, 2)) -> list[Arm]:
    """Copies of *base* differing only in LightGBM seed — their spread is the seed noise floor."""
    return [dataclasses.replace(base, name=f"{base.name}@s{s}", seed=s) for s in seeds]
