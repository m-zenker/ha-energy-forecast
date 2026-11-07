"""Run mechanics shared by every backtest: paths, AppDaemon stub, frozen clock, LightGBM seed override.

Always import the app as ``energy_forecast.*`` (``apps/`` on ``sys.path``), never ``apps.energy_forecast.*``:
they are separate module objects, and patches applied to one don't reach the other.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
TZ = "Europe/Zurich"

if str(REPO / "apps") not in sys.path:
    sys.path.insert(0, str(REPO / "apps"))
if "hassapi" not in sys.modules:  # AppDaemon only exists inside HA; the app only needs a base class
    _stub = ModuleType("hassapi")
    _stub.Hass = MagicMock  # type: ignore[attr-defined]
    sys.modules["hassapi"] = _stub


@contextlib.contextmanager
def frozen_now(ts: pd.Timestamp, tz: str = TZ) -> Iterator[None]:
    """Pin ``pd.Timestamp.now`` to naive-local *ts* (aware when called with ``tz=``); restore on exit.

    Saves/restores the class-dict descriptor: ``pd.Timestamp.now`` is a classmethod, so attribute access
    returns a fresh bound method each time and can't be used for the restore.
    """
    orig = pd.Timestamp.__dict__["now"]
    naive = pd.Timestamp(ts)

    def _now(cls, tz=None):
        return naive if tz is None else naive.tz_localize(tz, ambiguous=True, nonexistent="shift_forward")

    pd.Timestamp.now = classmethod(_now)
    try:
        yield
    finally:
        pd.Timestamp.now = orig


@contextlib.contextmanager
def lgbm_seed(seed: int | None) -> Iterator[None]:
    """Force ``random_state=seed`` on every ``LGBMRegressor.fit`` inside the block; ``None`` changes nothing."""
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
    if seed is None or lgb is None:
        yield
        return
    orig = lgb.LGBMRegressor.__dict__["fit"]

    def _fit(self, *args, **kwargs):
        self.random_state = seed
        return orig(self, *args, **kwargs)

    lgb.LGBMRegressor.fit = _fit
    try:
        yield
    finally:
        lgb.LGBMRegressor.fit = orig
