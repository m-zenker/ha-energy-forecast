"""Backtest kit: replay the real EnergyForecastModel.train()/predict() under declarative arms.

``env`` is imported first so the AppDaemon stub and ``apps/`` path exist before any app import.
See backtest/README.md.
"""

from . import env  # noqa: F401
