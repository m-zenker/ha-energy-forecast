"""Pytest configuration — makes the AppDaemon apps directory importable."""
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

# Add apps/ to sys.path so tests can import energy_forecast as a package
# without installing it, mirroring how AppDaemon loads it at runtime.
sys.path.insert(0, str(Path(__file__).parent / "apps"))

# Stub out hassapi so energy_forecast.py can be imported without AppDaemon.
# Tests that exercise AppDaemon-specific behaviour should mock at a higher level.
if "hassapi" not in sys.modules:
    _hassapi_stub = ModuleType("hassapi")
    _hassapi_stub.Hass = MagicMock  # type: ignore[attr-defined]
    sys.modules["hassapi"] = _hassapi_stub
