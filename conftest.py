"""Pytest configuration — makes the AppDaemon apps directory importable."""
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

# Add apps/ to sys.path so tests can import energy_forecast as a package
# without installing it, mirroring how AppDaemon loads it at runtime.
sys.path.insert(0, str(Path(__file__).parent / "apps"))

# Stub out the `hassapi` module before any test imports energy_forecast.py.
# Without this stub, the top-level `import hassapi as hass` in energy_forecast.py
# raises an ImportError because the hassapi package is only available inside an
# AppDaemon process.  Tests that need real AppDaemon behaviour should mock the
# specific methods (e.g. self.set_state, self.log) at a higher level.
if "hassapi" not in sys.modules:
    _hassapi_stub = ModuleType("hassapi")
    _hassapi_stub.Hass = MagicMock  # type: ignore[attr-defined]
    sys.modules["hassapi"] = _hassapi_stub
