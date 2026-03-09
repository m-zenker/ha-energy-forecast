"""Pytest configuration — makes the AppDaemon apps directory importable."""
import sys
from pathlib import Path

# Add apps/ to sys.path so tests can import energy_forecast as a package
# without installing it, mirroring how AppDaemon loads it at runtime.
sys.path.insert(0, str(Path(__file__).parent / "apps"))
