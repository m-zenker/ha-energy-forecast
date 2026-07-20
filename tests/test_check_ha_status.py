"""Tests for scripts/check_ha_status.py's pure filter/classify/render functions.

Only pure functions are tested here — no network I/O. Follows the same
importlib module-loading idiom used in ha-energy-manager's test suite.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).parent.parent


@pytest.fixture
def status_module(monkeypatch):
    monkeypatch.setenv("EM_HA_TOKEN", "test-token")
    script_path = _REPO_ROOT / "scripts" / "check_ha_status.py"
    if not script_path.exists():
        pytest.skip("scripts/check_ha_status.py is gitignored/local-only and not present in this checkout")
    spec = importlib.util.spec_from_file_location("check_ha_status", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_ha_status"] = mod
    spec.loader.exec_module(mod)
    yield mod
    del sys.modules["check_ha_status"]


def _state(entity_id: str, state: str, attributes: dict | None = None) -> dict:
    return {"entity_id": entity_id, "state": state, "attributes": attributes or {}}


class TestIndexByEntityId:
    def test_indexes_by_entity_id(self, status_module):
        states = [_state("sensor.a", "1")]
        assert status_module.index_by_entity_id(states) == {"sensor.a": states[0]}


class TestCheckAddonState:
    def test_reads_state_from_addon_info(self, status_module):
        assert status_module.check_addon_state({"data": {"state": "started"}}) == "started"

    def test_unknown_when_missing(self, status_module):
        assert status_module.check_addon_state({}) == "unknown"


class TestCheckModelPhase:
    def test_reads_model_phase_attribute(self, status_module):
        states_by_id = status_module.index_by_entity_id(
            [_state("sensor.ha_energy_forecast_energy_forecast_today", "12.3", {"model_phase": "physics_ml_blend"})]
        )
        assert status_module.check_model_phase(states_by_id) == "physics_ml_blend"

    def test_none_when_entity_missing(self, status_module):
        assert status_module.check_model_phase({}) is None


class TestCheckSetupStatus:
    def test_reads_setup_status_state(self, status_module):
        states_by_id = status_module.index_by_entity_id(
            [_state("sensor.ha_energy_forecast_energy_forecast_setup_status", "ok")]
        )
        assert status_module.check_setup_status(states_by_id) == "ok"

    def test_not_found_when_missing(self, status_module):
        assert status_module.check_setup_status({}) == "not_found"


class TestRenderStatusSection:
    def test_renders_all_fields(self, status_module):
        section = status_module.render_status_section(
            run_ts="2026-07-13 14:32",
            addon_state="started",
            model_phase="physics_ml_blend",
            setup_status="ok",
        )
        assert "## 2026-07-13 14:32" in section
        assert "**AppDaemon add-on**: started" in section
        assert "**model_phase**: physics_ml_blend" in section
        assert "**setup_status**: ok" in section

    def test_renders_missing_placeholder(self, status_module):
        section = status_module.render_status_section(
            run_ts="2026-07-13 14:32", addon_state="started", model_phase=None, setup_status="not_found"
        )
        assert "**model_phase**: MISSING" in section
        assert "**setup_status**: not_found" in section
