# Physics-ML Hybrid — Plan B: Phase 1 Integration (Physics-as-Feature) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `ThermalPhysicsModel` (Plan A) into the existing LightGBM pipeline as a de-risking feature — `physics_kwh` joins `_FEATURES_BASE`, LightGBM still trains on `gross_kwh` unchanged, and a holdout diagnostic reports whether the physics signal is trustworthy enough to justify Phase 2 (Plan C).

**Architecture:** `energy_forecast.py` parses the new `physics:` config block, fetches the additional sensor histories physics needs, and owns one `ThermalPhysicsModel` instance (`self._physics_model`, `None` when `physics:` block absent). `model.py`'s `train()`/`predict()` gain a `physics_model: ThermalPhysicsModel | None` parameter; internally they call `predict_training_series()`/`predict_series()` to get `physics_kwh_series` and thread it through `_engineer_features()` exactly like the existing `regime_kwh_series` pattern.

**Tech Stack:** No new dependencies. Depends on Plan A (`apps/energy_forecast/physics.py` must be merged first).

**Base branch:** `dev`, from a rebase that includes Plan A's branch (or `dev` after Plan A has merged). Branch name: `feat/physics-phase1-integration ha-energy-forecast`.

## Global Constraints

- `physics_kwh` is added to `_FEATURES_BASE` **only** when `physics_model is not None` at train time — a constant-zero column is never added — spec §2.1, §5.1.
- Absent `physics:` config block → model behaviour identical to current `v0.11.7` — spec §4.4.
- Phase 1 target is unchanged: LightGBM trains on `gross_kwh`, log1p transform stays active — spec §2.1.
- Model-artifact portability: if a saved model has `physics_kwh` in `feature_cols` but `physics_model is None` at predict time, fill the column with `0.0` and log WARNING — never raise — spec §5.1.
- Cold-start gate: even if `use_physics_residual: true` is set in config, Phase 2 must not activate until `ThermalPhysicsModel.is_cold_start_gated` is `False`. This plan only needs to **read and log** that gate (Plan C implements the branch it guards) — spec §2.1.
- `heating_buffer_temp` is added to `_FEATURES_BASE` only when `heating_buffer_temp_sensor` is configured — spec §5.1.

---

### Task 1: `physics:` config ingest + `ThermalPhysicsModel` instantiation

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (add to `initialize()`, after the existing `self._climate_room_areas` block at line 252)
- Test: `tests/test_energy_forecast.py` (create if it doesn't already cover `initialize()`; otherwise extend the existing config-parsing test module — confirm which by running `grep -l "def initialize\|_climate_room_areas" tests/*.py` first)

**Interfaces:**
- Consumes: `ThermalPhysicsModel` from Plan A (`from .physics import ThermalPhysicsModel`).
- Produces: `self._physics_model: ThermalPhysicsModel | None`, `self._physics_config: dict`, `self._room_thermostats: list[dict]` on the `EnergyForecast` app instance.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_energy_forecast_physics_config.py
"""Tests for physics: config ingest in energy_forecast.py initialize()."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_app(args: dict):
    """Build an EnergyForecast app instance with .args pre-set and hassapi.Hass mocked out."""
    from energy_forecast.energy_forecast import EnergyForecast

    app = EnergyForecast.__new__(EnergyForecast)  # bypass Hass.__init__
    app.args = args
    app.logger = MagicMock()
    # Stub the AppDaemon methods initialize() calls that aren't under test here
    app.register_service = MagicMock()
    app.listen_event = MagicMock()
    app.run_hourly = MagicMock()
    app.run_every = MagicMock()
    return app


class TestPhysicsConfigIngest:
    def test_absent_physics_block_disables_model(self, tmp_path, monkeypatch):
        from energy_forecast import energy_forecast as ef_module

        monkeypatch.setattr(ef_module.Path, "__truediv__", ef_module.Path.__truediv__)
        app = _make_app({"energy_sensor": "sensor.grid_import"})
        app.initialize()
        assert app._physics_model is None

    def test_physics_block_creates_model(self, tmp_path, monkeypatch):
        app = _make_app(
            {
                "energy_sensor": "sensor.grid_import",
                "physics": {
                    "cop_sensor": "sensor.kermi_cop",
                    "dhw_tank_temp_sensor": "sensor.kermi_dhw_buffer_temp",
                    "room_thermostats": [
                        {"climate_entity": "climate.living_room", "temp_sensor": "sensor.netatmo_living_room_temp", "area_m2": 35}
                    ],
                    "use_physics_residual": False,
                },
            }
        )
        app.initialize()
        assert app._physics_model is not None
        assert app._room_thermostats == [
            {"climate_entity": "climate.living_room", "temp_sensor": "sensor.netatmo_living_room_temp", "area_m2": 35.0}
        ]

    def test_room_thermostats_missing_required_key_skipped_with_warning(self):
        app = _make_app(
            {
                "energy_sensor": "sensor.grid_import",
                "physics": {"room_thermostats": [{"climate_entity": "climate.living_room"}]},  # missing temp_sensor
            }
        )
        app.initialize()
        assert app._room_thermostats == []

    def test_defaults_applied_when_physics_block_partial(self):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        assert app._physics_model is not None
        assert app._physics_config["dhw_tank_volume_l"] == 200
        assert app._physics_config["internal_gains_fraction"] == 0.8
        assert app._physics_config["use_physics_residual"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: FAIL — `app._physics_model` raises `AttributeError` (attribute doesn't exist yet)

- [ ] **Step 3: Write minimal implementation**

In `apps/energy_forecast/energy_forecast.py`, add near the top of the file (module level, alongside other imports):

```python
from .physics import ThermalPhysicsModel
```

In `initialize()`, immediately after the existing block that sets `self._climate_room_areas` (line 252), add:

```python
        # ── Physics model config (physics-ml-hybrid) ──────────────────────────
        physics_raw = self.args.get("physics") or {}
        if not isinstance(physics_raw, dict):
            _LOGGER.warning("physics: config block must be a dict — ignoring")
            physics_raw = {}

        default_physics_config = {
            "cop_sensor": None,
            "dhw_tank_temp_sensor": None,
            "heating_buffer_temp_sensor": None,
            "heating_curve_sensor": None,
            "cop_formula": {"a": 2.5, "b": 0.07},
            "dhw_tank_volume_l": 200,
            "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8,
            "heating_curve_points": [[-20, 55.5], [-5, 46.0], [5, 39.5], [20, 25.0]],
            "room_thermostats": [],
            "use_physics_residual": False,
        }
        self._physics_config: dict = {**default_physics_config, **physics_raw}

        room_therm_raw = self._physics_config.get("room_thermostats") or []
        self._room_thermostats: list[dict] = []
        for item in room_therm_raw:
            if not isinstance(item, dict) or "climate_entity" not in item or "temp_sensor" not in item:
                _LOGGER.warning(f"room_thermostats: skipping invalid entry {item!r} — needs climate_entity and temp_sensor")
                continue
            self._room_thermostats.append(
                {
                    "climate_entity": str(item["climate_entity"]),
                    "temp_sensor": str(item["temp_sensor"]),
                    "area_m2": float(item.get("area_m2", 20.0)),
                }
            )
        self._physics_config["room_thermostats"] = self._room_thermostats

        physics_enabled = bool(physics_raw)  # any physics: block present, even empty dict, enables the model
        self._physics_model: ThermalPhysicsModel | None = None
        if physics_enabled:
            physics_model_dir = Path(__file__).parent / "models"
            self._physics_model = ThermalPhysicsModel(physics_model_dir, self._physics_config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "feat: ingest physics: config block and instantiate ThermalPhysicsModel"
```

---

### Task 2: Sensor history fetch wiring

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (extend the training-data-fetch method around line 1161-1214, and the prediction-cache-fetch method around line 1320-1420 — confirm exact method names via `grep -n "_climate_entities\[" apps/energy_forecast/energy_forecast.py` before editing since these are large methods)
- Test: extend `tests/test_energy_forecast_physics_config.py`

**Interfaces:**
- Consumes: `ha_data.fetch_generic_sensor_history(app, entity_id, cache_path, column_name=..., timezone=...)` (existing, used today for `dhw_df`/`heating_active_df`), `ha_data.fetch_climate_history(app, entity_id, cache_path, timezone=...)` (existing).
- Produces: `self._room_thermostat_temp_dfs: dict[str, pd.DataFrame]` (`{temp_sensor: df[timestamp, value]}`), `self._physics_dhw_tank_df`, `self._physics_heating_buffer_df`, `self._physics_cop_df` — all `pd.DataFrame | None`, fetched once per training cycle and cached for the prediction cycle the same way `dhw_df`/`heating_active_df` already are.

This task's fetch calls must be **skipped entirely** when `self._physics_model is None`, to keep the absent-`physics:`-block behaviour identical to `v0.11.7` per the Global Constraints.

- [ ] **Step 1: Write the failing test**

```python
class TestPhysicsSensorFetch:
    def test_no_physics_model_skips_all_physics_fetches(self, monkeypatch):
        from energy_forecast import ha_data as hd

        fetch_generic = MagicMock(return_value=None)
        fetch_climate = MagicMock(return_value=None)
        monkeypatch.setattr(hd, "fetch_generic_sensor_history", fetch_generic)
        monkeypatch.setattr(hd, "fetch_climate_history", fetch_climate)

        app = _make_app({"energy_sensor": "sensor.grid_import"})  # no physics: block
        app.initialize()
        app._fetch_physics_sensor_histories()
        fetch_generic.assert_not_called()
        fetch_climate.assert_not_called()

    def test_physics_model_present_fetches_configured_sensors(self, monkeypatch):
        import pandas as pd
        from energy_forecast import ha_data as hd

        empty_df = pd.DataFrame(columns=["timestamp", "value"])
        fetch_generic = MagicMock(return_value=empty_df)
        fetch_climate = MagicMock(return_value=pd.DataFrame(columns=["timestamp", "current_temp", "setpoint"]))
        monkeypatch.setattr(hd, "fetch_generic_sensor_history", fetch_generic)
        monkeypatch.setattr(hd, "fetch_climate_history", fetch_climate)

        app = _make_app(
            {
                "energy_sensor": "sensor.grid_import",
                "physics": {
                    "dhw_tank_temp_sensor": "sensor.kermi_dhw_buffer_temp",
                    "heating_buffer_temp_sensor": "sensor.kermi_heating_buffer",
                    "cop_sensor": "sensor.kermi_cop",
                    "room_thermostats": [
                        {"climate_entity": "climate.living_room", "temp_sensor": "sensor.netatmo_living_room_temp", "area_m2": 35}
                    ],
                },
            }
        )
        app.initialize()
        app._fetch_physics_sensor_histories()
        assert fetch_generic.call_count == 4  # dhw_tank, heating_buffer, cop, and the one room temp_sensor
        fetch_climate.assert_called_once()  # the room_thermostat's climate_entity, for setpoint projection
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py::TestPhysicsSensorFetch -v`
Expected: FAIL with `AttributeError: 'EnergyForecast' object has no attribute '_fetch_physics_sensor_histories'`

- [ ] **Step 3: Write minimal implementation**

Add a new method to the `EnergyForecast` class in `energy_forecast.py` (place it near the existing `_fetch_dhw_history`-style helpers, e.g. just above the method containing line 1161):

```python
    def _fetch_physics_sensor_histories(self) -> None:
        """Fetch the additional sensor histories the physics model needs. No-op if physics is disabled."""
        self._room_thermostat_temp_dfs: dict[str, "pd.DataFrame"] = {}
        self._physics_dhw_tank_df = None
        self._physics_heating_buffer_df = None
        self._physics_cop_df = None
        self._physics_climate_dfs: dict[str, "pd.DataFrame"] = {}

        if self._physics_model is None:
            return

        cfg = self._physics_config

        if cfg.get("dhw_tank_temp_sensor"):
            self._physics_dhw_tank_df = ha_data.fetch_generic_sensor_history(
                self, cfg["dhw_tank_temp_sensor"], self._cache_path("physics_dhw_tank"), column_name="buffer_temp",
                timezone=self._timezone,
            )
        if cfg.get("heating_buffer_temp_sensor"):
            self._physics_heating_buffer_df = ha_data.fetch_generic_sensor_history(
                self, cfg["heating_buffer_temp_sensor"], self._cache_path("physics_heating_buffer"),
                column_name="heating_buffer_temp", timezone=self._timezone,
            )
        if cfg.get("cop_sensor"):
            self._physics_cop_df = ha_data.fetch_generic_sensor_history(
                self, cfg["cop_sensor"], self._cache_path("physics_cop"), column_name="cop", timezone=self._timezone,
            )

        for rt in self._room_thermostats:
            temp_df = ha_data.fetch_generic_sensor_history(
                self, rt["temp_sensor"], self._cache_path(f"physics_temp_{rt['temp_sensor']}"),
                column_name="current_temp", timezone=self._timezone,
            )
            self._room_thermostat_temp_dfs[rt["climate_entity"]] = temp_df
            self._physics_climate_dfs[rt["climate_entity"]] = ha_data.fetch_climate_history(
                self, rt["climate_entity"], self._cache_path(f"physics_climate_{rt['climate_entity']}"),
                timezone=self._timezone,
            )
```

`self._cache_path(...)` must already exist as a helper (the file has one for climate entities at line 451 — confirm its exact signature with `grep -n "_cache_path\|def _.*cache" apps/energy_forecast/energy_forecast.py` and adapt the calls above to match; if it takes no override argument, follow the existing per-entity cache-path pattern used for `dhw_buffer_sensor` instead).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "feat: fetch physics sensor histories (DHW tank, heating buffer, COP, room temps)"
```

---

### Task 3: `physics_kwh_series` threaded through `_engineer_features()` + `_FEATURES_BASE`

**Files:**
- Modify: `apps/energy_forecast/model.py`
  - `_FEATURES_BASE` (lines 77-157): no static change — `physics_kwh` and `heating_buffer_temp` are appended conditionally at runtime, not hardcoded into the list (see Step 3).
  - `_engineer_features()` signature (lines 2419-2434): add `physics_kwh_series: pd.Series | None = None` and `heating_buffer_temp_series: pd.Series | None = None` parameters.
  - Merge logic (after line 2765, following the `regime_kwh` pattern): add the `physics_kwh` and `heating_buffer_temp` merges.
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing new from other tasks.
- Produces: `_engineer_features(..., physics_kwh_series=..., heating_buffer_temp_series=...)`. The returned `df` gains a `physics_kwh` column **only** when `physics_kwh_series is not None`, and a `heating_buffer_temp` column **only** when `heating_buffer_temp_series is not None`.

- [ ] **Step 1: Write the failing test**

```python
class TestPhysicsFeatureIntegration:
    def test_physics_kwh_present_when_series_given(self):
        ts = pd.date_range("2026-01-15", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 5})
        weather = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 5})
        physics_series = pd.Series([0.5] * 5, index=ts.floor("1h"))
        result = _engineer_features(df, weather, outdoor_df=None, physics_kwh_series=physics_series)
        assert "physics_kwh" in result.columns
        assert result["physics_kwh"].iloc[0] == pytest.approx(0.5)

    def test_physics_kwh_absent_not_zero_filled_when_none(self):
        ts = pd.date_range("2026-01-15", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 5})
        weather = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 5})
        result = _engineer_features(df, weather, outdoor_df=None, physics_kwh_series=None)
        assert "physics_kwh" not in result.columns

    def test_heating_buffer_temp_present_when_series_given(self):
        ts = pd.date_range("2026-01-15", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 5})
        weather = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 5})
        buffer_series = pd.Series([42.0] * 5, index=ts.floor("1h"))
        result = _engineer_features(df, weather, outdoor_df=None, heating_buffer_temp_series=buffer_series)
        assert "heating_buffer_temp" in result.columns

    def test_heating_buffer_temp_absent_when_none(self):
        ts = pd.date_range("2026-01-15", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [1.0] * 5})
        weather = pd.DataFrame({"timestamp": ts, "temp_c": [5.0] * 5})
        result = _engineer_features(df, weather, outdoor_df=None)
        assert "heating_buffer_temp" not in result.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestPhysicsFeatureIntegration -v`
Expected: FAIL with `TypeError: _engineer_features() got an unexpected keyword argument 'physics_kwh_series'`

- [ ] **Step 3: Write minimal implementation**

In `model.py`, extend the `_engineer_features()` signature (line 2419-2434) by adding two parameters at the end:

```python
    physics_kwh_series: pd.Series | None = None,  # hourly physics baseline, absent when physics disabled
    heating_buffer_temp_series: pd.Series | None = None,  # direct sensor feature, absent when sensor not configured
) -> pd.DataFrame:
```

Immediately after the existing `regime_kwh` merge block (lines 2757-2765), add:

```python
    # ── Physics baseline (optional) ──────────────────────────────────────────
    if physics_kwh_series is not None:
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(physics_kwh_series.to_frame("physics_kwh"), left_on="_ts_floor", right_index=True, how="left")
        df.drop(columns=["_ts_floor"], inplace=True, errors="ignore")
        df["physics_kwh"] = df["physics_kwh"].fillna(0.0)

    # ── Heating buffer temp (optional direct sensor feature) ──────────────────
    if heating_buffer_temp_series is not None:
        df["_ts_floor"] = df["timestamp"].dt.floor("1h")
        df = df.merge(
            heating_buffer_temp_series.to_frame("heating_buffer_temp"), left_on="_ts_floor", right_index=True, how="left"
        )
        df.drop(columns=["_ts_floor"], inplace=True, errors="ignore")
        df["heating_buffer_temp"] = df["heating_buffer_temp"].ffill().fillna(df["heating_buffer_temp"].median())
```

Note both blocks intentionally do **not** have an `else: df["physics_kwh"] = 0.0` branch (unlike `regime_kwh`) — per the Global Constraints, the column must be absent, not zero-filled, when the input series is `None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py::TestPhysicsFeatureIntegration -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "feat: thread physics_kwh_series and heating_buffer_temp through _engineer_features"
```

---

### Task 4: `train()` — physics calibration trigger, feature list inclusion, sample-weight down-weighting

**Files:**
- Modify: `apps/energy_forecast/model.py` (`train()`, lines 303-724: add `physics_model` param; call calibration + `predict_training_series()` before the `_engineer_features()` call at line 474-490; extend `base_features` construction at lines 492-515; fold `detect_open_windows()` into the `hourly_weights` computation from Task 4 of Plan A's design)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `ThermalPhysicsModel.calibrate()`, `.calibration_stale`, `.predict_training_series()`, `.check_zone_boundary()`, `.detect_open_windows()` (Plan A).
- Produces: `EnergyForecastModel.train(..., physics_model: ThermalPhysicsModel | None = None)`.

- [ ] **Step 1: Write the failing test**

```python
class TestTrainWithPhysics:
    def test_physics_kwh_in_feature_cols_when_physics_model_given(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": False,
        })
        model, _ = _make_trained_model(tmp_path / "model", physics_model=pm)
        assert "physics_kwh" in model.feature_cols

    def test_physics_kwh_absent_from_feature_cols_when_none(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model", physics_model=None)
        assert "physics_kwh" not in model.feature_cols

    def test_calibrate_called_when_calibration_stale(self, tmp_path):
        from unittest.mock import patch
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": False,
        })
        assert pm.calibration_stale is True
        with patch.object(pm, "calibrate", wraps=pm.calibrate) as mock_calibrate:
            _make_trained_model(tmp_path / "model", physics_model=pm)
            mock_calibrate.assert_called_once()

    def test_calibrate_skipped_when_fresh(self, tmp_path):
        from unittest.mock import patch
        from energy_forecast.physics import ThermalPhysicsModel, _atomic_write_json, _default_calibration

        model_dir = tmp_path / "physics_models"
        model_dir.mkdir(parents=True)
        fresh_calib = {**_default_calibration(), "calibrated_at": pd.Timestamp.now().isoformat()}
        _atomic_write_json(model_dir / "physics_calibration.json", fresh_calib)
        pm = ThermalPhysicsModel(model_dir, {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": False,
        })
        assert pm.calibration_stale is False
        with patch.object(pm, "calibrate") as mock_calibrate:
            _make_trained_model(tmp_path / "model", physics_model=pm)
            mock_calibrate.assert_not_called()
```

Extend `_make_trained_model()` in `tests/test_model.py` (lines 48-84) to accept and forward `physics_model=None`:

```python
def _make_trained_model(tmp_path, n: int = 600, timezone: str = "Europe/Zurich", physics_model=None) -> tuple:
    # ... existing body unchanged up to the model.train(...) call ...
    model.train(energy, weather, outdoor_df=None, weight_halflife_days=0, physics_model=physics_model)
    return model, forecast_df
```

(Read the full existing helper first — `Read tests/test_model.py` lines 48-90 — before editing, so the rest of the fixture body is preserved exactly.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestTrainWithPhysics -v`
Expected: FAIL with `TypeError: train() got an unexpected keyword argument 'physics_model'`

- [ ] **Step 3: Write minimal implementation**

In `model.py`, add two parameters to `train()`'s signature (line 303-322), as the last parameters:

```python
    physics_model: "ThermalPhysicsModel | None" = None,
    heating_buffer_temp_df: pd.DataFrame | None = None,  # cols: timestamp, heating_buffer_temp — from self._physics_heating_buffer_df
```

Add `from .physics import ThermalPhysicsModel` under `TYPE_CHECKING` at the top of `model.py` (avoids a circular import — `physics.py` only imports from `model.py` inside function bodies, per Plan A Task 5/8's local-import pattern).

Immediately before the existing `_engineer_features()` call in `train()` (line 474), insert:

```python
        physics_kwh_series = None
        heating_buffer_temp_series = None
        open_window_flags = None
        if physics_model is not None:
            physics_model.check_zone_boundary(list(climate_dfs.keys()) if climate_dfs else [])
            if physics_model.calibration_stale:
                physics_model.calibrate(
                    energy_df, weather_df, climate_dfs, dhw_df,
                    holdout_cutoff=energy_df["timestamp"].max() - pd.Timedelta(days=int(len(energy_df) / 24 * 0.1) or 1),
                    heating_active_df=heating_active_df, away_df=away_df,
                )
            physics_model._tau_hours = self._tau_hours
            physics_kwh_series = physics_model.predict_training_series(
                energy_df, weather_df, climate_dfs=climate_dfs, dhw_df=dhw_df, room_areas=room_areas
            )
            if climate_dfs:
                open_window_flags = physics_model.detect_open_windows(climate_dfs, weather_df, room_areas)

        if heating_buffer_temp_df is not None and not heating_buffer_temp_df.empty:
            heating_buffer_temp_series = heating_buffer_temp_df.set_index(
                pd.to_datetime(heating_buffer_temp_df["timestamp"])
            )["heating_buffer_temp"]
```

Update the `_engineer_features()` call (line ~483) to pass the new series:

```python
        df = _engineer_features(
            df, weather_df, outdoor_df, canton=canton, country=country,
            likely_ev_hours=likely_ev_hours, away_df=away_df, presence_df=presence_df,
            climate_dfs=climate_dfs_for_features, dhw_df=dhw_df, tau_hours=self._tau_hours,
            room_areas=room_areas, regime_kwh_series=regime_kwh_series, heating_active_df=heating_active_df,
            physics_kwh_series=physics_kwh_series, heating_buffer_temp_series=heating_buffer_temp_series,
        )
```

(Match this call's existing argument list exactly — read lines 474-490 first; only the two new kwargs are additions.)

Update `base_features` construction (lines 492-515) to conditionally include the new columns:

```python
        if physics_kwh_series is not None:
            base_features = [*base_features, "physics_kwh"]
        if heating_buffer_temp_series is not None:
            base_features = [*base_features, "heating_buffer_temp"]
```

(Insert immediately after the existing `base_features = [...]` construction, before it's used to build `feature_cols`.)

Fold `open_window_flags` into the existing `hourly_weights` computation (after line 406, where `hourly_weights` is finalized):

```python
        if open_window_flags is not None and hourly_weights is not None:
            ow = open_window_flags.reindex(df["timestamp"].values if "timestamp" in df else hourly_weights.index)
            down_weight = pd.Series(np.where(ow.fillna(False), 0.5, 1.0), index=hourly_weights.index)
            hourly_weights = hourly_weights * down_weight
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: all pass including the 4 new `TestTrainWithPhysics` tests

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "feat: wire physics calibration trigger and physics_kwh into train()"
```

---

### Task 5: `predict()` — physics feature at prediction time + model-artifact portability fallback

**Files:**
- Modify: `apps/energy_forecast/model.py` (`predict()`, lines 976-1026; `_prepare_prediction_X()` — the internal helper `predict()` and `shap_summary()` both call, locate via `grep -n "_prepare_prediction_X" apps/energy_forecast/model.py`)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `ThermalPhysicsModel.predict_series()` (Plan A).
- Produces: `EnergyForecastModel.predict(..., physics_model: ThermalPhysicsModel | None = None)`. Also updates `_prepare_prediction_X()` with the same parameter, since `shap_summary()` (Task 6) reuses it.

- [ ] **Step 1: Write the failing test**

```python
class TestPredictWithPhysics:
    def test_physics_kwh_filled_with_zero_when_model_disabled_at_predict_time(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": False,
        })
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)
        assert "physics_kwh" in model.feature_cols

        # simulate physics disabled at predict time (sensor outage / config change)
        result = model.predict(forecast_df, live_temp=5.0, physics_model=None)
        assert not result.empty  # no exception

    def test_physics_kwh_computed_when_model_present_at_predict_time(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": False,
        })
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)
        result = model.predict(forecast_df, live_temp=5.0, physics_model=pm)
        assert not result.empty

    def test_predict_without_physics_ever_trained_unaffected(self, tmp_path):
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=None)
        result = model.predict(forecast_df, live_temp=5.0, physics_model=None)
        assert not result.empty
        assert "physics_kwh" not in model.feature_cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestPredictWithPhysics -v`
Expected: FAIL with `TypeError: predict() got an unexpected keyword argument 'physics_model'`

- [ ] **Step 3: Write minimal implementation**

Add two parameters to `predict()`'s signature (line 976-991) and to `_prepare_prediction_X()`'s signature (find via grep above — same parameter list pattern):

```python
    physics_model: "ThermalPhysicsModel | None" = None,
    heating_buffer_temp_recent: pd.DataFrame | None = None,  # cols: timestamp, heating_buffer_temp — most recent reading(s)
```

Inside `_prepare_prediction_X()`, wherever the feature dataframe `feat_df` is finalized just before being sliced to `self.feature_cols` (this is the method `_engineer_features()`-equivalent call site for prediction — locate the `_engineer_features(` call inside `_prepare_prediction_X`), add:

```python
        physics_kwh_series = None
        if physics_model is not None:
            physics_kwh_series = physics_model.predict_series(
                forecast_df, climate_recent=climate_recent, dhw_recent=dhw_recent, room_areas=room_areas,
                heating_active_series=heating_active_series, setpoint_on=setpoint_on, setpoint_off=setpoint_off,
            )

        heating_buffer_temp_series = None
        if heating_buffer_temp_recent is not None and not heating_buffer_temp_recent.empty:
            # hold the most recent reading flat across the forecast horizon — same "recent" pattern
            # used for dhw_recent/climate_recent elsewhere in this method
            latest_val = float(heating_buffer_temp_recent.sort_values("timestamp")["heating_buffer_temp"].iloc[-1])
            heating_buffer_temp_series = pd.Series(latest_val, index=pd.DatetimeIndex(forecast_df["timestamp"]))
```

Pass `physics_kwh_series=physics_kwh_series, heating_buffer_temp_series=heating_buffer_temp_series` into the `_engineer_features()` call inside `_prepare_prediction_X()`, matching Task 3's new parameters.

Immediately after `_engineer_features()` returns inside `_prepare_prediction_X()`, before the dataframe is sliced down to `self.feature_cols`, add the portability fallback:

```python
        if "physics_kwh" in self.feature_cols and "physics_kwh" not in feat_df.columns:
            feat_df["physics_kwh"] = 0.0
            _LOGGER.warning("physics_kwh in trained feature list but physics_model disabled at predict time — filling with 0.0")
```

(`feat_df` is whatever local variable name the existing method uses for the post-`_engineer_features()` dataframe — confirm the exact name when reading the method before editing.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: all pass including the 3 new `TestPredictWithPhysics` tests

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "feat: wire physics_kwh into predict() with model-artifact portability fallback"
```

---

### Task 6: Phase 1 validation diagnostic (SHAP top-5 + OLS calibration slope)

**Files:**
- Modify: `apps/energy_forecast/model.py` (`train()`, after the holdout MAE computation around line 702)
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `EnergyForecastModel._validate_physics_phase1(X_holdout, y_holdout_gross, physics_kwh_holdout) -> dict` — returns `{"shap_rank": int | None, "shap_top5": bool, "ols_slope": float | None, "calibration_good": bool}`; logged at INFO/WARNING, stored on `self.physics_phase1_diagnostic` for the `recalibrate_physics` service (Plan B Task 7) and future operator visibility. **Does not** flip `use_physics_residual` — that remains a manual config decision per spec §2.1 ("proceed to Phase 2" is an operator action, not automatic).

- [ ] **Step 1: Write the failing test**

```python
class TestPhase1Validation:
    def test_calibration_good_when_top5_and_slope_in_range(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model")
        X = pd.DataFrame({"physics_kwh": [1.0, 2.0, 3.0], "other_feat": [0.1, 0.2, 0.3]})
        y_gross = np.array([1.05, 2.1, 2.9])  # slope ~1.0, physics_kwh tracks gross_kwh closely
        physics_vals = np.array([1.0, 2.0, 3.0])
        result = model._validate_physics_phase1(X, y_gross, physics_vals)
        assert result["ols_slope"] == pytest.approx(1.0, rel=0.1)
        assert result["calibration_good"] is True

    def test_calibration_bad_when_slope_out_of_range(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model")
        X = pd.DataFrame({"physics_kwh": [1.0, 2.0, 3.0], "other_feat": [0.1, 0.2, 0.3]})
        y_gross = np.array([3.0, 6.0, 9.0])  # slope 3.0, badly miscalibrated
        physics_vals = np.array([1.0, 2.0, 3.0])
        result = model._validate_physics_phase1(X, y_gross, physics_vals)
        assert result["calibration_good"] is False

    def test_physics_kwh_not_in_features_returns_none_rank(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model")
        X = pd.DataFrame({"other_feat": [0.1, 0.2, 0.3]})
        result = model._validate_physics_phase1(X, np.array([1.0, 2.0, 3.0]), None)
        assert result["shap_rank"] is None
        assert result["calibration_good"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestPhase1Validation -v`
Expected: FAIL with `AttributeError: 'EnergyForecastModel' object has no attribute '_validate_physics_phase1'`

- [ ] **Step 3: Write minimal implementation**

Add to `EnergyForecastModel`:

```python
    def _validate_physics_phase1(
        self, X_holdout: pd.DataFrame, y_holdout_gross: np.ndarray, physics_kwh_holdout: np.ndarray | None
    ) -> dict:
        result = {"shap_rank": None, "shap_top5": False, "ols_slope": None, "calibration_good": False}
        if "physics_kwh" not in X_holdout.columns or physics_kwh_holdout is None or self.model is None:
            return result

        if self.engine == "LightGBM":
            contrib = self.model.predict(X_holdout, pred_contrib=True)
            mean_abs = np.abs(contrib[:, :-1]).mean(axis=0)
            ranked = pd.Series(mean_abs, index=X_holdout.columns).sort_values(ascending=False)
            rank = int(ranked.index.get_loc("physics_kwh")) + 1  # 1-indexed
            result["shap_rank"] = rank
            result["shap_top5"] = rank <= 5

        x = np.asarray(physics_kwh_holdout, dtype=float)
        y = np.asarray(y_holdout_gross, dtype=float)
        if np.sum(x**2) > 1e-9:
            slope = float(np.sum(x * y) / np.sum(x**2))
            result["ols_slope"] = slope
            result["calibration_good"] = result["shap_top5"] and (0.8 <= slope <= 1.2)

        if result["calibration_good"]:
            _LOGGER.info(f"Phase 1 physics validation: PASS (shap_rank={result['shap_rank']}, ols_slope={result['ols_slope']:.2f}) — physics signal is well-calibrated and predictive")
        else:
            _LOGGER.info(f"Phase 1 physics validation: not yet ready for Phase 2 (shap_rank={result['shap_rank']}, ols_slope={result['ols_slope']})")
        return result
```

Wire it into `train()` right after the existing holdout MAE block (line 702), only when `physics_model is not None`:

```python
        self.physics_phase1_diagnostic: dict | None = None
        if physics_model is not None and "physics_kwh" in feature_cols and holdout_mae is not None:
            X_ho = X.iloc[split:]
            y_ho_gross = y[split:]  # gross_kwh, not log-transformed
            physics_ho = X_ho["physics_kwh"].values if "physics_kwh" in X_ho.columns else None
            self.physics_phase1_diagnostic = self._validate_physics_phase1(X_ho, y_ho_gross, physics_ho)
```

(`split`, `X`, `y` are the existing local variables from the holdout block at lines 688-702 — reuse them directly, don't recompute.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "feat: add Phase 1 physics validation diagnostic (SHAP rank + OLS slope)"
```

---

### Task 7: `recalibrate_physics` service + cold-start-gate logging + wire train()/predict() call sites in `energy_forecast.py`

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (service registration near line 296; the `train()`/`predict()`/`shap_summary()` call sites — search `self._ml_model.train(` and `self._ml_model.predict(` and `self._ml_model.shap_summary(` to find all of them, there are several per the earlier research)
- Test: extend `tests/test_energy_forecast_physics_config.py`

**Interfaces:**
- Produces: AppDaemon service `energy_forecast/recalibrate_physics`; every `self._ml_model.train(...)`, `.predict(...)`, `.shap_summary(...)` call site gains `physics_model=self._physics_model`.

- [ ] **Step 1: Write the failing test**

```python
class TestRecalibratePhysicsService:
    def test_service_registered_when_physics_enabled(self):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        calls = [c.args[0] for c in app.register_service.call_args_list]
        assert "energy_forecast/recalibrate_physics" in calls

    def test_service_not_registered_when_physics_disabled(self):
        app = _make_app({"energy_sensor": "sensor.grid_import"})
        app.initialize()
        calls = [c.args[0] for c in app.register_service.call_args_list]
        assert "energy_forecast/recalibrate_physics" not in calls

    def test_recalibrate_service_calls_physics_calibrate_and_updates_calibrated_at(self, tmp_path):
        from unittest.mock import patch

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._fetch_physics_sensor_histories = MagicMock()
        app._physics_climate_dfs = {}
        app._physics_dhw_tank_df = None
        # minimal energy/weather fetch stand-ins
        app._cached_energy_df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10, freq="1h"), "gross_kwh": [1.0] * 10})
        app._cached_weather_df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=10, freq="1h"), "temp_c": [5.0] * 10, "direct_radiation_wm2": [0.0] * 10})
        with patch.object(app._physics_model, "calibrate") as mock_calibrate:
            app._recalibrate_physics_cb("default", "energy_forecast", "recalibrate_physics", {})
            mock_calibrate.assert_called_once()

    def test_cold_start_gate_logs_warning_when_residual_requested_but_gated(self, caplog):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {"use_physics_residual": True}})
        app.initialize()
        assert app._physics_model.is_cold_start_gated is True  # fresh model, no calibration windows yet
        with caplog.at_level("WARNING"):
            app._effective_use_physics_residual()
        assert any("cold-start" in r.message.lower() or "cold start" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py::TestRecalibratePhysicsService -v`
Expected: FAIL — `register_service` never called with `"energy_forecast/recalibrate_physics"`

- [ ] **Step 3: Write minimal implementation**

In `initialize()`, near the existing `self.register_service("energy_forecast/get_scenario", ...)` call (line 296), add:

```python
        if self._physics_model is not None:
            self.register_service("energy_forecast/recalibrate_physics", self._recalibrate_physics_cb)
```

Add the callback and the cold-start-gate helper as new methods:

```python
    def _recalibrate_physics_cb(self, namespace: str, domain: str, service: str, kwargs: dict) -> None:
        if self._physics_model is None:
            _LOGGER.warning("recalibrate_physics called but physics is not configured — ignoring")
            return
        try:
            self._fetch_physics_sensor_histories()
            energy_df = self._cached_energy_df
            weather_df = self._cached_weather_df
            holdout_cutoff = energy_df["timestamp"].max() - pd.Timedelta(days=max(int(len(energy_df) / 24 * 0.1), 1))
            self._physics_model.calibrate(
                energy_df, weather_df,
                climate_dfs=self._physics_climate_dfs, dhw_df=self._physics_dhw_tank_df,
                holdout_cutoff=holdout_cutoff,
            )
            _LOGGER.info("Physics recalibration complete")
        except Exception as exc:
            _LOGGER.error(f"recalibrate_physics failed: {exc}")

    def _effective_use_physics_residual(self) -> bool:
        """Config intent AND-ed with the cold-start gate. Plan C reads this to decide Phase 1 vs 2."""
        requested = bool(self._physics_config.get("use_physics_residual", False)) if self._physics_model else False
        if requested and self._physics_model.is_cold_start_gated:
            _LOGGER.warning(
                "use_physics_residual=true but cold-start gate not satisfied "
                f"({self._physics_model._calib.get('n_calibration_windows_ua_eff', 0)}/30 UA_eff windows) — "
                "holding at Phase 1"
            )
            return False
        return requested
```

Locate every `self._ml_model.train(`, `self._ml_model.predict(`, and `self._ml_model.shap_summary(` call site (`grep -n "_ml_model\.\(train\|predict\|shap_summary\)(" apps/energy_forecast/energy_forecast.py`) and add `physics_model=self._physics_model` to each call's kwargs. Additionally add `heating_buffer_temp_df=self._physics_heating_buffer_df` to every `self._ml_model.train(` call site (the parameter added to `train()` in Task 4), and `heating_buffer_temp_recent=self._physics_heating_buffer_df` to every `self._ml_model.predict(`/`.shap_summary(` call site (the parameter added in Task 5) — both source from the same fetch, `train()` merges the full history while `predict()`/`shap_summary()` only need the latest reading.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "feat: add recalibrate_physics service and cold-start gate check"
```

---

### Task 8: Full-suite regression + self-review

**Files:** none new — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass, including every pre-existing test in `tests/test_model.py` and `tests/test_energy_forecast.py` (if it exists) — Phase 1 must be fully backward compatible when `physics:` is absent from config.

- [ ] **Step 2: Manually verify absent-config parity**

Run a quick sanity check: train a model with `physics_model=None` and confirm `feature_cols` is byte-identical to what `v0.11.7` produces (no `physics_kwh`, no `heating_buffer_temp`). This is the single most important regression guard for this plan — flag any deviation before merging.

- [ ] **Step 3: Update CHANGELOG.md and README.md**

Use `@changelog-writer` per project CLAUDE.md. Document the new `physics:` config block, `recalibrate_physics` service, and that it's opt-in (absent block = no behavior change).

- [ ] **Step 4: Open PR against `dev`**

Follow `superpowers:finishing-a-development-branch`. Title: `feat: Phase 1 physics-as-feature integration`. This PR should only be mergeable after Plan A's PR has merged into `dev`.
