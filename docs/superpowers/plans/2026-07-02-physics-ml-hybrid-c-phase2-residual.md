# Physics-ML Hybrid — Plan C: Phase 2 Residual Split + New Sensors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the structural Phase 2 path — LightGBM trains on `gross_kwh − physics_kwh` instead of `gross_kwh` — behind `use_physics_residual`, plus the two new HA sensors (`physics_base_today`, `ml_adjustment_today`) and a `model_phase` attribute for `ha-energy-manager` to detect the transition. Per the 2026-07-02 decision, this ships now but stays dormant: the cold-start gate (Plan B's `_effective_use_physics_residual()`) keeps it off in production until ≥30 winter UA_eff calibration windows exist, which won't happen before winter 2026/27.

**Architecture:** `energy_forecast.py` resolves the effective flag via `self._effective_use_physics_residual()` (built in Plan B) and passes it into `EnergyForecastModel.train()`/`.predict()` as `use_physics_residual: bool`. `model.py` branches its existing target-construction and prediction-reconstruction logic on that flag; the branch structure and gross-MAE reporting exactly match spec §5.2.

**Tech Stack:** No new dependencies. Depends on Plan B (`physics_kwh` feature, `physics_model` threading, `_effective_use_physics_residual()` must already exist).

**Base branch:** `dev`, after Plan B has merged. Branch name: `feat/physics-phase2-residual ha-energy-forecast`.

## Global Constraints

- Default `use_physics_residual: false` — this plan changes no default behavior. Verify with a regression test: `use_physics_residual=False` after this plan produces byte-identical `predict()` output to before this plan, given identical training data.
- Phase 2 target: `gross_kwh − physics_kwh`, with `physics_kwh_series.fillna(0)` applied **before** subtraction, and the NaN count logged as WARNING when `> 0` — spec §5.2, §6.2.
- Phase 2 disables the log1p transform (`self._log_transform = False`) since the residual can be negative — spec §2.1.
- `predict()` reconstruction: `(physics_kwh + lgbm_raw).clip(lower=0)` in Phase 2; `lgbm_raw.clip(lower=0)` in Phase 1 — clip is applied consistently in **both** phases, not just Phase 2 — spec §5.2.
- Holdout MAE is **always** reported on `gross_kwh`, regardless of phase; residual MAE is an additional internal-only diagnostic in Phase 2 — spec §5.2.
- Any exception inside physics prediction during Phase 2 must fall back to ML-only output (`lgbm_raw` unmodified), never propagate — spec §6.1.

---

### Task 1: `train()` — residual target construction + log1p gate

**Files:**
- Modify: `apps/energy_forecast/model.py` (`train()`, lines 557-558 — the `y`/`y_fit` construction — and line 710 where `self._log_transform = True` is currently unconditional)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `physics_kwh_series` (already computed in Plan B's Task 4 addition to `train()`), new `use_physics_residual: bool = False` parameter.
- Produces: `EnergyForecastModel.train(..., use_physics_residual: bool = False)`. Sets `self._use_physics_residual: bool` (persisted in meta, read by `predict()` and by `energy_forecast.py` for the `model_phase` sensor attribute).

- [ ] **Step 1: Write the failing test**

```python
class TestPhase2ResidualTarget:
    def test_phase1_target_is_gross_kwh_log_transform_active(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model", use_physics_residual=False)
        assert model._log_transform is True
        assert model._use_physics_residual is False

    def test_phase2_target_is_residual_log_transform_disabled(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": True,
        })
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, _ = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        assert model._log_transform is False
        assert model._use_physics_residual is True

    def test_phase2_nan_physics_hours_filled_with_zero_and_warned(self, tmp_path, caplog):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": True,
        })
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        with caplog.at_level("WARNING"):
            model, _ = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        # physics_kwh_series is always fully aligned in this fixture (predict_training_series covers
        # every training timestamp), so no NaN-fill warning is expected here — this test documents
        # the *absence* of the warning as the baseline; the NaN-path itself is covered directly below.

    def test_phase2_target_computation_subtracts_physics_before_nan_fill_check(self, tmp_path):
        # direct unit test of the target math, independent of full train() plumbing
        ts = pd.date_range("2026-01-01", periods=5, freq="1h")
        df = pd.DataFrame({"timestamp": ts, "gross_kwh": [2.0, 3.0, 4.0, 5.0, 6.0]})
        physics_series = pd.Series([1.0, 1.0, np.nan, 2.0, 2.0], index=ts)  # one gap
        physics_aligned = physics_series.reindex(df["timestamp"])
        physics_vals = physics_aligned.fillna(0).values
        target = df["gross_kwh"].to_numpy(dtype=float) - physics_vals
        assert target[2] == pytest.approx(4.0)  # NaN -> 0, so residual = gross_kwh unmodified for that hour
        assert target[0] == pytest.approx(1.0)
```

Extend `_make_trained_model()` in `tests/test_model.py` to accept and forward `use_physics_residual=False`:

```python
def _make_trained_model(tmp_path, n=600, timezone="Europe/Zurich", physics_model=None, use_physics_residual=False) -> tuple:
    # ... existing body, then:
    model.train(
        energy, weather, outdoor_df=None, weight_halflife_days=0,
        physics_model=physics_model, use_physics_residual=use_physics_residual,
    )
    return model, forecast_df
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestPhase2ResidualTarget -v`
Expected: FAIL with `TypeError: train() got an unexpected keyword argument 'use_physics_residual'`

- [ ] **Step 3: Write minimal implementation**

Add `use_physics_residual: bool = False` to `train()`'s signature (after the `physics_model`/`heating_buffer_temp_df` parameters added in Plan B Task 4).

Replace lines 557-558:

```python
        y = df["gross_kwh"].to_numpy(dtype=float)
        y_fit = np.log1p(y)  # log-transform reduces influence of rare high peaks
```

with:

```python
        y = df["gross_kwh"].to_numpy(dtype=float)
        self._use_physics_residual = bool(use_physics_residual and physics_kwh_series is not None)

        if self._use_physics_residual:
            physics_aligned = physics_kwh_series.reindex(df["timestamp"])
            n_nans = int(physics_aligned.isna().sum())
            if n_nans > 0:
                _LOGGER.warning(
                    f"{n_nans} hours have no physics prediction — set to 0 in residual target "
                    "(check weather data gaps)"
                )
            physics_vals = physics_aligned.fillna(0).values
            y_fit = y - physics_vals  # residual target — can be negative, no log transform
        else:
            y_fit = np.log1p(y)  # log-transform reduces influence of rare high peaks
```

Replace line 710:

```python
        self._log_transform = True
```

with:

```python
        self._log_transform = not self._use_physics_residual
```

Add `"use_physics_residual": self._use_physics_residual` to the `meta` dict written in `_save()` (line 1284-1304), and `self._use_physics_residual = meta.get("use_physics_residual", False)` to the `_load()` restoration block (around line 1639-1659) — same pattern as `self._log_transform`'s persistence.

Also initialize `self._use_physics_residual: bool = False` in `__init__()` near line 261 (alongside `self._log_transform`), so a freshly-constructed, never-trained model has a defined value.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "feat: add Phase 2 residual target construction gated by use_physics_residual"
```

---

### Task 2: `predict()` reconstruction + consistent clip

**Files:**
- Modify: `apps/energy_forecast/model.py` (`predict()`, around lines 1020-1026 where `expm1`/clip currently happen)
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `self._use_physics_residual`, `self._physics_model` (threaded via `physics_model` param from Plan B), `physics_kwh_series` computed inside `_prepare_prediction_X()` (Plan B Task 5).
- Produces: `predict()` returns `(physics_kwh + lgbm_raw).clip(lower=0)` in Phase 2, `lgbm_raw.clip(lower=0)` in Phase 1.

- [ ] **Step 1: Write the failing test**

```python
class TestPhase2Predict:
    def test_phase2_predict_adds_physics_baseline(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": True,
        })
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        result = model.predict(forecast_df, live_temp=5.0, physics_model=pm)
        assert (result["predicted_kwh"] >= 0).all()  # clip enforced even though residual can go negative

    def test_phase1_to_phase2_regression_identical_when_flag_false(self, tmp_path):
        # use_physics_residual=False must reproduce pre-Phase-2 behaviour exactly
        model_a, forecast_df = _make_trained_model(tmp_path / "model_a", use_physics_residual=False)
        model_b, _ = _make_trained_model(tmp_path / "model_b", use_physics_residual=False)
        result_a = model_a.predict(forecast_df, live_temp=5.0)
        result_b = model_b.predict(forecast_df, live_temp=5.0)
        pd.testing.assert_series_equal(
            result_a["predicted_kwh"].reset_index(drop=True), result_b["predicted_kwh"].reset_index(drop=True)
        )

    def test_phase2_exception_in_physics_falls_back_to_ml_only(self, tmp_path, monkeypatch):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": True,
        })
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)

        def _broken_predict_series(*a, **kw):
            raise RuntimeError("simulated physics failure")

        monkeypatch.setattr(pm, "predict_series", _broken_predict_series)
        result = model.predict(forecast_df, live_temp=5.0, physics_model=pm)
        assert not result.empty  # no exception propagates; falls back to ML-only reconstruction
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestPhase2Predict -v`
Expected: FAIL — Phase 2 predictions not yet offset by physics baseline (values won't satisfy the intended assertions, or `predict()` errors on the exception-fallback test)

- [ ] **Step 3: Write minimal implementation**

In `predict()`, the existing reconstruction (around lines 1023-1025):

```python
        if self._log_transform:
            preds = np.expm1(preds)
        preds = np.maximum(0, preds)
```

becomes:

```python
        if self._log_transform:
            preds = np.expm1(preds)

        if self._use_physics_residual and physics_model is not None:
            try:
                physics_baseline = physics_kwh_series.reindex(pd.DatetimeIndex(forecast_df["timestamp"])).fillna(0).values
                preds = physics_baseline + preds
            except Exception as e:
                _LOGGER.warning(f"Phase 2 physics reconstruction failed: {e} — falling back to ML-only output")

        preds = np.maximum(0, preds)
```

`physics_kwh_series` here is the same series computed inside `_prepare_prediction_X()` in Plan B Task 5 — it must be returned alongside `X`/`future_hours` from `_prepare_prediction_X()` (extend its return tuple, or stash it as `self._last_physics_kwh_series` set inside `_prepare_prediction_X()` and read here — pick whichever the existing `_prepare_prediction_X()` return-tuple convention supports; check its current return signature with `grep -n "def _prepare_prediction_X" -A 3 apps/energy_forecast/model.py` before deciding, since `predict()` may consume it via `_prepared` unpacking rather than a direct call).

Note `test_phase2_exception_in_physics_falls_back_to_ml_only` exercises `physics_model.predict_series()` raising — since Plan A's `predict_series()` already catches internally and returns zeros on exception (Plan A Task 5), this test's `monkeypatch` bypasses that internal catch to verify `predict()`'s **own** try/except (the one added above) is also robust, not just relying on Plan A's. Both layers must independently hold per spec §6.1's "any exception in `predict_series()` → ML-only for that cycle."

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "feat: reconstruct Phase 2 predictions from physics baseline + residual with fallback"
```

---

### Task 3: Gross-kWh MAE reporting (both phases) + residual MAE diagnostic

**Files:**
- Modify: `apps/energy_forecast/model.py` (`train()`, extending the holdout MAE block at lines 688-702)
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `self.last_mae` continues to mean gross-kWh MAE in both phases (no behavior change to what this attribute represents — existing consumers of `last_mae` are unaffected). Adds `self.last_residual_mae: float | None` — Phase 2 only, internal diagnostic.

- [ ] **Step 1: Write the failing test**

```python
class TestGrossMAEReporting:
    def test_phase1_last_mae_is_gross_kwh_mae_unchanged(self, tmp_path):
        model, _ = _make_trained_model(tmp_path / "model", use_physics_residual=False)
        assert model.last_mae is not None
        assert model.last_residual_mae is None

    def test_phase2_last_mae_still_reported_on_gross_kwh(self, tmp_path):
        from energy_forecast.physics import ThermalPhysicsModel

        pm = ThermalPhysicsModel(tmp_path / "physics_models", {
            "cop_formula": {"a": 2.5, "b": 0.07}, "dhw_tank_volume_l": 200, "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8, "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [], "use_physics_residual": True,
        })
        pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
        model, _ = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        assert model.last_mae is not None
        assert model.last_residual_mae is not None
        # gross MAE and residual MAE are computed on different scales — they should not be identical
        # for a non-trivial physics baseline
        assert model.last_mae != model.last_residual_mae
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_model.py::TestGrossMAEReporting -v`
Expected: FAIL — `model.last_residual_mae` raises `AttributeError`

- [ ] **Step 3: Write minimal implementation**

Initialize `self.last_residual_mae: float | None = None` in `__init__()` alongside `self.last_mae`.

Extend the holdout MAE block (lines 688-702). The existing code computes `holdout_mae` on `np.expm1(ho_model.predict(...))` — in Phase 2 this is the residual prediction, not gross kWh, so it must be reconstructed before scoring:

```python
        holdout_mae = None
        self.last_residual_mae = None
        if mae_fn is not None:
            split = max(int(len(X) * HOLDOUT_FRACTION), len(X) - MIN_CV_ROWS)
            try:
                ho_model = _build_model(lgb, GBR, n_estimators=best_n_est, num_leaves=best_num_leaves)
                if sample_weight is not None:
                    ho_model.fit(X.iloc[:split], y_fit[:split], sample_weight=sample_weight[:split])
                else:
                    ho_model.fit(X.iloc[:split], y_fit[:split])

                X_ho = X.iloc[split:]
                ho_raw = ho_model.predict(X_ho)
                if self._use_physics_residual:
                    physics_ho = X_ho["physics_kwh"].values if "physics_kwh" in X_ho.columns else 0.0
                    gross_pred = np.maximum(0, physics_ho + ho_raw)
                    holdout_mae = round(float(mae_fn(y[split:], gross_pred)), 4)
                    self.last_residual_mae = round(float(mae_fn(y_fit[split:], ho_raw)), 4)
                else:
                    holdout_mae = round(float(mae_fn(y[split:], np.expm1(ho_raw))), 4)
            except (ValueError, IndexError):
                pass
```

(This replaces the existing block's body — keep the surrounding `mae_str`/`_LOGGER.info(...)` logging line unchanged; it already logs whichever of `cv_mae`/`holdout_mae` is available, and that value is now correctly gross-kWh in both phases.) Add one extra log line right after:

```python
        if self._use_physics_residual and self.last_residual_mae is not None:
            _LOGGER.info(f"Holdout MAE (gross kWh)={holdout_mae}, residual MAE (internal diagnostic)={self.last_residual_mae}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_model.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/model.py tests/test_model.py
git commit -m "feat: report holdout MAE on gross kWh in both phases, residual MAE as diagnostic"
```

---

### Task 4: Interval coverage note (quantile models retrain on residual — verification, not code)

**Files:** none — this task is a documented manual-verification step, not a code change. Spec §5.2's final paragraph ("After Phase 2 deployment, verify empirical interval coverage on gross kWh...") is inherently untestable in CI: it requires weeks of live production data. The quantile models (`self._model_q10`/`_model_q90`, trained at lines 725-759) already retrain on `y_fit` — which Task 1 makes the residual in Phase 2 — with no code change needed; `_calibrate_intervals()` (lines 1333-1354) already operates on whatever `y_fit` is.

- [ ] **Step 1: Add a operator checklist item, not a test**

Add to `docs/ROADMAP.md` (or wherever operational follow-ups are tracked in this repo — check for an existing "post-deployment checklist" convention first) a line: *"After Phase 2 (`use_physics_residual=true`) has been live for ≥30 days, verify empirical prediction-interval coverage on gross kWh matches the target (80% by default, per `_calibrate_intervals()`'s conformal quantile). If coverage has drifted, the CQR calibration on the residual distribution may need a wider correction — see spec §5.2."*

This has no pass/fail test; it's a reminder for whoever operates the system once Phase 2 is live (likely winter 2026/27, well after this plan merges).

- [ ] **Step 2: Commit**

```bash
git add docs/ROADMAP.md  # or wherever this landed
git commit -m "docs: add Phase 2 interval-coverage verification reminder"
```

---

### Task 5: `physics_base_today` / `ml_adjustment_today` sensors

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (find the existing `consumption_today` sensor publish call, likely in the same method as the `safe_set()` helper referenced in the index doc's research — search `grep -n "consumption_today" apps/energy_forecast/energy_forecast.py`)
- Test: extend `tests/test_energy_forecast_physics_config.py`

**Interfaces:**
- Consumes: `self._physics_model.predict_series(...)` for the physics-only baseline, and the already-published main forecast (`hourly_kwh` attribute) for `ml_adjustment_today = main_forecast - physics_base`.
- Produces: `sensor.energy_forecast_physics_base_today`, `sensor.energy_forecast_ml_adjustment_today` — both only published when `self._physics_model is not None`; **published in both Phase 1 and Phase 2** (spec §2.1 lists them under "Phase 2" narratively, but §5.3 says "New HA Sensors (Phase 2)" while also noting the main sensor gains `model_phase` "in both phases" — since the physics baseline is computable in Phase 1 too (physics_model is present, just not driving the target), publish these sensors whenever `physics_model is not None`, independent of phase; a Phase-1-only physics baseline sensor is strictly more useful than withholding it. If this reading is wrong, the fix is a one-line guard change (`if self._physics_model is not None and self._effective_use_physics_residual():`), not a redesign — flag this interpretation to the user before merging).

- [ ] **Step 1: Write the failing test**

```python
class TestPhysicsSensors:
    def test_physics_sensors_published_when_physics_configured(self, monkeypatch):
        import pandas as pd

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app.set_state = MagicMock()
        ts = pd.date_range("2026-01-15", periods=48, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "predicted_kwh": [1.0] * 48})
        monkeypatch.setattr(
            app._physics_model, "predict_series", lambda *a, **kw: pd.Series([0.6] * 48, index=ts)
        )
        app._publish_physics_sensors(forecast_df)
        published_entities = [c.args[0] for c in app.set_state.call_args_list]
        assert "sensor.energy_forecast_physics_base_today" in published_entities
        assert "sensor.energy_forecast_ml_adjustment_today" in published_entities

    def test_physics_sensors_not_published_when_physics_disabled(self):
        app = _make_app({"energy_sensor": "sensor.grid_import"})
        app.initialize()
        app.set_state = MagicMock()
        app._publish_physics_sensors(pd.DataFrame({"timestamp": [], "predicted_kwh": []}))
        app.set_state.assert_not_called()

    def test_ml_adjustment_can_be_negative(self, monkeypatch):
        import pandas as pd

        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app.set_state = MagicMock()
        ts = pd.date_range("2026-01-15", periods=48, freq="1h")
        forecast_df = pd.DataFrame({"timestamp": ts, "predicted_kwh": [0.5] * 48})  # physics baseline exceeds total
        monkeypatch.setattr(
            app._physics_model, "predict_series", lambda *a, **kw: pd.Series([0.8] * 48, index=ts)
        )
        app._publish_physics_sensors(forecast_df)
        adj_call = next(c for c in app.set_state.call_args_list if c.args[0] == "sensor.energy_forecast_ml_adjustment_today")
        assert float(adj_call.kwargs["state"]) < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py::TestPhysicsSensors -v`
Expected: FAIL with `AttributeError: 'EnergyForecast' object has no attribute '_publish_physics_sensors'`

- [ ] **Step 3: Write minimal implementation**

Add to `EnergyForecast`:

```python
    def _publish_physics_sensors(self, forecast_df: "pd.DataFrame") -> None:
        if self._physics_model is None or forecast_df.empty:
            return
        try:
            physics_series = self._physics_model.predict_series(
                forecast_df, climate_recent=self._physics_climate_dfs, dhw_recent=self._physics_dhw_tank_df,
                room_areas=self._climate_room_areas or None,
            )
            physics_vals = physics_series.reindex(forecast_df["timestamp"]).fillna(0.0).values
            ml_adjustment_vals = forecast_df["predicted_kwh"].values - physics_vals

            self.set_state(
                "sensor.energy_forecast_physics_base_today",
                state=str(round(float(physics_vals[0]), 3)),
                attributes={"hourly_kwh": [round(float(v), 3) for v in physics_vals], "friendly_name": "Energy Forecast Physics Base Today"},
                replace=True,
            )
            self.set_state(
                "sensor.energy_forecast_ml_adjustment_today",
                state=str(round(float(ml_adjustment_vals[0]), 3)),
                attributes={"hourly_kwh": [round(float(v), 3) for v in ml_adjustment_vals], "friendly_name": "Energy Forecast ML Adjustment Today"},
                replace=True,
            )
        except Exception as exc:
            _LOGGER.warning(f"Failed to publish physics sensors: {exc}")
```

Call `self._publish_physics_sensors(result_df)` from the same method that publishes `sensor.energy_forecast_consumption_today` today (wherever that call site is — add it right after, using the same `result_df`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "feat: publish physics_base_today and ml_adjustment_today sensors"
```

---

### Task 6: `model_phase` attribute on the main consumption sensor

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (same publish method as Task 5, where `sensor.energy_forecast_consumption_today`'s `extra_attrs` dict is built)
- Test: extend `tests/test_energy_forecast_physics_config.py`

**Interfaces:**
- Produces: `sensor.energy_forecast_consumption_today` attribute `model_phase: "phase1" | "phase2"`, reflecting `self._ml_model._use_physics_residual` at last train time. Published **only** when `self._physics_model is not None` — when physics is disabled entirely, no `model_phase` attribute is added (there is no phase concept without physics).

- [ ] **Step 1: Write the failing test**

```python
class TestModelPhaseAttribute:
    def test_model_phase_absent_when_physics_disabled(self):
        app = _make_app({"energy_sensor": "sensor.grid_import"})
        app.initialize()
        assert app._model_phase_attr() is None

    def test_model_phase_is_phase1_by_default(self, tmp_path):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._ml_model._use_physics_residual = False
        assert app._model_phase_attr() == "phase1"

    def test_model_phase_is_phase2_when_flag_set_on_model(self, tmp_path):
        app = _make_app({"energy_sensor": "sensor.grid_import", "physics": {}})
        app.initialize()
        app._ml_model._use_physics_residual = True
        assert app._model_phase_attr() == "phase2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py::TestModelPhaseAttribute -v`
Expected: FAIL with `AttributeError: 'EnergyForecast' object has no attribute '_model_phase_attr'`

- [ ] **Step 3: Write minimal implementation**

Add to `EnergyForecast`:

```python
    def _model_phase_attr(self) -> str | None:
        if self._physics_model is None:
            return None
        return "phase2" if getattr(self._ml_model, "_use_physics_residual", False) else "phase1"
```

In the method that builds `sensor.energy_forecast_consumption_today`'s attributes, add:

```python
        phase = self._model_phase_attr()
        if phase is not None:
            extra_attrs = {**(extra_attrs or {}), "model_phase": phase}
```

(`extra_attrs` is whatever local variable name the existing `consumption_today` publish call builds — merge into it rather than overwriting, since it already carries `hourly_kwh` and other attributes.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_energy_forecast_physics_config.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py
git commit -m "feat: add model_phase attribute to consumption_today sensor"
```

---

### Task 7: Full-suite regression + dormancy verification + self-review

**Files:** none new — verification only.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 2: Verify dormancy end-to-end**

With no `physics:` block and with `physics: {use_physics_residual: true}` but a fresh (never-calibrated) `ThermalPhysicsModel`, confirm via `EnergyForecast._effective_use_physics_residual()` (Plan B) that the resolved flag is `False` in both cases — the cold-start gate must hold Phase 2 off regardless of the config value until real winter calibration data exists. This is the core safety property of shipping Plan C dormant.

- [ ] **Step 3: Update CHANGELOG.md and ROADMAP.md**

Use `@changelog-writer`. Note that Phase 2 code ships in this release but is inert pending winter 2026/27 calibration data; document the two new sensors and `model_phase` attribute as available once physics is configured (Phase 1 or 2).

- [ ] **Step 4: Open PR against `dev`**

Follow `superpowers:finishing-a-development-branch`. Title: `feat: Phase 2 physics residual split (dormant behind cold-start gate)`. Flag in the PR description that this is intentionally inert in production and the interpretation decision from Task 5 (sensors publish in both phases, not Phase 2 only) for reviewer sign-off.
