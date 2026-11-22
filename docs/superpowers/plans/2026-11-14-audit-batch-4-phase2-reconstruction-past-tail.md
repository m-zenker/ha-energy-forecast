# Phase 2 Reconstruction and Physics Past Tail (ROADMAP #113) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

OWNER-APPROVED docs/superpowers/plans/2026-11-14-audit-batch-4-phase2-reconstruction-past-tail.md 2026-11-15

**Goal:** a Phase 2 (`use_physics_residual`) model returns gross kWh for the point forecast and the intervals on a live-shaped forecast frame, and the serving physics simulation starts at the first predicted hour instead of days in the past.

**Architecture:** three small fixes around `EnergyForecastModel._prepare_prediction_X`: (1) the physics baseline is aligned on the 48 predicted hours, not on the whole weather frame; (2) `predict_intervals()` adds the same baseline; (3) the weather frame handed to `ThermalPhysicsModel.predict_series()` is cut to the predicted hours. The diagnostic sensor publisher gets the same cut and a floor at 0.

**Tech Stack:** Python 3.13, pandas, numpy, LightGBM, pytest.

**Spec:** `ROADMAP.md` #113 (on `train/next`); `docs/analysis/2026-11-13-forecast-improvement-audit.md`; `docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md` §5; index `docs/superpowers/plans/2026-11-14-audit-batch-index.md`.

## Global Constraints

- Base branch `train/next`; branch `fix/phase2-reconstruction`; worktree `.worktrees/phase2-reconstruction`.
- Tests only via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest` (`$PY`).
- `use_physics_residual` stays off everywhere. This phase does not enable Phase 2; it removes the failure that would occur if it were enabled.
- A Phase 1 model (the live case) must produce the same `predicted_kwh` as before for a 48-row forecast frame. For a frame with a past tail, only the `physics_kwh` feature column may change.
- `predict()`'s existing fallback (physics failure → log a warning, publish the ML output) stays as it is.
- No signature change that breaks existing callers: new parameters are keyword-only with default `None`.

## Review Focus

1. Live forecast frame = about 3 days of past rows plus the future (≈ 170 rows), not 48. Every Phase 2 code path must be exercised with such a frame. → Tasks 1–3 tests.
2. Forecast frame that ends before the 48th predicted hour (weather API returned a short horizon): missing baseline hours are filled with 0 as today, no exception. → Task 1 test.
3. Phase 2 model, physics baseline unavailable at interval time (`predict_intervals` called without `_prepared` and without a physics model): return `None`, never residual-scale bands. → Task 2 test.
4. Forecast frame with no row at or after the first predicted hour: physics is skipped with a warning, the ML forecast still returns. → Task 3 test.
5. A pending DHW override delta more negative than the physics baseline: `physics_base_today` shows 0.0, not a negative value. → Task 4 test.

---

## Session setup

```bash
REPO=/home/jovyan/work/ha-energy-forecast
git -C $REPO worktree add $REPO/.worktrees/phase2-reconstruction -b fix/phase2-reconstruction train/next
cd $REPO/.worktrees/phase2-reconstruction
PY=/home/jovyan/my_envs/ha-energy-forecast/bin/python
$PY -m pytest tests/ -q          # baseline: 0 failed
```

## File Structure

- Modify `apps/energy_forecast/model.py`: `_prepare_prediction_X` (≈ 1034–1337), `predict` (≈ 1339–1408), `predict_intervals` (≈ 1410–1466).
- Modify `apps/energy_forecast/energy_forecast.py`: `_publish_physics_sensors` (≈ 2737–2817).
- Modify `tests/test_model.py` (new class `TestPhase2LiveShapedFrame` after `TestPhase2Predict`, ≈ line 7379), `tests/test_energy_forecast.py`.

Background:
- `_prepare_prediction_X` builds `future_hours = pd.date_range(now.floor("1h"), periods=48, freq="1h")` and returns `(future_hours, X)`. It calls `physics_model.predict_series(forecast_df, ...)` and stores the result in `self._last_physics_kwh_series` (indexed by the timestamps of the frame it was given).
- `predict()` then does `physics_kwh_series.reindex(pd.DatetimeIndex(forecast_df["timestamp"]))` and adds it to 48 predictions. With a 170-row frame that raises `operands could not be broadcast together with shapes (170,) (48,)`, is caught, and the bare residual is published. This is the bug.
- `ThermalPhysicsModel.predict_series` seeds the DHW tank simulation at the frame's first timestamp with the latest tank reading. With a past tail, that is a reading from now placed 3 days ago.
- Test helper `_make_trained_model(tmp_path, physics_model=pm, use_physics_residual=True)` in `tests/test_model.py` returns `(model, forecast_df)` with a 48-row frame starting at the current hour. Physics model construction: copy it from `TestPhase2Predict.test_phase2_predict_adds_physics_baseline`.

Shared test helpers for this plan (put them at the top of the new test class's section):

```python
def _phase2_physics(tmp_path):
    from energy_forecast.physics import ThermalPhysicsModel

    pm = ThermalPhysicsModel(
        tmp_path / "physics_models",
        {
            "cop_formula": {"a": 2.5, "b": 0.07},
            "dhw_tank_volume_l": 200,
            "dhw_power_w": 4000,
            "internal_gains_fraction": 0.8,
            "heating_curve_points": [[-20, 55.5], [20, 25.0]],
            "room_thermostats": [],
            "use_physics_residual": True,
        },
    )
    pm._calib.update(UA_eff=150.0, Q_base_el=0.35)
    return pm


def _with_past_tail(forecast_df: pd.DataFrame, hours: int = 122) -> pd.DataFrame:
    """Live-shaped frame: *hours* past rows in front of the 48 future rows (same columns, same values)."""
    first = forecast_df["timestamp"].iloc[0]
    tail = pd.concat([forecast_df.iloc[[0]]] * hours, ignore_index=True)
    tail["timestamp"] = pd.date_range(first - pd.Timedelta(hours=hours), periods=hours, freq="1h")
    return pd.concat([tail, forecast_df], ignore_index=True)
```

---

### Task 1: `predict()` aligns the baseline on the predicted hours

**Files:**
- Modify: `apps/energy_forecast/model.py` (`predict`)
- Test: `tests/test_model.py`

- [ ] **Step 1: Write the failing tests**

```python
class TestPhase2LiveShapedFrame:
    """ROADMAP #113: Phase 2 must work on a forecast frame that carries a past tail (170 rows live)."""

    def test_predict_adds_baseline_on_170_row_frame(self, tmp_path, caplog):
        pm = _phase2_physics(tmp_path)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        long_df = _with_past_tail(forecast_df)
        assert len(long_df) == 170
        with caplog.at_level("WARNING", logger="energy_forecast"):
            result = model.predict(long_df, live_temp=5.0, physics_model=pm)
        assert "Phase 2 physics reconstruction failed" not in caplog.text
        assert len(result) == 48
        baseline = model._last_physics_kwh_series.reindex(pd.DatetimeIndex(result["timestamp"])).fillna(0).to_numpy()
        assert baseline.sum() > 0
        # gross forecast is at least the clipped baseline-plus-residual, and clearly above the bare residual
        raw = model.model.predict(model._prepare_prediction_X(long_df, 5.0, None, physics_model=pm)[1])
        np.testing.assert_allclose(result["predicted_kwh"].to_numpy(), np.maximum(0, baseline + raw), rtol=1e-9)

    def test_short_frame_fills_missing_baseline_hours_with_zero(self, tmp_path, caplog):
        pm = _phase2_physics(tmp_path)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        with caplog.at_level("WARNING", logger="energy_forecast"):
            result = model.predict(forecast_df.iloc[:40], live_temp=5.0, physics_model=pm)
        assert len(result) == 48
        assert "Phase 2 physics reconstruction failed" not in caplog.text
```

- [ ] **Step 2: Run to verify the first fails** — `$PY -m pytest tests/test_model.py::TestPhase2LiveShapedFrame -q`. Expected: both fail on the `caplog` assertion (the warning text is present; shapes `(170,)` vs `(48,)` and `(40,)` vs `(48,)`). Checked against b03b19d on 2026-11-14.

- [ ] **Step 3: Fix `predict()`**

In the `if self._use_physics_residual and physics_model is not None:` block, reindex on the predicted hours instead of the frame: `physics_kwh_series.reindex(pd.DatetimeIndex(future_hours)).fillna(0).values`. Nothing else in the block changes.

- [ ] **Step 4: Run** — both pass; `$PY -m pytest tests/test_model.py -q -k "Phase2 or Physics"` → 0 failed.

- [ ] **Step 5: Commit** — `git commit -am "fix(model): align Phase 2 physics baseline on the predicted hours (#113)"`

---

### Task 2: `predict_intervals()` adds the baseline in Phase 2

**Files:**
- Modify: `apps/energy_forecast/model.py` (`predict_intervals`)
- Test: `tests/test_model.py` (`TestPhase2LiveShapedFrame`)

**Interfaces:**
- Produces: `predict_intervals(..., physics_model: ThermalPhysicsModel | None = None, heating_buffer_temp_recent: pd.DataFrame | None = None, _prepared: tuple | None = None)`. The two new parameters are forwarded to `_prepare_prediction_X` when `_prepared` is None. Return value unchanged in Phase 1. In Phase 2: `low_kwh`/`high_kwh` are gross kWh, or the method returns `None` when the baseline is unavailable.

- [ ] **Step 1: Write the failing tests** (add to `TestPhase2LiveShapedFrame`)

```python
    def test_intervals_add_the_baseline_in_phase2(self, tmp_path):
        pm = _phase2_physics(tmp_path)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        long_df = _with_past_tail(forecast_df)
        prepared = model._prepare_prediction_X(long_df, 5.0, None, physics_model=pm)
        base = model._last_physics_kwh_series
        a = model.predict_intervals(long_df, 5.0, _prepared=prepared)
        model._last_physics_kwh_series = base + 2.0
        b = model.predict_intervals(long_df, 5.0, _prepared=prepared)
        assert a is not None and len(a) == 48
        unclipped = a["high_kwh"].to_numpy() > 0
        assert unclipped.any()
        np.testing.assert_allclose((b["high_kwh"] - a["high_kwh"]).to_numpy()[unclipped], 2.0)
        assert (a["low_kwh"] <= a["high_kwh"]).all() and (a["low_kwh"] >= 0).all()

    def test_intervals_without_prepared_use_the_physics_model(self, tmp_path):
        pm = _phase2_physics(tmp_path)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        point = model.predict(forecast_df, live_temp=5.0, physics_model=pm)
        bands = model.predict_intervals(forecast_df, 5.0, physics_model=pm)
        assert bands is not None
        # gross scale: the upper band is not below the point forecast by a whole baseline
        assert bands["high_kwh"].sum() > 0.5 * point["predicted_kwh"].sum()

    def test_intervals_return_none_when_phase2_baseline_is_missing(self, tmp_path, caplog):
        pm = _phase2_physics(tmp_path)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm, use_physics_residual=True)
        with caplog.at_level("WARNING", logger="energy_forecast"):
            assert model.predict_intervals(forecast_df, 5.0) is None  # no physics model, no _prepared
        assert "Phase 2 intervals skipped" in caplog.text

    def test_phase1_intervals_unchanged(self, tmp_path):
        model, forecast_df = _make_trained_model(tmp_path / "model")
        prepared = model._prepare_prediction_X(forecast_df, 5.0, None)
        a = model.predict_intervals(forecast_df, 5.0, _prepared=prepared)
        model._last_physics_kwh_series = pd.Series(9.0, index=pd.DatetimeIndex(prepared[0]))
        b = model.predict_intervals(forecast_df, 5.0, _prepared=prepared)
        pd.testing.assert_frame_equal(a, b)  # Phase 1 never adds a baseline
```

- [ ] **Step 2: Run to verify they fail** — the first fails (difference is 0, not 2.0), the second with `TypeError: unexpected keyword argument 'physics_model'`, the third returns a frame instead of `None`.

- [ ] **Step 3: Implement**

- Add the two keyword parameters and forward them in the `_prepare_prediction_X` call.
- After the quantile predictions and the existing log-space branch, and **before** `np.maximum(0, ...)`: when `self._use_physics_residual`, read `self._last_physics_kwh_series`; if it is `None`, log `_LOGGER.warning("Phase 2 intervals skipped: physics baseline unavailable")` and `return None`; otherwise add `series.reindex(pd.DatetimeIndex(future_hours)).fillna(0).values` to both `low` and `high`.

- [ ] **Step 4: Run** — class passes; `$PY -m pytest tests/test_model.py -q -k "Interval"` → 0 failed.

- [ ] **Step 5: Commit** — `git commit -am "fix(model): Phase 2 intervals carry the physics baseline (#113)"`

---

### Task 3: Physics serving simulation starts at the first predicted hour

**Files:**
- Modify: `apps/energy_forecast/model.py` (`_prepare_prediction_X`)
- Test: `tests/test_model.py` (`TestPhase2LiveShapedFrame`)

- [ ] **Step 1: Write the failing tests**

```python
    def test_physics_series_is_simulated_from_the_first_predicted_hour(self, tmp_path, monkeypatch):
        pm = _phase2_physics(tmp_path)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)
        seen = {}
        real = pm.predict_series

        def spy(frame, **kw):
            seen["first"] = pd.Timestamp(frame["timestamp"].iloc[0])
            seen["rows"] = len(frame)
            return real(frame, **kw)

        monkeypatch.setattr(pm, "predict_series", spy)
        future_hours, _ = model._prepare_prediction_X(_with_past_tail(forecast_df), 5.0, None, physics_model=pm)
        assert seen["first"] == future_hours[0]
        assert seen["rows"] == 48

    def test_past_tail_does_not_change_the_physics_feature(self, tmp_path):
        pm = _phase2_physics(tmp_path)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)
        _, x_short = model._prepare_prediction_X(forecast_df, 5.0, None, physics_model=pm)
        _, x_long = model._prepare_prediction_X(_with_past_tail(forecast_df), 5.0, None, physics_model=pm)
        np.testing.assert_allclose(x_long["physics_kwh"].to_numpy(), x_short["physics_kwh"].to_numpy())

    def test_frame_without_future_rows_skips_physics(self, tmp_path, caplog):
        pm = _phase2_physics(tmp_path)
        model, forecast_df = _make_trained_model(tmp_path / "model", physics_model=pm)
        past_only = forecast_df.copy()
        past_only["timestamp"] = past_only["timestamp"] - pd.Timedelta(days=5)
        with caplog.at_level("WARNING", logger="energy_forecast"):
            result = model.predict(past_only, live_temp=5.0, physics_model=pm)
        assert len(result) == 48
        assert "no forecast rows at or after the first predicted hour" in caplog.text
```

(These three run in the same clock hour as `_make_trained_model`; if one fails only when the hour rolls over mid-test, rerun before investigating. Existing tests in this file share that property.)

- [ ] **Step 2: Run to verify they fail** — the spy sees the tail's first timestamp and 170 rows; the feature comparison differs; the warning text is absent.

- [ ] **Step 3: Implement**

In `_prepare_prediction_X`, in the block starting `# ── Physics-ML hybrid: physics baseline series at prediction time`: build `_serving_df = forecast_df[pd.to_datetime(forecast_df["timestamp"]) >= future_hours[0]]`. If it is empty, log `_LOGGER.warning("Physics baseline skipped: no forecast rows at or after the first predicted hour")` and leave `physics_kwh_series = None`; otherwise pass `_serving_df` (not `forecast_df`) to `physics_model.predict_series`. Add a two-line comment: the frame carries a past tail (`past_days=3`); simulating over it seeds the tank days ago (same reasoning as the override-delta fix in `energy_forecast.py`, alpha-34). Do not touch `_extended` (the weather look-back frame) or `heating_buffer_temp_series`.

- [ ] **Step 4: Run** — class passes; full suite `$PY -m pytest tests/ -q` → 0 failed. If a `tests/test_scenario_service.py` or `TestScenarioDhwDelta` assertion changes, stop and report it with the numbers instead of adjusting the assertion.

- [ ] **Step 5: Commit** — `git commit -am "fix(model): simulate serving physics from the first predicted hour (#113)"`

---

### Task 4: `_publish_physics_sensors` — same cut, floor at 0

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py` (`_publish_physics_sensors`)
- Test: `tests/test_energy_forecast.py`

- [ ] **Step 1: Write the failing tests**

Find the existing tests of `_publish_physics_sensors` (search that name in `tests/test_energy_forecast.py` and `tests/test_energy_forecast_physics_config.py`) and reuse their fake app. Add:

```python
    def test_physics_base_is_floored_at_zero_with_a_negative_override_delta(self):
        # physics predict_series -> 0.4 kWh for every hour; override_delta = -1.0 at hour 3, 0 elsewhere
        # published "hourly_kwh" of energy_forecast_physics_base_today: [0.4, 0.4, 0.4, 0.0, 0.4, ...]; min >= 0
        # ml_adjustment hourly_kwh[3] == predicted_kwh[3] - 0.0

    def test_physics_sensor_weather_frame_is_cut_to_the_forecast_hours(self):
        # self._cached_forecast_df has 122 past rows + 48 future rows
        # the frame handed to predict_series starts at forecast_df["timestamp"].iloc[0]
```

Write them as real assertions with the fakes those existing tests use.

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement**

- Cut `weather_df` to rows with `timestamp >= forecast_df["timestamp"].iloc[0]` before calling `predict_series` (keep the existing fallback to `forecast_df` when `_cached_forecast_df` is None; if the cut is empty, use the uncut frame).
- After the override delta is added, `physics_vals = np.maximum(0.0, physics_vals)`. `ml_adjustment_vals` is computed after that, so the two still add up to `predicted_kwh`.
- Extend the docstring by one sentence for each.

- [ ] **Step 4: Run** — full suite 0 failed; `ruff check . && ruff format --check .`

- [ ] **Step 5: Commit** — `git commit -am "fix(physics): cut the diagnostic simulation to forecast hours, floor physics base at 0 (#113)"`

---

### Task 5: Capture replay check

No code. This shows the live (Phase 1) forecast is unchanged.

- [ ] **Step 1:** From the main checkout, with this worktree's code and the env that matches the HA host:

  ```bash
  cd /home/jovyan/work/ha-energy-forecast
  W=$PWD/.worktrees/phase2-reconstruction
  PYTHONPATH=$W/apps:$W /home/jovyan/my_envs/ha-energy-forecast-live/bin/python -m backtest.capture_replay data/captures
  ```

  Expected: every `predict` row PASS (max abs difference ≤ 1e-6 kWh/h), as before this branch. Reason: Task 3 changes only the `physics_kwh` feature, which has 0 gain in the live model.
- [ ] **Step 2:** If any row FAILs, do not adjust code to make it pass. Record the capture id and the difference in the hand-back and stop before Task 6: it means `physics_kwh` carries weight in a live model, which changes the deploy sequencing (owner decision). If `data/captures` is missing or the `-live` env is unavailable, say so and continue.

---

### Task 6: Docs and merge to the train

- [ ] **Step 1:** CHANGELOG `[Unreleased]` via `/changelog`: three fixes; no forecast change in Phase 1; `physics_base_today` no longer negative; tag BEHAVIOUR for the serve-time `physics_kwh` feature (now seeded at the current hour).
- [ ] **Step 2:** ROADMAP #113: status → "fixed on `train/next` (date)"; keep the sentence that Phase 2 activation still needs the per-season check on the capture frame; Pending Summary row 113. In "Physics Phase 2 interval-coverage check" note that Phase 2 intervals are still not conformally calibrated (the correction applies only in log space).
- [ ] **Step 3:** `/home/jovyan/work/ha-energy-forecast/MEMORY.md`: one line, including the capture-replay result from Task 5.
- [ ] **Step 4:** Merge and push:
  ```bash
  git -C /home/jovyan/work/ha-energy-forecast/.worktrees/train merge --no-ff fix/phase2-reconstruction
  cd /home/jovyan/work/ha-energy-forecast/.worktrees/train && $PY -m pytest tests/ -q
  git push origin train/next
  ```

**Out of scope:** enabling `use_physics_residual`; calibrating Phase 2 intervals; `predict_scenario`'s use of `committed_override` for the natural baseline (it now inherits the Task 3 cut through `predict()`; its semantics are a cross-repo question with `ha-energy-manager`); #122 (override legs); #94.
