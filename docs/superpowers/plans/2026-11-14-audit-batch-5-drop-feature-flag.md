# Drop-Feature Flag for the Thermal-Pressure Family (ROADMAP #120) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a config key lets the model train without the thermal-pressure features (and, as a second arm, without `heating_buffer_temp`); default is off, and the walk-forward kit can score the shadow comparison with a mechanical verdict.

**Architecture:** `train()` gets a `drop_features` argument that removes names from the feature list after it is assembled; the list is already persisted in `meta.pkl` (`feature_cols`) and used for slicing at predict time, so serving needs no flag. Named groups live in `const.py`; the app maps the `drop_feature_groups` config key to feature names. The backtest arms switch from monkeypatching to the real argument, and `wf_score` gains `shadow_verdict()`.

**Tech Stack:** Python 3.13, pandas, LightGBM, pytest; `backtest.walkforward`, `backtest.wf_score`.

**Spec:** `ROADMAP.md` #120 (on `train/next`); `docs/analysis/2026-11-13-forecast-improvement-audit.md`; index `docs/superpowers/plans/2026-11-14-audit-batch-index.md`.

## Global Constraints

- **Prerequisite:** phase 4 (`fix/phase2-reconstruction`) is merged into `train/next` (both edit `model.py`). Check: `git log train/next --oneline | grep -i "#113"` prints lines. If not, stop.
- Base branch `train/next`; branch `feat/drop-feature-groups`; worktree `.worktrees/drop-feature-groups`.
- Tests only via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest` (`$PY`).
- **Default off, provably:** with no `drop_feature_groups` key, `feature_cols` of a trained model is identical to today's, in the same order.
- Groups, verbatim: `thermal_pressure` = `thermal_pressure`, `thermal_pressure_max`, `thermal_pressure_std`, `thermal_pressure_cop`, `thermal_pressure_net`, `infiltration_pressure`; `heating_buffer` = `heating_buffer_temp`.
- `sensor.energy_forecast_thermal_pressure_net` is consumed by `ha-energy-manager`. Its value must not change when the features are dropped from the model.
- Shadow success criteria, verbatim from ROADMAP #120: hourly and 3 h MAE non-inferior at 2 %; next-day swing at least 25 % lower; cold-day bias not higher.
- This phase does not turn the flag on in the live `apps.yaml` and does not judge the shadow.

## Review Focus

1. A typo in `drop_feature_groups` (`thermal-pressure`): the app must refuse to start with a clear error, not silently train the full model. → Task 3 test.
2. A model trained with the flag, then the flag removed and AppDaemon restarted before the next retrain: serving keeps using the persisted (reduced) `feature_cols`; nothing crashes. → Task 2 test.
3. `heating_buffer` dropped while the buffer sensor is configured: `heating_buffer_temp` must not be re-added to the feature list, and the "sensor unavailable … filling with 0.0" warning must not appear at predict time. → Task 2 test.
4. Dropping a name `train()` does not know (`drop_features=("lag_24h",)`): ValueError naming the feature, not a silent no-op. → Task 2 test.
5. Shadow run with no cold day in the window: verdict INCONCLUSIVE, not PASS. → Task 5 test.

---

## Session setup

```bash
REPO=/home/jovyan/work/ha-energy-forecast
git -C $REPO worktree add $REPO/.worktrees/drop-feature-groups -b feat/drop-feature-groups train/next
cd $REPO/.worktrees/drop-feature-groups
PY=/home/jovyan/my_envs/ha-energy-forecast/bin/python
$PY -m pytest tests/ -q          # baseline: 0 failed
```

## File Structure

- Modify `apps/energy_forecast/const.py` — `FEATURE_GROUPS`.
- Modify `apps/energy_forecast/model.py` — `train()` signature and feature assembly (≈ 334–360, 713–721), `_prepare_prediction_X` end (≈ 1336), `predict` (≈ 1385–1390), `_save`/`_load` meta (≈ 1767–1789, 2216–2238).
- Modify `apps/energy_forecast/energy_forecast.py` — config read (near `self._baseline_mode`, ≈ 346), `_validate_config` (≈ 588), `_retrain` `train_kwargs` (≈ 1958).
- Modify `backtest/walkforward.py`, `backtest/wf_score.py`, `backtest/README.md`.
- Tests: `tests/test_const.py`, `tests/test_model.py`, `tests/test_energy_forecast.py`, `tests/test_backtest_capture_frame.py`, `tests/test_backtest_wf_score.py`.

---

### Task 1: `FEATURE_GROUPS` in `const.py`

**Interfaces:**
- Produces: `FEATURE_GROUPS: dict[str, tuple[str, ...]]` with exactly the two groups from Global Constraints; `resolve_feature_groups(groups: Iterable[str]) -> tuple[str, ...]` — feature names in group order, no duplicates; `ValueError(f"unknown feature group(s) {bad}; known: {sorted(FEATURE_GROUPS)}")` for unknown names.

- [ ] **Step 1: Write the failing tests** (`tests/test_const.py`)

```python
def test_feature_groups_content():
    from energy_forecast.const import FEATURE_GROUPS

    assert FEATURE_GROUPS["thermal_pressure"] == (
        "thermal_pressure", "thermal_pressure_max", "thermal_pressure_std",
        "thermal_pressure_cop", "thermal_pressure_net", "infiltration_pressure",
    )
    assert FEATURE_GROUPS["heating_buffer"] == ("heating_buffer_temp",)


def test_resolve_feature_groups():
    from energy_forecast.const import FEATURE_GROUPS, resolve_feature_groups

    assert resolve_feature_groups([]) == ()
    both = resolve_feature_groups(["thermal_pressure", "heating_buffer", "thermal_pressure"])
    assert both == (*FEATURE_GROUPS["thermal_pressure"], "heating_buffer_temp")
    with pytest.raises(ValueError, match="thermal-pressure"):
        resolve_feature_groups(["thermal-pressure"])
```

- [ ] **Step 2: Run to verify they fail** — ImportError.
- [ ] **Step 3: Implement** both in `const.py`.
- [ ] **Step 4: Run** — pass.
- [ ] **Step 5: Commit** — `git commit -am "feat(const): named feature groups for drop_feature_groups (#120)"`

---

### Task 2: `train(drop_features=...)`

**Files:**
- Modify: `apps/energy_forecast/model.py`
- Test: `tests/test_model.py` (new class `TestDropFeatures` at the end)

**Interfaces:**
- Consumes: nothing from Task 1 (the model takes feature names, not group names).
- Produces: `train(..., drop_features: Sequence[str] = ())`; attribute `self._drop_features: tuple[str, ...]` (what the loaded model was trained without); meta key `"drop_features"` (list, default `[]` on load).

Droppable names: every name in `_FEATURES_BASE` that is not a `lag_*h` column, plus `heating_buffer_temp`. Anything else raises `ValueError(f"cannot drop {bad}: not a droppable feature")` at the top of `train()`, next to the `regime_mode` validation.

- [ ] **Step 1: Write the failing tests**

```python
class TestDropFeatures:
    """ROADMAP #120: train() can leave out named features; default is unchanged."""

    TP = (
        "thermal_pressure", "thermal_pressure_max", "thermal_pressure_std",
        "thermal_pressure_cop", "thermal_pressure_net", "infiltration_pressure",
    )

    def _train(self, tmp_path, **kw):
        energy, weather = train_frames()
        m = EnergyForecastModel(tmp_path, timezone="Europe/Zurich")
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0, **kw)
        return m

    def test_default_feature_list_is_unchanged(self, tmp_path):
        a = self._train(tmp_path / "a")
        b = self._train(tmp_path / "b", drop_features=())
        assert a.feature_cols == b.feature_cols
        assert set(self.TP) <= set(a.feature_cols)
        assert a._drop_features == ()

    def test_dropped_features_leave_the_model(self, tmp_path):
        full = self._train(tmp_path / "a")
        m = self._train(tmp_path / "b", drop_features=self.TP)
        assert not set(self.TP) & set(m.feature_cols)
        assert m.feature_cols == [c for c in full.feature_cols if c not in self.TP]  # order preserved
        assert m._drop_features == self.TP

    def test_unknown_or_lag_feature_raises(self, tmp_path):
        with pytest.raises(ValueError, match="lag_24h"):
            self._train(tmp_path / "a", drop_features=("lag_24h",))
        with pytest.raises(ValueError, match="no_such_feature"):
            self._train(tmp_path / "b", drop_features=("no_such_feature",))

    def test_drop_survives_save_and_load_and_predicts(self, tmp_path):
        m = self._train(tmp_path / "m", drop_features=self.TP)
        reloaded = EnergyForecastModel(tmp_path / "m", timezone="Europe/Zurich")
        assert reloaded._drop_features == self.TP
        assert reloaded.feature_cols == m.feature_cols
        _, forecast = _make_trained_model(tmp_path / "other")  # only for a 48 h forecast frame
        out = reloaded.predict(forecast, live_temp=None)
        assert len(out) == 48 and (out["predicted_kwh"] >= 0).all()

    def test_heating_buffer_drop_is_not_re_added(self, tmp_path, caplog):
        energy, weather = train_frames()
        buf = pd.DataFrame({"timestamp": energy["timestamp"], "heating_buffer_temp": 32.0})
        m = EnergyForecastModel(tmp_path, timezone="Europe/Zurich")
        m.train(energy, weather, outdoor_df=None, weight_halflife_days=0,
                heating_buffer_temp_df=buf, drop_features=("heating_buffer_temp",))
        assert "heating_buffer_temp" not in m.feature_cols
        _, forecast = _make_trained_model(tmp_path / "other")
        with caplog.at_level("WARNING", logger="energy_forecast"):
            m.predict(forecast, live_temp=None)
        assert "heating_buffer_temp in trained feature list" not in caplog.text

    def test_meta_without_drop_features_loads_as_empty(self, tmp_path):
        m = self._train(tmp_path / "m")
        import pickle

        meta_path = tmp_path / "m" / "meta.pkl"
        meta = pickle.loads(meta_path.read_bytes())
        meta.pop("drop_features")
        meta_path.write_bytes(pickle.dumps(meta))
        from energy_forecast.model import _write_hash

        _write_hash(meta_path)
        assert EnergyForecastModel(tmp_path / "m", timezone="Europe/Zurich")._drop_features == ()
```

Before writing `test_heating_buffer_drop_is_not_re_added`, check in `TestHeatingBufferSmoothing` how a buffer frame is passed so `heating_buffer_temp` really enters `feature_cols` without the drop (add that as the first assertion of the test, on a second model trained without `drop_features`). Adjust the frame construction to match; keep the assertions.

- [ ] **Step 2: Run to verify they fail** — `TypeError: train() got an unexpected keyword argument 'drop_features'`.

- [ ] **Step 3: Implement**

- Signature: add `drop_features: Sequence[str] = ()` after `regime_oof_cv`. Validate at the top.
- Feature assembly (search `base_features = [c for c in _FEATURES_BASE if c not in all_lag_cols]`): after `physics_kwh` and `heating_buffer_temp` have been appended to `base_features` and before `sensor_features` is built, filter: `base_features = [c for c in base_features if c not in drop_set]`.
- `__init__`: `self._drop_features: tuple[str, ...] = ()`. In `train()`, where `self.feature_cols = feature_cols` is set: `self._drop_features = tuple(drop_features)`.
- `_save`: add `"drop_features": list(self._drop_features)` to the meta dict. `_load`: `self._drop_features = tuple(meta.get("drop_features", []))`.
- Log one INFO line at train time when the tuple is non-empty: `"Training without %d dropped feature(s): %s"`.

- [ ] **Step 4: Run** — class passes; full suite 0 failed.
- [ ] **Step 5: Commit** — `git commit -am "feat(model): train(drop_features=...) with persisted drop list (#120)"`

---

### Task 3: Keep the thermal-pressure sensor alive; app config key

**Files:**
- Modify: `apps/energy_forecast/model.py` (`_prepare_prediction_X`, `predict`)
- Modify: `apps/energy_forecast/energy_forecast.py`
- Test: `tests/test_model.py` (`TestDropFeatures`), `tests/test_energy_forecast.py`

**Interfaces:**
- Consumes: `const.resolve_feature_groups` (Task 1), `train(drop_features=)` (Task 2).
- Produces: config key `drop_feature_groups` (list of group names, default `[]`); `self._drop_features: tuple[str, ...]` on the app; `train_kwargs["drop_features"]` in `_retrain`.

Problem to solve first: `predict()` reads `self._latest_thermal_pressure_net` from the model matrix `X` and falls back to `0.0` when the column is missing. With the family dropped, the published `sensor.energy_forecast_thermal_pressure_net` would go to 0.

- [ ] **Step 1: Write the failing tests**

```python
    # in TestDropFeatures
    def test_thermal_pressure_sensor_value_survives_the_drop(self, tmp_path, monkeypatch):
        import energy_forecast.model as mm

        m = self._train(tmp_path / "m", drop_features=self.TP)
        real = mm._engineer_features

        def fake(*a, **kw):
            df = real(*a, **kw)
            df["thermal_pressure_net"] = 1.23
            return df

        monkeypatch.setattr(mm, "_engineer_features", fake)
        _, forecast = _make_trained_model(tmp_path / "other")
        m.predict(forecast, live_temp=None)
        assert m._latest_thermal_pressure_net == pytest.approx(1.23)
```

`tests/test_energy_forecast.py` (reuse `_make_app(args)` at the top of the file; if it needs more required args, copy them from a neighbouring test):

```python
def test_drop_feature_groups_config_maps_to_feature_names():
    app = _make_app({..., "drop_feature_groups": ["thermal_pressure"]})
    assert app._drop_features == (
        "thermal_pressure", "thermal_pressure_max", "thermal_pressure_std",
        "thermal_pressure_cop", "thermal_pressure_net", "infiltration_pressure",
    )


def test_drop_feature_groups_default_is_empty():
    assert _make_app({...})._drop_features == ()


def test_unknown_drop_feature_group_fails_startup():
    with pytest.raises(ValueError, match="thermal-pressure"):
        _make_app({..., "drop_feature_groups": ["thermal-pressure"]})
```

Plus one test that `_retrain` passes the value on: find an existing test that inspects the kwargs `_retrain` gives to `train()` (search `train.call_args` or `heating_feature_df` in `tests/test_energy_forecast.py`) and add an assertion `kwargs["drop_features"] == app._drop_features` in a new test built the same way.

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement**

- `_prepare_prediction_X`: immediately before `X = feat_df[self.feature_cols].fillna(0)`, set `self._latest_thermal_pressure_net = float(feat_df["thermal_pressure_net"].iloc[0]) if "thermal_pressure_net" in feat_df.columns else 0.0` (use `0.0` for a NaN value too).
- `predict()`: delete the `if "thermal_pressure_net" in X.columns: ... else: ...` block. The value is now set where the features are built, for every caller.
- App `initialize`, next to `self._baseline_mode`: `self._drop_features: tuple[str, ...] = resolve_feature_groups(self.args.get("drop_feature_groups") or [])` (import from `.const` like the file's other const imports). A ValueError here stops startup, which is intended; if `_validate_config` is where the file raises config errors, do the resolve there instead and keep the attribute default `()` in `initialize`.
- Startup log line (search `baseline_mode=%s`): add `drop_feature_groups=%s`.
- `_retrain`: add `drop_features=self._drop_features,` to `train_kwargs`.

- [ ] **Step 4: Run** — full suite 0 failed; `ruff check . && ruff format --check .`

  Also run the capture kit's tests explicitly, since retrain captures pickle `train_kwargs`: `$PY -m pytest tests/test_capture.py tests/test_energy_forecast_capture.py tests/test_backtest_capture_replay.py tests/test_backtest_capture_frame.py -q` → 0 failed. Old captures have no `drop_features` key; `capture_frame.train` merges captured kwargs with overrides, so the default applies.

- [ ] **Step 5: Commit** — `git commit -am "feat(app): drop_feature_groups config key; thermal pressure sensor independent of the model matrix (#120)"`

---

### Task 4: Walk-forward arms use the real argument

**Files:**
- Modify: `backtest/walkforward.py`
- Test: `tests/test_backtest_capture_frame.py`

**Interfaces:**
- Produces: `ARMS["droptp"]` = `WfArm("droptp", train_kwargs={"drop_features": FEATURE_GROUPS["thermal_pressure"]})`; new `ARMS["droptp_nb"]` = same plus `"heating_buffer_temp"`. The module constant `THERMAL_PRESSURE` becomes `FEATURE_GROUPS["thermal_pressure"]` (imported from `energy_forecast.const`; `backtest.env` puts `apps/` on the path, as `backtest/arms.py` shows).

- [ ] **Step 1: Update the test** — the existing assertion `assert "thermal_pressure" in ARMS["droptp"].drop` (≈ line 123) becomes:

```python
    assert ARMS["droptp"].drop == ()
    assert "thermal_pressure" in ARMS["droptp"].train_kwargs["drop_features"]
    assert "heating_buffer_temp" in ARMS["droptp_nb"].train_kwargs["drop_features"]
    assert set(ARMS["droptp"].train_kwargs["drop_features"]) < set(ARMS["droptp_nb"].train_kwargs["drop_features"])
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement.** Keep the `drop=` field and the `feature_drop` patch path on `WfArm` for ad-hoc experiments; only the two named arms change. Update the README command to list `droptp_nb`.
- [ ] **Step 4: Run** — `$PY -m pytest tests/test_backtest_capture_frame.py tests/test_backtest_cli.py -q` → 0 failed.
- [ ] **Step 5: Commit** — `git commit -am "feat(backtest): droptp arms train through drop_features; add droptp_nb (#120)"`

---

### Task 5: `wf_score.shadow_verdict`

**Files:**
- Modify: `backtest/wf_score.py`
- Test: `tests/test_backtest_wf_score.py`

**Interfaces:**
- Consumes: `paired(table, arm, ref, metric)`, `summarize(table)`, `next_day_swing(rows)` (all existing).
- Produces: `shadow_verdict(table: pd.DataFrame, swing: pd.DataFrame, arm: str, ref: str = "base", ni_pct: float = 2.0, swing_drop_pct: float = 25.0) -> dict` with keys `mae_pct`, `mae3_pct`, `swing_pct`, `cold_bias_arm`, `cold_bias_ref`, `checks` (dict of the four booleans `mae`, `mae3`, `swing`, `cold_bias`; a check that cannot be computed is `None`), `verdict`.

Rules: `mae` passes when `paired(..., "mae")["pct"] <= ni_pct`; `mae3` likewise; `swing_pct = 100 × (swing.loc[arm, "range"] / swing.loc[ref, "range"] − 1)` passes when `<= -swing_drop_pct`; `cold_bias` passes when `abs(cold_bias_arm) <= abs(cold_bias_ref)`. `verdict` = `"INCONCLUSIVE"` if any check is `None`, else `"PASS"` if all are True, else `"FAIL"`.

- [ ] **Step 1: Write the failing tests**

```python
def _shadow_inputs(arm_mae=1.0, arm_mae3=1.0, arm_bias=0.5, arm_range=0.6, cold=True):
    days = pd.date_range("2026-11-15", periods=14, freq="D")

    def block(name, mae, mae3, bias):
        return pd.DataFrame(
            {"n": 24, "mae": mae, "mae3": mae3, "mae6": 2 * mae3, "bias": bias, "y": 24.0, "cold": cold},
            index=pd.MultiIndex.from_product([[name], days], names=["arm", "day"]),
        )

    table = pd.concat([block("base", 1.0, 1.0, 1.0), block("droptp", arm_mae, arm_mae3, arm_bias)])
    swing = pd.DataFrame({"step": [0.5, 0.3], "range": [1.0, arm_range], "range_p90": [2.0, 1.2]}, index=["base", "droptp"])
    return table, swing


def test_shadow_verdict_pass():
    v = ws.shadow_verdict(*_shadow_inputs(), arm="droptp")
    assert v["verdict"] == "PASS" and all(v["checks"].values())
    assert v["swing_pct"] == pytest.approx(-40.0)


def test_shadow_verdict_fails_each_criterion():
    assert ws.shadow_verdict(*_shadow_inputs(arm_mae=1.03), arm="droptp")["checks"]["mae"] is False
    assert ws.shadow_verdict(*_shadow_inputs(arm_mae3=1.03), arm="droptp")["checks"]["mae3"] is False
    assert ws.shadow_verdict(*_shadow_inputs(arm_range=0.8), arm="droptp")["checks"]["swing"] is False
    v = ws.shadow_verdict(*_shadow_inputs(arm_bias=-1.5), arm="droptp")
    assert v["checks"]["cold_bias"] is False and v["verdict"] == "FAIL"


def test_shadow_verdict_inconclusive_without_cold_days():
    v = ws.shadow_verdict(*_shadow_inputs(cold=False), arm="droptp")
    assert v["checks"]["cold_bias"] is None and v["verdict"] == "INCONCLUSIVE"
```

(Use the module alias the test file already imports `wf_score` under.)

- [ ] **Step 2: Run to verify they fail** — AttributeError.
- [ ] **Step 3: Implement** to the rules above. One-line docstring citing ROADMAP #120 and stating that `cold_bias` compares absolute values.
- [ ] **Step 4: Run** — full suite 0 failed; ruff clean.
- [ ] **Step 5: Commit** — `git commit -am "feat(backtest): shadow_verdict for the #120 criteria"`

---

### Task 6: Docs, shadow runbook, merge to the train

- [ ] **Step 1:** `README.md` config reference: `drop_feature_groups` (list; groups `thermal_pressure`, `heating_buffer`; default empty; takes effect at the next retrain; removing it also takes effect at the next retrain). Mark it experimental.
- [ ] **Step 2:** `backtest/README.md`, new section "Shadow check for #120" with this runbook, verbatim:

  ```bash
  # needs captures covering >= 14 target days after 2026-11-12 (the earlier holdout is spent)
  # start one week before the first scored day (burn-in)
  PYTHONPATH=apps:. /home/jovyan/my_envs/ha-energy-forecast-live/bin/python -m backtest.walkforward \
      data/captures --start <first scored day - 7 d> --end <last day> \
      --arms base jit droptp droptp_nb --out data/backtest/wf_shadow
  ```
  ```python
  from backtest import wf_score as ws
  rows = ws.attach_actuals(rows, frame.actuals(), daily_temp)
  rows = rows[rows["day"] >= "<first scored day>"]
  table = ws.day_table(ws.add_naive(ws.day_ahead(rows)))
  swing = ws.next_day_swing(rows)
  for arm in ("droptp", "droptp_nb", "jit"):
      print(arm, ws.shadow_verdict(table, swing, arm))
  ```
  State: `jit` is the noise floor (an arm whose numbers are no further from `base` than `jit` shows no effect); PASS on `droptp` is the trigger for the owner to set `drop_feature_groups: [thermal_pressure]`; the flip is a forecast-changing deploy and gets its own window.
- [ ] **Step 3:** CHANGELOG `[Unreleased]` via `/changelog` (new opt-in config key; default behaviour unchanged).
- [ ] **Step 4:** ROADMAP #120: status → "flag and shadow tooling on `train/next` (date); shadow not started — needs ≥ 14 capture days after 2026-11-12"; Pending Summary row 120.
- [ ] **Step 5:** `/home/jovyan/work/ha-energy-forecast/MEMORY.md`: one line with the earliest shadow date (2026-11-27 at the earliest, if captures are complete from 11-13).
- [ ] **Step 6:** Merge and push:
  ```bash
  git -C /home/jovyan/work/ha-energy-forecast/.worktrees/train merge --no-ff feat/drop-feature-groups
  cd /home/jovyan/work/ha-energy-forecast/.worktrees/train && $PY -m pytest tests/ -q
  git push origin train/next
  ```

**Out of scope:** running or judging the shadow; setting the key in the live `apps.yaml`; removing the thermal-pressure feature code or the sensor; #108.
