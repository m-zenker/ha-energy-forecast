# Physics Heating Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the zero-value physics parts (`physics_kwh`, Phase 2, `UA_eff` heating calibration, the two diagnostic sensors, `model_phase`) from hef, keep the DHW override correction, and make EM always add its own legionella cycle energy.

**Architecture:** Pure removal. Order: `model.py` API and its callers → `physics.py` heating code → hef app surface (sensors, config) → docs and verification → train merge. EM is a separate branch in its own repo. `train(physics_model=...)` keeps today's order. No new mechanisms beyond one startup discovery-clear publish.

**Tech Stack:** Python 3.13, AppDaemon 4.x, LightGBM, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-physics-heating-removal-design.md` (commit a38c02c)

**Base branches:**
- hef: `feat/physics-heating-removal`, already cut from `train/next`, worktree `ha-energy-forecast/.worktrees/physics-removal`.
- EM: new `fix/hef-physics-removal`, cut from EM `train/next`, in worktree `ha-energy-manager/.worktrees/hef-physics-removal`.

## Global Constraints

- Tests run only through the dedicated envs: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v` (hef), `/home/jovyan/my_envs/ha-energy-manager/bin/python -m pytest tests/ -v` (EM). Full suite after every task.
- `ruff check . && ruff format --check .` clean before every commit (pre-commit runs both).
- Forecast neutrality: the predict round trip of the capture replay stays at max |Δ| ≤ 1e-9 (spec §6.3).
- Keep unchanged: `train()`'s order (override delta → subtract → τ (`:484`) → `calibrate()` (`:506`) → `detect_open_windows` → weights), `compute_*_override_delta`, `_dhw_kwh_series`, `override_history`, `set_dhw_schedule`, `recalibrate_physics`, open-window detection, and the `get_scenario` / `set_dhw_schedule` service names and call shapes.
- Keep config keys: `room_thermostats`, `heating_buffer_temp_sensor`, `heating_sub_meter_sensor` (read by `heating_season.py`), `dhw_tank_temp_sensor`, `cop_formula`, `heating_curve_points`, `dhw_tank_volume_l`, `dhw_power_w`.
- Do not touch untracked files under `scripts/` (gitignored, so deletion cannot be undone).
- **Accepted live behaviour change (owner, 2026-11-16):** EM's legionella boost hand-off gates on v2 live (`heat_pump.py:5583`, `:5612`). Once hef stops publishing `model_phase`, v2 adds the cycle energy (1.5 + 3.5 kWh), with or without the EM change. A replay of the 3 past hand-offs: 10-15 5.54→1.39 pass, 10-31 3.34→0.72 pass, 11-09 0.94→0.20 **pass→below_min**. This goes in both CHANGELOGs and the deploy note.
- Gitignored memory lives in the main checkout only: `/home/jovyan/work/ha-energy-forecast/MEMORY.md`, `/home/jovyan/work/ha-energy-forecast/memory/`.
- No deploy, no tag, no push to `dev` or `main`. The plan ends with a `train/next` merge (hef: only after #124 has shipped). CHANGELOG via the `/changelog` skill on the main agent.
- Commit messages: single line, ending with the co-author lines from the session's attribution reminder.

## Review Focus

1. **A saved model whose `feature_cols` contain `physics_kwh`** (the live model until the next retrain) predicts exactly as before and logs no WARNING per cycle. Test: Task 1 `test_legacy_physics_kwh_model_predicts_without_warning`.
2. **A live `apps.yaml` that still has `use_physics_residual`, `cop_sensor`, `internal_gains_fraction`**: the app starts, nothing crashes, `cop_sensor` is not fetched. Test: Task 3 `test_removed_physics_keys_are_ignored`.
3. **An existing `physics_calibration.json` with `UA_eff` / `solar_gain_area` / `n_calibration_windows_ua_eff`** loads, and the DHW override delta still computes. Test: Task 2 `test_legacy_calibration_file_loads_and_dhw_delta_works`.
4. **An old capture whose `train_kwargs` contain `use_physics_residual` / `heating_sub_meter_df`** still replays through `capture_replay` and `CaptureFrame`. Test: Task 1 `test_strip_legacy_train_kwargs` + `test_capture_frame_strips_legacy_train_kwargs`.
5. **EM reads a hef `today` sensor that still carries a stale retained `model_phase`** (minutes after the hef deploy): v2 still adds the cycle and the bridge does not forward `model_phase`. Tests: Task 6 `test_v2_detail_has_no_physics_present`, Task 7 `test_model_phase_never_forwarded`.

## File map

| File | Change | Why (spec §) |
|---|---|---|
| `apps/energy_forecast/model.py` | drop `physics_kwh`, Phase 2, Phase 1 diagnostic, serve-time physics args, `train()` params `use_physics_residual` / `heating_sub_meter_df`; fill-0 log → DEBUG | §4.2 |
| `apps/energy_forecast/physics.py` | drop heating API, `COLD_START_MIN_WINDOWS`, `calibrate()` heating inputs and keys | §4.1 |
| `apps/energy_forecast/const.py` | drop `UA_EFF_*` | §4.4 |
| `apps/energy_forecast/energy_forecast.py` | call sites (Task 1); physics sensors (Task 2, they call `predict_series`); `model_phase`, `cop_sensor`, `heating_curve_sensor` (no reader), config defaults, retire discovery, `get_scenario` ignores `dhw_schedule` (Task 3) | §4.3 |
| `backtest/env.py` | add `strip_legacy_train_kwargs` (shared by replay and frame) | §4.4 |
| `backtest/{capture_replay,capture_frame,audit_features,walkforward,arms}.py`, `backtest/README.md` | drop serve-time physics args | §4.4 |
| `scripts/backtest_regime.py` | drop `physics_model` from predict inputs (`:394`) | §4.4 |
| `scripts/analyze_cop_scale.py`, `tests/test_analyze_cop_scale.py` | delete | §4.4 |
| `tests/test_physics_removal.py` | new: all new hef tests | §6 |
| tests of removed behaviour | delete or adapt | §4.4 |
| README, CHANGELOG, ROADMAP, 2026-06-22 spec, MEMORY | docs | §4.4 |
| EM `loads/legionella_coverage.py`, `loads/heat_pump.py` | v2 always adds cycle; delete `_legionella_physics_present`, `physics_present` | §5 |
| EM `forecast/consumption_forecast_app.py` | drop mirrors and `model_phase`; clear current discovery ids | §5 |
| EM `dashboard/ha-energy-manager.yaml` (`model_phase` row only; there are no physics sensor cards), `config.example.yaml`, `CLAUDE.md`, CHANGELOG | cleanup | §5 |

---

### Task 1: `model.py` API and all its callers

**Files:**
- Modify: `apps/energy_forecast/model.py` (`train` 344-1067, `_prepare_prediction_X` 1069-1386, `predict` 1388-1455, `predict_intervals` 1457-1528, `predict_scenario` 1530-1615, `_validate_physics_phase1` 1617-1667, `shap_summary` 1669-1720, `_engineer_features` 3244-3630, meta save `:1837-1838`, load `:2287-2288`, `__init__` `:282`, `:288-289`)
- Modify: `apps/energy_forecast/energy_forecast.py` call sites:
  - `:1298-1300` (`predict_scenario`)
  - `:1998-2005` (`train`)
  - `:2246` (`_prepare_prediction_X`)
  - `:2252` (`predict_intervals`)
  - `:2442` (`shap_summary`)
  - delete `_effective_use_physics_residual` (`:1624-1644`)
- Modify: `backtest/env.py`, `backtest/capture_replay.py` (`:112-115`, `:164`), `backtest/capture_frame.py` (`:62`, `:257-275`), `backtest/audit_features.py` (`:26` unused `_retarget_physics` import, `:82-84`), `backtest/walkforward.py` (`:47`, `:126`), `backtest/arms.py` (`:29` docstring), `backtest/README.md` (`:27`), `scripts/backtest_regime.py` (`:393-394`)
- Test: `tests/test_physics_removal.py` (new); adapt `tests/test_model.py` (incl. `:1167`, `:1889`, `:2926`), `tests/test_backtest_capture_replay.py`, `tests/test_backtest_capture_frame.py`, `tests/test_scenario_service.py`, `tests/test_backtest_arms.py`

**Interfaces:**
- Produces:
  - `EnergyForecastModel.train(..., physics_model: ThermalPhysicsModel | None = None, ...)` without `use_physics_residual` / `heating_sub_meter_df`.
  - `predict`, `predict_intervals`, `_prepare_prediction_X`, `predict_scenario`, `shap_summary` without `physics_model` / `dhw_schedule_override` (`heating_buffer_temp_recent` stays).
  - `backtest.env.strip_legacy_train_kwargs(kwargs: dict) -> dict`, which returns a copy without `use_physics_residual` and `heating_sub_meter_df`.
  - `capture_frame.predict(frame, model, work, origin, kwargs=...)` without `physics`; `WfArm` without `physics`.

- [ ] **Step 1: Record the baseline before any edit**

Run (worktree root): `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -q 2>&1 | tail -3` and `PYTHONPATH=apps:. /home/jovyan/my_envs/ha-energy-forecast/bin/python -m backtest.capture_replay /home/jovyan/work/ha-energy-forecast/data/captures`
Expected: suite green. Save the full replay table to `/tmp/claude-1000/replay-baseline.txt`. Run it in the background; six retrain replays are slow.
- Predict rows are the neutrality yardstick. Retrain rows may already sit outside the 0.1 % tolerance; record them as they are.
- If the captures dir has no checkable captures, run `python scripts/pull_ha_data.py` from the main checkout first.

- [ ] **Step 2: Write the failing tests in `tests/test_physics_removal.py`**

```python
import inspect, logging
from energy_forecast.model import EnergyForecastModel
from backtest.env import strip_legacy_train_kwargs

SERVE = ("predict", "predict_intervals", "_prepare_prediction_X", "predict_scenario", "shap_summary")

def test_serving_api_has_no_physics_args():
    for name in SERVE:
        params = inspect.signature(getattr(EnergyForecastModel, name)).parameters
        assert "physics_model" not in params and "dhw_schedule_override" not in params, name

def test_train_keeps_physics_model_only():
    params = inspect.signature(EnergyForecastModel.train).parameters
    assert "physics_model" in params
    assert "use_physics_residual" not in params and "heating_sub_meter_df" not in params

def test_trained_model_has_no_physics_kwh(...):  # reuse test_model.py's physics-enabled train fixture
    assert "physics_kwh" not in model.feature_cols
    assert model._log_transform is True
    assert not hasattr(model, "physics_phase1_diagnostic") and not hasattr(model, "_use_physics_residual")

def test_legacy_physics_kwh_model_predicts_without_warning(monkeypatch, caplog, ...):  # replaces test_model.py:1889
    # Build a live-like legacy model: during train(), monkeypatch energy_forecast.model._FEATURES_BASE to
    # include "physics_kwh" and wrap _engineer_features to add a constant column physics_kwh=0.144
    # (constant -> 0 splits, like the live model). Then, without the patches:
    #   X = model._prepare_prediction_X(...)[1]; assert (X["physics_kwh"] == 0.0).all()
    #   preds equal predictions from the same model with X["physics_kwh"] = 0.144 (0 splits -> identical)
    #   assert no caplog record with levelno >= logging.WARNING mentions "physics_kwh"

def test_strip_legacy_train_kwargs():
    out = strip_legacy_train_kwargs({"use_physics_residual": False, "heating_sub_meter_df": None, "a": 1})
    assert out == {"a": 1}

def test_capture_frame_strips_legacy_train_kwargs(...):  # build CaptureFrame from a payload whose
    # train_kwargs carry both legacy keys; frame.train_kwargs(cutoff) has neither key.
```

- [ ] **Step 3: Run the new tests to see them fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics_removal.py -v`
Expected: FAIL (`physics_model` still in signatures; `strip_legacy_train_kwargs` import error).

- [ ] **Step 4: Change `model.py`**

In `train()`, delete:
- the `predict_training_series` call (`:518`), the `check_zone_boundary` call (`:500`) and the `heating_sub_meter_df=` argument to `calibrate()` (`:513`);
- the `physics_kwh_series` plumbing into `_engineer_features` (`:721`) and `feature_cols` (`:745-746`);
- every `_use_physics_residual` branch (`:801-815`, `:866-871`, `:925-930`, `:983-985`, `:1015-1017`) and the Phase 1 diagnostic block (`:1020-1031`).

Then:
- Set `self._log_transform = True`.
- Keep `physics_model._tau_hours = self._tau_hours` (`:517`); `detect_open_windows` reads it.
- In `_prepare_prediction_X`, delete the physics baseline block and `_last_physics_kwh_series`. Keep the fill-0 line for a legacy `physics_kwh` column, with `_LOGGER.debug`.
- Delete the Phase 2 blocks in `predict` (`:1444`) and `predict_intervals` (`:1516`).
- `shap_summary`: drop `physics_model` and stop forwarding it to `_prepare_prediction_X`.
- In `predict_scenario`, make one `predict()` call and return `_composite_forecast(baseline_df, schedule, self._appliance_signatures)`.
- Delete `_validate_physics_phase1`, `last_residual_mae` and `_use_physics_residual`.
- Remove the `physics_kwh_series` parameter of `_engineer_features` and the `"use_physics_residual"` meta key on save and load. Load keeps reading `log_transform`.

- [ ] **Step 5: Update the callers**

- `energy_forecast.py`: drop the removed kwargs at the listed call sites. `_get_scenario_cb` keeps reading `dhw_schedule` from kwargs but no longer passes it (Task 3 adds the DEBUG line and test). Delete `_effective_use_physics_residual`.
- `backtest/env.py`: add `strip_legacy_train_kwargs`. Call it in `capture_replay.replay_train` (`:164`) and in `CaptureFrame.__init__` (`self.tk = strip_legacy_train_kwargs(payload["train_kwargs"])`).
- `replay_predict` and `audit_features`: stop passing `physics_model` (leave `_retarget_physics` only if `replay_train` still uses it).
- `capture_frame.predict`: drop the `physics` parameter.
- `walkforward`: drop `WfArm.physics` and the `physics=` argument.
- `backtest_regime.py:394`: `inp = inputs_cache[o]`.
- Fix the doc mentions of `physics_kwh` in `arms.py:29` and `README.md:27`.

- [ ] **Step 6: Adapt existing tests**

Delete tests whose subject is removed. To find them: `grep -n "use_physics_residual\|_use_physics_residual\|physics_phase1\|last_residual_mae\|Phase 2\|phase2\|heating_sub_meter_df" tests/*.py`.
- `test_model.py:1167` (`test_train_forwards_ev_df_and_heating_sub_meter_to_calibrate`): keep its `ev_df` assertion, drop the sub-meter part.
- Tests that only passed `physics_model=` into `predict*` / `shap_summary` lose that argument. Every test whose name contains `override`, `open_window` or `dhw` must pass unmodified, apart from removing the `physics_model=` predict argument.

- [ ] **Step 7: Run the full suite and the replay**

Run the full suite, then the replay command from Step 1 (background).
Expected:
- The suite is green.
- Every `predict` row has `max_abs_diff` ≤ 1e-9 and the same result as in the baseline file.
- `retrain` rows may differ (88 → 87 columns, spec §6.3). Note them and do not count them as failures.

- [ ] **Step 8: Commit**

```bash
git add -A apps/ backtest/ scripts/backtest_regime.py tests/
git commit -m "refactor(model): remove physics_kwh, Phase 2 and serve-time physics args"
```

### Task 2: `physics.py` heating removal

**Files:**
- Modify: `apps/energy_forecast/physics.py` (incl. `COLD_START_MIN_WINDOWS` `:36`)
- Modify: `apps/energy_forecast/const.py` (`:61-68`)
- Modify: `apps/energy_forecast/energy_forecast.py`:
  - `_recalibrate_physics_cb` (`:~1591-1601`): drop `climate_dfs` and `heating_sub_meter_df`.
  - Delete the two physics sensors here, because they call `predict_series`: discovery entries `:1106-1131`, `_publish_physics_sensors` `:2805-~2900` and its call `:2309`.
  - Keep `_override_delta` computed and applied at `:2267-2306`.
- Modify: `apps/energy_forecast/model.py`: in `train()`'s `calibrate()` call (`:506-513`), drop the positional `climate_dfs` argument.
- Test: `tests/test_physics_removal.py`; delete heating tests in `tests/test_physics.py` (incl. `climate_dfs=None` callers `:1680`, `:1693`, `:1710`, which lose that argument) and `tests/test_energy_forecast_physics_config.py` (incl. `TestPhysicsSensors` `:745-951`)

**Interfaces:**
- Consumes: Task 1 (`train()` no longer calls `predict_training_series` / `check_zone_boundary`).
- Produces: `ThermalPhysicsModel.calibrate(...)`. Today's parameters, names, order and kinds stay, except that `climate_dfs` and `heating_sub_meter_df` are removed; update every caller.

- [ ] **Step 1: Write the failing tests**

```python
from energy_forecast.physics import ThermalPhysicsModel, _default_calibration

GONE = ("predict_series", "predict_training_series", "_space_heating_kwh", "_cop_series", "_calibrate_ua_eff",
        "_calibrate_solar_gain_area", "_resolve_heating_eligibility", "is_cold_start_gated", "check_zone_boundary")

def test_physics_has_no_heating_api():
    for name in GONE:
        assert not hasattr(ThermalPhysicsModel, name), name

def test_calibrate_has_no_heating_inputs():
    params = inspect.signature(ThermalPhysicsModel.calibrate).parameters
    assert "climate_dfs" not in params and "heating_sub_meter_df" not in params

def test_default_calibration_has_no_heating_keys():
    keys = set(_default_calibration())
    assert keys.isdisjoint({"UA_eff", "solar_gain_area", "n_calibration_windows_ua_eff",
                            "q_base_el_calibrated", "room_thermostats_at_calibration"})
    assert {"Q_base_el", "Q_dhw_daily", "UA_dhw"} <= keys

def test_physics_sensors_never_published(...):
    # MQTT-mode app with a physics model, one forecast cycle: no non-empty mqtt/publish to any topic
    # containing "physics_base_today" or "ml_adjustment_today"; no set_state for those entity ids.

def test_legacy_calibration_file_loads_and_dhw_delta_works(tmp_path):
    # write physics_calibration.json containing UA_eff=123.0, solar_gain_area=2.0,
    # n_calibration_windows_ua_eff=40 plus valid DHW keys; build ThermalPhysicsModel(tmp_path, cfg);
    # commit_dhw_schedule({"legionella": [<tomorrow iso>, 12]}); compute_serving_override_delta(...)
    # over a 48 h window returns a Series with value > 0 at that hour.
```

- [ ] **Step 2: Run them to see them fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics_removal.py -v -k "physics_has_no or calibrate_has or default_calibration or legacy_calibration or sensors_never"`
Expected: FAIL on the first three; the fourth may already pass.

- [ ] **Step 3: Remove the heating code**

- Delete the `GONE` methods.
- In `_default_calibration()`, delete the heating keys from the test.
- In `calibrate()`, delete the `UA_eff` / `solar_gain_area` branches, the `climate_dfs` / `heating_sub_meter_df` parameters, and the writes of `q_base_el_calibrated`, `n_calibration_windows_ua_eff` and `room_thermostats_at_calibration`.
- Keep `_t_flow_c` and `_cop_formula_value` (DHW COP), `_t_indoor_for_window`, `_area_weighted_t_indoor`, `_calibrate_base_load`, `_calibrate_dhw_daily` and `_calibrate_ua_dhw`.
- Delete `UA_EFF_*` from `const.py` and `COLD_START_MIN_WINDOWS`.
- Update the two `calibrate()` callers.
- Delete the two physics sensors (discovery entries, `_publish_physics_sensors`, its call). Keep both ids in `_cleanup_legacy_states`.

- [ ] **Step 4: Delete heating tests, then run the full suite**

Delete the tests that fail only because they reference a removed symbol: `grep -n "UA_eff\|solar_gain\|space_heating\|cold_start\|zone_boundary\|predict_series\|predict_training_series\|heating_eligib\|_cop_series\|climate_dfs=\|physics_base_today\|ml_adjustment_today\|_publish_physics_sensors" tests/*.py`. Then run the full suite.
Expected: green. Every remaining `override` / `dhw` / `legionella` / `open_window` test in `tests/test_physics.py` passes unmodified.

- [ ] **Step 5: Commit**

```bash
git add -A apps/ tests/
git commit -m "refactor(physics): remove space-heating model, UA_eff calibration and physics diagnostic sensors"
```

### Task 3: hef app surface: discovery retirement, `model_phase`, config, `get_scenario`

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py`
  - config defaults `:464-477`: drop `cop_sensor`, `internal_gains_fraction`, `use_physics_residual`, `heating_curve_sensor` (no reader)
  - `cop_sensor` fetch `:1536-1546`, `_physics_cop_df` `:508`, `:1494`
  - `_model_phase_attr` `:1646-1655` and attributes `:2968-3007`
  - `_get_scenario_cb` `:1227-1310`
  - startup hook after `_mqtt_publish_all_discovery()` (`:558`)
- Test: `tests/test_physics_removal.py`; adapt `tests/test_energy_forecast.py` and `tests/test_scenario_service.py` (use their existing app fixtures)

**Interfaces:**
- Produces: `EnergyForecast._retire_physics_sensors() -> None`. In MQTT mode it publishes `payload=""`, `retain=True` to `f"{self._mqtt_discovery_prefix}/sensor/energy_forecast_physics_base_today/config"` and `.../energy_forecast_ml_adjustment_today/config` through `call_service("mqtt/publish", ..., namespace=self._mqtt_namespace)`. It is called once per startup, right after `_mqtt_publish_all_discovery()`, and logs a warning instead of raising on failure (same pattern as `_mqtt_publish_discovery`).

- [ ] **Step 1: Write the failing tests**

```python
def test_get_scenario_ignores_dhw_schedule(...):
    # same app state; call _get_scenario_cb with schedule only, then with schedule + dhw_schedule
    # {"legionella": [today_iso, 12]}; the published scenario values/attributes are identical.

def test_retired_physics_sensors_cleared_at_startup(...):
    # MQTT-mode app: after startup, call_service saw mqtt/publish with payload "" and retain=True on both
    # config topics above; no non-empty publish to any topic containing "physics_base_today"
    # or "ml_adjustment_today".

def test_today_attributes_have_no_model_phase(...):
    # after one forecast cycle with a physics model configured: no attributes payload/set_state
    # for the today sensor contains the key "model_phase".

def test_removed_physics_keys_are_ignored(...):
    # args["physics"] = {"use_physics_residual": True, "cop_sensor": "sensor.cop",
    #                    "internal_gains_fraction": 0.8, "dhw_tank_temp_sensor": "sensor.tank"}
    # initialize() succeeds; a history-fetch cycle never requests "sensor.cop".
```

- [ ] **Step 2: Run them to see them fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_physics_removal.py -v -k "scenario or retired or model_phase or removed_physics"`
Expected: FAIL (`_retire_physics_sensors` missing, `model_phase` present, `sensor.cop` fetched). `test_get_scenario_ignores_dhw_schedule` may already pass after Task 1. Keep it as a lock.

- [ ] **Step 3: Implement**

- Delete `_model_phase_attr` and every `model_phase` attribute write.
- Delete the `cop_sensor` fetch and `_physics_cop_df`, and drop `cop_sensor`, `internal_gains_fraction`, `use_physics_residual` and `heating_curve_sensor` from `default_physics_config`.
- Add `_retire_physics_sensors()` per Interfaces.
- In `_get_scenario_cb`, when `dhw_schedule` is given, log `_LOGGER.debug("get_scenario: dhw_schedule ignored (no physics scenario delta)")` and do nothing else with it.
- Keep `"sensor.energy_forecast_physics_base_today"` and `"…ml_adjustment_today"` in `_cleanup_legacy_states`.

- [ ] **Step 4: Adapt existing tests and run the full suite**

Delete or adapt tests asserting `model_phase`, the `cop_sensor` fetch or `heating_curve_sensor`: `grep -n "model_phase\|cop_sensor\|_physics_cop_df\|heating_curve_sensor" tests/*.py`. Do not touch `tests/test_check_ha_status.py`; it tests an untracked script and skips without it. Run the full suite.
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add -A apps/ tests/
git commit -m "refactor(app): retire physics diagnostic sensors and model_phase, drop dead physics config"
```

### Task 4: Cleanup, docs, verification

**Files:**
- Delete: `scripts/analyze_cop_scale.py`, `tests/test_analyze_cop_scale.py`
- Modify: `README.md` (`physics:` section), `CHANGELOG.md` (via `/changelog`), `ROADMAP.md`, `docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md` (header line)
- Modify (gitignored, main checkout): `/home/jovyan/work/ha-energy-forecast/MEMORY.md`, `/home/jovyan/work/ha-energy-forecast/memory/project_v1_release_criteria.md`; move `/home/jovyan/work/ha-energy-forecast/memory/project_physics_kwh_low_importance.md` to `memory/_archive/`

- [ ] **Step 1: Delete the COP-scale tool**

`git rm scripts/analyze_cop_scale.py tests/test_analyze_cop_scale.py`.

- [ ] **Step 2: Grep for leftovers**

Run: `grep -rn "physics_kwh\|use_physics_residual\|UA_eff\|model_phase\|physics_base_today\|ml_adjustment_today\|predict_training_series\|is_cold_start_gated" apps/ backtest/ tests/ README.md`
Expected: only the fill-0 legacy line in `model.py`, the `_cleanup_legacy_states` / `_retire_physics_sensors` ids, the legacy-kwargs helper, and tests of those.

- [ ] **Step 3: Update the docs**

- **README:** `physics:` configures DHW tank and override modelling. Add a "Removed in 0.12.0" list: `use_physics_residual`, `internal_gains_fraction`, `cop_sensor`, `heating_curve_sensor`, plus sensors `physics_base_today`, `ml_adjustment_today` and the `model_phase` attribute.
- **CHANGELOG** (`/changelog`, `[Unreleased]`): Removed (`physics_kwh`, Phase 2, `UA_eff` heating calibration, the two sensors, `model_phase`, dead `cop_sensor` fetch); Fixed (`get_scenario` no longer implies a DHW delta it never delivered; EM must add its own cycle energy). **Behaviour change** note: EM's legionella boost hand-off becomes stricter once `model_phase` is gone; give the replay figures from Global Constraints.
- **ROADMAP:**
  - Close #92, #103 and the Phase 2 rows. Close #105 with the note "continues offline only; a PASS re-enters as an ML feature".
  - Close the "Physics Phase 2 interval-coverage check" item.
  - Close the 13-dead-features note for `physics_kwh`.
  - Replace the v1.0 gate with "config schema, service signatures (`get_scenario`, `set_dhw_schedule`, `recalibrate_physics`) and sensor names unchanged for 4 weeks after v0.12.0 on `main`".
  - Add the §8 follow-ups as new rows.
- **2026-06-22 spec:** first line `> Superseded in part (heating, Phase 2) by 2026-11-16-physics-heating-removal-design.md.`
- **MEMORY:**
  - Rewrite `project_v1_release_criteria` to the new criterion.
  - Archive `project_physics_kwh_low_importance`.
  - Index line: "physics heating removal implemented, awaiting train merge after #124 ships".
  - Note for deploy day:
    - Remove `use_physics_residual`, `cop_sensor`, `internal_gains_fraction`, `heating_curve_sensor` from the live `apps.yaml`.
    - Check that both sensors are gone from HA.
    - Watch the first legionella hand-offs (`python scripts/legionella_report.py` in EM).
    - The untracked `scripts/check_ha_status.py` still checks `model_phase` and the two sensors; update it then.

- [ ] **Step 4: Final verification**

Run the full suite, the replay command (Task 1 Step 1, background) and `ruff check . && ruff format --check .`. Then check the training override delta on the latest retrain capture (spec §6.3). Run from the worktree root with `PYTHONPATH=apps:.`:

```python
import tempfile
from pathlib import Path
from unittest import mock
from backtest import capture_replay
from energy_forecast.physics import ThermalPhysicsModel

sums, orig = [], ThermalPhysicsModel.compute_training_override_delta
def spy(self, *a, **k):
    s = orig(self, *a, **k); sums.append(float(s.sum())); return s
with mock.patch.object(ThermalPhysicsModel, "compute_training_override_delta", spy):
    capture_replay.replay_train(Path("/home/jovyan/work/ha-energy-forecast/data/captures"),
                                "20261115T154303205403Z", Path(tempfile.mkdtemp()))
print(sums)
```

Expected:
- The suite is green.
- Every replay `predict` row has `max_abs_diff` ≤ 1e-9.
- The spy prints `[0.0]`.
- Ruff is clean.
- `git diff --stat train/next` shows `physics.py` shrinking and no new modules except `tests/test_physics_removal.py`.

- [ ] **Step 5: Commit**

```bash
git add -A README.md CHANGELOG.md ROADMAP.md docs/ scripts/ tests/
git commit -m "docs: physics heating removal (README, CHANGELOG, ROADMAP); drop COP-scale tool"
```

### Task 5: Merge hef to the train (gated)

- [ ] **Step 1: Check the gate**

Run:
```bash
git -C /home/jovyan/work/ha-energy-forecast fetch origin
git -C /home/jovyan/work/ha-energy-forecast log --oneline origin/dev..origin/train/next
```
Expected: no output. Everything on the train, including #124 (`16180c4`) and the daily-bias fix (`1ef86da`), has shipped to `dev`. If anything is listed, stop. The branch waits, so this change ships alone (spec §7).
Also confirm the owner has seen the hand-off behaviour change (Global Constraints) before merging.

- [ ] **Step 2: Merge, test, push** (per `_shared/ha-appdaemon-workflow.md`)

```bash
git -C /home/jovyan/work/ha-energy-forecast/.worktrees/train merge --no-ff feat/physics-heating-removal
cd /home/jovyan/work/ha-energy-forecast/.worktrees/train && /home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -q
git -C /home/jovyan/work/ha-energy-forecast/.worktrees/train push origin train/next
```

Expected: the merge is clean (CHANGELOG `[Unreleased]` conflict → keep both, newest first; any other conflict → stop and ask). The suite is green. Shipping happens at a `/pm evening` with this change alone.

### Task 6: EM — v2 always adds the cycle

**Setup:** `git -C /home/jovyan/work/ha-energy-manager worktree add .worktrees/hef-physics-removal -b fix/hef-physics-removal train/next`; all EM paths below are relative to that worktree.

**Files:**
- Modify: `apps/energy_manager/loads/legionella_coverage.py:77-88`
- Modify: `apps/energy_manager/loads/heat_pump.py`: `:4032`, `:4193-4194`, `:4207-4210`, delete `_legionella_physics_present` `:4220-4240`, `:5606`
- Test: `tests/test_legionella_coverage.py` (`:64-78`, `:103`), `tests/test_heat_pump.py` (`:6822-6928` stubs, `:7032-7217`)

**Interfaces:**
- Produces: `v2_coverage(*, solar: Sequence[float], baseline: Sequence[float | None], cycle_kwh: Sequence[float]) -> float | None` with denominator `sum(baseline) + sum(cycle_kwh)`. The coverage `detail` dict has no `physics_present` key, and the `ll_event` kwargs at `:4032` / `:5606` drop `physics_present`.

- [ ] **Step 1: Write the failing tests**

```python
def test_v2_always_adds_cycle():
    assert v2_coverage(solar=[2.0, 2.0], baseline=[1.0, 1.0], cycle_kwh=[1.5, 3.5]) == pytest.approx(4.0 / 7.0)

def test_v2_signature_has_no_physics():
    assert "physics" not in inspect.signature(v2_coverage).parameters

# tests/test_heat_pump.py, in the class at :7032:
def test_v2_detail_has_no_physics_present(self, tmp_path):
    # hef today sensor state carries attributes {"model_phase": "phase1"} (stale retained);
    # coverage detail has no "physics_present" key and its denominator == sum(baseline) + sum(cycle).
```

- [ ] **Step 2: Run them to see them fail**

Run: `/home/jovyan/my_envs/ha-energy-manager/bin/python -m pytest tests/test_legionella_coverage.py tests/test_heat_pump.py -v -k "v2"`
Expected: FAIL (`physics` still required).

- [ ] **Step 3: Implement**

- Drop the `physics` parameter and its branch from `v2_coverage` and the duplicate denominator at `heat_pump.py:4207`.
- Delete `_legionella_physics_present`.
- Remove `physics_present` from `detail` and from both `ll_event` calls.
- Update the docstrings to say EM always adds its own cycle energy, because hef's scenario frame never contains it.

- [ ] **Step 4: Adapt existing tests and run the full EM suite**

- Remove the `physics=` kwargs in `test_legionella_coverage.py:64-78`. The first case becomes `4.0 / 10.0`.
- Remove the `_legionella_physics_present` stubs.
- Delete `test_physics_present_reads_hef_model_phase`, `test_physics_present_prefers_registry_key` and `test_physics_unknown_without_any_source_is_false` (`:7200-7217`).
- Drop the `physics` parameter of the `_app` helper (`:7032`; its default was `True`) and the `physics_present` assertions (`:7046`, `:7114`, `:7173-7179`).
- These numeric assertions assumed `physics=True`: `:7044`, `:7070`, `:7088`, `:7099`, `:7116`, `:7198`. Recompute each expectation by hand as `numerator / (old_denominator + preheat_kwh + boost_kwh)`, using that test's fixture config (preheat 0 when adopted). Do not refit by copying the new output.

Run the full EM suite. Expected: green.

- [ ] **Step 5: Commit**

```bash
git add -A apps/ tests/
git commit -m "fix(legionella): v2 coverage always adds the cycle energy; drop hef physics detection"
```

### Task 7: EM — bridge cleanup, dashboard, docs, train merge

**Files:**
- Modify: `apps/energy_manager/forecast/consumption_forecast_app.py`: `_PHYSICS_SENSORS` `:79-83` and its uses (`:258-259` mirror list, `:353-357` discovery), `model_phase` `:139`, `:177`, `:463-473`, `:610-622`, `_cleanup_old_mqtt_discovery` `:234-273`
- Modify: `dashboard/ha-energy-manager.yaml` (the `model_phase` row `:1124-1128` only), `config.example.yaml:564-566`, `CLAUDE.md` (hef physics/DHW-schedule dependency section), `CHANGELOG.md`
- Test: `tests/test_consumption_forecast_app.py` (`:257-258`; discovery count `:1060-1061` 22 → 20; physics tests `:1066-~1190`; `:1201-1204`; `model_phase` tests `:2024-~2110`)

**Interfaces:**
- Consumes: nothing from Task 6.
- Produces: `_cleanup_old_mqtt_discovery` specs additionally contain `("sensor", f"{d}_physics_base_today")` and `("sensor", f"{d}_ml_adjustment_today")`, the current device-scoped node ids. No other current-format id is added.

- [ ] **Step 1: Write the failing tests**

```python
def test_current_physics_discovery_ids_cleared(...):
    # after _cleanup_old_mqtt_discovery(): cleared node ids include
    # "em_consumption_forecast_physics_base_today" and "em_consumption_forecast_ml_adjustment_today"

def test_physics_mirrors_not_published(...):
    # hef states for {PREFIX}physics_base_today / ml_adjustment_today present; after a bridge cycle no
    # set_state / discovery publish targets an entity or topic containing those suffixes (except clears)

def test_model_phase_never_forwarded(...):
    # hef {PREFIX}today attributes {"model_phase": "phase1"}; the source sensor's attributes lack "model_phase"
```

- [ ] **Step 2: Run them to see them fail**

Run: `/home/jovyan/my_envs/ha-energy-manager/bin/python -m pytest tests/test_consumption_forecast_app.py -v -k "physics or model_phase"`
Expected: FAIL.

- [ ] **Step 3: Implement**

- Delete `_PHYSICS_SENSORS` and its uses, `_read_model_phase`, `_last_model_phase` and the pass-through. Add the two specs to `_cleanup_old_mqtt_discovery`.
- Dashboard: remove the "hef Modellphase" row; validate with `python -c "import yaml; yaml.safe_load(open('dashboard/ha-energy-manager.yaml'))"` (no deploy).
- `config.example.yaml`: delete the `hef_forecast_today` lines `:564-566`.
- `CLAUDE.md`: the compatibility signal is the `set_dhw_schedule` / `get_scenario` services, not `model_phase`.

- [ ] **Step 4: Adapt the existing tests and run the full EM suite**

- Delete `PHYSICS_SUFFIXES` tests (`:1066-~1190`) and the `model_phase` tests (`:2024-~2110`).
- Drop the two ids from the list at `:257-258`; the discovery count at `:1060-1061` becomes 20.
- Keep `:1201-1204` (legacy ids are still cleared).

Run the full EM suite. Expected: green.

- [ ] **Step 5: CHANGELOG and commit**

Run `/changelog` for EM: `[Unreleased]` Fixed (legionella v2 coverage overstated because hef's scenario frame never contained the cycle; **the boost hand-off gates on v2, so it becomes stricter**: replay 11-09 0.94 → 0.20 would have been below_min), Removed (hef physics mirrors, `model_phase` bridging).

```bash
git add -A apps/ tests/ dashboard/ config.example.yaml CLAUDE.md CHANGELOG.md
git commit -m "refactor(consumption-bridge): drop hef physics mirrors and model_phase"
```

- [ ] **Step 6: Merge to the EM train**

```bash
git -C /home/jovyan/work/ha-energy-manager/.worktrees/train merge --no-ff fix/hef-physics-removal
cd /home/jovyan/work/ha-energy-manager/.worktrees/train && /home/jovyan/my_envs/ha-energy-manager/bin/python -m pytest tests/ -q
git -C /home/jovyan/work/ha-energy-manager/.worktrees/train push origin train/next
```

Expected: the suite is green. Deploy order versus hef does not matter (spec §5). Reset the v1/v2 legionella review count in EM memory when it ships.
