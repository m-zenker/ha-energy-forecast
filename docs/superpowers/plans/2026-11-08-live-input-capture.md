# Live Input Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Live records exactly what it passes to `train()` and to the prediction path, plus the model each retrain produced, so an offline replay can reproduce live bit-for-bit.

**Architecture:** A new `apps/energy_forecast/capture.py` owns writing, listing, loading and pruning gzip-pickle captures with JSON sidecars under `captures/` next to `regime_log.jsonl`. `energy_forecast.py` calls it at two hook points (before `train()`, before `_prepare_prediction_X()`), behind the `capture_inputs` flag. `backtest/capture_replay.py` replays captures and checks the two round-trip criteria.

**Tech Stack:** Python 3.13, AppDaemon 4.x, pandas, LightGBM, stdlib `pickle` + `gzip` + `json`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-11-08-live-input-capture-design.md`

**Base branch:** `train/next` (worktree `.worktrees/live-input-capture`, branch `feat/live-input-capture`). Ships through the release train; the owner decided it ships before the open A+B / #110 checks close.

## Global Constraints

- No change to training, prediction or any published value: with the flag on or off, sensor output is byte-identical.
- Flag `capture_inputs` in `apps.yaml`, default `false`.
- Every capture write is wrapped like `_append_regime_log` (`energy_forecast.py:3324`): log a warning, never raise, never delay the update beyond the write itself.
- Inputs are captured **before** `train()` / `_prepare_prediction_X()` mutate anything; the physics model is deep-copied at capture time.
- Retention: **45 days** for retrain and predict captures alike, plus a **1 GB** hard cap (oldest first).
- No new dependencies on the HA side.
- `scripts/` is gitignored: `scripts/pull_ha_data.py` changes are local-only.
- Tests: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`, full suite after every task; `ruff check apps/ tests/ backtest/` clean.

## Review Focus

1. **Unwritable or full capture dir**: the hourly update still publishes, with one warning. Test in Task 3 (`test_capture_failure_does_not_break_update`).
2. **DST fall-back: two hourly runs at the same local hour**: they must not overwrite each other. Filenames use UTC. Test in Task 1 (`test_capture_ids_unique_across_dst_fallback`).
3. **Crash mid-write leaves a partial file**: listing and pruning must skip it, never crash. Write to `.tmp` and rename. Test in Task 1 (`test_partial_files_ignored_and_pruned`).
4. **Physics disabled (`_physics_model is None`)**: captures still written with `physics_model=None`, and replay works. Tests in Tasks 3 and 4.
5. **Load under a different library version**: `load_capture` warns (names the package and both versions) instead of failing silently. Test in Task 1 (`test_load_warns_on_version_mismatch`).

---

## File Structure

- Create `apps/energy_forecast/capture.py`: capture IO (write/list/load/prune, versions, sidecar). No app imports.
- Modify `apps/energy_forecast/const.py`: `CAPTURE_DIR = PRED_HISTORY_PATH.parent / "captures"`.
- Modify `apps/energy_forecast/weather.py`: tag the returned frame with `attrs["weather_source"]`.
- Modify `apps/energy_forecast/energy_forecast.py`: flag in `initialize()`, retrain hook, predict hook, `_retrain_running` marker.
- Modify `conftest.py`: autouse redirect of `CAPTURE_DIR` to `tmp_path`, as for `REGIME_LOG_PATH` (`conftest.py:57`).
- Create `backtest/capture_replay.py`: replay + round-trip checks + CLI.
- Tests: `tests/test_capture.py`, `tests/test_energy_forecast_capture.py`, `tests/test_backtest_capture_replay.py`.
- Docs: `README.md`, `CHANGELOG.md`, `backtest/README.md`, `apps.yaml` example (if the repo has one with flags listed).

---

### Task 1: Capture IO module

**Files:**
- Create: `apps/energy_forecast/capture.py`
- Modify: `apps/energy_forecast/const.py:30` (add `CAPTURE_DIR`)
- Test: `tests/test_capture.py`

**Interfaces:**
- Produces:
  - `RETENTION_DAYS = 45`, `MAX_TOTAL_BYTES = 1_000_000_000`
  - `library_versions() -> dict[str, str]`: keys `python, pandas, numpy, sklearn, lightgbm` (`"missing"` if not importable).
  - `write_capture(root: Path, kind: Literal["retrain", "predict"], payload: dict, meta: dict, now_utc: pd.Timestamp) -> str | None`: writes `<root>/<kind>/<capture_id>.pkl.gz` plus `<capture_id>.json`, and returns `capture_id`. `capture_id = now_utc.strftime("%Y%m%dT%H%M%S%fZ")`. The sidecar = `meta` plus `{"capture_id", "kind", "ts_utc", "versions", "bytes"}`. It writes to `*.tmp` and then `os.replace`, never raises (warning, returns `None`), and calls `prune()` after a successful write.
  - `attach_reference(root: Path, capture_id: str, model_dir: Path) -> None`: copies every regular file in `model_dir` (not `archive/`) into `<root>/retrain/<capture_id>.ref/`, updates the sidecar's `bytes`, and sets `"reference": true`. Never raises.
  - `update_sidecar(root: Path, kind: str, capture_id: str, **fields) -> None`: merges fields into the sidecar (atomic rewrite); never raises.
  - `list_captures(root: Path, kind: str) -> list[dict]`: sidecars sorted by `ts_utc`. It skips `.tmp` files and sidecars whose pickle is missing.
  - `load_capture(root: Path, kind: str, capture_id: str) -> tuple[dict, dict]`: `(payload, sidecar)`. It emits `warnings.warn` per package whose version differs from `library_versions()`.
  - `prune(root: Path, now_utc: pd.Timestamp, retention_days: int = RETENTION_DAYS, max_bytes: int = MAX_TOTAL_BYTES) -> int`: deletes captures older than retention, then oldest-first until the total is ≤ `max_bytes`, plus orphan `.tmp` files older than 1 h. Returns the count removed and never raises.

- [ ] **Step 1: Write the failing tests** in `tests/test_capture.py`:
  - `test_write_then_load_roundtrip`: a payload with a DataFrame, a Series, `None` and a nested dict loads back equal (`pd.testing`), and the sidecar has `kind`, `ts_utc`, `versions`, `bytes > 0` and the caller's meta keys.
  - `test_capture_ids_unique_across_dst_fallback`: two writes at `2026-10-25 00:30Z` and `01:30Z` (both 02:30 local) give two distinct ids and `len(list_captures(...)) == 2`.
  - `test_update_sidecar_merges_fields`: `update_sidecar(..., last_trained_after="x")` → the sidecar keeps its old keys and gains `last_trained_after == "x"`.
  - `test_write_never_raises`: root is a file, not a dir → returns `None` and logs one warning (caplog).
  - `test_partial_files_ignored_and_pruned`: a stray `x.pkl.gz.tmp` (mtime 2 h ago) and a sidecar with no pickle are absent from `list_captures`; `prune` removes the `.tmp`.
  - `test_prune_retention`: captures at now−46 d and now−44 d → only the 44-day one is kept.
  - `test_prune_size_cap`: three captures and `max_bytes` smaller than two of them → the oldest go first until under the cap.
  - `test_attach_reference_copies_model_dir_without_archive`: a model dir with `energy_model.pkl`, `physics_schedule.json`, `archive/old.pkl` → `.ref/` holds the first two only, and the sidecar has `reference is True`.
  - `test_load_warns_on_version_mismatch`: rewrite the sidecar `versions["pandas"] = "0.0"` → `pytest.warns(UserWarning, match="pandas")`.
- [ ] **Step 2: Run** `…/bin/python -m pytest tests/test_capture.py -v` → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Implement** `capture.py` per the Interfaces block. Use `pickle.HIGHEST_PROTOCOL` inside `gzip.open(..., compresslevel=6)`. Add `CAPTURE_DIR` to `const.py`.
- [ ] **Step 4: Run** the test file → PASS; `ruff check` clean.
- [ ] **Step 5: Commit** `feat(capture): add capture IO with retention, size cap and version sidecar`

---

### Task 2: Wire the flag and the retrain hook

**Files:**
- Modify: `apps/energy_forecast/weather.py:fetch_forecast` (set `df.attrs["weather_source"] = "srg" | "open_meteo"` on each return path; frames are otherwise untouched)
- Modify: `apps/energy_forecast/energy_forecast.py` (`initialize()` near line 295; `_retrain()` around line 1827; `_retrain_cb` line 993)
- Modify: `conftest.py` (redirect `CAPTURE_DIR`)
- Test: `tests/test_energy_forecast_capture.py`

**Interfaces:**
- Consumes: `capture.write_capture`, `capture.attach_reference`, `capture.library_versions`, `const.CAPTURE_DIR`.
- Produces:
  - `self._capture_inputs: bool = bool(self.args.get("capture_inputs", False))`
  - `self._retrain_running: bool`: set `True` at the top of `_retrain()` and reset in a `finally`. Initialised `False` in `initialize()`.
  - Retrain payload keys: `"train_kwargs"` (the exact dict passed to `train()`, built once and splatted into the call so the two cannot drift), `"physics_model"` (`copy.deepcopy(self._physics_model)` or `None`), `"now_local"` (naive local `pd.Timestamp`), `"model_init"` (`{"model_archive_count", "timezone"}`), `"args"` (`dict(self.args)` minus keys containing `secret`, `token` or `password`).
  - Sidecar meta: `{"last_trained_before": <str|None>, "physics": bool}`.
  - After `train()` returns: `attach_reference(CAPTURE_DIR, capture_id, model_dir)`. Then `capture.update_sidecar(CAPTURE_DIR, "retrain", capture_id, last_trained_after=str(self._ml_model.last_trained))`.

- [ ] **Step 1: Write the failing tests** using the `_FakeRetrain` pattern (`tests/test_energy_forecast.py:4045`) with a stub `_ml_model` whose `train` records its kwargs:
  - `test_retrain_capture_off_by_default`: no `captures/` dir exists after `_retrain()`.
  - `test_retrain_capture_records_exact_train_kwargs`: flag on → one retrain capture, and the payload's `train_kwargs` is the same set of keys **and the same objects' values** (`pd.testing` per frame) as the stub received.
  - `test_retrain_capture_physics_copied_before_train`: the stub's `train` mutates `physics_model._tau_hours`; the captured copy keeps the pre-train value.
  - `test_retrain_capture_drops_secrets`: `args` has `srg_client_secret` → absent from the payload.
  - `test_retrain_capture_attaches_reference`: after `train`, `.ref/` mirrors the model dir.
  - `test_retrain_running_flag_reset_on_exception`: the stub `train` raises → `_retrain_running is False` afterwards.
  - `test_fetch_forecast_tags_source`: SRG path → `"srg"`; SRG failure fallback → `"open_meteo"` (monkeypatch the two fetchers).
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.** The capture call sits between `_fetch_physics_sensor_histories(...)` (line 1825) and `train(**train_kwargs)`, inside its own try/except.
- [ ] **Step 4: Run** the new file plus the full suite → PASS.
- [ ] **Step 5: Commit** `feat(capture): record exact train() inputs and the resulting model per retrain`

---

### Task 3: Predict hook

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:_update_sensors` (read `forecast_df.attrs["weather_source"]` right after `fetch_forecast`, before `drop_dst_duplicates/_strip_tz` can drop attrs; build the payload just before `_prepare_prediction_X` at line 2064; write once after `aggregated` exists, next to the regime-log write at ~line 2268)
- Test: `tests/test_energy_forecast_capture.py`

**Interfaces:**
- Consumes: Task 1 IO; Task 2's `self._capture_inputs`, `self._retrain_running`.
- Produces predict payload keys:
  - `"predict_kwargs"`: the raw arguments as passed to `_prepare_prediction_X`, i.e. `forecast_df, live_temp, recent_actuals, sub_sensors_recent, away_series, people_home_series, climate_recent, dhw_recent, room_areas, heating_active_series, setpoint_on, setpoint_off, heating_buffer_temp_recent`. Build one dict, splat it into both `_prepare_prediction_X` and `predict`, and keep `physics_model=` / `_prepared=` outside the dict.
  - `"physics_model"`: a deepcopy taken before `_prepare_prediction_X`, or `None`.
  - `"now_local"`; `"model_predictions"` (the `predictions` frame right after `predict()`, before the override delta); `"published_predictions"` (after the override delta); `"tomorrow_kwh"` (`aggregated["tomorrow"]`).
  - Sidecar meta: `{"last_trained": str(self._ml_model.last_trained), "weather_source": <str|None>, "retrain_in_progress": self._retrain_running and the call did not come from _retrain_cb, "physics": bool}`. For the `_retrain_cb` → `_update_sensors()` call, pass the reason explicitly: add the keyword `_update_sensors(self, *, after_retrain: bool = False)`, have `_retrain_cb` pass `True`, and record `after_retrain` in the sidecar.

- [ ] **Step 1: Write the failing tests** with `_FakeUpdateSensors` and `_patch_update_sensors_deps` (`tests/test_energy_forecast.py:4375`, `~4585`):
  - `test_predict_capture_off_by_default`: no capture files.
  - `test_published_output_identical_with_flag_on`: run once with the flag off and once on (same inputs) → identical regime-log record (minus `ts`), identical published state/attribute payloads (capture the MQTT/`set_state` mock calls), identical saved `pred_history.json`.
  - `test_predict_capture_contents`: the payload has all `predict_kwargs` keys, and `model_predictions` equals the `predict()` stub's return value.
  - `test_capture_failure_does_not_break_update`: `capture.write_capture` monkeypatched to raise → the regime-log line is still written and one warning is logged.
  - `test_after_retrain_marked`: `_update_sensors(after_retrain=True)` → sidecar `after_retrain is True`.
  - `test_predict_capture_without_physics`: `_physics_model=None` → payload `physics_model is None`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new file plus the full suite → PASS.
- [ ] **Step 5: Commit** `feat(capture): record raw prediction inputs and published vector each update`

---

### Task 4: Replay and round-trip checks

**Files:**
- Create: `backtest/capture_replay.py`
- Test: `tests/test_backtest_capture_replay.py`

**Interfaces:**
- Consumes: `capture.load_capture`, `capture.list_captures`, `backtest.env.frozen_now`, `backtest.env` hassapi stub.
- Produces:
  - `replay_predict(root: Path, predict_id: str, model_dir: Path) -> pd.DataFrame`: loads `EnergyForecastModel(model_dir, …model_init)` and, under `frozen_now(now_local)`, runs `_prepare_prediction_X(**predict_kwargs, physics_model=…)` then `predict(..., _prepared=…)`. Returns the predictions frame (pre-override, comparable to `model_predictions`).
  - `reference_dir_for(root: Path, predict_sidecar: dict) -> Path | None`: the `.ref/` of the retrain capture whose `last_trained_after == predict_sidecar["last_trained"]`.
  - `replay_train(root: Path, retrain_id: str, scratch: Path) -> Path`: copies the retrain capture's pre-train state into an empty `scratch` model dir. Seed `scratch` with the files the **previous** retrain's `.ref/` holds, if one exists, because `train()` reads prior artifacts. Then, under `frozen_now(now_local)`, it calls `EnergyForecastModel(scratch, …).train(**train_kwargs, physics_model=<captured copy>)` and returns `scratch`.
  - `check(root: Path) -> list[dict]`: rows of `{"predict_id", "kind": "predict"|"train", "max_abs_diff", "next_day_total_pct"}`.
    - Criterion 1 runs for every non-`retrain_in_progress` predict capture that has a reference.
    - Criterion 2 runs for each retrain capture paired with its `after_retrain` predict capture.
  - CLI: `python -m backtest.capture_replay <captures_dir>` prints the rows and a PASS/FAIL line. Criterion 1: `max_abs_diff ≤ 1e-6` kWh/h. Criterion 2: `|next_day_total_pct| ≤ 0.1`. Exits 0 even on FAIL: it reports, the same report-only stance as `backtest/repro.py`.
  - Note: `model.py:961` uses `datetime.now()` for `last_trained`. Ignore `last_trained` in comparisons; don't patch `datetime`.

- [ ] **Step 1: Write the failing tests** using a small synthetic training set. Reuse an existing `test_model.py` training fixture if one fits; otherwise ~60 days of hourly synthetic load + weather.
  - Produce captures through the real `capture.write_capture` with the same kwargs dicts used for a real `train()` and `predict()` on a tmp model dir, then attach the reference.
  - `test_replay_predict_matches_exactly`: `max_abs_diff ≤ 1e-6`.
  - `test_replay_train_reproduces_next_day_total`: `|pct| ≤ 0.1`.
  - `test_replay_without_physics`: the same two checks with `physics_model=None`.
  - `test_check_skips_retrain_in_progress`: a sidecar with `retrain_in_progress: true` is not in `check()` rows.
  - `test_cli_prints_pass`: `subprocess` run on the fixture dir → exit 0, output contains `PASS`.
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** the new file plus the full suite → PASS; `ruff check apps/ tests/ backtest/` clean.
- [ ] **Step 5: Measure sizes.** Run a local end-to-end capture against the `data/` mirror (symlink `../../data` into the worktree if absent) through the existing harness pattern (`scripts/regime_harness.py` in the main checkout, real `_retrain()` + `build_prediction_inputs`) with `capture_inputs: true`. Report the retrain capture size (including `.ref/`) and the predict capture size, and extrapolate to 45 days (assume 1 retrain/day, 24 predicts/day). Expected: well under the 1 GB cap. If the projection is over 1 GB, stop and report. Run `python -m backtest.capture_replay` on that dir and record the result; criterion 1 must PASS locally.
- [ ] **Step 6: Commit** `feat(backtest): replay live captures and check the two round-trip criteria`

---

### Task 5: Docs, local pull, deploy handoff

**Files:**
- Modify: `README.md` (an "Input capture (diagnostics)" subsection: the flag, what is stored, where, retention/cap, that secrets are dropped, disk estimate from Task 4 Step 5)
- Modify: `backtest/README.md` (how to pull captures and run `python -m backtest.capture_replay`)
- Modify: `CHANGELOG.md` via `@changelog-writer`
- Modify (local only, gitignored): `scripts/pull_ha_data.py`, which pulls `captures/` incrementally (skips ids already present locally) into `data/captures/`
- Memory: a new `memory/project_live_input_capture.md` plus a `MEMORY.md` index line

- [ ] **Step 1: Write the docs and the pull change**; run `pull_ha_data.py --help` (or a dry run) to show the change loads.
- [ ] **Step 2: Full suite + lint** → all pass; record the count.
- [ ] **Step 3: Commit** `docs(capture): document input capture and replay`
- [ ] **Step 4: Hand off to the train.** Merge `feat/live-input-capture` into `train/next` (no-ff) in `.worktrees/train`, run the full suite there, and push `train/next`. It ships with the next `/pm evening`. **Before that deploy**: with the owner's confirmation at deploy time, set `capture_inputs: true` in the live `apps.yaml` over Samba. `deploy.py` won't add single keys (`scripts/deploy.py:101-120`).
- [ ] **Step 5: Post-deploy check (first hours after deploy).**
  - Pull captures and confirm the startup retrain capture with `.ref/`, plus hourly predict captures with `weather_source` set.
  - Run `python -m backtest.capture_replay data/captures` and record criterion 1 and 2 results in the memory file.
  - A criterion 2 FAIL on live captures is a finding to report (non-input difference), not something to tune away.
