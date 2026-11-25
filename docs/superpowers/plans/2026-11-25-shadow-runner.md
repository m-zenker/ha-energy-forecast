# Shadow Runner (capture-based, idempotent) Implementation Plan

OWNER-APPROVED docs/superpowers/plans/2026-11-25-shadow-runner.md 2026-11-25 (native execution)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** one command, safe to run any time, trains frozen arms on live retrain captures, forecasts from live predict captures (real weather forecast included) and appends the forecasts to a log that `wf_score` can score.

**Architecture:** `backtest/shadow.py` loops over captures not yet logged and reuses `capture_replay` (inputs, frozen clock) and `walkforward.ARMS` (arm definitions). Work already logged is skipped, so a missed night is caught up by the next run. Scheduling is a thin shell wrapper, kept out of the Python.

**Tech Stack:** Python 3.13 env `ha-energy-forecast-live`, pandas, LightGBM, pytest, `flock`.

**Spec:** this document; background `memory/project_overforecast_diagnosis_2026_11_25.md` and `backtest/README.md` ("Cold-bias candidates", "Shadow check for #120").

## What was verified (2026-11-25)

- In this Jupyter container there is no `cron`, `crond`, `systemd`, `at`, `supercronic` or Docker socket. Nothing in the container survives a recreate except the volumes `/home/jovyan/work` and `/home/jovyan/my_envs`; both the code and the live env live there.
- The compose file (`homelab/compose_files/jupyter_docker_compose.txt`) names the container `jupyter-scipy`, sets `EM_SMB_USER` / `EM_SMB_PASSWORD` / `EM_HA_TOKEN` in the container environment (so `docker exec` should inherit them, not verified) and mounts `/home/martin/library/notebooks` and `/home/martin/library/jupyter_envs`.
- The host cannot be inspected from here. Host cron or a systemd timer calling `docker exec` is therefore **unverified**; Task 4 has the owner run the check.
- Captures: one retrain capture every 1-2 days, 24-29 predict captures a day. HA keeps 45 days (1 GB cap), local copies are kept for good.
- Consequence: the frozen arms can also be replayed from captures after the fact. The nightly run adds convenience and tamper evidence, not new data. Only the captures must not be lost: a gap over 45 days is the real risk.

## Global Constraints

- Base: `train/next` (the arms are there, not on `dev`); branch `feat/shadow-runner`, worktree `.worktrees/shadow-runner`. Tooling only, risk tag NONE, nothing under `apps/` changes, HA is only read.
- Arms: `base jit` is replaced by `base s1` here (a captured retrain cannot be shifted an hour; `s1` is base with LightGBM seed 1, the noise floor), plus `droptp droptp_nb shapecl clrecent rescale`. Definitions come from `walkforward.ARMS` unchanged.
- Origins: the first predict capture with no retrain in progress in each of the hours 00, 06, 12, 18 local.
- Output: `data/backtest/shadow/rows/<capture_id>.pkl` with the walkforward columns `arm, origin, cutoff, ts, lead, p50, lo, hi` plus `logged_at`, `blind`, `code`, `app_code`. Written atomically; an existing file means done.
- `blind` is True only for rows with `logged_at < ts`. `code` is a hash of `backtest/walkforward.py`, `backtest/regime_arms.py`, `backtest/shadow.py` and `backtest/capture_replay.py`; `app_code` is a hash of `apps/energy_forecast/*.py` (a repo commit would change on every docs merge). `score` refuses mixed `code`; a changed `app_code` is listed, because a live deploy legitimately changes it.
- Each arm has its own seed chain in `data/backtest/shadow/models/<arm>/<retrain_id>/` (the newest 3 kept); the first link is seeded from `seed_dir_for`.
- `base` must reproduce the captured `model_predictions` within 1e-6 (criterion 1). A mismatch marks the capture `parity: FAIL` in `status.json`, exits 2 and still logs the rows.
- Tests: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest`.

## Review Focus

1. Second run right after the first writes nothing and changes nothing. Task 1.
2. Killed half-way (no file, or a `.tmp`): the next run redoes that capture. Task 1.
3. A predict capture whose retrain capture is missing or unseeded is skipped and reported, never logged with a wrong model. Task 1.
4. Two runs at once: the second exits 0 without touching anything. Task 3.
5. HA unreachable: local captures are still processed, exit code non-zero, status says so. Task 3.

## File Structure

- Create `backtest/shadow.py`; modify `backtest/capture_replay.py` (`replay_train` gains `seed_dir`, `patch`, `energy_transform`, `train_kwargs`, defaults unchanged).
- Create `scripts/shadow_nightly.sh` (tracked by force-add only if the owner agrees; `scripts/` is gitignored) and `scripts/pull_captures.py` (local only, like `pull_ha_data.py`).
- Modify `backtest/README.md`, `CHANGELOG.md`. Test: `tests/test_backtest_shadow.py`.

### Task 1: core loop

**Interfaces — Produces:** `run(root: Path, out: Path, arms: Sequence[str], now: pd.Timestamp, since: pd.Timestamp) -> dict` (status: `processed`, `skipped`, `parity`, `errors`); `select_origins(sidecars: list[dict]) -> list[dict]`; `replay_train(root, retrain_id, scratch, *, allow_unseeded=False, seed_dir=None, patch=None, energy_transform=None, train_kwargs=None) -> Path`. The default call of `replay_train` stays identical (existing tests unchanged).

- [ ] **Step 1: failing tests** on a synthetic capture root (the fixture style of `tests/test_backtest_capture_replay.py`): `test_select_origins_first_clean_capture_per_slot`; `test_base_matches_captured_predictions`; `test_second_run_is_a_noop` (file mtimes unchanged); `test_tmp_file_is_redone`; `test_missing_retrain_is_skipped_and_reported`; `test_blind_flag_per_row`; `test_code_hash_ignores_docs_changes`; `test_parity_mismatch_exits_2_and_logs`; `test_replay_train_defaults_unchanged`; `test_replay_train_patch_not_active_after`.
- [ ] **Step 2:** run → FAIL. **Step 3:** implement. **Step 4:** PASS, full suite. **Step 5:** commit `feat(backtest): capture shadow runner core`.

### Task 2: arm chains and scoring entry

**Interfaces — Produces:** `python -m backtest.shadow run|score|status`; `score(out, root, since) -> dict` returning `shadow_verdict` for `droptp`/`droptp_nb` and `candidate_verdict` for the three candidates, from the rows files and `CaptureFrame.actuals()` of the newest retrain capture. Default `--since 2026-11-26`.

- [ ] **Step 1: failing tests:** `test_each_arm_keeps_its_own_seed_chain`; `test_keeps_newest_three_models`; `test_score_drops_days_without_full_actuals`; `test_score_refuses_mixed_code` (INCONCLUSIVE plus the reason); `test_score_filters_before_swing`; `test_status_lists_last_run_and_gaps` (a gap over 2 days is listed).
- [ ] **Step 2-5:** as above, commit `feat(backtest): shadow arms and scoring`.

### Task 3: wrapper and pull

- [ ] `scripts/pull_captures.py`: only `_pull_captures` (no CSV overwrite), timeout 10 min, exit 1 on error. `scripts/shadow_nightly.sh`: `flock -n` on `data/backtest/shadow/.lock` (busy → exit 0), pull (failure recorded, run continues), `python -m backtest.shadow run`, output to `data/backtest/shadow/log/<date>.log` (30 kept). No credential is written anywhere; the script needs `EM_SMB_USER` and `EM_SMB_PASSWORD` in its environment.
- [ ] Tests for the wrapper: bash run with a stub `python` covering lock busy, pull failure and success exit codes.
- [ ] CHANGELOG via `/changelog`, README section, commit.

### Task 4: scheduling, own container (owner deploys)

Decision 2026-11-25 (proposed): a separate small container instead of cron or systemd inside `jupyter-scipy`.

- `compose_files/hef_shadow_docker_compose.txt` (homelab repo): image `quay.io/jupyter/scipy-notebook:latest` (already on the host), `container_name: hef-shadow`, `user: "1000:100"`, `restart: unless-stopped`, no ports, `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]`, `mem_limit: 3g`, `cpus: 1.5`.
- Mounts: `/home/martin/library/notebooks/ha-energy-forecast` -> the same path under `/home/jovyan/work/` read-only; its `data/captures` and `data/backtest/shadow` read-write (created beforehand, owner 1000:100); `/home/martin/library/jupyter_envs` -> `/home/jovyan/my_envs` read-only. Same paths, because conda envs are not relocatable.
- Environment: only `EM_SMB_USER`, `EM_SMB_PASSWORD` (from an `env_file` outside git, mode 600), `TMPDIR` on a tmpfs, `PYTHONDONTWRITEBYTECODE=1`. No HA token.
- Command: run `scripts/shadow_nightly.sh`, sleep 6 h, repeat. No cron: the run is idempotent, so a missed slot is caught up.
- `smb` is not installed in the live env; the pull runs with `/home/jovyan/my_envs/ha-energy-forecast/bin/python`, the arms with the live env.
- Acceptance: `docker logs` shows two runs, parity PASS, the second a no-op; restart the container and it continues; stop it for a day and it catches up; writing outside the two data folders fails.
- If the container is not wanted: run `python -m backtest.shadow run` by hand at least every 30 days (HA prunes captures at 45).

## After this plan

Merge to `train/next` on an owner click. **2026-12-10 or later** `python -m backtest.shadow score` is the second, real-weather reading next to the walk-forward verdicts; where the two disagree, report both.
