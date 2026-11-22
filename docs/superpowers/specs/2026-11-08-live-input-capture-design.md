# Live input capture — design

Date: 2026-11-08 · Status: approved by owner 2026-11-08 · Base branch: `train/next`

## Intent

The live forecast loses to a naive "same hour, mean of days −2..−8" forecast in every segment since late
October (2026-11-08 review: post-10-28 model 0.546 vs naive 0.458 kWh/h; correlation 0.30 vs 0.42), and it
over-forecasts cold days by about +2.5 kWh/day. No offline tool reproduces either: the best harness replay
(`scripts/regime_harness.py`, real `_retrain()` with stubbed HA) sits 6–12 % below live on next-day totals,
and replays show +0.08 to +0.26 kWh/day bias instead of +2.5. Until a replay reproduces live, every backtest
compares variants of a model that is not the one running.

The gap is believed to sit in inputs that cannot be rebuilt after the fact: away/presence history, the 30-day
energy resync, the SRG weather frame actually served, physics state before a retrain, and library versions.
This change makes live **record exactly what it passed to the model**, so a replay can load identical inputs
instead of reconstructing them.

This is step 1 of two. Step 2 (separate design, after this ships) is a backtest-kit frame mode that loads these
captures, a naive-forecast arm as a standing guardrail, and the reproduction gate. Owner decision: **the
recording ships first**, ahead of the open checks (A+B re-baseline rescore ~11-09/11-12, #110 stability
check ~11-19), accepting that the deploy restarts AppDaemon and triggers a startup retrain.

## Goals

- At every retrain, persist every argument passed to `EnergyForecastModel.train()` plus the state needed to
  rerun it: physics model state before training, the frozen "now", the model constructor/config args, and
  installed `pandas`/`numpy`/`scikit-learn`/`lightgbm`/`python` versions.
- At every hourly sensor update, persist every argument passed to `predict()` (including the `forecast_df`
  weather frame actually served, SRG or Open-Meteo fallback, and which source it was), the frozen "now",
  the `last_trained` of the model used, and the final published 48 h vector (after the DHW override delta
  and any post-model correction).
- Captures are retrievable over Samba next to `pred_history.json` / `regime_log.jsonl`.
- **Success criterion (verifiable locally before deploy, and again on the first live captures):**
  1. *Predict round trip:* loading the live model files plus one predict capture and calling `predict()` with
     the captured arguments and frozen clock reproduces the captured pre-override prediction vector to
     ≤ 1e-6 kWh per hour.
  2. *Train round trip:* calling `train()` on a retrain capture with the frozen clock, then predicting on a
     predict capture from right after that retrain, reproduces the live vector to ≤ 0.1 % on the next-day
     total. (Training is deterministic for a fixed cutoff — 0.0 % gap measured 2026-11-05.)
  If (2) fails on live captures, that is itself the key finding (a non-input difference: library versions,
  hidden state) and gets reported, not worked around.

## Non-goals

- No change to training, prediction, or any published value. Byte-identical sensor output with the flag on
  or off.
- Not the backtest frame mode, naive arm, or reproduction gate (step 2).
- No backfill of past periods; the current +2.5 kWh/day period is not recoverable.
- No HA sensor or attribute for capture status.

## Anti-goals (failure even if it "works")

- Any capture error that breaks or delays a retrain or sensor update. Every capture write is wrapped like
  `_append_regime_log` (`energy_forecast.py:3324`): log a warning, never raise.
- Capturing objects *after* `train()`/`predict()` mutated them (e.g. physics calibration written during
  training). The capture must reflect the inputs, not the post-call state.
- Unbounded disk growth on the HA host.
- Captures that silently can't be loaded later (version skew) with no way to tell.

## Constraints

- AppDaemon 4.x, Python 3.13; no new dependencies on the HA side (format must use what is installed —
  pickle via `joblib`/`pickle` with gzip; parquet only if `pyarrow` is confirmed installed in the live add-on).
- Flag `capture_inputs` in `apps.yaml`, default `false`; owner's live `apps.yaml` sets `true`.
  `scripts/deploy.py` patches missing entries — check it does not add the key with the default and
  override the owner's intent.
- `scripts/` is gitignored (`.gitignore:38`); changes to `scripts/pull_ha_data.py` are local-only, as with
  the `regime_log` line.
- Ships through the release train (`train/next`, `/pm evening`) like other changes; this is a behaviour-free
  change and needs no gate, but it does restart AppDaemon.
- Tests via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`, full suite.

## Decided how

- **Two hook points, no model code touched:** just before `self._ml_model.train(...)`
  (`energy_forecast.py:1827`) and just before `self._ml_model._prepare_prediction_X(...)`
  (`energy_forecast.py:2064`), plus the final published vector once it is known in the same update.
  The predict capture stores the **raw** inputs, not `_prepared`, so a replay re-runs feature engineering
  (where the gap is suspected) and rebuilds the hidden state that call sets (`_last_regime_info`,
  `_last_physics_kwh_series`).
- **Exact arguments, not re-fetches:** capture the very objects passed, so the 30-day resync, away/presence,
  climate and EV handling are all included as live saw them.
- **New module** `apps/energy_forecast/capture.py` holding the write/prune logic and the loader the step-2
  frame will use; `energy_forecast.py` only calls it.
- **Location:** `captures/` directory next to `REGIME_LOG_PATH` (`const.py:30`), so Samba pulls find it.

## My calls (owner can override)

- **Physics state in every capture, not only at retrain:** predict reads `physics_schedule.json`
  (`committed_override`, `override_history`) and calibration, which service calls change between retrains,
  and the hourly-refreshed `self._physics_*` sensor histories (`energy_forecast.py:~2020`). Each predict
  capture therefore stores the physics JSON state and those histories as passed. Retrain captures store the
  physics state as a deep copy taken before `train()`.
- **Reference model per retrain:** each retrain capture also stores the resulting model files and post-train
  physics JSONs, clearly marked as *reference outputs* (the inputs-only anti-goal is about not snapshotting
  inputs after mutation, not about refusing outputs). This lets any predict capture replay against the model
  that served it (criterion 1) without first passing criterion 2, and lets criterion 2 compare the retrained
  model to the live one directly. Each predict capture records the `last_trained` it was served by, which
  keys it to its retrain capture.
- **Pre-train state per retrain (added 2026-11-11):** each retrain capture also stores `<id>.pre/`, the model dir
  as it was before `train()` (regular files, not `archive/`; empty when the dir is missing, e.g. a fresh install).
  `train()` depends on prior model state (the first live replay from an empty dir gave 152 trees vs live 174 and
  a 0.79 % next-day-total gap; seeded with live state it matched exactly), so criterion 2 seeds its replay from
  `.pre/`, falling back to the previous retrain's `.ref/` for captures that predate this, and reports a capture
  with neither as SKIP (no seed), not FAIL.
- **Retention matches across kinds:** retrain and predict captures are both kept for the same number of
  days, so no predict capture is left without its model (adaptive retrains can run ~daily). Window length:
  see open question below.
- **Concurrency:** captures do not take the retrain lock (that would change `_update_cb` timing). A predict
  capture taken while a retrain is running is marked `retrain_in_progress: true` in its sidecar and skipped
  by replays. Criterion 2's pairing uses the `_update_sensors()` call `_retrain_cb` makes straight after
  training (`energy_forecast.py:1002`).
- **Clock:** replays freeze "now" by monkeypatching (`datetime.now`, `pd.Timestamp.now`), as the backtest
  kit already does in `backtest/env.py`. No clock parameter is threaded through `model.py`/`physics.py`.
- **Flag rollout:** `deploy.py` only appends whole missing app entries and never single keys
  (`scripts/deploy.py:101-120`), so it won't turn the flag on. The deploy step therefore sets
  `capture_inputs: true` in the live `apps.yaml` over Samba, after the owner confirms at deploy time.
  The default stays `false` for other installs.
- **Format:** gzip pickle per capture plus a small JSON sidecar (timestamp, kind, versions, `last_trained`,
  weather source, `retrain_in_progress`, file size) so captures can be listed and version-checked without
  unpickling.
- **Size:** measure real capture sizes in the tests against the local `data/` mirror and report them before
  deploy; the estimate is a few MB per retrain capture including model files, and tens to low hundreds of KB
  per hourly capture with physics histories.

## Owner decisions (2026-11-08)

- Design approved; recording ships first, ahead of the open checks.
- Retention **45 days** for both retrain and predict captures. Disk budget: anything below 1 GB is fine.
  Pruning additionally enforces a 1 GB hard cap (oldest first) as a safety net.

## Left to the builder

- File naming, the exact sidecar schema, pruning trigger (on write is fine).
- Whether to capture via a thin wrapper object or a kwargs dict.
- Test design, as long as it covers: flag off → no files written and outputs unchanged; flag on → outputs
  unchanged; capture failure (e.g. unwritable dir) → warning only; pruning; the two round-trip criteria
  against a small fixture model.
- How `pull_ha_data.py` (local) fetches `captures/` incrementally.
