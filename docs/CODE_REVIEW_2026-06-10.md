# Full-Codebase Code Review — 2026-06-10

Detailed multi-angle review of all app modules (~7,300 lines), deploy/backfill scripts,
conftest, and project config. Static reading only — the test suite and ruff were **not**
run as part of this review; run both before acting on any fix.

Status legend: `[ ]` open · `[x]` resolved · `[~]` won't fix / accepted

---

## High severity

### [x] H1 — `get_scenario` service crashes when `sub_energy_sensors` is configured

`apps/energy_forecast/energy_forecast.py:782-784`

```python
valid_prefixes = set(self._ml_model._appliance_signatures.keys()) | set(
    (self.args or {}).get("sub_energy_sensors", {}).keys()
)
```

- `sub_energy_sensors` is parsed as a **list** of strings/dicts (`initialize`, lines 182–194).
  `.keys()` on a list raises `AttributeError`, swallowed by the outer `except Exception`
  → the whole scenario API silently fails for exactly the users who have appliances to schedule.
- Even if it were a dict, the keys would be entity IDs, not the `sub_*` prefixes the schedule
  uses, so the union is semantically wrong.
- Masked in tests because the app object is a `MagicMock` (its `.keys()` yields an empty iterator).

**Fix:** `{self._sub_sensor_prefix(e) for e in self._sub_energy_sensors}` and add a regression
test that uses a real list-typed `args["sub_energy_sensors"]`.

---

## Medium severity

### [x] M1 — Training lags are positional, prediction lags are temporal

`apps/energy_forecast/model.py:1707-1731` (training) vs `model.py:1734-1761` (prediction)

Training computes `lag_24h = gross_kwh.shift(24)` — 24 *rows* back, not 24 *hours* back.
The training frame has holes:

- the fetch filter drops zero-import hours (`ha_data.py:176`, `gross_kwh > 0`),
- `_retrain` drops EV-adjacent hours (`energy_forecast.py:936`),
- outages create gaps.

Around every hole, every lag and rolling feature is shifted off its true hour. At predict
time the same features are computed correctly by timestamp reindex → systematic
train/predict semantic mismatch.

**Fix:** reindex the training series to a continuous hourly grid before `shift()`
(NaNs then drop naturally via the existing `dropna`), or compute training lags by
timestamp like the prediction path.

**Related data concern:** for a PV household, hours with zero grid import are dropped
*before* the solar target correction runs, so fully solar-covered consumption hours never
enter training at all — `_apply_target_correction` only fixes hours that survived the
`> 0` filter.

### [x] M2 — The 5 freshest training days get median-imputed weather

`apps/energy_forecast/energy_forecast.py:1014`, `apps/energy_forecast/model.py:2476`

`end_date = min(…, today − 5 days)` because the Open-Meteo archive lags, but energy rows
run to now. Those last 5 days merge to NaN weather and are filled with column medians —
and with a 90-day half-life they are the *highest-weighted* rows in the training set.
This systematically blunts weather sensitivity where it matters most.

**Fix:** fetch the recent tail from the Open-Meteo forecast API with `past_days`
(already used at predict time in `fetch_open_meteo`) and stitch it onto the archive data.

### [x] M3 — Adaptive retrain bypasses the training lock

`apps/energy_forecast/energy_forecast.py:1524-1550`

`_update_cb` runs lock-free by design, but `_update_sensors → _maybe_adaptive_retrain →
self._retrain()` performs a full training run **without** `self._lock`, so it can race
the weekly `_retrain_cb`. AppDaemon's default thread pinning usually serializes this, but
the lock exists because that's not guaranteed (events/services can arrive on other threads).

Also: `train()` mutates shared state non-atomically (`_weather_tail` and signatures
mid-train; `self.model` then `self.feature_cols` at `model.py:673-674`), so the
"atomic model swap" comment in `_update_cb` overstates the guarantee — a concurrent
predict can see a new model with old feature columns.

**Fix:** route the adaptive path through the same non-blocking `acquire` pattern;
optionally assign `(model, feature_cols, medians, …)` as a single tuple/object swap.

### [x] M4 — "Today" blending is EV-inconsistent outside baseline_mode

`apps/energy_forecast/energy_forecast.py:1945-1957`

EV subtraction in training is unconditional → predictions are always EV-free baseline.
But `blended_actuals` only gets EV stripped when `baseline_mode=True`. In default mode the
`today` sensor sums EV-laden actuals for elapsed hours plus EV-free predictions for future
hours.

**Fix:** decide one semantic ("today = actual usage so far + expected baseline" vs
"today = baseline only") and apply it consistently in both modes.

### [x] M5 — Rolling-MAE sensors measure ~47h-ahead, not "day-ahead"

`apps/energy_forecast/energy_forecast.py:1262-1271`

Keep-first means each target hour is stored when it first enters the 48-hour window — at
horizon ≈ 47, not ≈ 24 as the comments claim. Consequences:

- `mae_7d` / `mae_30d` are 2-day-ahead metrics (doc/semantics drift).
- The adaptive-retrain comparison `live_MAE > threshold × cv_MAE` compares unlike
  quantities: CV MAE is computed at `hours_ahead=0` with real lags; live MAE at h≈47 with
  median/HOD-filled lags. Live > CV is structurally expected, not necessarily drift —
  threshold 2.0 may fire spuriously.

**Fix:** store predictions at a fixed ~24h horizon (skip the first 24 rows of each cycle),
or recalibrate the adaptive threshold acknowledging the horizon mismatch.

---

## Low severity

### [x] L1 — `shap_summary` uses the system clock instead of the configured timezone

`apps/energy_forecast/model.py:1136` — `pd.Timestamp.now().normalize()` is UTC in the
container; everywhere else `self._timezone` is used deliberately (documented in
`_maybe_adaptive_retrain`). The "today" SHAP slice is wrong for up to 2 h around midnight.
Same pattern (display-only, harmless): `last_trained` / `hours_since_trained` use
`datetime.now()`.

### [x] L2 — Holdout MAE is in-sample

`apps/energy_forecast/model.py:664-671` — the final model is fit on *all* rows, then
evaluated on the last 10% it has already seen. It's the `last_mae` fallback when CV is
skipped (< 500 rows), so new installs report optimistic accuracy.

### [x] L3 — SRG token not invalidated on auth failure

`apps/energy_forecast/weather.py:96-111` — a revoked/early-expired token causes silent
Open-Meteo fallback for the rest of the 55-minute cache window. Clear `_srg_token` on any
401/`RequestException` before falling back.

### [x] L4 — Total weather failure crashes the hourly update on fresh installs

`fetch_open_meteo` returns a column-less empty DataFrame on failure; with `_weather_tail`
also None (fresh install / old meta), `_engineer_features` raises `KeyError: 'timestamp'`
(`model.py:2413`). Caught and logged, but the hourly update is skipped. Return an
`_empty_weather_df()`-shaped frame instead so median-based prediction degrades gracefully.

### [x] L5 — `predict_scenario` drops `room_areas`

`apps/energy_forecast/model.py:1047-1086` — the scenario baseline computes thermal
pressure with `DEFAULT_ROOM_AREA_M2` for every room while the published forecast uses
configured areas → the scenario delta isn't measured against the same baseline.
Plumb `room_areas` through `predict_scenario` and `_get_scenario_cb`.

### [x] L6 — Scenario sensors bypass MQTT discovery

`apps/energy_forecast/energy_forecast.py:851-868` — `_publish_scenario_forecast` always
uses `set_state`, recreating the ghost-entity problem `_cleanup_legacy_states` exists to
remove in MQTT mode; the cleanup list also doesn't include the scenario entities.

### [x] L7 — Doc drift

- [x] `_resolve_programs_for_series` docstring updated: 1 h → 2 h (`ha_data.py`).
- [~] `HOLDOUT_FRACTION` name: existing comment already explains the inversion; name kept for backward compat — won't fix.
- [x] `_pred_history` comments: comment removed; `_accumulate_pred_history` docstring documents the correct 24-25h window.

### [x] L8 — deploy.py nits

`scripts/deploy.py`

- `repo_data` (line 110) is assigned and never used.
- Deploy never deletes remote files removed locally — stale modules keep loading until
  manually cleaned.
- `_ensure_dir` swallows all exceptions, hiding genuine permission errors.

---

## Performance

### [x] P1 — Feature matrix built three times per hourly update

`_update_sensors` calls `predict()`, `predict_intervals()`, and `shap_summary()`, each of
which runs `_prepare_prediction_X` from scratch (feature engineering, RC indoor-temp
projection, regime prediction). Computing `(future_hours, X)` once and passing it into all
three would cut the hourly callback cost by roughly two-thirds — relevant given the
documented AppDaemon 10 s callback concern.

### [x] P2 — `find_optimal_k` fits an informational RegimePredictor

`apps/energy_forecast/clustering.py:331-344` — full RandomForest + TimeSeriesSplit CV run
purely to produce one log line during auto-K selection.

---

## Security notes (no action required, recorded for awareness)

- Secrets come from env vars (deploy) and apps.yaml (SRG); backfill SQL is parametrized
  with a hardcoded column name; MQTT payloads are JSON-encoded. All good.
- The SHA-256 sidecars on the model pickles detect *corruption*, not tampering — anyone
  who can write the pkl writes the sidecar too. Fine for this trust boundary; don't read
  it as a security control.
- `pickle.load` of model files means the model directory is part of the attack surface if
  the Samba share is ever exposed.

---

## Repo hygiene

### [x] R1 — Stray artifacts in the tree

- `.ipynb_checkpoints` copies of app and test modules inside `apps/energy_forecast/` and
  `tests/` (deploy.py special-cases skipping them).
- `temp/`, `htmlcov/`, `.coverage`, `logs/`, `work/` in the repo root.
- Verify these are gitignored; `--cov=apps` in pyproject also counts checkpoint files.

### [x] R2 — `weather.py` imports pandas/numpy/requests at module top level

Every other module deliberately defers heavy imports for AppDaemon startup; harmless but
inconsistent with the established pattern.

---

## Positives (for the record)

- Disciplined DST handling — documented accepted edge cases instead of silent bugs.
- `shift(1)`-before-rolling leakage prevention mirrored correctly between training and
  prediction.
- Textbook CQR implementation (random calibration split, exchangeability rationale
  written down).
- Atomic pred-history saves; consistent "HA wins" cache-merge semantics; integrity-checked
  model persistence.

## Not covered by this review

The ~5,000 lines of tests, `scripts/analyze_*` tooling, `dashboard/`, and notebook
content were not deep-reviewed. The test suite and ruff were not executed — run both
before and after applying any of the fixes above.
