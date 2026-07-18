---
name: project-zero-diff-cache-gap-fix
description: Fixed 2026-07-17 on fix/zero-diff-cache-gap — hourly grid-import diffs of exactly 0.0 (real since solar) were silently dropped from the cache/training set instead of kept as zero; same root cause as [[project_actuals_history_freeze]]
metadata:
  type: project
---

**Found and fixed 2026-07-17**, branch `fix/zero-diff-cache-gap` (commits `3bc215b`, `b87b381`), off `dev`. **Merged and deployed 2026-07-17 as v0.12.0-alpha-4.**

**Root cause:** `_raw_to_kwh_diff()` in `ha_data.py` (used by both `fetch_energy_history()` and `fetch_recent_energy()`) filtered hourly grid-import diffs with `(diff > 0) & (diff < max_kwh)` — a diff of exactly `0.0` was silently dropped from the cache entirely, not recorded as a zero row. Before solar this never mattered (household always drew some baseline power every hour). Since SolarEdge commissioning (2026-07-16), an hour can legitimately have zero net grid import — solar covers the whole load — so real holes started forming in `energy_history.csv`. Reproduced by replaying the exact gap pattern from the live cache against `_add_lag_and_rolling_prediction`'s lookup logic: got `33/48` NaN for `lag_24h`, matching the live `32/48` warning within one row (cache freshness at time of pull). Cross-checked one gap window against raw HA sensor history via the REST API and confirmed `sensor.gplugk_z_ei` really did go flat/unavailable for ~2h during commissioning — this is also the root cause of [[project_actuals_history_freeze]] (same bug, different trigger: a longer real sensor outage vs. an hour of solar fully covering load).

**Confirmed as a genuine inconsistency, not by-design:** the sub-sensor fetch path (`fetch_recent_sub_sensor`/`fetch_sub_sensor_history`, same file) already keeps zero-diff rows — its docstring says so explicitly, and its filter has no lower-bound exclusion. Only the main energy-sensor path had the `>0` bug.

**Fix — 4 call sites, `>0` → `>=0`:**
1. `ha_data.py` `_raw_to_kwh_diff()` — cache-population filter.
2. `ha_data.py` `validate_energy_cache()` — health-check range widened `(0, MAX]` → `[0, MAX]` to match.
3. `energy_history_backfill.py` — same pattern in the standalone historical-backfill script.
4. `model.py` `EnergyForecastModel.train()` — training itself also dropped `gross_kwh==0` rows before fitting; fixed so genuine zero-consumption hours are kept as training examples.

**Follow-up fix from code review (`b87b381`):** the naive `clip(lower=0)` then `>=0` filter made a *negative* raw diff (meter reset) indistinguishable from a genuine zero — both got recorded as `gross_kwh=0.0`. Fixed by filtering on the pre-clip raw diff: a negative raw diff is dropped (as before, matching old meter-reset handling), while only a raw diff that is truly `0` produces a valid zero-consumption row. Applied in both `ha_data.py` and `energy_history_backfill.py`.

**Not fixed / explicitly deferred:** the EV-split (`ha_data.split_ev_charging`) and anomaly/MAE consumers were checked and don't assume `gross_kwh > 0`, so no changes needed there. The sub-sensor path's own pre-existing "keeps zeros" behavior (including for negative diffs there) was left untouched — out of scope, unrelated to this bug.

**Tests:** `tests/test_ha_data.py::TestRawToKwhDiff`, `tests/test_ha_data.py::TestValidateEnergyCache`, `tests/test_backfill.py::TestBackfillZeroConsumptionKept`, `tests/test_model.py::TestZeroConsumptionRowsKeptInTraining` — each new/changed test verified RED against pre-fix code before the fix made it GREEN. Full suite (887 tests) + ruff clean.

**How to apply:** Once merged/deployed, verify `lag_24h`/`lag_168h` NaN warnings stop recurring during normal operation, and that `_actuals_history` keeps advancing hourly (see [[project_actuals_history_freeze]] for the specific verification step). If a *different* NaN-lag warning shows up later, check first whether it's a genuine multi-hour sensor outage (this fix doesn't recover data that was never fetched at all) before assuming a regression.

**Follow-up (2026-07-18):** this exact "different NaN-lag warning" case showed up — not a multi-hour outage, but a sensor that stopped emitting states *entirely* (not just a `0.0`-diff between two real states). This fix only widened a filter on rows `resample()` had already produced; it couldn't catch a sensor that produces no rows at all past its last real push. See [[project_trailing_sensor_silence_fix]] for the distinct root cause and fix (an `end_time` param that extends the resample forward to "now").
