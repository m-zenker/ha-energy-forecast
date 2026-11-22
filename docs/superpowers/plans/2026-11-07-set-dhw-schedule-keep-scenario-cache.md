# Keep the Scenario Cache on `set_dhw_schedule` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the commit-induced gap only: a `set_dhw_schedule` commit no longer makes `get_scenario` return nothing until the next hourly cycle. See "Not addressed" for other paths to the same symptom.

**Architecture:** Remove the `_cached_forecast_df = None` line from `_set_dhw_schedule_cb`. That frame is the raw weather forecast (`weather.fetch_forecast`, `energy_forecast.py:1870`), and the hourly cycle overwrites it unconditionally (`:2051`). `predict_scenario` reads the live `committed_override` on each call (`model.py`, natural-baseline block), so a commit takes effect in scenarios without the reset.

**Tech Stack:** Python 3.13, AppDaemon 4.x, pytest.

**Spec:** `docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md` (§ around lines 350, 352, 459). **This plan amends the spec**: it mandates "immediately invalidates `_cached_forecast_df`" so "the next HA sensor publish reflects the updated schedule". The stated purpose stays met, but the mechanism is dropped. Owner approval of this plan covers that amendment.

OWNER-APPROVED docs/superpowers/plans/2026-11-07-set-dhw-schedule-keep-scenario-cache.md 2026-11-07

**Base branch:** `dev`
**Size:** XS
**Risk tags:** `BEHAVIOUR`. No `REMOVES` (no key, entity or flag), no `UI`, no `CONFIG-STEP`.

## Evidence

Live 2026-11-07, add-on log (local time):

```
05:04:55.446 energy_forecast: DHW schedule committed: {'comfort_boost': ['2026-11-08', 12, 54.0]}
06:00:00.084 WARNING energy_forecast: get_scenario called before first forecast cycle
06:00:05.088 WARNING heat_pump_optimiser: DHW scenario timeout for candidate hour 12
06:00:05.091 heat_pump_optimiser: Legionella candidate hour 12 scored N/A (timeout) — passes min coverage (0.50)
06:00:05.105 heat_pump_optimiser: Legionella cycle scheduled at 12:00 (forced=False)
```

The refill was due at 06:01:00. This is the only such warning in the log retained since 2026-10-20, and the first `hef_timeout` in EM's `legionella_log/`. Comfort-boost commits are not rare (10-31, 11-05, 11-07 in the same log), so exposure depends on one landing in the hour before a scoring run.

**No data-simulation evidence:** this is a control-flow fix, supported by code reading and the log above. It changes no forecast values outside the gap.

## Behaviour change to accept

Inside the gap (commit until the next `:01`), scenario callers get a real score instead of a 5 s timeout:
- **Legionella:** a candidate can now be `below_min` and move to the next hour or defer, where it would have been `timeout_pass` and booked. This restores the intended gate.
- **Appliance scheduler:** a candidate is scored instead of skipped, and no longer counts towards `scenario_timeout`.
- **Comfort boost and legionella fallback:** get a real hef answer instead of their timeout path.
- **Weather age:** scores in the gap use a weather frame up to about 1 h old, the same as any scenario call between cycles.
- **Coverage baseline:** EM's `_dhw_coverage`/v1 denominator comes from its published bridge sensors (`heat_pump.py` `_baseline_kwh_for_hour`), last refreshed at `:01` and without a just-made commit, while hef's delta is relative to the commit-inclusive baseline. In the gap the denominator can be slightly off, with no double count. A candidate is never scored against its own commit, because EM commits after scoring.

## Not addressed

These paths can give the same 5 s timeout and stay out of scope:
- **After a hef restart:** the cache is empty until the first cycle at `run_in(130)` (`energy_forecast.py:448`). If that cycle returns early (`:1284`) or aborts (`:1291`, "Sensor update failed"), it stays empty until the next `:01` succeeds.
- **Empty weather frame (unverified):** `weather.py:352-361` returns an empty frame on Open-Meteo failure, which passes the `is None` guard; `predict_scenario` may then raise (`:1155`, no event fired).
- **Slow scoring (unverified):** a DHW-override scenario runs `predict` twice (`model.py:1500`, `:1525`), which could exceed EM's 5 s timeout.

A future `before first forecast cycle` warning that does not follow a restart means this fix is incomplete.

## Global Constraints

- Tests run only through `/home/jovyan/my_envs/ha-energy-forecast/bin/python`.
- Work in worktree `/home/jovyan/work/ha-energy-forecast/.worktrees/fix-keep-scenario-cache`, branch `fix/set-dhw-schedule-keep-scenario-cache`, cut from `dev`. Use `git -C <path>` and absolute paths; check `branch --show-current` before committing; no `stash/checkout/reset/clean/restore/--amend`.
- No change in ha-energy-manager: EM's fail-open on timeout stays as it is.
- Finished work merges into `train/next` only after the owner's ready-to-merge click.

## Review Focus

1. **Before the first cycle after a restart:** `get_scenario` still warns and fires no event. The guard at `:1096` stays. Already pinned by `tests/test_scenario_service.py::…::test_no_cache_logs_warning`, which must stay green unmodified.
2. **Clear commits (`{kind: None}`):** same path as a set; the cache is kept. Pinned by `test_clear_commit_keeps_scenario_cache`.
3. **Physics not configured:** the callback still returns early with its warning. Existing tests cover this.
4. **Failed commit:** `commit_dhw_schedule` raising leaves the cache in place, as it already did because the reset came after the commit. Pinned by `test_failed_commit_keeps_scenario_cache`.
5. **Next publish reflects the commit (the spec's purpose):** already pinned by `TestServingOverrideCorrection` (`tests/test_energy_forecast_physics_config.py:1146`; publish path reads `committed_override`). `tests/test_model.py::…::test_set_dhw_schedule_then_get_scenario_with_different_override` pins the premise that scenarios read the override at call time. Both must stay green unmodified.

---

### Task 1: Keep the cache and amend the spec

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:1091` (docstring), `:1450-1466` (`_set_dhw_schedule_cb`)
- Modify: `docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md` (lines 66, 350, 352, 459, 469)
- Modify: `README.md` (lines 189, 1126-1127)
- Modify: `CHANGELOG.md` (`[Unreleased]`, via `@changelog-writer`; leave the historical `:374-375` entry alone)
- Test: `tests/test_energy_forecast_physics_config.py` (`TestSetDhwScheduleService`, `:1031`)

**Interfaces:**
- Consumes: `_make_app(args)` test helper; `_set_dhw_schedule_cb(namespace, domain, service, kwargs)`; `_get_scenario_cb(namespace, domain, service, kwargs)`, called directly in tests.
- Produces: nothing new. The `set_dhw_schedule` and `get_scenario` service contracts are unchanged.

- [ ] **Step 1: Write the failing tests**

Rename `test_commits_schedule_and_invalidates_cache` to `test_commit_keeps_scenario_cache`. Hold the seeded frame in a local `df`. Keep its commit assertion and replace the last assertion with an identity check, `assert app._cached_forecast_df is df`. In `test_comfort_boost_3elem_payload_forwards_unmodified` (`:1115`), make the same change and drop the "cache invalidated" comment. Add:

- `test_clear_commit_keeps_scenario_cache`: kwargs `{"dhw_schedule": {"legionella": None}}`; asserts `is df`.
- `test_failed_commit_keeps_scenario_cache`: `commit_dhw_schedule` patched with `side_effect=RuntimeError`; asserts no raise and `is df`.
- `test_get_scenario_answers_after_commit`: bind `app._get_scenario_cb = EnergyForecast._get_scenario_cb.__get__(app, type(app))` (`_make_app` does not bind it), commit, then call it with `app._ml_model.predict_scenario` patched to a small frame and `app.fire_event` patched; asserts `fire_event` is called once with `"energy_forecast_scenario_result"`. This proves the guard no longer trips, not scenario accuracy.

- [ ] **Step 2: Run them and confirm the failures**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py -v --no-cov -k "keeps_scenario_cache or answers_after_commit or comfort_boost_3elem"`
Expected: FAIL for `test_commit_keeps_scenario_cache`, `test_clear_commit_keeps_scenario_cache`, `test_comfort_boost_3elem_payload_forwards_unmodified` and `test_get_scenario_answers_after_commit`. PASS for `test_failed_commit_keeps_scenario_cache`, which pins existing behaviour.

- [ ] **Step 3: Remove the reset**

Delete line `:1463`. Reword the `_set_dhw_schedule_cb` docstring: it commits the override, keeps the cached weather frame, and the next hourly cycle publishes the new schedule. In the `get_scenario` docstring (`:1091`), replace "this app only invalidates its own _cached_forecast_df" with a statement that this app holds no schedule-dependent scenario cache. The callers' duty to flush their own cache stays. Leave the `:1096` guard and the `:2625` fallback as they are.

- [ ] **Step 4: Run the targeted tests and the spec-purpose test**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast_physics_config.py tests/test_model.py tests/test_scenario_service.py -v --no-cov -k "scenario_cache or answers_after_commit or comfort_boost_3elem or TestServingOverrideCorrection or test_no_cache_logs_warning or test_set_dhw_schedule_then_get_scenario_with_different_override"`
Expected: all PASS.

- [ ] **Step 5: Amend the spec and README**

Spec lines 66, 350, 352, 459, 469: state that `set_dhw_schedule` keeps `_cached_forecast_df` (raw weather), and the next hourly publish reflects the schedule because the committed override is read at prediction time. Add one dated note: invalidation removed 2026-11-07 because it blocked `get_scenario` between a commit and the next cycle. Keep the callers' secondary-cache flush duty unchanged. README lines 189 and 1126-1127: same statement, keeping the caller cache contract.

- [ ] **Step 6: Run the full suite**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -q --tb=short --no-cov`
Expected: 0 failed. Baseline 1316 passed and 11 skipped (`train/next` ae9ee8a, 2026-11-07), plus 3 net new tests.

- [ ] **Step 7: Update the changelog and commit**

Add a `[Unreleased]` Fixed entry through `@changelog-writer`, then:

```bash
git -C /home/jovyan/work/ha-energy-forecast/.worktrees/fix-keep-scenario-cache add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast_physics_config.py docs/superpowers/specs/2026-06-22-physics-ml-hybrid-design.md README.md CHANGELOG.md
git -C /home/jovyan/work/ha-energy-forecast/.worktrees/fix-keep-scenario-cache commit -m "fix(scenario): keep forecast cache on set_dhw_schedule so get_scenario answers between cycles"
```

Add the session's Co-Authored-By and Claude-Session trailers. Execution stops here at ready-to-merge.

## Post-deploy verification

Deploy away from the 06:00 legionella check and comfort-boost walks, since the restart reopens the ~130 s empty-cache window. Verification is passive: it completes only after a natural commit is followed by a scoring run within the hour. Do not probe with an artificial `set_dhw_schedule`, which would mutate `committed_override` and `override_history`.

- `grep "get_scenario called before first forecast cycle"` in the add-on log: only allowed right after a restart.
- `legionella_log/`: no `hef_timeout: true` candidate following a commit in the prior hour.
- Next `DHW schedule committed` line: the following `:01` cycle publishes normally.
