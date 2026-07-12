# Scenario-Result `request_id` Echo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `_get_scenario_cb`'s `energy_forecast_scenario_result` event never echoes back a `request_id`, even when the caller supplied one — so any caller trying to correlate its own in-flight `get_scenario` request with the right response has nothing to match on. Echo the incoming `request_id` (or `None` if the caller didn't send one) back in the fired event.

**Architecture:** No design change — one new kwarg on an existing `fire_event` call, sourced directly from the same `kwargs` dict `_get_scenario_cb` already reads `schedule`/`dhw_schedule`/`publish` from. No new state, no new config, no behavior change for callers that don't send `request_id` (they get `request_id=None` in the response, same as receiving no key at all from a caller's `dict.get("request_id")` perspective).

**Why now:** discovered while running `ha-energy-manager`'s "hef Physics Adoption Plan 2" Deployment Precondition checks (`ha-energy-manager`'s `docs/superpowers/specs/2026-07-10-hef-physics-adoption-design.md`). That plan built a shared `ScenarioScorer` class on the EM side that sends a `request_id` with every `get_scenario` call and — as of a recent hardening (its own finding #17) — **rejects** any response missing one, specifically to prevent two concurrently in-flight scoring runs (EM now has two independent callers: `appliance_scheduler.py` and a new legionella DHW-schedule gate in `heat_pump.py`) from accepting each other's answers. Since this repo never echoes `request_id` back, every EM-side response is currently rejected as unidentifiable — 100% of `get_scenario` calls from EM time out once that branch deploys. This is the fix on hef's side; EM keeps its reject-on-missing/mismatched logic as originally designed, no compatibility shim needed there.

**Tech Stack:** Python 3.13, pytest, `unittest.mock.MagicMock` (matches this repo's existing `_get_scenario_cb` test pattern in `tests/test_scenario_service.py` — no new dependencies.

## Global Constraints

- Base branch: `dev` (this is what's currently deployed and what EM's Deployment Precondition check confirmed is live — `main` doesn't have `_set_dhw_schedule_cb`/the physics scenario path at all yet, so a fix based on `main` wouldn't reach the instance that needs it).
- Branch name: `fix/scenario-result-request-id` — create it with `/feat fix/scenario-result-request-id ha-energy-forecast` (per `CLAUDE.md` — pass the bare `TYPE/NAME [PROJECT]` form, no prose, or it prints its own instructions instead of running).
- Run the **full** suite (`python -m pytest tests/ -v`) after the fix, not just the new tests — per project `CLAUDE.md`.
- **Scope is exactly one kwarg on one `fire_event` call.** Do not touch `_set_dhw_schedule_cb` (`energy_forecast.py:1354`, on this same `dev` branch) — it never fires an event in response to its call (it's a one-way commit, callers don't wait on or correlate a reply), so a `request_id` echo doesn't apply there. Do not add `request_id` handling to any other service callback in this file.
- Do not change `_get_scenario_cb`'s validation/error-handling behavior in any other way — the `no cache → WARNING, return early`, `invalid schedule → WARNING, return early`, and `predict_scenario raises → ERROR, no fire_event` paths (all covered by existing tests in `tests/test_scenario_service.py`) must remain exactly as they are; none of them reach the `fire_event` call, so this fix cannot affect them, but don't refactor anything nearby while you're in the function.
- After this fix ships and deploys, `ha-energy-manager`'s `ScenarioScorer` (`apps/energy_manager/forecast/scenario_scorer.py`, on its `fix/hef-physics-plan1-sensor-bridging` branch) needs no further changes — its existing "reject missing or mismatched `request_id`" logic starts working correctly against this repo's real behavior once this lands. That's a separate repo/branch; not part of this plan's scope, just context for why the deploy ordering matters (see "After implementation" below).

---

## Current State Reference

Read before starting — these are the exact anchors this plan diffs against, on `dev`:

- `apps/energy_forecast/energy_forecast.py:1036-1110` — `_get_scenario_cb`, the full method. Reads `kwargs.get("schedule", {})`, `kwargs.get("dhw_schedule")`, `kwargs.get("publish", False)` — this plan adds a fourth `kwargs.get("request_id")` read, used only at the `fire_event` call.
- `apps/energy_forecast/energy_forecast.py:1105-1108` — the exact `fire_event` call this plan modifies:
  ```python
  self.fire_event(
      "energy_forecast_scenario_result",
      forecast=result_df.to_dict("records"),
  )
  ```
- `tests/test_scenario_service.py:46-131` — `TestGetScenarioCb`, the test class this plan adds to. `test_fires_result_event_with_forecast` (`:60-86`) is the closest existing precedent: builds a cached baseline df, calls `_get_scenario_cb` directly as an unbound method (`EnergyForecast._get_scenario_cb(app, "homeassistant", "energy_forecast", "get_scenario", {kwargs})`), asserts on `app.fire_event.call_args[0][0]` (event name) and `app.fire_event.call_args[1]` (kwargs dict).
- `tests/test_scenario_service.py:21-35` — `_make_app(cached_df=None)`, the shared `MagicMock`-based app fixture this plan's new tests reuse unchanged.
- `tests/test_scenario_service.py:38-40` — `_make_baseline_df(start=..., n=48)`, the shared cached-forecast-df fixture this plan's new tests reuse unchanged.

---

### Task 1: Echo `request_id` in the scenario-result event

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:1105-1108` (inside `_get_scenario_cb`)
- Modify: `tests/test_scenario_service.py` (two new tests in `TestGetScenarioCb`)
- Modify: `CHANGELOG.md` (new entry under `## [Unreleased]`)

**Interfaces:**
- Consumes: nothing new — reads the same `kwargs: dict` parameter `_get_scenario_cb` already has.
- Produces: nothing new consumed by later tasks — this is the only task.

- [ ] **Step 1: Write the failing tests**

Add these two tests to `tests/test_scenario_service.py`'s `TestGetScenarioCb` class, immediately after `test_fires_result_event_with_forecast` (after line 86, before `test_room_areas_forwarded_to_predict_scenario`):

```python
    def test_echoes_request_id_when_caller_supplies_one(self):
        """A caller-supplied request_id must be echoed back verbatim in the
        result event, so concurrent callers can correlate their own request
        with the right response (ha-energy-manager's ScenarioScorer rejects
        any response it can't correlate — see this plan's docstring)."""
        from energy_forecast.energy_forecast import EnergyForecast

        cached_df = _make_baseline_df()
        app = _make_app(cached_df=cached_df)

        scenario_result = cached_df.copy()
        scenario_result["delta_kwh"] = 0.0
        app._ml_model.predict_scenario.return_value = scenario_result

        EnergyForecast._get_scenario_cb(
            app,
            "homeassistant",
            "energy_forecast",
            "get_scenario",
            {"schedule": {}, "publish": False, "request_id": "abc-123"},
        )

        app.fire_event.assert_called_once()
        _event_name, kwargs = app.fire_event.call_args[0][0], app.fire_event.call_args[1]
        assert kwargs["request_id"] == "abc-123"

    def test_request_id_is_none_when_caller_omits_it(self):
        """Backward compatibility: a caller that never sends request_id (e.g.
        an older integration, or a manual service call from HA's UI) must
        keep working exactly as before — no request_id key required in the
        call, and the response carries request_id=None rather than raising
        or omitting the key inconsistently."""
        from energy_forecast.energy_forecast import EnergyForecast

        cached_df = _make_baseline_df()
        app = _make_app(cached_df=cached_df)

        scenario_result = cached_df.copy()
        scenario_result["delta_kwh"] = 0.0
        app._ml_model.predict_scenario.return_value = scenario_result

        EnergyForecast._get_scenario_cb(
            app,
            "homeassistant",
            "energy_forecast",
            "get_scenario",
            {"schedule": {}, "publish": False},
        )

        app.fire_event.assert_called_once()
        _event_name, kwargs = app.fire_event.call_args[0][0], app.fire_event.call_args[1]
        assert kwargs["request_id"] is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/jovyan/work/ha-energy-forecast && python -m pytest tests/test_scenario_service.py -v -k "request_id"`
Expected: FAIL — both new tests raise `KeyError: 'request_id'` at `kwargs["request_id"]`, since the fired event currently carries no such key at all.

- [ ] **Step 3: Implement the fix**

In `apps/energy_forecast/energy_forecast.py`, replace the `fire_event` call (lines 1105-1108):

```python
            self.fire_event(
                "energy_forecast_scenario_result",
                forecast=result_df.to_dict("records"),
            )
```

with:

```python
            self.fire_event(
                "energy_forecast_scenario_result",
                forecast=result_df.to_dict("records"),
                request_id=kwargs.get("request_id"),
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/jovyan/work/ha-energy-forecast && python -m pytest tests/test_scenario_service.py -v`
Expected: PASS — full file, including all pre-existing `TestGetScenarioCb`/`TestGetScenarioCbErrors`/`TestPublishScenarioForecast`/`TestGetScenarioValidation`/`TestGetScenarioCbSubSensors`/`TestMqttSwVersion`/`TestPublishScenarioMqtt` tests (this catches any accidental regression in the surrounding validation logic, even though this change shouldn't touch it).

- [ ] **Step 5: Update CHANGELOG.md**

Add this entry under `## [Unreleased]` (top of the file, before the most recent dated release section):

```markdown
### Fixed
- `apps/energy_forecast/energy_forecast.py` — `_get_scenario_cb`'s `energy_forecast_scenario_result`
  event never echoed back a caller-supplied `request_id`, even though the service accepts one.
  Callers with more than one `get_scenario` request potentially in flight at once (e.g.
  `ha-energy-manager`'s `ScenarioScorer`, used by both its appliance scheduler and its legionella
  DHW-schedule gate) had no way to tell which in-flight request a given response event answered,
  and had started rejecting every response as uncorrelated. `request_id` (or `None`, if the caller
  didn't send one) is now echoed back verbatim in the fired event.
```

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_scenario_service.py CHANGELOG.md
git commit -m "fix: echo request_id in energy_forecast_scenario_result event"
```

---

### Task 2: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `cd /home/jovyan/work/ha-energy-forecast && python -m pytest tests/ -v`
Expected: 100% pass, no regressions outside `tests/test_scenario_service.py` (this plan touches no other file, so any other failure indicates a pre-existing/unrelated issue — investigate before proceeding, don't assume it's unrelated).

- [ ] **Step 2: Run ruff, if available in this environment**

Run: `ruff check apps/energy_forecast/energy_forecast.py tests/test_scenario_service.py && ruff format --check apps/energy_forecast/energy_forecast.py tests/test_scenario_service.py`
Expected: no violations. If `ruff` is unavailable in this environment (a known gap observed on the `ha-energy-manager` side of this same dev container — see that repo's `MEMORY.md`), note that explicitly rather than silently skipping, and flag it as a pre-merge CI/environment gap, not a code issue.

- [ ] **Step 3: Commit any formatting fixes**

```bash
git add -A
git commit -m "chore: ruff format after request_id echo fix" --allow-empty
```

(Use `--allow-empty` only if Step 2 found nothing to fix or ruff was unavailable — otherwise this commits real formatting changes.)

---

## After implementation

Per project workflow (`CLAUDE.md`): update `ROADMAP.md` if applicable and `MEMORY.md`, then merge to `dev` and deploy (`@deploy-agent`) — `dev` is what's live, and this fix needs to reach the live instance before (or in the same deploy window as) `ha-energy-manager`'s "hef Physics Adoption Plan 2" branch (`fix/hef-physics-plan1-sensor-bridging`) goes live. **Deploy ordering matters**: if EM's branch deploys first, its `get_scenario` calls will time out (rejected as uncorrelated) until this fix reaches the live hef instance too — not a crash, EM's legionella gate fails open safely, but its appliance scheduler will silently stop scheduling anything until both sides are live. Coordinate the two deploys, or deploy this one first.
