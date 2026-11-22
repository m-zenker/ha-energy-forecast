# Open-Meteo Archive Retry/Backoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a single transient 5xx/429/timeout from the Open-Meteo Archive API from degrading an entire retrain to median-imputed weather features (GitHub Issue #21) — add a bounded, backed-off retry inside `fetch_historical_weather()`, with no change to the caller's existing fallback behavior on genuine, sustained failure.

**Architecture:** A `for`/`else` retry loop wraps only the `requests.get()` + `raise_for_status()` call inside `fetch_historical_weather()` (`apps/energy_forecast/weather.py`). 5xx and 429 responses, plus `ConnectionError`/`Timeout`, retry up to 3 times with a fixed 2/4/8s backoff; any other 4xx, or JSON-parsing failures downstream of a successful response, fail immediately as today. On exhaustion, the last exception is re-raised unchanged — the existing caller-side `except (OSError, KeyError, ValueError)` in `energy_forecast.py:_retrain()` (median-imputation fallback) is untouched.

**Tech Stack:** Python 3.11 (verification sandbox) / 3.13 (project target per `pyproject.toml`'s `ruff` config — no syntax used here is version-sensitive), `requests` (already a dependency), stdlib `time` (already imported in `weather.py`). No new dependencies.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-27-openmeteo-archive-retry-backoff-design.md` (rev. 2, multi-stakeholder-reviewed). Every task below implements a specific section of it — cited inline.
- Run tests via this project's dedicated env: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v` — never bare `python`/`pytest`.
- **Test baseline (re-verified 2026-09-28 on this machine):** the full suite is green: 1134 passed, 0 failed, on `fix/partial-bucket-cache-poisoning`, which this branch is stacked on. The cloud sandbox reported one pre-existing failure (`TestLoadExcludedRanges::test_different_timezone_changes_spring_forward_detection`, Python 3.11, 11 skipped), but it does not reproduce in the project env. Every full-suite step below expects **0 failures**.
- All new tests go in `tests/test_weather.py` (Task 1) and `tests/test_energy_forecast.py` (Task 2) — no new test files.
- Every HTTP-error mock in the new tests must attach a `response` with an explicit plain-`int` `status_code` (never a bare `HTTPError(...)` with no `.response`, and never an unconfigured `MagicMock()` as the response) — this is the exact rev.-1 mocking bug the spec's multi-stakeholder review caught (spec §5, §7 findings SWE-1/Test-1/Test-2/Test-4). Task 1 defines a shared `_http_error(status)` helper for this; use it, don't hand-roll a mock inline.
- Terminal "success" responses in retry tests reuse `TestFetchHistoricalWeather()._make_archive_response(n)` (`tests/test_weather.py:208`), not a bare `MagicMock()` (spec §5, finding Test-6).
- `time.sleep` is monkeypatched in every retry test — no test may actually sleep.
- **Verified against real code before this plan was written:** every task's code and test snippets below were applied and run in a separate verification sandbox (same repo state, `dev` @ `4b5f9d1`) — 49/49 passed in `test_weather.py`, 2/2 in the new integration class, and the full suite showed exactly baseline (1103) + 11 new = 1114 passed, same single pre-existing failure. The verification edits were then reverted (not committed) — this plan re-derives the same edits fresh. If anything in this repo's `weather.py`/`energy_forecast.py`/tests has changed since `dev` @ `4b5f9d1`, re-check line numbers before applying Step 3 of each task.

---

### Task 1: Retry/backoff in `fetch_historical_weather()` + unit tests

Implements spec §2 (retry loop, retryable/non-retryable classification including the 429 addition) and §5 (unit test plan, corrected per §7's review findings).

**Files:**
- Modify: `apps/energy_forecast/weather.py` (`fetch_historical_weather`, currently lines 58-105; new module-level constants near `_WEATHER_COLUMNS`, currently lines 28-37)
- Test: `tests/test_weather.py` (new `_http_error` helper and `TestFetchHistoricalWeatherRetry` class, added after the existing `TestFetchHistoricalWeather` class, currently ending at line 266)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: no change to `fetch_historical_weather`'s signature or return type — only its internal retry behavior and the two new module-level constants (`_ARCHIVE_MAX_ATTEMPTS`, `_ARCHIVE_BACKOFF_S`). Task 2 depends on this task's retry-then-reraise behavior existing, but calls no new interface directly — it patches `fetch_historical_weather` as a whole.

- [ ] **Step 1: Write the failing tests**

Add `import requests` and `import pytest` to the top-level imports of `tests/test_weather.py` (currently just `from unittest.mock import MagicMock, patch` and `import pandas as pd`):

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
from energy_forecast import weather
from energy_forecast.weather import _supplement_from_open_meteo
```

Then add this, after the existing `class TestFetchHistoricalWeather:` block ends (currently line 266, right before the `# ── _supplement_from_open_meteo ──` section header):

```python
# ── fetch_historical_weather retry/backoff (#97) ───────────────────────────────


def _http_error(status: int) -> requests.HTTPError:
    """Build an HTTPError carrying a real status code, matching what
    res.raise_for_status() attaches on an actual requests.Response. The
    bare-HTTPError("404")-style mocks used elsewhere in this file (see
    TestFetchOpenMeteoNetworkErrors) leave .response as None and must NOT be
    reused for these tests — see spec
    2026-09-27-openmeteo-archive-retry-backoff-design.md §5/§7 (findings
    SWE-1, Test-1, Test-2).
    """
    err = requests.HTTPError(f"{status} error")
    err.response = MagicMock(status_code=status)
    return err


class TestFetchHistoricalWeatherRetry:
    """#97 — bounded retry with exponential backoff on transient Archive API failures."""

    def _archive_response(self, n: int = 3) -> MagicMock:
        return TestFetchHistoricalWeather()._make_archive_response(n)

    def _dates(self):
        from datetime import date

        return date(2026, 1, 1), date(2026, 1, 1)

    def test_503_then_200_retries_once(self, monkeypatch):
        start, end = self._dates()
        sleeps = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        with patch("requests.get", side_effect=[_http_error(503), self._archive_response()]):
            df = weather.fetch_historical_weather(47.0, 8.0, start, end)
        assert not df.empty
        assert sleeps == [2]

    def test_three_503s_then_200_succeeds_on_last_attempt(self, monkeypatch):
        start, end = self._dates()
        sleeps = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        with patch(
            "requests.get",
            side_effect=[_http_error(503), _http_error(503), _http_error(503), self._archive_response()],
        ) as mock_get:
            df = weather.fetch_historical_weather(47.0, 8.0, start, end)
        assert not df.empty
        assert sleeps == [2, 4, 8]
        assert mock_get.call_count == 4

    def test_four_503s_exhausts_all_attempts(self, monkeypatch):
        start, end = self._dates()
        sleeps = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        with patch("requests.get", side_effect=[_http_error(503)] * 4) as mock_get:
            with pytest.raises(requests.HTTPError):
                weather.fetch_historical_weather(47.0, 8.0, start, end)
        assert mock_get.call_count == 4
        assert sleeps == [2, 4, 8]

    def test_429_retried_like_5xx(self, monkeypatch):
        """New in rev. 2 — spec §7 finding SWE-3: 429 is the most realistic
        transient 4xx a free-tier API returns."""
        start, end = self._dates()
        monkeypatch.setattr("time.sleep", lambda s: None)
        with patch("requests.get", side_effect=[_http_error(429), self._archive_response()]):
            df = weather.fetch_historical_weather(47.0, 8.0, start, end)
        assert not df.empty

    def test_404_fails_immediately(self, monkeypatch):
        """The actual regression test for the status_code < 500 comparison
        (spec §7 finding Test-2) — must use _http_error(404), not a bare
        HTTPError, or this would pass for the wrong reason."""
        start, end = self._dates()
        sleeps = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        with patch("requests.get", side_effect=[_http_error(404)]) as mock_get:
            with pytest.raises(requests.HTTPError):
                weather.fetch_historical_weather(47.0, 8.0, start, end)
        assert mock_get.call_count == 1
        assert sleeps == []

    def test_response_none_fails_immediately(self, monkeypatch):
        """An HTTPError with no attached response object at all -- kept as its
        own test, separate from the 404 case above (spec §7 finding Test-5)."""
        start, end = self._dates()
        sleeps = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        bare_err = requests.HTTPError("unknown")  # .response defaults to None
        with patch("requests.get", side_effect=[bare_err]) as mock_get:
            with pytest.raises(requests.HTTPError):
                weather.fetch_historical_weather(47.0, 8.0, start, end)
        assert mock_get.call_count == 1
        assert sleeps == []

    def test_connection_error_then_200_retries(self, monkeypatch):
        start, end = self._dates()
        monkeypatch.setattr("time.sleep", lambda s: None)
        with patch("requests.get", side_effect=[requests.ConnectionError("no route"), self._archive_response()]):
            df = weather.fetch_historical_weather(47.0, 8.0, start, end)
        assert not df.empty

    def test_timeout_then_200_retries(self, monkeypatch):
        start, end = self._dates()
        monkeypatch.setattr("time.sleep", lambda s: None)
        with patch("requests.get", side_effect=[requests.Timeout("timed out"), self._archive_response()]):
            df = weather.fetch_historical_weather(47.0, 8.0, start, end)
        assert not df.empty

    def test_malformed_json_on_first_response_not_retried(self, monkeypatch):
        """Missing 'hourly' key is outside the retry loop's try block --
        must propagate immediately, not be retried (spec §2's stated scope)."""
        start, end = self._dates()
        sleeps = []
        monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
        mock = MagicMock()
        mock.raise_for_status = MagicMock()
        mock.json.return_value = {"metadata": "no hourly key"}
        with patch("requests.get", side_effect=[mock]) as mock_get:
            with pytest.raises(KeyError):
                weather.fetch_historical_weather(47.0, 8.0, start, end)
        assert mock_get.call_count == 1
        assert sleeps == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_weather.py::TestFetchHistoricalWeatherRetry -v`
Expected: every test in this class fails — `test_503_then_200_retries_once` etc. raise `requests.HTTPError` on the first call (no retry exists yet); `test_404_fails_immediately` and `test_response_none_fails_immediately` currently pass already (single-attempt behavior is the status quo) but re-run them anyway as part of this class so Step 4 catches any regression introduced by Step 3.

- [ ] **Step 3: Implement the retry loop**

In `apps/energy_forecast/weather.py`, add two module-level constants immediately after `_WEATHER_COLUMNS` (currently lines 28-37, right before the blank line preceding `def _parse_sunshine_min`):

```python
_WEATHER_COLUMNS = [
    "timestamp",
    "temp_c",
    "precipitation_mm",
    "sunshine_min",
    "wind_kmh",
    "cloud_cover_pct",
    "direct_radiation_wm2",
    "humidity",
]

_ARCHIVE_MAX_ATTEMPTS = 4  # 1 initial + 3 retries
_ARCHIVE_BACKOFF_S = (2, 4, 8)  # delay before each retry attempt (2nd, 3rd, 4th)
```

Then replace the current single-shot fetch inside `fetch_historical_weather` (currently lines 86-88):

```python
    res = requests.get(url, timeout=30)
    res.raise_for_status()
    h = res.json()["hourly"]
```

with:

```python
    last_exc: Exception | None = None
    for attempt in range(_ARCHIVE_MAX_ATTEMPTS):
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            break
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is None or (status < 500 and status != 429):
                raise  # 4xx other than 429 (or no response object) — not transient, fail immediately
            last_exc = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        if attempt < _ARCHIVE_MAX_ATTEMPTS - 1:
            delay = _ARCHIVE_BACKOFF_S[attempt]
            _LOGGER.info(
                "Open-Meteo archive request failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1,
                _ARCHIVE_MAX_ATTEMPTS,
                last_exc,
                delay,
            )
            time.sleep(delay)
    else:
        _LOGGER.warning(
            "Open-Meteo archive request failed after %d attempts: %s",
            _ARCHIVE_MAX_ATTEMPTS,
            last_exc,
        )
        raise last_exc

    h = res.json()["hourly"]
```

No new import needed — `time` is already imported at module level (`weather.py:4`) and `requests` is already imported inside this function (line 72, unchanged).

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_weather.py::TestFetchHistoricalWeatherRetry -v`
Expected: `10 passed`

- [ ] **Step 5: Run the full `test_weather.py` file to check for regressions in this module**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_weather.py -v`
Expected: all tests pass (existing `TestFetchHistoricalWeather`, `TestFetchOpenMeteo`, `TestFetchOpenMeteoNetworkErrors` classes untouched by this change — `fetch_open_meteo` is explicitly out of scope, spec §1).

- [ ] **Step 6: Run the full suite to check for regressions project-wide**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: baseline 1134 + 10 new = 1144 passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add apps/energy_forecast/weather.py tests/test_weather.py
git commit -m "feat: retry Open-Meteo archive fetch with backoff on transient 5xx/429/timeout"
```

---

### Task 2: Integration test — caller-side fallback still works after retry exhaustion

Implements spec §5's new `tests/test_energy_forecast.py` case (§7 finding Test-3): verifies the safety argument in spec §2 — "every retryable exception this loop can re-raise is already covered by the existing caller `except`" — against the real `_retrain()` caller, not just class-hierarchy prose.

**Files:**
- Modify: none (test-only task).
- Test: `tests/test_energy_forecast.py` (new `TestRetrainWeatherFetchRetryExhaustion` class, added after the existing `TestRetrainExcludedRanges` class)

**Interfaces:**
- Consumes: Task 1's `fetch_historical_weather` (patched to simulate post-retry-exhaustion failure, so this test never actually exercises the retry loop itself — that's Task 1's job. This test only exercises what happens *after* it re-raises).
- Produces: nothing for later tasks (last code task in this plan).

- [ ] **Step 1: Write the failing test**

Add this class to `tests/test_energy_forecast.py`, immediately after `class TestRetrainExcludedRanges:` ends (search for that class name; insert before whatever class follows it). Reuses the existing `_FakeRetrain`, `_make_energy_df`, `_empty_weather` helpers already defined earlier in this file (around lines 4004-4079):

```python
class TestRetrainWeatherFetchRetryExhaustion:
    """#97 — after fetch_historical_weather() exhausts its retries and re-raises,
    _retrain()'s existing except (OSError, KeyError, ValueError) fallback must
    still catch it and let the retrain proceed on median-imputed weather
    features, exactly as it does today for any other archive-fetch failure.
    Spec §7 finding Test-3: this was previously asserted only as prose about
    requests.RequestException subclassing OSError, never actually tested."""

    def _patch_retrain_deps(self, monkeypatch, energy_df, weather_exc):
        import energy_forecast.ha_data as ha_data_mod
        import energy_forecast.weather as weather_mod

        empty_df = pd.DataFrame()

        def _raise_weather_exc(*a, **kw):
            raise weather_exc

        monkeypatch.setattr(ha_data_mod, "fetch_energy_history", lambda *a, **kw: energy_df)
        monkeypatch.setattr(ha_data_mod, "split_ev_charging", lambda df, *a, **kw: (df, empty_df))
        monkeypatch.setattr(weather_mod, "fetch_historical_weather", _raise_weather_exc)
        monkeypatch.setattr(weather_mod, "fetch_open_meteo", lambda *a, **kw: _empty_weather())
        monkeypatch.setattr(ha_data_mod, "fetch_boolean_entity_history", lambda *a, **kw: empty_df)
        monkeypatch.setattr(ha_data_mod, "fetch_presence_history", lambda *a, **kw: empty_df)
        monkeypatch.setattr(ha_data_mod, "fetch_energy_history_15m", lambda *a, **kw: None)

    def test_http_error_after_retry_exhaustion_falls_back_to_median_imputation(self, tmp_path, monkeypatch, caplog):
        import logging

        import requests
        from energy_forecast.energy_forecast import EnergyForecast

        energy_df = _make_energy_df(200)
        exhausted_exc = requests.HTTPError("503 error after 4 attempts")
        self._patch_retrain_deps(monkeypatch, energy_df, exhausted_exc)

        stub = _FakeRetrain(tmp_path / "energy_history.csv")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._retrain(stub)  # must not raise

        assert any("historical weather fetch failed" in r.message.lower() for r in caplog.records)
        stub._ml_model.train.assert_called_once()

    def test_connection_error_after_retry_exhaustion_falls_back_to_median_imputation(
        self, tmp_path, monkeypatch, caplog
    ):
        """Same as above but for the other retried-then-exhausted exception type
        (ConnectionError/Timeout branch, not the HTTPError branch)."""
        import logging

        import requests
        from energy_forecast.energy_forecast import EnergyForecast

        energy_df = _make_energy_df(200)
        exhausted_exc = requests.ConnectionError("no route after 4 attempts")
        self._patch_retrain_deps(monkeypatch, energy_df, exhausted_exc)

        stub = _FakeRetrain(tmp_path / "energy_history.csv")
        with caplog.at_level(logging.WARNING, logger="energy_forecast"):
            EnergyForecast._retrain(stub)  # must not raise

        assert any("historical weather fetch failed" in r.message.lower() for r in caplog.records)
        stub._ml_model.train.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py::TestRetrainWeatherFetchRetryExhaustion -v`
Expected: both tests currently **pass already** — Task 1 didn't change `_retrain()`'s caller-side exception handling at all, only `fetch_historical_weather`'s internal retry behavior, and `requests.HTTPError`/`requests.ConnectionError` already subclass `OSError` today. This step is a checkpoint confirming that fact is actually true against the real code (not just spec prose), not a red-green cycle for new production code — same pattern as the reference plan's Task 4 Step 2. If either test fails here, stop: it means the class-hierarchy assumption in spec §2 is wrong and needs to be re-examined before proceeding, not patched over.

- [ ] **Step 3: (no production code change — see Step 2)**

- [ ] **Step 4: Run the full suite to check for regressions**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: 1146 passed, 0 failed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_energy_forecast.py
git commit -m "test: verify _retrain()'s median-imputation fallback still catches exhausted archive retries"
```

---

### Task 3: CHANGELOG entry and spec status update

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `docs/superpowers/specs/2026-09-27-openmeteo-archive-retry-backoff-design.md` (status line only)

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing (end of plan).

- [ ] **Step 1: Add a CHANGELOG.md entry**

Add to the `### Fixed` section under `## [Unreleased]` (#97 is a robustness fix for reported GitHub Issue #21, and sits next to the #24 fix entries) (follow the existing entries' level of detail — see e.g. the #92/UA_eff entry for the house style):

```markdown
- `apps/energy_forecast/weather.py` — `fetch_historical_weather()` (Open-Meteo Archive API,
  used during backfill/retrain) now retries up to 3 times with exponential backoff (2/4/8s) on
  HTTP 5xx, HTTP 429, and connection/timeout errors before falling back to the existing
  median-imputation path (ROADMAP #97, GitHub Issue #21). Other 4xx responses and JSON-parsing
  failures still fail immediately, unchanged.
```

- [ ] **Step 2: Update the spec's status line**

In `docs/superpowers/specs/2026-09-27-openmeteo-archive-retry-backoff-design.md`, change:

```
**Status:** Proposed — pending approval before implementation
```

to:

```
**Status:** Implemented (this plan) — see docs/superpowers/plans/2026-09-27-openmeteo-archive-retry-backoff.md
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md docs/superpowers/specs/2026-09-27-openmeteo-archive-retry-backoff-design.md
git commit -m "docs: changelog entry and spec status update for #97 (archive retry/backoff)"
```
