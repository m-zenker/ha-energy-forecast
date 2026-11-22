# Open-Meteo Archive Retry/Backoff on Transient 5xx — Design Spec

**Date:** 2026-09-27 (rev. 2 — post multi-stakeholder review)
**Status:** Implemented (this plan) — see docs/superpowers/plans/2026-09-27-openmeteo-archive-retry-backoff.md
**Branch base:** `dev`
**Roadmap item:** #97
**Source:** GitHub Issue #21 (2026-09-17)

**Revision note:** rev. 1 was reviewed in parallel by three independent reviewers (correctness/
software engineering, reliability/operations, test-quality) against the same draft. Two findings
were raised independently by more than one reviewer: (1) the proposed tests, if built by copying
this codebase's existing `HTTPError`-mocking pattern, would either fail to retry at all or silently
pass for the wrong reason, because that pattern never sets `.response`; and (2) the "~14s worst
case" latency claim ignored the unchanged `timeout=30` on the `Timeout`/`ConnectionError` retry
path, understating worst case by roughly 10x. Both are fixed below (§5, §2/§6). See §7 for the full
findings-to-changes mapping.

## 1. Problem & Motivation

A user hit a `503` from the Open-Meteo Archive API during backfill/retrain. The reporter's own
diagnosis (`sunshine_duration` being daily-only) was checked by replaying the exact failing request
and found wrong — the response shape was fine. The real gap: `weather.fetch_historical_weather()`
(`apps/energy_forecast/weather.py:58`) makes a single `requests.get()` call with no retry. One
flaky upstream response fails the whole fetch.

The failure isn't silent — the caller (`energy_forecast.py:1689-1701`) already catches
`(OSError, KeyError, ValueError)` (covers `requests.RequestException`, which subclasses `OSError`)
and falls back to median-imputed weather features with a `WARNING` log. So this isn't a crash bug;
it's an avoidable accuracy hit — a transient 503 currently degrades an entire retrain's weather
features when a few seconds' wait would have gotten a clean response.

**Scope note:** this applies only to `fetch_historical_weather()` (the Archive API,
`archive-api.open-meteo.com`, used for backfill/retrain). `fetch_open_meteo()`
(`weather.py:271`, the live forecast API) already wraps its call in its own
`try/except (requests.RequestException, KeyError, ValueError)` that returns an empty DataFrame —
different failure shape, called far more often (hourly), and out of scope for this issue. Not
touched here.

## 2. Retry Logic

New retry loop inside `fetch_historical_weather()`, wrapping only the `requests.get()` +
`raise_for_status()` call — parsing (`res.json()["hourly"]`, column construction) stays outside
the loop and outside retry scope, since a malformed body is not a transient condition retrying
would fix.

```python
_ARCHIVE_MAX_ATTEMPTS = 4  # 1 initial + 3 retries
_ARCHIVE_BACKOFF_S = (2, 4, 8)  # delay before each retry attempt (2nd, 3rd, 4th)

def fetch_historical_weather(...) -> pd.DataFrame:
    ...
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
                attempt + 1, _ARCHIVE_MAX_ATTEMPTS, last_exc, delay,
            )
            time.sleep(delay)
    else:
        _LOGGER.warning(
            "Open-Meteo archive request failed after %d attempts: %s",
            _ARCHIVE_MAX_ATTEMPTS, last_exc,
        )
        raise last_exc

    h = res.json()["hourly"]
    ...  # unchanged
```

(`time` is already imported at module level in `weather.py:4`; no new import needed.)

### Retryable vs. not

| Condition | Action |
|---|---|
| HTTP 5xx (`HTTPError` with `response.status_code >= 500`) | Retry |
| HTTP 429 (`HTTPError` with `response.status_code == 429`) | Retry — **added after review** (§7, finding SWE-3): Open-Meteo is a free-tier public API, and 429 is the most realistic transient 4xx it can return; unlike 400/401, waiting is exactly the correct response. No `Retry-After` parsing — the fixed backoff schedule is used as-is. |
| `requests.ConnectionError` (e.g. DNS failure, connection refused) | Retry |
| `requests.Timeout` | Retry |
| HTTP 4xx other than 429 (`HTTPError` with `response.status_code < 500`) | **Fail immediately** — a client error (bad params, auth) will not resolve itself by waiting |
| `HTTPError` with no `response` object at all (`exc.response is None`) | **Fail immediately** — not a case actually observed against this API (`requests` always attaches a response when `raise_for_status()` raises); treated conservatively as non-transient rather than guessed at |
| `KeyError` / `ValueError` (malformed JSON, missing `hourly` key) | **Fail immediately, unchanged** — not part of the retry loop at all; existing caller `except` still catches these |

After exhausting all attempts, the last exception is re-raised — the existing caller-side
`except (OSError, KeyError, ValueError)` in `energy_forecast.py:1694` is untouched and still
provides the median-imputation fallback (`requests.RequestException`, the parent of `HTTPError`/
`ConnectionError`/`Timeout`, subclasses `OSError`, so every retryable exception this loop can
re-raise is already covered by that existing `except`; §5 adds a test asserting this directly
rather than relying on the class hierarchy as unverified prose). **Net effect:** total failure
behavior is unchanged; only the transient-failure path now attempts to self-heal first.
**Timing correction from rev. 1** (§7, findings SWE-2/Reliability-1): the backoff sleeps alone
total ~14 s (2+4+8), but `requests.get(url, timeout=30)` is unchanged per attempt — a hung
connection (`Timeout`/`ConnectionError`, not a fast-failing 5xx) can cost up to 30 s *per attempt*,
so the true worst case is closer to **4×30 s + 14 s ≈ 134 s**, not 14 s. This is still acceptable
(see §6) but rev. 1 materially understated it.

## 3. Config

None. No new `apps.yaml` key — retry count and backoff are fixed constants
(`_ARCHIVE_MAX_ATTEMPTS`, `_ARCHIVE_BACKOFF_S` at module level in `weather.py`, next to
`_WEATHER_COLUMNS`). This is a one-shot backfill/retrain call (not the hourly hot path), so even
the corrected ~134 s worst case (§2) on a genuine outage is a non-issue, and a config knob would be
unused complexity for a single fixed call site. **Confirmed on review** (§7, finding
Reliability-4): weighed against this project's "personal-first" audience (`ROADMAP.md` Design
Decisions), hardcoding is the right call — the constants are named and module-level, so the rare
user who wants to tune them can still hand-edit `weather.py` directly, a sufficient release valve
without adding `apps.yaml` surface for effectively zero users.

## 4. Logging

- Each retry: `INFO` — attempt number, exception, delay before next attempt. Visible in normal
  logs without being alarming, and cheap to `grep` for if a user wants to check how often
  transient failures are actually occurring. (**Correction from rev. 1** — §7, finding
  Reliability-6: this was previously justified by analogy to `#95`'s `_warn_once()`, but that
  function implements a different pattern — dedup-by-key, WARNING-once-then-INFO-on-repeat — not
  an attempts-vs-exhaustion retry pattern. No existing precedent in this codebase directly
  matches; the levels chosen here stand on their own reasoning.)
- Final exhaustion: `WARNING` with attempt count and last exception, immediately before re-raising
  — this is what the caller's existing `WARNING` log will follow.
- 4xx / parse errors: no retry-related logging (they never enter the loop's retry branch); existing
  caller-side warning still fires as today.

## 5. Testing

`tests/test_weather.py`, new test class alongside the existing `TestFetchHistoricalWeather`.
**Deliberately does *not* copy `TestFetchOpenMeteoNetworkErrors`'s (line 583) bare-`HTTPError`
mocking pattern for the 5xx/429/404 cases** — that pattern never sets `.response`, so
`exc.response` is `None` under it. This spec's retry logic branches on `exc.response.status_code`,
so a bare `HTTPError("503")` would take the *fail-immediately* path (no response object → not
transient) instead of retrying, and a bare `HTTPError("404")` would "pass" its fail-immediately
assertion for the wrong reason — both caught independently by two of the three rev.-1 reviewers
(§7, findings SWE-1, Test-1, Test-2). Every HTTP-error mock below **must** attach a response with
an explicit plain-`int` `status_code` (not an unconfigured `MagicMock`, which raises `TypeError`
on the `status_code < 500` comparison — §7, finding Test-4):

```python
def _http_error(status: int) -> requests.HTTPError:
    err = requests.HTTPError(f"{status} error")
    err.response = MagicMock(status_code=status)
    return err
```

`time.sleep` is patched in every test so the suite doesn't actually wait. Terminal "success"
responses reuse the existing `_make_archive_response()` helper (`test_weather.py:208`), not a bare
`MagicMock()` — an unconfigured mock would fail inside `res.json()["hourly"]` with an error
unrelated to retry logic (§7, finding Test-6).

- **503 then 200:** `requests.get` side-effects `[_http_error(503), _make_archive_response(...)]`
  → returns the parsed DataFrame from the second call; `time.sleep` called once with `2`.
- **503, 503, 503, 200 (succeeds on last attempt):** 4th call succeeds → returns parsed data;
  `time.sleep` called with `2`, `4`, `8` in order.
- **503 four times (exhausts all attempts):** raises `requests.HTTPError`; `time.sleep` called
  exactly 3 times; `requests.get` called exactly 4 times (no 5th attempt).
- **429 then 200:** treated like a 5xx (new in this revision, §7 finding SWE-3) — retried and
  succeeds on the 2nd attempt.
- **404 (client error) — must use `_http_error(404)`, not a bare `HTTPError`:** raises
  immediately; `requests.get` called exactly once; `time.sleep` never called. This is the actual
  regression test for the `status_code < 500` comparison (§7, finding Test-2) — a bare-`HTTPError`
  version of this test would pass for the wrong reason and not catch a broken comparison.
- **`HTTPError` with `response=None` (no response object at all) — kept separate from the 404
  test above (§7, finding Test-5):** raises immediately, documenting that "no response" is treated
  as non-transient by design, not inferred from the 404 case.
- **`ConnectionError` then 200:** retried like a 5xx; succeeds on 2nd attempt.
- **`Timeout` then 200:** retried like a 5xx; succeeds on 2nd attempt.
- **Malformed JSON (missing `hourly` key) on the first, successful (200) response:** `KeyError`
  propagates immediately — not retried, since it's outside the loop's try block. `requests.get`
  called exactly once.
- **Existing `TestFetchHistoricalWeather` happy-path tests:** confirm they still pass unmodified
  (single `requests.get` call, no `time.sleep` needed on first-try success).

**New, `tests/test_energy_forecast.py` (§7, finding Test-3):** an integration-level check that the
safety argument in §2 ("every retryable exception this loop can re-raise is already covered by the
existing caller `except`") actually holds against the real caller, not just class-hierarchy prose —
patch `weather.fetch_historical_weather` to raise `requests.HTTPError` (post-exhaustion) from
within the `_retrain()` path and assert the existing median-imputation fallback and `WARNING` log
still fire, unchanged from today's behavior.

## 6. Known Limitations

- **Fixed backoff, not jittered.** A thundering-herd scenario (many installs retrying in lockstep
  after a shared outage) isn't a realistic concern here — Open-Meteo's Archive API is called only
  during backfill/retrain by a personal/small-community userbase, not at a scale where synchronized
  retries would matter.
- **No circuit breaker across calls.** Each `fetch_historical_weather()` invocation retries
  independently; a sustained multi-hour outage means every retrain attempt pays the full retry
  window before falling back. **Corrected worst case** (§7, findings SWE-2/Reliability-1): up to
  ~134 s (4 attempts × 30 s request timeout + 14 s of backoff sleeps), not the ~14 s originally
  stated — the difference matters specifically for a hung connection (`Timeout`/`ConnectionError`),
  not a fast-failing 5xx. Still acceptable given retrain frequency (weekly/adaptive-gated, not
  hourly) and that the fallback (median imputation) already exists and is cheap; a bounded
  ~2-minute delay before falling back doesn't justify a shorter per-attempt timeout or a circuit
  breaker for a single fixed call site.
- **`_retrain()`'s lock is held for the full retry duration.** `fetch_historical_weather()` runs
  inside `_retrain_cb`'s `self._lock` (`energy_forecast.py:994-1006`), acquired non-blocking. The
  only contenders are a manual `RELOAD_ENERGY_MODEL` event, `_rollback_model_cb`, and the
  adaptive-retrain threshold check — all already no-op silently (`DEBUG`-only log) on
  `acquire(blocking=False)` failure by existing design; the hourly sensor-update tick does not
  contend for this lock at all. Extending the lock-hold window by up to ~134 s (above) makes an
  already-existing, already-silent no-op path slightly more likely to occur during an active
  outage, but doesn't introduce a new failure mode (§7, finding Reliability-3).
- **Scoped to the Archive endpoint only.** `fetch_open_meteo()`'s own inline try/except (no retry)
  is left as-is and, unlike this fix, logs nothing at all on failure — it silently returns an empty
  DataFrame. It's called on the very next lines of the same retrain path
  (`energy_forecast.py:1705`) to stitch in the 5-day archive-lag gap, so a broader outage spanning
  both endpoints can still degrade a retrain's weather features without any visible signal beyond
  this fix's own `WARNING`, which only describes the archive-fetch half of the picture (§7, finding
  Reliability-2). Not folded into this spec's scope — a different call site, different existing
  failure shape, and folding it in would obscure this fix's narrow intent. Should get its own spec
  if it's ever independently reported as a problem.

---

## 7. Multi-Stakeholder Review — Findings and Disposition

Rev. 1 of this spec was reviewed in parallel by three independent reviewers — correctness/software
engineering, reliability/operations, and test-quality — each working from the same draft without
seeing the others' findings.

### Correctness / Software Engineering

| # | Sev | Finding | Disposition |
|---|---|---|---|
| SWE-1 | High | 5xx/404 retry tests, if built by mirroring this codebase's existing bare-`HTTPError` mock pattern (no `.response` set), would either fail to retry at all or pass the 404 case for the wrong reason | **Fixed** — §5 now requires an explicit `_http_error(status)` helper that sets `response.status_code` |
| SWE-2 | Medium | "~14s worst case" ignores the unchanged `timeout=30` per attempt on the `Timeout`/`ConnectionError` path; true worst case ≈134s | **Fixed** — §2, §3, §6 corrected |
| SWE-3 | Low | All 4xx treated as non-retryable, including 429 — a realistic transient response from a free-tier API | **Fixed** — §2 now retries 429 alongside 5xx |
| — | — | `for...else`/`last_exc` control flow, attempt/backoff counts, and retry-loop scope (excludes JSON parsing) verified correct as designed | **Confirmed, no change** |

### Reliability / Operations

| # | Sev | Finding | Disposition |
|---|---|---|---|
| Reliability-1 | Medium | Same "~14s" understatement as SWE-2, found independently | **Fixed** — see SWE-2 |
| Reliability-2 | Medium | `fetch_open_meteo()`, called immediately after on the same retrain path, fails completely silently (no log at all); a partial outage can still degrade a retrain invisibly even after this fix | **Documented** — §6 adds an explicit Known Limitations note; out of scope for this spec (different call site/failure shape) |
| Reliability-3 | Low | Retry extends `_retrain()`'s lock-hold time; existing contenders already no-op silently on non-blocking acquire failure | **Documented** — §6 adds a note; confirmed no new failure mode, just a slightly wider existing silent-no-op window |
| Reliability-4 | Low | Hardcoding retry constants (no config) — evaluated both sides | **Confirmed correct, no change** — §3 adds the reviewer's own rationale (advanced users can hand-edit the module constants) |
| Reliability-5 | Low | Retrying on `Timeout`/`ConnectionError` doesn't meaningfully risk masking a sustained outage, once the timing figure is corrected | **Confirmed, no change** |
| Reliability-6 | Low | The `#95`/`_warn_once()` citation for the INFO/WARNING logging convention doesn't actually match (`_warn_once` is a dedup pattern, not a retry pattern) | **Fixed** — §4 citation corrected |

### Test Quality

| # | Sev | Finding | Disposition |
|---|---|---|---|
| Test-1 | High | Same root cause as SWE-1: bare `HTTPError` mocks leave `.response` as `None`, breaking the 503-retry tests as literally described | **Fixed** — see SWE-1 |
| Test-2 | High | The 404 test could pass "for the wrong reason" (hits the `response is None` branch, not the `status_code < 500` branch) unless explicitly constructed otherwise | **Fixed** — §5 the 404 test now explicitly requires `_http_error(404)`; a separate `response=None` test added so each branch has its own dedicated coverage |
| Test-3 | Medium | No test verifies the final re-raised exception is actually caught by the existing caller-side `except (OSError, KeyError, ValueError)` — the whole safety argument rested on unverified class-hierarchy prose | **Fixed** — §5 adds an integration-level test in `test_energy_forecast.py` |
| Test-4 | Medium | An unconfigured `MagicMock()` as `.response` would raise `TypeError` on the `status_code < 500` comparison rather than exercising the intended branch | **Fixed** — §5 now mandates a plain `int` `status_code` |
| Test-5 | Low | No dedicated test for the `response=None` edge distinct from the (now-fixed) 404 case | **Fixed** — added alongside Test-2's fix |
| Test-6 | Low | Terminal "success" responses left unspecified — a bare `MagicMock()` would fail inside JSON parsing with an unrelated error | **Fixed** — §5 specifies reusing `_make_archive_response()` |

No further review round required: every High and Medium finding across all three reviewers is
Fixed or Documented with a concrete spec change; remaining Low items are either Fixed or explicitly
Confirmed as correct-as-designed with the reviewer's own supporting rationale folded in.
