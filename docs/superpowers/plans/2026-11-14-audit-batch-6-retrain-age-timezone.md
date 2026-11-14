# Retrain Age Off by the UTC Offset (ROADMAP #116) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** the retrain-age gates compute the true age of the model, whatever time zone the container runs in.

**Architecture:** `model.py` stamps `last_trained` with naive `datetime.now()`, i.e. the container's system-local time. `_last_trained_local()` wrongly assumes that stamp is UTC. The fix interprets a naive stamp as system-local (`datetime.astimezone()` with no argument does exactly that) and converts it to the app's configured time zone. No change to what is stored, so no migration.

**Tech Stack:** Python 3.13, pandas, pytest (`time.tzset` to pin the system zone in tests; Linux only, which is what CI and HA run).

**Spec:** `ROADMAP.md` #116 (on `train/next`); index `docs/superpowers/plans/2026-11-14-audit-batch-index.md`.

## Global Constraints

- Base branch `train/next`; branch `fix/retrain-age-timezone`; worktree `.worktrees/retrain-age-timezone`.
- Tests only via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest` (`$PY`).
- `meta.pkl` format and `model.py` are not changed. `hours_since_trained()` in `model.py` compares naive with naive and is already correct.
- Constants unchanged: startup-retrain gap 6 h (`STARTUP_RETRAIN_MIN_GAP_HOURS`), adaptive cooldown 24 h.
- `datetime.min` ("never trained") must keep passing through unchanged.

## Review Focus

1. Container in local time (the live HA add-on: Europe/Zurich): age must be exact in winter (CET) and summer (CEST). → Task 1 tests.
2. Container in UTC with the app configured for Europe/Zurich (plain Docker installs): the stamp is shifted by +1 h / +2 h to local, as the old code intended. → Task 1 test.
3. A tz-aware `last_trained` (not produced today, possible in a future change): converted, not localized twice. → Task 1 test.
4. `datetime.min` stays `datetime.min`; `astimezone()` on it would raise `OverflowError`. → Task 1 test.
5. A stamp inside the DST fall-back hour (02:00–03:00 on the last Sunday of October) must not raise. → Task 1 test.

---

## Session setup

```bash
REPO=/home/jovyan/work/ha-energy-forecast
git -C $REPO worktree add $REPO/.worktrees/retrain-age-timezone -b fix/retrain-age-timezone train/next
cd $REPO/.worktrees/retrain-age-timezone
PY=/home/jovyan/my_envs/ha-energy-forecast/bin/python
$PY -m pytest tests/ -q          # baseline: 0 failed
```

## File Structure

- Modify `apps/energy_forecast/energy_forecast.py`: `_last_trained_local` (≈ 1149–1164). Its three callers (`_seed_adaptive_cooldown`, `_maybe_startup_retrain`, the resync at the end of `_retrain`) need no change.
- Modify `tests/test_energy_forecast.py`: the existing `_last_trained_local` tests (≈ 3590–3680; search `tz_localize("UTC").tz_convert("Europe/Zurich")`) and new ones.

---

### Task 1: Interpret the naive stamp as system-local time

**Interfaces:**
- Produces: `_last_trained_local(self) -> datetime` — unchanged signature; returns the stamp as naive wall-clock time in `self._timezone`.

- [ ] **Step 1: Write the failing tests**

Add next to the existing `_last_trained_local` tests, using the same fake app those tests use (it exposes `_ml_model.last_trained`, `_timezone` and binds `EnergyForecast._last_trained_local`):

```python
@pytest.fixture
def system_tz(monkeypatch):
    """Pin the process time zone; restores the original on teardown."""
    import time

    def _set(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield _set
    monkeypatch.undo()
    time.tzset()


class TestLastTrainedLocalSystemZone:
    """ROADMAP #116: last_trained is naive system-local time, not UTC."""

    def _fake(self, stamp, tz="Europe/Zurich"):
        ...  # build the fake exactly as the neighbouring tests do; set _timezone = tz, _ml_model.last_trained = stamp

    def test_local_container_winter_is_not_shifted(self, system_tz):
        system_tz("Europe/Zurich")
        fake = self._fake(datetime(2026, 1, 15, 12, 0))
        assert fake._last_trained_local() == datetime(2026, 1, 15, 12, 0)

    def test_local_container_summer_is_not_shifted(self, system_tz):
        system_tz("Europe/Zurich")
        fake = self._fake(datetime(2026, 7, 15, 12, 0))
        assert fake._last_trained_local() == datetime(2026, 7, 15, 12, 0)

    def test_utc_container_is_converted_to_app_zone(self, system_tz):
        system_tz("UTC")
        assert self._fake(datetime(2026, 1, 15, 12, 0))._last_trained_local() == datetime(2026, 1, 15, 13, 0)
        assert self._fake(datetime(2026, 7, 15, 12, 0))._last_trained_local() == datetime(2026, 7, 15, 14, 0)

    def test_aware_stamp_is_converted_once(self, system_tz):
        from datetime import UTC

        system_tz("Europe/Zurich")
        fake = self._fake(datetime(2026, 7, 15, 12, 0, tzinfo=UTC))
        assert fake._last_trained_local() == datetime(2026, 7, 15, 14, 0)

    def test_never_trained_passes_through(self, system_tz):
        system_tz("Europe/Zurich")
        assert self._fake(datetime.min)._last_trained_local() == datetime.min

    def test_fall_back_hour_does_not_raise(self, system_tz):
        system_tz("Europe/Zurich")
        out = self._fake(datetime(2026, 10, 25, 2, 30))._last_trained_local()
        assert out == datetime(2026, 10, 25, 2, 30)

    def test_startup_gate_sees_the_true_age(self, system_tz):
        # the live symptom: "last retrain was -0.9h ago" for a model trained 9 minutes earlier in CET
        system_tz("Europe/Zurich")
        fake = self._fake(datetime.now() - timedelta(minutes=9))
        age_h = (pd.Timestamp.now("Europe/Zurich").tz_localize(None) - fake._last_trained_local()).total_seconds() / 3600
        assert 0.1 < age_h < 0.2
```

The returned object may be a `pd.Timestamp` (it compares equal to `datetime`); do not assert on the type.

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_energy_forecast.py::TestLastTrainedLocalSystemZone -q`
Expected: the two "local container" tests fail by +1 h and +2 h; `test_startup_gate_sees_the_true_age` fails with a negative age; `test_aware_stamp_is_converted_once` fails with `TypeError: Cannot localize tz-aware Timestamp`. The UTC-container and never-trained tests already pass.

- [ ] **Step 3: Implement**

In `_last_trained_local`, replace the final `return` with: for a naive stamp, `aware = last_trained.astimezone()` (attaches the system zone); for an aware stamp, use it as is; then `return pd.Timestamp(aware).tz_convert(self._timezone).tz_localize(None)`. Keep the `datetime.min` early return.

Rewrite the docstring: `last_trained` is stamped by `datetime.now()` in `model.py`'s `train()`, which is the container's system-local time (local time in the HA add-on, UTC in a plain Docker container); this converts it to the app's configured zone so it is comparable with `pd.Timestamp.now(self._timezone).tz_localize(None)`. Mention the old behaviour in one sentence: it assumed UTC, which made the age read 1–2 h short on HA (#116).

- [ ] **Step 4: Update the existing tests**

The older tests build `last_trained` as `datetime.now(UTC).replace(tzinfo=None)` and expect a UTC→Zurich shift. They only pass when the test process itself runs in UTC. Give each of them the `system_tz` fixture and call `system_tz("UTC")` first, so they state their assumption and pass on any machine. Do not change their assertions.

- [ ] **Step 5: Run** — `$PY -m pytest tests/test_energy_forecast.py -q -k "LastTrained or StartupRetrain or AdaptiveCooldown or adaptive"` → 0 failed; then the full suite → 0 failed; `ruff check . && ruff format --check .`

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast.py
git commit -m "fix(retrain): read last_trained as system-local time, not UTC (#116)"
```

---

### Task 2: Docs and merge to the train

- [ ] **Step 1:** CHANGELOG `[Unreleased]` via `/changelog`: the startup-retrain skip gate (< 6 h) and the 24 h adaptive-retrain cooldown now use the true model age; on HA they previously ran 1 h (CET) or 2 h (CEST) late. Tag BEHAVIOUR (retrain timing).
- [ ] **Step 2:** ROADMAP #116: status → "fixed on `train/next` (date)"; Pending Summary row 116.
- [ ] **Step 3:** `/home/jovyan/work/ha-energy-forecast/MEMORY.md`: one line; also correct the note "`last_trained` is naive LOCAL" if it still describes the bug as open.
- [ ] **Step 4:** Merge and push:
  ```bash
  git -C /home/jovyan/work/ha-energy-forecast/.worktrees/train merge --no-ff fix/retrain-age-timezone
  cd /home/jovyan/work/ha-energy-forecast/.worktrees/train && $PY -m pytest tests/ -q
  git push origin train/next
  ```

**Post-deploy check (owner/PM):** after the next AppDaemon restart within 6 h of a retrain, the log line "Startup retrain skipped — last retrain was X h ago" shows a positive X equal to the real age.

**Out of scope:** storing `last_trained` as tz-aware UTC (would need a `meta.pkl` shim and touches the capture sidecars that match on the stamp's string form); retrain cadence.
