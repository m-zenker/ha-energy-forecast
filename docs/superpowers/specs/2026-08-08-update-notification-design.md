# Daily Update-Check + Notification — Design Spec

**Date:** 2026-08-08
**Status:** Approved — ready for implementation planning
**Branch base:** `dev`
**Roadmap item:** #91

## 1. Problem & Motivation

Users on `main` only receive stable releases (merged from `dev` after a local test period), so
they have no signal that a new release exists unless they actively watch the repo. The project
already maintains a public GitHub mirror (`github.com/m-zenker/ha-energy-forecast`) for community
engagement (Discussions, issues), and that mirror's Releases API is live and populated — confirmed
by fetching `https://api.github.com/repos/m-zenker/ha-energy-forecast/releases/latest`, which
currently returns `v0.11.10`, matching the released `main` version. This gives a reliable,
zero-maintenance "you're behind" signal without adding any new external service.

## 2. Config

New optional `apps.yaml` key:

- `update_check_enabled` (bool, default `true`) — when `false`, the daily job is never scheduled
  (not scheduled-then-skipped): zero outbound network calls, zero daily wakeups. Exists so
  privacy-conscious or fully offline instances can opt out without editing code.

New constant in `const.py`, next to `CACHE_PATH`:

```python
GITHUB_RELEASES_URL = "https://api.github.com/repos/m-zenker/ha-energy-forecast/releases/latest"
```

## 3. Check Logic

New method `_check_for_update_cb(self, kwargs)` on `EnergyForecastApp`
(`apps/energy_forecast/energy_forecast.py`), scheduled from `initialize()`:

```python
if self._update_check_enabled:
    self.run_daily(self._check_for_update_cb, time(9, 0, 0))
```

(`09:00` local, arbitrary but off the top of the hour to avoid clustering with `run_hourly`'s
`00:01:00` update tick.)

Behavior:

1. **Dev-track guard:** if `__version__` contains `-alpha` or `-beta`, log at `DEBUG` and return
   immediately. The maintainer's own dev/`dev`-branch system always runs ahead of `main` and must
   never nag itself; this check is purely for `main`-track users.
2. **Fetch:** `requests.get(GITHUB_RELEASES_URL, timeout=10)`, lazily imported inside the method —
   matches the existing pattern in `weather.py` (`import requests` inside the function body, no
   module-level dependency).
3. **Parse & compare:** read `tag_name` from the JSON body, strip a leading `v` if present.
   Compare to `__version__` with plain string inequality — `main`-track versions only ever advance
   forward via releases, so any mismatch means "a newer release exists." No semver library is
   needed for this one-directional comparison. **If the remote tag equals `__version__`, the
   instance is already up to date: return immediately — no dedup check, no state read/write.**
4. **Dedup (only reached when step 3 found a newer remote tag):** load
   `{"last_notified_tag": ...}` from the state file (§4). If the remote tag equals
   `last_notified_tag`, skip — already notified about this version. Otherwise, proceed to notify.
5. **Notify:** on a new, not-yet-notified tag, call:
   ```python
   self.call_service(
       "persistent_notification/create",
       title=f"HA Energy Forecast {tag} available",
       message=f"A new version ({tag}) is available. You are running {__version__}.",
       notification_id="hef_update_available",
   )
   ```
   The fixed `notification_id` means a later check replaces the existing notification instead of
   stacking duplicates.
6. **Persist:** on successful notify, write the new tag as `last_notified_tag` to the state file.

## 4. State / Dedup File

`self._cache_path.parent / "update_check_state.json"`, holding a single key:
`{"last_notified_tag": "v0.12.0"}`.

Load/save follow the existing `_load_pred_history` / `_save_pred_history` pattern
(energy_forecast.py:2928 area): wrapped in `try/except`, malformed or missing state is logged at
`WARNING` and treated as "no prior notification" (empty state) rather than raising. A save failure
is logged at `WARNING` and does not crash the callback — worst case, the same version is
re-notified on the next run, which is harmless (same fixed `notification_id`).

## 5. Error Handling

Any failure in steps 2–3 (`requests.RequestException`, non-200 status via `raise_for_status()`,
`json.JSONDecodeError`, missing `tag_name` key) is caught in a single `try/except`, logged at
`WARNING`, and the callback returns without notifying. The next day's scheduled run retries
automatically — no backoff or retry-within-day logic needed given the low frequency and low
stakes.

The callback never raises out of AppDaemon's scheduler.

## 6. Testing

`tests/test_energy_forecast.py`, new test class for `_check_for_update_cb`:

- Local version contains `-alpha`/`-beta` → no HTTP call made, no notification.
- `update_check_enabled: false` → `run_daily` never called during `initialize()`.
- Remote tag differs from local version and from `last_notified_tag` → notification fires with
  expected `title`/`message`/`notification_id`; state file updated.
- Remote tag equals `last_notified_tag` (already notified) → no duplicate notification.
- Remote tag changes again after a prior notification (new release since last notify) →
  notifies again, state file updated to the newer tag.
- Remote tag equals local `__version__` (already up to date) → no notification, state unchanged.
- `requests.get` raises `RequestException` → warning logged, no notification, no exception
  propagates.
- Response JSON missing `tag_name` → warning logged, no notification, no exception propagates.
- Corrupt/missing state file on load → treated as no prior notification, no exception.
- State file write failure (e.g. read-only path in test) → warning logged, callback still
  completes without raising.

## 7. Known Limitations

- **No mobile push.** Only `persistent_notification` (HA UI panel) is implemented. A user who
  doesn't check the HA UI regularly won't see it promptly. Deferred — can be added later as an
  optional `notify_service` config key if requested.
- **Simple string-inequality version comparison.** Correct for this project's actual release
  pattern (linear forward progression on `main`), but would misbehave if `main` ever needed a
  point release out of order or a tag were manually rolled back. Not a realistic scenario given
  the existing release workflow (`/release` skill, Forgejo-tag-then-mirror), so a full semver
  comparator is not justified.
- **GitHub mirror freshness is an external dependency.** The check assumes the GitHub mirror stays
  in sync with Forgejo releases; if that sync process ever lapses, the notification would fire
  late or not at all. Outside this feature's scope to guard against.
