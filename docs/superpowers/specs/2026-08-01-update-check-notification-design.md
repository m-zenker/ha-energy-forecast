# Design: Daily Update-Check + Notification (Roadmap #91)

**Date:** 2026-08-01
**Status:** Approved, ready for implementation planning
**Roadmap item:** `#91 — Daily Update-Check + Notification` (`ROADMAP.md` on `dev`, added 2026-07-18 in `5769e85`)

## Context

Other users of this project are expected to stay on `main`, which only receives
stable releases merged from `dev` after a local test period on the maintainer's
own instance. Those users have no reason to watch the repo, so there's no
built-in signal telling them a new stable release exists. `main` and `dev`
share a public GitHub mirror at `github.com/m-zenker/ha-energy-forecast`
(confirmed live: real Discussions threads, stars, issue links), so GitHub's
Releases API is a legitimate, publicly reachable source of truth — unlike the
project's actual git remote, which is a local-only Forgejo instance.

This feature adds a quiet, reliable "you're behind" signal for main-track
users: once a day, check GitHub's latest release against the running
`__version__`, and notify in Home Assistant if the user is behind.

## Goals

- Main-track users are notified, once per new release, when they're behind.
- Zero new required configuration.
- The maintainer's own `dev`/alpha/beta instance never triggers this — it is
  always ahead of `main` and checking would be noise, not signal.
- No new dependency (`requests` is already used by `weather.py`).

## Non-goals

- No manual/on-demand trigger (service call) — daily schedule only.
- No opt-out config key — always-on for any non-prerelease install.
- No automatic dismissal of a stale notification once the user updates — HA
  persistent_notifications are user-dismissed by design; this feature doesn't
  change that convention.
- No mobile push (`notify.*` service) — HA's built-in `persistent_notification`
  panel is the only delivery channel.

## Architecture

Two pieces, split by testability, matching the existing convention where
`weather.py` / `physics.py` / `clustering.py` hold pure logic that
`energy_forecast.py` (the AppDaemon entry point) orchestrates.

### `apps/energy_forecast/update_check.py` (new module — pure logic)

```python
class UpdateInfo(NamedTuple):
    tag: str    # e.g. "v0.12.0", as returned by GitHub ("v" prefix included)
    url: str    # release page html_url, for the notification message

def check_for_update(
    current_version: str,
    repo: str = "m-zenker/ha-energy-forecast",
    timeout: int = 10,
) -> UpdateInfo | None: ...
```

- Calls `GET https://api.github.com/repos/{repo}/releases/latest`
  (unauthenticated; 1 req/day/install is negligible against the 60/hr
  unauthenticated rate limit). This endpoint already excludes prereleases and
  drafts by GitHub's own semantics, so the response is always a stable tag.
- Parses `tag_name` and `html_url` from the JSON body.
- Version comparison is genuine semver ordering, not string inequality: a
  private `_parse_version(s: str) -> tuple[int, int, int]` strips an optional
  leading `v` (repo tags are `vX.Y.Z`, e.g. `v0.11.10`; local `__version__` in
  `apps/energy_forecast/__init__.py` has no `v` prefix, e.g. `"0.11.10"`),
  splits on `.`, and takes the first three numeric components. Returns
  `UpdateInfo` only if `remote > local`; returns `None` if remote `<=` local.
- **All failure modes return `None`, never raise**: network errors, timeouts,
  non-200 responses, malformed JSON, and unparsable version strings are all
  caught (`requests.RequestException, KeyError, ValueError` — the same catch
  shape `weather.py` already uses) and logged via `_LOGGER.warning` before
  returning `None`.

### `apps/energy_forecast/energy_forecast.py` (orchestration only)

In `initialize()`:

```python
if "-alpha" not in __version__ and "-beta" not in __version__:
    self.run_daily(self._check_for_updates_cb, time(3, 0, 0))
```

The guard is checked once, before scheduling — a prerelease-tagged instance
(the maintainer's own `dev` deployment) never even registers the daily job,
so it costs nothing beyond the one string check at startup.

`_check_for_updates_cb(self, kwargs)`:

1. `info = check_for_update(__version__)` — if `None`, return immediately
   (nothing to do; any failure was already logged inside the module).
2. Load dedup state from a new file, `self._cache_path.parent /
   "update_check.json"` — deliberately separate from `pred_history.json`,
   which is scoped to forecast-accuracy tracking, not update metadata.
   Missing file or `JSONDecodeError` is treated as `{}` (never notified),
   mirroring the existing defensive-load pattern used for
   `_load_pred_history`.
3. If `state.get("last_notified_tag") == info.tag`: skip — already notified
   for this exact release.
4. Otherwise, fire the notification and persist the new dedup state:

```python
self.call_service(
    "persistent_notification/create",
    title=f"HA Energy Forecast {info.tag} available",
    message=f"You're on {__version__}. See release notes: {info.url}",
    notification_id="hef_update_available",
)
```

   The fixed `notification_id` means a later release replaces the existing
   notification instead of stacking a second one. State is then written as
   `{"last_notified_tag": info.tag}`.

## Data flow summary

```
run_daily @ 03:00 (skipped entirely if local version is -alpha/-beta)
  -> update_check.check_for_update(__version__)
       -> GET github.com/.../releases/latest
       -> parse + semver-compare
       -> UpdateInfo | None
  -> if UpdateInfo and not already notified for this tag:
       -> persistent_notification/create (fixed notification_id)
       -> write update_check.json {"last_notified_tag": tag}
```

## Error handling

| Failure | Behavior |
|---|---|
| Network error / timeout / non-200 | Caught in `check_for_update()`, logged as warning, returns `None`. Retried automatically at the next `run_daily` firing — nothing is persisted on failure. |
| Malformed JSON / missing `tag_name` / unparsable version string | Same as above — caught, warned, `None`. |
| `update_check.json` missing or corrupt | Treated as `{}` (never notified before); does not block notifying. |
| `persistent_notification/create` call itself fails | Not specially wrapped — matches how other `run_daily`/`run_hourly` callbacks in this codebase behave; AppDaemon catches and logs exceptions from scheduled callbacks. |

## Testing

**`update_check.py`** — pure unit tests, no AppDaemon stub required, mocking
`requests.get` the same way `weather.py`'s existing tests do:
- remote strictly newer → returns `UpdateInfo` with correct `tag`/`url`
- remote equal or older → `None`
- malformed `tag_name` → `None`, warning logged
- network error / timeout → `None`, warning logged

**`energy_forecast.py` orchestration** — existing `hassapi` stub / MagicMock
app fixture pattern:
- first detection of a new tag → notification fired with correct
  `notification_id` and message content; dedup file written with that tag
- repeat check against the same tag → notification not re-fired
- corrupt/missing dedup file → treated as empty, notification still fires
- local `__version__` contains `-alpha`/`-beta` → `run_daily` is never
  registered in `initialize()`

## Key decisions (from interview)

- **Delivery channel**: `persistent_notification` only — no `notify.*`
  mobile push, no config key for one.
- **Prerelease guard**: hardcoded on `-alpha`/`-beta` substring in
  `__version__`, not configurable.
- **Version comparison**: real semver ordering (`remote > local`), not string
  inequality — avoids false "update available" signals from a rolled-back or
  re-tagged release.
- **Check timing**: fixed off-peak time (03:00), no jitter, no config key —
  matches the existing fixed-time pattern of `_update_cb`'s
  `run_hourly(time(0, 1, 0))`.
- **Dedup storage**: new `update_check.json`, not a key inside
  `pred_history.json` — keeps concerns separate.
- **Notification content**: links to the GitHub release page (`html_url`,
  already in the API response — no second request).
- **Stale notification cleanup**: none — left for the user to dismiss
  manually, consistent with normal HA `persistent_notification` UX.
- **Manual trigger**: none — daily schedule only, to keep scope inside the
  roadmap's 1–1.5h estimate.
- **Opt-out**: none — always-on for any non-prerelease install; the cost (one
  daily API call, one dismissible notification) doesn't justify a config key.
