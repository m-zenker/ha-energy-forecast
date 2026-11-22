# Counter-Reset-Aware Sub-Sensor kWh — Implementation Plan (A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hours in which a cumulative kWh sensor resets keep their real energy instead of becoming 0 kWh.

**Architecture:** Replace `hourly.last().diff().clip(lower=0)` in the two sub-sensor fetchers with a shared helper that diffs the *raw* samples. A large drop that *stays* low is a counter restart (increment = new value). A drop that recovers at the next sample is a transient glitch and is ignored. Small jitter dips are clipped to 0, and increments are summed per hour. This affects every sub-sensor, not only the battery: the dishwasher, washer and dryer counters reset to 0 after each cycle (7/3/2 resets in 10 days, verified 2026-10-15), and a new cycle's first-hour energy was being lost too. The heating/DHW meters never reset. A one-off local script then repairs already-cached reset hours from HA long-term statistics (whose hourly `change` already handles resets).

**Tech Stack:** Python 3.13, pandas, pytest; HA WebSocket API (`websockets`) + pysmb for the repair script. Both run in the `ha-energy-forecast` env; `websockets` must be installed there first (Task 2 Step 3). It is in neither env today (verified 2026-10-15).

**Spec:** `docs/superpowers/plans/2026-10-15-cold-onset-forecast-fixes-index.md` (row A) + `memory/project_cold_onset_overforecast_2026_10_15.md`.

**Base branch:** `dev` → `fix/counter-reset-increments`.

## Global Constraints
- Tests via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v` (never bare python). Ruff via `/home/jovyan/my_envs/ha-energy-forecast/bin/ruff` (not on PATH).
- Behaviour for monotonic counters must equal today's within float tolerance (telescoping sum == last-minus-last; verified on a random series with gaps and empty hours).
- Known, accepted residuals: a jitter dip (10.00 → 9.99 → 10.50) over-counts by the dip size (0.01). A glitch spanning ≥ 2 consecutive low samples is classified as a restart; its recovery jump then exceeds `MAX_HOURLY_KWH` on lifetime counters and that hour is dropped (as today for spikes).
- First hourly bucket of a fetch window stays dropped (no prior reading), as today.
- `MAX_HOURLY_KWH` spike filter unchanged.
- `scripts/` is gitignored — the repair script is local-only, never committed.

## Review Focus
1. Reset exactly at an hour boundary (last sample of H-1 high, first sample of H ≈ 0) → H gets the post-reset value, H-1 unaffected.
1b. Transient glitch on a lifetime counter (1203 → 0.0 → 1207, e.g. the wallbox `total_energy`) → ignored. The old hourly `last()` hid it; diffing raw samples must not turn it into a 1207 kWh spike that the `MAX_HOURLY_KWH` filter then drops, losing an EV hour.
1c. NaN sample (`float("nan")` parses in `_fetch_history`) → dropped, energy not lost.
2. Reset to a small non-zero value (seen live: `2.012 -> 0.027`) → increment 0.027, not 0.
3. Small jitter dip (e.g. 10.00 → 9.99 → 10.50) → treated as noise, not a reset (no +9.99 kWh spike).
4. Unsorted raw samples (single-window `_fetch_history` doesn't sort) → helper sorts first.
5. Repair script run while AppDaemon writes the cache → script never touches the last 2 days, re-reads the file just before upload.

---

### Task 1: Reset-aware hourly increments helper + wire into both fetchers

**Files:**
- Modify: `apps/energy_forecast/const.py` (add `COUNTER_RESET_FRACTION`). Rationale: constants live here and are never redefined locally (project rule).
- Modify: `apps/energy_forecast/ha_data.py` (new helper above `fetch_sub_sensor_history`; replace diff blocks at ~L980–986 and ~L1046–1052). Rationale: both sub-sensor fetchers carry the reset bug; one shared helper keeps the 30-day and 2-day paths identical.
- Test: `tests/test_ha_data.py` (new class `TestHourlyCounterIncrements`, after `TestFetchSubSensorHistory`). Rationale: next to the existing fetcher tests, which must stay green unchanged (they prove monotonic equivalence).

**Interfaces:**
- Produces: `ha_data._hourly_counter_increments(raw_ha: pd.DataFrame) -> pd.DataFrame` — input `_fetch_history` frame (`timestamp` tz-aware, `value` cumulative kWh); output columns `["timestamp", "kwh"]`, hourly, tz as input, first bucket dropped.
- Produces: `const.COUNTER_RESET_FRACTION = 0.9`

- [ ] **Step 1: Write the failing tests**

```python
class TestHourlyCounterIncrements:
    """Cumulative counters that reset (SolarEdge b1 battery meters reset ~06:50/~20:55 daily)."""

    def test_monotonic_matches_last_minus_last(self):
        raw = make_ha_raw(
            ["2026-01-10T07:10:00Z", "2026-01-10T08:05:00Z", "2026-01-10T08:50:00Z", "2026-01-10T09:30:00Z"],
            [10.0, 10.4, 11.0, 11.5],
        )
        out = ha_data._hourly_counter_increments(raw).set_index("timestamp")["kwh"]
        # local 08:00 bucket is first → dropped; 09:00 = 11.0-10.0; 10:00 = 11.5-11.0
        assert list(out.round(3)) == [1.0, 0.5]

    def test_reset_mid_hour_keeps_pre_and_post_reset_energy(self):
        # October: UTC+2. Local 06:00 bucket = 04:00–05:00 UTC.
        raw = make_ha_raw(
            ["2026-10-14T03:30:00Z", "2026-10-14T04:10:00Z", "2026-10-14T04:40:00Z",
             "2026-10-14T04:50:00Z", "2026-10-14T04:55:00Z", "2026-10-14T05:20:00Z"],
            [2.0, 3.135, 3.302, 0.0, 0.045, 0.2],
        )
        out = ha_data._hourly_counter_increments(raw).set_index("timestamp")["kwh"]
        out.index = out.index.tz_localize(None)
        assert out[pd.Timestamp("2026-10-14 06:00")] == pytest.approx(1.135 + 0.167 + 0.045)
        assert out[pd.Timestamp("2026-10-14 07:00")] == pytest.approx(0.155)

    def test_reset_to_small_nonzero_value_counts_that_value(self):
        raw = make_ha_raw(
            ["2026-10-14T17:30:00Z", "2026-10-14T18:40:00Z", "2026-10-14T18:59:00Z"],
            [1.9, 2.012, 0.027],
        )
        out = ha_data._hourly_counter_increments(raw)["kwh"]
        assert out.iloc[-1] == pytest.approx(0.112 + 0.027)

    def test_reset_at_hour_boundary(self):
        raw = make_ha_raw(
            ["2026-10-14T02:30:00Z", "2026-10-14T03:10:00Z", "2026-10-14T03:59:00Z",
             "2026-10-14T04:01:00Z", "2026-10-14T04:30:00Z"],
            [2.8, 3.0, 3.3, 0.01, 0.2],
        )
        out = ha_data._hourly_counter_increments(raw).set_index("timestamp")["kwh"]
        # 04:00 bucket dropped (first); 05:00 = 0.2 + 0.3 (pre-reset, unaffected); 06:00 = 0.01 + 0.19
        assert list(out.round(3)) == [0.5, 0.2]

    def test_transient_glitch_on_lifetime_counter_ignored(self):
        raw = make_ha_raw(
            ["2026-10-14T03:30:00Z", "2026-10-14T04:05:00Z", "2026-10-14T04:10:00Z",
             "2026-10-14T04:20:00Z", "2026-10-14T04:40:00Z"],
            [1200.0, 1203.0, 0.0, 1207.0, 1208.0],
        )
        out = ha_data._hourly_counter_increments(raw)["kwh"]
        assert out.iloc[-1] == pytest.approx(8.0)  # not 1210

    def test_nan_sample_dropped_not_losing_energy(self):
        raw = make_ha_raw(
            ["2026-01-10T07:10:00Z", "2026-01-10T08:05:00Z", "2026-01-10T08:30:00Z", "2026-01-10T08:50:00Z"],
            [10.0, 10.4, float("nan"), 11.5],
        )
        out = ha_data._hourly_counter_increments(raw)["kwh"]
        assert out.iloc[-1] == pytest.approx(1.5)

    def test_appliance_cycle_counter_new_cycle_counted(self):
        # Dishwasher per-cycle counter: 0.6 at end of last cycle → 0.0 → next cycle 0.1, 0.4 within the hour
        raw = make_ha_raw(
            ["2026-10-14T03:30:00Z", "2026-10-14T04:05:00Z", "2026-10-14T04:20:00Z", "2026-10-14T04:50:00Z"],
            [0.6, 0.0, 0.1, 0.4],
        )
        out = ha_data._hourly_counter_increments(raw)["kwh"]
        assert out.iloc[-1] == pytest.approx(0.4)   # old method: 0.4 − 0.6 < 0 → 0.0

    def test_small_dip_is_jitter_not_reset(self):
        raw = make_ha_raw(
            ["2026-01-10T07:10:00Z", "2026-01-10T08:05:00Z", "2026-01-10T08:20:00Z", "2026-01-10T08:40:00Z"],
            [10.0, 10.0, 9.99, 10.5],
        )
        out = ha_data._hourly_counter_increments(raw)["kwh"]
        assert out.iloc[-1] == pytest.approx(0.51)

    def test_unsorted_input_is_sorted(self):
        raw = make_ha_raw(
            ["2026-01-10T08:50:00Z", "2026-01-10T07:10:00Z", "2026-01-10T08:05:00Z"],
            [11.0, 10.0, 10.4],
        )
        out = ha_data._hourly_counter_increments(raw)["kwh"]
        assert out.iloc[-1] == pytest.approx(1.0)

    def test_fetch_recent_sub_sensor_keeps_reset_hour(self, mock_app, tmp_path):
        raw = make_ha_raw(
            ["2026-10-14T03:30:00Z", "2026-10-14T04:40:00Z", "2026-10-14T04:50:00Z", "2026-10-14T04:55:00Z"],
            [2.0, 3.302, 0.0, 0.045],
        )
        with patch.object(ha_data, "_fetch_history", return_value=raw):
            result = ha_data.fetch_recent_sub_sensor(mock_app, "sensor.b1_export", tmp_path / "battery_discharge.csv")
        row = result[result["timestamp"] == pd.Timestamp("2026-10-14 06:00")]
        assert row.iloc[0]["kwh"] == pytest.approx(1.302 + 0.045)

    def test_fetch_sub_sensor_history_keeps_reset_hour(self, mock_app, tmp_path):
        raw = make_ha_raw(
            ["2026-10-14T03:30:00Z", "2026-10-14T04:40:00Z", "2026-10-14T04:50:00Z", "2026-10-14T04:55:00Z"],
            [2.0, 3.302, 0.0, 0.045],
        )
        with patch.object(ha_data, "_fetch_history", return_value=raw):
            result = ha_data.fetch_sub_sensor_history(mock_app, "sensor.b1_export", tmp_path / "battery_discharge.csv")
        row = result[result["timestamp"] == pd.Timestamp("2026-10-14 06:00")]
        assert row.iloc[0]["kwh"] == pytest.approx(1.302 + 0.045)
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py::TestHourlyCounterIncrements -v`
Expected: FAIL — `AttributeError: module 'energy_forecast.ha_data' has no attribute '_hourly_counter_increments'` (and the two fetch tests fail with kwh 0.0).

- [ ] **Step 3: Implement**

`const.py` (next to `MAX_HOURLY_KWH`):
```python
# A drop to below this fraction of the previous reading is a counter restart
# (SolarEdge b1 battery meters reset twice daily); smaller dips are jitter.
COUNTER_RESET_FRACTION = 0.9
```

`ha_data.py` — extend the `.const` import with `COUNTER_RESET_FRACTION`, then add above `fetch_sub_sensor_history`:
```python
def _hourly_counter_increments(raw_ha: pd.DataFrame) -> pd.DataFrame:
    """Per-hour kWh from a cumulative counter's raw samples, reset-aware.

    Diffs consecutive raw samples (not hourly last values) so a mid-hour counter
    restart keeps both the pre-reset increments and the post-reset value. A drop
    below COUNTER_RESET_FRACTION × previous reading is a restart (increment = new
    value) unless it recovers at the next sample (glitch, ignored); smaller dips are
    jitter (increment 0). For a monotonic counter the hourly sum telescopes to
    last(H) − last(H−1), i.e. unchanged behaviour.
    The first bucket has no prior reading and is dropped.
    """
    s = raw_ha.set_index("timestamp")["value"].sort_index().dropna()
    # A dip that recovers at the very next sample is a transient glitch (e.g. a lifetime
    # counter briefly reading 0 on integration restart), not a restart — drop the sample.
    prev, nxt = s.shift(1), s.shift(-1)
    glitch = (s < COUNTER_RESET_FRACTION * prev) & (nxt >= COUNTER_RESET_FRACTION * prev)
    s = s[~glitch]
    prev = s.shift(1)
    inc = s - prev
    restart = (inc < 0) & (s < COUNTER_RESET_FRACTION * prev)
    inc = inc.where(~restart, s).clip(lower=0)
    hourly = inc.resample("1h").sum().iloc[1:]
    out = hourly.reset_index()
    out.columns = ["timestamp", "kwh"]
    return out
```

Replace in **both** `fetch_sub_sensor_history` and `fetch_recent_sub_sensor`:
```python
        hourly = raw_ha.set_index("timestamp")["value"].resample("1h").last().ffill()
        diff = hourly.diff().clip(lower=0).reset_index()
        diff.columns = ["timestamp", "kwh"]
```
with:
```python
        diff = _hourly_counter_increments(raw_ha)
```
(the following `tz_localize(None)` and `MAX_HOURLY_KWH` filter lines stay).

- [ ] **Step 4: Run new + existing sub-sensor tests**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py -v`
Expected: all PASS, incl. the unchanged `TestFetchSubSensorHistory` and `TestFetchSubSensorHistoryWithProgram` (proves monotonic behaviour is identical).

- [ ] **Step 5: Full suite + ruff**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -q && /home/jovyan/my_envs/ha-energy-forecast/bin/ruff check apps tests && /home/jovyan/my_envs/ha-energy-forecast/bin/ruff format --check apps tests`
Expected: 1146 + 11 new passed, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/const.py apps/energy_forecast/ha_data.py tests/test_ha_data.py
git commit -m "fix: reset-aware hourly kWh for cumulative sub/correction sensors"
```

---

### Task 2: Changelog + deploy, then repair cached reset hours (local script, not committed)

**Files:**
- Modify: `CHANGELOG.md` (via `@changelog-writer`)
- Create (gitignored, local only): `scripts/repair_counter_reset_cache.py`. Rationale: cached rows older than HA's recorder retention can't be re-derived by the app; long-term statistics handle resets correctly.

**Interfaces:**
- Consumes: Task 1 deployed live (otherwise the next 30-day retrain fetch re-zeroes the recent window).
- Consumes: `scripts/pull_ha_data.py` → `_smb_connect()`, `_smb_read(conn, path)`, `SMB_SHARE`, `FORECAST_REMOTE`.

- [ ] **Step 1: Changelog entry** — dispatch `@changelog-writer`: "Fixed: cumulative sub-sensor counter resets zeroed the whole reset hour — battery charge/discharge (SolarEdge b1 meters reset twice daily, so actuals read ~0 kWh at 06:00/20:00) and per-cycle appliance counters (first-hour energy of a new cycle was lost). Transient counter glitches are ignored. Sub-sensor history re-derived from HA (last ~10–30 days) will differ slightly from older cached rows." Commit `docs: changelog — counter-reset fix`.

- [ ] **Step 2: Finalize via deploy-agent Workflow A** (merge → dev no-ff, next `v0.12.0-alpha-N`, deploy, push, cleanup). Verify after the next hourly run: `data/` re-pull shows `battery_discharge.csv` 06:00/20:00 rows of the last 2 days > 0.

- [ ] **Step 3: Install `websockets` into the project env, then write the repair script**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pip install websockets` (local env only; not a project dependency).

```python
#!/usr/bin/env python3
"""One-off: overwrite cached battery-counter hours with HA long-term statistics `change` (reset-aware).

Usage: EM_HA_TOKEN=… EM_SMB_PASSWORD=… /home/jovyan/my_envs/ha-energy-forecast/bin/python \
         scripts/repair_counter_reset_cache.py [--apply]
Without --apply: dry run (prints the rows it would change). Never touches the last 2 days (app-owned).
Only raises cached values (repairs zeroed/partial reset hours); never lowers them.
"""
from __future__ import annotations

import asyncio, io, json, os, pathlib, sys
from datetime import datetime

import pandas as pd
import websockets

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import pull_ha_data as p  # noqa: E402

TZ = "Europe/Zurich"
FILES = {
    "battery_discharge.csv": "sensor.solaredge_i1_b1_energy_export",
    "battery_charge.csv": "sensor.solaredge_i1_b1_energy_import",
}
START = "2026-07-16T00:00:00Z"
BACKUP = pathlib.Path(__file__).parent.parent / "temp" / f"cache_backup_{datetime.now():%Y-%m-%dT%H%M%S}"


async def fetch_stats(ids: list[str]) -> dict:
    async with websockets.connect("ws://homeassistant:8123/api/websocket", max_size=2**25) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "auth", "access_token": os.environ["EM_HA_TOKEN"]}))
        assert json.loads(await ws.recv())["type"] == "auth_ok"
        await ws.send(json.dumps({"id": 1, "type": "recorder/statistics_during_period", "start_time": START,
                                  "statistic_ids": ids, "period": "hour", "types": ["change"]}))
        resp = json.loads(await ws.recv())
        if not resp.get("success"):
            raise SystemExit(f"statistics request failed: {resp.get('error')}")
        return resp["result"]


def to_series(rows: list[dict]) -> pd.Series:
    start = pd.Series([r["start"] for r in rows])
    ts = pd.to_datetime(start, unit="ms", utc=True) if start.dtype.kind in "if" else pd.to_datetime(start, utc=True)
    return pd.Series([r["change"] for r in rows], index=ts.dt.tz_convert(TZ).dt.tz_localize(None).values)


def main() -> None:
    apply = "--apply" in sys.argv
    if apply and datetime.now().minute < 5:
        raise SystemExit("refusing to write in the first 5 min of the hour (app rewrites caches at :00) — retry")
    stats = asyncio.run(fetch_stats(list(FILES.values())))
    conn = p._smb_connect()
    cutoff = pd.Timestamp.now(tz=TZ).tz_localize(None).floor("1h") - pd.Timedelta(days=2)
    for fname, eid in FILES.items():
        if eid not in stats or not stats[eid]:
            print(f"{fname}: no long-term statistics for {eid} (state_class total/total_increasing?) — skipped")
            continue
        remote = f"{p.FORECAST_REMOTE}/{fname}"
        raw = p._smb_read(conn, remote)
        df = pd.read_csv(io.BytesIO(raw))
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        st = to_series(stats[eid])
        if df["timestamp"].duplicated().any():
            print(f"  ABORT {fname}: duplicate timestamps (DST fall-back?) — handle before repairing"); continue
        cached = df.set_index("timestamp")["kwh"]
        common = cached.index.intersection(st.index)
        common = common[common < cutoff]
        delta = (st[common] - cached[common]).abs()
        clean = delta[cached[common] > 0.02]  # hours not zeroed by a reset
        print(f"{fname}: {len(common)} overlapping h, median |Δ| on clean hours {clean.median():.4f}, "
              f"{(delta > 0.05).sum()} hours differ > 0.05 kWh (Σ {(st[common] - cached[common]).sum():+.2f} kWh)")
        if len(clean) < 48 or not clean.median() <= 0.01:  # also catches NaN median
            print(f"  ABORT: statistics not aligned with cache on clean hours (n={len(clean)}, "
                  f"median |Δ|={clean.median()}) — check hour labelling / 5-min compilation noise")
            continue
        fix = common[(st[common] - cached[common]) > 0.05]   # raise only: zeroed/partial reset hours
        print(f"  {len(fix)} rows to raise; by local hour: {pd.Series(fix.hour).value_counts().sort_index().to_dict()}")
        print(pd.DataFrame({"cached": cached[fix], "stats": st[fix]}).head(20).round(3).to_string())
        if apply:
            BACKUP.mkdir(parents=True, exist_ok=True)
            (BACKUP / fname).write_bytes(raw)
            df = df.set_index("timestamp")
            df.loc[fix, "kwh"] = st[fix]
            buf = io.BytesIO(df.reset_index().to_csv(index=False).encode())
            if p._smb_read(conn, remote) != raw:
                print("  ABORT: remote changed during run — rerun"); continue
            conn.storeFile(p.SMB_SHARE, remote, buf)
            print(f"  wrote {len(fix)} rows (backup in {BACKUP})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Dry run.** Run without `--apply`. Expected: median |Δ| on clean hours ≤ 0.01, and rows-to-raise concentrated at local hours 06/07 and 20/21 (discharge) and 20/21 (charge) since 2026-07-16. Note the first affected date. If ABORT on alignment, inspect a few hours by hand: HA compiles statistics from 5-min samples, so a slightly looser threshold (≤ 0.02) is acceptable if residuals are symmetric. Otherwise stop and report.

Note: after Task 1 is live, every retrain's 30-day fetch re-derives the window that HA still holds (recorder default ≈ 10 days) with the new helper, and HA wins the merge. The repair therefore matters for rows older than recorder retention; repaired rows inside it are overwritten by helper values, which are within a few Wh.

- [ ] **Step 5: Apply + verify** — run with `--apply`, then `pull_ha_data.py` and re-run the 06h/20h check from the investigation (hourly mean of `battery_discharge` by hour since 10-08 should no longer show 0.00 at 06 and 20). Force a retrain (restart AppDaemon or wait for the nightly) so training sees repaired rows.

- [ ] **Step 6: Memory** — update `memory/project_cold_onset_overforecast_2026_10_15.md` (A deployed + repaired, alpha tag) and MEMORY.md line.
