"""One-off: rebuild energy_history.csv's SolarEdge-sourced tail after switching
energy_sensor from sensor.gplugk_z_ei to sensor.solaredge_i1_m1_ac_energy_imported.

Context (see docs/superpowers/plans/2026-09-11-solaredge-grid-meter-transition-plan.md,
Task 2.2): a full-window comparison (2026-07-16 commissioning -> 2026-09-11 gplugk
freeze) found gplugk and the SolarEdge meter agree within ~1% (export) / ~10% (import,
small base) from 2026-07-25 onward, but disagree by 400-600% in the first ~9 days
post-commissioning (documented phase-C sign-flip + lifetime-counter-reset events in
memory/project_solaredge_modbus_live_entities.md). So this does NOT touch rows before
the cutover -- only rebuilds the tail from the cutover date forward using SolarEdge,
which is also what a fresh live deploy will produce going forward once energy_sensor
is repointed.

Reads long-term hourly statistics via HA's WebSocket API (recorder/statistics_during_period)
rather than mounting the live ~1.1GB home-assistant_v2.db directly -- same underlying
`statistics` table energy_history_backfill.py reads via SQLite, just over the API since
this runs outside the AppDaemon container.

Usage: EM_HA_TOKEN=... python3 scripts/backfill_compare_solaredge.py [--apply]
Without --apply, writes to data/energy_history_solaredge_compare.csv for review only.
With --apply, merges into the live-pulled energy_history.csv (passed via --base) and
writes the merged result for upload.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import websockets

MAX_HOURLY_KWH = 50.0
CUTOVER_LOCAL = "2026-07-25 00:00:00"  # Europe/Zurich naive local, matches CSV convention
TIMEZONE = "Europe/Zurich"  # matches apps.yaml's explicit timezone override (see const.py resolve_timezone)
ENTITY_ID = "sensor.solaredge_i1_m1_ac_energy_imported"


async def fetch_statistics(entity_id: str, start_iso: str, end_iso: str) -> list[dict]:
    token = os.environ["EM_HA_TOKEN"]
    async with websockets.connect("ws://homeassistant:8123/api/websocket", max_size=50 * 1024 * 1024) as ws:
        hello = json.loads(await ws.recv())
        assert hello["type"] == "auth_required", hello
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        auth_resp = json.loads(await ws.recv())
        assert auth_resp["type"] == "auth_ok", auth_resp
        await ws.send(
            json.dumps(
                {
                    "id": 1,
                    "type": "recorder/statistics_during_period",
                    "start_time": start_iso,
                    "end_time": end_iso,
                    "statistic_ids": [entity_id],
                    "period": "hour",
                    "types": ["sum"],
                }
            )
        )
        resp = json.loads(await ws.recv())
        assert resp.get("success"), resp
        return resp["result"].get(entity_id, [])


def stats_to_hourly_kwh(rows: list[dict], tz: str) -> pd.DataFrame:
    """Mirror energy_history_backfill.py's diff/filter/timezone logic exactly."""
    df = pd.DataFrame([{"epoch": r["start"] / 1000.0, "cumsum": r["sum"]} for r in rows if r.get("sum") is not None])
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "gross_kwh"])
    df["timestamp"] = pd.to_datetime(df["epoch"], unit="s", utc=True).dt.tz_convert(tz).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    raw_diff = df["cumsum"].diff()
    df["gross_kwh"] = raw_diff.clip(lower=0)
    valid = raw_diff.notna() & (raw_diff >= 0) & (df["gross_kwh"] < MAX_HOURLY_KWH)
    return df[valid][["timestamp", "gross_kwh"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="merge into --base and write the mergeable output")
    parser.add_argument("--base", type=Path, help="existing energy_history.csv to merge into (required with --apply)")
    parser.add_argument("--out", type=Path, default=Path("data/energy_history_solaredge_compare.csv"))
    args = parser.parse_args()

    cutover_utc = (
        pd.Timestamp(CUTOVER_LOCAL).tz_localize(ZoneInfo(TIMEZONE)).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    now_utc = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"Fetching {ENTITY_ID} statistics from {cutover_utc} to {now_utc} …", file=sys.stderr)
    rows = asyncio.run(fetch_statistics(ENTITY_ID, cutover_utc, now_utc))
    print(f"  {len(rows)} raw statistic rows", file=sys.stderr)

    new_tail = stats_to_hourly_kwh(rows, TIMEZONE)
    print(f"  {len(new_tail)} clean hourly rows after diff/spike-filter", file=sys.stderr)

    if not args.apply:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        new_tail.to_csv(args.out, index=False)
        print(f"Wrote comparison-only tail to {args.out} (not merged — pass --apply --base <existing csv> to merge)")
        return

    if not args.base:
        parser.error("--apply requires --base <existing energy_history.csv path>")

    existing = pd.read_csv(args.base, parse_dates=["timestamp"])
    cutover_ts = pd.Timestamp(CUTOVER_LOCAL)
    kept = existing[existing["timestamp"] < cutover_ts]
    dropped_count = len(existing) - len(kept)
    print(
        f"Keeping {len(kept)} pre-cutover rows unchanged, dropping {dropped_count} existing "
        f"rows at/after {CUTOVER_LOCAL} in favour of the SolarEdge-sourced tail",
        file=sys.stderr,
    )

    merged = (
        pd.concat([kept, new_tail])
        .drop_duplicates(subset=["timestamp"], keep="last")  # new_tail (SolarEdge) wins post-cutover, by design
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    span = f"{merged['timestamp'].min()} .. {merged['timestamp'].max()}"
    print(f"Wrote merged CSV ({len(merged)} rows, {span}) to {args.out}")


if __name__ == "__main__":
    main()
