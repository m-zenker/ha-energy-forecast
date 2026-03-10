"""
energy_history_backfill.py — One-off AppDaemon app to backfill energy_history.csv
from Home Assistant long-term statistics via direct SQLite access.

HOW TO USE
----------
1. Copy this file into your AppDaemon apps folder alongside energy_forecast.py.

2. Add the following block to apps.yaml (no token needed):

       energy_history_backfill:
         module: energy_forecast.energy_history_backfill
         class: EnergyHistoryBackfill
         energy_sensor: sensor.gplugk_z_ei
         ha_db_path: /config/home-assistant_v2.db   # default for HAOS

3. AppDaemon will start the app automatically. Watch the AppDaemon log for
   the final summary line.

4. Once you see "Backfill complete — remove this app from apps.yaml", do exactly
   that and delete this file.

DETAILS
-------
- Reads the HA `statistics` table directly via sqlite3 — no REST API, no token.
  The `statistics` table is never purged by the recorder.
- Joins `statistics_meta` to filter by statistic_id (= entity_id).
- The `sum` column is the cumulative kWh total; we diff() to get per-hour kWh.
- Negative diffs (meter reset) are clipped to 0.
- Hours > MAX_HOURLY_KWH (50 kWh) are dropped as spikes/resets.
- Merges into the existing CSV — CSV wins on timestamp conflicts.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import hassapi as hass

from .const import CACHE_PATH

# ── Tunables ──────────────────────────────────────────────────────────────────
LOOKBACK_YEARS  = 1
MAX_HOURLY_KWH  = 50.0
DEFAULT_DB_PATH = "/config/home-assistant_v2.db"


class EnergyHistoryBackfill(hass.Hass):
    """Runs once on startup, backfills the CSV from the HA SQLite DB."""

    def initialize(self) -> None:
        self.log("=" * 60)
        self.log("EnergyHistoryBackfill starting…")
        self.run_in(self._run, 5)

    def _run(self, kwargs: dict) -> None:
        try:
            self._backfill()
        except Exception as exc:  # noqa: BLE001
            self.log(f"Backfill FAILED: {exc}", level="ERROR")
            import traceback
            self.log(traceback.format_exc(), level="ERROR")

    # ── Main logic ────────────────────────────────────────────────────────────

    def _backfill(self) -> None:
        import sqlite3
        import pandas as pd

        entity_id = self.args["energy_sensor"]
        db_path   = self.args.get("ha_db_path", DEFAULT_DB_PATH)

        if not Path(db_path).exists():
            raise FileNotFoundError(
                f"HA database not found at '{db_path}'. "
                "Check ha_db_path in apps.yaml. "
                "Common paths: /config/home-assistant_v2.db (HAOS addon), "
                "/homeassistant/home-assistant_v2.db (some setups)."
            )

        self.log(f"Reading statistics for {entity_id} from {db_path} …")

        cutoff_ts = (
            datetime.now(tz=timezone.utc) - timedelta(days=365 * LOOKBACK_YEARS)
        ).timestamp()

        # ── 1. Query the statistics table ─────────────────────────────────────
        # HA stores timestamps as Unix epoch in `start_ts` (since ~2022.10).
        # Older installs used a `start` datetime column — we handle both.
        con = sqlite3.connect(db_path)
        try:
            cols = {
                row[1]
                for row in con.execute("PRAGMA table_info(statistics)")
            }

            if "start_ts" in cols:
                ts_expr    = "s.start_ts"
                where_col  = "s.start_ts"
                cutoff_val: float | str = cutoff_ts
            else:
                cutoff_val = datetime.utcfromtimestamp(cutoff_ts).strftime("%Y-%m-%d %H:%M:%S")
                ts_expr    = "strftime('%s', s.start)"
                where_col  = "s.start"

            # where_col is a hardcoded column reference, not user input.
            # cutoff_val is passed as a bound parameter to prevent injection.
            query = f"""
                SELECT
                    {ts_expr}  AS epoch,
                    s.sum      AS cumsum
                FROM statistics s
                JOIN statistics_meta sm ON s.metadata_id = sm.id
                WHERE sm.statistic_id = ?
                  AND {where_col} >= ?
                ORDER BY epoch
            """
            rows = con.execute(query, (entity_id, cutoff_val)).fetchall()
        finally:
            con.close()

        if not rows:
            raise ValueError(
                f"No statistics rows found for '{entity_id}' in the last "
                f"{LOOKBACK_YEARS} year(s). "
                "Check that the entity has state_class: total_increasing and "
                "has been tracked by the recorder."
            )

        self.log(f"Retrieved {len(rows)} raw statistic rows from DB.")

        # ── 2. Convert to hourly kWh ──────────────────────────────────────────
        df = pd.DataFrame(rows, columns=["epoch", "cumsum"])
        df["epoch"]  = pd.to_numeric(df["epoch"], errors="coerce")
        df["cumsum"] = pd.to_numeric(df["cumsum"], errors="coerce")
        df = df.dropna()

        # Convert Unix epoch → naive Europe/Zurich timestamp
        df["timestamp"] = (
            pd.to_datetime(df["epoch"], unit="s", utc=True)
            .dt.tz_convert("Europe/Zurich")
            .dt.tz_localize(None)
        )

        df = df.sort_values("timestamp").reset_index(drop=True)

        # Diff cumulative sum → per-hour kWh; clip negatives (meter resets)
        df["gross_kwh"] = df["cumsum"].diff().clip(lower=0)
        df = df.dropna(subset=["gross_kwh"])
        df = df[(df["gross_kwh"] > 0) & (df["gross_kwh"] < MAX_HOURLY_KWH)]

        df_new = df[["timestamp", "gross_kwh"]].reset_index(drop=True)
        self.log(f"After diff & filtering: {len(df_new)} clean hourly rows.")

        # ── 3. Load existing CSV ──────────────────────────────────────────────
        df_cache = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        if CACHE_PATH.exists():
            try:
                df_cache = pd.read_csv(CACHE_PATH)
                ts = pd.to_datetime(df_cache["timestamp"])
                if ts.dt.tz is not None:
                    ts = ts.dt.tz_convert("Europe/Zurich").dt.tz_localize(None)
                df_cache["timestamp"] = ts
                self.log(f"Loaded {len(df_cache)} existing rows from {CACHE_PATH.name}.")
            except (OSError, pd.errors.ParserError) as exc:
                self.log(
                    f"Could not read existing CSV ({exc}) — will create fresh.",
                    level="WARNING",
                )

        # ── 4. Merge — CSV wins on conflicts ──────────────────────────────────
        combined = (
            pd.concat([df_new, df_cache])          # cache last → keep="last" favours it
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
            .dropna(subset=["timestamp", "gross_kwh"])
            .reset_index(drop=True)
        )

        added      = len(combined) - len(df_cache)
        date_range = (
            f"{combined['timestamp'].min().date()} → "
            f"{combined['timestamp'].max().date()}"
        )

        # ── 5. Save ───────────────────────────────────────────────────────────
        combined.to_csv(CACHE_PATH, index=False)

        self.log(
            f"Saved {len(combined)} rows to {CACHE_PATH.name} "
            f"({added:+d} rows added). Range: {date_range}."
        )
        self.log("=" * 60)
        self.log(
            "Backfill complete — remove 'energy_history_backfill' from "
            "apps.yaml and delete energy_history_backfill.py."
        )
        self.log("=" * 60)
        