"""COP-scale analysis: which COP source matches the heat pump's own heat/electricity counters?

Background (memory ``project_ua_eff_null_cause_and_path``): the physics UA_eff calibrator uses
the config COP formula (min(0.45 * Carnot, a + b*T)), which gives ~3.1 at 8 C, while the live
Kermi ``cop_heating_live`` sensor reads ~5.  UA scales with COP, so the choice matters for #105.

The Kermi controller also exposes cumulative counters for delivered heat (heating + DHW,
``lifetime_heat``) and for electricity split into heating and DHW.  Per local day we fit

    heat = COP_h * el_heating + COP_d * el_dhw          (no intercept)

which gives an energy-weighted heating COP independent of both the formula and the live sensor,
then compare it per outdoor-temperature bin against the formula and the el-weighted live sensor.
Finally UA is fitted three ways (counter heat, formula COP, sensor COP) on daily degree-hours.

Caveat: the Kermi heat counter is estimated by the controller, not a calibrated heat meter.

Read-only: one HA WebSocket request (recorder/statistics_during_period).  Token from
``EM_HA_TOKEN`` in the environment, never from the command line.

Usage:
    python scripts/analyze_cop_scale.py --start 2026-03-01 [--end 2026-11-08] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HA_WS = "ws://homeassistant:8123/api/websocket"
TZ = "Europe/Zurich"

HEAT = "sensor.em_kermi_bridge_kermi_lifetime_heat"
EL_TOTAL = "sensor.em_kermi_bridge_kermi_lifetime_electricity"
EL_HEAT = "sensor.em_kermi_bridge_kermi_electricity_heating"
EL_DHW = "sensor.em_kermi_bridge_kermi_electricity_dhw"
COP_LIVE = "sensor.em_kermi_bridge_kermi_cop_heating_live"
# Live hef uses the Kermi outdoor average, but it has no state_class, so no long-term statistics.
# First candidate with statistics wins.
T_OUT_CANDIDATES = ["sensor.em_kermi_bridge_kermi_outside_temperature_avg", "sensor.outdoor_netatmo_temperature"]
PV = "sensor.pv_energy_2"

COUNTERS = {"heat": HEAT, "el_total": EL_TOTAL, "el_heat": EL_HEAT, "el_dhw": EL_DHW, "pv": PV}
MAX_HOURLY_KWH = {"heat": 40.0, "el_total": 12.0, "el_heat": 12.0, "el_dhw": 12.0, "pv": 20.0}

TEMP_BINS = [-30, 0, 4, 8, 12, 16, 40]
MIN_EL_HEAT_DAY_KWH = 2.0
MIN_HOURS = 22


# ── data access ──────────────────────────────────────────────────────────────


async def fetch_statistics(ids: list[str], start: str, end: str, token: str) -> dict[str, list[dict]]:
    import websockets  # dev-only dependency, imported lazily so the pure functions stay testable

    async with websockets.connect(HA_WS, max_size=None) as ws:
        json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        auth = json.loads(await ws.recv())
        if auth.get("type") != "auth_ok":
            raise RuntimeError(f"HA auth failed: {auth.get('type')}")
        await ws.send(
            json.dumps(
                {
                    "id": 1,
                    "type": "recorder/statistics_during_period",
                    "start_time": start,
                    "end_time": end,
                    "statistic_ids": ids,
                    "period": "hour",
                    "types": ["sum", "mean"],
                }
            )
        )
        resp = json.loads(await ws.recv())
        if not resp.get("success"):
            raise RuntimeError(f"statistics request failed: {resp.get('error')}")
        return resp["result"]


def _local_index(rows: list[dict]) -> pd.DatetimeIndex:
    epoch = [r["start"] / 1000.0 for r in rows]
    return pd.to_datetime(epoch, unit="s", utc=True).tz_convert(TZ).tz_localize(None)


def rows_to_hourly_delta(rows: list[dict], max_hourly: float) -> pd.Series:
    """Hourly increments of a total_increasing statistic.

    The increment for hour h is sum[h] - sum[h-1].  Negative steps (counter reset, which HA's
    ``sum`` already compensates, or a glitch) and implausibly large steps are dropped (NaN),
    so a day containing one is excluded by the completeness filter rather than silently skewed.
    """
    rows = [r for r in rows if r.get("sum") is not None]
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    s = pd.Series([r["sum"] for r in rows], index=_local_index(rows)).sort_index()
    s = s[~s.index.duplicated(keep="last")]  # DST fall-back produces two identical local hours
    d = s.diff()
    return d.where((d >= 0) & (d <= max_hourly))


def rows_to_hourly_mean(rows: list[dict]) -> pd.Series:
    rows = [r for r in rows if r.get("mean") is not None]
    if not rows:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    s = pd.Series([r["mean"] for r in rows], index=_local_index(rows)).sort_index()
    return s.groupby(level=0).mean()


# ── analysis ─────────────────────────────────────────────────────────────────


def daily_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate an hourly frame (counter deltas + means) to complete local days.

    ``cop_live_w`` is the el_heat-weighted mean of the live COP, i.e. the COP that would turn the
    day's heating electricity into the day's heat if the sensor were right.
    """
    h = hourly.copy()
    h["cop_live"] = pd.to_numeric(h["cop_live"], errors="coerce")
    h["cop_x_el"] = h["cop_live"] * h["el_heat"]
    h["el_heat_with_cop"] = h["el_heat"].where(h["cop_live"].notna())
    g = h.groupby(h.index.normalize())
    d = pd.DataFrame(
        {
            "heat": g["heat"].sum(min_count=MIN_HOURS),
            "el_total": g["el_total"].sum(min_count=MIN_HOURS),
            "el_heat": g["el_heat"].sum(min_count=MIN_HOURS),
            "el_dhw": g["el_dhw"].sum(min_count=MIN_HOURS),
            "pv": g["pv"].sum(min_count=MIN_HOURS) if "pv" in h else np.nan,
            "t_out": g["t_out"].mean(),
            "t_in": g["t_in"].mean() if "t_in" in h else np.nan,
            "cop_live_w": g["cop_x_el"].sum() / g["el_heat_with_cop"].sum().replace(0.0, np.nan),
            "n_heat": g["heat"].count(),
        }
    )
    return d.dropna(subset=["heat", "el_heat", "el_dhw", "t_out"])


def fit_split_cop(d: pd.DataFrame) -> tuple[float, float]:
    """Least squares heat = cop_h * el_heat + cop_d * el_dhw, no intercept."""
    X = d[["el_heat", "el_dhw"]].to_numpy(float)
    y = d["heat"].to_numpy(float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def bootstrap(d: pd.DataFrame, fn, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95 % percentile interval of fn over day-resampled frames."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(d), len(d))
        try:
            vals.append(fn(d.iloc[idx]))
        except (np.linalg.LinAlgError, ZeroDivisionError, ValueError):
            continue
    lo, hi = np.nanpercentile(np.asarray(vals, float), [2.5, 97.5])
    return float(lo), float(hi)


def heating_cop_given_dhw(d: pd.DataFrame, cop_d: float) -> float:
    """Energy-weighted heating COP after removing DHW heat at cop_d: sum(heat - cop_d*el_dhw)/sum(el_heat)."""
    return float((d["heat"] - cop_d * d["el_dhw"]).sum() / d["el_heat"].sum())


def formula_cop_fn(physics_cfg: dict):
    """Vectorised wrapper over the production ThermalPhysicsModel._cop_formula_value (no shift)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps"))
    from energy_forecast.physics import ThermalPhysicsModel

    model = ThermalPhysicsModel(Path(tempfile.mkdtemp(prefix="cop_scale_")), physics_cfg)
    return lambda temps: np.array([model._cop_formula_value(float(t), None) for t in np.atleast_1d(temps)])


def bin_table(d: pd.DataFrame, cop_d: float, formula) -> pd.DataFrame:
    rows = []
    hd = d[d["el_heat"] >= MIN_EL_HEAT_DAY_KWH]
    for lo, hi in zip(TEMP_BINS[:-1], TEMP_BINS[1:], strict=True):
        b = hd[(hd["t_out"] >= lo) & (hd["t_out"] < hi)]
        if len(b) < 3:
            continue
        cop_h = heating_cop_given_dhw(b, cop_d)
        ci = bootstrap(b, lambda x: heating_cop_given_dhw(x, cop_d), n=1000)
        live = b.dropna(subset=["cop_live_w"])
        rows.append(
            {
                "t_out bin": f"[{lo},{hi})",
                "days": len(b),
                "t_out mean": b["t_out"].mean(),
                "COP_h counters": cop_h,
                "95% CI": f"{ci[0]:.2f}-{ci[1]:.2f}",
                "COP formula": float(np.average(formula(b["t_out"]), weights=b["el_heat"])),
                "COP live (el-wtd)": (
                    float(np.average(live["cop_live_w"], weights=live["el_heat"])) if len(live) else np.nan
                ),
                "live days": len(live),
            }
        )
    return pd.DataFrame(rows)


def fit_ua(d: pd.DataFrame, q_heat_kwh: pd.Series, with_pv: bool) -> tuple[float, float]:
    """Daily heat balance q = UA*24*dT/1000 (- k*pv). Returns (UA W/K, R^2)."""
    dt = (d["t_in"] - d["t_out"]).clip(lower=0) * 24.0 / 1000.0
    X = np.c_[dt, -d["pv"]] if with_pv else dt.to_numpy()[:, None]
    y = q_heat_kwh.to_numpy(float)
    coef, *_ = np.linalg.lstsq(X.astype(float), y, rcond=None)
    yhat = X @ coef
    r2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return float(coef[0]), float(r2)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--start", required=True, type=pd.Timestamp)
    ap.add_argument("--end", default=None, type=pd.Timestamp)
    ap.add_argument("--season-start", default="2026-10-01", type=pd.Timestamp)
    ap.add_argument("--apps-yaml", default="data/apps.yaml")
    ap.add_argument("--csv", default=None, help="write the daily frame here")
    args = ap.parse_args()

    token = os.environ.get("EM_HA_TOKEN")
    if not token:
        print("NO-CREDENTIAL: EM_HA_TOKEN not set", file=sys.stderr)
        return 2

    cfg = yaml.safe_load(open(args.apps_yaml))["energy_forecast"]
    physics_cfg = cfg.get("physics", {})
    rooms = physics_cfg.get("room_thermostats") or []
    room_ids = {r["temp_sensor"]: float(r.get("area_m2", 1.0)) for r in rooms}

    end = args.end or pd.Timestamp.now(tz=TZ).normalize().tz_localize(None)
    start_iso = args.start.tz_localize(TZ).isoformat()
    end_iso = end.tz_localize(TZ).isoformat()
    ids = [*COUNTERS.values(), COP_LIVE, *T_OUT_CANDIDATES, *room_ids]
    stats = asyncio.run(fetch_statistics(ids, start_iso, end_iso, token))
    t_out_id = next((e for e in T_OUT_CANDIDATES if stats.get(e)), None)
    if t_out_id is None:
        print("no outdoor temperature statistics found", file=sys.stderr)
        return 1

    cols = {k: rows_to_hourly_delta(stats.get(eid, []), MAX_HOURLY_KWH[k]) for k, eid in COUNTERS.items()}
    cols["cop_live"] = rows_to_hourly_mean(stats.get(COP_LIVE, []))
    cols["t_out"] = rows_to_hourly_mean(stats[t_out_id])
    room = pd.DataFrame({eid: rows_to_hourly_mean(stats.get(eid, [])) for eid in room_ids})
    if not room.empty:
        w = pd.Series(room_ids)[room.columns]
        cols["t_in"] = (room * w).sum(axis=1) / (room.notna() * w).sum(axis=1)
    hourly = pd.DataFrame(cols).sort_index()

    print("statistics coverage:")
    for k, eid in [*COUNTERS.items(), ("cop_live", COP_LIVE), ("t_out", t_out_id)]:
        rows = stats.get(eid, [])
        first = _local_index(rows[:1])[0].date() if rows else None
        print(f"  {k:9s} {len(rows):6d} h  from {first}")
    print(f"  rooms     {len(room.columns)}/{len(room_ids)} sensors with statistics")

    d = daily_frame(hourly)
    if args.csv:
        d.to_csv(args.csv)
    print(f"\ncomplete days: {len(d)}  ({d.index.min().date()} .. {d.index.max().date()})")
    other = (d["el_total"] - d["el_heat"] - d["el_dhw"]).dropna()
    print(f"el_total - (el_heat + el_dhw): median {other.median():.2f} kWh/day, max {other.abs().max():.2f}")
    print(f"whole-period heat / el_total: {d['heat'].sum() / d['el_total'].sum():.2f}")

    formula = formula_cop_fn(physics_cfg)
    for label, sub in [("all days", d), (f"season >= {args.season_start.date()}", d[d.index >= args.season_start])]:
        if len(sub) < 5:
            print(f"\n{label}: too few days ({len(sub)})")
            continue
        cop_h, cop_d = fit_split_cop(sub)
        ci_h = bootstrap(sub, lambda x: fit_split_cop(x)[0])
        ci_d = bootstrap(sub, lambda x: fit_split_cop(x)[1])
        print(f"\n== {label}: n={len(sub)} days")
        print(f"split fit: COP_heating {cop_h:.2f} [{ci_h[0]:.2f}, {ci_h[1]:.2f}]", end="   ")
        print(f"COP_dhw {cop_d:.2f} [{ci_d[0]:.2f}, {ci_d[1]:.2f}]")
        dhw_only = sub[(sub["el_heat"] < 0.2) & (sub["el_dhw"] > 0.5)]
        if len(dhw_only) >= 3:
            ratio = dhw_only["heat"].sum() / dhw_only["el_dhw"].sum()
            print(f"DHW-only days (el_heat<0.2): n={len(dhw_only)}  heat/el_dhw {ratio:.2f}")
        tbl = bin_table(sub, cop_d, formula)
        if not tbl.empty:
            print(tbl.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

        hd = sub[(sub["el_heat"] >= MIN_EL_HEAT_DAY_KWH)].dropna(subset=["t_in", "pv"])
        if len(hd) >= 5:
            print(f"UA from daily heat balance (heating days with indoor temp, n={len(hd)}):")
            q_counter = hd["heat"] - cop_d * hd["el_dhw"]
            q_formula = hd["el_heat"] * formula(hd["t_out"])
            live = hd["cop_live_w"].fillna(pd.Series(formula(hd["t_out"]), index=hd.index))
            q_live = hd["el_heat"] * live
            for name, q in [("counter heat", q_counter), ("formula COP", q_formula), ("live COP", q_live)]:
                ua0, r20 = fit_ua(hd, q, with_pv=False)
                ua1, r21 = fit_ua(hd, q, with_pv=True)
                print(f"  {name:13s} UA {ua0:6.1f} W/K (R2 {r20:.2f})   with PV term: UA {ua1:6.1f} W/K (R2 {r21:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
