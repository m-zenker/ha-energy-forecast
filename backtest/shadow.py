"""Shadow runner: frozen backtest arms on live captures, logged as they happen.

Every live predict capture holds the exact serving inputs (with the real weather forecast) and every retrain capture
the exact training inputs, so the arms of ``walkforward`` can be trained and served on what live saw. ``run`` is
idempotent: it only does captures without a rows file, so a missed night is caught up by the next run.

Layout of the output folder: ``rows/<predict capture>.pkl``, ``models/<arm>/<retrain capture>/``, ``status.json``.
Needs the env that matches the HA host, like every capture replay. HA is never contacted here.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from . import capture_replay as cr
from . import wf_score as ws
from .env import REPO, TZ, frozen_now, lgbm_seed
from .walkforward import ARMS as WF_ARMS
from .walkforward import WfArm

ORIGIN_HOURS = (0, 6, 12, 18)
BURN_IN = pd.Timedelta(days=7)  # arm chains start this long before the first logged origin
KEEP_MODELS = 3
MIN_DAYS = 14  # below this every verdict is INCONCLUSIVE
META = "shadow_meta.json"
PARITY_TOL_KWH = 1e-6
ARM_FILES = (
    "backtest/walkforward.py",
    "backtest/regime_arms.py",
    "backtest/shadow.py",
    "backtest/capture_replay.py",
    "backtest/env.py",  # lgbm_seed (the s1 arm) and frozen_now
)
# "s1" is base with another LightGBM seed: the noise floor (a captured retrain cannot be shifted an hour like jit)
ARMS: dict[str, WfArm] = {
    **{name: WF_ARMS[name] for name in ("base", "droptp", "droptp_nb", "shapecl", "clrecent", "rescale")},
    "s1": WfArm("s1", patch=lambda: lgbm_seed(1)),
}
ARM_NAMES = ("base", "s1", "droptp", "droptp_nb", "shapecl", "clrecent", "rescale")


class SkipCapture(Exception):
    """This predict capture cannot be served by a faithful model; the message says why."""


def _hash(paths: Sequence[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:12]


def code_hash(repo: Path = REPO) -> str:
    """Hash of the files that define the arms and how they are run."""
    return _hash([Path(repo) / f for f in ARM_FILES])


def app_hash(repo: Path = REPO) -> str:
    return _hash(sorted((Path(repo) / "apps" / "energy_forecast").glob("*.py")))


def _naive_local(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_convert(TZ).tz_localize(None) if ts.tzinfo else ts


def select_origins(sidecars: Sequence[dict]) -> list[dict]:
    """The first predict capture without a running retrain in each of the hours 00, 06, 12, 18 local."""
    slots: dict[pd.Timestamp, dict] = {}
    for sc in sorted(sidecars, key=lambda s: s["ts_utc"]):
        hour = _naive_local(sc["ts_utc"]).floor("h")
        if hour.hour in ORIGIN_HOURS and not sc.get("retrain_in_progress"):
            slots.setdefault(hour, sc)
    return list(slots.values())


def _replace_dir(tmp: Path, final: Path) -> None:
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)


def _read_meta(model_dir: Path, code: str | None, start: str | None) -> dict | None:
    """Meta of a finished model dir trained by this arm code from this chain start, else None (rebuild it)."""
    try:
        meta = json.loads((model_dir / META).read_text())
    except (OSError, ValueError):
        return None
    return meta if (meta["code"], meta["start"]) == (code, start) else None


def _ensure_model(
    root: Path,
    out: Path,
    name: str,
    chain: Sequence[dict],
    rc: dict,
    code: str | None = None,
    app: str | None = None,
) -> tuple[Path, dict]:
    """Model dir (and its meta) of arm *name* after retrain *rc*, training forward from the newest usable link.

    ``base`` starts every retrain from the captured pre-train state, so it reproduces live even across a missing
    capture. Every other arm continues from its own previous model; its first link starts from the captured state.
    Links trained by other arm code or from another chain start are rebuilt; an app-code change is not a rebuild,
    because live also continues from the model it had. A link that cannot be trained is skipped and counted in
    ``chain_skips``, so one bad retrain capture does not block every later one; *rc* itself must train.
    """
    arm = ARMS[name]
    if arm.drop:
        raise ValueError(f"arm {name!r}: feature_drop arms are not supported here, use train_kwargs")
    folder = out / "models" / name
    ids = [c["capture_id"] for c in chain]
    if rc["capture_id"] not in ids:
        raise SkipCapture(f"retrain capture {rc['capture_id']} is before the chain start")
    start = None if name == "base" else ids[0]
    upto = ids[: ids.index(rc["capture_id"]) + 1]
    if name == "base":
        upto = upto[-1:]
    metas = {i: m for i, cid in enumerate(upto) if (m := _read_meta(folder / cid, code, start)) is not None}
    newest = max(metas) if metas else None
    previous = folder / upto[newest] if newest is not None else None
    meta = metas[newest] if newest is not None else {}
    skips = meta.get("chain_skips", 0)
    for cid in upto[(newest + 1 if newest is not None else 0) :]:
        tmp = folder / f"{cid}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            cr.replay_train(
                root,
                cid,
                tmp,
                seed_dir=previous,
                patch=arm.patch,
                energy_transform=arm.energy_transform,
                train_kwargs=arm.train_kwargs,
            )
        except cr.NoSeedError as exc:
            raise SkipCapture(f"no seed: {exc}") from exc
        except Exception:
            if cid == upto[-1]:
                raise
            skips += 1
            shutil.rmtree(tmp, ignore_errors=True)
            continue
        meta = dict(code=code, app=app, start=start, chain_skips=skips)
        (tmp / META).write_text(json.dumps(meta))
        _replace_dir(tmp, folder / cid)
        previous = folder / cid
    return folder / upto[-1], meta


def _prune_models(out: Path, name: str, chain: Sequence[dict]) -> None:
    folder = out / "models" / name
    ids = [c["capture_id"] for c in chain if (folder / c["capture_id"]).is_dir()]
    for cid in ids[:-KEEP_MODELS]:
        # rename first: a kill inside rmtree must not leave a half-deleted dir under a usable name
        doomed = folder / f"{cid}.tmp"
        if doomed.exists():
            shutil.rmtree(doomed, ignore_errors=True)
        os.replace(folder / cid, doomed)
        shutil.rmtree(doomed, ignore_errors=True)


def _forecast(model_dir: Path, payload: dict, init: dict) -> pd.DataFrame:
    """48 h forecast of the model in *model_dir* on a predict payload: ``ts, p50`` and, if available, ``lo, hi``."""
    from energy_forecast.model import EnergyForecastModel

    with tempfile.TemporaryDirectory() as tmp, frozen_now(payload["now_local"], init.get("timezone", TZ)):
        for p in model_dir.iterdir():
            if p.is_file():
                shutil.copy2(p, Path(tmp) / p.name)
        model = EnergyForecastModel(Path(tmp), **init)
        kwargs = copy.deepcopy(payload["predict_kwargs"])
        prepared = model._prepare_prediction_X(**kwargs)
        point = model.predict(**kwargs, _prepared=prepared)
        interval_kwargs = {k: v for k, v in kwargs.items() if k != "heating_buffer_temp_recent"}
        intervals = model.predict_intervals(**interval_kwargs, _prepared=prepared)
    ts = pd.to_datetime(point["timestamp"]).dt.tz_localize(None).to_numpy()
    rows = pd.DataFrame({"ts": ts, "p50": point["predicted_kwh"].to_numpy(dtype=float)})
    if intervals is not None:
        rows["lo"] = intervals["low_kwh"].to_numpy(dtype=float)
        rows["hi"] = intervals["high_kwh"].to_numpy(dtype=float)
    return rows


def _process(root: Path, out: Path, arms: Sequence[str], sc: dict, chain: Sequence[dict], now, code: str, app: str):
    """Rows of *arms* for one predict capture; ``parity`` is set when base is among them."""
    capture = cr._capture_mod()
    want = sc.get("last_trained")
    rc = next((c for c in chain if want is not None and c.get("last_trained_after") == want), None)
    if rc is None:
        raise SkipCapture(f"no finished retrain capture in the chain for last_trained {want}")
    payload, _ = capture.load_capture(root, "predict", sc["capture_id"])
    init = cr._model_init_for(root, want)
    origin = _naive_local(payload["now_local"]).floor("h")
    cutoff = _naive_local(rc["ts_utc"]).floor("h")
    live = payload["model_predictions"]["predicted_kwh"].to_numpy(dtype=float)
    parts, parity = [], None
    for name in arms:
        model_dir, meta = _ensure_model(root, out, name, chain, rc, code, app)
        rows = _forecast(model_dir, payload, init)
        if name == "base":
            ok = len(rows) == len(live) and abs(rows["p50"].to_numpy() - live).max() <= PARITY_TOL_KWH
            parity = "PASS" if ok else "FAIL"
        rows.insert(0, "arm", name)
        rows["model_app_code"], rows["chain_skips"] = meta.get("app"), meta.get("chain_skips", 0)
        parts.append(rows)
    df = pd.concat(parts, ignore_index=True)
    df["origin"], df["cutoff"] = origin, cutoff
    df["lead"] = ((df["ts"] - origin) / pd.Timedelta(hours=1)).astype(int)
    df["logged_at"], df["blind"] = now, now < df["ts"]
    df["code"], df["app_code"] = code, app
    return df, parity


def _write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_pickle(tmp, compression=None)
    os.replace(tmp, path)


def run(
    root: Path,
    out: Path,
    arms: Sequence[str],
    now: pd.Timestamp,
    since: pd.Timestamp,
    log: Callable[[str], None] = lambda _msg: None,
    pull_exit: int | None = None,
) -> dict:
    """Log every selected predict capture from *since* on that lacks an arm; returns the status of this run.

    ``base`` is always included. A slot already logged stays with the capture it has; rows of arms logged before are
    kept as they are.
    """
    capture = cr._capture_mod()
    root, out = Path(root), Path(out)
    rows_dir = out / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    arms = list(dict.fromkeys(("base", *arms)))
    now, since = _naive_local(now), _naive_local(since)
    code, app = code_hash(), app_hash()
    chain = [
        c
        for c in capture.list_captures(root, "retrain")
        if c.get("last_trained_after") and _naive_local(c["ts_utc"]) >= since - BURN_IN
    ]
    status: dict = dict(
        last_run=str(now), since=str(since), pull_exit=pull_exit, processed=[], skipped={}, parity={}, errors={}
    )
    predicts = capture.list_captures(root, "predict")
    if not predicts:
        status["errors"]["captures"] = f"no predict captures under {root}"
    logged = {}
    for path in sorted(rows_dir.glob("*.pkl")):
        rows = pd.read_pickle(path, compression=None)
        logged[rows["origin"].iloc[0]] = path, rows
    for sc in select_origins(predicts):
        cid = sc["capture_id"]
        at = _naive_local(sc["ts_utc"])
        if at < since:
            continue
        path, old = logged.get(at.floor("h"), (rows_dir / f"{cid}.pkl", None))
        if old is not None and path.stem != cid:
            # the slot was logged from another capture: complete it from that one, never from this one
            sc = next((p for p in predicts if p["capture_id"] == path.stem), None)
            if sc is None:
                continue
            cid = path.stem
        missing = [a for a in arms if old is None or a not in set(old["arm"])]
        if not missing:
            continue
        try:
            df, parity = _process(root, out, missing, sc, chain, now, code, app)
            df["parity"] = parity if old is None else old["parity"].iloc[0]
            _write(df if old is None else pd.concat([old, df], ignore_index=True), path)
        except SkipCapture as exc:
            status["skipped"][cid] = str(exc)
        except Exception as exc:  # noqa: BLE001  one broken capture must not stop the catch-up
            status["errors"][cid] = f"{type(exc).__name__}: {exc}"
        else:
            status["processed"].append(cid)
            if parity is not None:
                status["parity"][cid] = parity
            log(f"{cid}: {len(df)} rows, parity {parity}")
    for name in arms:
        with contextlib.suppress(OSError):
            _prune_models(out, name, chain)
    tmp = out / "status.json.tmp"
    tmp.write_text(json.dumps(status, indent=1))
    os.replace(tmp, out / "status.json")
    return status


def exit_code(status: dict) -> int:
    """2 when base did not reproduce live in this run, 1 when a capture errored, else 0 (skips are only reported)."""
    if "FAIL" in status["parity"].values():
        return 2
    return 1 if status["errors"] else 0


DEFAULT_SINCE = "2026-11-26"  # first target day of the fresh window (earlier days were used to find the cause)
SHADOW_ARMS = ("droptp", "droptp_nb")
CANDIDATE_ARMS = ("shapecl", "clrecent", "rescale", "s1")
GAP = pd.Timedelta(days=2)


def load_rows(out: Path) -> pd.DataFrame:
    """All logged rows, empty frame when there are none."""
    files = sorted((Path(out) / "rows").glob("*.pkl"))
    return (
        pd.concat([pd.read_pickle(f, compression=None) for f in files], ignore_index=True) if files else pd.DataFrame()
    )


def score_rows(rows: pd.DataFrame, actuals: pd.Series, daily_temp: pd.Series, since: pd.Timestamp) -> dict:
    """#120 shadow verdicts and cold-bias candidate verdicts for target days from *since* on.

    Dates are filtered before the day table and the swing, so burn-in days never count. Rows from different arm code
    make every verdict INCONCLUSIVE; a changed app code is only listed, because a live deploy changes it. Origins
    where base did not reproduce live, or which lack an arm, are left out for every arm, so all arms are compared on
    the same origins.
    """
    res: dict = dict(days=0, reason=None, codes=[], app_codes=[], blind_share=None, table=None, swing=None)
    res.update(parity_failed=[], incomplete_origins=0, chain_skips=0, origins=None)
    res["shadow"] = {a: {"verdict": "INCONCLUSIVE"} for a in SHADOW_ARMS}
    res["candidate"] = {a: {"verdict": "INCONCLUSIVE"} for a in CANDIDATE_ARMS}
    if rows.empty:
        return {**res, "reason": "no rows logged"}
    failed = rows.loc[rows["parity"] != "PASS", "origin"].unique()
    res["parity_failed"] = sorted(pd.Timestamp(o) for o in failed)
    rows = rows[~rows["origin"].isin(failed)]
    if rows.empty:
        return {**res, "reason": "no origin where base reproduced live"}
    per_origin = rows.groupby("origin")["arm"].nunique()
    short = per_origin.index[per_origin < rows["arm"].nunique()]
    res["incomplete_origins"] = len(short)
    rows = rows[~rows["origin"].isin(short)]
    res["chain_skips"] = int(rows["chain_skips"].max()) if len(rows) else 0
    scored = ws.attach_actuals(rows, actuals, daily_temp)
    scored = scored[scored["day"] >= pd.Timestamp(since)]
    table = ws.day_table(ws.day_ahead(scored))
    if table.empty:
        return {**res, "reason": "no rows with complete actuals in the window"}
    scored = scored[scored["day"].isin(table.index.get_level_values("day"))]
    swing = ws.next_day_swing(scored)
    res.update(
        days=int(table.xs("base", level="arm").shape[0]) if "base" in table.index.get_level_values("arm") else 0,
        codes=sorted(scored["code"].unique()),
        app_codes=sorted(scored["app_code"].unique()),
        blind_share=float(scored["blind"].mean()),
        table=table,
        swing=swing,
        origins=scored[["arm", "origin"]].drop_duplicates(),
    )
    if len(res["codes"]) > 1:
        return {**res, "reason": f"rows come from different arm code: {', '.join(res['codes'])}"}
    res["shadow"] = {a: ws.shadow_verdict(table, swing, a) for a in SHADOW_ARMS}
    if res["days"] < MIN_DAYS:  # shadow_verdict has no day minimum of its own
        for verdict in res["shadow"].values():
            verdict["verdict"] = "INCONCLUSIVE"
    res["candidate"] = {a: ws.candidate_verdict(table, swing, daily_temp, a) for a in CANDIDATE_ARMS}
    return res


def score(out: Path, root: Path, since: pd.Timestamp) -> dict:
    """``score_rows`` with actuals and archive temperature of the newest retrain capture."""
    from .capture_frame import CaptureFrame

    frame = CaptureFrame.from_capture(Path(root))
    weather = frame.weather
    daily_temp = weather["temp_c"].groupby(pd.to_datetime(weather["timestamp"]).dt.normalize()).mean()
    return score_rows(load_rows(out), frame.actuals(), daily_temp, since)


def status_text(out: Path, now: pd.Timestamp) -> str:
    """Last run, its problems, and gaps of more than two days between logged origins."""
    path = Path(out) / "status.json"
    if not path.exists():
        return "no run yet"
    st = json.loads(path.read_text())
    lines = [f"last run: {st['last_run']} (now {now})"]
    for key in ("errors", "skipped"):
        lines += [f"{key[:-1] if key == 'errors' else key}: {cid}: {msg}" for cid, msg in st.get(key, {}).items()]
    if st.get("pull_exit"):
        lines.append(f"pull failed (exit {st['pull_exit']}): HA drops captures after 45 days")
    for path in sorted((Path(out) / "rows").glob("*.pkl")):  # from the rows: a later run must not hide it
        if not (pd.read_pickle(path, compression=None)["parity"] == "PASS").all():
            lines.append(f"parity FAIL: {path.stem}")
    rows = load_rows(out)
    origins = [] if rows.empty else sorted(rows["origin"].unique())
    lines.append(f"logged origins: {len(origins)}" + (f", newest {pd.Timestamp(origins[-1])}" if origins else ""))
    for a, b in zip(origins, origins[1:], strict=False):
        if pd.Timestamp(b) - pd.Timestamp(a) > GAP:
            lines.append(f"gap: no origin between {pd.Timestamp(a)} and {pd.Timestamp(b)}")
    if origins and _naive_local(now) - pd.Timestamp(origins[-1]) > GAP:
        lines.append(f"gap: newest origin is {_naive_local(now) - pd.Timestamp(origins[-1])} old")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m backtest.shadow", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("run", "score", "status"):
        p = sub.add_parser(cmd)
        if cmd != "status":
            p.add_argument("captures_dir", type=Path)
            p.add_argument("--since", default=None, help=f"first target day (default {DEFAULT_SINCE})")
        if cmd == "run":
            p.add_argument("--pull-exit", type=int, default=None, help="exit code of the capture pull, for the status")
            p.add_argument("--arms", nargs="+", default=list(ARM_NAMES), choices=sorted(ARMS))
        p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    now = pd.Timestamp.now(tz=TZ).tz_localize(None).floor("s")
    if args.cmd == "status":
        print(status_text(args.out, now))
        return 0
    since = pd.Timestamp(args.since or DEFAULT_SINCE)
    if args.cmd == "run":
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / ".run.lock", "w") as lock:  # also covers a run by hand next to the container
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("another shadow run is active")
                return 0
            # day-ahead rows of the first target day come from the two origin days before it
            first = since - pd.Timedelta(days=2)
            status = run(args.captures_dir, args.out, args.arms, now, first, log=print, pull_exit=args.pull_exit)
        print(status_text(args.out, now))
        return exit_code(status)
    res = score(args.out, args.captures_dir, since)
    print(f"days {res['days']}, blind rows {res['blind_share']}, arm code {res['codes']}, app code {res['app_codes']}")
    print(f"left out: parity FAIL {res['parity_failed']}, incomplete origins {res['incomplete_origins']}")
    print(f"most skipped retrain links in a chain: {res['chain_skips']}")
    if res["reason"]:
        print(f"INCONCLUSIVE: {res['reason']}")
    for kind in ("shadow", "candidate"):
        for arm, verdict in res[kind].items():
            print(kind, arm, verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
