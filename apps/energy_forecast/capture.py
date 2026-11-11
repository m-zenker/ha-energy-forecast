"""Capture IO: persist exactly what the live app feeds its ML model, for offline replay.

Layout: ``<root>/<kind>/<capture_id>.pkl.gz`` (payload) + ``<capture_id>.json`` (sidecar),
optionally ``<capture_id>.ref/`` (copy of the model dir, retrain captures only).
All write-side functions never raise; failures are logged as warnings.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import pickle
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Literal

import pandas as pd

_LOGGER = logging.getLogger("energy_forecast")

RETENTION_DAYS = 45
MAX_TOTAL_BYTES = 1_000_000_000
_TMP_MAX_AGE_S = 3600
_KINDS = ("retrain", "predict")
_SIDE_DIRS = (".ref", ".pre")  # model-file dirs stored next to a retrain capture


def library_versions() -> dict[str, str]:
    """Versions of the packages that affect model behaviour ("missing" if not importable)."""
    out = {"python": ".".join(map(str, sys.version_info[:3]))}
    for key, mod in (("pandas", "pandas"), ("numpy", "numpy"), ("sklearn", "sklearn"), ("lightgbm", "lightgbm")):
        try:
            out[key] = str(__import__(mod).__version__)
        except Exception:
            out[key] = "missing"
    return out


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, default=str))
    os.replace(tmp, path)


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def write_capture(
    root: Path,
    kind: Literal["retrain", "predict"],
    payload: dict,
    meta: dict,
    now_utc: pd.Timestamp,
) -> str | None:
    """Write payload + sidecar atomically; return the capture_id, or None on failure."""
    try:
        root = Path(root)
        capture_id = now_utc.strftime("%Y%m%dT%H%M%S%fZ")
        d = root / kind
        d.mkdir(parents=True, exist_ok=True)
        pkl = d / f"{capture_id}.pkl.gz"
        tmp = d / f"{capture_id}.pkl.gz.tmp"
        with gzip.open(tmp, "wb", compresslevel=6) as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, pkl)
        sidecar = {
            **meta,
            "capture_id": capture_id,
            "kind": kind,
            "ts_utc": now_utc.isoformat(),
            "versions": library_versions(),
            "bytes": pkl.stat().st_size,
        }
        try:
            _atomic_write_json(d / f"{capture_id}.json", sidecar)
        except Exception:
            pkl.unlink(missing_ok=True)
            raise
    except Exception as exc:
        _LOGGER.warning("capture: write failed (%s): %s", kind, exc)
        return None
    prune(root, now_utc, protect=capture_id)
    return capture_id


def update_sidecar(root: Path, kind: str, capture_id: str, **fields) -> None:
    """Merge fields into an existing sidecar (atomic rewrite)."""
    try:
        path = Path(root) / kind / f"{capture_id}.json"
        data = json.loads(path.read_text())
        data.update(fields)
        _atomic_write_json(path, data)
    except Exception as exc:
        _LOGGER.warning("capture: sidecar update failed (%s/%s): %s", kind, capture_id, exc)


def _capture_bytes(d: Path, capture_id: str) -> int:
    """Pickle size plus every model-file side dir of one capture."""
    total = (d / f"{capture_id}.pkl.gz").stat().st_size
    for suffix in _SIDE_DIRS:
        side = d / f"{capture_id}{suffix}"
        if side.is_dir():
            total += _dir_size(side)
    return total


def _attach_model_dir(root: Path, capture_id: str, model_dir: Path, suffix: str, flag: str) -> None:
    """Copy the model dir's regular files (not archive/) to ``<capture_id><suffix>`` and flag the sidecar."""
    d = Path(root) / "retrain"
    side = d / f"{capture_id}{suffix}"
    side.mkdir(parents=True, exist_ok=True)
    if Path(model_dir).is_dir():
        for p in Path(model_dir).iterdir():
            if p.is_file():
                shutil.copy2(p, side / p.name)
    update_sidecar(root, "retrain", capture_id, bytes=_capture_bytes(d, capture_id), **{flag: True})


def attach_reference(root: Path, capture_id: str, model_dir: Path) -> None:
    """Copy the model dir's regular files (not archive/) next to a retrain capture (the model train() produced)."""
    try:
        _attach_model_dir(root, capture_id, model_dir, ".ref", "reference")
    except Exception as exc:
        _LOGGER.warning("capture: attach_reference failed (%s): %s", capture_id, exc)


def attach_pre_state(root: Path, capture_id: str, model_dir: Path) -> None:
    """Copy the model dir as it is *before* train() runs. A missing dir yields an empty ``.pre`` (fresh install)."""
    try:
        _attach_model_dir(root, capture_id, model_dir, ".pre", "pre_state")
    except Exception as exc:
        _LOGGER.warning("capture: attach_pre_state failed (%s): %s", capture_id, exc)


def list_captures(root: Path, kind: str) -> list[dict]:
    """Sidecars sorted by ts_utc; skips partial (.tmp) files and sidecars without a pickle."""
    d = Path(root) / kind
    out: list[dict] = []
    if not d.is_dir():
        return out
    for sp in d.glob("*.json"):
        if not (d / f"{sp.stem}.pkl.gz").exists():
            continue
        try:
            out.append(json.loads(sp.read_text()))
        except Exception:
            continue
    return sorted(out, key=lambda s: s.get("ts_utc", ""))


def load_capture(root: Path, kind: str, capture_id: str) -> tuple[dict, dict]:
    """Return (payload, sidecar); warn per package whose version differs from the current env."""
    d = Path(root) / kind
    sidecar = json.loads((d / f"{capture_id}.json").read_text())
    with gzip.open(d / f"{capture_id}.pkl.gz", "rb") as fh:
        payload = pickle.load(fh)
    current = library_versions()
    for pkg, ver in (sidecar.get("versions") or {}).items():
        if current.get(pkg) != ver:
            warnings.warn(
                f"capture {capture_id}: {pkg} version {ver} at capture time, {current.get(pkg)} now",
                UserWarning,
                stacklevel=2,
            )
    return payload, sidecar


def _remove_capture(d: Path, capture_id: str) -> None:
    (d / f"{capture_id}.json").unlink(missing_ok=True)
    (d / f"{capture_id}.pkl.gz").unlink(missing_ok=True)
    for suffix in _SIDE_DIRS:
        shutil.rmtree(d / f"{capture_id}{suffix}", ignore_errors=True)


def _old_enough(path: Path) -> bool:
    try:
        return time.time() - path.stat().st_mtime > _TMP_MAX_AGE_S
    except OSError:
        return False


def _parse_ts(value) -> pd.Timestamp | None:
    """Parse a tz-aware timestamp; None if missing, unparseable or naive."""
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts) or ts.tzinfo is None:
        return None
    return ts


def prune(
    root: Path,
    now_utc: pd.Timestamp,
    retention_days: int = RETENTION_DAYS,
    max_bytes: int = MAX_TOTAL_BYTES,
    protect: str | None = None,
) -> int:
    """Drop captures past retention, then oldest-first until under max_bytes.

    Also clears stale .tmp files and orphan .pkl.gz / .ref dirs lacking a sidecar (older than 1 h).
    A capture with an unreadable sidecar is treated as the oldest. ``protect`` is never removed.
    """
    removed = 0
    try:
        root = Path(root)
        cutoff = now_utc - pd.Timedelta(days=retention_days)
        oldest = pd.Timestamp.min.tz_localize("UTC")
        entries: list[tuple[pd.Timestamp, Path, str, int]] = []
        for kind in _KINDS:
            d = root / kind
            if not d.is_dir():
                continue
            for tmp in d.glob("*.tmp"):
                if _old_enough(tmp):
                    tmp.unlink(missing_ok=True)
            for orphan in list(d.glob("*.pkl.gz")) + [p for sfx in _SIDE_DIRS for p in d.glob(f"*{sfx}")]:
                cid = orphan.name.split(".")[0]
                if not (d / f"{cid}.json").exists() and _old_enough(orphan):
                    _remove_capture(d, cid)
                    removed += 1
            for sp in d.glob("*.json"):
                cid = sp.stem
                if cid == protect or not (d / f"{cid}.pkl.gz").exists():
                    continue
                try:
                    size = _capture_bytes(d, cid) + sp.stat().st_size
                    try:
                        ts = _parse_ts(json.loads(sp.read_text()).get("ts_utc"))
                    except Exception:
                        ts = None
                    entries.append((ts if ts is not None else oldest, d, cid, size))
                except Exception as exc:
                    _LOGGER.warning("capture: prune skipped %s: %s", sp.name, exc)
        entries.sort(key=lambda e: e[0])
        keep = []
        for e in entries:
            if e[0] < cutoff:
                _remove_capture(e[1], e[2])
                removed += 1
            else:
                keep.append(e)
        total = sum(e[3] for e in keep)
        for e in keep:
            if total <= max_bytes:
                break
            _remove_capture(e[1], e[2])
            total -= e[3]
            removed += 1
    except Exception as exc:
        _LOGGER.warning("capture: prune failed: %s", exc)
    return removed
