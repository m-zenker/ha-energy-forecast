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
        _atomic_write_json(d / f"{capture_id}.json", sidecar)
    except Exception as exc:
        _LOGGER.warning("capture: write failed (%s): %s", kind, exc)
        return None
    prune(root, now_utc)
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


def attach_reference(root: Path, capture_id: str, model_dir: Path) -> None:
    """Copy the model dir's regular files (not archive/) next to a retrain capture."""
    try:
        ref = Path(root) / "retrain" / f"{capture_id}.ref"
        ref.mkdir(parents=True, exist_ok=True)
        for p in Path(model_dir).iterdir():
            if p.is_file():
                shutil.copy2(p, ref / p.name)
        pkl = Path(root) / "retrain" / f"{capture_id}.pkl.gz"
        total = pkl.stat().st_size + _dir_size(ref)
        update_sidecar(root, "retrain", capture_id, bytes=total, reference=True)
    except Exception as exc:
        _LOGGER.warning("capture: attach_reference failed (%s): %s", capture_id, exc)


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
    shutil.rmtree(d / f"{capture_id}.ref", ignore_errors=True)


def prune(
    root: Path,
    now_utc: pd.Timestamp,
    retention_days: int = RETENTION_DAYS,
    max_bytes: int = MAX_TOTAL_BYTES,
) -> int:
    """Drop captures past retention, then oldest-first until under max_bytes; clear stale .tmp files."""
    removed = 0
    try:
        root = Path(root)
        cutoff = now_utc - pd.Timedelta(days=retention_days)
        entries: list[tuple[str, Path, str, int]] = []
        for kind in _KINDS:
            d = root / kind
            if not d.is_dir():
                continue
            for tmp in d.glob("*.tmp"):
                if time.time() - tmp.stat().st_mtime > _TMP_MAX_AGE_S:
                    tmp.unlink(missing_ok=True)
            for sc in list_captures(root, kind):
                cid = sc["capture_id"]
                size = (
                    (d / f"{cid}.pkl.gz").stat().st_size
                    + (d / f"{cid}.json").stat().st_size
                    + (_dir_size(d / f"{cid}.ref") if (d / f"{cid}.ref").is_dir() else 0)
                )
                entries.append((sc.get("ts_utc", ""), d, cid, size))
        entries.sort(key=lambda e: pd.Timestamp(e[0]) if e[0] else pd.Timestamp.min.tz_localize("UTC"))
        keep = []
        for e in entries:
            ts = pd.Timestamp(e[0]) if e[0] else None
            if ts is not None and ts < cutoff:
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
