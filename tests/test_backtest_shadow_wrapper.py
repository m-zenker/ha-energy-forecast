"""backtest/shadow_nightly.sh with stubbed pull and run commands."""

import fcntl
import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "backtest" / "shadow_nightly.sh"


def _stub(path: Path, body: str) -> str:
    path.write_text(f"#!/bin/bash\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _call(tmp_path, pull="exit 0", run="exit 0"):
    data = tmp_path / "data"
    (data / "captures").mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "SHADOW_DATA": str(data),
        "SHADOW_PULL": _stub(tmp_path / "pull", f'echo pulled >> "{tmp_path}/calls"; {pull}'),
        "SHADOW_RUN": _stub(tmp_path / "run", f'echo "ran $*" >> "{tmp_path}/calls"; {run}'),
    }
    res = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
    calls = (tmp_path / "calls").read_text().splitlines() if (tmp_path / "calls").exists() else []
    return res, calls, data


def test_success_pulls_then_runs(tmp_path):
    res, calls, data = _call(tmp_path)
    assert res.returncode == 0 and calls[0] == "pulled"
    assert calls[1] == f"ran run {data}/captures --out {data}/backtest/shadow"
    assert len(list((data / "backtest/shadow/log").glob("*.log"))) == 1


def test_failed_pull_still_runs_and_is_reported(tmp_path):
    res, calls, data = _call(tmp_path, pull="exit 7")
    assert res.returncode == 1 and len(calls) == 2
    assert "pull failed" in next((data / "backtest/shadow/log").glob("*.log")).read_text()


@pytest.mark.parametrize("pull", ["exit 0", "exit 7"])
def test_run_exit_code_wins(tmp_path, pull):
    assert _call(tmp_path, pull=pull, run="exit 2")[0].returncode == 2


def test_busy_lock_does_nothing(tmp_path):
    shadow = tmp_path / "data" / "backtest" / "shadow"
    shadow.mkdir(parents=True)
    with open(shadow / ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        res, calls, _ = _call(tmp_path)
    assert res.returncode == 0 and calls == []


def test_keeps_thirty_logs(tmp_path):
    log = tmp_path / "data" / "backtest" / "shadow" / "log"
    log.mkdir(parents=True)
    for i in range(40):
        (log / f"2026-01-{i:02d}.log").write_text("x")
    _call(tmp_path)
    names = sorted(p.name for p in log.iterdir())
    assert len(names) == 30 and "2026-01-00.log" not in names


def test_credentials_are_not_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("EM_SMB_PASSWORD", "s3cret-value")
    _, _, data = _call(tmp_path)
    assert "s3cret-value" not in next((data / "backtest/shadow/log").glob("*.log")).read_text()
