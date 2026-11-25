#!/bin/bash
# One shadow run: pull new captures from HA (read-only), then log the arms' forecasts. Safe to run any time.
# Needs EM_SMB_USER / EM_SMB_PASSWORD in the environment for the pull. Exit: run's code, else 1 if the pull failed.
# SHADOW_DATA, SHADOW_PULL and SHADOW_RUN override the defaults (used by the tests).
set -u
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA=${SHADOW_DATA:-/home/jovyan/work/ha-energy-forecast/data}
ENVS=/home/jovyan/my_envs
OUT=$DATA/backtest/shadow
mkdir -p "$OUT/log"

exec 9>"$OUT/.lock"
flock -n 9 || { echo "another shadow run is active"; exit 0; }

LOG=$OUT/log/$(date +%F).log
cd "$CODE" || exit 1
export PYTHONPATH=apps:. PYTHONDONTWRITEBYTECODE=1
{
  echo "== $(date -Is) shadow run"
  # smb is only in the test env; the arms need the env that matches the HA host
  if [ -n "${SHADOW_PULL:-}" ]; then timeout 600 "$SHADOW_PULL"
  else timeout 600 "$ENVS/ha-energy-forecast/bin/python" "$DATA/../scripts/pull_captures.py"; fi
  pull=$?
  [ $pull -eq 0 ] || echo "pull failed (exit $pull); continuing with local captures"
  if [ -n "${SHADOW_RUN:-}" ]; then "$SHADOW_RUN" run "$DATA/captures" --out "$OUT"
  else "$ENVS/ha-energy-forecast-live/bin/python" -m backtest.shadow run "$DATA/captures" --out "$OUT"; fi
  run=$?
  echo "== $(date -Is) done: pull $pull, run $run"
} >>"$LOG" 2>&1

ls -1 "$OUT"/log/*.log | sort | head -n -30 | xargs -r rm -f
[ $run -ne 0 ] && exit $run
[ $pull -eq 0 ] || exit 1
