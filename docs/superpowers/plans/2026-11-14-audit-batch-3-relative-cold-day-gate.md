# Relative Cold-Day Gate on Live Data (ROADMAP #118) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** one command reads the live `pred_history.json` and returns PASS / FAIL / INCONCLUSIVE for the new gate: pooled cold-day bias within −12 % to +6 % of consumption over ≥ 14 cold days, on appliance-free actuals.

**Architecture:** new tracked module `backtest/live_gate.py` turns `pred_history.json` into the row format `backtest/wf_score.py` already scores, and reuses its `day_table`, `summarize` and `cold_gate`. The local, gitignored `scripts/check_cold_onset_gate.py` drops its own absolute-band verdict and calls the module.

**Tech Stack:** Python 3.13, pandas, pytest; `backtest.wf_score`, `backtest.data`.

**Spec:** `ROADMAP.md` #118 and Design Decisions "Cold-day bias gate" (on `train/next`); index `docs/superpowers/plans/2026-11-14-audit-batch-index.md`.

## Global Constraints

- **Prerequisite:** phases 1 and 2 are merged into `train/next` (`git log train/next --oneline | grep -iE "appliance-free|accuracy"` shows both). If not, stop.
- Base branch `train/next`; branch `feat/live-cold-gate`; worktree `.worktrees/live-cold-gate`.
- Tests only via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest` (`$PY`).
- Gate values, verbatim from ROADMAP: low −12 %, high +6 %, minimum 14 cold days, fewer = INCONCLUSIVE.
- Cold day: daily mean outdoor temperature ≤ 10.0 °C (`backtest.wf_score.COLD_C`), overridable.
- A day counts only with ≥ 22 scored hours (`wf_score.MIN_DAY_HOURS`).
- Actuals without the marker `"actuals_basis": "target"` (written by phase 1) force INCONCLUSIVE unless explicitly overridden.
- No changes under `apps/`. Nothing here is deployed to HA.
- The module must be importable in the test env without network access; only the CLI may fetch weather.

## Review Focus

1. `pred_history` stores the first forecast made within 25 h of the target hour. Hours already stored before the change under test keep their old value, so "after the anchor" means target hours ≥ anchor + 24 h, rounded up to the hour. → Task 1 test.
2. Mixed file: migrated marker missing (phase 1 not deployed yet, or migration pending) must not yield PASS/FAIL. → Task 1 test.
3. Days with no temperature (weather cache ends before the newest actuals) are not cold days; they must not crash or count. → Task 1 test.
4. A bias 0.1 % inside a band edge is PASS and 0.1 % outside is FAIL, on both sides (the edges themselves are inclusive in `wf_score.cold_gate`). → Task 1 test.
5. Fewer than 14 cold days with a bias far outside the band is still INCONCLUSIVE, not FAIL. → Task 1 test.

---

## Session setup

```bash
REPO=/home/jovyan/work/ha-energy-forecast
git -C $REPO worktree add $REPO/.worktrees/live-cold-gate -b feat/live-cold-gate train/next
cd $REPO/.worktrees/live-cold-gate
PY=/home/jovyan/my_envs/ha-energy-forecast/bin/python
$PY -m pytest tests/ -q          # baseline: 0 failed
```

## File Structure

- Create `backtest/live_gate.py`, `tests/test_backtest_live_gate.py`.
- Modify `backtest/README.md`, `ROADMAP.md`, `CHANGELOG.md`.
- Modify (main checkout, gitignored, no commit): `/home/jovyan/work/ha-energy-forecast/scripts/check_cold_onset_gate.py`, `/home/jovyan/work/ha-energy-forecast/scripts/pull_ha_data.py`.

Background: `wf_score.attach_actuals(rows, actual, daily_temp)` needs `rows` with a `ts` column and adds `y`, `naive`, `day`, `cold`. `wf_score.day_table(rows)` needs `arm`, `ts`, `p50`, `y`, `day`, `cold`. `wf_score.add_naive(rows, ref)` appends a `naive` pseudo-arm. `wf_score.summarize(table)` returns per arm `days, mae, mae3, mae6, daily_abs, bias, cold_days, cold_bias, cold_bias_pct`. `wf_score.cold_gate(row, low_pct, high_pct, min_days)` returns the verdict string.

---

### Task 1: `backtest/live_gate.py`

**Files:**
- Create: `backtest/live_gate.py`
- Test: `tests/test_backtest_live_gate.py`

**Interfaces:**
- Produces:
  - `ARM = "live"`
  - `load_pred_history(path: Path) -> tuple[pd.Series, pd.Series, str | None]` — `(pred, actual, actuals_basis)`, both Series float, indexed by naive hourly `pd.Timestamp`, sorted.
  - `live_rows(pred: pd.Series, actual: pd.Series, daily_temp: pd.Series, since: pd.Timestamp | None = None, cold_below: float = wf_score.COLD_C) -> pd.DataFrame` — columns `arm, ts, p50, y, naive, day, cold`; one row per hour in `pred`; with `since`, only `ts >= (since + 24 h).ceil("1h")`; `cold = daily_temp[day] <= cold_below` (False when the day has no temperature).
  - `verdict(rows: pd.DataFrame, actuals_basis: str | None, low_pct: float = -12.0, high_pct: float = 6.0, min_days: int = 14, allow_mixed_actuals: bool = False) -> dict` with keys `gate` (`"PASS" | "FAIL" | "INCONCLUSIVE"`), `reason` (str, empty on PASS), `cold_days`, `cold_bias`, `cold_bias_pct`, `days`, `mae`, `mae3`, `daily_abs`, `bias`, `naive_mae`, `naive_mae3`.
  - `main(argv: Sequence[str] | None = None) -> int` — CLI, always returns 0.

`cold_days` and `days` are ints (0 when there are none; `wf_score.summarize` gives NaN there, convert it). `reason` values: `"actuals not on the target basis (#117 not live or migration pending)"`; `"only N cold day(s), need 14"`; `"cold-day bias +X.X % outside [-12, +6] %"`.

- [ ] **Step 1: Write the failing tests** — `tests/test_backtest_live_gate.py`

```python
import json

import pandas as pd
import pytest

from backtest import live_gate

START = pd.Timestamp("2026-11-01")


def _series(days: int, value: float) -> pd.Series:
    return pd.Series(value, index=pd.date_range(START, periods=24 * days, freq="1h"))


def _temp(days: int, value: float = 5.0) -> pd.Series:
    return pd.Series(value, index=pd.date_range(START, periods=days, freq="D"))


def _gate(pred_value, days=20, temp=5.0, basis="target", **kw):
    rows = live_gate.live_rows(_series(days, pred_value), _series(days, 1.0), _temp(days, temp))
    return live_gate.verdict(rows, basis, **kw)


def test_pass_inside_band():
    v = _gate(1.02)
    assert v["gate"] == "PASS" and v["reason"] == ""
    assert v["cold_days"] == 20 and v["cold_bias_pct"] == pytest.approx(2.0)
    assert v["cold_bias"] == pytest.approx(0.48)  # 0.02 kWh/h x 24


def test_fail_above_and_below_band():
    assert _gate(1.10)["gate"] == "FAIL"
    low = _gate(0.85)
    assert low["gate"] == "FAIL" and "outside" in low["reason"]


def test_just_inside_and_outside_the_band_edges():
    # not exactly 1.06 / 0.88: float rounding puts those a hair outside the inclusive edge
    assert _gate(1.059)["gate"] == "PASS" and _gate(1.061)["gate"] == "FAIL"
    assert _gate(0.881)["gate"] == "PASS" and _gate(0.879)["gate"] == "FAIL"


def test_inconclusive_below_min_cold_days_even_with_large_bias():
    v = _gate(1.50, days=10)
    assert v["gate"] == "INCONCLUSIVE" and "10 cold day" in v["reason"]


def test_warm_days_do_not_count():
    v = _gate(1.02, temp=15.0)
    assert v["gate"] == "INCONCLUSIVE" and v["cold_days"] == 0


def test_unmarked_actuals_are_inconclusive_unless_allowed():
    assert _gate(1.02, basis=None)["gate"] == "INCONCLUSIVE"
    assert "target basis" in _gate(1.02, basis="gross")["reason"]
    assert _gate(1.02, basis=None, allow_mixed_actuals=True)["gate"] == "PASS"


def test_since_starts_24h_after_the_anchor_rounded_up():
    rows = live_gate.live_rows(
        _series(20, 1.0), _series(20, 1.0), _temp(20), since=pd.Timestamp("2026-11-05 17:28")
    )
    assert rows["ts"].min() == pd.Timestamp("2026-11-06 18:00")


def test_days_without_temperature_are_not_cold():
    rows = live_gate.live_rows(_series(20, 1.02), _series(20, 1.0), _temp(15))
    assert live_gate.verdict(rows, "target")["cold_days"] == 15


def test_cold_below_override():
    rows = live_gate.live_rows(_series(20, 1.02), _series(20, 1.0), _temp(20, 12.0), cold_below=13.5)
    assert live_gate.verdict(rows, "target")["cold_days"] == 20


def test_load_pred_history_round_trip(tmp_path):
    p = tmp_path / "pred_history.json"
    p.write_text(json.dumps({
        "pred": {"2026-11-02T01:00:00": 1.5, "2026-11-02T00:00:00": 1.0},
        "actuals": {"2026-11-02T00:00:00": 0.9},
        "actuals_basis": "target",
    }))
    pred, actual, basis = live_gate.load_pred_history(p)
    assert basis == "target" and list(pred) == [1.0, 1.5] and actual.index[0] == pd.Timestamp("2026-11-02")
    p.write_text(json.dumps({"pred": {}, "actuals": {}}))
    assert live_gate.load_pred_history(p)[2] is None


def test_naive_guardrail_is_reported():
    v = _gate(1.02)
    assert v["naive_mae3"] == pytest.approx(0.0)  # constant load: the naive mean is exact
    assert v["mae3"] == pytest.approx(0.06)
```

- [ ] **Step 2: Run to verify they fail** — `$PY -m pytest tests/test_backtest_live_gate.py -q` → `ImportError: cannot import name 'live_gate'`

- [ ] **Step 3: Implement `backtest/live_gate.py`**

- `live_rows`: build `DataFrame({"arm": ARM, "ts": pred.index, "p50": pred.values})`, apply the `since` cut, call `wf_score.attach_actuals(rows, actual, daily_temp)`, then overwrite `cold` with the `cold_below` rule.
- `verdict`: `table = wf_score.day_table(wf_score.add_naive(rows, ref=ARM))`; `summary = wf_score.summarize(table)`; take the `ARM` row for the gate and metrics and the `naive` row for `naive_mae`/`naive_mae3`. Gate string from `wf_score.cold_gate(row, low_pct, high_pct, min_days)`, overridden to INCONCLUSIVE first when `actuals_basis != "target"` and not `allow_mixed_actuals`. When no day is scored at all, return INCONCLUSIVE with `cold_days` 0 and NaN metrics instead of raising (an empty `table` has no `ARM` row).
- `main`: arguments `data_dir` (positional, default `backtest.data.DATA_DIR`), `--since` (local timestamp of the deploy or retrain under test), `--cold-below` (float, default 10.0), `--min-days` (int, default 14), `--allow-mixed-actuals`, `--no-fetch`. Daily temperature = daily mean of `temp_c` from `backtest.data.load_weather(data_dir, fetch_tail=not args.no_fetch)`. Print: the anchor and first scored hour, the per-day table for the live arm (`day, n, mae, mae3, bias, y, cold`), the summary for `live` and `naive`, and a final line `GATE: <verdict> — <reason>`. Module docstring carries the usage line:

  ```
  PYTHONPATH=apps:. /home/jovyan/my_envs/ha-energy-forecast/bin/python -m backtest.live_gate data --since "2026-11-05 17:28"
  ```

- [ ] **Step 4: Run** — `$PY -m pytest tests/test_backtest_live_gate.py -q` → 11 passed; full suite 0 failed; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add backtest/live_gate.py tests/test_backtest_live_gate.py
git commit -m "feat(backtest): relative cold-day gate on live pred_history (#118)"
```

---

### Task 2: Smoke-run on real data

No code. This checks the module against the real file.

- [ ] **Step 1:** In the main checkout (`cd /home/jovyan/work/ha-energy-forecast`), run with the worktree on the path:

  ```bash
  W=/home/jovyan/work/ha-energy-forecast/.worktrees/live-cold-gate
  PYTHONPATH=$W/apps:$W $PY -m backtest.live_gate data --since "2026-11-05 17:28" --no-fetch
  ```

  Expected today: `GATE: INCONCLUSIVE — actuals not on the target basis ...` (phase 1 is not deployed yet, so the pulled file has no marker). Then add `--allow-mixed-actuals`: it must print a per-day table and a verdict without a traceback. If `data/pred_history.json` is missing, skip this task and say so in the hand-back.
- [ ] **Step 2:** Paste both outputs into the hand-back message. Do not record them in ROADMAP as a gate result: mixed actuals are not a valid reading.

---

### Task 3: Local scripts (main checkout, gitignored — edit, do not commit)

**Files:**
- Modify: `/home/jovyan/work/ha-energy-forecast/scripts/check_cold_onset_gate.py`
- Modify: `/home/jovyan/work/ha-energy-forecast/scripts/pull_ha_data.py`

- [ ] **Step 1: `check_cold_onset_gate.py`** — keep the per-day table, the hour-of-day profile and the band-coverage section. Remove the constant `BIAS_BAND` and the section printed under `"── Post-deploy cold days vs criterion"` (the per-day PASS/FAIL against [−3, +1] and the "gate needs ≥ 3" line). In its place:

  ```python
  try:
      from backtest import live_gate
  except ImportError:
      print("gate verdict needs backtest.live_gate: run with PYTHONPATH=.worktrees/train/apps:.worktrees/train")
  else:
      live_gate.main([str(DATA_DIR), "--since", args.deploy, "--no-fetch"])
  ```

  Update the module docstring: point 1 now says the verdict is the #118 relative gate from `backtest.live_gate`; the 06/07/20/21 h mask and the 13.5 °C threshold apply only to the descriptive per-day table.
- [ ] **Step 2: `pull_ha_data.py`** — in `_pull_captures`, a retrain capture can also have a `<cid>.pre/` directory (pre-train model snapshot, since alpha-38). Mirror the existing `<cid>.ref/` handling for `.pre`: include it in the "already complete locally" check and copy its files. Run `$PY scripts/pull_ha_data.py` only if the owner is available to confirm (it reads from the live HA share); otherwise leave it unrun and say so.
- [ ] **Step 3:** `$PY -m py_compile scripts/check_cold_onset_gate.py scripts/pull_ha_data.py` → no output.

---

### Task 4: Docs and merge to the train

- [ ] **Step 1:** `backtest/README.md`: new short section "Gate on live data" after "Walk-forward on a capture" with the command, the three verdicts and the `actuals_basis` rule.
- [ ] **Step 2:** ROADMAP: #118 status → "gate implemented in `backtest/live_gate.py` (date)"; Pending Summary row 118; in "Cold-onset sequencing" step 3 replace "Re-read with the #118 gate once #117 is live" with the command.
- [ ] **Step 3:** CHANGELOG `[Unreleased]` via `/changelog` (developer tooling; no `apps/` change).
- [ ] **Step 4:** `/home/jovyan/work/ha-energy-forecast/MEMORY.md`: one line with the command and that the first valid reading needs #117 deployed plus ≥ 14 cold days.
- [ ] **Step 5:** Merge and push:
  ```bash
  git -C /home/jovyan/work/ha-energy-forecast/.worktrees/train merge --no-ff feat/live-cold-gate
  cd /home/jovyan/work/ha-energy-forecast/.worktrees/train && $PY -m pytest tests/ -q
  git push origin train/next
  ```

**Out of scope:** re-reading the A+B gate (owner step once data exists); band-coverage logic; any change to `wf_score.cold_gate`.
