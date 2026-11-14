# Audit Batch (ROADMAP #117, #114, #118, #113, #120, #116) — Overarching Plan

> **For agentic workers:** this file is the index. Each phase has its own sub-plan, written to be executed in a separate session by a mid-tier model. Open the sub-plan for your phase and follow it task by task; read this file once for context and the shared rules.

**Goal:** fix the yardstick the forecast is judged by, then remove the two known blockers behind it (Phase 2 reconstruction, unstable thermal-pressure features), in the order agreed on 2026-11-13.

**Source of truth:** `ROADMAP.md` on `train/next` (items #113–#123 and "Audit sequencing") and `docs/analysis/2026-11-13-forecast-improvement-audit.md`. The `ROADMAP.md` in the main checkout (`dev`) is older and stops at #116.

## Why this batch

The 2026-11-13 audit found that hourly MAE is at its day-ahead floor and no model arm beats current code. What is wrong is the measurement: stored actuals include appliance energy the model excludes (every bias reading is about 0.8 kWh/day too low), the live sensors report only hourly MAE (which rewards under-forecasting), and the cold-day gate is an absolute band that does not scale with load. #105 (daily `UA_eff`) and everything behind it is on hold until that is fixed. #119 (capture-frame backtest) is already on `train/next` (b03b19d).

Not in this batch, by decision: #121, #122, #123 (low priority, "as capacity allows"; #122 needs ≥ 5 overrides in the captures first), #105 (held until phases 1–3 are live and the gate passes), #111 (parked), everything HACS/community.

## Phases

| # | Sub-plan | ROADMAP | Branch | Touches | Forecast changes? | Size |
|---|---|---|---|---|---|---|
| 1 | `2026-11-14-audit-batch-1-appliance-free-actuals.md` | #117 | `fix/appliance-free-actuals` | `energy_forecast.py` | No. `mae_*`, bias and anomaly readings step once | S |
| 2 | `2026-11-14-audit-batch-2-block-daily-naive-metrics.md` | #114 | `feat/block-daily-accuracy` | new `accuracy.py`, `energy_forecast.py`, `backtest/runner.py`, `backtest/score.py` | No. Four new sensors | M |
| 3 | `2026-11-14-audit-batch-3-relative-cold-day-gate.md` | #118 | `feat/live-cold-gate` | new `backtest/live_gate.py`; local `scripts/` | No. Tooling only | S |
| 4 | `2026-11-14-audit-batch-4-phase2-reconstruction-past-tail.md` | #113 | `fix/phase2-reconstruction` | `model.py`, `energy_forecast.py` | Not while `physics_kwh` has 0 gain (verify by capture replay) | S–M |
| 5 | `2026-11-14-audit-batch-5-drop-feature-flag.md` | #120 | `feat/drop-feature-groups` | `const.py`, `model.py`, `energy_forecast.py`, `backtest/walkforward.py`, `backtest/wf_score.py` | No. Flag defaults off | S–M |
| 6 | `2026-11-14-audit-batch-6-retrain-age-timezone.md` | #116 | `fix/retrain-age-timezone` | `energy_forecast.py` | Retrain gates fire 1–2 h earlier (correct time) | S |

## Order and dependencies

```
1 (#117) ──► 2 (#114) ──► 3 (#118)        yardstick: strictly in this order
4 (#113) ──► 5 (#120)                      both edit model.py; run 5 after 4 is on train/next
6 (#116)                                   independent; any time
```

- 1 → 2: both edit the same stretch of `_update_sensors`; phase 2's numbers are only meaningful on phase 1's actuals.
- 2 → 3: phase 3 reads the `actuals_basis` marker phase 1 writes, and reuses the metric definitions phase 2 pins by test.
- 4 → 5: independent in logic, but both edit `model.py`. Sequential avoids a merge conflict a lower-tier session would have to resolve.
- Tracks {1, 2, 3}, {4, 5} and {6} can run in parallel sessions. Within a track, start a phase only after the previous one is merged into `train/next`.

## Shared rules for every phase

1. **Base branch: `train/next`** (not `dev`): the work depends on unshipped train content (#119 tooling, ROADMAP #117–#123) and on earlier phases of this batch. Never check out `train/next` in the main working tree; create a worktree:
   ```bash
   REPO=/home/jovyan/work/ha-energy-forecast
   git -C $REPO worktree add $REPO/.worktrees/<slug> -b <branch> train/next
   cd $REPO/.worktrees/<slug>
   ```
2. **Tests:** always `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -q`, never bare `python`. Run the full suite before the first change (baseline: 0 failed; about 1474 passed, 12 skipped on b03b19d) and after every task.
3. **Lint:** `ruff check . && ruff format --check .` must be clean before each commit (line length 120).
4. **TDD:** write the failing test, see it fail for the stated reason, implement, see it pass, commit.
5. **Commits:** one concise subject line, plus the co-author trailer your session specifies.
6. **Docs at the end of each phase:** CHANGELOG `[Unreleased]` (use the `/changelog` skill), ROADMAP status of the item, README when a sensor or config key is added, and `MEMORY.md` in the main checkout (`/home/jovyan/work/ha-energy-forecast/MEMORY.md`, gitignored, never `git add`).
7. **Finish = merge to the train, nothing more:** from `$REPO/.worktrees/train`, `git merge --no-ff <branch>`, run the full suite there, `git push origin train/next` (plain push, never force). No tag, no deploy, no merge to `dev` or `main`. CHANGELOG `[Unreleased]` conflicts: keep both entries, newest first. Any other conflict: stop and ask. If a push is blocked by the permission mode, hand the owner the exact command prefixed with `!`.
8. **Line numbers** in the sub-plans are from `train/next` b03b19d and will drift. Each anchor also quotes text to search for; trust the text.
9. **Stop and ask** if a baseline test fails, if an anchor's quoted text cannot be found, or if a step would change published forecast values where the plan says it must not.

## Decisions taken in this plan (D1–D8 confirmed by the owner, 2026-11-14)

| # | Decision | Why | Alternative |
|---|---|---|---|
| D1 | Phase 1 migrates the 30 days of already-stored actuals once (subtracting the cached sub-meter history) and writes `"actuals_basis": "target"` into `pred_history.json`. | Without it the 30-day readings mix both definitions for a month and the #118 gate could not be read before mid-December. | No migration; wait 30 days. |
| D2 | Phase 2 adds four sensors (`block_mae_7d/30d`, `daily_bias_7d/30d`); the naive guardrail and the relative bias are attributes, not extra sensors. | Keeps the entity count down; the roadmap asks for the numbers, not for entities. | Separate naive and relative sensors (8 more). |
| D3 | Phase 3 puts the gate in tracked, tested `backtest/live_gate.py`; the local `scripts/check_cold_onset_gate.py` only calls it. | `scripts/` is gitignored, so logic there cannot be reviewed or tested in CI. | Edit the local script only. |
| D4 | Phase 3's cold day is a daily mean ≤ 10 °C (`backtest.wf_score.COLD_C`), not the old script's < 13.5 °C. | The −12 %/+6 % band was derived from audit numbers computed with 10 °C. | Keep 13.5 °C (`--cold-below 13.5` is available). |
| D5 | Phase 4 makes `predict_intervals()` return `None` when a Phase 2 model has no physics baseline, instead of publishing residual-scale bands. | A missing band is visible; a band 5 kWh/day too low is not. | Fall back like `predict()` does. |
| D6 | Phase 5 "cold-day bias not higher" is read as `abs(arm) <= abs(base)`. | The roadmap wording is ambiguous for a signed number. | Signed comparison. |
| D7 | Phase 5 ships only the flag, the config key and the shadow tooling. The ≥ 14-day shadow and any flip are owner steps. | The holdout through 2026-11-12 is spent; new days must accumulate. | — |
| D8 | All six phases may ride one train. | None changes the published forecast with default config (phase 4 is checked by capture replay), so the "one forecast-changing step per window" rule is not triggered. | One train per phase. |

## After the batch (owner / PM steps, not sessions)

1. Ship the train (`/pm evening`). After the first hourly update, check that `pred_history.json` carries `"actuals_basis": "target"` and that the four new sensors have values.
2. Expect `mae_7d`/`mae_30d`, relative MAE and the anomaly sensor to step once (appliance energy leaves the actuals). Note the timestamp in ROADMAP so later comparisons do not span it unknowingly.
3. Re-read the A+B re-baseline with `python -m backtest.live_gate data --since "<anchor>"` once ≥ 14 cold days exist on the new basis. PASS releases #105 from hold.
4. Shadow #120: when the captures hold ≥ 14 target days after 2026-11-12, run the walk-forward in phase 5's last task and decide the flip.
5. Phase 2 of the physics hybrid stays off. Phase 4 only removes the crash; activation still needs the per-season check on the capture frame.

## Batch success criteria

- Full suite green on `train/next` after each merge; ruff clean.
- `backtest.capture_replay data/captures` still reports PASS on existing captures after phase 4 (run in the `-live` env).
- With default config, `feature_cols` of a freshly trained model is identical before and after phase 5.
- ROADMAP rows #113, #114, #116, #117, #118, #120 updated; CHANGELOG has one entry per phase.
