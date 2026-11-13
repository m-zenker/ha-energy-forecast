# Backtest kit

Replays the real `EnergyForecastModel.train()` / `predict()` under declarative **arms** and compares each arm with a
`base` arm. Use it to test a feature, transform or train-kwarg change before shipping it. It never touches HA, and
`scripts/deploy.py` doesn't upload it.

Data comes from `data/`, pulled from HA with `scripts/pull_ha_data.py`. The archive weather cache is
`data/regime_backtest_weather.pkl`.

## Writing an experiment

Copy `backtest/experiments/temp_ablation.py` to `backtest/experiments/<name>.py` and edit `ARMS`:

```python
from backtest.arms import Arm, climatology

ARMS = [
    Arm("base"),                                    # ARMS[0] is the reference and must be named "base"
    Arm("-temp_c", drop=("temp_c",)),               # drop static _FEATURES_BASE names
    Arm("clim", weather=climatology),               # transform the weather frame (train + predict)
    Arm("hl30", train_kwargs={"weight_halflife_days": 30.0}),  # override train() kwargs
]
# optional: FOLDS = [...]  (default backtest.data.DEFAULT_FOLDS), ORIGINS = (6, 18)
```

Train kwargs default to what live `_retrain()` passes, read from `data/apps.yaml`. Only static `_FEATURES_BASE`
names can be dropped. A drop that `train()` re-adds (`lag_*h`, `physics_kwh`, sub-sensor lags) raises an error
rather than silently doing nothing.

## Running

```bash
/home/jovyan/my_envs/ha-energy-forecast/bin/python -m backtest <name> --repro --seeds 1 2
```

- `--seeds 1 2` adds `base@s1`, `base@s2`: base with other LightGBM seeds. Their spread is the noise floor.
- `--repro` replays base at 25 h lead over the last 28 days, in weekly retrains, and compares it with live
  `pred_history.json`. It only reports numbers, with no pass/fail.
- `--smoke` runs the last fold for 2 days. `--arms`, `--folds` and `--no-fetch` narrow the run.
- Rows are written to `data/backtest/<name>_<stamp>.pkl` after every arm.

## Replaying live captures

With `capture_inputs: true` on the live app (see the main README), pull `captures/` (the local `scripts/pull_ha_data.py` copies it to `data/captures/`), then:

```bash
/home/jovyan/my_envs/ha-energy-forecast/bin/python -m backtest.capture_replay data/captures
```

- `predict` rows re-run feature building plus `predict()` on a captured update, against the model files saved with the retrain that served it. PASS means a max absolute difference of at most 1e-6 kWh/h from what live computed.
- `train` rows re-run `train()` on a retrain capture, seeded from the model dir as it was before that retrain (`<id>.pre/`, else the previous retrain's `.ref/`, else SKIP). They then predict on the update made right after that retrain. PASS means a next-day total within 0.1 %.
- Updates taken while a retrain was running are skipped. The command reports only and always exits 0. A FAIL is a finding: some difference beyond the inputs, such as library versions (sidecars record them, and loading warns on a mismatch) or hidden state.

## Walk-forward on a capture (live-equivalent frame)

The lean frame above has no climate, tank or sub-meter inputs and no intervals. For anything that touches those,
use the capture frame: one retrain capture holds the full live history, so truncating it at a cutoff gives the
inputs live had then.

```bash
PYTHONPATH=apps:. /home/jovyan/my_envs/ha-energy-forecast-live/bin/python -m backtest.walkforward \
    data/captures --start 2026-10-01 --end 2026-11-12 --arms base jit droptp --out data/backtest/wf
```

- It runs the real `train()` / `predict()` / `predict_intervals()`: daily retrain at 18:00 seeded from the previous
  model dir, four origins a day, 48 h each. Arms are listed in `backtest/walkforward.py` (`ARMS`); add a `WfArm`
  there for a feature drop, a `train()` kwarg, another cadence or live's 3-day weather tail.
- **Always run `jit` with `base`.** It is the same model retrained one hour earlier. Its distance from `base` is
  the noise floor: about ±1 % on hourly MAE, ±2 % on 3 h blocks, and up to ±13 % on daily |bias|.
- **Burn-in.** The first model of a run is unseeded and differs from a seeded one by about as much as the jitter
  arm. Start a week before the days you want to score, and compare arms only on runs with the same start.
- Needs the env that matches the HA host (`ha-energy-forecast-live`), like every capture replay. About 17 s per
  simulated day on one core. Run arms in parallel as separate processes; `--n-jobs 1` (default) keeps LightGBM
  from oversubscribing.
- Checked against live on 2026-11-13: serve inputs within 0.007 kWh/h of captured live forecasts; a retrain at the
  capture time is identical to the live model.

Score the rows with `backtest.wf_score`:

```python
from backtest import wf_score as ws
rows = ws.attach_actuals(rows, frame.actuals(), daily_temp)      # daily_temp: mean temp_c per day
table = ws.day_table(ws.add_naive(ws.day_ahead(rows)))            # newest forecast with >= 24 h lead
ws.summarize(table); ws.paired(table, "droptp", metric="mae3"); ws.next_day_swing(rows)
```

- **Judge on `mae3` (3 h-block totals), `daily_abs` and `bias`; `mae` (hourly) is secondary.** Hourly MAE rewards
  under-forecasting, and `ha-energy-manager` consumes 3 h blocks.
- `cold_gate()` applies the ROADMAP #118 rule: pooled cold-day bias within −12 % to +6 % of consumption over at
  least 14 cold days, else INCONCLUSIVE.
- `naive` is the same-hour mean of days −2..−8, the baseline a change has to beat.
- `backtest.audit_intervals.report(rows)` gives interval coverage as shipped, recalibrated on served errors, and
  for the summed block and daily bands.
- `python -m backtest.audit_features data/captures` pairs every feature as served with the value later trained on,
  by lead. Run it after any change to feature building.

Limits: perfect weather (measured cost of real forecast weather: 1–2 %); history is today's view of the past; the
physics state is the capture's at every cutoff; days before a sensor existed have no such inputs, as live had none.

## Reading the summary

- `dMAE%` / `95% CI` is the paired difference against base. The CI resamples (fold, ISO-week) blocks, not days.
- `worse` is the number of days the arm was worse. `folds` is how many folds agree with the sign of the mean.
- `d|bias|%` is the same comparison on absolute daily bias. It caught arms that looked good on MAE before.
- `~` marks an arm inside the seed noise floor; treat it as "no effect". The floor is computed **per segment**:
  seed spread is larger in the thin WARM segment.
- The CI treats each fold's trained model as fixed, so it **excludes training randomness**. Seed-only arms can get
  CIs that exclude 0. Count an effect only when the CI excludes 0 **and** the arm is outside the floor.
- `bias` / `|bias|` are kWh/day; `mae` is kWh/h.
- The ALL / COLD / WARM segments split on a daily mean temperature of ≤ 10 °C.

## Accepted risks

- **Perfect (archive) weather.** Weather-feature gains are upper bounds, and noise-robustness transforms are scored
  unfairly. There is no day-ahead mode yet.
- **Lean frame.** There are no physics, climate, DHW or sub-meter inputs. Relative comparisons are invalid for
  features or kwargs that overlap them: the temperature group, heating degrees, heating/DHW buffer and `regime_kwh`.
  A live-equivalent frame is the next plan.
- **Constant features in the lean frame.** Dropping one of these gives "no effect" because it carries no signal
  here, not because the feature is useless live: `heating_active`, `dhw_buffer_temp`, `dhw_pressure`, `is_away`,
  `people_home`, `thermal_pressure*`, `infiltration_pressure`, and `regime_kwh` when regimes are off. Nothing
  warns about this.
- **Drops leave derived information.** Dropping a column doesn't remove information computed from it. For example,
  `lag_*_tgated` stays discounted by temperature deltas after `-T`. For a truly "without X" arm, use a `weather`
  transform.
- **One train per fold.** The replay model is older than live's weekly retrain. This is equal across arms, so it
  only affects the absolute numbers.
- **Fold reuse.** Repeated experiments on the same 4 folds slowly overfit them. See the holdout rule below.
- **The WARM segment is thin,** mostly Oct26. Check `days` before reading it.

## Holdout rule

After a ship/reject decision made on these folds, score the next ≥ 14 new days once, as a holdout, before treating
the result as settled.
