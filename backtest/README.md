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
- `train` rows re-run `train()` on a retrain capture, seeded from the previous retrain's saved model files. They then predict on the update made right after that retrain. PASS means a next-day total within 0.1 %.
- Updates taken while a retrain was running are skipped. The command reports only and always exits 0. A FAIL is a finding: some difference beyond the inputs, such as library versions (sidecars record them, and loading warns on a mismatch) or hidden state.

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
