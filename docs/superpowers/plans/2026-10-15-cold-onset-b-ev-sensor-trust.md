# Trust Wallbox-Sensor EV Subtraction — Implementation Plan (B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Days/hours whose EV load was subtracted exactly via `ev_charging_sensor` are no longer discarded from regime clustering, `likely_ev_hour` learning, or MAE tracking — only threshold-inferred EV rows (which can carry residue) still are.

**Out of scope (verified unaffected):** physics `_calibrate_base_load` (`physics.py:637-639`) and the UA fit (`physics.py:765-767`) exclude every EV hour by timestamp. That stays conservatively correct; switching them to `residual_ev_rows` would only be an optional data gain. No other `ev_df` consumer reads anything but `timestamp`/`gross_kwh`, and none writes it to CSV.

**Architecture:** Tag every EV row with its source (`ev_source` = `"sensor"` | `"threshold"`) inside the three split functions; one helper `residual_ev_rows()` returns only threshold rows (or all rows if untagged = conservative). The three consumers call the helper. Training rows themselves are unchanged (already kept; ±1h padding already threshold-only since 2026-08-28).

**Tech Stack:** Python 3.13, pandas, pytest.

**Spec:** index row B; user request 2026-10-15 ("with the precise EV sensor this shouldn't be needed anymore").

**Base branch:** `dev` → `fix/ev-sensor-trust` (independent of A; if A merged first, rebase).

## Global Constraints
- Tests via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`.
- Users without `ev_charging_sensor` (threshold-only) must see zero behaviour change.
- Untagged `ev_df` (e.g. from old callers/tests) is treated as threshold-sourced.
- Evidence gate already run (scratch `bt_regime.py`, R1): Sept fold MAE −3.9 %, cold fold +0.2 % → GO.

## Review Focus
1. Hybrid split (sensor coverage starts 2026-07-27): EV days before coverage still excluded from clustering, after coverage not.
2. Live update path uses `split_ev_charging_from_sensor` (not hybrid) → MAE exclusion set becomes empty; threshold fallback (sensor fetch empty) still excludes ±1h.
3. `ev_df` empty / None → helper returns it unchanged, no KeyError.
4. Extra `ev_source` column must not leak into anything that sums/concats numeric columns (EV sensor publishing uses `gross_kwh` only).
5. Clustering's "<14 non-EV days → include all" fallback becomes rarer — no assertion depends on it.

---

### Task 1: Tag EV source + `residual_ev_rows` helper

**Files:**
- Modify: `apps/energy_forecast/ha_data.py` — `split_ev_charging` (~L715), `split_ev_charging_from_sensor` (~L747), new helper after `split_ev_charging_hybrid`. Rationale: the split functions are the only place that knows whether a row's EV kWh is exact (sensor) or inferred (threshold); tagging there makes the distinction available to every consumer.
- Test: `tests/test_ha_data.py` (new class `TestEvSource`)

**Interfaces:**
- Produces: `ev_df["ev_source"]` ∈ {`"threshold"`, `"sensor"`} on every frame returned by the three split functions.
- Produces: `ha_data.residual_ev_rows(ev_df: pd.DataFrame | None) -> pd.DataFrame | None`

- [ ] **Step 1: Failing tests**

```python
class TestEvSource:
    def _energy(self):
        ts = pd.date_range("2026-07-26 10:00", periods=4, freq="h").append(pd.date_range("2026-07-28 10:00", periods=2, freq="h"))
        return pd.DataFrame({"timestamp": ts, "gross_kwh": [9.5, 0.4, 0.3, 0.3, 3.0, 0.5]})

    def test_threshold_rows_tagged(self):
        _, ev = ha_data.split_ev_charging(self._energy(), 7.0, charger_kw=9.0)
        assert set(ev["ev_source"]) == {"threshold"}

    def test_sensor_rows_tagged(self):
        evk = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-28 10:00"]), "kwh": [2.5]})
        _, ev = ha_data.split_ev_charging_from_sensor(self._energy(), evk)
        assert list(ev["ev_source"]) == ["sensor"]

    def test_hybrid_mixes_sources_by_coverage(self):
        evk = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-27 10:00", "2026-07-28 10:00"]), "kwh": [0.0, 2.5]})
        _, ev = ha_data.split_ev_charging_hybrid(self._energy(), evk, 7.0, charger_kw=9.0)
        src = ev.set_index("timestamp")["ev_source"]
        assert src[pd.Timestamp("2026-07-26 10:00")] == "threshold"
        assert src[pd.Timestamp("2026-07-28 10:00")] == "sensor"

    def test_residual_ev_rows_keeps_threshold_only(self):
        ev = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-26 10:00", "2026-07-28 10:00"]),
                           "gross_kwh": [9.5, 2.5], "ev_source": ["threshold", "sensor"]})
        out = ha_data.residual_ev_rows(ev)
        assert list(out["timestamp"]) == [pd.Timestamp("2026-07-26 10:00")]

    def test_residual_ev_rows_untagged_is_conservative(self):
        ev = pd.DataFrame({"timestamp": pd.to_datetime(["2026-07-28 10:00"]), "gross_kwh": [2.5]})
        assert len(ha_data.residual_ev_rows(ev)) == 1

    def test_residual_ev_rows_none_and_empty(self):
        assert ha_data.residual_ev_rows(None) is None
        empty = pd.DataFrame(columns=["timestamp", "gross_kwh"])
        assert ha_data.residual_ev_rows(empty).empty
```

- [ ] **Step 2: Run** `…/bin/python -m pytest tests/test_ha_data.py::TestEvSource -v` → FAIL (`KeyError: 'ev_source'`, `AttributeError: residual_ev_rows`).

- [ ] **Step 3: Implement**

In `split_ev_charging`, after `ev_df = df[ev_mask].copy()`:
```python
    ev_df["ev_source"] = "threshold"
```
In `split_ev_charging_from_sensor`, after `ev_df["gross_kwh"] = ev_kwh[ev_mask]`:
```python
    ev_df["ev_source"] = "sensor"
```
(`split_ev_charging_hybrid` concatenates both, so its rows inherit the tags.)

New helper after `split_ev_charging_hybrid`:
```python
def residual_ev_rows(ev_df: pd.DataFrame | None) -> pd.DataFrame | None:
    """EV rows whose subtraction was inferred (threshold), i.e. whose baseline may still
    carry EV residue. Wallbox-sensor rows are exact and returned as clean. Untagged
    frames are treated as threshold-sourced (conservative)."""
    if ev_df is None or ev_df.empty or "ev_source" not in ev_df.columns:
        return ev_df
    return ev_df[ev_df["ev_source"] == "threshold"]
```

- [ ] **Step 4: Run** `…/bin/python -m pytest tests/test_ha_data.py -v` → all PASS (existing EV split tests ignore the new column; if one asserts exact columns, add `"ev_source"` to its expectation — that's the intended contract change).

- [ ] **Step 5: Commit** `git commit -m "feat: tag EV rows with ev_source (sensor vs threshold)"`

---

### Task 2: Consumers use `residual_ev_rows` (clustering, likely_ev_hour, MAE)

**Files:**
- Modify: `apps/energy_forecast/model.py` — import (~L52), L491, L576. Rationale: the two training-side consumers that currently treat every EV row as contamination (EV-hour learning, regime clustering).
- Modify: `apps/energy_forecast/energy_forecast.py` — new module function near `_fetch_correction_dfs` (~L3365); replace L1950–1956 block, net sensor EV hours in the `_actuals_history` loop (~L2181–2188). Rationale: the MAE/anomaly consumer; it stores gross actuals, so sensor hours must be netted, not just un-excluded.
- Test: `tests/test_model.py`, `tests/test_energy_forecast.py`

**Interfaces:**
- Consumes: `ha_data.residual_ev_rows`, `ev_source` tag (Task 1).
- Produces: `energy_forecast._ev_mae_excluded_hours(ev_detected: pd.DataFrame | None) -> set[pd.Timestamp]`
- Produces: `energy_forecast._ev_sensor_kwh_by_hour(ev_detected: pd.DataFrame | None) -> dict[pd.Timestamp, float]` — wallbox kWh per hour for sensor-tagged rows. **Why it's needed:** `_actuals_history` is filled from the **gross** `full_actuals` (`energy_forecast.py:2172-2188`, "pre-split"), while `_pred_history` holds EV-free baseline predictions. Today EV hours are simply excluded. Once sensor EV hours are kept, they must be stored net of the exact wallbox kWh, or wallbox load would be scored against EV-free predictions and inflate `mae_*` and the anomaly sigma (review finding 2026-10-15).

- [ ] **Step 1: Failing tests**

`tests/test_energy_forecast.py` (add `_ev_mae_excluded_hours` to the existing `from energy_forecast.energy_forecast import (...)` block):
```python
class TestEvMaeExcludedHours:
    def test_sensor_rows_not_excluded(self):
        ev = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-14 12:00"]), "gross_kwh": [3.0], "ev_source": ["sensor"]})
        assert _ev_mae_excluded_hours(ev) == set()

    def test_threshold_rows_excluded_with_neighbours(self):
        ev = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-14 12:00"]), "gross_kwh": [9.5], "ev_source": ["threshold"]})
        assert _ev_mae_excluded_hours(ev) == {
            pd.Timestamp("2026-10-14 11:00"), pd.Timestamp("2026-10-14 12:00"), pd.Timestamp("2026-10-14 13:00")}

    def test_none_and_empty(self):
        assert _ev_mae_excluded_hours(None) == set()
        assert _ev_mae_excluded_hours(pd.DataFrame(columns=["timestamp"])) == set()


class TestEvSensorKwhByHour:
    def test_only_sensor_rows_floored_to_hour(self):
        ev = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-14 12:00", "2026-10-14 13:00"]),
                           "gross_kwh": [3.0, 9.5], "ev_source": ["sensor", "threshold"]})
        assert _ev_sensor_kwh_by_hour(ev) == {pd.Timestamp("2026-10-14 12:00"): 3.0}

    def test_none_empty_untagged(self):
        assert _ev_sensor_kwh_by_hour(None) == {}
        assert _ev_sensor_kwh_by_hour(pd.DataFrame(columns=["timestamp", "gross_kwh"])) == {}
        untagged = pd.DataFrame({"timestamp": pd.to_datetime(["2026-10-14 12:00"]), "gross_kwh": [3.0]})
        assert _ev_sensor_kwh_by_hour(untagged) == {}   # untagged = threshold semantics → excluded, not netted
```
(Add `_ev_mae_excluded_hours, _ev_sensor_kwh_by_hour` to the existing `from energy_forecast.energy_forecast import (...)` block at `tests/test_energy_forecast.py:20`, in alphabetical order — ruff isort is on.)

Create **`tests/_frames.py`** (shared synthetic frames, also used by Plans C/D; skip if another plan already created it):
```python
"""Shared synthetic training frames for model tests (Plans 2026-10-15 B/C/D)."""
import numpy as np
import pandas as pd


def train_frames(n: int = 960) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h")
    energy = pd.DataFrame({"timestamp": ts, "gross_kwh": rng.uniform(0.5, 5.0, size=n)})
    weather = pd.DataFrame({"timestamp": ts, "temp_c": rng.uniform(-5, 25, size=n), "precipitation_mm": 0.0,
                            "sunshine_min": 30.0, "wind_kmh": 10.0, "cloud_cover_pct": 50.0, "direct_radiation_wm2": 100.0})
    return energy, weather
```
`tests/test_model.py` — capture what clustering and the EV-hour learner receive (add `from tests._frames import train_frames` to the imports):
```python
class TestEvSourceInTraining:
    def test_only_threshold_ev_rows_reach_clustering_and_ev_hours(self, tmp_path, monkeypatch):
        from energy_forecast import clustering
        from energy_forecast import model as model_mod

        seen: dict = {}
        real_fit = clustering.DailyProfileClusterer.fit
        real_how = model_mod._compute_likely_ev_hours

        def spy_fit(self, df, ev_day_dates=None):
            seen["ev_day_dates"] = set(ev_day_dates or ())
            return real_fit(self, df, ev_day_dates=ev_day_dates)

        def spy_how(base, ev):
            seen["ev_rows"] = 0 if ev is None else len(ev)
            return real_how(base, ev)

        monkeypatch.setattr(clustering.DailyProfileClusterer, "fit", spy_fit)
        monkeypatch.setattr(model_mod, "_compute_likely_ev_hours", spy_how)
        energy_df, weather_df = train_frames()
        ev_ts = energy_df["timestamp"].iloc[[300, 301, 700]]
        ev_df = pd.DataFrame({"timestamp": ev_ts.values, "gross_kwh": [2.0, 2.0, 9.5],
                              "ev_source": ["sensor", "sensor", "threshold"]})
        m = model_mod.EnergyForecastModel(tmp_path, timezone="Europe/Zurich")
        m.train(energy_df, weather_df, outdoor_df=None, weight_halflife_days=0, ev_df=ev_df,
                enable_regimes=True, regime_count=2)
        assert seen["ev_day_dates"] == {pd.Timestamp(ev_ts.iloc[2]).date()}
        assert seen["ev_rows"] == 1
```
If `clustering.SKLEARN_AVAILABLE` is False in the env, skip like `test_auto_k_selects_valid_k` does.

- [ ] **Step 2: Run both** → FAIL (`ImportError: _ev_mae_excluded_hours / _ev_sensor_kwh_by_hour`; clustering sees 2 dates, EV learner 3 rows).

- [ ] **Step 3: Implement**

`model.py` imports (after `from . import clustering`):
```python
from .ha_data import residual_ev_rows
```
L491:
```python
        likely_ev_hours = _compute_likely_ev_hours(energy_df, residual_ev_rows(ev_df))
```
L576:
```python
            _ev_resid = residual_ev_rows(ev_df)
            ev_day_dates: set = (
                set(_ev_resid["timestamp"].dt.date) if _ev_resid is not None and len(_ev_resid) > 0 else set()
            )
```
`energy_forecast.py` new module-level function:
```python
def _ev_mae_excluded_hours(ev_detected: pd.DataFrame | None) -> set:
    """Hours dropped from the MAE/actuals history because their EV subtraction may leave residue.

    Only threshold-inferred EV hours (plus ±1h ramp) qualify; wallbox-sensor hours are
    subtracted exactly and are scored like any other hour.
    """
    resid = ha_data.residual_ev_rows(ev_detected)
    if resid is None or resid.empty:
        return set()
    hours = set(pd.to_datetime(resid["timestamp"]).dt.floor("1h"))
    return hours | {ts + pd.Timedelta(hours=d) for ts in hours for d in (-1, 1)}
```
Second module-level function next to it:
```python
def _ev_sensor_kwh_by_hour(ev_detected: pd.DataFrame | None) -> dict:
    """Wallbox kWh per hour for sensor-tagged EV rows — used to store MAE actuals net of EV."""
    if ev_detected is None or ev_detected.empty or "ev_source" not in ev_detected.columns:
        return {}
    s = ev_detected[ev_detected["ev_source"] == "sensor"]
    return dict(zip(pd.to_datetime(s["timestamp"]).dt.floor("1h"), s["gross_kwh"].astype(float)))
```
Replace L1950–1956 (`ev_hour_set` … `ev_mae_excluded: set = ev_hour_set | _ev_adjacent`) with the following (`ev_hour_set`/`_ev_adjacent` have no other uses, verified by review):
```python
        ev_mae_excluded: set = _ev_mae_excluded_hours(ev_detected)
        ev_sensor_kwh: dict = _ev_sensor_kwh_by_hour(ev_detected)
```
In the `_actuals_history` population loop (~L2181-2188), store sensor EV hours net of the wallbox:
```python
                if _ts not in ev_mae_excluded:
                    self._actuals_history[_ts] = max(0.0, _kwh - ev_sensor_kwh.get(_ts, 0.0))
```
Update the comment above the loop ("Use full_actuals (pre-split …)"): sensor EV hours are netted, threshold EV hours ±1 h are excluded. The loop is keep-last over every run's `full_actuals` window, so a wallbox reading that arrives an hour late still corrects the stored value on the next run. `_actuals_for_retrain` (~L2196) already uses the split `recent_actuals` and needs no change.

- [ ] **Step 4: Full suite + ruff** → all PASS.

- [ ] **Step 5: Commit** `git commit -m "fix: keep wallbox-sensor EV days in clustering, EV-hour learning and MAE tracking"`

---

### Task 3: Docs + finalize

- [ ] `@changelog-writer`: "Changed: with `ev_charging_sensor` configured, EV-charging days are no longer excluded from regime clustering and EV hours are no longer dropped from MAE tracking — the wallbox subtraction is exact. Threshold-only setups unchanged." README: if it documents the EV-day exclusion / MAE exclusion, update that sentence.
- [ ] deploy-agent Workflow A → next alpha. Verify in the AppDaemon log after retrain: `Clustering: excluding N EV days` shows N ≤ number of pre-2026-07-27 threshold EV days (≈ 39 × decay-window share), not Oct days.
- [ ] After deploy, verify on the next EV-charging day: `pred_history.json` `actuals` for charging hours ≈ (grid + PV − battery net) − wallbox, i.e. household load in the same range as neighbouring hours, not 3–11 kWh. If any charging hour shows gross EV load, roll back (deploy-agent) — the MAE sensors would otherwise inflate.
- [ ] Memory: note B deployed in the cold-onset memory file. Expect `mae_*` to shift slightly (more pairs: EV-day hours are now scored, netted) — note this so it isn't misread as a regression.
