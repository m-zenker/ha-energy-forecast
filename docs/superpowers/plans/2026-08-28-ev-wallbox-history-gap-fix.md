# EV Wallbox History Gap + Adjacent-Hour Padding Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Base branch:** `dev` (create `fix/ev-wallbox-history-gap` off `dev` via `/feat fix/ev-wallbox-history-gap ha-energy-forecast` before starting Task 1).

**Goal:** Fix two EV-detection regressions introduced by switching from threshold-based EV inference to the wallbox `ev_charging_sensor`: (A) EV sessions from before the wallbox sensor existed are silently un-excluded from training, and (B) the ±1h adjacent-hour padding — sized for the old flat-charger tapering estimate — over-trims long solar-tracking wallbox sessions, pushing most post-wallbox EV days below clustering's 18h/day completeness floor.

**Architecture:** Both fixes live entirely in `apps/energy_forecast/ha_data.py` (new pure functions) and `apps/energy_forecast/energy_forecast.py::_retrain()` (wiring). No change to `clustering.py`, `model.py`, or the live/recent-actuals path (`_update_sensors()`) — its EV lookback window (400h tail rows, `ha_data.py:490`) never reaches back before the wallbox's installation date, so it isn't affected by (A), and it isn't in scope for (B) per the diagnosis below.

**Tech Stack:** Python 3.13, pandas, pytest. Dedicated env: `/home/jovyan/my_envs/ha-energy-forecast`.

**Spec:** No separate spec doc — this plan is derived directly from a debugging session; the diagnosis is captured in the "Root cause" sections of Task 1 and Task 3 below, each with the exact evidence that established it.

## Global Constraints

- Run all tests through `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v` — never bare `python`.
- `pytest.warns(None)` is invalid (pytest ≥7); omit the context manager to assert no warnings.
- Follow existing test conventions in `tests/test_ha_data.py` (small hand-built DataFrames, `abs(x - y) < 1e-6` for float comparisons) and `tests/test_energy_forecast.py` (`_FakeRetrain` stub + `monkeypatch.setattr(ha_data_mod, ...)` + inspect `stub._ml_model.train.call_args.args[0]`).
- No behavior change to the three existing fallback branches in `_retrain()`'s EV block (no sensor configured / sensor fetch raised / sensor returned empty) — only the "sensor returned non-empty history" branch changes.
- Do not touch `_update_sensors()`'s EV logic (recent actuals / blended actuals) — verified out of scope (see Architecture above).

---

### Task 1: `ev_sensor_coverage()` — coverage-window helper

**Files:**
- Modify: `apps/energy_forecast/ha_data.py` (insert after `split_ev_charging_from_sensor`, currently ending at line 753)
- Test: `tests/test_ha_data.py` (insert after `TestSplitEvChargingFromSensor`, currently ending at line ~730)

**Interfaces:**
- Produces: `ev_sensor_coverage(ev_kwh_df: pd.DataFrame | None) -> tuple[pd.Timestamp, pd.Timestamp] | None` — used by Task 2's `split_ev_charging_hybrid()` and by Task 4's padding-skip logic in `energy_forecast.py`.

**Root cause this supports:** `fetch_sub_sensor_history()`'s on-disk cache for `ev_charging_sensor` only ever contains rows from whenever that entity was first configured onward — it never backfills earlier dates. `split_ev_charging_from_sensor()`'s left-join `fillna(0.0)` treats any `energy_df` row outside that cache's range as "0 kWh charged", silently un-flagging genuine pre-wallbox EV sessions. Confirmed against the live `data/` pull: `energy_history.csv` has 38 distinct dates between 2025-10-12 and 2026-07-07 with the classic 7–13 kWh/h evening/night spike signature of the old flat charger, none of which are in `ev_se_one_ev_charger_..._total_energy.csv` (which only starts 2026-07-27) — all 38 currently train into the model as ordinary baseline load.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ha_data.py — add after TestSplitEvChargingFromSensor (after its last test, ~line 730)


# ── ev_sensor_coverage ───────────────────────────────────────────────────────


class TestEvSensorCoverage:
    def test_returns_floored_min_and_max(self):
        df = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-03-10 03:37"), pd.Timestamp("2026-03-12 21:05")],
                "kwh": [1.0, 2.0],
            }
        )
        start, end = ha_data.ev_sensor_coverage(df)
        assert start == pd.Timestamp("2026-03-10 03:00")
        assert end == pd.Timestamp("2026-03-12 21:00")

    def test_empty_df_returns_none(self):
        assert ha_data.ev_sensor_coverage(pd.DataFrame(columns=["timestamp", "kwh"])) is None

    def test_none_input_returns_none(self):
        assert ha_data.ev_sensor_coverage(None) is None

    def test_single_row_start_equals_end(self):
        df = pd.DataFrame({"timestamp": [pd.Timestamp("2026-03-10 03:00")], "kwh": [1.0]})
        start, end = ha_data.ev_sensor_coverage(df)
        assert start == end == pd.Timestamp("2026-03-10 03:00")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py::TestEvSensorCoverage -v`
Expected: FAIL with `AttributeError: module 'energy_forecast.ha_data' has no attribute 'ev_sensor_coverage'`

- [ ] **Step 3: Implement `ev_sensor_coverage()`**

```python
# apps/energy_forecast/ha_data.py — insert after split_ev_charging_from_sensor (after line 753, before _merge_sub_sensor_frames)


def ev_sensor_coverage(ev_kwh_df: pd.DataFrame | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """Return the (start, end) hour-range a wallbox kWh cache actually covers.

    Both bounds are floored to the hour, matching how split_ev_charging_from_sensor
    aligns timestamps. Returns None when *ev_kwh_df* is None or empty — callers
    should then treat every row as outside coverage (fall back to threshold
    detection everywhere), since the sensor has no data to be a source of truth for.

    fetch_sub_sensor_history()'s on-disk cache only ever contains rows from
    whenever the sensor entity was first configured onward — it never backfills
    dates before that. This coverage window is how callers know which part of a
    longer energy_df the sensor can actually speak for.
    """
    if ev_kwh_df is None or ev_kwh_df.empty:
        return None
    ts = pd.to_datetime(ev_kwh_df["timestamp"]).dt.floor("1h")
    return ts.min(), ts.max()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py::TestEvSensorCoverage -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add apps/energy_forecast/ha_data.py tests/test_ha_data.py
git commit -m "feat: add ev_sensor_coverage() helper for wallbox history-gap fix"
```

---

### Task 2: `split_ev_charging_hybrid()` — per-row threshold/sensor split

**Files:**
- Modify: `apps/energy_forecast/ha_data.py` (insert immediately after Task 1's `ev_sensor_coverage()`)
- Test: `tests/test_ha_data.py` (insert after `TestEvSensorCoverage`)

**Interfaces:**
- Consumes: `ev_sensor_coverage(ev_kwh_df)` (Task 1), `split_ev_charging(df, threshold_kwh, charger_kw=9.0)` (existing), `split_ev_charging_from_sensor(energy_df, ev_kwh_df)` (existing).
- Produces: `split_ev_charging_hybrid(energy_df: pd.DataFrame, ev_kwh_df: pd.DataFrame, threshold_kwh: float, charger_kw: float = 9.0) -> tuple[pd.DataFrame, pd.DataFrame]` — same `(baseline_df, ev_df)` contract as its two siblings. Used by Task 3's `_retrain()` wiring.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ha_data.py — add after TestEvSensorCoverage


# ── split_ev_charging_hybrid ─────────────────────────────────────────────────


class TestSplitEvChargingHybrid:
    def _energy_df(self) -> pd.DataFrame:
        """6 hourly rows, 2026-03-10 00:00..05:00. Row 1 (01:00) is a pre-coverage
        threshold-shaped spike; row 3 (03:00) is a sub-threshold, in-coverage hour
        the wallbox sensor reports as charging."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-10 00:00", periods=6, freq="1h"),
                "gross_kwh": [3.0, 10.0, 3.0, 3.0, 2.0, 3.0],
            }
        )

    def _ev_kwh_df(self) -> pd.DataFrame:
        """Wallbox coverage starts 03:00 — the sensor was only installed then."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-03-10 03:00", periods=3, freq="1h"),
                "kwh": [2.5, 0.0, 0.0],
            }
        )

    def test_pre_coverage_hour_uses_threshold_detection(self):
        """A spike before the wallbox existed must still be caught by the threshold."""
        baseline, ev = ha_data.split_ev_charging_hybrid(
            self._energy_df(), self._ev_kwh_df(), threshold_kwh=4.5, charger_kw=9.0
        )
        row = baseline[baseline["timestamp"] == pd.Timestamp("2026-03-10 01:00")].iloc[0]
        assert abs(row["gross_kwh"] - 1.0) < 1e-6  # 10.0 - 9.0 charger_kw
        assert pd.Timestamp("2026-03-10 01:00") in set(ev["timestamp"])

    def test_in_coverage_subthreshold_hour_uses_sensor_detection(self):
        """A sub-threshold hour within coverage must still be caught via the wallbox reading."""
        baseline, ev = ha_data.split_ev_charging_hybrid(
            self._energy_df(), self._ev_kwh_df(), threshold_kwh=4.5, charger_kw=9.0
        )
        row = baseline[baseline["timestamp"] == pd.Timestamp("2026-03-10 03:00")].iloc[0]
        assert abs(row["gross_kwh"] - 0.5) < 1e-6  # 3.0 - 2.5 (sensor kWh, not charger_kw)
        ev_row = ev[ev["timestamp"] == pd.Timestamp("2026-03-10 03:00")].iloc[0]
        assert abs(ev_row["gross_kwh"] - 2.5) < 1e-6

    def test_no_double_counting_or_dropped_rows(self):
        baseline, _ = ha_data.split_ev_charging_hybrid(
            self._energy_df(), self._ev_kwh_df(), threshold_kwh=4.5, charger_kw=9.0
        )
        assert len(baseline) == 6
        assert list(baseline["timestamp"]) == list(self._energy_df()["timestamp"])

    def test_empty_ev_kwh_df_falls_back_to_pure_threshold(self):
        """No sensor data at all -> identical result to calling split_ev_charging directly."""
        energy_df = self._energy_df()
        expected_baseline, expected_ev = ha_data.split_ev_charging(energy_df, threshold_kwh=4.5, charger_kw=9.0)
        baseline, ev = ha_data.split_ev_charging_hybrid(
            energy_df, pd.DataFrame(columns=["timestamp", "kwh"]), threshold_kwh=4.5, charger_kw=9.0
        )
        pd.testing.assert_frame_equal(
            baseline.reset_index(drop=True), expected_baseline.reset_index(drop=True)
        )
        assert len(ev) == len(expected_ev)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py::TestSplitEvChargingHybrid -v`
Expected: FAIL with `AttributeError: module 'energy_forecast.ha_data' has no attribute 'split_ev_charging_hybrid'`

- [ ] **Step 3: Implement `split_ev_charging_hybrid()`**

```python
# apps/energy_forecast/ha_data.py — insert immediately after ev_sensor_coverage()


def split_ev_charging_hybrid(
    energy_df: pd.DataFrame,
    ev_kwh_df: pd.DataFrame,
    threshold_kwh: float,
    charger_kw: float = 9.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split EV charging using the wallbox sensor where it has coverage, threshold
    detection everywhere else. Same (baseline_df, ev_df) contract as
    split_ev_charging() / split_ev_charging_from_sensor().

    Rationale: fetch_sub_sensor_history()'s cache for ev_charging_sensor only
    covers dates from whenever that entity was configured onward.
    split_ev_charging_from_sensor() alone would silently treat every row before
    that as "0 kWh charged" (its left-join fillna(0.0)), un-excluding real
    pre-wallbox EV sessions from training. This function keeps the sensor as the
    source of truth inside its coverage window (exact per-hour kWh — catches
    variable-power solar-surplus charging the threshold would miss) and falls
    back to threshold detection for rows outside it.
    """
    coverage = ev_sensor_coverage(ev_kwh_df)
    if coverage is None:
        return split_ev_charging(energy_df, threshold_kwh, charger_kw=charger_kw)

    cov_start, cov_end = coverage
    ts_floor = pd.to_datetime(energy_df["timestamp"]).dt.floor("1h")
    in_coverage = (ts_floor >= cov_start) & (ts_floor <= cov_end)

    sensor_baseline, sensor_ev = split_ev_charging_from_sensor(energy_df[in_coverage], ev_kwh_df)
    threshold_baseline, threshold_ev = split_ev_charging(
        energy_df[~in_coverage], threshold_kwh, charger_kw=charger_kw
    )

    baseline_df = (
        pd.concat([sensor_baseline, threshold_baseline]).sort_values("timestamp").reset_index(drop=True)
    )
    ev_df = pd.concat([sensor_ev, threshold_ev]).sort_values("timestamp").reset_index(drop=True)
    return baseline_df, ev_df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py::TestSplitEvChargingHybrid -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the full ha_data test file to check for regressions**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_ha_data.py -v`
Expected: PASS (all tests, including the pre-existing `TestSplitEvCharging` / `TestSplitEvChargingFromSensor` classes)

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/ha_data.py tests/test_ha_data.py
git commit -m "feat: add split_ev_charging_hybrid() to cover pre-wallbox EV history"
```

---

### Task 3: Wire the hybrid split into `_retrain()` (fixes issue A)

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:1573-1574`
- Test: `tests/test_energy_forecast.py` (new class, add after `TestEvChargingSensor`, ~line 645)

**Interfaces:**
- Consumes: `ha_data.split_ev_charging_hybrid(energy_df, ev_kwh_df, threshold_kwh, charger_kw=9.0)` (Task 2), `ha_data.ev_sensor_coverage(ev_kwh_df)` (Task 1).
- Produces: `_retrain()`'s local `_ev_sensor_coverage: tuple[pd.Timestamp, pd.Timestamp] | None` variable — consumed by Task 4 in the same method.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_energy_forecast.py — add after class TestEvChargingSensor (after its last test, ~line 645)


class TestRetrainCoversPreWallboxEvHistory:
    """Regression test (found 2026-08-28): once ev_charging_sensor is configured,
    _retrain() ran split_ev_charging_from_sensor() on the FULL energy_df history,
    not just the sensor's own coverage window. Its left-join fillna(0.0) then
    silently treated every pre-installation date as "0 kWh charged", un-excluding
    real threshold-shaped EV sessions from before the wallbox existed. Fixed by
    routing through split_ev_charging_hybrid(), which falls back to threshold
    detection outside the sensor's coverage window.
    """

    def _patch_deps(self, monkeypatch, energy_df, ev_kwh_df):
        import energy_forecast.ha_data as ha_data_mod
        import energy_forecast.weather as weather_mod

        empty_df = pd.DataFrame()

        monkeypatch.setattr(ha_data_mod, "fetch_energy_history", lambda *a, **kw: energy_df)
        monkeypatch.setattr(ha_data_mod, "fetch_sub_sensor_history", lambda *a, **kw: ev_kwh_df)
        monkeypatch.setattr(weather_mod, "fetch_historical_weather", lambda *a, **kw: _empty_weather())
        monkeypatch.setattr(weather_mod, "fetch_open_meteo", lambda *a, **kw: _empty_weather())
        monkeypatch.setattr(ha_data_mod, "fetch_boolean_entity_history", lambda *a, **kw: empty_df)
        monkeypatch.setattr(ha_data_mod, "fetch_presence_history", lambda *a, **kw: empty_df)
        monkeypatch.setattr(ha_data_mod, "fetch_energy_history_15m", lambda *a, **kw: None)
        # split_ev_charging / split_ev_charging_from_sensor / split_ev_charging_hybrid
        # deliberately left unpatched — this test exercises the real hybrid split.

    def test_pre_wallbox_ev_spike_still_excluded_from_training(self, tmp_path, monkeypatch):
        from energy_forecast.energy_forecast import EnergyForecast

        energy_df = _make_energy_df(60)  # 2024-01-01 00:00..02:00 (60h), flat 1.0 kWh
        # Threshold-shaped spike at 10:00, well before the wallbox existed.
        energy_df.loc[energy_df["timestamp"] == pd.Timestamp("2024-01-01 10:00"), "gross_kwh"] = 10.0

        # Wallbox sensor coverage only starts 2024-01-02 06:00 (idx30) — after the spike.
        ev_kwh_df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-02 06:00", periods=11, freq="1h"), "kwh": [0.0] * 11}
        )

        self._patch_deps(monkeypatch, energy_df, ev_kwh_df)

        stub = _FakeRetrain(tmp_path / "energy_history.csv")
        stub._ev_threshold = 7.0
        stub._ev_charger_kw = 9.0
        stub._ev_charging_sensor = "sensor.se_one_ev_charger_total_energy"

        EnergyForecast._retrain(stub)

        trained_df = stub._ml_model.train.call_args.args[0]
        spike_row = trained_df[trained_df["timestamp"] == pd.Timestamp("2024-01-01 10:00")]
        assert len(spike_row) == 1, "the EV hour itself must remain in the training frame"
        assert abs(spike_row.iloc[0]["gross_kwh"] - 1.0) < 1e-6, (
            "pre-wallbox spike must be threshold-detected (10.0 - 9.0 charger_kw = 1.0), "
            f"got {spike_row.iloc[0]['gross_kwh']} — it is leaking into training as raw baseline load"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py::TestRetrainCoversPreWallboxEvHistory -v`
Expected: FAIL — `spike_row.iloc[0]["gross_kwh"]` is `10.0` (untouched), not `1.0`, because `_retrain()` still calls `split_ev_charging_from_sensor()` directly on the full history.

- [ ] **Step 3: Wire in the hybrid split**

In `apps/energy_forecast/energy_forecast.py`, replace:

```python
                else:
                    baseline_df, ev_df = ha_data.split_ev_charging_from_sensor(energy_df, _ev_hist_stripped)
```

(line 1573-1574) with:

```python
                else:
                    baseline_df, ev_df = ha_data.split_ev_charging_hybrid(
                        energy_df, _ev_hist_stripped, self._ev_threshold, charger_kw=self._ev_charger_kw
                    )
                    _ev_sensor_coverage = ha_data.ev_sensor_coverage(_ev_hist_stripped)
```

This requires a variable to hold `_ev_sensor_coverage` across the whole `if self._ev_charging_sensor:` block (Task 4 reads it right after, and it must exist — as `None` — on every other branch). Add the initialization immediately before that `if`. Replace:

```python
        # ── Subtract EV charging from gross import ────────────────────────────
        if self._ev_charging_sensor:
```

(line 1558-1559) with:

```python
        # ── Subtract EV charging from gross import ────────────────────────────
        _ev_sensor_coverage: tuple | None = None
        if self._ev_charging_sensor:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py::TestRetrainCoversPreWallboxEvHistory -v`
Expected: PASS

- [ ] **Step 5: Run the full existing EV/_retrain test suites to check for regressions**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py -k "Ev or Retrain or FifteenMinCache" -v`
Expected: PASS (all — includes `TestRetrainEvCachePathBug`, `TestRetrainCorrectsBeforeEvSplit`, `TestRetrainCoversPreWallboxEvHistory`, `TestFifteenMinCache`, `TestRetrainExcludedRanges`)

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast.py
git commit -m "fix: cover pre-wallbox EV history via split_ev_charging_hybrid in _retrain"
```

---

### Task 4: Skip ±1h adjacent-hour padding for sensor-detected EV hours (fixes issue B)

**Files:**
- Modify: `apps/energy_forecast/energy_forecast.py:1588-1603`
- Test: `tests/test_energy_forecast.py` (new class, add after `TestRetrainCoversPreWallboxEvHistory`)

**Interfaces:**
- Consumes: `_ev_sensor_coverage` (Task 3's local variable, `None` on every branch except the hybrid-split branch).

**Root cause this supports:** the ±1h pad exists because `split_ev_charging()`'s flat `charger_kw` subtraction can't tell exactly when a session ramps up/down, so its immediate neighbor hours might still carry a partly-charging co-load. `split_ev_charging_from_sensor()` has no such ambiguity — it subtracts the wallbox's own exact per-hour kWh, so an hour it *doesn't* flag genuinely had 0 kWh charged, no padding needed. Applying the pad unconditionally strips far more than intended from real (long, solar-tracking) wallbox sessions: verified against the live `data/` pull, all 8 real post-wallbox EV days have a full 24 hourly rows before padding, but only 2 still have ≥18 (clustering's daily-profile completeness floor) after it — the other 6 fall to 13–17 rows and silently vanish from `clustering.DailyProfileClusterer.fit()`'s `pivoted` matrix instead of being logged as excluded.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_energy_forecast.py — add after class TestRetrainCoversPreWallboxEvHistory


class TestRetrainSkipsPaddingForSensorDetectedEv:
    """Regression test (found 2026-08-28): _retrain() dropped the ±1h hours
    adjacent to EVERY EV hour, regardless of detection method. That padding
    exists to cover split_ev_charging()'s flat-charger-kw tapering uncertainty;
    split_ev_charging_from_sensor() has no such uncertainty (exact per-hour kWh),
    so padding sensor-detected hours only discards good training rows — and for
    long solar-tracking wallbox sessions, discards enough of the day to push it
    below clustering's 18h/day completeness floor. Threshold-detected hours
    (including the pre-wallbox-coverage ones from Task 3) must still be padded.
    """

    def _patch_deps(self, monkeypatch, energy_df, ev_kwh_df):
        TestRetrainCoversPreWallboxEvHistory()._patch_deps(monkeypatch, energy_df, ev_kwh_df)

    def test_sensor_detected_ev_hour_neighbors_are_kept(self, tmp_path, monkeypatch):
        from energy_forecast.energy_forecast import EnergyForecast

        energy_df = _make_energy_df(60)  # 2024-01-01 00:00.., flat 1.0 kWh

        # Wallbox coverage 2024-01-02 06:00..16:00 (idx30..idx40); charges at 11:00 (idx35).
        ev_kwh_df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-02 06:00", periods=11, freq="1h"), "kwh": [0.0] * 11}
        )
        ev_kwh_df.loc[ev_kwh_df["timestamp"] == pd.Timestamp("2024-01-02 11:00"), "kwh"] = 5.0

        self._patch_deps(monkeypatch, energy_df, ev_kwh_df)

        stub = _FakeRetrain(tmp_path / "energy_history.csv")
        stub._ev_threshold = 7.0
        stub._ev_charger_kw = 9.0
        stub._ev_charging_sensor = "sensor.se_one_ev_charger_total_energy"

        EnergyForecast._retrain(stub)

        trained_ts = set(stub._ml_model.train.call_args.args[0]["timestamp"])
        assert pd.Timestamp("2024-01-02 10:00") in trained_ts, "sensor-detected EV hour's left neighbor must be kept"
        assert pd.Timestamp("2024-01-02 12:00") in trained_ts, "sensor-detected EV hour's right neighbor must be kept"

    def test_threshold_detected_ev_hour_neighbors_are_still_padded(self, tmp_path, monkeypatch):
        """Same retrain call, but the pre-coverage threshold-detected hour (Task 3's
        scenario) must still lose its ±1h neighbors — padding is still needed there."""
        from energy_forecast.energy_forecast import EnergyForecast

        energy_df = _make_energy_df(60)
        energy_df.loc[energy_df["timestamp"] == pd.Timestamp("2024-01-01 10:00"), "gross_kwh"] = 10.0

        ev_kwh_df = pd.DataFrame(
            {"timestamp": pd.date_range("2024-01-02 06:00", periods=11, freq="1h"), "kwh": [0.0] * 11}
        )

        self._patch_deps(monkeypatch, energy_df, ev_kwh_df)

        stub = _FakeRetrain(tmp_path / "energy_history.csv")
        stub._ev_threshold = 7.0
        stub._ev_charger_kw = 9.0
        stub._ev_charging_sensor = "sensor.se_one_ev_charger_total_energy"

        EnergyForecast._retrain(stub)

        trained_ts = set(stub._ml_model.train.call_args.args[0]["timestamp"])
        assert pd.Timestamp("2024-01-01 09:00") not in trained_ts, "threshold-detected EV hour's left neighbor must still be dropped"
        assert pd.Timestamp("2024-01-01 11:00") not in trained_ts, "threshold-detected EV hour's right neighbor must still be dropped"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py::TestRetrainSkipsPaddingForSensorDetectedEv -v`
Expected: `test_sensor_detected_ev_hour_neighbors_are_kept` FAILs (10:00/12:00 are currently dropped unconditionally); `test_threshold_detected_ev_hour_neighbors_are_still_padded` already PASSes (no regression yet, current code always pads) — confirms the test correctly isolates the behavior being changed.

- [ ] **Step 3: Make the padding conditional on detection method**

In `apps/energy_forecast/energy_forecast.py`, replace:

```python
        if len(ev_df):
            _LOGGER.info(
                "EV filter: %d charging hours detected (%.1f kWh gross). Sessions on: %s",
                len(ev_df),
                ev_df["gross_kwh"].sum(),
                sorted(ev_df["timestamp"].dt.date.unique().tolist()),
            )
            # Drop ±1h adjacent hours (ramp-up/down) — split_ev_charging subtracts the
            # charger load from exact EV hours but can't cleanly correct adjacent hours
            # whose elevation comes from a tapering charger.  Dropping 1–2 rows per
            # session (≈15% of days) has negligible training impact; NaN lags are
            # filled by feature medians in meta.pkl.
            _ev_adj_ts: set = {
                ts + pd.Timedelta(hours=d) for ts in pd.to_datetime(ev_df["timestamp"]).dt.floor("1h") for d in (-1, 1)
            }
            baseline_df = baseline_df[~pd.to_datetime(baseline_df["timestamp"]).dt.floor("1h").isin(_ev_adj_ts)]
```

(lines 1588-1603) with:

```python
        if len(ev_df):
            _LOGGER.info(
                "EV filter: %d charging hours detected (%.1f kWh gross). Sessions on: %s",
                len(ev_df),
                ev_df["gross_kwh"].sum(),
                sorted(ev_df["timestamp"].dt.date.unique().tolist()),
            )
            # Drop ±1h adjacent hours (ramp-up/down) — but only for threshold-detected
            # EV hours. split_ev_charging()'s flat charger_kw subtraction can't tell
            # exactly when a session ramps up/down, so its immediate neighbors may
            # still carry a partial co-load. split_ev_charging_from_sensor() has no
            # such ambiguity (exact per-hour wallbox kWh) — padding those hours too
            # only discards good rows, and for long solar-tracking sessions discards
            # enough of the day to push it below clustering's 18h/day completeness
            # floor (found 2026-08-28: 6 of 8 real post-wallbox EV days were silently
            # dropped from regime clustering entirely, not counted as excluded).
            ev_ts_floor = pd.to_datetime(ev_df["timestamp"]).dt.floor("1h")
            if _ev_sensor_coverage is not None:
                cov_start, cov_end = _ev_sensor_coverage
                pad_ts = ev_ts_floor[(ev_ts_floor < cov_start) | (ev_ts_floor > cov_end)]
            else:
                pad_ts = ev_ts_floor
            _ev_adj_ts: set = {ts + pd.Timedelta(hours=d) for ts in pad_ts for d in (-1, 1)}
            baseline_df = baseline_df[~pd.to_datetime(baseline_df["timestamp"]).dt.floor("1h").isin(_ev_adj_ts)]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py::TestRetrainSkipsPaddingForSensorDetectedEv -v`
Expected: PASS (both tests)

- [ ] **Step 5: Run the existing ±1h-padding regression tests to confirm the no-sensor path is untouched**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_energy_forecast.py -k "adjacent or Retrain or Ev or FifteenMinCache" -v`
Expected: PASS (all — includes `test_retroactive_eviction_removes_stale_ev_adjacent_actuals`, `test_actuals_for_retrain_excludes_ev_adjacent_hours`, `test_retrain_baseline_excludes_ev_adjacent_hours`, which don't call `_retrain()` and are unaffected, plus everything from Task 3)

- [ ] **Step 6: Commit**

```bash
git add apps/energy_forecast/energy_forecast.py tests/test_energy_forecast.py
git commit -m "fix: skip ±1h EV padding for sensor-detected hours in _retrain"
```

---

### Task 5: Full suite verification + docs

**Files:**
- Modify: `README.md:441` (table row), `README.md:751` (prose paragraph)
- Modify: `CHANGELOG.md` (via `@changelog-writer` subagent)
- Modify: `MEMORY.md` / `memory/*.md` (per repo convention — do not `git add MEMORY.md`)

- [ ] **Step 1: Run the full test suite**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: PASS, 0 failures (note the exact count — this repo's baseline count should be documented in the task's completion notes so a regression is obvious)

- [ ] **Step 2: Update README.md's `ev_charging_sensor` table row**

Replace (line 441):

```
| `ev_charging_sensor` | No | — | Entity ID of a cumulative wallbox energy meter (`total_increasing`, e.g. `sensor.wallbox_total_energy`). When set, actual sensor readings replace threshold-based EV detection — required for variable-power solar-surplus charging where grid import never exceeds the threshold. Falls back to threshold detection if the sensor returns no data. |
```

with:

```
| `ev_charging_sensor` | No | — | Entity ID of a cumulative wallbox energy meter (`total_increasing`, e.g. `sensor.wallbox_total_energy`). When set, actual sensor readings replace threshold-based EV detection within the sensor's own history — required for variable-power solar-surplus charging where grid import never exceeds the threshold. Falls back to threshold detection if the sensor returns no data at all, and automatically for any date before the sensor was first configured (its cache never backfills earlier history). |
```

- [ ] **Step 3: Update README.md's solar-surplus wallbox paragraph**

Replace (line 751):

```
**Solar-surplus wallboxes** charge at variable power tracking the PV surplus, so grid import never rises enough to trigger the threshold. For these setups, configure `ev_charging_sensor` with a cumulative kWh meter from the wallbox (e.g. `sensor.wallbox_total_energy`). Actual sensor readings replace threshold inference; the threshold path remains as an automatic fallback if the sensor goes offline.
```

with:

```
**Solar-surplus wallboxes** charge at variable power tracking the PV surplus, so grid import never rises enough to trigger the threshold. For these setups, configure `ev_charging_sensor` with a cumulative kWh meter from the wallbox (e.g. `sensor.wallbox_total_energy`). Actual sensor readings replace threshold inference inside the sensor's coverage window; the threshold path remains as an automatic fallback if the sensor goes offline, and is used automatically for any training history that predates the sensor's installation — so switching to a wallbox meter mid-history doesn't un-exclude EV sessions recorded before it existed.
```

- [ ] **Step 4: Update CHANGELOG.md**

Dispatch the `@changelog-writer` subagent with: "Add a CHANGELOG entry for two EV-detection fixes in `_retrain()`: (1) `ev_charging_sensor` history predating the sensor's installation is now threshold-detected instead of silently un-excluded from training (new `ha_data.split_ev_charging_hybrid()` / `ev_sensor_coverage()`), (2) the ±1h adjacent-hour padding around EV hours is now skipped for sensor-detected hours (only applied to threshold-detected ones), fixing regime clustering silently dropping most post-wallbox EV days below its 18h/day completeness floor instead of counting them as excluded."

- [ ] **Step 5: Update project memory**

Check `memory/*.md` and `MEMORY.md` for any existing entry describing the wallbox EV detection setup or the "only 2 excluded EV days" symptom; update or add a `project`-type memory noting this fix landed, with a link to this plan file. Do not `git add MEMORY.md` (gitignored).

- [ ] **Step 6: Commit docs**

```bash
git add README.md CHANGELOG.md
git commit -m "docs: document EV wallbox history-gap and padding fixes"
```

---

## Self-Review Notes

- **Spec coverage:** Issue A (pre-wallbox history un-excluded) → Task 1–3. Issue B (±1h padding over-trimming sensor-detected sessions) → Task 4. Docs/changelog/memory → Task 5. Both issues from the diagnosis are covered; no gaps identified.
- **Scope boundary verified:** `_update_sensors()`'s recent/blended-actuals EV paths use `fetch_recent_sub_sensor()`, capped at `_FETCH_RECENT_TAIL_ROWS = 400` (~16.7 days) — always within the wallbox's coverage window given it was installed ~32 days ago, so issue A cannot occur there. Left untouched, confirmed by explicit constraint in Global Constraints.
- **Existing regression tests re-checked line-by-line:** `test_retrain_ev_split_operates_on_corrected_energy_df` and `test_retrain_applies_correction_before_ev_split` string-match `"ha_data.split_ev_charging(\n                energy_df, self._ev_threshold"` (16-space indent) and the bare first occurrence of `"ha_data.split_ev_charging("` respectively — both only match the *no-sensor* `else` branch and the *sensor-empty* branch, neither of which Task 3/4 touch. Confirmed unaffected.
