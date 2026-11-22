# Tau & UA Estimation Rework — Implementation Plan (D)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estimate the building's thermal time constant τ and its effective heat-loss coefficient UA from data available during heating season. Today τ only learns from season-switch-off hours (almost none since the 2026-09-26 automation) with a biased fit, and UA stays `null`.

**Architecture:**
1. Cache each Netatmo room's `heating_power_request` (valve demand) so "no heat input" is known per room and per hour.
2. New pure module `tau_estimation.py`: pooled per-room ODE regression `ΔT_in = (T_out − T_in)/τ` over valve-closed night hours, area-weighted by heat-loss rate. **Airing is filtered without an extra sensor:** a room-night whose cooling rate k = 1/τ is > 2.5× that night's median k across rooms (rooms with k > 0, needs ≥ 3) is dropped. Weather and envelope are shared on a given night; an open window is room-specific. The rule uses k, not τ: τ = ∞ for barely-cooling rooms would make a τ-median meaningless.
3. Physics `UA_eff` from a **daily** heat balance on the heating sub-meter (rolling 28 days ending at the data end), refreshed weekly.
4. A gated ML feature for this season's heating intensity.

Every behaviour change sits behind a config flag whose default keeps today's behaviour. Branches can therefore merge and deploy safely, and each gate only flips a flag.

**Tech Stack:** Python 3.13, numpy, pandas, pytest; HA history API (attributes).

**Spec / evidence** (2026-10-15 investigation, `memory/project_cold_onset_overforecast_2026_10_15.md`; analysis scripts gitignored in `scripts/`):
- Live τ = 11.64 h. The log-ΔT window fit reads the outdoor morning warm-up as room cooling: 06:00 windows show no indoor drop yet return τ 7–15 h, while evening windows give 41–106 h (`analyze_tau_valve_windows.py`).
- Netatmo exposes `heating_power_request` (0–100). Valves are closed in 190–250 of ~248 h per room; 91 % of heat-pump heating kWh falls in hours where at least one room requests heat.
- Pooled ODE on valve-closed night hours (`analyze_tau_valve_ode.py`, using **5-min time-weighted** temperatures): per room 41–800 h, area-weighted ≈ 92 h. Biased high by thermal mass and neighbouring rooms. Only 10 days of data (recorder retention). The shipped estimator uses the cache's hourly last value (0.5 °C steps), so ≈ 92 h is **not** guaranteed to reproduce. The Task 3 gate decides.
- Bedroom airing (2026-10-05..13, per-night τ from valve-closed 21–06 h): airing nights (humidity drop > 0.5 g/m³) gave τ 18–62 h; the two clearly closed nights 116 h and 219 h; other rooms 67–204 h the same nights. The bedroom is aired on most nights, so a threshold relative to the room's *own* history fails (its median ≈ 40 h is the airing regime). The **cross-room** rule on k (all 10 rooms; 7 of 9 nights evaluable with ≥ 3 rooms k > 0) flags **5 bedroom nights**: every short-τ bedroom night among the evaluable ones, and neither of the two closed ones. It also flags **1 bathroom night** (27 h on 10-06, plausibly post-shower airing) and **0** room-nights in the other 8 rooms. User decision 2026-10-15: no humidity sensor (one less measurement to track). The existing `detect_open_windows` (`physics.py:1036`) is a single-hour shock detector and misses windows left ajar for hours; it stays as is.
- Hourly `_calibrate_ua_eff` (`physics.py:714`) fails for two reasons. The buffer tank and heat-pump cycling decouple hourly input from loss. And the sub-meter tier keeps only hours with `kwh > HEATING_SUB_METER_MIN_KWH`, which is selection on the target.
- Daily balance (`analyze_ua_daily_heat_balance.py`, 51 heating days) works but is seasonal: UA/COP 21–28 W/K in spring (R² 0.50–0.58), 8–9 W/K in October (n = 8).
- Physics uses formula COP (≈ 3.2 at 10 °C), so October's effective UA ≈ 28 W/K, below `UA_EFF_SANITY_MIN_W_PER_K = 50`.
- **Shipped-estimator check on real cache data (2026-10-15, hourly last values, 10 nights):** house τ ≈ 131 h (rel_se 0.16); at ratio 2.5 the airing filter drops 6 bedroom + 2 dining-room room-nights. Bedroom τ goes 46 h → 95 h with the filter on. Four interior rooms come out k ≤ 0 (barely cooling; consistent with 300–2600 h in the 5-min analysis) and count as k = 0 with their weight kept. House τ is insensitive to the ratio (131–134 h at 2.0–2.5).
- **Coupling hazard (review F3, verified):** `physics._space_heating_kwh` uses `c_building = config.c_building_wh_k or UA·τ` (`physics.py:229-230`), and `model.py:467` copies the ML τ into physics. With UA ≈ 28 and τ 11.6 → 92 h, `c_building` grows 8× and the clipped `q_mass` term inflates physics heating by ~40 %.
- **Cutoff hazard (review F1, verified):** `model.py:~456` passes `holdout_cutoff = data end − 10 % of span` (≈ 37 days back live). The 28 days before it contain 0 heating days, so a window ending at the ML holdout would keep UA `null` until late November.
- **Cross-repo check (verified 2026-10-15):** ha-energy-manager reads only the **state** of `sensor.energy_forecast_thermal_pressure_net` (`ha-energy-manager/apps/energy_manager/loads/heat_pump.py:861`), never its `tau_hours` attribute. A τ change doesn't affect it. Mention the τ change in the CHANGELOG anyway.

**Base branch / branches** (all off `dev`, each rebased on `dev` after the previous alpha lands):
- `feat/climate-valve-cache` — Task 1. **Independent of all other plans; ship first:** data has to accumulate (≥ 14 valve-closed nights) and HA keeps only ~10 days.
- `feat/tau-pooled` — Task 3, flag `tau_estimator` (default `windows`). After Plan C.
- `feat/ua-eff-daily` — Task 4, flag `physics.ua_eff_method` (default `hourly`).
- `feat/heating-intensity-feature` — Task 5, flag `heating_intensity_feature` (default `false`).

τ is **not on the critical path** of the cold-day over-forecast: after Plan C, τ only matters for heating-off hours and warm rooms cooling down.

## Global Constraints
- Tests via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`. Ruff via `/home/jovyan/my_envs/ha-energy-forecast/bin/ruff`.
- With all flags at their defaults, behaviour is identical to today (asserted by the existing suites staying green unchanged).
- Users without `heating_power_request` (non-Netatmo) or without a heating sub-meter can't enable the new paths. The flags then fall back with a warning.
- The climate CSV gains an optional `heating_power_request` column. Old rows are NaN, and every existing reader keeps working.
- Physics COP stays formula-based (`_cop_series(..., cop_sensor_series=None)`), the same as serve and training. UA is "effective UA in formula-COP units".
- Gates on ≤ 7-day folds are **sanity checks, not proof**. Each needs a paired daily sign test (variant better on ≥ 5 of 7 days) on top of the MAE threshold. An opt-in *feature* (Task 5) is only switched on live after ≥ 14 cold post-deploy days confirm it (see the index's verification section).
- Commit messages end with the session's `Co-Authored-By` / `Claude-Session` trailers. `scripts/` is gitignored: gate scripts are local-only.

## Review Focus
1. Climate history with `heating_power_request` missing for some states (unavailable / old firmware) → NaN, never treated as 0 (= "closed"); the carry stops at a null state. Quiet hours (no state change) **carry** the last state, because HA logs only changes.
2. Method switch on a model whose meta has no `tau_method` (all existing installs) → treated as `"windows"`. A switch only happens when the flag is set **and** the estimate is confident, and it resets both drift anchors (otherwise the ±35 %/30 d cap would take ~6 months to reach ~60–90 h).
3. Rooms with noisy `k ≤ 0` → k clipped to 0 with their area weight **kept** (dropping them would bias the house k up). If the house k is ≤ 0 → None. They never enter the airing median and are never flagged as aired.
3b. A habitually aired room (bedroom: most nights) → its aired nights are dropped by the cross-room rule even though its *own* median is short. A night with < 3 rooms with k > 0 → no airing filter that night.
4. Daily UA on a day with partial climate **or meter** coverage (< 22 h) → day dropped, not under-counted.
5. Raising τ must not change physics `c_building`; it is decoupled in Task 4.

---

### Task 1: Cache `heating_power_request` in the climate cache (ship first, `feat/climate-valve-cache`)

**Files:**
- Modify: `apps/energy_forecast/ha_data.py` — `fetch_climate_history` (~L1196–1206) and `fetch_recent_climate` (~L1255–1263). Rationale: both build the hourly climate frame; one shared helper keeps them in sync.
- Modify (local, gitignored): `scripts/pull_ha_data.py`. Rationale: the gate scripts need `climate_*`, `dhw_*` and `physics_*` caches locally, and the current prefix filter (L78–85) doesn't pull them.
- Test: `tests/test_ha_data.py` (new class `TestClimateValveDemand`)

**Interfaces:**
- Produces: climate frames/CSVs with optional column `heating_power_request` (float 0–100). Hourly value = max over the states in the hour **and** the state in force at the hour's start. HA history records only *changes*, so a state persists across hours without rows. An explicit null/unavailable state ends the carry. NaN only where truly unknown. (Verified 2026-10-15: a one-hour-deep carry left 20–70 % of real hours NaN; the persistent carry leaves 0 NaN, with 170–230 closed hours per room in 10 days.)
- Produces: `ha_data._hourly_climate(ha_df: pd.DataFrame) -> pd.DataFrame`

- [ ] **Step 1: Failing tests**

```python
def _climate_raw(rows):
    """rows: (utc_iso, current_temperature, temperature, heating_power_request | None | "missing")."""
    out = []
    for ts, cur, sp, hpr in rows:
        r = {"timestamp": pd.Timestamp(ts).tz_convert("Europe/Zurich").floor("1h"),
             "current_temperature": cur, "temperature": sp}
        if hpr != "missing":
            r["heating_power_request"] = hpr
        out.append(r)
    return pd.DataFrame(out)


class TestClimateValveDemand:
    def test_hpr_hourly_max_includes_carried_state(self, mock_app, tmp_path):
        raw = _climate_raw([
            ("2026-10-14T20:50:00Z", 20.0, 20.0, 40),   # local 22:xx → valve open
            ("2026-10-14T21:10:00Z", 20.0, 20.0, 0),    # local 23:xx → closes
            ("2026-10-14T23:30:00Z", 19.5, 20.0, 0),    # local 01:xx
        ])
        with patch.object(ha_data, "_fetch_history", return_value=raw):
            out = ha_data.fetch_climate_history(mock_app, "climate.room", tmp_path / "climate_room.csv")
        h = out.assign(timestamp=pd.to_datetime(out["timestamp"])).set_index("timestamp")["heating_power_request"]
        assert h[pd.Timestamp("2026-10-14 22:00")] == 40
        assert h[pd.Timestamp("2026-10-14 23:00")] == 40   # open at hour start (carried), closed during the hour
        assert h[pd.Timestamp("2026-10-15 00:00")] == 0
        assert h[pd.Timestamp("2026-10-15 01:00")] == 0

    def test_carry_does_not_cross_missing_attribute(self, mock_app, tmp_path):
        raw = _climate_raw([
            ("2026-10-14T20:10:00Z", 20.0, 20.0, 0),
            ("2026-10-14T21:10:00Z", 20.0, 20.0, None),  # attribute present but null (unavailable)
            ("2026-10-14T23:10:00Z", 20.0, 20.0, 0),
        ])
        with patch.object(ha_data, "_fetch_history", return_value=raw):
            out = ha_data.fetch_climate_history(mock_app, "climate.room", tmp_path / "climate_room.csv")
        h = out.assign(timestamp=pd.to_datetime(out["timestamp"])).set_index("timestamp")["heating_power_request"]
        assert pd.isna(h[pd.Timestamp("2026-10-15 00:00")])   # no state in hour, carry blocked by the null

    def test_state_persists_across_quiet_hours(self, mock_app, tmp_path):
        raw = _climate_raw([
            ("2026-10-14T20:10:00Z", 20.0, 20.0, 0),     # local 22:xx closed
            ("2026-10-15T00:10:00Z", 19.0, 20.0, 0),     # local 02:xx — no rows at 23, 00, 01
        ])
        with patch.object(ha_data, "_fetch_history", return_value=raw):
            out = ha_data.fetch_climate_history(mock_app, "climate.room", tmp_path / "climate_room.csv")
        h = out.assign(timestamp=pd.to_datetime(out["timestamp"])).set_index("timestamp")["heating_power_request"]
        assert list(h[pd.Timestamp("2026-10-14 23:00"):pd.Timestamp("2026-10-15 01:00")]) == [0, 0, 0]

    def test_null_state_hour_itself_uses_start_state_only(self, mock_app, tmp_path):
        raw = _climate_raw([
            ("2026-10-14T20:10:00Z", 20.0, 20.0, 30),
            ("2026-10-14T21:10:00Z", 20.0, 20.0, None),  # local 23:10 → unavailable
        ])
        with patch.object(ha_data, "_fetch_history", return_value=raw):
            out = ha_data.fetch_climate_history(mock_app, "climate.room", tmp_path / "climate_room.csv")
        h = out.assign(timestamp=pd.to_datetime(out["timestamp"])).set_index("timestamp")["heating_power_request"]
        assert h[pd.Timestamp("2026-10-14 23:00")] == 30      # open at hour start, then unknown

    def test_without_attribute_column_absent(self, mock_app, tmp_path):
        raw = _climate_raw([("2026-10-14T20:50:00Z", 20.0, 20.0, "missing"), ("2026-10-14T22:50:00Z", 20.0, 20.0, "missing")])
        with patch.object(ha_data, "_fetch_history", return_value=raw):
            out = ha_data.fetch_climate_history(mock_app, "climate.room", tmp_path / "climate_room.csv")
        assert "heating_power_request" not in out.columns or out["heating_power_request"].isna().all()

    def test_old_cache_without_column_merges(self, mock_app, tmp_path):
        p = tmp_path / "climate_room.csv"
        pd.DataFrame({"timestamp": ["2026-10-13 10:00:00"], "current_temp": [20.0], "setpoint": [20.0]}).to_csv(p, index=False)
        raw = _climate_raw([("2026-10-14T20:50:00Z", 20.0, 20.0, 0)])
        with patch.object(ha_data, "_fetch_history", return_value=raw):
            out = ha_data.fetch_climate_history(mock_app, "climate.room", p)
        assert len(out) >= 2 and pd.isna(out.iloc[0]["heating_power_request"])

    def test_recent_climate_also_caches_hpr(self, mock_app, tmp_path):
        raw = _climate_raw([("2026-10-14T20:50:00Z", 20.0, 20.0, 15)])
        with patch.object(ha_data, "_fetch_history", return_value=raw):
            out = ha_data.fetch_recent_climate(mock_app, "climate.room", tmp_path / "climate_room.csv")
        assert out["heating_power_request"].iloc[-1] == 15
```
(`raw_ha` timestamps are already floored to the hour by the fetchers before this point. The fixture mirrors that.)

- [ ] **Step 2: Run** `…/bin/python -m pytest tests/test_ha_data.py::TestClimateValveDemand -v` → FAIL (`KeyError: 'heating_power_request'`).

- [ ] **Step 3: Implement.** New helper above `fetch_climate_history`:

```python
def _hourly_climate(ha_df: pd.DataFrame) -> pd.DataFrame:
    """Hourly climate frame from attribute rows (timestamps already floored to the hour).

    current_temp / setpoint: last state in the hour, forward-filled (unchanged behaviour).
    heating_power_request (Netatmo valve demand, optional): max over the states in the hour
    AND the state in force at the start of the hour (a valve open at :00 that closes at :10
    still delivered heat). HA history only records changes, so a state persists across hours
    without rows — it is carried until the next row. A row whose attribute is null
    (unavailable) ends the carry: hours after it stay NaN until a real value arrives, so
    unknown is never turned into 0 (closed).
    """
    import numpy as np
    import pandas as pd

    g = ha_df.set_index("timestamp").sort_index()
    hourly = g[["current_temp", "setpoint"]].resample("1h").last().ffill()
    if "heating_power_request" in g.columns:
        _NULL = -1.0  # sentinel: an explicit null state, distinct from "no row this hour"
        hpr = pd.to_numeric(g["heating_power_request"], errors="coerce")
        in_hour_max = hpr.resample("1h").max()
        end_state = hpr.fillna(_NULL).resample("1h").last()       # NaN only for hours without rows
        end_state = end_state.ffill()                             # state persists across empty hours
        start_state = end_state.shift(1).where(lambda x: x != _NULL)
        hourly["heating_power_request"] = np.fmax(in_hour_max, start_state)
    hourly = hourly.reset_index()
    if hourly["timestamp"].dt.tz is not None:
        hourly["timestamp"] = hourly["timestamp"].dt.tz_localize(None)
    return hourly
```
(A null-state sentinel distinguishes "no row this hour" (the state persists) from "the attribute was null" (unknown): `fillna(_NULL)` before the hourly `last()`, `ffill()` across empty hours, then sentinel → NaN.)

In **both** fetchers, replace the block from `ha_df = raw_ha[["timestamp", "current_temperature", "temperature"]].copy()` through the `tz_localize(None)` lines with:
```python
        cols = ["timestamp", "current_temperature", "temperature"]
        if "heating_power_request" in raw_ha.columns:
            cols.append("heating_power_request")
        ha_df = raw_ha[cols].rename(columns={"current_temperature": "current_temp", "temperature": "setpoint"})
        df_new = _hourly_climate(ha_df)
```
`scripts/pull_ha_data.py` (local): `f.startswith(("sub_", "ev_", "heating_active_"))` → `f.startswith(("sub_", "ev_", "heating_active_", "climate_", "dhw_", "physics_"))`.

- [ ] **Step 4: Full suite + ruff → PASS** (existing climate tests unchanged).
- [ ] **Step 5: Commit** `feat: cache Netatmo heating_power_request in climate cache`; `@changelog-writer`; deploy-agent Workflow A. The next day, run `pull_ha_data.py` and confirm `data/climate_*.csv` has non-NaN `heating_power_request` for new rows in most hours, and that in valve-closed hours it reads 0, not NaN. Expect ≈ 0 NaN hours (verified on the 10-day attribute history).

---

### Task 2: (dropped 2026-10-15) — humidity-based ventilation detection

Replaced by the cross-room airing filter inside Task 3's estimator (user decision: avoid tracking another sensor). Task numbering is kept so cross-references stay valid.

---

### Task 3: Pooled valve-closed τ estimator behind `tau_estimator` flag (`feat/tau-pooled`)

**Files:**
- Create: `apps/energy_forecast/tau_estimation.py`. Rationale: pure functions, no HA access; `model.py` is already ~176 KB and the gate script imports it directly.
- Modify: `apps/energy_forecast/model.py`:
  - `_calibrate_tau` (~L1816–2122): EMA/drift tail → `_blend_tau`; pooled path behind the flag. Rationale: reuse the existing rate limits after the one-time switch.
  - `__init__` (~L294) + meta save (~L1713) / load (~L2160): `tau_method`.
  - Call site (~L433). Rationale: the pooled path doesn't need `heating_active_df`.
  - `train(..., tau_estimator="windows")`.
- Modify: `apps/energy_forecast/energy_forecast.py` — config `tau_estimator: windows|valve_pooled` (default `windows`), validated, passed to `train`.
- Modify: `apps/energy_forecast/const.py` — `TAU_POOLED_MIN_NIGHTS = 14`, `TAU_POOLED_FULL_CONF_HOURS = 150`, `TAU_POOLED_MIN_HOURS_PER_NIGHT = 4`
- Test: `tests/test_tau_estimation.py`, `tests/test_model.py` (new `TestTauMethodSwitch`; `TestCalibrateTau`, `TestTauCalibrationSafeguards` and `TestTauDriftCap` must stay green unchanged)

**Interfaces:**
- Produces: `tau_estimation.TauEstimate(tau_hours: float, per_room: dict[str, float], n_hours: int, n_nights: int, rel_se: float, n_airing_dropped: int)`
- Produces: `tau_estimation.estimate_tau_pooled(climate_dfs, weather_df, room_areas=None) -> TauEstimate | None`; `TauEstimate.n_airing_dropped: int` (room-nights removed by the cross-room rule)
- Produces: `EnergyForecastModel._blend_tau(self, old_tau: float, tau_raw: float, confidence: float, latest_ts: pd.Timestamp) -> float`. It covers the EMA (`new_weight = confidence · _TAU_EMA_MAX_NEW_WEIGHT`) and both drift caps only. The windows-path logging stays in `_calibrate_tau`.
- Produces: `self._tau_method` ∈ {`"windows"`, `"valve_pooled"`}; meta key `tau_method`

- [ ] **Step 1: Failing tests** (`tests/test_tau_estimation.py`, new file):

```python
import numpy as np
import pandas as pd
import pytest

from energy_forecast import tau_estimation as te


def _room(tau_true, hours=24 * 21, t_out=8.0, t0=21.0, seed=0, airing_tau=None, airing_every=0):
    """Synthetic room: valve closed 19–07, free-cooling with tau_true during 21–06, reheated to t0 from 07.
    With airing_every=n, every n-th night cools with airing_tau instead (window open)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-01", periods=hours, freq="1h")
    t = np.empty(hours); t[0] = t0
    for i in range(1, hours):
        cooling = idx[i].hour >= 21 or idx[i].hour < 7       # through 06:xx — reheat starts with the valve at 07
        night_no = (idx[i] - pd.Timedelta(hours=7)).dayofyear
        tau = airing_tau if (airing_every and night_no % airing_every == 0) else tau_true
        t[i] = t[i - 1] + (t_out - t[i - 1]) / tau if cooling else t0
    hpr = np.where((idx.hour >= 19) | (idx.hour < 7), 0.0, 50.0)
    cur = np.round((t + rng.normal(0, 0.05, hours)) * 2) / 2   # 0.5 °C quantisation like Netatmo
    return pd.DataFrame({"timestamp": idx, "current_temp": cur, "setpoint": t0, "heating_power_request": hpr})


def _weather(idx, t_out=8.0):
    return pd.DataFrame({"timestamp": idx, "temp_c": t_out, "humidity": 80.0})


@pytest.mark.parametrize("tau_true", [30.0, 60.0, 120.0])
def test_recovers_tau_within_25_percent(tau_true):
    c = _room(tau_true)
    est = te.estimate_tau_pooled({"climate.a": c}, _weather(c["timestamp"]))
    assert est is not None and est.tau_hours == pytest.approx(tau_true, rel=0.25)
    assert est.n_nights >= 14


def test_area_weighting_uses_mean_k_not_mean_tau():
    a, b = _room(40.0, seed=1), _room(120.0, seed=2)
    est = te.estimate_tau_pooled({"climate.a": a, "climate.b": b}, _weather(a["timestamp"]),
                                 room_areas={"climate.a": 10.0, "climate.b": 10.0})
    assert est.tau_hours == pytest.approx(60.0, rel=0.3)   # mean k = (1/40 + 1/120)/2 → 60 h (mean τ: 80 h)


def test_room_without_hpr_column_is_skipped():
    c = _room(60.0).drop(columns=["heating_power_request"])
    assert te.estimate_tau_pooled({"climate.a": c}, _weather(c["timestamp"])) is None


def test_hpr_nan_is_not_closed():
    c = _room(60.0); c["heating_power_request"] = np.nan
    assert te.estimate_tau_pooled({"climate.a": c}, _weather(c["timestamp"])) is None


def test_airing_room_nights_dropped_by_cross_room_rule():
    rooms = {f"climate.r{i}": _room(80.0, seed=i) for i in range(3)}
    rooms["climate.bed"] = _room(80.0, seed=9, airing_tau=12.0, airing_every=2)   # window open every other night
    w = _weather(rooms["climate.r0"]["timestamp"])
    est = te.estimate_tau_pooled(rooms, w)
    assert est.per_room["climate.bed"] == pytest.approx(80.0, rel=0.3)          # airing nights removed
    assert est.n_airing_dropped >= 8


def test_cross_room_rule_needs_three_rooms():
    a = _room(80.0, seed=1); b = _room(80.0, seed=2, airing_tau=12.0, airing_every=2)
    est = te.estimate_tau_pooled({"climate.a": a, "climate.b": b}, _weather(a["timestamp"]))
    assert est.n_airing_dropped == 0      # < 3 rooms that night → no reliable median → no filtering


def test_warming_room_clipped_keeps_weight():
    a = _room(60.0)
    warm = a.copy(); warm["current_temp"] = np.round(np.linspace(18, 22, len(warm)) * 2) / 2   # "closed" yet warming
    est = te.estimate_tau_pooled({"climate.a": a, "climate.w": warm}, _weather(a["timestamp"]),
                                 room_areas={"climate.a": 10.0, "climate.w": 10.0})
    assert est.tau_hours == pytest.approx(120.0, rel=0.3)   # k_w clipped to 0 but still weighted → house k halves
    assert te.estimate_tau_pooled({"climate.w": warm}, _weather(a["timestamp"])) is None


def test_positive_outlier_hours_trimmed():
    c = _room(60.0)
    night = (c["timestamp"].dt.hour == 2)
    c.loc[night, "current_temp"] += 1.0          # nightly +1 K blips (sun/neighbour/body heat artefacts)
    est = te.estimate_tau_pooled({"climate.a": c}, _weather(c["timestamp"]))
    assert est.tau_hours == pytest.approx(60.0, rel=0.35)
```
`tests/test_model.py`. Copy `_room`/`_weather` into a small module-level helper, or import them: `from tests.test_tau_estimation import _room, _weather` works if `tests/` is a package (`tests/__init__.py` exists).
```python
class TestTauMethodSwitch:
    def _model(self, tmp_path):
        m = EnergyForecastModel(tmp_path, timezone="Europe/Zurich")
        m._tau_hours, m._tau_method = 11.64, "windows"
        m._tau_anchor_hours, m._tau_anchor_ts = 11.64, pd.Timestamp("2026-10-01")
        m._tau_long_anchor_hours, m._tau_long_anchor_ts = 11.64, pd.Timestamp("2026-10-01")
        return m

    def test_flag_off_keeps_windows_path(self, tmp_path):
        m = self._model(tmp_path); c = _room(60.0)
        m._calibrate_tau({"climate.a": c}, pd.DataFrame(), _weather(c["timestamp"]), tau_estimator="windows")
        assert m._tau_method == "windows" and m._tau_hours == pytest.approx(11.64)

    def test_confident_pooled_switches_and_resets_anchors(self, tmp_path):
        m = self._model(tmp_path); c = _room(60.0)
        tau = m._calibrate_tau({"climate.a": c}, pd.DataFrame(), _weather(c["timestamp"]), tau_estimator="valve_pooled")
        assert tau == pytest.approx(60.0, rel=0.25) and m._tau_method == "valve_pooled"
        assert m._tau_anchor_hours == pytest.approx(tau) and m._tau_long_anchor_hours == pytest.approx(tau)

    def test_weak_confidence_does_not_switch(self, tmp_path):
        m = self._model(tmp_path); c = _room(60.0, hours=24 * 15)   # ≥ 14 nights but few hours → low confidence
        with patch("energy_forecast.model.TAU_POOLED_FULL_CONF_HOURS", 10_000):
            tau = m._calibrate_tau({"climate.a": c}, pd.DataFrame(), _weather(c["timestamp"]), tau_estimator="valve_pooled")
        assert tau == pytest.approx(11.64) and m._tau_method == "windows"

    def test_second_retrain_blends_with_drift_cap(self, tmp_path):
        m = self._model(tmp_path); c = _room(60.0)
        first = m._calibrate_tau({"climate.a": c}, pd.DataFrame(), _weather(c["timestamp"]), tau_estimator="valve_pooled")
        m._tau_hours = first
        c2 = _room(200.0, seed=5)  # implausible jump
        second = m._calibrate_tau({"climate.a": c2}, pd.DataFrame(), _weather(c2["timestamp"]), tau_estimator="valve_pooled")
        assert first < second <= first * 1.35 + 1e-6   # EMA + 30-day cap from the reset anchor

    def test_meta_roundtrip_keeps_method_and_defaults(self, tmp_path):
        import pickle
        from energy_forecast.model import _write_hash
        m = EnergyForecastModel(tmp_path); m._tau_method = "valve_pooled"; m._save()
        m2 = EnergyForecastModel(tmp_path); m2._load()
        assert m2._tau_method == "valve_pooled"
        with open(m._meta_path, "rb") as fh:
            meta = pickle.load(fh)
        meta.pop("tau_method")
        with open(m._meta_path, "wb") as fh:
            pickle.dump(meta, fh)
        _write_hash(m._meta_path)
        m3 = EnergyForecastModel(tmp_path); m3._load()
        assert m3._tau_method == "windows"
```
(Mirror `test_country_defaults_to_ch_when_absent_from_meta` at `tests/test_model.py:3130` for the exact tamper/rehash calls; adjust `_write_hash` usage to match it.)

- [ ] **Step 2: Run** → FAIL (`AttributeError: estimate_tau_pooled`, `_tau_method`, `tau_estimator`).

- [ ] **Step 3: Implement.** `tau_estimation.py` (new module):

```python
"""Thermal time-constant (τ) estimation from per-room valve state — Plan 2026-10-15 D.

Pure functions (no HA access) so they are unit-testable and reusable by gate scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .const import DEFAULT_ROOM_AREA_M2

NIGHT_START_H, NIGHT_END_H = 21, 6
MAX_STEP_K_PER_H = 0.8           # |ΔT_in| per hour above this is not passive cooling: a drop = window flung
                                 # open, a rise = sensor blip / sun / someone heating. Symmetric on purpose:
                                 # a one-sided filter turned a +1 K nightly blip into τ 135 h instead of 60 h.
MIN_INDOOR_EXCESS_K = 3.0        # need a real indoor–outdoor gradient for a usable slope
MIN_ROOM_HOURS = 24
MIN_HOURS_PER_NIGHT = 4
TRIM_SIGMA = 3.0
AIRING_K_RATIO = 2.5             # room-night cooling rate above this × the night's cross-room median k → window open
                                 # (2.0 flagged the fast-cooling living room on real hourly data; 3.0 missed the bedroom)
AIRING_MIN_ROOMS = 3             # need this many rooms with k > 0 that night for a reliable median


@dataclass
class TauEstimate:
    tau_hours: float
    per_room: dict[str, float]
    n_hours: int
    n_nights: int
    rel_se: float
    n_airing_dropped: int = 0


def _drop_airing_nights(per_room_hours: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], int]:
    """Remove room-nights whose free-cooling rate k = 1/τ is > AIRING_K_RATIO × that night's median k
    across rooms with k > 0 (≥ AIRING_MIN_ROOMS of them). Weather and envelope are shared on a night;
    an open window is room-specific, so a much faster-cooling room that night was aired. No extra
    sensor needed. Validated 2026-10-05..13: flags 5 bedroom nights + 1 bathroom night, 0 elsewhere.
    Rooms with k ≤ 0 (barely cooling / warming) are never flagged and don't enter the median."""
    k_rn: dict[tuple, float] = {}
    for eid, s in per_room_hours.items():
        for night, g in s.groupby("night"):
            if len(g) < MIN_HOURS_PER_NIGHT:
                continue
            k_rn[(eid, night)] = float((g["x"] * g["dT"]).sum()) / float((g["x"] ** 2).sum())
    pos_by_night: dict = {}
    for (eid, night), k in k_rn.items():
        if k > 0:
            pos_by_night.setdefault(night, []).append(k)
    drop = {key for key, k in k_rn.items()
            if len(pos_by_night.get(key[1], [])) >= AIRING_MIN_ROOMS
            and k > AIRING_K_RATIO * float(np.median(pos_by_night[key[1]]))}
    kept = {eid: s[~s["night"].map(lambda n, e=eid: (e, n) in drop)] for eid, s in per_room_hours.items()}
    return kept, len(drop)


def _usable_hours(c: pd.DataFrame, t_out: pd.Series) -> pd.DataFrame:
    h = (c.assign(timestamp=pd.to_datetime(c["timestamp"]).dt.floor("1h"))
           .groupby("timestamp")[["current_temp", "heating_power_request"]].last().asfreq("1h"))
    hpr = h["heating_power_request"]
    closed = (hpr == 0) & (hpr.shift(1) == 0) & (hpr.shift(2) == 0) & (hpr.shift(-1) == 0)  # NaN → False
    hours = h.index.hour
    night = (hours >= NIGHT_START_H) | (hours < NIGHT_END_H)
    t_in = h["current_temp"]
    d_t = t_in.shift(-1) - t_in
    x = t_out.reindex(h.index, method="nearest", tolerance=pd.Timedelta("1h")) - t_in
    ok = closed & night & (d_t.abs() < MAX_STEP_K_PER_H) & (-x >= MIN_INDOOR_EXCESS_K) & d_t.notna() & x.notna()
    out = pd.DataFrame({"x": x[ok], "dT": d_t[ok]})
    out["night"] = (out.index - pd.Timedelta(hours=NIGHT_END_H + 1)).date
    return out


def _fit_k(s: pd.DataFrame) -> tuple[float, float, int]:
    """OLS through origin of dT on x, one symmetric 3σ residual trim. Returns (k, se, n)."""
    def ols(d):
        sxx = float((d["x"] ** 2).sum())
        k = float((d["x"] * d["dT"]).sum()) / sxx
        resid = d["dT"] - k * d["x"]
        return k, resid, sxx
    k, resid, _ = ols(s)
    sigma = float(resid.std(ddof=1)) or 0.0
    if sigma > 0:
        s = s[(resid.abs() <= TRIM_SIGMA * sigma).values]
    k, resid, sxx = ols(s)
    se = float(np.sqrt((resid**2).sum() / max(len(s) - 1, 1) / sxx))
    return k, se, len(s)


def estimate_tau_pooled(
    climate_dfs: dict[str, pd.DataFrame],
    weather_df: pd.DataFrame,
    room_areas: dict[str, float] | None = None,
) -> TauEstimate | None:
    """House τ from per-room OLS-through-origin of ΔT_in on (T_out − T_in) over valve-closed night
    hours, after dropping aired room-nights (cross-room rule). Rooms combine by area-weighted mean of k = 1/τ (heat-loss rates
    add). Rooms with k ≤ 0 contribute k = 0 but keep their weight (dropping them would bias k up).
    Returns None when no room has usable valve data or the house k is ≤ 0.
    """
    w = weather_df.assign(timestamp=pd.to_datetime(weather_df["timestamp"]).dt.floor("1h"))
    t_out = w.groupby("timestamp")["temp_c"].mean()
    usable = {eid: _usable_hours(c, t_out) for eid, c in climate_dfs.items()
              if not c.empty and "heating_power_request" in c.columns}
    usable, n_airing = _drop_airing_nights(usable)
    ks, ses, areas, per_room, n_total = [], [], [], {}, 0
    night_hours: dict = {}
    for eid, s in usable.items():
        if len(s) < MIN_ROOM_HOURS:
            continue
        k, se, n = _fit_k(s)
        ks.append(max(k, 0.0)); ses.append(se)
        areas.append((room_areas or {}).get(eid, DEFAULT_ROOM_AREA_M2))
        if k > 0:
            per_room[eid] = 1.0 / k
        n_total += n
        for d in s["night"]:
            night_hours[d] = night_hours.get(d, 0) + 1
    if not ks:
        return None
    wts = np.asarray(areas) / float(np.sum(areas))
    k_house = float(np.dot(wts, ks))
    if k_house <= 0:
        return None
    se_house = float(np.sqrt(np.dot(wts**2, np.asarray(ses) ** 2)))
    n_nights = sum(1 for v in night_hours.values() if v >= MIN_HOURS_PER_NIGHT)
    return TauEstimate(1.0 / k_house, per_room, n_total, n_nights, se_house / k_house, n_airing)
```

`model.py`:
1. **Refactor:** `_blend_tau(self, old_tau, tau_raw, confidence, latest_ts) -> float` holds `new_weight = confidence * self._TAU_EMA_MAX_NEW_WEIGHT`, the EMA formula, and the nested `_apply_drift_cap` (closing over `old_tau`/`latest_ts`) applied for both caps. It returns the capped result; when `old_tau` is None/≤ 0 it returns `tau_raw` (the old `else: tau_result = tau_median` branch). In the windows path, keep the confidence computation and the "~0 % confidence" / "τ EMA blend" log lines in `_calibrate_tau`, emitted before calling `_blend_tau(old_tau, tau_median, confidence, combined.index.max())`. Then run `TestCalibrateTau`, `TestTauCalibrationSafeguards`, `TestTauDriftCap` (incl. `test_tau_calibration_log_shows_actual_percentage`) → must be green with no test edits. Commit this refactor on its own (`refactor: extract _blend_tau`).
2. Signature: `_calibrate_tau(self, climate_dfs, heating_active_df, weather_df, room_areas=None, tau_estimator="windows")`. Insert the pooled block **after** the function-local `import numpy as np` / `import pandas as pd`:
```python
        if tau_estimator == "valve_pooled":
            from . import tau_estimation as te

            pooled = te.estimate_tau_pooled(climate_dfs, weather_df, room_areas)
            if pooled is not None and pooled.n_nights >= TAU_POOLED_MIN_NIGHTS:
                confidence = min(1.0, pooled.n_hours / TAU_POOLED_FULL_CONF_HOURS) * float(
                    np.clip(1.0 - (pooled.rel_se - 0.25) / 0.25, 0.0, 1.0))
                latest_ts = max(pd.to_datetime(c["timestamp"]).max() for c in climate_dfs.values() if not c.empty)
                _LOGGER.info("τ (valve_pooled): raw %.1f h from %d h / %d nights (%d aired room-nights dropped), "
                             "rel_se %.0f%%, confidence %.0f%%, per-room %s",
                             pooled.tau_hours, pooled.n_hours, pooled.n_nights, pooled.n_airing_dropped,
                             pooled.rel_se * 100, confidence * 100,
                             {k.split(".")[-1]: round(v) for k, v in pooled.per_room.items()})
                if self._tau_method != "valve_pooled":
                    if confidence < 0.5:
                        return self._tau_hours
                    _LOGGER.warning("τ method switch windows → valve_pooled: %s h → %.1f h (anchors reset)",
                                    self._tau_hours, pooled.tau_hours)
                    self._tau_method = "valve_pooled"
                    self._tau_anchor_hours = self._tau_long_anchor_hours = pooled.tau_hours
                    self._tau_anchor_ts = self._tau_long_anchor_ts = latest_ts
                    return pooled.tau_hours
                return self._blend_tau(self._tau_hours, pooled.tau_hours, confidence, latest_ts)
            if self._tau_method == "valve_pooled":
                _LOGGER.info("τ: valve data insufficient this retrain — keeping stored %s h", self._tau_hours)
                return self._tau_hours
            _LOGGER.info("τ: tau_estimator=valve_pooled but no usable valve data yet — legacy estimator")
        if heating_active_df is None or heating_active_df.empty:
            return self._tau_hours
```
   Import `TAU_POOLED_MIN_NIGHTS, TAU_POOLED_FULL_CONF_HOURS` in the module's `from .const import (...)` block (the existing `_TAU_*` are class attributes and stay there).
3. `__init__`: `self._tau_method: str = "windows"`. Meta save `"tau_method": self._tau_method`; load `self._tau_method = meta.get("tau_method", "windows")`.
4. Call site (~L433): `if climate_dfs: self._tau_hours = self._calibrate_tau(climate_dfs, heating_active_df, weather_df, room_areas=room_areas, tau_estimator=tau_estimator)`. Keep the existing debug logs for the skip cases. Behaviour with the default flag and an empty `heating_active_df` is unchanged (returns stored τ, as the skipped call did).
5. `energy_forecast.py`: `self._tau_estimator = str(self.args.get("tau_estimator", "windows"))`, validated ∈ {windows, valve_pooled}, passed to every `self._ml_model.train(`.

- [ ] **Step 4: Full suite + ruff → PASS. Commit** `feat: valve-closed pooled τ estimator behind tau_estimator flag`.

- [ ] **Step 5: Gate (local, `scripts/gate_tau_pooled.py`)**. Run once ≥ 14 nights of `heating_power_request` are cached. It uses the **shipped** estimator on the **cached** (hourly) data:

```python
#!/usr/bin/env python3
"""Plan D Task 3 gate: 9 h free-cooling prediction of indoor temp on held-out valve-closed nights."""
import pathlib, sys, types
from datetime import date, timedelta
from unittest.mock import MagicMock
REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "apps"))
_s = types.ModuleType("hassapi"); _s.Hass = MagicMock; sys.modules.setdefault("hassapi", _s)
import numpy as np, pandas as pd, yaml  # noqa: E402
from energy_forecast import tau_estimation as te  # noqa: E402
from energy_forecast.weather import fetch_historical_weather  # noqa: E402

cfg = yaml.safe_load((REPO / "data/apps.yaml").read_text())["energy_forecast"]
areas = cfg.get("climate_room_areas") or {}
clim = {}
for eid in cfg["climate_entities"]:
    d = pd.read_csv(REPO / f"data/climate_{eid.split('.', 1)[1]}.csv")
    d["timestamp"] = pd.to_datetime(d["timestamp"], format="mixed")
    if "heating_power_request" in d.columns:
        clim[eid] = d
if not clim:
    raise SystemExit("no heating_power_request in cached climate files yet")
start = min(d.loc[d["heating_power_request"].notna(), "timestamp"].min() for d in clim.values()).date()
w = fetch_historical_weather(cfg["latitude"], cfg["longitude"], start, date.today() - timedelta(days=3), timezone="Europe/Zurich")
nights = sorted({(t - pd.Timedelta(hours=7)).date() for d in clim.values()
                 for t in d.loc[d["heating_power_request"].eq(0) & d["timestamp"].dt.hour.isin([22, 23, 0, 1, 2]), "timestamp"]})
cut = pd.Timestamp(nights[int(len(nights) * 0.75)])
est = te.estimate_tau_pooled({k: d[d["timestamp"] < cut] for k, d in clim.items()}, w[w["timestamp"] < cut], areas)
if est is None:
    raise SystemExit("GATE: STOP — pooled estimator returned None on training nights")
tau_old = float(sys.argv[1]) if len(sys.argv) > 1 else 11.64
t_out = w.set_index("timestamp")["temp_c"]
variants = {"old": lambda e: tau_old, "pooled": lambda e: est.tau_hours,
            "pooled_per_room": lambda e: est.per_room.get(e, est.tau_hours),
            "fixed_40": lambda e: 40.0, "fixed_90": lambda e: 90.0, "persistence": lambda e: np.inf}
errs = {k: [] for k in variants}; wins = {k: 0 for k in variants}
for eid, d in clim.items():
    h = d.set_index(d["timestamp"].dt.floor("1h"))[["current_temp", "heating_power_request"]].groupby(level=0).last().asfreq("1h")
    for n in [x for x in nights if pd.Timestamp(x) >= cut]:
        idx = pd.date_range(pd.Timestamp(n) + pd.Timedelta(hours=21), periods=10, freq="1h")
        seg = h.reindex(idx)
        if seg["heating_power_request"].ne(0).any() or seg["current_temp"].isna().any() or t_out.reindex(idx[:-1]).isna().any():
            continue
        night_err = {}
        for name, f in variants.items():
            tau = f(eid); t = seg["current_temp"].iloc[0]; pred = [t]
            for ts in idx[:-1]:
                t = t + (t_out[ts] - t) / tau; pred.append(t)
            night_err[name] = float(np.sqrt(np.mean((np.array(pred) - seg["current_temp"].values) ** 2)))
            errs[name].append(night_err[name])
        wins["pooled"] += night_err["pooled"] < night_err["old"]
n = len(errs["pooled"])
print({k: (round(float(np.nanmean(v)), 3), len(v)) for k, v in errs.items()}, "τ_pooled", round(est.tau_hours, 1),
      f"pooled beats old on {wins['pooled']}/{n} room-nights")
m = {k: float(np.nanmean(v)) for k, v in errs.items()}
go = n >= 20 and m["pooled"] <= 0.8 * m["old"] and m["pooled"] <= m["persistence"] and wins["pooled"] >= 0.7 * n
print("GATE:", "GO" if go else "STOP", "| per-room better than house τ by ≥10 %:", m["pooled_per_room"] < 0.9 * m["pooled"])
```
GO → set `tau_estimator: valve_pooled` in live apps.yaml (backup first) and restart. Then confirm the `τ method switch` warning in the log after the next retrain, and that `sensor…thermal_pressure_net` attr `tau_hours` changed. STOP → the flag stays `windows`; report the table. If per-room beats house τ by ≥ 10 %, open a follow-up for `tau_by_room` in `_project_indoor_temps` (not in this plan). Also record `n_airing_dropped` and the bedroom's per-room τ with the rule on vs off (patch `AIRING_MIN_ROOMS` to 99 to disable it). Expect the bedroom's τ to move from ≈ 40 h toward the other rooms' range and roughly 1 flagged room-night per aired night. If rooms other than bedroom/bathroom get flagged on > 10 % of their nights, raise `AIRING_K_RATIO` to 2.5 and re-run.

---

### Task 4: Daily heat-balance `UA_eff` behind `physics.ua_eff_method` flag (`feat/ua-eff-daily`)

**Files:**
- Modify: `apps/energy_forecast/physics.py`:
  - new `_calibrate_ua_eff_daily` and `_area_weighted_t_indoor_strict`;
  - `calibrate()` (~L1134–1185) gains `room_areas` and `ua_window_end`, and picks the method by `self._config["ua_eff_method"]`;
  - new `ua_calibration_stale` + `recalibrate_ua`;
  - `_space_heating_kwh` (~L229): decouple `c_building` from the ML τ.
  - Rationale: UA calibration lives here; the decoupling prevents review hazard F3.
- Modify: `apps/energy_forecast/model.py` (~L452–467). Rationale: pass `ua_window_end=energy_df["timestamp"].max() + pd.Timedelta(hours=1)` and `room_areas`; weekly `recalibrate_ua` when not otherwise stale.
- Modify: `apps/energy_forecast/energy_forecast.py`. Rationale: physics config default `"ua_eff_method": "hourly"` (next to `heating_sub_meter_sensor` at ~L338), validated ∈ {hourly, daily}.
- Modify: `apps/energy_forecast/const.py` — `UA_EFF_DAILY_WINDOW_DAYS = 28`, `UA_EFF_DAILY_MIN_DAYS = 10`, `UA_EFF_DAILY_MIN_MEAN_DT_K = 8.0`, `UA_EFF_DAILY_MIN_R2 = 0.4`, `UA_EFF_DAILY_MIN_W_PER_K = 15.0`, `UA_EFF_RECAL_DAYS = 7`, `C_BUILDING_DEFAULT_TAU_H = 8.0`
- Test: `tests/test_physics.py` (new class `TestUaEffDaily`)

**Interfaces:**
- Produces: `ThermalPhysicsModel._calibrate_ua_eff_daily(climate_dfs, weather_df, heating_sub_meter_df, window_end, room_areas=None, heating_active_df=None) -> tuple[float | None, int]`. It returns UA in W/K and the number of eligible heating days in the **whole** history, which feeds `n_calibration_windows_ua_eff` / the Phase 2 cold-start gate. For new installs, that gate's unit therefore changes from night-hour windows to heating days; document it in the CHANGELOG. Live has ~57 eligible days, so it stays ≥ 30.
- Produces: `_calib["ua_calibrated_at"]` (ISO), `_calib["ua_eff_method"]`; `ua_calibration_stale -> bool`; `recalibrate_ua(energy_df, climate_dfs, weather_df, heating_sub_meter_df, window_end, room_areas=None, heating_active_df=None) -> None`

**Design decisions** (from review):
- **Window end = data end**, not the ML holdout. The UA fit is a one-parameter through-origin fit on the heating meter, a different target from the ML model, so holdout leakage is irrelevant. Using the holdout would leave 0 heating days in the window today (F1).
- **Day eligibility uses x, not y:** a day counts if it has ≥ 22 h of both climate and meter coverage, **and** is a heating day by `heating_active_df` (daily mean ≥ 0.5) when available, else by mean ΔT ≥ `UA_EFF_DAILY_MIN_MEAN_DT_K` (8 K ≈ outdoor daily mean < 12–13 °C). No `kwh ≥` threshold, because that selects on the target. Verified on real data: 4 K admitted 104 mild days with the heat pump off and dragged UA to 14 W/K; 8 K reproduces the switch-based spring value (61.8 vs 60.5 W/K).
- **`c_building` decoupled:** `c_building = config.c_building_wh_k or UA · C_BUILDING_DEFAULT_TAU_H` (8 h = today's τ-less default), no longer `UA · self._tau_hours`. The room-air cooling τ from Task 3 is not the building-mass τ (F3). This is the only change to `_space_heating_kwh`.
- **`recalibrate_ua` updates UA only.** It doesn't re-run `solar_gain_area` (never active live; it runs on the 30-day full `calibrate()` as today). The gate logs its value once UA is non-null.
- Training `physics_kwh` uses the *current* UA for all history (as today's design does for any UA). The feature's scale therefore drifts weekly. The Step 5 gate covers that.

- [ ] **Step 1: Failing tests**

```python
def _daily_fixture(days=40, start="2026-04-01", t_in=20.0):
    idx = pd.date_range(start, periods=days * 24, freq="1h")
    rng = np.random.default_rng(3)
    t_out = 8.0 + 4 * np.sin(np.arange(len(idx)) * 2 * np.pi / 24) + rng.normal(0, 1, len(idx))
    weather = pd.DataFrame({"timestamp": idx, "temp_c": t_out, "direct_radiation_wm2": 0.0})
    clim = {"climate.a": pd.DataFrame({"timestamp": idx, "current_temp": t_in, "setpoint": t_in})}
    return idx, weather, clim, t_out


def _meter_for(m, idx, t_out, ua):
    cop = m._cop_series(idx, pd.Series(t_out, index=idx), cop_sensor_series=None).values
    ua = np.broadcast_to(ua, len(idx))
    kwh = ua * np.clip(20.0 - t_out, 0, None) / cop / 1000.0
    daily = pd.Series(kwh, index=idx).resample("1D").sum()
    bursts = pd.Series(0.0, index=idx)            # buffer tank: energy in 2-h bursts, daily sum unchanged
    for d, v in daily.items():
        bursts[d + pd.Timedelta(hours=13)] = v / 2
        bursts[d + pd.Timedelta(hours=14)] = v / 2
    return pd.DataFrame({"timestamp": idx, "kwh": bursts.values})


class TestUaEffDaily:
    def _m(self, tmp_path):
        return ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)

    def test_recovers_ua_in_formula_cop_units(self, tmp_path):
        m = self._m(tmp_path); idx, weather, clim, t_out = _daily_fixture()
        ua, n = m._calibrate_ua_eff_daily(clim, weather, _meter_for(m, idx, t_out, 80.0), window_end=idx[-1] + pd.Timedelta(hours=1))
        assert ua == pytest.approx(80.0, rel=0.05) and n == 40

    def test_partial_climate_day_dropped(self, tmp_path):
        m = self._m(tmp_path); idx, weather, clim, t_out = _daily_fixture(days=12)
        clim["climate.a"] = clim["climate.a"].iloc[5:]          # day 1 has only 19 h
        _, n = m._calibrate_ua_eff_daily(clim, weather, _meter_for(m, idx, t_out, 80.0), window_end=idx[-1] + pd.Timedelta(hours=1))
        assert n == 11

    def test_partial_meter_day_dropped(self, tmp_path):
        m = self._m(tmp_path); idx, weather, clim, t_out = _daily_fixture(days=12)
        meter = _meter_for(m, idx, t_out, 80.0).iloc[5:]        # meter missing first 5 h
        _, n = m._calibrate_ua_eff_daily(clim, weather, meter, window_end=idx[-1] + pd.Timedelta(hours=1))
        assert n == 11

    def test_zero_kwh_heating_day_still_counts(self, tmp_path):
        m = self._m(tmp_path); idx, weather, clim, t_out = _daily_fixture(days=12)
        meter = _meter_for(m, idx, t_out, 80.0)
        meter.loc[meter["timestamp"].dt.date == idx[0].date(), "kwh"] = 0.0   # no selection on y
        _, n = m._calibrate_ua_eff_daily(clim, weather, meter, window_end=idx[-1] + pd.Timedelta(hours=1))
        assert n == 12

    def test_too_few_days_returns_none_but_counts(self, tmp_path):
        m = self._m(tmp_path); idx, weather, clim, t_out = _daily_fixture(days=6)
        ua, n = m._calibrate_ua_eff_daily(clim, weather, _meter_for(m, idx, t_out, 80.0), window_end=idx[-1] + pd.Timedelta(hours=1))
        assert ua is None and n == 6

    def test_rolling_window_uses_last_28_days_only(self, tmp_path):
        m = self._m(tmp_path); idx, weather, clim, t_out = _daily_fixture(days=60)
        ua_true = np.where(idx < idx[0] + pd.Timedelta(days=32), 150.0, 60.0)   # seasonal drop
        ua, n = m._calibrate_ua_eff_daily(clim, weather, _meter_for(m, idx, t_out, ua_true), window_end=idx[-1] + pd.Timedelta(hours=1))
        assert ua == pytest.approx(60.0, rel=0.05) and n == 60

    def test_low_effective_ua_accepted_with_warning(self, tmp_path, caplog):
        m = self._m(tmp_path); idx, weather, clim, t_out = _daily_fixture()
        ua, _ = m._calibrate_ua_eff_daily(clim, weather, _meter_for(m, idx, t_out, 28.0), window_end=idx[-1] + pd.Timedelta(hours=1))
        assert ua == pytest.approx(28.0, rel=0.05) and "below envelope plausibility" in caplog.text

    def test_strict_indoor_leaves_gaps_nan(self, tmp_path):
        m = self._m(tmp_path)
        ts = pd.date_range("2026-04-01", periods=6, freq="1h")
        clim = {"climate.a": pd.DataFrame({"timestamp": ts[[0, 1, 5]], "current_temp": [20.0, 20.0, 21.0]})}
        t_in = m._area_weighted_t_indoor_strict(clim, ts, None)
        assert t_in.isna().tolist() == [False, False, True, True, True, False]

    def test_calibrate_daily_flag_with_real_cutoff_shape(self, tmp_path):
        cfg = {**DEFAULT_CONFIG, "ua_eff_method": "daily"}
        m = ThermalPhysicsModel(tmp_path / "models", cfg)
        idx, weather, clim, t_out = _daily_fixture(days=60, start="2026-08-20")
        meter = _meter_for(m, idx, t_out, 80.0)
        meter.loc[meter["timestamp"] < pd.Timestamp("2026-10-08"), "kwh"] = 0.0     # summer: no heating
        warm = weather["timestamp"] < pd.Timestamp("2026-10-08")
        weather.loc[warm, "temp_c"] = 22.0                                            # not heating days by ΔT
        energy = pd.DataFrame({"timestamp": idx, "gross_kwh": 0.5})
        m.calibrate(energy, weather, clim, None,
                    holdout_cutoff=idx[-1] - pd.Timedelta(days=37),   # the ML holdout — must NOT bound the UA window
                    heating_sub_meter_df=meter, ua_window_end=idx[-1] + pd.Timedelta(hours=1))
        assert m._calib["ua_eff_method"] == "daily" and m._calib["UA_eff"] == pytest.approx(80.0, rel=0.1)
        assert "ua_calibrated_at" in m._calib

    def test_default_flag_keeps_hourly(self, tmp_path):
        m = self._m(tmp_path); idx, weather, clim, t_out = _daily_fixture()
        energy = pd.DataFrame({"timestamp": idx, "gross_kwh": 0.5})
        m.calibrate(energy, weather, clim, None, holdout_cutoff=idx[-1], heating_sub_meter_df=_meter_for(m, idx, t_out, 80.0))
        assert m._calib["ua_eff_method"] == "hourly"

    def test_ua_staleness_weekly_and_missing_key_is_stale(self, tmp_path):
        m = self._m(tmp_path)
        assert m.ua_calibration_stale
        m._calib["ua_calibrated_at"] = (pd.Timestamp.now() - pd.Timedelta(days=8)).isoformat()
        assert m.ua_calibration_stale
        m._calib["ua_calibrated_at"] = (pd.Timestamp.now() - pd.Timedelta(days=2)).isoformat()
        assert not m.ua_calibration_stale

    def test_c_building_independent_of_ml_tau(self, tmp_path):
        m = self._m(tmp_path); m._calib["UA_eff"] = 28.0
        idx, weather, clim, t_out = _daily_fixture(days=2)
        t_in = pd.Series(20.0 + 0.2 * np.sin(np.arange(len(idx))), index=idx)
        args = dict(t_indoor=t_in, t_outdoor=pd.Series(t_out, index=idx), ghi=pd.Series(0.0, index=idx),
                    cop=pd.Series(3.2, index=idx))
        m._tau_hours = 11.6; a = m._space_heating_kwh(**args)
        m._tau_hours = 92.0; b = m._space_heating_kwh(**args)
        pd.testing.assert_series_equal(a, b)
```
(`_space_heating_kwh(self, t_indoor, t_outdoor, ghi, cop)`, verified.) **No existing test needs updating:** `test_calibrate_ua_eff_prefers_sub_meter_when_available` and siblings call `_calibrate_ua_eff` directly, and the default flag keeps `calibrate()` on the hourly path (verified by review).

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement** (`physics.py`):

```python
    def _area_weighted_t_indoor_strict(
        self, climate_data: dict[str, pd.DataFrame], timestamps: pd.DatetimeIndex, room_areas: dict[str, float] | None
    ) -> pd.Series:
        """Like _area_weighted_t_indoor, but hours without a reading within ±1 h stay NaN and the
        weighting uses only rooms present at each hour (no unbounded 'nearest' fill)."""
        num = pd.Series(0.0, index=timestamps); den = pd.Series(0.0, index=timestamps)
        for eid, cdf in (climate_data or {}).items():
            if cdf.empty:
                continue
            c = cdf.assign(timestamp=pd.to_datetime(cdf["timestamp"])).groupby("timestamp")["current_temp"].last()
            v = c.reindex(timestamps, method="nearest", tolerance=pd.Timedelta("59min"))
            a = (room_areas or {}).get(eid, DEFAULT_ROOM_AREA_M2)
            num += v.fillna(0.0) * a * v.notna(); den += a * v.notna()
        return (num / den).where(den > 0)

    def _calibrate_ua_eff_daily(self, climate_dfs, weather_df, heating_sub_meter_df, window_end,
                                room_areas=None, heating_active_df=None) -> tuple[float | None, int]:
        """Effective UA (W/K, formula-COP units) from a DAILY heat balance on the heating sub-meter:
            meter_kWh(day) = UA/1000 · Σ_h max(0, T_in − T_out)/COP_h
        Daily sums absorb buffer-tank and HP-cycling lag that defeat the hourly fit. Rolling window of
        UA_EFF_DAILY_WINDOW_DAYS ending at window_end (data end — not the ML holdout): effective UA is
        seasonal (2026: spring 21–28 W/K electrical, October 8–9). Days are selected on x (heating
        season / ΔT), never on the meter value. Returns (UA | None, eligible heating days in history).
        """
        if not climate_dfs or heating_sub_meter_df is None or heating_sub_meter_df.empty:
            return None, 0
        hs = heating_sub_meter_df.assign(timestamp=pd.to_datetime(heating_sub_meter_df["timestamp"]).dt.floor("1h"))
        meter = hs.groupby("timestamp")["kwh"].sum()
        meter = meter[meter.index < window_end]
        if meter.empty:
            return None, 0
        ts = pd.date_range(meter.index.min().floor("1D"), window_end.floor("1h") - pd.Timedelta(hours=1), freq="1h")
        w = weather_df.set_index(pd.to_datetime(weather_df["timestamp"]))
        t_out = w["temp_c"].reindex(ts, method="nearest", tolerance=pd.Timedelta("1h"))
        t_in = self._area_weighted_t_indoor_strict(climate_dfs, ts, room_areas)
        cop = self._cop_series(ts, t_out, cop_sensor_series=None)
        m = meter.reindex(ts)
        frame = pd.DataFrame({"x": np.clip(t_in - t_out, 0, None) / cop, "dt": (t_in - t_out),
                              "ok": (t_in.notna() & t_out.notna() & m.notna()).astype(int), "kwh": m.fillna(0.0)})
        if heating_active_df is not None and not heating_active_df.empty:
            ha = heating_active_df.assign(timestamp=pd.to_datetime(heating_active_df["timestamp"]).dt.floor("1h"))
            frame["on"] = ha.groupby("timestamp")["heating_active"].last().reindex(ts).ffill()
        day = frame.resample("1D").agg({"x": "sum", "dt": "mean", "ok": "sum", "kwh": "sum",
                                        **({"on": "mean"} if "on" in frame else {})})
        heating = day["on"] >= 0.5 if "on" in day else day["dt"] >= UA_EFF_DAILY_MIN_MEAN_DT_K
        day = day[(day["ok"] >= 22) & heating]
        n_days = len(day)
        win = day[day.index >= window_end.floor("1D") - pd.Timedelta(days=UA_EFF_DAILY_WINDOW_DAYS)]
        if len(win) < UA_EFF_DAILY_MIN_DAYS:
            _LOGGER.info("UA_eff daily: %d heating days in last %d d (need %d) — skipping",
                         len(win), UA_EFF_DAILY_WINDOW_DAYS, UA_EFF_DAILY_MIN_DAYS)
            return None, n_days
        x, y = win["x"].values, win["kwh"].values
        slope = float(np.sum(x * y) / np.sum(x * x))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - float(np.sum((y - slope * x) ** 2)) / ss_tot if ss_tot > 1e-9 else 0.0
        ua = slope * 1000.0
        _LOGGER.info("UA_eff calibration (daily, %d d window): %d days, UA_eff=%.1f W/K, R²=%.2f",
                     UA_EFF_DAILY_WINDOW_DAYS, len(win), ua, r2)
        if r2 < UA_EFF_DAILY_MIN_R2 or not (UA_EFF_DAILY_MIN_W_PER_K <= ua <= UA_EFF_SANITY_MAX_W_PER_K):
            _LOGGER.warning("UA_eff daily: rejected (UA %.1f W/K, R² %.2f)", ua, r2)
            return None, n_days
        if ua < UA_EFF_SANITY_MIN_W_PER_K:
            _LOGGER.warning("UA_eff daily %.1f W/K below envelope plausibility (%.0f) — effective value absorbs "
                            "formula-COP error and gains; used as-is for consistency", ua, UA_EFF_SANITY_MIN_W_PER_K)
        return ua, n_days
```
(With the fixture's constant T_in = 20 and T_out ≈ 8 ± 4, `dt` ≈ 12 ≥ 8, so all days are heating days. In `test_calibrate_daily_flag_with_real_cutoff_shape` the warm summer days have dt ≈ −2, so they're not heating days; that test also exercises `window_end` vs the holdout.)

Add to `physics.py`'s `from .const import (...)` block: `C_BUILDING_DEFAULT_TAU_H, DEFAULT_ROOM_AREA_M2, UA_EFF_DAILY_MIN_DAYS, UA_EFF_DAILY_MIN_MEAN_DT_K, UA_EFF_DAILY_MIN_R2, UA_EFF_DAILY_MIN_W_PER_K, UA_EFF_DAILY_WINDOW_DAYS, UA_EFF_RECAL_DAYS`.

`calibrate()`: add params `room_areas: dict | None = None, ua_window_end: pd.Timestamp | None = None`. Replace the `ua_eff, n_windows = self._calibrate_ua_eff(...)` block with:
```python
            method = self._config.get("ua_eff_method", "hourly")
            if method == "daily" and heating_sub_meter_df is not None and not heating_sub_meter_df.empty:
                ua_eff, n_windows = self._calibrate_ua_eff_daily(
                    climate_dfs, weather_df, heating_sub_meter_df,
                    ua_window_end if ua_window_end is not None else pd.to_datetime(energy_df["timestamp"]).max() + pd.Timedelta(hours=1),
                    room_areas=room_areas, heating_active_df=heating_active_df)
            else:
                if method == "daily":
                    _LOGGER.warning("physics.ua_eff_method=daily needs physics.heating_sub_meter_sensor — using hourly")
                    method = "hourly"
                ua_eff, n_windows = self._calibrate_ua_eff(
                    energy_df, weather_df, climate_dfs, dhw_df, holdout_cutoff,
                    heating_active_df=heating_active_df, heating_sub_meter_df=heating_sub_meter_df,
                    ev_df=ev_df, away_df=away_df)
            self._calib["n_calibration_windows_ua_eff"] = n_windows
            self._calib["ua_eff_method"] = method
            self._calib["ua_calibrated_at"] = pd.Timestamp.now().isoformat()
            if ua_eff is not None:
                self._calib["UA_eff"] = ua_eff
            else:
                _LOGGER.warning("UA_eff calibration unavailable — heating component will be skipped")
```
`ua_calibration_stale`: True if `ua_calibrated_at` is missing or unparsable, or older than `UA_EFF_RECAL_DAYS` (mirror `calibration_stale`). `recalibrate_ua(...)`: runs the daily branch only, updates `UA_eff` (if not None), `n_calibration_windows_ua_eff`, `ua_eff_method`, `ua_calibrated_at`, and saves via the same save call `calibrate()` ends with. It's a no-op unless `ua_eff_method == "daily"`.
`_space_heating_kwh`: `c_building = self._config.get("c_building_wh_k") or (ua * C_BUILDING_DEFAULT_TAU_H)`; drop the `tau = self._tau_hours or 8.0` line if nothing else uses it there.
`model.py` (~L452): pass `room_areas=room_areas, ua_window_end=energy_df["timestamp"].max() + pd.Timedelta(hours=1)` to `calibrate`, then add `elif physics_model.ua_calibration_stale: physics_model.recalibrate_ua(energy_df, climate_dfs, weather_df, heating_sub_meter_df, energy_df["timestamp"].max() + pd.Timedelta(hours=1), room_areas=room_areas, heating_active_df=heating_active_df)`.

- [ ] **Step 3b: NaN-robust indoor averaging + DHW ODE guard** (folded in 2026-10-19; separate commit `fix: NaN-robust indoor temperature for physics series and DHW ODE`).

  **Why:** found 2026-10-19. `physics_kwh` has 0 gain / 0 splits (rank 88/88) in the live model. `_area_weighted_t_indoor` uses `np.average`, so one room's NaN `current_temp` makes that hour's house average NaN. The live climate CSVs have 429 such hours across 9 of 10 rooms, first at 2026-04-12 07:00. `_dhw_kwh_series` carries `t_tank` forward, so the first NaN poisons every later hour: comparisons are False, `el_kwh` is 0, and the training `physics_kwh` stays a constant 0.144 (Q_base_el). Once this task makes `UA_eff` non-null, the same NaN hours would also make `_space_heating_kwh`, and so `physics_kwh`, NaN at those hours; the model.py merge fills them with 0, dropping base load too. Step 3's strict helper covers only the daily calibrator, so it doesn't fix this. Evidence: `memory/project_physics_kwh_low_importance.md`.

  **Files:**
  - `physics.py`: rewrite `_area_weighted_t_indoor` to share the per-hour num/den weighting with `_area_weighted_t_indoor_strict`, through one private helper with a `tolerance` argument:
    - strict: `tolerance=59min`, gaps stay NaN;
    - non-strict: unbounded `nearest`, then `.ffill().bfill()`, then `DEFAULT_AMBIENT_C`. It never returns NaN, and keeps the existing `None` when there are no rooms.
  - `physics.py` `_dhw_kwh_series`: treat a non-finite `t_ambient.iloc[i]` as `DEFAULT_AMBIENT_C` (defence in depth for any caller).
  - `physics.py` solar-gain calibration (~L873) and hourly `_calibrate_ua_eff` (~L781): pass `room_areas` through instead of `None`, now that `calibrate()` takes it (Step 3). This aligns calibration with the area weighting used at forecast time.

  **Tests** (`tests/test_physics.py`, new class `TestIndoorNanRobustness`):
  ```python
  def test_one_room_nan_uses_remaining_rooms(self, tmp_path):
      m = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
      ts = pd.date_range("2026-04-01", periods=3, freq="1h")
      clim = {"climate.a": pd.DataFrame({"timestamp": ts, "current_temp": [20.0, np.nan, 20.0]}),
              "climate.b": pd.DataFrame({"timestamp": ts, "current_temp": [22.0, 22.0, 22.0]})}
      t = m._area_weighted_t_indoor(clim, ts, {"climate.a": 10.0, "climate.b": 10.0})
      assert t.tolist() == pytest.approx([21.0, 22.0, 21.0])

  def test_all_rooms_nan_hour_is_filled(self, tmp_path):
      m = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
      ts = pd.date_range("2026-04-01", periods=3, freq="1h")
      clim = {"climate.a": pd.DataFrame({"timestamp": ts, "current_temp": [20.0, np.nan, 21.0]})}
      assert not m._area_weighted_t_indoor(clim, ts, None).isna().any()

  def test_dhw_ode_survives_nan_ambient(self, tmp_path):
      m = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
      ts = pd.date_range("2026-04-01", periods=24 * 10, freq="1h")
      amb = pd.Series(20.0, index=ts); amb.iloc[5] = np.nan
      el, traj, final = m._dhw_kwh_series(ts, amb, 50.0, None)
      assert not traj.isna().any() and np.isfinite(final)
      assert (el.iloc[24:] > 0).sum() >= 3          # keeps cycling after the gap (was 0 before the fix)

  def test_training_series_varies_despite_room_gap(self, tmp_path):
      m = ThermalPhysicsModel(tmp_path / "models", DEFAULT_CONFIG)
      ts = pd.date_range("2026-04-01", periods=24 * 10, freq="1h")
      energy = pd.DataFrame({"timestamp": ts, "gross_kwh": 0.5})
      weather = pd.DataFrame({"timestamp": ts, "temp_c": 10.0, "direct_radiation_wm2": 0.0})
      temps = np.full(len(ts), 20.0); temps[5] = np.nan
      clim = {"climate.a": pd.DataFrame({"timestamp": ts, "current_temp": temps})}
      s = m.predict_training_series(energy, weather, climate_dfs=clim)
      assert s.iloc[24:].nunique() > 1
  ```
  Check whether `DEFAULT_CONFIG`'s Q_dhw_daily/UA_dhw defaults make the tank cycle within 10 days; if not, set `m._calib["Q_dhw_daily"] = 2.8` in the two ODE tests, matching the live calibration. Run the tests and confirm they fail before implementing.

  **Effect without UA (expected, not a win):** this brings back a DHW sawtooth in the training `physics_kwh`, but that open-loop ODE barely matches real reheat timing: 1 overlap out of 16 pre-NaN, while real reheats are solar-gated at 12–15 h. Don't expect importance to rise from 3b alone; the value comes from the heating term this task enables. Whether to keep DHW inside `physics_kwh` is a separate, open decision.

- [ ] **Step 4: Full suite + ruff → PASS. Commit** `feat: daily rolling heat-balance UA_eff behind physics.ua_eff_method`.

Expected live behaviour (verified on real data 2026-10-15): with the season switch, 35 eligible heating days in history (Phase 2 count stays ≥ 30), but only 6 in the last 28 days, so UA stays `null` until ~10 October heating days accumulate (~2026-10-18/19). That's correct. A spring window ending 05-21 gives 60.5 W/K (R² 0.60); one ending 05-01 is rejected at R² 0.38.

- [ ] **Step 5: Gate (local).** Requires the full-feature backtest harness `scripts/backtest_heat_loss_feature.py` from Plan C Task 4 Step 5. Create it from that plan's code if C Task 4 wasn't executed. The `heat_loss_feature` kwarg then stays False.
  - Add `physics_model=ThermalPhysicsModel(tmp / "phys", {**DEFAULT_PHYSICS_CONFIG, "ua_eff_method": v})` and `heating_sub_meter_df=<sub_em_kermi_bridge_kermi_electricity_heating>` to its `train` call, for v ∈ {hourly, daily}.
  - GO requires all of:
    - the 24 h-lead cold-fold MAE doesn't worsen by > 2 %, and the paired daily sign test is not worse (≤ 3 of 7 days worse);
    - physics' daily heating component tracks the heating meter with daily MAE < 50 % of the meter's daily mean on the held-out days;
    - the `solar_gain_area` printed after calibration is within [0, 20] m².
  - Both arms include Step 3b. Also run a third arm with `hourly` on the pre-3b code: that's the live baseline, which isolates 3b's effect. For each arm, record `physics_kwh` gain % and rank (`booster_.feature_importance("gain")`). Report it for information; it's not a GO criterion. Live baseline 2026-10-19: rank 88/88, 0 gain.
  - GO → set `physics.ua_eff_method: daily` live (backup first) and force recalibration (delete `physics_calibration.json`'s `calibrated_at`, or wait ≤ 7 days). STOP → flag stays `hourly`; record the table.

---

### Task 5: Gated ML feature — this season's heating intensity (`feat/heating-intensity-feature`)

**Files:**
- Modify: `apps/energy_forecast/model.py`:
  - `_heating_intensity` helper;
  - `train(..., heating_intensity_feature=False, heating_meter_prefix=None)`;
  - `_prepare_prediction_X` reads the meter from the existing `sub_sensors_recent[self._heating_meter_prefix]`;
  - meta persists `heating_meter_prefix` (a string, not a data tail).
  - Rationale: no new predict parameter or data tail is needed; `fetch_recent_sub_sensor` already returns the full merged cache to the predict path (review F7).
- Modify: `apps/energy_forecast/energy_forecast.py`. Rationale: config `heating_intensity_feature` (default false); pass `heating_meter_prefix=self._sub_sensor_prefix(physics.heating_sub_meter_sensor)` when configured and present in `sub_energy_sensors`, else warn and skip.
- Test: `tests/test_model.py` (`TestHeatingIntensity`)

**Why:** the same outdoor ΔT costs ~⅓ of the electricity in October that it did in spring (daily balance above). The model's only handle on that today is lags of the heating meter. A weather-normalised trailing ratio gives it the season's heating intensity directly.

**Interfaces:**
- Produces: `_heating_intensity(meter: pd.Series, t_out: pd.Series, at: pd.DatetimeIndex, lag_hours: int) -> pd.Series`. It computes Σ meter kWh / Σ max(0, 18 − T_out) over the 168 h window ending `lag_hours` before each `at`. Only hours with **both** meter and weather count, so a weather gap between the stored `_weather_tail` and the forecast can't bias the ratio. It needs ≥ 120 such hours and ≥ 24 degree-hours, else 0.0. On real data it gives spring 0.015–0.022 and October 0.004–0.012 kWh/degree-hour.
- Training uses `lag_hours=24` (what is known at a 24 h lead, the horizon `pred_history` scores). Serving uses `lag_hours=0` with `at=[now]` broadcast to all 48 rows. The mean serve lag over the 48 h window is ~24 h, so train and serve are matched on average; document this in the docstring.

- [ ] **Step 1: Failing tests**

```python
class TestHeatingIntensity:
    def _series(self):
        idx = pd.date_range("2026-10-01", periods=24 * 10, freq="1h")
        t_out = pd.Series(8.0, index=idx)                 # 10 degree-hours per hour
        meter = pd.Series(0.5, index=idx)                 # 0.05 kWh per degree-hour
        meter[idx >= pd.Timestamp("2026-10-09")] = 5.0    # spike in the final day
        return idx, t_out, meter

    def test_training_lag_excludes_last_24h(self):
        from energy_forecast.model import _heating_intensity
        idx, t_out, meter = self._series()
        at = pd.DatetimeIndex([pd.Timestamp("2026-10-09 12:00")])
        assert _heating_intensity(meter, t_out, at, lag_hours=24).iloc[0] == pytest.approx(0.05)

    def test_serve_lag_zero_includes_recent(self):
        from energy_forecast.model import _heating_intensity
        idx, t_out, meter = self._series()
        at = pd.DatetimeIndex([pd.Timestamp("2026-10-09 12:00")])
        assert _heating_intensity(meter, t_out, at, lag_hours=0).iloc[0] > 0.05

    def test_gaps_not_zero_filled(self):
        from energy_forecast.model import _heating_intensity
        idx, t_out, meter = self._series()
        sparse = meter[::3]                               # only 1/3 of hours present
        at = pd.DatetimeIndex([pd.Timestamp("2026-10-08 00:00")])
        assert _heating_intensity(sparse, t_out, at, lag_hours=0).iloc[0] == 0.0   # < 120 h coverage → 0, not biased low

    def test_warm_week_is_zero(self):
        from energy_forecast.model import _heating_intensity
        idx = pd.date_range("2026-07-01", periods=24 * 10, freq="1h")
        out = _heating_intensity(pd.Series(0.0, index=idx), pd.Series(25.0, index=idx), idx[-1:], lag_hours=0)
        assert out.iloc[0] == 0.0

    @pytest.mark.parametrize("flag", [False, True])
    def test_flag_controls_feature_and_serve_path(self, tmp_path, flag):
        energy_df, weather_df = train_frames()
        meter = energy_df.rename(columns={"gross_kwh": "kwh"}).assign(kwh=0.3)
        m = EnergyForecastModel(tmp_path, timezone="Europe/Zurich")
        m.train(energy_df, weather_df, outdoor_df=None, weight_halflife_days=0,
                sub_sensors_dict={"sub_heat": meter}, heating_meter_prefix="sub_heat", heating_intensity_feature=flag)
        assert ("heating_intensity_7d" in m.feature_cols) is flag
        if flag:
            fc = weather_df.tail(48).assign(timestamp=pd.date_range(pd.Timestamp.now().floor("1h"), periods=48, freq="1h"))
            recent_ts = pd.date_range(pd.Timestamp.now().floor("1h") - pd.Timedelta(days=10), periods=240, freq="1h")
            recent_meter = pd.DataFrame({"timestamp": recent_ts, "kwh": 0.3})
            m._weather_tail = pd.DataFrame({"timestamp": recent_ts, "temp_c": 8.0})   # cold recent week
            _, X = m._prepare_prediction_X(fc, None, energy_df.tail(400), sub_sensors_recent={"sub_heat": recent_meter})
            assert X["heating_intensity_7d"].nunique() == 1
            assert X["heating_intensity_7d"].iloc[0] == pytest.approx(0.03, rel=0.05)   # 0.3 kWh / 10 degree-hours
```
(`from tests._frames import train_frames` — see the index's "Shared test helpers". `_prepare_prediction_X(forecast_df, live_temp, recent_actuals, sub_sensors_recent=…)` and its `(future_hours, X)` return are verified.)

- [ ] **Step 2: Run** → FAIL.

- [ ] **Step 3: Implement**

```python
def _heating_intensity(meter: Any, t_out: Any, at: Any, lag_hours: int) -> Any:
    """Weather-normalised heating intensity: Σ heating-meter kWh / Σ max(0, 18 − T_out) over the 168 h
    ending `lag_hours` before each timestamp in `at`. Training uses lag 24 (known at the 24 h lead
    pred_history scores); serving uses lag 0 at `now` for all 48 rows (mean serve lag ≈ 24 h).
    Windows with < 120 meter hours or < 24 degree-hours → 0.0 (gaps are not zero-filled)."""
    import numpy as np
    import pandas as pd

    m = meter.copy(); m.index = pd.DatetimeIndex(m.index).floor("1h"); m = m.groupby(level=0).sum()
    grid = pd.date_range(min(m.index.min(), pd.DatetimeIndex(at).min() - pd.Timedelta(hours=168 + lag_hours)),
                         pd.DatetimeIndex(at).max(), freq="1h")
    m = m.reindex(grid)
    dh = np.maximum(0.0, 18.0 - t_out.sort_index().groupby(pd.DatetimeIndex(t_out.index).floor("1h")).mean().reindex(grid))
    both = m.notna() & dh.notna()                      # only hours with BOTH meter and weather count —
    num = m.where(both).rolling(168, min_periods=1).sum()   # a weather gap (e.g. between the stored
    cnt = both.astype(int).rolling(168, min_periods=1).sum()  # weather tail and the forecast) must not
    den = dh.where(both).rolling(168, min_periods=1).sum()    # bias the ratio
    ratio = (num / den).where((cnt >= 120) & (den >= 24.0), 0.0).fillna(0.0).shift(lag_hours)
    return ratio.reindex(pd.DatetimeIndex(at)).fillna(0.0)
```
Training (`train()`), directly after the `if physics_kwh_series is not None:` feature append (~L674); new kwargs `heating_intensity_feature: bool = False, heating_meter_prefix: str | None = None`:
```python
        if heating_intensity_feature:
            _meter = (sub_sensors_dict or {}).get(heating_meter_prefix) if heating_meter_prefix else None
            if _meter is not None and not _meter.empty:
                _ms = _meter.assign(timestamp=pd.to_datetime(_meter["timestamp"])).set_index("timestamp")["kwh"]
                _to = weather_df.assign(timestamp=pd.to_datetime(weather_df["timestamp"])).set_index("timestamp")["temp_c"]
                df["heating_intensity_7d"] = _heating_intensity(
                    _ms, _to, pd.DatetimeIndex(df["timestamp"]), lag_hours=24
                ).values
                base_features = [*base_features, "heating_intensity_7d"]
                self._heating_meter_prefix = heating_meter_prefix
            else:
                _LOGGER.warning("heating_intensity_feature: heating meter %r not in sub_sensors — feature skipped",
                                heating_meter_prefix)
```
`__init__`: `self._heating_meter_prefix: str | None = None`; meta save `"heating_meter_prefix": self._heating_meter_prefix`, load `meta.get("heating_meter_prefix", None)`.
Serving (`_prepare_prediction_X`), directly before `feat_df["hours_ahead"] = np.arange(len(feat_df))` (~L1229):
```python
        if "heating_intensity_7d" in self.feature_cols:
            _val = np.nan  # → training median via the NaN-fill below
            _meter = (sub_sensors_recent or {}).get(self._heating_meter_prefix)
            if _meter is not None and not _meter.empty:
                _ms = _meter.assign(timestamp=pd.to_datetime(_meter["timestamp"])).set_index("timestamp")["kwh"]
                _w = pd.concat([self._weather_tail, forecast_df]) if self._weather_tail is not None else forecast_df
                _to = (_w.assign(timestamp=pd.to_datetime(_w["timestamp"])).drop_duplicates("timestamp", keep="last")
                       .set_index("timestamp")["temp_c"])
                _val = float(_heating_intensity(_ms, _to, pd.DatetimeIndex([future_hours[0]]), lag_hours=0).iloc[0])
            else:
                _LOGGER.warning("heating_intensity_7d: heating meter %r missing at predict time — median fill",
                                self._heating_meter_prefix)
            feat_df["heating_intensity_7d"] = _val
```
`energy_forecast.py`: `self._heating_intensity_feature = bool(self.args.get("heating_intensity_feature", False))`. Pass `heating_intensity_feature=…, heating_meter_prefix=self._sub_sensor_prefix(<physics.heating_sub_meter_sensor>)`, but only when that sensor is configured **and** in `sub_energy_sensors`; otherwise log a warning and pass the flag as False.

- [ ] **Step 4: Full suite + ruff → PASS. Commit** `feat: opt-in heating_intensity_7d feature`.
- [ ] **Step 5: Gate** with the full-feature harness (see Task 4 Step 5) extended by one variant (`heating_intensity_feature=True`, `heating_meter_prefix`, `sub_sensors_dict` incl. the heating meter). Sanity GO: 24 h-lead cold-fold MAE ≤ 95 % of baseline, Sept ≤ 102 %, and the variant better on ≥ 5 of 7 cold days. Even on GO, only set `heating_intensity_feature: true` live after the index's post-deploy verification has ≥ 14 cold days of baseline after Plans C/A/B. Then judge it on the next ≥ 14 cold days per the index's attribution rules. STOP → default stays off; record the table.

---

### Task 6: Docs + finalize (per branch)

- [ ] Per branch, after rebase on `dev`: `@changelog-writer` (Added: `heating_power_request` caching; `tau_estimator` (incl. cross-room airing filter); `physics.ua_eff_method`; `heating_intensity_feature`; note that the Phase 2 cold-start count is heating days with `ua_eff_method: daily`). README config table rows for each new key. Mention the τ value change for anyone reading `tau_hours` (ha-energy-manager doesn't).
- [ ] Live apps.yaml (backup first, only when the respective gate is GO): the flags.
- [ ] deploy-agent Workflow A per branch. Record the alpha tag + deploy timestamp in the index's deploy log.
- [ ] Memory: τ before/after, UA_eff, gate tables. Mark the "τ only from switch-off hours" issue resolved once Task 3 is GO.
