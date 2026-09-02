# Cooling Mode / AC Support — Plan D: Feature Registration & Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register the two new columns as trained features, prove the whole series didn't regress heating-only behavior, and close out docs — the terminal plan in this series.

**Architecture:** `_FEATURES_BASE`/SHAP-label registration, then a no-regression test comparing `X.columns` composition and holdout MAE against a pre-change `dev` baseline captured via a temporary git worktree.

**Tech Stack:** Python 3.13, pandas, numpy, LightGBM, pytest, git worktrees.

**Spec:** `docs/superpowers/specs/2026-09-01-cooling-mode-ac-support-design.md` (rev. 3) — §5, §6.
**Plan index:** `docs/superpowers/plans/2026-09-01-cooling-mode-ac-support-index.md`.
**Depends on:** Plan A, Plan B, and Plan C must all be merged first.

**Base branch:** `dev`, branched from Plan C's merged commit. Branch name: `feat/cooling-mode-finalize ha-energy-forecast`.

## Global Constraints

- `cooling_active`, `cooling_load_sum_24h` are added to `_FEATURES_BASE` unconditionally — constant-zero for the current heating-only install base, by design, indefinitely. (`cooling_load_sum_168h` was also originally planned but was removed pre-Plan-D — see `2026-09-02-cooling-load-sum-drop-168h.md`.)
- Task 1's baseline-capture script (`scripts/capture_cooling_regression_baseline.py`) must run against a pristine pre-Plan-A `dev` checkout (via `git worktree`), never against the feature branch.
- Test files run via `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`.

---

### Task 1: `_FEATURES_BASE` + SHAP labels (§5)

Adds `cooling_active` and `cooling_load_sum_24h` to `_FEATURES_BASE` **unconditionally** — they
will be constant-zero for effectively the entire current install base (heating-only Central
European users), by design, indefinitely; not "the same pattern as `heating_active`" (which
defaults to 1 only for the *unconfigured* subset and is a genuinely variable signal for
deployments that configure it). `cooling_load_sum_168h` was removed before this task ever ran
(#96, see the design spec's §4.7 history note) — only these 2 columns exist. Per Implementation
Decision #7, the per-row conditional SHAP label swap the spec describes for a "single-prediction
narrative" has no existing call site to attach to (the only consumer, `_build_shap_narrative()`, is
fed exclusively by the aggregate `shap_summary()` path the spec itself says should keep the static
label) — only the two new static dict entries are added.

**Files:**
- Modify: `apps/energy_forecast/model.py` (`_FEATURES_BASE`, ~lines 79-159)
- Modify: `apps/energy_forecast/energy_forecast.py` (`_SHAP_FEATURE_LABELS`, ~lines 63-147)
- Test: `tests/test_model.py`, `tests/test_energy_forecast.py`

**Interfaces:**
- Consumes: Plan A's Task 4's `cooling_active` column, Plan B's Task 4's `cooling_load_sum_24h` column (post-#96, `cooling_load_sum_168h` no longer exists).
- Produces: the two columns are now selected as trained features. Consumed by Task 2's no-regression test.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_model.py`:
```python
class TestCoolingFeaturesRegistered:
    def test_cooling_columns_in_features_base(self):
        assert "cooling_active" in _FEATURES_BASE
        assert "cooling_load_sum_24h" in _FEATURES_BASE

    def test_training_with_constant_zero_cooling_columns_does_not_crash(self, tmp_path):
        """No cooling entities configured -> both columns are constant-zero;
        LightGBM/TreeSHAP must handle this without error (spec §5's cost claim)."""
        from energy_forecast.model import EnergyForecastModel

        ts = pd.date_range("2026-01-01", periods=200, freq="1h")
        energy_df = pd.DataFrame(
            {"timestamp": ts, "gross_kwh": np.random.default_rng(3).uniform(0.5, 2.0, len(ts))}
        )
        weather_df = _make_weather_df(ts)
        model = EnergyForecastModel(model_dir=tmp_path)
        model.train(energy_df, weather_df, outdoor_df=None)
        assert "cooling_active" in model.feature_cols
        assert "cooling_load_sum_24h" in model.feature_cols
```

Add to `tests/test_energy_forecast.py`:
```python
class TestCoolingShapLabels:
    def test_cooling_labels_present(self):
        from energy_forecast.energy_forecast import _SHAP_FEATURE_LABELS

        assert "cooling_active" in _SHAP_FEATURE_LABELS
        assert "cooling_load_sum_24h" in _SHAP_FEATURE_LABELS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_model.py::TestCoolingFeaturesRegistered tests/test_energy_forecast.py::TestCoolingShapLabels -v`
Expected: FAIL — `assert "cooling_active" in _FEATURES_BASE` / `in _SHAP_FEATURE_LABELS` both fail (not added yet).

- [ ] **Step 3: Add to `_FEATURES_BASE`**

In `apps/energy_forecast/model.py`, in the `_FEATURES_BASE` list (lines 79-159), add `"cooling_active",` immediately after `"heating_active",  # seasonal on/off from heating_system_active_entity`:
```python
    "heating_active",  # seasonal on/off from heating_system_active_entity
    "cooling_active",  # AC on/off from hvac_mode_entity/cooling_system_active_entity — #96
```
Add `"cooling_load_sum_24h",` immediately after `"heating_deg_sum_168h",  # #50 accumulated heating debt`:
```python
    "heating_deg_sum_24h",
    "heating_deg_sum_168h",  # #50 accumulated heating debt
    "cooling_load_sum_24h",  # #96 accumulated cooling debt (AC-active-gated)
```

- [ ] **Step 4: Add to `_SHAP_FEATURE_LABELS`**

In `apps/energy_forecast/energy_forecast.py`, in the `_SHAP_FEATURE_LABELS` dict (lines 63-147), add immediately after `"heating_active": "seasonal heating on/off",`:
```python
    "cooling_active": "seasonal cooling on/off",
```
Add immediately after `"heating_deg_sum_168h": "accumulated heating demand (7d)",`:
```python
    "cooling_load_sum_24h": "accumulated cooling demand (24h)",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_model.py::TestCoolingFeaturesRegistered tests/test_energy_forecast.py::TestCoolingShapLabels -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Run the full model and energy_forecast test suites**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_model.py tests/test_energy_forecast.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add apps/energy_forecast/model.py apps/energy_forecast/energy_forecast.py tests/test_model.py tests/test_energy_forecast.py
git commit -m "feat: register cooling columns in _FEATURES_BASE and SHAP labels (#96, §5)"
```

---

### Task 2: No-regression test suite + docs (§6, §6's 2nd bullet, CHANGELOG/README/ROADMAP/MEMORY)

Closes the two §6 requirements not already covered by earlier tasks' own tests: (1) a `X.columns`-composition + holdout-MAE diff against pre-change `dev` output, with `cooling_mode_enabled` both absent and explicitly `false` — the actual risk the 2 unconditionally-added columns pose; (2) the "enabled but no active entity configured" branch (already exercised at the fetch layer by Plan A's Task 3's `test_neither_entity_configured_returns_empty`, extended here to a full-chain confirmation). Then updates docs per the project's standard finishing workflow.

**Files:**
- Create: `tests/fixtures/cooling_regression_baseline.json` (golden baseline, captured from `dev` — see Step 1)
- Create: `scripts/capture_cooling_regression_baseline.py` (one-time capture script, self-contained so it can run unmodified against a pre-change worktree)
- Create: `tests/test_cooling_regression.py`
- Modify: `CHANGELOG.md`, `ROADMAP.md`, `MEMORY.md` (this project's, at `memory/*.md` + `MEMORY.md` index)
- Test: the new `tests/test_cooling_regression.py` itself is the deliverable

**Interfaces:**
- Consumes: the complete feature from Plan A, Plan B, and Plan C (all 14 preceding tasks).
- Produces: nothing consumed by further tasks — this is the terminal task.

- [ ] **Step 1: Capture the pre-change baseline from `dev` via a git worktree**

Write `scripts/capture_cooling_regression_baseline.py` — self-contained (no imports from `tests/`, since it must also run correctly checked out against `dev`, where none of this plan's test helpers exist yet):

```python
"""One-time baseline capture for the cooling-mode no-regression test (#96).

Run this script from a pristine pre-cooling-change checkout (the `dev` branch,
before Plan A's Task 1 of the cooling-mode-ac-support plan set
(docs/superpowers/plans/2026-09-01-cooling-mode-ac-support-index.md) lands) to produce tests/fixtures/cooling_regression_baseline.json. The
no-regression test (tests/test_cooling_regression.py) then compares current
code's output against this frozen baseline — it does NOT re-run this script.

Usage (from the repo root, dedicated env active):
    python scripts/capture_cooling_regression_baseline.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))

import numpy as np
import pandas as pd

from energy_forecast.model import EnergyForecastModel


def _build_fixture_dataset():
    """Deterministic synthetic heating-season dataset — no climate/cooling
    entities configured, matching the current install base's typical shape."""
    rng = np.random.default_rng(96)
    ts = pd.date_range("2026-01-01", periods=24 * 120, freq="1h")  # 120 days
    hour = ts.hour.values
    base = 0.8 + 0.6 * np.sin((hour - 6) / 24 * 2 * np.pi) ** 2
    noise = rng.normal(0, 0.1, len(ts))
    gross_kwh = np.clip(base + noise, 0.05, None)
    energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": gross_kwh})

    temp_c = 5.0 + 8.0 * np.sin((ts.dayofyear.values / 365) * 2 * np.pi - np.pi / 2)
    weather_df = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": temp_c,
            "precipitation_mm": np.zeros(len(ts)),
            "sunshine_min": np.clip(30 * np.sin((hour - 6) / 12 * np.pi), 0, None),
            "wind_kmh": np.full(len(ts), 10.0),
            "humidity": np.full(len(ts), 70.0),
            "cloud_cover_pct": np.full(len(ts), 50.0),
            "direct_radiation_wm2": np.clip(300 * np.sin((hour - 6) / 12 * np.pi), 0, None),
        }
    )
    return energy_df, weather_df


def main():
    energy_df, weather_df = _build_fixture_dataset()
    split = int(len(energy_df) * 0.9)
    train_energy, holdout_energy = energy_df.iloc[:split], energy_df.iloc[split:]

    model = EnergyForecastModel(model_dir=Path("/tmp/cooling_baseline_model"))
    model.train(train_energy, weather_df.iloc[:split], outdoor_df=None)

    future_ts = holdout_energy["timestamp"]
    forecast_df = weather_df[weather_df["timestamp"].isin(future_ts)].reset_index(drop=True)
    preds = model.predict(forecast_df, live_temp=None, recent_actuals=train_energy.tail(48))
    merged = preds.merge(holdout_energy, on="timestamp", suffixes=("_pred", "_actual"))
    mae = float(np.mean(np.abs(merged["gross_kwh_pred"] - merged["gross_kwh_actual"])))

    baseline = {"columns": sorted(model.feature_cols), "mae": mae}
    out_path = Path(__file__).parent.parent / "tests" / "fixtures" / "cooling_regression_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(baseline, indent=2))
    print(f"Wrote baseline: {len(baseline['columns'])} columns, MAE={mae:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
```

Run it against a pristine `dev` checkout (not the feature branch — this must reflect pre-change code):
```bash
git worktree add /tmp/cooling-baseline-worktree dev
cd /tmp/cooling-baseline-worktree
/home/jovyan/my_envs/ha-energy-forecast/bin/python scripts/capture_cooling_regression_baseline.py
cp tests/fixtures/cooling_regression_baseline.json /home/jovyan/work/ha-energy-forecast/tests/fixtures/cooling_regression_baseline.json
cd /home/jovyan/work/ha-energy-forecast
git worktree remove /tmp/cooling-baseline-worktree
```
(If `scripts/capture_cooling_regression_baseline.py` doesn't exist yet on `dev` — it won't, since it's part of this task — copy just the script into the worktree checkout manually before running it there: `cp scripts/capture_cooling_regression_baseline.py /tmp/cooling-baseline-worktree/scripts/` from the feature branch, then run it inside the worktree.)

- [ ] **Step 2: Commit the baseline fixture and capture script**

```bash
git add scripts/capture_cooling_regression_baseline.py tests/fixtures/cooling_regression_baseline.json
git commit -m "test: capture pre-change baseline for cooling-mode no-regression test (#96)"
```

- [ ] **Step 3: Write the no-regression test**

Create `tests/test_cooling_regression.py`:

```python
"""No-regression guarantee for #96: cooling_mode_enabled absent/false must
reproduce pre-change dev output exactly (X.columns composition + holdout MAE),
per spec §6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from energy_forecast.model import EnergyForecastModel

_BASELINE_PATH = Path(__file__).parent / "fixtures" / "cooling_regression_baseline.json"


def _build_fixture_dataset():
    """Must stay byte-identical to scripts/capture_cooling_regression_baseline.py's
    _build_fixture_dataset() — duplicated deliberately, since the capture script
    must also run unmodified against a pre-change dev checkout that doesn't have
    this test file."""
    rng = np.random.default_rng(96)
    ts = pd.date_range("2026-01-01", periods=24 * 120, freq="1h")
    hour = ts.hour.values
    base = 0.8 + 0.6 * np.sin((hour - 6) / 24 * 2 * np.pi) ** 2
    noise = rng.normal(0, 0.1, len(ts))
    gross_kwh = np.clip(base + noise, 0.05, None)
    energy_df = pd.DataFrame({"timestamp": ts, "gross_kwh": gross_kwh})

    temp_c = 5.0 + 8.0 * np.sin((ts.dayofyear.values / 365) * 2 * np.pi - np.pi / 2)
    weather_df = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": temp_c,
            "precipitation_mm": np.zeros(len(ts)),
            "sunshine_min": np.clip(30 * np.sin((hour - 6) / 12 * np.pi), 0, None),
            "wind_kmh": np.full(len(ts), 10.0),
            "humidity": np.full(len(ts), 70.0),
            "cloud_cover_pct": np.full(len(ts), 50.0),
            "direct_radiation_wm2": np.clip(300 * np.sin((hour - 6) / 12 * np.pi), 0, None),
        }
    )
    return energy_df, weather_df


@pytest.fixture(scope="module")
def _baseline():
    if not _BASELINE_PATH.exists():
        pytest.skip(
            f"{_BASELINE_PATH} missing — run scripts/capture_cooling_regression_baseline.py "
            "against a pristine dev checkout first (see Task 2, Step 1 of the cooling-mode plan)."
        )
    return json.loads(_BASELINE_PATH.read_text())


class TestCoolingModeNoRegression:
    @pytest.mark.parametrize("cooling_mode_enabled_arg", [None, False], ids=["absent", "explicit_false"])
    def test_columns_and_mae_match_pre_change_baseline(self, tmp_path, _baseline, cooling_mode_enabled_arg):
        """cooling_mode_enabled absent and explicitly False are both untouched by
        this plan's changes at the fetch/config layer, so cooling_active_df/
        cooling_active_series are never built — only the 2 unconditionally-added
        columns (cooling_active, cooling_load_sum_24h) differ from baseline,
        and they must be constant-zero and not move MAE."""
        energy_df, weather_df = _build_fixture_dataset()
        split = int(len(energy_df) * 0.9)
        train_energy, holdout_energy = energy_df.iloc[:split], energy_df.iloc[split:]

        model = EnergyForecastModel(model_dir=tmp_path)
        model.train(train_energy, weather_df.iloc[:split], outdoor_df=None)  # no cooling_active_df passed

        baseline_cols = set(_baseline["columns"])
        current_cols = set(model.feature_cols)
        new_cooling_cols = {"cooling_active", "cooling_load_sum_24h"}

        assert current_cols - baseline_cols == new_cooling_cols, (
            f"Unexpected column set change beyond the 2 cooling additions: "
            f"{(current_cols - baseline_cols) - new_cooling_cols}"
        )
        assert baseline_cols - current_cols == set(), (
            f"Columns present in baseline but missing now: {baseline_cols - current_cols}"
        )

        future_ts = holdout_energy["timestamp"]
        forecast_df = weather_df[weather_df["timestamp"].isin(future_ts)].reset_index(drop=True)
        preds = model.predict(forecast_df, live_temp=None, recent_actuals=train_energy.tail(48))
        merged = preds.merge(holdout_energy, on="timestamp", suffixes=("_pred", "_actual"))
        mae = float(np.mean(np.abs(merged["gross_kwh_pred"] - merged["gross_kwh_actual"])))

        assert mae == pytest.approx(_baseline["mae"], rel=0.02), (
            f"Holdout MAE moved beyond a 2% tolerance: {mae:.4f} vs baseline {_baseline['mae']:.4f} "
            "— the 2 unconditionally-added constant-zero columns should not perturb tree structure."
        )

    def test_enabled_but_no_active_entity_reproduces_heating_only_columns(self, tmp_path, _baseline):
        """§6's 2nd required bullet: cooling_mode_enabled=True with no
        hvac_mode_entity/cooling_system_active_entity configured must reproduce
        the same output as disabled — extends Plan A's Task 3's fetch-layer-only
        test_neither_entity_configured_returns_empty to the full training chain."""
        energy_df, weather_df = _build_fixture_dataset()
        split = int(len(energy_df) * 0.9)
        train_energy = energy_df.iloc[:split]

        # cooling_mode_enabled=True is an energy_forecast.py-layer config flag —
        # at the model.py layer exercised directly here, "enabled but unconfigured"
        # means cooling_active_df simply stays None (Plan A's Task 3's _fetch_hvac_active_history
        # never populates it without an entity), which is indistinguishable from
        # the disabled case at this layer by construction (Implementation Decision #1).
        model = EnergyForecastModel(model_dir=tmp_path)
        model.train(train_energy, weather_df.iloc[:split], outdoor_df=None, cooling_active_df=None)

        new_cooling_cols = {"cooling_active", "cooling_load_sum_24h"}
        assert set(model.feature_cols) - set(_baseline["columns"]) == new_cooling_cols
```

- [ ] **Step 4: Run the no-regression test**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/test_cooling_regression.py -v`
Expected: PASS (3 tests: 2 parametrized + 1). If the MAE assertion fails outside the 2% tolerance, this is a real signal that one of Plan A's Task 4 onward, or a task in Plan B or Plan C, changed heating-only behavior — do not loosen the tolerance to make it pass; use `superpowers:systematic-debugging` to find which task's change leaked into the disabled path.

- [ ] **Step 5: Run the ENTIRE test suite**

Run: `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`
Expected: PASS — every test added across all 16 tasks, plus zero regressions to the pre-existing suite.

- [ ] **Step 6: Update CHANGELOG.md**

Delegate to `@changelog-writer` with a summary of this feature: cooling-mode/AC-support ML-feature layer (#96) — config keys, `cooling_active` threading through the full train/predict call chain, mode-aware `thermal_pressure`/`thermal_pressure_net`/`thermal_pressure_cop`, `cooling_load_sum_24h`, `defrost_risk` cooling exclusion, τ-calibration passive-window guard, regime-clustering cooling-day exclusion, model-artifact rollback fallback, `_FEATURES_BASE`/SHAP label registration. Note it is opt-in (`cooling_mode_enabled: false` by default) and physics-baseline forecasts stay heating-only.

- [ ] **Step 7: Update ROADMAP.md**

Check `ROADMAP.md` for an existing #96 entry (the spec's own header says "#96... remains correctly deprioritized behind higher-impact items"); update its status to reflect that the ML-feature-layer implementation described in the spec is now done, while noting `physics.py`'s cooling-capable baseline (§2's non-goal) remains a distinct, unscoped follow-up.

- [ ] **Step 8: Update this project's MEMORY.md**

Add a `memory/` entry (type `project`) noting: cooling-mode/AC support implemented per `docs/superpowers/specs/2026-09-01-cooling-mode-ac-support-design.md` and this plan; opt-in via `cooling_mode_enabled`; known limitations carried forward from spec §7 (EER placeholder, no latent-load signal, solar-lag approximation, dry/fan_only false-positive risk on the binary-entity path, no dual-mode blending, physics baseline stays heating-only). Add the one-line pointer to `MEMORY.md`'s index.

- [ ] **Step 9: Final commit**

```bash
git add CHANGELOG.md ROADMAP.md tests/test_cooling_regression.py
git commit -m "docs: update CHANGELOG/ROADMAP for cooling-mode support (#96)"
```
(`memory/*.md` and the project's `MEMORY.md` are gitignored per the global memory rule — do not `git add` them.)

---

## Self-Review Notes (covers the full 4-plan series)

**Spec coverage:** §3 (Plan A Task 2) · §4.1/§4.1a (Plan A Tasks 3-5) · §4.2 (Plan A Task 6) · §4.3 (Plan B Task 1) · §4.4 (Plan B Task 2) · §4.5 (no task anywhere in the series — spec states zero additional code needed once §4.3 is mode-aware, confirmed correct: `infiltration_pressure` derives from `thermal_pressure`, which Plan B Task 1 already makes mode-aware) · §4.6 (Plan B Task 3) · §4.7 (Plan B Task 4) · §4.8 (Plan B Task 5) · §4.9 (Plan C Task 1) · §4.10 (Plan C Task 2) · §4.11 (Plan C Task 3) · §5 (this plan's Task 1) · §6 (every plan's own tests plus this plan's Task 2's no-regression + unconfigured-branch tests) · §7 (documented inline as code comments throughout the series, consolidated in this plan's Task 2's CHANGELOG/MEMORY update) · §8 (the series' 4 plans × 16 total tasks map closely to the spec's own effort-table line items) · §9 (explicitly not attempted anywhere in the series — no task builds a dedicated cooling regime cluster) · §10 (no gaps to address — spec's own alignment check found none).

**Placeholder scan:** no `TBD`/`TODO`/"add appropriate error handling" patterns across any of the four plan documents; every step has literal code.

**Type consistency across the series:** `cooling_active_df` (`pd.DataFrame | None`, cols `timestamp`/`cooling_active`) is produced by Plan A Task 3 and consumed with that exact name/shape through Plan A Tasks 4/6, Plan C Tasks 1/2. `cooling_active_series` (`pd.Series | None`, indexed by timestamp) is produced by Plan A Task 5 and consumed identically through Plan B Tasks 1-4's prediction-time paths. `cooling_conflict_warned` (`set | None`) is threaded identically in Plan A Task 6 and Plan B Task 3. `is_training` (`bool`, default `False`) is introduced in Plan B Task 3 and reused identically in Plan B Task 4. `warn_once(logger, warned, key, msg, *args)`'s signature (Plan A Task 1) matches every call site in Plan A Task 6 and Plan B Task 3.

**Final release note:** once all four plans have merged to `dev` and passed the full suite, this feature is ready for a `dev`-tag alpha release per the project's standard `deploy-agent` workflow (`@deploy-agent`'s Workflow A). It is **not** ready for a `main` release / public tag until a maintainer explicitly decides to promote it — the spec's own header frames this as "community-contribution candidate... no implementation timeline committed," so building it does not itself imply shipping it to `main`.
