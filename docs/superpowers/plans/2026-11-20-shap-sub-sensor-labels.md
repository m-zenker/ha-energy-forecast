# SHAP narrative labels for dynamic sub-sensor features

Branch: `fix/shap-sub-sensor-labels` (cut from `dev`; train/next does not touch this code). No deploy, no tag.

## Problem
`sub_em_kermi_bridge_kermi_electricity_heating_lag_168h` appears in the descriptive SHAP narrative as its raw
column name. `_SHAP_FEATURE_LABELS` (`apps/energy_forecast/energy_forecast.py:69`) is static; sub-sensor
columns are generated per entity (`model.py:720-727`) as `<prefix>_{lag_24h,lag_168h,active_24h,runs_7d}` with
`prefix = "sub_" + entity_slug` (`_sub_sensor_prefix`). Every configured sub-sensor hits this. These are the only
unlabelled dynamic columns (`heating_buffer_temp`, `outdoor_temp_live`, `temp_bias` are dynamic but already labelled;
climate/generic sensors use fixed names). Independent review confirmed the suffix set is complete.

## Design
Pattern resolution, not per-entity dict entries.

### 1. `apps/energy_forecast/energy_forecast.py`
- Add `import re` (the module has none; review blocker).
- Add `_SUB_SENSOR_SUFFIX_LABELS: dict[str, str]` (templates with `{name}`):
  - `lag_24h` -> "yesterday's {name} consumption"
  - `lag_168h` -> "last week's {name} consumption"
  - `active_24h` -> "{name} active in past 24h"
  - `runs_7d` -> "{name} starts in past 7d"
- Add `_SUB_SENSOR_RE = re.compile(r"^sub_(?P<slug>.+)_(?P<suffix>lag_24h|lag_168h|active_24h|runs_7d)$")`
  (greedy slug + anchored suffix, so a slug that itself contains `lag`/`runs` still parses on the final suffix).
- Add `_humanize_sub_slug(slug)`: strip a leading `em_` then a leading `kermi_bridge_` (intentional: both are EM-publisher prefixes, not part of the appliance name) (only when something remains
  after stripping), replace `_` with spaces. If the result would be empty, use the unstripped slug.
- Add `_shap_label(feat)`: static dict first; then the regex; else the raw `feat`.
- `_build_shap_narrative` uses `_shap_label` instead of `.get`.
- Expected: `sub_em_kermi_bridge_kermi_electricity_heating_lag_168h` -> "last week's kermi electricity heating consumption".

### 2. `tests/test_energy_forecast.py` (extend `TestShapFeatureLabelCoverage`)
- The reported feature resolves to the exact string above.
- All four suffixes resolve for one slug.
- Slug without `em_`/`kermi_bridge_` (`sub_dishwasher_lag_24h`) -> "yesterday's dishwasher".
- Slug containing a suffix-like token (`sub_heat_runs_7d_lag_24h`) parses on the final suffix only.
- Unknown / non-matching name (`sub_foo`, `lag_999h`) falls back to the raw column.
- Static label still wins over the regex (e.g. `lag_24h_tgated`).
- `_build_shap_narrative` end-to-end with a mixed static + sub-sensor dict has no raw `sub_` in the output.
- Coverage guard (no model.py refactor; there is no suffix constant): call `_add_sub_sensor_lags_training` on a
  >268-row frame, diff added columns vs input, assert every added column resolves to a label != raw. Fails if a
  fifth suffix is added without a label.
- Real prefix round-trip: `sub_em_kermi_bridge_kermi_electricity_heating` (as asserted in
  `tests/test_energy_forecast_physics_config.py:549`) + each suffix resolves.
- Edge: `sub_em_lag_24h` (slug `em` strips to empty -> falls back to "em"), `sub__lag_24h` (no match -> raw).
- Regression guard: `shap_top_features` keys passed through `_build_shap_narrative` are not mutated (raw).

### 3. Docs
- CHANGELOG `[Unreleased]` via `/changelog` (Fixed). ROADMAP: extend row #90 (line 769, currently Unreleased) with the
  sub-sensor pattern labels. README: add a sub-sensor example to the narrative text near README.md:576.

## Success criteria
- Reported feature renders as a readable label; full suite green via
  `/home/jovyan/my_envs/ha-energy-forecast/bin/python -m pytest tests/ -v`; `ruff check` clean.

## Out of scope
- `shap_top_features` attribute keys and `dashboard/shap-importance.yaml` keep raw column names (stable identifiers;
  relabelling would break consumers). Only the narrative text is humanized.
- Renaming the columns themselves (would invalidate persisted models / `sub_sensor_prefixes` metadata).
- Changing which features SHAP surfaces.
