# Fix Strategy — HA Energy Forecast

Generated from a senior developer code review. Issues are grouped into phased
milestones ordered by impact. Each fix lists the target file(s), nature of the
change, effort (S/M/L), and dependencies.

---

## Milestone 1 — Security ✅ DONE

| ID  | Fix | File(s) | Effort | Status |
|-----|-----|---------|--------|--------|
| 1.1 | Parameterise SQL `where_clause` | `energy_history_backfill.py:96–117` | S | Done |
| 1.2 | SHA-256 integrity check before `pickle.load()` | `model.py:_save/_load` | S | Done |

### 1.1 — SQL injection (energy_history_backfill.py)
The `where_clause` was f-string interpolated directly into the query. Replaced
with a `?` placeholder (`where_col` stays interpolated as it is a hardcoded
column reference, not user input). `cutoff_val` is now a bound parameter:
```python
con.execute(query, (entity_id, cutoff_val))
```

### 1.2 — Pickle integrity (model.py)
Added `_write_hash(path)` and `_verify_hash(path)` helpers using `hashlib.sha256`
(stdlib). `_save()` writes a `.sha256` sidecar alongside each `.pkl`. `_load()`
verifies the sidecar before unpickling; mismatch triggers a warning and cold
start. Missing sidecar (pre-upgrade install) is allowed to load.

---

## Milestone 2 — Correctness

| ID  | Fix | File(s) | Effort | Status |
|-----|-----|---------|--------|--------|
| 2.1 | Unify CSV merge strategy + shared helper | `ha_data.py:57,114` / `energy_history_backfill.py:169` | M | TODO |
| 2.2 | Config validation at startup | `energy_forecast.py:45–56` | S | TODO |
| 2.3 | Guard/warn on empty weather DataFrame before training | `energy_forecast.py:126–131` | S | TODO |

### 2.1 — CSV merge inconsistency
`ha_data.py` and `energy_history_backfill.py` both use `drop_duplicates(keep="last")`
but with reversed `concat` order, making fresh data win in one place and cached
data win in the other. Decision:
- Live fetch (`ha_data.py`): fresh HA data wins — `[df_cache, df_new]`, `keep="last"` ✓
- Backfill (`energy_history_backfill.py`): cached CSV wins — `[df_new, df_cache]`, `keep="last"` ✓

The logic is correct but unintuitive. Extract a `_merge_timeseries(df_primary, df_secondary)`
helper called from both sites with intent documented in comments. Add minimal
pytest fixtures that mock `hass.Hass` and exercise the merge with known CSVs
**before** making this change.

### 2.2 — Config validation
Add `_validate_config()` called from `initialize()`. Validate:
- `-90 ≤ lat ≤ 90`, `-180 ≤ lon ≤ 180`
- `weight_halflife_days > 0`
- `ev_charging_threshold_kwh > 0`
- `ev_charger_kw > 0`

Raise `ValueError` with a human-readable message on failure. Log all confirmed
values at INFO level on startup.

### 2.3 — Empty weather DataFrame
When `weather_df = _empty_weather_df()` fallback is taken, emit a named WARNING
listing the impaired features (`temp_c`, `heating_degree`, `cooling_degree`, etc.)
rather than silently proceeding. Consider skipping the training run on first
startup when no weather data is available at all.

---

## Milestone 3 — Observability

| ID  | Fix | File(s) | Effort | Status |
|-----|-----|---------|--------|--------|
| 3.1 | Replace broad `except Exception` with specific types (10+ sites) | all files | M | TODO |
| 3.2 | Warn when lag features are NaN-filled | `model.py:392–397` | S | TODO |
| 3.3 | Log active ML engine at startup | `model.py:91–98` | S | TODO |

### 3.1 — Specific exception types
Each catch site needs the right type(s). Keep broad `except Exception` only at
the outermost AppDaemon callback boundary (`_retrain_cb`, `_update_cb`).

| Location | Specific exceptions |
|----------|-------------------|
| `ha_data.py` cache load/save | `OSError`, `pd.errors.ParserError` |
| `model.py` CV/holdout | `ValueError`, `np.linalg.LinAlgError` |
| `model.py` pickle load | `pickle.UnpicklingError`, `EOFError`, `OSError` |
| `weather.py` fetch | `requests.exceptions.RequestException`, `KeyError`, `ValueError` |

### 3.2 — NaN lag feature warning
After building each lag column in `_add_lag_and_rolling_prediction()`, count NaN
values and log WARNING if >50% are NaN:
```
WARNING: lag_168h has 36/48 NaN values — recent_actuals doesn't reach back 168h
```

### 3.3 — ML engine startup log
One line in `ensure_ml_packages()` after the engine name is determined:
```python
_LOGGER.info("ML engine: %s", engine)
```

---

## Milestone 4 — Code Quality

| ID  | Fix | File(s) | Effort | Status |
|-----|-----|---------|--------|--------|
| 4.1 | Extract magic numbers to constants | `model.py:179,227` → `const.py` | S | TODO |
| 4.2 | Make `CACHE_PATH` config-driven; deduplicate constant | `ha_data.py:17`, `energy_forecast.py:58` | M | TODO |
| 4.3 | Replace `Any` with `pd.DataFrame` in type hints | `model.py:417–421`, `ha_data.py` | S | TODO |
| 4.4 | Remove redundant `tz_convert` | `model.py:263–264` | S | TODO |
| 4.5 | Vectorise lag lookups with `reindex` | `model.py:392–397` | S | TODO |

### 4.1 — Magic numbers
Add to `const.py`:
```python
MIN_TRAINING_ROWS = 100   # model.py:179
HOLDOUT_FRACTION  = 0.9   # model.py:227
MIN_CV_ROWS       = 500   # model.py:227
```

### 4.2 — Hardcoded paths
`CACHE_PATH` is a module-level constant duplicated in both `ha_data.py` and
`energy_history_backfill.py`. Move it to a parameter of `fetch_energy_history()`
and `fetch_recent_energy()` with the current value as default. Pass from
`initialize()` where it can be overridden via `self.args`. Depends on 2.1.

### 4.3 — Type hints
Use `TYPE_CHECKING` guard for pandas import and replace `Any` with `pd.DataFrame`
on all public function parameters and return types across `model.py` and `ha_data.py`.

### 4.4 — Redundant tz_convert (model.py:263–264)
```python
# Before
now_naive = now_aware.tz_convert("Europe/Zurich").tz_localize(None)
# After — tz_convert is a no-op; now_aware is already in Europe/Zurich
now_naive = now_aware.tz_localize(None)
```

### 4.5 — Vectorise lag lookups (model.py:392–397)
```python
# Before
future_df[f"lag_{lag}h"] = [actuals.get(t, np.nan) for t in lag_times]
# After
future_df[f"lag_{lag}h"] = actuals.reindex(lag_times).values
```

---

## Milestone 5 — DST Hardening (Deferred)

| ID  | Fix | File(s) | Effort | Status |
|-----|-----|---------|--------|--------|
| 5.1 | Detect and warn on duplicate naive timestamps after DST fall-back | `ha_data.py`, `energy_forecast.py` | M | TODO |

Add `_check_dst_duplicates(df, logger)` that logs WARNING when
`df["timestamp"].duplicated().any()` after a merge. Document the spring-forward
gap (filled by `ffill()`) as an accepted behaviour. Requires DST-specific test
cases before merging.

---

## Sequencing

```
M1: 1.1 → 1.2              (no deps — single PR)
M2: 2.2 → 2.3 → 2.1        (write tests first, then merge logic)
M3: 3.3 → 3.1 → 3.2        (trivial log, then exception audit)
M4: 4.4 → 4.1 → 4.3 → 4.5 → 4.2   (single-liners first, path refactor last)
M5: 5.1                     (requires test infra from M2)
```

## Additional Recommendation

There is no test suite. Before implementing M2, create a minimal `tests/`
directory with pytest fixtures that mock AppDaemon's `hass.Hass` and exercise
`ha_data.fetch_energy_history()` with known input CSVs. This is a prerequisite
for safe execution of Fix 2.1 and Fix 5.1.
