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

## Milestone 2 — Correctness ✅ DONE

| ID  | Fix | File(s) | Effort | Status |
|-----|-----|---------|--------|--------|
| 2.1 | Unify CSV merge strategy + shared helper | `ha_data.py:57,114` / `energy_history_backfill.py:169` | M | Done |
| 2.2 | Config validation at startup | `energy_forecast.py:45–56` | S | Done |
| 2.3 | Guard/warn on empty weather DataFrame before training | `energy_forecast.py:126–131` | S | Done |

### 2.1 — CSV merge inconsistency
Added `_merge_energy_frames(df_winner, df_loser)` helper to `ha_data.py`.
Concatenates loser first so `keep="last"` always selects the winner's row.
Both `fetch_energy_history()` and `fetch_recent_energy()` now call it with
`df_winner=df_new` — fresh HA data wins on conflicts. The backfill file
(`energy_history_backfill.py`) was already correct (cache wins, with a comment)
and was left unchanged.

### 2.2 — Config validation
Added `_validate_config()` to `EnergyForecast`, called from `initialize()`
immediately after all config values are assigned. Validates:
- `-90 ≤ lat ≤ 90`, `-180 ≤ lon ≤ 180`
- `weight_halflife_days > 0`
- `ev_charging_threshold_kwh > 0`
- `ev_charger_kw > 0`

Raises `ValueError` with a human-readable message on failure. Logs all
confirmed values at INFO level on startup.

### 2.3 — Empty weather DataFrame
The weather fetch fallback `WARNING` now names the specific impaired features
(`temp_c`, `heating_degree`, `cooling_degree`, `temp_rolling_3d`) so operators
know forecast quality will be reduced.

---

## Milestone 3 — Observability ✅ DONE

| ID  | Fix | File(s) | Effort | Status |
|-----|-----|---------|--------|--------|
| 3.1 | Replace broad `except Exception` with specific types (10+ sites) | all files | M | Done |
| 3.2 | Warn when lag features are NaN-filled | `model.py` | S | Done |
| 3.3 | Log active ML engine at startup | `model.py` | S | Done |

### 3.3 — ML engine startup log
Added `_LOGGER.info("ML engine: %s", engine)` in `ensure_ml_packages()` after
the engine name is determined.

### 3.1 — Specific exception types
Replaced all broad inner catches. Outermost callback boundary (`_retrain_cb`,
`_update_cb`) retains `except Exception` as intended.

| Location | Specific exceptions used |
|----------|--------------------------|
| `ha_data.py` cache load | `(OSError, pd.errors.ParserError)` |
| `ha_data.py` cache save | `OSError` |
| `model.py` CV MAE | `(ValueError, np.linalg.LinAlgError)` |
| `model.py` holdout MAE | `(ValueError, IndexError)` |
| `model.py` pickle load | `(pickle.UnpicklingError, EOFError, OSError)` |
| `weather.py` fetches | `(requests.RequestException, KeyError, ValueError)` |
| `energy_forecast.py` weather/actuals fetch | `(OSError, KeyError, ValueError)` — `requests.RequestException` is a subclass of `OSError` in Python 3 |

### 3.2 — NaN lag feature warning
After building each lag column in `_add_lag_and_rolling_prediction()`, logs
WARNING if >50% of the 48 forecast hours are NaN, naming the lag and
explaining the fallback to training medians.

---

## Milestone 4 — Code Quality

**Next milestone to implement.**
Branch: `fix/milestone-1-security` (continue on same branch).
Suggested order: 4.4 → 4.1 → 4.3 → 4.5 → 4.2.

| ID  | Fix | File(s) | Effort | Status |
|-----|-----|---------|--------|--------|
| 4.1 | Extract magic numbers to constants | `model.py:200,248` → `const.py` | S | TODO |
| 4.2 | Make `CACHE_PATH` config-driven; deduplicate constant | `ha_data.py:17`, `energy_forecast.py:58` | M | TODO |
| 4.3 | Replace `Any` with `pd.DataFrame` in type hints | `model.py:437–441`, `ha_data.py` | S | TODO |
| 4.4 | Remove redundant `tz_convert` | `model.py:284–285` | S | TODO |
| 4.5 | Vectorise lag lookups with `reindex` | `model.py:428–432` | S | TODO |

### 4.1 — Magic numbers
Add to `const.py`:
```python
MIN_TRAINING_ROWS = 100   # model.py:200
HOLDOUT_FRACTION  = 0.9   # model.py:248
MIN_CV_ROWS       = 500   # model.py:248
```

### 4.2 — Hardcoded paths
`CACHE_PATH` is a module-level constant duplicated in both `ha_data.py` and
`energy_history_backfill.py`. Move it to a parameter of `fetch_energy_history()`
and `fetch_recent_energy()` with the current value as default. Pass from
`initialize()` where it can be overridden via `self.args`. Depends on 2.1.

### 4.3 — Type hints
Use `TYPE_CHECKING` guard for pandas import and replace `Any` with `pd.DataFrame`
on all public function parameters and return types across `model.py` and `ha_data.py`.

### 4.4 — Redundant tz_convert (model.py:284–285)
```python
# Before
now_naive = now_aware.tz_convert("Europe/Zurich").tz_localize(None)
# After — tz_convert is a no-op; now_aware is already in Europe/Zurich
now_naive = now_aware.tz_localize(None)
```

### 4.5 — Vectorise lag lookups (model.py:428–432)
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
M1: 1.1 → 1.2              ✅ done
M2: tests → 2.2 → 2.3 → 2.1  ✅ done
M3: 3.3 → 3.1 → 3.2        ✅ done
M4: 4.4 → 4.1 → 4.3 → 4.5 → 4.2  ← NEXT
M5: 5.1                     (requires DST test cases first)
```

All work is on branch `fix/milestone-1-security`. Tests: 18/18 passing.

## Additional Recommendation ✅ DONE

A test suite has been added:
- `conftest.py` (repo root) — adds `apps/` to `sys.path` so `energy_forecast`
  is importable without AppDaemon installed.
- `tests/test_ha_data.py` — 18 tests across three classes:
  - `TestMergeEnergyFrames` — winner selection, empty inputs, NaN dropping,
    sort order, multiple conflicts
  - `TestFetchEnergyHistory` — HA-only, cache-only, HA wins on conflict, old
    cache rows preserved, both-empty raises ValueError, spike filter, cache save
  - `TestFetchRecentEnergy` — same merge contract verified independently

`_fetch_history` is patched in all fetch tests; no AppDaemon or live HA needed.
Fix 5.1 (DST hardening) should extend this suite with DST-specific test cases
before merging.
