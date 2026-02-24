# Technical Debt Resolution Plan — HIGH PRIORITY

**Source audit:** `reports/investigations/2026-02-21_technical_debt_audit.md`
**Date:** 2026-02-21
**Status:** Planned — not yet executed

Execute fixes in order. Run `conda run -n views-hydranet-env pytest tests/ -q` after each group.

---

## GROUP 1 — Correctness Blockers (CRITICAL)

### Fix 1 — TD-022 + TD-030: `"mean"` crashes at inference
**Files:** `views_hydranet/utils/utils_config.py:198`, `views_hydranet/utils/volume_handler.py:382`

`validate_agg_method` maps `"mean"` → `"geometric_mean"`, but `collapse_to_point()` has no
`"geometric_mean"` branch and raises `NotImplementedError`. Fix the mapper:

```python
# BEFORE
mapper = {"mean": "geometric_mean", "median": "median", "max_aposteriori": "median"}

# AFTER
mapper = {"mean": "arithmetic_mean", "median": "median", "max_aposteriori": "median"}
```

No change needed to `collapse_to_point` — `"arithmetic_mean"` already works.

---

### Fix 2 — TD-024: Dead first `scaffold_cols` block
**File:** `views_hydranet/utils/volume_handler.py:506–516`

Delete the first dead assignment of `scaffold_cols` and its duplicate `# 4. Initialize Polars
Scaffold` comment. Keep only the second (live) identical block at lines 513–516.

---

### Fix 3 — TD-026: `is_forecast=False` always bypasses forecast continuity check
**File:** `views_hydranet/manager/hydranet_manager.py:279`

In `_forecast_model_artifact` only, change:
```python
# BEFORE
sniffer.sniff_forecast_alignment(df, handler, is_forecast=False)

# AFTER
sniffer.sniff_forecast_alignment(df, handler, is_forecast=True)
```

Leave the call in `_evaluate_model_artifact` (line 204) unchanged — `is_forecast=False` is
correct for evaluation.

---

### Fix 4 — TD-027: `index_names` required at runtime but absent from schema
**File:** `views_hydranet/utils/utils_config.py`

Add to `HydraNetConfig` in the "3. Spatiotemporal Topology" section:
```python
index_names: list[str] = Field(..., description="Column names used as the DataFrame index")
```

---

### Fix 5 — TD-029: Shuffle-vulnerable fallback must fail loud
**File:** `views_hydranet/utils/volume_handler.py:571–580`

Replace the silent position-fallback `else` branch in `_reconstruct_from_provider` with a
`raise ValueError` (ADR-008):
```python
else:
    err_msg = (
        f"VolumeHandler._reconstruct_from_provider: Prediction channel '{name}' has no "
        "watermarked identity scaffold. Cannot safely reconstruct without risking identity "
        "scramble. Ensure wrap_predictions() is used before reconstruction."
    )
    logger.error(err_msg)
    raise ValueError(err_msg)
```

---

## GROUP 2 — API Integrity (HIGH)

### Fix 6 — TD-021: Typo `evalution_mode` baked into canonical field name
**Files:** `views_hydranet/utils/utils_config.py`, `views_hydranet/utils/inference_orchestrator.py`

1. Rename the Pydantic field: `evalution_mode` → `evaluation_mode`
2. Reverse the shim in `handle_typos`: translate old typo → correct name:
   ```python
   if "evalution_mode" in data and "evaluation_mode" not in data:
       data["evaluation_mode"] = data["evalution_mode"]
   ```
3. Update validator decorator: `@field_validator("evaluation_mode")`
4. Update `inference_orchestrator.py` line 126: `.get("evalution_mode")` → `.get("evaluation_mode")`
5. Grep for any other consumers of the typo key.

---

### Fix 7 — TD-028: Preflight head-count mismatch warns but does not fail
**File:** `views_hydranet/manager/hydranet_manager.py:71–78`

Change `logger.warning` + `print` to `raise ValueError` per ADR-008:
```python
if n_reg != 3 or n_class != 3:
    err_msg = (
        f"ARCHITECTURE MISMATCH: Model expects 3+3 heads, "
        f"Config has {n_reg}+{n_class}. Aborting."
    )
    logger.error(err_msg)
    raise ValueError(err_msg)
```

---

## GROUP 3 — ADR-008 Fail Loud Violations

### Fix 8 — TD-001: Dead duplicate `execute_freeze_h_option` in `utils.py`
**File:** `views_hydranet/utils/utils.py:132–153`

Delete the entire `execute_freeze_h_option` function. The live version is
`HydraNetInference.execute_freeze_h_option()` in `hydranet_inference.py`.

---

### Fix 9 — TD-008: Unknown metric silently returns `0.0` in `TrainingForensics`
**File:** `views_hydranet/utils/training_forensics.py:138`

```python
# BEFORE
return 0.0   # silent zero

# AFTER
err_msg = f"TrainingForensics: Unknown regression metric '{name}'."
logger.error(err_msg)
raise ValueError(err_msg)
```

After this fix, add RED GATE test to `tests/test_training_forensics.py`:
`test_forensics_unknown_metric_raises` — passes unknown metric name, expects `ValueError`.

---

### Fix 10 — TD-023: Silent autocorrect for `"stocastic"` typo
**File:** `views_hydranet/utils/utils_config.py:185–186`

Delete the two silent coercion lines:
```python
if v == "stocastic":    # DELETE
    return "stochastic" # DELETE
```

---

## GROUP 4 — Dead Code Removal

### Fix 11 — TD-002: Orphaned `IntegrityGuardian.check_weights`
**File:** `views_hydranet/utils/integrity_guardian.py:63–74`

Delete the `check_weights` static method. Never called.

---

### Fix 12 — TD-003: Dead `columns` parameter
**File:** `views_hydranet/train/train_model.py:232, 404, 421`

Remove `columns: list[str] | None = None` from `training_loop` and `train_model_artifact`
signatures and from the forwarding call inside `train_model_artifact`.

---

### Fix 13 — TD-004 + TD-018: Dead attributes in `VolumeSampler`
**File:** `views_hydranet/utils/volume_sampler.py:42–43`

Delete:
```python
self._buffer: List[VolumeHandler] = []
self.windows_per_lesson = config["windows_per_lesson"]
```

---

### Fix 14 — TD-025 + TD-037: Redundant `h_max` rebind
**File:** `views_hydranet/utils/volume_sampler.py:96–97`

Delete the second binding:
```python
h_max, _ = train_vh.shape[1], train_vh.shape[2]  # DELETE — already set at line 64
```

---

### Fix 15 — TD-013 + TD-038: Buggy bare string literal in `mtloss.py`
**File:** `views_hydranet/utils/mtloss.py:68–79`

Delete the entire triple-quoted string literal "usage" example block.
It contains incorrect API (`torch.stack(a, b, c)` instead of `torch.stack([a, b, c])`).

---

## GROUP 5 — `print()` → `logger.*()` (MEDIUM)

### Fix 16 — TD-020: Replace `print()` with structured logging
**Files:** `data_sniffer.py`, `data_fetcher.py`, `utils_device.py`, `hydranet_manager.py`

- `data_sniffer.py`: Delete all `print("")` blank-line separators
- `data_fetcher.py`: Delete `print("")` separators
- `utils_device.py:41`: `print(...)` → `logger.info(...)`
- `hydranet_manager.py`: Delete all `print("")` block separators; remove duplicate `print(f"\n{msg}\n")` in `_run_preflight_check` (becomes moot after Fix 7 converts it to a raise)

---

## GROUP 6 — Structural Mismatch

### Fix 17 — TD-039: Span check vs. absolute index check in `DataSniffer`
**File:** `views_hydranet/utils/data_sniffer.py:241–257`

Change the span-based check to mirror `VolumeHandler.from_df` exactly:
```python
# BEFORE
r_span = df[y_col].max() - df[y_col].min()
if r_span >= height:

# AFTER
row_offset = config.get("row_offset", 0)
col_offset = config.get("col_offset", 0)
r_max_idx = (df[y_col] - row_offset).max()
c_max_idx = (df[x_col] - col_offset).max()
if r_max_idx >= height or c_max_idx >= width:
```

---

## GROUP 7 — Import Hygiene (LOW)

### Fix 18 — TD-005/006/007: Promote deferred imports to module level
- `views_hydranet/train/train_model.py:34` — move `import functools` to top
- `views_hydranet/utils/feature_scaler.py:62, 219` — move `import numpy as np`, `import torch` to top
- `views_hydranet/utils/visual_diagnostics.py:36` — move `from datetime import datetime` to top

---

## GROUP 8 — Comment / Marker Cleanup (LOW)

### Fix 19 — TD-009/010/011/012: Stale comments
- `hydranet_inference.py:427` — Delete `# REFACTOR:` comment
- `volume_handler.py:691` — Grep callers of `permute`; if used in tests, remove the "NOTE: Review needed" text; if unused, delete the method
- `utils_device.py:42` — Delete `# not sure you need to return it...` comment
- `visual_diagnostics.py:271–274` — Delete the three commented-out pseudo-code lines

---

## GROUP 9 — Test Coverage Gaps

### Fix 20 — TD-031: Active GREEN GATE for `biopsy_volume`
**File:** `tests/test_visual_diagnostics.py`

Add `test_vd_biopsy_volume_saves_png`: minimal `VolumeHandler` (1×4×4×2, THWC, one identity +
one feature channel), call `biopsy_volume(vh, "Test Stage")` in active mode, assert PNG exists.

### Fix 21 — TD-036: Unit test for `FeatureScaler.inverse_transform_volume`
Apply `log1p` transform to known value, call `inverse_transform_volume`, assert:
- `lr_` channels: `expm1` correctly applied
- `by_` channels: skipped
- `pred_` prefix: stripped before lookup

### Fix 22 — TD-035: Metric math + unknown-metric RED GATE tests
**File:** `tests/test_training_forensics.py`

- `test_forensics_unknown_metric_raises` — enabled by Fix 9
- `test_forensics_rmsle_correctness` — known values
- `test_forensics_msle_correctness` — known values

### Fix 23 — TD-034: Failure-path test for `sniff_pure_state_parity`
**File:** `tests/test_pure_state_integrity.py`

Add test where output df has corrupted `c_id` column; assert `sniff_pure_state_parity` raises.

### Fix 24 — TD-032/033: Active-mode tests for `biopsy_dataframe` and `biopsy_sample`
**Deferred** — both require complex fixture setup (full spatial DataFrame; two calibrated
VolumeHandlers). Defer to follow-up PR.

---

## Expected Suite Delta

| Before | After | Delta |
|--------|-------|-------|
| 152 passing | ~160 passing | +~8 new tests |

---

## Verification

```bash
# After all groups complete:
conda run -n views-hydranet-env pytest tests/ --tb=short -q
```
