# Implementation Guide: Adopting PredictionFrame in views-hydranet

> **This document replaces the previous architectural plan (DoD #1 is now complete).
> This is a self-contained guide to ship to an agent working on views-hydranet.**

## Context

HydraNet currently returns `list[pd.DataFrame]` from both `_evaluate_model_artifact()`
and `_forecast_model_artifact()`. The pipeline (ADR-033, DoD #1 — now merged) expects
`Dict[str, List[PredictionFrame]]` and `Dict[str, PredictionFrame]` respectively when
`"prediction_format": "prediction_frame"` is declared in config.

HydraNet is the primary target for this migration because it has 6 output heads
(3 regression + 3 classification). The dict+PF interface makes target-to-prediction
alignment explicit and removes all ambiguity about which predictions belong to which target.

---

## What Ships with This Guide

| File | Role |
|------|------|
| `views_pipeline_core/data/prediction_frame.py` | PredictionFrame class to construct |
| `views_pipeline_core/modules/validation/adapter.py` | EvaluationAdapter (consumes PF) |
| `views_pipeline_core/managers/prediction/prediction_frame_dispatcher.py` | Bridge + audit |
| `documentation/CICs/EvaluationAdapter.md` | Contract for the adapter |

---

## Key Types

```python
# Evaluation return type (prediction_frame path)
Dict[str, List[PredictionFrame]]
# └── target_name → [pf_window_0, pf_window_1, ..., pf_window_K-1]
# K = rolling-origin windows (MAX_SHIFT_COUNT + 1 = 13 for calibration)

# Forecast return type (prediction_frame path)
Dict[str, PredictionFrame]
# └── target_name → single PF for the forecast horizon (one origin)
```

For HydraNet, both dicts have 6 keys:
`"lr_sb_best"`, `"lr_ns_best"`, `"lr_os_best"`, `"by_sb_best"`, `"by_ns_best"`, `"by_os_best"`

---

## PredictionFrame Constructor

```python
from views_pipeline_core.data.prediction_frame import PredictionFrame

pf = PredictionFrame(
    y_pred=np.ndarray,          # shape (N, S) — N observations, S posterior samples
    identifiers={
        "time": np.ndarray,     # shape (N,)  — month_id, integer, no NaN
        "unit": np.ndarray,     # shape (N,)  — priogrid_gid, integer, no NaN
    },
)
```

**Construction raises `ValueError` automatically if:**
- `y_pred.ndim != 2`, `N == 0`, or `S == 0`
- `"time"` or `"unit"` key is missing
- Any identifier array length ≠ N
- Any identifier array contains NaN

---

## Source Data: Current HydraNet DataFrame Format

Each `df` in `list_df_predictions` (from `orchestrator.generate_forecasts()` →
`PureStateAdapter.enforce_pure_state_list()`) has:

- **MultiIndex:** `(month_id, priogrid_gid)` — level 0 = time, level 1 = unit
- **Prediction columns:** `pred_lr_sb_best`, `pred_lr_ns_best`, `pred_lr_os_best`,
  `pred_by_sb_best`, `pred_by_ns_best`, `pred_by_os_best`
- **Cell format:** each cell is a Python `list` of `S` floats (posterior samples)

```python
# Extraction per target per rolling-origin window
col    = f"pred_{target}"
y_pred = np.stack(df[col].values)              # (N, S): stack list-cells into 2D array
time   = df.index.get_level_values(0).values   # (N,): month_id
unit   = df.index.get_level_values(1).values   # (N,): priogrid_gid
```

---

## Changes Required in `hydranet_manager.py`

### 1 — New imports (top of file)

```python
import numpy as np
from views_pipeline_core.data.prediction_frame import PredictionFrame
```

### 2 — Add config key in the model config file

In whichever config file declares `regression_targets` / `classification_targets`, add:

```python
"prediction_format": "prediction_frame",
```

### 3 — Add private conversion helper to `HydranetManager`

```python
def _to_pf_dict(
    self,
    list_dfs: list[pd.DataFrame],
    all_targets: list[str],
) -> dict[str, list[PredictionFrame]]:
    """
    Convert list[pd.DataFrame] → Dict[str, List[PredictionFrame]].

    Each DataFrame in list_dfs is one rolling-origin window.
    y_pred shape (N, S): N pgm cells, S posterior samples (from list-in-cell column).
    identifiers["time"] = month_id from MultiIndex level 0.
    identifiers["unit"] = priogrid_gid from MultiIndex level 1.
    """
    result: dict[str, list[PredictionFrame]] = {t: [] for t in all_targets}
    for df in list_dfs:
        time_arr = df.index.get_level_values(0).values
        unit_arr = df.index.get_level_values(1).values
        for target in all_targets:
            y_pred = np.stack(df[f"pred_{target}"].values)  # (N, S)
            result[target].append(
                PredictionFrame(
                    y_pred=y_pred,
                    identifiers={"time": time_arr, "unit": unit_arr},
                )
            )
    return result
```

### 4 — Update `_evaluate_model_artifact()`

Change **only the return type annotation and the final return statement**.
All existing inference logic (ingestion → scaling → VolumeHandler → InferenceOrchestrator
→ PureStateAdapter → VisualDiagnostics) is **unchanged**.

```python
# Old signature:
def _evaluate_model_artifact(
    self, eval_type: str, artifact_name: str | None = None
) -> list[pd.DataFrame]:

# New signature:
def _evaluate_model_artifact(
    self, eval_type: str, artifact_name: str | None = None
) -> dict[str, list[PredictionFrame]]:
```

Replace the final `return list_df_predictions` with:

```python
log_prediction_summary(list_df_predictions)

all_targets = (
    self.configs.get("regression_targets", [])
    + self.configs.get("classification_targets", [])
)
return self._to_pf_dict(list_df_predictions, all_targets)
```

### 5 — Update `_forecast_model_artifact()`

Change **only the return type annotation and the final return statement**.
All inference logic is **unchanged**.

```python
# Old signature:
def _forecast_model_artifact(
    self, artifact_name: str | None = None
) -> list[pd.DataFrame]:

# New signature:
def _forecast_model_artifact(
    self, artifact_name: str | None = None
) -> dict[str, PredictionFrame]:
```

Replace the final `return list_df_predictions` with:

```python
log_prediction_summary(list_df_predictions)

all_targets = (
    self.configs.get("regression_targets", [])
    + self.configs.get("classification_targets", [])
)
pf_dict_of_lists = self._to_pf_dict(list_df_predictions, all_targets)
# Forecast has exactly one origin → unwrap list to get single PF per target
return {target: pf_list[0] for target, pf_list in pf_dict_of_lists.items()}
```

---

## What Does NOT Change

- `_train_model_artifact()`, `_evaluate_sweep()`, `prepare_actuals_df()` — all unchanged
- All data ingestion, scaling, `InferenceOrchestrator`, `PureStateAdapter`, `VisualDiagnostics`
- Storage format: pipeline internally converts PF → legacy DataFrame for disk saves (bridge)
- `_run_preflight_check()` — unchanged

---

## Pipeline Behaviour After the Change (informational)

The `ForecastingModelManager` in views-pipeline-core performs these steps automatically:

1. **Type guard**: verifies `isinstance(raw_preds, dict)` ✓
2. **Step window check**: extracts first target's sequences to check temporal bounds
3. **Save loop**: for each target, calls `to_legacy_dfs([pf], target)` → saves DataFrame
4. **Parity audit**: builds EF via PF path AND via legacy DF path, asserts bit-wise equality
5. **Metrics**: calls `EvaluationAdapter.from_prediction_frames()` per target → EvaluationFrame

The parity audit runs on every execution during the migration window. If both paths produce
identical EvaluationFrames, the audit passes silently. If not, it raises `ValueError` with
`"Parity Failure"` prefix and the diverging field.

---

## Verification

```bash
# Run a calibration evaluation end-to-end
python views_hydranet/main.py --run_type calibration

# Confirm in logs:
#   - No "multi-target PF output is not yet supported" warning (old warning is gone)
#   - Parity audit passes for all 6 targets
#   - 6 prediction parquet files saved at data/generated/
#   - No ModelEvaluationException raised
```

---

## Quick Reference

| Method | Old return | New return |
|--------|-----------|-----------|
| `_evaluate_model_artifact()` | `list[pd.DataFrame]` | `dict[str, list[PredictionFrame]]` |
| `_forecast_model_artifact()` | `list[pd.DataFrame]` | `dict[str, PredictionFrame]` |

| Dict key | Value (eval) | Value (forecast) |
|----------|-------------|-----------------|
| `"lr_sb_best"` | `[pf_w0, ..., pf_wK]` | `pf` |
| `"lr_ns_best"` | `[pf_w0, ..., pf_wK]` | `pf` |
| `"lr_os_best"` | `[pf_w0, ..., pf_wK]` | `pf` |
| `"by_sb_best"` | `[pf_w0, ..., pf_wK]` | `pf` |
| `"by_ns_best"` | `[pf_w0, ..., pf_wK]` | `pf` |
| `"by_os_best"` | `[pf_w0, ..., pf_wK]` | `pf` |

| `PredictionFrame` field | Source |
|------------------------|--------|
| `y_pred` shape `(N, S)` | `np.stack(df[f"pred_{target}"].values)` |
| `identifiers["time"]` shape `(N,)` | `df.index.get_level_values(0).values` |
| `identifiers["unit"]` shape `(N,)` | `df.index.get_level_values(1).values` |

---

## Files to Modify in views-hydranet

| File | Change |
|------|--------|
| `views_hydranet/manager/hydranet_manager.py` | Steps 1, 3, 4, 5 above |
| Config file that declares `regression_targets` | Step 2: add `"prediction_format"` key |

The previous architectural plan content follows (retained for ADR-033 context):

---

# Previous Plan Content (ADR-033 Architectural Plan — DoD #1 COMPLETE)

## Status: Where We Are

**DoD #1 is substantially reached for single-target models.** The ADR-033 Strangler Fig
implementation already supports both paths, two-level parity auditing, and a clean adapter
boundary. The remaining gap that blocks DoD #1 completeness is:

1. **Multi-target PF is not supported** — only the primary target is extracted. This is
   unacceptable: multi-target output (e.g. HydraNet: `ged_sb`, `ged_ns`, `ged_os`) is a
   primary feature of the system.
2. **No type enforcement** on abstract method returns — a misconfigured model produces a
   late, cryptic error instead of an early, clear one.
3. **Dispatch is repeated 4×** in model.py — DRY violation, minor but real.

---

## Three-Expert Decision: Multi-Target PF Interface

**The question**: how should `_evaluate_model_artifact()` return multi-target PF predictions?

**Three options considered:**

| Option | Return type | PredictionFrame change | Risk |
|--------|------------|----------------------|------|
| A | `Dict[str, List[PredictionFrame]]` | None | Low |
| B | `List[PredictionFrame]` with `y_pred: Dict[str, ndarray]` | Breaking — changes y_pred shape | High |
| C | `List[PredictionFrame]` with `target: str` field; manager groups | Adds field; ordering ambiguous | Medium |

**Verdict: Option A — `Dict[str, List[PredictionFrame]]`**

> **Uncle Bob**: "The current `List[PredictionFrame]` is ambiguous — the list indexes
> sequences but is silent on targets. `Dict[str, List[PredictionFrame]]` is
> self-documenting: 'a sequence-list per named target.' Fix the contract now. Zero PF
> models are in production. This is the cheapest moment."
>
> **Senior ML Engineer**: "HydraNet runs ONE forward pass and produces all targets
> simultaneously. Targets share identical identifiers. The dict matches the model's natural
> structure. Single-target models return a single-key dict — no magic. The DF path already
> does this implicitly (all columns in one DataFrame); the dict makes it explicit."
>
> **Principal Developer**: "Option B mutates PredictionFrame's `y_pred: ndarray(N,S)`
> invariant — every consumer must handle two shapes. Option C creates ordering ambiguity
> between sequences and targets that couples step_mapping assignment to list position.
> Option A is localized: manager receives the dict, adapter is unchanged, PredictionFrame
> stays pure."

---

## 1. Minimal Prediction Interface

**Transitional (DoD #1):**
```python
# eval — PF path
def _evaluate_model_artifact(...) -> Dict[str, List[PredictionFrame]]
# {target_name: [pf_seq0, pf_seq1, ...]}  — one PF per rolling-origin sequence, per target

# eval — DF path (unchanged)
def _evaluate_model_artifact(...) -> List[pd.DataFrame]

# forecast — PF path
def _forecast_model_artifact(...) -> Dict[str, PredictionFrame]
# {target_name: pf}

# forecast — DF path (unchanged)
def _forecast_model_artifact(...) -> pd.DataFrame
```

**Canonical (DoD #2, after DF path retired):**
```python
def _evaluate_model_artifact(...) -> Dict[str, List[PredictionFrame]]
def _forecast_model_artifact(...) -> Dict[str, PredictionFrame]
```

Single-target models use a single-key dict: `{"ged_sb": [pf0, pf1, ...]}`. No magic; no
special-casing for single vs. multi-target in the manager.

---

## 2. Module Responsibility Boundaries

| Layer | Responsibility | Change required |
|-------|---------------|----------------|
| `ForecastingModelManager` | Dispatch by `_prediction_format` property; iterate targets from dict keys | Yes — receive dict, iterate targets |
| `PredictionFrameDispatcher` | Bridge, parity audit, structural audit — per target | Minor — accept `(target, List[PF])` instead of `(List[PF], target)` |
| `EvaluationAdapter` | Convert single-target `List[PredictionFrame]` → `EvaluationFrame` per target call | No change needed |
| `PredictionFrame` | Self-validating container, zero dependencies | **No change** |
| `EvaluationManager` | Metrics on pure numpy | No change |

---

## 3. Changes Required for DoD #1

### A — Abstract method return types

**`_evaluate_model_artifact` PF path:**
```python
# Before:
def _evaluate_model_artifact(...) -> Union[List[pd.DataFrame], List[PredictionFrame]]

# After:
def _evaluate_model_artifact(...) -> Union[List[pd.DataFrame], Dict[str, List[PredictionFrame]]]
```

Docstring and return type annotation updated. Contract note: PF path returns a dict with
one key per target; each value is a list of `len(sequences) == MAX_SHIFT_COUNT + 1` PFs.

**`_forecast_model_artifact` PF path:**
```python
# Before:
def _forecast_model_artifact(...) -> Union[pd.DataFrame, PredictionFrame]

# After:
def _forecast_model_artifact(...) -> Union[pd.DataFrame, Dict[str, PredictionFrame]]
```

### B — `_prediction_format` property (DRY fix)

Extract 4× repeated `self.configs.get("prediction_format", "dataframe")` into:
```python
@property
def _prediction_format(self) -> str:
    return self.configs.get("prediction_format", "dataframe")
```

### C — Type enforcement guards (fail-loud, early)

After each abstract call, verify the returned type matches `_prediction_format`:
```python
# eval
if self._prediction_format == "prediction_frame":
    if not isinstance(raw_preds, dict):
        raise ModelEvaluationException(
            f"prediction_format='prediction_frame' declared but "
            f"_evaluate_model_artifact() returned {type(raw_preds).__name__}, "
            f"expected Dict[str, List[PredictionFrame]]. Model contract violation."
        )
```

### D — Manager dispatch updated for multi-target dict

**`_execute_model_evaluation` (PF path, save loop):**
```python
# Before (primary-target only):
_primary_target, _all_targets = self._get_primary_target()
for i, df_for_save in enumerate(
    _dispatcher.to_legacy_dfs(list_df_predictions, _primary_target)
):
    self._save_predictions(df_for_save, ...)

# After (all targets):
for target, pf_sequence_list in raw_preds_dict.items():
    for i, df_for_save in enumerate(
        _dispatcher.to_legacy_dfs(pf_sequence_list, target)
    ):
        self._save_predictions(df_for_save, ...)
```

**`_evaluate_prediction_dataframe` (PF path, EF build loop):**
```python
# Before: manager receives List[PF] and selects target from config
# After: manager receives Dict[str, List[PF]]; target loop is driven by dict keys
for target, pf_sequence_list in raw_preds_pf.items():
    step_mappings = self._get_evaluation_step_mappings(n_sequences=len(pf_sequence_list))
    ef = _disp.build_evaluation_frame(actual_slice, pf_sequence_list, target, step_mappings)
    # ... rest of target loop unchanged
```

**`_execute_model_forecasting` (PF path):**
```python
# Before (primary-target only):
_pf = self._forecast_model_artifact(...)
_primary_target, _ = self._get_primary_target()
df_predictions = PredictionFrameDispatcher().to_legacy_dfs([_pf], _primary_target)[0]

# After (all targets, each saved separately):
pf_dict = self._forecast_model_artifact(...)  # Dict[str, PredictionFrame]
for target, _pf in pf_dict.items():
    _disp = PredictionFrameDispatcher()
    df_target = _disp.to_legacy_dfs([_pf], target)[0]
    _disp.audit_prediction_structure(_pf, df_target, target)
    # undo_all_transformations + _save_predictions per target
```

**`_get_primary_target()` and multi-target warning: deleted.** No longer needed — the dict
drives target iteration directly. The warning was a placeholder for this fix.

---

## 4. TDD Strategy

### New tests required (write RED first):

**`test_model_manager_prediction_format.py`:**

| Test | What it proves |
|------|---------------|
| `test_pf_eval_returns_dict_keyed_by_target` | PF path calls abstract method, receives dict, dispatches each target |
| `test_pf_eval_multi_target_saves_all_targets` | Two-target dict → two save calls, one per target |
| `test_pf_forecast_dict_dispatches_all_targets` | Forecasting PF dict → save called once per target |
| `test_type_guard_eval_pf_declared_df_returned` | Raises `ModelEvaluationException` on List[DF] when dict expected |
| `test_type_guard_eval_df_declared_dict_returned` | Raises on dict when List[DF] expected |
| `test_type_guard_forecast_pf_declared_df_returned` | Raises on type mismatch in forecast path |

**Parity closure (existing `TestParityClosure` in `test_evaluation_adapter.py`):**
Already covers single-target. Add one multi-target variant:
| `test_pf_and_legacy_df_parity_multi_target` | Two targets × 3 sequences: both adapter paths produce identical EFs per target |

### Existing coverage retained (no changes):
`TestAuditParityEf`, `TestBuildEvaluationFrame`, `TestParityClosure` (single-target) — all unchanged.

---

## 5. DoD #2 — Scaffolding Removal Plan

**Trigger**: All model subclasses migrate to `prediction_format: prediction_frame`.

**Removal commit sequence** (each keeps tests green):

```
Commit 1: Delete Level 1 parity (adapter-level)
  - PredictionFrameDispatcher.build_evaluation_frame(): direct to from_prediction_frames()
  - Delete: audit_parity_ef(), to_legacy_dfs(), EvaluationAdapter.from_dataframes()
  - Delete: EvaluationAdapter.from_prediction_frame() (singular)
  - Delete: tests for deleted methods

Commit 2: Delete Level 2 parity (manager-level)
  - Remove evaluate(ef=None) call and _audit_parity()
  - Keep only evaluate(ef=ef)

Commit 3: Remove DF dispatch branches
  - Delete all `else: (DF path)` blocks in model.py
  - Delete CorePredictionSniffer calls (DF only)
  - Delete _prediction_format property (single path, no dispatch needed)
  - Delete ThreadPoolExecutor sniffer block

Commit 4: Simplify abstract signatures
  - _evaluate_model_artifact() → Dict[str, List[PredictionFrame]]
  - _forecast_model_artifact() → Dict[str, PredictionFrame]
  - Remove Union types; remove type enforcement guards (single type, can't mismatch)

Commit 5: Documentation
  - EvaluationAdapter.md: remove deprecation notes
  - ADR-033: mark closed, record final state
```

**Post-removal final state:**
```
_evaluate_model_artifact() → Dict[str, List[PredictionFrame]]
    ↓
for target, pf_list in predictions.items():
    EvaluationAdapter.from_prediction_frames(actual, pf_list, target, step_mappings)
    → EvaluationFrame
    → EvaluationManager.evaluate(ef=ef) → metrics
```

Zero conditionals. Zero bridges. Zero parity audits.

---

## 6. Risks

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| Existing single-target PF models break on dict interface | None — zero PF models in production | Greenfield change; safe now |
| DF path models unaffected | Confirmed — DF branch unchanged | No migration required |
| Transformation undo per target in forecasting | Low complexity | Apply undo separately per target DF |
| Type enforcement too strict | Low | Guard checks isinstance; dict with wrong value type still passes at model runtime |
| Multi-target step_mapping assignment | None | step_mapping is per-sequence; dict groups sequences by target naturally |

---

## Immediate Implementation Order (DoD #1 complete)

1. **TDD RED**: Write the 7 new tests listed above — all fail
2. **Step A**: `_prediction_format` property
3. **Step B**: Update abstract method signatures + docstrings
4. **Step C**: Type enforcement guards
5. **Step D**: Manager dispatch updated (eval + forecast PF branches)
6. **Step E**: Delete `_get_primary_target()` and multi-target warning (no longer needed)
7. **TDD GREEN**: All tests pass
8. **Verify**: 782 + 7 - 0 = 789 passing

**Files modified**: `model.py` only (all changes are in the manager layer).
**`PredictionFrame`, `EvaluationAdapter`, `PredictionFrameDispatcher` — unchanged.**
