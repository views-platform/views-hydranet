# Context: HydraNet Streaming Evaluation — Ticket-002

**Repo:** `views-hydranet`
**Depends on:** `views-pipeline-core` (Steps 1–3, already complete)
**Branch:** `feature/samples_for_fao` (pipeline-core); use an equivalent feature branch in views-hydranet

---

## 1. Why This Work Exists

### The Problem

HydraNet passes memory tests in isolation but crashes with OOM when run inside the VIEWS
pipeline at `pgm` scale with S ≥ 32 samples.

Standalone tests use:
- M = 1 rolling origin
- `cm` scale (country-month: ~192 spatial cells)

Pipeline evaluation uses:
- M = 13 rolling origins
- `pgm` scale (PRIO-grid month: ~2.3 million spatial cells)
- T = 6 targets (e.g. `lr_sb`, `lr_sb_best`, `lr_ns_best`, `by_sb_1m`, `by_sb_3m`, `by_sb_6m`)
- S = 32 samples per prediction (posterior Monte Carlo draws)

**Memory arithmetic at pgm/S=32:**

| Quantity | Value |
|---|---|
| Spatial cells per origin window | ~2,300,000 |
| Samples per cell (S) | 32 |
| Bytes per value (float32) | 4 |
| Targets per origin (T) | 6 |
| Size of one PredictionFrame | 2,300,000 × 32 × 4 = **294 MB** |
| Size of one origin (all T targets) | 6 × 294 MB = **1.76 GB** |
| All M=13 origins in memory simultaneously | 13 × 1.76 GB = **22.9 GB → OOM** |

The standalone tests never hit the M-fold multiplication, which is why they pass.

---

## 2. Root Cause: Two Gaps in the Pipeline

### Gap 1 — No native PredictionFrame persistence

`PredictionFrame` had no `save()` / `load()` methods. The only way to write a
`PredictionFrame` to disk was via `PredictionFrameConverter.to_prediction_df()`, which
converts it back to a list-in-cell `pd.DataFrame` parquet. That format is 8–10× larger
than the float32 numpy array on read-back.

This meant: even if the model _produced_ PFs efficiently, writing them to disk was a RAM
sink rather than a RAM relief.

### Gap 2 — Batch interface forces full accumulation

`_execute_model_evaluation()` (pipeline-core) called `_evaluate_model_artifact()` once and
received **all M origins × T targets** as a single `dict[str, list[PredictionFrame]]`.
It held that entire structure in RAM while saving each one sequentially, and held it again
during the metrics computation phase.

The accumulation happened in two places:

**HydraNet side** (`inference_orchestrator.py`):
```python
list_pf_dicts: List[Dict[str, Any]] = []
for i, origin in enumerate(origins):
    # ... compute pf_dict for this origin ...
    list_pf_dicts.append(pf_dict)   # ← accumulator: grows to M entries
return list_pf_dicts
```

**Manager side** (`hydranet_manager.py`):
```python
list_pf_dicts = orchestrator.generate_prediction_frames(...)
result = {t: [d[t] for d in list_pf_dicts] for t in all_targets}  # ← M×T simultaneous
return result
```

Both copies exist in memory at the same time.

---

## 3. Fix Architecture: Two Parallel Tracks

The fix is a streaming interface. Instead of collecting all M origins and returning them
in bulk, the model calls a callback (`origin_sink`) immediately after computing each origin,
then frees the memory before computing the next.

Each origin writes two files and then the in-memory PF is deleted:

```
INFERENCE (per rolling origin)
    │
    ▼
PredictionFrame ──── Track A: Computation Format (.npy) ──────────────────────
                      pf.save(staging_path / f"origin_{i}" / target_name)
                      → y_pred.npy        float32 array, shape (N, S), compact
                      → identifiers.npz   int arrays: time, unit, …
                      Used by: metrics reload phase via PredictionFrame.load(mmap=True)
                      Peak memory on reload: only accessed pages via OS page cache

                 ──── Track B: Delivery Format (.parquet) ──────────────────────
                      converter.to_prediction_df(pf, target)
                      → predictions_eval_origin_{i}_{target}.parquet
                      Used by: downstream consumers, reporting, dashboards
                      Format: unchanged from current behaviour
    │
    ▼
del pf_dict
gc.collect()         ← PF freed; both tracks are on disk
```

After all M origins are processed:
- Peak in-memory PFs: exactly 1 origin's worth (1.76 GB instead of 22.9 GB)
- Track A files on disk enable the metrics phase to reload each origin with `mmap=True`,
  so only the pages accessed by the metrics computation enter RAM at any time

---

## 4. What Was Done in `views-pipeline-core` (Steps 1–3, COMPLETE)

These steps are **already implemented and tested**. You do not need to repeat them.

### Step 1 — `PredictionFrame.save()` / `PredictionFrame.load()`

**File:** `views_pipeline_core/data/prediction_frame.py`

```python
def save(self, directory: Path) -> None:
    """Write y_pred and identifiers to directory as numpy files (Track A)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "y_pred.npy", self.y_pred)
    np.savez(directory / "identifiers.npz", **self.identifiers)

@classmethod
def load(cls, directory: Path, mmap: bool = False) -> "PredictionFrame":
    """Read a PredictionFrame from a directory written by save()."""
    directory = Path(directory)
    mmap_mode = "r" if mmap else None
    y_pred = np.load(directory / "y_pred.npy", mmap_mode=mmap_mode)
    with np.load(directory / "identifiers.npz") as f:
        identifiers = dict(f)
    return cls(y_pred=y_pred, identifiers=identifiers)
```

Key property: when `mmap=True`, `y_pred` is a `np.memmap` — a read-only view from disk.
Pages enter RAM only when accessed. Peak memory is bounded by the access pattern, not the
file size.

### Step 2 — `_evaluate_model_artifact_streaming()` base class hook

**File:** `views_pipeline_core/managers/model/model.py`

```python
def _evaluate_model_artifact_streaming(
    self,
    eval_type: str,
    artifact_name: str,
    origin_sink: Callable[[int, Dict[str, PredictionFrame]], None],
) -> None:
```

This is a **virtual method** on `ForecastingModelManager`. The default implementation
wraps the existing batch `_evaluate_model_artifact()` for backward compatibility (all models
that haven't adopted streaming still work). Subclasses override it to stream origins one at
a time.

**HydraNet's task is to provide this override.**

### Step 3 — `_execute_model_evaluation()` refactored to streaming

**File:** `views_pipeline_core/managers/model/model.py`

The PF path in `_execute_model_evaluation()` now:
1. Defines `_origin_sink(origin_idx, pf_dict)` — writes Track A + Track B per origin, then frees
2. Calls `self._evaluate_model_artifact_streaming(...)` which invokes the sink once per origin
3. After all origins: reloads from Track A staging files via `mmap=True` for metrics
4. Cleans up staging files with `shutil.rmtree(staging_path, ignore_errors=True)`

Full suite status: **861 passed, 1 skipped**.

---

## 5. What Remains: Steps 4 and 5 (THIS TICKET, in `views-hydranet`)

### Step 4 — `HydranetManager._evaluate_model_artifact_streaming()` (TDD)

**File:** `views_hydranet/manager/hydranet_manager.py`

HydraNet must override the base class hook to produce PFs one origin at a time.

The current `_evaluate_model_artifact()` method contains two logical sections:
1. **Setup**: model fetch, data fetch, scaling, VolumeHandler creation, sniffer, partition
   slicing, rolling origin computation — this is shared between batch and streaming
2. **Execution**: the `orchestrator.generate_prediction_frames()` call and the accumulation
   into `list_pf_dicts` / `result` — this is replaced by streaming

The refactor is:
- Extract section 1 into `_setup_evaluation(eval_type, artifact_name) -> dict`
- Keep `_evaluate_model_artifact()` intact by calling `_setup_evaluation()` (backward compat)
- Add `_evaluate_model_artifact_streaming()` that calls `_setup_evaluation()` then delegates
  to `orchestrator.generate_prediction_frames_streaming()` — passing `origin_sink` through

### Step 5 — `InferenceOrchestrator.generate_prediction_frames_streaming()` (TDD)

**File:** `views_hydranet/utils/inference_orchestrator.py`

Add a new method alongside `generate_prediction_frames()` (the batch method is **untouched**).

The streaming method follows the identical ADR 039 six-step inference sequence:
1. PREDICT — `inference.generate_posterior_samples()`
2. TEMPORAL ALIGNMENT — `handler.slice_time()` / `handler.extrapolate_time()`
3. WRAP — `window_handler.wrap_predictions()`
4. INVERT — `scaler.inverse_transform_volume()`
5. COLLAPSE — `pred_handler.collapse_to_point()` (conditional)
6. RECONSTRUCT — `pred_handler.to_evaluation_pf()`

The only difference from the batch method:
```python
# Batch method (DO NOT CHANGE):
list_pf_dicts.append(pf_dict)   ← accumulates

# Streaming method (NEW):
origin_sink(i, pf_dict)         ← delegates immediately
del pf_dict                      ← frees immediately
gc.collect()
```

---

## 6. The Interface Contract

### What `origin_sink` expects

The pipeline-core `_origin_sink` (defined inside `_execute_model_evaluation()`) expects:

```python
origin_sink(
    origin_idx: int,              # 0-based sequential: 0, 1, 2, …, M-1
    pf_dict: Dict[str, PredictionFrame],  # {target_name → PredictionFrame}
)
```

**`pf_dict` must contain exactly one `PredictionFrame` per target** — not a list.
The PF's `y_pred` shape must be `(N, S)` where:
- `N` = number of spatial cells in the prediction window
- `S` = number of posterior samples (`n_posterior_samples` from config), or `1` in point mode

### What `_evaluate_model_artifact_streaming()` must do

```
Call origin_sink(i, pf_dict) exactly once per rolling origin.
Call it with sequential i = 0, 1, 2, ..., M-1.
Do NOT hold pf_dict after calling origin_sink.
Do del pf_dict and gc.collect() between origins.
```

### Type guard in the default base class implementation

If `_evaluate_model_artifact()` returns a non-dict (wrong return type), the default
`_evaluate_model_artifact_streaming()` raises:

```
ModelEvaluationException:
    "prediction_format='prediction_frame' declared but
     _evaluate_model_artifact() returned <type>, expected
     Dict[str, List[PredictionFrame]]. Model contract violation."
```

HydraNet's override does not call `_evaluate_model_artifact()` at all — it calls
`_setup_evaluation()` directly, so this guard does not apply.

---

## 7. Invariants — What Must NOT Change

| Component | Status |
|---|---|
| `_evaluate_model_artifact()` in HydraNet | **Untouched** — other callers rely on it |
| `generate_prediction_frames()` in InferenceOrchestrator | **Untouched** — forecast path uses it |
| `generate_forecasts()` in InferenceOrchestrator | **Untouched** — DataFrame forecast path |
| `PredictionFrameConverter.to_prediction_df()` | **Untouched** |
| `EvaluationAdapter.from_prediction_frames()` | **Untouched** |
| All existing HydraNet tests | **Must continue to pass** |

---

## 8. Repos and Paths

```
views-pipeline-core/  ← pipeline-core: Steps 1–3 DONE (read-only reference)
  views_pipeline_core/
    data/prediction_frame.py        ← PredictionFrame.save() / load()
    managers/model/model.py         ← _evaluate_model_artifact_streaming() base class
  tests/
    test_data/test_prediction_frame_persistence.py
    test_managers/test_streaming_evaluation.py

views-hydranet/       ← your work lives here
  views_hydranet/
    manager/hydranet_manager.py     ← Step 4: override + _setup_evaluation()
    utils/inference_orchestrator.py ← Step 5: generate_prediction_frames_streaming()
  tests/
    test_inference_memory_hygiene.py    ← extend with TestStreamingEvalInterface
    test_inference_orchestrator.py      ← extend with TestStreamingOrchestrator
```

---

## 9. Verification Commands

```bash
# Confirm pipeline-core is still clean (should be untouched)
cd /path/to/views-pipeline-core
conda run -n views_pipeline pytest tests/ -q
# Expected: 861 passed, 1 skipped

# HydraNet — after implementing Steps 4 and 5
cd /path/to/views-hydranet
conda run -n views_pipeline pytest tests/test_inference_memory_hygiene.py -v
conda run -n views_pipeline pytest tests/test_inference_orchestrator.py -v
conda run -n views_pipeline pytest tests/ -q
# Expected: all pass
```
