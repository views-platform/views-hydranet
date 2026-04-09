# Implementation Guide: HydraNet Streaming Evaluation — Ticket-002

**TDD recipe. Follow each numbered step in order. Do not skip ahead.**

---

## 0. Prerequisites

Before writing any code, verify the starting state.

```bash
# In views-pipeline-core: confirm steps 1–3 are complete and tests pass
cd /path/to/views-pipeline-core
conda run -n views_pipeline pytest tests/ -q
# Expected: 861 passed, 1 skipped

# In views-hydranet: confirm all existing tests pass before you touch anything
cd /path/to/views-hydranet
conda run -n views_pipeline pytest tests/ -q
# Expected: all pass (note the count — this is your baseline)
```

Read these files before writing a single line:

1. `views_pipeline_core/managers/model/model.py` — search for `_evaluate_model_artifact_streaming`. Read the full docstring and implementation (about 35 lines). This is the contract you are implementing.
2. `views_pipeline_core/data/prediction_frame.py` — read `save()` and `load()`. These are what the pipeline's `_origin_sink` calls after you emit each `pf_dict`.
3. `views_hydranet/manager/hydranet_manager.py` — read `_evaluate_model_artifact()` from top to bottom. Understand every variable. This is the method you are refactoring.
4. `views_hydranet/utils/inference_orchestrator.py` — read `generate_prediction_frames()` from top to bottom. Understand every step. The streaming method is a structural clone with one difference.
5. `tests/test_inference_memory_hygiene.py` — read the whole file. Understand `MEMORY_CFG`, `_MinimalModel`, `_make_mock_inference()`, `_make_handler()`. You will extend this file.

---

## Step 4A: Write the failing tests for `_evaluate_model_artifact_streaming()`

**File:** `tests/test_inference_memory_hygiene.py` (extend — do not replace)

**Rule: Red first.** Run these tests and confirm they FAIL before implementing anything.
A test that passes before you write the code is a broken test.

Add a new class at the bottom of the file:

```python
# ─── Step 4 tests: _evaluate_model_artifact_streaming() ─────────────────────

class TestStreamingEvalInterface:
    """
    TDD tests for HydranetManager._evaluate_model_artifact_streaming().

    RED before Step 4B implementation: the method does not exist yet.
    GREEN after Step 4B: the override calls origin_sink once per origin with
    the correct structure, and does not accumulate PFs.
    """

    def test_streaming_calls_sink_once_per_origin(self):
        """origin_sink must be called exactly M times — once per rolling origin."""
        ...

    def test_streaming_emits_correct_origin_indices(self):
        """origin_idx argument to sink must be 0, 1, 2, …, M-1 (sequential, 0-based)."""
        ...

    def test_streaming_pf_dict_contains_all_targets(self):
        """Every pf_dict emitted to sink contains all targets (reg + cls)."""
        ...

    def test_streaming_frees_pf_after_sink(self):
        """
        No PredictionFrame should be alive after origin_sink returns.
        Uses weakref to detect leaks.
        """
        ...
```

**How to test a method that calls the orchestrator and real data fetchers:**

For `_evaluate_model_artifact_streaming()` on `HydranetManager`, the challenge is that the
real `_setup_evaluation()` fetches data from disk. The correct approach is to:

1. Patch `_setup_evaluation()` to return a dict with toy objects
2. Patch `orchestrator.generate_prediction_frames_streaming()` to call `origin_sink` directly

The test then verifies the orchestration logic without running any real I/O.

Alternatively — and this is the simpler TDD approach — test the streaming behaviour at the
`InferenceOrchestrator` level (Step 5A), and test `HydranetManager._evaluate_model_artifact_streaming()`
with a narrower contract test that verifies it delegates to `generate_prediction_frames_streaming()`.

**Confirm tests FAIL before moving on:**

```bash
conda run -n views_pipeline pytest tests/test_inference_memory_hygiene.py::TestStreamingEvalInterface -v
# Expected: FAILED (method does not exist)
```

---

## Step 4B: Implement `_setup_evaluation()` and `_evaluate_model_artifact_streaming()`

**File:** `views_hydranet/manager/hydranet_manager.py`

### Sub-step 4B-1: Extract `_setup_evaluation()`

Find `_evaluate_model_artifact()` in the file. Everything from the first line
(`log_device_report(...)`) up to and including `orchestrator = InferenceOrchestrator(...)`
is **Section A: Setup**. Cut this block out and paste it into a new method:

```python
def _setup_evaluation(self, eval_type: str, artifact_name: str | None = None) -> dict:
    """
    Shared setup for both batch and streaming evaluation.

    Returns a dict with keys: handler, scaler, origins, all_targets, orchestrator.
    All I/O happens here; callers receive ready-to-use objects.

    Memory: the raw DataFrame (df) is freed with del/gc.collect() before return.
    """
    # ... (see CODE_EXAMPLES.md for the verbatim implementation) ...
```

The return value must be exactly:
```python
return dict(
    handler=handler,
    scaler=scaler,
    origins=origins,
    all_targets=all_targets,
    orchestrator=orchestrator,
)
```

### Sub-step 4B-2: Update `_evaluate_model_artifact()` to call `_setup_evaluation()`

Replace Section A in `_evaluate_model_artifact()` with a call to `_setup_evaluation()`:

```python
def _evaluate_model_artifact(
    self, eval_type: str, artifact_name: str | None = None
) -> "Union[dict[str, list[PredictionFrame]], List[pd.DataFrame]]":
    ctx = self._setup_evaluation(eval_type, artifact_name)
    handler     = ctx["handler"]
    scaler      = ctx["scaler"]
    origins     = ctx["origins"]
    all_targets = ctx["all_targets"]
    orchestrator = ctx["orchestrator"]

    prediction_format = self.configs.get("prediction_format", "prediction_frame")

    if prediction_format == "prediction_frame":
        list_pf_dicts = orchestrator.generate_prediction_frames(
            handler, scaler, origins=origins, all_targets=all_targets
        )
        result: dict[str, list[PredictionFrame]] = {
            t: [d[t] for d in list_pf_dicts] for t in all_targets
        }
        logger.info(...)
        return result
    else:
        list_df_predictions = orchestrator.generate_forecasts(
            handler, scaler, origins=origins
        )
        logger.info(...)
        return list_df_predictions
```

**This refactor must not change any existing behaviour.** Run the full suite to confirm:

```bash
conda run -n views_pipeline pytest tests/ -q
# Expected: same pass count as before
```

### Sub-step 4B-3: Add `_evaluate_model_artifact_streaming()`

Add after `_evaluate_model_artifact()`:

```python
def _evaluate_model_artifact_streaming(
    self,
    eval_type: str,
    artifact_name: str | None,
    origin_sink: Callable[[int, Dict[str, PredictionFrame]], None],
) -> None:
    """
    Override of ForecastingModelManager.  Streams one origin at a time.

    Calls origin_sink(i, pf_dict) immediately after each origin is produced.
    The pipeline's _origin_sink writes Track A (.npy) + Track B (.parquet),
    then frees the PF before this method advances to the next origin.

    Peak memory: one origin's worth of PredictionFrames (not M × T).
    """
    ctx = self._setup_evaluation(eval_type, artifact_name)
    ctx["orchestrator"].generate_prediction_frames_streaming(
        ctx["handler"],
        ctx["scaler"],
        origins=ctx["origins"],
        all_targets=ctx["all_targets"],
        origin_sink=origin_sink,
    )
```

**Run the new tests:**

```bash
conda run -n views_pipeline pytest tests/test_inference_memory_hygiene.py::TestStreamingEvalInterface -v
# Expected: GREEN (all pass)
```

**Run full suite:**

```bash
conda run -n views_pipeline pytest tests/ -q
# Expected: all pass (same or higher count than baseline)
```

---

## Step 5A: Write the failing tests for `generate_prediction_frames_streaming()`

**File:** `tests/test_inference_orchestrator.py` (extend the existing file)
or create a new file `tests/test_streaming_orchestrator.py`.

Add a new test class. These tests work with `_make_mock_inference()` and `_make_handler()`
from `test_inference_memory_hygiene.py` (import them or duplicate them).

```python
class TestStreamingOrchestrator:
    """
    TDD tests for InferenceOrchestrator.generate_prediction_frames_streaming().

    RED before Step 5B: method does not exist.
    GREEN after Step 5B: method streams origins one at a time.
    """

    def test_streaming_calls_sink_once_per_origin(self):
        """sink is called exactly len(origins) times."""
        ...

    def test_streaming_pf_dict_contains_correct_target_keys(self):
        """Every pf_dict passed to sink has the right target keys."""
        ...

    def test_streaming_frees_posterior_arrays_after_each_origin(self):
        """
        After each origin_sink call, the large intermediate arrays
        (posterior_zstack, pred_handler, window_handler) must be freed.
        Use weakref on the pf_dict itself to confirm it is freed.
        """
        ...
```

**Also extend `TestStructuralMemoryHygiene` with two structural tests:**

```python
def test_generate_prediction_frames_streaming_deletes_pf_dict(self):
    """del pf_dict must appear in generate_prediction_frames_streaming source."""
    source = inspect.getsource(InferenceOrchestrator.generate_prediction_frames_streaming)
    assert "del pf_dict" in source, (
        "generate_prediction_frames_streaming() must explicitly 'del pf_dict' after "
        "calling origin_sink(). Without this, the PredictionFrame stays live until the "
        "next gc pass, overlapping with the next origin's allocations."
    )

def test_generate_prediction_frames_streaming_calls_gc_collect(self):
    """gc.collect() must appear in generate_prediction_frames_streaming source."""
    source = inspect.getsource(InferenceOrchestrator.generate_prediction_frames_streaming)
    assert "gc.collect()" in source, (
        "generate_prediction_frames_streaming() must call gc.collect() inside the "
        "origin loop to promptly release memory after each origin."
    )
```

**Confirm tests FAIL before moving on:**

```bash
conda run -n views_pipeline pytest tests/ -k "TestStreamingOrchestrator or test_generate_prediction_frames_streaming" -v
# Expected: FAILED (method does not exist)
```

---

## Step 5B: Implement `generate_prediction_frames_streaming()`

**File:** `views_hydranet/utils/inference_orchestrator.py`

Add the new method **immediately after** `generate_prediction_frames()`. Do not modify
`generate_prediction_frames()` at all.

The implementation is a structural clone of `generate_prediction_frames()` with one change:

**Remove:**
```python
list_pf_dicts: List[Dict[str, Any]] = []
# ... (inside loop) ...
list_pf_dicts.append(pf_dict)
# ... (after loop) ...
return list_pf_dicts
```

**Replace with:**
```python
# ... (inside loop, after to_evaluation_pf() call) ...
origin_sink(i, pf_dict)
del pf_dict
gc.collect()
# (no return value — method is None)
```

**Signature:**
```python
def generate_prediction_frames_streaming(
    self,
    handler: VolumeHandler,
    scaler: "FeatureScaler",
    origins: List[int],
    all_targets: List[str],
    origin_sink: Callable[[int, Dict[str, Any]], None],
) -> None:
```

See `CODE_EXAMPLES.md` for the complete, verbatim implementation.

**Run the new tests:**

```bash
conda run -n views_pipeline pytest tests/ -k "TestStreamingOrchestrator or test_generate_prediction_frames_streaming" -v
# Expected: GREEN (all pass)
```

---

## Step 6: Verify the Full Suite

```bash
conda run -n views_pipeline pytest tests/ -v
# Expected: all pass (old tests unaffected, new tests green)

# Spot-check the memory hygiene file specifically
conda run -n views_pipeline pytest tests/test_inference_memory_hygiene.py -v
```

---

## Step 7: Structural Test Checklist

The HydraNet test suite enforces memory discipline via `inspect.getsource()` assertions.
Before declaring this ticket complete, verify these structural invariants are present in
the new code and covered by tests:

| Invariant | Test location | Status |
|---|---|---|
| `del pf_dict` in `generate_prediction_frames_streaming` | `TestStructuralMemoryHygiene` | Add |
| `gc.collect()` in `generate_prediction_frames_streaming` | `TestStructuralMemoryHygiene` | Add |
| `del pf_dict` in `_evaluate_model_artifact_streaming` (if applicable) | Optional | Consider |

---

## Key Rules and Pitfalls

### Rule 1: Never modify the batch methods

`generate_prediction_frames()` and `_evaluate_model_artifact()` must not change.
Other callers (the forecast path, tests, any external code) depend on them.

### Rule 2: `del pf_dict` + `gc.collect()` inside the loop

This is the mechanism that makes streaming effective. If you forget `del pf_dict`, the PF
stays alive until the next iteration starts — you do not save memory. If you forget
`gc.collect()`, CPython's reference counting will handle simple cycles, but explicit
collection is a safety net for any cross-references.

### Rule 3: `origin_sink` owns the pf_dict after the call

After `origin_sink(i, pf_dict)` returns, the pipeline has written the PF to disk (Track A
and Track B). Your `del pf_dict` in the streaming method is correct — the pipeline's sink
does not hold a reference either (it calls `del pf_dict` inside the sink itself).

### Rule 4: origin index must be 0-based and sequential

The pipeline names staging directories `origin_0`, `origin_1`, ..., `origin_{M-1}`.
If you pass a non-sequential or non-zero-based index, the metrics phase reload will fail.
`enumerate(origins)` gives you `(i, origin)` — use `i` as the `origin_sink` index, not
`origin` (which is the time index into the data, not a sequence counter).

### Rule 5: `device` argument to `HydraNetInference` is a string

```python
# Correct:
HydraNetInference(self.model, self.config, device=str(self.device), ...)

# Wrong:
HydraNetInference(self.model, self.config, device=self.device, ...)  # torch.device not str
```

### Rule 6: `is_backtest` flag

```python
is_backtest = len(origins) > 1
```

This is the same logic as in `generate_prediction_frames()`. Copy it exactly.
It is passed to `inference.generate_posterior_samples(..., is_evaluation=is_backtest)`.

### Rule 7: The diagnostic biopsy only runs on origin 0

```python
if i == 0:
    self.viz.biopsy_volume(pred_handler, f"Stage 6: Raw Predicted Volume (Origin {origin})")
```

Keep this in the streaming method — it provides a visual diagnostic without significant
memory cost (it runs once at the start, before the PF is freed).

### Rule 8: Collapse step is conditional

```python
if self.config.get("evaluation_mode") == "point":
    pred_handler = pred_handler.collapse_to_point(
        method=self.config["aggregate_method"]
    )
```

Do not unconditionally collapse. In stochastic mode (`evaluation_mode != "point"`),
the samples are passed through as-is and `y_pred.shape[-1] == S`.

---

## Summary: Files Modified

| File | Change |
|---|---|
| `views_hydranet/manager/hydranet_manager.py` | Add `_setup_evaluation()`; refactor `_evaluate_model_artifact()` to call it; add `_evaluate_model_artifact_streaming()` |
| `views_hydranet/utils/inference_orchestrator.py` | Add `generate_prediction_frames_streaming()` (batch method untouched) |
| `tests/test_inference_memory_hygiene.py` | Add `TestStreamingEvalInterface`; add 2 structural tests to `TestStructuralMemoryHygiene` |
| `tests/test_inference_orchestrator.py` | Add `TestStreamingOrchestrator` |

## Summary: Files NOT Modified

| File | Why |
|---|---|
| `views_hydranet/manager/hydranet_manager.py` → `_evaluate_model_artifact()` body | Unchanged externally — still calls `generate_prediction_frames()` internally |
| `views_hydranet/utils/inference_orchestrator.py` → `generate_prediction_frames()` | Completely untouched |
| `views_hydranet/utils/inference_orchestrator.py` → `generate_forecasts()` | Completely untouched |
| All `views-pipeline-core` files | Read-only reference |
