# HydraNet Memory Investigation

**To**: HydraNet repo agent
**From**: views-pipeline-core memory audit (2026-03-10)
**Subject**: Two confirmed memory issues in `_evaluate_model_artifact()` to investigate and fix

A comprehensive memory audit of the views-pipeline-core evaluation loop confirmed that the
dominant memory consumers are in the **model repo** (HydraNet), not the pipeline-core.
Two specific hypotheses are empirically supported and need to be investigated in HydraNet's
`_evaluate_model_artifact()`.

---

## H2 — PyTorch Caching Allocator Does Not Release Between Sequences

**What the audit showed**: For numpy arrays, `del array + gc.collect()` immediately returns
memory to the OS (via mmap). The pipeline-core's own allocations behave correctly.

**The unresolved question**: PyTorch uses a **custom caching allocator** (C10 CachingAllocator)
that is fundamentally different. When a PyTorch tensor is deleted in Python, PyTorch returns
the memory to its **internal pool** — but NOT to the OS. `psutil.rss` stays elevated between
sequences even if all Python tensor references are deleted.

**Where to look in HydraNet**: Inside the rolling-origin loop in `_evaluate_model_artifact()`:

```python
# Current pattern (suspected):
for seq_idx in range(13):
    X = prepare_sequence_data(seq_idx)
    pred = model(X)             # ← PyTorch creates computation graph + activations
    pf = PredictionFrame(pred.detach().cpu().numpy(), identifiers)
    results[target].append(pf)
    # ← PyTorch tensors may still be alive here if no explicit cleanup
```

**What to check**:

1. Is `torch.no_grad()` used during inference? Without it, PyTorch retains the full
   computation graph for every forward pass — this holds all intermediate activation tensors
   in memory until Python GC runs, which may not happen between sequences.
   ```python
   with torch.no_grad():
       pred = model(X)   # activations freed immediately after forward pass
   ```

2. Are tensors explicitly deleted after conversion to numpy?
   ```python
   pred_np = pred.detach().cpu().numpy()
   del pred          # ← without this, tensor stays alive
   gc.collect()      # ← on CPU, this matters
   ```

3. Is `torch.cuda.empty_cache()` called between sequences? (Only relevant if GPU is used.)
   On CPU: `gc.collect()` is the equivalent.

4. Does HydraNet use any internal state (hidden state, buffers) that accumulates across calls?

**Expected impact if H2 is true**: Each sequence holds the forward-pass activation graph
until explicitly freed. For HydraNet at 32 samples, 180×180, this could be 1–5 GB of retained
activations per sequence. Over 13 sequences without cleanup, this alone could saturate 32 GB.

**How to measure**: Add a probe inside the loop:

```python
import psutil, os, gc
proc = psutil.Process(os.getpid())

for seq_idx in range(13):
    rss_before = proc.memory_info().rss / 1e6
    X = ...
    with torch.no_grad():
        pred = model(X)
    pred_np = pred.detach().cpu().numpy()
    del pred; gc.collect()
    rss_after = proc.memory_info().rss / 1e6
    print(f"seq {seq_idx:02d}: before={rss_before:.0f}MB  after={rss_after:.0f}MB  delta={rss_after-rss_before:+.0f}MB")
    pf = PredictionFrame(pred_np, ...)
    results[target].append(pf)
```

If RSS after `del+gc` is not close to RSS before `model(X)`, PyTorch is retaining memory
(H2 confirmed).

---

## H4 — Input Sequence Data (X) Not Freed Between Sequences

**What the audit showed**: Empirically confirmed +2,145 MB excess accumulation when viewser
input data is not explicitly freed after each sequence. 13 sequences × ~178 MB per slice =
+2.3 GB.

**Where to look in HydraNet**: The data loading step inside the loop:

```python
for seq_idx in range(13):
    X = load_sequence_data(seq_idx)    # ← loads viewser slice into memory
    pred = model(X)
    pf = ...
    results[target].append(pf)
    # ← X stays alive until overwritten by next iteration.
    # 'X = load_sequence_data(seq_idx+1)' does NOT guarantee
    # the old X is freed before the new one is allocated (overlap window).
```

**What to do**: Explicitly free X after inference, before loading the next sequence:

```python
for seq_idx in range(13):
    X = load_sequence_data(seq_idx)
    with torch.no_grad():
        pred = model(X)
    pred_np = pred.detach().cpu().numpy()
    del pred, X          # ← free BOTH before appending the PF
    gc.collect()
    pf = PredictionFrame(pred_np, ...)
    results[target].append(pf)
    del pred_np          # ← also free the raw numpy after PF construction
```

**Note on float32**: `PredictionFrame.__init__` now casts `y_pred` to float32 (views-pipeline-core
Phase 1 change). Even if `pred_np` is float64, the PF stores a float32 copy. After PF
construction, `pred_np` is safe to delete.

**Expected impact**: ~2.3 GB reduction in peak RSS during the inference loop.

---

## Context: Pipeline-Core Memory Budget (After Phase 1 — float32 enforcement)

After `_evaluate_model_artifact()` returns, pipeline-core holds:

| Component | Size |
|-----------|------|
| raw_preds (13 seqs × 2 targets × float32) | ~3.9 GB |
| First `to_prediction_df()` call (Python list explosion) | +4.8 GB transient |
| `from_prediction_frames()` peak per target | +3.1 GB |

Pipeline-core peak is ~8–11 GB. If HydraNet's inference adds another 10+ GB from H2 + H4,
the 32 GB limit is hit during inference — before pipeline-core even starts processing results.

---

## Priority Order

1. **Add `torch.no_grad()` context manager** to the inference call (zero-cost correctness fix)
2. **Add explicit `del pred; gc.collect()` after each sequence** (frees PyTorch allocator pool on CPU)
3. **Add explicit `del X; gc.collect()` after each sequence** (frees viewser data slice)
4. **Run the RSS probe** (print before/after each sequence) to empirically confirm H2 and H4

Fix in this order: `torch.no_grad()` is free. The del+gc calls are two lines. The probe makes
it measurable. Do not guess — measure.
