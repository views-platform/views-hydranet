# Computational Bottleneck: Spatiotemporal DF Reconstruction

## 1. Problem Statement
The final stage of the HydraNet pipeline—converting internal 5D stochastic volumes back into row-oriented `pd.DataFrame` objects—has become a terminal bottleneck for consumer-grade hardware. 

**CRITICAL CLARIFICATION:** This is a **System RAM** failure, not a GPU VRAM issue. While the model runs efficiently on the GPU, the process of "dressing" the data for the CPU-bound evaluation domain causes a massive heap-memory explosion that kills the host machine.

## 2. The Final DataFrame Schema (ViEWS Outbound Contract)
To be bit-perfect and compatible with downstream evaluation, the `VolumeHandler` must produce a DataFrame with the following exact structure:

### 2.1 Index Structure
*   **Type:** `pd.MultiIndex`
*   **Levels:** `['month_id', 'priogrid_gid']`
*   **Invariants:** Must be topologically complete (no missing cells within the spatiotemporal cube).

### 2.2 Column Domain (Per Target)
For every target feature (e.g., `sb`), the DF contains a triplet of columns:
1.  **`sb` (Actual):** The ground-truth count from the history provider (Type: `float` or `int`).
2.  **`pred_sb_raw` (Regression):** The model's intensity prediction.
3.  **`pred_sb_prob` (Classification):** The model's probability-of-event signal.

### 2.3 Data Payload (Stochastic Mode)
In stochastic mode (`n_posterior_samples > 1`), the prediction cells (`_raw` and `_prob`) do not contain scalars. They contain **Python Lists of Floats**:
*   **Cell Content:** `[sample_1, sample_2, ..., sample_N]`
*   **Length:** Exactly equal to `config["n_posterior_samples"]`.

## 3. Technical Root Cause: The "Object Explosion"
The bottleneck is located in the `VolumeHandler` reconstruction bridges (`to_evaluation_df`, `to_forecast_df`). 

### 3.1 RAM Object Overhead
When satisfying the "List-in-Cell" requirement, we are moving from a contiguous NumPy block to a fragmented Python Heap.
*   **The Math of Failure:** A 180x180 grid with 36 months and 100 samples produces ~1.1 million cells. If each cell contains a Python list of 100 floats, the Python interpreter must manage **millions of individual list and float objects** in RAM.
*   **Memory Fragmentation:** Unlike NumPy arrays which are contiguous, these millions of small objects fragment the RAM heap, causing the OS to trigger the OOM killer even if the total "data" is only a few gigabytes.

### 3.2 The Reconstruction Gate
The current implementation in `VolumeHandler._reconstruct_from_ledger` uses a dictionary-to-dataframe approach:
```python
# This line is the "Kill Point" for RAM
df_out = pd.DataFrame(reconstructed)
```
The `reconstructed` dictionary holds the entire spatial grid in memory as raw Python objects before pandas even begins its internal consolidation.

## 4. Plan for Empirical Diagnostic (The Memory Fingerprint Audit)
We will implement a script (`tests/memory_fingerprint_audit.py`) to quantify this explosion.

### 4.1 Measurement States
1.  **State A (Contiguous):** Raw NumPy 5D volume. Mathematically minimal.
2.  **State B (Fragmented):** Intermediate Python dictionary holding millions of `tolist()` outputs. 
3.  **State C (The Monolith):** The final `pd.DataFrame` consolidated by pandas.

### 4.2 The "Object Tax" Metric
We will calculate the ratio: `RAM_Used_By_DF / RAM_Used_By_NumPy`. 
A high ratio (>10x) proves that the "List-in-Cell" contract is the primary blocker for system scalability.

## 5. Status
**NOT FOR IMPLEMENTATION.** This document serves only to acknowledge the architectural limit of the current "Boring" implementation. We are prioritizing bit-perfect stability over performance in the current cycle.

## 6. Discussion Starters
*   **Vectorized List-Creation:** Can we bypass the dictionary-of-lists and use a multi-step NumPy reshape to feed pandas?
*   **Lazy Reconstruction:** Should we move away from "The Big DF" and only reconstruct slices on-demand?
*   **Dask/Polars:** Is the ViEWS Outbound Contract fundamentally incompatible with the memory limits of a single-threaded pandas DataFrame?