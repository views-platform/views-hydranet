# Diagnosis: The Random Number Generator Incident

**Date:** 2026-02-19
**Subject:** Root Cause Analysis of Performance Collapse
**Status:** Investigation

---

## Executive Summary
The model, previously SOTA, is now producing output indistinguishable from random noise or static. This behavior typically arises from one of three mathematical failures: **Input Scrambling** (Spatial/Temporal), **Gradient Explosion/Collapse** (Scaling/Init), or **Architectural Disconnect** (Config vs Model).

Based on a deep code review, the most probable cause is **Spatial Scrambling** due to incorrect `row_offset`/`col_offset` configuration, followed closely by a **Scaling Handshake Failure**.

---

## 1. The Spatial Scrambling Hypothesis (High Probability)
**Mechanism:** `VolumeHandler.from_df` uses numpy fancy indexing:
`r_idx = (df[y_col] - row_offset).astype(int).values`

**The Fault Line:**
- If `row_offset` is incorrect (e.g., set to `0` when data starts at index `100` on a global grid), `r_idx` becomes correct `100`.
- **CRITICAL:** If `row_offset` is set to `100` but the dataframe contains local coordinates (starting at 0), `r_idx` becomes `-100`.
- **Result:** Numpy wraps negative indices. Data is written to the *end* of the array. The map is spatially inverted or scrambled.
- **Symptom:** A CNN fed spatially scrambled data will produce uncorrelated noise ("Random Numbers"), as the spatial autocorrelation it relies on is destroyed.

---

## 2. The Scaling Handshake Failure
**Mechanism:** `FeatureScaler` applies transformations based on the `transform` dictionary in the config.

**The Fault Line:**
- `HydraNetConfig` allows extra keys (`extra="allow"`).
- If the "Integration Refactor" changed the config structure (e.g., `classification_targets` missing from `transform`), the scaler might skip transformation for those columns.
- **Result:** The model (initialized for normalized inputs ~0-1) receives Raw Counts (0-5000+).
- **Symptom:** Immediate gradient explosion or ReLU saturation, leading to `NaN`s or constant zero outputs.

---

## 3. The "Hardcoded Heads" Conflict
**Mechanism:** `HydraBNUNet06_LSTM4` hardcodes 3 Regression + 3 Classification heads. `utils.py` hardcodes a 6-channel loss mask `[T,T,T,F,F,F]`.

**The Fault Line:**
- If the configuration provided by the new orchestration layer specifies fewer than 3 targets (e.g., testing with just `sb`), the loss function will misalign.
- **Result:** The model optimizes `Head 1` against `Target 2`'s labels.
- **Symptom:** The model fails to converge and outputs mean-value garbage.

---

## 4. The Autoregressive Lobotomy (Rule Out?)
**Mechanism:** `t0 = t1_pred.detach()`.
**Status:** **Ruled Out**. You confirmed the model uses only endogenous conflict features (3 channels). The input/output shapes match (3 in, 3 out).

---

## 5. The Sorting Vacuum
**Mechanism:** `DataFetcher` does not explicitly sort the DataFrame by Time/ID.
**The Fault Line:** `VolumeHandler` relies on `df` being a consistent stream for `DataSniffer` continuity checks. While `VolumeHandler` placement is robust (fancy indexing), unsorted data can cause the `DataSniffer` to falsely report success or fail to catch gaps.

---

## 6. Next Steps
The remediation requires answering the 6 Foundational Questions to isolate which of these fault lines has fractured.
