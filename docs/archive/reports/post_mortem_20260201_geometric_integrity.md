# Post-Mortem: Restoration of Geometric Integrity and Pipeline Hardening
**Date:** 01-02-2026  
**Status:** Canonical State Restored  
**Focus:** Data Ingestion, Volume Construction, and Numerical Stability

---

## 1. Executive Summary

Between January 30 and February 1, 2026, the HydraNet pipeline experienced a series of cascading failures during training, culminating in consistent CUDA Out-of-Memory (OOM) errors even on small datasets. Initial investigations incorrectly attributed this to sequence length or batch sizing. 

A forensic audit revealed that the true cause was an **Activations Explosion** triggered by a regression in the data pipeline's geometric logic. Specifically, recent refactors had destroyed the "Structural Invariants" of the spatiotemporal volumes, causing the model to receive raw PrioGrid IDs (integers ~250,000) as features instead of log-scaled counts (~7.0).

To resolve this, we transitioned the architecture from "Smart/Dynamic" logic to a "Stable/Boring" foundation centered around two new authoritative classes: `VolumeHandler` and `VolumeSampler`. We have successfully restored the system to a bit-perfect symmetric state where data can be rasterized and vectorized without losing geographic or temporal identity.

## 2. The Crisis: The Activation Explosion & OOM

### Symptoms
During standard training runs on the 'purple_alien' dataset, the GPU memory would max out almost instantly upon entering the first batch. 
* **The Clue:** Progress bars indicated the model was making it through roughly 250 months before crashing.
* **The Misdirection:** Because the crash happened during long sequences, we initially suspected that the RNN (LSTM) was holding too many gradients in memory. 

### Discovery
Detailed audit logging revealed that the Magnitude of the input tensors was the variable that actually broke the hardware. In a ReLU-based architecture, receiving a value like **250,000** (a PrioGrid ID) into a layer with even small weights results in an activation value so large that the resulting gradients bloat the computation graph beyond the 8GB VRAM capacity.

We learned that **GPU OOM is often a symptom of numerical instability**, which itself is a symptom of data-contract violation.

## 3. Root Cause Analysis: The Fall of the Structural Invariant

The audit of the git history identified three primary failure points:

### A. The "Smart Discovery" Regression
We replaced a hardcoded positional index (Index 5) with a "smart" helper (`_get_feature_indices`) that scanned column names. However, when the data reached the trainer as a NumPy array, the column names were missing. The helper fell back to unvalidated defaults, which caused a **Channel Shift**. The model began "reading" PrioGrid IDs as if they were conflict features.

### B. The Configuration "Split-Brain"
The introduction of a complex "Handshake" created two sources of truth: the core `self.configs` and a shadow `self._hydranet_config`. Over time, these drifted. Validation became a side-effect of execution rather than a gate, allowing misaligned volumes to be built without a "Fail-Fast" trigger.

### C. Relative vs. Absolute Geometry
The volume builder was using `row - row.min()`. While memory-efficient, this made the coordinate `(0,0)` in the array geographically ambiguous—it depended on the dataset slice. This "Dynamic Shifting" made it impossible to verify the North-Up flip without knowing the specific bounds of every training batch.

## 4. The Solution: Building the 'Boring' Foundation

We restored the architecture by implementing four "Immovable Reference Points":

### 1. VolumeHandler (The Custodian)
We moved away from dynamic logic to **Absolute Anchoring**. The `VolumeHandler` now uses `row` and `col` directly as indices relative to a fixed geographic datum (e.g. Row 87). It owns an **Immutable Ledger** that tracks axis labels, channel maps, and spatial offsets. Every volume is now "Born with a Name."

### 2. VolumeSampler (The Lens)
We decoupled the act of stochastic windowing from the immutable global volume. The trainer no longer handles naked arrays; it receives windowed `VolumeHandler` objects. This ensures that even a tiny 32x32 sample carries its own coordinate pedigree and channel map.

### 3. Sacred Source (Conf-Purity)
We purged the handshake shims. The Manager now consumes `self.configs` directly from the Pipeline Core. Redundant secondary states were deleted. Temporal logic (time_steps) is now explicitly calculated as `len(steps)` at the call site, eliminating mutable drift.

### 4. IntegrityGuardian (The Sentinel)
We added a passive monitor that performs "Numerical Forensic Audits" on every sequence. It provides a hard stop (RuntimeError) if loss becomes NaN or if predictions exceed plausible magnitudes (>10,000), preventing the "Garbage In, Garbage Out" cycle from wasting GPU time.

## 5. Lessons Learned: Precision over Cleverness

1. **Boring is Reliable:** In spatiotemporal systems, "Clever" logic (like relative shifting or smart discovery) is a liability. Precision and positional invariants must be the foundation.
2. **The "Carrier" Pattern:** Tensors should never travel alone. Always wrap them in a metadata object (like `VolumeHandler`) that answers the question "Where is this value located?" at any point in the stack.
3. **Visibility is Security:** The `visual_audit()` method is as important as a unit test. If the map looks wrong, the model will be wrong.
4. **Trust the Source:** Configuration should be read-only and authoritative. Calculation should happen at the call site rather than being cached in derived state.

**Conclusion:** The pipeline is now significantly more robust, traceable, and "Rust-like" in its enforcement of data contracts. 🖖
