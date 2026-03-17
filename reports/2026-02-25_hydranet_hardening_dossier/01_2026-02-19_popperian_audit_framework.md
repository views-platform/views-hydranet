# Popperian Audit Framework: Spatiotemporal Volume Integrity

**Objective:** Rigorously test the `VolumeHandler` implementation against two core hypotheses to find the truth, with no bias toward verification or falsification.

---

### Hypothesis 1: Specification Ambiguity
**The Claim:** The definitions of "Historical", "Evaluation", and "Forecast" are conceptually leaky and allow for overlapping or undefined behavior.

#### Falsification Test 1.1: The "Temporal Overlap" Probe
*   **Vector:** Attempt to call `to_evaluation_df` using a `start_idx` that causes the prediction window to extend *partially* beyond the end of the history provider.
*   **Success Criteria:** The system must either strictly reject the call (Contract Enforcement) or provide a deterministic, documented behavior for the "Hybrid" state. Silent truncation or index errors verify the hypothesis of ambiguity.

#### Falsification Test 1.2: The "Identity Authority" Probe
*   **Vector:** Provide a prediction volume that contains its own identity channels (e.g., a `month_id` channel) but with values that contradict the History Provider.
*   **Success Criteria:** The system must demonstrate an authoritative hierarchy. If the implementation allows the "Signal" volume to redefine its own "Place in Time" during an evaluation call, the specification is ambiguous.

---

### Hypothesis 2: Implementation Divergence (Spirit vs. Reality)
**The Claim:** The implementation pays lip service to the "Ledger" but relies on hidden assumptions, magic strings, or positional invariants.

#### Falsification Test 2.1: The "Alias" Audit (The String Test)
*   **Vector:** Use a configuration where `priogrid_gid` is renamed to `cell_index` and `month_id` is renamed to `temporal_step`.
*   **Success Criteria:** Bit-perfect reconstruction using arbitrary names. If the code crashes looking for "priogrid_gid", the "Boring/Ledger" spirit has been violated.

#### Falsification Test 2.2: The "Geographic Shift" Audit (The Absolute Anchor Test)
*   **Vector:** Create a History volume anchored at `(0,0)` and a Prediction volume anchored at `(10,10)`.
*   **Success Criteria:** If the system maps the prediction's `[0,0]` to the history's `[0,0]` despite different geographic offsets, the claim of "Absolute Anchoring" is a falsified implementational myth.

#### Falsification Test 2.3: The "Dimensionality Collision" Audit (The 5D Test)
*   **Vector:** Inject a 5D Probabilistic Tensor (Samples dimension) into the reconstruction methods.
*   **Success Criteria:** The system must handle the 5th dimension via an explicit convention or raise a specific `ContractViolationError`. Unhandled `IndexError` or silent dimension loss verifies Hypothesis 2.

---

### The Audit Roadmap

| Script Name | Target | Focus |
| :--- | :--- | :--- |
| `audit_temporal_boundaries.py` | H1 | Window overflows and temporal leakage. |
| `audit_ledger_independence.py` | H2 | Custom column names and permuted channel maps. |
| `audit_spatial_rigor.py` | H2 | Mismatched offsets and absolute coordinate alignment. |
| `audit_probabilistic_bridge.py` | H1/H2 | 5D tensor handling and data-type preservation. |
