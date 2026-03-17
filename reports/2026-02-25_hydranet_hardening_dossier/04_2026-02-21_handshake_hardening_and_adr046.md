# Post-Mortem: Evaluation Handshake Hardening and ADR 046 Implementation
**Date:** 2026-02-21  
**Author:** HydraNet Lead Engineer  
**Status:** Resolved  

---

## 1. Executive Summary
This work package began as a fix for a `RuntimeError` (channel mismatch) and evolved into a fundamental architectural refactor of the `views-hydranet` feature lifecycle. We successfully transitioned from an implicit, discovery-based data model to an explicit **Instructional Blueprint (ADR 046)**.

This transition revealed a deep "Ontological Clash" with the `views_pipeline_core` and `views-evaluation` libraries, necessitating a coordinated hardening of the ecosystem's evaluation handshake.

---

## 2. The Incidents

### 2.1 The "Channel Mismatch" (The Trigger)
*   **Symptom:** `RuntimeError: Given groups=1, weight of size [4, 3, 3, 3], expected input[1, 6, 32, 32]...`
*   **Root Cause:** The model architecture expected 3 input channels, but the data loader blindly passed everything it found (6 channels).
*   **Resolution:** Implemented explicit index mapping in `train_model` and `hydranet_inference`, ensuring the model only receives the features defined in `config["features"]`.

### 2.2 The "KeyError" (The Ontological Clash)
*   **Symptom:** `KeyError: "['by_sb_best', 'by_ns_best', 'by_os_best'] not in index"` during evaluation.
*   **Root Cause:** `views_pipeline_core` assumes all targets exist as columns on disk ("Naked Load"). However, HydraNet now manufactures binary targets (`by_`) on-the-fly using the Blueprint.
*   **Resolution:** 
    1.  **Core Hardening:** Introduced `prepare_actuals_df` hook in `ModelManager` (pipeline core).
    2.  **Edge Fulfillment:** Overrode this hook in `HydranetManager` to call `DataFetcher.apply_blueprint`, manufacturing the missing targets just-in-time for evaluation.

### 2.3 The "Ontological Ceiling" (The Evaluator)
*   **Symptom:** `ValueError: Target by_sb_best is not a valid target`.
*   **Root Cause:** `views-evaluation` had a hardcoded whitelist of valid prefixes (`ln`, `lx`, `lr`). It rejected the new `by_` prefix.
*   **Resolution:** Proposed "Ontology Liberation" for `views-evaluation`—making the evaluator data-agnostic (Passenger vs. Gatekeeper).

### 2.4 The "Success Trap" (CUDA OOM)
*   **Symptom:** `CUDA out of memory` during inference on large grids.
*   **Root Cause:** Fixing the functional bugs allowed the model to run the full sequence for 100 posterior samples. The massive accumulation of activation graphs (and redundant GPU transfers) overwhelmed VRAM.
*   **Resolution:** 
    1.  Moved `full_tensor` to GPU **once**.
    2.  Wrapped MC Dropout loop in `torch.no_grad()`.
    3.  Added explicit `torch.cuda.empty_cache()` between samples.

---

## 3. Key Architectural Deliverables

### 3.1 Symmetric Feature Lifecycle (ADR 046)
We moved from "Magic" to "Instruction."
*   **Transformations:** Mathematical scaling (e.g., `log1p`).
*   **Derivations:** Creation of new signals (e.g., `binary` targets) from existing ones.
*   **Impact:** Training and Evaluation now use the *exact same code path* (`DataFetcher.apply_blueprint`) to generate ground truth, guaranteeing bit-perfect alignment.

### 3.2 The Handshake Protocol
We established a new contract between the generic core and the specific model:
*   **Core:** "I provide the raw material."
*   **Model:** "I manufacture the actuals."
*   **Evaluator:** "I measure the difference."

---

## 4. Lessons Learned
1.  **Success Unmasks Debt:** Fixing the first crash often reveals the deeper scaling limit (the OOM).
2.  **Boundaries Matter:** Trying to hack the handshake at the edge (`fetch_viewser_df` override) failed. The only robust solution was to formalize the protocol in the foundation (`prepare_actuals_df`).
3.  **Ontology vs. Reality:** Systems that assume data schemas (like the evaluator's whitelist) will always block innovation. Data-agnostic components are more resilient.

## 5. Future Watchlist
*   **Scaling:** As grids grow to 360x720, we may need to implement "Tiled Inference" to break the spatial dimension, as VRAM will again be the bottleneck.
*   **Core Merge:** The changes to `views_pipeline_core` and `views-evaluation` must be merged before this branch can be fully deployed in production.

---
**Signed:** HydraNet Engineering Team
