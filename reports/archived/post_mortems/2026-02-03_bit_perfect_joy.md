# Post-Mortem: The Bit-Perfect Restoration & Joyful Refactor
**Date:** 03-02-2026  
**Subject:** Resolution of VRAM OOM, Naming Entropy, and Configuration Hardening  
**Status:** System Green / Architecture Hardened

---

## 1. Executive Summary
This session successfully resolved a critical hardware-block (VRAM OOM) and a growing technical debt in naming conventions. By applying the "Boring Architecture" philosophy (ADR 003), we moved from a fragile, "ghost-parameter" driven state to a bit-perfect, schema-guaranteed architecture. The result is a **Tolerant Handshake** that allows for seamless pipeline integration while strictly enforcing the physics of the HydraNet domain.

---

## 2. The Crises

### 2.1 The VRAM OOM Barrier
Researchers encountered immediate OOM crashes when processing the "Mixed Salad" (3 windows of 336 steps). 
*   **Root Cause:** Activation graphs were being accumulated across the entire lesson before `.backward()` was called, consuming ~6.1GB of VRAM just for activations on an 8GB card.
*   **Identification:** Analysis of `train_model.py` revealed that the optimizer was correctly gated, but the gradient graph was leaking across iterations.

### 2.2 The Naming & Symmetry Jungle
The expansion to a 6-head output topology (3x Regression, 3x Classification) triggered a cascade of naming collisions.
*   **Root Cause:** Internal model heads were being mapped to semantic names that collided with "Actuals" (ground truth) during spatiotemporal reconstruction in the `VolumeHandler`.
*   **Entropy:** The `HydranetManager` had become a "Rename Jungle," attempting to juggle `prefix`, `surfix`, and `signal` strings to prevent data loss.

---

## 3. The Interventions

### 3.1 Memory-Safe Accumulation (ADR 014 Hardening)
We implemented **Immediate Backpropagation**. The system now calls `.backward()` immediately after each window to purge activation graphs from VRAM while preserving the lesson-level weight update. This allows the model to scale to arbitrary temporal depths on consumer hardware.

### 3.2 The Symmetry Engine (ADR 020)
We formalized the 6-head output topology. 
*   **Internalization:** Naming invariants (`pred_`, `_raw`, `_prob`) were hardcoded as internal architectural secrets.
*   **Collision Protection:** `VolumeHandler` was hardened to use an `ACTUAL_INTERNAL_` prefix during reconstruction, ensuring ground-truth data is never overwritten by predictions.

### 3.3 The Tolerant Handshake (ADR 009 Evolution)
We initially attempted a "Selfish Component" model (`extra="forbid"`), which crashed due to pipeline baggage. 
*   **Synthesis:** We updated `HydraNetConfig` to exhaustively document all 46 keys used by the domain logic (removing "ghosts") while setting `extra="allow"` to tolerate external metadata.
*   **Checksum Law:** We mandated explicit redundancy for critical couplings (e.g., `input_channels == len(features)`), ensuring logical drift is caught at the configuration gate.

---

## 4. The Popperian Proof
To falsify the hypothesis of hallucination, we implemented `tests/popperian_audit.py`, which enforces four "Hard Gates":
1.  **Gate 1 (Symmetry):** Verified 6-head naming and actuals protection.
2.  **Gate 2 (Topology):** Verified automatic MultiIndex restoration.
3.  **Gate 4 (Memory):** Verified VRAM graph clearance.
4.  **Gate 4 (Strictness):** Verified the Pydantic handshake.

**Results:** All gates passed. The architecture is proven.

---

## 5. Lessons Learned
*   **Redundancy is Boring (and Good):** Demanding both a list and its length as a checksum prevents silent, hard-to-debug failures in data slicing.
*   **Internalize Invariants:** Parameters that define the "Physics" of the package (like naming conventions) should be hidden from the researcher to reduce the configuration surface area.
*   **Sanitize, Don't Seclude:** A "Tolerant Handshake" is superior to strict exclusion in a complex ecosystem. We should strictly validate what we *need* and ignore what we *don't*.

---

## 6. Artifact Trail
*   `views_hydranet/utils/utils_config.py`: The hardened schema.
*   `views_hydranet/utils/volume_handler.py`: The Internalized Naming Engine.
*   `views_hydranet/manager/hydranet_manager.py`: The simplified Orchestrator.
*   `tests/popperian_audit.py`: The permanent Truth Engine.
*   `ADR 009`, `ADR 020`: The updated architectural laws.

**System State:** bit-perfect. 🖖
