# Implementation Plan: ADR Integrity and CIC Rollout

**Date:** 2026-02-19  
**Status:** Approved  
**Objective:** Resolve architectural discrepancies identified in the Cross-Review Audit and establish a complete set of Class Intent Contracts (CICs) to prevent semantic drift.

---

## Stage 1: Foundation Hardening (The Legal Fix) - [COMPLETED]
*Goal: Ensure the "Constitutional" layer of the repository is bit-perfect and free of numbering collisions.*

### 1.1 Resolution of Numbering Collisions - [DONE]
- **Action:** Renumber `docs/ADRs/active/007_volume_ledger_and_topology.md` to **ADR-012**.
- **Rationale:** ADR-007 is the authoritative "Silicon Protocol" (Constitutional). The Volume Ledger fits better in the `010-014` range dedicated to Spatiotemporal Topology.

### 1.2 Global Reference Patching - [DONE]
- **Action:** Perform a surgical multi-file replace to update stale ADR references:
    - `ADR-042` → `ADR-008` (Standardized Error Propagation)
    - `ADR-015` → `ADR-003` (Philosophy of Engineering)
    - `ADR-008` (Old context) → `ADR-009` (Boundary Contracts)
    - `ADR-013` (Old context) → `ADR-025` (Invariance/Scale)
- **Tooling:** Use `grep` and `replace` to ensure zero broken links in the `docs/` directory.

### 1.3 Terminology Unification - [DONE]
- **Action:** Standardize on **"Ontological Categories"** (from ADR-001) and retire "Functional Zones" in ADR-002 and ADR-044.
- **Goal:** Linguistic consistency across all architectural artifacts.

---

## Stage 2: Class Intent Contract (CIC) Rollout - [COMPLETED]
*Goal: Declare the "What" and "Why" for all non-trivial classes to govern silicon and carbon contributions.*

### Phase 1: The Spinal Components (Fortress Spine) - [DONE]
- **Targets:** `VolumeHandler`, `HydranetManager`, `DataSniffer`.
- **Logic:** These are the most critical "Custodian," "Orchestrator," and "Sentinel" roles that define the pipeline's stability.

### Phase 2: The Neural Bridge (Mathematical Actors) - [DONE]
- **Targets:** `InferenceOrchestrator`, `FeatureScaler`, `ModelArtifactFetcher`.
- **Logic:** Define the "Symmetry Engine" and the bit-perfect reversible math required for scientific integrity.

### Phase 3: Ingestion & Training (The Muscles) - [DONE]
- **Targets:** `DataFetcher`, `CurriculumLearner`, `VolumeSampler`.
- **Logic:** Formalize the "Mixed Salad" strategy and the "Dumb Mechanic" law of extraction.

---

## Stage 3: Verification and Closure
*Goal: Empirically prove that the system matches the new documentation.*

### 3.1 ADR Compliance Audit
- **Action:** Use `docs/audits/adr_compliance_audit_template.md` to generate a report verifying that the code actually implements the renumbered laws.

### 3.2 Commit and Sync
- **Action:** Merge `doc/architectural-docs-and-license` into `development` once all contracts are signed.

---

## Success Criteria
1.  **Zero Broken Links:** No "Dead" ADR references in any documentation.
2.  **Unique IDs:** No duplicate ADR numbers in the `active/` directory.
3.  **Traceability:** Every class in `views_hydranet/` that orchestrates or modifies data has a corresponding `.md` contract in `docs/CICs/`.
