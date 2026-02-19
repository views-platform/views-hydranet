# The Hardened Protocol: Contributor Governance

This document defines the mandatory engineering and mathematical standards for the `<Project Name>` repository. Adherence to this protocol is required for all contributions to guarantee absolute scientific integrity and reproducibility.

---

## 1. Core Principles

### A. The Authority of Declarations (ADR-003)
**"Never infer; only trust declarations."**
All meaningful semantics (architectures, loss functions, scaling strategies, seeds) must be explicitly declared in the Configuration Manifest (DNA). 
- **Prohibited:** Filename-based logic, directory-structure inference, or shape-based guessing.
- **Requirement:** If a parameter affects model identity, it must be a mandatory gene in the `ReproducibilityGate`.

### B. The Fail-Loud Mandate (ADR-008)
**"A crash is a successful defense of scientific integrity."**
Silent failures, implicit fallbacks, and "best-effort" corrections are forbidden. 
- **Requirement:** Violations of physical, temporal, or configuration invariants must raise an explicit `ReproducibilityError` or `NumericalSanityError` immediately.
- **Prohibited:** Using `nan_to_num`, silent clipping, or "sensible defaults" for critical parameters.

### C. The Numerical Airlock (ADR-010)
All data entering the system must pass through a numerical airlock.
- **Requirement:** Downcast all input to `float32` immediately.
- **Requirement:** Detect and raise errors on NaNs or Infs at every boundary (Data entry, Loss calculation, Prediction output).

### D. Physical Symmetrical Architecture (ADR-025)
**"1 Class, 1 File, 1 Name."**
Organizational Zen is a requirement for maintainability.
- **Requirement:** Every non-trivial class must live in its own file named after the class in `snake_case`.
- **Requirement:** Heterogeneous logic (callbacks, patches, exceptions) must be consolidated into pre-defined symmetrical hubs (`utils/callbacks.py`, `utils/patches.py`).

---

## 2. Contributor Requirements

### Adding a New Component (Model, Algorithm, Transform)
1.  **Define the Genome:** Register mandatory hyperparameters in the `ReproducibilityGate` or Config Manifest.
2.  **Symmetrical Entry:** Create the file following the 1-Class-1-File rule.
3.  **Create Specs/CICs:** Write the **Specification Card** and **Class Intent Contract (CIC)**.
4.  **Register in Catalog:** Add instantiation logic to the appropriate catalog or factory.

---

## 3. Mandatory Testing Taxonomy (ADR-005)

Every Pull Request must include tests covering the following three perspectives:

### 🟩 Green Team (Stability & Correctness)
*   **Goal:** Ensure the system works as intended and remains stable.
*   **Examples:** Gradient verification, bit-identical reloads, scaling integrity.

### 🟫 Beige Team (DNA & Human Error)
*   **Goal:** Catch failures caused by common configuration mistakes or missing parameters.
*   **Examples:** Manifest audits (blocking runs if a gene is missing), input shape/type verification.

### 🟥 Red Team (Adversarial)
*   **Goal:** Expose failure modes by deliberately trying to make the system lie or fail.
*   **Examples:** Injecting temporal holes, future-peeking injections, numerical poisoning (NaN/Inf injection).

---

## 4. Operational Invariants

- **Hardware Self-Healing:** Models/components must implement Verify-and-Restore to prevent hardware drift.
- **Entropy Locking:** All probabilistic operations must be preceded by a seed reset via a centralized reproducibility utility.

---

🖖 **"In this repository, we value bit-perfect reproducibility over convenient execution."**
