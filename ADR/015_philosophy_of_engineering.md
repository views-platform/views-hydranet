# ADR 015: Philosophy of Engineering (Boring Architecture)

**Status:** Active  
**Context:** Spatiotemporal modeling is high-entropy and high-risk. To manage this complexity, we reject "clever" software engineering in favor of a "Boring" architecture. This document defines the philosophical laws that govern every line of code in this repository.

---

## 1. Law 1: Fail Loud and Proud (Anti-Fragility)
We strictly reject silent error handling, magic fallbacks, and "best-effort" execution.
*   **The Principle:** If the data or configuration violates its contract, the system MUST crash immediately with a descriptive error.
*   **Prohibited:** 
    *   Magic default values (e.g., assuming a spatial resolution if not provided).
    *   Silent truncation (e.g., evaluation ignoring months if history ends prematurely).
    *   Try-Except blocks that swallow structural errors.
*   **Goal:** Ensure that every successful run is mathematically valid. A crash is a feature that protects the integrity of the science.

---

## 2. Law 2: Zero-Magic (Total Traceability)
The code is a passive executor of instructions; it is not a simulation of "Hidden Physics."
*   **The Principle:** No hardcoded strings, magic indices, or positional assumptions.
*   **The Ledger:** Every spatiotemporal role (Time, ID, Row, Col) must be explicitly mapped in a Ledger at initialization. 
*   **Goal:** Decouple the logic from the dataset. The code should be able to process any grid (VIEWS, NASA, Caster) simply by updating the configuration.

---

## 3. Law 3: Explicit Scaffolding (No Orphaned Data)
Tensors are meaningless without their spatiotemporal context.
*   **The Principle:** Data never travels alone. It must always be wrapped in a **Custodian** (`VolumeHandler`) that carries its own metadata.
*   **The Scaffold:** Predicted signals must always be "dressed" in identities provided by an authoritative history scaffold.
*   **Goal:** Eliminate "Topological Drift." Ensure that any prediction can be traced back to its geographic and temporal origin with bit-perfect precision.

---

## 4. Law 4: The Mixed Salad (Symmetry)
Multi-task learning requires simultaneous, not sequential, optimization.
*   **The Principle:** Every parameter update must be informed by a diverse representative batch of all conflict tasks.
*   **Mechanism:** High-frequency subject oscillation and gradient accumulation (The Optimization Gate).
*   **Goal:** Stability. Prevent the model from "forgetting" one task while learning another.

---

## 5. Law 5: Explicit Transformation (No Silent Logic)
Data content modification (scaling, augmentation, binarization) is a strategic act and must be treated as such.
*   **The Principle:** Logic that changes cell values or row counts must never be "bundled" into ingestion or structural bridges.
*   **The Mechanism:** Transformations must be implemented in specialized classes (e.g., `FeatureScaler`) and invoked explicitly by the `HydranetManager`.
*   **The Constraint:** Every transformation must be triggered by a specific `config` entry. If the config is silent, the data remains raw.
*   **Goal:** Traceability. A researcher must be able to look at the `Manager` and see exactly when and where the "Raw" data became "Semantic" data.

---

## 6. Law 6: The Prefix-Purity Law (No False Comfort)
We reject the practice of renaming columns (e.g., `lr_` to `ln_`) as they undergo mathematical transformations. 
*   **The Principle:** Column prefixes describe **Semantic Intent** (Linear vs. Binary), not **Numerical Scale** (Logged vs. Raw).
*   **Authority:** The `config` Fit/Transform state is the only authoritative record of a column's current scale. 
*   **False Comfort:** Changing a prefix does not prove a transformation occurred; it only creates a misleading audit trail. We preserve the original name and trust the `config` to govern the math.
*   **The Edge Exception:** Standard prefixes (`pred_lr_`, `pred_by_`) are enforced only at the **I/O Edges** to satisfy external consumers.

---

## 7. Consequences
By following these laws, we trade **Initial Speed** for **Permanent Trust**. 
*   It takes longer to set up a run (because the config is verbose).
*   It is harder to write new features (because they must be verified against ADRs).
*   **BUT:** We achieve bit-wise reproducibility and absolute confidence in our spatiotemporal predictions.
