# ADR 012: Specification for CurriculumLearner (The Planner)

**Status:** Proposed  
**Context:** To prevent mode collapse and stochastic instability, the training path must be governed. This ADR defines the `CurriculumLearner` as the strategic authority responsible for the training trajectory.

---

## 1. Functional Categorization (The "Zones")

### Zone 1: Configuration (The Handshake)
*   **Responsibility:** Initializing the trajectory based on hyperparameters.
*   **Input:** `config: dict`, `ledger: VolumeMetadata`.
*   **Validation:** Verifies that all targets defined in `config` exist in the ledger's `feature_cols`.

### Zone 2: Trajectory Calculation (The Cooling)
*   **Responsibility:** Implementing the "Mathematical Cooling" of the search threshold.
*   **Logic:** Uses the linear decay function (`my_decay`) to map the current `sample_idx` to a `min_events` threshold.
*   **Goal:** Enforce Signal Anchorage (high signal early, sparse data late).

### Zone 3: Subject Rotation (The Oscillation)
*   **Responsibility:** Ensuring balanced learning across all conflict tasks.
*   **Logic:** Alternates the search target across `sb`, `ns`, and `os` based on the **global window index** (Sample index * Batch size + Batch element index).
*   **Goal:** Enforce Multi-Task Priming at the highest possible frequency (The Mixed Salad).

---

## 2. Structural Invariants (The "Spirit")

1.  **Strategic Isolation:** The Planner knows "What" to look for and "How Hard" to look for it, but it never touches the numerical data directly.
2.  **Stateless Math:** Given the same `sample_idx` and `config`, the Planner must return the identical `Lesson` (Target + Threshold).
3.  **Zero-Magic:** All targets are looked up via the Ledger's names, never hardcoded VIEWS strings.

---

## 3. Data Flow Topology
`Trainer` → **`CurriculumLearner`** → `Lesson (Target, Threshold)` → **`VolumeSampler`**.

---

## 4. Contractual Precision (The "Constraints")

### `get_lesson(sample_idx: int) -> Tuple[str, int]`
*   **Pre-condition:** `sample_idx` must be within the bounds of `total_samples`.
*   **Post-condition:** Returns a tuple of `(target_column_name, event_threshold)`.

---

## 5. Semantic Naming
*   `Lesson`: The atomic unit of training instruction for a single epoch/sample.
*   `Cooling`: The process of reducing the threshold over time.
*   `Subject`: The specific conflict feature targeted for "Busy-Search."
