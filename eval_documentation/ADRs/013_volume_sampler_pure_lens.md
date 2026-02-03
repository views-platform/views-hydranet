# ADR 013: Specification for VolumeSampler (The Pure Lens)

**Status:** Proposed  
**Context:** Supersedes ADR 009. To achieve 100% separation of concerns, the `VolumeSampler` is now a pure geometric tool. It has no knowledge of training progress or schedules; it simply extracts windows based on explicit instructions.

---

## 1. Functional Categorization (The "Zones")

### Zone 1: Identification (The Busy-Search)
*   **Responsibility:** Scanning the global volume for cells that meet an explicit difficulty threshold.
*   **Contract:** Receives a `target_name` and a `threshold`. It returns a list of geographic coordinates `(y, x)` that satisfy `activity[target_name] >= threshold`.

### Zone 2: Extraction (The Bridge)
*   **Responsibility:** Slicing the global volume into a local `VolumeHandler`.
*   **Contract:** Receives geographic coordinates. It must return a **mini-VolumeHandler** with a correctly adjusted `spatial_offset`.

---

## 2. Structural Invariants (The "Spirit")

1.  **The Mini-Custodian Law:** A sample is NEVER a "naked" tensor. It is a fully functional `VolumeHandler`. Data and Ledger must never be decoupled. This ensures every 32x32 window knows its own geographic truth.
2.  **Dumb Mechanic Law:** The Sampler never "decides" what is a good sample. It only filters based on the `Lesson` provided by the Planner (`CurriculumLearner`).
3.  **Local Reproducibility:** The Sampler uses a local `np.random.Generator` to ensure that spatial jitter within a selected window is deterministic for a given seed.

---

## 3. Data Flow Topology
`CurriculumLearner` → **`VolumeSampler`** → `Tuple[List[VolumeHandler], int]` → `Trainer`.

---

## 4. Contractual Precision (The "Constraints")

### `get_batch(target_name: str, threshold: int, batch_size: int) -> Tuple[List[VolumeHandler], int]`
*   **Pre-condition:** `target_name` must exist in the Ledger.
*   **Post-condition:** Returns a tuple containing:
    1.  A list of bit-perfect spatial windows anchored to the correct geographic offsets.
    2.  An integer `qualified_count` representing the number of "Busy" cells found globally for transparency.

---

## 5. Semantic Naming
*   `The Lens`: The Sampler's role as a viewer of a subset of the world.
*   `Jitter`: The stochastic spatial offset applied to a "Busy" cell to prevent center-bias overfitting.
