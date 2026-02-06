# ADR 033: Naming Invariants and Semantic Intent

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | The Life Cycle of a Column Name |
| ADR Number          | 033               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 06.02.2026        |

## 1. Context
To maintain absolute stability across a multi-tasking spatiotemporal pipeline, we require a naming convention that is deterministic, immutable, and independent of mathematical scale. Previous iterations where column names changed to reflect transformations (e.g., `ln_` for logged data) led to "Logic Drift" and broken joins.

## 2. Decision: The "Immutable Semantic" Law
We enforce a naming protocol where column names describe **What the data represents** (Intent) rather than **How it is currently scaled** (Mathematics).

### 2.1 The Core Subjects
The atomic subjects of the system are: `sb_best`, `ns_best`, and `os_best`. These names represent the fundamental conflict types.

### 2.2 The Three Sacred Prefixes
Only three prefixes are permitted to modify a core subject:

1.  **`lr_` (Linear/Identity):** Signifies raw count intent. This is the "Identity" prefix. It is inherited from upstream and **must never be altered** to reflect numerical scaling (like `log1p` or `asinh`).
2.  **`by_` (Binary):** Signifies occurrence intent (1 if count > 0, else 0). This is the only valid transformation of the `lr_` prefix for actuals.
3.  **`pred_` (Prediction):** Signifies model output intent. This is prepended at the I/O edge.

### 2.3 The Evolution of a Name
The system recognizes only four valid states for a target (e.g., `sb_best`):

| State | Name | Intent |
| :--- | :--- | :--- |
| **Linear Actual** | `lr_sb_best` | The ground truth count. (May be raw or scaled). |
| **Binary Actual** | `by_sb_best` | The ground truth occurrence. |
| **Linear Prediction** | `pred_lr_sb_best` | The model's regression output. |
| **Binary Prediction** | `pred_by_sb_best` | The model's classification output (probability). |

## 3. Structural Invariants (The "Why")

### 3.1 Why we don't change `lr_` for math
Following **ADR 015 Law 6**, we reject "Mathematics-in-the-Name." If a column is named `lr_sb_best`, it remains `lr_sb_best` even if it is logged. 
*   **The Rationale:** The **Configuration** is the only authoritative record of scale. Changing the name creates a false sense of security; keeping the name ensures that internal components (like the `VolumeHandler`) can always find the correct target without parsing complex naming history.

### 3.2 The Symmetry Engine
Because naming is deterministic, the `VolumeHandler` can automatically "dress" model outputs. If the model is training on `lr_sb_best`, it knows exactly how to name the outbound heads: `pred_lr_sb_best` and `pred_by_sb_best`.

### 3.3 The 12-Feature Ledger (ADR 032)
For the standard 3-task model, the output DataFrame is guaranteed to have exactly 12 feature columns (6 actuals, 6 predictions). This strictness ensures join-safety with the `views-evaluation` library.

## 4. Consequences

**Positive Effects:**
- **Absolute Traceability:** You always know which target you are looking at.
- **Join Stability:** Names don't break when switching between logged and raw runs.
- **Simplicity:** No complex string manipulation logic is needed in the core architecture.

**Negative Effects:**
- **Verbosity:** Config files must be explicit about transformations because the column names won't tell you.

## 5. Rationale
We trade "at-a-glance" math visibility (which is often misleading) for **Systemic Integrity**. The name identifies the **Subject**; the Config identifies the **State**.
