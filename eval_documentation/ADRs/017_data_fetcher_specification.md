# ADR 017: Specification for DataFetcher (The Ingestor)

**Status:** Proposed  
**Context:** Data ingestion is the first point of contact with the "Wild World." To maintain topological integrity, we require a strictly specified component that standardizes diverse data formats into a canonical sparse table.

---

## 1. Functional Categorization (The "Zones")

### Zone 1: Physical Retrieval (I/O)
*   **Responsibility:** Locating and reading raw DataFrames from disk.
*   **The Handshake:** It must validate the physical existence of the data partition (`calibration`, `validation`, `forecasting`) before attempting ingestion.

### Zone 2: Structural Standardization (The Bridge)
*   **Responsibility:** Mapping complex MultiIndex structures to a flat, role-based table.
*   **The Law:** The Fetcher never "guesses" which index level represents Time or Space. It must pull the authoritative names from the `index_names` entry in the configuration.

### Zone 3: Structural Gateway (Flat Exit)
*   **Responsibility:** Standardizing the MultiIndex into a format consumable by the VolumeHandler.
*   **The Law of Content Preservation:** The Fetcher is a "Passive Ingestor." It is strictly prohibited from dropping rows, filling NaNs, or scaling values. It only resets the index to provide a flat table.

---

## 2. Structural Invariants (The "Spirit")

1.  **Zero-Inference Handshake:** If `index_names` is missing from the config, the Fetcher must fail immediately. 
2.  **Explicit Transformation Only:** Content modification (e.g., scaling) is the responsibility of the `FeatureScaler`, not the Fetcher.
3.  **Flat Output:** The final product of the Fetcher must always be a flat `pd.DataFrame` with a standard numeric index. MultiIndices must be reset after validation.

---

## 3. Data Flow Topology
`Disk` → **`DataFetcher` (Structure)** → `Sparse DF` → **`FeatureScaler` (Content)** → `Semantic DF` → **`VolumeHandler` (Topology)**.

---

## 4. Contractual Precision (The "Constraints")

### `fetch_df() -> pd.DataFrame`
*   **Pre-condition:** Physical data path must be valid and accessible.
*   **Post-condition:** Returns a raw DataFrame as stored on disk.

### `standardize_raw_df(df: pd.DataFrame, config: dict) -> pd.DataFrame`
*   **Pre-condition:** The `df` must possess a MultiIndex matching the `index_names` in the config.
*   **Post-condition:** Returns a flattened, geographically sanitized DataFrame (Ocean cells removed).

---

## 5. Semantic Naming
*   `index_names`: The authoritative list of MultiIndex levels providing spatiotemporal context.
*   `sanitization`: The removal of invalid or non-geographic (Ocean) rows.
