# ADR 017: The Inbound Handshake (Ingestion and Validation)

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Standardizing Data Retrieval and Integrity |
| ADR Number          | 017               |
| Status              | Proposed          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Functional Categorization (The Ingestor and Sentinel)

### Zone 1: Physical Retrieval (The `DataFetcher`)
*   **Responsibility:** Locating and reading raw DataFrames from disk.
*   **The Law of Structure:** The Fetcher never "guesses" which index level represents Time or Space. It pulls the authoritative names from the `index_names` entry in the configuration.
*   **The Flat Exit:** The final product must be a flat `pd.DataFrame` with a standard numeric index to provide a clean "Raw Space" starting point.

### Zone 2: Integrity Validation (The `DataSniffer`)
*   **Responsibility:** Passive observer that verifies data state at major pipeline boundaries.
*   **The Ingestion Suite:** Verifies column existence, (Time, ID) uniqueness, and 100% finiteness of mandatory identity columns.
*   **The Alignment Suite:** Verifies spatiotemporal continuity (e.g., `history_end + 1`) and absolute geographic anchoring.

---

## 2. Structural Invariants (The "Spirit")

1.  **Passive Observer Law:** Both Fetcher and Sniffer are strictly prohibited from modifying the data content (no dropping rows, no scaling).
2.  **Zero-Inference Handshake:** If mandatory config keys (`index_names`) are missing, the handshake must fail immediately.
3.  **Fail Loud and Proud:** Any violation of a contract MUST result in an immediate, descriptive exception.

---

## 3. Data Flow Topology
`Disk` → **`DataFetcher` (Retrieval)** → `DF` → **`DataSniffer` (Validation)** → `HydranetManager`.

---

## 4. Rationale
Spatiotemporal integrity is easily corrupted by silent data drift. By unifying Fetching and Sniffing into a single "Handshake," we ensure that the model never learns from "Broken Physics" or incorrect geographic alignment.
