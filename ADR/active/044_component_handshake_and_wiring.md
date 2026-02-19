# ADR 044: Component Handshake and Wiring

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Managing Dependencies and Inter-Component Data Flow |
| ADR Number          | 044               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. The Component Gluing (The Handshake)
This specification defines how specialized "Boring" components are wired together to form the HydraNet pipeline.

### 1.1 The "Boring" Actors
*   **`DataFetcher`**: Standards-compliant retrieval from disk.
*   **`VolumeHandler`**: The spatiotemporal custodian.
*   **`InferenceOrchestrator`**: The unified symmetry engine (ADR 038).

### 1.2 The Law of Delegation
The Manager is prohibited from performing math or data transformation directly. It trusts the specialized components to return bit-perfect, topologically correct, and correctly named DataFrames.

### 1.3 Zero-Management of Keys
The Manager is prohibited from performing manual string renaming or index manipulation. If a column needs a prefix or a MultiIndex needs restoration, it must be handled by the specialized gate inside the `VolumeHandler` (ADR 043).

---

## 2. Data Flow Topology

### Training Flow:
`Config` → **`HydranetManager`** → `[Fetcher -> Sniffer -> Scaler -> Handler]` → `[Sampler -> Planner -> Trainer]` → `Artifact`.

### Evaluation Flow:
`Artifact` → **`HydranetManager`** → `[Fetcher -> Sniffer -> Scaler -> Handler]` → `[Retriever -> Evaluator]` → `Contract DF`.

---

## 3. Rationale
By moving to a "Wiring-Only" orchestrator, we ensure that as the complexity of individual components (like `DataSniffer`) grows, the central manager remains simple, readable, and easy to maintain.
