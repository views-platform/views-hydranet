# ADR 002: Topology and Dependency Rules

| ADR Info            | Details           |
|---------------------|-------------------|
| Subject             | Enforcing Directional Dependencies |
| ADR Number          | 002               |
| Status              | Accepted          |
| Author              | Gemini CLI        |
| Date                | 19.02.2026        |

---

## 1. Context
Architectural fragility emerges from uncontrolled dependencies. Without explicit rules, high-level modules begin depending on low-level details, creating circular dependencies and constraining system evolution.

---

## 2. Decision: Strict Directional Topology
This repository enforces a strict, hierarchical dependency structure.

### 2.1 The Dependency Law
**Dependencies must follow declared architectural direction.** No component may depend on a layer above it or a layer outside its functional zone.

### 2.2 Hierarchical Layers
1.  **Manager (Top):** Orchestrates Actors and Custodians. Depends on everything.
2.  **Inference/Train (Mid):** Implements logic. Depends on Utils and Custodians.
3.  **Utils/Actors (Low):** Specialized mathematical or logical units. Must be self-contained.
4.  **Custodian (Foundation):** The `VolumeHandler`. Must never depend on the Manager or Trainer.

---

## 3. Rationale
By enforcing a "One-Way Street" for dependencies, we ensure that the system remains modular and that changes to low-level components do not cause cascading failures in the orchestration layer.
