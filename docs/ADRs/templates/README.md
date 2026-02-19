
# ADR README and Governance Map

This repository uses Architectural Decision Records (ADRs) to govern
structural, semantic, and operational behavior.

ADRs are divided into two categories:

1. **Constitutional ADRs (001–009)**  
   Foundational architectural rules that apply across the system.

2. **Project-Specific ADRs (010+)**  
   Domain, implementation, or feature-level decisions.

---

## Constitutional ADRs

These ADRs define system philosophy and governance:

- **ADR-001** — Ontology of the Repository  
  Defines what concepts exist.

- **ADR-002** — Topology and Dependency Rules  
  Defines structural dependency direction.

- **ADR-003** — Authority of Declarations Over Inference  
  Defines where semantic authority lives.

- **ADR-004** — Rules for Evaluation and Stability (Deferred)

- **ADR-005** — Testing as Mandatory Critical Infrastructure  
  Defines red / beige / green test doctrine.

- **ADR-006** — Intent Contracts for Non-Trivial Classes  
  Requires declared class-level purpose.

- **ADR-007** — Silicon-Based Agents as Untrusted Contributors  
  Governs automated modification.

- **ADR-008** — Observability and Explicit Failure  
  Defines fail-loud + log requirements.

- **ADR-009** — Boundary Contracts and Configuration Validation  
  Defines explicit interface contracts and configuration validation.

These ADRs form the architectural constitution of the repository.

---

## Project-Specific ADRs

ADRs numbered 010 and above define:

- Domain-specific evaluation strategy
- Implementation details
- Infrastructure decisions
- Feature-level trade-offs

These must comply with the constitutional ADRs above.

---

## Governance Structure (Conceptual Map)

- **Ontology (001)** defines what exists.
- **Topology (002)** defines structural direction.
- **Authority (003)** defines who owns meaning.
- **Boundary Contracts (009)** define interaction rules.
- **Observability (008)** enforces failure semantics.
- **Testing (005)** verifies system integrity.
- **Intent Contracts (006)** bind class-level behavior.
- **Automation Governance (007)** constrains silicon-based agents.

Together, these define the invariant layer of the system.
