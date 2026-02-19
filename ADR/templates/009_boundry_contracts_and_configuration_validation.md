
# ADR-009: Boundary Contracts and Configuration Validation

**Status:** --template--  
**Date:** YYYY-MM-DD  
**Deciders:** <roles / team>  

---

## Context

Complex systems fail most often at boundaries:

- between modules,
- between configuration and runtime,
- between data producers and consumers,
- between planning and execution.

Ambiguous configuration, hidden defaults, and implicit contracts
introduce silent semantic drift and runtime fragility.

To preserve architectural integrity and fail-loud guarantees (ADR-003),
all external and internal boundaries must be explicit and validated.

---

## Decision

This repository adopts the following invariants:

> All architectural boundaries must declare explicit contracts.  
> All configuration must be validated at entry.  
> No semantic defaults may exist silently.

---

## 1. Boundary Contracts

Every boundary between components must define:

- Explicit input schema
- Explicit output schema
- Declared invariants
- Failure semantics

Boundaries include:

- Configuration → runtime
- Data ingestion → processing
- Planning → execution
- Internal modules → external interfaces

Implicit contracts are prohibited.

If a boundary assumption cannot be declared clearly,
the boundary is ill-defined and must be redesigned.

---

## 2. Configuration as First-Class Artifact

Configuration is not a convenience layer.
It is an architectural artifact.

Configuration must:

- Be explicit
- Be versionable
- Be externally inspectable
- Be validated before execution
- Not rely on hidden defaults

Changing configuration must not silently alter system meaning.

---

## 3. Validation at Entry (Handshake Principle)

All configuration and external inputs must be validated at the system boundary.

Validation must occur:

- Before state mutation
- Before execution begins
- Before orchestration proceeds

The system must fail early if:

- Required fields are missing
- Types are incorrect
- Redundant parameters disagree
- Declared invariants are violated

Borrowed or assumed state is prohibited.

---

## 4. Separation of Configuration Domains

Configuration domains must be separated conceptually.

Examples (illustrative, not prescriptive):

- Operational parameters (affect computation)
- Behavioral parameters (affect runtime behavior)
- Metadata or documentation parameters (informational only)

Cross-domain coupling must be explicit.

Configuration that affects behavior must not be disguised as documentation.

---

## 5. Redundancy and Consistency Checks

Where ambiguity risk is high, explicit redundancy is preferred.

Examples:

- Declaring both dimensionality and shape
- Declaring both type and interpretation
- Declaring both mode and permitted operations

Redundant declarations must be validated for consistency.

Silent derivation is discouraged where semantic meaning is involved.

---

## 6. Failure Semantics

Configuration validation failures must:

- Be logged (ADR-008)
- Be raised explicitly (ADR-008)
- Halt execution

Warnings are insufficient for structural configuration errors.

---

## Consequences

### Positive

- Eliminates hidden configuration drift
- Reduces boundary fragility
- Strengthens fail-loud guarantees
- Improves reproducibility and traceability

### Negative

- Requires explicit schemas
- Adds validation boilerplate
- Increases up-front configuration clarity requirements

These costs are accepted.

---

## Notes

This ADR does not prescribe:

- Specific file layouts
- Specific configuration libraries
- Specific schema frameworks

Operational configuration structures may vary by project,
provided they comply with the invariants defined here.
