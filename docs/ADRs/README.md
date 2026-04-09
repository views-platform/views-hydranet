# Architectural Decision Records (ADRs)

This directory contains the Architectural Decision Records (ADRs) for the HydraNet project. Records are organized by their lifecycle status to ensure the codebase remains governed and traceable.

## Directory Structure

- **[`active/`](./active/)**: The current "Law of the Land." These ADRs are accepted, implemented, and rigorously verified. They define the structural invariants of the system.
- **[`proposed/`](./proposed/)**: Emerging specifications and strategies currently under review or awaiting implementation.
- **[`archive/`](./archive/)**: Legacy records, superseded specifications, and rejected proposals preserved for historical context and rationale.
- **[`templates/`](./templates/)**: Constitutional ADR templates from base_docs used as foundations for new repositories.

---

## [Active (The Law)](./active/)

### Constitutional ADRs (000-009)

Governance foundation adopted from base_docs. These define HOW the project governs itself.

- **000**: Standard for ADRs
- **001**: Ontology and Topology of the HydraNet Repository
- **002**: Topology and Dependency Rules
- **003**: Philosophy of Engineering and Semantic Authority
- **004**: Rules for Evaluation and Stability
- **005**: Testing as Mandatory Critical Infrastructure (Red/Beige/Green)
- **006**: Intent Contracts for Non-Trivial Classes (CICs)
- **007**: Protocol for Silicon-Based Contributors
- **008**: Standardized Error Propagation (Double-Lock)
- **009**: Boundary Contracts and Configuration Validation

### Project-Specific ADRs (010+)

Architectural decisions specific to the HydraNet system.

- **010**: Spatiotemporal Dual-Representation
- **011**: Curriculum and Training Topology
- **012**: Volume Ledger and Topology
- **016**: Orchestration Lifecycle Management
- **019**: Feature Scaler Specification
- **020**: Multi-Task Output Topology
- **021**: Volume Dimension Reduction
- **022**: Generative Tool Safety
- **025**: Spatiotemporal Invariance and Scale
- **026**: Model Artifact Fetcher Specification
- **027**: Autoregressive Inference Strategy
- **028**: Numerical Stability Guards
- **030**: Dynamic Slicing Handshake
- **031**: Virtual Target Augmentation
- **032**: Authoritative Output Schema (The Pure State)
- **034**: Prediction Diagnostic Summary
- **035**: Training Health Audit
- **036**: Structured Dependency Scaffolding
- **037**: Geometric Health Visualization
- **038**: Unified Inference Pipeline (Forecast-is-Backtest)
- **039**: Symmetry Engine Specification
- **040**: Pure State Adapter
- **043**: Spatiotemporal Reconstruction Bridges
- **044**: Component Handshake and Wiring
- **045**: Visual Diagnostics Directory Structure
- **046**: Symmetric Feature Lifecycle
- **047**: Pandas-Free Prediction Output
- **048**: Technical Risk Register

## [Proposed (Emerging)](./proposed/)

Specifications currently under review or awaiting implementation:

- **029**: Geographic Anchors

## [Archive (Historical)](./archive/)

Legacy iterations and superseded designs:

- **001-006**: Early evaluation and manager specifications (superseded by constitutional ADRs)
- **007**: Original Volume Ledger (superseded by 012)
- **009**: Original VolumeSampler (superseded by 011)
- **011-014**: Curriculum mechanics iterations (superseded by 011 unified)
- **016**: Original Orchestration (superseded by 044)
- **017-018**: Inbound Handshake and Data Sniffer iterations
- **023**: Polars Bridge (preserved research)
- **024**: Legacy Backtest Orchestrator (superseded by 038)
- **033**: Naming Invariants (superseded by 032)
- **041**: Redundant Targets (superseded by unified configuration)

---

## Usage & Quality

Every new ADR must follow the [ADR Template](./templates/adr_template.md) and satisfy the quality criteria defined in **[ADR 000](./active/000_standard_for_adrs.md)**.
