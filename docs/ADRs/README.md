# Architectural Decision Records (ADRs)

This directory contains the Architectural Decision Records (ADRs) for the HydraNet project. Records are organized by their lifecycle status to ensure the codebase remains governed and traceable.

## 📁 Directory Structure

- **[`active/`](./active/)**: The current "Law of the Land." These ADRs are accepted, implemented, and rigorously verified. They define the structural invariants of the system.
- **[`proposed/`](./proposed/)**: Emerging specifications and strategies currently under review or awaiting implementation.
- **[`archive/`](./archive/)**: Legacy records, superseded specifications, and rejected proposals preserved for historical context and rationale.
- **[`templates/`](./templates/)**: Generic ADR templates used as foundations for new repositories.

---

## 🟢 [Active (The Law)](./active/)
These documents define the current structure of the HydraNet pipeline:
- **000**: Standard for ADR Quality.
- **005**: Testing as Mandatory Critical Infrastructure (Red/Beige/Green).
- **007**: The Volume Ledger & Topology.
- **008**: Operational Configuration Specification (Unified).
- **015**: Philosophy of Engineering (Boring Architecture).
- **016**: Orchestration Lifecycle Management.
- **019**: The Normalizer (`FeatureScaler`).
- **020**: Multi-Task Output Topology.
- **021**: Volume Dimension Reduction.
- **032**: Authoritative Output Schema (The Pure State).
- **038**: Unified Inference Pipeline (Forecast-is-Backtest).
- **042**: Standardized Error Propagation (Double-Lock).
- **043**: Spatiotemporal Reconstruction Bridges.
- **044**: Component Handshake and Wiring.
- **045**: Ontology of the HydraNet Repository (Roles).
- **046**: Protocol for Silicon-Based Contributors.

## 🟡 [Proposed (Emerging)](./proposed/)
Specifications currently in the design or implementation phase:
- **010**: Spatiotemporal Dual-Representation Reasoning.
- **011**: Curriculum Learning and Training Topology (Unified).
- **017**: The Inbound Handshake (Ingestion and Validation).
- **029**: Geographic Anchors.

## 🔴 [Archive (Historical)](./archive/)
Legacy iterations and superseded designs:
- **001-006**: Early evaluation and manager specifications.
- **009**: Original VolumeSampler (superseded by 013).
- **012-014**: Curriculum mechanics (superseded by 011).
- **018**: Data Sniffer (superseded by 017).
- **023**: Polars Bridge (Preserved research).
- **024**: Legacy Backtest Orchestrator (superseded by 038).
- **033**: Naming Invariants (superseded by 032).
- **041**: Redundant Targets (superseded by 008).

---

## Usage & Quality
Every new ADR must follow the [ADR Template](./adr_template.md) and satisfy the quality criteria defined in **[ADR 000](./active/000_standard_for_adrs.md)**.
