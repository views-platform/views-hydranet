# Architectural Decision Records (ADRs)

## Overview

This folder houses the Architectural Decision Records (ADRs) of our project. ADRs are critical documents capturing essential architectural decisions, along with their contexts and consequences. These records aim to provide insights into the architectural direction, promote consistent decision-making, and enhance comprehension for both new and existing project contributors.

## Purpose

ADRs are instrumental in:
- Documenting the rationale behind specific architectural decisions.
- Outlining considered alternatives and assessed trade-offs.
- Ensuring transparency and maintaining architectural integrity as the project evolves.

## Usage Guidelines

### For New Developers
- **Orientation**: Begin by reviewing existing ADRs to grasp the foundational decisions shaping the project.
- **Contextual Learning**: Gain insights into the problems addressed and the solutions devised through each ADR.

### For Experienced Developers
- **Reference Tool**: Refer to ADRs to understand and remember the reasons behind past architectural choices, particularly useful when updating or expanding the system.
- **Documentation of Changes**: Contribute new ADRs when suggesting significant architectural changes or introducing new patterns.

## Status of ADRs

This repository is currently transitioning to a **"Boring Architecture"** standard governed by **ADR 000**. 

### 🟢 Active (The Law of the Land)
These documents are rigorously verified against the implementation and define the current structural invariants:
*   **000**: Standard for ADR Quality.
*   **007**: The Custodian (`VolumeHandler`).
*   **008**: Operational Configuration Specification.
*   **010**: Dual-Representation (Semantic vs. Hardware).
*   **011**: Curriculum Learning Strategy.
*   **012**: The Planner (`CurriculumLearner`).
*   **013**: The Pure Lens (`VolumeSampler`).
*   **014**: The Optimization Gate (Gradient Accumulation).

### 🔴 Legacy (Historical Reference)
These documents represent previous iterations and may contain magic numbers or deprecated logic. Use with caution:
*   **001-006**: Early evaluation and manager specifications.
*   **009**: Original VolumeSampler (superseded by 012 & 013).

## Structure of ADRs

Each ADR follows a structured markdown format:
- **Title**: Concise and indicative of the decision content.
- **Status**: Current state of the ADR (e.g., proposed, accepted, rejected, deprecated).
- **Context**: Description of the problem and the need for a decision.
- **Decision**: The specific resolution or change implemented.
- **Consequences**: Both positive and negative outcomes resulting from the decision.

### Naming Convention

ADRs are sequentially named with a numeric prefix (e.g., `001-initial-decision.md`) to maintain order and facilitate easy navigation through the records.

## Contributing to ADRs

To add a new ADR:
1. Fork the repository and create a feature branch for your ADR.
2. Draft your ADR in markdown using the standard template.
3. Open a pull request with a comprehensive description of the proposed record.
4. Participate in the review process to address any feedback and finalize the ADR.

## Feedback and Inquiries

For questions or comments regarding the ADRs or the documentation process, please initiate a discussion through a repository issue or bring it up in our regular team meetings.
