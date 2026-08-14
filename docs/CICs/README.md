# Class Intent Contracts README

This directory contains **Intent Contracts** as defined in ADR-006.

An Intent Contract is a human-readable, unambiguous declaration of:

- what a non-trivial class is meant to do,
- what it must never do,
- its invariants,
- and its failure semantics.

Intent Contracts are architectural artifacts.
They are not implementation documentation.

---

## When Is an Intent Contract Required?

An Intent Contract is mandatory for:

- Core domain classes
- Architectural boundary classes
- Orchestration components
- State-owning components
- Classes that enforce invariants
- Classes that modify semantics or transformation

Trivial value objects and pure utility functions do not require one.

---

## Structure of an Intent Contract

Each contract must define:

1. Purpose
2. Responsibility Boundary
3. Invariants
4. Explicit Non-Responsibilities
5. Failure Semantics
6. Observable Effects (if applicable)

Contracts must be clear enough that:

- Tests (ADR-005) can be derived from them.
- Architectural violations can be detected.
- Silicon-based agents cannot reinterpret intent (ADR-007).

---

## Active Contracts

### Model & Training Core
- `HydraBNUNet06LSTM4.md` — the recurrent BN-U-Net + Quad-ConvLSTM model (per-timestep `forward`).
- `LockedDropout.md` — consistent-mask (variational) dropout for the AR rollout (ADR-057).
- `TrainingEngine.md` — the recurrent training loop, scheduled-sampling feedback, BN recalibration.
- `CurriculumLearner.md`
- `ScheduledSamplingMixer.md`
- `BodySupervisionResolver.md`
- `TrainingForensics.md`

### Distribution Families & Output Heads (ADR-067)
- `DistributionFamily.md` — the per-cell output-distribution ABC.
- `DistributionRegistry.md` — the `name → family` registry + self-zeroed mirror.
- `NBCore.md` — the shared Negative-Binomial count-math authority.
- `ForecastComposer.md` — gate × body composition (ADR-069).
- `PosteriorCubeSampler.md` — the D×K posterior-cube sampler `to_cube_samples` (ADR-067/070).

### The Spine (Orchestration & Data)
- `HydranetManager.md`
- `VolumeHandler.md`
- `VolumeSampler.md`
- `DataSniffer.md`
- `DataFetcher.md`
- `FeatureScaler.md`

### Inference & Prediction Output
- `InferenceOrchestrator.md`
- `PredictionFrameAssembler.md`
- `ModelArtifactFetcher.md`

### Config, Guards & Diagnostics
- `HydraNetConfig.md`
- `ConfigInitializer.md`
- `IntegrityGuardian.md`
- `VisualDiagnostics.md`

---

## Governance Relationship

Intent Contracts are governed by:

- ADR-006 (Intent Contracts for Non-Trivial Classes)
- ADR-003 (Philosophy of Engineering and Semantic Authority)
- ADR-005 (Testing as Mandatory Critical Infrastructure)

If a class changes meaning, its Intent Contract must be updated.
