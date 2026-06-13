# ADR-060: Static & Architectural Input Channels

**Status:** Accepted
**Date:** 2026-06-13
**Deciders:** Simon (chair), Claude (pair)
**Extends:** ADR-046 (Symmetric Feature Lifecycle)
**Supersedes:** ADR-029 (Geographic Anchors — jointly with ADR-061)
**Depends on:** ADR-003 (Zero Magic), ADR-008 (Fail Loud)
**Related:** ADR-056 (off-path bit-identity precedent), ADR-061 (first instance — coordinate channels)

---

## 1. Context

**Problem.** ADR-046 formalized two feature lifecycles: **Transformations** (in-place, value-level,
*invertible*, key `transformations`) and **Derivations** (additive, a *new* identity derived from
*existing columns*, *not* invertible, key `derivations`). Both describe signals that live in the
"Pure State" DataFrame and are either prediction **targets** or the raw inputs behind them. The
autoregressive head consumes and predicts the *same* channels (the conflict histories), so input and
output have so far been one coupled set.

**Assumption no longer true.** That *every* input channel is also a prediction target, and that every
channel is born either as a raw feature or a derivation-of-features. The spatial-grounding work needs
channels that are **input-only**: injected into the model, never predicted, never inverted, never a
target — and sourced not from a data column but from the tensor's **own geometry** (coordinates), and
later from **external static rasters** (population, terrain, ocean mask).

**Urgency.** Coordinate channels (ADR-061) are the first such channel and static covariates the
second. With no contract, the only way to add a channel is the `features`/target list — which would
silently make coordinates a *prediction target* and route them into the prediction frame and the
inverse-transform. That is a category error. We pin the ontology **now**, before implementation, so it
informs the CICs and does not drift.

---

## 2. Decision

**Statement.** *We will extend the feature lifecycle with a **Static** channel class — input-only
channels that are injected into the model, never predicted, never inverted, and never targets — and
classify every channel along two axes so the config, the engine, and the CICs treat each kind
correctly.*

### 2.1 The two axes

**Axis A — Dynamic (endogenous) vs Static (exogenous):**

| | Dynamic (endogenous) | Static (exogenous) |
|---|---|---|
| Predicted by the head? | **Yes** | **No** |
| Fed back in the rollout? | Yes (own output) | **Re-injected as true values every step** |
| In `regression_/classification_targets`? | Yes | **Never** |
| Counted in `output_channels`? | Yes | **Never** |
| Inverse-transformed / in the prediction frame? | If transformed, yes | **Never** |
| Example | `lr_sb_best`, `lr_ns_best`, `lr_os_best` | coordinates; (future) population |

**Axis B — within Static: Architectural/Derived vs Measured:**

| | Architectural / Derived | Measured / Exogenous-data |
|---|---|---|
| Source | the **tensor geometry** (grid index) | external **static rasters** |
| Availability | always; deterministic; no data dependency | fetched, grid-aligned, per-channel scaled |
| Example | coordinate channels (ADR-061) | population, terrain, ocean (future ADR) |

> **The land/water mask straddles Axis B.** A binary land mask is *derivable* (from `priogrid_gid > 0`,
> per ADR-029) yet encodes a covariate-like structural prior. It is therefore the **cheapest first
> covariate** in the escalation: derivable like a coordinate channel, but base-rate-bearing like a
> measured covariate. It is the natural rung between coordinates (ADR-061) and fetched rasters.

> **The *measured* sub-class owes a handshake (unspecified here).** Sourcing a measured channel (a
> fetched raster) additionally requires a grid-alignment **handshake** — raster→PRIO-grid registration,
> missingness handling, per-channel scaling — that this ADR does **not** specify; it is owed by the
> future covariate ADR. Derived/architectural channels (coordinates) have no such handshake (I4), which
> is why they go first.

This is the missing third lifecycle alongside ADR-046's `transformations` and `derivations`: a Static
channel is **additive** (like a derivation) and **never inverted** (like a derivation), but unlike a
derivation it is **not a DataFrame column, not derived from other columns, and never a target** — it
is injected at the model boundary.

### 2.2 Invariants (the CIC-citable contract)

- **I1 — Never a target.** A static channel never appears in `regression_targets` /
  `classification_targets`, and `output_channels` counts only dynamic channels. The config validator
  **MUST fail loud** if a static-channel name appears in a target list.
- **I2 — No inversion, not in the frame.** A static channel is never passed through `transformations`
  / inverse-transform and never appears in any `PredictionFrame`.
- **I3 — Static across the rollout.** In the autoregressive loop a static channel is re-injected with
  its **true** values at every step; it is never overwritten by model output.
- **I4 — Alignment by construction.** An Architectural/Derived channel is computed over the **full**
  tensor/grid and sliced with the **same window indices** as the dynamic channels. Global alignment is
  guaranteed by *derive-then-slice*, not by routing coordinate columns through the data path.
- **I5 — Off-path bit-identity.** With static channels disabled, the pipeline is **bit-identical** to
  the pre-ADR model (cf. ADR-056 invariant 5).
- **I6 — Augmentation sync.** Any spatial transform applied to dynamic channels (training-time
  flips/rotations, the North-Up orientation flip) **MUST** be applied identically to static channels, so
  absolute position stays consistent with the data. *(Inherited from ADR-029's "deterministic
  augmentation" requirement — a real correctness condition: coordinates/masks that don't flip with the
  features encode the wrong position.)*

### 2.3 Scope

- **In-scope:** the Static channel class, the two-axis ontology, invariants I1–I6, and the
  config-surface principle — *a static channel is declared in its own block, never in `features`/targets.*
- **Out-of-scope:** the specific channels (coordinates → ADR-061; static covariates → a future ADR);
  the concrete config field names and the in-model injection mechanism (delegated to the implementing
  ADR/CICs).

---

## 3. Rationale & Integrity Impact

- **Logic.** ADR-046 separated the *mathematics of scale* (transform) from the *ontology of identity*
  (derivation). It left a third axis unnamed: a channel's **role** (predicted-dynamic vs injected-static)
  and its **source** (data vs geometry). Without this axis the only door to a new channel is `features`,
  which couples it to the prediction targets — wrong for an input-only signal. *Correctness > convenience:*
  we refuse to smuggle a non-target through the target machinery.
- **Fortress State.** A static channel cannot be misconfigured into a target (I1, fail-loud); derived
  channels are aligned by construction (I4), eliminating a class of silent windowing bugs; off-path
  bit-identity (I5) keeps the existing baseline reproducible.
- **Fail-Loud.** I1 raises on static-channel-in-targets; any required parameter that is missing fails
  loud per ADR-008 (no silent/magic defaults — the ADR-046 rule carries over).

---

## 4. Consequences

### ✅ Positive (Benefits)
- A **reusable seam** — coordinates now, static covariates later — instead of a per-feature hack.
- The config stops "lying" about what is a target (the ADR-046 spirit, extended to inputs).
- The **dynamic/static** distinction becomes explicit in the rollout loop and the output path.
- CICs gain a precise, citable contract (I1–I6).

### ⚠️ Negative (Costs)
- A **third** feature category to teach the config schema and the engine.
- The network must accept **`input_channels > output_channels`** (dynamic + static in, dynamic out)
  — a real change to the model's I/O contract.

---

## 5. Validation

- **Invariants:** I1–I6 above.
- **Tests (ADR-005):**
  - *Red:* a config with a static channel in a target list → validator raises (I1); a static channel
    surfacing in any `PredictionFrame` → test fails (I2).
  - *Beige:* rollout test asserts a static channel is identical at every step and equals its source
    (I3); windowing test asserts the derived-channel slice uses the same window indices as the dynamic
    channels (I4); an augmentation test asserts a flipped/rotated window's static channels match the
    transformed grid position (I6).
  - *Green:* with static channels disabled, full-pipeline output is **byte-identical** to baseline (I5).
- **Failure Mode:** any static channel leaking into a target list, the prediction frame, or an
  inverse-transform; a static channel differing across rollout steps; or disabling static channels
  **not** being bit-identical ⇒ the seam is wrong and this ADR must be reconsidered.

---

## 6. Implementation Notes

- **Location:** the config schema (`HydraNetConfig`) + its validator; the network class I/O contract;
  `VolumeHandler` / `VolumeSampler` windowing; `FeatureScaler` (must ignore static channels). Enforced
  in code + config validation + CI tests.
- **CIC impact (updated by the implementing ADR, 061):** `HydraNetConfig`, `VolumeHandler` /
  `VolumeSampler`, `FeatureScaler`, the network-class CIC, and possibly a **new** CIC if the injection
  becomes its own class/mixin (cf. `ScheduledSamplingMixer`).
- **Function-level contracts → CICs.** This ADR fixes *architectural* invariants (I1–I6); per-method
  pre/post-conditions are delegated to the CICs (ADR-006), updated per the CIC-impact list above.
- **Status note:** **Accepted** as the contract (the decision to add a Static channel class is
  committed). I1–I6 are the **conformance tests** the implementation must pass — a static channel
  leaking into a target / frame / inversion, drifting across the rollout, or a non-bit-identical
  off-path, is a *contract violation*, not a metric question. (The metric question — does any specific
  static channel help — belongs to ADR-061.)
- **References:** ADR-046 (extended), ADR-003 / ADR-008 (Zero Magic / Fail Loud), ADR-056 (off-path
  bit-identity precedent), ADR-061 (first instance), and the coordinate-grounding dossier
  (`reports/2026-06-11_coordinate_grounding_dossier/`).
