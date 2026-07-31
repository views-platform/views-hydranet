# ADR-062: Channel-Role Accessors + VolumeHandler Decomposition

**Status:** Proposed
**Date:** 2026-06-13
**Deciders:** Simon (chair), Claude (pair)
**Extends:** ADR-060 (Static & Architectural Input Channels — implements its ontology as a contract)
**Continues:** D-01 (VolumeHandler partial split — PredictionFrameAssembler extraction)
**Depends on:** ADR-003 (Zero Magic), ADR-008 (Fail Loud), ADR-046 (Symmetric Feature Lifecycle)
**Addresses:** C-156 (root), C-157, C-158, C-159, C-36, C-37, C-75
**Related:** ADR-061 (coordinate channels — the first static instance that exposed the gap)

---

## 1. Context

ADR-060 defined the **ontology** — a channel is Dynamic (predicted) or Static (input-only), and the
config/engine/CICs "must treat each kind correctly." The #108 implementation gave that ontology **no
first-class home**: it represented "static" by *appending the static names to `feature_cols`*
(`volume_handler.py:159`, `kept_feature_cols = features + static_channels`) and selecting model
channels by membership (`to_pytorch`, `:277-278`).

**The defect (C-156).** `feature_cols` now carries **two roles at once** — "channels fed to the model"
*and* "channels the model predicts/trains on." For the bounded baseline those sets were identical, so
the overload was invisible. The static-channel seam broke the identity but only *some* consumers were
taught the difference. The census (Phase 1.1, `tests/test_channel_role_census.py`) pins the fallout
**empirically** (running code, not static analysis):

| Consumer | Reads | Verdict |
|---|---|---|
| `curriculum.py:45` | `list(handler.feature_cols)` as training subjects | **C-158** silent mis-train — rotates statics in as prediction targets |
| `training_engine.py:435` (Stage-5 biopsy) | `idx.feat` only, into a static-widened model | **C-157** crash |
| `train_model.py:75-85` (arch sidecar) | `arch_keys` omit `static_channels` | **C-159** crash on reload (rebuilds wrong width) |
| `visual_diagnostics.py` | `feature_cols` for the plot list | benign — plots statics as signal |
| `to_pytorch:278` | statics included in model input | safe / by-design |

The root is one missing abstraction, surfacing once per consumer — and once more for **every future
input-only channel** (the covariate roadmap). Patching each consumer (the abandoned `2d2bde9`) treats
symptoms; this ADR fixes the root. Separately, the register flags the Custodian itself: **C-36** (a
451-edge "god node" bridging 16+ communities; "any signature change ripples") and **C-37** (no
abstract interface; a *deliberately accepted* trade-off, now reconsidered because the seam means the
interface is no longer "rarely changing").

---

## 2. Decision

### 2.1 Channel roles become first-class on the Custodian (single source of truth)

`VolumeHandler` exposes **explicit role accessors**, each derived from one authoritative classification
(the metadata ledger + the ADR-060 ontology), never re-derived by consumers:

- **`model_input_cols`** — dynamic features ⧺ static channels, **in feed order** (exactly what reaches
  the model; what `to_pytorch` selects).
- **`target_cols`** — `regression_targets` + `classification_targets` (what the head predicts / loss
  trains on).
- **`static_cols`** — the input-only static channels (ADR-060 Static class).

`feature_cols` is **retained but demoted to a derived alias** of `model_input_cols` (back-compat), and
new code reads the explicit roles. Every consumer is rewired to ask for the role it means:

| Consumer | Was | Becomes |
|---|---|---|
| curriculum subjects | `feature_cols` | **`target_cols`** (statics can never be a subject) → resolves C-158 |
| training biopsy input | `idx.feat` | **`model_input_cols`** (dynamic ⧺ static) → resolves C-157 |
| arch sidecar | hand-listed `arch_keys` | **model-declared constructor params** incl. `static_channels` → resolves C-159 (see 2.3) |
| `to_pytorch` | `feature_cols` membership | `model_input_cols` (semantics unchanged, name honest) |

The classification obeys ADR-060 I1–I6 unchanged; this ADR only gives them a typed home so a consumer
**cannot** read the wrong set. **I5 (off-path bit-identity) is the standing guard**: with
`static_cols == []`, `model_input_cols == target_cols == feature_cols`, so behavior is byte-identical
to the proven baseline (verified by the end-to-end parity gate, not unit tests alone).

### 2.2 VolumeHandler decomposition (C-36 / C-37) — principled, not split-for-its-own-sake

Continuing D-01, separate the Custodian's responsibilities into cohesive units around a thin
role-bearing core. **Honor Ousterhout: the core volume ops (transpose/flip/slice/extrapolate/collapse +
the ledger) are a genuinely deep module — they stay together.** We extract only what has *distinct*
cohesion:

- **`ChannelLedger` (role core):** the channel_map + role classification + the accessors in 2.1. The
  single place that knows "what kind is each channel." Small, pure, heavily tested.
- **Ingestion (`from_df`)** delegates derivation + static-fill but stays a `VolumeHandler` factory.
- **`DerivationEngine`** (extracted): the `_execute_derivations` logic — **also resolves C-75** (the
  DataFetcher↔VolumeHandler duplication, guarded by `test_derivation_parity.py`).
- **`IVolumeHandler` Protocol** (resolves the C-37 residual): a typed interface for the core ops, so
  consumers depend on the contract, not the concrete class.
- PredictionFrame assembly stays in `PredictionFrameAssembler` (D-01, done).

The *exact* module boundaries and the **staging order** are specified in the build plan (1.5); this ADR
commits to the direction and the role contract. The decomposition is gated by the characterization net
(Phase 3) + the before/after **parity run** (Phases 2/5) — the only trustworthy check on a 451-edge node.

### 2.3 The sidecar derives from the model, not a hand-list (C-159)

The artifact sidecar's persisted keys are **derived from the model's actual constructor signature**
(so writer and reader cannot drift), and the save path **self-validates** by reloading the just-written
artifact before declaring success. `static_channels` is therefore persisted by construction.

---

## 3. Must-preserve invariants (the refactor regresses none)

- **ADR-060 I1–I6**, especially **I5** (off-path bit-identity) — enforced by the parity gate.
- **C-11** — consumers use properties, never `_metadata.*`.
- **C-13 / C-14** — derived-handler methods return *new* instances (immutability).
- **C-39** — no `views_pipeline_core` / framework imports in the Custodian.
- **SpatialConvention propagation** through all 8 creation sites (the 40-test flip/convention suite
  stays green).
- **C-98 / C-105** — the `input_channels` and `features ⊆ targets` laws (the role accessors should
  *strengthen*, not weaken, these).

---

## 4. Consequences

**Positive:** C-156 dissolved at the root; C-157/158/159 fixed *by construction* (the census xfails
flip to XPASS with no per-consumer patch beyond reading the right role); the covariate roadmap
(population, land mask) inherits a correct seam; C-36/C-37/C-75 reduced. The Custodian's contract
becomes legible.

**Cost / risk:** touches a 451-edge node — broad blast radius (C-36). Mitigated by: (a) the
characterization net pinning current behavior first, (b) byte-identical-when-off discipline at every
step, (c) the before/after end-to-end **parity gate** (bit-identical no-coords runs). Reversing C-37's
prior "accepted trade-off" is deliberate.

**Out of scope:** the **Measured** static sub-class (data-sourced covariates — population/terrain).
ADR-060 §2.2 owes that a separate handshake ADR; this ADR keeps coordinates (Architectural/Derived)
correct and does not preclude covariates, but does not build them.

---

## 5. Alternatives considered

- **Per-consumer patches (rejected):** the `2d2bde9` approach — three spot-fixes. Treats symptoms;
  re-breaks on the next input-only channel. Explicitly abandoned.
- **Minimal accessors, no decomposition (rejected by chair):** add roles but leave the god-class.
  Viable and lower-risk, but leaves C-36/C-37 standing; chair chose to take on the split now.
- **Revert the seam entirely (rejected):** discards correct dormant capability (#108/#109) the
  covariate roadmap needs; the floor already exists without it.

---

## 6. Test / verification plan

- **Census (done):** `tests/test_channel_role_census.py` — C-157/158/159 as `xfail(strict)`; flip to
  XPASS when 2.1 lands (no other consumer change).
- **Characterization net (Phase 3):** extend the census + the existing seam I1–I6, hard-gate, and
  40-test flip/convention suites to the rewired consumers.
- **Parity gate (Phases 2/5):** a no-coords run on current code == a no-coords run on refactored code,
  **bit-for-bit** (locked seeds) — the proof the decomposition preserved baseline behavior.
- **Coord smoke (Phase 7):** 1-lesson train **and** eval-reload clean end-to-end with coords on.
