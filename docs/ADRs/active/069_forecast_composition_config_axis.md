# ADR-069: `forecast_composition` — a first-class config axis for how gate + body compose into the emitted forecast

**Status:** Active
**Date:** 2026-07-24 (accepted 2026-07-24)
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers
**Epic:** #183 · **Reconciles:** ADR-066 (withdrawn) · **Builds on:** ADR-067 (family subsystem), ADR-068 (arm names)

## Summary (read this first — self-contained)

A HydraNet forecast for a map cell is built from a **body** (how many deaths if any) and — for a
non-self-zeroed body (one that does not produce its own zeros) — a **gate** (the chance any violence happens). *How* those two combine into the one
number the model emits is the **composition**. Today composition is **not a model setting**: the model
emits the body and the gate as two separate arrays, and they are combined only *after the fact* in the
scoring script. That has two costs: (1) the scoring composition we used never matched the locked
definition of `gated_NB` (it never actually gated the body), so our arm comparison was on an object the
model cannot produce; and (2) the autoregressive rollout (T>0) has no composed forecast to feed back.

This ADR makes composition a **first-class config axis**, `forecast_composition`, with three values —
`self_zeroed`, `soft_gate`, `threshold_gate` (+ a threshold `gate_threshold` τ) — applied **inside the
model at emit time**, to **both** the sample cube and the point / autoregressive-feedback (AR) path. This
turns ZINB, gated_NB and th_gated_NB into real, honestly-composed model outputs, and is the infrastructure
the **T>0 bloom fix** (repairing the 36-month autoregressive rollout that runs away — C-113) needs.

## 1. Context

- **Composition is score-time only.** For a family run the model emits the body samples
  (`posterior_magnitudes_zstack`) and the gate `P(y>0)` (`posterior_probabilities_zstack`) as **separate**
  arrays (`views_hydranet/utils/hydranet_inference.py:658-696`); the point emit `_emit_magnitude` returns
  **ungated μ** (`:227-235`). Nothing composes them.
- **The re-score never matched the locked definitions.** The frozen lodestar ruler computes count-CRPS on
  the raw body samples and feeds the gate only into AP/Brier (`lodestar_score.py:114-118`). The glossary
  (`reports/GLOSSARY.md` §2c) defines `gated_NB` as **per-draw `Bernoulli(gate) × body`** — which was
  never implemented. So the banked "gated_NB" is the *ungated* body, and the th_gated_NB win rests on a
  composition the model cannot emit.
- **ADR-066 was withdrawn** (its `body_family × zero_handling` idea folded into ADR-067). This ADR revives
  the missing half — the composition axis — in ADR-067's structure, using ADR-068's arm names.
- **The bloom (T>0)** needs the model to feed back the *composed* forecast (`plan_bloom_fix_sparse_feedback.md`).

## 2. Decision

Add an orthogonal config axis to `HydraNetConfig`:

**`forecast_composition`** — how the body and gate combine into the emitted forecast:

| value | meaning | arm (with family) |
|---|---|---|
| `self_zeroed` | passthrough — the body carries its own zeros; no external gate | `zinb` → **ZINB** |
| `soft_gate` | per-draw `Bernoulli(gate) × body` (sample cube); `gate × E[Y]` (point) | `nb` → **gated_NB** |
| `threshold_gate` | full body where `gate ≥ τ`, else 0 — whole cell (both paths) | `nb` → **th_gated_NB** |

**`gate_threshold`** (τ) — a float in `(0, 1)`, consumed **only** by `threshold_gate`. Fixed a-priori
(never fit on scored months — Goodhart). Boundary is inclusive: `gate ≥ τ` is kept.

**Applied at emit time, in one authority, to BOTH outputs:**
1. the **D×K sample cube** (`to_cube_samples` / a new composer), using the gate the model already computes;
2. the **point emit / AR feedback** (`_emit_magnitude`).

**Determinism:** `soft_gate`'s per-draw Bernoulli uses the existing seeded `torch.Generator`
(`hydranet_inference.py:664`, the S2 #121 determinism gate).

**Validators (fail-loud; authority = the family's `self_zeroed` attribute).** Config validation must
stay torch-free (CRP), and the `self_zeroed` attribute lives on the torch-importing family class, so
config reads a **torch-free mirror** in the registry (`self_zeroed_family_names()`); a parity test
(`test_registry`) asserts the mirror equals `{n for n in family_names() if get_family(n).self_zeroed}`,
so the attribute stays the ground truth and the two cannot drift. The rules:
- A **self-zeroed** family (`self_zeroed=True`, i.e. `zinb`) with `soft_gate`/`threshold_gate` → **RAISE**
  (double-counts the zeros: π + an external gate do the same job).
- A **non-self-zeroed** family (`nb`) with `self_zeroed` → **RAISE** (nb has no self-zeroing; per the
  glossary §2 matrix, NB has no self-zeroed standalone — it *must* be gated).
- `threshold_gate` without `gate_threshold ∈ (0,1)` → **RAISE**; `gate_threshold` set with a non-threshold
  composition → **RAISE**.
- Composition only constrains **family** runs (`output_distribution ∈ family_names()`); for legacy heads
  the field is inert (the composer runs only on the family path).

**Default:** `forecast_composition = "self_zeroed"` — inert for legacy heads, correct for `zinb`, and
**forces every `nb` config to declare** its composition explicitly (the point of the axis).

**Retire `emit_family_core`** — the inference flag for the KILLED gated_ZINBcore arm (dead code).

## 3. Rationale & integrity impact

- **Honesty.** The three arms become objects the model actually emits, produced the same way — the
  precondition for a trustworthy comparison and any ensemble.
- **`gated_NB`'s numbers WILL move** (expected, not a regression): per-draw Bernoulli zeros some draws on
  low-gate cells, lowering crps-none vs the ungated re-score. This is the correction, surfaced and logged.
- **Single authority** for "is this self-zeroed?" is the `DistributionFamily.self_zeroed` attribute
  (already set for ZINB) — no hardcoded name lists (mirrors the ADR-065/067 validator discipline).
- **Byte-identity preserved** for legacy heads and the `self_zeroed` (zinb) path (passthrough ⇒ unchanged);
  proven by the S2 characterization goldens.
- **Determinism** unchanged (seeded generator).

## 4. Consequences

### ✅ Positive
- ZINB · gated_NB · th_gated_NB are real, comparable model outputs → an honest three-arm base + ensemble.
- Emit-time composition is exactly what the T>0 bloom fix (sparse feedback) needs.
- Retires the dead `emit_family_core` flag; reconciles ADR-066's withdrawn design.

### ⚠️ Negative
- **Breaking for `nb` configs:** they must now declare `forecast_composition` (previously implicit at score
  time). Intentional — the implicitness was the bug.
- The banked score-time gated_NB numbers are superseded; S8 re-scores and logs the movement.
- One more config axis to reason about (mitigated by fail-loud validators + the glossary).

## 5. Validation
- TDD red-first per story: config validators (S3), composition correctness on both paths (S4), contract +
  byte-identity parity (S6), full suite + per-composition smoke (S7), real T=0 eval-only re-score (S8).
- Determinism gate (seeded generator) and the ruler `--selftest` remain green.

## 6. Implementation notes
- **Composer** — a single authority (`views_hydranet/distributions/composition.py` or a family-agnostic
  helper) taking (body samples/mean, gate, composition, τ) → composed count sample cube / composed log1p point.
- **Wiring sites:** `to_cube_samples` call at `hydranet_inference.py:683` (pass the gate `prob_zstack`,
  `:692`) and `_emit_magnitude` (`:214-258`).
- **Config:** `HydraNetConfig` fields + validators in `views_hydranet/utils/config_initializer.py`
  (mirror `validate_output_distribution`, the family/legacy validators `:700-770`).
- **CIC:** new `docs/CICs/ForecastComposer.md` (or extend `DistributionFamily`/`InferenceOrchestrator`) — S6.
- Scope: T=0 calibration + eval-only (no retrain); M3 validation, new distributions and the bloom are OUT.

## Glossary
Uses the locked arm names from `reports/GLOSSARY.md` §2c (ADR-068): **ZINB** (self-zeroed), **gated_NB**
(soft), **th_gated_NB** (hard threshold). `forecast_composition` is the config realisation of the
"occurrence rule" knob defined there (self / soft gate / threshold gate).
