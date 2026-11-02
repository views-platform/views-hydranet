# ADR-075: A run can be asked to emit the body, which is otherwise unrecoverable

**Status:** Proposed
**Date:** 2026-11-02
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Requires:** ADR-074 (an emitted artifact records what it is)
**Related:** ADR-069 (composition as a config axis), ADR-067 (distribution family subsystem), ADR-073 (which family is emitted)

## Summary (read this first — self-contained)

The model produces forecasts. A forecast is built from two parts: a **gate**, the chance that any
violence occurs in a cell, and a **body**, how many deaths if it does. How they combine is the
**composition**, and the combined number is what the model emits.

The **gate** is already emitted. It is written out as its own array, it is a declared target, and
it is scored against ground truth, because whether violence occurred is something the world records.

The **body is not**, and it is the one part of a forecast with all three of these properties:

- **it cannot be observed** — nothing in the world records "how many deaths, given that there were
  any", so no ground truth exists for it;
- **it therefore cannot be scored**, which is why it can never be a pipeline target;
- **it cannot be recovered** from the forecast, because the composition that produced the forecast
  discards it wherever the gate is low.

For audit, evaluation, experiments and continuous improvement we need the body anyway: questions
about whether the model is losing cells or losing magnitude are questions about one part while the
other is held still, and a combined forecast conflates them.

This ADR lets a run be **asked** to emit the body, from the same inference pass and the same draws
as the forecast. It is a diagnostic output, not a forecast, and never a target — written under a
name that says so, `diagnostic_body_<target>`, by a path that cannot turn it into one. It changes
no default.

---

## 1. Context

**The body is discarded before anything is written.** Composition runs inside the inference pass,
and the composed array is what reaches disk. Under the thresholding composition in ordinary use the
body is set to zero wherever the gate falls below the threshold, for every draw. Above the threshold
it survives untouched.

So the body can be reconstructed exactly above the threshold and not at all below it. On a global
grid at the threshold in use, the cells above it are **well under one percent** of the grid.

**Diagnostics need the parts, not the whole.** Whether the gate is calibrated, whether the body
carries the spatial structure it should, whether a composition choice helps or harms — each is a
question about one part. Every combined readout answers them with both mixed together.

**The question cannot be answered by running twice.** Two runs at the same seed draw identically,
so a second pass under a different composition does not give a second sample. It gives a run whose
*rollout* diverges from the first, because composition also determines the field fed back into the
next step — so beyond the first horizon the two runs are different models, not two views of one.

**What exists today is a mean, reachable only by hand.** A diagnostic dump of the body's posterior
mean exists, but it is switched on by setting an attribute on the inference orchestrator from a
driver script. The ordinary entry point never sets it, so a run launched normally — including every
run on rented hardware — produces nothing. A mean also cannot support any distributional readout.

---

## 2. Decision

**Statement.** *We will let a configuration ask a run to emit the body alongside the forecast, from
the same pass and the same draws.*

### 2.1 The field

A configuration field declares which quantities a run writes. It is a **set**, so asking for the
forecast and the body yields both. Its default is the set emitted today, which already includes the
gate.

### 2.2 Emission is a separate field from composition

The field that selects emission is not the field that selects composition, and it does not reach
the autoregressive feedback path. Output beyond the first horizon is identical whether it is set or
not.

### 2.3 The body is a diagnostic, never a target

The body is emitted outside the pipeline's target system: not declared as a regression or
classification target, and not entering the target set handed to evaluation.

The gate's position is different and unchanged — it is a declared classification target, correctly,
because ground truth for it exists.

### 2.4 What it is called, and the two things the name has to survive

An emitted diagnostic is named **`diagnostic_<quantity>_<target>`** — so the body for a given
target is `diagnostic_body_<target>`. Targets are the model's existing set, and that set may grow.

Two constraints decide this, and both are correctness rather than taste.

**The name must not be caught by a selection meant for something else.** Downstream tooling selects
artifacts by matching a path component against the forecast and gate prefixes. A name beginning
with either — including a compound such as the forecast prefix followed by `body` — is **silently
swept into the forecast archive**. Measured: that compound matches; `diagnostic_body_…` does not.

**It must not be written by the target assembler.** That assembler gives every target a
prediction-prefixed column, and a serving layer downstream adopts prediction-prefixed columns **by
prefix, with no allow-list** — so a diagnostic that passed through it would become servable as a
forecast. A diagnostic therefore has its **own writer**, and never acquires a prediction column.

⇒ Two independent protections, which is the right number for a quantity that would be mistaken for
a forecast if it leaked: a name no selection catches, and a write path that cannot mint a
prediction column.

**The namespace is open.** `diagnostic_<quantity>_<target>` admits other quantities should they be
wanted — a bulk or a tail of a distribution, for instance — without a further decision about
naming. Adding one is still a decision about whether to emit it; this ADR only fixes the shape of
the name.

⚠️ One possibility the namespace permits and this ADR does **not** propose: the gate is currently a
declared target with its own ground truth, and correctly so. If it ever stopped being scored as a
target it could move into this namespace instead. That is a long way off, is not planned, and would
need its own decision.

### 2.5 Scope

**In scope:** the field, its validation, and the model honouring it.

**Out of scope:** which horizons a run emits (ADR-078); which *family* a run emits from
(ADR-073); what is transported off rented hardware, which the models repository owns; which models
set the field; and composition semantics, which ADR-069 governs unchanged.

---

## 3. Rationale & Integrity Impact

| alternative | why it loses |
|---|---|
| **select emission with the composition field** | that field is also read by the autoregressive feedback, so using it to choose what is written would change what the rollout feeds back, and so every horizon beyond the first. Shapes and row counts would stay correct and nothing would report the difference. The only alternative here that can corrupt a result rather than fail |
| **set an attribute on the orchestrator, as the existing diagnostics do** | unreachable from a configuration. The ordinary entry point sets no such attribute, so a normally launched run writes nothing — the situation today |
| **declare the body a pipeline target** | evaluation slices ground truth by target name and no observable corresponds to the body. Guards in the pipeline and the evaluator do refuse it — but the stronger reason is downstream: a serving layer adopts prediction columns **by prefix with no allow-list**, and its documented fail-loud check does not distinguish a body from a forecast. A body admitted to the target system would be served as a forecast, silently |
| **run inference twice under different compositions** | the draws are seeded and identical, so this yields no second sample — only a second rollout. It answers a different question |
| **reconstruct the body from the forecast and the gate** | exact above the composition's threshold, impossible below it, and below it is almost the whole grid |
| **build a general artifact-selection language here** | the models repository already has a declaration vocabulary for what is produced and shipped. A second would fork it, and one field settles this |

**Reproducibility.** A comparison between the body and the forecast is drawn from one pass over one
set of draws, so it carries no sampling difference at all.

**Fail-loud, as intended.** A requested quantity the configured family cannot produce should be
refused when the configuration is built, and a run whose additional output will not fit should be
refused before allocation rather than terminated during it. Neither refusal exists yet; §5 names
them as tests to write.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] The body becomes available at all, and available together with the forecast from one pass.
- [ ] Emission is declarable from an ordinary model configuration, so it works wherever a model
      runs, including rented hardware.
- [ ] Diagnostic questions about the body alone become answerable without retraining.
- [ ] The distinction between choosing what to emit and choosing how to compose becomes structural
      rather than a matter of care.

### ⚠️ Negative (Costs)

- [ ] **Memory and storage.** The body is another array of the forecast's shape, raising peak memory
      by a substantial fraction and adding a comparable volume per origin.
- [ ] Two quantities in the same units exist on disk, only one of which is a forecast. ADR-074 is
      what keeps them distinguishable, which is why it is a prerequisite.
- [ ] One more configuration field on a schema that is already large.
- [ ] A request that cannot be met fails when the configuration is built, rather than producing
      something adjacent to what was asked for.

---

## 5. Validation

**Invariants.**
- Output beyond the first horizon is bit-identical with the field set and unset.
- A configuration that does not set the field produces output byte-identical to a run before this
  change, at the same seed.

**Tests.**
- *Red:* a requested quantity the configured family cannot produce is refused at configuration time.
- *Red:* the memory preflight refuses an oversized request before allocating; assert the refusal,
  not the failure that follows it.
- *Green:* under the **thresholding** composition, the emitted body masked by the gate at that
  threshold reproduces the emitted forecast exactly above the threshold. This establishes that the
  emitted body is the one the forecast was built from, rather than an independent draw. It is
  specific to that composition: the sampling composition masks per draw, and the self-zeroing one
  makes the body the forecast, so each needs its own equality.
- *Green:* ADR-074's manifest records which quantities were emitted.

**Not yet enforced.** None of these tests exists, and neither does the emission path.

**What would reopen this.** A quantity that cannot be expressed as a member of a set — one needing
its own inference pass, or its own loss — would mean the field had become a dispatch mechanism
rather than a declaration.

---

## 6. Implementation Notes

- **Location.** The field and its validator belong with the configuration contract (ADR-009). The
  emission path honours it; the artifact's self-description is ADR-074's.
- **Ordering.** ADR-074 first. A second quantity in the forecast's own units, written beside it with
  nothing recording which is which, is worse than not emitting it.
- **Evidence.** `reports/2026-11-02_adr_claim_verification/00_CLAIMS.md` §2 holds every factual
  claim in this ADR with the source that settles it, including the measured above-threshold
  fraction, the seeded-draw finding, and the evidence that the gate is already emitted.

---

## References

**This repository:** ADR-009 (configuration validation) · ADR-067 (distribution family subsystem) ·
ADR-069 (composition as a config axis) · ADR-073 (which family is emitted) · ADR-074 (an emitted
artifact records what it is) · ADR-078 (horizon selection).

**Other repositories** — the contracts this ADR works within:

- [views-models ADR-019 — delivery declaration](https://github.com/views-platform/views-models/blob/main/docs/ADRs/019_delivery_declaration.md). The existing vocabulary for declaring what is produced and shipped, which §3 declines to fork.
- [views-models ADR-024 — pod runner contract](https://github.com/views-platform/views-models/blob/main/docs/ADRs/024_pod_runner_contract.md). What a run on rented hardware guarantees, and why a switch reachable only from a driver script is a dead end.
- [views-pipeline-core ADR-011 — separation of configs](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/011_seperation_of_configs.md). Where a model's settings live, and so where this field belongs.
