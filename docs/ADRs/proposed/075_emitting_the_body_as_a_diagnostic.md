# ADR-075: Which quantities a run emits is a config axis, distinct from composition

**Status:** Proposed
**Date:** 2026-11-01
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Sibling of:** ADR-069 (composition as a config axis) · **Requires:** ADR-074 (emission provenance)
**Independent of:** ADR-078 (origin and horizon selection — a separate axis, deliberately not bundled here)
**Builds on:** ADR-067 (distribution family subsystem), ADR-009 (boundary contracts and configuration validation)
**Respects:** the standing constraint that no new emitting family is introduced

## Summary (read this first — self-contained)

A HydraNet cell forecast has two parts: a **body** — how many, if any — and a **gate** — the chance
of any violence at all. ADR-069 made *how they combine* a configuration setting: the
**composition**.

What ADR-069 did not make configurable is **which of the resulting quantities a run writes to
disk**. That is currently an either/or. One composition setting yields the gated forecast; another
yields the plain body. A single run cannot produce both, because composition is applied before the
array is populated. The gate is written alongside whichever is chosen, so a run yields two of the
three quantities and cannot choose which two.

This matters because **the body is not recoverable from the gated forecast.** Under a thresholding
composition the body is zeroed outright on the overwhelming majority of cells — the information is
gone, not merely noisy.

This ADR makes the **set of emitted quantities** a configuration field, and makes it **additive**:
one run can yield the gated forecast and the body, from the same draws. It is deliberately a
*separate* setting from composition, because that setting is also read by the autoregressive
rollout — driving emission from it would silently change every forecast horizon beyond the first.

It changes no default. A configuration that does not set the field emits exactly what it emits
today.

**Scope note.** Selecting *which origins and which horizons* a run emits is a different axis with a
different risk profile, and is ADR-078. This ADR carries the hazard; that one carries none, and the
two are decided separately so that the harmless half does not wait on the careful half.

---

## 1. Context

**The problem: emission is an either/or, and the body is the casualty.**

Composition is applied inside the sample-cube construction, before the magnitude array is
populated. So the array on disk is **already composed**. Under a thresholding composition the mask
is a deterministic per-cell zero-or-one: cells at or above the threshold pass through untouched,
cells below it are zeroed for every draw.

Reconstruction is therefore possible only on cells at or above the threshold, which on a real global
gate posterior is a small minority of the grid. The measurement is recorded in the blast-radius
investigation under `reports/`.

The existing body-mean diagnostic argues this case from a *sampling* composition's variance
inflation. Under the thresholding composition actually in use the loss is **total, not noisy** — the
conclusion is right and the stated reason does not apply. That docstring should be corrected when
the file is next touched.

**Assumption that is no longer true.** ADR-069 assumed one composition per run is what a run needs,
because the composed forecast was the object under study. With the body itself now an object of
study, a run must be able to emit more than one view of a single inference pass. Paying for a second
pass is both wasteful and **not comparable**, because dropout draws differ between passes — two
passes give two samples of the posterior, not two views of one.

**Urgency.** The next experiment needs the body, and the routes available today are a driver script
that cannot run on rented hardware (§3) or a second inference pass that answers a different
question.

---

## 2. Decision

**Statement.** *We will make the set of quantities a run emits a first-class configuration field —
additive, and separate from composition — so that one inference pass can yield every view of itself
an experiment needs.*

### 2.1 The quantities field

A configuration field declares which of the **gated forecast**, the **gate** and the **body** are
written, and therefore which directory prefixes appear. Its default is today's set, so existing
configurations are byte-identical.

**Additive, not exclusive.** The field is a **set**. Asking for the gated forecast and the body
yields both, from one pass, from the same draws — which is the point, and is impossible today.

### 2.2 Emission is a separate setting from composition, and never touches the feedback path

The composition setting is read at four places, and two of them are not emission: the scored sample
cube, the point forecast **fed back** into the rollout, the sampled feedback draw, and a
consistency check.

Expressing *"also emit the body"* by varying composition would therefore change **what the
autoregressive rollout feeds back**, and so every horizon beyond the first. The run would complete,
write correct shapes and correct row counts, and not be the same model. Nothing would report it.

**So the quantities field is its own setting. It selects what is written. It does not reach the
feedback path, and a test asserts that output beyond the first horizon is bit-identical with it set
and unset.**

This is the one rule in this ADR whose violation corrupts a result rather than failing a run, and it
is the reason this axis is decided separately from ADR-078's.

### 2.3 The body is a diagnostic quantity, never a pipeline target

The body is emitted **outside** the target system. It is not added to the regression or
classification target lists, and it does not enter the combined target set the pipeline hands to
evaluation.

This is a correctness decision, not a convenience. Evaluation slices ground truth by target name,
and **there is no observable called "the ungated body."** It is a component of a forecast, not a
forecast of anything in the world. Declaring it a target is a category error, and four independent
guards already refuse it: the architectural invariant tying input channels to the regression head
count, the fleet-wide roster conformance pins, the evaluator's refusal of an undeclared target, and
the actuals-preparation contract itself.

No new emitting family is introduced. The body is an existing quantity of the existing family —
already computed on the inference path, and already dumped as a mean by the existing diagnostic.
Nothing new is fitted, trained or scored.

### 2.4 In-scope / out-of-scope

**In scope:** the quantities field, its validation, and HydraNet honouring it.

**Out of scope:**
- **Origin and horizon selection.** ADR-078. A different axis, and it cannot corrupt a result.
- **What comes home.** Transport and archive selection belong to the models repository, which
  already has a declaration vocabulary for deciding what ships. This ADR does not fork a second
  selection language; it gives that one something honest to select.
- **Which models set the field.** A models-repository decision.
- **The on-disk form of the body artifact.** Implementation, constrained by ADR-074.
- **Composition semantics.** ADR-069 stands unchanged. No composition value is added or changed.

---

## 3. Rationale & Integrity Impact

| alternative | why rejected |
|---|---|
| **overload the composition setting** | it is read by the rollout feedback (§2.2). Would silently alter every horizon beyond the first while every shape and row count stayed correct. The only option that corrupts rather than fails |
| **use the orchestrator-attribute route, as the existing diagnostics do** | measured unreachable in production. Every diagnostic on the orchestrator is an attribute defaulting to none, set by a driver script, and the manager sets none of them before running inference — so any run through the ordinary entry point, including every run on rented hardware, writes nothing. A configuration field is forced, not preferred |
| **declare the body a pipeline target** | category error (§2.3). Correctly refused by four independent guards |
| **run inference twice, once per composition** | wasteful, and **not comparable**: dropout draws differ between passes, so the two views would be of different posteriors |
| **reconstruct the body from the gated forecast and the gate** | possible only above the threshold, a small minority of cells (§1) |
| **a general artifact-selection language in this repository** | machinery that does not earn itself, and it would fork the models repository's existing vocabulary. One declarative field is the least that solves the problem |

**Reproducibility.** An experiment needing the body and the gated forecast gets both from **one**
pass over **one** set of draws, so the comparison is of the same posterior rather than of two
samples of it. With ADR-074, what was emitted is also recorded.

**Numerical stability.** Unchanged for any configuration that does not set the field, and the
guarantee is asserted by the §5 invariant rather than assumed.

**Fail-loud.** Validation refuses rather than degrades: a requested quantity the configured family
cannot produce raises at configuration construction, and the memory preflight raises before
allocation rather than letting the host terminate the run — a failure already observed on this work.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] The body becomes obtainable at all, and obtainable **alongside** the gated forecast from a
      single pass — the thing that is impossible today.
- [ ] Emission becomes declarable from an ordinary model configuration, so it works on rented
      hardware. No driver script, no orchestrator attribute, no never-modify file touched.
- [ ] The emission-versus-composition confusion is closed by construction rather than by care.
- [ ] A diagnostic question about the body's own behaviour becomes answerable without retraining.

### ⚠️ Negative (Costs)

- [ ] **Memory and disk.** Emitting the body adds a third cube of the magnitude cube's shape: half
      again the peak memory, and a substantial per-origin cost on the global grid. Sized in the
      blast-radius investigation and mitigated by a hard preflight, not by hope.
- [ ] One more field on a configuration schema that is already large, and it is then on the path of
      every later change. Accepted: a declarative field with a default, not a guard.
- [ ] A request that cannot be satisfied now fails at configuration time rather than producing
      something adjacent. Intended, and louder than today.
- [ ] Two quantities in count space now exist on disk, one of which is not a forecast. ADR-074 is
      what keeps them distinguishable, which is why it is a prerequisite rather than a companion.

---

## 5. Validation

**Invariants that must remain bit-perfect.**
- **With the quantities field set and unset, forecast output beyond the first horizon is
  bit-identical.** This guards §2.2 and is the most important assertion in this ADR.
- A configuration that does not set the field produces output byte-identical to a pre-ADR baseline
  at the same seed.

**Tests.**
- *Red:* a requested quantity the configured family cannot produce raises at configuration
  construction.
- *Red:* the cube-fit preflight — already variadic over the shapes it is given — raises **before
  allocation** when the extra cube does not fit. Assert the raise, not an out-of-memory kill.
- *Green:* with the body emitted, the body masked by the gate-at-threshold reproduces the gated cube
  **exactly** on the cells at or above the threshold. This proves the emitted body is the body that
  was composed, not a second draw.
- *Green:* ADR-074's manifest records the quantities actually emitted, for every combination tested.

**Honest gap.** None of these tests exists yet, and neither does the emission path they would check.
This ADR is currently intent with no enforcement, which is stated rather than implied.

**Failure mode that would trigger reconsideration.** If a future emitted quantity cannot be
expressed as a member of a set — one requiring its own inference pass, or its own loss — then the
field has become a dispatch mechanism rather than a declaration, and the axis needs rethinking
rather than extending.

---

## 6. Implementation Notes

- **Location.** Field and validator in the configuration contract layer (ADR-009). Honouring it is
  in the inference path and the orchestrator; the writer is the module ADR-074 introduces.
- **Ordering.** ADR-074 is a prerequisite. Emitting a third quantity without recording which prefix
  carries which quantity would make the ambiguity it fixes strictly worse — and the body is the
  hazard precisely because it shares the forecast's count space while not being the forecast.
- **Relationship to ADR-078.** Independent. Neither requires the other, and either can land first.
  They were originally one document; they were split because this axis can corrupt a result and
  that one cannot, so bundling would have made the harmless decision wait on the careful one.
- **References.** ADR-067, ADR-068, ADR-069, ADR-074, ADR-078. Cross-repository: the models
  repository's delivery declaration and its runner contract.
- **Evidence.** The six-repository blast-radius investigation under `reports/` holds the
  recoverability arithmetic, the cube sizing, and the measurement that the orchestrator attribute is
  unreachable in production. Numbers live there, not here.
