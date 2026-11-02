# ADR-075: A run can be asked to emit the gate and the body alongside the forecast

**Status:** Proposed
**Date:** 2026-11-02
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Requires:** ADR-074 (an emitted artifact declares what it is)
**Related:** ADR-069 (composition as a config axis), ADR-067 (distribution family subsystem)

## Summary (read this first — self-contained)

The model produces forecasts. A forecast is made of two parts — a **gate**, the chance that any
violence occurs in a cell, and a **body**, how many deaths if it does — which can be combined in
several ways. The combination is the **composition**, and the combined number is what the model
emits.

Neither part can be fully reconstructed from the combined forecast. Composition is applied before
anything is written, and the common case discards the body wherever the gate is low.

For audit, evaluation, experiments and continuous improvement of the model we need the gate and the
body **independently**. That is a different thing from getting the forecast out: it is emitting
**subcomponents of a forecast, for diagnostic purposes.** The subcomponents are not forecasts and
must never be read as forecasts.

This ADR makes the set of emitted quantities a configuration field, so that a single run can produce
the forecast, the gate and the body together from the same draws. It changes no default: a
configuration that does not set the field emits exactly what it emits today.

---

## 1. Context

**A subcomponent cannot be recovered from the forecast.** Composition runs inside the inference pass
and the composed array is what reaches disk. Under the thresholding composition in ordinary use, the
body is set to zero wherever the gate falls below the threshold, for every draw. Above the threshold
the body survives untouched, so recovery is possible there and nowhere else — and on a global grid
the cells above threshold are a small minority. The measured fraction is recorded in the
blast-radius investigation under `reports/`.

**Diagnostics need the parts, not the whole.** Questions about whether the model is losing cells or
losing magnitude, whether the gate is calibrated, and whether a composition choice helps or harms,
are all questions about one part while the other is held still. A combined forecast cannot answer
them, because every combined readout conflates the two.

**A second inference pass is not a substitute.** Running the model again under a different
composition yields a different set of dropout draws, so the two outputs are samples of the posterior
rather than two views of one. Any comparison between them carries that difference as a confound.

**What exists today.** A diagnostic dump of the body's posterior *mean* exists, reachable only by
setting an attribute on the inference orchestrator from a driver script. The ordinary entry point
never sets it, so a run launched normally — including every run on rented hardware — produces
nothing. A mean also cannot support a distributional readout.

---

## 2. Decision

**Statement.** *We will make the set of quantities a run emits a configuration field, so one
inference pass can yield the forecast, the gate and the body from the same draws.*

### 2.1 The field

A configuration field declares which of the **forecast**, the **gate** and the **body** a run
writes. It is a **set**, so asking for the forecast and the body yields both. Its default is the
set emitted today.

### 2.2 Emission is a separate field from composition

The field that selects emission is not the field that selects composition, and the emission field
does not reach the autoregressive feedback path. Output beyond the first horizon is identical
whether the emission field is set or not.

### 2.3 Subcomponents are diagnostics, not targets

The gate and the body are emitted outside the pipeline's target system. They are not declared as
regression or classification targets, and they do not enter the target set handed to evaluation.

A subcomponent is not a forecast of anything in the world, so no ground truth exists to score it
against.

### 2.4 Scope

**In scope:** the field, its validation, and the model honouring it.

**Out of scope:** which origins and horizons a run emits (ADR-078); what is transported off rented
hardware, which the models repository owns; which models set the field, which is a configuration
decision; composition semantics, which ADR-069 governs unchanged.

---

## 3. Rationale & Integrity Impact

| alternative | why it loses |
|---|---|
| **select emission with the composition field** | that field is also read by the autoregressive feedback, so using it to choose what is emitted would change what the rollout feeds back, and therefore every horizon beyond the first. Shapes and row counts would stay correct and nothing would report the difference. This is the only alternative that can corrupt a result rather than fail |
| **set an attribute on the orchestrator, as existing diagnostics do** | unreachable from a configuration. The ordinary entry point sets no such attribute, so a normally launched run writes nothing — which is the situation today |
| **declare the subcomponents as pipeline targets** | evaluation slices ground truth by target name, and no observable corresponds to a subcomponent. Several independent guards refuse it, correctly |
| **run inference twice under different compositions** | the two runs draw differently, so the outputs are not two views of one posterior |
| **reconstruct the body from the forecast and the gate** | possible only where the gate exceeds the threshold, a small minority of cells |
| **build a general artifact-selection language here** | the models repository already has a declaration vocabulary for deciding what is produced and shipped. A second one would fork it, and one field is enough for this decision |

**Reproducibility.** A diagnostic comparison between a subcomponent and the forecast is drawn from
one pass over one set of draws, so the comparison carries no sampling confound.

**Fail-loud.** A requested quantity the configured distribution family cannot produce is refused
when the configuration is built. A run whose additional output would not fit in memory is refused
before allocation rather than terminated during it.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] The gate and the body become available at all, and available together with the forecast from
      one pass.
- [ ] Emission is declarable from an ordinary model configuration, so it works wherever a model
      runs, including rented hardware.
- [ ] Diagnostic questions about one subcomponent become answerable without retraining.
- [ ] The distinction between choosing what to emit and choosing how to compose becomes structural
      rather than a matter of care.

### ⚠️ Negative (Costs)

- [ ] **Memory and storage.** Each additional quantity is another array of the forecast cube's
      shape, raising peak memory by a substantial fraction and adding a comparable volume per
      origin. Sized in the blast-radius investigation and refused by preflight when it will not fit.
- [ ] Two quantities in the same units now exist on disk, only one of which is a forecast. ADR-074
      is what keeps them distinguishable, which is why it is a prerequisite.
- [ ] One more configuration field, on a schema that is already large.
- [ ] A request that cannot be met now fails when the configuration is built, rather than producing
      something adjacent to what was asked for.

---

## 5. Validation

**Invariants.**
- Output beyond the first horizon is bit-identical with the emission field set and unset.
- A configuration that does not set the field produces output byte-identical to a run before this
  change, at the same seed.

**Tests.**
- *Red:* a requested quantity the configured family cannot produce is refused at configuration time.
- *Red:* the memory preflight refuses an oversized request before allocating, and the refusal is
  asserted rather than the resulting failure.
- *Green:* the emitted body, masked by the gate at the composition's threshold, reproduces the
  emitted forecast exactly on the cells above threshold. This establishes that the emitted
  subcomponent is the one the forecast was built from, not an independent draw.
- *Green:* ADR-074's manifest records which quantities were emitted.

**Not yet enforced.** None of these tests exists, and neither does the emission path. This ADR is
intent without enforcement until they are written.

**What would reopen this.** A quantity that cannot be expressed as a member of a set — one needing
its own inference pass, or its own loss — would mean the field has become a dispatch mechanism
rather than a declaration, and the axis would need redesigning rather than extending.

---

## 6. Implementation Notes

- **Location.** The field and its validator belong with the configuration contract (ADR-009). The
  emission path honours it; the artifact's self-description is written by the module ADR-074
  introduces.
- **Ordering.** ADR-074 first. A subcomponent in the forecast's own units, written beside it without
  anything recording which is which, is worse than not emitting it.
- **References.** ADR-009, ADR-067, ADR-069, ADR-074, ADR-078. Cross-repository: the models
  repository's delivery declaration and runner contract.
- **Evidence.** `reports/2026-11-01_body_output_blast_radius/00_REPORT.md` holds the recoverability
  arithmetic, the memory sizing, and the finding that the existing diagnostic attribute is
  unreachable from an ordinary run.
