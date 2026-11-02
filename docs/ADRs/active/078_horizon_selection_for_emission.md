# ADR-078: Horizons are selectable for emission, never for computation

**Status:** Accepted
**Date:** 2026-11-02 · accepted 2026-11-02
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Requires:** ADR-074 (an emitted artifact records what it is)
**Related:** ADR-075 (emitting the body), ADR-009 (configuration validation)

## Summary (read this first — self-contained)

A forecast run rolls out to many **horizons** — one, two, three months ahead and so on. It writes
every one of them, always. There is no way to say that a particular piece of analysis needs only
the first and the last.

On a global grid that is a large volume of output, most of which a given analysis does not read,
and all of which must be stored and moved.

This ADR makes horizons selectable for emission, with one constraint that is easy to misread:
**selection applies to what is written, not to what is computed.** A rollout cannot skip months —
the twelfth horizon exists only because the first eleven were computed — so asking for one horizon
does not shorten the run. It shortens the output.

It also closes a trap. A configuration field already exists that **looks** like a horizon selector
and is not: only the *number* of entries in the declared step list is read, so a sparse list
produces that many *consecutive* horizons instead of the ones named. Downstream the mismatch is
caught on evaluation runs and not on forecasting runs. This ADR requires the list to be contiguous
and routes genuine selection to the new field.

It changes no default: a configuration that does not set the field emits exactly what it emits
today.

---

## 1. Context

**Nothing subsets anything.** Both inference paths iterate the full origin list and the full target
list, and every cell of the posterior becomes a row. The only per-origin gate anywhere is a
visualisation one, limited to the first origin.

**The step list appears to select horizons and does not.** Inside this repository only its *length*
is consumed — by the training-window slice, by a checksum against the declared count, and by the
reproducibility record. The values are read nowhere on the inference path, and nothing checks that
they are contiguous.

So a configuration declaring a sparse step list produces that many **consecutive** horizons. What
happens next depends on the run type, which is the part worth knowing:

- on an **evaluation** run the pipeline builds a month-to-step mapping **from the declared values**
  and raises on any predicted month outside it, so a sparse list fails loudly;
- on a **forecasting** run that check is not called, so the mismatch stands.

Nothing in either repository attaches a step label to an emitted row, so a sparse list does not
mislabel anything — the declared values are simply ignored here and then contradicted there.

**Why now.** Horizons acquire a stated mapping under ADR-074. Until they are named, selecting them
cannot be expressed; once they are, the selection is a small addition, and the step list is the only
remaining thing that appears to offer it.

---

## 2. Decision

**Statement.** *We will make horizons selectable for emission, require that the selection never
change what is computed, and require the declared step list to be a contiguous count rather than a
selector.*

### 2.1 The field, and its name

A configuration field declares which horizons a run emits. Its default is all of them.

**Its name must say that it selects emission**, not horizons in general — so it reads as
*emit*-something rather than as a bare horizon or step setting. This is not cosmetic. The field it
sits beside, the declared step list, is already a setting that *appears* to select horizons and
does not (§2.3), and the single most likely misuse of this one is to expect it to shorten a run
(§2.2). A name that states which of the two it does is the cheapest defence against both, and the
only one present at the moment someone types it.

### 2.2 Selection applies to output, not to computation

The field chooses what is written. It does not change what is computed.

⇒ **This constraint is defended in three places, deliberately**, because an ADR cannot reach the
person who misreads it: in the field's **name** (§2.1), in the field's own **documentation**, and
by the **test** in §5 that a selected subset is bit-identical to the corresponding part of a full
run — which *proves* no computation was skipped rather than asserting it.

### 2.3 The declared step list is a count, not a selector

The horizon field is the only way to select horizons. The declared step list must be contiguous and
ascending from one, or the configuration is refused.

### 2.4 Scope

**In scope:** the field, its validation, the step-list constraint, and the model honouring them.

**Out of scope:** selecting **origins**, for the reason in §3; which quantities a run emits
(ADR-075); what is transported off rented hardware, which the models repository owns; which models
set the field; and shortening a run, which §2.2 explicitly does not offer.

---

## 3. Rationale & Integrity Impact

| alternative | why it loses |
|---|---|
| **let selection reduce computation too** | a rollout cannot skip months, so for horizons it is impossible. For origins it would change which draws are taken, because the dropout stream is seeded once per run and consumed in order — so an origin subset would not be a subset of a full run's output, it would be different numbers |
| **also select origins** | **not available.** The pipeline requires an evaluation run to return exactly one sequence per expected origin and raises otherwise, so an origin subset is refused on the runs where output volume actually costs. The one existing origin-selection parameter is dead — nothing anywhere passes it — and it fixes the *last* origin and rolls backwards, so it changes what is computed, which §2.2 forbids. This is excluded on availability, not on preference, and would become available if that contract were relaxed |
| **express sparse horizons through the step list** | that is what the list appears to offer. Making it real would require a non-contiguous rollout, which §2.2 rules out; leaving it as it is keeps a field that misdescribes the run on one of the two run types |
| **subset afterwards with a separate tool** | the present arrangement, and it has produced one bespoke slicing script per artifact class. A declared selection is honoured by one path; a post-hoc slicer is rewritten for each new need |
| **decide this together with which quantities are emitted** | that axis is read by the autoregressive feedback and can change a result; this one cannot. Different risk, different evidence, and a reviewer should be able to accept one without the other |

**Reproducibility.** A selected subset is bit-identical to the corresponding part of a full run,
which §5 asserts rather than assumes.

**Fail-loud.** The step-list constraint is refused when the configuration is built rather than
discovered during analysis, and a selection naming a horizon that does not exist is refused rather
than quietly emitting less than was asked for.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] Output volume becomes controllable, which matters most where it is most expensive.
- [ ] A latent trap closes: the step list stops appearing to select horizons, on both run types
      rather than only the one that currently catches it.
- [ ] Removes the standing reason to write another bespoke post-hoc slicing script.

### ⚠️ Negative (Costs)

- [ ] **The field will be read as a speed-up**, despite §2.2 and its own documentation. There is no
      defence beyond documentation, which has failed at this kind of thing here before.
- [ ] A new refusal rejects a configuration shape that is accepted today. No current configuration
      is affected, but a future sparse step list must move to the new field.
- [ ] Output volume is only half controllable: the expensive dimension is origins, and origins are
      not available (§3). The saving is real but smaller than the problem.
- [ ] A subset is less than a full run, so analysis wanting more later must re-run.

---

## 5. Validation

**Invariants.**
- A configuration that does not set the field produces output byte-identical to a run before this
  change, at the same seed.
- A selected subset is bit-identical to the corresponding part of a full run at the same seed. This
  establishes that selection did not perturb the rollout.

**Tests.**
- *Green:* a horizon subset changes only which artifacts exist; those that exist match the full run
  exactly.
- *Red:* a sparse step list is refused when the configuration is built; a contiguous one is not.
- *Red:* selecting a horizon outside the run's range is refused, rather than emitting fewer
  artifacts than requested.
- *Green:* ADR-074's manifest records which horizons were emitted, so a subset is declared rather
  than inferred from which files happen to be present.

**Not yet enforced.** None of these tests exists, and neither does the field.

**What would reopen this.** A need to shorten a *run* rather than its output — a cheap preflight at
one horizon, say — would make §2.2 the wrong constraint for that use and would need its own
decision. Separately, if the pipeline's sequence-count contract were relaxed, origin selection
would become available and §3's exclusion would no longer hold.

---

## 6. Implementation Notes

- **Location.** The field and its validator belong with the configuration contract (ADR-009). The
  field gates the write, not the loop.
- **Ordering.** ADR-074 first: a horizon must be named before it can be selected, and a subset must
  be declared in the manifest or it cannot be told apart from a partial failure.
- **Noted, not addressed.** Forecasting runs write no per-origin directory level while other run
  types do, and the step-list mismatch is caught on one and not the other. Both are real; neither is
  fixed here.
- **Evidence.** `reports/2026-11-02_adr_claim_verification/00_CLAIMS.md` §5 holds every factual claim
  in this ADR with its source, including the sequence-count contract, the dead origin parameter, and
  which run type catches a sparse step list.

---

## References

**This repository:** ADR-009 (configuration validation) · ADR-074 (an emitted artifact records what
it is) · ADR-075 (emitting the body).

**Other repositories** — the contracts this ADR works within:

- [views-models ADR-019 — delivery declaration](https://github.com/views-platform/views-models/blob/main/docs/ADRs/019_delivery_declaration.md). The existing vocabulary for declaring what is produced and shipped; a selection made here is what it would select from.
- [views-models ADR-024 — pod runner contract](https://github.com/views-platform/views-models/blob/main/docs/ADRs/024_pod_runner_contract.md). What a run on rented hardware guarantees, and why a selection reachable only from a driver script would be unusable where output volume costs.
- [views-pipeline-core ADR-013 — prediction naming convention](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/013_prediction_naming_convention.md). The directory layout a selection subsets, and the layout that differs between run types.
