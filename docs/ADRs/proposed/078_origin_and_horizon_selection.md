# ADR-078: Origins and horizons are selectable for emission, never for computation

**Status:** Proposed
**Date:** 2026-11-02
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Requires:** ADR-074 (emission provenance — horizons must be named before they can be selected)
**Independent of:** ADR-075 (which quantities a run emits — a separate axis, deliberately not bundled)
**Builds on:** ADR-009 (boundary contracts and configuration validation)

## Summary (read this first — self-contained)

An inference run produces a forecast for every **origin** it rolls from and every **horizon** it
rolls to. Today it writes all of them, always. There is no way to say *"this run only needs the
last origin"* or *"only the first and last horizons."*

This ADR makes origins and horizons selectable, with one constraint that is easy to get wrong:
**selection applies to what is written, never to what is computed.** An autoregressive rollout
cannot skip months — a late horizon exists only because the earlier ones were computed — so
selecting one horizon does not shorten the run. It only shortens the output.

It also closes a trap. A configuration field already exists that *looks* like a horizon selector
and is not: only the **length** of the declared step list is ever read, so a sparse list rolls out
that many **consecutive** months and labels them as though they were the horizons named. Nothing
reports it. This ADR requires that list to be contiguous, and routes genuine horizon selection to
the new field instead.

It changes no default. A configuration that sets neither field emits exactly what it emits today.

**Scope note.** This axis cannot corrupt a result — it only subsets output. The axis that *can*,
which quantities a run emits, is ADR-075. The two were originally one document and were split so
this harmless half need not wait on the careful one.

---

## 1. Context

**Nothing selects anything.** Neither inference path subsets targets, origins or horizons. Both
loop the full origin list and the full target list, and the whole posterior cube becomes rows. The
only existing per-origin gate is a visualisation one, which fires for the first origin only.

**One of the seams is already built and unwired.** The origin-sequence helper already accepts a
parameter that fixes the last origin. Only tests pass it; the manager never does. So an
origin-selection mechanism exists, has existed, and is unreachable from a configuration.

**The step list looks like a horizon selector and is not.** Only its **length** is read — by the
training-window slice, by the checksum against the declared step count, and by the reproducibility
record. The **values** are read nowhere on the inference path, and there is no contiguity check and
no value check.

So a configuration declaring a sparse step list passes every gate, rolls out that many consecutive
months, and labels them as though they were the horizons named. The run completes, the row counts
are correct, and nothing reports the mismatch.

Latent, not firing: every live configuration declares a contiguous range from one, so **no past
result is affected.** It is recorded here because it is a trap with the same shape as the ones that
have already cost us — a declaration nothing enforces.

**Why now.** Horizons acquire a stated mapping under ADR-074. Until they are named, selecting them
is not expressible; once they are, the selection is a small addition and the step-list trap becomes
the only remaining way to *appear* to select them.

---

## 2. Decision

**Statement.** *We will make origins and horizons selectable for emission, require that selection
never affect computation, and require the declared step list to be a contiguous count rather than a
selector.*

### 2.1 The two fields

| field | declares | default |
|---|---|---|
| **origins** | all origins, or a named subset. Wires the existing fixed-last-origin seam rather than duplicating it | all |
| **horizons** | all horizons, or a subset | all |

Both default to all, so an existing configuration is unaffected.

### 2.2 Compute-all, emit-some

An autoregressive rollout cannot skip months. A late horizon exists only because the earlier ones
were computed, and an origin's forecast depends on the state rolled forward to it.

**So these fields select what is written, never what is computed.** They are not a speed-up.

This must be stated in each field's own documentation, not only here — otherwise someone will
select a single late horizon expecting a short run, get a full-length one, and conclude the field
is broken.

### 2.3 The declared step list is a count, not a selector

**The horizons field is the only way to select horizons.**

The declared step list must be **contiguous ascending from one**, or the configuration raises. A
sparse set of horizons is expressed with the horizons field, never by relabelling the step list.

This belongs with §2.2 rather than with provenance, because it is the same rule from the other
side: a field that *appears* to select horizons while only setting their number is the inverse of
selecting emission without selecting computation. The two are only coherent stated together.

### 2.4 In-scope / out-of-scope

**In scope:** the two fields, their validation, the step-list constraint, and HydraNet honouring
them.

**Out of scope:**
- **Which quantities are emitted.** ADR-075.
- **What comes home.** Transport and archive selection belong to the models repository, which
  already has a declaration vocabulary for deciding what ships.
- **Which models set the fields.** A models-repository decision.
- **Shortening a run.** Explicitly not offered, by §2.2. If run length ever needs reducing, that is
  a different decision about the rollout itself.

---

## 3. Rationale & Integrity Impact

| alternative | why rejected |
|---|---|
| **let selection subset computation too** | impossible for horizons in an autoregressive rollout, and for origins it would change the state rolled forward. A field that silently meant different things for the two axes would be worse than no field |
| **express sparse horizons by relabelling the step list** | this is what the current field appears to offer and does not. Making it real would mean a non-contiguous rollout, which §2.2 rules out; leaving it as-is keeps a label that lies |
| **keep the origin seam driver-only, as it is today** | it is unreachable from a configuration, so it cannot be used on rented hardware — the same dead end ADR-075 documents for its own axis |
| **do nothing, and subset after the fact with a separate tool** | this is the status quo, and it has produced one bespoke slicing script per artifact class. A declared selection is read by one path; a post-hoc slicer is re-invented per need |
| **bundle this with ADR-075** | originally done, and rejected on review: that axis can silently corrupt a result and this one cannot, so bundling would make the harmless decision wait on the careful one |

**Reproducibility.** Unaffected in substance, and improved in practice: a selected subset is
bit-identical to the corresponding slice of a full run, which §5 asserts rather than assumes.

**Numerical stability.** Untouched. Nothing in the computation path changes, by §2.2.

**Fail-loud.** The step-list constraint raises at configuration construction rather than at analysis
time, and a selection naming an origin or horizon that does not exist raises rather than silently
emitting less.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] Output volume becomes controllable, which matters most where it is most expensive — the
      global grid on rented hardware.
- [ ] An already-built origin-selection seam becomes reachable rather than duplicated.
- [ ] A latent trap closes: the step list stops appearing to select horizons.
- [ ] Removes the standing reason to write another bespoke post-hoc slicing script.

### ⚠️ Negative (Costs)

- [ ] **The fields will be misread as a speed-up**, despite §2.2. Documentation is the only defence,
      and documentation has already failed at this kind of thing in this codebase.
- [ ] A new refusal will reject a configuration shape that passes today. Latent, so no live
      configuration is affected — but a future sparse step list must move to the horizons field.
- [ ] Two more fields on a configuration schema that is already large.
- [ ] A selected subset is less than a full run, so an analysis wanting more later must re-run. The
      alternative is emitting everything always, which is the current cost.

---

## 5. Validation

**Invariants that must remain bit-perfect.**
- A configuration setting neither field produces output byte-identical to a pre-ADR baseline at the
  same seed.
- **A selected subset is bit-identical to the corresponding slice of a full run at the same seed.**
  This is §2.2's compute-all guarantee, and it is the assertion that proves selection did not
  perturb the rollout.

**Tests.**
- *Green:* an origin or horizon subset changes only *which* artifacts exist; those that exist match
  the full run's corresponding slices exactly.
- *Red:* a sparse step list raises at configuration construction; a contiguous one does not.
- *Red:* selecting an origin or horizon outside the run's range raises, rather than emitting fewer
  artifacts than requested.
- *Green:* ADR-074's manifest records the origins and horizons actually emitted — so a subset is
  declared in the artifact, not inferred from which directories happen to be present.

**Honest gap.** None of these tests exists yet, and neither do the fields. This ADR is currently
intent with no enforcement, which is stated rather than implied.

**Failure mode that would trigger reconsideration.** If a legitimate need arises to *shorten* a run
rather than its output — for instance a cheap preflight at one horizon — then §2.2 is the wrong
constraint for that use, and it needs its own decision rather than a quiet reinterpretation of
these fields.

---

## 6. Implementation Notes

- **Location.** Fields and validators in the configuration contract layer (ADR-009). The origin
  field wires the existing sequence-helper parameter; the horizon field gates the write, not the
  loop.
- **Ordering.** ADR-074 is a prerequisite: horizons must be named before they can be selected, and
  a subset must be declared in the manifest or it is indistinguishable from a partial failure.
- **Relationship to ADR-075.** Independent. Neither requires the other, and either can land first.
- **Noted, not fixed.** Forecasting runs write no per-origin directory level while the other run
  types do, so a per-origin selection means something different between them. Real, recorded,
  **not addressed here.**
- **References.** ADR-009, ADR-074, ADR-075. Cross-repository: the models repository's delivery
  declaration and its runner contract.
- **Evidence.** The six-repository blast-radius investigation under `reports/` records that nothing
  in either repository selects targets, origins or horizons, that the origin seam is test-only, and
  that only the length of the step list is read.
