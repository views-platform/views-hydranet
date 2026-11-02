# ADR-078: Origins and horizons are selectable for emission, never for computation

**Status:** Proposed
**Date:** 2026-11-02
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Requires:** ADR-074 (an emitted artifact records what it is)
**Related:** ADR-075 (emitting subcomponents), ADR-009 (configuration validation)

## Summary (read this first — self-contained)

An inference run rolls forward from several **origins** — the months it starts forecasting from —
and out to many **horizons** — how far ahead each forecast reaches. It writes every origin and every
horizon, always. There is no way to ask for only the last origin, or only the first and last
horizons.

On a global grid that is a large volume of output, most of which a given piece of analysis does not
need, and all of which must be stored and moved.

This ADR makes origins and horizons selectable, with one constraint that is easy to misread:
**selection applies to what is written, not to what is computed.** A forecast at a late horizon
exists only because the earlier ones were computed, so asking for one horizon does not shorten the
run — it shortens the output.

It also closes a trap. A configuration field already exists that appears to select horizons and does
not: only the number of entries in the declared step list is read, so a sparse list produces that
many *consecutive* horizons under the labels given. This ADR requires that list to be contiguous and
routes genuine selection to the new field.

It changes no default: a configuration setting neither field emits exactly what it emits today.

---

## 1. Context

**Nothing selects anything.** Both inference paths iterate the full origin list and the full target
list, and every cell of the posterior becomes a row. The only per-origin gate anywhere is a
visualisation one, limited to the first origin.

**An origin-selection mechanism already exists and is unreachable.** The helper that chooses origins
accepts a parameter fixing the last one. Only tests pass it; nothing in the ordinary path does. So
the capability exists but cannot be requested from a configuration.

**The step list appears to select horizons and does not.** Only its length is consumed — by the
training-window slice, by a checksum against the declared count, and by the reproducibility record.
The values are read nowhere on the inference path, and nothing checks that they are contiguous or
that they mean anything.

A configuration declaring a sparse step list therefore passes every gate, produces that many
consecutive horizons, and labels them as the horizons named. The run completes, the row counts are
right, and the mismatch is reported nowhere.

Every current configuration declares a contiguous range from one, so no result produced so far is
affected. The trap is recorded because its shape — a declaration nothing enforces — is the one that
has already cost this project time.

**Why now.** Horizons acquire a stated mapping under ADR-074. Until they are named, selecting them
cannot be expressed; once they are, the selection is a small addition, and the step list becomes the
only remaining way to appear to select them.

---

## 2. Decision

**Statement.** *We will make origins and horizons selectable for emission, require that selection
never change computation, and require the declared step list to be a contiguous count rather than a
selector.*

### 2.1 The two fields

A configuration field declares which **origins** a run emits, and another declares which
**horizons**. Both default to all, so an existing configuration is unaffected. The origin field uses
the existing origin-selection parameter rather than a second mechanism.

### 2.2 Selection applies to output, not to computation

Both fields choose what is written. Neither changes what is computed: a rollout cannot skip months,
and an origin's forecast depends on the state carried forward to it.

This is stated in each field's own documentation as well as here.

### 2.3 The declared step list is a count, not a selector

The horizon field is the only way to select horizons. The declared step list must be contiguous and
ascending from one, or the configuration is refused.

### 2.4 Scope

**In scope:** the two fields, their validation, the step-list constraint, and the model honouring
them.

**Out of scope:** which quantities a run emits (ADR-075); what is transported off rented hardware,
which the models repository owns; which models set the fields; and shortening a run, which §2.2
explicitly does not offer.

---

## 3. Rationale & Integrity Impact

| alternative | why it loses |
|---|---|
| **let selection reduce computation as well as output** | impossible for horizons, since a rollout cannot skip months, and for origins it would change the state carried forward. A field meaning one thing for horizons and another for origins would be worse than no field |
| **express sparse horizons through the step list** | that is what the field appears to offer. Making it real would require a non-contiguous rollout, which §2.2 rules out; leaving it as it is keeps a label that misdescribes the output |
| **leave the origin mechanism reachable only from a driver script** | it cannot then be used wherever models actually run, including rented hardware |
| **subset afterwards with a separate tool** | the present arrangement, and it has produced one bespoke slicing script per artifact class. A declared selection is honoured by one path; a post-hoc slicer is rewritten for each new need |
| **decide this together with which quantities are emitted** | that axis can silently change a result and this one cannot. Combining them would make a harmless decision wait on a careful one |

**Reproducibility.** A selected subset is bit-identical to the corresponding part of a full run,
which §5 asserts rather than assumes.

**Fail-loud.** The step-list constraint is refused when the configuration is built rather than
discovered during analysis, and a selection naming an origin or horizon that does not exist is
refused rather than quietly emitting less than was asked for.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] Output volume becomes controllable, which matters most where it is most expensive.
- [ ] An existing origin-selection mechanism becomes usable rather than duplicated.
- [ ] A latent trap closes: the step list stops appearing to select horizons.
- [ ] Removes the standing reason to write another bespoke post-hoc slicing script.

### ⚠️ Negative (Costs)

- [ ] **The fields will be read as a speed-up**, despite §2.2 and the field documentation. This has
      no defence beyond documentation, which has failed at this kind of thing before.
- [ ] A new refusal rejects a configuration shape that is accepted today. No current configuration
      is affected, but a future sparse step list must move to the horizon field.
- [ ] Two more configuration fields, on a schema that is already large.
- [ ] A subset is less than a full run, so analysis wanting more later must re-run. The alternative
      is emitting everything always, which is the current cost.

---

## 5. Validation

**Invariants.**
- A configuration setting neither field produces output byte-identical to a run before this change,
  at the same seed.
- A selected subset is bit-identical to the corresponding part of a full run at the same seed. This
  is the assertion that establishes selection did not perturb the rollout.

**Tests.**
- *Green:* a subset changes only which artifacts exist; those that exist match the full run exactly.
- *Red:* a sparse step list is refused when the configuration is built; a contiguous one is not.
- *Red:* selecting an origin or horizon outside the run's range is refused, rather than emitting
  fewer artifacts than requested.
- *Green:* ADR-074's manifest records which origins and horizons were emitted, so a subset is
  declared rather than inferred from which directories happen to be present.

**Not yet enforced.** None of these tests exists, and neither do the fields.

**What would reopen this.** A legitimate need to shorten a *run* rather than its output — a cheap
preflight at one horizon, for instance — would make §2.2 the wrong constraint for that use, and
would need its own decision rather than a reinterpretation of these fields.

---

## 6. Implementation Notes

- **Location.** Fields and validators belong with the configuration contract (ADR-009). The origin
  field wires the existing selection parameter; the horizon field gates the write rather than the
  loop.
- **Ordering.** ADR-074 first: a horizon must be named before it can be selected, and a subset must
  be declared in the manifest or it cannot be told apart from a partial failure.
- **Noted, not addressed.** Forecasting runs write no per-origin directory level while other run
  types do, so a per-origin selection means something different between them.
- **References.** ADR-009, ADR-074, ADR-075. Cross-repository: the models repository's delivery
  declaration and runner contract.
- **Evidence.** `reports/2026-11-01_body_output_blast_radius/00_REPORT.md` records that nothing in
  either repository selects targets, origins or horizons, that the origin parameter is test-only,
  and that only the length of the step list is read.
