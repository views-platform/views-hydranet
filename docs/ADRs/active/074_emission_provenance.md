# ADR-074: An emitted artifact records what it is

**Status:** Accepted
**Date:** 2026-11-01 (accepted 2026-11-02)
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers · views-frames maintainers
**Required by:** ADR-075 (emitting subcomponents), ADR-078 (origin and horizon selection)
**Related:** ADR-003 (declarations over inference), ADR-032 (output schema), ADR-008 (explicit failure)

## Summary (read this first — self-contained)

An inference run leaves prediction arrays on disk. Each row carries the month it refers to and the
grid cell it refers to. Nothing else travels with the bytes.

So an artifact cannot say which composition produced it, which forecast horizon a row belongs to,
which origin it was rolled from, or which model and configuration made it. All of that lives in
directory names and in the memory of whoever launched the run. A reader who was not that person
cannot establish what they are holding, and a reader who was can only recall it.

This ADR requires a run to write a **manifest** stating what it produced, and requires the **origin**
of each artifact to be recorded rather than inferred from where the artifact sits.

The manifest records what the run *did*. A configuration records what it was *asked* to do. Both are
kept, because the value of either depends on being able to compare them.

Nothing about what is computed or emitted changes, and no reported number moves.

---

## 1. Context

**Identity is carried by location, not by content.** The target is a leaf directory name, the origin
is a directory name, the run type and a timestamp are a parent directory name. The spatial level is
not stored at all and must be supplied by whoever loads the file.

Absent from the bytes: the composition settings, the sample widths, the seeds, the model name, the
artifact digest, any configuration digest, the origin, the origin's first month, and the horizon.

The trained-model sidecar does not close this. It records architecture and seeds, but not the
settings applied at emission — by design, because one trained model is legitimately evaluated under
several compositions. Composition belongs to the emission, not to the weights.

**Two coordinates are inferred rather than declared, and both inferences can be wrong silently.**

The horizon is taken as a row's offset from the earliest month present. That assumes an origin's
months are contiguous and that its first horizon is present. Several consumers derive it
independently; nothing asserts either premise. A gap in an origin's months would make a given
horizon a different month in different origins, with correct row counts throughout.

The origin is taken from a directory's position in a sorted list, which is a sequence number rather
than the origin itself. The origin and its first month appear nowhere in the output, while a
separate diagnostic records the origin directly — so two artifacts of one run can number their
origins differently, and joining them requires re-deriving how origins were chosen.

**The repository already forbids this.** ADR-003 holds that meaningful semantics must be declared,
that inferring them across a component boundary is forbidden, and that where they are missing the
system must not guess.

**What is not the problem.** ADR-032 fixes output to the scale of the input — counts in, counts out.
Internal transformations are invisible outside the model because a prediction is returned
un-transformed. A forecast is therefore correctly named and unambiguous in kind, and that rule is
unaffected by this one. What ADR-032 does not cover is a quantity that is **not a forecast**:
ADR-075 introduces such quantities, one of them in the forecast's own units, and nothing today would
distinguish it from a forecast.

---

## 2. Decision

**Statement.** *We will require every inference run to write a manifest declaring what it produced,
and require origin to be recorded rather than inferred.*

### 2.1 The manifest

One file per run, written into the run's generated-data directory, declaring:

| group | source | records |
|---|---|---|
| the **resolved configuration**, in full — composition settings, sample widths, step count, grid extent, spatial level, seeds | the validated configuration object the run used | fact |
| a **commit identifier for each repository involved**, each with a flag stating whether that working tree was modified | version control, at launch | intent |
| the **quantities emitted**, and the directory prefix each was written under | the emission path | fact |
| the **origins emitted** — for each, its sequence number, its origin, and its first month | the run | fact |
| the **horizon-to-month map**, per origin | the run | fact |
| the **expected row counts**, per horizon and per origin | the run | fact |
| **identity** — model name, run type, artifact filename, artifact digest, package version | the artifacts | fact |

The resolved configuration is recorded whole rather than filtered. A filter is a judgement about
what will matter later.

### 2.2 Origin is recorded; horizon follows from it

Origin is the primitive. A horizon is a function of an origin and a month, so both are not recorded
as independent facts about a row. For an artifact holding one origin, the origin is a property of
the artifact and belongs in its header.

The horizon-to-month map is nevertheless stated in the manifest, because the layer that holds the
month treats it as an opaque integer and validates only structure. It cannot establish that a
horizon means the same month across origins, whatever identifiers it carries.

### 2.3 Contiguity is asserted

A run fails if an origin's months are not contiguous and ascending. A failed assertion fails the
run, not the reader: it must not be possible to publish a manifest whose horizon map cannot be
relied on.

### 2.4 The manifest's values come from the objects that produced them

Values are taken from the objects the computation used, not by re-reading configuration files, not
by re-deriving, and not by recomputing.

### 2.5 Scope

**In scope:** what a run declares about its own output, and the validation that keeps the
declaration true.

**Out of scope:** what is transported off rented hardware, which the models repository owns;
retrofitting artifacts already written; and any change to what is computed or emitted.

**Adjacent, and available.** The prediction-frame layer's metadata model already anticipates this
extension: its identifiers are a required month-and-unit pair, and a further per-row axis — step,
origin, scenario — is explicitly provided for as an *optional* identifier added when a consumer
needs one. Its header type carries producer-generic fields, extensible the same way. An origin
plausibly qualifies as producer-generic; a composition does not, being this model family's own
vocabulary. The route for origin is therefore a request to that repository, and that layer's own
serialisation already writes a header carrying the spatial level and producer metadata which the
pipeline currently does not use.

---

## 3. Rationale & Integrity Impact

| alternative | why it loses |
|---|---|
| **a commit identifier instead of a manifest** | it requires an external lookup, at a revision that may have been rewritten, in one of several repositories — so the artifact's identity again depends on something outside the artifact. The test: can a reader establish which quantity an artifact is, and under what composition, holding only the artifact, offline? A resolved configuration answers; a commit identifier does not. One commit per run is also not enforceable, and a commit carries no execution facts at all |
| **the configuration files themselves** | a configuration is executable: it computes, branches, reads the environment, and the schema supplies defaults for what it omits. The files record the program; the run consumed its output |
| **path conventions plus a reference document** | the present arrangement. A naming table is not a property of the bytes and must be kept in step by hand |
| **the trained-model sidecar** | written at training time, while one model is evaluated under several compositions. Identity belongs to the emission |
| **a lookup from a configuration digest** | needs a registry and a reader with access to it. The artifact still cannot answer for itself |

**Reproducibility.** A result becomes traceable to a composition, a sample width, a seed and a model
digest without consulting anything outside the delivery. Whether two results may be compared becomes
checkable.

**Fail-loud.** §2.3 refuses rather than publishing an unreliable map, and §2.4 is enforced by a test
rather than by discipline.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] An artifact can be identified from its own content: which quantity, which composition, which
      origin, which horizons.
- [ ] Two artifacts of one run become joinable, because both name the same origin.
- [ ] Horizon-indexed readouts become expressible, because horizons acquire a stated mapping.
- [ ] Several independent derivations of the horizon offset get one authority to read instead.
- [ ] Makes ADR-075 safe: a quantity in the forecast's units, written beside the forecast, stays
      distinguishable from it.

### ⚠️ Negative (Costs)

- [ ] One more artifact per run, which must be kept true. A manifest that is trusted and wrong is
      worse than none, which is why §5 tests it adversarially.
- [ ] Information that belongs with the rows is duplicated for now. Accepted rather than blocking on
      another repository's release; §2.5 names the route that removes it.
- [ ] The contiguity assertion can fail a long run late, which is preferred to a silently wrong
      horizon.
- [ ] A manifest invites being read in place of the code. It records what a run did; it does not
      describe what the code does.

---

## 5. Validation

**Invariants.**
- A run produces prediction arrays byte-identical to a run before this change, at the same seed. The
  manifest is additive.
- Nothing on the inference or feedback path changes.

**Tests.**
- *Green:* every manifest field agrees with the configuration and with the artifacts on disk, on a
  run with several origins.
- *Red:* a deliberately falsified manifest field fails the test. A manifest is worth only as much as
  the test that can catch it lying.
- *Red:* altering the resolved configuration object after it is built changes the manifest. This is
  the only test that separates a manifest recording what happened from one re-reading what was
  asked, and one that re-read would agree with the configuration by construction.
- *Green:* an artifact's recorded origin joins its directory to that origin's first month.
- *Red:* non-contiguous months in an origin fail the run.
- *Beige:* a manifest from an earlier version is readable, or refused with a version message, and
  never silently misread.

**Not yet enforced.** None of these tests exists, and neither does the manifest.

**What would reopen this.** If the manifest and the prediction-frame identifiers ever disagreed
about a row's horizon or origin, this ADR would have created a second source of truth, which ADR-003
forbids. The duplication in §2.5 would then have to be resolved in favour of the identifiers.

---

## 6. Implementation Notes

- **Location.** One module owning the record and its writer, called from the inference path: one
  function and one data class, with no registry and no abstraction over artifact kinds. Validation
  belongs with the configuration contract (ADR-009).
- **Sequencing.** Before ADR-075 and ADR-078.
- **Noted, not addressed.** The inference module carries several diagnostics and is large, and the
  package holding it has a generic name. Recorded here, not solved here.
- **References.** ADR-003, ADR-008, ADR-009, ADR-032, ADR-047, ADR-075, ADR-078. Cross-repository:
  the prediction-frame metadata and identifier model, and the pipeline's prediction-output layout.
- **Evidence.** `reports/2026-11-01_body_output_blast_radius/00_REPORT.md` holds the measured basis
  for §1 — the absent fields, the two origin numberings, and the independent horizon derivations.
