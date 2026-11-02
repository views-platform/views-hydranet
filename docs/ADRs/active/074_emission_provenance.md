# ADR-074: An emitted artifact must declare what it is

**Status:** Accepted
**Date:** 2026-11-01 (accepted 2026-11-02)
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers · views-frames maintainers
**Builds on:** ADR-003 (authority of declarations over inference), ADR-008 (observability and explicit failure)
**Leaves intact:** ADR-032 (authoritative output schema — counts in, counts out)
**Required by:** ADR-075 (emitting the body — the change that creates emitted intermediaries), ADR-078 (origin and horizon selection)

## Summary (read this first — self-contained)

When HydraNet finishes an inference run it leaves prediction arrays on disk. Each row carries a
month identifier and a grid-cell identifier. Nothing else travels with the bytes: not which
composition produced them, not which forecast horizon a row belongs to, not which origin, not which
model or configuration. All of that lives in directory names and in the memory of whoever launched
the run.

**What is not broken.** ADR-032's rule — the output is on the same scale as the input, *counts in,
counts out* — is intact and remains the rule. Whatever the model does internally is invisible
outside it, because a prediction is un-transformed before anyone sees it. Today's emitted forecast
is therefore correctly named and unambiguous as to what kind of thing it is.

**What is new.** We now want **intermediate quantities** out of the model as diagnostics — things
that are not the forecast. The **gate** is an occurrence probability, so its scale distinguishes it.
The **body** is in counts and is *still not the forecast*, so neither scale nor naming convention
separates the two. That hazard is created by ADR-075, and it is why this ADR exists.

**This ADR requires** one **emission manifest** per inference run, declaring what the run produced;
and it requires **origin** to be recorded rather than inferred, with the horizon-to-month map stated
explicitly. It changes nothing about what is computed or emitted, and no number we have reported.

---

## 1. Context

**The problem.** An emitted artifact cannot be identified from itself.

The prediction-frame index is, by design, a month identifier and a unit identifier per row. The
spatial level is not persisted either, and is supplied by the caller at load time. Everything else
is path convention: the target is a leaf directory name, the origin is a directory name, the run
type and a model-artifact timestamp are a parent directory name.

Absent from the bytes entirely: the composition settings, the posterior and head sample widths, the
seeds, the model name, the artifact digest, any configuration digest, the origin index, the origin's
first month, and the horizon.

The checkpoint sidecar does not close the gap. It records architecture and seeds, but **not** the
emit-time settings — deliberately, since the configuration layer documents them as read at
evaluation rather than baked into the artifact. One checkpoint is legitimately evaluated under
several compositions, so the composition belongs to the *emission*, not to the weights.

**What ADR-032 settled, and why it is not the gap.** ADR-032 makes target names state-independent:
a `lr_` prefix signifies count intent, and the name holds even if the underlying data is logged or
transformed. That is not an absence of state-tracking; it is an **invariant that makes
state-tracking unnecessary**, because the output is returned on the input's scale. The name
suffices precisely because the scale is guaranteed. That rule stands, and this ADR neither amends
nor completes it.

**What is genuinely new.** Intermediate quantities that are not the forecast:

| quantity | relation to the forecast | separable by scale? |
|---|---|---|
| the **gate** | a binary transformation — an occurrence probability | yes, it is not in count space |
| the **body** | not a transformation at all: counts, but not the forecast | **no** |

The body is the hazard. It shares the forecast's space and is not the forecast, so no existing law
separates them — ADR-032 governs forecasts and had no reason to speak about a quantity that is not
one. A new category needs its own declaration rather than an amendment to the law covering a
different category.

**The repository already forbids the consequence.** ADR-003 holds that all meaningful semantics
must be explicitly declared, that inference of semantics across component boundaries is forbidden,
and that where semantics are missing or ambiguous the system must not guess. Measured against that
rule, the emission boundary is in standing violation at three points:

| inferred, not declared | consequence |
|---|---|
| **horizon**, as the offset of a row's month from the earliest month present | assumes an origin's months are contiguous and that its first horizon is present. Three separate consumers re-derive this independently; no guard, assertion or test anywhere establishes either premise. A gap in an origin's months would make a given horizon a *different* month in different origins, with correct row counts throughout |
| **origin**, as a position in a sorted list | the directory name is a sequence number, not the origin index. The origin's own index and its first month appear nowhere in the prediction output. A separate diagnostic dump records the real index instead, so two artifacts of one run number their origins differently and cannot be joined without re-deriving the origin function |
| **composition**, nowhere at all | which composition produced a forecast is absent from the artifact, the checkpoint and the directory name. Narrower than a quantity ambiguity: the artifact *is* the forecast in counts, correctly named — what cannot be established is which composition made it |

**Not in violation today: quantity.** There is one emitted quantity, it is the forecast, and
ADR-032 names it correctly. Quantity becomes ambiguous only when ADR-075 puts non-forecast
intermediaries on disk beside it.

**Urgency.** Two reasons, and only the first is retrospective.

1. Composition, origin and horizon are already unrecorded, so analysis has been spent on artifacts
   whose provenance was assumed rather than read.
2. ADR-075 introduces emitted intermediaries. From that point the *kind* of thing an array is
   becomes ambiguous for the first time, and the body is indistinguishable from the forecast by
   both scale and naming convention. This ADR must therefore land **before** ADR-075.

---

## 2. Decision

**Statement.** *We will require every inference run to emit a manifest declaring what it produced,
and we will record origin explicitly rather than let consumers infer it — so that an artifact's
identity is a fact on disk rather than a reconstruction.*

### 2.1 The emission manifest

One file per inference run, written into the run's generated-data directory.

A configuration, and its commit, record what *should* have happened — **intent**. The manifest
records what *did* — **fact**. Both are needed, and they must be comparable, so that a divergence
between declaration and execution is detectable rather than invisible. A manifest that merely
restated the configuration would record intent twice and fact not at all.

| group | source | kind |
|---|---|---|
| **resolved configuration, in full** — including the composition fields, the sample-width fields, the step count, the grid extent, the spatial level and the seeds | the validated configuration object in memory | fact |
| **declared provenance** — a commit identifier for each repository involved, each with an explicit dirty-tree flag | version control, at launch | intent |
| **quantities** — each quantity emitted, and the directory prefix it was written under | the emission path | fact |
| **origins** — per emitted origin: its sequence number, its origin index, and its first month | the run | fact |
| **horizons** — per origin, the explicit horizon-to-month map | the run | fact |
| **rows** — expected rows per horizon and per origin | the run | fact |
| **identity** — model name, run type, artifact filename, artifact digest, package version | the artifacts | fact |

The resolved configuration is recorded **in full** rather than filtered to whichever fields seem
important today. It is small, so curating it would be a false economy, and a filter is a judgement
that will be wrong later.

The **dirty-tree flag** is load-bearing: a commit identifier taken from a modified working tree is a
false provenance claim, and the flag degrades it honestly to *approximately this revision, plus
unrecorded edits*.

### 2.2 Origin is recorded; horizon is derived from it

**Origin is the primitive.** Horizon is a function of an origin and a month, so recording both
invites the two to disagree. The need for an explicit horizon arises only with rolling origins —
and what is missing in that case is *which origin a row came from*. Once origin is known, horizon
follows.

So: **origin is recorded as a fact about the artifact.** For an artifact holding a single origin it
is a property of the whole artifact, not of each row, and belongs in a header rather than in a
per-row array. A per-row origin earns its place only where several origins are stacked together.

**The horizon-to-month map nevertheless remains in the manifest**, for a reason that is not
redundancy: the prediction-frame layer treats a month as an opaque integer and validates structure
only — never epoch, range or monotonicity. It cannot guarantee that a given horizon means the same
month in every origin, whatever identifiers it carries. The map is the only place that can state it.

### 2.3 Contiguity is asserted, not assumed

The run fails if an origin's months are not contiguous ascending. The current writer always produces
contiguous months, so this asserts a property already relied on and never checked.

A failed assertion fails the **run**, not the reader. It must not be possible to publish a manifest
this ADR cannot stand behind.

### 2.4 The manifest records fact, taken from the objects that produced it

The manifest's values come from the same objects the computation used. Not from re-reading
configuration files, not from re-deriving a value, not from recomputing anything.

A manifest assembled by reading the configuration would agree with it **by construction**, which
destroys the comparability §2.1 requires and makes the artifact decorative.

### 2.5 In-scope / out-of-scope

**In scope:** what HydraNet declares about its own emissions, and the validation that keeps those
declarations true.

**Out of scope:**
- **Transport.** What comes home from rented hardware, the archive selection and the run manifest
  there belong to the models repository.
- **Retrofitting existing artifacts.** Already-written output keeps its layout.
- **Changing any emitted number.** This ADR adds a description.
- **Where the emitted intermediaries come from.** ADR-075.

**Adjacent, and explicitly *not* out of reach.** The prediction-frame layer's own metadata model
anticipates exactly this extension: identifiers are a fixed, required month-and-unit pair, and any
future per-row axis — step, origin, scenario — is added as an **optional** identifier, as a minor
change, when a concrete consumer requires it. Its typed, all-optional header already carries
producer-generic fields, and new optional fields there are likewise minor. A forecast origin
plausibly qualifies as producer-generic; a composition and a gate-versus-body distinction do not,
being this model family's own vocabulary, and stay in this manifest.

So the correct route for origin is **a request to that repository**, filed as an issue — not a
workaround recorded here as somebody else's problem. Worth noting alongside it: that layer's own
serialisation already writes a header carrying the spatial level and producer metadata, and the
pipeline writes its own layout instead, which is why level and provenance vanish on disk. Using the
layer's serialisation, or carrying its header, would deliver part of this ADR with no change
anywhere.

---

## 3. Rationale & Integrity Impact

### 3.1 Why not simply a commit identifier

A commit identifier is **corroboration, not identity**:

1. **It requires an external lookup** — the right repository, at a revision that may have been
   rebased or force-pushed. A model's configuration and the inference code live in *different*
   repositories, so "the commit" is ambiguous without saying which. That makes the artifact's
   identity depend on something outside the artifact, which is this ADR's own defect displaced by
   one step.
   **The test:** can a reader establish which quantity an artifact is, and under what composition,
   holding only the delivered artifact, offline? With a resolved-configuration snapshot, yes. With
   a commit identifier alone, no.
2. **One commit per run is not enforceable.** Usual, but not guaranteed — so the claim is
   probabilistic where the rest of the manifest is not.
3. **It carries no execution facts.** Which origins ran, which months each covered, and whether
   contiguity held are properties of execution, present in no configuration at any revision.

### 3.2 Why not the configuration files

A model's configuration is **executable code**, not data: it computes, branches, reads the
environment, and the schema supplies defaults for whatever it omits. Recording the files records
*the program, not its output* — and the run consumed the output. Hence the resolved object.

### 3.3 Why not path conventions and a reference document

This is the status quo, and it produced the three misidentified artifacts in §1. A documented naming
table is not a property of the bytes, and it must be kept in step by hand.

### 3.4 Why not the checkpoint sidecar

The checkpoint is written at training time, and one checkpoint is legitimately evaluated under
several compositions. Identity belongs to the emission, not to the weights.

### 3.5 Integrity

**Reproducibility.** A result becomes traceable to a composition, a sample width, a seed and a
checkpoint digest without consulting anything outside the delivery. Comparability becomes checkable
rather than remembered.

**Numerical stability.** Untouched. Nothing in the computation path changes.

**Fail-loud.** §2.3 and §2.4 are both refusals: the contiguity assertion fails the run rather than
emitting an unverifiable manifest, and the fact-not-intent constraint is enforced by a test rather
than by care.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] An artifact can be identified from its own bytes — which quantity, which composition, which
      origin, which horizons.
- [ ] Two artifacts of one run become joinable, closing the dual origin-numbering defect.
- [ ] Horizon-indexed readouts become expressible, because horizons acquire a stated mapping.
- [ ] Three independent re-derivations of the horizon offset get one authority to read instead.
- [ ] Makes ADR-075 safe to land: ADR-032 keeps the *forecast* unambiguous, and this keeps an
      emitted *intermediary* from being mistaken for one.

### ⚠️ Negative (Costs)

- [ ] One more artifact per run, and one more thing that must be kept true. A manifest that is
      trusted and wrong is worse than none — addressed by §5's falsification tests, not by care.
- [ ] Duplicates, for now, information that belongs with the rows. Accepted as the cost of not
      blocking on another repository's release cycle; §2.5 names the route that removes it.
- [ ] The contiguity assertion can fail a long run late. Preferred over a silently wrong horizon.
- [ ] A manifest invites being read instead of the code. It records what a run did; it is not a
      substitute for reading what the code does.

---

## 5. Validation

**Invariants that must remain bit-perfect.**
- A run whose configuration sets none of ADR-075's fields produces prediction arrays
  byte-identical to a pre-ADR baseline at the same seed. The manifest is purely additive.
- Nothing in the inference or feedback path is touched.

**Tests.**
- *Green:* every manifest field agrees with the configuration and with the artifacts on disk, on a
  real multi-origin run.
- *Red:* a **deliberately falsified manifest field fails the test.** A manifest is worth only as
  much as the test that can catch it lying.
- *Red:* **mutate the resolved configuration object after it is built, and the manifest must follow
  the object, not the file on disk.** This is the only test that distinguishes a manifest recording
  fact from one re-reading intent — and one that silently re-read would agree with the
  configuration by construction, passing every other check.
- *Green:* the origin record joins an artifact's directory to its origin index and first month.
- *Red:* non-contiguous months in an origin fail the run.
- *Beige:* a manifest written by an older version is readable, or refused with a version message —
  never silently misread.

**Honest gap.** None of these tests exists yet, and neither does the manifest. This ADR is
currently **intent with no enforcement** — the condition that let a predecessor's claim go stale
unnoticed. Stated rather than implied.

**Failure mode that would trigger reconsideration.** If the manifest and the prediction-frame
identifiers ever disagree about a row's horizon or origin, this ADR has created a second source of
truth, which is what ADR-003 forbids. That observation means the duplication in §2.5 must be
resolved in favour of the identifiers rather than papered over.

---

## 6. Implementation Notes

- **Location.** One module owning the record and its writer, called from the inference path. One
  function and one data class: no registry, no writer hierarchy, no abstraction over artifact kinds.
  Validation lives with the rest of the configuration contract (ADR-009).
- **Sequencing.** Before ADR-075, which is what makes emitted intermediaries exist.
- **Noted, not fixed.** The inference module already carries several diagnostics and is large, and
  the package it sits in has a generic name. The boundary problem is real and is recorded here
  rather than solved here.
- **References.** ADR-003, ADR-008, ADR-009, ADR-032, ADR-047, ADR-067, ADR-069, ADR-075.
  Cross-repository: the prediction-frame metadata and identifier model, and the pipeline's
  prediction-output layout.
- **Evidence.** The six-repository blast-radius investigation under `reports/` holds the measured
  basis for §1 — the absent fields, the dual origin numbering, the three independent horizon
  re-derivations, and the recoverability arithmetic behind ADR-075's case for emitting the body.
  Numbers live there, not here.
