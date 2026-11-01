# ADR-074: Emission provenance — an emitted artifact must declare what it is

**Status:** Proposed
**Date:** 2026-11-01
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Builds on:** ADR-003 (authority of declarations over inference) · **Leaves intact:** ADR-032 (authoritative output schema — counts in, counts out; this ADR governs a category ADR-032 does not cover) · **Aligns with:** ADR-008 (observability and explicit failure), ADR-047 (pandas-free prediction output) · **Required by:** ADR-075 (emission selection — the change that creates emitted intermediaries)

## Summary (read this first — self-contained)

When HydraNet finishes an inference run it leaves directories of `.npy` files on disk. Those files
carry **two** pieces of information about themselves: a month id per row and a grid cell id per row.
Nothing else. Not which composition produced them, not which forecast horizon a row belongs to, not
which origin, not which model or which config. All of that lives in **directory names** and in the
memory of whoever launched the run.

⭐ **To be clear about what is *not* broken.** ADR-032's rule — the output is on the same scale as
the input, **counts in, counts out** — is intact and remains the rule. Whatever the model does
internally is invisible outside it, because a prediction is un-transformed before anyone sees it.
So today's emitted forecast is correctly named and unambiguous as to *what kind of thing it is*.

**What changes is that we now want intermediate quantities out of the model as diagnostics** —
things that are not the forecast. The **gate** is an occurrence probability, so its scale
distinguishes it. The **body** is in counts and is **still not the forecast** — so nothing, neither
scale nor naming convention, separates the two. That is a new hazard, created by ADR-075, and it is
why this ADR exists.

Alongside that, provenance gaps that are already costing us: three times in three months we
analysed an artifact that was not what we believed. A cached pull scoped to Africa and the Middle
East rather than global land. A gate posterior sliced at the first horizon with no record that the
other 35 were left behind. And — the instructive one — a `lr_*` directory read as the raw body,
which was *my* error importing a distinction the emitted output does not yet make, not a failure of
ADR-032. In every case the bytes were fine. What was missing was any way to ask them what they were.

This ADR requires HydraNet to write **one emission manifest per inference run**, stating what it
produced, and to **record origin and the horizon-to-month map explicitly instead of letting
consumers derive them**. It does not change what is computed, what is emitted, or any number we
have ever reported.

## 1. Context

**Problem.** No artifact HydraNet writes can be identified after the fact.

`identifiers.npz` holds exactly two arrays. `views-frames/src/views_frames/index.py:143-145`:

```python
def identifiers(self) -> dict[str, IntArray]:
    """The integer identifier arrays, keyed by name."""
    return {"time": self._time, "unit": self._unit}
```

written verbatim by `views-pipeline-core/views_pipeline_core/managers/prediction/prediction_frame_io.py:26-27`.
That module says outright at `:5-11` that even the spatial `level` is not persisted and must be
supplied by the caller.

Everything else is path convention: the target is a leaf directory name, the origin is
`origin_{i}`, the run type and a 15-character model-artifact timestamp are a parent directory name
(`model.py:1457`, `:1459-1465`).

**Absent from disk entirely:** `forecast_composition`, `gate_threshold`, `emit_family_core`,
`emit_distribution`, `evaluation_mode`, `n_posterior_samples`, `n_head_samples`, the seeds, the
model name, the artifact sha256, any config digest, the origin index, the origin's first month, and
the horizon.

And the model's own sidecar does not close it. `train/train_model.py:138-144` writes
`{...}.pt.config.json` with architecture and seeds but **no emit-time settings** — by design:
`config_initializer.py:238-239` records that `emit_family_core` is *"Emit-time only (read at eval,
not baked in the artifact)"*. So not even the checkpoint can tell you the composition of a forecast
made from it.

**What ADR-032 settled, and why it is not the gap.** ADR-032 §2.2 makes target names
state-independent:

> **`lr_` (Linear/Identity):** Signifies ground truth count intent. **This name remains constant
> even if the underlying data is logged or transformed.**

⭐ **That is not an absence of state-tracking. It is an invariant that makes state-tracking
unnecessary:** the output is on the same scale as the input. For conflict fatalities, **counts in,
counts out**. Whatever the model does internally — log transforms, binarisation, normalisation,
standardisation — is invisible outside the model, because a prediction is un-transformed before it
reaches a consumer. The name suffices precisely *because* the scale is guaranteed.

⇒ **That rule is intact and remains the rule.** This ADR does not amend it, complete it, or fill a
hole in it. There was no hole.

**What is actually new.** We now want to retrieve **intermediate quantities** from the model as
diagnostics — things that are *not* the forecast:

| quantity | relation to the forecast | distinguishable by scale? |
|---|---|---|
| the **gate** | a binary transformation — an occurrence probability, not counts | ✅ yes, it is not in count space |
| the **body** | not a transformation at all. It is in counts, and it is **still not the forecast** | ⛔ **no** |

⛔ **The body is the hazard, and it is a new one.** It lives in the same space as the forecast and
is not the forecast, so neither scale nor ADR-032's naming law separates them — ADR-032 governs
*the forecast*, and has no reason to say anything about a quantity that is not one.

⇒ So the requirement this ADR answers is **created by ADR-075**, which is what makes
non-forecast intermediaries exist on disk at all. ADR-032 governs the emitted forecast; **nothing
governs an emitted intermediary, because until now there were none.** A new category needs its own
declaration, not an amendment to the law covering a different category.

⚠️ **And this corrects the account of my own error.** I read a `lr_*` directory as the raw body.
Under ADR-032 it is the **forecast, in counts** — correctly named, exactly as specified. My mistake
was importing a body-versus-forecast distinction that *does not yet exist* in the emitted output,
and ADR-032 is not at fault for failing to encode a distinction it was never asked to make. What
*was* genuinely missing is narrower: **which composition produced that forecast** — a legitimate
ADR-069 axis that changes a forecast's value without changing what kind of thing it is.

**The repository already forbids the consequence.** ADR-003
(`docs/ADRs/templates/003_authority_of_declarations_over_inferance.md`):

> All meaningful semantics must be explicitly declared. Inference of semantics across component
> boundaries is forbidden. … If required semantics are missing, ambiguous, or contradictory, the
> system **must not guess**.

Measured against that rule, the current emission boundary is in standing violation at three points:

| inferred, not declared | where | consequence |
|---|---|---|
| **horizon** = `time − min(time) + 1` | re-derived independently at `reports/2026-07-25_t0_rollout_skill_dossier/tools/rollout_skill_score.py:60-64`, `scripts/proper_score_audit.py:119-120`, and `views-models/tools/podrun/slice_gate_posterior.py:113` | assumes contiguous months **and** that h=1 is present. No guard, assertion or test anywhere establishes either. |
| **origin** = position in a sorted list | `inference_orchestrator.py:308` `for i, origin in enumerate(origins)` → `origin_{i}` | `origin_0` is the *earliest* origin, not origin index 0. The real volume index and its month are nowhere in the prediction dirs. The body-mean dump records the **real** index instead (`hydranet_inference.py:1014,1019`), so two artifacts from one run number origins differently and are **not joinable** without re-deriving `get_rolling_origin_indices` (`utils/utils_orchestration.py:7-41`). |
| **composition** = not recorded anywhere | the emit-time axes of ADR-067/068/069 | which composition produced a forecast is absent from the artifact, from the checkpoint sidecar and from the directory name. ⚠️ Note this is **narrower than a quantity ambiguity**: the artifact *is* the forecast in counts, correctly named; what cannot be established is which composition made it |

✅ **Not in violation today: quantity.** There is one emitted quantity, it is the forecast, and
ADR-032 names it correctly. Quantity becomes ambiguous only when ADR-075 puts non-forecast
intermediaries on disk beside it — and the body, being in counts and not a forecast, is the case
that no existing law separates.

**Urgency.** Two distinct reasons, and only one of them is retrospective:
1. ⚠️ **Already costing us:** composition, origin and horizon are not recorded, so analysis has
   been spent on artifacts whose provenance was assumed rather than read.
2. ⛔ **Prospective, and larger:** ADR-075 introduces emitted intermediaries. From that point the
   *kind* of thing an array is becomes ambiguous for the first time, and the body is
   indistinguishable from the forecast by both scale and naming convention. ⇒ This ADR must land
   **before** ADR-075, not alongside it.

---

## 2. Decision

**Statement.** *We will require every HydraNet inference run to emit a manifest that declares what
it produced, and we will name horizons and origins explicitly rather than letting consumers derive
them, so that an artifact's identity is a fact on disk rather than an inference.*

### 2.1 The emission manifest

One file per inference run, written by HydraNet into the run's generated-data directory.

#### 2.1.1 ⭐ A config and a manifest are not the same claim

The intent-versus-fact split that governs this ADR's own authority (§0) governs the manifest's
contents too:

| | records | authority |
|---|---|---|
| **a config, and its commit** | what *should* have happened — **intent** | the config |
| **the manifest** | what *did* happen — **fact** | **the code, at the point of use** |

⇒ **Both** are required, and they must be **comparable**, so that a divergence between what was
declared and what ran is detectable rather than invisible. A manifest that merely restated the
config would record intent twice and fact not at all.

#### 2.1.2 Why it is not simply a commit identifier

A commit identifier is **corroboration, not identity**:

1. ⛔ **It requires an external lookup** — the right repository, at a revision that may have been
   rebased or force-pushed. A model's config and the inference code live in **different
   repositories**, so "the commit" is ambiguous without saying which. That makes the artifact's
   identity depend on something outside the artifact, which is this ADR's own defect displaced by
   one step.
   ⇒ **The test:** *can a reader answer "which quantity is this, under what composition?" holding
   only the delivered artifact, with no network and no repository?* With a resolved-config
   snapshot, yes. With a commit identifier alone, no.
2. ⚠️ **One commit per run is not enforceable.** Usual, but not guaranteed — so the claim is
   probabilistic where the rest of the manifest is not.
3. ⛔ **It cannot carry execution facts at all.** Which origins ran, which months each covered, and
   whether contiguity held are properties of **execution**, present in no config at any revision.

#### 2.1.3 Why it is not the config files either

A model's config is **executable code**, not data: it may compute, branch, and read the
environment, and the schema supplies defaults for whatever it omits. ⇒ Recording the config *files*
records **the program, not its output** — and the run consumed the output.

⇒ So the manifest records the **resolved, validated configuration object**: the same object the
inference path read. It is small enough that curating it would be a false economy, so it is
recorded **in full** rather than filtered to whichever fields seem important today.

#### 2.1.4 Contents

| group | source | kind |
|---|---|---|
| **resolved configuration, in full** — including the composition fields (`forecast_composition`, `gate_threshold`, `emit_family_core`, `emit_distribution`, `evaluation_mode`, `aggregate_method`), the shape fields (`n_posterior_samples`, `n_head_samples`, their product, `time_steps`, grid extent, `level`) and the seeds | the validated config object **in memory** | fact |
| **declared provenance** — a commit identifier for **each** repository involved, each with an explicit **dirty-tree flag** | version control, at launch | intent |
| **quantities** — each quantity emitted, and the directory prefix it was written under | the emission path | fact |
| **origins** — per emitted origin: its sequence number (the directory it occupies), its volume index, and its first month | the run | fact |
| **horizons** — per origin, the explicit horizon-to-month map | the run | fact |
| **rows** — expected rows per horizon and per origin | the run | fact |
| **identity** — model name, run type, artifact filename, artifact digest, package version | the artifacts | fact |

⭐ **The dirty-tree flag is load-bearing.** A commit identifier taken from a modified working tree
is a false provenance claim. The flag degrades it honestly to *"approximately this revision, plus
unrecorded edits"* — still useful, no longer a claim of provenance.

#### 2.1.5 ⛔ Implementation constraint, not a style preference

**The manifest's values are taken from the same objects the computation used.** Not by re-reading
configuration files, not by re-deriving a value, not by recomputing anything.

⇒ A manifest assembled by reading the configs would record intent a second time and fact zero
times, and would agree with the config **by construction** — destroying the comparability §2.1.1
requires and making the artifact decorative.

### 2.2 Horizons and origins are named, never derived

- The manifest's horizon→month map is **authoritative**. A consumer reads it rather than computing
  `time − min(time) + 1`.
- The origin table carries **both** numbering schemes, so any two artifacts from one run are
  joinable without reconstructing the origin function.

### 2.3 Contiguity is asserted, not assumed

The run **fails** if an origin's months are not contiguous ascending. The current writer always
produces contiguous months, so this asserts a property we already rely on and have never checked.

⇒ A failed assertion fails the **run**, not the reader. It must not be possible to write a manifest
this ADR cannot stand behind.

### 2.4 Named invariant — `steps` is a horizon COUNT, not a horizon SELECTOR

The `steps` config field's **values are never read** anywhere in views-hydranet. Only its length is:
`volume_sampler.py:60-66` (`slice_time(0, total_t - len(steps))`), `config_initializer.py:639-642`
(the checksum `time_steps == len(steps)`), and `reproducibility_gate.py:35` (recorded, never acted
on). There is no contiguity validator and no value check.

⇒ A config declaring `steps = [1, 6, 12]` passes every gate today, rolls out **3 consecutive
months**, and labels them as if they were horizons 1, 6 and 12.

**Decision:** `steps` must be contiguous ascending from 1, or the config raises. Selecting a sparse
set of horizons is `emit_horizons` (ADR-075), never a relabelling of `steps`.

⚠️ This is recorded here as a named invariant rather than as its own ADR because it is one
validator and a direct corollary of §2.2 — if horizon identity must be declared, a field that
*appears* to declare horizons but does not control them is the same defect. It is called out
explicitly so it is not later mistaken for an unrecorded decision.

✅ **Latent, not firing.** Every live config is contiguous from 1 (`violet_visitor`:
`list(range(1, 37))`; `heavy_freighter`, `purple_alien` likewise). **No past result is affected.**

### 2.5 In-scope / out-of-scope

**In scope:** what HydraNet writes about its own emissions, and the config validation that keeps
those statements true.

**Out of scope, deliberately:**
- **Extending `identifiers.npz`.** Horizon and origin belong with the rows, but that file is
  written by pipeline-core from views-frames' two-array contract — another repository. Filed as a
  views-models/pipeline-core issue. The manifest does not wait on it.
- **Transport.** What comes home from a pod, the archive selection and the run `MANIFEST` are
  views-models' (see ADR-075 §5 and the issues filed alongside it).
- **Retrofitting existing artifacts.** Already-written directories keep their current layout; the
  body-mean npz keeps its two-layout quirk (S8/#361 back-compat).
- **Changing any emitted number.** This ADR adds a description. It does not touch computation.

---

## 3. Rationale & Integrity Impact

**Logic — why this over the alternatives.**

| alternative | why rejected |
|---|---|
| keep path conventions + the run guide's prose table | this *is* the status quo, and it produced three misidentified artifacts. A markdown table a human must consult separately is not a property of the bytes. |
| put horizon/origin in `identifiers.npz` | correct destination, wrong repository. Blocking on a cross-repo contract change would leave the hole open meanwhile. Filed as an issue; the manifest is additive and compatible with it later. |
| bake emit-time settings into the `.pt` sidecar | the checkpoint is written at **train** time and one checkpoint is legitimately evaluated under several compositions. Identity belongs to the *emission*, not the weights. |
| derive identity at read time from a config digest lookup | requires a registry mapping digests to configs, and a reader that has access to it. The artifact still cannot answer for itself. |

**Fortress state.** Reproducibility improves directly: a result becomes traceable to a composition,
a sample width, a seed and a checkpoint hash without consulting anything outside the delivery.
Numerical stability is untouched — nothing in the computation path changes.

**Fail-loud.** §2.3 and §2.4 are both refusals. The contiguity assertion fails the run rather than
emitting an unverifiable manifest; the `steps` validator fails at config construction rather than
at analysis time three weeks later. This follows views-models' ADR-025 — *a check that cannot see
must refuse, not pass*.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] An artifact can be identified from its own bytes. The question *"is this the body or the
      gated forecast?"* becomes answerable without knowing which config ran.
- [ ] Two artifacts from one run become joinable (closes the dual origin-numbering defect).
- [ ] "All 36 horizons" becomes expressible, because horizons acquire names. This is the
      prerequisite for every horizon-indexed readout, including the h36 half of `P-23`.
- [ ] Three independent re-derivations of `time − min(time) + 1` get one authority to read instead.
- [ ] A latent config trap (`steps` as a false selector) is closed before it fires.
- [ ] Makes ADR-075 safe to land. ADR-032's counts-in-counts-out rule keeps the *forecast*
      unambiguous; this ADR is what keeps an emitted **intermediary** from being mistaken for one.

### ⚠️ Negative (Costs)

- [ ] One more file per run, and one more thing that must be kept true. **A manifest that is
      trusted and wrong is worse than no manifest** — mitigated by §5's deliberately-wrong-field
      test, not by care.
- [ ] A new refusal (`steps` contiguity) will reject a config shape that passes today. Latent, so
      no live config is affected, but a future sparse-`steps` config must move to `emit_horizons`.
- [ ] The contiguity assertion can fail a long run late. Preferred over a silent wrong horizon.
- [ ] Duplicates, for now, information that belongs in `identifiers.npz`. Accepted as the cost of
      not blocking on another repository.

---

## 5. Validation

**Invariants that must remain bit-perfect.**
- A run whose config sets none of ADR-075's new keys produces `lr_*` and `by_*` arrays
  **byte-identical** to a pre-ADR baseline at the same seed. The manifest is additive.
- Nothing in the inference or feedback path is touched: horizons h≥2 are bit-identical.

**Tests.**
- *Green:* the manifest's fields agree with the config and with the artifacts actually on disk,
  on a real multi-origin run.
- ⭐ *Red:* a **deliberately falsified manifest field fails the test.** A manifest is only worth
  as much as the test that can catch it lying; without this one the rest is decoration.
- ⭐ *Red, for §2.1.5:* **mutate the resolved config object after it is built, and the manifest
  must follow the object, not the file on disk.** This is the only test that can distinguish a
  manifest recording *fact* from one re-reading *intent* — and a manifest that passes every other
  check while silently re-reading configs would agree with them by construction, which is the
  failure §2.1.1 exists to prevent.
- *Green:* the origin table joins `origin_N` to the volume index and to that origin's first month.
- *Red:* non-contiguous months in an origin fail the run.
- *Red:* `steps=[1, 6, 12]` raises at config construction; `steps=list(range(1, 37))` does not.
- *Beige:* a manifest written by an older version is readable, or refused with a version message —
  never silently misread.

**Failure mode that would trigger reconsideration.** If the manifest and `identifiers.npz` ever
disagree about a row's horizon or origin, this ADR has created a second source of truth, which is
precisely what ADR-003 forbids. That observation means the duplication in §2.5 must be resolved in
favour of the identifiers, not papered over.

---

## 6. Implementation Notes

- **Location.** One module in views-hydranet owning the record and its writer, called from the
  inference path. One function, one dataclass: no registry, no writer hierarchy, no abstraction
  over artifact kinds. Validation lives with the rest of the config contract in
  `utils/config_initializer.py` (ADR-009).
- **Note, not fixed (out of scope).** `utils/hydranet_inference.py` already carries five
  diagnostics and is ~1800 lines, and `views_hydranet/utils/` is a generic package name. The
  boundary problem is real and is recorded here rather than solved here.
- **References.** ADR-003, ADR-008, ADR-009, ADR-032, ADR-047, ADR-067, ADR-069; views-models
  ADR-019 (delivery declaration — the transport-side vocabulary this manifest is designed to be
  selectable by) and ADR-025 (a check that cannot see must refuse).
- **Evidence.** `reports/2026-11-01_body_output_blast_radius/00_REPORT.md` — the six-repo
  blast-radius investigation that established the facts cited above.
