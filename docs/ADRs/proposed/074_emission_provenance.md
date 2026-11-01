# ADR-074: Emission provenance — an emitted artifact must declare what it is

**Status:** Proposed
**Date:** 2026-11-01
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Builds on:** ADR-003 (authority of declarations over inference) · **Completes:** ADR-032 (authoritative output schema) · **Aligns with:** ADR-008 (observability and explicit failure), ADR-047 (pandas-free prediction output) · **Sibling:** ADR-075 (emission selection)

## Summary (read this first — self-contained)

When HydraNet finishes an inference run it leaves directories of `.npy` files on disk. Those files
carry **two** pieces of information about themselves: a month id per row and a grid cell id per row.
Nothing else. Not which quantity they are (the gated forecast? the gate? the body?), not how that
quantity was composed, not which forecast horizon a row belongs to, not which origin produced it,
not which model or which config.

All of that currently lives in **directory names** and in the memory of whoever launched the run.

Three times in three months we have therefore analysed an artifact that was not the thing we
believed it to be: a cached pull scoped to Africa and the Middle East rather than global land; a
`lr_*` directory read as the raw body when under `threshold_gate` it is the **gated forecast**; and
a gate posterior sliced at the first horizon with no record that the other 35 were left behind.
In every case the bytes were fine. What was missing was any way to ask them what they were.

This ADR requires HydraNet to write **one emission manifest per inference run**, stating what it
produced, and to **name horizons and origins explicitly instead of letting consumers derive them**.
It does not change what is computed, what is emitted, or any number we have ever reported.

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

**Assumption that is no longer true.** ADR-032 §2.2 made target names deliberately
state-independent:

> **`lr_` (Linear/Identity):** Signifies ground truth count intent. **This name remains constant
> even if the underlying data is logged or transformed.**

That was the right call — it is the "Immutable Semantic Law", and it stops a name drifting every
time a transform changes. But ADR-032 then **never designated where the state lives**. The
assumption was that intent-only names were sufficient because a human would know the rest. With
rolling origins, three emit-time axes (ADR-067/068/069), two run-type directory shapes and
artifacts crossing machines on rented hardware, they are not.

⇒ `lr_sb_best` failing to tell you whether composition was applied is **ADR-032 working as
specified with nothing on the other side of it.** This ADR supplies the other side.

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
| **quantity** = a directory prefix | ADR-032 §2.2 | state-independent by law; see above |

**Urgency.** Weeks of analysis have been spent on artifacts whose identity was assumed. The next
experiment adds a third emitted quantity (ADR-075), which makes the prefix→quantity mapping
genuinely ambiguous rather than merely undeclared.

---

## 2. Decision

**Statement.** *We will require every HydraNet inference run to emit a manifest that declares what
it produced, and we will name horizons and origins explicitly rather than letting consumers derive
them, so that an artifact's identity is a fact on disk rather than an inference.*

### 2.1 The emission manifest

One file per inference run, written by HydraNet into the run's generated-data directory. It declares:

| group | contents |
|---|---|
| **quantities** | each quantity emitted by this run, and the directory prefix it was written under |
| **composition** | `forecast_composition`, `gate_threshold`, `emit_family_core`, `emit_distribution`, `evaluation_mode`, `aggregate_method` |
| **shape** | `n_posterior_samples` (D), `n_head_samples` (K), `S = D×K`, `time_steps`, grid `H×W`, `level` |
| **origins** | per emitted origin: the **sequence number** (its `origin_N` directory), the **volume index**, and its **first month** |
| **horizons** | per origin, the explicit horizon→month map |
| **identity** | model name, run type, artifact filename, artifact sha256, views-hydranet version, config digest, seeds |
| **rows** | expected rows per horizon and per origin |

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
- [ ] Completes ADR-032 — intent-only names become safe, because state now lives somewhere.

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
