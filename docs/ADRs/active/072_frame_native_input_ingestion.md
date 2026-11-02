# ADR-072: Input data enters the model as a frame, not as a pandas DataFrame

**Status:** Accepted
**Date:** 2026-08-13 (rewritten and accepted 2026-11-02)
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers
**Builds on:** ADR-047 (the output path, already pandas-free), ADR-046 (feature lifecycle), ADR-019 (feature scaler), ADR-060 (static channels)
**Related:** ADR-076 (the migration bar). This ADR assumes the current data provider, which every model already declares

## Summary (read this first — self-contained)

Feature data currently enters the model as a **pandas DataFrame** and leaves as a numpy array. The
output half was made pandas-free some time ago; the input half was deliberately left for later.

This ADR is that later. Input arrives instead as a **frame** — a small numpy-only container of
values plus an integer index of months and grid cells — so pandas leaves the pipeline entirely.

It is a **refactor, not a behaviour change.** The guarantee is that the model receives a
**bit-for-bit identical** input array before and after, proven against the current path, and pandas
is not deleted until that proof passes.

Why bother, given pandas is declared honestly as a dependency and works: it is a large library
carried for one step of one path, it is pinned to a single version by an upstream constraint, and
every conversion between it and the numpy volume is a place where an index can be silently
reordered. The output path has already been through this and the input path is the remaining half.

---

## 1. Context

**The input path already separates into four jobs** — fetch the raw features, derive and scale them,
validate them, and pack them into the dense volume the model consumes. Only the *container* passed
between those jobs is pandas; the arithmetic in each is already numpy.

**The output path is the precedent.** It was converted on the same terms — a parity gate first, the
old path deleted only after — and that conversion is what established the volume as the
authoritative internal representation.

**pandas is pinned, not merely present.** An upstream constraint holds it at a single version, so
the dependency is not just weight: it is a version the project cannot move while anything needs it.

**What would make this a mistake.** If the conversion cannot be shown to produce an identical
volume, the refactor is not a refactor. That is why the gate, not the deletion, is the decision.

---

## 2. Decision

**Statement.** *Input data enters as a frame rather than a pandas DataFrame, and pandas is removed
from the input path only once a parity gate proves the resulting volume is bit-for-bit identical.*

### 2.1 The container

The frame and its integer index become the authoritative input container. No DataFrame is built
before the volume on a training or evaluation run.

### 2.2 The gate decides, not the plan

The volume built from a frame must be **bit-for-bit identical** to the volume built from a DataFrame
on the same data. Until that holds, both paths exist and the pandas path is the one in use.

### 2.3 Two properties that are not the same, and must be checked separately

**Uniqueness** — no two rows describe the same month and cell — and **completeness** — every month
and cell the model needs is present — are different properties. The frame's integer index rules out
a *missing label on a row that exists*; it says nothing about a *row that is absent*. Each needs its
own check.

### 2.4 Scope

**In scope:** the container on the input path, and the gate that admits it.

**Out of scope:** the provider the data comes from (ADR-076); the output path (ADR-047, already
done); the diagnostic and audit code that uses pandas off the production path; and any
reorganisation of the modules involved, which is recorded in the dossier and not decided here.

---

## 3. Rationale & Integrity Impact

| alternative | why it loses |
|---|---|
| **keep pandas on the input path** | it is pinned to a single version by an upstream constraint, so it blocks movement as well as adding weight; and every DataFrame-to-numpy conversion is a place an index can be silently reordered. The output path already paid this cost down |
| **convert and delete in one step** | the whole value is the parity proof. Deleting first makes the proof unobtainable, because there is nothing to compare against |
| **accept approximate agreement** | a refactor that changes a number is not a refactor. Nothing downstream could distinguish the intended change from a defect |
| **use pandas with a faster backend** | addresses the weight and not the pinning, and leaves every conversion boundary in place |
| **build a frame at the boundary and convert to pandas inside** | keeps both dependencies and both conversion costs, and the conversions are the risk |
| **reorganise the modules first** | a larger change with its own risks, and it would make the parity diff unreadable. Recorded in the dossier as a separate question |

**Reproducibility.** The gate is the guarantee: a bit-for-bit identical volume means no result moves.

**Fail-loud.** §2.3 splits a check that currently reads as one property into the two it actually is,
so an absent row is detected rather than assumed away by a dtype.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] pandas leaves the pipeline, so the project is no longer held at one pinned version of it.
- [ ] Every DataFrame-to-numpy conversion on the input path disappears, and with it the chance of a
      silent index reordering.
- [ ] The input and output paths use the same representation, which the output path already proved.
- [ ] Uniqueness and completeness become separately checkable.

### ⚠️ Negative (Costs)

- [ ] Both paths exist until the gate passes, which is more code to maintain in the interim.
- [ ] The parity gate is the expensive part of the work, and it buys no new capability.
- [ ] Diagnostic and audit code keeps using pandas, so the dependency does not leave the repository
      — only the production input path.
- [ ] The derived channels and the scaler change their keying from column names to feature names,
      which is a contract change for anything that configures them.

---

## 5. Validation

**Invariants.**
- The volume built from a frame equals the volume built from a DataFrame, bit for bit, on the same
  data.
- No result moves. If one does, the refactor has failed rather than improved something.

**Tests.**
- *Green:* the parity gate above. ⚠️ **It does not exist yet, and cannot until the frame path does** —
  there is nothing to compare. It is the decision's precondition, not its consequence.
- *Green:* the existing derivation-parity, static-channel-seam and data-backed-static-channel tests
  already cover the transform step and must stay green unchanged.
- *Red:* a frame with a duplicate month-and-cell pair is refused.
- *Red:* a frame missing a month the model needs is refused — the check §2.3 separates out.

**What would reopen this.** If the parity gate cannot be made to pass, the conversion is not a
refactor and the decision lapses rather than being relaxed.

---

## 6. Implementation Notes

- **Status of the work.** The decision is taken: the input path goes pandas-free. **The work is
  not scheduled**, and nothing below is built. An accepted ADR states what the code owes, not what
  it currently does, and the parity gate in §2.2 is the precondition for *deleting* pandas rather
  than for taking the decision.
- **Location.** A new constructor on the volume builder, taking a frame; the fetch, transform and
  validate steps swap their container. The volume-write loop itself does not change.
- **Noted, not addressed.** The module layout, the size of the volume builder, and the one pandas
  idiom remaining in the assemble step are all recorded in the dossier.
- **Evidence.** `reports/2026-11-02_frame_native_ingestion_dossier/` holds the worked example, the
  code walkthrough, the reorganisation proposal and the self-audit — along with a correction list,
  because four implementation claims in those notes are false and eight of their line references
  have drifted. `reports/2026-11-02_adr_claim_verification/00_CLAIMS.md` §6 holds the sources.

---

## References

**This repository:** ADR-019 (feature scaler) · ADR-046 (feature lifecycle) · ADR-047 (the
pandas-free output path) · ADR-060 (static channels) · ADR-076 (the provider).

**Other repositories** — the contracts this ADR works within:

- [views-frames ADR-013 — metadata and identifier model](https://github.com/views-platform/views-frames/blob/main/docs/ADRs/013_metadata_and_identifier_model.md). The index contract the input container relies on.
- [views-pipeline-core ADR-015 — model-specific input querysets](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/015_model_specific_inputdata_querysets.md). Where the fetch this path consumes is declared.
