# 07 — Experiment log

Popperian record for this dossier. **Negatives are first-class**: an entry that records a failed parity gate
is worth more than one that records a pass, because the pass was the prediction. Every entry links to the
pre-registration clause it answers (05) and names which falsifier fired, or that none did.

Entry template: `id · date · prereg clause · the ONE variable · artifact location · readout · verdict vs
falsifiers · decision`.

---

## E0 — 2026-10-19 — verification of the dossier's own claims (not an experiment; the harness audit)

**Prereg clause:** none — this precedes 05's lock, and closing PF-9 is one of its outputs.
**The one variable:** nothing was changed. Read-only re-verification of 00/02/04/05 against the code as it
stands, two months after they were drafted by an agent who did not run them.
**Artifact:** `03_harness.md` §1, every claim with a `file:line`.

**Readout.** 8 load-bearing claims hold. **6 corrections to the dossier**, of which **5 are also new
against ADR-072** and are amended into it (ADR-072 Amendment 1), plus **one finding about the ADR itself**.

⛔ **Attribution first, because I nearly got it wrong.** ADR-072 is a much stronger document than the
dossier that was supposed to execute it. It *already* corrects the `get_feature_frame` naming (`:89-91`,
"Correction to an earlier draft") and *already* states the uniqueness obligation in full, citing
views-frames C-21 and calling it *"a real design obligation, not a free lunch"* (`:227-234`). Corrections 1
and 3 below are corrections to the **dossier**, contradicting its own ADR. I had them written up as
discoveries before checking — the third instance of that failure mode in this repo (`RESULTS.md §12`).

In descending order of consequence:

| # | correction | effect on the program |
|---|---|---|
| 2 | pipeline-core already ships the entire frame fetch half (`data_format: feature_frame` dispatch, `_get_cached_frame_path`, `load_frame_cache`) | **shrinks it** — the inbound change is a like-for-like disk read, not a cross-repo integration |
| 4 | `apply_blueprint`'s `by_*` columns never reach the volume; `_execute_derivations` (numpy) is already the only live derivation path | **shrinks it** — the derivation migration the dossier centred on is already done; the pandas half is deleted, not ported |
| 1 | `SpatioTemporalIndex` does **not** validate row-uniqueness (views-frames C-21) and `validate_values` does **not** reject NaN | **grows it** — gates ported, not inherited; 02's "free" claim struck. ⚠️ **Not new: ADR-072 `:227-234` already said so.** The dossier contradicted its own ADR |
| 5 | neither datafactory adapter supplies `row`/`col`; the frame path refuses to by contract, and `identity_cols` needs them as channel values as well as scatter indices | **grows it** — one new decoder, needed twice, with `dataloaders.py:537-542` as an exact oracle |
| 6 | the pandas path scales in **float64** and downcasts at the volume write; a frame-native scaler on float32 values would differ in the last bit | **changes the prereg** — `atol=0` is unreachable without a precision policy, so F1 would have fired for a non-container reason |
| 3 | the API names | ⚠️ **Not new: ADR-072 `:89-91` already corrected this.** What *is* new is *who calls* — hydranet calls neither; the manager does, upstream |
| 7 | **three of ADR-072's own citations no longer resolve** — `dataloaders.py:1112` (cited 3×) points past the end of a 1010-line file | the ADR is not reviewable by following its citations; re-pin before PF-1 |

**Verdict vs falsifiers:** 05's falsifiers were not yet live. But correction 6 is a **pre-emptive F1 false
positive** caught before it could stop the program with the wrong diagnosis attached, and correction 1 is a
C-303 instance (prose asserting a guard the code does not implement) in a document whose own ADR had the
guard right.

**Decision.** The plan is executable. The remaining blockers are **PF-1** (human ADR review) and **PF-3** (a
one-key views-models descriptor change ⇒ GitHub issue, per the standing no-commits-in-views-models rule).
Everything else in the pre-flight checklist is mechanical and mine. 05 is re-locked with the precision
constraint inside the intervention definition.

**Honest note on this entry.** None of the seven findings required running anything. All were available to a
reader of the code. The dossier shipped a plan whose central worry (derivation parity) was already solved,
whose central simplification (uniqueness for free) was false **and already contradicted by its own ADR**,
and which never noticed that the upstream half had been built. Two failure modes in one document: *never
describe an artifact you have not opened*, and *search the prior art in this repository before claiming a
discovery*. I reproduced the second one myself while writing this entry, and caught it only by opening
ADR-072 — which is why the attribution note is above the table and not in a footnote.

---

## Open slots (one per stage; a stage is not done until its entry is written)

| id | stage | prereg clause | state |
|---|---|---|---|
| **E1** | PF-0 — pandas declared in `pyproject.toml` | — (honesty fix, separable) | ☐ not run |
| **E2** | PF-6 — source dtype at the zarr boundary | 05 P1 precondition | ☐ not run — **a STOP if float64** |
| **E3** | PF-5 — the pandas path's NaN-fill count on the anchor | 05 P5 / §1.7 | ☐ not run |
| **E4** | PF-8 — `row_col_from_priogrid` vs the pipeline-core oracle | 05 P1 | ☐ not run |
| **E5** | S1 — golden anchor captured (volume + predictions, checksummed) | 05 method | ☐ not run |
| **E6** | S2 — red tests for the frame-native path, failing for the right reason | 05 P1/P5 | ☐ not run |
| **E7** | S3 — `from_feature_frame` implemented behind the flag | 05 intervention | ☐ not run |
| **E8** | **S4 — THE BYTE-IDENTICAL GATE** | 05 P1/P2/P3, F1/F2 | ☐ not run |
| **E9** | S5 — default flipped, hot-path pandas retired | 05 P4, F3 | ☐ not run |
| **E10** | S6 — import purity + smoke verify + close | 05 P4, F3 | ☐ not run |

Log integrity rule for this dossier: **no stage may be marked done in 00_README's checklist before its
entry exists here.** The dossier's first two months are the argument for that rule — four documents
describing a program, no record of a single thing executed.
