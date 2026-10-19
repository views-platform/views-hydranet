# 05 — Pre-analysis plan (parity pre-registration) — DRAFT (lock at S0 greenlight)

> This is the commit-before-you-look contract for the migration. It is **draft** until ADR-072 is accepted;
> at S0 it gets locked (no edits after the anchor is captured). The migration is a **refactor**, so the
> hypothesis is *equivalence*, not improvement — the only acceptable outcome is "identical behaviour, minus
> pandas."

## Hypothesis
Replacing the pandas input path (`fetch_df`→`pd.DataFrame`→`from_df`) with the frame-native path
(`get_feature_frame`→`FeatureFrame`→`from_feature_frame`) yields a **byte-identical model input volume
tensor** and **byte-identical predictions** on the v2 baseline, while removing pandas from the hot path.

## Intervention (the ONE variable)
The input container/path: `pd.DataFrame` → `FeatureFrame`. Everything else — data values, config, grid,
channel order, derivations, scaling semantics, model weights, seeds, output path — held fixed.

## Predictions (committed)
- **P1 — tensor parity:** `from_feature_frame(ff).volume` equals `from_df(df).volume` element-wise within a
  pre-registered `atol` (target: **exactly 0** — bit-identical; fall-back `atol=0` unless a specific
  reduction-order channel forces a tiny, named, justified tolerance).
- **P2 — derivation parity:** the `by_*` channels from `_execute_derivations` equal today's
  `apply_blueprint` output (existing `test_derivation_parity` stays green).
- **P3 — prediction parity:** for one frozen origin, end-to-end predictions match the pandas-path golden
  reference within model-forward float tolerance (deterministic seed).
- **P4 — import purity:** a training/eval run imports **no pandas** from the hot-path modules after S5.
- **P5 — loud failure:** every malformed `FeatureFrame` (missing month, dup `(time,unit)`, off-grid, NaN,
  wrong `feature_names`) raises a named exception at the handshake.

## Falsifiers (pre-committed — any one fires ⇒ STOP or fix-before-proceed)
- **F1 — parity break:** P1 fails and cannot be reconciled by matching reduction order ⇒ the container swap
  changes the data ⇒ **STOP**; the migration is not free. Do **not** widen `atol` to pass.
- **F2 — silent divergence:** a channel differs but no gate/test catches it (found only by eyeballing) ⇒ the
  parity harness is inadequate ⇒ strengthen the oracle before any flip.
- **F3 — hidden pandas:** after S5, an import-purity probe shows pandas still pulled on the hot path ⇒ the
  removal is incomplete ⇒ not done.
- **F4 — actuals-seam leak:** `prepare_actuals_df` turns out to be a pipeline-core contract we cannot change
  unilaterally ⇒ descope that seam (keep a pandas shim off the input path), do not force it.
- **F5 — scope creep:** the change is found entangling the provider swap (ADR-071) or the
  column-name-as-role coupling (C-173/C-174) ⇒ back out the entanglement; those are separate programs.

## Method
S1 anchor (freeze golden volume+preds) → S2 red tests → S3 build behind flag → **S4 byte-identical GATE** →
S5 flip default + retire pandas + CIC/pyproject/ADR updates → S6 import-purity + smoke verify. (Roadmap 04.)

## Decision rules
- **P1–P5 hold, F1–F5 quiet** → accept ADR-072; flip default; retire hot-path pandas; close.
- **F1 fires** → STOP; keep pandas; record why the swap isn't byte-identical (a real finding).
- **F4 fires** → ship the input-path removal, leave the actuals seam on a pandas shim, register it.
- **Honest close:** whatever ships, record that the OUTPUT side was already pandas-free (ADR-047) and this
  completes the INPUT side; note the residual audit-only pandas island and C-173/C-174 explicitly.

## Scope guards (non-negotiable)
- **Refactor, not experiment:** no scientific claim rides on this; success = *no behaviour change*.
- **After the provider swap, separate from it** (ADR-071 §S9 sequencing lock).
- **Never widen the parity tolerance to make the gate pass.**
- **Cross-repo IDs cited as `repo#id`** (lesson from the C-132/C-286 collision).

---

# Amendment 1 — 2026-10-19 — the precision policy, and three clauses that rested on false premises

Written before anything was executed, after the verification pass recorded in `03_harness.md §1` and
`07_experiment_log.md E0`. The body above is **unchanged**; this amendment adds to the intervention
definition and narrows three clauses. Nothing is relaxed.

## A1.1 — The intervention definition gains a precision clause (⛔ load-bearing)

The intervention above says "the input container/path … everything else held fixed." That was not
sufficient to specify, because the two containers carry **different dtypes**: the pandas path delivers
float64 (`ensure_float64`, `dataloaders.py:577`) and `FeatureFrame` is float32 by contract. The scaling
transform therefore runs in a different precision in each path, and `float32(log1p_f64(x))` ≠
`log1p_f32(float32(x))` in the last bit for some `x`.

**The intervention is amended to read:** the frame-native path **upcasts feature values to float64 before
the scaling transform and performs its only downcast at the volume scatter-write**, mirroring the pandas
path step for step.

This is not a tolerance. It is a specification of what "everything else held fixed" means, committed
before the anchor. Without it, **P1 fails and F1 fires on a precision artefact** — a false STOP with the
wrong diagnosis attached.

**New precondition, pre-committed as a STOP:** if the values are float64 at the zarr boundary and the
`FeatureFrame` coerces them to float32 before we receive them, information is lost upstream of us and
`atol=0` is structurally unreachable. Measure this at **PF-6, before S1**. If it fires, the honest outcome
is "the frame boundary is lossy for this source" — a finding, reported, not worked around.

## A1.2 — P5 no longer leans on the index (it never could)

P5 pre-registers a named exception for a duplicated `(time, unit)` row. The body above implied
`SpatioTemporalIndex` supplies that. **It does not** — views-frames register C-21: duplicates are
*allowed*, uniqueness is *not validated at construction*, and `has_unique_rows()` is opt-in and "not run by
default" (`index.py:25-38`, `:245-254`). `validate_values` likewise does not reject NaN or Inf
(`_validation.py:79-106`).

**P5 is amended:** the five malformed-input cases are gates **we implement at the handshake**, ported from
`DataSniffer._check_spatiotemporal_uniqueness` and `_check_non_finite`. P5 is now a test of our code, which
is what makes it falsifiable. The prediction is unchanged; only the mechanism it relies on is corrected.

## A1.3 — P2 is now a regression test, not a migration test

P2 pre-registers derivation parity as a thing the migration must achieve. Re-checked: the numpy
`_execute_derivations` is **already** the only derivation path feeding the volume — the `by_*` columns
`apply_blueprint` writes are not in `channel_map` and are never read into the tensor (`03_harness.md §1.4`).

**P2 is amended:** it predicts that `test_derivation_parity` stays green while `apply_blueprint` is
**deleted from the input path**. That is a weaker prediction than written, and saying so is the point —
the clause was claiming credit for work already done.

## A1.4 — F1's STOP is narrowed by one named exclusion

F1 says a P1 failure that "cannot be reconciled by matching reduction order" is a STOP. **The precision
policy in A1.1 is the only pre-approved reconciliation.** Any other reconciliation — reordering a
reduction, changing a dtype anywhere else, touching the scaler's semantics — is **not** covered and F1
stands. This closes the obvious Goodhart route: "reconcile" must not become a licence to tune until the
gate passes.

## A1.5 — What locks this plan

Locked when **PF-1** (human review of ADR-072) and **PF-3** (`data_format: feature_frame` on the target
descriptor) close. From the moment the S1 anchor is captured, no further amendment: a change after that
point is a new plan, not an amendment to this one.
