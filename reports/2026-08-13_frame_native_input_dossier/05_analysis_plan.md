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
