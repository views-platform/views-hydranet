# SCOPE — what this dossier does and does not do

## In scope
Measure **which statistic of the fed-back field rollout skill depends on**, by fingerprinting the generated
field (E1), degrading the real field one axis at a time (E2), testing realism against correctness (E3),
splitting the occurrence and magnitude channels (E4), and testing a coherent sampler (EXP-05). Emit-only on
a saved artifact.

## Explicitly out of scope

| # | Excluded | Why |
|---|---|---|
| 1 | **Building any fix** | This decides what a fix would have to target. Professor Forcing, K-step rollout training and distribution matching are all untouched. |
| 2 | Changing the **scored** product | The correlated sampler applies to the feedback path only; `to_cube_samples` keeps independent sampling, so an effect cannot be the ruler being handed a prettier cube. |
| 3 | Retraining | Every arm is emit-only on `calibration_model_20260814_003058.pt`. |
| 4 | The temporal axis | `shuffle_months` was voided (F6). Real conflict is geographically sticky over years, so month-permutation inside a 36-month window is inert **by construction** — a design limitation, recorded, not worked around. |
| 5 | `violet_visitor` confirmation | 160 lessons, 13 origins. **Not run.** Every claim is INDICATIVE. |
| 6 | Multi-seed | One seed. The ordering of axes is the result; the magnitudes are not calibrated. |

## Confounds stated before the run, not after

- **`spatial_scramble` cannot be clean.** Destroying clustering necessarily breaks the field's alignment with
  the static channels, because plausible locations *are* the clustering. Read it as "spatial structure **and
  its geographic grounding**".
- **`thin` / `inject` are not orthogonal.** Both perturb clustering as a side effect (0.447 → ~0.136).
  Part of their measured damage is likely clustering-mediated.
- **A torus roll was built and rejected** for the persistence axis: it preserves the statistics but the grid
  is a map of Africa, not a torus, so a rolled blob lands in another country while the coordinate channels
  stay fixed.

## The railguard that held
**Fix nothing the probe finds.** Everything built here is instrumentation. No model was retrained, no
remedy implemented, and the one intervention tested (the coherent sampler) was run as a *diagnostic* and
came back null.
