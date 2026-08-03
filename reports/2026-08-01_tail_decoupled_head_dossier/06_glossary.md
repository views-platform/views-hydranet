# 06 — Glossary (new terms only)

The locked `reports/GLOSSARY.md` governs shared vocabulary (gate, body, gated forecast, crps-all/
events/none, size-ratio, MCR, baseline). This file defines ONLY the terms this program introduces.

- **mixture-density NB head** — a per-cell 2-component negative-binomial mixture,
  `P(y)=w·NB(μ1,θ1)+(1−w)·NB(μ2,θ2)`, emitted as a single ADR-067 family (`output_distribution=
  "mixture_nb"`); the tail-decoupled head under test.
- **bulk component / tail (surge) component** — component 1 (`μ1`, smaller) / component 2 (`μ2`, larger);
  the mixing weight `w` is the bulk share.
- **ordered-means constraint** — `μ2 = μ1 + softplus(Δ)`, forcing `μ2 > μ1` by construction; the
  identifiability fix that pins component 2 as the heavier tail (removes label switching).
- **mean-decoupled tail** — a tail whose scale is not tied to the bulk mean (what a single NB cannot do,
  variance `μ+μ²/θ`); the property the mixture adds and the whole point of the test.
- **within-family vs real (the wall)** — the binary question: is the amount-ceiling magnitude wall an
  artifact of the NB *family's* mean-tied light tail (within-family — a richer head cracks it), or a
  genuine data limit (real — no head cracks it)?
- **ex-ante risk stratum** — a subset of cells/timesteps defined by a covariate known *before* the
  outcome (recent realized conflict intensity / persistence), used to condition the evaluation. Contrast
  with subset-on-outcome (improper).
- **stratified-proper CRPS** — the ordinary proper `crps_all` computed *within* an ex-ante stratum;
  proper because it partitions the sample space by a covariate, not by the outcome.
- **GW test (Giacomini–White conditional predictive ability)** — the pre-registered decision statistic;
  tests equal conditional predictive ability of two forecasting *methods*, valid for nested models
  (NB ⊂ mixture).
