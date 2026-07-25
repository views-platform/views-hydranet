# 02 — Design: the T>0 Rollout Skill ruler

**Status:** DRAFT (2026-07-25) — precedes an `expert-method-review`, then `preregister`. Graduates to a
proposed ADR on `promote`.

## 1. The question it must answer

For an autoregressive rollout from origin month `o`, at each horizon `h = 1..36` the model emits a
predictive distribution (D×K samples per cell) for month `o+h`. **Is that distribution an accurate forecast
of the realized `truth[o+h]` — and out to what horizon?** Decomposed into the two questions that matter:

- **Q1 (skill & depth):** does the deployed **free-running** rollout beat a climatology/persistence baseline
  at horizon h — and where does it cross below (the *usable rollout depth*)?
- **Q2 (bug vs ceiling):** is the rollout's decay caused by **exposure bias** (the bloom — a *fixable* bug)
  or by an **intrinsic predictability ceiling** (the features can't see further — *not* fixable by any
  feedback trick)? Answered by the **free-running − teacher-forced gap**.

## 2. The construction

For a fixed origin set `O` (G4) and the frozen truth parquet, for each target ∈ {sb, ns, os} and each
horizon `h = 1..36`, score three+ **rollout variants** and two **baselines** on identical (origin,cell)
support, with the frozen lodestar functions:

**Rollout variants** (same trained artifact; differ only in what is fed back):
- **free-running** — feed back the emitted prediction (today's deployed behavior; the bloom driver). *Data
  already on disk.*
- **teacher-forced-oracle** — feed back the realized `truth[o+t]` each step. Upper bound on rollout skill
  given the features; measures the predictability ceiling. *Needs G2 + a small re-run.*
- **(later) fix-variants** — sample-feedback (H-SAMPLE), τ-gated, GTF — each scored the same way, once the
  ruler exists. These are the *experiments*; the ruler is the *instrument*.

**Baselines** (per horizon, same support):
- **climatology** (white_ranger) — per-cell historical distribution; horizon-independent (flat in h); the
  hard-to-beat long-horizon reference.
- **persistence** — `truth[o]` held for all h; trivial, strong at short h.

**Metrics per (variant, target, h):** crps-all / crps-events / crps-none (magnitude), size-ratio
(magnitude calibration), AP + Brier (occurrence). Same definitions as the frozen ruler — indexed by h.

## 3. The two read-outs (what we actually plot / decide on)

1. **Skill-vs-horizon curve.** crps-all(h) for free-running vs climatology vs persistence, per target. The
   **crossover horizon h_x** where free-running rises above climatology = the usable rollout depth. (crps:
   lower = better; climatology is flat; free-running is expected to start below it and cross above as it
   decays.)
2. **Bloom-cost decomposition.** `gap(h) = crps_all_free(h) − crps_all_oracle(h)`.
   - **gap large & growing with h** ⇒ the decay is **exposure bias (the bloom) — a fixable bug**; a
     rollout fix that closes the gap is the win. This is the H-SAMPLE / GTF thesis.
   - **gap ≈ 0 while both curves rise toward climatology** ⇒ **intrinsic ceiling**; the model can't roll
     out usefully regardless of feedback, and the honest conclusion is "T=0 is the product; long-horizon is
     climatology." (Consistent with the amount-ceiling wall finding — magnitude predictability is
     structurally limited.)
   - The realistic answer is a **mix** — a crossover horizon below which the fix helps and above which the
     ceiling dominates. The ruler *locates* that horizon; that is its whole value.

## 4. Faithfulness & honesty guardrails (baked in)

- **h=1 == lodestar T=0** (byte-exact) — the loader sanity check.
- **Identical support across h** (G4) — the same (origin,cell) set at every horizon; N dropped is logged.
- **Occurrence threshold** (for the th_gated variants' AP) uses the pre-registered a-priori τ / base-rate,
  **never** derived from the scored months (the lodestar Goodhart rule).
- **Multi-seed for any KEEP claim.** The bloom_investigation blooms are s44-only; any skill conclusion that
  informs a decision must be ≥3 seeds (42/43/44). Single-seed reads are labeled INDICATIVE.
- **STABILITY≠SKILL restated at every readout** — a bounded curve that sits at/above climatology for all h
  is *stable but skill-less*; the ruler will say so plainly.

## 5. What this design deliberately is NOT

- Not a retraining harness. It scores existing/lightly-re-emitted rollouts. (GTF, which *does* retrain, is
  a later experiment scored *by* this ruler.)
- Not a new metric. It is the frozen lodestar, indexed by horizon.
- Not a rollout *fix*. It is the *instrument* that lets a fix be judged on skill. The fixes
  (sample-feedback, τ, GTF, spectral-norm) come after, each pre-registered against this ruler.

## 6. Open design questions for the method-review (flagged, not settled)

- **DQ1 — is per-horizon crps-all the right skill scalar,** or should the headline be a proper *skill
  score* (e.g. CRPSS = 1 − crps_model/crps_climatology) so "beats climatology" is a single sign-test per h?
  (Leaning: report both — crps-all for continuity with the lodestar, CRPSS for the crossover.)
- **DQ2 — origin-set size vs horizon reach.** A 36-month future window shrinks `O` to the earliest calib
  origins. Is `|O|` large enough for a stable per-cell crps mean at h=36, or do we accept widening CIs with
  h (and report them)? (Leaning: accept + plot bootstrap CIs; the crossover is robust even if h=36 is
  noisy.)
- **DQ3 — is teacher-forced-per-step the right ceiling,** or a weaker "feed back the emit-mean of a
  truth-conditioned step"? (Leaning: realized-truth feedback is the cleanest, most interpretable ceiling.)
- **DQ4 — climatology adequacy.** white_ranger is per-cell resample; is it the right long-horizon
  reference, or do we also need the mixture baseline (red/green/yellow_ranger) at horizon? (Leaning: add
  the mixture baseline; it is the stronger reference and already exists.)
