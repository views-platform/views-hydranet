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

**Rollout variants** (same trained artifact; differ only in what is fed back). *Revised per method-review
§6b — the on-disk mean-feedback rollout is NOT the deployed object:*
- **current mean-feedback** — feed back the emit-mean (today's on-disk behavior; the bloom driver). *Data
  already on disk (GPU-free).* Scored as a **diagnostic of current behavior**, honestly labeled — **NOT
  "deployed skill"** (a probabilistic model rolled out by feeding back its mean is broken by construction;
  Salinas2020).
- **ancestral (sample-feedback)** — feed back a per-cell family sample each step (the H-SAMPLE probe). **This
  is the true deployed object** and the arm the *skill verdict* is gated on. *Needs the sample-feedback
  re-run (small GPU).*
- **teacher-forced one-step-conditioned ceiling** — feed back the realized `truth[o+t]` each step. Upper
  bound on rollout skill *given the trained one-step map* (relabeled from "predictability ceiling" per §6b:
  if the one-step map is biased, the oracle inherits it). *Needs G2 + a small re-run.*
- **(later) fix-variants** — τ-gated, GTF — scored the same way once the ruler exists. Experiments, not the
  instrument.

**Baselines** (per horizon, same support):
- **climatology** (white_ranger) — per-cell historical distribution; horizon-independent (flat in h); the
  hard-to-beat long-horizon reference.
- **persistence** — `truth[o]` held for all h; trivial, strong at short h.

**Metrics per (variant, target, h):** the frozen-ruler set, indexed by h — **crps_all / crps_events /
crps_none** (the split is the Goodhart guard: on a 99.7%-zero DGP, crps_all alone lets a timid
conservative-zero rollout look skillful, so crps_all is NEVER the headline — it is read with the
events/none split), **size-ratio** (magnitude calibration), **AP + Brier** (occurrence), **MCR** and
**QS99** (the locked FAO-02 calibration/tail guardrails). **NO twCRPS, NO PIT, NO LogScore** — FAO-02
rejected them and the lab tested them negative; they return only if a fresh test re-earns them. Per-horizon
calibration is read via **MCR**, not PIT. CRPSS (= 1 − crps/crps_clim) is computed **only** for the
crossover visualization (horizon-comparability), never optimized on and never a decision metric — raw CRPS
+ the split drive decisions. (Method-review §6b chair ruling, 2026-07-25.)

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

## 6. Open design questions — RESOLVED by the method-review (`02b`, 2026-07-25)

- **DQ1 — skill scalar. RESOLVED:** headline = the **crps_all/events/none split** + locked **Brier/MCR/QS99**
  (no twCRPS/PIT — FAO-02 rejected, chair-ruled §6b); **CRPSS only for the crossover plot**, never a
  decision metric. (See §2 metrics.)
- **DQ2 — origin-set size. RESOLVED:** accept widening CIs with h, but compute them with a **block bootstrap
  over origins** (|O|≈12 with overlapping 36-month futures ⇒ severe temporal autocorrelation; iid-cell CIs
  are fiction). No significance/KEEP claim without the block bootstrap.
- **DQ3 — the ceiling. RESOLVED:** keep teacher-forced realized-truth feedback, but **relabel it
  "one-step-conditioned ceiling"** and interpret `gap(h)` as **input-exposure-bias ⊕ induced state-drift**
  (the ConvLSTM `h_t` diverges too; cite the inert `freeze_h` result as evidence the input path dominates,
  so the gap is still interpretable). Not pure exposure bias.
- **DQ4 — baselines. RESOLVED:** add the **mixture baseline** (red/green/yellow_ranger) alongside
  white_ranger + persistence — score against the *strongest* reference (Bracher2021 hub ethos), not a
  strawman. **Direct-multi-horizon is NOT a baseline** — parked as an architectural alternative that a large
  oracle gap would motivate (§6b; the gap already diagnoses recursion's error-accumulation cost).

## 7. Blocker — CLEARED (2026-07-25, C-217)

**Partition discipline: verified, no leakage.** `config_partitions.py`: calibration = train **(121, 456)**,
test **(457, 504)**. The 13 rolling origins (T=0 = 457–469) each roll 36 steps covering months 457–504 —
**entirely inside the held-out calibration test window**; the model trained only on ≤456. Input history
uses ≤456 (allowed); every scored horizon-truth (457–504) is held out. **No leakage.**
- **Remaining guard (carried into `05`):** assert the re-scored artifacts are **calibration**-partition
  trained (train 121–456), NOT validation-partition (train 121–504 — that WOULD have seen 457–504). A
  one-line check of each artifact's `config.json` partition.
- **Note (H≈335):** origin index for T=0=457 ≈ 456−121 = 335 (the biopsy's "origin 335") ⇒ the rollout
  digests ~335 months of history before decoding — context for the C-223 recursive-vs-direct cost (direct
  saves only the 35 extra decode passes ≈ ~10% of inference, not 36×).
