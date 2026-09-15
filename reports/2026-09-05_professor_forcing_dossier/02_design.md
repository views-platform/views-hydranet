# 02 — Design

**Written:** 2026-09-05, after `expert-method-review` (7 independent seats: Hochreiter, Shi,
Goodfellow, Gneiting, Hyndman, Sutton, operational). **Status: the panel ruled AGAINST building
Professor Forcing first.** This document records the rulings, the surviving dissent, and the
sequence the panel converged on. The F1–F6 rulings are kept because they become live again if PF is
ever reached.

## The headline: the target is right, the instrument is wrong

Every seat agreed the programme is now attacking the **correct quantity** — the recurrent state
rather than the fed-back field. Hyndman, the hostile seat, put it plainly: *"the diagnostic chain
M50 → M54 → M60 → M56 is genuinely good work, and it correctly identifies the state. The target is
right. The instrument is wrong."*

**Five of seven ruled: do not build the adversary first.** Three seats independently proposed the
same replacement without seeing each other.

## The disqualifying evidence, from PF's own paper

**C-265**: *"Professor Forcing shows no improvement on word-level Penn Treebank or on speech
sequences shorter than 100 steps, suggesting the benefit scales with the importance of long-term
dependencies."* Its §4.6 is titled **"Negative Results on Shorter Sequences."**

**Our horizon is 36.** Worse, the regime is inverted. PF's selling case (C-263) is *train short,
generate long* — 50-step segments → 1000-step generation. HydraNet trains on a **~383-step**
teacher-forced graph and generates **36**. We are train-long / generate-short: the regime the paper
reports no benefit in.

Cost was also understated. Lamb reports **3× teacher-forcing training time**; the dossier's
~4 h/arm extrapolated pushforward's ×1.76, which buys one extra step and **no discriminator**.
Realistic: ~10 h/arm.

## The alternative the panel converged on — GTF (#294)

`Hess2023_GeneralizedTeacherForcing`, **already in the library, never considered by this dossier**.
Proposed independently by **Sutton, Shi and Hyndman**.

`z̃ = (1−α)z + α·z̄` — interpolate the free-running state toward the teacher-forced state each step.
**C-499**: with `α = 1 − 1/σ_max` this **strictly bounds the Jacobian product series at arbitrary
horizon** — a theorem addressing exactly the divergence that killed #308. **C-502** (aGTF) derives α
per batch, removing any need for the Lyapunov exponent — the dependency that disqualified Horizon
Forcing.

Why it dominates PF here: same target (the state), one interpolation line, **no discriminator, no
second optimiser, no adversarial VOID branch, no degenerate equilibrium**. `σ_max ≈ 7.76` is already
measured (#294) ⇒ α ≈ 0.871.

**It is not scheduled sampling.** SS interpolates the *fed field* — the lever six failures died on
(M45). GTF interpolates the **state**, the axis nothing has touched. This distinction is the whole
argument and must survive into the pre-registration.

## Rulings on F1–F6

| # | ruling | basis |
|---|---|---|
| **F1** | **Discriminate `[hs_1..hs_4, tanh(hl_1)..tanh(hl_4)]` — bounded coordinates. NEVER raw `hl`.** | **The panel's sharpest disagreement; see below.** |
| **F2** | **Short: 4–7 steps.** Not 1 (forfeits the mechanism; C-265 says a 1-step PF is not PF), not 36 (that is #308). Shi: L=4, since M50's clamped arm is flat from step 6. Hochreiter: ~7, the cell's measured drain half-life (41× over 35 steps ⇒ 0.899/step ⇒ 6.5). **A config field (C-85), one pre-registered value — a sweep at n=1 is not an experiment.** | M50, C-458, M61 |
| **F3** | **`C_f` only. Unanimous.** `C_t` pulls the teacher-forced pass toward the known-degenerate free-running mode. M55 says that pass produces a model beating fair persistence 2.97× at h36 with the gap widening. There is no case for perturbing it. | M55, Lamb §2.2 |
| **F4** | **Spectral-normalised patch discriminator + dilated temporal conv head. NOT a bidirectional ConvRNN.** GroupNorm only — **never BatchNorm in D** (C-328 would gain a fifth instance). **Patch sampling is the decision the dossier omitted:** at 99.94% zeros a full-field D trains on background where the modes are near-identical and wins on global norm. Sample K=64 patches of 32×32, half within ±3 cells of a target event, half uniform, **identical coordinates for the paired TF/FR members**. | Miyato C-302, C-328 |
| **F5** | **The adversarial gradient reaches the RECURRENCE but MUST NOT reach the fed field.** These are two paths and `03_harness` §B conflated them — **the dossier's own line was wrong and reconstructed #308.** Keep: term → free-running `hl_t` → the LSTM weight matrices. Detach: term → the sampled feedback field → the emission head (M61/M62/M63). Lamb advertises avoiding that path as a feature (C-261). | **corrected by the panel** |
| **F6** | **Four VOID rules, pre-registered, measured every lesson on a fixed balanced probe.** V1 D never learned (median acc < 0.60 after L20). V2 D won (acc ≥ 0.99 in ≥30% of lessons). **V3 trivial win — a fixed logistic classifier on one scalar (log mean\|h\|) reaches ≥0.95, so the learned D added nothing.** V4 creep (rolling median pre-clip grad norm rises >10× in a 25-lesson window). Loss form: **least-squares (LSGAN, C-440)**, not the saturating variant; 1 D-step : 1 G-step on cached tensors. | Goodfellow; SeqGAN C-426; M61 |

## The live disagreement — F1, and it is the panel's most valuable output

**Hochreiter (discriminate the cell, raw `hl`):** `hs = o_t · tanh(hl)`. With `max|hl| = 65.6`,
`∂hs/∂hl = o_t · sech²(65.6) ≈ 4e-57`. A discriminator on `hs` cannot see the drain, and the
adversarial gradient routes entirely into `o_t` — **training the model to fake the readout while the
cell keeps draining.**

**Shi (discriminate `hs`, never raw `hl`) — and he found the code fact:**
`HydraBNrecurrentUnet_06_LSTM4.py:604` is `x = torch.cat([x, hs_1..hs_4], 1)`. **The U-Net encoder
sees only the hidden half.** The cell reaches the forecast exclusively through `hs`. So a raw-`hl`
discriminator separates instantly on scale — a quantity the emission path is nearly blind to, since
`tanh(65.6) = 1.0000` and `tanh(1.6) = 0.9217` — and the generator's cheapest fool is to rescale
`hl`, a near-no-op through `tanh` costing the NLL nothing. That yields **a PF arm that trains
cleanly, logs healthy discriminator accuracy inside any F6 band, and moves nothing** — an
interpretable-looking null that is really VOID and will not be called VOID.

**Resolution, and the measurement that settles it:** both warn of a *different cheap fake*, so the
answer is the coordinate system where neither is available — **bounded channels, `hs` and
`tanh(hl)`, none able to dominate on norm.** Verified against M50, which supports Shi on the
premise: the hidden half **does** drain 0.98 → 0.56 free-running (0.98 → 0.93 clamped), a 43% shift
in an unsaturated, emission-facing quantity.

**A correction to this dossier's own reasoning, from Shi:** `hs_j` is recomputed unconditionally
every step, and the roll instrument perturbs the state *before* `forward()`, so a rolled `hs` is
overwritten within the same step. **M60's "input 0/26 / hidden 0/26" therefore measures the
persistence of a perturbation, not causal importance for emission**, and #309's framing overstates
it. *(Verified, with a qualification: the rolled `hs` does still enter that step's gates via
`o_t`/`i_t`/`f_t`, so the effect is one-step and indirect rather than nil.)*

## The strongest objection — and the cheapest kill in the programme

**C-319 (Hochreiter, independently echoed by Shi).** M54 rolled the cell 90 cells: the forecast moved
90 cells **intact**, r ≈ 0.90, and skill collapsed **48×**. The ledger's own conclusion is that *a
displaced forecast is internally perfectly coherent, so no internal statistic can see it.*

**A PF discriminator is an internal statistic.** Distribution matching is **permutation-blind in
space**. So PF can achieve its own published success criterion in full — C-264's 40% divergence
reduction — while AP does not move, and we would be unable to distinguish *"PF engaged and placement
isn't what it fixes"* from *"PF never engaged."*

**The test costs no training.** Train the proposed discriminator offline on frozen state dumps from
the existing L=300 checkpoints, then feed it the **M54 rolled-cell dumps**. If it cannot separate
rolled from unrolled, the discriminator is provably blind to the failure we are trying to fix —
**and that kills the whole distribution-matching family, not just this configuration.** Run this
first.

## What the evaluation must change (Gneiting), independent of which method is built

1. **`AP@h18` stays the decision statistic** — truth-referenced, placement-sensitive, and six prior
   negatives are denominated in it. **But add an unnormalised *fair* CRPS co-primary** (`CRPS − ½E|X−X′|`,
   Ferro C-100) with equal `m`, plus its reliability–resolution–uncertainty decomposition
   (Hersbach C-13). The h36 artifact was not CRPS's fault: it was a **skill score** (improper even on
   a proper base rule, Bolin C-97), an **unfair ensemble** (finite-`m` bias), and a DGP whose
   answer is decomposition, not abandonment.
2. **`crps_events` is improper as a comparative criterion** — conditioning the evaluation set on
   `y>0` makes the score-optimal report `P(Y|Y>0)`, which *rewards inflation*. Demote to descriptive.
3. **The `act_ratio` prediction as written is a hedge and must be repaired before S2.** It has no
   threshold on the success side and no outcome that costs anything. Replace with a numeric
   two-sided equivalence band **and a stated abandonment condition**. Better still, bet on the
   **horizon shape** C-265 implies: the PF−control gap should be ≈0 at h1 and **increase
   monotonically with horizon**; flat-in-horizon falsifies the mechanism even if h18 is positive.
4. **A non-inferiority floor on `AP@h1` and the T=0 score, with numeric margins.** The adversarial
   gradient enters the shared recurrence; h18 can improve while h1 falls and nothing in the proposed
   readout would flag it. Breach = failure, not a trade-off argued after the numbers land.
5. **n=1 cannot detect this programme's own best effect.** MDE ≈ 3.52σ. Even on the optimistic
   cross-run σ ≈ 0.012 computed from existing artifacts, MDE ≈ 0.043 > the cell freeze's **+0.039**.
   **2 seeds × 2 arms, paired within seed** is the cheapest design that can conclude anything
   positive. **Recompute σ properly from artifacts already on disk before spending any GPU** — it
   costs nothing and may be the cheapest decision in the programme.

## The panel's sequence

1. **The C-319 blindness probe** — offline, no training, could kill the entire family for minutes of GPU.
2. **Recompute σ** from existing artifacts; re-derive the MDE.
3. **The emit-only state-restoration sweep** (norm restoration, leaky clamp, w-dial at L=300,
   periodic re-anchor) — answers M50's own stated open question, *"an intervention stopping the fade
   without clamping"*, at zero training cost, n=4 seeds.
4. **Productionise `freeze_recurrent='cell'`** — the operational seat's ruling. It is the programme's
   only replicated win (+0.0591 AP@h36, 4/4 seeds), needs no retraining, and **has never been used in
   a delivered forecast** (`hydranet_inference.py:178` marks it *"Diagnostic only"*; the production
   constructor never sets it). Disclose M58: the clamp compresses predicted-change dispersion
   (−0.46 at h18, −0.66 at h36) — better at *where*, flatter about *how much*.
5. **GTF (#294)** if a training-time state intervention is still wanted.
6. **PF last**, and on C-265 arguably not at all at a 36-step horizon.

## The dissent kept live

The operational seat, which ruled against building it now, added the strongest argument *for*
eventually running it: **the `act_ratio` prediction is a genuinely well-formed falsifier, and a
seventh instance of the firing failure would itself be a finding worth having** — evidence that the
firing lever is a property of the *data* (99.94% zeros) rather than of the feedback loop. Sutton
offers the sharper version: the direct multi-horizon head (#310) never feeds itself, so **if it too
loses AP by firing more, the feedback loop is exonerated** — and that is the more important finding.
