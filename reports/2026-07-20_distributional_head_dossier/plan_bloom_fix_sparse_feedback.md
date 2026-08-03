# Forward plan: the t=1…t=36 bloom — the sparse-feedback fix ladder

> **PROMOTED 2026-07-25 → `reports/2026-07-25_t0_rollout_skill_dossier/`.** Prior art. The fix ladder + the
> H-SAMPLE probe are absorbed there; per the ruler-first decision, the T>0 skill ruler is built BEFORE any
> fix is scored (a fix judged only on boundedness = the corrupted-probe trap). §NEXT (H-SAMPLE) becomes a
> Phase-4 experiment, scored on the new ruler.

**Status (updated 2026-07-25):** FIRST PASS RUN — rungs 1–2 probed inference-only. **Findings + full
epistemic caveats: `bloom_investigation.md` (read that first).** Short version: sparsity IS the lever
(confirmed); τ≥0.8 threshold feedback keeps the 36-step rollout bounded (3 seeds) AND beats the foundation
on T=0 crps-all — BUT **stability ≠ skill** (we don't score the rollout), the bloom comparisons are
single-seed, ZINB blooms too (learned π decalibrates), and the `feedback_clamp` rail was inert (unknown
cause). τ is now framed as a **conservative-point *tool*, not the solution.** Next real probe = §NEXT
(sample-feedback rollout).

_(Original brainstorm, recorded 2026-07-24, retained below for the ladder framing.)_

## The problem (the bloom, C-113)
At **T=0** the forecast is correct and sparse (matches the 99.7%-zero truth). Over the 36-month
autoregressive rollout it **blooms**: the eval biopsy (Stage 6, origin 335) shows *both* heads
decalibrate —
- the **gate (`pred_by_*`) saturates toward 1** — fires everywhere; even the *minimum* gate is ~0.28 by
  T=35 (no cell is confidently zero anymore);
- the **magnitude (`pred_lr_*`) fills the map** with a dense, noisy, static-like field (sb μ 0.01→0.91,
  ns →1.27).

The T=35 forecast is a continent-wide activation vs a mostly-empty truth.

## What's known (do NOT re-derive)
- Root = **C-113 autoregressive feedback / exposure bias**, NOT the loss or gate spec. The `freeze_h`
  ablation localized it to the **input→output map** (gain > 1 on its own fed-back prediction); the
  io-gain diagnostic showed the free-running map settles at an out-of-range attractor.
- `feedback_clamp_log1p` is a **safety rail, not a fix** (pins to the ceiling → F2). asinh ruled out.
- We score **T=0 only**, so the bloom has been parked; it becomes the top prize once T=0 plateaus.

## The core insight (why the distributional head is the enabler)
The bloom is driven by feeding back the **diffuse emit-mean `E[y]`** — a dense field that looks nothing
like the sparse (99.7%-zero) real data → input goes out-of-distribution → errors compound over 36 steps.
The fix is to feed back something **SPARSE / in-distribution**. Framed as "corrective *samples* not
mean" — but the load-bearing property is **sparsity, not stochasticity**. A point head can only emit the
diffuse mean; the **distributional/count head is what lets us feed back a sparse field**. So this fix is
**general across every count arm** (ZINB, gated_NB, th_gated_NB, gated_ZINBcore), NOT specific to any one
run. (Legacy point heads can't do it → out of scope.) Bonus: only the **magnitude (`lr_*`)** channel is
fed back — the gate is computed *from* the input, so a sparse fed-back magnitude keeps the gate
calibrated **for free** (no more saturation). One change fixes both heads.

## Compute framing (width vs length)
The full stochastic version = each of `S = D×K` samples feeds *itself* back → S independent
sample-**paths** → an ensemble-of-trajectories (proper Monte-Carlo / ancestral rollout). Its cost is
**WIDTH** (parallel batch → S hidden states + activations in VRAM), **not length** — the 36
autoregressive steps are unchanged. On a **starved box** width is the binding constraint (VRAM; cf. the
`assert_cube_fits` preflight + the 18 GB publish-OOM scars); if S doesn't fit you tile → width converts
back to length (time). **So the rich version is deferred until compute allows** — but it is the *same*
design with the width knob turned up, so it is latent, not a new idea.

## The ladder (escalate only as needed)
1. **[FRUGAL — do first] Sparse deterministic feedback.** Feed back a **`th_gated`-masked** magnitude
   (zero below τ, body above) instead of the diffuse mean. **Single trajectory, ~zero extra compute**
   vs today's mean rollout; **inference-only (no retrain)**. Attacks the density-bloom (gate saturation)
   directly. Reuses the `th_gated` composition + the `feedback_clamp_log1p` rail we already have.
2. **[cheap tune] Diagnose the residual after #1.** If only the **magnitude creeps on retained cells**
   (μ still climbs) → tighten τ with horizon and/or engage `feedback_clamp_log1p` (cheap). Only escalate
   if it **still blooms structurally**.
3. **[train match inference] Scheduled sampling → pushforward / GTF.** Teach the model to *recover from
   its own sparse feedback* — the training-time **completion of #1** (they stack). Cheapest rung —
   **scheduled sampling (`ss_epsilon`, ADR-056) — is already partially wired.** Requires retraining.
4. **[LAST — on RISK] Bound the input→output gain (spectral-norm / Lipschitz).** Make the free-running
   map a contraction. Last resort because a Lipschitz cap **limits expressivity — the very capacity we
   spent earning at T=0** — so it can *cost* T=0 quality. The heavy hammer.

**Read as:** `sparse-feedback (frugal) → τ/clamp tune → GTF (stacks on #1) → spectral-norm (last, on risk)`.

## Deferred (latent in #1, not a separate project)
The full **S sample-path rollout** — rich uncertainty propagation through all 36 steps — is unlocked
later by compute budget or system optimization. Same lever, width turned up.

## Cross-refs
C-113 (the bloom / autoregressive feedback); `feedback_clamp_log1p` (the rail); ADR-056 (scheduled
sampling); the `th_gated` / `gated_ZINBcore` arms (ADR-068 + the `05_analysis_plan` PRE-DATA amendment);
the io-gain / `freeze_h` diagnostics; `observation_flat_loss_moving_internals` (the μ-still-climbing
companion — same μ that would feed the retained-cell creep in rung #2).

---

## §NEXT — the scoped experiment: sample-feedback (ancestral) rollout

**Motivation (from the 2026-07-25 discussion + the ZINB result).** Every *learned* gate (classifier gate,
ZINB π) decalibrates out of teacher-forcing → blooms. Only a *hard* rule (τ) stayed stable, but τ collapses
the predictive distribution to one conservative point. A **sample** is *also* a hard, sharp, in-distribution
realization (a sparse count draw) — so it should resist decalibration like τ, but WITHOUT discarding the
distribution. This is the principled version of rung 1; τ is its frugal proxy.

**Hypothesis (H-SAMPLE).** Feeding back a *sample* (not the mean) keeps each trajectory sparse and
in-distribution → the gate stays calibrated → no runaway; and the spread across S sample-paths is the
honest predictive uncertainty (sharp per-path, diffuse in aggregate) — recovering the correct diffuse
*marginal* without ever feeding a blurry object back.

**The ONE change.** In the AR feedback (`hydranet_inference.py` ~442/86), feed back a per-cell **draw**
from the family (`family.sample`, composed / self-zeroed as appropriate) instead of `_emit_magnitude`'s
`compose_mean`. Everything downstream unchanged. Inference-only (no retrain) for the single-path version.

**Pre-registered readouts (measure the SAME per-step trajectory + more):**
1. **Runaway?** per-step magnitude + gate trajectory (as in `bloom_investigation.md`). H-SAMPLE predicts
   BOUNDED (like τ≥0.8) — but via honest sampling, not thresholding.
2. **Sharpness of a single path** at T=8/17/35: does one sample-path *look like* a plausible sparse
   conflict map (few active cells, realistic magnitudes), NOT a diffuse blob? (Biopsy render + active-cell
   count / spatial autocorrelation.)
3. **Diffuse in aggregate:** the *mean over S paths* should spread with horizon (honest marginal
   uncertainty) even though each path is sharp. Contrast: mean-feedback is diffuse because it's broken;
   sample-feedback-aggregate should be diffuse because it's *correct*.

**Pre-committed falsifiers.**
- F-S1: sample-feedback ALSO runs away (bounded fails) → the instability is intrinsic to the map (io-gain
  >1 even on in-distribution inputs), not the mean-feedback → escalate to rung 4 (bound the gain) or accept
  the model can't roll out.
- F-S2: single paths are NOT sharp (still blobby) → sampling didn't buy in-distribution-ness → diagnose.
- F-S3: determinism/seed handling breaks the S2 #121 gate → fix before trusting.

**Hard caveat carried from `bloom_investigation.md` §5.1 — STABILITY ≠ SKILL.** Even if H-SAMPLE holds, we
still have **no T>0 scoring against truth**. Bounded, sharp, in-distribution paths are necessary but do not
prove the rollout is *accurate*. A companion question (bigger, separate): build a T>0 scoring readout
(does a sample-path's occurrence/magnitude match the realized future?), acknowledging the deeper
features/world-model ceiling — the model only knows conflict-history, so long-horizon *skill* may be
capped regardless of rollout method.

**Compute.** Single-path = ~one eval (cheap, inference-only). The S-path ensemble = WIDTH (S hidden states
in VRAM); on a starved box, tile → time. Start single-path (decisive on the runaway question), scale S only
if H-SAMPLE holds and we want the uncertainty spread.

**Deferred rung 3 (if sampling only partly helps):** scheduled sampling / GTF retrain — teach the model to
recover from its own sampled feedback (ADR-056, partially wired). Needs training.
