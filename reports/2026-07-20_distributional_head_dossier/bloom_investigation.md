# Bloom investigation — the T>0 autoregressive runaway (C-113)

> **PROMOTED 2026-07-25 → `reports/2026-07-25_t0_rollout_skill_dossier/`.** The bloom is now its own epic.
> This doc is retained as prior art (the STABILITY≠SKILL confession + the τ measurements); new bloom work,
> the T>0 skill ruler, and the sample-feedback probe live in that dossier.

**Dates:** 2026-07-24 → 2026-07-25 · **Status:** ACTIVE, first pass done, NOT closed.
**Scope of every measurement below:** calibration partition, eval-only re-inference of already-trained
nb/zinb artifacts (NO retrain), stealth floor-config trap-restore (md5 verified before+after each run).
**What was measured:** per-step mean of the emitted `lr_sb` (magnitude, log1p space) and `by_sb` (raw gate
`P(y>0)`), averaged over the 13 calibration origins, at rollout steps T=0…35. This is a **trajectory-
stability diagnostic**, NOT a scored rollout (see the ⚠️ caveat in §5 — this is the most important
limitation in this whole note).

---

## 0. Epistemic summary — read this first

| Claim | Confidence | Basis |
|---|---|---|
| th_gated_NB @ τ≥0.8 beats the *foundation* on T=0 crps-all (sb, ns; ties os) | **High** | 3 seeds (42/43/44) |
| Higher τ monotonically lowers T=0 crps-all (via crps-none collapse) | **High** | 3 seeds |
| At τ≥0.8 the 36-step rollout stays bounded/flat (no runaway) | **High** | 3 seeds |
| Under soft_gate (dense mean feedback) the rollout runs away catastrophically | **Medium** | 1 seed (s44) |
| Under τ=0.5 the rollout still runs away | **Medium** | 1 seed (s44) |
| ZINB (self-zeroed) also blooms — its learned π decalibrates in rollout | **Medium** | 1 seed (s44) |
| `feedback_clamp_log1p` had ZERO effect (byte-identical with/without) | **Observed, cause UNKNOWN** | 1 seed (s44) |
| "The bloom is a symptom, not the disease; the fix is sample-feedback" | **Interpretation / hypothesis** | motivated, partly supported, NOT proven |
| The stable high-τ rollout is a *good* forecast | **UNKNOWN — not measured** | we do not score T>0 |

**One-line honest state:** we found a real, seed-stable *knob* (τ) that both improves the T=0 aggregate
score and keeps the long rollout numerically bounded — but we have **not** shown the bounded rollout is
*skilful*, several of the bloom comparisons are single-seed, one tool (the clamp) silently did nothing for
reasons we haven't diagnosed, and the framing that points to the "real" fix is a hypothesis we have **not
yet tested**.

---

## 1. What triggered this

Epic #183 made forecast composition a real config axis applied at emit time. We verified
(`hydranet_inference.py` ~442–450) that the **autoregressive feedback consumes the composed emit**
(`fb = t1_pred = _emit_magnitude(...)`). So `forecast_composition=threshold_gate` already feeds back a
*sparse* (thresholded) magnitude — i.e. the drafted "frugal fix" (sparse feedback) was implemented for
free. That let us probe the bloom with inference-only re-scores instead of new training.

## 2. The core measurements (nb seed 44, per-step, 13-origin mean)

Same trained model; **only the feedback composition/τ differs.** Magnitude is mean `lr_sb` in log1p space;
the count column is `expm1` of the T=35 value (so a per-cell average predicted death count).

| feedback setting | gate μ @T35 | magnitude log1p @T35 | ⇒ count/cell @T35 |
|---|---|---|---|
| soft_gate (`gate×mean`, dense) | 0.914 | 24.10 | **29,000,000,000** |
| threshold τ=0.5 | 0.301 | 8.78 | 6,486 |
| threshold τ=0.5 **+ clamp 7** | 0.301 | 8.78 | 6,486 *(identical — clamp inert)* |
| threshold τ=0.8 | 0.017 | 0.23 | **0.3** |
| threshold τ=0.9 | 0.008 | 0.07 | **0.1** |
| ZINB (self-zeroed, `(1−π)μ`) | — | 21.75 | **2,800,000,000** |

Rollout stability at τ=0.8 / τ=0.9 replicated on seeds 42 and 43 (count @T35 all ≤ 0.5). The bloom rows
(soft, τ=0.5, ZINB) are **seed 44 only** — not yet replicated.

**Reading:** sparsity is the lever. A *dense* feedback (soft_gate, or ZINB once π decalibrates) compounds
into an unbounded runaway; a *sufficiently sparse* feedback (τ≥0.8) stays flat for all 36 steps. τ=0.5 is
sparse enough to slow it but not to stop it.

## 3. T=0 cost of a tighter τ (the surprise — 3 seeds)

Tightening τ does **not** cost T=0 accuracy on the aggregate metric — it *improves* it. crps-all,
count-only, mean over cells:

| τ | s42 (sb/ns/os) | s43 | s44 | foundation |
|---|---|---|---|---|
| 0.5 | — | — | 0.141/0.081/0.031 | 0.137/0.083/0.028 |
| 0.8 | 0.134/0.078/0.028 | 0.131/0.079/0.028 | 0.136/0.081/0.028 | " |
| 0.9 | 0.132/0.079/0.027 | 0.130/0.079/0.027 | 0.134/0.079/0.028 | " |

**th_gated_NB @ τ≥0.8 beats the foundation on sb and ns on all three seeds, ties/beats on os.** This is
the first arm in the program to beat the foundation on the primary metric.

**Why — and the honest asterisk.** crps-all is dominated by the ~99.7% zero cells. A higher τ zeros the
lukewarm cells (mostly genuinely empty), collapsing crps-none (sb 0.5:~0.015 → 0.9:0.003). But it wins by
being **decisively conservative**: crps-events *creeps up* (sb 16.4→16.9 as some real events get zeroed)
and the occurrence signal drops (frac-of-samples AP 0.25→0.17). So the aggregate win is a **confident-zero
strategy** — better on the score that weights zeros, *not* better at sizing or flagging actual conflict.
This is Goodhart-adjacent and must be stated whenever the "beats foundation" number is quoted.

## 4. The ZINB probe (why it matters)

Hypothesis: ZINB's *learned* structural-zero π is a per-cell soft gate, so ZINB might stay sparse — and
thus bloom-resistant — for free, without any τ. **Falsified (s44):** ZINB blooms to ~2.8 billion/cell,
right alongside soft_gate. Its π **decalibrates over the rollout exactly like the classifier gate** — once
the fed-back input drifts out-of-distribution, π stops believing in zeros, mass leaks onto every cell, and
it compounds. Dynamically, ZINB is a mean-feedback arm. **Only τ — a *hard, external, fixed* rule — stayed
stable; every *learned* gate (classifier or π) decalibrated.**

## 5. ⚠️ What we do NOT know / open questions — the load-bearing section

1. **STABILITY ≠ SKILL (the biggest caveat).** We do **not** score the T>0 rollout against truth (the
   frozen ruler scores T=0 only; there is no T>0 scoring harness). "The bloom is fixed" here means *the
   trajectory statistics stay bounded* — NOT that the rollout is a good forecast. A τ=0.9 rollout is
   bounded partly *because it predicts almost nothing* (count/cell ~0.1). A model that confidently
   forecasts near-zero everywhere is perfectly stable and potentially useless. **We have not shown the
   stable rollout is accurate.** Any claim that τ "fixes the forecast" is unsupported; it fixes the
   *numerical runaway*, which is necessary but not sufficient.
2. **Single-seed bloom cases.** soft, τ=0.5, and ZINB bloom are s44 only. The *stability* at τ≥0.8 is
   3-seed; the *instability* comparisons are not yet replicated.
3. **The inert clamp — unexplained.** `feedback_clamp_log1p=[7,7,7]` produced a byte-identical trajectory
   to no clamp. The rail either isn't wired into the family eval rollout or isn't consumed. Cause not
   diagnosed. Registered as a risk (see §8). It does not change any conclusion (τ is the lever), but a
   safety rail that silently does nothing is a real latent hazard.
4. **io-gain: in- vs out-of-distribution.** We attribute the runaway to the input→output map having gain
   >1 on the OOD (dense mean) input. We have NOT confirmed the gain is ≤1 on in-distribution (sparse)
   inputs. If the model learned gain>1 even on realistic inputs, sample-feedback would only partially help.
5. **Sample-feedback is untested.** The proposed principled fix (§7) is a hypothesis; no measurement yet.
6. **Everything is calibration-partition, T=0-scored.** No validation partition (M3). The "beats
   foundation" result is in-sample by the program's own F3 standard until M3 runs.

## 6. Interpretation (LENS — not proven): the bloom as a symptom

*(This section is deliberately flagged as interpretation. It organizes the facts; it is not itself a
measured result.)*

The model is a ConvLSTM fed **only conflict-history features** and trained **one step ahead**. Its entire
inductive bias is "conflict persists and diffuses locally." Rolled out many steps on its own output, a
*correct* probabilistic model's **mean** map SHOULD be diffuse — the marginal over many futures is blurry
(cf. multi-step video prediction; the motivation behind JEPA-style world models). So *some* spreading is
honest predictive uncertainty, and picking a sharp map from a high τ is choosing one **conservative point
summary** of the posterior — useful, but a summary, not the distribution.

**However**, what we measured is *beyond* honest blur. Honest diffusion conserves mass (spreads a fixed
total); here the **total explodes** (~10⁹ over reality) and the gate saturates to 0.91 on ~99.7%-empty
terrain. That is decalibration + a dynamical instability, not calibrated uncertainty — a genuine bug
riding on top of the legitimate diffusion. The ZINB result supports the "symptom" reading: even the
model's own learned zero-structure collapses out of teacher-forcing, i.e. **the model does not know how to
propagate conflict beyond "outward," and its confidence decalibrates the moment it leaves the training
regime.** This suggests the deeper ceiling is **features + world-model** (the model has no covariates that
say *where* conflict actually goes), which no rollout trick fixes — but this is a hypothesis about the
ceiling, not something we measured here.

## 7. The τ tool — status and honest limits

- **Real and seed-stable:** τ≥0.8 gives a bounded rollout AND beats the foundation on T=0 crps-all.
- **It is a tool, not a solution.** It works by collapsing the predictive distribution to its conservative
  core (a hard, decalibration-proof cutoff). Costs: lower recall / higher crps-events; stability that may
  partly be under-prediction (see §5.1); calibration-only. Document it; it may be worth shipping as a
  "confident-core" map; do NOT present it as "the bloom is solved."

## 8. Next probe (scoped): sample-feedback (ancestral) rollout

**See `plan_bloom_fix_sparse_feedback.md` (updated) for the full scoping.** In one line: feed back a
**sample** (a sparse count draw from the family head) instead of `compose_mean` — a hard, sharp, in-
distribution realization that resists decalibration like τ does, but *without* collapsing the distribution.
Run S paths → the spread is the honest uncertainty. This is the "deferred rich version" of the plan; the
distributional head we built is what enables it. **Untested — this note does not pre-judge it.**

**Risk to register:** `feedback_clamp_log1p` inert on the family eval rollout (§5.3) — a C-113 safety rail
that silently no-ops; Tier 3 (no wrong output, but a guard that does nothing is a latent hazard + misleads
future rung-2 work).
