# 07 — Experiment log

---

## SCREEN-1 — BPTT-SA · **VOID: the experiment tested nothing** · 2026-09-03

**Pre-registration:** [`05_analysis_plan.md`](05_analysis_plan.md), locked `4424f26` before training.
Two arms, 300 lessons each, 276 min, 2/2 trained with no failures. Then scored free-running on the
standard 13-origin support.

### The result that could not be a result

| h | ε=0 baseline | A (wire cut) | B (wire on) | B − A |
|---|---|---|---|---|
| 1 | 0.4779 | 0.4502 | 0.4502 | **+0.0000** |
| 18 | 0.3298 | 0.3064 | 0.3064 | **+0.0000** |
| 36 | 0.2208 | 0.1908 | 0.1908 | **+0.0000** |

`crps_events`, `size_ratio` and `act_ratio` are identical too. **The trained weights hash
identically.** Two models trained with different flags cannot agree to the last bit — so this was
read as a defect, not as "Δ ≤ 0, H is dead", which is what the locked decision rule would otherwise
have said.

### Diagnosis: the change is INERT on the path production uses

The gradient through the feedback, measured directly:

| `ss_feedback` | `d(fed)/d(params)` |
|---|---|
| `"mean"` | **167.8** |
| `"sample"` | **exactly 0.0** |

A **draw** from the family is not reparameterised. It carries a `grad_fn` from the `log1p` wrapper —
so it *looks* connected — and delivers nothing. Removing `.detach()` from a tensor whose gradient is
already zero changes nothing, which is why the two arms are byte-identical.

And **C-259 requires `ss_feedback="sample"` whenever ε > 0.** The repo's own guard forces the single
mode in which the intervention cannot act.

### Why the tests passed anyway — the same error as C-323, one layer up

`tests/test_bptt_sa.py` asserted the wire was connected by checking `grad_fn is not None` on the
**point-head** path (`family=None`), where the fed value is the raw prediction and genuinely
differentiable. Production uses a **family head with sampled feedback**. *The property was verified
on a path production never takes* — which is precisely **C-323**, now recurring in the test suite
rather than in an ablation arm.

A `grad_fn` is not evidence that a wire carries anything. Both facts are now pinned by tests with
the measured numbers in them, so a future reparameterised or straight-through feedback flips them
and is noticed.

### What this does and does not tell us about H

**H is NOT refuted.** It was never tested. The pre-registered rule is not applied, because its
premise — that the two arms differ in the intended variable — is false.

**What is real:** both arms sit **~0.025 below** the ε=0 baseline at every horizon. Scheduled
sampling still hurts, on today's code, on this vehicle. Consistent with M26–M33. ⚠️ Not clean
evidence: the baseline was trained in August on different code, so part of that gap may be drift.

### What it would take to test H properly

The feedback has to be differentiable. Three routes, none free:

1. **`ss_feedback="mean"`** — differentiable today (167.8), but C-259 forbids it at ε>0, and for a
   stated reason: EXP-4 argued train exposure should match deploy exposure, which samples.
2. **Reparameterised draw** — NB is gamma–Poisson; `torch.distributions.Gamma` has `rsample`, the
   Poisson step does not. Needs a relaxation.
3. **Straight-through estimator** — sample forward, pass the mean's gradient backward.

Route 3 is the smallest and is what most of this literature does in practice. **None of this is
BPTT-SA-the-idea failing; it is BPTT-SA-my-implementation being a no-op.**

**Cost of the lesson: 276 min of training plus ~13 min of emit.**

---

## SCREEN-2 — BPTT-SA, straight-through · **ARM B DID NOT TRAIN** · 2026-09-03/04

**Pre-registration:** `05_analysis_plan.md` (`4424f26`) + amendment **A1** (`88fb223`, the STE-bias
caveat, recorded before any result). Potency gate passed at launch — `off=0, on=66.63`.

### What happened

| arm | wire | outcome |
|---|---|---|
| A `ssdetached` | **cut** (plain scheduled sampling) | trained 300 lessons, 184 min, **artifact produced** |
| B `ssattached` | **connected** (BPTT-SA, straight-through) | **`[FATAL GRADIENT EXPLOSION] NaN/Inf in enc_conv0.weight at Lesson 48`** |

Single variable. Same ε, seed, data, code. **Connecting the gradient path destabilises training.**
Gradient clipping is on in this config and did not save it — clipping bounds large values, it cannot
repair a NaN. The `IntegrityGuardian` caught it and failed loud, which is the system working.

### The pre-registered rule CANNOT be applied

It needs `AP@h18` from both arms. Arm B has no artifact. This is a **fourth outcome the plan did not
enumerate** — not `Δ ≥ +0.02`, not `Δ ≤ 0`, not the grey zone, but *"the arm does not train"*.
Another **C-320**-class gap: a decision rule that enumerates results and not failures.

**H is still not tested.** Twice now.

### Three candidate mechanisms tested, none reproduces it

| hypothesis | test | result |
|---|---|---|
| gradient explodes with sequence length (Jacobian product) | grad-norm vs 1…31 steps, toy head | ratio **0.999–1.000**, no growth |
| …on the real architecture | same, `HydraBNUNet06_LSTM4` + nb, 31 steps | ratio **1.001**, all finite |
| the surrogate overflows where the draw cannot | `mu` swept to 99 via raw-scale ×80 | **both finite throughout** |

At initialisation the extra gradient path contributes **~0.1%** of the total norm. So this is an
**emergent training-dynamics instability**, not an initialisation-time or overflow one, and the
mechanism is **unknown**. Recorded as unknown rather than attributed to the nearest plausible story
(σ_max ≈ 7.76 from #294 predicts an exploding Jacobian product — *consistent, but not evidence*, and
the two static tests above are mildly against it).

### What is genuinely established

* Connecting the feedback gradient **destabilises training on this vehicle** at ε=0.5. Clean
  single-variable comparison.
* **Arm A is a usable by-product**: a fresh scheduled-sampling arm on *today's* code, which removes
  the "the ε=0 baseline was trained in August on different code" confound that SCREEN-1's reading
  carried. Worth scoring on its own.

### What is NOT established, and must not be written as if it were

Per **amendment A1**, committed before this result: the straight-through estimator is **biased**, so
*"BPTT-SA is unstable here"* is **not** a supported claim. The supported claim is *"this
straight-through implementation of BPTT-SA is unstable here."* An unbiased reparameterised feedback
(#308's follow-up) could behave differently, and the two discrete sampling steps are why that route
is not free.

---

## GRAD-TRAJ — the mechanism family: **CREEP**, not JUMP · 2026-09-04

**Pre-registration:** the rule, its windows, its thresholds and its AMBIGUOUS branch were committed
in `fe856b1` **before the run**, and all three verdict branches were fired on synthetic fixtures
first (CREEP 4.0×/11.1×, JUMP 1.04×/1.00×, AMBIGUOUS 1.59×/1.68×). 36 min, two throwaway clones.

### The result

| | median grad norm, lessons 15–25 | median, lessons 38–47 | Spearman ρ(grad, lesson) |
|---|---|---|---|
| **attached** (wire on) | 133,465 | **9.4 × 10⁹** | **+0.561** |
| **detached** (control) | 859 | **312** | **−0.764** |

`late/early = 70,468×` · `attached/control at the late window = 3.0 × 10⁷×`

**VERDICT: CREEP.** Both bars (≥3× on each comparison) cleared by four to seven orders of magnitude.

**The control is the finding, as much as the treatment.** Its gradient norm *falls* over training —
ρ = **−0.764**, 859 → 312 — which is what a healthy run looks like. The attached arm's *rises*. And
the two have already separated **155×** by the early window (lessons 15–25), i.e. immediately after
ε saturates at 0.5 (lesson 15). So the divergence begins the moment the dose stops changing, and
compounds for 33 lessons until float32 gives out: 7.6 × 10¹⁸ squared is 5.8 × 10³⁷, against a
float32 ceiling of 3.4 × 10³⁸.

### Two things this kills

**"It reproduced exactly."** Lesson 48 again, on a fresh clone. The failure is deterministic, not a
numerical fluke — which JUMP would have needed.

**The loss tells you nothing.** `loss_reg` *falls* 655 → 255 across the same 48 lessons and the gate
logit drifts smoothly −6.87 → −6.67. A run whose gradient is at 10¹⁸ looks, on every curve a human
would watch, like a run that is training well. Gradient clipping is on and bounds the *step*; it
cannot bound the *intermediate gradient*, and it is the intermediate that overflows.

### ⛔ Correction to this dossier's own SCREEN-2 entry

SCREEN-2 recorded three candidate mechanisms as "tested, none reproduces it", the first two being
gradient-norm-vs-sequence-length on a toy head (ratio 0.999) and on the real `HydraBNUNet06_LSTM4`
at 31 steps (1.001). **Those measurements are correct and the conclusion drawn from them was wrong.**
Both were taken on a network at **initialisation**. The instability is something *training builds* —
at init the extra path is ~0.1% of the gradient norm, exactly as measured, and 48 lessons later it
is seven orders of magnitude above the control. **An untrained network cannot exhibit an instability
that training creates**, so those two tests never had the power to see it. Registered as **C-325**.

The σ_max ≈ 7.76 reading (#294) is now **consistent and no longer contradicted** — but it is still
not *established* as the mechanism. CREEP names a family, not a cause: any route to a compounding
gradient produces this signature, and this probe cannot separate them.

### What is now worth GPU, and what still is not

A stabiliser is worth buying — that was **not** true before this line, and would not have been if the
verdict had been JUMP. The measured failure is specifically **intermediate-gradient overflow inside
the recurrence**, not an unbounded optimiser step, because clipping already bounds the step. So the
fix must bound the gradient **per step inside the feedback wire**, not at the end of the lesson.

Still out of scope, per amendment **A1**: none of this licenses a claim about **BPTT-SA**. The
estimator is biased; what has been characterised is *this straight-through implementation*.

---

## CLIP-CHECK — the per-step limiter holds · 2026-09-04

One arm, wire connected, `ss_feedback_grad_clip = 1.0` (the same bound the model already applies to
its whole-model gradient), capped at 80 lessons. 30 min. The unclipped arm died at lesson 48 twice.

| arm | n | grad L15–25 | grad L38–47 | late/early | ρ(grad, lesson) | outcome |
|---|---|---|---|---|---|---|
| attached, no clip | 48 | 133,465 | 9.4 × 10⁹ | **70,468×** | **+0.561** | **NaN at L48** |
| detached (control) | 69 | 859 | 312 | 0.363× | −0.836 | stable |
| **attached + clip 1.0** | **80** | **913** | **834** | **0.914×** | **−0.306** | **stable, capped** |

The clipped arm's gradient sits **on the control's scale** — 913 against the control's 859 at the
same window — and trends *down*, not up. Peak over all 80 lessons is 13,021 against the unclipped
arm's 7.6 × 10¹⁸: **fifteen orders of magnitude**. `loss_reg` keeps falling (655 → 109).

**The limiter is acting, not decorating.** The new `fed_grad_max` column records the feedback
gradient *before* clipping: median **5.7** (L15–25), **12.5** (L38–47), **17.5** (L70–79), peak
**1,774**, against a threshold of 1.0. So it bites on essentially every step.

### ⚠️ Pre-registered BEFORE the screen (amendment **A2**)

That last number is the catch. Clipping at 1.0 against a natural norm of 5–18 **attenuates the
feedback gradient by roughly an order of magnitude on every step**. The wire is connected, but it is
carrying a fraction of its signal. Therefore:

> **A null result on the screen may NOT be reported as "BPTT-SA does not help here."** "The clip was
> too tight" is a live alternative explanation, and it is the *same* class of confound as **A1**'s
> estimator bias — an intervention weakened until it cannot show an effect is indistinguishable from
> an intervention that has none.

A null obliges a **threshold ladder** (e.g. 1 / 10 / 100) before any conclusion about the idea. A
*positive* result is not affected: an attenuated wire that still helps has helped.

This is recorded now, before the data, because it is exactly the kind of caveat that reads as an
excuse when written afterwards.

### What this does and does not establish

**Does:** the instability is *controllable*, and controllable by bounding precisely the quantity
GRAD-TRAJ measured. That is a second, independent confirmation of the CREEP diagnosis — a defect in
the straight-through arithmetic (the JUMP branch) would not have been fixed by a gradient bound.

**Does not:** say anything about skill. `loss_reg` is in-sample *training* loss, and this dossier's
own M32/M45 history is a series of arms that improved a training number and lost AP. The screen is
still the only thing that can answer #308, and it has still never run to completion.

---

## SCREEN-3 — the screen finally RAN. **The wire HURTS, substantially.** · 2026-09-04

Both arms trained 300 lessons. Weight hashes differ (`54b5b215…` vs `0d9b59a5…`) so the treatment
acted — not void by the C-324 route. Neither arm crashed, so **BRANCH 0 does not apply**. This is
the first time #308's question has actually been answered.

### Primary — `AP@h18`, `sb`, free-running, 13-origin support

| | control (wire cut) | treated (wire on, bounded) | Δ |
|---|---|---|---|
| **AP@h18** | **0.3064** | **0.1997** | **−0.1066** |

**−0.1066 is not a null.** It is **5×** the pre-registered `+0.02` decision threshold and roughly
**1.8×** the ~20% training variance (C-119/C-184) that limits this n=1 design. A 35% relative loss.

### It is worse at every horizon, and the mechanism is FIRING

| h | ΔAP | Δact_ratio | Δ crps_events | Δ size_ratio |
|---|---|---|---|---|
| 1 | −0.0715 | **+0.1020** | +0.80 | −0.007 |
| 18 | **−0.1066** | **+0.0709 (0.036 → 0.107, ×3.0)** | −0.01 | 0.000 |
| 36 | −0.0904 | **+0.0223 (0.0026 → 0.0249, ×9.6)** | −0.04 | 0.000 |

The treated model **fires far more and places far worse**. Magnitude is untouched (`size_ratio` 0.000
in both at h18/h36) and `crps_events` is unchanged, so nothing about the body moved — this is
entirely an occurrence/placement effect.

**That is M45 again, exactly.** "AP loss scales with how much the model FIRES" — `truncated_nb`
×1170 firing cost −0.238 AP. Here ×3.0 firing at h18 costs −0.107. BPTT-SA-via-straight-through has
landed on the same lever four other interventions already fell off, and it falls the same way.

**Vehicle sanity check:** the control's 0.3064 is where it should be. The untouched L=300
free-running reference is 0.3318 (M34–M37) and M30–M33 measured plain SS at ε=0.5 costing −0.0426,
predicting ≈0.289. The control sits between, within seed noise. The ruler is not broken.

### ⚡ The result discriminates between its own two pre-registered confounds

A2 committed the required next step on a null: a threshold ladder, because the clip attenuates the
feedback gradient ~10×. **The data now argues against A2 being the explanation, and the argument is
directional, not convenient:**

> Attenuating the feedback gradient toward zero makes the treated arm **approach the control** — with
> no signal on the wire it *is* plain scheduled sampling. Throttling predicts **convergence**. What
> was measured is a 35% **divergence**. Attenuation cannot manufacture harm of this size; at worst it
> manufactures sameness.

**A1 survives and is now the leading explanation.** A biased estimator does not merely weaken the
signal, it points it in a *wrong direction* — and more of a wrong direction is actively harmful. The
straight-through surrogate substitutes the composed **mean's** gradient for the **draw's**. That is a
systematically incorrect credit signal, and this is what training on one looks like.

**So the ladder changes purpose.** A2 wrote it as a rescue attempt. It is now a **discriminating
test** with opposed predictions, which is worth far more:

* if **A1 (bias)** is the story, raising the clip feeds *more* wrong gradient ⇒ **worse, or unstable**
* if **A2 (throttling)** is the story, raising the clip feeds *more real* gradient ⇒ **better**

### What is established, and what is still not

**Established:** on this vehicle, at this seed, connecting the feedback gradient through a
straight-through estimator and bounding it at 1.0 **costs 0.107 AP@h18** and does so by making the
model fire more. The effect is far outside what this design can confuse with noise.

**NOT established, and the amendments still bind:** this is **not** "BPTT-SA does not work". A1's
bias is unrefuted and is now the *prime suspect for the harm itself* — which means the honest reading
is that **the straight-through approximation may be the thing that failed, and the idea underneath it
is untested.** An unbiased feedback (the reparameterisation follow-up) remains the open question.
n=1, one seed, one configuration.

---

## EXP-BIAS (Phase 1) — **A1 REFUTED. The estimator points the right way.** · 2026-09-04

Rule pre-registered in **A3** (`5202cf8`), committed before the instrument existed. n=1024 draws,
three handoffs, both trained 300-lesson artifacts, on each arm's **own real data** through the
production pipeline (5,034,240 rows), 905,289 parameters, `nb` + `soft_gate`.

### Readout — `cos(Δ_ST, Δ_SF)`

| arm | step 1 | step 3 | step 5 |
|---|---|---|---|
| **attached** (trained WITH the wire — where the harm happened) | **+0.882** | **+0.765** | **+0.756** |
| detached (control) | +0.213 | +0.707 | +0.782 |

Threshold was **≥ +0.3 ⇒ A1 weakened, A3 leading**. Five of six handoffs clear it by a wide margin;
**all three on the arm that actually trained under the estimator are ≥ +0.75.**

### Every instrument gate passed

| gate | result |
|---|---|
| split-half reliability of `Δ_SF` | **0.79 – 0.99** (parity and permuted splits agree: 0.86/0.89, 0.94/0.87, 0.84/0.84) |
| negative control `cos(Δ_ST, random)` | **≤ 0.0018** in all six |
| forward identity `L_cut == L_on` | **≤ 5.5e-07** — float32 noise; straight-through touches only the backward |
| exact-truth Bernoulli control | passes (`dE/dp = f(1) − f(0)`) |
| exact-truth Gaussian control | passes (`dE[z²]/dμ = 2μ`) |
| plateau | 3 of 6 formally; see below |

**The plateau gate is the one soft spot, and its failure direction is known.** The cosine *rises*
with N (attached step 5: −0.579 → +0.014 → +0.530 → +0.670 → +0.754 → +0.756) because a noisy
reference can only drag a correlation **toward zero**. So an observed +0.76 is a **lower bound**:
more samples move these numbers further above the threshold, never below. The verdict direction does
not depend on closing that gate.

**⛔ My earlier n=256 reading was wrong and the gate caught it.** At n=256 the control's split-half
came back +0.50 / −0.06 / −0.17, and I suspected my even/odd split was manufacturing it from
adjacent RNG seeds. It was not — a permuted split at n=1024 agrees with parity to ~0.05. It was
simply sample size. Recorded because I published the suspicion.

### What this establishes

**A1 is refuted.** The straight-through estimator does **not** point the wrong way; it aligns with
the true gradient at 0.76–0.88 on the arm trained under it. The −0.1066 AP@h18 loss is therefore
**not** explained by my approximation being backwards.

**A3 leads: the objective genuinely does this.** BPTT-SA on this problem really does train the model
to fire more — and firing is the lever M32/M45 and three other interventions have already shown to
cost AP. This is a fifth confirmation of that finding, not a novel failure.

⚠️ `cos ≈ 0.8` is not 1.0; ~0.6 rad of misalignment remains. The claim is "not backwards", **not**
"unbiased". A perfectly unbiased feedback could still behave differently, and #308's follow-up
(reparameterised / REINFORCE) is untouched by this.

### ⛔ DEFECT FOUND IN MY OWN PHASE-2 DESIGN (see C-327)

A2/A4 pre-registered the clip ladder as a **discriminator**, predicting AP *decreasing* under A1 and
*increasing* under A3. **That derivation was wrong.** Under A3 the gradient is correct and points at
an optimum that involves MORE firing, so more of it also gives **worse** AP. A1 and A3 make the
**same** prediction and the ladder cannot separate them. It never could have.

It did not cost anything — Phase 1 answered the question directly, and the ladder was never run —
but a gate that cannot deliver its verdict is exactly **C-320**, and this is the fifth instance.
