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
