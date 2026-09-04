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
