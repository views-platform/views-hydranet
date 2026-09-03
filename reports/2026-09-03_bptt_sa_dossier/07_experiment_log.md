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
