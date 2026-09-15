# Decision criteria — the three zero-GPU falsifier checks

> ## ⚠️ This is a DECISION MEMO, not a blind test. Read this before citing anything below.
>
> The falsifiers for **#288**, **#290** and **#291** were registered in those GitHub issues on
> 2026-08-22, **before any answer was known** — that part is genuine and the issue timestamps prove it.
>
> But they were registered **qualitatively** (*"if the failure is concentrated in the first ~6 steps"*),
> and **C-305** — registered the same day — is precisely the failure that a qualitative rule enables: a
> registered branch fired, I overrode it with a numeric criterion the rule did not contain, and wrote it
> up as though no branch had matched.
>
> **So this document exists to make the criteria numeric. And it is written AFTER exploration surfaced
> the values.** Three consequences, stated so nobody has to discover them:
>
> 1. **#290 and #291 are not blind.** I have seen the inputs. The thresholds below are set from
>    **external principled grounds** — Lamb's own scaling argument, and the conventional 2σ/1σ bar —
>    not reverse-engineered from our numbers. That is a mitigation, not a repair.
> 2. **The expected outcome of each check is stated up front.** If a check lands where expected, that is
>    weak evidence; the criterion was chosen knowing roughly where the data sat.
> 3. **#288 was genuinely blind** — nobody had traced the sampling path. It is also the only one of the
>    three that gates a whole research branch, and it is a *fact about code*, not a measurement, so no
>    threshold applies.

---

## Why these three, and why now

The retention programme closed with PR #292. `freeze_recurrent='cell'` ships; **77% of the oracle gap
stays open**. Five investigative issues carry the remainder. Ranked by **expected information gain per
unit cost**, three carry falsifiers answerable with **zero GPU**, and they reshape where the expensive
budget goes:

| issue | falsifier | cost | prior | decides |
|---|---|---|---|---|
| **#288** | is the fed-back draw differentiable at all? | a code read | **~50/50, genuinely unknown** | whether the branch is alive |
| **#290** | where along the horizon does damage concentrate? | data on disk | ~75/25 it closes | closes an issue for nothing |
| **#291** | does M27 contradict its coupling premise? | data on disk | ~80/20 it closes | closes an issue for nothing |

**#287** (increasing-TF curriculum) is the highest-value if it works, but a 4v4 costs **~36 GPU-hours**.
Committing that before reading the code in #288 would spend hours a free measurement could redirect.

---

## Check A — #290, Professor Forcing

**The registered falsifier** (issue #290, 2026-08-22): *"if our rollout's failure is concentrated in the
first ~6 steps … the method is aimed at the wrong part of the curve."*

Lamb et al. 2016 report **no benefit below ~100 steps**, and argue the benefit **scales with the
importance of long-term dependencies**. Our readout horizon is 36. The question is therefore whether our
damage is a *short-horizon* phenomenon (Professor Forcing aimed wrong) or *accumulates over the horizon*
(its mechanism plausibly applies).

**Metric.** With `gap(h) = oracle_AP(h) − free_running_AP(h)` on a shared support:

```
rate_early = (gap(6)  − gap(1))  / 5      # AP lost per step, steps 1→6
rate_late  = (gap(36) − gap(6))  / 30     # AP lost per step, steps 6→36
front_loading = rate_early / rate_late
```

**Criterion.**

| `front_loading` | verdict |
|---|---|
| **≥ 2.0** | **front-loaded ⇒ CLOSE #290.** The first block loses AP at more than double the later rate — a different regime, not a marginal difference. A method whose benefit scales with long-term dependencies is aimed at the wrong part of the curve. |
| **≤ 1.2** | accumulating ⇒ **KEEP OPEN** despite the 36 < 100 length argument. |
| between | **INCONCLUSIVE.** The length argument alone deprioritises; it does not close. |

**Why 2.0, chosen from Lamb and not from our data:** Lamb's claim is about a *regime* (long-term
dependencies mattering), not a marginal effect. A doubling of the per-step damage rate is the smallest
ratio that distinguishes a regime from a gradient. 1.2 is the corresponding floor for "indistinguishable
from uniform accumulation".

⚠️ **Expected: ~2.2–3.5 ⇒ close.** Exploration has already reported ~0.017 AP/step early against
0.003–0.008 late.

**Scope, stated rather than resolved.** Score CSVs exist only on the grid {1,6,12,18,24,30,36}.
**Horizons 2–5 do not exist anywhere in the repo**, so this is a *block-level* answer and cannot resolve
step 2 vs step 5. Resolving it needs a re-emit; the block-level evidence is what this check rests on.

## Check B — #291, Horizon Forcing

**The registered falsifier** (issue #291, 2026-08-22): *"M27 is the pre-registered falsifier for the
premise. If T=0 and retention are independent in our vehicle, 'controlling long-term error necessitates
controlling short-term error' is false here."*

Zhuang et al. 2025 state the premise with its own scope condition: *"**in chaotic systems** controlling
long-term error necessitates controlling short-term error."* Conflict counts are not chaotic in the
Lyapunov sense, so the question is whether the coupling holds here.

**Metric.** Across the lesson-count sequence (40 → 160 → 300 → 600), for each step:

```
ΔT0        = AP_h1(L_next) − AP_h1(L)          against σ_T0        (seed sd at L=160, n=6)
Δretention = R(L_next) − R(L),  R = AP_h18/AP_h1   against σ_retention (seed sd at L=160, n=6)
```

**Criterion.** **Decoupled ⇒ CLOSE #291** if there exists a step where **ΔT0 > 2·σ_T0** *and*
**Δretention < 1·σ_retention**: one-step skill moves materially while robustness does not.

**Why 2σ/1σ:** the conventional bars for "real movement" and "indistinguishable from noise". Not derived
from our values.

⚠️ **Expected: the 300→600 step gives ΔT0 = 2.8σ and Δretention = 0.03σ ⇒ close.**

**Robustness requirement.** M26/M27 quote L=300 from **one seed**, but **four ε=0 seeds now exist**. The
check must be run on the multi-seed L=300 retention mean as well; if the two disagree on the verdict,
report both and do not close.

## Check C — #288, BPTT-SA

**Not a threshold — a fact about code.** The registered question (issue #288, item 3): *"Does the
gradient even exist? A sample from a Gamma/NB draw is not differentiable without a reparameterisation."*

**Method.** Trace the feedback path and *execute* the decisive operations in this environment rather
than reasoning about them. Report:

1. where the gradient is severed, by which operation, at which line;
2. whether removing the two `.detach()` calls would restore it, or fail **silently**;
3. whether any reparameterised path exists, reachable or not.

**Consequences, registered before writing them up:**

* **If the path is severed and un-detaching fails loudly** → #288 re-scopes to "find a reparameterised
  count path", a much larger piece of work.
* **If it fails *silently* (zero gradient, no error)** → that is a **trap**, and the issue must carry a
  mandatory graph assertion for any future arm, because `tests/train/test_feedback_parity.py` pins
  *values only* and would pass a zero-gradient no-op.
* **If a differentiable path exists** → #288 is live and cheap, and the arms should be named.

## Falsifiers on the checks themselves

1. **h1 must be bit-identical between free-running and oracle on every seed.** There is no feedback at
   step 1, so the two cannot differ. If they do, the dossiers are not describing the same vehicle and
   Check A is unreadable.
2. **`N` and `n_event` must match** between the free-running and oracle CSVs at every horizon, or the
   subtraction in `gap(h)` is between different supports.
3. **M27 must reproduce from the CSVs**, not from the ledger prose: T=0 300→600 = +0.02127,
   σ_T0 = 0.0077, Δretention = +0.00143.

## Out of scope

No GPU. No change to `views_hydranet/`. No confidence interval on `gap(h)` — that needs two cubes
present simultaneously and every `predictions_*` cube has been deleted; the point estimates need
nothing. #287's off-the-floor arm.

---

# AMENDMENT 1 — Check D (#294, GTF): the σ_max estimator

*Added 2026-08-23, **before the measurement**. The threshold itself was registered earlier, in GitHub
issue #294 (created 2026-08-22), whose timestamp precedes this file. **Check D is therefore genuinely
blind**, unlike Checks A and B.*

## The registered threshold (from #294, restated)

Hess et al. 2023 derive `α = 1 − 1/σ_max`, **requiring σ_max ≥ 1**. M41 measured our empirical optimum
at **w ≈ 0.1**, which if `w ≡ α` implies **σ_max ≈ 1.11**.

| measured σ_max | verdict |
|---|---|
| **1.05 – 1.20** | striking — a derived α lands on a hand-swept optimum. Correspondence likely real. |
| **≫ 1.2** | correspondence superficial; needs explaining before anything is built. |
| **< 1.0** | **α is undefined for us — CLOSE #294.** The paper's formula yields a negative α below 1. |

## The estimator, chosen before running

σ_max is *"the supremum of spectral norms over all RNN Jacobians"*. The recurrent state map for LSTM
cell *k* (`HydraBNrecurrentUnet_06_LSTM4.py:524-529`), with state `(hs, hl)` and input `x`:

```
i = σ(Wxi·x + Whi·hs)   f = σ(Wxf·x + Whf·hs)   c̃ = tanh(Wxc·x + Whc·hs)
hl' = f ⊙ hl + i ⊙ c̃    o = σ(Wxo·x + Who·hs)   hs' = o ⊙ tanh(hl')
```

**The four cells are independent given `x`** — cell *k* reads only `hs_k, hl_k` — so the Jacobian is
**block-diagonal** and `σ_max = max_k σ_max(block k)`.

**Step 1 (data-free, exact for the operator): spectral norms of the 16 recurrent convolutions.** Power
iteration on each `Wh*` as a *convolution operator* at the real spatial size, not on the weight matrix —
a conv's operator norm is not its kernel's matrix norm. This needs **no data at all**.

**Step 2: the analytic Jacobian bound.** Using `|σ'| ≤ ¼`, `|tanh'| ≤ 1`, `|f|,|i|,|o| ≤ 1`,
`|c̃|,|tanh(hl')| ≤ 1`:

```
‖∂hl'/∂hl‖ ≤ max f  < 1                      (the cell's self-coupling is ALWAYS contractive)
‖∂hl'/∂hs‖ ≤ ¼·M·‖Whf‖ + ¼·‖Whi‖ + ‖Whc‖     M = max|hl|, the one term needing data
‖∂hs'/∂hl‖ ≤ max(o · tanh'(hl') · f) ≤ 1
‖∂hs'/∂hs‖ ≤ ¼·‖Who‖ + ‖∂hl'/∂hs‖
```

**What each outcome licenses — registered now so it cannot be chosen later:**

| result | reading |
|---|---|
| the bound is **< 1 even at generous M** | **σ_max < 1 definitively ⇒ CLOSE #294.** An upper bound below 1 is decisive; no data needed. |
| the bound is **> 1** | **INCONCLUSIVE — an upper bound above 1 licenses nothing.** The bound is loose (it assumes every gate saturates simultaneously). Reporting "σ_max ≥ 1, so GTF applies" from a loose bound would be C-306 again: the wrong yardstick for the question. A verdict then requires **power iteration on the true Jacobian at states from a real rollout**, which needs an inference pass. |

**M is the one term requiring data.** `max|hl|` is unbounded analytically for an LSTM. The bound will be
reported **as a function of M**, with the M at which it crosses 1 stated explicitly, so the reader can
see how much cell-state magnitude the contractive conclusion would tolerate.

**Falsifier on the check itself:** the 16 conv operator norms must be **stable under power-iteration
count** (report at 50 and 200 iterations; a drift > 1% means the iteration has not converged and no
bound built on it is readable).
