# 08 — Post-mortem: a change that ran, passed everything, and did nothing

**Scope:** 2026-09-03, issue **#308**. One implementation, six errors, 276 minutes of training
producing two models with byte-identical weights.

Written at the chair's request after he asked the question this episode actually turned on:

> *"How can we know it is fixed? ... your error rate is so high that I don't know how to know if the
> implementation is correct. As you say the fact that the result was impossible rather than merely
> disappointing was a blessing. Otherwise we would have been fooled into thinking we know something
> which we do not know. I estimate that we have rejected many ideas that would have improved
> performance due to wrong implementations."*

This covers process and verification, not science — there is no science, because the hypothesis was
never tested. The scientific record is `07_experiment_log.md` (SCREEN-1: **VOID**).

---

## 1. The headline

| | |
|---|---|
| Change size | **3 lines** |
| Safeguards passed while it did nothing | unit tests (7), mutation testing (**5/5**), lint, full suite (**1,901**) |
| GPU burned | **276 min** training + 13 min emit |
| Distinct errors in the episode | **6** |
| Of those, the *same* error repeated | **3** |
| What actually caught it | the result being **impossible** — i.e. luck |

The last line is the post-mortem. Every deliberate safeguard passed. The thing that saved us was
that two independently trained models came out identical to the last bit, which cannot happen.

---

## 2. What was supposed to happen

The training loop feeds the model its own prediction (scheduled sampling), and that fed value was
`.detach()`-ed — the gradient could not reach the step that produced it. Removing the detach should
let credit flow back through the handoff (**BPTT-SA**, `Vlachas2023_LearningFromPredictions`), which
is also a mechanical account of why scheduled sampling failed here before (M26–M33).

## 3. What actually happened

`d(draw)/d(params)` is **exactly 0**. A draw from the family is not reparameterised. It carries a
`grad_fn` from a `log1p` wrapper — so it *looks* connected — and delivers nothing. Removing
`.detach()` from a tensor whose gradient is already zero changes nothing.

And **C-259 requires sampled feedback whenever ε > 0**. The repo's own guard forces the single mode
in which the change cannot act.

---

## 4. The six errors, in order

| # | error | why it survived |
|---|---|---|
| 1 | removing `.detach()` was **inert** on the production path | nothing measured whether the knob moved anything |
| 2 | tests verified the wire on the **point-head** path (`family=None`) | production is a **family head**; the tested path is differentiable, the real one is not |
| 3 | mutation testing was **5/5 green while the change did nothing** | mutations only probe code the author already thought to test |
| 4 | the first fix used `mean` mode, which **ignores the gate** | it is the analogue of an *uncomposed* draw — a different quantity from what the forward pass produces |
| 5 | the NaN diagnosis was **itself wrong** — my fixture fed raw params where production supplies activated ones | I debugged a symptom my own test harness invented |
| 6 | the first integration attempt left **4 of 5 call-site mutations alive**, including the original bug reintroduced verbatim | the tests exercised *helpers*, and the bug lives at the *call site* |

**Errors 2, 5 and 6 are the same error**: verifying on a path production never takes.

That is **C-323**, which I wrote into the risk register **the previous day**, about ablations on a
component the architecture regenerates. I wrote the general rule and then broke it three times
inside twenty-four hours, in a different part of the system each time. Knowing a failure mode
abstractly does not confer the habit of checking for it.

---

## 5. Biases weighed — intellectual-honesty audit

**Confirmation bias, concretely.** I wanted the change to be small and elegant. "It's three lines
and the plumbing already exists" was true and became a reason not to look further. The satisfying
shape of the fix substituted for evidence that it did anything.

**Green-suite anchoring.** Seven passing tests and 5/5 mutations produced *more* confidence than a
change of that risk warranted. I treated mutation testing as a guarantee when it is a coverage
measure — it probes the code you thought to test, and I had not thought to test the family path.

**A confound I did not rule out and reported anyway.** SCREEN-1's arms both sit ~0.025 below the ε=0
baseline. I noted that as "scheduled sampling still hurts, consistent with M26–M33". The baseline
was trained in **August, on different code**, so an unknown part of that gap is drift. It is logged
with the caveat but it should not have been reported as corroboration at all.

**The counter-hypothesis I should have held.** Before running, the live question was *"does this
knob do anything?"*, not *"does BPTT-SA help?"*. I went straight to the second and never asked the
first. The pre-registration is meticulous about the *hypothesis* and silent about the *apparatus* —
`05_analysis_plan.md` names the measure, the noise floor, and the decision rule, and nowhere asks
whether the treatment is capable of acting.

**Where I would still be wrong if the result had been merely bad.** If the knob had been
half-connected, Δ would have been small, the locked rule would have returned *"Δ ≤ 0 → H is dead"*,
and I would have written a ledger row saying BPTT-SA does not work here. The chair's estimate that
this programme has rejected workable ideas on broken implementations is **supported by this
episode's own mechanism**, and I have no basis for assuming it is the first time.

---

## 6. What did and did not catch it

**Did not:** unit tests · mutation testing · lint · the 1,901-test suite · code review · type
correctness. Every one green.

**Did:** the result being impossible.

**What now catches it — measured, not asserted:**

| safeguard | evidence it works |
|---|---|
| **Potency gate** (`scripts/potency_check.py`) — prove the knob moves a number **on the arm's own config** | reproduces the failure exactly: `off=0.0, on=0.0 → INERT`. 7 tests, 5/5 mutations. |
| **Integration test at the call site, on the production path** | with helper-level tests only, **4/5 call-site mutations survived** including the original bug verbatim; adding **one** family-head integration test caught **4 of those 5** |
| **Positive control** (`assert_control_fires`) — prove the *readout* can see a known effect | a null from a blind harness is not a null |

The fifth mutation is left uncaught **deliberately**: C-259 rejects its configuration at validation,
so it is unreachable, and a test now pins that reachability claim so the justification fails loudly
if C-259 is ever relaxed.

---

## 7. The general lesson — C-324, Tier 1

> **A broken implementation must not produce the same signature as a null result.**

Where *"the knob does nothing"* and *"the knob does nothing useful"* look alike in the readout,
every null is ambiguous, and no after-the-fact analysis separates them.

Registered **Tier 1** — the only class in the register whose failure mode is a **wrong scientific
conclusion recorded as fact, with no error signal anywhere**. It corrupts the ledger, not a run.

---

## 8. What worked

**The chair's refusal to accept the fix at face value.** After the diagnosis he did not ask for the
patch; he asked how anyone could *know* a patch was correct given the demonstrated error rate. The
potency gate, the integration-test discipline and C-324 all exist because of that question. The most
valuable artifact of the episode came from the question, not from the code.

**Checking before running, once prompted.** Errors 4, 5 and 6 were all caught *before* any GPU time,
by measuring instead of assuming. That is the only part of the loop that behaved.

**The honest library answer.** The library was searched for guidance and does not contain it — it
has Clean Architecture, Ousterhout, Nygard, Kleppmann, Kitchenham, and nothing on manipulation
checks. Saying so, rather than dressing an adjacent hit as an answer, left a real gap on the record.

---

## 9. Disposition

* **Stays:** the straight-through fix (`bf6b50f`) — forward bit-identical to the draw
  (`max|diff| = 0.0`), gradient 0.0 → 82.0, verified on the production path.
* **Abandoned:** SCREEN-1's two arms and their artifacts. Both retrained, not reused — the detached
  arm's code path is *textually* unchanged, and "textually unchanged so it must be fine" is the
  exact assumption this episode punished.
* **Redirects toward:** the potency gate as a hard launch precondition in `run_screen.sh`, not a
  remembered habit. Every future arm in this programme proves its knob is potent before it costs
  anything.
* **Unchanged:** the pre-registration. Same arms, same variable, same rule, still **n=1**, still a
  **SCREEN with no verdict authority**. Only the implementation under test is different.
