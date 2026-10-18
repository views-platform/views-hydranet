# R&D PLAN — the gate under-dispersion programme, Phase 2 onward

**Date:** 2026-10-18 · **Owner of this document:** views-hydranet seat · **Executing seat:** views-models
**Gate:** ⛔ **Simon, 2026-10-18 — nothing runs until (a) the results record, (b) this plan and (c) a Phase-2 pre-registration exist.** (a) is `RESULTS.md`; (c) is `../2026-10-18_gate_phase2_prereg.md`.
**Inputs:** `RESULTS.md` (the complete Phase-1 record) · `PREREG.md` (Phase 1, 14 amendments) · `../2026-10-16_pos_weight_arm_outcome/OUTCOME.md` rev 3 (13 issues)

⛔ **This document is a dependency map and a sequencing argument. It authorises nothing.** Every item names what it needs, what it costs, and whether that cost has been measured.

---

## 0. In plain words

We know the gate's probabilities are squashed on the two models we studied. We have a one-parameter fix that works on one of them and fails on the other. **We do not know whether the defect exists on any of the eight models actually being served**, and the one piece of evidence we have says it does not generalise — two models, one config key apart, and the fix works on only one.

So the next step is to **measure, not to fix.** Everything after that depends on what the measurement says.

---

## 1. ⛔ The sequencing constraint, from Simon

> *"when/if we think we have a solution to the gate's under-dispersion then we (you, I and Hydranet session) move to the body."*

**Gate first, body second.** The body defect is **#530** — under-predicts totals by ~10×, ~0% interval coverage — and is **not in scope** here.

⚠️ **But the two are not independent in the delivered product, and this plan must say so** (§6).

---

## 2. Phase 2 — measure the served models. THE NEXT STEP.

**What:** gate calibration (`b`, `a`, held-out, bin-free and binned) on the eight `rusty_bucket` members.
**Why it cannot be skipped:** ⛔ **zero gate-calibration numbers exist for any served model.** Everything in `RESULTS.md` is amber and copper — `n = 2`, same architecture, **one config key apart, and R1 works on one and fails the other.** Generalising from that is unjustified.
**Cost:** ~1.5 h local CPU, **$0, no GPU** — *the other seat's estimate.* ⚠️ **UNMEASURED by this seat, and nine of nine prior estimates across both seats were optimistic. Measure the first model and report before running the remaining seven.**

**Pre-registered in (c).** Nothing else in this plan proceeds until it returns.

---

## 3. ⛔ TWO PATHS — and the hard-gate path is NOT a treatment path

The other seat proposed "two treatment paths". **This seat's correction: there is one treatment path and one leave-alone path.**

### Path A — the 7 `soft_gate` models. A treatment path.

`amber_kestrel`, `copper_kestrel`, `blue_stranger`, `pink_pirate`, `purple_alien`, `bright_starship`, `heavy_freighter`.

A soft gate passes the distortion **straight through**: `compose_samples` draws `Bernoulli(gate)` and `compose_mean` multiplies by `gate`, so a squashed gate **is** a squashed firing probability and a squashed expected count. ⛔ **These are the models the defect can actually harm.**

### Path B — the 3 `threshold_gate` models. ⛔ LEAVE THEM ALONE.

`violet_visitor` τ=0.20, `blazing_meteor` τ=0.16, `bold_comet` τ=0.14.

**DERIVED and verified exactly** (`RESULTS.md` §8): temperature scaling and moving τ are **the same knob** —

```
sigmoid(logit(g)/T) ≥ τ   ⟺   g ≥ sigmoid(T·logit(τ))     [200,000/200,000 identical decisions]
```

— and under-dispersion is a **monotone** distortion, so a threshold gate is **invariant to it up to the choice of τ.**

⛔ **And τ was tuned empirically (#465 moved it off 0.5), so the tuning already absorbed the distortion.** These models have had the correction applied implicitly. Therefore:

- **They need no fix.**
- ⛔ **Applying R1 while keeping τ would CHANGE their firing set for no reason** — moving the effective threshold from 0.2302 back to 0.20 on `violet_visitor`. **A correction that is a no-op on the right model is a regression on the wrong one.**
- **If anyone ever does touch them, τ must be re-derived, and the gain is zero by construction.**

⚠️ **Simon, 2026-10-18: too soon to conclude, and he is not planning to change them — *"if anything the opposite."*** This path exists in the plan to be **excluded**, not worked.

---

## 4. ⛔ The fix is not a fix — it is a per-model choice, and the selection rule must be pre-committed

`RESULTS.md` §7: **R1 works on amber (71%) and FAILS on copper** (worse than raw). The reason is in the bin-free intercepts — amber **+0.16** (slope-only), copper **+1.30** (slope **and** level) — and **one parameter cannot address two components.** R3's two fix copper exactly.

So Phase 3 is not "apply R1". It is: **per model, choose between R1 (1 param), R3 (2 params) and no correction.** ⛔ **And the selection rule must be written down BEFORE the Phase-2 numbers are seen**, or we will be fitting a remedy family to the data it is judged on — the `implied_w` error a third time.

**Proposed rule, for (c) to pre-commit:** choose by **held-out calibration error under origin-fold cross-validation**, with R1 preferred on ties within one seed sd, and **R3 only where the bin-free `|a|` exceeds a pre-set threshold** — because that is the measurable signature that distinguishes the two cases, and it was already in the record before either arm's remedy was fitted.

---

## 5. ⛔ F11 — the firing-rate check. NOT free, and the plan must stop calling it a gate.

**This is the last check before any correction reaches production**, and it is the one that would catch a repeat of the §3 CRPS artefact.

⛔ **But it is not free and it is not currently possible.** It needs `act_ratio` and CRPS on the **composed** forecast. Tier 2 does not build the composed forecast, **and the uncomposed body output was not retained** — verified: nothing on disk, and the library's `_dump_body_mean` path was not enabled for these runs.

**So F11 requires a re-run.** A re-run needs a GPU. ⛔ **Therefore "the last gate before production" is a Tier-3 item, not a cheap one, and any plan that lists it as a free check is wrong.** Either:
- **(i)** enable `_dump_body_mean` and re-run the two arms — GPU, Simon's; or
- **(ii)** recompose from stored draws if the magnitude cube is retained — ⚠️ **unverified; this seat has not checked whether `draws/` holds the uncomposed body.** **Check before costing it.**

---

## 6. ⛔ THE INTERACTION NEITHER SEAT HAS NAMED: a gate fix makes the known body defect WORSE in the product

Simon's sequencing is gate-then-body, and that is right for *diagnosis*. ⛔ **But they are coupled in the delivered numbers and the plan must state it.**

- The composed forecast is `gate × magnitude` (`compose_mean`) or `Bernoulli(gate) × magnitude` (`compose_samples`).
- **R1 on amber lowers the mean gate by 21.3%** (1.209× truth → 0.951× truth; **REPRODUCED here**). Correct for occurrence — it moves *toward* the true rate.
- ⛔ **But #530 says the body already under-predicts totals by ~10×.** A 21% lower gate is a **~21% lower expected count**, on a product already an order of magnitude low.

⛔ **So the honest statement: R1 makes the occurrence probabilities honest and the delivered totals worse.** On copper it is unambiguous — raw 0.945× truth → **0.733×**, away from truth on both counts.

**Consequences for sequencing:**
1. **Calibrating the gate in isolation improves a diagnostic and degrades the delivered product.** That is not a reason to skip it, but it **must not be presented to anyone as a product improvement** until the body is fixed.
2. ⛔ **The gate fix and the body fix should ship TOGETHER, or the gate fix should ship as a SEPARATE occurrence product that does not touch the count forecast.** The second is free and touches nothing shipped; the first needs #530 done.
3. This is the strongest argument for the other seat's recommendation that the corrected gate be published as a **new occurrence product** rather than used to correct the counts. ✅ **This seat agrees, on this ground.**

---

## 7. ⛔ A GOVERNANCE BLOCKER — the eligibility criteria are satisfiable by the artefact

`RESULTS_LEDGER.md:613-623` defines the shipping gate: **CRPS ≥ 5% better than baseline**, with guardrails **QS99, Brier, MCR**.

⛔ **There is no occurrence guardrail — and `OUTCOME.md §3` measured that halving the firing rate buys 37–83% CRPS.** So a model that halves its firing rate **passes criterion 1 outright**, and none of the three guardrails reliably catches it: MCR is the pooled magnitude ratio that **`audit_run.py:137` explicitly demotes** (*"DO NOT read as calibration"*), and Brier on a 99.7%-zero field is dominated by zeros (the documented #258 blindness, *"under-activate 4–115× yet MCR ≈ 1"*).

⚠️ **This blocks validation, not measurement.** Phase 2 can run. But **no fix can be declared an improvement against criteria the artefact itself satisfies.**

**Proposed, as issues for Simon — not edited by this seat, because eligibility is governance:**
1. Add an **occurrence guardrail** (`act_ratio`, read with `precision_at_k`) to the eligibility conjunction.
2. Correct the ledger's metric table, which lists MCR as *"guardrail — magnitude calibration"* against `audit_run.py:137`.
3. Promote both limbs of the standing rule (`OUTCOME.md §3.5`) into the ledger, where eligibility is defined.

---

## 8. Out of scope, and why

| | why |
|---|---|
| **The body (#530)** | Simon's sequencing: gate first. ⚠️ But see §6 — coupled in the product. |
| **The root cause (train/deploy geometry)** | Training crops are 32×32 activity-seeking; inference is one untiled 360×720 pass. **Training-side, GPU, never scoped by this programme.** `RESULTS.md` §6. |
| **`h ≥ 2`** | The gate posterior is not retained (#558 / C-161). Everything is h=1; the delivery is 36 months. **This programme's inability to reach h≥2 is the evidence for prioritising #558.** |
| **H2 (`weight_decay: 0.1`)** | Live, untested, and the premise this seat argued it from is damaged. A training arm — Tier 3. |
| **H3 (resolution limit)** | Live, untested. ⛔ **F7 firing cannot confirm it — AP is unchanged by construction, and a test that cannot fail cannot confirm.** |
| **Tuning the recal distribution** | ⛔ **Banned** (`RESULTS.md` §6) — a nuisance parameter fitted to its own endpoint, and no fixed statistic is correct. |

---

## 9. The exit

**#563** — production-readiness gate and exit criteria (filed in views-models at Simon's instruction). The route: fix gate → fix body → swap the `rusty_bucket` members → flip maturity.

⛔ **On the current evidence this plan cannot say when that is reachable**, and will not estimate it: the gate fix is validated on **one** model, fails on its sibling, has an **unrun** firing-rate gate that needs a re-run, and the body defect is untouched.

---

## 10. Sequencing, with dependencies

```
   Phase 2   measure the 8 served models            FREE*, next, pre-registered in (c)
      |          *estimate is the other seat's and UNMEASURED — measure model 1 first
      v
   Phase 3   per-model remedy selection             FREE — rule pre-committed BEFORE Phase 2 reads out
      |      (R1 / R3 / none; Path B excluded)
      v
   Phase 4   F11 firing-rate + CRPS on the          ⛔ NOT FREE — needs the composed forecast,
      |      composed forecast                         which needs a re-run (GPU, Simon's)
      v
   Phase 5   publish as a SEPARATE occurrence       free; touches nothing shipped (§6)
             product  — OR hold for #530
      |
      v
   blocked   anything presented as a product        ⛔ blocked on §7 (governance) and §6 (the body)
             improvement
```

**Parallel and independent of all of the above:** the 13 issues in `OUTCOME.md §9` and the three proposed in §7. **Simon's repos, Simon's call.**

---

## 11. What this plan commits to about its own methods

Carried forward from Phase 1, because each was learned by being wrong:

1. ⛔ **No cost or power figure without measuring one first.** Nine for nine optimistic across both seats.
2. ⛔ **Tier-1 falsifiers with pre-committed consequences**, and **exhaustive decision tables** — Phase 1's `F1⁗` had no branch for the outcome that occurred.
3. ⛔ **The sweep rule:** before any scalar is reported, sweep the free parameter it depends on and show the family does not cross the value being tested.
4. ⛔ **Provenance before reuse:** a value's setter answered some question; if that differs from the question now asked, re-derive. Tags propagate through inputs that **stand in for** the population a claim is about.
5. ⛔ **Record search, four places:** `reports/`, `scripts/`, **metric definitions inside scoring functions**, and anything marked *"superseded"* — which retires a conclusion, never a tool, an instrument or a falsifier.
6. ⛔ **The executing seat does not decide which of its own objections survive, and neither does the authoring seat.** Phase 1's close was refused once, correctly, and Tier 2 changed the answer twice.
