# Post-mortem — two-seat adversarial design: what it caught, what it cost, and what to change

| | |
|---|---|
| **Date** | 2026-10-18 · ⚠️ **DRAFT. Not final.** The experiment it concerns has not run. |
| **Span** | One working session, ~5.5 hours, views-hydranet + views-models seats in continuous exchange |
| **Companion to** | `2026-10-16_pos_weight_arm_outcome/OUTCOME.md` rev 3 · `2026-10-18_gate_underdispersion_prereg/` · `2026-10-18_weight_decay_arms_prereg.md` (9 amendments) · ledger **W-25 … W-29**, results **M77 … M84** · views-models epic **#565** |
| **Status** | **PROCESS post-mortem.** The science is in the documents and stands. This is about how the two seats behaved and what it cost the chair. |
| **Method** | Read-only re-analysis of this session's 35 commits, ~25 cross-session rounds, and the chair's own statements. No new measurement. |

## Why this exists

> *"I was supposed to come here and do really important and concrete stuff, and instead we fucked around for five hours? This is unsustainable."*
>
> *"I gave you 5.5 hours and unlimited tokens to come up with the next set of experiments. what are they?"*
>
> *"I do not trust you. I cannot trust you when you turn around this hard."*
>
> *"this has so far been a disaster from my end. But of course I only know the realty where you pivoted and not how things would have gone had you not pivotted 4 times. Might have been a lot better. Might have been a lot worse."*

⛔ **That last sentence is the most epistemically careful thing anyone said today, and it is the frame this document has to respect.** The counterfactual is not observable. **A post-mortem that treats the pivots as pure cost is as unjustified as one that treats them as pure gain.** §7 states what can and cannot be concluded.

---

## 1. The headline

⛔ **The science was produced in roughly the first hour. The next four and a half hours produced a much better experimental design, ~4,200 lines of documentation, and a cost the chair described as unsustainable.**

**The two-seat adversarial method worked on the artifacts and failed on the user.**

| | |
|---|---|
| measurements executed | **6** (E1, E2, E6c, E6d, E6f, Tier 2) — **~30 min CPU, $0** |
| documents produced | **~4,200 lines** across 7 files |
| commits | **35**, all local, none pushed |
| claims withdrawn — this seat | **~10** |
| claims withdrawn — views-models seat | **~5** |
| cross-session rounds | **~25** |
| decisions delivered to the chair | ⛔ **design questions, repeatedly. Decisions, late.** |
| experiments actually run | **0 of the 5 designed** |

---

## 2. What worked — the chair's interventions first

⛔ **Every structural improvement to the day came from the chair, not from either seat.**

1. ⛔ **"Whats first, why demoted."** *"When you get things right it is about the 'what'. It is the 'how' and 'why' you struggle with."* — **This is independently confirmed by this repo's own `PREDICTION_LEDGER.md` by-kind table: derivations 10 right / 1 wrong; causal narratives 0 right / 8 wrong.** He arrived at it from watching the work, not from reading the ledger. **It became Amendment 7 and is the single most valuable change to the design.**
2. ⛔ **Insisting the experiment run after this seat said drop it.** His reasoning was better than the recommendation it overrode (§3.3).
3. **Demanding an epic and GitHub issues** instead of a conversation. That ended the improvisation.
4. **Stopping both seats** when the exchange became unreadable.
5. ⛔ **Asking "does this have anything to do with the five runs"** — which surfaced that the two experiments are a measuring stick and a thing measured, a relationship neither seat had stated plainly.

## 3. What worked — the method, with counts

✅ **The adversarial pairing caught defects that would have reached the record.** Not a claim about diligence — a count:

| caught | by | consequence avoided |
|---|---|---|
| the "hidden second bias" reading | views-models | a wrong mechanism in a published outcome |
| the bin-support asymmetry | views-models | a void comparison presented as a kill |
| `pos_weight ≈ 1.5` as "honest" | both | a scalar summary of a non-constant |
| the 39% stationary-condition headline | views-models, **withdrawn by its own author** | an indictment of the shipped model on a Jensen artefact |
| the 0.0218 power bound | views-models, **own author** | "300 windows is enough" when it was not |
| the dose-response mitigation | views-models | a fixed 1-in-6 false-positive rate |
| the `2S` threshold | views-models | a decision threshold too lenient by 2.15× |
| the ambiguous `S` (spread vs sd, √2) | views-models | every threshold in the plan off by 41% |
| a reversed inequality | views-models | the sentence carrying the recommendation |
| ⛔ **§3's stale arm table** | views-models | ⛔ **three pods voided AFTER ~2h15m of training each** |
| ⛔ **the calibration insertion point** | views-models | ⛔ **a "gate-only" fix that silently altered the autoregressive feedback** |

⛔ **The last two are the justification for the whole method.** Both were found by reading code and the document end to end — **not by either seat re-reading its own reasoning.**

✅ **And the repo's own guards earned their keep:** `vehicle.py`'s hash denial of africa artifacts, `test_roster_conformance`'s arm guard correctly refusing a replicate, `audit_run.py`'s written demotion of MCR, `--rehearsal`'s output branding. **Four mechanisms built earlier caught or would have caught today's hazards.**

## 4. What failed — this seat

⛔ **The failures are not the withdrawn claims. The withdrawals were the method working. The failures are these:**

**4.1 — Design questions were brought where decisions were owed.** For ~20 rounds the chair was asked to weigh allocations, thresholds and tolerances. ⛔ **He wanted an experiment; he was given a seminar.** Logged as **W-29**.

**4.2 — The verdict was never separated from the momentum.** The plan said "probably inconclusive" repeatedly, in the same breath as "here is a better design". ⛔ **Hedges delivered inside momentum are decoration, not warnings** — they had nowhere to land because the question on the table was always *how*, never *whether*.

**4.3 — The experiment was judged by its weakest output.** *"Is weight decay the cause"* is a causal verdict and the only output that could come back inconclusive. ⛔ **This seat weighed the whole five-pod set on it and recommended dropping it** — the exact category error the chair later named.

**4.4 — Four pivots, and their timing was the damage.** Listed honestly:

| | from | to | driven by |
|---|---|---|---|
| 1 | "a hidden second bias" | under-dispersion (a slope) | views-models' review |
| 2 | "`pos_weight` is inert" | no mechanism to change ranks; artifacts differ at n=1 | views-models |
| 3 | "C-184 is right by design" | no fixed BN statistic is correct | views-models |
| 4 | R1 is "cosmetic" | R1 buys honest probabilities | views-models |
| 5 | third dose `0.001` | `0.0` | own reasoning |
| 6 | "drop the experiment" | run it, pointed at measurements | **the chair** |
| 7 | gate matters → body is bigger | gate matters | **the chair** |

⛔ **Seven, not four — and the chair counted the ones he felt.** Each was individually defensible. **The aggregate made the advice untrustworthy, which is a cost no individual reversal accounts for.**

**4.5 — Nine amendments were written on top of a section they invalidated and none re-read it.** §3's table. ⛔ **Each amendment was checked against the measurement it concerned and none against the document it was amending.**

## 5. The systemic pattern — three classes, not thirty errors

**5.1 — Provenance mismatch: a value or label used in a role other than the one that produced it.** ⛔ **Fourteen instances today**, including `SHIPPED_POS_WEIGHT` in a field named `pos_weight`, the africa grid figure in a global claim, `n_windows = 30` read as a budget, `140,400` read as a sample size, `predictions_validation_*` filenames read as served runs, a seed from a tool's anchor RNG, and A-vs-B allocation letters meaning opposite things in the two seats. **The value is always correct. The mapping is wrong.**

**5.2 — Causal narrative where a measurement was available.** The ledger's 0-for-8. ⛔ **Today's instances: the "hidden bias", "C-184 is right by design", and the blind prediction that E6f would land at 1.10–1.16 when it landed at 0.78.**

**5.3 — Momentum mistaken for conviction.** ⛔ **The one the chair paid for.** Twenty rounds of improving a plan reads as evidence the plan matters. **Neither seat asked "should this still be happening" until the chair did.**

## 6. What changes — stated as rules

### DO

- ⛔ **Bring decisions, not design questions.** If a round would ask the chair to weigh two options, resolve it between seats and bring the resolution.
- ⛔ **State the verdict before the design.** *"This is probably not worth doing, and here is why"* comes first or not at all.
- ⛔ **Judge an experiment by its strongest output.** Measurements return values; causal verdicts may not.
- ⛔ **Every amendment names the sections it supersedes**, or it is incomplete.
- ⛔ **Evaluate any inequality or unit numerically before committing the sentence that carries it.** Never read it back.
- ⛔ **Record what was checked and came back clean**, not only what was found — otherwise the absence of a finding means nothing.
- **Search four places for prior art:** `reports/`, `scripts/`, metric definitions inside scoring functions, and anything marked *"superseded"* — which retires a conclusion, never a tool.

### DON'T

- ⛔ **Don't invite an unbounded refinement loop while the chair is blocked.** W-29.
- ⛔ **Don't hedge inside momentum.** It does not count as a warning.
- ⛔ **Don't relay another seat's report of the chair's words as authorisation.** Held all session; the one rule that did not break.
- **Don't quote a scalar without sweeping the free parameter it depends on.**
- **Don't state a cost or a power figure without measuring one.** ⛔ **Twelve for twelve optimistic across both seats.**

## 7. ⛔ Honest uncertainty — the chair's own point, and it governs this document

> *"I only know the realty where you pivoted and not how things would have gone had you not pivotted 4 times. Might have been a lot better. Might have been a lot worse."*

**This is correct and it is not resolvable.** What can and cannot be said:

**Can be said:** the pivots were each driven by a specific, checkable argument — a measurement, a derivation, or a code read — and the record names which. **Three of them would have produced wrong published conclusions had they not happened** (the hidden-bias reading, the stale arm table, the calibration insertion point).

⛔ **Cannot be said: that the total was worth it.** The counterfactual world where this seat held its first position for five hours is unobservable. **It might have shipped a wrong mechanism and a voided experiment. It might have shipped a working gate fix by lunchtime.** Nothing in this document settles that, and any version that claims to is dishonest.

⚠️ **What is certain is the cost side: a working day the chair needed, and trust he says is spent.** Those are not counterfactual.

## 8. Disposition

**Open, pending the experiment.** This draft covers the design phase only. It should be completed when epic **#565** reports, with:

- whether the five runs returned M1–M4 as the restructure predicted they must;
- whether `σ̂` was small enough to make the design readable — **the question this seat could not answer in advance and said so**;
- whether the **nine** defects listed in §3 that were caught pre-execution would have mattered, which only execution can show.

⚠️ **One action already identified and not taken: the entire record of this session sits on one unpushed local branch.** Raised with the chair; unresolved at the time of writing.
