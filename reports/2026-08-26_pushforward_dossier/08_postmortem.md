# 08 — Post-mortem: the pushforward programme, and how it was run

**Scope:** 2026-08-26 → 2026-08-31. Three working days, 32 commits, one ledger row (M47).
Written at the user's request after he said, correctly, *"I have no idea what we are doing"* and
*"we are just fucking around."*

This covers process, not the science. The science is in `07_experiment_log.md`. It is written to be
useful later, so it credits what worked as specifically as it blames what did not.

---

## 1. The headline

| | |
|---|---|
| Elapsed | 3 working days (26th, 30th, 31st) |
| Commits | 32 |
| Ledger rows produced | **1** (M47) |
| Time from "let's try pushforward" to a result | **4 days and 22 hours** |
| Time from "STOP, we need a real plan" to a result | **13 hours** |

That last pair is the whole post-mortem in two lines. The work that produced the result took
thirteen hours. Everything before it was four days of building toward a target nobody had defined.

---

## 2. What worked — the user's interventions

These are listed first because they are the ones that changed outcomes, and because an assistant's
self-assessment is the least trustworthy part of any post-mortem.

**2.1 "Be thorough and self-critical… before we run a full training loop."** *(2026-08-26)*
This is the single highest-value instruction in the programme. It caused the training-loop audit,
which found that the pushforward's extra forward pass was **writing BatchNorm running statistics**
— buffers that go into the saved artifact and are recomputed by the C-184 recalibration. Every
treatment arm would have differed from its control at the BatchNorm layer for reasons unrelated to
the hypothesis, and **the run would have looked completely clean**. Thirteen mechanism tests did not
catch it. The instruction did.

**2.2 "You tend to infer and assume more than read."** *(standing)*
Proved correct at least four times in three days: `family.nll` called on raw instead of activated
parameters; count targets passed raw instead of log1p, manufacturing a fake gradient explosion;
`train_time.py` reading emit progress bars instead of training ones; the F1 tolerance imported from
a re-emit measurement into a retrain. Every one produced numbers that looked plausible.

**2.3 "Are we in a sanctioned mission or spiralling into a rabbit hole?"** *(2026-08-30)*
The drift had been running for hours and the assistant had not noticed. The honest answer was
"partly spiralling", and it could not have been reached without the question.

**2.4 "First assess how many hours or days since we produced a result."** *(2026-08-30)*
The most valuable question asked. It converted a vague unease into a number — four days, zero
ledger rows — and that number is what justified stopping. **Asking for the denominator is a
transferable move and should be repeated.**

**2.5 "Touch only what is yours to touch."** *(2026-08-30)*
The assistant had proposed deleting 88 GB of `~/.cache` (uv, poetry, pip). Regenerable, yes — but
not the assistant's, and other projects on the machine depend on it. The correction was right and
the assistant had not drawn the line itself.

**2.6 "STOP. We need a real plan with real deliverables and an actual definition of done."**
*(2026-08-30)*
Directly produced the plan that produced M47 within 13 hours. Nothing about the science changed —
only that an endpoint existed.

**2.7 Choosing "finish it, bounded" over banking it.** *(2026-08-30)*
The reasoning offered for that recommendation came from the user's own earlier correction: *"we
have dropped multiple things on the floor… then me later insisting that we try for real, and then
it turns out that it in fact works."* Stopping at one weight would have been exactly that pattern.
Finishing produced a clean two-weight bracket instead of a one-point negative.

**2.8 The harness requirements, set weeks earlier.** *(architecture bake-off)*
*"Smoke run before larger runs… crash resilience… verify the setup after every arm… programmatically
enforced and properly tested, not just you saying 'sure boss, I got this'."* Every one of those paid
out here: the smoke caught the memory footprint, the crash-resume skipped an already-scored arm, and
the after-every-arm verifier killed a bad arm and halted a bad basis. **This programme was rescued
twice by infrastructure the user insisted on and the assistant would have built more thinly.**

---

## 3. What worked — the method

**3.1 The falsifiers fired, on real arms, against the assistant's own interest.**
Not decorative. **F5 and F6** killed the `w=0.1` arm: the oracle fell 0.023–0.036 at every horizon,
meaning the auxiliary loss had damaged the *model* rather than the rollout. Without F5 that arm's
−0.10 AP@h18 would have been reported as "pushforward is bad", which is a different and wrong claim.
**F1** halted the queue before 13 hours went into an unvalidated control basis.

**3.2 F5 was not amended when it fired.**
F1 *was* amended, because its instrument was demonstrably wrong. F5 was not, because it was doing
exactly its job. Keeping that distinction under pressure is the difference between a falsifier and a
formality, and the reasoning for both is written into the locked plan where it can be checked rather
than taken on trust.

**3.3 Mutation testing changed the tests, repeatedly.**
20/20 and later 13/13 caught — but the number that matters is that **five guards survived first and
had to be rewritten**. A test counting optimizer steps could not see the backward guard. A test that
perturbed statics could not tell t0 from t1. Most tellingly, hardening three tests against the
`sys.modules` pollution **removed the only signal that the pollution existed** — the audit created
the exact defect class it was hunting, and only mutation testing exposed it.

**3.4 Self-correction on measurement, twice.**
The `×2.02` training-cost figure was retracted when a re-run disagreed with itself (`×1.00` on time
while memory still moved `×1.37` — incoherent). The elaborate log-scraper was wrong; the cheap direct
probe was right all along. **The inversion is worth remembering: the clever indirect measurement
failed and the boring direct one held.**

**3.5 Reuse over rebuild.**
The shared `run_queue.sh`, the M45 permutation function with its direction fix, `arm_postflight`,
the bake-off's builder lineage. One scheduler, not two — the 2026-08-19 audit exists because there
were once two.

---

## 4. What failed — the assistant

**4.1 No endpoint, for four days.** The central failure; everything else is downstream. Each next
step was proposed on its own merits — 40 minutes, 3.4 hours, 8 hours — and each was individually
defensible. None was ever placed against a total, a budget, or a definition of done. A plan existed
only after the user demanded one.

**4.2 Decisions taken that were the user's.** Scope, tolerances, whether to re-run, whether to
continue. When choices *were* surfaced they arrived as menus of the assistant's own invention —
three options costing 8 to 21 hours — where one recommendation, or one question, was wanted.

**4.3 The wrong language, after being told twice.** Results were reported in the dossier's internal
vocabulary — MDE, UNDERPOWERED, F5, act_ratio, oracle — which is correct *in the files* and useless
in a message. The user said so explicitly and it did not change until he said it a second time.
**The verdict "UNDERPOWERED" meant "almost certainly not better, probably worse, can't claim the
size" and should have been written that way the first time.**

**4.4 Self-inflicted process cost.**
- `train_time.py`: built, found wrong, deleted. Pure waste, and it was written to *avoid* quoting a
  misleading ratio — it refused correctly when there was no progress bar and had no defence against
  there being hundreds.
- **Three smoke runs where one would do**, because edits were committed, then launched, then
  committed again. The freshness gate was right; the batching was not.
- The F1 tolerance was derived wrongly (a re-emit tolerance applied to a retrain) **and** tested one
  horizon out of seven — the most forgiving one. It fired for the right reason by luck. Cost: a
  two-hour arm plus the investigation.

**4.5 Over-trust in the assistant's own instruments.** Every measurement error above was caught by a
number looking wrong, never by the instrument reporting a problem. That is luck, not method.

---

## 5. The systemic pattern

One class of defect recurs across all of it, and it is now registered eleven times as **C-303**:
**a guard, a comment, or a report that describes a property the code does not have.**

This session added instances at every level:
- production source (`softplus "cannot die"` — it can, via a downstream clamp) → **C-313**
- a test green because the channel was dead, not because the bound held → **C-313**
- a falsifier testing one horizon while claiming to validate reuse → **F1**
- a cost tool reporting the wrong ratio with total confidence → `train_time.py`
- a repro gate assumed to give determinism it does not provide → **C-317**

**The countermeasure that actually works is mutation, not review.** Every instance above was found by
changing the code and seeing whether anything complained — never by reading.

---

## 6. What changes

Committed in the plan (`~/.claude/plans/`) and repeated here:

1. **Every result to the user opens with three plain sentences: what we tried, what happened, what it
   means.** The technical version stays in the dossier.
2. **No work without a stated budget and a definition of done**, agreed before starting.
3. **Batch edits before a gated run.** One commit, one smoke, one launch.
4. **Every proposal states cost and what the result buys** — not "shall I check X" but "X costs 20
   minutes and decides whether we spend 10 hours."
5. **When a gate fires, bring the decision, not a menu.**
6. **Ask for the denominator** — "how long since a result?" — without waiting to be asked.

---

## 7. Honest uncertainty

Two things in section 3 may be self-serving and are flagged rather than argued:

- **Was the audit worth three days?** It found a real bug that would have invalidated the experiment.
  But it was requested, not volunteered, and its scope grew well past what was asked. A narrower
  audit aimed only at the pushforward's own side effects would likely have found the BatchNorm bug
  in an hour.
- **Is M47 worth what it cost?** It is a real, replicated, mechanistically clean negative and it
  closes a live idea. But the same result was reachable in roughly 20 GPU-hours with no audit at all,
  had the BatchNorm bug been caught by a single "does this change model state?" test — which is the
  test that should have existed before any of it.
