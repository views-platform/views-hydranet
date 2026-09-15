# Post-mortem — implementation practice: what caught defects, what did not, and what to change

| | |
|---|---|
| **Date** | 2026-09-05 |
| **Span** | 2026-09-03 → 2026-09-05, issues **#308** (closed) and **#311** (closed) |
| **Companion to** | `2026-09-03_bptt_sa_dossier/`, `2026-09-04_input_noise_dossier/`; ledger **M61–M64**; register **C-324 (Tier 1)**, **C-325**, **C-326**, **C-327**, **C-328** |
| **Status** | PROCESS post-mortem. Both hypotheses were answered; the science is in the logs. This is about how the implementations behaved. |
| **Method** | Read-only re-analysis of three independent mutation audits, five register entries, and the run logs. No new training. |

## Why this exists

> *"I need you to read the post mortums and read the relevant papers and really really look at the
> code base here to get it right; audit, test, mutate, double check, etc."* — and, earlier: *"your
> error rate is so high that I don't know how to know if the implementation is correct… I estimate
> that we have rejected many ideas that would have improved performance due to wrong
> implementations."*

That estimate is the thing worth testing. This document collects the evidence from two experiments
run under deliberately heavy verification and asks: **which checks actually caught defects, which
did not, and what should change.**

**Boundary:** process and verification only. The scientific record is in the two dossiers' `07`
logs. Nothing here revises a result.

---

## 1. The headline

| | |
|---|---|
| Experiments completed | 2 (#308 BPTT-SA, #311 input noise) |
| Independent audit rounds | 3, each in a clean context by a non-author |
| Mutation scores, in order | **52.8% → 81.3% → 80.4%** |
| Rounds that found real defects | **3 of 3** |
| Register entries opened | **5** (one Tier 1) |
| Defects found **after** I judged the code correct | **at least 15** |
| GPU wasted on a defect that shipped | **276 min** (#308's inert implementation) |
| GPU that would have been wasted but for a late catch | **~4 h** (#311's BatchNorm leak) |
| Defects found by my own review | **0** |

**The single most important number is the last one.** Across three rounds and five register
entries, my own review caught nothing that independent checking did not. Every defect in this
document was found by a mechanism, not by me looking harder.

---

## 2. What worked — the chair's interventions, first

Listed first because an assistant's self-assessment is the least trustworthy part of any
post-mortem, and because in both experiments the decisive intervention was a question, not a fix.

* *"how can we know it is fixed? … your error rate is so high that I don't know how to know"* —
  produced the **potency gate** and **C-324**. Without it, #308's re-run would have had no
  precondition and the second inert implementation would have been indistinguishable from a null.
* *"I want a real check! I don't need a cheap check if the knowledge generated is not valid."* —
  killed a cheap probe that would have produced an invalid answer.
* *"third audit"* — I had proposed stopping at two. Round 3 found **C-328**, the BatchNorm leak that
  would have confounded the entire #311 screen. **I was wrong about when to stop, and the cost of
  being wrong was the whole experiment.**
* *"fix them all then re-run the audit"* — the re-run is what proved the fixes worked (9 of 9 gaps
  closed, verified in a throwaway copy) rather than being asserted to.

---

## 3. What worked — the method, with counts

| mechanism | evidence from this span |
|---|---|
| **Independent audit, clean context, non-author** | 3 rounds, 3 hits. Round 2 found errors introduced *by round 1's fix*. The auditor was denied commit messages, dossier and design docs on purpose — it must not inherit the author's model of failure |
| **Turning untested prose into a test** | Found **C-328**, which three rounds of mutation testing missed. The CIC asserted "the biopsy is never noised"; writing that assertion down as a test failed immediately, on a different call path |
| **The deletion test** — remove the feature, see which tests still pass | Found 2 vacuous assertions and 1 zero-power test. A test that survives the feature's removal is not testing the feature |
| **Reading the writer instead of assuming its output** | Caught `-s` = `--sweep` (every arm would have written no artifact) *before* launch; the score-CSV filename bug *after* it cost one aborted chain |
| **A result being impossible** | #308's two arms trained to byte-identical weights. Luck, and it must be named as luck — a *half*-connected knob would have produced a plausible number |
| **Control arms** | GRAD-TRAJ's control is what made the treatment's curve legible (it *fell* 859→312 while the treatment rose to 9.4e9). `floor_gate` on the control, at zero extra GPU |
| **Potency gate at a trained checkpoint** | C-324 + C-325. #311's knob moved the loss 265.80 → 106.27 on the arm's own config |
| **Proving the harness can fire** | The round-1 mutation harness reported *every* mutation as surviving — it was swallowing output. Noticed only because everything surviving is as implausible as identical weights |

---

## 4. What failed — the assistant

| # | error | why it survived my checking |
|---|---|---|
| 1 | Tested the **helper**, not the **seam** | 17/22 helper mutations caught, **2/22 seam mutations caught**. Unit tests feel like coverage |
| 2 | Derived a test's expected value from **the expression under test** | The expectation and the mutant agree by construction. Done **twice** — the second time *inside the fix for the first* |
| 3 | Asserted `all(...)` over a collection never checked non-empty | Vacuously true. Both such tests passed with the feature deleted |
| 4 | Asserted two tensors had the same shape — which they always do | Cannot fail. The docstring named a tell the assertion did not check |
| 5 | A reproducibility test with **zero discriminating power** | Passes for any deterministic implementation, feature present or absent |
| 6 | Docstring claimed a fix closed a survivor it did not | C-303 **inside the C-303 fix** |
| 7 | Docstring cited a test file that does not exist | Prose is unchecked by definition |
| 8 | Assumed a filename convention instead of reading the writer | Twice: #308's scorer, then #311's chain |
| 9 | Assumed `-s` meant "save" | It means `--sweep`, which saves nothing |
| 10 | Cited a **memory note** instead of the artifact | "pushforward has never been run" — it had run twice, the second time at 4 seeds |
| 11 | Wired a **harness** failure into a **scientific** branch | The chain announced "BRANCH 0: VOID" when the emit had succeeded |
| 12 | Reported lint green when the repo's own gate was red | I ran ruff myself instead of running the gate |
| 13 | Ruled a mechanism out by measuring an **untrained** network | C-325. The numbers were correct; the conclusion was not |
| 14 | Let a training augmentation reach the **BatchNorm recalibration** pass | C-328. Would have confounded #311 at the BN layer with every log clean |
| 15 | Pre-registered a "discriminating" test whose two hypotheses predict the **same** outcome | C-327. Two rows pointing opposite ways *looked* like a design |

---

## 5. The systemic pattern — three classes, not fifteen errors

**Class A — verified on a path or state the phenomenon never occupies.** C-323, C-324 (Tier 1),
C-325, C-328. Four register entries, and the dominant class by cost. The helper instead of the seam;
the untrained network instead of the trained one; the main forward instead of the auxiliary one.
Every instance had *correct measurements* and a *wrong conclusion*.

**Class B — prose asserting what the code does not do.** C-303, now at ten-plus occurrences and
extended this span by items 6, 7 and 12 above. Docstrings, verdict strings, commit messages, and a
plan describing its own provenance. **Prose is the only interface most readers have to a rule, so a
false description is a false result that no test catches.**

**Class C — tests that cannot fail.** Tautology, vacuity, zero power (items 2–5). Not previously
named in the register. The third audit made it visible by asking a question mutation testing does
not: *would this test pass if the feature were deleted?*

Class C is the one that most deserves a name, because **it is invisible to mutation testing by
construction** — a test that cannot fail cannot be shown to fail by mutating the code it does not
examine.

---

## 6. What changes — the practice, stated as rules

### DO

1. **Test the seam before the function.** Where does production *assemble the arguments*? That layer
   first. Evidence: 2/22 vs 17/22.
2. **Run the deletion test on every new feature.** Delete the config field, the call site, and the
   assembly; run the tests. Anything still green is not testing the feature. Cheap, and it found
   three non-tests immediately.
3. **Turn every prose invariant into a test.** Any CIC line, docstring or comment that says the
   system *does* something. This found the one defect three mutation rounds missed.
4. **Have a non-author audit in a clean context**, denied the commit message, the dossier and the
   design docs. 3 rounds, 3 hits, including errors introduced by the previous round's fix.
5. **Read the writer before consuming its output** — filenames, flags, formats. Three defects here
   came from assuming instead of reading, and reading takes a minute.
6. **Check the artifact, not the note about the artifact.** Memory notes go stale; this one was five
   days old and would have cost an unnecessary arm.
7. **Measure at the state the phenomenon occupies.** An untrained network cannot exhibit what
   training creates.
8. **Prove a gate can fire before trusting its silence.** Fire every branch on synthetic fixtures
   before real data exists; include a sanity mutation that must be caught.
9. **Ask which *other* forwards traverse the path you just changed** — auxiliary losses, diagnostic
   probes, recalibration passes. C-328 and #289's BN bug are the same question unasked.

### DON'T

1. **Don't derive an expected value from the expression under test.** If the test computes
   `list(range(len(features)))` and so does the code, the test agrees with every mutant.
2. **Don't assert only that something *changed*.** `off != on` survives almost any corruption.
   Assert what it changed *to*.
3. **Don't write `all(...)` without first asserting the collection is non-empty.**
4. **Don't wire a harness failure into a scientific branch.** "No artifact" is not "the hypothesis
   failed" unless you have proven the arm actually ran.
5. **Don't claim a fix closes a defect without a test that fails when the fix is removed.**
6. **Don't edit shared source between two arms of a comparison** — not even a docstring. They must
   run identical code.
7. **Don't report a gate green from your own invocation.** Run the gate the repo runs.
8. **Don't stop auditing because the score plateaued.** Rounds 2 and 3 scored the same (81.3%,
   80.4%) and round 3 found the worst defect of the span.

---

## 7. The gate that should have existed

Every item in §6 is cheap; none was skipped for cost. They were skipped because **nothing asked for
them at the right moment.** The register's own structural finding applies: *no gate in this repo is
repo-wide; each is opted into per launcher, and a dossier that forgets one gets no warning.*

The minimum durable change is a **pre-run checklist in the dossier's `03_harness`**, with these four
lines added to the existing set, because all four are mechanical and each caught something this span:

- [ ] deletion test run; no test survives the feature's removal
- [ ] every prose invariant in the touched CIC has a test that fails when the invariant is broken
- [ ] independent audit by a non-author in a clean context; survivors dispositioned in writing
- [ ] the repo's own lint/clean gates run **as tests**, not as my own command

**What is not proposed:** a new framework, a new abstraction, or a general-purpose verification
layer. The evidence says the failures were of *attention at a specific moment*, not of missing
machinery.

---

## 8. Honest uncertainty

* **Three audit rounds is not "enough" — it is where we stopped.** Round 3 still left 21 survivors,
  9 provably equivalent and 12 genuine. Score plateaued at ~80% while severity did not decline. **I
  do not know the shape of the tail**, and the one datum I have — that round 3 found the worst
  defect — argues against assuming it is thin.
* **The chair's estimate may still be right.** *"We have rejected many ideas due to wrong
  implementations."* This span produced two clean negatives, but #308's first implementation *was*
  inert, and only an impossible result exposed it. **How many earlier negatives rest on
  implementations nobody audited this way is unknown, and this document does not establish that they
  are sound.**
* **Self-report bias.** §3 and §6 are my account of what worked. The counts are real; the selection
  of what to count is mine, and §2 exists because the interventions I did not choose were decisive.
* **The BatchNorm catch was partly luck.** I wrote the biopsy test for a *different* invariant and
  grad state happened to be the tell. A test aimed directly at the BN pass would have been better,
  and I did not think to write one.

---

## 9. Disposition

| what | disposition |
|---|---|
| The four checklist lines (§7) | **Adopt** into the standing pre-flight; they are mechanical and each has evidence |
| Class C — "tests that cannot fail" | **Register as a new class.** It is invisible to mutation testing by construction and is not yet named |
| `input_noise_dropout`, `ss_feedback_grad_clip` | Stay merged, default-off, tested. Both are seams a follow-up needs |
| Three-round audit as the default for a new training-path feature | **Adopt**, with the caveat in §8: three is where we stopped, not a demonstrated sufficiency |
| Earlier negatives (pre-#308) | **Not revisited here.** Whether any rest on inert implementations is open, and it is the most expensive open question in this document |

---

*The transferable lesson: every defect in this span was found by a mechanism that did not trust the
author — an independent auditor, a deletion, a control arm, an impossible result — and none was found
by the author checking more carefully.*
