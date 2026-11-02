# The ADRs are written to their own author. Plan to fix that.

**Date:** 2026-11-02 · **Status:** plan, nothing executed

---

## The charge, in Simon's words

> *"It is 100% you explaining to yourself with your current context why this document looks
> different from its predecessors. It is not you explaining to anybody without your current context
> the decision we are here taking."*

He stopped reading ADR-075 at page 2 of 6. Correctly.

---

## 1. Diagnosis

### The failure mode

Each ADR written today was **edited in response to feedback, five or six times.** Every round added
a layer explaining why this version differs from the last one. The result is a document whose
content is partly a changelog of its own drafting.

A reader who was not present for the drafting cannot use that content. Worse, they cannot tell it
apart from content they *do* need, so it does not merely waste their time — it makes the document
harder to trust.

### The specific sentences, so this is not vague

From ADR-075 alone, in the first two pages:

| sentence | what it actually is |
|---|---|
| *"Scope note. Selecting which origins and horizons… is ADR-078."* | justifying a split made an hour earlier. §2.4's out-of-scope list already covers it |
| *"The existing body-mean diagnostic argues this case from a sampling composition's variance inflation… the conclusion is right and the stated reason does not apply."* | a correction to someone's docstring. An issue, not a decision |
| *"Assumption that is no longer true. ADR-069 assumed one composition per run…"* | positioning against a predecessor rather than stating a need |
| *"Relationship to ADR-078… They were originally one document; they were split because…"* | editing history |
| *"Respects: the standing constraint that no new emitting family is introduced"* | defensive, aimed at a rule the author was anxious about |

And the Summary spends three paragraphs on what ADR-069 did and did not do before reaching the
problem.

### The same disease as the rest of today

A document whose meaning depends on knowing its own history is **not self-contained** — the same
defect as an artifact whose identity depends on the directory it sits in, or an ADR that depends on
a config value. The template's own section heading is *"Summary (read this first —
self-contained)"*, and it is not.

### Root cause

**Editing rather than rewriting.** Feedback arrived, a passage was adjusted, and a justification for
the adjustment was added beside it. That is a reasonable way to answer a reviewer and a
catastrophic way to produce a document, because the justification is addressed to the reviewer and
stays in the artifact after the review ends.

---

## 2. The rule

> **An ADR is written for a reader who has never seen a previous version of it, has not read the
> ADRs around it, and was not party to the discussion that produced it.**

Operationally — these do not belong in an ADR:

| smell | where it goes instead |
|---|---|
| any comparison to a previous version of this document | nowhere. The git history holds it |
| any explanation of why a section is where it is, or why a split was made | nowhere, or one line in the commit message |
| any correction of another document's or another file's error | an issue |
| any defensive citation of a rule the author was worried about violating | nowhere |
| the author's own mistakes, and what they learned | the prediction ledger or a dossier postmortem |
| *"originally"*, *"previously"*, *"used to"*, *"I had"*, *"was rejected on review"*, *"this corrects"* | these words are the tell |

**The test, and it is a reading test rather than a checklist:**

> Could someone who has read none of the surrounding ADRs, and was not in the room, say what is
> being decided and why, from the Summary alone?

A rationale section may name and reject alternatives — that is what it is for. The line is whether
the alternative was *considered on the merits* (keep) or *was a previous draft of this document*
(cut).

---

## 3. What to do

### Scope: all five ADRs written today, not just 075

| ADR | status | why it is in scope |
|---|---|---|
| **074** | **Accepted** | carries *"And this corrects the account of my own error. I read a `lr_*` directory as the raw body"* — the author's mistake, in a governing document. Passed review before this complaint was articulated |
| **075** | Proposed | the document that triggered this |
| **076** | Proposed | carries forward 071's reasoning with commentary about the bundling fault |
| **077** | Proposed | its §1 is substantially about why the ADR did not exist until now |
| **078** | Proposed | records in §3 that bundling was tried and rejected |

### Method: rewrite, do not edit

Each one gets **written once, from a blank file**, starting from the plain statement of the problem
and the decision — not from the current text with the offending sentences removed. Editing is what
produced the defect; a sixth editing pass would add a seventh layer.

For 075 the plain statement already exists, in Simon's words, and it is the whole document:

> The model produces forecasts. A forecast is made of a gate and a body, which can be combined in
> various ways. Neither the gate nor the body can be fully reconstructed from the forecast. So for
> audit, evaluation, experiments and continuous improvement, we need to be able to get the gate and
> the body out independently. That is different from getting the forecast out — it is **emitting
> subcomponents of a forecast, for diagnostic reasons.**

⚠️ That is also a better title than the one it has.

### What each rewrite keeps

Only what a cold reader needs: the problem, the decision, the alternatives rejected on their
merits, the costs, how it will be verified, and where the evidence lives. Nothing about how the
document came to look the way it does.

### Order

1. **075** — the one that failed, and the one whose plain statement is already written.
2. **074** — Accepted, so the hygiene edit is legitimate but the *decision* must not move. If a
   rewrite changes what is decided rather than how it reads, it stops being an edit and needs
   Simon.
3. **078**, then **076**, **077**.

⚠️ **Not a rewrite-everything licence.** Each pass is a reading pass first: if a document already
passes the cold-reader test, it is left alone. 076 and 077 may be close.

---

## 4. Verification

| check | how |
|---|---|
| cold-reader test | the Summary, read alone, says what is decided and why. Judged by a reader, not a regex |
| no self-reference | no sentence in the document refers to a previous version of it, or to why it is structured as it is |
| no foreign corrections | corrections to other files have left the ADR and exist as issues |
| the decision is unchanged | for 074 specifically: a diff of *what is decided* before and after, confirming the rewrite moved only prose |
| the earlier rules still hold | zero `file:line`, zero model names, zero values, two emoji — the checks from `00_PLAN.md` |

⚠️ The first two cannot be mechanised and should not be faked. A regex for *"originally"* would
catch the word and miss the habit.

---

## 5. Risks

1. ⚠️ **A rewrite of an Accepted ADR can silently change what was decided.** For 074 the content
   must be held fixed while the prose changes, and that is a judgement a diff cannot make. If in
   doubt it goes to Simon rather than being decided by the author.
2. ⚠️ **This plan can commit its own sin.** It is itself a document about why other documents are
   wrong, written by their author. It belongs in `reports/`, not in an ADR, and the rule it
   produces — §2 — is the only part that should reach `ADR-000`.
3. ⚠️ **Five rewrites is a lot of churn for documents that are days old.** The alternative is
   leaving five governing documents that a new reader cannot use, which is worse — but the churn is
   real and is the author's fault, not the reviewer's.
