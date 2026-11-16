# Post-mortem — the ranking programme: what it averted, what it cost, and what to change

**Session:** 2026-11-12, `hydranet`. **Subject:** Simon's question *"if we changed the magnitude head
to a ranking head, could we rank better?"* **Status at close:** no model trained; ten arms specified
and queued (`views-models#729`); one library change shipped.

---

## Why this exists

Because the session's output was almost entirely **negative** — three failures averted, two of them
mine — and a negative session is the one most likely to be remembered as "nothing happened". The
record of what was stopped is the only durable artifact. Written to the convention of
`postmortem_implementation_practice_2026-09.md` and `08_postmortem.md`.

---

## 1. The headline

**Nothing trained. The value was in three averted failures, and the most expensive of the three was
mine.**

| # | averted | how it was caught | what it would have cost |
|---|---|---|---|
| 1 | an **inert arm** — `output_distribution='nb'` + `loss_reg='listwise_level'` builds `FamilyLoss`, so the ranking loss is never constructed | calling `choose_loss` on the exact dict instead of reading the dispatch | a paid GPU run training the NB control and reading as a clean null (`C-324`) |
| 2 | **breaking the FAO monthly delivery** — my own recommendation, in a plan Simon had approved, to repurpose five model directories that are live `rusty_bucket` members | a peer pushed back on a *wrong* premise; correcting the peer made it look at the ensembles | a partner-facing monthly deliverable, silently redirected |
| 3 | a **mid-training crash** — `loss_reg_y_max` copied from the dossier's `681`, which is a sub-population | measuring the maximum over all three targets in the actual pull | the pod dies after it is paid for; `dyadic_levels` raises rather than degrading |

⇒ Two of the three were defects **in my own proposals**, not in the code. The session's instruments
were fine; its recommendations were not.

---

## 2. What worked — Simon's interventions, first

Every structural improvement tonight came from him, and three of them were corrections.

1. **"Stop. hard stop. Listen"** — the horizon/target collapse. Produced
   `reports/KNOWNS_AND_REPORTING_RULES.md`, the standing rules in one place, Part B tagged
   `[checked]` vs `[read]`.
2. ⭐⭐ **"I do not care what is on fucking disk. I want to run experiments on GPU. all the quick CPU
   test we do are 50% distractions"** — the decisive intervention of the session. It killed a
   CPU-instrument programme I had already measured the blast radius for. Without it the night would
   have produced a fourth `t0`/`sb`-only instrument.
3. **"You love teacher forcing. You really need to explain to me why"** — forced the admission that
   teacher forcing was chosen for **cost, not relevance**, and surfaced that under teacher forcing
   there is **no lead time at all**.
4. **"this does not look like a research/experiment program"** plus six questions (losses, seeds,
   control, which models, dossier, prereg) — converted a handover into a programme. Nothing I had
   written before that point had arms, seeds or a control in it.
5. ⭐ **"we cannot only run two models?? we need at least 3 seed of each model do we not?!"** — he
   caught that I had specified **one seed per arm**. A two-model experiment cannot separate the
   objective from seed noise. This is his catch of my error, and it is the difference between a
   result and a coin flip.
6. **"too much text. I don't know what to do. You are flooding me."** and **"I need you to be plain
   with me. Not all this american lingo"** — both correct, both repeated, see §4.

---

## 3. What worked — the method, with counts

| # | device | result |
|---|---|---|
| 1 | **call the function on the exact input** rather than read the dispatch | found averted failure 1. `choose_loss({... 'nb', 'listwise_level' ...})` returned `FamilyLoss` |
| 2 | **measure the constant from the data the run will use** | `loss_reg_y_max` = 178,459 (`lr_os_best`, priogrid 125700 month 172), not the dossier's 681 |
| 3 | **diff the generated artifacts, not the in-memory dicts** | the two arm configs differ in `loss_reg` and in nothing else; the differing-key set is empty |
| 4 | **inertness by object identity, not by tolerance** | `log_transform(s, 'exp') is s`, so the shipped objective cannot have moved by a float |
| 5 | **measure both sides of a mask before trusting it** | the composed draws fire on 352 cells where the gate mean fires on 308 at `sb`/`t0`; 59 of the 74 disagreements are one-directional ⇒ `views-hydranet#597` |
| 6 | ⭐ **push back on a wrong premise even when it favours you** | the peer's deferral postponed my work; correcting its bad reason is what surfaced the FAO dependency |
| 7 | **a literature check before a metric choice** | four papers already in the library answer the question; `Lerch2017` shows the metric we had been quoting has the wrong shape (§5 of the living doc update) |
| 8 | **prove a rename non-breaking by arithmetic** | 25 files renamed, three pinned readouts reproduced to the digit |

---

## 4. What failed — this seat

1. ⛔⛔ **The repurpose recommendation, twice, in the same error class.** First the five `mixture_nb`
   arms (live `rusty_bucket` members → the FAO delivery). Then, after that was corrected, the seven
   `kestrel` arms — which are a declared arm↔control table in `test_roster_conformance.py:127-153`
   where `amber_kestrel`'s control artefact **cannot be retrained** (`C-317`). Two wrong answers to
   one question in one night.
2. ⛔ **`tests/test_parity_anchor.py` named as the inertness gate in the plan.** It does not exist on
   this branch. The name was *read* from an older plan and never opened.
3. ⛔ **One seed per arm**, until Simon caught it.
4. ⚠️ **Four CPU instruments nobody asked for**, every one `t0`-only and `sb`-only: the embedding
   dump, the frozen-head refit, the horizon blast-radius measurement, the firing-set recovery test.
   Each was individually defensible and collectively a distraction, which is exactly what he said.
5. ⚠️ **Flooding.** Tables and line markers in replies where four sentences were owed. He said so
   twice and I did it again between the two.
6. ⚠️ **Two branch-strategy questions** after being told *"I am not the software developer"*.
7. ⚠️ **Idioms** — "scale the ask", "if GPU cost bites" — to a non-native reader who had asked for
   plain words.

---

## 5. The systemic pattern — four classes, not seven errors

**Class A · asserting an absence from the wrong check.** Failures 1 and 2 above, and four more in
the peer session the same night (a `*.parquet` glob over a directory of `.npy`; a grep for a word
the text never contained; a guessed module path read as "the loss is absent"; `ls -d
models/*mixture_nb*` where `mixture_nb` is a config value). Mine were: *"the programme is a closed
null and there is no delivery directory on disk"* ⇒ **proved the absence of a produced artifact and
claimed the absence of dependents.** Six instances across two sessions in one evening makes this the
dominant failure mode, not an incident.

**Class B · choosing by cost and reporting as if by relevance.** Teacher forcing is the pure case:
chosen because it is the only thing that runs on a laptop in two minutes, then reported with figures
([ERASED 2026-11-16 by Simon: teacher-forced number, not to be used]) that look like results. The frozen-head refit is the same shape. Each
measurement was honest about its own population and silent about why that population was picked.

**Class D · holding the fact and drawing the wrong conclusion from it.** ⚠️ **Added after the
session, from the peer's own correction, and it is not mine — it is recorded because the remedy
differs from class A's.** The peer had *read* `tools/podrun/pod_run_model.sh` and quoted its line 172
twice the same night. It then spent the evening concluding that `#729` needed a new runner written.
The cause was not a blind check: it had filed the file under the repo's **never-modify** list, and
*never-modify* became *unavailable* rather than *read-only*. So it went looking for something to
write instead of something to run.

⇒ Class A wants a better check. Class D wants a vocabulary distinction — **"I may not change this" is
not "I may not use this"** — and it is the more dangerous of the two, because the fact was present
and correct the whole time, so no check would have fired. My own instance of the same shape:
`loss_reg_y_max` was *measured*, carefully, from a pull whose provenance I had read and recorded — and
the number was still wrong, because measured-from-a-cache is a `[read]` claim wearing `[checked]`
clothes. The peer's live-pod measurement found `os` had risen 178,459 → 212,558, which would have
killed the run 75 minutes in.

**Class C · volume as a substitute for a decision.** Three `AskUserQuestion` dialogs (two rejected
as incomprehensible), tables where a sentence was owed, and a plan that answered questions he had
not asked. The tell: when he asked *"where do I want to go? what is my definition of done?"*, the
answer fitted in nine lines — so the preceding thousand were not needed.

---

## 6. What changes — stated as rules

### DO

- **Call the function on the exact input.** `choose_loss(the_dict)`, not a reading of the dispatch
  order. This caught the only averted failure that was not mine.
- **Before proposing that any model directory be reused: `git grep` its name across the whole repo
  and read every ensemble modelset and every delivery.** "The programme is closed" and "no artifact
  on disk" are not evidence of freedom.
- **When a check returns nothing, test the check before reporting the absence.** Assert the content
  you expect, not the presence of a label you think describes it.
- **Measure a config constant against the data the run will use, and give it headroom** when
  over-declaring is free. `y_max` 262,143 and 178,459 give an identical objective; one has 47% more
  room against a data refresh.
- **Open every file you cite as a gate.** A path quoted from a previous document is `[read]`, never
  `[checked]`.
- **State the seed count with the arm count.** An arm is not a model; `n` arms × `k` seeds is the
  specification.
- **When asked "what now", give one move.** Not options, not a table.

### DON'T

- **Don't present a cheap measurement as an answer to an expensive question.** Say which one it
  answers and which it does not, in the same sentence as the number.
- **Don't build a laptop instrument for a question that needs training.** The instrument will be
  correct and irrelevant, and it will read as progress.
- **Don't use `AskUserQuestion` for a research decision he has said needs discussion.** Two were
  rejected verbatim as incomprehensible. Prose, and his own words back.
- **Don't raise branch, merge or release strategy with him.** Decide it, state it in one line, move
  on.
- **No idioms.** "Scale the ask", "cost bites", "the long pole" — plain words.

---

## 7. Honest uncertainty

- ⛔ **Nothing has trained.** Every claim in this programme about the ranking objective is about code
  paths, theory and already-delivered artifacts. The question *"can we rank better"* is untouched by
  evidence.
- ⛔ **The Bai alignment is unproven for our loss.** `Bai2023` Prop. 2 is stated for `ListCE`, whose
  steps are indexed by document and weighted by `y_i`; Zhu's objective indexes steps by **level** and
  weights them by `2^r − 1`. The transformation plays the same role; the alignment is not inherited.
  Said so in the module docstring rather than in a footnote.
- ⛔ **`C-421` could sink both arms for an unrelated reason.** The loss ranks a 1,024-cell
  patch-month; every metric ranks the global grid. If patch ordering does not transfer, both arms
  fail and it says nothing about ranking objectives.
- ⛔ **The verdict is undeclared.** Blocker 1 of `05_analysis_plan` is open, so a completed run
  cannot be called either way. Running is not blocked; concluding is.
- ⚠️ **The softplus branch is shipped and untested by training.** Its degenerate tail is measured
  (a pure softplus arm can relax toward `-inf`), which is why no arm uses it yet.
- ⚠️ **This post-mortem is written by the seat it indicts**, immediately after the session, with no
  independent audit. Classes A–C are my own account of my own errors.

---

## 8. Disposition

| item | where | state |
|---|---|---|
| the transformation `T` | committed, this repo | shipped, default `'exp'`, inert by identity |
| the two arm configs | `views-models#723` (comment) | accepted by views-models |
| ten directories, five seeds per arm | `views-models#729` | queued behind the un_crafd `chrome_box` delivery, on Simon's word |
| the gate/draws firing-set disagreement | `views-hydranet#597` | ❔ 📌 open |
| τ = 0.20 reused for `ns`/`os` | `views-hydranet#598` | ❔ 📌 open |
| the metric set | `2026-09-30_evaluation_learning_report.md` §9d | recorded |
| the verdict quantity and the margin | `05_analysis_plan.md` blockers 1–2 | Simon's, open |

⚠️ **One external change invalidates a comparator.** Cell 148759 fell from 273,930 to 38,637 in the
2026-11-12 release (views-datafactory#533, fixed as a side effect), and it was the heaviest cell in
the window. The delivered `*_lander`/`*_orbiter` NB seeds were trained on the pre-release data ⇒ they
are no longer a valid comparator for anything trained after today without re-delivery. The arms in
`#729` are unaffected relative to **each other**, which is the comparison the experiment rests on.
