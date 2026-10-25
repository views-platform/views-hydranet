# What we have learned about evaluating HydraNets

> ⛔ **PROVENANCE CORRECTION, 2026-10-20 — "99.5%" was never our number.** Every "99.5%-zero" in this
> report (except the synthetic-fixture description, which keeps its own value) has been changed to
> **99.7%**, our own measurement. `99.5%` is **Hegre et al. (2024)**'s figure for the VIEWS Prediction
> Challenge — their PRIO-GRID, their period — confirmed verbatim in their source text (*"99.5% of the
> historical values are zero"*) by the `library` seat on request. Ours, measured: the observed positive
> rate on **all land cell-months is 0.003448 ⇒ 99.655% zero**, and on the **32×32 activity-seeking
> curriculum windows** the median positive fraction is 0.004508 ⇒ **99.55%**. Neither rounds to 99.5%.
> ⚠️ A third, different statistic is also in circulation and must not be substituted for these:
> **98.59% of land CELLS never fire at all** (D3, cells-ever-active, not cell-months). No argument in
> this report depends on which figure is used — the correction is made because a borrowed figure
> standing in for a measurement we hold is the same error class as
> `2026-10-20_d3_static_channel_ceiling.md`'s three self-inflicted ones.

**Date:** 2026-09-30 · **Last revised 2026-10-25** · **Status:** ⛔ **LIVING DOCUMENT — expected to carry open rows; not a verdict and not finished.** Consolidation only: **no new measurement by this document** (§0.2 and §5.0 carry tables measured 2026-09-30 by the `models` session, attributed there) ·
**Supersedes as the living document:** `reports/2026-08-13_evaluation_pitfalls_and_metric_battery_DRAFT.md`
(retained as dated provenance; its taxonomy letters A1–I4 are preserved here unrenumbered).

---

## What this document is for

**A living starting point, not a verdict.** Its job was to stop eight months of evaluation knowledge
being lost, and that is done. From here it is the base for three things, each of which should be worked
on its **own branch, cut from this one**:

| | downstream use | where it starts |
|---|---|---|
| 📄 | **an academic paper** on evaluating high-resolution conflict forecasts from both point-estimate and sample-estimate models | **§8.3** (the diagnostic suite, as a proposal) and **Appendix E** (the literature, including where it contradicts us) |
| 📐 | **new metrics** | **§5.8** (what is computed and read by nothing), **§8.3**, and `Appendix E.3`/`E.5.2` — three instruments the literature has that we built by hand, plus one usable score we are not using |
| ⚖️ | **a new evaluation scheme** | **§8** (what our own scars demand of a promotion gate) and **§1** (the two regimes) |

⛔ **Open questions are not blockers here.** This document is expected to carry unresolved items for
weeks; naming them is the point. It is **not** definitive, it does **not** need to be finished before it
is useful, and it merges with its open rows intact.

---

## How much weight each part carries

**Everything below has been attacked.** One clean-context adversarial review (2026-10-11) and **fifteen
independent falsification audits, one per section (2026-10-19/20)** — each auditor in a clean context,
denied the author's reasoning, required to record a risky prediction before every probe.

**Fourteen of fifteen sections came back FALSIFIED. None survived.** **143 findings, 80 of them hard
(~70 distinct after cross-auditor duplicates).** ⛔ **And zero
wrong measurements** — every ratio, percentage and verdict count any auditor recomputed reproduced
exactly, one of them to **17 significant digits**.

✅ **So the single most useful thing to know about this report:**

> **The numbers are reliable. The sentences about what exists are not.**
>
> Every defect found is a claim that something *is called*, *is never called*, *does not exist*, *is
> classified as X*, or *that the record says X*. Three audits independently named it **cluster 21 —
> *"prose with a self-dating expiry: true when written, named its own expiry condition, never re-read
> when that event landed"*** — the class this document itself catalogues.

| part | weight it carries |
|---|---|
| **Measurements, ratios, tables, CSV-derived figures** | ✅ **Trust them.** Fifteen audits, nothing wrong. |
| **§0 items, §4's verdicts, §5's blind spots, §7** | 🟡 **Attacked, corrected, usable** — read the dated correction blocks inline. |
| **§1's amendments 2, 3 and 4** | ⛔ **Do not rely.** Amendment 4 is **WITHDRAWN** (2026-10-20 — the diagnostic ban it asked to lift does not exist) and amendment 2's QS99 source is corrected. **Amendment 3 still fails at the strength stated** — it is a reasoning finding, outside the corrected tier. |
| **§2.3's instrument claims** | 🟡 **Corrected in place 2026-10-20.** The ARTIFACT verdict has ruled **once**, from a gitignored tool, on one archived board — the section now says so. |
| **Appendices A, B, C, D** | 🟡 **Their false statements are corrected (2026-10-20); their INCOMPLETENESS is not.** A still omits the scorer that made the headline table and ~13 instruments; B is still missing ≥11 retractions including this report's own lead claim; C's open-findings table still shows the pre-adjudication state for six rows; D still covers 20 of 43 DRAFT entries while claiming "most". Read `F.3`. |
| **Appendix E's blockquotes** | ⚠️ **Treat as paraphrase.** Nine quote the library's machine-extracted summaries, not the papers' prose (`E.0.3`). The conclusions hold; the wording is not the sources'. |
| **Citations to `M66`–`M76`** | ⛔ **Not ledger rows.** The ledger's table rows stop at **M65**. The underlying measurements are sound; the citations are empty. |
| **Any literature claim** | ⚠️ `Appendix E.5` audited **seven of eight** flagged sources — only `Lerch2017` is unexamined. But that was a **content** audit; **nobody has opened a PDF to confirm a given sentence is supported at the page cited.** |

---

## Where the next branch should be cut

**Four findings from the 2026-10-11 review remain open** — rows **6, 7, 8 and 10** in `Appendix C`, and
they are exactly the four that reviewer marked as *reasoning* rather than citation mismatch. ⚠️ Row 3's
**REFUTED** verdict is **re-opened**: its stated basis (*"the report cites M9 at every site"*) is false —
`Appendix D.4` carries the figure with no identifier at all.

🔴 **And the audit left ~38 of the ~70 distinct hard findings unfixed** — grouped into 11 bullets in
`Appendix F.3`, which also breaks the ~70 down by kind: **~18 are false as written**, the rest overstated,
stale, miscounted, badly cited or incomplete. The cheapest high-value items:

- **Every absence claim needs a `grep` before it ships, and every count needs re-deriving.** That is the
  whole remedy for most of the ~70, and it is mechanical.
- ⛔ **A sensitivity analysis over cell size.** `Moller1998` (p. 452, endorsing Rathbun & Cressie 1994)
  establishes that a cell-constant-intensity model **has no continuum limit**, so `size_ratio`,
  `act_ratio`, zero-share and the ARTIFACT verdict are all defined against **one arbitrary quadrat
  size**, and nothing here says how any of them behave at 0.25° or 1.0°. Cheap, and it bears on every
  headline number.
- **Wire a score that already exists.** `Appendix E.5.2` names a strictly consistent score for **cell
  expectations** that needs only predicted and observed counts — it runs on a delivered cube.

---

## Audit history, in one place

- **2026-10-11** — clean-context adversarial review: **17 findings**. Five fixed on receipt, including
  ⛔ **the retraction of §0 item 1, this report's lead claim**. Ten recorded and not investigated.
  ⚠️ 5 + 10 = 15, and **the accounting has never closed for the other two**, nor reconciled with
  `Appendix C`'s separate *"seven independently verified"*. Left visible rather than rounded.
- **2026-10-19** — six of the ten adjudicated: **five CONFIRMED and fixed**, one refuted (and that
  refutation is now itself re-opened, above). Four remain open.
- **2026-10-19** — `Appendix E`: ten newly-ingested papers brought against the report. **Two contradict
  it** — `Walz2024_PrecipitationTropicalAfrica` (a deterministic U-Net plus post-hoc calibration beat
  postprocessed ECMWF ensembles **in northern tropical Africa**, on occurrence *and* amount) and
  `Resin2026` (for **binary** outcomes PIT calibration is provably equivalent to auto-calibration). A
  third, `Raftery2005`, was filed as a contradiction and is **demoted** — the transfer fails for a
  Bernoulli gate. Plus **two proven non-identifiability results** bounding what any marginal diagnostic
  can do. ⛔ Nothing in §0–§8 was edited to agree with the literature; each collision is stated as two
  positions and left standing.
- **2026-10-19/20** — fifteen falsification audits. `Appendix F`.
- **2026-10-25** — `/falsify` on the sharpness-ceiling measurement: **3 hard + 1 soft**, two of them
  retractions of numbers this project had held for about an hour. ⭐ The first audit in this history
  whose findings are mostly **about the report's own subject matter rather than its prose** — see
  **§5.5 L1–L3** (the ceiling of a normalised statistic is not the ceiling of its unnormalised cousin)
  and **§5.10 L4–L6**. Two retractions added to `Appendix B`; one new instrument to `Appendix A.1`; a
  fourth row to `Appendix A.3`. ⛔ Nothing was deleted.

⚠️ **Nine of the ~70 distinct findings were introduced by me on 2026-10-19, the day before the audit.** The
auditors found all nine; I would not have. They are itemised in `Appendix F.2`.

---

**What this is.** Eight months of trying to measure whether a HydraNet is any good, collected in one
place: 53 dossiers, ~350 risk-register entries, 76 numbered claims (⚠️ **the ledger's table rows stop at
M65**; M66–M76 exist only as prose, see `Appendix F`), 26 measurement scripts, **seven** postmortems. It is about **the ruler first** — which metrics, baselines, gates and experiment designs
told us the truth and which fooled us — and then about the verdicts that ruler produced, so a reader
can tell what state the models are actually in.

**What this is not.** It does not review the platform's evaluation schema or governance protocol as
documents. It reports what happened here. §8 states what our own scars demand of a promotion gate;
comparing that against the protocol is the reader's job.

**Two disciplines, carried from this repo's own practice.** Every claim is labelled **MEASUREMENT**
(a number a run produced; overturned only by a harness bug or a re-run) or **INFERENCE** (a story
built on measurements; overturned by thinking, and repeatedly has been). The distinction is
`RESULTS_LEDGER.md:181`'s, adopted because five mechanism stories were retracted on 2026-08-16/17
alone. And every claim carries its **scope** on the line, not in a footnote — most of this record is
single-seed, single-vehicle, `sb`-only, which is below this repo's own stated bar.

---

## §0 — The six things to know

⛔ **1. The sanctioned rule's PRIMARY METRIC ranks an empty map top. Its guardrails then veto it.**
*(MEASUREMENT · 4 arms × 3 targets, `2026-08-15` substrate, h36)* ⛔ **CORRECTED 2026-10-11 — an earlier
version of this item claimed the rule "applied faithfully, promotes an empty map." That is RETRACTED: the
rule would rule this forecast *Ineligible*.** See Appendix C for the retraction and how it stood.

**What is measured.** A forecast with `size_ratio` **0.0000** and `act_ratio` **7.0e-05** (it fires
~14,000× less often than reality) clears the 5% CRPS superiority bar with CRPSS **+0.0904**, a 90%
origin-block CI of **[−0.0892, −0.0841]** excluding zero, and an effect ~34× the MDE — because **74.6%
of the win is confident zeros**. At h36 the ARTIFACT verdict is **12/12 — every arm, every target**
(verified in `results/rescore.csv`: 4 arms × 3 targets, all ARTIFACT). The repo's own words, and note
how narrowly they are drawn: *"FAO-02's own selection rule, applied to **its own primary metric**
against its own mandated baseline, would have promoted this forecast."*
(`.../07_experiment_log.md:347`)

**What the rule actually requires**, and the step the earlier version of this item skipped
(`RESULTS_LEDGER.md:622`): eligibility is a **strict conjunction** — superiority **and** *"all guardrails
non-inferior: QS99 & Brier within 1% of baseline; MCR calibration-distance `|MCR−1|` at least 1%
closer-to-1 than baseline's."* ⛔ ***"Fail either → Ineligible (regardless of ranking)."*** On this row:

| guardrail | the forecast | the baseline | verdict |
|---|---:|---:|---|
| `\|MCR−1\|` (magnitude calibration) | **0.99999** | 0.635 | ⛔ **FAIL, catastrophically** |
| Brier | 0.010387 | 0.010280 | ⛔ **FAIL** — 1.04% worse, outside the 1% bar |
| QS99 | never computed | — | not evaluated (`SCOPE.md:47`) |

**So the conjunction rejects it, and MEASUREMENT already said so.** `technical_risk_register.md:654`
(C-126, 2026-06-22, **3 targets × 3 seeds**): *"the sharp/shrinkage body **wins CRPS on all 3 targets,
every seed**, while emitting ~1–4% of the true magnitude (MCR≈0.03) … i.e. CRPS-based selection picks
the timid prophet exactly as F5 warns, and **the QS99/MCR guardrails correctly veto it.**"*

⚠️ **The narrower claim that survives is still worth the amendments in §1, for a different reason.**
Three things, each measured: (a) the **primary ranking metric cannot discriminate** an empty map from a
real forecast — if anyone ranks on CRPS and reads the guardrails as advisory, they promote this; (b) the
protection here rests on **one** guardrail, because QS99 is structurally inapplicable on a 99.7%-zero
field (**C-224** — the unconditional 99th percentile *is* zero) and Brier failed by **0.04 percentage
points**; and (c) nothing in this repo has ever **run** the conjunction — the baseline field is still
*"TBD — set once the first clean eval lands"* (`RESULTS_LEDGER.md:638`), so the veto above is arithmetic
I performed on an archived CSV, not a gate anyone executed. MEASUREMENT for (a) and (b); (c) is a
statement about the code.

⛔ **2. Nothing in the apparatus scores the product that leaves the building.**
*(MEASUREMENT about the code · the served-surface numbers below are one model, 16 draws, Africa+ME —
not the pooled ensemble)* Every instrument in §2–§5 scores **AP** (gate discrimination) or **CRPS/Brier**
on a sample cube. What the FAO receives is a **collapsed point surface** plus HDI bounds — a different
object, and **no instrument scores it.** The concrete instance, measured on a 40-lesson rehearsal by the
`models` session: `tower_point` served **0.0 for all 2,330,712 cells on global land, in all three
targets**, and **every structural check passed** — valid manifest, right coverage, right months, file
size 98.1% of a real run's. A downstream service reproduced the zeros independently. ⛔ **No metric in
our inventory would have flagged it.** Nor is there any calibration of the predictive distribution at
any aggregation level: on one model's validation, 468 origin-months, 90% interval coverage was
**0% / 1% / 7%** where ~90% is wanted, and the cells behave as independent marginals
and ⛔ *corrected 2026-10-11:* an earlier version of this line read the dependence ratio as ≈0.95 and
concluded *"any aggregate interval is too narrow by construction."* **§5.0(b) retracts that, in our
favour** — measured, the pooled ratio is **9.81** on `sb`, so country and regional totals are *less*
falsely confident than predicted. The coverage failure above stands; the variance half does not, and
it was a one-model number generalised to "any aggregate".

⛔ **And the part that makes this worse than a gap. We prescribed the diagnostic that would have seen
it, built it, left it unseeded, and wired it to nothing.** ⚠️ *(This replaces a claim about "both
regimes banned it" that was wrong — see Appendix C.)* The chain, every link sourced:
the register **prescribed** a positive-cell PIT as a *conditional diagnostic* on 2026-06-21
(`technical_risk_register.md:821`, drawing the selection-vs-diagnostic distinction explicitly) → it was
**built**, carefully, with its own Czado2009 discrete caveat documented (`scripts/proper_score_audit.py:82`)
→ it is **not reproducible**, because its RNG salt is `hash(t + tag)` (`:248`), which Python randomises
per process → **no current ruler emits it**: it is absent from `HEADLINE_COLUMNS` and no gate reads it →
and a delivery served an **all-zero surface across global land** that a PIT would have shown instantly.
**This is §5.8's "computed and read by nobody" pattern landing on the one diagnostic family that would
have caught the delivery failure.** The fix is seeding one RNG and adding a column.

⛔ **And when the collapse is finally scored, the delivered one is the worst available.** Seven collapses
of the same cube, measured 2026-09-30: **AP spans 18.9×**, and `tower_point` — the estimator the FAO is
served — is **last on both AP and mass**, 9.9× worse on AP than a plain mean and delivering **0.07% of
observed mass**. §5.0 carries the table, the like-for-like comparison, and the one caveat that could
overturn the worst row.

⛔ **3. We run two evaluation regimes and have never reconciled them.**
*(MEASUREMENT for the dates, the VOID and the promotion rule)* The adopted protocol (CRPS
primary, QS99/Brier/MCR guardrails, strict eligibility conjunction) was adopted 2026-06-20 and its
eligibility rule was **last applied 2026-06-22**. Everything since 2026-07-17 runs on a different
ruler built in-repo: AP primary, the crps-all/events/none split, activation and size ratios,
retention, zero-share-of-gap, paired origin-block CIs. The 8 production models were promoted under
**neither** — see §1 for what they *were* promoted under, which is the sharper question and is
answerable: two kill criteria and no comparison to anything.

⛔ **4. Nearly every scalar we reached for first was blind.** *(MEASUREMENT)* `crps_all` put four arms inside 0.0007
of each other while gate AP spanned **13×** (M9). MCR read ≈1.0 while activation was 4–115× wrong
(#258). Across the roster CRPS agreed to three decimals from step 6 while MCR spanned 10×. The
pattern is not metric-specific: **on a 99.7%-zero field, an all-cell aggregate is a report about the
zeros.** MEASUREMENT.

⚠️ **5. Cheap probes are reliable at killing ideas and unreliable at establishing them.** This is
the repo's own standing rule (`RESULTS_LEDGER.md`, adopted 2026-08-17) and it was adopted from
evidence: *every* retired claim has stayed retired, while the programme's most-cited positive number
(I-A's 89%) is flagged as the least trustworthy line in the table. Six interventions have failed;
four failed on the same lever. INFERENCE, but well-supported.

⚠️ **6. The test suite does not find this class of defect; deleting the fix does.** *(MEASUREMENT)* 2,079 tests were
green when `/code-review max` returned **25 runtime-verified findings, none caught by the suite**
(Epic #353). Then five of my own new guards survived their first mutation. The ledger's conclusion:
*"the deletion test, not the test suite, is what finds this class."*

⛔ **And the conclusion this report records the facts for but had not drawn.** ⚠️ **Three of this
sentence's five premises are CORRECTED 2026-10-20 and the conclusion is weakened, not withdrawn:**
~~No forecast metric is computed in `views_hydranet/`~~ — **false**, `utils/training_forensics.py:344`
computes `average_precision_score` with **AP as the default**, so part of the ruler lives in-package;
the ruler's *scored* half does live in `scripts/` and `reports/*/tools/`; `reports/` is gitignored
**but 1,300+ files under it are force-tracked**, so "gitignored" overstates the practical
irreproducibility; **21** tracked tool files carry hardcoded `/home/simon/...` paths, not 22 (the 22nd
is a docstring in a test), **and none of them is in `scripts/`**; and the governing framework is a PDF
outside git whose ~~six~~ **one** in-repo citation is a dangling *path* — the other five mention
"FAO-02" and cite no path, so they cannot dangle. Therefore: **the apparatus
that judges the delivered product cannot be run by anyone else, on any other machine, or in CI — so no
result in this entire record is independently reproducible by a third party.** *(MEASUREMENT about the
code; the conclusion is INFERENCE, but I cannot see a way around it.)*

⛔ **And it is worse than "outside the package": the ruler depends on a package nothing declares.**
Found 2026-09-30 by the `views-evaluation` session, verified here. **Four** scripts import
`views_evaluation` — `audit_run.py:27`, `crps_significance.py:27`, `mcr_readout.py:37`,
`proper_score_audit.py:52` — and it is **absent from `pyproject.toml`'s dependencies**. It arrives
transitively through `views-pipeline-core`, which pins it `>=2.0.0,<3.0.0`. So **any
views-evaluation 2.x release changes our numbers with no change in this repo and no pin of our own** —
and copying `scripts/` to another machine does not reproduce the ruler. ⚠️ **This is the identical
defect class as the `wandb` finding recorded in `pyproject.toml` two lines above the dependency list,
fixed eleven days earlier on 2026-09-19 (#381)** — the same shape, in the same file, unnoticed.

### The state of the models, in two lines

**What works:** at **h=1** the model genuinely beats climatology and persistence, and at L=300 it beats
a fairly-scored persistence on AP at **every** horizon to h36 (4 seeds). The ceiling — what it would
score on perfect inputs — is ~0.47 AP and **flat across all 36 steps**.
**What does not:** the free-running rollout throws away **two thirds** of that ceiling; the long-horizon
CRPS win over climatology is **ARTIFACT — 12/12 at h36 and 12/12 at h30**, though ⛔ *corrected
2026-10-11:* an earlier version of this line said *"every CRPS win beyond h1 is ARTIFACT"*, which is
false — `rescore.csv` returns **REAL** at h6 (5/12), h12 (3/12), h18 (2/12) and h24 (1/12), so the
decay is gradual and the win survives to about h18; magnitude has no skill (`size_ratio` → 0); six
interventions have failed, four on the same lever; and climatology overtakes on AP around **month 20**.

---

## §1 — The two regimes

| | **adopted protocol** | **working ruler** |
|---|---|---|
| source | FAO Pre-Release Note 05, Topics C & D — adopted `RESULTS_LEDGER.md:607` | built in-repo from `2026-07-17_lodestar_eval_dossier` onward |
| adopted / built | 2026-06-20 | 2026-07-17 |
| primary | **CRPS**, 5% superiority margin over baseline | **AP** (gate ranking), with paired origin-block CIs |
| guardrails | QS99, Brier, MCR — each within 1% | `crps_all` / `crps_events` / `crps_none` split, `act_ratio`, `size_ratio`, `retention`, `zero_share_of_gap`, the ceiling (the oracle's score) |
| eligibility | strict conjunction, written down, explicit | per-dossier pre-registered predictions + falsifiers |
| baseline | an empirical heuristic baseline | persistence **and** climatology, both emitted by the scorer |
| last used | eligibility conjunction: **no dossier after 2026-06-22** | every dossier from 2026-08-14 onward |

**Verification of those dates.** `Eligible`/`eligibility` appears in exactly four dossiers, all June
2026, the last `2026-06-22_hurdle_body_dossier`. `AP`/`delta_AP`/`crps_events` appears from
`2026-07-17_lodestar_eval_dossier` and then in every dossier from `2026-08-14` on. MEASUREMENT (grep).

**The divergence is explicit, not drift.** `2026-08-15_rollout_ruler_trust_dossier/SCOPE.md:47` places
**QS99 out of scope "anywhere, for any reason, including 'just as a diagnostic'"** — item 15, grouped
with twCRPS, SCRPS and LogScore. So one regime's guardrail is the other regime's banned term, in
writing.

⚠️ **Read item 15's scope precisely; an earlier draft of this report did not.** That clause covers
twCRPS/SCRPS/LogScore/QS99 **only**. The adjacent item 14 (PIT / active-cell calibration) carries a
*narrower* ground — *"FAO-02-rejected for selection; adding it now re-opens item 7 through the back
door"* — i.e. out of scope **for that dossier**, on selection grounds. **PIT as a diagnostic is not
banned anywhere in this repo; the register recommends it.** See §0 item 2 and Appendix C.

**And the protocol is not in the repo.** Same file, item 8: FAO-02's definition *"lives outside git (a
memory file + a PDF in `~/brain`) and every in-repo citation — e.g. `scripts/proper_score_audit.py:31`
— points at a path that does not exist in the tree."* Registered as a real problem; not fixed. So the
governing framework cannot be diffed, versioned, or checked by CI. MEASUREMENT.

**The baseline that was never set.** `RESULTS_LEDGER.md:633` — the reference that superiority and
guardrails are measured against — reads **"TBD — set once the first clean eval lands."** The run
ledger beneath it stops at **R5, 2026-06-20**. Nothing was ever entered under the eligibility scheme.
MEASUREMENT.

### What the 8 production models were actually promoted under

The sharper question, and it is answerable. `reports/2026-09-06_ensemble_roster_dossier/06_acceptance_criteria.md`
defines three criteria, all of them **kill** criteria, not rankings:

| | criterion | outcome |
|---|---|---|
| **K1** | STARVED at t=36 — fires if `recall_mass < 0.01` | **PASS 8/8** (0.016–0.057, the worst clearing the floor by 1.6×) |
| **K2** | BLOOM | **PASS 8/8** |
| **K3** | WORSE THAN CLIMATOLOGY | ⛔ **VOID** — the comparator produced no scores (views-models#445) |

So the de facto promotion rule was: **do not emit nothing, do not explode — and the one criterion that
compares the model to anything at all could not be evaluated.** The dossier's own verdict table is
explicit that VOID is *"not a pass and not a fail"*, and the log records it as *"VOID, not PASS … the
roster has not been checked against a baseline."*

⚠️ And `recall_mass`, the quantity K1 gates on, is **computed in exactly one script, read by nothing
else, absent from the locked glossary and from every score CSV**, and the criterion is applied by a
`print` against a hardcoded roster list. *(MEASUREMENT about the code.)*

**Why this matters more than any individual metric finding:** the protocol was never the thing being
tested. ⛔ **Corrected 2026-10-11:** an earlier version of this sentence said its one head-to-head
encounter with our data *"it lost"*. It did not — §0.1's conjunction **rejects** the empty map, and
C-126 measured its guardrails working on that pathology across 3 targets and 3 seeds. What is true is
that the conjunction has **never been run here**: the baseline it needs is unset (`:638`), K3 is VOID on
the roster, and the one time anything resembling it was applied to our data only its primary metric was
applied. The protocol was never the thing being tested, and it still has not been. Everything else in this report is
evidence about the ruler we built instead, and that ruler has itself been wrong repeatedly — §3.

### What this report recommends, since declining to say would be the larger omission

⚠️ Raised by the `views-evaluation` session: §1 documented two unreconciled regimes and **declined to
recommend which should win**, which on this report's own evidence may be the most consequential omission
in it. They are right. The evidence supports **four specific amendments** — three carried by measurements, the fourth by an argument and labelled as such — and it does not support
"the working ruler wins" — so that is not what is recommended.

**1. ⛔ CRPS cannot be the primary ranking metric on this field.** Not "is imperfect" — cannot. Four arms
inside **0.0007** at h18 while gate AP across the same four spans **13×** (M9); across the roster,
identical to **three decimals** from step 6 while MCR spanned 10×. A metric whose dynamic range across
the model set is smaller than the noise cannot rank that set. **Amendment: rank on AP; report CRPS.**

**2. ⛔ QS99 is a tail guardrail that never reaches the tail.** The 99th percentile of a 99.7%-zero field
**is zero**, so the guardrail sits inside the zero mass (C-224, **Tier 1**). ⚠️ **Read that precisely,
because it is easy to garble and this session garbled it once in conversation: the claim is NOT that QS99
evaluates to zero.** It does not — historical runs report values like 1.03 and 0.11 ⛔ **[CORRECTED 2026-10-20: the dossier cited for
these is WRONG. `2026-07-02_t0_calibration_screen_dossier` contains no QS99 value — `qs99` appears only
as prose in its `00_README.md` and `01_literature.md`, and its apparent "1.03" is a substring of
`41.034`. The real source is `reports/2026-07-16_bulk_calibration_dossier/07_experiment_log.md:116`,
and it is ONE arm-vs-baseline pair — "sb τ.8 CRPS 5.14 = 18× white_ranger (QS99 1.03 vs .11)" — not two
independent historical readings.]**
(`2026-07-02_t0_calibration_screen_dossier`). The claim is that **τ=0.99 lands on the zero/positive onset
boundary rather than on large-event magnitude**, because the real tail begins above the 99.7th percentile.
It is a well-behaved score answering the wrong question. And it is not merely a bad choice here:
C-224 cites Taillardat2023 and Brehmer & Strokorb for the claim that
**no proper-score *expectation*** separates different tail shapes — ⚠️ a citation whose status is
qualified in §3.2 and which the amendment does **not** need. **Amendment: drop QS99, or replace the tail
guardrail with something that is not an expectation of a proper score.** ✅ It rests on the arithmetic
alone.

**3. ⛔ No CRPS-based superiority claim without the decomposition beside it.** This is the one that is
*measured* rather than argued: the sanctioned rule, applied faithfully, promoted a forecast with
`size_ratio` 0.0000 (§0.1). What separated the real h1 win from the artifactual h36 one was
**`zero_share_of_gap` and the sign of ΔAP** — nothing in the CRPS number itself. **Amendment: a CRPS
superiority claim is inadmissible unless it ships with the zero-share of the gap and ΔAP.**

**4. ⚠️ Judge the twCRPS-as-diagnostic ban on its own merits, or lift it.** ⚠️ **Promoted to an amendment
2026-10-01 at Simon's instruction, and it is the only one of the four resting on an argument rather than a
number — that is stated here rather than left for a reader to notice.** The argument: §3.2 bans
`crps_events` from selection on **Lerch2017's** authority, and Lerch2017's own proposed remedy for the
problem it names is proper **weighted** scoring rules — twCRPS among them — which `SCOPE.md` item 15 bans
*"anywhere, for any reason, including 'just as a diagnostic'."* So **the literature's solution to the
problem we cite is the thing we excluded**, and it was excluded in the same clause as QS99 rather than
assessed separately. ⛔ **RETRACTED 2026-10-20 — THE DIAGNOSTIC BAN THIS AMENDMENT ASKS TO LIFT DOES NOT EXIST.** Three
live scripts prescribe twCRPS as a diagnostic **by name**, each citing the C-167 close-out:
`scripts/proper_score_audit.py:29` (*"this tool is a DIAGNOSTIC … twCRPS/PIT are read here only to
corroborate, never to decide"*), `scripts/tail_scorecard.py:4` (*"FAO-02 bans twCRPS + PIT for
SELECTION; this is a DIAGNOSTIC"*), and `scripts/mcr_readout.py`. ⚠️ **And the premise was a
misreading of scope:** `SCOPE.md` items 14 and 15 are adjacent bullets on **one dossier's own
out-of-scope list** under one header; this amendment read 14 as dossier-scoped and 15 as a repo-wide
ban. ⛔ **This report already retracted the identical argument for PIT and named the failure mode;
the amendment commits its mirror four lines away.** The amendment is withdrawn: nothing needs
lifting. Original text retained below.

**Amendment [WITHDRAWN]: the selection ban is defensible and should stand; the diagnostic ban
should be re-derived or dropped.** ⚠️ Be precise about which weighting — Gneiting2011 proves
**multiplicatively** weighted scores improper, while **threshold-weighted and quantile-weighted CRPS are
proper**, so "twCRPS is banned" has been imprecise about what it bans. ⛔ **The honest status: no
measurement supports this one.** It is a reading of two held papers against two of our own documents. If
it is wrong, it is wrong because the exclusion was deliberate and reasoned somewhere this report has not
found.

**What is deliberately not recommended.** Adopting the working ruler wholesale. §5 lists its live holes —
the method rule no scorer enforces, the floor-gate clause that never evaluates, the frozen instrument's
check that cannot fail, and the fact that nothing in it scores the delivered product. **Both rulers are
defective; only one of them has been measured failing on a promotion decision.**

⚠️ **And the actor, named precisely, because an earlier draft of this passage got it wrong.** C-224's
Tier-1 ask is FAO-02 owner sign-off — and **FAO-02 is Simon's own project**, a pre-release note under
`~/brain/2_projects/fao02/`. So this is **not** waiting on an external party, and the earlier wording
*"not ours to amend"* was misleading. What is true is narrower and still matters: **the document lives
outside the code repository**, so it cannot be version-controlled with the code, cannot be checked by CI,
and the repo's own references to it point at paths that do not exist in the tree (C-278). A developer in
the code cannot read the rules they are being held to.

⛔ **So all four amendments are Simon's to make, in a different project of his, not a technical step here.**
C-224's ask is recorded as *"unchanged, still open"* since July and, as far as this record shows, has never
been acted on.

---

## §2 — What works

Instruments that have repeatedly told the truth. ⛔ **[QUALIFIED 2026-10-20 for the rollout ruler
specifically (§2.3): its ARTIFACT verdict has ruled ONCE, not repeatedly. `verdict_token` has a single
caller (`2026-08-15_rollout_ruler_trust_dossier/tools/rescore_v2.py:301`); of 268 score CSVs under
`reports/`, exactly one carries `zero_share_of_gap` and one a `verdict` column — both the 2026-08-15
re-score of one archived board; and nothing in `scripts/` or `views_hydranet/` imports
`rollout_ruler_core` at all. The instrument is real, tested and correct, and it has been applied to
none of the 24 dossiers that followed. "Available" is the honest word, not "repeatedly".]** Each row
names the case where it caught something
nothing else could.

### 2.1 The oracle comparison — the single most valuable instrument we have

Emit the **same model twice**: free-running (`rollout_feedback='sample'`, it feeds on its own output)
and **the oracle** (`='teacher_forced'`, handed the real field each month). The oracle's score is
**the ceiling**. Both curves, per horizon.

- **M10:** fed the real field, gate AP holds **0.4745 → 0.4793 (h18) → 0.4577 (h36)** over 36 steps.
  It does not degrade *at all*. MEASUREMENT, one seed, `violet_visitor`, 160 lessons, `sb`.
- **M24:** the same model left to feed on itself goes **0.474 → 0.257 → 0.137**. So the ceiling is
  ~0.47 at *every* horizon and the rollout throws away two thirds of it. MEASUREMENT, same scope.

**Why it works:** it separates *"the model is bad at long horizons"* from *"the model is poisoned by
its own output"* — and the answer here was unambiguously the second. Nothing else we tried
distinguished those two, and for months the horizon was blamed. It costs **no training**, only a
second emit.

**What it does not do:** the ceiling says what perfect inputs would buy. It says nothing about what is
learnable from the data — that is the amount-ceiling question, §4.

### 2.2 AP, because it has dynamic range where CRPS does not

- **M9:** four arms score `crps_all` **0.1353 / 0.1352 / 0.1350 / 0.1346** at h18 — a span of
  **0.0007** — while gate AP across the same four spans **13×**. MEASUREMENT.
- Across the 8-model roster, CRPS was identical to **three decimals** from ~step 6 while MCR spanned
  **10×** (#258).

**The generalisable rule this yields:** *check a metric's dynamic range across your model set before
ranking on it.* A proper score can be simultaneously proper and non-discriminating. That is not a
criticism of CRPS; it is a statement about a 99.7%-zero field.

### 2.3 The zero-share decomposition and the ARTIFACT verdict

`scripts/rollout_ruler_core.py` (Epic #263, ⛔ **53** tests, not 86 — and it **does** skip (three `pytest.skip` calls; corrected 2026-10-20)). For a claimed win it reports
CRPSS, ΔAP, and **what fraction of the CRPS gap is true zeros**, then rules REAL / UNDECIDABLE /
ARTIFACT against a pre-registered rule.

This is the instrument that produced §0.1. ⛔ **[CORRECTED 2026-10-20: not "the only thing".** §0.1 does the same by a different mechanism and
two months earlier — the guardrail conjunction vetoing a CRPS winner, C-126, 2026-06-22, 3 targets ×
3 seeds. And `scripts/screen_verdict.py:104` has an explicit `"verdict": "KILL"` branch for a model
that met its own win criterion with a guardrail broken. **The defensible claim is that it is the only
instrument that decomposes the gap and rules on the decomposition.]** It is the only thing in the repo
that can look at a
statistically significant, correctly computed, properly scored win and say *the gap is real and the
win is not.* Applied to the whole board: at h1, **7 REAL / 0 ARTIFACT**; at h36, **0 / 12** — ARTIFACT
unanimous. MEASUREMENT.

**Its method rule matters and is easy to violate:** *the reference's S must equal the arms' cube
width.* `crps_ensemble`'s `2/(m·m)` bias does not cancel across unequal S, and AP is quantised to S+1
rank levels. Persistence scored at S=1 against arms at S>1 violates it — see §3 H-new.

### 2.4 The two ratios that made the original disaster legible

- **`act_ratio`** — how often the model fires versus how often reality does.
- **`size_ratio`** — the **median** of guess ÷ truth on conflict cells.

In #258 these were the only numbers that saw the defect: activation **4–115× wrong**, compensated by
over-magnitude, so MCR read ≈1.0 and looked calibrated. In §0.1 they are what makes "promoted an empty
map" a statement rather than an opinion: `act_ratio` 7.0e-05, `size_ratio` 0.0000.

⚠️ **`size_ratio` (median) and `pos-mcr` (mean) are different metrics and we once conflated them.** A
few large cells flatter the mean. `GLOSSARY.md` §4 records this explicitly.

### 2.5 Paired designs — the cheapest real power we have

- **M40:** pairing cut the MDE from **0.0541 → 0.0086, 6.3× on identical data.** The same effect reads
  UNDERPOWERED unpaired and EFFECT paired. This answered #281 by demonstration. MEASUREMENT.
- **M65:** the seed variance the programme had been designing against is **~5× too large**, and a
  paired 2-seed design detects effects **3× smaller** than the programme's best result. MEASUREMENT.

**The lesson is a design lesson, not a statistics lesson:** most of our "underpowered" verdicts were
self-inflicted by comparing arms that also differed by seed. Pairing is free.

⚠️ It does not rescue everything. The SS sweep's arms differ by seed *as well as* by treatment, so
pairing cannot be applied retroactively there (M40's own caveat).

### 2.6 The floor gate — refusing to measure on a vehicle that cannot measure

`scripts/floor_gate.py`. A control arm scoring at or below chance has no dynamic range, and an
experiment run on it is uninformative *in both directions*.

- **M28:** both 40-lesson arms fail the gate — **2.16× and 0.99× chance** — and their two seeds sit
  **2.3× apart**. A 40-lesson model *matches* climatology at h1 (0.97×/1.03×) and is **10–25× worse**
  at h18. MEASUREMENT, `scripts/floor_gate.py` + lesson-curve EXP-01.
- `postmortem_floor_limited_vehicle.md` is the consequence: the shared `truncated_smoke` vehicle
  scored **0.0070 at h18 against a prevalence of 0.009077 — 0.77×, below random ranking.** Nine claims
  (M1–M9) ⛔ **[CORRECTED 2026-10-20: wrong in both directions. `postmortem_floor_limited_vehicle.md`
  disposition table covers ELEVEN claims — M1, M3, M4, M5, M6-oracle, M7, M8, M9, I-B, I-C, I-E, plus
  M20 as confounded — of which eight are `M` rows. ⛔ `M2` appears NOWHERE in that file, and M2 is the
  row carrying the floored control value (h18 0.007) the whole postmortem is about. Verified: zero
  occurrences.]** had to be individually re-dispositioned, and one manipulation read **+0.9%** on the floored
  vehicle and **−93.7%** on a real one.

**This is the highest-leverage cheap gate in the repo** and it is the reason the h1-matches-climatology
observation is not evidence of anything.

### 2.7 Direct manipulation with a known answer

The probes that reliably killed hypotheses are the ones where the right answer was known in advance
because we constructed the input:

| probe | what it does | result |
|---|---|---|
| `spatial_scramble` | keeps count and magnitudes, moves the locations | AP **0.3008 → 0.0097** (M3); on a real vehicle 5× worse than control (M12) |
| `thin:0.75` | deletes ¾ of true events, keeps placement | recovers **95.5%** (M12); **97%** of the oracle (M45-era) |
| `wrong_month:-60` | a *real* field from the wrong month | 43% — and a verification pass nearly "corrected" this correct number |
| roll the cell state 90 cells | displaces the recurrent map | forecast moves **90 cells intact, r≈0.90**, skill collapses **48×** (M54) |

**Together these say: recall is nearly free, placement is everything.** MEASUREMENT, and replicated
across vehicles for the scramble/thin pair (M3/M4 → M11/M12).

### 2.8 Byte-identity, on the right object

- **M13:** the Epic #263 board reproduces bit-for-bit from preserved cubes — worst **|ΔAP| = 0.00e+00**
  over 7 horizons.
- **M14:** the post-2026-08-12 inference commits are a proven no-op on the free-running path — worst
  **|ΔAP| = 0.00e+00**.
- **M22:** retraining at HEAD gave **190 weight tensors with identical sha256** five days and five
  commits later.
- **M76:** ⛔ **[CORRECTED 2026-10-20: `M76` IS NOT A LEDGER ROW. `RESULTS_LEDGER.md`'s table rows
  stop at **M65**; M76 occurs once, as prose, at `:178` ("Row M76; `reports/2026-09-15_pypi_smoke/`").
  The same is true of M66–M75, which §4.1 also cites as rows. ✅ The MEASUREMENT is sound —
  `RESULTS_LEDGER.md:170-171` and `reports/2026-09-15_pypi_smoke/` both record four byte-identical
  artifacts, 190 tensors each, BatchNorm buffers included. The citation is empty; the finding is not.]**
  four roster models retrained from the **PyPI wheel** are byte-identical to checkout-trained
  artifacts, BatchNorm buffers included.

⚠️ **And the trap inside it:** the two byte-identical models of M22 have **different `.pt` file
shas** — torch stamps mtimes into the zip. **File hashes are an invalid identity check**; hash the
weight tensors.

### 2.9 Recomputing the headline numbers from the committed data

The verification pass of 2026-08-17 recomputed all nine headline figures from committed CSVs instead
of copying dossier prose. **Three of nine were wrong or misleading, and all three failed the same
way:** values pulled from different cells of the results grid and presented as a single comparison
(Moran's I paired the oracle with free-running; `thin` activation paired h18 with h36; a peak quoted
as a range). Registered **C-298**. None changed a conclusion; two had already propagated into the risk
register.

⚠️ **A verification pass produces false positives too.** "Occurrence merely being plausible — 43%"
looked wrong until traced. *The fix is to trace the number to its arm, not to correct it on suspicion.*

### 2.10 The MEASUREMENT / INFERENCE ledger

Not a metric — a bookkeeping discipline, and on the evidence the most effective single practice in the
record. It is what makes the retraction rate visible instead of embarrassing, and it carries the
repo's standing rule (§0.4). Its own self-description is the best argument for it: the first version
of the persistence section stated an unqualified claim that was wrong — *"the sentence this table
exists to prevent, written into the table itself on day one."*

### 2.11 VOID — a third outcome besides "it worked" and "it didn't"

The single most transferable idea in the repo's experimental practice, and it is not a metric. **VOID
means the instrument could not answer**, as distinct from the treatment not working. Harvested from the
pre-analysis plans, the triggers actually used are:

| trigger | what it means |
|---|---|
| the **control** arm fails the floor gate | the vehicle cannot resolve the effect we came to measure |
| two arms' **trained weight hashes are identical** | the treatment was inert (C-324) |
| `N ≠ 170430` or origins ≠ 13 in any row | the arms were scored on different supports and are not comparable |
| h=1 metrics not identical across arms to 1e-6 | step 1 has no feedback, so a spread means something other than the treatment differed |
| an arm produced **no scoreable artifact** | *"VOID, not negative. No Δ is quoted, estimated or implied; a crash is never evidence about H"* |
| the **confound control given the answer** fails to score near-perfect (or scores AUC ≥ 0.99) | the instrument is broken, in either direction |
| preserved cubes do not reproduce the shipped board exactly | the scorer, truth pin or cubes are not what produced it |

And the rule that gives it teeth: **"Any of F1–F5 firing makes that candidate VOID, never 'worse'."**

**Why it works:** without it, every one of those conditions produces a *number*, and a number gets
read as a result. C-324 is the proof — an inert treatment produced byte-identical weights, and *"a
half-connected knob would have produced a small number, and the pre-registered rule would have read it
as 'hypothesis dead'."*

⚠️ **The gap in it, named by the repo itself:** *"Every falsifier in the 2026-08-14 plan (F1, F2,
F-DEGEN) assumes the readout is valid. **None fires on an uninformative measurement.** That is the
hole, and it is generic: a pre-registration can be perfect about what would refute the hypothesis and
silent about whether the instrument can see anything at all."* The floor gate (§2.6) exists to close
exactly that, and §5.3 shows its decisive clause is not running.

⚠️ **And one discipline failure is on record:** a dossier notes a falsifier being **changed after it
fired**. That it is recorded is the practice working; that it happened is the practice failing.

### 2.12 Pre-registration — it works here, and the counts say how

Of the dossiers dated after 2026-08-13 — ⛔ **26, not 25, so every rate below has a wrong denominator. Re-derived 2026-10-20: ≥24 of 26 pre-registered a falsifier list (not 21) and 25 of 26 ran a scored arm (not 20), the sole exception being `2026-09-05_professor_forcing_dossier` ("No experiments run"). Numerators unverified.** Of the 25: **21 pre-registered a falsifier list**, 20 ran at least one
scored arm.

| | count |
|---|---|
| a pre-registered **named falsifier actually fired** on real data | **9 of 20** |
| at least one gate fired **or** a run was declared VOID | **13 of 20** |
| a pre-registered **prediction was falsified** — the verdict contradicted the dossier's own stated expectation | **9 of 20** |
| the verdict was **not** the one the dossier hoped for (negative / null / VOID / underpowered) | **15 of 20** |

⛔ **The decisive evidence that it is not decorative** is the counterfactual: in four cases a falsifier
voided an arm and the record states the *specific wrong claim* that would otherwise have been written.

> ⛔ **CORRECTED 2026-10-20 — TWO OF THESE FOUR ROWS DO NOT SUPPORT THE CLAIM, and one is evidence
> AGAINST it.**
>
> **Row 3 (BPTT-SA SCREEN-1) is a pre-registration FAILURE that luck caught, not a success.** The
> weight-hash post-condition is **absent from the plan locked at `4424f26`** before training; it enters
> the next day in `AMENDMENT A2.3` (2026-09-04), described there as *"the **existing** weight-hash
> post-condition"*. The dossier's own log says the locked rule would have published the wrong claim:
> *"this was read as a defect, not as 'Δ ≤ 0, H is dead', **which is what the locked decision rule would
> otherwise have said**."* ⛔ **And `Appendix B` of this report classifies that detector as *"the result
> being impossible — named as luck"*, not as a gate.** Three independent confirmations. The row stays
> **as the counterexample it actually is.**
>
> **Row 4 (roster K3) counts a gate that never ran.** The record says K3 *"was never evaluated"* because
> the comparator produced no scores — it did not fire; and *"a PASS"* is this report's reconstruction, not
> a quotation. The same dossier's weight-hash post-condition is recorded *"**Never computed** …
> pre-registered and it did not run"*.
>
> ✅ **Rows 1 and 2 survive verbatim.** The honest count is **two** documented counterfactuals, not four.

| arm | what fired | what would have been published |
|---|---|---|
| `shuffle_months` | F6 — the intervention moved its own target statistic only 5% | *"persistence matters somewhat"* — its −0.0415 "reads as a plausible middling effect" |
| pushforward w=0.1 | F5 **and** F6 — the oracle fell, so the term damaged the **model** | *"pushforward is bad"* from its −0.10 AP@h18 — **"a different and wrong claim"** |
| BPTT-SA SCREEN-1 | the weight-hash post-condition — two independently trained arms, **byte-identical** | *"BPTT-SA does not work here"* |
| roster K3 | the comparator produced no scores | a **PASS** — instead, *"the roster has not been checked against a baseline"* |

And the discipline held under pressure, which is the part that matters: *"F1 was amended, because its
instrument was demonstrably wrong. F5 was not, because it was doing exactly its job. **Keeping that
distinction under pressure is the difference between a falsifier and a formality.**"* ⛔ **[Corrected
2026-10-20: that third sentence is MINE, not the record's — it exists nowhere in this repository outside
this report, and it sat inside the quotation marks. The first two sentences do paraphrase
`2026-08-26_pushforward_dossier/05_analysis_plan.md:213`.]**

**Where the value actually comes from — not the thresholds, the verdict state space.** Four states
(EFFECT / NULL / UNDERPOWERED / VOID) produced the repo's two most honest labels and are the direct fix
for an earlier null being *"uninterpretable rather than merely negative."*

⚠️ **Its systematic blind spot, named twice independently: the apparatus, not the hypothesis.**
*"Every falsifier in the 2026-08-14 plan assumes the readout is valid. **None fires on an
uninformative measurement.**"* And: *"the pre-registration is meticulous about the **hypothesis** and
silent about the **apparatus**."* Every gate added since — the floor gate, the potency check, the weight
hash, BRANCH 0, the positive control — is an *apparatus* gate added after the fact.

⚠️ **And a pre-registered rule is not a tested rule.** Six distinct classes of broken decision rule are
on record. The worst: `truncated_nb`'s EFFECT clause used a permutation test implemented one-sided for
`treatment > control`, so a **−0.2376** effect scored **p = 1.0000** and the author's own
`EFFECT (NEGATIVE)` branch was **unreachable**. ⛔ **[CORRECTED 2026-10-20: the causality is INVERTED. Every gate named below predates the defect
it is said to respond to — `screen_verdict.py` 2026-07-18, `floor_gate.py` 08-17,
`lesson_curve_gate.py` 08-18, `ss_sweep_gate.py` 08-21, all before the truncated_nb dossier of
08-24 (first-commit dates, `git log --diff-filter=A`) — and the quoted justification is dated
2026-08-17, a week earlier. ⛔ And the rule with the unreachable branch was NEVER extracted: it is
still `reports/2026-08-24_truncated_nb_dossier/tools/verify_trunc.py`, with no `rule_md5`, no test in
`tests/`, inside the gitignored tree and therefore not in CI — the exact condition the quoted
justification exists to prevent. The PROPERTIES claim survives; the "exists because" does not.]** The
response — md5-pinned, unit-tested,
sabotage-verified rule modules in tracked `scripts/` — exists because *"a rule that lived only in the
dossier would be a rule with no test in CI."*

### 2.13 Validating the instrument on synthetic fields, before it sees data

The single cleanest instance in the record, and it changed a primary endpoint **before the plan was
finalised**: the sharpness dossier's planned primary metric, `fss_ratio`, *"was the planned primary and
is NOT a blur detector"* — measured on synthetic fields with known answers it falls **further for a
displaced field (0.267) than for a heavily blurred one (0.897)**. *"Used as intended it would have
reported displacement as blur."* `moran_i` was promoted instead because it separates blur, displacement
and thinning **by sign**. In the same pass, `conc1pct`'s direction was predicted, measured, and found
**wrong** — so the measured direction is what got registered.

Two siblings of the same practice: the **confound control** — a learner *given the answer*, which must
score near-perfect or the instrument is broken (it caught a numerical bug on 2026-07-14 before any
verdict was banked), and the discovery that a cited premise **did not exist**: C-190 had been cited as
measuring the model as a strong low-pass filter, and *"the producing code does not exist in the repo or
its history."*

---

## §3 — What does not work

The 2026-08-13 DRAFT organised this by **symptom** (A1–I4: metric blindness, Goodhart, instability,
composition, rollout, baselines, epistemics). Those letters are preserved and referenced below. But
the register has since found the better organising principle, by **cause**, and named it for the first
time on **2026-09-10**:

> **Cluster 18 — "The instrument did not measure what it claims."** 23 Tier-1/2/3 entries (⛔ **24**, not 23; corrected 2026-10-20), *"every one
> of which cost GPU time or produced a wrong ledger row, currently read as 23 unrelated incidents."*
> The register calls it **the highest-value group it has**.

And its proposed remedy is one question, asked before the run:

> ⛔ **"What would this measurement look like if the effect were absent?"**

Three of cluster 18's four shapes fail that question. That single line is the most valuable sentence in
this report.

### 3.1 The shapes of cluster 18 — ⛔ **FIVE since 2026-10-11, not four; shape (e) is C-345 and the register says it "inverts the cluster's usual remedy: the pre-flight question passes". So "three of four" is three of five, and the missing shape is the counterexample to the remedy this section calls its most valuable sentence.** The four shapes

| shape | the defect | worst instance |
|---|---|---|
| **(a) the instrument cannot see the thing** | the statistic cannot move when the thing under test moves | **C-319** — rolling the clamp anchor 90 cells leaves occurrence 0.718/0.715, magnitude 0.628/0.651, alignment 61.5/69.3 essentially unchanged while **AP@h18 falls 0.362 → 0.0075 (48×)**. Three statistics, all blind. (↔ DRAFT A1, A2) |
| **(b) the decision rule cannot deliver a verdict** | branches unreachable or the outcome space omits "no number" | **C-303 occ. 9** — a one-sided permutation p made the EFFECT branch unreachable for a negative Δ, so the rule printed **"NULL / UNDERPOWERED, p = 1.0000" for a mean ΔAP of −0.2376 — 7.5× the 3×MDE bar, 4/4 seeds unanimous.** (↔ DRAFT C2, the power-vs-null pattern) |
| **(c) measured on the wrong object, phase or training state** | correct arithmetic, wrong subject | **C-308** — a drift probe kept the *first six* forward calls, i.e. the teacher-forced warm-up, not the 36 free-running steps. Published σ_max **1.60**; true **7.7628**. It reported `max\|h\|` **rising** 0.000 → 2.867 where the real phase **collapses 40× (65.6 → 1.6)**. *"Every downstream guard passed."* (↔ DRAFT B1, B2, the wrong-object pattern) |
| **(d) the readout assembled wrongly** | well-formed number, different population than its label | **C-298** — 3 of 9 headline figures wrong: *"the results grid here is four-dimensional and the prose is one-dimensional, so every sentence silently collapses three axes."* (↔ DRAFT C7) |

### 3.2 The metrics themselves

⛔ **`crps_all` on a 99.7%-zero field ranks the zeros, not the model.** Four arms inside **0.0007** at
h18 while gate AP spans **13×** (M9). Across the roster, CRPS identical to **three decimals** from step
6 while MCR spanned **10×** — *"ranking these models on CRPS ranks noise"* (#258). (DRAFT A3/A4.)

⛔ **QS99 cannot work on this target, structurally.** **C-224 (Tier 1):** the 99th percentile of a
99.7%-zero field **is zero** — the guardrail sits *inside* the zero mass. And it is worse than a bad
choice: C-224 cites Taillardat2023 and Brehmer & Strokorb for the claim that **no proper-score
*expectation*** — including the banned twCRPS — separates different tail shapes ξ. **This is a guardrail
of the adopted protocol.** Its Tier-1 governance ask is recorded as *unchanged, still open*.

⚠️ **Citation status, flagged not fixed — Appendix D's discipline applied to this report.** Reviewed
2026-09-30 by the `library` session. **Brehmer & Strokorb is not in the library**, so half the
impossibility citation cannot be verified from our own holdings. Taillardat2023 **is** held, and its
extracted claim is narrower than the sentence needs: it concerns the expected **weighted** CRPS, not
proper-score expectations in general. Ehm2016 sharpens what the impossibility can mean — **elementary
scoring functions *can* distinguish forecasters pointwise** (Murphy diagrams) — so the barrier lives in
the **expectation**, not in the scoring framework. ✅ C-224 and this report both say *expectation*, so the
precision is intact; but the sentence currently rests on one verifiable paper about weighted CRPS plus one
unverifiable one. ⛔ **The QS99 half needs none of it:** *"the 99th percentile of a 99.7%-zero field is
zero"* is arithmetic.

⛔ **MCR is a first-moment ratio and hides compensating errors.** It read ≈1.0 while activation was
**4–115×** wrong, cancelled by over-magnitude (#258). (DRAFT A1, the compensating-error pattern.)

⚠️ **`crps_events` must never select, and our own rules disagree about this.** Subsetting a score on
the observed outcome is the Forecaster's Dilemma (Lerch2017). **C-224 (2026-07-30): display-only,
"exactly like the FAO-02-banned twCRPS."** But **C-167's close-out (2026-07-27) wired it into a
pre-registered decision rule** — *"a magnitude change WINS iff crps_events improves [proper, selects]
…"* — and that text was left standing. C-224 is later and supersedes; the register does not flag the
conflict. **✅ RESOLVED 2026-09-30, and the loop is worth recording.** This report flagged it as
*"unresolved in the documents"*; the `views-evaluation` session then asked whether the edit had actually
been made, which is what produced it. C-167 now carries a dated `SUPERSEDED IN PART by C-224` note
striking the `[proper, selects]` clause and stating that the rest of the rule — `size_ratio`→1 and the
FSS@1/conc1% spatial guard — stands. Register integrity tests green (54 passed).

⚠️ **Why it mattered more than a tidy:** C-167's was the one instance of the conflict wired into a
**pre-registered decision rule**, not merely into prose. It had stood for two months.

⚠️ **The theory predicted the CRPS finding, and the report had only measured it.** Supplied by the
`library` session: **Bolin2023** proves CRPS is **not locally scale invariant** — its scale function is
`σ^{α−2}·s(Q)`, so **discriminatory power grows with scale.** On a near-zero field the scale is tiny, so
the discriminatory power is tiny. That is the mechanism behind four arms sitting inside 0.0007. The
finding moves from *"we measured that CRPS does not work here"* to *"we measured it, and the theory says
it cannot."* ⚠️ And a caution on our own instrument from the same paper: **skill scores are in general
improper even when built on a proper rule.** `CRPSS` appears throughout this record as a headline
number — §2.3 uses it only inside a verdict that also requires ΔAP and the zero-share, which is the
right shape, but a bare CRPSS ranking would be unsound.

⚠️ **And a tension the report cites into existence without resolving.** §3.2 bans `crps_events` from
selection on Lerch2017's authority. **Lerch2017's own proposed remedy for the Forecaster's Dilemma is
proper *weighted* scoring rules — twCRPS among them** — which is the family banned by SCOPE.md item 15.
So the literature's solution to the problem we cite is the thing we have excluded. ⚠️ Be precise about
which "weighted": Gneiting2011 proves **multiplicatively** weighted scores improper, while
**threshold-weighted and quantile-weighted CRPS are proper.** The ban is defensible for *selection*; it
is **not** obviously defensible for *diagnostics*, and on the evidence here twCRPS-as-diagnostic looks
caught in the crossfire of the QS99 ban rather than independently judged. **Promoted to §1 amendment 4 on
2026-10-01**, carrying the label that it is the only amendment resting on an argument rather than a
measurement.

⚠️ **Any statistic computed from the forecast alone cannot judge placement.** Occurrence, body
magnitude and gate–body alignment are all preserved by a spatial roll that costs 48× in AP ⛔ **[the
locators in the 2026-10-11 correction below are off by one on EVERY line, and the substance
over-corrects — CORRECTED 2026-10-20. Actual `GLOSSARY.md`: `:155` the ceiling (a drift warning, NOT
C-318), `:156` occurrence (C-318), `:157` body magnitude (C-318), `:158` gate–body alignment (the
placement mark). ⛔ And all three metrics ARE documented as placement-blind — `:158`'s own text says
"The same holds for occurrence and body magnitude … No statistic in this row, or the two above it, can
distinguish a good forecast from the same forecast in the wrong place." So "one is documented as such"
and "the documentation claim was two-thirds wrong" are themselves wrong: the original error was the ROW
COUNT, not the coverage. The 48× is exact (0.362/0.0075 = 48.3).]** (C-319,
M53). `GLOSSARY.md` now carries this as a ⛔ annotation on three separate metric rows. ⛔ **[CORRECTED
2026-10-20: ONE row, not three. `GLOSSARY.md:158` (gate–body alignment) carries the single
"MEASURED BLIND TO PLACEMENT" mark, and its own text covers all three metrics: "The same holds for
occurrence and body magnitude … No statistic in this row, or the two above it, can distinguish a good
forecast from the same forecast in the wrong place." Rows `:156` and `:157` carry ⛔ marks about C-318
(sentinel averaging), a different defect. `RESULTS_LEDGER.md:411` says "the measured warning",
singular. ✅ The COVERAGE claim was right; the row count was wrong.]**

### 3.3 The classes the DRAFT did not have

Eight defect classes are in the register and not in the A1–I4 taxonomy. They are the ones that have
cost the most.

**N1 — The declared-but-inert knob.** ⛔ **C-324, Tier 1.** A BPTT-SA implementation was *"unit tested
(7), mutation tested 5/5, lint clean, suite green (1,901)"* — and **inert on the production config**.
Two arms trained **276 minutes to byte-identical weights**. It was caught only because the result was
**impossible**. The register's own comment is the lesson: *"A half-connected knob would have produced a
small number, and the pre-registered rule would have read it as 'hypothesis dead'."* Nine other
instances: C-216 (`feedback_clamp_log1p=[7,7,7]` → byte-identical rollout, and it had **already**
misled a result — "clamp doesn't help" vs "clamp wasn't applied"), C-302 (a log path silently dropped
by Pydantic `extra="ignore"`, so the failure mode is *an absent file that reads as "the diagnostic ran
and found nothing"*), C-331 (`freeze_recurrent_weight=0.0` logs `🧊 recurrent state CLAMPED` and
applies nothing — escalated to **production** by ADR-027 §2.1).

**N2 — The instrument perturbs or kills the run it measures.** ⛔ **C-344** — the gate probe drew from
the same RNG stream the feedback transforms use, so *"any arm measured with the probe on was not the
same arm as the headline run of the same name"*, and probe records already on disk were measured on a
differently-thinned field. **C-330, Tier 1** — a `register_hook` on a non-leaf fired on gradient
accumulated from every consumer, so a feedback grad-clip **rescaled the primary supervised gradient**.
⛔ **C-328 instance 4 is in every shipped artifact:** ⛔ **[CORRECTED 2026-10-20: FALSE for the
current roster. Instance 4 was FIXED 2026-09-05 (`register:3900`, verified in source on `development`
2026-09-10) and the eight roster models were retrained 2026-09-07/08 — after the fix. The same register
entry states it: "no roster artifact's buffers change". ⚠️ Which views-hydranet revision views-models
resolved on 09-07/08 was not verified; the fix was on `main` and `development` from 09-05.]** `random_flips` defaults True and reaches the
BatchNorm recalibration pass, so **~half of every BN running statistic in every shipped artifact was
accumulated on H/W-flipped fields.**

**N3 — The reference, not the instrument, is wrong.** The register quotes itself: *"The register has
extensive coverage of whether we measure the model correctly, and **none** of whether the thing we
measure against is correct."* ⛔ **C-338** — priogrid 149451 reads **27,413** in m550 and **32,505** in
m552 with a **2** between them, while the next largest cell-month anywhere is **1,100**. Filed upstream
(views-datafactory#484). Nothing checks the truth field for outliers at build time. And **#282** — the
first origin's persistence forecast was silently all zeros, understating the baseline by
**10.5%/6.5%/4.1%**, and the error **scales as 1/n_origins**, so a single-origin study scores
persistence as **exactly zero**, silently.

**N4 — MDE borrowed from another contrast; noise floor unmeasured.** **C-306** — M41 judged
`cell@0.5` vs `cell` (one blend weight apart) against an MDE computed for `cell` vs `none` (states
diverge completely), and *"both quantities are called 'the paired MDE at h18'."* **C-317** — retraining
the same config, same seed, same GPU, same torch reproduces its twin to **148%/114%/94%/78% of the
seed-to-seed sd**. **C-253/C-254** — plain Newey–West on an origin×cell panel *manufactures*
significance, and GW asymptotics live in origins (P=13), not in the cell cross-section.

**N5 — Provenance not recorded, so runs are not comparable.** **C-300** — all four SS-on roster models
trained **before** the validator that forbids their config landed, so they trained on an *ungated mean*
and rolled out on a *gated sample*. Their rows are un-rerunnable *and* were produced under a config the
repo now rejects. Conclusion, verbatim: **"There is no valid, non-floored SS measurement anywhere."** ⛔ **[CORRECTED
2026-10-20: that register quote (`technical_risk_register.md:2673`, ~2026-08-14) is STALE and this
report closes the question on the measurement that superseded it. M30–M33 ran `ss_feedback='sample'`
in every arm at L=300 across 4 seeds with all four floor gates passing
(`2026-08-17_ss_retention_dossier/05_analysis_plan.md:86`, `07_experiment_log.md:18`) — and §4.1 item 7
closes SS on exactly that, 25 lines above this row. §3's N5 repeats the same stale quote.]**

**N6 — A confident negative from a search whose non-empty case was never seen.** **C-309** — *"We have
not tested that. It is a real, untested axis"* — while `CurriculumLearner` implements exactly it, has
its own CIC, its own active ADR, config fields in a file edited earlier in the same session, and runs
in **every arm of every experiment**. Then the fix *itself* failed the same way: a literature search
returned "none of the 26 relevant papers has a single extracted claim" — all 26 had claims (70 across
20 checked); the glob path was wrong. ⛔ **The amended rule is the transferable one: a negative claim
requires a search plus a positive control, and a uniform-zero result across a heterogeneous population
should be treated as instrument failure until proven otherwise.**

**N7 — A cheap screen's NO recorded as a closure.** **C-307**, and the entry quotes the chair verbatim:

> *"we have dropped multiple things on the floor … by doing quick smart tests to see if real
> implementation makes sense, then dropping real implementation because the test told us to. Then me
> later insisting that we try for real, and then it turns out that it in fact works. This happens so
> much."*

The instance: #294 concluded "correspondence superficial" from σ_max measured **on a teacher-forced
model**, when α is a *training* parameter and training under teacher forcing reshapes the Jacobian. The
rule now: **a proxy closure must carry its false-negative mode and a reopen trigger.** Applied
retroactively to #290/#291/#294.

**N8 — The harness is green for a reason unrelated to the code.** **C-283** — three tests shelled out
to `ruff`, a binary the environment never installs. ⛔ **[CORRECTED 2026-10-20: present tense is
FALSE. `environment.yml:31` installs `ruff==0.14.14`, and `technical_risk_register.md:2340` says so
inside the very entry this sentence quotes: "Fixed in this PR by adding `ruff==0.14.14` to
`environment.yml`'s pip list." The scar is real as HISTORY; the register now files C-283 under cluster
21, "prose with a self-dating expiry" — which is this sentence.]** Red on **every CI run since at least 2026-08-14**:
*"every CI `test` result before 2026-08-15 should be read as red for this reason unless shown
otherwise."* **C-284** — `gpd_pwm_fit` saturates one-sidedly (true ξ 1.00/1.10/1.30 → 0.92/0.96/0.99)
and **all three published h=36 tail rows sit at the ceiling, 0.951–0.965**; the only test pinned ξ=0.5,
the last value where the estimator is still accurate.

### 3.4 What does not detect any of this

From 31 traceable retractions, ranked by yield:

| detector | hits | note |
|---|---|---|
| `/code-review` or `/expert-code-review` **by a non-author, in a clean context** | **9** | The single highest-yield detector. One round found errors *introduced by the previous round's fix*. |
| Two independently computed numbers disagreeing — **and the disagreement traced, not averaged** | 3 | |
| A systematic recompute of every published figure | 1 (found 3 of 9) | Reusable and cheap. |
| Re-running on a converged / non-floored vehicle | 2 | |
| A **control arm** or positive control | 2 | The control is what made the treatment legible. |
| Running the pre-registered rule's own decision table properly | 2 | The rule already had the answer. |
| An **impossible** result | 1 | Named in the source as **luck**. |
| User escalation | 3 | Produced the potency gate and C-324. |
| **The test suite** | **0** | |

⛔ **And the number that should govern practice:** in the September implementation postmortem,
*"defects found by my own review: **0**"* — every one of ≥15 defects found *after* the author judged
the code correct was found by a **mechanism**, not by looking harder. Against that: 2,079 green tests
and 25 runtime-verified findings, 0 caught; 27 of 40 tests passing with the call site deleted; 17 of 33
mutations surviving the fix for the entry about tests that cannot fail.
---

## §4 — Solved and not solved

"Solved" is used in two senses below and they are kept apart, because conflating them is how a
question gets closed by exhaustion and later reopened at cost. **CLOSED** = measured, with a stated
scope. **CLOSED-BY-EXHAUSTION** = we stopped asking; the evidence does not support closure.

### 4.1 Closed — measurement questions

| question | verdict | evidence | scope |
|---|---|---|---|
| Is the long-horizon CRPS win over climatology real? | **No — ARTIFACT, 12/12 at h36** | Epic #263, an exact identity decomposition of archived CSVs, **no GPU** | the 2026-07-29 board; all arms, all targets |
| Can a first-moment aggregate certify calibration? | **No.** MCR ≈1.0 with activation 4–115× wrong | #258 | the 8-model roster |
| Does the rollout decay because the horizon is long, or because the model eats its own output? | **Its own output.** The oracle does not degrade at all across 36 steps | M10, M24 | 1 seed, `violet_visitor`, 160L, `sb` |
| Where in the rollout does the damage go? | **Occurrence 89–95%; magnitude indistinguishable from zero** | I-A, replicated 4/4 vehicles, two body families, two compositions | 1 seed per vehicle — the repo calls this an **escalation trigger, not a conclusion** |
| Is it recall or placement? | **Placement.** `thin:0.75` deletes ¾ of true events and keeps 95–97%; scrambling locations costs **5×** (M12, `violet_visitor`) | M3/M4 → M11/M12 | replicated across two vehicles. ✅ **CORRECTED 2026-10-19 (open finding 4):** this row read **31×**, which is M3's figure on the **floor-limited** vehicle. `RESULTS_LEDGER.md:207` classes M3 *"understated, replicated … cite the corrected number"*, and the corrected number is M12's **5×** (0.0486 vs a 0.2569 control). §2.7 gave both; this row gave only the floored one, under M12's provenance |
| Does more training fix the rollout? | **No — training is closed as a retention lever.** Retention 0.03 → 0.600 (160L) → 0.683 (300L, n=4) → 0.692 (600L). T=0 skill and the ceiling keep rising | M26–M29 | shape from 5 lesson counts; the plateau itself is a 2-point claim |
| Does scheduled sampling help? | **No — it hurts, at every horizon** | M30–M33, 4v4 seeds, p=0.0286 | formally UNDERPOWERED (does not clear 3×MDE); direction established, magnitude not |
| Does the model beat persistence at long horizons on AP? | **Yes at L=300, all four seeds, every horizon** | M34–M37 | one vehicle, AP only; the CRPS win remains ARTIFACT |
| Does pairing buy power? | **Yes — MDE 0.0541 → 0.0086, 6.3× on identical data** | M40 | one horizon, two arms, one seed |
| Is the packaging boundary a source of drift? | **No — byte-identical, 190 tensors × 4 models** | M76 | this machine |
| Can a multi-τ roster beat the soft gate? | **No — algebraically impossible** | M69 | exact; zero GPU |
| Is the ensemble buying diversity? | **No on error diversity — it is buying sample count.** Errors on the **composed forecast** correlate 0.96–0.99 across all 28 pairs. ⛔ *Corrected 2026-10-11:* this row said *"**Gate** errors"* — the same operand error §7-S7 fixed twice; the §7 pass checked §0 and not §4. ⚠️ And see §7-S7: on the **delivered field** the members disagree substantially (r median 0.439), so what is dead is error diversity, not output diversity | M66, M67 | one artifact per config, one origin, `sb` |

### 4.2 Closed — and the closure is what should be read, not the number

| question | what is closed | what is **not** |
|---|---|---|
| Does freezing the cell state help? | **Direction: yes**, four seeds, paired CI excludes zero, +0.039 AP@h18 (M38/M48) | **Mechanism: no.** M43 refuted the out-of-distribution-state hypothesis; M52 is explicitly "CONSISTENT-WITH, not causal". And M8's original 13× magnitude is **retired** — the control was broken |
| Which half of the recurrent state matters? | The **cell** (M39, M56) | C-292: `hs = o⊙tanh(hl)` makes `cell ≈ all` **architecturally predetermined**, and ⛔ **[CORRECTED 2026-10-20: the caveats in this row are contradicted by the ledger row it cites.
`RESULTS_LEDGER.md:414` (M56): "the 89% attribution **is confirmed at n=4**; its pre-registered
falsifier (all > cell beyond spread) **did not fire**" — so "was withdrawn" is false. The same row
also records `hidden` alone as **real at long horizon** (+0.0186 AP@h36, 4/4 seeds), i.e. CONTESTED at
h18 rather than structurally null. Either M56 or this row is wrong, and §4 did not flag that they
disagree.]** C-323 makes the hidden-freeze null structurally guaranteed. The earlier "cell carries 89%" was withdrawn |
| Is the amount ceiling real? | **Yes** — spearman 0.303 < persistence 0.367, confound-clean | It is a statement about *magnitude*. Jump-risk/volatility **is** predictable (spearman ≈0.79) |

### 4.3 Not solved — and the reason each is stuck

| # | Open question | Why it is stuck | issue |
|---|---|---|---|
| ⛔ 1 | **Is the h=1 win real?** ⚠️ **TWO COMPARISONS, AND THIS ROW NAMED ONLY ONE — corrected 2026-10-19 (open finding 9).** Against **persistence** it reads **UNDECIDABLE for a purely procedural reason** — no bootstrap CI was computed; point estimates CRPSS +0.197, ΔAP +0.152. ✅ **Against CLIMATOLOGY the CI exists:** `2026-08-15_rollout_ruler_trust_dossier/results/rescore.csv`, `violet_visitor`/`sb`/`h=1` — `crpss_vs_clim` **0.3086**, `delta_AP` **0.1765**, `ci_excludes_zero` **True**, `verdict` **REAL**. ⛔ **So *"no bootstrap CI was computed"* is false for the climatology reference**, and D.5 states the distinction this row did not. ⚠️ **Two cautions this seat adds, neither in the finding:** that row's `zero_share_of_gap` is **0.7866** — 78.7% of the CRPS gap is true zeros, the same signature Epic #263 ruled **ARTIFACT** on at 74.6% — and its interval is wholly negative (`ci_lo` −0.0731, `ci_hi` −0.0426), so **the CI is on a loss difference, not on the ΔAP gain**; it must not be quoted as a CI on AP | the **persistence** half still needs a re-run emitting **per-origin** scores at matched S; the cubes were score-then-deleted | — |
| ⛔ 2 | **Which ledger rows rest on floor-limited vehicles?** M8's best case on the smoke vehicle (0.0912) is **3.6× below** an untreated 300-lesson baseline (0.3298) | the audit was only ever run on the rollout-collapse programme; the postmortem states that "this is the only floored measurement in the repo" is **not established** | **#280** |
| ⛔ 3 | **Can the design resolve the effects we chase at all?** SS: −0.0426 with MDE 0.0541. *"If placement interventions land in the same 0.03–0.05 band, every future experiment returns UNDERPOWERED"* | one of four candidate fixes (a paired statistic) changes the MDE at **zero GPU** and has not been adopted generally | **#281** |
| ⛔ 4 | **Is the truth data correct?** priogrid 149451 reads **27,413** then **32,505** with a **2** between them; the next largest cell-month anywhere is **1,100** | nothing checks the truth field for outliers at build time; filed upstream | C-338, views-datafactory#484 |
| ⚠️ 5 | **Is D=4 enough gate variation for a delivered point estimate?** The gate is deterministic given a pass's mask, so D is the only lever; on `blazing_meteor` the gate kills 1–3 of 4 passes in **407 of 502** signal-carrying cells | "no ledger entry answers it" | **#388** |
| ⚠️ 6 | **Is `aggregate_method` offering the wrong collapses?** `gate × mu` beats mean-of-16 by **+58% to +159%**; `median` is an empty map. ⛔ **SCOPE, added 2026-10-19 (open finding 1): `gate × mu` (`analytic_mean`) is non-zero on **100% of cells** — the dossier says so itself. It is therefore structurally the same object as `gate_only`, which §5.0 disqualifies as *"not like-for-like"*. The figure is a **dense-surface ranking** gain and may not be quoted as a collapse effect comparable to an intervention. ⛔ The like-for-like figure is `tower_point` vs `mean` — **9.9×**.** | indicative only — one artifact per config, one origin, `sb` | **#337** |
| ⚠️ 7 | ⛔ **[CORRECTED 2026-10-20 — see §3's N5: the C-300 quote this row rests on predates M30–M33, which ran `ss_feedback='sample'` in every arm at L=300 across 4 seeds with all floor gates passing, and §4.1 item 7 closes SS on it.]** **Was any scheduled-sampling verdict measuring anything?** All four SS-on roster models trained under a config the repo now rejects, and *"there is no valid, non-floored SS measurement anywhere"* | and #288 asks whether the SS feedback path is differentiable at all | **#295, #288** |
| ⚠️ 8 | **Do the direct-multi-horizon and recursive cubes score identically?** | *"Prove the ruler scores both cube types identically"* is a precondition nobody has met | **#310, #324** |
| ⚠️ 9 | **Is the architecture bake-off attributable?** DualStream replaces the identity top-skip; WideMemory silently drops `state_channels`; MaxBlurPool2d needs an 8×8 grid and dies at the repo's own `window_dim=4`; arms matched on parameters but **not compute** (+5% to +20% wall-clock) | four open issues, no re-run | **#366–#369** |
| ⛔ 12 | **Does AP@h18 predict anything about the served product?** ⚠️ **PARTLY ANSWERED 2026-09-30 and the answer is bad:** across seven collapses of one cube, **AP spans 18.9×**, and the delivered `tower_point` is **last**. So the optimisation target and the deliverable are connected by a transform with a larger effect than most interventions we have tested, and until 2026-09-30 nobody had scored it. ✅ **CLOSED 2026-10-01, and this row did not say so — corrected 2026-10-19 (open finding 5).** Both of the clauses that stood here as open are answered by §5.0's own resolution: run `20261001_012905`, **8/8 models, 300 lessons, pooled**, at the delivered **128 draws**. ⛔ **The row read "still open … queued" while the section 120 lines below it read "RESOLVED".** ⚠️ **What remains genuinely open is narrower:** AP is not computable on the *forecast* months (no truth), so a reading on the **delivered forecast frame** — as opposed to the validation partition — still does not exist | — | **#337, #388** |
| ⛔ 11 | **Has long-horizon training ever been tried?** Aceituno2025 Thm 4.6: minima found on short horizons do not generalise to long ones, gap `O(e^{λ(T_h−T_l)})`. ⚠️ **Scope caveat added 2026-09-30: the theorem is derived for feedforward MLPs with smooth loss and ergodic dynamics; extension to RNNs is discussed as plausible and not proven — and the HydraNet is an RNN.** The mechanism, exponential gradient scaling in a chaotic autoregressive system, is generic, so I believe it transfers; that is INFERENCE. Scheduled sampling, increased teacher forcing and `truncated_nb` all changed **what the model is fed**; **none changed what is optimised** | so three experiments resembled long-horizon training and none was it. ⚠️ INFERENCE, and the sharpest unexamined direction in the record | — |
| ⚠️ 10 | **Should QS99 be in the protocol at all?** C-224 is **Tier 1** and its governance ask is recorded as *unchanged, still open* | requires amending a framework that is not in the repo | C-224 |

### 4.4 Closed-by-exhaustion — flagged honestly

- **"Is training reproducible?"** C-119 is marked **RESOLVED** and cluster 2 "mostly closed", on a
  validation the postmortem itself scoped at **1 lesson / 3 windows** — a caveat the RESOLVED banner
  does not carry. **C-317 is the full-run confirmation and it fails:** retraining the same config, same
  seed, same GPU, same torch reproduces its twin to **148%/114%/94%/78% of the seed-to-seed sd**. The
  two entries contradict each other and both stand. ⛔ **Treat training reproducibility as OPEN.**
- **Whether `crps_events` may select.** C-167's close-out wired it into a decision rule (2026-07-27);
  C-224 forbade it three days later as the Forecaster's Dilemma. C-224 supersedes, C-167's text stands,
  the register does not flag it. ⛔ **[CORRECTED 2026-10-20: it does. C-167 carries
`Update 2026-09-30 — SUPERSEDED IN PART by C-224: crps_events must NOT select`, which names the
conflict, quotes the offending clause, and credits this report's §3.2 for raising it. `Appendix C`
claims that edit ("Q4 … produced the edit: C-167 now carries the supersede note") and never noticed
this sentence still asserts the absence.]**
- **⛔ **[CORRECTED 2026-10-20: not six — ALL FIFTEEN, and the error runs toward under-reporting.
`technical_risk_register.md:14`: "§Disagreements curated for the first time — 15 entries, **0 ever
marked resolved**"; `:4785`: "this section has never been curated. **All 15** entries read as live".
The curation pass then flagged five as plausibly settled by events. "Six" matches nothing in the
source, and the quote attached to it is about all fifteen.]** Six of fifteen register "disagreements" have never been curated** — *"not one carries a resolution
  marker, and several have been answered by events rather than by a ruling. A disagreement that events
  have settled but that still reads as open is the same defect class as C-303."*

### 4.5 A third disposition — "decided, built, unread"

⛔ **Neither CLOSED nor CLOSED-BY-EXHAUSTION, and it is worse than either.** §4.4's definition is *"we
stopped asking; the evidence does not support closure"* — and both of its entries fit it exactly, because
in each case **somebody concluded something** and the conclusion is what the evidence cannot bear
(a `RESOLVED` banner contradicted by C-317; a close-out contradicted three days later).

Calibration has no such conclusion. **Nobody decided it was fine.** It was prescribed, built, and then
simply never read. There is no banner to contradict and no closure to be honest about — and *that is why
it survived three months and a delivery.* A wrong closure at least leaves a claim someone can attack.
This left nothing.

So it is a third row, and §4.4's definition stays as it is rather than being widened.

#### And it is a class, not an incident — measured 2026-09-30

If the disposition exists once, the question is how many times. The candidate pool is §5.8's
"computed, orphaned" column. **The distinction that does the work: was the metric merely *built*, or was
it *prescribed* first?** A metric someone wrote on spec and nobody wired up is ordinary dead code. A
metric the register or a dossier explicitly **asked for**, which was then built and never read, is a
**decision that silently failed to execute** — which means the prescribing mechanism does not work.
⛔ **[CORRECTED 2026-10-20: `recall_mass` DID execute and its verdict was published.
`RESULTS_LEDGER.md:100`: "**K1 PASS 8/8** (`recall_mass` h36 0.016–0.057 vs a 0.01 floor, 13 origins —
the worst clears it by 1.6×)", attributed to M70. A criterion that produced a recorded 8/8 verdict
belongs in pile 3 ("read once by the dossier that built them, then dropped"), not pile 1. ⛔ Pile 1
then holds ONE member (PIT), so the load-bearing sentence "Two instances make it a class" FAILS on its
own arithmetic. What survives: it is applied by a `print()` against a hardcoded roster, with no exit
code — which is pile 3's complaint, not pile 1's.]**

Grepped the register and every dossier design / plan / acceptance document for each name. The pool
partitions four ways, and the piles are not the same finding:

| pile | metrics | prescribed where | state |
|---|---|---|---|
| ⛔ **1. Prescribed as a GATE, built, never wired** | **positive-cell PIT** · **`recall_mass`** | register `:821` (2026-06-21, *"as a conditional diagnostic only"*) · `2026-09-06_ensemble_roster_dossier/06_acceptance_criteria.md:27-28`, which **defines the formula and the threshold**: *"K1 FIRES if recall_mass < 0.01"* | PIT: built, RNG unseeded, in no `HEADLINE_COLUMNS`. `recall_mass`: computed in **one** dossier tool, the name appears in **no score CSV, no glossary row, and no other code**, and the criterion is applied by a **`print`** against a hardcoded roster |
| ⚠️ **2. Prescribed for REPORTING, built, emitted — but invisible to the decision rule** | `n_false_pos`, `mag_on_false_pos`, `mag_on_true_pos`, `act_pred`/`act_true` | `2026-08-24_truncated_nb_dossier/05_analysis_plan.md:83` requires them *"reported at every horizon"*; also `architecture_bakeoff/02_design.md:94`, `ensemble_roster/05_analysis_plan.md:47` | The reporting requirement **is met** — they are in the CSV. They are absent from `HEADLINE_COLUMNS` and unreadable by `verdict_token`, so no verdict can see them |
| ⚠️ **3. Read once, by the dossier that built them, then dropped** | `moran_i`, `conc1pct`, `fss_1` | `2026-08-31_sharpness_diagnostic_dossier/05_analysis_plan.md` — `moran_i` **primary**, `conc1pct` secondary, S3 a falsifier on their disagreement | Read **correctly and once**: S1 and S5 both fired and killed a hypothesis. Nothing has read them since |
| minor **4. Emitted to human-readable audit output, gated on by nothing** | `gate_auc`, `tail_miss`, `ratio_p10/p90` | nowhere — **zero** prescriptive mentions | Ordinary. They appear in per-run audit files; nothing claims they should do more |

✅ **And the control case that proves the repo can do this properly.** `waste_frac` sits in the same file
as `recall_mass`, and its docstring says: *"Reported alongside, **not as criteria**."* An explicit,
honest, recorded disposition. **The whole difference between `waste_frac` and `recall_mass` is that one
declared its status and the other did not** — so this is not a capability the repo lacks, it is one it
applied unevenly.

⛔ **The finding.** Pile 1 has **two** members and both are load-bearing: one is the calibration
diagnostic whose absence let an all-zero surface ship, and the other is **the production roster's own
acceptance criterion**. Two instances make it a class, and it is the second class in this record — after
the register's cluster 18 — that reads as N unrelated incidents until someone names it. *(MEASUREMENT for
the partition, which is a grep; INFERENCE that the prescribing mechanism is what fails, though I cannot
see another reading when a kill criterion is applied by a `print`.)*

---

## §5 — Blind spots

Axes we currently cannot see. Each is sourced, and each is a *present* state of the apparatus, not a
historical anecdote.

⚠️ **Basis, since this report's own discipline demands it:** every claim in §5 is a **MEASUREMENT about
the code** — a reading of the apparatus as of 2026-09-30, by grep and by tracing call sites — not a
measurement about a model. Where a claim rests on someone else's run, it says whose.

### 5.0 ⛔ The delivered product is not scored, and the predictive distribution is not calibrated

Two holes, and they are the largest in the report because they are about the output rather than the
research.

**(a) Nothing scores the served surface.** Every instrument in Appendix A takes a sample cube and ⛔ **[CORRECTED 2026-10-20: FALSIFIED BY
APPENDIX A ITSELF. `delivery_frame_audit.py` (A.1) reads "a delivered handover folder; no truth, so no
AP/CRPS/calibration" and is annotated "the only instrument that reads the DELIVERED product rather
than a sample cube — a partial answer to §5.0(a)". Four further A.1 rows take no cube and return none
of the three: `field_sharpness.py`, `potency_check.py`, `arm_identity_check.py`, `arm_postflight.py`.
A.1 gained that row on 2026-10-10 and this sentence was not updated. ✅ The weaker claim — nothing
SCORES the served surface — survives, since `delivery_frame_audit.py` has no truth.]**
returns AP, CRPS or Brier. The FAO receives a **collapsed point surface** plus HDI bounds. The collapse
is not a formatting step — M70 measured that `gate × mu` beats the mean of 16 draws by **+58% to +159%**
⛔ **(scope, 2026-10-19, open finding 1: `gate × mu` is non-zero on 100% of cells, so this is a dense-surface
ranking gain and is disqualified by the same argument this section applies to `gate_only` below — the
like-for-like figure is `tower_point` vs `mean`, 9.9×)**
at identical total mass, and `median` is an **empty map** (0.05–0.14% of cells non-zero). So the stage
with the largest measured effect on the delivered number has **no instrument at all**, and #388 asks
whether D=4 gives the gate enough variation to support a point estimate while recording that *"no ledger
entry answers it."*

The concrete instance, supplied by the `models` session from a 40-lesson rehearsal: the served
`tower_point` surface was **0.0 for all 2,330,712 cells on global land, in all three targets**, while
**every structural check passed** — manifest valid, coverage right, months right, file size 98.1% of a
real run's — and a downstream service reproduced the zeros independently. The model was undertrained and
the estimator was behaving correctly. ⛔ **The finding is that nothing in our inventory would have
flagged it.**

#### And when the collapse IS scored, the delivered one is the worst of seven

Measured 2026-09-30 by the `models` session. ⚠️ **Scope, on the line: `blue_stranger`, validation
partition, 6 origins, **16 draws**, `lr_sb_best`, Africa+ME. One model — NOT the pooled ensemble.**
Seven collapses of the **same cube**:

| collapse | AP | mass_ratio | nonzero_share |
|---|---:|---:|---:|
| `gate_only` | 0.29854 | 0.02864 | 1.00000 |
| `mean` | 0.15659 | 0.06506 | 0.01450 |
| `q95` | 0.15583 | 0.29371 | 0.01450 |
| `gate_x_condmean` | 0.15539 | 0.12760 | 0.01450 |
| `conditional_mean` | 0.12435 | 0.33015 | 0.01450 |
| `median` | 0.04582 | 0.00658 | 0.00096 |
| **`tower_point`** — **what the FAO is served** | **0.01578** | **0.00070** | **0.00009** |

`mass_ratio` = collapsed total ÷ observed total. **AP spans 18.9× across collapses of one cube.**

⛔ **The like-for-like comparison, and it needs no caveat: `tower_point` vs `mean`.** Both are magnitude
collapses of the same cube. `tower_point` is **9.9× worse on AP** and delivers **0.07% of observed mass**
against `mean`'s 6.5%. `mean` is roughly what a researcher looks at; `tower_point` is what the partner
receives.

⚠️ **Do not lead with `gate_only` winning** — the `models` session flagged this and they are right. AP
measures discrimination of occurrence and the gate *is* the occurrence model, so scoring it on AP is
close to scoring it on its own objective. `gate_only` 0.299 vs `mean` 0.157 is not like-for-like, and a
magnitude-aware score would treat them differently.

#### ⛔ RESOLVED 2026-10-01 at the delivered 128 draws, and it resolves the hard way

> ⛔ **SCOPE CORRECTION, 2026-10-20 — THE "EMPTY MAP" FRAMING BELOW IS OVERSTATED FOR THE DELIVERED
> PRODUCT, AND THIS REPO'S OWN TIER-1 REGISTER ENTRY ALREADY SAID SO.** Run `20261001_012905` is
> **validation, Africa+ME** — not the global delivery. `reports/technical_risk_register.md:4776`
> (**C-345, Tier 1**, 2026-10-11), whose `Location` field *names this section*:
>
> *"⛔ **That does not transfer to the delivery.** On the 2026-09-29 handover, **146,847** cell-months
> have ≥8/16 positive draws, so a point map here is sparse but **not** empty. The 'empty map' framing was
> overstated for this product; the *inversion* above is the durable finding and does not depend on it."*
>
> ⛔ **§5 cited C-345 zero times until this correction**, and the register was last updated 2026-10-11
> while this report was revised on the 19th. Independently,
> `reports/2026-10-11_conflictology_empty_mode_prereg.md` (on `development`) measures the live delivery:
> **18,407** `sb` cells have `map == 0` **and** `hdi95_upper > 0`, with `severe_scenario > 0` in **all**
> of them ⇒ *"nothing is lost on the wire"*.
>
> ✅ **What survives, and it is the durable half:** the *direction* — more draws destroyed the estimator
> rather than rescuing it — with a mechanism (the zero-fraction flip point moves from ~25–35% at S=16 to
> **10–20% at S=128**). ⛔ **What does not survive is "empty", and the defect re-scopes**: not *"the map is
> empty"* but *"the column named as the headline estimate reads zero while four siblings in the same row
> say otherwise, and nothing tells the reader which to trust."*

The caveat on the worst row was that **16 draws** might be unfair to a tip estimator on a 99.75%-zero
field — that `tower_point`'s near-zero could be small-sample behaviour of that estimator rather than a
property of the collapse. **It was not.** Run `20261001_012905`, 8/8 models, **300 lessons**, pooled:

| draws | 16 | 32 | 48 | **128 (delivered)** |
|---|---|---|---|---|
| cells `tower_point` keeps | 204 | 39 | 19 | **0** |

**More draws destroyed the estimator rather than rescuing it.** At the delivered draw count
`tower_point` returns **zero cells in all three targets**, and `mass_ratio` is **exactly 0** — not small,
zero. Against what the pooled cube actually carries:

| target | cells with signal | `mean` keeps | `tower_point` keeps |
|---|---|---|---|
| `lr_sb_best` | **46,488** | 46,488 | **0** |
| `lr_ns_best` | 14,603 | 14,603 | **0** |
| `lr_os_best` | 17,152 | 17,152 | **0** |

⛔ **So `tower_point` is now worse than `median`**, which M70 already called an empty map (`median` keeps
583 / 75 / 0). And the vehicle under this is no longer undertrained: **eight converged 300-lesson models,
46,488 cells of genuine signal in `sb` alone, and the collapse the partner receives discards every one.**

✅ **The `models` session, which raised the caveat, retracted it in the strongest terms once the series
completed.** Recording that: the caveat was theirs, it was a good caveat, it was pre-registered as the
thing that could demote the finding, and the measurement killed it. **§5.0 does not become a footnote. It
becomes the delivery finding.**

✅ **One independent confirmation:** `median` returns **0.096% non-zero** — an empty map — reproducing
M70 exactly at a different scope, by a different route, on a different vehicle.

**(b) No calibration of the predictive distribution, at any aggregation level.** §3 catalogues metric
blindness thoroughly and contains **nothing** about whether the distribution is honest. Measured by the
`models` session on `blue_stranger`, validation, 468 origin-months, summing draws:

| | measured | wanted |
|---|---|---|
| 90% interval coverage | **0% / 1% / 7%** | ~90% |
| PIT median | 1.000 (share >0.95: 99% / 96% / 79%) | ~0.5 |
| MCR | 0.092 / 0.062 / 0.147 | ~1.0 |
| `Var(total) / ΣVar(cell)` | ~0.95 | <1 if cells are dependent |

Two separable faults: **location** (predicted totals ~10% of observed — the timid body again) and
**variance** (cells behave as independent marginals, so any aggregate interval is too narrow).

⛔ **The variance half is CORRECTED 2026-10-01, and in our favour — I had it wrong.** I predicted the
pooled product would show no cross-cell dependence, reasoning from the per-model emit path, which contains
no spatial mechanism. **Measured, it shows a lot:** dependence ratio **9.81** on `sb`, 2.16 on `ns`, 1.88
on `os` — against an independent-sampling baseline of 1.0. The coupling is a **between-model level shift**
(one model hot across the whole field, another cold), and it enters at the **pooling** step, one level
above the path I read. ⛔ **That is this report's own §3.1 shape (c) — measurement on the wrong object —
committed by me, against my own pre-registration** (`reports/2026-09-30_dependence_ratio_prereg.md`, scored
there). ✅ **Consequence, and it is the one piece of good news in this section: country and regional totals
from the pooled product are *less* falsely confident than either session predicted.** The width half is
partly rescued, and pooling is what buys it. ⚠️ The **location** half is untouched — the interval is
honestly wide around a number that is still ~10× too low.

⚠️ **And the instrument for the first row already exists one repo over.** The `views-evaluation` session
reports `Coverage(alpha)` — exactly the 90%-interval coverage measured by hand above — as **implemented
and tested**, alongside `MIS(alpha)`, `QIS`, `Ignorance`, `Brier_rgs_sample` and `twCRPS(threshold)`.
§8.1 prefers wiring an existing metric to adding one; **these are existing, just not in this repo.**
⚠️ But read the limits with them: that library has **no confidence intervals, no bootstrap, no paired
tests, no significance machinery at all** (they grepped the package), **no reference forecasts,
baselines or skill scores**, and **nothing that scores a collapse**. So the paired origin-block CI
(§2.5) and the artifact verdict (§2.3) are strictly ahead of anything the platform has, and §8 row 2 has
zero platform support. A migration would close §5.7 and lose §2.3 and §2.5 unless they move too. ⚠️ **Scope, theirs:** one model, 16 draws, log1p space, Africa+ME — **not** the pooled
ensemble, which is the open question. Tooling and pre-registered falsifiers: views-models#530.

⚠️ **This finding is predicted, and its fix exists in the literature.** Supplied by the `library`
session: **Randahl2024**, written for conflict data, shows standard conformal prediction achieves correct
**marginal** coverage while systematically **over-covering dense regions and under-covering sparse ones**
on a skewed outcome — which is exactly the 0%/1%/7% shape measured above, the zeros over-covered and the
events under-covered. Candidate fixes, none evaluated here: **Wang2024_SAUC**, a *sparsity-aware*
calibration framework that calibrates zero and non-zero predictions **separately**, built for sparse
zero-inflated spatiotemporal data — the closest thing in the library to a purpose-built answer; and
**Kuleshov2018**'s post-hoc isotonic recalibration, which is cheap. ⚠️ **Guo2017** is worth reading beside
C-328: it finds BatchNorm **worsens** calibration, and this repo has an open finding that ~half of every
shipped artifact's BN statistics were accumulated on flipped fields.

⚠️ **And a distinction §5.0(a) does not yet draw, which changes what the fix is.** Is the median an empty
map because the **collapse** is wrong, or because the **distribution** is too zero-concentrated for any
point summary to carry content? Bracher2021 argues absolute error is the right point-forecast loss in
epidemic settings *because* it implies reporting the median — and if our median is structurally zero, then
no reasonable point-forecast loss applied to it would see anything. **The first is a choice of estimator;
the second is a property of the model, and only the first is fixable by changing `aggregate_method`.**
The 128-draw repeat bears on this directly.

⛔ Note the governance shape of (b), corrected: **the register prescribed exactly this diagnostic and
nothing ever banned it.** `technical_risk_register.md:821` (2026-06-21): *"Use integrand-weighted
threshold-weighted CRPS on ALL cells (proper) as the selector, **with a positive-cell PIT as a
conditional diagnostic only**"* — the selection-vs-diagnostic distinction, drawn correctly, three months
before the delivery. `:652` says the same. `scripts/proper_score_audit.py:82` implements
`randomized_pit` and its header states its own limits. What is true of FAO-02 is only that it rejects
PIT **for selection**. ⚠️ An earlier draft of this report claimed active-cell PIT was banned outright
by a scope document; that was a misreading of an adjacent clause — see Appendix C.

### 5.1 ⛔ The largest hole: nothing guards against a spatially displaced forecast

We have **measured** that occurrence, body magnitude and gate–body alignment are all preserved by a
90-cell roll that costs **48× in AP** (C-319). ⛔ *Corrected 2026-10-11:* an earlier version said
`GLOSSARY.md` carries this as a ⛔ on **three** metric rows. It carries it on **one** — `:157`,
`gate–body alignment`. Rows `:155` and `:156` do carry ⛔ marks, but for C-318 (conditioned statistics
on a vanishing denominator), an unrelated defect. And ⛔ **[CORRECTED 2026-10-20: FALSE — four tests plant one, and one asserts a metric responds.
`tests/test_field_sharpness.py:46` defines `_displaced()` as a double `np.roll`, used at `:70` (Moran's
I must NOT move), `:86` (`conc1pct` must not move) and decisively `:118`, inside
`test_the_fss_ratio_is_NOT_a_blur_detector`: `assert displaced < blurred` — a tracked test that plants
a rolled field and asserts a metric moves on it, with the docstring "This test exists so that reading
cannot be reintroduced without deliberately breaking it." `tests/test_roll_diagnosis.py` (6 tests) and
`tests/test_per_step_roll.py` (10) validate the instrument besides. ✅ What survives: no GATE reads any
of it.]** **no test plants a rolled field and asserts that
any metric fires.** The only artifact is a one-off cross-correlation script in one dossier.

So three field statistics are measured placement-blind (M53) and unguarded, of which **one** is
documented as such in the locked glossary. The measurement is unaffected; the documentation claim was
two-thirds wrong.
A future arm that improves all three and destroys placement would be reported as an improvement.

⚠️ **And it is a platform-wide hole, not a hydranet one.** Reported by the `views-evaluation` session:
the platform metric catalogue carries **Variogram** and **pEMDiv** — pEMDiv being the metric built for
exactly this problem — and both are `implemented=False` **stubs**. So no repo on this platform can
currently score spatial displacement, and the two candidate instruments are unbuilt rather than absent.
They have offered to implement them there rather than in `scripts/`, which is the right home.

### 5.2 ⛔ The method rule that no scorer enforces

*The reference's S must equal the arms' cube width.* It is stated in one module
(`rollout_ruler_core.reference_sample_width`) and **not called by either HEADLINE scorer.** ⛔ **CORRECTED 2026-10-20: three tools DO call it** — `2026-08-15_rollout_ruler_trust_dossier/tools/rescore_v2.py:44,65` (**the tool that wrote the `rescore.csv` this report's headline numbers come from**), `tools/mde.py:118,143`, `tools/tail_index.py:34,84`; `assert_sample_cube` is called by `tools/partition_audit.py:55,159`. Register: `technical_risk_register.md:2265`. The defensible claim is that the two HEADLINE scorers do not call it. Live violators:
`rollout_skill_score.score_horizons(add_persistence=True)` and
`score_v2_horizons.score_horizons_v2(add_persistence=True)` inject a **1-sample** persistence arm into
a table of **S=16** arms; `pool_ensemble` concatenates on the sample axis so the pool's S is
`n_members × S_member`, wider than any member it is compared against.

⚠️ **And the mechanism is narrower than it looks — a peer's proposed explanation was checked and does
not hold.** The `views-evaluation` session found that their kernel's `_guard_shapes` reshapes a 1-D
`y_pred` to `(N,1)`, so a **point** forecast handed to a **sample** metric is silently scored as a
one-member ensemble, and suggested that is where our silence comes from. Checked: **it is not.** Both
persistence-injecting scorers import `crps_ensemble` from the **frozen lodestar**, not
`calculate_crps_native`; and the persistence arm is built by `np.stack([...])` into shape **`(N,1)`**,
not 1-D, so nothing is being silently reshaped. Our frozen ruler accepts it because **arithmetically it
*is* a legitimate 1-member ensemble** — at `m=1` the `2/m²` spread term is exactly zero and CRPS reduces
to MAE.

⛔ **So the defect is not a silent reshape. It is that nothing compares S across arms at all** — which is
precisely what `reference_sample_width` was written to do and what no scorer calls. Their finding stands
as a **platform contract** finding (filed as views-evaluation#98, with our 31%-understatement figure as
the cost evidence); it is not the cause here.

Related and equally live: **"AP" means two different things** depending on whether an arm has a gate
cube — a gated arm is ranked on a continuous probability, a gate-less arm on a statistic taking S+1
values, which at S=1 is **two levels**. Exactly **one** instrument refuses the mismatch
(`ap_diff_origin_block_ci`); the two headline scorers do not.

### 5.3 ⚠️ FG-C — the floor-gate clause that matters — does not run on two of its three callers

> ⛔ **HEADING CORRECTED 2026-10-20.** It read "does not run", full stop. **Verified false:**
> `reports/2026-09-04_input_noise_dossier/tools/check_floor.py:56` passes `mde_ap=MDE_AP` (`:32`,
> `MDE_AP = 0.06`), so FG-C evaluates to PASS/FAIL there, and that file's own docstring says "FG-C is
> REPORTED, not binding … It is computed, printed, and carried into the write-up." ⚠️ Also: omitting
> `mde_ap` does **not** silently pass — `scripts/floor_gate.py:181-205` returns `PROVISIONAL` and flags
> `missing_binding`. ✅ The paragraph below is correct about the other two callers
> (`lesson_curve_gate.py:182`, `ss_sweep_gate.py:172`), and `Appendix A` already stated the
> correctly-scoped version.

`postmortem_floor_limited_vehicle.md:175` names FG-C as *"the one that names the real failure … not
'the model is bad' but 'the effect we came to measure is smaller than what this setup can resolve'."*
Both gates that call the floor gate — `lesson_curve_gate` and `ss_sweep_gate` — call it **without
`mde_ap`**, so **FG-C never evaluates on either path**. And FG-B (worse than climatology) is
**advisory and never binding**: a control worse than climatology still PASSes.

Separately, `check_floor.py` records that **no dossier invoked the floor gate between 2026-08-26 and
2026-09-04** — the C-299 mitigation lapsed for over a week, silently.

### 5.4 ⚠️ The red-team exists, and is pointed at the wrong target

⛔ **CORRECTED 2026-10-19 (open finding 2). This paragraph claimed more than exists, and three of the cases
it named were deliberately descoped.** The DRAFT's own superseding header says `tests/test_activation_metrics.py`
*"already implements **4 of its 6** degenerates"*, and
`2026-08-15_rollout_ruler_trust_dossier/SCOPE.md:50` says *"Do not build the spiky / frequency-right /
magnitude-right cases."* ⛔ **Verified against the tests: no spiky, frequency-right or magnitude-right case
exists in either file** — `test_activation_metrics.py` carries collapse, bloom, calibrated, a perfect oracle
and climatology; `test_rollout_ruler_core.py` carries `test_degenerate_all_zero_forecast_scores_badly_on_skill`
and the climatology battery. **The count of seven also counts the known-good reference as a degenerate; the
DRAFT counts six degenerates plus that reference.** ⚠️ **And the paragraph contradicted itself two sentences
later**, conceding the spiky case is tested *"only on a uniform magnitude field"* while asserting it was
implemented. **The section's real point survives intact and is strengthened: the cases that DO exist all
exercise `activation_metrics` in isolation.**

~~Better than expected: **all seven** degenerate forecasts from the DRAFT's §5 proposal are implemented —~~
all-zero, climatology, persistence (partial), over-concentrated spiky, frequency-right,
magnitude-right, and a known-good oracle — in `tests/test_activation_metrics.py` and
`tests/test_rollout_ruler_core.py`.

But: **every one of them exercises `activation_metrics` in isolation.** ⛔ **[CORRECTED 2026-10-20:
FALSE on both clauses. `tests/test_rollout_ruler_trust.py:211`,
`test_climatology_passes_through_metric_row_unchanged`, builds a climatology cube — itself one of the
two degenerates this section says DOES exist — and calls `score_v2_horizons._metric_row` directly,
asserting the 21-column contract and finite `crps_all`/`crps_none`.
`tests/test_score_v2_horizons.py:91-130` feeds a hand-built fixture through `score_horizons_v2` AND
`lodestar_score.score_models` with hand-computed assertions. ⚠️ Both skip if the gitignored dossier
tools are absent; both run here. ✅ What survives, and it is the real gap: **no test asserts the
emitted row SEPARATES the degenerates** — the DRAFT's Figure-1 table is still not done.]** Nothing
feeds a degenerate cube
through `score_v2_horizons._metric_row` or `lodestar_score.score_models` and asserts that the emitted
**row** separates the cases. The DRAFT's own TODO — *"tabulate which metric separates which degenerate
— that table is the paper's Figure 1"* — is **not done**. ⛔ **[CORRECTED 2026-10-20: this sentence is the self-contradiction the 2026-10-19 correction above
says it was fixing, left standing 12 lines below it. That correction states "no spiky,
frequency-right or magnitude-right case exists in either file" — verified true — while this sentence
presupposes a spiky case exists and is tested. The sentence is struck.]** ~~And the spiky case is
tested only on a
**uniform** magnitude field; no test feeds one 99,465-value cell, which is the shape that would leave
`size_ratio` (a median) unmoved and `mcr_all` at ≈1.~~

### 5.5 ⚠️ We audit whether we measured the model right, never whether the target is right

The register says this about itself, verbatim: *"The register has extensive coverage of whether we
measure the model correctly, and **none** of whether the thing we measure against is correct."*
Consequences on record: C-338 (an implausible truth outlier, filed upstream, unchecked at build time)
and **#282** — a baseline defect that **scales as 1/n_origins**, so a single-origin study scores
persistence as **exactly zero, silently**.

> ⭐⭐ **FIRST PARTIAL CLOSURE, 2026-10-25 — and the closure itself produced the deeper lesson.**
> `reports/2026-10-25_sharpness_ceiling_prereg.md` is the first instrument in this project pointed at
> the **target** rather than the model. It is free (laptop, **78 s**, $0) and it generalises to any
> normalised skill statistic. What follows is the transferable part; the measurement's own numbers are
> in that prereg and in `GLOSSARY.md §5d`.

#### ⭐⭐ L1 — a normalised score silently asserts that its own ceiling is 1.0, and on a noisy target that is an assumption, not a fact

`2026-10-21_body_scoreboard.md` B-1 reports *"the body carries **0.391** of its due spatial structure at
one cell"*. A ratio against truth reads as *"39% of the way to perfect"* — but **only if perfect is
attainable**. On a 99.7%-zero count field assembled from event reports, a large share of the truth
field's own spatial variation may be sampling noise that **no model could or should reproduce**. Nobody
had checked. Three days of experiment design rested on it.

**The method, and it is cheap and general.** Split the in-sample months into two independent halves —
two noisy observations of the same latent field `f`. Then `gamma_A = gamma_f + nug_A`,
`gamma_B = gamma_f + nug_B`, and the **cross**-variogram `gamma_cross = gamma_f`, because the two
nuggets are independent and mean-zero. So `gamma_cross/gamma_B` is the reproducible fraction, **with no
fitted parameter**. ⇒ Any ratio-to-truth metric can be given a measured ceiling this way, for the price
of computing it twice.

#### ⛔⛔ L2 — THE DEEPER LESSON: the ceiling must be measured with the SAME estimator it will be compared against

This is the finding, and it cost a wrong conclusion that was published for about an hour before
`/falsify` caught it.

The first measurement returned `C(1) = 0.498` and the prereg's own decision rule concluded *"the arm's
upside is capped at 0.498, not 1.0"* — i.e. **0.391 is already most of the way to the ceiling, don't run
the experiment.** That was wrong, and wrong in the direction that kills work.

| | estimator | what it measures |
|---|---|---|
| **B-1**, which produced `0.391` | `gamma_model/gamma_truth` **divided by the ratio of the two fields' variances** (the same-shape null) | **shape** |
| the ceiling as first measured, `0.498` | `gamma_cross/gamma_B`, **not** variance-normalised | **amplitude** |

Re-measured with **B-1's own** estimator on the two halves: **0.983 – 1.009 across `d = 1…32`**, within
1.7% of 1.0. ⇒ **The true ceiling for `0.391` is ~1.0.** The variance normalisation cancels exactly the
noise the first number measured: two noisy halves of the truth have the **same variogram shape** even
though each is individually noisy in **amplitude**.

⇒ **The generalisable rule.** *A normalising step in a metric changes what its ceiling is.* The ceiling
of a normalised statistic is **not** the ceiling of its unnormalised cousin, even when both are
dimensionless and both live in `[0, 1]`. ⛔ So "measure the ceiling" is not enough — **measure the
ceiling by running your actual metric on two independent halves of the truth.** Substitute half-A for
the model and half-B for the truth, change nothing else. Anything else answers a different question.

⚠️ **Both numbers are real; they bound different things.** `0.498` says slightly under half the
**amplitude** of the truth's one-cell variogram is signal rather than month-sampling noise — a genuine
fact about the data, and one that matters for any metric that is *not* variance-normalised. It simply
does not bound B-1. Locked as two distinct terms in `GLOSSARY.md §5d` for exactly this reason.

#### ⭐ L3 — a reliability correction is scale-specific; do not carry a 1/n noise argument across a nonlinear transform

The prereg also converted the half-sample number to a full-sample one with Spearman–Brown
(`C_full = 2C/(1+C)`). The algebra is right. ⛔ **The premise is not transferable.** "Additive noise with
variance ∝ 1/n" is a statement about **counts**; it was applied to a `log(rate + 1e-6)` field, where a
cell that is zero in both halves is pinned at `log(eps) = -13.8` regardless of `n` — error variance ~0,
not ∝ 1/n.

Its own pre-registered sweep refuted it: the correction gives **0.665** on the logged field and **0.262**
on raw counts, **3.3× apart**. Both cannot describe one noise process. ⇒ Withdrawn. **A split-half
reliability needs no scaling argument and should be reported at the `n` it was measured at.** The
correction was the only part of the measurement that tried to extrapolate, and it was the only part that
broke.

⛔ **What none of this removes.** A split-half comparison cancels noise **independent between the
halves** (event sampling, month-binning). **Systematic provider geocoding lands in the same cell in both
halves, survives, and is counted as reproducible structure.** ⇒ Every ceiling above, the ~1.0 included,
is an **UPPER BOUND**. A **low** ceiling is conclusive; a **high** one is not. This is the §5.5 blind
spot reappearing one level down: we have now audited whether the target's *noise* is reproducible, and
still not whether the target is *right*.

### 5.6 ⚠️ Two bootstrap CIs in the apparatus are not reproducible

`mcr_readout` and `proper_score_audit` derive their RNG salt from `hash(str)`, which Python randomises
per process, and no `PYTHONHASHSEED` is set anywhere in the repo. So their confidence intervals and
randomized-PIT numbers **differ run to run**. `gate_reliability` fixed exactly this and says so in a
comment — a stable per-target seed instead of `hash()` — and **the fix was never back-ported to its two
siblings. That is a five-minute change**, priced here because §8 prices everything else. This matters because the reproducibility runbook nominates
`mcr_readout` as the human-readable determinism cross-check.

### 5.7 ⚠️ The ruler is not part of the package, and the protocol is not in the repo

**No forecast metric is computed anywhere in `views_hydranet/` ⛔ **CORRECTED 2026-10-20: FALSE.** `views_hydranet/utils/training_forensics.py:344,359,364` computes `average_precision_score` and `roc_auc_score`, with **AP the default** (`:32`), reachable and imported by `train/training_engine.py:38`. ⛔ **The metric this universal negative misses is this report's own primary metric.**.** The entire ruler lives in `scripts/`
and `reports/*/tools/`, reading arrays this repo does not write. `reports/` is gitignored (tracked only
by `git add -f`), 22 tracked tool files carry hardcoded `/home/simon/...` paths (the register says 7),
and the frozen lodestar's one soft check — support coverage — **cannot fail**: it prints a warning,
continues, and the CSV carries no provisional flag.

⛔ **Said out loud, because the report is otherwise rigorous about self-application: that last one is
cluster 18 shape (b) — a decision rule that cannot deliver a verdict — sitting inside our own frozen
instrument.** It is the same defect class §3.1 catalogues, in the tool whose freeze is the ruler's
credibility.

And the governing framework itself is not in the repo: FAO-02 lives as *"a memory file + a PDF in
`~/brain`"* and **all six in-repo citations are dangling paths.** It cannot be diffed, versioned, or
checked by CI.

### 5.8 ⚠️ Metrics computed and read by nobody; metrics cited and computed by nobody

| computed, orphaned | cited, never computed |
|---|---|
| `recall_mass`, `waste_frac` — the K1 acceptance criterion for the production roster is applied by a `print` and a hardcoded roster list | **QS99** — an FAO-02 guardrail; **no current ruler emits it** |
| `gate_auc`, `ratio_p10/p90`, `tail_miss` — computed at every one of 36 steps, read by nothing | **the ceiling** (oracle AP) — load-bearing for two gates' falsifiers; no script computes it, it arrives as a CSV column |
| `mag_on_true_pos`, `n_false_pos`, `act_pred`, `act_true` — in the CSV, absent from `HEADLINE_COLUMNS`, unreadable by `verdict_token` | **occurrence, body magnitude, gate–body alignment** — three glossary-locked metrics, in no score CSV |
| `fss_1`, `fss_11`, `conc1pct`, `moran_i` — one CSV in one dossier, no gate reads them | **`mcr_all` and `mcr_none`** — the two columns the headline scorer emits have **no locked glossary definition**, while "MCR" already names four different quantities across the apparatus |

### 5.9 minor — gates that exist and nobody runs

`compare_run_determinism`: **one recorded invocation, 2026-06-15**, no test file — and it is the
determinism gate. `ReproducibilityGate.audit_manifest`: *"has no production caller — the genome audit
C-43 shipped never runs"* (C-271). `screen_verdict`: nothing has invoked it since its own dossier, and
its thresholds are frozen to a 300-lesson floor, so it is meaningless on any other vehicle.

---

### 5.10 ⚠️ Five lessons from 2026-10-25 that are not about the ceiling (added 2026-10-25)

#### ⛔ L4 — a VOID condition drafted globally over a deliberate sweep voids your own measurement

The ceiling prereg pre-registered a **sweep** over a free parameter (the log offset `eps`, plus a raw
untransformed slot) *and* predicted, as **Q-4**, that the raw slot would diverge from the logged one by
more than 0.10. It then wrote a void rule reading *"**any** `C(d) < 0` or `> 1`"*.

Both fired — on the raw and contiguous slots, exactly the ones Q-4 said would misbehave. The primary
slot was clean and strictly monotone. ⛔ **And the driver printed `VOID` and then printed its decision
verdict anyway**, resolving the contradiction silently in favour of reporting.

⇒ **The rule.** When a design includes a sweep over values chosen to probe degeneracy, **scope the void
conditions to the primary slot.** A swept diagnostic slot going degenerate is a *finding*; only the
primary slot going degenerate is a *void*. ⚠️ Related to §5.3 and to Appendix A's recurring theme — this
is a gate that **could** fire, fired correctly, and still meant nothing, which is a third failure mode
alongside "cannot fire" and "never run".

#### ⚠️ L5 — evaluation outputs can fail to leave the machine that produced them

On the 2026-10-25 control run, **three of five delivery directories did not exist** until an audit went
looking. The pod runner collects `$OUT` only, and both it and `tools/collapse` ship `lr_*` alone — so the
**entire gate side (`by_*`)**, the gate posterior (6.55 GB raw) and the `by_*` metricframes **reached
neither the parquets nor the draws archive**. They existed on the pod and nowhere else, and would have
died with it.

⇒ Most of this project's open evaluation questions are about the **gate**. A gate-calibration
measurement that discovers after terminate that `by_*` never came home is a wasted pod. ⛔ **Collection
of the artifact a measurement needs is part of the measurement's pre-registration, not an infrastructure
detail.** Reported by the `models` session's own independent `/falsify`.

#### ⚪ L6 — "unmeasured" must be representable distinctly from "measured zero"

The control run's `trajectory.csv` carries `fed_grad_max` **blank in all 300 rows**. That is correct and
deliberate. The column measures gradient through the autoregressive feedback wire, which is only ever fed
when `ss_backprop_through_feedback` is True — default False, set by no roster config. The column
**used** to initialise to `0.0`, which made *"the wire is off"* and *"the wire is on and the gradient
died"* **the same three characters** in a delivered CSV. Blank → `read_csv` NaN → "not measured", which
is the truth. Pinned by `tests/test_trajectory_log_end_to_end.py::test_fed_grad_max_is_BLANK_not_zero_when_the_feedback_wire_is_off`.

⇒ This is the same defect class the repo already records as *"an unmeasured input absorbed as a value —
fix by DELETING the representation, never by adding a guard"*, applied to an **output**. ⚠️ And a second
observation worth keeping: **two independent audits ran on 2026-10-25 and each flagged the other's
working guard as a defect** — this blank column, and a loud `aggregate_method is IGNORED` warning that
turns out to be a documented mode interaction with its own test. The usual failure here is guards that
cannot fire; this was the inverse, and it is cheaper to resolve but it costs attention. **A guard whose
visible signature is an absence will be read as a gap by anyone who did not write it**, so it needs to
say so at the site.

#### ⛔ L7 — a derived constant must carry its vehicle in the same sentence as its value

The structure arm's one derived constant, `struct_penalty_weight = 8.89624`, was committed into a
prereg under a ⭐, with the explicit rationale that deriving it **removes a free parameter rather than
multiplying the arm by a sweep**. It was wrong by **21×**, three independent ways at once: it was
measured on the **wrong family** (the `truncated_nb` vehicle, not the `nb` one the arm runs on), it
pinned **v1** of a script whose **v2 had already superseded it on disk**, and it was gradient-matched
against **one** of the **three** targets the loss sums over.

⇒ **The rule, and it is narrow on purpose.** A number *derived* from an artifact is a claim about that
artifact, so it must be written with its **checkpoint sha** and the **script version that produced it**
in the same sentence as its value. ⚠️ Note what did *not* protect it: the derivation was scripted,
committed, reproducible, cost-reported, carried an honest C-325 caveat about being measured at
initialisation, and **ran correctly every time**. Every error was in *what it was pointed at*, which no
amount of rigour inside the script can see. ⛔ **Deriving a constant removes the free parameter and
replaces it with a provenance obligation** — that is the trade, and only half of it was being paid.

⭐ **And the direction is the expensive one.** A 21×-too-large weight would have trained a $3 arm at a
term strength nothing had derived, and the resulting null would have read as evidence against *a
spatial penalty* rather than against *my arithmetic*. Compare `W-59`, where the error pointed at
**cancelling** a justified experiment. Both directions are live; this one burns money and then
mislabels the ashes.

#### ⚠️ L8 — two committed documents can contradict each other, and no gate in either one looks

The second of L7's three errors is the structural one. `2026-10-12_underfire_diagnosis_prereg.md`
recorded `8.89624` as **VOID** with its replacement, on **2026-10-21**. `2026-10-21_structure_penalty_arm_prereg.md`
§3b went on quoting `8.89624` as the pinned value. **Both were committed, both were in the repository,
and they disagreed for four days.**

⇒ Every pre-flight gate this project has asserts that **the config matches the prereg**. None asserts
that **the prereg matches the last derivation that ran**, and none asserts that **two preregs agree
about a number they both name**. ⚠️ This is a different shape from §5.9's "gates that exist and nobody
runs": the gates ran, passed, and were **scoped to one document**. ⛔ The cheap version of the fix is
not a new gate but a convention — **a derived constant gets exactly one home, and every other mention
is a link to it**, because a value duplicated across two documents will diverge and a grep is the only
thing that will ever notice.

---

## §6 — Where the unknown unknowns are

⚠️ **This section is INFERENCE throughout.** It reasons from where past surprises came from, to where
the next ones probably are. Nothing here is measured.

**The inference base:** of 31 traceable retractions, ⛔ **[CORRECTED 2026-10-20: the list below is NOT exhaustive and the
phrasing "the detectors were" reads as though it is. `Appendix B:1606-1612` gives NINE categories
totalling 23 hits; this reproduces five of them (15 hits) and silently omits **user escalation (3)**,
re-running on a converged vehicle (2), running the pre-registered decision table properly (2), and a
systematic recompute (1). It also drops Appendix B's two stated caveats — the categories are "**not**
mutually exclusive", and "about eight of the 31 record no single identifiable detector". ⛔ This
matters because the conclusion drawn below turns on whether a second computation existed, and the
largest omitted category — the user noticing — is not a second computation.]** the detectors were an
adversarial second pass (9),
two numbers disagreeing (3), a control arm (2), an impossible result (1), and the test suite (**0**). So
the unknowns are concentrated wherever **no second computation of the same quantity exists**.

| where | why I think it lurks | the cheapest probe |
|---|---|---|
| ⚠️ **The forecast path — DOWNGRADED 2026-09-30, it has now run** | `extrapolate_time` is forecast-only code and had produced **zero output ever** when this row was written. The `models` session reports a 300-lesson global-land forecasting run in flight on a rented pod — three models complete, `predictions_forecasting_*` and `forecasting_log.txt` present, 2,333,448 rows × 128 pooled draws, months 561–596, 64,818 cells — and a 40-lesson rehearsal that completed the whole chain and is being served. ⚠️ **Their measurement, not mine:** nothing on this machine holds forecast output but pytest fixtures, so I cannot verify it here. **The row is now "exercised once" rather than "never run", and its priority drops accordingly** — but one run is one run, and §5.0 is what that run actually exposed. | Score the surface it produced. That is §5.0, not this row. |
| ⛔ **The truth field** | §5.5. One outlier was found by eye, upstream, and filed. Nothing systematic has ever looked. A corrupt truth cell is invisible to every instrument we own, because every instrument scores *against* it. | An outlier scan at build time: any cell-month above the historical max by >10×. |
| ⛔ **Every negative result whose implementation nobody audited** | The September postmortem names this its *"most expensive open question"*: ≥15 defects found *after* the author judged the code correct, **0 by his own review**, and C-324 proves an implementation can be fully green and **completely inert**. The audited experiments are #308 and #311. Every earlier negative — and there are six failed interventions — rests on an unaudited implementation. | A potency check at a trained checkpoint on each retired arm. Cheap; no retraining. |
| ⚠️ **The in-package artifact, not the scorer** | ⛔ **[CORRECTED 2026-10-20: FIXED 2026-09-05; the roster was retrained 2026-09-07/08, after — so not "every shipped artifact".]** **C-328 instance 4 is already in production:** `random_flips` defaults True and reaches BN recalibration, so ~half of every BN running statistic in every shipped artifact was accumulated on flipped fields. That is a whole class — *"which non-training forwards traverse this path?"* — and it was asked once, late. | Enumerate every forward pass that is not a training step, and what augmentation each sees. |
| ⚠️ **The collapse from cube to delivered number** | ⛔ **CLAIM NARROWED 2026-10-19 (open finding 1).** The load-bearing figure here is **`tower_point` vs `mean` — 9.9× on AP**, the one like-for-like pair. ~~M70: `gate × mu` beats mean-of-16-draws by **58–159%**~~ — withdrawn as support, because `gate × mu` is non-zero on **100% of cells** and is the same object §5.0 disqualifies as `gate_only`. `median` is still an empty map. **The collapse step has the least scrutiny of any stage**, and on the like-for-like pair its effect is still larger than most interventions we tested. #388 asks whether D=4 is enough and *"no ledger entry answers it."* | Score all available collapses on one existing cube. Zero GPU. |
| ⚠️ **Anything measured once, by one instrument** | The three retractions caught by "two numbers disagreeing" only happened because a second computation existed. Where only one exists, the class is silent by construction: `recall_mass` (the production roster's own K1 criterion), `taillardat_index`, `hard_gate_threshold`. | For each single-source number that gates a decision, compute it a second way. |
| ⚠️ **The CI signal itself** | ⛔ **[2026-10-20: `environment.yml:31` installs `ruff==0.14.14`; the register says so in the same entry — this is history, not a present state.]** C-283: three tests shelled out to a binary the environment never installs, and CI was **red on every run for ≥2 weeks** — *"every CI `test` result before 2026-08-15 should be read as red for this reason unless shown otherwise."* A gate can be broken for reasons wholly unrelated to the code, and stay broken. | Assert the CI job's own preconditions (every external binary it invokes exists). |
| minor **Cross-model coupling in the pooled product** | The sample cube's randomness carries **no model identity** — `seed_t` is derived from `torch_seed` alone. Two roster models share `torch_seed` 42 and two share 43, and the 43 pair also share the composition rule. Pooled draws may therefore be less independent than the draw count implies. | Correlate the zero/nonzero mask patterns within each same-seed pair against a different-seed control pair. |
---

## §7 — What surprised us

Results that contradicted a stated prior. These are the most informative rows in the whole record and
they have never been listed together. Ordered by how much they should change how we measure.

| # | The prior | What happened | Why it matters to evaluation |
|---|---|---|---|
| S1 | The diagnosed defect was the problem, so removing it would help | **M45 — the double-counted zero was LOAD-BEARING.** The occurrence process was being applied twice (M44: the gate is hot, the composition halves it). Removing that defect **collapsed the rollout by 70%**. | ⛔ A correctly diagnosed defect can be a **brake**. "We found the bug" does not license "fixing it will help" — that is a separate, testable claim. |
| S2 | The dumb baseline beats us, so the model has no long-horizon value | **M34 overturns M1.** At L=300 the model beats a *fairly scored* persistence at **every** horizon (worst of 4 seeds 2.0–2.3×, gaps 2.1–4.3× MDE). Persistence itself moved ~1%. | ⚠️ **The ruler was stable; the model moved.** M1 was true of the 40-lesson vehicle it measured and nobody re-ran the race for two months. A baseline result has a *vehicle* scope, not just a metric scope. |
| S3 | A fix that restores the missing behaviour should improve the forecast | Scheduled sampling **largely fixed the zero collapse** — `act_ratio` at h18 **9.4×**, at h36 **28×** — and **lost AP at every horizon**. It answered more, in worse places, and lost. | ⛔ Corrects the earlier "the model just needs to answer" story (M15/M18). **Restoring a field statistic is not evidence of a better forecast.** |
| S4 | Sharper spatial structure is better | **M46 — 12 of 12 arms, six unrelated mechanisms:** sharpening the model's spatial view helps h1 and **hurts h36**. | ⚠️ h1 and h36 are different objects. An intervention validated at one horizon can be inverted at the other, and unanimity across mechanisms means it is the axis, not the implementation. |
| S5 | Our diagnostics describe the field, so they can judge it | **M53 / EXP-3 — occurrence, body magnitude and gate–body alignment are ALL preserved by a spatial roll that destroys the forecast.** Alignment reads **61.5** (vs the clamp's 69.3) while `AP@h18` collapses **0.362 → 0.0075**. | ⛔ **Three diagnostics, all blind to placement.** Any internal statistic — one computed from the forecast alone, without truth — cannot distinguish a good forecast from the same forecast in the wrong place. Judging placement needs a truth-referenced score. |
| S6 | The recurrent state is a memory of dynamics | **M54 — it is a MAP.** Roll it 90 cells and the forecast moves 90 cells **intact, r≈0.90**, while skill collapses **48×**. | ⚠️ A perfectly coherent, perfectly structured, perfectly self-consistent forecast can be entirely in the wrong place, and every structure metric will approve. |
| S7 | Ensembling buys diversity | **M66 — the 8-member pool beats its best member (AP@h18 0.2203 vs 0.1804), but subsampling the same pool back to 16 draws collapses it to 0.1462 — *below* the best member.** AP is monotone in S. **M67:** errors correlate **0.96–0.99 across all 28 pairs**. ⛔ **Corrected 2026-09-30 — this is not the gate.** The `library` session asked whether the correlation was on the gate or on the full output. Checked: `analyse_diversity.py:7,93` correlates `\|mean(draws) − truth\|` **in count space on the composed forecast**. So the correlation is on the composed forecast, not merely on the occurrence head. ⛔ **Corrected again 2026-10-10 — the previous wording of this entry was wrong.** It read *"diversity is dead on the delivered output"*, which conflates two quantities: M67 correlates **errors**, and on the **delivered field** the eight disagree substantially. Measured on the 2026-09-29 handover (`I_mean`, `pred_lr_sb_best`, 8 models × 13 origins, 30,334,824 rows each, row order asserted identical on the month×grid key): pairwise Pearson **r = 0.273–0.552, median 0.439** on the union support; of the **328,562** cell-months where any model fires, **33.0% is one model firing alone** and only **13.5%** has all eight; total predicted mass spans **3.64×** (bold_comet 1.681× the eight-model mean, bright_starship 0.462×). ⚠️ `sb` only, calibration partition, **no truth** — so it says nothing about which member is closer to what happened, or whether pooling beats choosing. What is dead is therefore **error** diversity, not output diversity: the members disagree about *where to put mass* and agree almost perfectly about *what they fail to catch*. ⚠️ **That reconciliation is INFERENCE** — error correlation cannot be recomputed on the handover, which carries no truth. It also weakens the Deep Ensembles (Lakshminarayanan2017) claim this entry used to make: random initialisation **did** supply output diversity here, so the independence condition is not falsified at the level of predictions; what fails is the weaker, more practical requirement that members' *errors* be decorrelated enough for averaging to help. | ⛔ The ensemble was buying **sample count, not member disagreement.** A pooled-vs-best comparison at unequal S measures S. Fix the draw count before claiming an ensemble effect. |
| S8 | A roster of different thresholds adds something | **M69 — mathematically it cannot.** `E_τ~U(0,1)[(gate ≥ τ)] = gate`, so averaging thresholds reconstructs the soft gate **from below** and never crosses it — within 0.5–0.7% at 64 τ. | minor — but a good example of a design question settled with algebra and zero GPU. Do that first. |
| S9 | The posterior mean is the natural point estimate | **M70 — `gate × mu` beats the mean of 16 draws by +58% to +159%** at identical total mass; `median` is an **empty map** (0.05–0.14% of cells non-zero). ⛔ **2026-10-19 (open finding 1): `gate × mu` is non-zero on 100% of cells**, so the surprise is real but the figure is a dense-surface ranking gain — the like-for-like collapse effect is `tower_point` vs `mean`, **9.9×**. | ⚠️ How you collapse the cube is a first-class modelling decision — **on the like-for-like pair** its effect is still larger than most interventions we tested. |
| S10 | Our failed approximation explains the failure | **M63 — the straight-through estimator is NOT backwards.** We tested our own excuse and it did not hold, so **the objective was wrong**, not the approximation. | ⚠️ Rare and valuable discipline: test the excuse. Without M63 the BPTT-SA null would have been attributed to an implementation detail and re-attempted. |
| S11 | A component that ablates to nothing is unimportant | **C-323 — you cannot ablate a regenerated component.** `hs = o ⊙ tanh(hl)` is recomputed from the cell every step, so a perturbation self-heals. **M60's "hidden 0/26" is architecturally guaranteed, not a finding.** | ⛔ A null from an ablation that the architecture makes impossible is not evidence. Check whether the manipulation can survive one forward pass before interpreting its null. |
| S12 | The mechanism story follows from the measurement | **"The recurrence diffuses the gate" was backwards on its own terms.** Smoothing *raises* spatial autocorrelation, so diffusion predicts Moran's I going **up** — it went **down** (M6). Falsified using data already on disk. | ⚠️ Four other mechanism stories were retracted the same week. A story that has not been asked to make a *directional* prediction has not been tested. |
| S13 | The rollout starts from a broken state | **M43 — it does not.** The rollout does not begin from an out-of-distribution state; the hypothesis is refuted and the one positive intervention (M38) still has **no mechanism**. | ⚠️ We are comfortable shipping an intervention that works without knowing why. That is defensible; pretending otherwise is not. |
| S14 | Fixing BatchNorm recalibration means giving it clean data | I switched dropout **off** during BN recal. **Inference runs MC-dropout ON** (ADR-059), so `running_var` came out **14–21% low at 14 of 15 BN layers**. Reverted before merge. | ⛔ The invariant was documented in **three places, all on the consumer side**, and none was read by a training-side edit. A "hygiene" fix to a measurement path changes the measurement. |
| S15 | Identical models have identical files | **M22 — two byte-identical models have different `.pt` shas.** torch stamps mtimes into the zip. | minor, but it invalidated an identity check we were relying on. Hash the weight tensors. |
| S16 | The test suite protects the measurement code | **2,079 green tests, 25 runtime-verified findings, 0 caught.** Then 5 of my own new guards survived their first mutation; a later audit found **17 of 33** mutations survived. | ⛔ Coverage is not sensitivity. The only reliable check is to **delete the fix and watch the test stay green**. |
| S17 | Packaging is a plausible source of drift | **M76 — the PyPI wheel reproduces the checkout byte-identically**, 190 tensors × 4 models, BN buffers included. A *positive* surprise. | minor — worth recording because it removes a whole class of suspicion, cheaply. |

---

## §8 — What our own experience demands of a promotion gate

**Rule for this section: no requirement without a scar.** Every row below is something that actually
bit us here, with the cost. Nothing is included because it is good practice elsewhere.

⚠️ **And read this as a menu with costs, not as a gate.** §8.1 argues against frameworks and this is a
fifteen-row table, which is a fair thing to point at. The resolution: **nothing here is proposed as a
mandatory checklist.** **Rows ★1, ★3, ★4 and ★6 are the four** that would have prevented the most
expensive failures on record and are cheap; the other eleven are priced so that whoever owns the protocol
can choose. A list of fifteen enforced gates would be the 303 → 1,167-line validator ⛔ **[CORRECTED 2026-10-20:
NOT in this repo's register. `grep` finds the figure nowhere in `reports/technical_risk_register.md`,
`docs/` or `CLAUDE.md` — it appears only in this report. Its actual home is the author's global
`~/.claude/rules/design-preferences.md`, which attributes it to "one repo here" and not to this
register. ⛔ This matters because §8's rule is that the scar is in THIS repository's record, and this
one is not.]** this repo already
warns about.

⚠️ The `views-evaluation` session asked whether the table would be stronger as **those four, with the
eleven in an appendix.** It would be, for a reader choosing what to adopt; it would be weaker as a record
of what bit us, which is what the scar rule is for. **Compromise: the four carry a ★. The eleven stay in
place and keep their scars.** If this ever becomes a proposal rather than a record, cut it to the four.

Whoever owns the promotion protocol can compare this against it. I make no claim about what the
protocol currently says.

⚠️ **§8.3 is a PROPOSAL and deliberately breaks this section's scar rule** — a diagnostic suite matched
to the factorization of the data-generating process, from Simon's notes of 2026-10-19, kept separate from
the fifteen rows so it cannot be mistaken for the record. It is the only forward-looking part of §8.

| # | Requirement | The scar that earns it | Cost to adopt |
|---|---|---|---|
| ⛔ ★1 | **Ask, before the run: "what would this measurement look like if the effect were absent?"** Record the answer. | Cluster 18 — **23 Tier-1/2/3 entries (⛔ **24**, not 23; corrected 2026-10-20)**, each of which cost GPU time or produced a wrong ledger row, and which the register read as 23 unrelated incidents until 2026-09-10. Three of its four shapes fail this one question. | One line in a pre-registration. |
| ⛔ 2 | **A reference forecast on every scored run — persistence *and* climatology — with the zero-share decomposition and the ARTIFACT verdict applied automatically.** No result may be reported against `none`. | §0.1: a forecast with `size_ratio` 0.0000 cleared the 5% bar on the primary metric, and 74.6% of the win was confident zeros. ⚠️ It would **not** have been promoted — the MCR guardrail vetoes it (corrected 2026-10-11) — but nothing computed the decomposition, and the ARTIFACT verdict stood unexamined **16 days**. This is the ledger's own recommendation #3 and it is **not implemented**. | Structural, cheap; the instruments exist (`rollout_ruler_core`, `climatology_resample`). |
| ⛔ ★3 | **A floor gate on the CONTROL arm before any treatment arm runs: FG-A `AP_ctrl ≥ 5 × prevalence`, and FG-C `(1−θ)·AP_ctrl ≥ 3 × MDE`.** | A 6-GPU-hour pre-registered sweep with falsifiers was **VOID as run**; 9 ledger rows and 5 inferences re-scoped, 3 claims retired. One manipulation read **+0.9%** on the floored vehicle and **−93.7%** on a real one. The condition was visible in a CSV **5 h 47 m before the last arm finished.** | Zero GPU beyond the control. ⛔ **And fix the existing gate:** ⛔ **[CORRECTED 2026-10-20: "both callers omit `mde_ap`, so FG-C
never evaluates" is FALSE. There are THREE callers and one passes it —
`reports/2026-09-04_input_noise_dossier/tools/check_floor.py:56`, `mde_ap=MDE_AP` where `MDE_AP = 0.06`
(verified) — and that file's docstring states FG-C's disposition: "REPORTED, not binding … computed,
printed, and carried into the write-up". Omitting it is also not a silent pass:
`scripts/floor_gate.py:181-205` returns `PROVISIONAL` and flags `missing_binding`. The true form: two
of three callers (`lesson_curve_gate.py:182`, `ss_sweep_gate.py:172`) omit it.]** |
| ⛔ ★4 | **Match S between arms and reference, enforced in the scorer, and refuse an AP comparison across gated/ungated arms.** | Persistence was published **understated by 31%** at h18 by two independent scoring defects. #282's share of it **scales as 1/n_origins**, so a single-origin study scores persistence as exactly zero, silently. | The check exists (`reference_sample_width`, `assert_sample_cube`). ⛔ **CORRECTED 2026-10-20: three tools DO call it** — `2026-08-15_rollout_ruler_trust_dossier/tools/rescore_v2.py:44,65` (**the tool that wrote the `rescore.csv` this report's headline numbers come from**), `tools/mde.py:118,143`, `tools/tail_index.py:34,84`; `assert_sample_cube` is called by `tools/partition_audit.py:55,159`. Register: `technical_risk_register.md:2265`. The defensible claim is that the two HEADLINE scorers do not call it. |
| ⛔ 5 | **Every metric that gates a decision must have a degenerate-forecast test that fires — run against the HEADLINE scorer, not the metric function in isolation.** Include the spatial roll and the single-huge-cell. ⛔ **[CORRECTED 2026-10-20: this row's "the seven
degenerates that ARE implemented all test `activation_metrics` in isolation" carries a count §5.4
STRUCK on 2026-10-19 — three of the seven named cases do not exist in either file, and the count
double-counted a known-good reference. And "all … in isolation" is false:
`tests/test_rollout_ruler_trust.py:211` and `tests/test_score_v2_horizons.py:91-130` both feed a cube
through the headline scorer. Open finding 2 was fixed in §5.4 only; this row kept the retracted
version.]** | The roll: three field statistics unchanged, **AP 48× worse**, and **no test guards it**. The seven degenerates that *are* implemented all test `activation_metrics` in isolation; nothing feeds a degenerate cube through the scorer that produces the published row. | Days, not weeks. Synthetic fields, no GPU. |
| ⛔ ★6 | **Prove the intervention is potent, at a trained checkpoint, before spending GPU on the arm.** | **C-324 (Tier 1):** 7 unit tests, 5/5 mutations caught, lint clean, 1,901-test suite green — and **inert**. Two arms trained **276 minutes to byte-identical weights**. Caught only because the result was impossible; a half-connected knob would have produced a plausible number and the pre-registered rule would have read it as "hypothesis dead". | `scripts/potency_check.py` exists. Two call sites in the whole repo. |
| ⛔ 7 | **Judge a guard by deleting the thing it guards, not by coverage.** | ⚠️ **Four separate studies, populations named, because quoting them together is the C-298 defect this report is about.** (i) Epic #353's release review: **2,079 green tests, 25 runtime-verified findings, 0 caught.** (ii) C-329's round-3 deletion test **on one feature**: **27 of 40 individual tests** passed with the call site deleted, and **1 of 4 test files** passed with the feature entirely removed. (iii) A `/ship-it` guard audit on **the fix for C-329**: **17 of 33 mutations survived.** (iv) Epic #353's **own new guards**, written by me: **5 survived their first mutation.** | Minutes per guard. |
| ⛔ 8 | **An adversarial pass by a non-author, in a clean context, denied the commit message and design docs.** | **The highest-yield detector in the record: 9 of 31 retractions.** Against *"defects found by my own review: 0."* One round found errors introduced by the previous round's fix. | Hours. It is the cheapest thing on this list relative to what it finds. |
| ⛔ 8b | **Every metric a plan prescribes must declare its status — gate, report-only, or diagnostic — and a prescribed gate must be wired to a verdict, not to a `print`.** | §4.5: **two** prescribed gates were built and never wired. One is the calibration diagnostic whose absence let an **all-zero surface ship**; the other is **`recall_mass`, the production roster's own K1 kill criterion**, whose name appears in no score CSV, no glossary row and no other code, and which is applied by a **`print`** against a hardcoded roster. `waste_frac`, in the same file, declares *"reported alongside, not as criteria"* — so the practice exists and was applied unevenly. | One line per metric. The `waste_frac` docstring is the template. |
| ⚠️ 9 | **A negative claim requires a search *plus* a positive control.** A uniform-zero result across a heterogeneous population is instrument failure until proven otherwise. | **C-309, twice** — once asserting an untested axis that was running in every arm of every experiment, and once in the *fix* for that, where "none of 26 papers has a claim" was a wrong glob path. | One extra query. |
| ⚠️ 10 | **A cheap screen's NO must be recorded with its false-negative mode and a reopen trigger, or it does not close anything.** | C-307, and the chair's own words: *"we have dropped multiple things on the floor … then me later insisting that we try for real, and then it turns out that it in fact works. This happens so much."* | Two lines per closure. |
| ⚠️ 11 | **Recompute every headline figure from the committed data before it leaves the dossier.** Trace a suspicious number to its arm; do not "correct" it on suspicion. | **3 of 9 headline figures wrong**, all failing the same way (cells of a 4-D grid quoted as one comparison); **two had already propagated into the register**. And the pass produced a false positive of its own. | One pass per dossier. |
| ⚠️ 12 | **Enumerate every forward pass that is not a training step, and what augmentation each one sees.** | **C-328 instance 4 is in every shipped artifact:** `random_flips` defaults True and reaches BN recalibration, so ~half of every BatchNorm running statistic in every production model was accumulated on H/W-flipped fields. *"Every log, gate and score would have looked clean."* | A checklist line. |
| ⚠️ 13 | **Audit the reference, not only the model.** An outlier scan on the truth field at build time. | The register on itself: *"extensive coverage of whether we measure the model correctly, and **none** of whether the thing we measure against is correct."* One truth cell reads 27,413 → 2 → 32,505 where the global maximum elsewhere is 1,100. | A single query at build time. |
| ⚠️ 14 | **Pairing by default**, and record `torch_seed`, `S` and the vehicle alongside every score. | MDE **0.0541 → 0.0086** from pairing alone on identical data; the noise the programme designed against was **~5× too large**. And because the sample cube's randomness carries **no model identity**, a paired reading and an independent one are indistinguishable in the CSV. | Free. |
| minor 15 | **Hash weight tensors, never `.pt` files.** | Two byte-identical models have different `.pt` shas — torch stamps mtimes. The invalid check *"would have shown 'still non-deterministic' even after a correct fix."* | Already implemented in `compare_run_determinism`. |

### 8.1 Three things this list deliberately does not ask for

- **A new framework.** The September postmortem's own conclusion: the failures were *"of attention at a
  specific moment, not of missing machinery"*, and its fix was **four checklist lines**, explicitly not
  a framework. The register separately warns that a guard-heavy config validator grew 303 → 1,167 lines.
- **More metrics.** The apparatus already computes `recall_mass`, `gate_auc`, `tail_miss`, `fss_1`,
  `moran_i`, `conc1pct`, `ratio_p10/p90` and more that **nothing reads**. Wiring an existing metric to a
  decision beats adding one.
- **Repo-wide gates.** ⛔ **[CORRECTED 2026-10-20: this quote is NOT the register's. It is from
  `reports/postmortem_implementation_practice_2026-09.md:171`.]** The structural finding is that
  *"no gate in this repo is repo-wide;
  each is opted into per launcher, and a dossier that forgets one gets no warning."* That is a real
  problem — but the demonstrated failure mode of centralising is C-283, where one broken shared check
  turned CI red for two weeks and nobody noticed. ⛔ **[CORRECTED 2026-10-20: BOTH HALVES WRONG, and
the scar does not entail the requirement. `technical_risk_register.md:2324-2340`: red "since at least
2026-08-14", **found and fixed 2026-08-15** by a PR #273 CI investigation — one day, and it WAS
noticed. ⛔ And C-283 is three tests shelling out to a `ruff` binary `environment.yml` never installed:
nothing centralised broke, because no shared check existed to break. A requirement not to centralise,
resting on a scar that demonstrates something else.]**

### 8.2 The two measurements, and which is which

⚠️ **The ordering, made explicit because a reader may be able to fund exactly one.** The **larger
finding** is §5.0 — the delivered product is unscored and the delivered collapse measured worst of
seven. The **cheapest decisive** measurement is the h=1 question below. They are not the same thing and
the report previously implied the second was the more important.

⚠️ **PARTLY SPENT ALREADY — corrected 2026-10-19 (open finding 5):** the 128-draw repeat below **was run on
2026-10-01**, 8/8 models, pooled (§5.0). What is left to fund is the aggregate reading, not the repeat.

**Fund §5.0 first if you can fund one:** ~~the 128-draw repeat of the collapse comparison, and~~ an aggregate
coverage number. Both are zero-GPU on an existing cube, and together they decide whether the partner is
being served the worst available collapse of a distribution that is also mis-calibrated. That is a
delivery finding, and delivery beats research here.

**The h=1 question is the cheapest thing that could change the research story:**

⛔ **Establish whether the h=1 win is real.** It is the only non-ARTIFACT signal in the entire
programme. ⚠️ **Scope corrected 2026-10-19 (open finding 9): this is the PERSISTENCE comparison.** Against
climatology `rescore.csv` already carries a CI that excludes zero and a **REAL** verdict at h=1 (see §4.3
item 1, with its two cautions). The persistence comparison reads **UNDECIDABLE for a purely procedural reason** — no bootstrap CI was
computed — while both point estimates say REAL (CRPSS +0.197, ΔAP +0.152). It needs a re-run emitting
**per-origin** scores at matched S, because the cubes were score-then-deleted.

It is cheap and it is decisive in both directions: it either establishes this project's first genuine
win over a naive baseline, or it removes the last one.

---

### 8.3 A diagnostic suite matched to the factorization — ⛔ PROPOSAL, not a scar

⚠️ **Status: this subsection breaks §8's own rule.** Every row above is earned by a scar. This one is a
**proposal**, from Simon's notes of 2026-10-19, and it is marked as such so it cannot be mistaken for the
record. It is here because §8 is where a reader asks *"so what should the gate measure?"* and the table
above answers only *"what bit us."* ⛔ **Nothing below has been validated on synthetic data with known
properties, and the proposal says so itself.**

#### Why this is not the "more metrics" §8.1 refuses

§8.1 declines new metrics on the ground that *"the apparatus already computes `recall_mass`, `gate_auc`,
`tail_miss` … and more that **nothing reads**. Wiring an existing metric to a decision beats adding one."*
That objection is **upheld, and it is the strongest argument for this subsection**: checked against the
apparatus, **two of the three proposed metrics already exist and nothing gates on either.**

| proposed | status in this repo | location |
|---|---|---|
| **M1 — activation calibration ratio** `R_A = Σ1(Ŷ>0) / Σ1(Y>0)` | ⚠️ **implemented as the DRAW-AVERAGED form**, not identically — see the naming note below. *(An earlier version of this cell said "identical definition", contradicting that note 130 lines later; corrected 2026-10-20.)* | `2026-07-29_v2_scoreboard_dossier/tools/activation_metrics.py:28-40`, as **`act_ratio`** |
| **M2 — total magnitude ratio** `R_T = Σ E[Y] / Σ Y` | ✅ **in use as a diagnostic**, as **`mass_ratio`** | §5.0's collapse table, this report |
| **M3 — conditional magnitude distance** `W₁(F₊, F̂₊)` | ⚠️ **`EMD` ships in views-evaluation but POINT-ONLY**; `pEMDiv` and Sinkhorn are listed *not implemented* | `reports/guides/eval_lib_implementation_details.md:184` |

So the proposal is **two-thirds wiring and one-third genuinely new** — and `act_ratio`'s own docstring
already states the case for it: *"`crps_all`/MCR go BLIND when the field is ~99% zero … these metrics
look at the thing that actually broke."* Written June 2026, independently of these notes.

⛔ **And the gate-axis conclusion (`2026-10-18_gate_underdispersion_prereg/CONCLUSION.md` §9) found the
hole this fills by name:** the eligibility conjunction has **no occurrence guardrail**, and halving the
firing rate buys **37–83% CRPS**, so the artefact satisfies the shipping criteria outright. **`R_A` is
the missing guardrail.** The same conclusion records that **`act_ratio` has no pre-registered numeric
band**; the proposal supplies one (`R_A = 1`, with a direction table). Two independent threads arriving
at the same missing instrument is the argument for wiring it.

#### ⛔ The forecaster's dilemma constrains SELECTION, not diagnosis

The standing objection to any positive-subset metric is FAO-02's: twCRPS and
positive-subset/extremes-conditioning *"were considered and REJECTED there (the forecaster's dilemma,
Lerch2017)"* — `technical_risk_register.md:655`.

⛔ **An earlier reading of that in this programme was too strong, and is corrected here.** Lerch's result
is that conditioning on the **realized outcome** destroys the **propriety of a rule used to rank
forecasters**: a forecaster who always cries wolf wins when only the wolf days are scored. It is a
theorem about **selection**. It is not a prohibition on conditional **diagnostics**, and this repo has
never behaved as though it were — **`crps-events` ("CRPS over cells that truly had conflict") and
`size-ratio` (conditioned on conflict cells) are both in the locked glossary and both in use.**

The distinction that makes this coherent, and the frame proposed for the article:

> **One proper rule, over the full outcome space, is reserved for ranking. The diagnostics are matched to
> the factorization of the data-generating process, and are never used to select.**

For a hurdle model the likelihood **factorizes** into an occurrence term and a conditional-magnitude term
— **Cragg1971** (held: `9_library/papers/Cragg1971_LimitedDependentVariables.pdf`), with **Mullahy1986**
for the binomial-plus-truncated-count form this repo's arm actually implements. Evaluating each factor
against the matching data is then not arbitrary subsetting; it is evaluating the model that was fitted.

⚠️ **The caveat, already on record and load-bearing:** Cragg's separability assumes **independently
parameterised** parts, and **our gate and body share a trunk.** The likelihood still factorizes, so the
*diagnostics* survive — but no claim that the two components can be independently **improved** survives,
and §7's allocation findings are the evidence.

> ⛔ **AND A SECOND BREAK, ON AN AXIS THIS SUBSECTION DOES NOT DEFEND — see `Appendix E.2`.** The
> `opponent` session attacks the factorization not on **shared parameters** but on **conditional
> independence across units.** Cragg splits *one observation's* likelihood; the panel likelihood is then a
> product over cell-months, and that product requires occurrence in one cell-month to be conditionally
> independent of its neighbours' **realised** history. Under a self-exciting DGP (`g ≠ 0` in Ogata's
> intensity) it is not — occurrence is driven by `∫g(t−s)dN_s`, a function of realised events — so **the
> likelihood does not factor over cells at all, and the "occurrence term" §8.3 proposes to diagnose is not
> the fitted model's occurrence term under that DGP.**
>
> ⚠️ **This is the `opponent` session's derivation, explicitly flagged by it as inference and as the
> load-bearing one: neither Hawkes nor Ogata mentions Cragg, hurdles or zero inflation.** It is recorded
> here to be **checked, not cited.** ⛔ **Nothing in §8.3 is withdrawn on the strength of it** — but if it
> holds, the factorization justification needs a conditional-independence assumption stated explicitly,
> and we do not currently state one.
>
> ✅ **What the same pass could NOT break:** the algebra. `mean = (n_positive/S) × E[y|y>0]` is an
> identity, three measurements carry two degrees of freedom, and **none of the ten papers touches it.** The
> consistency-check framing stands.

#### 🔴 The suite is a FACTORIZATION, not an orthogonal basis — and this report's own glossary proves it

The proposal hopes for *"largely orthogonal"* diagnostics and concedes they are *"complementary, but not
fully orthogonal."* ⛔ **They cannot be orthogonal.** `GLOSSARY.md`'s *positive-draw fraction* row states
the identity outright:

```
mean = (n_positive / S) × E[y | y>0]
```

so `R_T ≈ R_A × (conditional magnitude ratio)` — **three measurements, two degrees of freedom.** One line
of algebra refutes an orthogonality claim, and an article resting on one would be refuted in review.

⛔ **RETRACTED 2026-10-20 — the inference I drew from that identity was WRONG.** I wrote that the
factorization makes the third quantity *determined* by the other two, turning the suite into a
**consistency check** a compensating model could not pass, and called it "provable". It is not.

**The identity pins the conditional MEAN, not the conditional DISTRIBUTION.** `M3` is a Wasserstein
distance between conditional *distributions*, which is free with both ratios at 1. Counterexample,
verified numerically:

> Truth: 100 cells at 10. Forecast: 50 cells at 5, 50 at 15.
> **`R_A` = 1 · `R_T` = 1 · `W₁` = 5.**

Both ratios perfect, the magnitude distributions completely different. ⛔ **So there are three degrees of
freedom, not two, and a model CAN satisfy both ratios while compensating one magnitude error with
another** — which is exactly what the counterexample does.

⚠️ **And this subsection already contained its own refutation.** The validation list below demands a
synthetic case of **"correct total with too many small and too few extreme events"** and **"correct mean
with wrong tail"** — those *are* the third degree of freedom this argument denied.

**What survives.** The three metrics are **not orthogonal** — `R_T ≈ R_A ×` (conditional magnitude
ratio) is real, and two of the three do constrain each other. The honest claim is therefore **weaker than
orthogonality AND weaker than a consistency check**: a partial factorization with one free dimension, and
`M3` is the free one. That is still a reason to report all three; it is not a proof of anything.

#### ⛔ The latent-state extension has a worse trap than the dilemma

The proposal's natural extension — a state-based DGP (no conflict / pre-conflict / low / mid / high /
extreme / post-conflict), with magnitude and self-excitation differing by state — is substantively right
and has a hard evaluation problem: **a latent state cannot be conditioned on, because it is not observed.**
Condition on a state **inferred by the model under evaluation** and the model defines the partition it is
then scored on. That is circular, and it is worse than the dilemma, because nothing in the score reveals it.

⚠️ **CORRECTED 2026-10-20. The repo has a NEAR-MISS here, not an instance, and my earlier wording
overstated it twice.** I wrote that *"C-345's **recorded** scope limit is that 'it measured model-defined
cell sets, never truth-defined ones'"* — ⛔ **that string exists nowhere but this report; it is my
paraphrase inside quotation marks.** C-345's actual recorded scope
(`reports/technical_risk_register.md`) is *"**No truth involved** — this is a statement about the model's
own posterior, not about accuracy."*

⛔ **And C-345 is not an instance of the circularity.** It scored nothing against truth, so there is no
partition-defines-its-own-score loop in it — only a descriptive stratification by the model's own output.
The latent-state trap named above is real; **C-345 is not an example of it.** What C-345 *does* show is
the weaker and still useful point that a model-defined cell set answers a different question than a
truth-defined one — which is why the conflictology probe
(`reports/2026-10-11_conflictology_empty_mode_prereg.md`, on `development`) uses a **truth-defined** set
and a control with no fitted parameters.

**So the rule the extension needs:**
- ✅ **Condition on observed history** — e.g. *"did this cell have conflict in the prior 12 months"*. That
  is covariate conditioning, pre-registrable, and proper. It is exactly the partition
  `2026-06-23_body_sweep_dossier/15_decay_failure_confirmed.md:8-12` used to localise the decay failure.
- ⛔ **Never condition on a model-inferred state.**

#### What the suite cannot see, stated rather than discovered later

- ⛔ **Dependence, and therefore self-excitation.** All three metrics are marginal or pooled by
  construction and are blind to spatial and temporal dependence. The proposal raises self-excitation
  differing by state; **no metric in it can detect self-excitation at all.** That is a fourth axis, and
  `reports/2026-09-30_dependence_ratio_prereg.md` is where it already lives in this repo.
- ⛔ **Point-forecast models.** The proposal's scope is *"both point estimate models and sample estimate
  models"*, but `R_A` is **undefined for a point model** without a threshold — is a point forecast of 0.3
  fatalities an activation? — and the threshold-free variant `R_A^(p) = Σpᵢ / Σ1(Y>0)` requires an
  occurrence probability a pure point model does not emit. **Specify that seam or the suite silently covers
  half its stated scope.**
- ⚠️ **M3's conditional populations do not match**, and the proposal says so itself: the predicted positives
  are conditional on **the model firing**, the observed positives on **reality firing**. That is the
  version the dilemma objection does land on. ✅ **The repo already has the version that survives it:
  `crps-events`, conditioned on cells where TRUTH was positive.** So M3 should be scoped as a **shape
  diagnostic on the marginal severity distribution**, explicitly not conditional calibration, and
  `crps-events` should carry the conditional claim.

#### 🔴 A naming requirement, because this repo has already paid for it

The proposal treats "total magnitude ratio" as one quantity. **This repo has three, and the glossary warns
about confusing them:**

| name | definition | note |
|---|---|---|
| **`size-ratio`** | **median** of (guess ÷ truth) on conflict cells | the glossary's locked term |
| **`pos-mcr`** | **mean** of (guess ÷ truth) on conflict cells | *"a few big cells can flatter it (why we once distrusted it)"* |
| **`mass_ratio`** | ratio of **sums**: Σ guess ÷ Σ truth | ⛔ **this is the proposal's `R_T`** |

⛔ **`R_T` must enter under `mass_ratio`, not under a generic name.** Three claims have been withdrawn in
this programme for the mean-versus-median class of error, and
`feedback_never_mix_two_statistics` exists because a median-over-cells was compared to a pooled mean.
Likewise the proposal's own warning — that `R_A` on thresholded draws and `R_A^(p)` on probabilities
*"should not be conflated"* — must survive into the glossary, because `act_ratio` as implemented is
computed over **draws of the composed cube**, which is neither of the proposal's two definitions exactly.

#### What would have to happen before any of this gates a decision

The proposal's own list, adopted unchanged because it is the right one: synthetic data with known
properties, exercising **correct activation with underestimated magnitudes**, **correct total with too
many small and too few extreme events**, **correct pooled magnitude distribution with magnitudes assigned
to the wrong cases**, **correct mean with wrong tail**, and **correct marginals with wrong aggregates**.
⛔ That is **row 5 of the table above** — *"every metric that gates a decision must have a
degenerate-forecast test that fires"* — applied to this suite before it is trusted, and the degenerate
harness it names already exists.

⚠️ **Unaudited, and it matters here most.** The proposal cites Czado/Gneiting/Held on count-forecast
calibration and recent tail-calibration work. ⛔ **No literature citation in this report has been audited
by anyone**, and that caveat applies to these two as much as to §1's fourth amendment. **Lerch2017** and
**Machete2012** are held and were read for the dilemma argument above; the count-calibration and
tail-calibration citations are **not** verified. Treat the framing as this repo's reasoning, with the
literature as a claim to check.

---

## Appendix A — The instrument inventory

**Where the ruler lives.** ⚠️ **No forecast metric is computed anywhere in `views_hydranet/` ⛔ **CORRECTED 2026-10-20: FALSE.** `views_hydranet/utils/training_forensics.py:344,359,364` computes `average_precision_score` and `roc_auc_score`, with **AP the default** (`:32`), reachable and imported by `train/training_engine.py:38`. ⛔ **The metric this universal negative misses is this report's own primary metric.**.** The
entire apparatus is `scripts/` (26 files, 5,460 lines) plus `reports/*/tools/` (⛔ **164 files, 106 `.py` — 109 matches no counting rule; corrected 2026-10-20**), reading
`origin_*/<target>/y_pred.npy` + `identifiers.npz` — arrays this repo does not write. The only
evaluation artifact written in-package is the body-mean dump.

### A.1 Still load-bearing

| instrument | measures | scope | note |
|---|---|---|---|
| `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py` | AP, Brier, crps all/events/none, `size_ratio` (median), `pos_mcr` (mean) | **T=0 only**; all-cell; support intersected across models | **FROZEN.** `--selftest` is the freeze gate. Every later ruler imports it. ⚠️ Its one soft check (support coverage, tol 0.20) **cannot fail** — warns, continues, no flag in the CSV |
| `.../2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py` | the 21-column headline row | h∈{1,18,36}; all-cell | the current headline scorer; h=1 pinned byte-equal to lodestar |
| `.../2026-07-25_t0_rollout_skill_dossier/tools/rollout_skill_score.py` | the lodestar row indexed by horizon, h=1..36 | all-cell | hardcoded `/home/simon` path; `block_bootstrap_crps` single-arm support = **C-277**, `xfail(strict=True)` |
| `scripts/rollout_ruler_core.py` | `zero_share`, `crps_skill_score`, `climatology_resample`, `verdict_token`, `assert_sample_cube`, `reference_sample_width` | horizon-agnostic pure fns | **the current ruler contract**; 53 tests, never skips. `verdict_token` is the strongest guard in the repo. ⛔ `assert_sample_cube` / `reference_sample_width` are **called by no scorer** |
| `.../2026-07-29_v2_scoreboard_dossier/tools/activation_metrics.py` | `act_pred/true/ratio`, `precision_at_k`, `mag_on_false_pos/true_pos` | all cell×draws; threshold-free | **the only red-teamed metric module** |
| `scripts/ap_block_bootstrap.py` | origin-block CI + MDE on AP; AP ratio; AP diff | per horizon; 13 origins | the only instrument that **refuses** a gated-vs-ungated AP comparison |
| `.../2026-08-22_state_freeze_l300_dossier/tools/paired_ci.py` | paired origin-block CI, AP(cell) − AP(none) | one horizon, two arms | the 6.3× MDE result |
| `.../2026-08-21_persistence_reference_dossier/tools/fair_persistence.py` | persistence AP ranked by value vs by indicator | 7 horizons | repairs the S=1 AP quantisation defect — **not fixed at source** |
| `scripts/floor_gate.py` | FG-A, FG-B (advisory), FG-C | one readout horizon; control arm | 20 `FLOORGATE_*` tokens on disk incl. 2 `_FAIL`. ⛔ FG-C never evaluates on either gate that calls it |
| `scripts/potency_check.py` | does the knob move a number; does a positive control fire | pre-GPU, the arm's own config | **two call sites in the repo**, both post-dating the 276-wasted-minute incident |
| `scripts/audit_run.py` | 13 per-step metrics incl. `spearman_pos`, `range_compression`, `gate_ap` | **every step 0–35** | `screen_verdict` reads 7 of 13; four are computed 36× and read by nobody |
| `scripts/field_sharpness.py` | `fss_1`, `fss_11`, `fss_ratio`, `conc1pct`, `moran_i` | one field vs truth, threshold-matched | ⚠️ **contradicts itself**: module docstring says `moran_i` is primary and `fss_ratio` "must not be read as sharpness"; the function docstring 30 lines later calls `fss_ratio` "the primary readout" — a live C-303 instance |
| `scripts/lesson_curve_gate.py`, `scripts/ss_sweep_gate.py`, `scripts/input_noise_gate.py` | verdict tokens with falsifiers | per programme | all three **can fail** and have; `ss_sweep_gate`'s anchor guard was in the pre-registration and **missing from the code until 2026-08-21** |
| `scripts/arm_identity_check.py`, `scripts/arm_postflight.py` | config↔request identity; artifact completeness and support match | per arm | `arm_postflight`'s support-match check — its whole point — is **skipped when `reference` is omitted** |
| `.../2026-09-03_drivers_and_dynamics_dossier/tools/subset_ap.py` | AP on an onset universe vs a continuation universe | partition, not subset | the **only** instrument that separates onset from continuation |
| `.../2026-09-02_silence_vs_fade_dossier/tools/decompose.py` | the exact identity `mean(g·μ) = mean(g) × [Σgμ/Σg]` | all cells, nothing conditioned | the fix for C-318 (a conditional mean on a vanishing denominator, twice) |
| `reports/2026-10-10_delivery_frame_measurements/delivery_frame_audit.py` | integrity, frame ordering (`II,III ≥ I`), collapse mass-ratios, per-step mass **and** step-to-step L1 churn, member distinctness on the union support | a **delivered handover folder**; no truth, so no AP/CRPS/calibration | ⚠️ **NEW 2026-10-10, ungated lab code, no tests.** The only instrument that reads the *delivered* product rather than a sample cube — a partial answer to §5.0(a). Its (e) section exists because the first version of that measurement was wrong: whole-field agreement on a 99%-zero field measures shared zeros (Appendix C) |
| `reports/2026-10-25_sharpness_ceiling/ceiling.py` | **split-half variogram reliability** of the TRUTH: `gamma_A`, `gamma_B` and `gamma_cross` in one pass, over 2 split schemes × 3 transforms × 3 targets × {E–W, isotropic} | the 64,818-cell land grid, months 121–456; **no model artifact is read** | ⭐ **NEW 2026-10-25.** The **first instrument in the apparatus pointed at the target rather than the model** (§5.5). Free: 78.0 s, 4.76 GiB, $0, no GPU. ⛔ Reports an **amplitude** fraction (0.498 at `d=1`); the **shape** fraction that actually bounds B-1 (~1.0) needs the variance normalisation added — see §5.5 L2, and do not quote one for the other. ⛔ Its `c_full()` half→full correction is **WITHDRAWN** (§5.5 L3). Committed before it ran; no tests

### A.2 Superseded or dead

> ⛔ **ONE MISFILING, CORRECTED 2026-10-20: `gate_reliability` is not superseded or dead.** §5.6 of
> this report holds it up as the one instrument that **fixed** the `hash()`-salt reproducibility defect
> — "a stable per-target seed instead of `hash()`" — and as the template its two siblings should copy,
> calling the back-port "a five-minute change". It has also run
> (`reports/2026-06-20_gate_calibration_dossier/07_experiment_log.md:5`). ⚠️ And `RESULTS.md §12` on
> the gate axis records that it was **reinvented** because a "superseded" marker hid it — which is this
> row doing it again. Standing rule: **a "superseded" marker retires a CONCLUSION, never a tool, an
> instrument or a falsifier.**

`compare_run_determinism` (**one recorded run, 2026-06-15; no test file** — and it is the determinism
gate) · `crps_significance` (step-1 only; its iid-over-cells bootstrap is the construction C-221 later
banned) · `gate_reliability` (pre-registered and "never run" until it was) · `diagnose_io_gain`
(bloom-era, synthetic inputs by its own "HONESTY" docstring) · `hurdle_decompose` (still imported by
three probes; its only range guard is a bare `assert`, stripped under `-O`) · `mcr_readout` (superseded
as a readout, **load-bearing as the join library** for six scripts) · `screen_verdict` (thresholds
frozen to a 300-lesson floor; nothing has invoked it since its dossier) · `sharpness_scorecard`
(superseded as a ruler, load-bearing as a library) · `decay_attribution_probe`,
`decay_discrimination_probe`, `gate_resolution_probe`, `single_window_overfit`,
`probe_inverse_transform_memory` (**dead**) · `tail_scorecard` (**UNKNOWN/untrusted** — the ruler-trust
dossier lists it "tracked, C-224-labelled, untested", and it is the only prediction-reading instrument
with **no join guard**, so an unmatched cell is silently dropped from the positive set).

### A.3 The same name, different quantities

⛔ **"MCR" names five things** across the apparatus: `mcr_all` (all cells), `pos_mcr`/`mcr_events`
(**mean** of per-cell ratios on event cells), `MCR` in `mcr_readout` (a different estimator on the
positive subset), `mcr_none` (**absolute mass, not a ratio**), and `mcr` in `sharpness_scorecard`. The
locked glossary defines `pos-mcr` and `size-ratio` and warns they differ — but has **no row for
`mcr_all` or `mcr_none`, the two columns the headline scorer actually emits.**

⛔ **"fraction of its due spatial structure" names two things — added 2026-10-25, and it cost a wrong
verdict.** `0.391` (B-1) is a variogram ratio **divided by the ratio of the two fields' variances**;
`0.498` (the split-half ceiling, as first measured) is a variogram ratio **with no such normalisation**.
Both are dimensionless, both live in `[0, 1]`, and they were compared directly to conclude the
structure-penalty arm had almost no headroom. Re-measuring the ceiling **with B-1's own estimator** gives
**~1.0**, so the arm's headroom was real all along. ⇒ **A normalising step changes what a statistic's
ceiling is.** Now locked as two separate terms, `GLOSSARY.md §5d`, with the general rule in **§5e**:
a comparison must name, per side, the population, the transform, the estimator **and whether it is
normalised and by what**. ⚠️ The fourth row of this appendix, and the **third** time this project has
compared two same-named statistics built by different estimators (see `GLOSSARY.md §5e`).

⛔ **"AP" names two things**: a continuous gate probability for gated arms, and a statistic with S+1
levels — **two levels at S=1** — for gate-less arms. Four scorers form it the same way; one refuses the
mismatch.

✅ **Two CRPS implementations, and they are now known to be the same quantity — parity closed 2026-09-30.**
The `audit_run` → `screen_verdict` chain gates on `views_evaluation`'s `calculate_crps_native`; the
lodestar ruler uses its own frozen `crps_ensemble`. The `views-evaluation` session ran the check this
report asked for, and the check now lives in this repo at
`reports/2026-08-15_rollout_ruler_trust_dossier/tools/crps_parity_check.py` — **re-run here independently,
zero arguments, exit 0**: **eight cases compared with `np.array_equal`, not `allclose` — max absolute
difference `0.000e+00` on every one**, including 99.5%-zero with a Pareto tail, S=1, S=128, heavy ties, magnitude
1e6 and float32 input; scalar wrappers agree to twelve digits. Both compute the same sorted-ensemble
identity with the same **biased** `1/S²` spread convention, reached independently. ⚠️ Scope, theirs:
synthetic cubes, fixed seed, their draws — it establishes that the **estimators** agree, not that any
published number is right. ✅ It was written to this report's own standards: no hardcoded paths (§0 counts
22 tool files carrying them), **exit 0/1/2 so it can be wired to a gate rather than joining §5.8's
"computed and read by nobody"**, and a docstring that declares its own status and limits — the §4.5
template, applied by someone else to a file they wrote for us. ⚠️ And one of the two is **another repo's private function**:
`crps_significance.py:27` imports `_crps_ensemble_numpy`, underscore-prefixed and not in that package's
`__all__`, so it carries no stability guarantee and can change in a PATCH release. The reason is
legitimate — the public `calculate_crps_native` returns a scalar mean while a paired origin-block CI
needs the **per-cell array** — so this is a missing public capability upstream, not carelessness here.
Reported by the `views-evaluation` session, which is filing it.

✅ **A near-miss on the primary metric, and it was luck.** On 2026-09-18 views-evaluation changed average
precision on a group with no positive truth from `0.0` to `nan`. On a 99.7%-zero field *"this group has no
positives"* is the **ordinary case**, so a mean over groups would silently become a mean over fewer
groups — and AP is this repo's primary metric. **It does not touch us:** verified that
`calculate_ap_native` is imported nowhere here; AP comes from the frozen `lodestar_score.average_precision`
or from `sklearn`. ⚠️ Note what saved us was the frozen ruler's independence, not a pin or a test — and
`rollout_ruler_core.py:398` records that our own implementation already returns NaN on an all-zero label
vector, i.e. the two converged on the same semantics by accident.

⚠️ **twCRPS is computed here, in seven places, and the report should say so plainly** because a reader
could otherwise conclude a banned metric is live in the decision path. `calculate_twcrps_native` is
imported at **7 sites across `mcr_readout.py` and `proper_score_audit.py`** ⛔ **[CORRECTED
2026-10-20: "7 sites" is 7 token occurrences — **2 imports, 3 computations**. And "both classified" is
false: `scripts/proper_score_audit.py` appears in neither A.1 nor A.2, so it is classified NOWHERE,
despite being a live `scripts/` tool and one of the two non-reproducible-CI instruments §5.6 names.]**
— `mcr_readout.py` classified SUPERSEDED
or display-only in A.2, and both documenting that FAO-02 bars it from selection. So: **computed, never
selected on.** That is the correct disposition and it is the §4.5 template being followed.

⚠️ **One metric block copied verbatim into three files** ⛔ **[CORRECTED 2026-10-20: TWO files, and
they are NOT verbatim. `git grep "match the audit tool"` returns
`reports/2026-07-14_amount_ceiling_dossier/tools/score.py:5` and
`reports/2026-07-15_volatility_ceiling_dossier/tools/score.py:5` — and their `md5sum` differs
(`d93e7077…` vs `40e5d232…`). No third site exists under any wording.]**, each with a docstring saying
"they must match
the audit tool". **Nothing tests that they still match.**

---

## Appendix B — The retraction log

31 traceable retractions or overturns, 2026-05 → 2026-09. ⛔ **[UPDATED 2026-10-25: **33**, through
2026-10. The two added are the last two rows of the table below and are dated there. The prose counts in
this paragraph are NOT re-derived — they are the 2026-09 figures and are left as measured.]** **≥14 concern a published ledger row or a
delivered verdict.** Two independent internal estimates of the base rate exist and both are lower
bounds: a systematic recompute found **3 of 9 headline figures (33%)** wrong or misleading; a deletion
test found **27 of 40 tests (68%)** passing with the call site removed.

Full table with detection mechanism and how long each stood: see the source harvest. ⛔ **[CORRECTED
2026-10-20: NO SUCH ARTIFACT EXISTS. Verified — the only occurrence of "source harvest" anywhere in
the repository, on any branch, is this sentence. So the 21 retractions not shown in the table below,
their detectors and their durations, are cited to nothing. ⛔ This makes §6's inference base
unauditable: §6 derives its conclusion from the shape of that list.]** The ten most
consequential:

| claim announced | what it actually was | caught by | stood |
|---|---|---|---|
| "coords made it worse" + the FAO eligibility table + RESULTS_LOG deltas | the baseline alone swings **2.99–3.69** run-to-run from init-RNG drift | a **two-run parity gate** before a refactor | weeks |
| **gated_NB beats climatology at h36** | **ARTIFACT** — 75.9% ⛔ **[CORRECTED 2026-10-20: SUPERSEDED FIGURE. 75.9% is from
`2026-08-15_rollout_ruler_trust_dossier/02_design.md:20`, the pre-correction design doc. The shipped
canonical value in that dossier's `results/rescore.csv` is `zero_share_of_gap` = **0.7456**, and §0,
`Appendix C` twice and `Appendix C` row 9 all use **74.6%**. The retraction log was quoting the
retracted number.]** of the gap is `crps_none`, AP is *worse*, `size_ratio` 0.0000 | Epic #263, exact decomposition of archived CSVs, no GPU | **16 days** |
| `gated_ZINBcore` "structurally dead, 5–15× worse" — arm retired | the frozen ruler's `crps_all` is **gate-independent**, so it scored the body **ungated**; re-tested, **~6× better** and viable | a re-test | 5 days |
| "it is the long-term memory — **cell carries 89%**" | `hs = o⊙tanh(hl)` makes `cell ≈ all` architecturally predetermined | `/code-review medium` | ~1 day |
| "the gate diffuses"; "the model needs to answer" | properties of a floored smoke vehicle promoted to properties of the model; one ratio's denominator held **two cells** | reading an unrelated CSV for an unrelated question (**luck**) | 3 days |
| 3 of 9 feedback-realism headline figures | cells of a 4-D grid quoted as single comparisons; two had already propagated into the register | the Claims-Ledger verification pass | 1 day |
| **`truncated_nb` fixes the #258 double zero** (it was built for it) | at L=300, 4 seeds: ΔAP@h18 −0.219 to −0.256, **7.5× the 3×MDE bar**. The double-counted zero was an accidental **brake** | re-running on a converged vehicle after the floored one was voided | **10 days** |
| σ_max = 1.60 and "the first direct observation" of rising state drift | the probe sampled the **teacher-forced warm-up**; true σ_max **7.7628**, and the real phase **collapses 40×**. *"Every downstream guard passed"* | `/code-review medium`, incidentally | ~1 day |
| BPTT-SA "unit tested (7), mutation tested 5/5, suite green (1,901)" | **inert on the production path**; two arms trained to byte-identical weights | **the result being impossible** — named as luck | 276 GPU-min |
| C-58's forecast-sniffer remedy | made **every forecasting run impossible**; no forecasting run had ever completed in this repository | a fresh look, #390 | **months** |
| **"the sharpness ceiling is 0.498, so the structure arm's upside is capped"** (2026-10-25) | ⛔ **CATEGORY ERROR.** The ceiling was measured **unnormalised** and compared to B-1's **variance-normalised** 0.391. Re-measured with B-1's own estimator: **0.983–1.009** at every lag ⇒ the ceiling is **~1.0** and the arm's premise survives. The error pointed at **killing** a justified experiment | `/falsify` on the claim *"the run is done and we have everything we need"* — probe P5, the one predicted most likely to bite | **~1 hour** |
| **`C_full = 0.665`**, the 1/n-corrected ceiling (2026-10-25) | Spearman–Brown's *additive noise, variance ∝ 1/n* premise is a statement about **counts**, applied to a `log(rate+eps)` field. Gives **0.665** logged vs **0.262** raw. **Withdrawn** | the same audit — and the refuting evidence was the prereg's **own pre-registered Q-4** | **~1 hour** |
| **`struct_penalty_weight` = 8.89624**, the structure arm's one *derived* constant, committed into a prereg under a ⭐ (2026-10-21) | ⛔ **Wrong by 21× — correct value 0.416904 — three independent ways at once.** Derived on the **`truncated_nb`** vehicle, not the `nb` one the arm runs on; pinned **v1** where **v2 had already superseded it on disk**; gradient-matched against **one** of the **three** targets the loss sums over. The script was correct, committed and reproducible every time — every error was in **what it was pointed at** | ⭐ **Simon**, asking whether there is a penalty weight per violence type (there is not) — a question about the **schema**, which exposed all three | **4 days**, and it was ~1 hour from the pod |

⛔ **Detector yield.** ⚠️ **These counts do not sum to 31, and the reason is a finding rather than an
error — but it has to be stated, because a reader who adds them up should not have to guess.** The
categories are **not mutually exclusive** (three retractions were caught by two mechanisms at once — one
was found by a code review that was *also* a disagreement between two computed numbers), and **about
eight of the 31 record no single identifiable detector**: they surfaced from a later programme
contradicting an earlier one, with no one moment of detection. Treat the numbers as **hits per
mechanism, not a partition**:

non-author adversarial review **9** · two independently computed numbers disagreeing, and the
disagreement *traced* rather than averaged **3** · user escalation **3** · a control or positive control
**2** · re-running on a converged / non-floored vehicle **2** · running the pre-registered rule's own
decision table properly **2** · a systematic recompute of every published figure **1** (which itself
found 3) · a result being *impossible* **1** · ⛔ **the test suite: 0** — and that last one is exact, not
approximate.

---

## Appendix C — Provenance, and how much weight this report can carry

**Sources read.** `reports/RESULTS_LEDGER.md` (claims M1–M76, the MEASUREMENT/INFERENCE split, the FAO-05
criteria section and the abandoned run ledger) · `reports/technical_risk_register.md` (~340 entries,
the **21** causal clusters (⛔ corrected 2026-10-20 — cluster **21**, *"prose with a self-dating expiry"*, is the class of most of this audit's findings), the §Disagreements section) · the four postmortems · `reports/GLOSSARY.md` ·
`reports/2026-08-13_evaluation_pitfalls_and_metric_battery_DRAFT.md` · the 53 dossiers, prioritising the
25 dated after 2026-08-13 · all 26 `scripts/` and the 109 dossier `tools/` · the evaluation-contract
tests · GitHub issues, open and closed.

**What is verified here.** Every figure in this report was traced to a `file:line`, an M-number, a
C-number or an issue. Nine headline figures were spot-checked back to their source file. Two
load-bearing structural claims were verified independently of the harvest: cluster 18's pre-flight
question is quoted verbatim from `technical_risk_register.md:75`, and `assert_sample_cube` /
`reference_sample_width` have **zero callers** outside their own module and tests. ⛔ **CORRECTED 2026-10-20: three tools DO call it** — `2026-08-15_rollout_ruler_trust_dossier/tools/rescore_v2.py:44,65` (**the tool that wrote the `rescore.csv` this report's headline numbers come from**), `tools/mde.py:118,143`, `tools/tail_index.py:34,84`; `assert_sample_cube` is called by `tools/partition_audit.py:55,159`. Register: `technical_risk_register.md:2265`. The defensible claim is that the two HEADLINE scorers do not call it. ⛔ **This is one of the two claims this appendix advertises as independently verified, and it is false — see `Appendix F`.**

**The DRAFT's citations are audited, not inherited.** Its own header says several are from memory,
and the dossier that absorbed it **parked** the verification. **Appendix D closes most of it**: 9 verify
exactly, 4 carry a wrong C-number, 2 are mis-vehicled or misdated, 2 are misattributed within their own
dossier, 1 is superseded, 1 is unsourced, and 1 is superseded by a better measurement. Two of the
corrections change what the entry says — including the DRAFT's flagship scheduled-sampling result, which
rests on a **VOID** run. The A1–I4 letters are kept as cross-references, not as evidence.

**The scope ceiling on everything below the surface.** Most of the underlying record is **single-seed,
single-vehicle, `sb`-only, one origin** — below this repo's own stated bar of ≥3 seeds on validation.
The two exceptions that carry real weight are the persistence re-reference (n=4 seeds) and the
multi-vehicle decomposition (4 vehicles, 1 seed each, which the repo itself classes as an *escalation
trigger, not a conclusion*).

**Three things a reader should not conclude from this report.**

1. **Not** that the adopted protocol is wrong — on this target or anywhere. ⛔ **An earlier version of
   this item said it was "wrong on *this* target, demonstrated once, decisively." That is RETRACTED
   (2026-10-11).** What this record shows is narrower: its **primary metric** cannot discriminate an
   empty map (§0.1), while its **guardrails catch one** — both here by arithmetic and at C-126 by
   multi-seed measurement. Two real weaknesses survive and neither is "the protocol is wrong": one of
   its three guardrails is structurally inapplicable here (C-224, the 99th percentile of a 99.7%-zero
   field is zero), and **the conjunction has never been executed in this repo at all**, so nothing
   here is evidence about how it performs — in either direction.
2. **Not** that the working ruler is trustworthy because it caught the protocol out. §3 and §5 are
   about the working ruler's own failures, and there are more of them than of the protocol's — because
   it is the one we actually used.
3. **Not** that the retraction count is a sign of low quality. It is a sign of a *detection* practice.
   The alarming number is not 31 retractions; it is **0 caught by the test suite** and **0 found by the
   author's own review**.

### What the first outside read changed (2026-09-30)

The `models` session read the whole document at Simon's request and supplied 15 notes. It states
plainly that it is **not** the clean-context adversarial pass §8 row 8 calls for — it read the reasoning
before forming views, which is exactly what that pass forbids. Accepted and folded in:

| their note | what changed |
|---|---|
| The forecast path has now run | §6 row 1 **downgraded** from ⛔ to ⚠️ and rewritten. Their measurement, attributed; unverifiable from this machine |
| The `c_id`-frozen concern rests on a false premise — `c_id` is **constant across history too** (100% identical in early, mid and late 6-month windows), so freezing it forward introduces no shift | ⛔ **Concern withdrawn.** It was the only ⛔ in that row and it does not survive. Their stated limit: they checked the *input* frame, not `extrapolate_time`'s output |
| Nothing scores the delivered product; no calibration at any aggregation level | **New §0 item 2 and new §5.0** — the largest addition in this revision, with their served-surface and coverage numbers and their scope caveat |
| Appendix B's detector counts do not sum to 31 | Fixed, and the reason stated: overlapping categories, and ~8 retractions with no single detector |
| Three deletion-test numbers with unnamed populations — *"the exact defect C-298 is about"* | Fixed: **four** studies, each population named, in §8 row 7 |
| §8 is a 15-item framework while §8.1 disclaims frameworks | Resolved in the §8 preamble: a menu with costs, not a gate |
| §5 and §8 drop the MEASUREMENT/INFERENCE discipline | Basis line added at the head of §5; §0 items tagged |
| What were the 8 models promoted under, if neither regime? | **Answered in §1** — two kill criteria, and the one comparative criterion VOID |
| The apparatus cannot be run by anyone else ⇒ nothing is independently reproducible | Drawn as a conclusion and **promoted into §0** |
| The lodestar's cannot-fail check is cluster 18 shape (b) in our own frozen instrument | Said out loud in §5.7 |
| Does AP@h18 predict anything about the served product? | **New §4.3 item 12** — and they then **measured it**: seven collapses of one cube, AP spanning 18.9×, `tower_point` last. Folded into §5.0 with their scope, their warning not to lead with `gate_only`, and their asymmetric 16-draw caveat |
| *(their addition)* Both regimes banned the instrument that would have seen the all-zero surface | Promoted into §0 item 2 — then ⛔ **RETRACTED, see below** |
| A two-line model-state summary belongs in §0 | Added |

### ⛔ A claim in this report was retracted, and the mechanism is one this report catalogues

**Sequence, because it is the most useful thing in this appendix.** The `models` session proposed a
finding — *both regimes banned the diagnostic that would have caught the all-zero surface* — and I
promoted it into §0. I then flagged to them what I thought its strongest objection was: PIT-for-selection
and PIT-as-diagnostic are different asks, and FAO-02 only rejects the first. I believed the claim
survived because a scope document banned PIT *"anywhere, for any reason, including 'just as a
diagnostic'."* **I asked them to kill it if they read that clause differently. They did, on three counts,
and I verified all three against the primary sources:**

1. ⛔ **The clause is not about PIT.** `SCOPE.md` item **15** covers *twCRPS, SCRPS, LogScore, QS99* —
   *"anywhere, for any reason, including 'just as a diagnostic'."* Item **14** is PIT, and its ground is
   narrower: *"FAO-02-rejected for selection; adding it now re-opens item 7 through the back door"* —
   out of scope **for that one dossier**, on selection grounds. I had applied item 15's text to item 14's
   subject.
2. ⛔ **The register prescribes PIT as a diagnostic.** `:821` (2026-06-21): *"…with a positive-cell PIT
   as a **conditional diagnostic** only"* — the exact distinction my own objection had identified, drawn
   correctly three months before the delivery. `:652` says the same.
3. ⛔ **It is built.** `scripts/proper_score_audit.py:82` implements `randomized_pit`, and its header
   documents its own limits including the Czado2009 discrete caveat.

**What survives:** FAO-02 does reject PIT *for selection*. Only the diagnostic half was wrong.

**What replaced it is stronger**, and it is theirs: **prescribed → built → left unseeded → wired to
nothing → the failure it would have caught happened.** It needs no interpretation of anyone's ban and it
is entirely inside this repo's control. It also joins §5.6, §5.8 and §5.0 into one causal story instead
of three observations — which is what the register's cluster 18 did for 23 unrelated incidents.

⚠️ **And the defect class is one this document catalogues.** Applying a clause from item 15 to item 14's
subject is **C-298's shape at the level of prose** — text from one cell of a source quoted as though it
described another — in a report whose Appendix D audits exactly that in someone else's document. It was
caught by a non-author reading primary sources instead of my summary, which is **§8 row 8**, the
highest-yield detector in the record, working on this report. It stood for one commit.

**Rejected, with reason:** their note 13 (*"§2.12 claims the counts say how, but I could not find the
counts"*) — the counts are in §2.12's table; they read the file between the two commits.

**A second reviewer, 2026-09-30 — the `views-evaluation` session, 9 notes and 9 questions.** Facts it
could see and this repo could not, all verified here: **views-evaluation is an undeclared dependency of
the ruler** (four scripts import it, absent from `pyproject.toml`, arriving transitively — §0); one of
those imports is a **private** function; **twCRPS is computed at 7 sites** and the report had never said
so; the instruments §5.0(b) needs (`Coverage(alpha)` and five others) **already exist one repo over**;
the two instruments §5.1 needs (**pEMDiv**, **Variogram**) are **stubs there**, making that a
platform-wide hole; and the platform library has **no CIs, no baselines, no skill scores and nothing that
scores a collapse**, so §2.3 and §2.5 are strictly ahead of it. Plus a second independent computation of
§0.4's claim on a synthetic field: two statistically identical models, MSE preferred one in **45%** of 200
replicates and MSLE in **50%** — chance, reported as a plain number.

**The near-miss they surfaced, recorded as luck:** views-evaluation changed AP on a group with no positive
truth from `0.0` to `nan` on 2026-09-18. AP is this report's primary metric and on a 99.7%-zero field that
is the ordinary case, not an edge case. **It does not touch us** — verified, `calculate_ap_native` is
imported nowhere here — but what saved us was the frozen ruler's independence, not a pin or a test.

**Their questions changed four things.** The header now says *no new measurement **by this session***
(Q1). §1 now **recommends** rather than only documenting — three specific amendments, and it names Simon
as the only actor who can carry them to FAO-02's owner (Q5, Q6; Q6 was the sharpest, and they were right
that declining to conclude was the larger omission). §8.2 now separates **larger finding** from **cheapest
decisive** (Q3). §8's four load-bearing rows carry a ★ (Q7). And Q4 — *"has the register edit been
made?"* — **produced the edit**: C-167 now carries the supersede note, tests green.

**A second round with `models`, 2026-09-30.** They declined a suggestion I had made — that calibration join §4.4's
closed-by-exhaustion list — and the reason was better than the entry: **it is a third disposition, not a
widening.** §4.4's entries each have a *conclusion* the evidence cannot bear; calibration has none, which
is exactly why it survived three months and a delivery. They then proposed the check that turns it from an
incident into a class, and it is the most productive thing either of us did: **grep the orphaned-metric
pool for prescriptive mentions, and partition it by whether the metric was merely built or was
prescribed first.** I ran it. **§4.5 is the result** — four piles, two prescribed gates that were never
wired, and a control case (`waste_frac`) proving the repo can disposition a metric correctly and did not.
Their bet that `recall_mass` would be the second instance was right.

**Not done, and it is the one that matters:** the clean-context adversarial pass. Still owed.
⛔ **[CORRECTED 2026-10-20: FALSE, and this appendix documents the pass itself. It ran 2026-10-11,
returned 17 findings and retracted the lead claim — recorded ~150 lines below and in the front matter.
Fifteen further falsification audits ran 2026-10-19/20 (`Appendix F`). ⚠️ A second copy of this claim
sits at the end of this appendix ("This document has not itself been through the adversarial pass…").
Cluster 21, inside the section whose job is calibrating this report's weight.]**

⛔ **And the reviews made it harder, not easier — stated because it is the one cost of this exchange.**
The pass requires an auditor who has **not** read the reasoning that produced the artifact. Both reviewers
read the whole document before forming views, and both said so unprompted. So **the `models` session, the
`views-evaluation` session, and anyone who has read either review thread are now ineligible to run it.**
The pool of eligible auditors shrank by two today. Whoever eventually runs it must not have read this
report's reasoning, this appendix, or either review exchange.

**Two things that cannot be done on this machine**, recorded so nobody re-plans around them: **AP is not
computable on a forecasting run** — the forecast months have no observed values, so there is no label —
and the 8-model pooled-validation comparison **cannot be run here**: exactly one
`predictions_validation_*` directory survives on this machine (4.7 GB, in Trash, model unidentified),
because the roster cubes were deleted in a disk cleanup. The pooled 128-draw repeat is therefore queued
on the `models` side, without AP.

### ⛔⛔ THE LEAD CLAIM OF THIS REPORT WAS RETRACTED, 2026-10-11 — found by the adversarial pass

**§0 item 1 said: *"The sanctioned selection rule, applied faithfully, promotes an empty map."*** It was
the document's opening claim, §1's stated reason for recommending amendments, §8's row 2, and the one
thing Appendix C told a reader they *may* conclude. **It is false.**

**What the rule requires** (`RESULTS_LEDGER.md:622`): eligibility is a **strict conjunction** —
superiority **and** all guardrails non-inferior, with *"MCR calibration-distance `|MCR−1|` at least 1%
closer-to-1 than baseline's"*, and ***"Fail either → Ineligible (regardless of ranking)."*** On the §0.1
row (`violet_visitor`/`sb`/h36, from the dossier's own shipped `results/rescore.csv`): `|MCR−1|` =
**0.99999** against the baseline's **0.635**, and Brier **1.04%** worse against a 1% bar. **The
conjunction rules it Ineligible.** It would have been rejected, not promoted.

⛔ **And the repo already held the measured converse, which the report never cited.**
`technical_risk_register.md:654` (C-126, 2026-06-22, **3 targets × 3 seeds**): *"CRPS-based selection
picks the timid prophet exactly as F5 warns, and **the QS99/MCR guardrails correctly veto it.**"*

**The defect's shape, and it is one this document catalogues.** The dossier's sentence is drawn narrowly
and is correct: *"applied to **its own primary metric** against its own mandated baseline."* I quoted it
accurately and then restated it as *"applied faithfully"* — a phrase **§1's own table defines** as
*"strict conjunction, written down, explicit."* So the claim was manufactured by widening a correctly
scoped quotation one step, in the lead item, of a report whose **Appendix D audits exactly that defect in
someone else's document** and whose §3.1 names it as C-298's shape. It is the third instance of this
class by me in this document (after the `SCOPE.md` item-14/item-15 swap and the M67 operand), and the
most consequential by a wide margin.

⚠️ **It also propagated a second error the same way.** §0.1's headline numbers (CRPSS **+0.0786**, zero
share **83.5%**, effect ~12× MDE) are EXP-06's, which the dossier's own EXP-09 labels *"sliding, S=16,
seed 0 (as first built)"* at `:457` and then supersedes at `:462` — *"Re-scored headline under the
canonical convention (supersedes EXP-06's table)."* The shipped canonical values are **+0.0904**,
**74.6%**, ~34× MDE. ⛔ **The reason EXP-06 was superseded is a reference/arm sample-count mismatch —
the exact defect §2.3 elevates to a method rule.** §0 therefore quoted pre-fix numbers while §2.3 stated
the rule that invalidated them. Appendix B carries a third value for the same quantity (**75.9%**, from
the dossier's *design* document, i.e. pre-registration arithmetic rather than a result).

**What survives and why §1's amendments still stand** — on a narrower and better-evidenced footing: the
**primary ranking metric cannot discriminate** an empty map from a real forecast; the protection rests on
**one** guardrail here, since QS99 is structurally inapplicable on a 99.7%-zero field (C-224) and Brier
failed by 0.04 percentage points; and **the conjunction has never been executed in this repo**, the
baseline still reading *"TBD"* (`:638`). The amendments were never really about the protocol being
wrong; they are about what this data does to a rule whose tail guardrail cannot fire.

✅ **How it was caught, and the method point.** By the clean-context adversarial pass that §8 row 8 names
as the highest-yield detector in the record — run for the first time on this document, by a reader given
the report and one question (*which claims are wrong, overstated, or unsupported by the evidence they
cite?*) and **deliberately not given the author's view of where it was weak**. It went to
`rescore.csv` and `RESULTS_LEDGER.md:622` and did the arithmetic. It returned **17 findings**; the author
independently verified seven against primary sources before acting, and all seven held. ⛔ **The lead
claim had survived four rounds of peer review by sessions that had read the reasoning first.** That is
the strongest evidence in this document for its own §8 row 8, and it was bought at the cost of the
document's headline. The remaining findings are not yet dispositioned — see the note below.

✅ **The four other verified findings were fixed 2026-10-11** and each carries a dated correction in
place: §0's *"every CRPS win beyond h1 is ARTIFACT"* (false — REAL at h6 5/12, h12 3/12, h18 2/12, h24
1/12); §0 item 2's *"any aggregate interval is too narrow by construction"* (§5.0(b) retracts it, in our
favour); §5.1's *"three separate metric rows"* in `GLOSSARY.md` (there is **one**); and §4.1's
*"**Gate** errors correlate"* (the M67 operand, third instance — the §7 fix checked §0 and not §4).

### Open findings — recorded 2026-10-11 · ✅ six adjudicated 2026-10-19, four still open

The adversarial pass returned 17 findings. Five were fixed on receipt (above, plus the lead claim). The ten
below were recorded as-received and deliberately not chased.

> ✅ **2026-10-19 — the six CHECKABLE rows were adjudicated against primary sources.** Scope chosen in
> advance: the rows whose claim is a citation or internal-consistency mismatch — the category
> `PREDICTION_LEDGER.md`'s by-kind table says this seat is reliable in — and **not** the four the reviewer
> marked as reasoning. **Result: 5 CONFIRMED and fixed in place, 1 REFUTED.**
>
> | row | verdict | the evidence that settled it |
> |---|---|---|
> | **1** | ⛔ **CONFIRMED** | `2026-09-06_ensemble_roster_dossier/07_experiment_log.md` says it itself under M70: *"`analytic_mean` is non-zero on **100%** of cells for soft-gate models."* §5.0's own table gives `gate_only` `nonzero_share` **1.00000** and disqualifies it as not like-for-like. **Same object, opposite treatment, 36 lines apart.** Fixed at all four citation sites; the like-for-like **9.9×** now carries the claim |
> | **2** | ⛔ **CONFIRMED** | The DRAFT's superseding header: *"already implements **4 of its 6** degenerates."* `2026-08-15_rollout_ruler_trust_dossier/SCOPE.md:`**`50`** — ⚠️ **not `:48`; the finding's locator was off by two lines** and the line reads *"Do not build"* — *"Do not build the spiky / frequency-right / magnitude-right cases."* **And verified in the tests: none of those three cases exists in either file.** Fixed |
> | **3** | ⛔ **RE-OPENED 2026-10-20 — the stated basis is FALSE.** The ground given was that "two
> different 13× figures were conflated". **They are one figure:** `RESULTS_LEDGER.md:230` (M8) gives
> h18 `0.0070 → 0.0912`, and **0.0912 / 0.0070 = 13.03**; `:231` (M9) gives "gate AP spans **13×**"
> across the **same four freeze arms**, same `state-freeze EXP-02` source, same 40-lesson vehicle. And
> the report does **not** cite M9 at all five sites — `Appendix D.4` carries the figure with no
> M-number, C-number or issue at all, and D.4 is one of the five sites the finding named. ⚠️ The
> ledger-status half may still hold (its floor table does put M9 under "survives"), but M8's row says
> "MAGNITUDE RETIRED … **quote the direction, never these numbers**", so the reviewer's point — a
> retired magnitude quoted five times — stands and must be re-argued. Original disposition retained
> below. | ⛔ **The finding named the wrong measurement.** The report cites **M9** at all five sites, not M8. M9 is *"`crps_all` is blind to all of it"* — four arms inside 0.0007 while gate AP spans 13×. `RESULTS_LEDGER.md`'s floor-status table puts **M9 in the "survives" row** (*"M9 is about the metric, corroborated by Epic #263"*, confidence **High**) while M8 sits in *"uninformative and NOT re-derived"*. **Two different 13× figures were conflated — and §4.4 already scopes M8's retired one correctly.** ⛔ **This is itself the provenance-mismatch class, committed by the review.** No change made |
> | **4** | ⛔ **CONFIRMED** | `RESULTS_LEDGER.md:207` classes M3 *"understated, replicated … cite the corrected number"*; M12 gives **5×** (0.0486 against a 0.2569 control). §2.7 carried both; §4.1 carried only the floored **31×**, under M12's provenance. Fixed |
> | **5** | ⛔ **CONFIRMED** | §4.3 item 12 read *"still open … queued"* for the 128-draw question that §5.0, **120 lines later**, declares *"RESOLVED 2026-10-01"* on run `20261001_012905`, 8/8 models, pooled — and §8.2 then asked a reader to fund it. ⚠️ The finding's *"fund this first"* is a paraphrase of *"Fund §5.0 first if you can fund one"*. Both fixed; the genuinely-open residue is narrower and now stated |
> | **9** | ⛔ **CONFIRMED, numbers exact** | `rescore.csv`, `violet_visitor`/`sb`/`h=1`: `crpss_vs_clim` **0.30862112…**, `delta_AP` **0.17648526…**, `ci_excludes_zero` **True**, `verdict` **REAL** — so *"no bootstrap CI was computed"* is false for the climatology reference. ⚠️ **Two cautions this seat adds that the finding did not:** `zero_share_of_gap` is **0.7866**, and Epic #263 ruled ARTIFACT at **74.6%**; and the interval is **wholly negative** (−0.0731, −0.0426), so it is a CI on a loss difference and **must not be quoted as a CI on ΔAP**. Fixed with both cautions |
>
> ⚠️ **The review's hit rate on its own checkable claims: 5 of 6.** That is a good instrument — and the one
> miss was in the same class this report spends §4 cataloguing.

⛔ **Rows 6, 7, 8 and 10 below remain OPEN and unverified.** Each is the reviewer's claim, not a verified
defect. Anyone acting on a section they touch should close its row first.

| # | section | the reviewer's claim | status |
|---|---|---|---|
| 1 | §5.0, §6, §7-S9, §4.3-6 | M70's *"+58% to +159%"* is AP measured on `gate × mu`, which the dossier flags as non-zero on **100% of cells** — structurally the same object as `gate_only`, which §5.0 itself **disqualifies** as "not like-for-like". So *"the collapse has effects larger than most interventions we tested"* may not follow from the number | unverified; the number itself is not disputed |
| 2 | §5.4 | *"**all seven** degenerate forecasts are implemented"* — the DRAFT's own header says **4 of 6**, and `SCOPE.md:48` says *"Do NOT build the spiky / frequency-right / magnitude-right cases."* §5.4 then contradicts itself two sentences later | unverified |
| 3 | §0.4, §1 amd. 1, §2.2, §3.2, D.4 | the *"gate AP spans 13×"* figure is M8's ladder on the **40-lesson floored vehicle**, of which the ledger says *"DIRECTION CONFIRMED, MAGNITUDE RETIRED by M38 … quote the direction, never these numbers."* Cited five times, four without vehicle scope | unverified |
| 4 | §4.1 | *"scrambling locations costs **31×**"* is M3's uncorrected floored number; the ledger directs *"cite the corrected number"* = M12's **5×**. §2.7 gives both, so §4.1 is the slip | unverified |
| 5 | §4.3 item 12, §8.2 | both present as open/unfunded the question §5.0 declares **RESOLVED 2026-10-01** at 128 draws, pooled 8/8. §8.2's *"fund this first"* may already be spent | unverified (internal) |
| 6 | §0 item 2 | the PIT counterfactual — *"a PIT would have shown it instantly"* — may not follow: the all-zero surface is a **forecasting** run with no truth (Appendix C's own point about AP), and `randomized_pit` consumes a **sample cube**, the input class §5.0(a) says cannot see a collapsed surface | unverified (reasoning); the prescribed→built→unseeded chain is separately verified and holds |
| 7 | §7-S4 | *"six unrelated mechanisms ⇒ it is the axis, not the implementation"* is contradicted by §4.3 item 9 (two mechanisms absent or non-functional; arms not compute-matched) and by M46's own *"the cross-arm sign pattern is POST-HOC"* | unverified (internal) |
| 8 | §2.8, §4.1, §7-S17 vs §4.4 | retraining byte-identity is used as a load-bearing positive (M22, M76) while §4.4 says *"treat training reproducibility as OPEN"* on C-317. The report applies its "both entries stand" discipline to C-119/C-317 but not to M22/M76 | unverified (internal) |
| 9 | §4.3 item 1, §8.2 | *"the h=1 win reads UNDECIDABLE"* is the **persistence** comparison; against climatology `rescore.csv` gives CRPSS +0.3086, ΔAP +0.1765, CI excluding zero, verdict **REAL**. D.5 states the distinction; §4.3 and §8.2 may not | unverified |
| 10 | §0 | *"no result in this entire record is independently reproducible by a third party"* may be broader than its premises — Appendix A.3's `crps_parity_check.py` has no hardcoded paths and was re-run independently, exit 0 | unverified (reasoning); the premises are each verified |

**Smaller discrepancies the same pass reported, also unverified:** §2.3 says `rollout_ruler_core.py` has
**86** never-skipping tests, Appendix A says **53**; §5.8 says *"MCR"* names **four** things, A.3 says
**five**; A.3's twCRPS *"computed in seven places"* may overstate by ~2.3× (7 name occurrences, 2 imports,
3 computations); *"the 25 dossiers dated after 2026-08-13"* may be **26**; §0's wandb comment is *inside*
the dependency list, not two lines above it.

⚠️ **Four of the ten are the reviewer's reasoning rather than citation mismatches** (rows 6, 7, 8, 10) and
it said so. ✅ **Those four are exactly the four still open after 2026-10-19** — the pass above took the other
six because they were checkable, and the division the reviewer drew in advance turned out to be the right one. ⛔ **And it audited no literature citation at all** — Bolin2023, Lerch2017, Gneiting2011,
Taillardat2023, Brehmer & Strokorb, Ehm2016, Aceituno2025, Lakshminarayanan2017 and the rest are
unexamined by any reviewer, which matters most for §1's fourth amendment, the one resting on no
measurement of ours.

**This appendix's provenance claims should be read as weaker than they were before this pass, not
stronger.**

### ⛔ A second claim in this report was retracted, 2026-10-10 — the same M67 row, the second time

§7's **S7** row read *"diversity is dead on the **delivered output**"*. That was wrong, and it was my
wording, not M67's. **M67 correlates errors** (`analyse_diversity.py:7,93` — `|mean(draws) − truth|` in
count space). I restated an error correlation as a statement about the output.

**What falsified it.** The 2026-09-29 handover is on disk (`~/Documents/views_selector_frames_20260929/`,
312 parquet, collapses only). Measured directly — `I_mean`, `pred_lr_sb_best`, 8 models × 13 origins,
30,334,824 rows each, row order asserted identical on the month×grid key:

| | measured |
|---|---|
| pairwise Pearson r, 28 pairs, union support | **0.273–0.552, median 0.439** |
| union support (any model fires) | **328,562** cell-months (1.083% of rows) |
| of the union: one model firing alone | **33.0%** (108,446) |
| of the union: all eight firing | **13.5%** (44,287) |
| total predicted mass, max ÷ min member | **3.64×** (bold_comet 1.681× the eight-model mean, bright_starship 0.462×) |

⚠️ `sb` only, calibration partition, **no truth** — it cannot say which member is closer to what
happened, or whether pooling beats choosing.

**What survives, and it is a sharper claim than the one it replaces.** Error diversity is dead; output
diversity is not. The members disagree about *where to put mass* and agree almost perfectly about *what
they fail to catch* — which is consistent on a 99.9%-zero field, where the error is dominated by events
every member misses. ⚠️ **That reconciliation is INFERENCE**: error correlation cannot be recomputed on
the handover, which carries no truth. The Deep Ensembles (Lakshminarayanan2017) falsification the row
used to assert is also withdrawn in part — random initialisation **did** supply output diversity here,
so the independence condition does not fail at the level of predictions; what fails is the weaker,
practical requirement that members' errors be decorrelated enough for averaging to help.

⛔ **The pattern, because it is now a pattern and not an incident.** This is the **second** correction to
the same row. The `library` session caught the first (it was the composed forecast, not the gate). Both
have the same shape: **restating a correlation without naming the quantity it is on.** A correlation
coefficient is meaningless without its operand, and I twice carried one across a sentence boundary that
changed the operand. ⚠️ Neither was caught by a reviewer reading the report — the first came from a peer
asking what the correlation was *of*, the second from me measuring a different quantity for an unrelated
reason and noticing the two could not both be described the way I had described them. **Appendix B's
"two numbers disagreeing" detector, working on this report.** It stood 10 days.

**It also qualifies §0 and §6.** Nothing in §0 asserts the overstated form, so no change there. §6's
*"anything measured once, by one instrument"* row gains an instance: M67 was the only diversity
measurement in the record until this one, and it was read as answering a question it never measured.

**The report's own weakest point, stated rather than buried.** It is a synthesis, and a synthesis is
exactly the artifact class this repo has repeatedly found to drift — *"each dossier reasons locally and
correctly; nothing forced a claim to survive contact with the others, and that is exactly where
narrative drift lives."* ⛔ **[CORRECTED 2026-10-20: it HAS — 2026-10-11, 17 findings, lead claim retracted, documented in this very appendix; plus fifteen falsification audits 2026-10-19/20, `Appendix F`. Second copy of the same expired sentence.]** ~~This document has not itself been through the adversarial pass that §8.8
identifies as the highest-yield detector in the record. **It should be.**~~

---

## Appendix D — Auditing the 2026-08-13 DRAFT's own citations

The DRAFT's header states that several of its citations are from memory and unverified, and the dossier
that absorbed it **parked** the verification. This appendix closes most of it. Of the entries checked:
**9 verify exactly, 4 carry a wrong C-number, 2 are mis-vehicled or misdated, 2 are misattributed
within their own dossier, 1 is superseded, 1 is unsourced, and 1 is superseded by a better
measurement.**

### D.1 ⛔ Corrections that change what the entry says

| DRAFT | correction |
|---|---|
| **G6** — "scheduled sampling was falsified as the fix: ε∈{0,0.1,0.25,0.5} gave flat rollout AP and *harmed* T=0" | ⛔ **The cited run is VOID.** It ran on the floor-limited 40-lesson vehicle (control 0.77× prevalence, below random ranking); all four arms sit at 0.02–0.04 and nothing can discriminate. **Take 2 at L=300, 4v4 seeds, changes the content three ways:** SS is **worse, not flat** (−0.0426, p=0.0286, 4/4 pairs down); the **T=0 harm claim does not survive** — the anchor guard **passes** (ΔAP h1 −0.0277 against a 0.0440 limit), so it is a retention result, not a trade; and the mechanism **reverses** — SS largely *fixed* the zero collapse (`act_ratio` 9.4× at h18, 28× at h36) and lost AP anyway. The correct statement is **"it answered more, in worse places, and lost."** |
| **G5 / I4** — the oracle-input rollout, credited to "2026-08-14, africa truncated_nb", with oracle AP 0.30→0.27 and free-running 0.008 | ⚠️ **Priority and vehicle both wrong.** The oracle probe is **2026-07-25 on zinb**, three weeks earlier, where it had already decomposed occurrence-as-bug from magnitude-as-ceiling and said so. And the quoted numbers come from the **VOID** 40-lesson vehicle. The replicated figures are oracle **0.4745 → 0.4793 → 0.4577** and free-running **0.4745 → 0.2569 → 0.1370** (M10/M24). ✅ The lesson survives and is *strengthened*: the oracle does not decay **at all**. |
| **E1** — composed activation "12–69× below what the gate implies; `P(body>0)` 0.015–0.154 where it should be ≈1" | ⚠️ **Superseded by a 4-seed measurement, and the sign of the diagnosis flips.** The verified L=300 step-1 decomposition: **`G = 1.280` — the gate *over*-predicts occurrence by 28%** — attenuated **×0.332** by the body's own `P(draw>0)`, giving `C = 0.424`. That is a **~2.4× shortfall, not 12–69×**, and **the gate is hot, not approximately right.** |
| **I2** — "jump-risk IS predictable (spearman 0.79); quantiles 48% sharper; tail ξ≈0.8" | ⚠️ **Partly misattributed, and the actionable part is missing.** 0.792 is **not** a jump-risk spearman — it is how well the learner ranks realized \|resid\| on active cells (the ARCH result); jump *risk* is measured as AP-skill/AUC on exceedance. **48.5% sharper verifies exactly.** But **ξ is 0.711–0.879 depending on threshold and is roughly state-invariant across volatility tertiles (0.841/0.619/0.800)** ⇒ **tail heaviness is not a per-state learnable target; the head needs a *fixed* heavy tail.** That is the design consequence and the DRAFT does not carry it. |
| **H1** — "climatology overtakes by ~h18" | ⚠️ **Superseded, and it matters.** That is the v2 board's figure. On the corrected ruler the crossover is **≈ month 20** and **h18 is still a win (+0.032)**. State which substrate. The lesson is unaffected. |
| **C1** — nb "beats climatology" on `crps_events` ≈14 with `size_ratio` 0.0 | ⚠️ The win was on **`crps_all`**, not on `crps_events`. `crps_events` ≈14 is the *tied* number showing no arm cracks magnitude. Add the identifier: **C-219**. |

### D.2 ⛔ Four wrong C-numbers, and one unsourced figure

| DRAFT cites | what that number actually is |
|---|---|
| **C-72** for "`exp(µ)` overflowed float32 → 63% Inf" | a misspelled `requirments.txt` in the package root. ⛔ **And the "63%" appears nowhere in `reports/` except the DRAFT and a verbatim copy of it — the figure is self-citing.** Either find the run or drop it. |
| **C-85** for the stale-cache / all-equal-but-wrong instance | a demoted entry about a hardcoded flip probability. Better-sourced siblings exist: **C-321** (`--keep-cubes` disabling the multi-arm guard) and the `--resume` defect below. |
| **C-132** for the ensemble dropping the `by_*` gate channel | the wandb train-run lifecycle being dropped by a manager override. ✅ The phenomenon is real and better documented: the pool loops over a meta that listed only `lr_` targets — *"the framework CAN pool `by_` … it simply is never asked to."* AP 0.316 against members 0.38–0.47, **fixed by adding a list, no retrain, no code change.** |
| **C-286** for the same entry | `partition_audit` recording two fields that cannot express the state they name. |

### D.3 ✅ Verified exactly

**A7** (`size_ratio` ≈0.02, and it was called the next lever — the dossier's literal wording) · **C4**
(`body_mask=pos_cells` lifted `size_ratio` ×11–60 and blew `crps_all`, ns 0.083→24.6 — **and both F2 and
F3 fired**, which strengthens the point) · **F4** (the 2-NB mixture; understated if anything — 2.14% /
1.51% / 3.45% across three seeds, all CIs excluding zero, and component-2 *directly measured alive*:
median `w|active` 0.71–0.92, median `μ2:μ1` **14×–690×**) · **I1** (amount ceiling, 0.303 < 0.367,
confound control at 0.99999 — **and EXP-01 was VOID on an instrument fault first**, a clean
VOID-then-confirm) · **D5** (never pool the rollout — and **a verdict was reversed by it**) · **C7** (the
different-months bug — with the co-equal half the DRAFT omits: **terminology drift**, fixed by a locked
glossary alongside the frozen ruler) · **H3** (FAO-02's choices — and the tension is no longer a
hypothesis; see §0.1) · **D1**'s C-number (C-184 is right — but **M65 re-scopes its magnitude: the ~20%
figure used to size every recent screen is ~5× too large** for today's vehicle, measured 4.1% at h18).

### D.4 ⚠️ The DRAFT's largest unverified block

**A3, A5, A6, B1–B3, E1, G1–G4** all trace to issue #258 or ADR-070 rather than to a dossier, and
**none reproduces from `reports/`.** Two have exact dossier-sourced substitutes that should replace
them, because they are corroborated independently:

- **A3** (CRPS blindness): `crps_all` **0.1353 / 0.1352 / 0.1350 / 0.1346** across four arms at h18
  while gate AP spans **13×** and precision@k **37×**.
- **A6** (window artifact): `crps_all` agreeing to **four** decimals at h36 (0.8753–0.8755).

### D.5 Three things the DRAFT omits and should not

1. ⛔ **The degenerate *learner*, not just the degenerate forecast.** The amount- and volatility-ceiling
   dossiers both pre-registered a VOID-on-instrument-failure clause using a learner **given the answer**,
   which must score near-perfect or the instrument is broken. It fired, caught a real numerical bug, and
   predates the DRAFT's §5 red-team proposal by three weeks. It is the complementary half of the method.
2. ⛔ **`crps_all`'s win over *persistence* — not just climatology — is ARTIFACT at h6/h18/h36**
   (zero-share 73–82%, ΔAP negative at every horizon), and **h1 is the only non-artifact signal in the
   whole programme**, UNDECIDABLE purely because no bootstrap CI was computed. The structural remedy is
   already written down and unbuilt: *"make the collapsed control unusable as a reference — persistence
   and climatology emitted by the scorer on every run, the artifact verdict applied automatically."*
3. ⚠️ **One theorem reframes three failed programmes as one barrier.** Aceituno2025 Thm 4.6: minima found
   on short horizons do not generalise to long ones, with a gap of `O(e^{λ(T_h−T_l)})`. Scheduled
   sampling, increased teacher forcing and `truncated_nb` all changed **what the model is fed**; **none
   changed what is optimised.** So long-horizon training *"has never actually been tried here, despite
   three experiments that resembled it."* ⚠️ INFERENCE twice over — the theorem is proved for feedforward MLPs and the
   HydraNet is an RNN, so the transfer is argued rather than established. Still the sharpest unexamined
   direction in the record.

---

## Appendix E — The literature, and where it contradicts us

**Added 2026-10-19.** Ten papers entered `~/brain/9_library` on 2026-10-18 (621 papers, 2,927 extracted
claims). This appendix brings them against the report.

⛔ **THE RULE FOR THIS APPENDIX, and it is Simon's: COMPLEMENT, DO NOT REVISE.** Where a paper
contradicts something above, **nothing above is rewritten.** Both positions are stated — *this is what we
had, this is what the paper says* — and the collision is left standing. A challenge we lost is worth more
to this document than a citation that agrees with us. ⚠️ So **no claim in §0–§8 has been edited on the
strength of anything below.** If a reader finds the two in tension, that tension is the content.

**Provenance.** Every entry cites a `paper_id` and the locator the library's extraction carries. Claims
are labelled with the library's own confidence (`proven` / `argued` / `empirical`). ⚠️ **Where I extend a
paper's claim to our setting, that step is mine and is marked INFERENCE** — several of these papers are
about continuous weather variables and ours is a sparse count field, and that gap is not always small.

---

### E.0 ⛔ A RETRACTION IN THIS APPENDIX, AND A PROVENANCE WARNING ON ALL TEN SOURCES

#### E.0.1 ⛔ RETRACTED, same day: `Walz2026_MonotoneDependenceAUC` cannot be cited about AP

**The first version of this appendix (commit `2c2ad44`, 2026-10-19) cited `Walz2026` as *"our AP /
S-quantisation scar named by a theorem."* ⛔ That is withdrawn. It was wrong twice over**, and it is the
provenance-mismatch class this report devotes Appendix C to — committed by me within an hour of writing an
appendix warning about it. Found by the `opponent` session, **verified here independently** on the PDF's
own text layer (`pdftotext -W 2000 -H 2000`, 21,011 words):

| check | result |
|---|---|
| occurrences of **"precision"** | **0** |
| **"average precision" / "precision-recall" / "PR-AUC"** | **0** |
| **"prevalence" / "imbalance"** | **0** |
| what "AUC" means in the paper | **"receiver operating characteristic"** — ROC-AUC |

**Error 1 — wrong AUC.** Our workhorse is **AP, which is PR-AUC**. The paper is about **ROC-AUC**. ⛔ The
two diverge precisely on **prevalence**, which is our defining condition (99.7% zero), and the paper never
mentions prevalence at all.

**Error 2 — wrong discretisation.** The stability result *is* in the paper (⛔ **at extracted-text LINES
588 and 591, not printed pages — a 38-page arXiv preprint with no printed pagination; both sit on PDF
page 8. I cited my own `grep -n` output as page numbers, in the paragraph retracting a provenance error.
Corrected 2026-10-20.** Text: *"AGC(X, Y_k) depends much less on the discretization level k"* and *"a
factor that may be arbitrary"*) — but it concerns
discretising **the OUTCOME Y into ordinal levels.** Our scar is that **the PREDICTION is quantised to S+1
rank levels** by the sample-cube width. **Different quantities. I conflated them.**

✅ **Where `Walz2026` CAN be cited here**, and it is still useful:
- About **`gate_auc`** — which is ROC-AUC, and which §8.1 lists among the metrics *"nothing reads."*
- It upgrades our `ΔAP = 0.00000` from a measured quirk to a **structural property**: these are **copula
  functionals**, invariant under strictly increasing transformations of both margins, so
  calibration-blindness is definitional for the class, not incidental to our implementation. **MERELY
  COMPLEMENTS** — no contradiction.
- Its **DeLong-type analytic tests** for comparing two such measures against a shared outcome are worth a
  look given §5.6's two irreproducible bootstrap CIs. ⚠️ Whether a DeLong test survives spatially
  dependent origin blocks is **not addressed in the paper**, and the `opponent` session's expectation is
  that it does not.

#### E.0.3 ⛔ NINE "VERBATIM" QUOTATIONS IN THIS APPENDIX ARE NOT VERBATIM — one defect, one cause

**Found 2026-10-20 by an independent falsification audit that extracted all twelve available PDFs.** The
defect tracks provenance exactly:

- ✅ **Every `*"..."*` I read off a PDF page IS verbatim** — `Raftery2005` ×5 (printed p. 1155),
  `Hawkes1971` (p. 86, p. 90), `Moller1998` (p. 451, p. 452), `Resin2026` (p. 2), `Walz2026` ×2.
- ⛔ **Every `*"..."*` that came from the library's LLM-EXTRACTED CLAIM TEXT is a paraphrase or a
  two-sentence composite**, not the paper's prose: `Hawkes1971` in E.2.1, `Moller1998` in E.2.2,
  `Moller1998`'s `z(t)` claim, `Raftery2005`'s claim-2 block, `Resin2026`'s collapse quote,
  `Gneiting2023`'s CORP quote, `Walz2024_EasyUQ`, both `Walz2024_Precipitation` quotes, and
  `Gneiting2026`.

⛔ **I quoted a machine summary of a paper as if it were the paper.** The text layers were fully
extractable in every case, so this is not an extraction artefact — it is a method error, and it is the
same provenance class as E.0.1 and as `Appendix D`.

**What this does and does not cost.** The auditor's verdict on substance: *"I found no case where a paper
was made to say something it does not say — only cases where it was made to say it in words it did not
use, at a locator that does not hold."* ✅ So **every conclusion in this appendix stands**;
⛔ **its representation of having read the sources does not.** ⚠️ **Treat every blockquote below as
PARAPHRASE unless this note lists it as PDF-read.** Demoting them properly is a mechanical pass over the
appendix and is **not yet done** — recorded as debt rather than described as fixed.

**Three further locator defects from the same audit:**
- ⛔ `Gneiting2023`'s `S̄ = MCB − DSC + UNC` is **Theorem 2.23**, not 2.26 — and the paper credits it to
  **Dawid (1986) and Pohle (2020)**, an unflagged cite-from-elsewhere of the class E.2.3 polices.
- ⛔ `Moller1998`'s `z(t)` is **eq (5)/(6) on p. 458**, outside the cited "Theorem 1, eqs. 2–4, pp. 456–457".
- ⚠️ The corpus size is given as **621** here and **615** in three places; the actual counts are **617
  PDFs / 621 sidecars**, so three corpus-wide *absence* claims cite a figure matching neither.

#### E.0.2 ⛔ All ten sources have EMPTY stored evidence

`papers/_meta/<id>.json` carries **`"key_passages": []` for all ten.** The `opponent` session established
(library GitHub #53) that Phase 5 of `/library add` **never ran for the 18 October batch or the two
before it — 52 papers, 100% of each batch.**

**Consequence for this appendix:** every locator in it was read out of a PDF text layer, not from stored
verified passages. The ten were clipping-tested (`#51`) and came back clean — zero word delta on nine,
four words on `Raftery2005` — so **these particular text layers are sound, which is not a general property
of that corpus** (`Cox1955` lost 731 words to clipping). ⛔ **But a later reader cannot verify these
citations without re-reading the PDFs.** Get the key passages added before any of this is relied on
outside this appendix.

---

### E.1 ⛔ TWO PAPERS CONTRADICT US — and a third did not, by my error

⚠️ **This section was filed as three contradictions. E.1.1's gate half is DEMOTED** to a complement after
the `library` session answered the transfer question I had flagged, and the fault was a category error of
mine. **E.1.2 and E.1.3 stand. E.1.1's contradiction of our ensemble DESIGN also stands** — see the box.

#### E.1.1 ⛔ DEMOTED 2026-10-19 — this was a CATEGORY ERROR of mine, not a contradiction

> ⛔ **Read this box before the entry below it.** The entry was filed as the first of three
> contradictions. **It is not a contradiction, and the fault is mine.** The `library` session was asked
> whether Raftery's variance decomposition transfers to a Bernoulli gate — the step I had flagged
> INFERENCE — and its answer, checked by derivation here, is **no**:
>
> - **The transfer fails on a constraint.** For a Bernoulli outcome `Var(Y|p) = p(1−p)` is a deterministic
>   function of the mean. **There is no free within-forecast `σ²` to be missed**, so the mechanism
>   Raftery's decomposition describes has no analogue. `Raftery2005` C-3548 specifies **normal component
>   PDFs** explicitly, and **nothing in the 615-paper corpus treats underdispersion of an
>   ensemble-derived PROBABILITY.**
> - ⛔ **And the direction may reverse. DERIVED here:** a mixture of `Bernoulli(p_k)` over members **is
>   exactly `Bernoulli(p̄)`** — any distribution on `{0,1}` is Bernoulli — so there is no extra dispersion
>   to find. Further, by concavity `p̄(1−p̄) ≥ Σ w_k p_k(1−p_k)`, so the pooled forecast is if anything
>   **MORE** dispersed than its average member: **the opposite of Raftery's direction.** ⚠️ This corrects
>   the `library` session's phrasing too, which said the mixture *"is not Bernoulli"* — for a single
>   binary outcome it is.
> - ⛔ **The real error is that I matched two different quantities to one word.** Raftery's
>   "underdispersive" is about the **predictive interval at one location being too narrow.** Our
>   `b ≈ 1.19` is about the **slope of the calibration curve across cells** — the spread of predicted
>   probabilities over units being too concentrated. **Same word, different object.**
>
> ✅ **What survives is a complement, and we already measured it.** Averaging in probability space does
> pull predictions toward the middle, which is the one real point of contact — and `RESULTS.md §3`
> measured that contribution at **~8%**. So the literature names a mechanism this programme had already
> quantified and found minor. **`Raftery2005` does not contradict our gate finding.** It is reclassified
> **COMPLEMENTS**, and the entry is kept rather than deleted because the record of the mis-match is worth
> more than a clean appendix.
>
> ⚠️ **Only the gate half is demoted.** The paper's contradiction of our **ensemble design** (below) is
> unaffected — that part is about pooling rules and matched draw counts, and does not depend on the
> variance decomposition transferring.

**The original entry, retained:** our gate under-dispersion may be structural, not a defect — `Raftery2005_BayesianModelAveraging`

> **What we have.** The gate's logits are compressed: slope `b ≈ 1.1875` on the shipped model, measured six
> ways, surviving ten rounds of two-seat review (`2026-10-18_gate_underdispersion_prereg/RESULTS.md §1`).
> We concluded the compression is **in the network** (the reporting path explains ~8%), spent three days on
> candidate causes, and banked **temperature scaling** as the one-parameter fix.

> ⛔ **What the paper says.** `Raftery2005_BayesianModelAveraging`, claim 2, **PROVEN**: the BMA predictive
> variance decomposes into **between-forecast variance (ensemble spread)** plus **within-forecast variance
> (σ²)**, and this *"provides a theoretical explanation of the empirically observed coexistence of
> spread-error correlation and **underdispersion in forecast ensembles**: the ensemble spread captures only
> the first component, **systematically underestimating total predictive uncertainty**."* Empirically, the
> raw ensemble range contained **29%** of observations against a nominal 66.7%.

**The collision.** Our posterior is an ensemble — 16 MC-dropout draws. If under-dispersion is the
*structural* consequence of reading spread off an ensemble and ignoring within-member variance, then we
measured a known property of ensembles and diagnosed it as a defect of our network. ⛔ **And the remedy
differs:** BMA adds a σ² term; we proposed rescaling the log-odds. A temperature fit can match the
observed spread without supplying the missing variance component, which is the Goodhart shape this report
spends §4 cataloguing.

⚠️ **INFERENCE, and the gap is real.** Raftery's decomposition is for normal component PDFs on a
continuous outcome. Our gate is a Bernoulli probability, where "variance" is a function of the mean and
cannot be inflated independently. **Whether the decomposition transfers to a probability is not settled by
this paper and I have not derived it.** What survives transfer without argument is the weaker, still
uncomfortable point: *ensemble-derived uncertainty is expected to be too narrow, and we treated narrowness
as evidence of a bug.*

**Neither position is retracted.** Our `b` is a measurement and stands. The causal story is an inference
and now has a competitor from the literature.

⛔ **The same paper contradicts our ENSEMBLE DESIGN, and supplies a fix for a scar in §8.** Verbatim,
p. 1155: ensembles *"tend to be underdispersive"*; *"intervals based solely on the ensemble spread
incorporate the first component but not the second"*; *"the BMA weights can be used to assess the
usefulness of ensemble members"*; the BMA point forecast *"had root-mean-square errors 7% lower than the
best of the ensemble members"*; and *"the BMA PDF can be represented as an unweighted ensemble of any
desired size, by simulating from the BMA predictive distribution."*

Three consequences, all from the `opponent` session's reading:

1. ⛔ **Our benchmark is EQUAL-WEIGHT pooling, and Raftery's point is that a raw ensemble is the wrong
   object to score.** M67's 0.96–0.99 error correlation is a measurement that **between-forecast variance
   is near zero** — in this frame, that the raw ensemble supplies almost none of the variance a calibrated
   distribution needs. **Same measurement, read as a deficiency of the POOLING rather than of the
   ensemble.**
2. A **weighted** combination beat the best member by **7% RMSE**. Our subsample result shows **equal-weight**
   pooling does not — and §7-S7 generalises from the one rule tested to the class.
3. 🔴 **Our matched-S confound has a PUBLISHED solution.** §8's row ★4 exists because persistence was
   understated by **31%** partly through S-mismatch. BMA re-emits its PDF **as an unweighted ensemble of
   any desired size**, so arms and reference can be compared at equal draw count **without re-running
   anything.**

⚠️ **Inference flag, the `opponent` session's:** BMA weights on 8 members correlating 0.96–0.99 would
very likely degenerate onto one or two, which would **support** our conclusion. The paper does not report
that case — its simulations cover calibrated or overdispersed ensembles, **which is not our regime.**

#### E.1.2 A deterministic backbone plus post-hoc calibration beat the ensemble — in our region — `Walz2024_PrecipitationTropicalAfrica`

> **What we have.** §7-S7: the 8-member ensemble was buying **sample count, not member disagreement** —
> M66 (pooled beats best member at unequal S; subsample to 16 draws and it falls *below* the best member),
> M67 (errors correlate 0.96–0.99 across all 28 pairs). The conclusion drawn was about our ensemble's
> diversity.

> ⛔ **What the paper says.** `Walz2024_PrecipitationTropicalAfrica`, **EMPIRICAL**: a two-stage **U-Net
> CNN + EasyUQ** for 24-hour probabilistic precipitation over **northern tropical Africa** *"clearly
> outperforms all competing methods — including monthly probabilistic climatology, raw and postprocessed
> ECMWF ensemble forecasts, logistic regression, distributional index models"* — on **both occurrence
> (Brier) and amount (CRPS)**. And separately: *"The HRES+EasyUQ forecast — a single deterministic NWP run
> converted to probabilistic via EasyUQ — **outperforms the ensemble-based EPS+ISO forecast** for
> precipitation occurrence, demonstrating that a well-calibrated deterministic forecast can be"* competitive
> with an ensemble.

**The collision.** We asked *"is our ensemble diverse enough?"* The literature's answer in the same region,
on the same kind of sparse-extreme field, is that **the ensemble may be the wrong instrument**: one
deterministic backbone plus an isotonic post-hoc UQ layer beat a physics ensemble on occurrence, at a
fraction of the cost. ⛔ **That is not a refinement of M66/M67 — it questions whether the ensemble axis was
worth the eight members.**

⚠️ **Scope, stated because it is doing a lot of work:** 24-hour lead time against our 36 months; gridded
precipitation against fatality counts; IMERG/ERA5 predictors against conflict history. **Different target,
different horizon, same region and same architecture family.** Treat it as a strong prior on the *shape* of
the answer, not as a result about us.

#### E.1.3 For a binary gate, PIT calibration IS the strongest notion — and we rejected it — `Resin2026_HierarchiesCalibration`

> **What we have.** FAO-02 is LOCKED: CRPS primary, QS99/Brier/MCR as guardrails, and ⛔ **twCRPS,
> LogScore and PIT explicitly REJECTED** (`technical_risk_register.md:655`, on forecaster's-dilemma
> grounds, `Lerch2017`). §0 separately argues *"a PIT would have shown it instantly"* about the all-zero
> surface — a claim the 2026-10-11 adversarial pass flagged as **open finding 6**, still unverified.

> ⛔ **What the paper says.** `Resin2026_HierarchiesCalibration`, claim 4, **PROVEN** (Section 2.4,
> Figure 3): *"**For binary outcomes, numerous otherwise distinct notions of calibration collapse**:
> auto-calibration, class-wise calibration, class-wise confidence calibration, threshold calibration,
> conditional exceedance probability calibration, **probabilistic calibration**, double PIT calibration, and
> strong quantile calibration **all coincide**."* And `Gneiting2023_ConditionalCalibration`, claim 1,
> **PROVEN**: auto-calibration is the **strongest** notion, implying probabilistic calibration (uniform
> PIT) and marginal calibration.

⛔ **COLLISION LARGELY DISSOLVED, 2026-10-20 — and by prior art in our own register.** The locator I
cited, `technical_risk_register.md:655`, **rejects twCRPS and positive-subset conditioning, not PIT** —
and the line above it *recommends* PIT/coverage. The real PIT locator is **`:965`**, and it reads:
*"PIT is **FAO-02-REJECTED for *selection***… retained **as a diagnostic only, never as a gate**."*

**So the register already drew the selection-versus-diagnosis distinction I presented as a new insight —
on 2026-07-27.** ⛔ **That is the fourth instance in this programme of re-deriving this repository's own
prior art** (see `RESULTS.md §12` on the gate axis). The collision as filed is retracted.

**What survives, narrowly.** Probabilistic calibration *is* PIT calibration, and **our gate is binary**,
so for the gate the notion is **provably equivalent to auto-calibration — the strongest available.** That
makes PIT-as-diagnostic a stronger instrument for a binary gate than the register's wording implies, and
it is worth saying. It is **not** the governance contradiction I claimed. ⛔ Rejecting PIT for a binary gate is not the same decision as rejecting twCRPS for a
heavy-tailed magnitude, and FAO-02 made them as one decision.

⚠️ **What this does NOT say.** It does not say FAO-02 was wrong. The dilemma argument was about
**outcome-selected scoring used for ranking**, and PIT was rejected in that company. The paper establishes
a **logical equivalence for binary outcomes**, not that the rejection's reasoning fails. ⛔ **But the
rejection was applied to a binary gate on the strength of an argument about extremes, and the collapse
result means that application needs its own justification.** FAO-02 is governance and is not edited here.

✅ **Resin2026 also names our MCR failure in its own words** (p. 2): *"care needs to be taken to avoid
reliance on techniques that make forecasts appear calibrated with respect to a particular notion of
calibration, even though they might be far from the goal of auto-calibration."* **MCR ≈ 1.0 while
activation is 4–115× wrong is that sentence.** ⚠️ Our framing attributes it to aggregating over zeros;
Resin's frame says the cause is **position in the hierarchy**, so the remedy is **naming which notion each
guardrail certifies**, not adding guardrails. ⚠️ The mapping onto MCR is the `opponent` session's, not the
paper's.

✅ **And a held, read replacement for an unaudited citation.** `Resin2026` covers **count data and
double-PIT calibration explicitly**, extending Czado et al. (2009) — the count-calibration citation
**§8.3 flags as unaudited.** Cite Resin instead.

**It also sharpens open finding 6 without closing it.** Our *"a PIT would have shown it instantly"* is
about the **body's** all-zero surface, not the gate — and for a continuous or count outcome the collapse
does not hold, so the claim keeps its original problem. **Row 6 stays open.**

---

### E.2 ⛔ TWO PROVEN IMPOSSIBILITIES THAT BOUND §8.3 AND THE STATE-BASED EXTENSION

§8.3 proposes diagnostics matched to the factorization of the data-generating process, and notes that
**none of them can see dependence or self-excitation**. The literature is harsher than that: it says a
whole *class* of diagnostic cannot see it, as a theorem.

#### E.2.1 Self-exciting and doubly stochastic processes are indistinguishable at second order — `Hawkes1971_SelfExcitingSpectra`

> ⛔ **PROVEN** (Section 3, eqs. 12–18, pp. 85–86): a self-exciting process with exponential kernel
> `g(v) = α·exp(−βv)` has a covariance density *"**identical** to the covariance function of a doubly
> stochastic Poisson process whose intensity has the same covariance structure. Thus self-exciting and
> doubly stochastic processes with matching second-order properties **cannot be discriminated by methods
> based on the estimated point spectrum, covariance density, or variance-time function**."*

**What this challenges.** The hope — Simon's, and mine in §8.3 — that a *latent risk field* and a
*self-exciting dynamic* can be measured by separate, near-orthogonal diagnostics. ⛔ **At second order they
are the same process.** No marginal, pooled, spectral or variance-based diagnostic can separate "the risk
field was high here" from "an event here made more events likely". **That is the strongest single result
against the orthogonality programme, and it is a theorem, not a caution.**

⛔ **AND IT INVALIDATES AN INSTRUMENT ALREADY PRE-REGISTERED ON THIS BRANCH.** Hawkes's verbatim
sentence (printed p. 86, Biometrika 58(1)): *"given a set of data from a point process we **cannot expect
to discriminate** between a self-exciting and a doubly stochastic process on the basis of the estimated
point spectrum, covariance density or other equivalent functions such as the variance-time function"* —
and at p. 90, that he concentrated on second-order properties *"since these are most amenable to data
analysis for testing goodness of fit."* He then constructs the matching doubly stochastic process term for
term (eq. 18, same page).

⛔ **RETRACTED 2026-10-20 — I WROTE THIS ABOUT A FILE I HAD NOT OPENED, which is this programme's own
named failure mode.** The claim was that Hawkes invalidates
`reports/2026-09-30_dependence_ratio_prereg.md`. **It does not, because that prereg is not about
self-excitation at all:** it contains **zero** occurrences of self-excitation, contagion, diffusion,
clustering or Hawkes, and what it predicts is `Var_d(total) / Σ_cells Var_d(cell)` — a variance ratio
across **MC-dropout draws of our own forecast cube**, derived from reading the code. ⛔ **There is no
observed point process for Hawkes's indistinguishability theorem to apply to.** I amplified the
`opponent` session's conditional ("*if* its stated purpose is to establish self-excitation") into a
headline invalidation without checking whether the condition held. It does not.

✅ **What survives is the general warning, with no artifact attached:** a dependence ratio, a covariance
density, a spectrum and a variance-time function are **all second-order**, so **none of them can ever
establish self-excitation against a doubly stochastic alternative.** If this programme ever builds such an
instrument for that purpose, Hawkes's theorem applies. It does not apply to the prereg we have.

**What it leaves open, which is the constructive half.** Discrimination is possible, but only by
**likelihood on the conditional intensity**, not by a summary statistic.
`Ogata1982_LinearIntensityModels` is exactly that: a unified model
`λ(t|H_t) = μ + ∫g(t−s)dN_s + ∫h(t−s)dX_s` in which **`g = 0` gives a doubly stochastic Poisson process
driven by the external input and `h = 0` gives no external causation** (Section 2, eq. 2.3, pp. 102–103),
fitted by partial likelihood with **AIC over response-function orders** (Section 4, eq. 4.1). ⛔ **So the
separation Simon wants is a MODEL-SELECTION problem, not a metric problem.** That reframing is the most
useful thing in this appendix for the planned article.

⛔ **But be clear what it costs — this closes the question I put to both sessions.** Ogata's is a **model
class with a selection procedure, not a diagnostic runnable on a delivered cube.** It needs a point-process
intensity fitted to event times; a 360×720×months sample cube cannot be scored with it. **So the honest
answer is that there is NO usable evaluation instrument for self-excitation in either paper.** ⚠️ And the
compensator-based residual analysis one would reach for is **Ogata 1988, not Ogata & Akaike 1982** — not
held, and `Ogata1982` contains none of it. ⛔ **Do not cite `Ogata1982` for residual analysis.**

#### E.2.2 And a second non-identifiability, from the spatial side — `Moller1998_LogGaussianCoxProcesses`

> **EMPIRICAL** (Section 4, pp. 460–462): for certain parameters the pair correlation functions of a
> **Thomas process** (clustering by offspring) and an **LGCP** (clustering by a latent Gaussian field) are
> *"nearly indistinguishable, and **neither the F and G nearest-neighbor statistics nor the third-order
> characteristic z(t) reliably discriminate** between the two models in finite samples."*

**What this challenges.** The same hope, from the spatial direction: *clustering because the latent risk
field is smooth* and *clustering because events beget events* are not reliably separable in finite samples
either. ⛔ **Two independent literatures, two non-identifiability results, same conclusion.**

✅ **And one genuinely usable instrument, which we do not have.** `Moller1998`, claim 1, **PROVEN** (Theorem
1, eqs. 2–4, pp. 456–457): for a stationary LGCP all higher-order moments are determined by the first and
second order, which *"provides a **testable third-order summary statistic z(t) that identically equals 1
for all t > 0** under an LGCP."* **That is a falsifier for a model class** — the shape of thing §8's table
demands and our apparatus does not contain. ⚠️ Subject to E.2.2's own warning that `z(t)` does not
discriminate LGCP from Thomas; it tests the LGCP assumption, not the clustering mechanism.

⚠️ **INFERENCE, flagged.** Our field is a **discretised raster**, not a point pattern, and these results
are stated for point processes on a continuum. **Whether they transfer to PRIO-GRID cell-months is not
established by these papers and I have not derived it.**

#### E.2.3 ⛔ CORRECTION to this appendix: the grid question is PARTLY answered, and more narrowly than I asked it

**The first version of this appendix recorded the grid-discretisation challenge as "sought and NOT
found".** ⛔ **That was wrong — I missed the relevant passage.** `Moller1998`, printed **p. 452**, on
lattice models that hold intensity constant within square quadrats:

> *"such models do not converge to anything reasonable as the sides of the quadrats tend to zero."*

⚠️ **And a provenance caveat that must travel with it:** that clause is **attributed to Rathbun & Cressie
(1994)**, which is **not held**. Møller endorses it (*"as noticed by"*) and builds his design argument on
it, but he is not the primary source. ⛔ **Cite it as *`Moller1998` p. 452 endorsing Rathbun & Cressie
1994*, or fetch the primary.** (Named by the `opponent` session rather than let through — it is the
cite-from-elsewhere class this report's Appendix D exists for.)

**What it does and does NOT establish.** ⛔ It does **not** say our zeros are artefactual — a low-intensity
point process genuinely gives mostly-empty cells **at any resolution**, so the premise *"the grid
manufactures the zero inflation"* is **not** supported. What it says is narrower and more useful: a
**cell-constant-intensity model has no continuum limit**, so its fitted parameters estimate **no
grid-independent quantity.** PRIO-GRID at 0.5° is exactly that construction.

🔴 **The consequence for this report is a missing sensitivity analysis, not a refuted premise.**
`size_ratio`, `act_ratio`, zero-share and the ARTIFACT verdict are **all defined against one arbitrary
quadrat size**, and nothing in this report says how any of them behave at **0.25° or 1.0°**. ⚠️ The
`opponent` session judges this the most fundable item in its dossier — cheap, and it bears on every
headline number. **Recorded, not acted on.**

✅ **And Møller supplies a rival MECHANISM for our zero inflation** (p. 451): Cox processes model
aggregation *"due to a stochastic environmental heterogeneity"*, distinct from cluster processes where
*"clusters are generated by an unseen point process"*. On that reading, **excess zeros and overdispersion
are consequences of a varying rate and need no separate gate at all.** ⚠️ That mapping is the `opponent`
session's — **Møller nowhere discusses zero inflation or hurdle models.**

⛔ **And Hawkes's impossibility applies to this question too**, which neither session asked: if we ever try
to establish *why* the field clusters — environmental heterogeneity versus contagion — **the two readings
are second-order indistinguishable.** Only a likelihood-based nested fit separates them. That is a
modelling programme, not a metric.

---

### E.3 ✅ THREE INSTRUMENTS THE LITERATURE HAS AND WE BUILT BY HAND

Not contradictions — but each is a published, proven version of something this report derived
expensively, and §8.1's *"wiring an existing metric beats adding one"* applies to the literature too.

| | the literature | what we did instead |
|---|---|---|
| **Miscalibration vs discrimination** | `Gneiting2023_ConditionalCalibration`, **PROVEN** (§2.5–2.6, Thm 2.26): the score decomposition **`S̄ = MCB − DSC + UNC`** for *any* identifiable functional with a consistent scoring function, plus a universal **`R* = DSC/UNC`** nesting classical R². | ⛔ The gate programme spent three days establishing by derivation and measurement that **recalibration moves firing volume and cannot move placement** (`ΔAP = 0.00000` exactly). That is **MCB** moving while **DSC** is fixed. The decomposition was available as a theorem with an estimator. |
| ⛔ **And it answers §0 item 4 on cubes we ALREADY HAVE** | Same decomposition. *"`crps_all` within 0.0007 while gate AP spans 13×"* is, decomposed, the statement that **DSC varies 13-fold while the sum stays flat** — it would show which term carries it. ⛔ **CRPS is not blind on this field; its MEAN is.** (`opponent`'s reading, sharper than §0's.) ⚠️ Neither of us checked whether the decomposition is implemented in `views_evaluation`. | §0 item 4 states the blindness as a property of the metric, not of its mean. |
| **Reliability diagrams** | Same paper, **ARGUED** (§3.1): **CORP** — consistent, optimally binned, reproducible, **PAV-based** — giving *"non-decreasing reliability diagrams with guaranteed nonnegative MCB and DSC components."* | We have `scripts/gate_reliability.py`, a **binned** diagram with bootstrap CIs — which `RESULTS.md §12` records was itself *reinvented* because a "superseded" marker hid it. ⛔ **Binned, not CORP**, so the non-negativity guarantee does not hold for it. |
| **Comparing a point model against a sample model** | `Gneiting2026_FairComparisonAIWPNWP`, **PROVEN**: **PCRPS** — apply EasyUQ (isotonic distributional regression) post hoc to deterministic output, then take mean CRPS. Purpose-built for *"fair comparison of single-valued output"* against ensembles. | ⛔ **§8's row ★4 exists because we got this wrong**: persistence was published **understated by 31%** at h18 by two scoring defects, one of which is S-mismatch between arms and reference. PCRPS is the literature's answer to that exact comparability problem — **and to §8.3's unsolved "both point and sample models" seam.** |

⛔ **`Walz2026_MonotoneDependenceAUC` — see E.0.1. The version of this paragraph that cited it against
AP is RETRACTED.** What survives: **CMA reduces to ROC-AUC for a dichotomous outcome and to ½(ρ+1) for
continuous outcomes** (Thm 2.4, eqs. 17–18), so one coefficient spans a binary gate and a continuous body
— which is the seam §8.3 leaves unsolved — and its mixture representation
`CMA = 6∫α(1−α)·AUC^(α)dα` (Thm 2.5, eq. 20) **integrates discrimination over every dichotomisation
threshold**, where our roster picks one `τ` empirically and the gate conclusion then found that **τ had
silently absorbed a calibration defect**. ⚠️ **It is ROC-AUC throughout, so it speaks to `gate_auc`, not
to AP**, and it says nothing about prevalence.

---

### E.4 ✅ ONE PLACE OUR MEASUREMENT MEETS THE LITERATURE AND OURS STANDS

`Walz2024_EasyUQ`, **PROVEN**: EasyUQ is *"a special case of **isotonic distributional regression** (IDR)
that leverages the **pool-adjacent-violators** algorithm under the stochastic monotonicity constraint."*

⛔ **We tested this and it lost.** The gate programme's Tier-2 remedies compared temperature scaling (R1)
against **isotonic** recalibration on the shipped model and recorded that **isotonic cannot beat R1**
(`RESULTS.md §7`, §10). So on our data, for our gate, the one-parameter method beat the IDR-family method.

**Stated as a complement, not a victory.** It is one model, one target, h=1, and `n = 2` models overall —
far below the bar this report sets for a generalisation. ✅ But it is a case where this record **anticipated
the literature's instrument and measured against it**, and the result is ours to keep.

---

### E.5 ✅ The citation audit — and the report mostly survives it

**2026-10-19.** The `library` session audited the seven sources this report's Appendix C names as
**"unexamined by any reviewer"**. ⚠️ **It could not see the report text**, so it returned *conditional*
drift risks — "if the report says X, note that the paper proves Y". **Those conditions were then checked
against the text here.**

⛔ **Result: four of the seven warnings were ALREADY CAVEATED IN THE REPORT, two of them in the library's
own terms.** That is the finding. This document was more careful with its literature than its own
Appendix C gives it credit for — because a `library` session had already been consulted once and the
record says so (*"Supplied by the `library` session: Bolin2023 proves…"*).

| source | the library's drift warning | state in this report |
|---|---|---|
| **`Gneiting2011_Comparing`** | multiplicative weighting makes a proper score **improper** (Thm 1); **threshold- and quantile-weighted CRPS are proper.** Different operations | ✅ **ALREADY THERE, twice, near-verbatim**: *"Be precise about which weighting — Gneiting2011 proves **multiplicatively** weighted scores improper, while **threshold-weighted and quantile-weighted CRPS are proper**"* |
| **`Bolin2023`** | CRPS is not locally scale invariant; the mechanism is the scale function `σ^{α−2}·s(Q)`, not a plain variance bias. And **skill scores built on proper rules are in general improper** (C-97) | ✅ **ALREADY THERE, both halves**, with the scale function quoted exactly, and the skill-score caution applied to our own `CRPSS` |
| **`Aceituno2025`** | theory derived for **feedforward MLPs only**; extension to RNNs discussed, not proved. And it proves a performance–**learnability trade-off** | ✅ **Scope caveat already present** (*"added 2026-09-30"*). ⚠️ **The trade-off half is NOT**: the landscape roughens exponentially, so long-horizon training is *harder to optimise*, not merely better. **New, and it qualifies §8's row 11.** |
| **`Taillardat2023` / `Brehmer & Strokorb`** | C-1479 **PROVES** expected **weighted** CRPS cannot discriminate different extremal tail behaviours — so citing it *for* CRPS-variant tail evaluation inverts it | ✅ **ALREADY FLAGGED as an overreach** — *"its extracted claim is narrower than the sentence needs: it concerns the expected **weighted** CRPS, not proper-score expectations in general"* |
| **`Ehm2016`** | clean; but the paper separates **quantile** and **expectile** theory, and C-1861's uniqueness result is about expectiles at `α > 1/2` | ✅ **Used precisely** — cited for elementary scoring functions distinguishing forecasters *pointwise* via Murphy diagrams, locating the barrier in the **expectation** |
| **`Lakshminarayanan2017`** | ⛔ C-1475 **explicitly distinguishes deep ensembles (model combination) from BMA (soft model selection)**; deep ensembles beat BMA **under misspecification**. `M=5` is empirical, not theoretical | ⚠️ **The report is careful. `Appendix E` is NOT — see E.5.1.** |
| **`Taillardat2023`, further** | the `T_u` index works **ONLY under calibration** (C-1484) — an uncalibrated extremist can beat the ideal on it | 🆕 **New.** Not in the report, and it is the shape of defect §4 catalogues |

#### E.5.1 ⛔ A CONFLATION I INTRODUCED IN THIS APPENDIX

`Lakshminarayanan2017` C-1475 **explicitly distinguishes deep ensembles from BMA.** ⛔ **`Appendix E.1.1`
reads `Raftery2005`'s BMA results onto our 8-member random-initialisation ensemble**, which the report
discusses under Deep Ensembles. **Treating the two as the same object contradicts both papers.**

**What that does and does not cost.** `Raftery2005`'s *"re-emit the BMA PDF as an unweighted ensemble of
any desired size"* remains a real answer to our matched-S scar, because it is a property of BMA's own
construction and we would be *adopting* BMA, not describing our current pool. ⛔ **But the reading of
M67's 0.96–0.99 error correlation as "a deficiency of the pooling rather than the ensemble" imports a BMA
frame onto a deep ensemble, and the paper that governs deep ensembles says that is a different object.**
Recorded as an open question, not resolved: **does the between/within variance framing apply to a deep
ensemble at all?** Neither session has an answer and the corpus has none.

#### E.5.2 ✅ THE CORRECTION THAT MATTERS MOST: there IS a score-time instrument, and we already cite its author

⛔ **`Appendix E.2.1` states that "there is NO usable evaluation instrument for self-excitation in either
paper." True of those two papers — and I generalised it too far.** The corpus has score-time instruments
for exactly our situation, and they are in a paper this report **already cites, for something else**:

⛔ **CORRECTED 2026-10-20 — I CONFLATED TWO PAPERS, and §3.2 was right all along.** The tail-shape
impossibility §3.2 cites is **Brehmer & Strokorb (2019)**, which is **NOT held in the library**
(verified: the only Brehmer PDFs are the 2023 and 2026 papers). `Brehmer2023` is *"Comparative evaluation
of point process forecasts"* by Brehmer, **Gneiting**, Herrmann, Marzocchi, Schlather and Strokorb — a
**different paper that merely shares an author**. ⛔ **So my "note the inversion" below — that the report
cites this author for an impossibility while his constructive result goes unused — is VOID.** §3.2's
claim that the impossibility paper is not held stands.

✅ **The instrument itself is real and still worth having**, it just is not the paper the report cites:

- **`Brehmer2023_PointProcessForecasts`** (Brehmer, Gneiting, Herrmann, Marzocchi, Schlather, Strokorb) C-3567: the Poisson scoring function
  `S = Σ(−φ(B_i)·log λ_i + λ_i)` is **strictly consistent for CELL EXPECTATIONS regardless of whether the
  process is Poisson.** ⛔ **It needs only the predicted expected count per cell and the observed count —
  no fitted intensity. It runs on a delivered cube.** ⛔ **The "inversion" sentence that stood here is VOID — see the correction above. Two different papers.**
- **`Brehmer2026_EarthquakeForecastEvaluation`** C-3591–3594: **logarithmic Murphy diagrams** for
  cell-level expected counts; **isotonic-regression reliability diagrams on expected counts with no
  parametric assumptions**; and **Diebold–Mariano tests with lag-adjusted variance for temporal
  dependence.** 🔴 **That last one bears directly on §5.6's two irreproducible bootstrap CIs and on the
  origin-block dependence problem** — and it is the state of the art for what CSEP does.

⛔ **What is still absent, and now confirmed against 615 papers.** Compensator- and residual-based
goodness-of-fit (time-rescaling; `Reinhart2018` C-618–622 reviews them) **all require the fitted
conditional intensity `λ*(t|H_t)`, not cell expectations.** A delivered cube gives marginal expected
counts per cell-period; **the self-exciting temporal structure is integrated out and cannot be
reconstructed from the cube.** ✅ So the E.2.1 conclusion survives *for self-excitation specifically*,
and fails for **cell-expectation evaluation**, which is a different and achievable thing.

#### E.5.3 The grid-discretisation argument: NOT in the corpus, searched properly

The `library` session searched **all 615 papers** semantically and by grep — MAUP, ecological fallacy,
change of support, aggregation bias, raster artefact, discretisation + zero. ⛔ **The argument is not
held.** Nearest misses, each explicitly *not* making it: `Browning2024` C-1754 (spatial/temporal
aggregation error framed as a tractability trade-off), `Courgeau2012` (ecological fallacy, for
individual-vs-aggregate inference), and four zero-inflation papers (`Lambert1992`, `Wang2024`,
`Kong2020`, `Xu2023`) that model zero inflation without tracing it to discretisation.

**Named leads, not held:** **Gotway & Young (2002)** on change-of-support for spatial data, and
**Diggle (2003/2013)** on point-process aggregation. ⇒ **This is a fetch, not a search.** Taken with
`E.2.3`'s Møller finding, it is the one literature gap in this appendix with a concrete next action.

#### E.5.4 ⚠️ What this audit still does not do

It is a **content audit against the papers' extracted claims, not a locator audit of our text.** Nobody
has opened a PDF and confirmed that a specific sentence in §1–§8 is supported at the page this report
cites. ⛔ **Appendix C's warning is narrowed, not lifted:** the citations are now checked for *drift of
meaning* and found largely sound; they are **still unchecked for provenance** — which is the failure class
`E.0.1` demonstrated on my own work, same day.

---

### E.6 What this appendix does not do

- ⛔ **It audits none of the report's pre-existing citations.** Bolin2023, Gneiting2011, Taillardat2023,
  Brehmer & Strokorb, Ehm2016, Aceituno2025, Lakshminarayanan2017 remain **unexamined by any reviewer**, and
  the standing warning in Appendix C is unchanged. `Lerch2017`, `Machete2012`, `Cragg1971` and
  `Mullahy1986` were read for §8.3 and for E.1.3.
- ⛔ **It closes none of the four open findings** (6, 7, 8, 10). E.1.3 sharpens row 6 and explicitly leaves
  it open.
- ⚠️ **It establishes no result about our models.** Every entry is either a theorem about a class of
  process or an empirical finding on another dataset. **The two INFERENCE steps are marked where they occur**
  (E.1.1's transfer to a Bernoulli gate, E.2.2's transfer from point patterns to a raster) and neither is
  derived.
- ⛔ **The grid-discretisation challenge is PARTLY answered — see E.2.3, which corrects this appendix's
  first version.** The strong form (*the grid manufactures our zero inflation*) is **not** supported by any
  of the ten. The narrow form is: a cell-constant-intensity model **has no continuum limit**, so its
  parameters estimate no grid-independent quantity. 🔴 **The live consequence is a missing sensitivity
  analysis** — every headline number here is defined against one arbitrary quadrat size.
- **Provenance of the adversarial pass.** §E.0, E.1.2's reading, E.2.1's prereg consequence, E.2.3, and
  the factorization attack in §8.3 come from the `opponent` session's dossier
  (`9_library/opponent/dossiers/collisions_hydranet_evalreport_2026-10-19.md`, commit `fe82cba`), which
  states its own budget: it read §0, §8.2 and §8.3 in full, the headings of §4/§5/§7, and the text layers
  of **seven of the ten** papers — **not** `Gneiting2026`, `Walz2024_EasyUQ` or `Walz2024_Precipitation`,
  because none of the six questions put to it pointed there. ⛔ **So E.1.2 and E.3's PCRPS row rest on my
  reading alone, not on two.** Its closing table records **six hoped-for collisions as ABSENT**.


---

## Appendix F — The 2026-10-19/20 falsification audit, and what it did to this report

**Fifteen independent auditors, one per section, each in a clean context and each denied the author's
reasoning, commit messages and session transcript** — because an author's justification for a sentence is
the thing under test. Each was asked to record a **risky prediction before running every probe**, to open
every cited `file:line` rather than trust it, to extract PDFs rather than trust a citation, and to report
what it did **not** check as explicitly as what it did.

### F.1 The result

| | |
|---|---|
| pieces audited | **15 of 15** (§0–§8, front matter, Appendices A–E) |
| **FALSIFIED** | **14** |
| CONTESTED | 1 (§7) |
| SURVIVED | **0** |
| ⛔ **wrong measurements found** | **ZERO** |

#### The finding count, derived rather than asserted

⚠️ **An earlier version of this appendix said "~70 hard findings" without saying whether that was raw or
distinct, and separately claimed "~45 remaining". Neither was derived. Corrected 2026-10-20** — and the
irony of an un-derived count inside the appendix about un-derived counts is not lost.

| audit | hard | soft |
|---|---:|---:|
| §0 | 6 | 6 |
| §8 (incl. §8.1–8.3) | 10 | 5 |
| Appendix E | 6 | 6 |
| §4 | 4 | 5 |
| Appendix C | 8 | 11 |
| §1 + §3 | 8 | 8 |
| §5 | 9 | 6 |
| front matter, §6, §7, Appendices A/B/D | 19 | 8 |
| §2 | 10 | 8 |
| **RAW TOTAL** | **80** | **63** |

> ✅ **UPDATE 2026-10-20, second pass — the FALSE tier is closed.** 34 flatly-false statements were
> enumerated, verified at their files and corrected, plus 5 second copies. With the first pass's ~32,
> **~66 of the ~70 distinct hard findings are now addressed.** What remains is what Simon's scope
> decision excluded: overstatements, miscounts, expired sentences, incompleteness and reasoning — see
> `F.3`. ⚠️ **And my "~18 false" was an undercount; the enumerated tier was 34.**

**143 findings in total; 80 of them hard.** ⚠️ Roughly **10 of the 80 are the same defect found
independently by different auditors** — *"called by no scorer"* was reported by **four** of them, as was
the `views_hydranet` metric claim — so the **distinct** hard count is **~70**. About **32** were addressed
on 2026-10-20 (`F.2`), leaving **~38**, not the 45 previously stated. ⚠️ `F.3` lists **11 bullets**, each
covering several items; the remainder has never been enumerated one by one, and that is a gap.

#### ⛔ And "wrong claims" is the wrong description. The ~70 by KIND:

| kind | ≈ | what it means | example |
|---|---:|---|---|
| **Flatly false** | 18 | contradicted by the code or the record | *"No forecast metric is computed in `views_hydranet/`"* — it is, and it is **AP**, this report's primary metric |
| **True but overstated** | 15 | holds in a narrower form | *"FG-C never evaluates"* — true of two of its three callers |
| **Counts wrong** | 12 | a number nobody re-derived | 25 dossiers → **26**; 86 tests → **53**; 109 tool files → **164** |
| **Stale — true when written** | 10 | superseded by a dated update nobody re-read | *"a `ruff` binary the environment never installs"* — added since, and the cited entry says so |
| **Quotation / attribution** | 6 findings, ~15 instances | paraphrase inside quotation marks; wrong locator | nine `Appendix E` blockquotes are the library's machine summaries, not the papers' prose |
| **Incomplete, not wrong** | 8 | an inventory or log missing members | `Appendix A` omits the scorer that produced the headline table |
| **Reasoning / strength** | 8 | an inference the evidence does not carry | §1's amendments 2–4; §8.3's retracted degrees-of-freedom argument |

⛔ **So the honest statement is NOT "~70 wrong claims".** It is: **on the order of 18 statements are false
as written**, and roughly 50 more are overstated, stale, miscounted, badly cited or incomplete. ✅ **And
across all 143 findings, not one is a wrong measurement.**

⛔ **That last row is the finding.** Every ratio, percentage, multiplier and verdict count that any
auditor recomputed reproduced **exactly** — `rescore.csv` to **17 significant digits**, the ARTIFACT
board counts to the row, the roster seed table, all 53 dossiers, 350 register entries, 26 scripts, the
`#282` figures *more* precisely than the register's own. **The numbers in this report are sound.**

✅ **And every single defect is in one class: a sentence asserting that something exists, does not exist,
is called, is never called, is classified as X, or that the record says X.** Three of the audits noted
independently that this is **cluster 21 — *"prose with a self-dating expiry: text that was true when
written, named its own expiry condition, and is never re-read when that event lands"*** — the class this
document itself catalogues.

### F.2 What was fixed here (2026-10-20)

| | fix |
|---|---|
| front matter | the mangled *"THREE OF TWO OF THEM"* sentence; the stale `Last revised`; the *"no literature citation audited"* claim; the false *"M9 at every site"*; the postmortem count; the never-closing 17 = 5 + 10 accounting |
| §0 / §5.7 / A.1 | *"no forecast metric in `views_hydranet/`"* — **false** at all three sites, and the metric it misses is this report's primary one; plus the four sibling premises in §0's reproducibility conclusion |
| §2.12 | **two of the four pre-registration counterfactuals withdrawn** — row 3 is a pre-registration *failure* that luck caught, row 4 counts a gate that never ran; the authored sentence inside a quotation |
| §2.3 / §5.2 / §8 ★4 / A.1 / Appendix C | *"called by no scorer"* / *"zero callers"* — **three tools call it**, one of them the tool that wrote this report's headline CSV |
| §2.12 counts | 26 dossiers not 25, ≥24 not 21, 25 not 20 — **every published rate had a wrong denominator** |
| §3 / §8 | cluster 18 has **five** shapes, not four — and the fifth is the counterexample to the sentence §3 calls its most valuable |
| §5.0 | ⛔ **the "empty map" is overstated for the delivered product** — C-345, Tier 1, names this section and says so; §5 cited it zero times |
| §8.3 | ⛔ **the degrees-of-freedom argument RETRACTED** — counterexample verified; the C-345 latent-state claim corrected |
| Appendix E | **nine quotations identified as paraphrase**; my locators were `grep -n` line numbers; the PIT collision dissolved against prior art; the Brehmer conflation |
| counts | 53 tests not 86 (and it skips); 164 tool files not 109; 21 clusters not 20; 24 cluster-18 entries not 23; 2/3 twCRPS sites not 7 |

#### Second pass, 2026-10-20 — the FALSE tier, enumerated and closed

⛔ **Simon's scope decision: the flatly-false statements only.** Overstatements, miscounts,
stale-but-once-true sentences and incompleteness remain in `F.3`.

⚠️ **And a correction to my own framing before the work started:** I quoted the false tier as
**"~18"**. **Enumerating it gave 34** — my 18 was an estimate presented as a count, which is the defect
this whole exercise is about. Simon approved a scope roughly double what he was quoted, with the
enumerated list in front of him.

**34 false statements corrected, plus 5 second copies** found by the plan's own grep step — a claim
corrected at one site and left standing at another is how the first pass left three copies of one
expired sentence. By section: §1 (2) · §2 (5) · §3 (3+2) · §4 (5+1) · §5 (5) · §6 (1) · §7 (2) ·
§8 (5) · App A (3) · App B (2) · App C (2+1).

**The seven that matter most**, each verified at its file before the correction was written:

| the false statement | what the file says |
|---|---|
| §8 ★3 / §5.3 — "FG-C never evaluates" | `check_floor.py:56` passes `mde_ap=MDE_AP`; omitting it returns `PROVISIONAL`, not a silent pass |
| §2.6 — "nine claims (M1–M9) re-dispositioned" | `M2` appears **zero** times in that postmortem — and M2 carries the floored control value the postmortem is about |
| §4.2 — the 89% "was withdrawn" | `RESULTS_LEDGER.md:414` — "**confirmed at n=4**; its pre-registered falsifier did not fire" |
| §4.5 — `recall_mass` "silently failed to execute" | `RESULTS_LEDGER.md:100` — **K1 PASS 8/8**, published. So "two instances make it a class" fails on its own arithmetic |
| §2.12 — the gates "exist **because**" of the truncated_nb defect | all four predate it (07-18, 08-17, 08-18, 08-21 vs 08-24), the quoted justification is a week earlier, and the offending rule was never extracted |
| App B — "see the source harvest" | **no such artifact on any branch** — so 21 of 31 retractions cite nothing, and §6's inference base is unauditable |
| App C row 3 — REFUTED because "two different 13× figures were conflated" | **one figure**: `0.0912/0.0070 = 13.03`, same run, same four arms. Verdict **re-opened** |

⚠️ **Three corrections of corrections**, because the first pass over-reached: §5.1's 2026-10-11 glossary
fix was off by one on every line **and** over-corrected the substance (all three metrics *are*
documented as placement-blind, in one row covering three); §5.4's 2026-10-19 fix left standing the very
sentence it diagnosed; and §8.3's C-345 claim put my paraphrase inside quotation marks.

**Nine of the ~70 distinct findings were introduced by me on 2026-10-19**, the day before the audit — the
`Walz2026` mis-citation, the nine paraphrase-as-quotation, the line-number locators, the PIT collision,
the Brehmer conflation, the degrees-of-freedom argument, the mangled front-matter sentence, the stale
date, and ruling open finding 3 *REFUTED* on a basis that does not hold. **The auditors found all nine. I
would not have.**

### F.3 ⛔ What was NOT fixed — the honest list

**This is debt, not a clean close — but the false tier IS closed.** Re-derived 2026-10-20 from the two
tables rather than carried forward: of the **~70 distinct** hard findings, **~32 were addressed on
2026-10-20 (first pass) and 34 more in the second**, so the ~70 is now accounted for as **~66 addressed**
and the remainder below.

⛔ **What remains is, by Simon's scope decision, everything that was NOT false as written:** ~15
overstatements that hold in a narrower form, ~12 counts needing re-derivation, ~10 sentences that were
true when written, ~8 incompleteness findings, ~8 reasoning findings. ⚠️ **The 11 bullets below group
them and they have still never been enumerated one by one** — the enumeration was done for the false
tier only. That gap is narrower than it was and it is not closed.

- ⛔ **The nine Appendix E blockquotes are LABELLED as paraphrase, not rewritten.** They still sit inside
  quotation marks. Demoting them is a mechanical pass and is not done.
- ⛔ **`Appendix B` is still missing ≥11 retractions**, headed by **the retraction of this report's own
  lead claim**, and its row 2 still quotes the superseded **75.9%** against the shipped **74.6%**. Its
  *"see the source harvest"* pointer resolves to **no artifact on any branch**, so 21 of its 31
  retractions are cited to nothing — which makes **§6's entire inference base unauditable**, and §6
  additionally presents five of Appendix B's nine detector categories as if exhaustive, dropping *"the
  user noticing"*, the largest omitted one.
- ⛔ **`Appendix A` omits the scorer that produced this report's headline table** (`rescore_v2.py`) and
  ~13 further instruments including `mde.py`, behind every `×MDE` bar in the document. `proper_score_audit.py`
  is classified in neither A.1 nor A.2. `gate_reliability` is filed *"superseded or dead"* while §5.6
  holds it up as the working template.
- ⛔ **`Appendix D` claims to close *"most"* of the DRAFT: it accounts for 20 of 43 entries (47%)**, and
  its own counts do not reconcile (D.3 lists eight where the summary says nine).
- ⛔ **`Appendix C`'s open-findings table still reads "unverified" for all six rows adjudicated above it**,
  so a reader following its own instruction gets the pre-adjudication state. It also still asserts, twice,
  that the adversarial pass is *"still owed"* — inside the appendix documenting it.
- ⛔ **M66–M76 are not ledger rows.** The ledger's table rows stop at **M65**; M66–M76 exist only as prose.
  §2.8 and §4.1 cite `M76` as a row. The underlying measurement is sound; the citation is empty.
- ⛔ **§2.3's ARTIFACT verdict has ruled exactly once**, on one archived board, from a gitignored tool —
  268 score CSVs, one carries the zero-share column, none carries a verdict, and nothing in `scripts/`
  imports the module. It is credited under *"instruments that have repeatedly told the truth."*
- ⛔ **§1's amendments 2, 3 and 4 do not hold at the strength stated.** Amendment 4 asks to lift a
  twCRPS-as-diagnostic ban that **does not exist** — three live scripts prescribe it as a diagnostic by
  name — and repeats a mechanism this report already retracted for PIT. Amendment 3, the one labelled
  *"measured rather than argued"*, is contradicted by the §0.1 it cites.
- ⛔ **§7's S16 pools three studies §8 row 7 explicitly separates**, *"because quoting them together is
  the C-298 defect this report is about"* — 37 lines apart.
- ⛔ **§2.7 and §4.1 mix two statistics** (a gap-recovery share with an oracle-level ratio) into one
  *"95–97%"* range, against this repo's own standing rule.
- ⚠️ Four §4 findings, six §5 findings, and the §2 findings H4/H9 are recorded in the audit but not
  reflected in the text.

### F.4 ⚠️ What the audit itself did not establish

- **No auditor opened a PDF except Appendix E's**, so the literature beyond `Appendix E.5`'s seven
  sources is still unverified — and even that was a **content** audit, not a **locator** audit.
- **Where two auditors agreed, nothing re-checked them.** Where two *disagreed* — on whether M66–M76 are
  ledger rows — one was wrong, and the wrong one had already been relayed. ⛔ **So the single-pass
  findings carry single-pass confidence.**
- **One auditor's shell `grep` was `ugrep --ignore-files`, which silently skips the gitignored `reports/`
  tree**; it caught this after three false negatives and re-ran. Any re-run must use `command grep`.
- **Numerators were not re-derived** for §2.12's sub-table, and ~15 findings in the audits are marked
  "unverified in both directions".

### F.5 The one rule this produces

⛔ **Every absence claim in this document needs a `grep` before it ships, and every count needs
re-deriving.** Not one measurement failed; the sentences about what exists failed roughly seventy times.
The remedy is mechanical, and it is cheaper than the audit that found the need for it.
