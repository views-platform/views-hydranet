# What we have learned about evaluating HydraNets

**Date:** 2026-09-30 · **Status:** consolidation report, **no new measurement by this session** (§0.2 and §5.0 carry tables measured 2026-09-30 by the `models` session, attributed there) ·
**Supersedes as the living document:** `reports/2026-08-13_evaluation_pitfalls_and_metric_battery_DRAFT.md`
(retained as dated provenance; its taxonomy letters A1–I4 are preserved here unrenumbered).

**What this is.** Eight months of trying to measure whether a HydraNet is any good, collected in one
place: 53 dossiers, ~340 risk-register entries, 76 numbered claims, 26 measurement scripts, four
postmortems. It is about **the ruler first** — which metrics, baselines, gates and experiment designs
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
(`Var(total)/ΣVar(cell)` ≈ 0.95), so any aggregate interval is too narrow by construction.

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
pattern is not metric-specific: **on a 99.5%-zero field, an all-cell aggregate is a report about the
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

⛔ **And the conclusion this report records the facts for but had not drawn.** No forecast metric is
computed in `views_hydranet/`; the whole ruler lives in `scripts/` and `reports/*/tools/`; `reports/`
is gitignored; 22 tracked tool files carry hardcoded `/home/simon/...` paths; and the governing
framework is a PDF outside git whose six in-repo citations are dangling. Therefore: **the apparatus
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
**What does not:** the free-running rollout throws away **two thirds** of that ceiling; every CRPS win
beyond h1 is **ARTIFACT**; magnitude has no skill (`size_ratio` → 0); six interventions have failed,
four on the same lever; and climatology overtakes on AP around **month 20**.

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
evaluates to zero.** It does not — historical runs report values like 1.03 and 0.11
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
assessed separately. **Amendment: the selection ban is defensible and should stand; the diagnostic ban
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

Instruments that have repeatedly told the truth. Each row names the case where it caught something
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
criticism of CRPS; it is a statement about a 99.5%-zero field.

### 2.3 The zero-share decomposition and the ARTIFACT verdict

`scripts/rollout_ruler_core.py` (Epic #263, 86 never-skipping tests). For a claimed win it reports
CRPSS, ΔAP, and **what fraction of the CRPS gap is true zeros**, then rules REAL / UNDECIDABLE /
ARTIFACT against a pre-registered rule.

This is the instrument that produced §0.1. It is the only thing in the repo that can look at a
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
  (M1–M9) had to be individually re-dispositioned, and one manipulation read **+0.9%** on the floored
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
- **M76:** four roster models retrained from the **PyPI wheel** are byte-identical to checkout-trained
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

Of the 25 dossiers dated after 2026-08-13: **21 pre-registered a falsifier list**, 20 ran at least one
scored arm.

| | count |
|---|---|
| a pre-registered **named falsifier actually fired** on real data | **9 of 20** |
| at least one gate fired **or** a run was declared VOID | **13 of 20** |
| a pre-registered **prediction was falsified** — the verdict contradicted the dossier's own stated expectation | **9 of 20** |
| the verdict was **not** the one the dossier hoped for (negative / null / VOID / underpowered) | **15 of 20** |

⛔ **The decisive evidence that it is not decorative** is the counterfactual: in four cases a falsifier
voided an arm and the record states the *specific wrong claim* that would otherwise have been written.

| arm | what fired | what would have been published |
|---|---|---|
| `shuffle_months` | F6 — the intervention moved its own target statistic only 5% | *"persistence matters somewhat"* — its −0.0415 "reads as a plausible middling effect" |
| pushforward w=0.1 | F5 **and** F6 — the oracle fell, so the term damaged the **model** | *"pushforward is bad"* from its −0.10 AP@h18 — **"a different and wrong claim"** |
| BPTT-SA SCREEN-1 | the weight-hash post-condition — two independently trained arms, **byte-identical** | *"BPTT-SA does not work here"* |
| roster K3 | the comparator produced no scores | a **PASS** — instead, *"the roster has not been checked against a baseline"* |

And the discipline held under pressure, which is the part that matters: *"F1 was amended, because its
instrument was demonstrably wrong. F5 was not, because it was doing exactly its job. **Keeping that
distinction under pressure is the difference between a falsifier and a formality.**"*

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
`EFFECT (NEGATIVE)` branch was **unreachable**. The response — md5-pinned, unit-tested,
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

> **Cluster 18 — "The instrument did not measure what it claims."** 23 Tier-2/3 entries, *"every one
> of which cost GPU time or produced a wrong ledger row, currently read as 23 unrelated incidents."*
> The register calls it **the highest-value group it has**.

And its proposed remedy is one question, asked before the run:

> ⛔ **"What would this measurement look like if the effect were absent?"**

Three of cluster 18's four shapes fail that question. That single line is the most valuable sentence in
this report.

### 3.1 The four shapes of cluster 18

| shape | the defect | worst instance |
|---|---|---|
| **(a) the instrument cannot see the thing** | the statistic cannot move when the thing under test moves | **C-319** — rolling the clamp anchor 90 cells leaves occurrence 0.718/0.715, magnitude 0.628/0.651, alignment 61.5/69.3 essentially unchanged while **AP@h18 falls 0.362 → 0.0075 (48×)**. Three statistics, all blind. (↔ DRAFT A1, A2) |
| **(b) the decision rule cannot deliver a verdict** | branches unreachable or the outcome space omits "no number" | **C-303 occ. 9** — a one-sided permutation p made the EFFECT branch unreachable for a negative Δ, so the rule printed **"NULL / UNDERPOWERED, p = 1.0000" for a mean ΔAP of −0.2376 — 7.5× the 3×MDE bar, 4/4 seeds unanimous.** (↔ DRAFT C2, the power-vs-null pattern) |
| **(c) measured on the wrong object, phase or training state** | correct arithmetic, wrong subject | **C-308** — a drift probe kept the *first six* forward calls, i.e. the teacher-forced warm-up, not the 36 free-running steps. Published σ_max **1.60**; true **7.7628**. It reported `max\|h\|` **rising** 0.000 → 2.867 where the real phase **collapses 40× (65.6 → 1.6)**. *"Every downstream guard passed."* (↔ DRAFT B1, B2, the wrong-object pattern) |
| **(d) the readout assembled wrongly** | well-formed number, different population than its label | **C-298** — 3 of 9 headline figures wrong: *"the results grid here is four-dimensional and the prose is one-dimensional, so every sentence silently collapses three axes."* (↔ DRAFT C7) |

### 3.2 The metrics themselves

⛔ **`crps_all` on a 99.5%-zero field ranks the zeros, not the model.** Four arms inside **0.0007** at
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
magnitude and gate–body alignment are all preserved by a spatial roll that costs 48× in AP (C-319,
M53). `GLOSSARY.md` now carries this as a ⛔ annotation on three separate metric rows.

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
⛔ **C-328 instance 4 is in every shipped artifact:** `random_flips` defaults True and reaches the
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
repo now rejects. Conclusion, verbatim: **"There is no valid, non-floored SS measurement anywhere."**

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
to `ruff`, a binary the environment never installs. Red on **every CI run since at least 2026-08-14**:
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
| Is it recall or placement? | **Placement.** `thin:0.75` deletes ¾ of true events and keeps 95–97%; scrambling locations costs 31× | M3/M4 → M11/M12 | replicated across two vehicles |
| Does more training fix the rollout? | **No — training is closed as a retention lever.** Retention 0.03 → 0.600 (160L) → 0.683 (300L, n=4) → 0.692 (600L). T=0 skill and the ceiling keep rising | M26–M29 | shape from 5 lesson counts; the plateau itself is a 2-point claim |
| Does scheduled sampling help? | **No — it hurts, at every horizon** | M30–M33, 4v4 seeds, p=0.0286 | formally UNDERPOWERED (does not clear 3×MDE); direction established, magnitude not |
| Does the model beat persistence at long horizons on AP? | **Yes at L=300, all four seeds, every horizon** | M34–M37 | one vehicle, AP only; the CRPS win remains ARTIFACT |
| Does pairing buy power? | **Yes — MDE 0.0541 → 0.0086, 6.3× on identical data** | M40 | one horizon, two arms, one seed |
| Is the packaging boundary a source of drift? | **No — byte-identical, 190 tensors × 4 models** | M76 | this machine |
| Can a multi-τ roster beat the soft gate? | **No — algebraically impossible** | M69 | exact; zero GPU |
| Is the ensemble buying diversity? | **No — it is buying sample count.** Gate errors correlate 0.96–0.99 across all 28 pairs | M66, M67 | one artifact per config, one origin, `sb` |

### 4.2 Closed — and the closure is what should be read, not the number

| question | what is closed | what is **not** |
|---|---|---|
| Does freezing the cell state help? | **Direction: yes**, four seeds, paired CI excludes zero, +0.039 AP@h18 (M38/M48) | **Mechanism: no.** M43 refuted the out-of-distribution-state hypothesis; M52 is explicitly "CONSISTENT-WITH, not causal". And M8's original 13× magnitude is **retired** — the control was broken |
| Which half of the recurrent state matters? | The **cell** (M39, M56) | C-292: `hs = o⊙tanh(hl)` makes `cell ≈ all` **architecturally predetermined**, and C-323 makes the hidden-freeze null structurally guaranteed. The earlier "cell carries 89%" was withdrawn |
| Is the amount ceiling real? | **Yes** — spearman 0.303 < persistence 0.367, confound-clean | It is a statement about *magnitude*. Jump-risk/volatility **is** predictable (spearman ≈0.79) |

### 4.3 Not solved — and the reason each is stuck

| # | Open question | Why it is stuck | issue |
|---|---|---|---|
| ⛔ 1 | **Is the h=1 win real?** It is the only non-ARTIFACT signal in the entire programme, and it reads **UNDECIDABLE for a purely procedural reason** — no bootstrap CI was computed. Both point estimates say REAL (CRPSS +0.197, ΔAP +0.152) | needs a re-run emitting **per-origin** scores at matched S; the cubes were score-then-deleted | — |
| ⛔ 2 | **Which ledger rows rest on floor-limited vehicles?** M8's best case on the smoke vehicle (0.0912) is **3.6× below** an untreated 300-lesson baseline (0.3298) | the audit was only ever run on the rollout-collapse programme; the postmortem states that "this is the only floored measurement in the repo" is **not established** | **#280** |
| ⛔ 3 | **Can the design resolve the effects we chase at all?** SS: −0.0426 with MDE 0.0541. *"If placement interventions land in the same 0.03–0.05 band, every future experiment returns UNDERPOWERED"* | one of four candidate fixes (a paired statistic) changes the MDE at **zero GPU** and has not been adopted generally | **#281** |
| ⛔ 4 | **Is the truth data correct?** priogrid 149451 reads **27,413** then **32,505** with a **2** between them; the next largest cell-month anywhere is **1,100** | nothing checks the truth field for outliers at build time; filed upstream | C-338, views-datafactory#484 |
| ⚠️ 5 | **Is D=4 enough gate variation for a delivered point estimate?** The gate is deterministic given a pass's mask, so D is the only lever; on `blazing_meteor` the gate kills 1–3 of 4 passes in **407 of 502** signal-carrying cells | "no ledger entry answers it" | **#388** |
| ⚠️ 6 | **Is `aggregate_method` offering the wrong collapses?** `gate × mu` beats mean-of-16 by **+58% to +159%**; `median` is an empty map | indicative only — one artifact per config, one origin, `sb` | **#337** |
| ⚠️ 7 | **Was any scheduled-sampling verdict measuring anything?** All four SS-on roster models trained under a config the repo now rejects, and *"there is no valid, non-floored SS measurement anywhere"* | and #288 asks whether the SS feedback path is differentiable at all | **#295, #288** |
| ⚠️ 8 | **Do the direct-multi-horizon and recursive cubes score identically?** | *"Prove the ruler scores both cube types identically"* is a precondition nobody has met | **#310, #324** |
| ⚠️ 9 | **Is the architecture bake-off attributable?** DualStream replaces the identity top-skip; WideMemory silently drops `state_channels`; MaxBlurPool2d needs an 8×8 grid and dies at the repo's own `window_dim=4`; arms matched on parameters but **not compute** (+5% to +20% wall-clock) | four open issues, no re-run | **#366–#369** |
| ⛔ 12 | **Does AP@h18 predict anything about the served product?** ⚠️ **PARTLY ANSWERED 2026-09-30 and the answer is bad:** across seven collapses of one cube, **AP spans 18.9×**, and the delivered `tower_point` is **last**. So the optimisation target and the deliverable are connected by a transform with a larger effect than most interventions we have tested, and until 2026-09-30 nobody had scored it. **Still open:** whether it holds at the delivered 128 draws, and whether it holds on the pooled ensemble | the 128-draw repeat is queued on the `models` side; AP is not computable on the forecast months (no truth), so it needs **pooled validation** draws — which do not exist on one machine today | **#337, #388** |
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
  the register does not flag it.
- **Six of fifteen register "disagreements" have never been curated** — *"not one carries a resolution
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

**(a) Nothing scores the served surface.** Every instrument in Appendix A takes a sample cube and
returns AP, CRPS or Brier. The FAO receives a **collapsed point surface** plus HDI bounds. The collapse
is not a formatting step — M70 measured that `gate × mu` beats the mean of 16 draws by **+58% to +159%**
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
90-cell roll that costs **48× in AP** (C-319; `GLOSSARY.md:157` carries it as a ⛔ on three metric
rows). And **no test plants a rolled field and asserts that any metric fires.** The only artifact is a
one-off cross-correlation script in one dossier.

So three glossary-locked field statistics are *documented* placement-blind and *unguarded* against it.
A future arm that improves all three and destroys placement would be reported as an improvement.

⚠️ **And it is a platform-wide hole, not a hydranet one.** Reported by the `views-evaluation` session:
the platform metric catalogue carries **Variogram** and **pEMDiv** — pEMDiv being the metric built for
exactly this problem — and both are `implemented=False` **stubs**. So no repo on this platform can
currently score spatial displacement, and the two candidate instruments are unbuilt rather than absent.
They have offered to implement them there rather than in `scripts/`, which is the right home.

### 5.2 ⛔ The method rule that no scorer enforces

*The reference's S must equal the arms' cube width.* It is stated in one module
(`rollout_ruler_core.reference_sample_width`) and **called by no scorer.** Live violators:
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

### 5.3 ⛔ FG-C — the floor-gate clause that matters — does not run

`postmortem_floor_limited_vehicle.md:175` names FG-C as *"the one that names the real failure … not
'the model is bad' but 'the effect we came to measure is smaller than what this setup can resolve'."*
Both gates that call the floor gate — `lesson_curve_gate` and `ss_sweep_gate` — call it **without
`mde_ap`**, so **FG-C never evaluates on either path**. And FG-B (worse than climatology) is
**advisory and never binding**: a control worse than climatology still PASSes.

Separately, `check_floor.py` records that **no dossier invoked the floor gate between 2026-08-26 and
2026-09-04** — the C-299 mitigation lapsed for over a week, silently.

### 5.4 ⚠️ The red-team exists, and is pointed at the wrong target

Better than expected: **all seven** degenerate forecasts from the DRAFT's §5 proposal are implemented —
all-zero, climatology, persistence (partial), over-concentrated spiky, frequency-right,
magnitude-right, and a known-good oracle — in `tests/test_activation_metrics.py` and
`tests/test_rollout_ruler_core.py`.

But: **every one of them exercises `activation_metrics` in isolation.** Nothing feeds a degenerate cube
through `score_v2_horizons._metric_row` or `lodestar_score.score_models` and asserts that the emitted
**row** separates the cases. The DRAFT's own TODO — *"tabulate which metric separates which degenerate
— that table is the paper's Figure 1"* — is **not done**. And the spiky case is tested only on a
**uniform** magnitude field; no test feeds one 99,465-value cell, which is the shape that would leave
`size_ratio` (a median) unmoved and `mcr_all` at ≈1.

### 5.5 ⚠️ We audit whether we measured the model right, never whether the target is right

The register says this about itself, verbatim: *"The register has extensive coverage of whether we
measure the model correctly, and **none** of whether the thing we measure against is correct."*
Consequences on record: C-338 (an implausible truth outlier, filed upstream, unchecked at build time)
and **#282** — a baseline defect that **scales as 1/n_origins**, so a single-origin study scores
persistence as **exactly zero, silently**.

### 5.6 ⚠️ Two bootstrap CIs in the apparatus are not reproducible

`mcr_readout` and `proper_score_audit` derive their RNG salt from `hash(str)`, which Python randomises
per process, and no `PYTHONHASHSEED` is set anywhere in the repo. So their confidence intervals and
randomized-PIT numbers **differ run to run**. `gate_reliability` fixed exactly this and says so in a
comment — a stable per-target seed instead of `hash()` — and **the fix was never back-ported to its two
siblings. That is a five-minute change**, priced here because §8 prices everything else. This matters because the reproducibility runbook nominates
`mcr_readout` as the human-readable determinism cross-check.

### 5.7 ⚠️ The ruler is not part of the package, and the protocol is not in the repo

**No forecast metric is computed anywhere in `views_hydranet/`.** The entire ruler lives in `scripts/`
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

## §6 — Where the unknown unknowns are

⚠️ **This section is INFERENCE throughout.** It reasons from where past surprises came from, to where
the next ones probably are. Nothing here is measured.

**The inference base:** of 31 traceable retractions, the detectors were an adversarial second pass (9),
two numbers disagreeing (3), a control arm (2), an impossible result (1), and the test suite (**0**). So
the unknowns are concentrated wherever **no second computation of the same quantity exists**.

| where | why I think it lurks | the cheapest probe |
|---|---|---|
| ⚠️ **The forecast path — DOWNGRADED 2026-09-30, it has now run** | `extrapolate_time` is forecast-only code and had produced **zero output ever** when this row was written. The `models` session reports a 300-lesson global-land forecasting run in flight on a rented pod — three models complete, `predictions_forecasting_*` and `forecasting_log.txt` present, 2,333,448 rows × 128 pooled draws, months 561–596, 64,818 cells — and a 40-lesson rehearsal that completed the whole chain and is being served. ⚠️ **Their measurement, not mine:** nothing on this machine holds forecast output but pytest fixtures, so I cannot verify it here. **The row is now "exercised once" rather than "never run", and its priority drops accordingly** — but one run is one run, and §5.0 is what that run actually exposed. | Score the surface it produced. That is §5.0, not this row. |
| ⛔ **The truth field** | §5.5. One outlier was found by eye, upstream, and filed. Nothing systematic has ever looked. A corrupt truth cell is invisible to every instrument we own, because every instrument scores *against* it. | An outlier scan at build time: any cell-month above the historical max by >10×. |
| ⛔ **Every negative result whose implementation nobody audited** | The September postmortem names this its *"most expensive open question"*: ≥15 defects found *after* the author judged the code correct, **0 by his own review**, and C-324 proves an implementation can be fully green and **completely inert**. The audited experiments are #308 and #311. Every earlier negative — and there are six failed interventions — rests on an unaudited implementation. | A potency check at a trained checkpoint on each retired arm. Cheap; no retraining. |
| ⚠️ **The in-package artifact, not the scorer** | **C-328 instance 4 is already in production:** `random_flips` defaults True and reaches BN recalibration, so ~half of every BN running statistic in every shipped artifact was accumulated on flipped fields. That is a whole class — *"which non-training forwards traverse this path?"* — and it was asked once, late. | Enumerate every forward pass that is not a training step, and what augmentation each sees. |
| ⚠️ **The collapse from cube to delivered number** | M70: `gate × mu` beats mean-of-16-draws by **58–159%**, and `median` is an empty map. The collapse step has effects larger than most interventions we tested and the least scrutiny of any stage. #388 asks whether D=4 is enough and *"no ledger entry answers it."* | Score all available collapses on one existing cube. Zero GPU. |
| ⚠️ **Anything measured once, by one instrument** | The three retractions caught by "two numbers disagreeing" only happened because a second computation existed. Where only one exists, the class is silent by construction: `recall_mass` (the production roster's own K1 criterion), `taillardat_index`, `hard_gate_threshold`. | For each single-source number that gates a decision, compute it a second way. |
| ⚠️ **The CI signal itself** | C-283: three tests shelled out to a binary the environment never installs, and CI was **red on every run for ≥2 weeks** — *"every CI `test` result before 2026-08-15 should be read as red for this reason unless shown otherwise."* A gate can be broken for reasons wholly unrelated to the code, and stay broken. | Assert the CI job's own preconditions (every external binary it invokes exists). |
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
| S9 | The posterior mean is the natural point estimate | **M70 — `gate × mu` beats the mean of 16 draws by +58% to +159%** at identical total mass; `median` is an **empty map** (0.05–0.14% of cells non-zero). | ⚠️ How you collapse the cube is a first-class modelling decision with effects larger than most interventions we tested. |
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
can choose. A list of fifteen enforced gates would be the 303 → 1,167-line validator this repo already
warns about.

⚠️ The `views-evaluation` session asked whether the table would be stronger as **those four, with the
eleven in an appendix.** It would be, for a reader choosing what to adopt; it would be weaker as a record
of what bit us, which is what the scar rule is for. **Compromise: the four carry a ★. The eleven stay in
place and keep their scars.** If this ever becomes a proposal rather than a record, cut it to the four.

Whoever owns the promotion protocol can compare this against it. I make no claim about what the
protocol currently says.

| # | Requirement | The scar that earns it | Cost to adopt |
|---|---|---|---|
| ⛔ ★1 | **Ask, before the run: "what would this measurement look like if the effect were absent?"** Record the answer. | Cluster 18 — **23 Tier-2/3 entries**, each of which cost GPU time or produced a wrong ledger row, and which the register read as 23 unrelated incidents until 2026-09-10. Three of its four shapes fail this one question. | One line in a pre-registration. |
| ⛔ 2 | **A reference forecast on every scored run — persistence *and* climatology — with the zero-share decomposition and the ARTIFACT verdict applied automatically.** No result may be reported against `none`. | §0.1: a forecast with `size_ratio` 0.0000 cleared the 5% bar on the primary metric, and 74.6% of the win was confident zeros. ⚠️ It would **not** have been promoted — the MCR guardrail vetoes it (corrected 2026-10-11) — but nothing computed the decomposition, and the ARTIFACT verdict stood unexamined **16 days**. This is the ledger's own recommendation #3 and it is **not implemented**. | Structural, cheap; the instruments exist (`rollout_ruler_core`, `climatology_resample`). |
| ⛔ ★3 | **A floor gate on the CONTROL arm before any treatment arm runs: FG-A `AP_ctrl ≥ 5 × prevalence`, and FG-C `(1−θ)·AP_ctrl ≥ 3 × MDE`.** | A 6-GPU-hour pre-registered sweep with falsifiers was **VOID as run**; 9 ledger rows and 5 inferences re-scoped, 3 claims retired. One manipulation read **+0.9%** on the floored vehicle and **−93.7%** on a real one. The condition was visible in a CSV **5 h 47 m before the last arm finished.** | Zero GPU beyond the control. ⛔ **And fix the existing gate: both callers omit `mde_ap`, so FG-C never evaluates.** |
| ⛔ ★4 | **Match S between arms and reference, enforced in the scorer, and refuse an AP comparison across gated/ungated arms.** | Persistence was published **understated by 31%** at h18 by two independent scoring defects. #282's share of it **scales as 1/n_origins**, so a single-origin study scores persistence as exactly zero, silently. | The check exists (`reference_sample_width`, `assert_sample_cube`) and **no scorer calls it**. |
| ⛔ 5 | **Every metric that gates a decision must have a degenerate-forecast test that fires — run against the HEADLINE scorer, not the metric function in isolation.** Include the spatial roll and the single-huge-cell. | The roll: three field statistics unchanged, **AP 48× worse**, and **no test guards it**. The seven degenerates that *are* implemented all test `activation_metrics` in isolation; nothing feeds a degenerate cube through the scorer that produces the published row. | Days, not weeks. Synthetic fields, no GPU. |
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
- **Repo-wide gates.** The register's structural finding is that *"no gate in this repo is repo-wide;
  each is opted into per launcher, and a dossier that forgets one gets no warning."* That is a real
  problem — but the demonstrated failure mode of centralising is C-283, where one broken shared check
  turned CI red for two weeks and nobody noticed.

### 8.2 The two measurements, and which is which

⚠️ **The ordering, made explicit because a reader may be able to fund exactly one.** The **larger
finding** is §5.0 — the delivered product is unscored and the delivered collapse measured worst of
seven. The **cheapest decisive** measurement is the h=1 question below. They are not the same thing and
the report previously implied the second was the more important.

**Fund §5.0 first if you can fund one:** the 128-draw repeat of the collapse comparison, and an aggregate
coverage number. Both are zero-GPU on an existing cube, and together they decide whether the partner is
being served the worst available collapse of a distribution that is also mis-calibrated. That is a
delivery finding, and delivery beats research here.

**The h=1 question is the cheapest thing that could change the research story:**

⛔ **Establish whether the h=1 win is real.** It is the only non-ARTIFACT signal in the entire
programme. It currently reads **UNDECIDABLE for a purely procedural reason** — no bootstrap CI was
computed — while both point estimates say REAL (CRPSS +0.197, ΔAP +0.152). It needs a re-run emitting
**per-origin** scores at matched S, because the cubes were score-then-deleted.

It is cheap and it is decisive in both directions: it either establishes this project's first genuine
win over a naive baseline, or it removes the last one.

---

## Appendix A — The instrument inventory

**Where the ruler lives.** ⚠️ **No forecast metric is computed anywhere in `views_hydranet/`.** The
entire apparatus is `scripts/` (26 files, 5,460 lines) plus `reports/*/tools/` (109 files), reading
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

### A.2 Superseded or dead

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
precision on a group with no positive truth from `0.0` to `nan`. On a 99.5%-zero field *"this group has no
positives"* is the **ordinary case**, so a mean over groups would silently become a mean over fewer
groups — and AP is this repo's primary metric. **It does not touch us:** verified that
`calculate_ap_native` is imported nowhere here; AP comes from the frozen `lodestar_score.average_precision`
or from `sklearn`. ⚠️ Note what saved us was the frozen ruler's independence, not a pin or a test — and
`rollout_ruler_core.py:398` records that our own implementation already returns NaN on an all-zero label
vector, i.e. the two converged on the same semantics by accident.

⚠️ **twCRPS is computed here, in seven places, and the report should say so plainly** because a reader
could otherwise conclude a banned metric is live in the decision path. `calculate_twcrps_native` is
imported at **7 sites across `mcr_readout.py` and `proper_score_audit.py`** — both classified SUPERSEDED
or display-only in A.2, and both documenting that FAO-02 bars it from selection. So: **computed, never
selected on.** That is the correct disposition and it is the §4.5 template being followed.

⚠️ **One metric block copied verbatim into three files**, each with a docstring saying "they must match
the audit tool". **Nothing tests that they still match.**

---

## Appendix B — The retraction log

31 traceable retractions or overturns, 2026-05 → 2026-09. **≥14 concern a published ledger row or a
delivered verdict.** Two independent internal estimates of the base rate exist and both are lower
bounds: a systematic recompute found **3 of 9 headline figures (33%)** wrong or misleading; a deletion
test found **27 of 40 tests (68%)** passing with the call site removed.

Full table with detection mechanism and how long each stood: see the source harvest. The ten most
consequential:

| claim announced | what it actually was | caught by | stood |
|---|---|---|---|
| "coords made it worse" + the FAO eligibility table + RESULTS_LOG deltas | the baseline alone swings **2.99–3.69** run-to-run from init-RNG drift | a **two-run parity gate** before a refactor | weeks |
| **gated_NB beats climatology at h36** | **ARTIFACT** — 75.9% of the gap is `crps_none`, AP is *worse*, `size_ratio` 0.0000 | Epic #263, exact decomposition of archived CSVs, no GPU | **16 days** |
| `gated_ZINBcore` "structurally dead, 5–15× worse" — arm retired | the frozen ruler's `crps_all` is **gate-independent**, so it scored the body **ungated**; re-tested, **~6× better** and viable | a re-test | 5 days |
| "it is the long-term memory — **cell carries 89%**" | `hs = o⊙tanh(hl)` makes `cell ≈ all` architecturally predetermined | `/code-review medium` | ~1 day |
| "the gate diffuses"; "the model needs to answer" | properties of a floored smoke vehicle promoted to properties of the model; one ratio's denominator held **two cells** | reading an unrelated CSV for an unrelated question (**luck**) | 3 days |
| 3 of 9 feedback-realism headline figures | cells of a 4-D grid quoted as single comparisons; two had already propagated into the register | the Claims-Ledger verification pass | 1 day |
| **`truncated_nb` fixes the #258 double zero** (it was built for it) | at L=300, 4 seeds: ΔAP@h18 −0.219 to −0.256, **7.5× the 3×MDE bar**. The double-counted zero was an accidental **brake** | re-running on a converged vehicle after the floored one was voided | **10 days** |
| σ_max = 1.60 and "the first direct observation" of rising state drift | the probe sampled the **teacher-forced warm-up**; true σ_max **7.7628**, and the real phase **collapses 40×**. *"Every downstream guard passed"* | `/code-review medium`, incidentally | ~1 day |
| BPTT-SA "unit tested (7), mutation tested 5/5, suite green (1,901)" | **inert on the production path**; two arms trained to byte-identical weights | **the result being impossible** — named as luck | 276 GPU-min |
| C-58's forecast-sniffer remedy | made **every forecasting run impossible**; no forecasting run had ever completed in this repository | a fresh look, #390 | **months** |

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
the 20 causal clusters, the §Disagreements section) · the four postmortems · `reports/GLOSSARY.md` ·
`reports/2026-08-13_evaluation_pitfalls_and_metric_battery_DRAFT.md` · the 53 dossiers, prioritising the
25 dated after 2026-08-13 · all 26 `scripts/` and the 109 dossier `tools/` · the evaluation-contract
tests · GitHub issues, open and closed.

**What is verified here.** Every figure in this report was traced to a `file:line`, an M-number, a
C-number or an issue. Nine headline figures were spot-checked back to their source file. Two
load-bearing structural claims were verified independently of the harvest: cluster 18's pre-flight
question is quoted verbatim from `technical_risk_register.md:75`, and `assert_sample_cube` /
`reference_sample_width` have **zero callers** outside their own module and tests.

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
truth from `0.0` to `nan` on 2026-09-18. AP is this report's primary metric and on a 99.5%-zero field that
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

⚠️ **Not yet dispositioned (2026-10-11).** Ten of the seventeen findings are unverified and unfixed, and
four more are verified and unfixed: §0's *"every CRPS win beyond h1 is ARTIFACT"* (false — `rescore.csv`
has REAL verdicts at h6 5/12, h12 3/12, h18 2/12, h24 1/12); §0 item 2's *"any aggregate interval is too
narrow by construction"* (§5.0(b) retracts it, in our favour, and both statements are still in the
document); §5.1's *"three separate metric rows"* in `GLOSSARY.md` (there is **one**); and §4.1's
*"**Gate** errors correlate 0.96–0.99"* (the M67 operand error, third instance, after the §7 fix checked
§0 and not §4). The reviewer also flagged four of its own findings as reasoning rather than citation
mismatches, and audited **no** literature citation. **This appendix's provenance claims should be read as
weaker than they were before this pass, not stronger.**

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
narrative drift lives."* This document has not itself been through the adversarial pass that §8.8
identifies as the highest-yield detector in the record. **It should be.**

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
