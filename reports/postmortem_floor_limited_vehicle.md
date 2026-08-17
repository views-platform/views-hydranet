# Post-Mortem — Three Days of Experiments Ran on a Vehicle With No Dynamic Range (C-299)

**Date:** 2026-08-17
**Companion to:** risk register C-299 (new), C-300 (new), C-293 (collapsed *reference* — the sibling defect), C-298 (cross-cell figures); dossiers `2026-08-14_scheduled_sampling`, `2026-08-15_state_freeze`, `2026-08-16_feedback_realism`, `2026-08-17_vehicle_replication`, `2026-08-17_placement_intervention`; `reports/RESULTS_LEDGER.md` §Claims Ledger
**Status:** **ROOT CAUSE FOUND + PREVENTIVE GATE DESIGNED AND VALIDATED.** The gate is not yet wired into any driver. One correctly-designed experiment is **VOID**; nine ledger rows and five inference rows are re-scoped; three claims are retired.
**Method:** read-only re-analysis of committed score CSVs, plus one controlled replication on a second vehicle (`2026-08-17_vehicle_replication_dossier`) and a six-model roster probe (`2026-08-17_placement_intervention_dossier` EXP-03). No new training.

---

## 1. How it surfaced (the trigger)

Not from a failure. Everything ran green for three days.

It surfaced while reading `reports/2026-08-15_rollout_ruler_trust_dossier/results/rescore.csv` to answer a
different question ("is the h1 win over persistence real"). That file already contained `violet_visitor`
scored against climatology with an origin-block CI at every horizon — and its rollout numbers were nothing
like the ones I had been working with:

| sb gate AP, free-running, identical setup | h1 | h6 | h18 | h36 |
|---|--:|--:|--:|--:|
| `violet_visitor` (160 lessons, `nb`) | 0.4745 | 0.3924 | 0.2569 | 0.1370 |
| `climatology` | 0.2980 | 0.2620 | 0.2251 | 0.1667 |
| **`truncated_smoke`** (40 lessons, `truncated_nb`) | **0.2979** | **0.0284** | **0.0070** | **0.0083** |

`truncated_smoke`'s h1 AP equals climatology's to four decimals — checked, and a genuine coincidence at
4 s.f., not a bug. It has **no occurrence skill at any horizon**, and it collapses 42× by h18. Its own
config says so: `total_lessons: 40,  # SMOKE (not a scored result)`.

Every arm of the state-freeze probe (#277) and all thirteen feedback-realism arms (#278) ran on it.

## 2. The suspect space

The obvious reading was "the smoke vehicle is just worse, so the effects are smaller but the ordering
holds." That is wrong in a specific and asymmetric way, and the asymmetry is the whole defect:

- a **null** measured on a control at the floor is *uninformative* — nothing can move a number already at
  zero, and the measurement cannot distinguish "no effect" from "no resolution";
- a **positive** measured against a control at the floor is *understated* — a degradation arm cannot fall
  below a control that has already fallen.

Neither shows up as an error. Both produce well-formed CSVs, plausible tables and green falsifiers.

## 3. The decisive evidence

`spatial_scramble` feeds back a perfect field with its **locations permuted** — active count and
magnitudes held byte-identical (verified: `active_fraction` differs by 0.00e+00). The only thing that
changes between the two rows below is the vehicle:

| h18, target `sb` | oracle | control | scrambled | scramble's share of the gap |
|---|--:|--:|--:|--:|
| `truncated_smoke` | 0.3008 | 0.0070 | 0.0097 | **+0.9%** |
| `violet_visitor` | 0.4793 | 0.2569 | 0.0486 | **−93.7%** |

On smoke, scrambling read as *marginally helpful*. On a vehicle with skill it is **catastrophic** — five
times worse than feeding the model its own flawed output. The +0.9% was never a measurement of placement's
importance; it was the distance between two numbers both pinned near zero.

**This points at the mechanism precisely:** the failure is not that the smoke model is bad. It is that its
control sat **below the random-ranking floor**, so a downward-acting intervention had nowhere to go.

## 4. What it cost — and the experiment it voided

The expensive casualty is not a misread figure. It is `reports/2026-08-14_scheduled_sampling_dossier/`,
which is a **correctly-designed experiment**: one variable (`ss_epsilon_max`), seed fixed at 42, four
doses, everything else pinned, pre-registered with falsifiers. Six hours of GPU.

| arm | AP h1 | AP h18 | retention | AP h18 ÷ prevalence |
|---|--:|--:|--:|--:|
| ε=0.0 | 0.2979 | 0.0070 | 0.024 | **0.77×** |
| ε=0.1 | 0.2851 | 0.0070 | 0.025 | **0.77×** |
| ε=0.25 | 0.2779 | 0.0095 | 0.034 | **1.05×** |
| ε=0.5 | 0.2039 | 0.0089 | 0.043 | **0.98×** |

Prevalence at h18 is 1547/170430 = 0.009077. **Three of the four arms score at or below random ranking**,
and all four sit inside a band narrower than the noise. The experiment could not have discriminated
between any hypothesis and any other. It was unanswerable before it started.

**And the condition was visible in the control's own score CSV 5 h 47 min before the sweep finished.**
`results/score_eps0.0.csv` has mtime 04:51:10; the last arm completed at 10:38:29.

## 5. Biases weighed (intellectual-honesty audit)

**The sharpest entry is not mine.** The 2026-08-14 pre-registration's skepticism ledger, item 2,
*predicted the floor* — "the bloom is the *gate* over-firing on its own drift (AP→0.01 by h36)" — and the
experiment log then promoted it to **"the invariant is the gate has no rollout precision (AP 0.30→0.01)"**.
The condition that made the measurement impossible was written down, and reclassified as the result. That
is the most instructive failure in this whole episode and it was made before I touched the programme.

**Mine, in order of severity:**

1. **I inherited the vehicle across four dossiers without once asking whether it could show an effect.**
   The vehicle was chosen for speed and never re-examined. I wrote "INDICATIVE, one vehicle" in every
   scope section and then reasoned as though it were not — the caveat was ritual, not load-bearing.
2. **I read a floor as a finding, twice.** "The gate diffuses" and "the model stops committing" were both
   properties of the smoke vehicle promoted to properties of the model. Both are now retired.
3. **I quoted a ratio without checking its denominator.** Top-K's "14–19× headroom" was real as a ratio
   and worthless as a lever — measured on a field holding two cells. I was one step from recommending a
   build on it.
4. **Confounds not ruled out:** the `truncated_smoke` ↔ `violet_visitor` contrast differs on **three**
   axes — `total_lessons` (40/160), `output_distribution` (`truncated_nb`/`nb`) and **`body_supervision`
   (`active`/`all`)**. Three documents state "two axes"; that is corrected with this postmortem. Nothing
   in the record disentangles them, so *nothing here attributes the floor to any one of the three.*
5. **Counter-hypothesis not tested:** that the smoke vehicle is representative and `violet_visitor` is the
   outlier. Partially addressed by EXP-03 (six models: retention 0.02–0.54, smoke at the bottom with
   `pink_pirate`), but `violet_visitor` *is* a config outlier on six axes, and that is not nothing.

## 6. What is / isn't established

**Established.**
- The smoke vehicle's control at h18 sits **below random ranking** (0.77× prevalence). Arithmetic, from
  committed CSVs.
- On a vehicle with range, the same intervention reads **−93.7%** where it read **+0.9%**. One controlled
  replication, identical code, identical arms.
- The 2026-08-14 SS sweep **cannot discriminate** and is VOID as run.
- A gate on the control's own score CSV separates the two vehicles by **36×** (0.77× vs 28.30× prevalence).

**Not established — and this half is longer on purpose.**
- **Which of the three axes causes the floor.** Length, distribution and body-supervision all differ. A
  40-lesson `nb` model has never been trained; that single run would isolate length.
- **That `violet_visitor` is representative.** It is a six-axis config outlier. Retention across the roster
  spans 11× with no measured predictor.
- **That any re-scoped claim is false.** They are *unlicensed*, not refuted. M7, M8, I-B, I-C and I-E have
  simply not been re-derived on a vehicle that can carry them.
- **That the gate's thresholds are right.** R=5 is defensible and sits inside a (0.8, 28) separating band,
  but it has been validated against exactly two vehicles.
- **That this is the only floored measurement in the repo.** Only the rollout-collapse programme was
  audited. Earlier dossiers were not.

## 7. Disposition

| what | disposition |
|---|---|
| `2026-08-14_scheduled_sampling_dossier` | **VOID as run.** Its design is sound; its vehicle was not. Superseded by `2026-08-17_ss_retention_dossier` as *take 2*. |
| M1, M9 | **survive.** M1 is a baseline the floor makes *true*; M9 is a statement about the metric, independently corroborated by Epic #263. |
| M3, M4, M6-oracle | **understated, replicated.** Keep with the corrected numbers from the replication dossier. |
| M5 | **uninformative as run**, rescued by the M17 re-run on `violet_visitor`. |
| **M7, M8, I-B, I-C, I-E** | **uninformative as run and NOT re-derived.** Marked owed in the ledger. |
| M20 (the SS lead) | **confounded** — see C-300; rewritten. |
| "the gate diffuses" | **RETIRED** (EXP-03: Moran's I holds 70–86% in all six roster models). |
| "the model needs to answer" | **RETIRED** (EXP-03: commitment spans 24× and does not track skill). |
| C-152's mechanism update | carries floored numbers as a class statement — **annotated**, not deleted. |
| **C-299** | new. The defect class. Tier 2. |
| **C-300** | new. Four roster models trained under a config the codebase now rejects — see §8. |

## 8. The second finding, which the audit surfaced

All four scheduled-sampling-*on* roster models were trained **2026-08-12/13**. The C-259 validator that
couples `ss_feedback` to `rollout_feedback` landed **2026-08-14 04:19** (`c07a352`). `ss_feedback` defaults
to `"mean"` (`config_initializer.py:178`), and `training_engine.py:231` returns an **ungated** mean for any
mode ≠ `sample`.

So those four trained on an **ungated mean field while rolling out on a gated sample** — precisely the
mismatch C-259 now forbids. Their `rescore.csv` rows are therefore compromised twice over: **un-rerunnable**
(their configs no longer load — views-models#404) and **trained under a configuration the codebase rejects**.

This also reframes the observational lead. "Scheduled sampling hurts retention" is very likely the wrong
reading; what the roster shows is four models trained under a forbidden mismatch. **A sweep with
`ss_feedback='sample'` tests a different intervention and cannot settle it** — pre-registered as a scope
limit on the successor dossier.

## 9. Meta-lesson — the gate that should have existed

Every falsifier in the 2026-08-14 plan (F1, F2, F-DEGEN) assumes the readout is valid. **None fires on an
uninformative measurement.** That is the hole, and it is generic: a pre-registration can be perfect about
*what would refute the hypothesis* and silent about *whether the instrument can see anything at all*.

The fix is one cheap, objective clause, checked on the control arm before any treatment arm runs:

> **FG-A.** `AP_control(h*) ≥ 5 × prevalence(h*)` — the ranker must demonstrably beat chance at the
> readout horizon.
> **FG-C.** `(1 − θ) · AP_control(h*) ≥ 3 × MDE_AP(h*)` — the pre-registered effect, applied to *this*
> control, must exceed the measurement's resolution.

FG-C is the one that names the real failure. It is not "the model is bad"; it is **"the effect we came to
measure is smaller than what this setup can resolve."**

Validated on data already committed: `truncated_smoke` **0.77×** (FAIL), `violet_visitor` **28.30×**
(PASS). Cost: zero GPU beyond the control arm you were running anyway.

*The measurements from these three days were all real. What was wrong was the licence I gave them. A
result that cannot move is not a result, and the cheapest moment to discover that is before the second arm
starts — not five hours after the last one finishes.*
