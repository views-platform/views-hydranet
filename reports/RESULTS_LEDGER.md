# Results Ledger — HydraNet experiments (C-113 program & beyond)

**Purpose:** a durable, human-curated record of *what we ran, what we got, and what we now
know* — keyed to the **parameters/architecture** of each run, including things wandb does
**not** track (architecture variant, loss family, config knobs, qualitative read). This is our
sanity record; wandb holds the curves, this holds the *conclusions*.

**Living doc.** Append rows; never rewrite history. git-tracked via `git add -f` (reports/ is gitignored).

> ⚠️ **GAP: 2026-06-20 → 2026-08-17.** This ledger was not maintained for two months. Everything in
> that window — the ZINB epic (#167), the bloom epic (#193), the composition axis (#183), the v2
> scoreboard, the rollout-ruler artifact verdict (#263), the state-freeze probe (#277) and the
> feedback-realism probes (#278) — was recorded **only in per-dossier logs**. Each dossier reasons
> locally and correctly; nothing forced a claim to survive contact with the others, and that is
> exactly where narrative drift lives. The §Claims Ledger below is back-filled for the
> **rollout-collapse programme only**. The rest of the window is **NOT back-filled** — treat its
> conclusions as living in their dossiers until someone does the work, and do not read silence here
> as agreement.

---

## 🏁 TRAINING LENGTH — ANSWERED 2026-08-20

**Retention plateaus at ~300 lessons; training is closed as a retention lever.**
`0.03 (40L, no skill) → 0.600 (160L) → 0.683 (300L, n=4) → 0.692 (600L)`. The last step is +0.0084
against a seed sd of 0.046. **T=0 skill and the ceiling keep rising** (+0.0213 vs sd 0.0077), so the model keeps
improving and stops becoming more robust. The gate still loses **31%** of itself to its own feedback and
no amount of training recovers it. **40 lessons = smoke only** (matches climatology at h1, 10–25× worse
at h18). **160 lessons is not converged** and is only 14% above climatology at h18 — the parked SS sweep
must move to L=300. Rows **M26–M29**; dossier `reports/2026-08-18_lesson_curve_dossier/`.

---

## 🧪 SCHEDULED SAMPLING — ANSWERED 2026-08-21

**Training on its own output makes the rollout WORSE, and it is the PLACEMENT that costs.**
At L=300, 4v4 seeds, one variable: `ε=0.5` lowers AP@h18 from **0.3257 → 0.2831** (**−0.0426**,
exact one-sided **p = 0.0286**), retention −0.053, all four seed-pairs down on both endpoints, and the
**anchor guard passes** (ΔAP h1 −0.0277 vs a 0.0440 limit) — so this is retention, not a trade.
Formally **UNDERPOWERED**: significant, but the drop does not clear 3×MDE (0.0541). Direction
established, magnitude not.

**The twist:** SS largely **fixed** the zero collapse — `act_ratio` at h18 went **0.0093 → 0.0875
(9.4×)**, at h36 **28×** — and AP fell at *every* horizon anyway. The placement probe splits the damage
**56% the model itself / 44% the field it emits**, and `thin:0.75` shows the SS model uses a *good*
field slightly **better** than the control. **It answered more, in worse places, and lost.**
Rows **M30–M33**; dossier `reports/2026-08-17_ss_retention_dossier/`.

---

## 🏆 PERSISTENCE RE-REFERENCE — 2026-08-21 — the dumb baseline no longer beats us

**M1 is overturned for a converged model.** At L=300 the model beats a *fairly-scored* persistence at
**every horizon out to 36 months** — worst of 4 seeds h1 1.99×, h18 2.16×, h36 2.22×, every gap 2.1–4.3× the MDE. M1
was true of the **40-lesson** vehicle it measured (**M28**: a smoke test) and nobody had re-run the
race. Persistence itself barely moved (matches M1's column to ~1%) — **the ruler is stable; the model
changed**, free-running h18 0.007 → 0.3318, **47×**. Along the way the baseline was found to be
understated by **two** scoring defects (binary ranking at S=1; `m0-1` never loaded → the first
origin's persistence is silently all-zeros, **#282**), together worth **+31%** at h18 — so M1 was
*stronger* than written, not weaker. **All four ε=0 seeds ran** — the **worst** seed still beats persistence at every horizon (2.1–4.3× MDE, seed sd ≤0.019). **Remaining caveats: one vehicle, AP only** — the `crps_all` ARTIFACT verdict is untouched, and retention is still stuck at 0.69. Rows **M34–M37**; dossier
`reports/2026-08-21_persistence_reference_dossier/`.

---

## 🧊 STATE-FREEZE AT L=300 — 2026-08-22 — it helps, it is the CELL, and pairing settles it

**Freezing the cell state buys +0.039 AP@h18 on a converged model** (both seeds; +0.060 at h36), and
the **paired origin-block CI excludes zero at h6/h18/h36**. **M8 is direction-confirmed and
magnitude-retired** — its 13× recovery was a broken control, not a bigger effect. **`hidden` does
nothing, `cell` does everything** (M39), and the anchor is a **switch that saturates at w≈0.1** — no
dial to tune (M41). And pairing cut the MDE **0.0541 → 0.0086, 6.3× on identical
data** — the same effect reads UNDERPOWERED unpaired and EFFECT paired, which **answers #281 by
demonstration** (it does *not* rescue the SS sweep, whose arms differ by seed too). ⚠️ Caveats: CI is
one seed, one vehicle, **AP only**; this is a rollout-time intervention, **not a fix** — C-293's
static-map worry is untouched. Rows **M38–M41**; dossier
`reports/2026-08-22_state_freeze_l300_dossier/`.

---

## Claims Ledger — the rollout collapse (#258 / #262)

**Why this section exists.** The run ledger below is per-*run* and the narrative is chronological, and
chronological prose is what drifts: five mechanism stories were retracted on 2026-08-16/17 alone. This
table is per-*claim* and answers one question — **what do we believe right now, and how much weight can
it carry?**

**The distinction that does the work:**

- **MEASUREMENT** — a number a run produced. Overturned only by a bug in the harness or a re-run.
- **INFERENCE** — a story built on measurements. Overturned by thinking, and repeatedly has been.

Every claim below carries its scope. **The entire rollout-collapse programme to date is single-seed,
single-vehicle**, which is *below this project's own evidentiary bar* (≥3 seeds on validation; the v2
scoreboard ran 3 seeds × 300 lessons). Nothing here is a verdict. The dossiers say INDICATIVE and then
get reasoned from as though they did not — this table exists to make that impossible.

> ⛔ **FLOOR STATUS (2026-08-17) — read `reports/postmortem_floor_limited_vehicle.md` before citing M1–M9.**
> The shared vehicle below, `truncated_smoke`, has **no dynamic range**: its control at h18 scores 0.0070
> against a prevalence of 0.009077 — **0.77×, below random ranking**. That breaks measurements in both
> directions (nulls uninformative, positives understated), and it is why `spatial_scramble` read **+0.9%**
> here and **−93.7%** on `violet_visitor`. Per-row disposition:
>
> | status | rows | meaning |
> |---|---|---|
> | **survives** | M1, M9 | M1 is a baseline the floor makes *true*; M9 is about the metric, corroborated by Epic #263 |
> | **understated, replicated** | M3, M4, M6-oracle | re-measured on `violet_visitor`; cite the corrected number |
> | **rescued by re-run** | M5 | uninformative as run; re-derived as M17 |
> | ⛔ **uninformative and NOT re-derived** | **M7, M8, I-B, I-C, I-E** | **owed a re-derivation on a vehicle with range. Do not cite as support.** (I-A **was** re-derived — 4/4 vehicles, 2026-08-18.) |
> | ⛔ **confounded** | M20 | see its row — the roster's SS-on models were trained under a C-259-forbidden mismatch (C-300) |
>
> C-299 is the defect class. I-C's floored form was written into the register as C-152's mechanism update
> and is annotated there.

**Shared scope unless a row says otherwise:** seed 42 · 40 lessons · `truncated_smoke` ·
artifact `calibration_model_20260814_003058.pt` · 13 origins · 4 posterior samples · 35 steps ·
target `sb` where a single target is named · calibration partition.

### LIVE — measurements

| # | Claim | Evidence | Confidence |
|---|-------|----------|------------|
| M1 | **On occurrence (AP), persistence beats every arm from h6 on.** h6 0.112 / h18 0.108 / h36 0.083 vs the best held arm 0.087 / 0.091 / 0.069 and free-running 0.028 / 0.007 / 0.008. At h1 the model wins (0.298 vs 0.146). **On `crps_all` the arms beat persistence at every horizon (CRPSS +0.20/+0.45/+0.41/+0.11) — and that win is an ARTIFACT at h6/18/36** by the audited rule; see §The persistence re-reference. | state-freeze EXP-03 | **High** — a baseline, not an inter-arm comparison. Metric-qualified: an unqualified version of this row was wrong. **⚠️ VEHICLE-QUALIFIED by M34 (2026-08-21): true for the 40-lesson vehicle it measured, FALSE at L=300.** This row's persistence column was also **understated** by two scoring defects (binary ranking + #282), so as written the claim was *stronger* than it should have been, not weaker. |
| M2 | **Gate AP collapses steeply then saturates:** 0.298 (h1) → 0.028 (h6) → 0.007 (h18), flat thereafter. ~5 steps hold most of the damage. | realism EXP-01 | **High** — large effect, reproduced across every arm's control. |
| M3 | **Scrambling only the LOCATIONS of a perfect field reproduces the collapse:** AP 0.3008 → 0.0097, against free-running 0.0070 — with active count and magnitudes held identical. | realism EXP-03 | **High** — 31× effect; direct manipulation. Confounded with geographic grounding (C-291). |
| M4 | **Sparsity alone is survivable.** At matched horizon, `thin:0.75` fires at a *similar* rate to the collapse and scores far better — h18: AP 0.2244 vs 0.0070 (**32×**) at `act_ratio` 0.332 vs 0.291; h36: AP 0.1898 vs 0.0083 (**23×**) at 0.317 vs 0.266. | realism EXP-03 | **High** — large, direct, holds at both horizons. ⚠️ EXP-03 quoted "0.33 vs 0.27", which pairs `thin` at **h18** with `identity` at **h36** — a cross-horizon conflation. The conclusion survives at matched horizons; the quoted pair did not exist in any single cell. |
| M5 | **Clustering spanning 100× moves AP not at all.** Fed clustering 0.011 → 1.064 (brackets the real 0.449); AP flat at ~0.007. | realism EXP-05 | **Medium-high** — a null over a wide dose range, but arms were **not byte-paired** (C-296), so read at one significant figure. |
| M6 | **The recurrence does not smear the gate.** Oracle holds Moran's I flat over 35 steps (sb 0.507 → 0.494 → 0.516); free-running over identical steps falls 0.409 → 0.192 → 0.178. | realism EXP-04 data, analysed 2026-08-17 | **High** — same architecture, same kernel, same step count; only the fed content differs. |
| M7 | **The gate's ranking stays structured while the draw does not.** At equal expected count, top-K clustering vs independent-draw clustering: 4.4× (step 1) → 15.5× (step 6) → 26.8× (step 12). | realism EXP-04 | **High** — same gate, same count, two draw rules. |
| M8 | **Freezing recurrent state partially recovers AP:** h18 0.0070 → 0.0912, h36 0.0083 → 0.0693. Ordering `all` ≥ `cell` > `hidden` > `none`. ⚠️ **DIRECTION CONFIRMED, MAGNITUDE RETIRED by M38 (2026-08-22).** Measured on a 40-lesson vehicle **M28** calls a smoke test; the 13× recovery looked spectacular only because the control was broken — this row's *recovered* 0.0912 is **3.6× BELOW** an L=300 model's untouched free-running 0.3298. On a converged vehicle the same intervention buys **~+13% relative**. Quote the direction, never these numbers. | state-freeze EXP-02 | **Medium** — real and pre-registered, but see I-D and C-292 for what it does *not* license. |
| M9 | **`crps_all` is blind to all of it.** Four arms score 0.1353 / 0.1352 / 0.1350 / 0.1346 at h18 while gate AP spans **13×**. | state-freeze EXP-02 | **High** — corroborates Epic #263 independently. |

### LIVE — inferences

| # | Claim | Rests on | Confidence |
|---|-------|----------|------------|
| I-A | **Occurrence carries ~90–95% of the gap; magnitude carries ZERO.** ⬆️ **UPGRADED 2026-08-18 — replicated 4/4 vehicles.** occurrence 89.5–95.3%, magnitude −1.0% to +1.4%, across two body families, two compositions, four seeds and a 3× range in baseline retention. All four predictions and all falsifiers passed on every vehicle. | multivehicle EXP-01 (+ M3, M4) | **High for the ORDERING.** The occurrence share is stable (89–95%); magnitude is indistinguishable from zero everywhere. ⚠️ `spatial_scramble`'s magnitude is **not** stable (−21% to −94%, 4× spread) — quote its sign, never its size. Different configurations, not different seeds. |
| I-B | **Clustering is a proxy for correct placement, not an independently sufficient property.** Right places + no clustering → collapse (M3); wrong places + right clustering → no recovery (M5). | M3 + M5 | **Medium-high** — the two arms bracket the claim from opposite ends. |
| I-C | **Coordinate channels never helped because they act on marginals while the failure is joint.** | M3, M5, C-152's 3-seed negative | **Medium** — a mechanism fitted to an already-established null. It explains, it does not predict. |
| I-D | **Some of the gap flows through the recurrent state (~23% of the oracle gap).** ⚠️ The **23%** is the floor-limited figure; on a converged vehicle it is **+0.039 of a ~0.17 oracle gap at h18** (M38). | M8 | **Low.** INDICATIVE. Recovers 23% *relative to a collapsed control* and still does not reach persistence (M1), so it is not a skill claim. **Which memory half is NOT established** (C-292). |
| I-E | **The independent Bernoulli draw discards usable ranking information.** | M7 | **Medium** — the information gap is measured (M7). That fixing it would help is **untested**, and M5 is a warning that a plausible fix can do nothing. |

### REPLICATED on a vehicle with skill — 2026-08-17

`reports/2026-08-17_vehicle_replication_dossier/`. Six arms on **`violet_visitor`** (160 lessons, `nb`,
REAL against climatology through h18 by the audited verdict), all falsifiers pass, GREEN. This directly
tests whether M3/M4/I-A above were artifacts of an undertrained vehicle. **They were not.**

| # | Claim | Evidence | Confidence |
|---|-------|----------|------------|
| M10 | **The oracle does not degrade at all.** Fed the real field, gate AP holds 0.4745 → 0.4793 (h18) → 0.4577 (h36) over 36 steps. All free-running decay is attributable to fed-back content, not to the recurrence or the horizon. | replication EXP-02 | **High** — direct, large, and corroborates M6 on the headline metric. |
| M11 | **Occurrence is ~95% of the gap; magnitude ~0%.** E4a (real occurrence × the model's own **71%-inflated** magnitudes) recovers 95.3% at h18; E4b recovers 1.4% and is **negative at 4 of 6 horizons**. On smoke: 88.6 / 7.9. | replication EXP-02 | **Medium-high** — replicates across two vehicles, near-additive (86–99%). Still one seed each. |
| M12 | **Wrong placement is worse than the model's own errors.** `spatial_scramble` scores 0.0486 at h18 against a control of 0.2569 — 5× worse — while `thin:0.75` discards ¾ of true events and recovers 95.5%. | replication EXP-02 | **High** — 5× effect, direct manipulation, F6 confirms the transform bit. |
| M13 | **The Epic #263 board reproduces bit-for-bit** from preserved cubes: worst \|ΔAP\| = 0.00e+00 over 7 horizons, `N` identical. | replication EXP-00 | **High** — exact. |
| M14 | **The post-2026-08-12 inference commits are a no-op on this vehicle's free-running path** (incl. `a2eabeb` per-site LockedDropout): `identity` today vs cubes from 08-12, worst \|ΔAP\| = 0.00e+00. | replication EXP-02 | **High** — exact, on 7 horizons. |

**Two corrections this forces on earlier work:**

* **M3/I-A were measured under a floor effect.** On `truncated_smoke` the control was already at 0.0070,
  so `spatial_scramble`'s "+0.9% of the gap" was the distance between two numbers both pinned near zero —
  never a measurement of placement's importance. The smoke run **understated** it.
* **The share statistic `(arm − control)/(oracle − control)` does not apply to arms outside that
  interval.** `spatial_scramble` falls *below* the control on both vehicles, so its share is negative and
  meaningless as a fraction. Any decomposition must check the interval before quoting a share.

**Re-scoping, per the pre-committed decision rule:** I-A moves from "single-seed decomposition, do not
quote the number" to **replicated across two vehicles with the ordering robust and magnitude's share
falling to ~0**. It remains one seed per vehicle, so under the standing rule this is an **escalation
trigger** — second seed, third vehicle — not a conclusion.

---

### THE SKILL ENVELOPE — 2026-08-18 — where the model is actually worth using

| # | Claim | Evidence | Confidence |
|---|-------|----------|------------|
| M23 | **The model beats climatology to ~month 18 and loses from month 24.** Gate AP, `sb`, `violet_visitor` free-running vs `climatology`: h1 +0.176, h6 +0.130, h12 +0.078, h18 +0.032, **h24 −0.006, h30 −0.028, h36 −0.030**. The crossover sits around **month 20**. Past it you would do better with "fighting happens where it usually happens". | `rescore.csv` (Epic #263, origin-block CI) | **High** — this is the shipped board's own data, stated plainly for the first time. It is implicit in its ARTIFACT verdicts from h24; the plain statement was missing. |
| M24 | **The ceiling is ~0.47 at EVERY horizon, and the rollout throws away two thirds of it.** Fed the true field each month, the same 160-lesson model holds AP 0.474 → 0.479 → 0.458 across 36 steps. Left to feed on itself it goes 0.474 → 0.257 → 0.137. **160 lessons does not fix the rollout; it produces a model with skill to lose.** | replication EXP-02 + M10 | **High** — same model, same steps, only the fed content differs. |
| M25 | **Extra training buys robustness, not accuracy.** 40L vs 160L on an otherwise identical config: at h1 the ratio is **1.6×** (0.289 → 0.474), at h18 it is **13×** (0.020 → 0.257). The one-step-ahead ability barely moves; the ability to survive its own output transforms. ⚠️ The *ratio* is sensitive because 0.020 is small; the absolute gap (0.237 AP) is the sturdier statement. | ss-retention Stage A vs A′ | **Medium-high** — single-variable, one seed each, but the effect dwarfs seed spread. |

**The practical read.** The gap worth attacking is **~0.2 AP at h36** — the distance between the rollout
(0.137) and the ceiling (0.458). That is **larger than the entire distance between the current model and
climatology at any horizon**, and per I-A it is ~90–95% a *placement* problem.

---

### TRAINING LENGTH — 2026-08-17 — steep below 160, UNKNOWN above it

| # | Claim | Evidence | Confidence |
|---|-------|----------|------------|
| M21 | **Training length is the dominant cause of the floor.** A clone of `violet_visitor` differing in **exactly one key** (`total_lessons` 160 → 40, verified by symmetric-difference on the resolved config dicts) collapses retention **0.54 → 0.068**. The residual 0.068 → 0.02 is what `truncated_nb` and `body_supervision` contribute. | ss-retention Stage A | **High** — single-variable, 8× effect, one seed but far above seed spread. |
| M22 | **Training is bit-reproducible at fixed seed on this box.** Retraining violet at HEAD gave **190 weight tensors with an identical sha256** and predictions matching to 15 d.p. at all 7 horizons, 5 days and 5 commits later. The 8 commits touching the training path since 2026-08-12 are **no-ops for this configuration**. ⚠️ The two identical models have **different `.pt` file shas** — torch stamps mtimes into the zip, so file shas are an invalid identity check. | ss-retention Stage A′ | **High** — exact. |

**✅ RESOLVED 2026-08-20 by `reports/2026-08-18_lesson_curve_dossier/` EXP-01 — see M26–M29 below.**
Retention plateaus at ~300 lessons; T=0 skill and the ceiling keep rising; 160 is **not** converged for
retention. The block below is kept as the question that was asked.

**⛔ What WAS not established (superseded, kept for the record).**

*Nothing above 160 lessons is known.* Specifically:

* **600 lessons: no evidence exists in this repo** — no config, no result, one passing mention in a
  planning doc. The historical recollection that it worked well is currently the only evidence.
* **300 lessons: the v2 board is not a controlled comparison.** It is `gated_NB`, 3 seeds, trained
  2026-07-29 on a different data snapshot from `violet_visitor`. Comparing it to violet at 160L
  compares two unrelated runs, not two lesson counts. Its own seed spread at h36 is **0.048 — 34% of
  the mean** — and violet at 160L sits *above* the whole 300L range at h18 and *below* it at h36, i.e.
  outside the spread in **opposite directions**, which is what noise looks like.

**So the claim "160 → 300 buys nothing" is NOT supported.** It was stated twice in conversation on
2026-08-17 and does not survive checking; recorded here so it is not repeated.

**Why it matters:** the ladder 600 → 300 → 160 → 40 was a cost-saving decision, never validated as
monotone. M21 shows the bottom of it is catastrophic, which is direct evidence the curve is *not* flat
where it was assumed to be. **Whether 160 is on the plateau or still on the slope is unknown** — and if
it is still climbing, every experiment at 160 (including the parked SS sweep) measures a
partially-trained model, and a null there may only mean "this does not help a model that has not
finished learning."

### TRAINING LENGTH — ANSWERED 2026-08-20 (EXP-01, lesson-curve dossier)

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| **M26** | **Retention saturates at ~300 lessons.** `AP(h18)/AP(h1)`, free-running, `sb`: 40L **0.03–0.07** (2 seeds, both floor-gate FAIL) → 160L **0.600 ± 0.046** (n=6) → 300L **0.683 ± 0.032 (n=4)** → 600L **0.692**. The 300→600 step is **+0.0084**, one **fifth** of the anchor's seed sd. ⚠️ **CORRECTED 2026-08-22:** this row originally read 300L **0.690** and the step **+0.0014** from a **single seed**; three more ε=0 L=300 seeds landed later via the SS-retention amendment, and the multi-seed mean is 0.683. **The plateau claim is unaffected** — the step is still far inside the seed sd — but the number was stale (falsifier-checks CHECK B). | lesson-curve EXP-01 | **High for the shape** (5 lesson counts, anchor n=6, seed noise measured). One seed at 300 and 600, so the plateau itself is a two-point claim. |
| **M27** | **T=0 skill and the ceiling do NOT saturate.** 300→600 moves T=0 **+0.0213** against a seed sd of **0.0077** (~3×, real) and the ceiling (oracle h18) 0.4974 → 0.5072. **The model keeps improving with training; it stops becoming more robust.** Every gain past 300 lessons comes from being better to begin with, not from surviving its own output better. | lesson-curve EXP-01 | **Medium-high** — one seed per point above 160, but the T=0 move is ~3× the measured seed noise. |
| **M28** | **A 40-lesson model has NO skill, at any horizon — it is a smoke test, never a result.** Month 1 it *matches* climatology (0.97× / 1.03×); month 18 it is **10–25× WORSE** than climatology (0.09× / 0.04×). Both 40L arms fail the floor gate (2.16× and 0.99× chance) and their two seeds sit 2.3× apart. | lesson-curve EXP-01 + `scripts/floor_gate.py` | **High** — two seeds, both floored, and the gate rejects them by an objective threshold. |
| **M29** | **160 lessons is NOT converged for retention, and is marginal at the rollout horizon.** Retention 0.600 against its own 0.690 plateau, and h18 AP beats climatology by only **14%** (vs 47% at 300L). **Any rollout/retention experiment at 160 measures a partially-robust model that barely clears the trivial baseline where it is read out.** | lesson-curve EXP-01 | **High** — direct, and it re-scopes the parked SS sweep. |

**What this closes and what it opens.** **Training is closed as a retention lever**: it took retention
from ~0.03 to 0.69 and stopped. The residual **31%** the gate loses to its own feedback is structural
and needs a different lever. **L=900 is deprioritised** — retention is the quantity of interest and it
stopped moving two rungs earlier; 900 would measure T=0 and the ceiling, which is not what the rollout
programme is trying to fix. It becomes interesting again only after retention moves by some other means.

⚠️ **Action carried to the parked SS sweep:** `2026-08-17_ss_retention_dossier` must run at **L=300**,
not 160 (M29). Same design, one config value.

---

**Being answered: `reports/2026-08-18_lesson_curve_dossier/` (pre-registered 2026-08-18, LOCKED).**
Controls **and** oracles at L = 160 / 300 / 600 (900 conditional), one variable (`total_lessons`),
splitting any change into a **ceiling** part and a **retention** part via `log F = log C + log R`. It
spends its first stage measuring **σ_seed** — the seed-to-seed SD of retention across four training runs
at L=160, which this programme has never measured on this vehicle — because without it a one-seed
lesson point is a number rather than an effect or a null. Pre-registered θ = 0.14 (30% of the measured
0.4687 gap from R(160)=0.5415 to the oracle's 1.0101) and four decision states, so **PLATEAU is
declarable rather than assumed**. σ_seed is a deliverable in its own right: it bounds what *any*
single-seed experiment on this vehicle can see.

⚠️ Confound declared before the run (**C-301**): `curriculum.py:85` normalises the difficulty schedule
by `total_lessons`, so a longer run is the same curriculum **stretched**, not continued. The experiment
answers "does a longer budget help", never "do more gradient steps help".

---

### SCHEDULED SAMPLING — ANSWERED 2026-08-21 (EXP-01/EXP-02, SS-retention dossier)

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| **M30** | **Scheduled sampling makes the free-running rollout WORSE.** L=300, seeds 42–45, `ss_epsilon_max` 0 vs 0.5 as the only variable: mean AP@h18 **0.3257 → 0.2831** (**−0.0426**), retention **0.6833 → 0.6303** (−0.053), **exact one-sided p = 0.0286**, and **all four seed-pairs fall on both endpoints**. The §4 **anchor guard passes** (ΔAP h1 −0.0277 against a 0.0440 limit), so it is not a one-step-for-many trade. Formally **UNDERPOWERED** — the drop does not clear 3×MDE (0.0541), so the DIRECTION is established and the MAGNITUDE is not. | SS EXP-01, rule md5 `d1432db9a7611cf349f1009225365027` | **High for direction** (4v4, pre-registered, one-sided, guard passed, unanimous by seed). **None for magnitude** — the gate says so by its own rule. |
| **M31** | **SS largely FIXED the zero collapse and lost skill anyway.** `act_ratio` (fed-back activity ÷ real): h18 **0.0093 → 0.0875 (9.4×)**, h36 **0.0007 → 0.0204 (28×)**. AP fell at **every** horizon (h1 −0.029, h6 −0.040, h18 −0.050, h36 −0.032). **The under-firing symptom and the skill loss are separable — treating the symptom did not treat the disease.** | SS EXP-01 | **High** — a large, monotone move on the diagnostic, against a uniform loss on the endpoint. |
| **M32** | **The damage splits ~50/50 between the model and the field it emits, and placement is the half that was actionable.** Handing both models perfect occurrence leaves **56%** of the h18 gap standing (+0.0132 of +0.0234); **44%** (+0.0102) is attributable to the field SS emits. The **ceiling** drop (oracle h18, −0.0149) independently reproduces the residual to 0.002. Pre-registered P1 (>60% placement) and P2 (<30%) **both FAIL** — the answer is "both, roughly half each". **This CORRECTS M15**: the model does not merely need to answer; *where* it answers decides the score. | SS EXP-02 (placement probe) | **Medium-high** — two independent routes to the residual agree, but **one seed**; the sweep's significance rests on the other four. |
| **M33** | **SS did NOT damage the model's ability to use a good input — that hypothesis is falsified.** `thin:0.75` recovers **90%** of the control's own gap and **93%** of the SS model's (P4 HOLDS). `spatial_scramble` sits far below both controls (−0.237, −0.211) — destroying placement is worse than either model's own output, replicating M12 on a third pair (P3 HOLDS). | SS EXP-02 | **Medium-high** — clean, large margins; one seed; `spatial_scramble` inherits C-291's confound. |

**What this closes and what it opens.** **Exposure-bias training is closed as a retention lever on this
vehicle**, at this dose, in this form — the one untested lever from the 2026-08-17 review is now tested
and it points the wrong way. Consistent with **Huszár (2015)**: scheduled sampling is a statistically
inconsistent estimator, and here its target is partly unlearnable (the model is asked for the real next
month from a degraded field that does not contain it) while the compounding regime is wrong (ε=0.5 gives
an expected run of 2 synthetic steps against an 18–36-step inference face). **What it opens** is the
narrower question M32 poses: a lever that improves *placement* without also making the model worse as a
model. Dose shape (ε ∈ {0.1, 0.25}) was pre-registered as *after* an effect is demonstrated — the effect
is demonstrated, in the wrong direction, so a smaller dose now asks "is there a dose that does not hurt"
rather than "is there a dose that helps".

### PERSISTENCE RE-REFERENCE — 2026-08-21 (EXP-01/02/03, persistence-reference dossier)

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| **M34** | **A 300-lesson model BEATS persistence at every horizon out to 36 months — on ALL FOUR seeds.** One shared support (`sb`, N=170430 on every seed). **The WORST of four seeds** beats persistence everywhere: h1 0.4716 vs 0.2364 (**1.99×**), h6 (**2.26×**), h18 0.3058 vs 0.1416 (**2.16×**), h36 0.2108 vs 0.0951 (**2.22×**) — margins **2.1×–4.3× the MDE**. Seed sd 0.0035–0.0189, an order of magnitude below the gap. **This overturns M1 for a converged model**; M1 remains true of the 40-lesson vehicle it measured. | persistence-ref EXP-01/02/**03** | **High** — n=4, worst-case not mean-case, one shared support, and the re-emits reproduced two archived controls exactly (seed 43 to <5e-5; seed 42's h18 AP 0.3298 on the nose). Still **one vehicle** and **AP only**. |
| **M35** | **The persistence baseline was being understated by two independent scoring defects, both fixed here.** (a) `_persistence_gathered` supplies no gate, so AP ranked it on a **two-level** `(cs>0).mean(1)` at S=1 while gated arms got a continuous probability; ranking by the persisted *value* lifts h18 0.1152 → 0.1416. (b) `score_v2_horizons` never loads month `m0-1`, so the **first origin's persistence forecast is silently all-zeros** (**#282**); loading it lifts h18 0.1077 → 0.1152. Combined **+31%** at h18. | persistence-ref EXP-02; #282 reproduced to 4 dp | **High** — the defect reproduces the scorer's exact output, and the direction is asserted over 200 random draws. |
| **M36** | **Persistence is a stable ruler, and that is what makes the comparison legitimate.** Persistence on our origins matches M1's column to ~1% (h6 0.1122 vs 0.112, h18 0.1077 vs 0.108, h36 0.0834 vs 0.083) when scored **the same defective way**. Persistence is truth-only, so this is what it must do iff the origin and cell sets are comparable — they are. **The ruler did not move; the model did**: free-running h18 went **0.007 → 0.3318, a factor of 47.** | persistence-ref EXP-01 | **High** — three independent horizons agreeing to ~1% on a truth-only baseline. |
| **M37** | **Seed variance is not a threat to M34, and the n=1 draw was mildly flattering.** Across 4 seeds the arm's AP sd is **0.0035 (h1) to 0.0189 (h12)**, while the gap it would have to cross is **0.14–0.24**. The single seed reported first (43) read **2.41×** at h36 against a 4-seed worst of **2.22×** — high, but inside the noise. **The aggregator refuses to summarise unless the support matches across seeds** — it compares both the per-horizon row count `N` **and the per-seed persistence AP**, the latter being the sharper test because persistence is truth-only and must return one number per support. | persistence-ref EXP-03 | **High** — direct measurement of the quantity in question. |

**What this closes and what it opens.** The programme has spent weeks under the belief that the
long-horizon rollout had **no demonstrated value** because a dumb baseline beat it. That was measured
on a vehicle **M28** now classifies as a smoke test, and **nobody re-ran the race**. It cost 7 minutes
of GPU to find out. **What it does NOT do** is rescue retention: the model still loses 31% of itself
to its own feedback (**M26**), it is **one vehicle**, and this is an **AP** claim only — the `crps_all`
ARTIFACT verdict stands untouched. **The n=1 caveat is now discharged** (EXP-03, 41 min): all four ε=0 seeds ran and the **worst** of them still beats persistence everywhere. What remains open is **one vehicle** and **AP only**.

### STATE-FREEZE AT L=300 — 2026-08-22 (EXP-01/02, state-freeze-l300 dossier)

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| **M38** | **Freezing the CELL state helps a converged model — +0.039 AP@h18, and the interval excludes zero.** L=300, 2 seeds, `sb`: h18 mean `none` 0.3308 → `cell` 0.3666 (**+0.036**, both seeds agree: +0.0391 / +0.0323); h36 **+0.060**. Paired origin-block 90% CI (seed 43, 13 origins, 400 reps): h6 **[+0.0163, +0.0286]**, h18 **[+0.0297, +0.0469]**, h36 **[+0.0500, +0.0704]** — **all exclude zero**, effect **4.5× its own MDE**. **M8's direction replicates on a real vehicle; its magnitude framing does not** (13× there vs ~+13% relative here). | state-freeze-l300 EXP-01/02 | **High for direction** — 2 seeds, both falsifiers passed (h1 identical across arms; every `none` arm reproduces its published free-running value), interval excludes 0 at three horizons. **Medium for size** — the CI is one seed. |
| **M39** | **It is the CELL state, and only the cell.** `hidden` alone does **nothing** (−0.005), `cell` alone does **everything** (+0.036), `all` adds **+0.001** over `cell`. This does **not** contradict **C-292** — that entry retires the *decomposition* claim (`hs = o ⊙ tanh(hl)` makes freezing cell constrain hidden by construction, so "cell carries 89%" was predetermined). An **ablation ordering in the other direction** is not predicted by that argument and stands. | state-freeze-l300 EXP-01 | **High** — a clean 4-arm ablation, replicated on 2 seeds. |
| **M40** | **The PAIRED construction resolves what the unpaired one cannot — #281 answered by demonstration.** Identical data, identical 13 origins: unpaired MDE **0.0541** (the SS sweep's between-seed design) vs paired **0.0086** at h18 — **6.3× tighter**. The same +0.039 effect reads **UNDERPOWERED** unpaired and **EFFECT** paired under the identical `3 × MDE` rule. ⚠️ **Does NOT retroactively rescue the SS sweep** — those arms differ by seed *as well as* treatment, so they are not pairable. The lesson is for **design**, not for re-reading old results. | `scripts/ap_block_bootstrap.ap_diff_origin_block_ci`, 9 tests | **High** — measured on real arms; the estimator's own test bootstraps two arms independently and asserts the paired interval is tighter. |
| **M41** | **The cell anchor is a SWITCH, not a dial — and it saturates at w≈0.1.** Convex weight `w·anchor + (1−w)·new` on the cell half, seed 43, `sb`, AP@h18: w=0 **0.3318** → 0.10 **0.3643 (83% of the full-clamp gain)** → 0.25 0.3678 (92%) → 0.50 0.3716 → 0.75 0.3731 → 1.00 0.3709. **Verdict CONFIRMED on the correct yardstick 2026-08-22 (§5a).** The paired interval for the actual contrast — `cell@0.5` vs `cell`, MDE **0.0045**, not the 0.0086 wrongly cited (C-306) — **includes zero at h18** ([−0.0039, +0.0051]) and h36, so there is no resolvable interior optimum. **At h6 the hard clamp is significantly BETTER** (−0.0061, [−0.0107, −0.0011], excludes zero), which the coarser wrong yardstick hid: the interior point does not merely fail to win, it **loses** at short horizons. Two process defects stand regardless: the registered rule fired branch 1 (0.3715866 > 0.3709158) and was **overridden on grounds it did not contain**, then written up as "no branch matched" (**C-305**) — the +0.0007 turns out to be noise, so the override was right by luck, not by rule. The curve is **sharply saturating**, which says the cell **drifts** rather than diverging and a light restoring force nearly matches a hard clamp. **Consequence for learning w:** headroom above a crude constant is ~17% of a +0.039 effect ≈ **0.007 AP, below the MDE** — a learned *scalar* has nothing measurable to win; only a state- or horizon-dependent function could (#290, #291). | state-freeze-l300 EXP-03 + §5a | **Medium-high** — the switch verdict is now established on the correct paired interval (MDE 0.0045) and is *stronger* than first stated (the clamp wins outright at h6). Falsifier passed (h1 identical **per seed**). Still **one seed**, and the two process defects C-305/C-306 are recorded against how it was reached. The conclusion is a negative and the shipping decision is unchanged, so it does not need the replication a positive would. |
| **M42** | **ITF fails too — and the variable looks like DOSE, not direction.** L=300, `sb`, 2 seeds, ε decaying 0.5→0: AP@h18 **0.3298→0.3125** (seed 42, −1.30σ) and **0.3318→0.3105** (seed 43, −1.59σ) — §4's rule returns **ITF FAILS TOO**. **Both arms passed the floor gate**, so this is not Teutsch's premature-termination failure; ITF trained properly and did worse. **The reframing:** mean ε over training is control **0.000** / ITF **0.251** / SS **0.487**, and *both* endpoints order monotonically by it — AP `ctrl > ITF > SS` and `act_ratio` `ctrl < ITF < SS`, **12 of 12 orderings at h6/h18/h36 across both seeds**. ITF's exposure is ~half SS's and its damage ~half (−0.019 vs −0.043). **This extends M31**: SS *fixed* the zero collapse (`act_ratio` ×9.4) and lost AP anyway; ITF does the same at half dose (×1.4, half the AP loss). Reversing the ramp changes the *amount* of exposure, not the sign of its effect. **The AP ordering already holds at h1** (0.4779/0.4761/0.4502; 0.4774/0.4601/0.4435) — the step with **no feedback at all** — so the curricula damaged the *weights*, not just the trajectory. | itf-pilot EXP-01 | **Medium for the verdict** — a 2v2 screen (exact one-sided permutation floor **0.167**), so direction-and-magnitude only, never a p-value; anchor guard clean; both floor gates PASS. **Low for the dose hypothesis** — ITF and SS differ in mean ε *and* schedule shape, so the ordering does not isolate dose; 2 seeds, 3 points. **Registered false-negative mode (C-307): ε started at 0.5, not 1.0** — a null cannot separate "ITF fails" from "we did not run real ITF". Reopen triggers in `07` §7. |
| **M43** | **The rollout does NOT begin from an out-of-distribution state — H refuted, and M38 still has no mechanism.** L=300, 2 seeds, `sb`, states only, no training. `f` = fraction of deployment (R2) per-cell state values outside their channel's [1%,99%] interval built from training-like input (R1: 32×32 patches drawn by the PRODUCTION anchor strategy at three curriculum ratios, same months). **A 98% interval puts f=0.02 outside by construction; measured f = 0.0030 / 0.0032 (cell), 0.0033 / 0.0036 (hidden) — ~7× BELOW chance.** No seed split, no half split. So the zero collapse is a **dynamical** property of free-running, not distribution shift at the origin — and the out-of-range explanation for **M38** (+0.039 AP from freezing the cell) is **excluded**. Byproduct: **the effective receptive field is far smaller than the theoretical one** — patch-vs-full-grid interior `rel_abs_diff` 0.0006 vs pooled 0.10, a 142× gap, because the recurrence *contracts* (F2: 65.6 → 1.8, **35.8×**, over 30 free steps). Byproduct 2: **the curriculum barely separates the training diet** — mean event density 4.548/5.985/3.966 across thresholds 143/75/10, i.e. **+15% and non-monotone** over a ratio range of 0.665→0.05. | state-range EXP-01 | **High for the negative** — 2 seeds, both halves, all five falsifiers passed (F3 the geometry HARD STOP among them), effect ~7× below chance so it is not a near-miss. ⚠️ **`f` measures EXCURSION, not DEGENERACY**: a state collapsed toward zero reads trivially 'in range' of an interval spanning zero, so this scopes to *the origin state is not an OOD excursion* and says nothing about whether the collapsed state is representative. ⚠️ Registered false-negative mode (§7): R1 uses FINAL weights, so this closes *the deployed model is run out of range*, NOT *training never saw a different state regime*. ⚠️ A first pass used `predict()`'s `seq_len-1=383` fallback instead of the production origin **335** and reported |R2|max as 21.59 vs the true **66.08** (published 65.6) — a 3× error with every falsifier still green, same shape as C-308. |
| **M44** | **The gate is HOT and the composition halves it — the occurrence process is applied twice.** L=300, `sb`, **4 seeds**, step 1 (input entirely real, no feedback yet), no training. `soft_gate` emits `body_sample × bernoulli(gate)`, and the NB body carries its own mass at zero, so a cell survives only if the gate fires AND the NB draw is non-zero — re-answering the question the gate was trained on. Measured: **mean G = mean(gate)/truth = 1.280** (the gate OVER-predicts occurrence by 28%), attenuated **×0.332** by `P(NB>0)` to **mean C = 0.424**, reproducing the observed emitted ratio (0.371–0.427) on **4/4 seeds** within ±0.05. **The shortfall is in the COMPOSITION LAYER, not the gate and not training.** Upstream of every rollout result: the ×0.78/step compounding starts from a step-1 value already 2.4× too low. Reframes M30–M33/M42 — every SS arm fed back a field whose occurrence was already halved. | front-loading EXP-01 | **High for the decomposition** — 4 seeds, all four falsifiers pass, and `C` reproduces an independently measured quantity it was not fitted to. ⚠️ **OCCURRENCE ONLY, one step, no AP measured** — *that fixing composition improves AP is NOT established here* and needs its own pre-registration. ⚠️ `C` sits ~0.03 ABOVE observed on all 4 seeds — a consistent sign, so a real small bias: this tool runs dropout OFF, the `fedfield` CSVs come from the production path with locked dropout ON. ⚠️ F2 was broken as registered (2% tolerance against a single realisation carrying ~15% MC error); fixed by averaging 200 draws, tolerance unchanged. |
| **M45** | **The double-counted zero was LOAD-BEARING — removing it collapses the rollout by 70%.** L=300, `sb`, **4 seeds**, one key changed (`output_distribution` nb→truncated_nb), verified single-variable against each seed's own `fullzero_*` control. AP@h18 **0.3298→0.1113 / 0.3318→0.0970 / 0.3058→0.0497 / 0.3352→0.0943**; mean Δ **−0.2376** = **7.5× the 3×MDE bar**, exact one-sided permutation **p=0.0143 — the 1/C(8,4) FLOOR**, i.e. all four treatment arms below all four controls. **All four floor gates PASS**, so the vehicles had dynamic range. **The MODEL is barely worse — the FEEDBACK LOOP is destroyed**: given a real field it scores 0.4625–0.4720 vs the control's 0.4910–0.5014 (~5% down), but free/oracle retention falls **0.66 → 0.19**. **F2 passed**: step-1 `act_ratio` moved 0.36→**1.16–1.75×** truth (past the gate's own 1.28× over-prediction, exactly as M44 predicts), then compounded **×8–22 at h18** and **×21–69 at h36**. Every guardrail worsened (`crps_all` h18 0.134→0.520; `size_ratio` 0.00→1.90; `precision_at_k` 0.385→0.166). **M44 is NOT overturned** — the zero process IS applied twice; what is falsified is *therefore removing it helps*. **Completes a 4-point dose-response**: ITF (act_ratio ×1.6, ΔAP −0.019), SS (×5, −0.043), truncated_nb (×1170, −0.238) all lose AP in proportion to how much they make the model FIRE, while the only rollout intervention that ever worked — the cell-state freeze, **+0.039** — is the only one that does not touch firing. **Raising recall in the rollout is harmful in proportion to the dose.** Vindicates M21's confounded floor-limited hint. | truncated-nb EXP-01 | **High** — 4 seeds, single-variable verified against each control, permutation at the exact floor, effect 7.5× MDE, all falsifiers passed. ⚠️ **§7 false-negative mode:** the gate retrains alongside the body, so this closes *swapping the body fixes rollout skill*, NOT *the double-zero diagnosis was wrong*. ⚠️ **Process defect disclosed:** §5's permutation test was implemented one-sided for `treatment > control`, so a negative effect scored p=1.0 and first rendered as *NULL/UNDERPOWERED*; found AFTER seeing the numbers, fixed in AMENDMENT 1 from the locked prose, no threshold moved. |
| **M46** | **Sharpening the model's spatial view helps h1 and hurts h36 — 12 of 12 arms, six unrelated mechanisms.** L=300, `sb`, 6 architectures × 2 seeds vs the existing `fullzero_*` controls, one key changed (`model`), 12/12 floor gates PASS. **§4 verdict: NO PROMOTION** — 5 INCONCLUSIVE, 1 REJECT (ShallowPool, −0.0178/−0.0150 @h18). The finding is the **monotone sign gradient**: arms beating control number **9, 9, 5, 5, 2, 0, 0** at h1/6/12/18/24/30/36 — **12 of 12 NEGATIVE at both h30 and h36** (2⁻¹² per column), and the damage is ordered by how aggressively each architecture preserves resolution (DualStream −0.0567 @h36, ShallowPool −0.0475). **The models are NOT worse** — every oracle matches or beats the control's ceiling (WideMemory best at 0.5077/0.5094 vs 0.4974/0.5014) and Δ`crps_all` is within ±0.006, so the body is untouched and this is purely the gate's placement degrading through the rollout. **Predicted by `Aceituno2025` Thm 4.6** (short-horizon minima do not generalise to long horizons, gap `O(e^{λΔT})`): we train at ONE step and evaluate at 36. **Reframes M30–33, M42 and M45** — all three changed what the model is FED; none changed the HORIZON OF THE LOSS, so long-horizon training has never been tried here. `MillerHardt2019` explains WideMemory's null: a contractive recurrence (35.8× state collapse measured) is in the stable regime where truncated feed-forward approximates it, so memory capacity is unused — and that is also why freezing the cell state (M38) helps. | arch-bakeoff EXP-01 | **High for the pattern** — 12 arms, 6 mechanisms, monotone, all falsifiers passed, mechanism tests verified to fire against 7 neutering mutations. ⚠️ **The cross-arm sign pattern is POST-HOC** and cannot promote anything under §4. ⚠️ **Per-candidate results are a 2-seed SCREEN**: the permutation floor is 0.167, INCONCLUSIVE means unresolvable, NOT 'no effect'. ⚠️ **My primary endpoint h18 was the worst available** — it sits on the sign change (5 better/7 worse) and reported 'nothing happening'; chosen by convention, not because it discriminates. |
| **M47** | **The pushforward trick does NOT improve the rollout — it makes it consistently worse, while leaving the model itself untouched.** L=300, `sb`, **4 seeds**, one key changed (`pushforward_weight` 0.0→0.01, verified single-variable against each seed's own `fullzero_*`), 4/4 floor gates PASS. **§4 verdict: UNDERPOWERED — and that is the honest label, not a hedge.** ΔAP@h18 = **−0.0220**, **all four seeds negative**, exact one-sided permutation in the observed direction **p = 0.0429** (floor 0.0143), paired 95% interval **[−0.0308, −0.0131]** — which excludes zero but does **not** exclude the pre-registered MDE of 0.024, and §4 requires `ΔAP ≤ −MDE` for EFFECT NEGATIVE. So: a real, replicated, significant negative effect whose magnitude cannot be claimed to reach the threshold we declared meaningful before running. **The mechanism is clean and is the actual finding: the oracle does not move** (Δ = −0.0065 to +0.0019 across seeds, all inside σ=0.0134, F5 PASS 4/4) and h1 is unchanged (+0.0012), so the model is *equally good teacher-forced and worse free-running* — the precise opposite of what Brandstetter's pushforward is for. **F7 does NOT fire**: firing rose 1.35–5.50× across seeds while ΔAP stayed flat at −0.015 to −0.027, giving r=+0.008 — so this is **not** an M45 firing-dose artifact, and the two findings are independent. **A prior weight (0.1) is VOID**, not negative: it moved the oracle −0.023 to −0.036 at every horizon (F5+F6 fired), i.e. it damaged the model rather than the rollout — recorded as dose evidence. | pushforward EXP-01 (VOID) + EXP-02 | **Medium-high** — n=4, one shared support, all falsifiers mechanised and F5/F6 demonstrated live by killing the w=0.1 arm. ⚠️ Two caveats on the record: the paired interval is a **seed-level t-interval, NOT the pre-registered origin-block bootstrap** (the prediction cubes are deleted after scoring, so the registered statistic was not computable), and the controls were trained before PR #303 — reuse validated by a fresh control within 1.48 sd (Amendment 1), not by bit-reproduction, which **C-317** shows is unavailable here. |
| **M48** | **The cell anchor holds at four seeds — the only positive result the rollout programme has produced, and it needs no retraining.** L=300, `sb`, **4 seeds**, **emit only** on the existing `fullzero_*` artifacts (`freeze_recurrent='cell'` is an `InferenceOrchestrator` setting, not a config key and not a trained behaviour — artifact mtimes unchanged, G3 PASS). **§4 verdict: CONFIRMED** — the rule required 4/4 seeds positive **and** the paired 90% interval excluding zero on ≥3 of 4; **both were 4/4**. AP@h18: 42 **+0.0323** [0.0228, 0.0421] · 43 **+0.0391** [0.0297, 0.0469] · 44 **+0.0460** [0.0327, 0.0561] · 45 **+0.0292** [0.0185, 0.0412] — every effect **3–4× its own paired MDE** (0.0086–0.0117). **The horizon shape is the corroboration**: mean Δ is **exactly 0.0000 at h1** and rises monotonically to **+0.0591 at h36** (+26% relative), 4/4 seeds positive at every horizon from h6 on. Nothing to fix at the first step, most to fix at the last — the signature of drift accumulating through the rollout, not of a better model. It recovers **~22% of the oracle gap** (0.4974 vs 0.3257). **This confirms M38 (2 seeds) and, with M39 (cell only) and M41 (a switch, saturating at w≈0.1), completes the state axis.** Set against M42/M45/M46/M47 — six architectures and four feeding schemes, all negative, oracle never moving — the programme's shape is now: **the model is not the problem, the free-running state is, and this is the one lever that touches it.** | state-freeze L300 EXP-04 | **High** — n=4, one shared support (170,430 cells, identical across arms), paired origin-block CI per seed, emit-only so no training variance enters. **G1 is unusually strong**: seeds 42/43 re-emitted on today's code reproduced their archived August values to the last digit (0.3298395823400329), which both validates the vehicle and demonstrates that *emitting* is bit-deterministic even though *training* is not (**C-317**). ⚠️ **The mechanism is unknown** — M43 refuted the out-of-distribution-state explanation, so this is an effect without a cause. `sb` and AP only; w=1.0 (the full clamp), M41's dial not re-tested. |
| **M49** | **The rollout field does NOT blur — it goes quiet. One explanation for M48 is excluded.** L=300, `sb`, seed 42, **emit only**, two arms on one artifact: `identity` (free-running) vs `use_real` (real field fed back, so blur cannot accumulate by construction). Harness check passed exactly — at h1 both arms predict from real data and were **byte-identical on 13/13 origins** (max |Δ| = 0.000e+00), differing by 0.944 at h36. **§6 verdict: S1 FIRES, hypothesis dead.** Moran's I under `identity` **falls** 0.6554 → 0.4904; `conc1pct` falls 0.4999 → 0.3001. The instrument was validated on synthetic fields *before* seeing data (`tests/test_field_sharpness.py`), and blur **raises** both (Moran's I 0.64 → 0.97 at σ=2) while **thinning lowers** them (0.64 → 0.29) — so both detectors agree *against* blur and point at the quieting signature (**S5**). `use_real` is flat on both (0.6554 → 0.6492), so **S2 does not fire**: this is genuine rollout accumulation, not a property of later months — the control worked. **Confound stated, not buried:** `act_ratio` under `identity` collapses 0.3627 → 0.000246 (**1,475×**, the known #258 collapse) while `use_real` holds ~0.36, and since thinning also lowers Moran's I, *this design cannot separate 'the field decoheres' from 'the field empties'* — the alternative reading is **UNRESOLVED**. **M48 is untouched**; what is excluded is one *explanation* for it. Stage 2 was pre-registered to run only if Stage 1 survived, and did not run. | sharpness diagnostic EXP-01 | **High for the negative, none for the alternative** — n=1 seed but the comparison is internal (same artifact, same origins, same code, one flag), the h1 byte-identity check validates the arms, and the falsifying direction was fixed by synthetic test before any data. ⚠️ One seed, `sb` only, gate field only. The §7 prediction was that Moran's I would RISE; it fell — the third wrong prediction in this programme, and the argument for validating instruments before trusting stories. **Cost: ~25 min, no training.** |
| **M50** | **The rollout FADES, and the cell clamp stops it — M48 finally has a mechanism.** Emit only, no training, each level within its own seed. **Field (seed 42):** firing collapses **1,547×** over 35 steps free-running (0.000612 → 0.0000004) but only **2×** with the cell clamped (0.000612 → 0.000315), and the clamped arm is **flat from step 6 onward**; at step 30 it fires **308×** more than the control **and the magnitude claim in the first version of this row was WRONG — see the correction below.** **State (seed 43):** `max|h|` drains **41×** (65.6 → 1.6) free-running and **holds** under the clamp (65.6 → 66.1). **The registered trap F5 is the actual finding**: the *cell* half is held by construction and proves nothing, but the **hidden** half — which evolves freely — drains 0.98 → 0.56 without the clamp and 0.98 → 0.93 with it, so **holding the cell stabilises the half it does not touch** (C-292 predicted this was possible; it is now measured). F3 identity check passed byte-identical (`AP@h18 = 0.3621885544392029`, the archived M48 `cell` value). **The chain is now end-to-end: clamp → state stops draining → field stops collapsing → AP rises 0.000 at h1 to +0.059 at h36.** This supplies what **M43** and **M49** each failed to: both asked the wrong property — M43 whether the state left its *range* (it does not; collapsing toward zero is inside it), M49 whether the field *blurred* (it does not; a vanishing field has no structure to smear). **Neither asked about magnitude.** It also explains four negatives in one line: M42, M45, M47 and M26–M33 all changed *how the model decides to fire* while the signal driving that decision drained away. | fade mechanism EXP-01 | **High for the mechanism, none for causation** — within-seed at both levels, one flag changed, F3 byte-identical, and the two falsifiers that could have killed it were registered before running and did not fire. ⚠️ **Does NOT establish that the fade CAUSES the AP loss** — both could follow from a third property of holding the state; that needs an intervention stopping the fade *without* clamping. Field is seed 42, state is seed 43: consistent, but sibling vehicles. Anomaly recorded not explained: unclamped fed magnitude goes **negative** by step 30 in a log1p space where it should not. **Cost: ~15 min, emit only.** ⚠️ **CORRECTED 2026-09-02 (C-318), by an `/expert-code-review` that read the CSVs rather than the write-up.** The first version of this row said the fed field collapses *on both axes*, occurrence **and** magnitude (18.4 → −0.8). **Magnitude does not collapse.** `mean_magnitude_on_active` carries an explicit `-1.0` UNDEFINED sentinel when `n_active == 0` (`hydranet_inference.py:527-533`), and by step 30 **97% of the control's records are that sentinel** — a mean of clamped non-negatives cannot be negative, which is the tell I missed. Filtered to records that actually fired, magnitude is **18.4 → 13.5 unclamped and 18.4 → 13.8 clamped — essentially identical.** The comparison of the clamped arm's 15.7 against the control's −0.7 put a real mean beside a 97%-sentinel column. **The finding stands and is sharper for it: the collapse is PURELY occurrence** (active fraction 0.000612 → 0.0000004, **1,547×**, unclamped versus **2×** clamped — both computed from `active_fraction`, which is never a sentinel), and the state numbers come from a separate capture path and are untouched. **What changes: the clamp preserves *where the model fires*, not *how loudly*.** | ⛔ **RETRACTED IN PART 2026-09-02 by M51 — the second correction to this row.** The corrected sentence above ("the collapse is PURELY occurrence"; "the clamp preserves *where* the model fires, not *how loudly*") is **FALSE**. It rested on `mean_magnitude_on_active`, a statistic conditioned on cells that fired — and M51 measures that at h36 the conditioned set holds **1 cell** at τ=0.1 and **0** at every higher τ, against 3840 at h1. Filtering the sentinel out left `n=2 of 156`; the statistic had no support either way. Measured unconditioned over every cell, magnitude falls **4.5×** (plain body mean) — the field collapses on **both** axes after all. **What survives:** the clamp/state chain and every `active_fraction` number, which were never conditioned. **What dies:** the occurrence-only reading, twice asserted from the same faulty instrument.
| **M51** | **The rollout collapses on BOTH axes — 'fewer, not smaller' is FALSE — and the gate stops knowing where the big values are.** L=300, `sb`, seed 42, **emit only**, 13 origins, one flag (`feedback_transform`: `identity` free-running vs `use_real` control). Measured over **every cell with no conditioning**, via the exact identity `mean(g·μ) = mean(g) × [Σgμ/Σg]`, h1 → h36: **occurrence 4.146e-03 → 1.492e-04 (×0.036, 28× fewer)** and **magnitude, plain mean of μ, 0.0961 → 0.0213 (×0.222, 4.5× smaller)**. Pre-registered **F1 fired at 0.0144 against a 0.5 threshold** — a 35× margin, not marginal. **Control flat on all three** (×1.19 / ×0.91 / ×1.12 across 36 horizons) and the arms are **identical at h1** as they must be before feedback acts, so this is free-running, not the readout. **The unpredicted third finding may matter most: the ALIGNMENT between firing and magnitude decays 66.6× → 4.3×** (`gate-weighted μ / plain-mean μ`). At h1 the model fires where it expects a lot; by h36 the gate is nearly uninformative about where the large values are, so what keeps firing is disproportionately the **small** stuff — survivorship running *downward*, the opposite of the rival that was pre-registered. That is why gate-weighted magnitude falls **69×** while the plain mean falls **4.5×**: the extra collapse is the gate landing in the wrong places, not the body shrinking further. **Sharpens M32/M45** (placement is everything) by naming *how* placement fails, and gives **M48**'s clamp a specific thing to be tested against. | silence-vs-fade EXP-1 | **High for the falsification, none for replication** — pre-registration locked (`db20868`) and amended (`8e3b99c`) before any GPU second; instrument default-off, parity-tested, **15/15 mutations caught**; F7 anchor exact (`AP@h18 = 0.3298395823400329`); gate cross-instrument agreement **byte-identical (0.000e+00)** across two separate runs. ⚠️ **Seed 42 only — 05 §7 makes replication a CONDITION of the finding, so this is not yet established.** ⚠️ **F3 FIRED and is left fired on the record**: its 10% band is tighter than the reference's own sampling noise and fires on the known-good control (2/36 horizons); proceeding past it was the chair's explicit call, not a silent override. The magnitude channel cannot be cross-validated at h36 by any sampling reference — the field yields **0 nonzero draws in 210,000 slots**, which is itself a measurement. **Cost: ~2 h wall clock, emit only, no training.** |
| **M52** | **The cell clamp completely stops the alignment decay — M48's first mechanism story to survive a real test, but it is CONSISTENT-WITH, not causal.** L=300, `sb`, seed 42, **emit only**, 13 origins, 355.6 s, one flag added to M51's arm (`freeze_recurrent='cell'`). `A(h) = gate-weighted μ / plain-mean μ` — how concentrated firing is on the cells the body thinks are big — runs **66.6 → 4.3** unclamped and **66.6 → 69.3 clamped**, i.e. **flat across all 36 horizons**. Pre-registered recovery fraction **R = 1.042** against a 0.25 bar for support and 0.07 for death. Both collapses also arrested: occurrence ×0.036 → **×0.715**, plain magnitude ×0.222 → **×0.651**. **What the arithmetic says precisely:** alignment is a ratio, so "preserved" means the two magnitudes decline *proportionally* under the clamp (0.677/0.651 = 1.04), whereas unclamped the gate-weighted magnitude falls **15× further** than the plain mean. **The clamp does not stop the field fading — it loses ~29% of its firing and ~35% of its magnitude anyway — it stops the field firing in the WRONG PLACES.** Together with M51 this is the first end-to-end account of M48 that survived contact with a validated instrument, and it sharpens M32/M45 (placement is everything) into a mechanism: **the cell state carries *where it will be big*, and free-running drains that specifically.** | silence-vs-fade EXP-2 | **Low as confirmation, and registered that way BEFORE running** — `05b` §2 declared this a strong falsifier and a **weak confirmer**, and the weak branch is what happened. ⚠️ **NOT causal.** AP was already known to rise (M48, 4 seeds); alignment is now also known to hold. That is **two known effects of one intervention**, not evidence one produces the other. **AP is not evidence here** and was used only as the identity check — reading it as support would be circular. ⚠️ **The outcome was the EXPECTED one, which is what makes it weak**: `hs = o ⊙ tanh(hl)` (C-292) means holding the cell can bound the hidden half (M50 measured it), so restored alignment discriminates poorly. Had F-A fired it would have been decisive. ⚠️ **Seed 42 only.** Gates passed exactly: F-B anchor `AP@h18 = 0.3621885544392029` (archived M48 `cell` value), F-C h1 identical to 9 s.f. **Settling it needs an intervention that restores alignment WITHOUT clamping, or clamps WITHOUT restoring it** — until then "the clamp works by preserving alignment" and "the clamp does something else that also preserves alignment" fit equally well. | ⚠️ **QUALIFIED 2026-09-03 by M53.** "The clamp preserves alignment" is true and is **not an explanation**: M53's rolled-anchor arms preserve alignment (61.5 vs the clamp's 69.3), occurrence (0.718 vs 0.715) and magnitude (0.628 vs 0.651) just as well, and score **AP@h18 0.0075 against 0.362** — 1/48th the skill. **Alignment is NOT SUFFICIENT for skill**, so the sufficiency reading of this row is dead. It may still be necessary; M53 does not refute that. The sentence "the cell state carries *where it will be big*" survives only in the weak sense — M53 shows the anchor's spatial content IS load-bearing, but no statistic in this row measures whether that content is *correct*.
| **M53** | **Holding the state about the WRONG PLACES is far worse than not holding it — and every field statistic we have is blind to the difference.** L=300, `sb`, seed 42, **emit only**, 13 origins, one flag (`freeze_anchor_roll ∈ {3,15,90}`) on M52's arm. The roll is a **permutation**, so the anchor keeps its norm, mean, variance, per-channel distribution and internal spatial structure exactly, and loses only its correspondence to geography. **AP@h18: unclamped 0.32984 · clamp 0.36219 · roll3 0.04040 · roll15 0.01062 · roll90 0.00753** — the roll is not merely benefit-free, it is **44× below the unclamped baseline**, and monotone in distance. **⇒ H-scale REFUTED**: the clamp does not work by steadying the state's magnitudes; the anchor's **spatial content is load-bearing**. **The bigger finding came from a falsifier that fired.** FR-4 required alignment to fall when the map was wrong; it did not (61.5 vs 69.3), because **alignment is an INTERNAL statistic** — it measures whether the gate and body agree *with each other*, not whether those cells are *right*. A rolled model is perfectly self-consistent about the wrong places, and occurrence (0.718 vs 0.715) and magnitude (0.628 vs 0.651) are equally blind. **No statistic in M51/M52/M53 can distinguish a good forecast from the same forecast in the wrong place** — glossary now carries the measured warning (`0d4d079`). **Qualifies M52** (alignment is not sufficient) and re-grounds M32/M45's "placement is everything" as the only truth-referenced reading. | silence-vs-fade EXP-3 | **High for the refutation, and it does not depend on the fired gate** — FR-1 (`B(90) ≥ 0.7`, which would have killed H-place) did not fire at −9.96; FR-3 h1 identity passed to 9 s.f. across all five arms; FR-2 did not fire, the dose being monotone. Instrument: 8 tests, **8/8 mutations caught**, three silent no-ops (zero roll, whole-period roll, roll without a clamp) all fail loud. ⚠️ **FR-4 FIRED and is recorded fired** — the SECOND mis-specified gate in this dossier after F3's band, both from a falsifier written on an untested assumption about what a statistic measures. ✅ **The off-distribution reading is now EXCLUDED by measurement (EXP-3b, same day).** The proposed re-score-against-rolled-truth test was **refused as invalid** — `torch.roll` wraps on a torus while only 13,110 of 32,400 cells are study cells, the input was never rolled, and AP scores study cells only, so a displaced forecast could score zero merely by landing outside the mask. Circular cross-correlation of the FIELDS (no mask, no truth, no score) instead: the peak sits at **exactly the roll distance at every horizon for every dose — (3,3), (15,15), (90,90) — at r ≈ 0.90–0.93, with r ≈ 0 at zero offset**, while the clamp-vs-unclamped control peaks at (0,0). **The rolled model is not broken; its forecast is the clamp's forecast, displaced.** ⇒ **the cell state functions as a MAP**: move it 90 cells and the forecast moves 90 cells while skill collapses 48×. ⚠️ Seed 42 only. **Cost: ~30 min, emit only.** |
| **M54** | **The ConvLSTM cell state is a MAP: move it 90 cells and the forecast moves 90 cells, intact.** Re-analysis of M53's field dumps — **no new GPU** — with a circular cross-correlation instrument validated on synthetic conflict-shaped fields *before* touching real data. **The cross-correlation peak sits at EXACTLY the roll distance — (3,3), (15,15), (90,90) — at every horizon for every dose, r = 0.90–0.93, with r ≈ 0 at zero offset.** Control: clamp vs unclamped peaks at (0,0), r = 0.71–0.78, so the instrument distinguishes *different* from *displaced*. At h2 every arm still matches the clamp exactly, as it must, since the blend applies to the state carried forward. **⇒ the rolled model is NOT broken** — it emits a structurally intact forecast displaced by exactly the distance its memory was displaced, which **excludes the off-distribution reading of M53 by measurement rather than argument**. Completes the chain from **M48**: the clamp helps because the cell state carries a *spatial layout of where conflict will be*, free-running loses it, and holding it hands the model a correct map back. **Nothing about the state's scale is doing the work** (M53 refuted H-scale; this shows what H-place actually means mechanically). It also explains M53's fired FR-4: a displaced forecast is internally perfectly coherent, so no internal statistic can see it (**C-319**). | silence-vs-fade EXP-3b | **High** — the readout is instrument-internal (no truth, no mask, no score), which is precisely why it evades the three ways the obvious AP-based test would have been invalid: `torch.roll` wraps on a torus while only 13,110 of 32,400 cells are study cells; the model's INPUT was never rolled; and AP scores study cells only, so a displaced forecast could score zero merely by landing outside the mask. Instrument: 9 tests, **7/7 mutations caught**, recovers planted shifts at r=1.0, holds the peak under 25% noise, scores unrelated fields < 0.5, refuses constant fields. ⚠️ **INDICATIVE ONLY, not pre-registered:** the peak is ~0.90 not 1.00 while the input was never rolled, which *suggests* the recurrent state dominates the emitted field's spatial structure over the incoming data — but r=0.9 does not decompose linearly into "90% memory", and this needs a designed experiment. ⚠️ Seed 42, `sb`, gate field. **Cost: zero GPU.** |
| **M55** | **The clamp is NOT buying persistence — it beats it ~3× at h36 and the gap WIDENS.** No new GPU: the fair-persistence baseline was computed 2026-08-21, before this dossier, and the support was verified identical at every horizon (`N = 170430`; `n_event` 1343/1336/1379/1547/1590/1655/1779 matching exactly) before any number was read. **AP@h36: persistence 0.0951 · unclamped 0.2208 · clamped 0.2828 ⇒ 2.97×.** The ratio to persistence **rises** across the horizon (2.02× at h1 → 2.97× at h36); a persistence-equivalent state would give a flat or falling ratio. Said more directly, **fraction of h1 skill retained at h36: persistence 40.2%, unclamped 46.2%, clamped 59.2%** — **the clamped model degrades more slowly than the observed field its own anchor was built from**, which a persistence fallback cannot do. **Answers the load-bearing 'bad reasons' worry**: the anchor pins state built from the last real observations, and conflict is persistent, so the clamp could have been a placement-persistence prior dressed as skill. It is not. With **M54** the reading is that the cell state holds a **learned** spatial prior, not a copy of the last observation. | silence-vs-fade EXP-3c | **Moderate** — the comparison is exact (same cells, same truth, same scorer, support checked first) and the baseline is the *corrected* value-ranked persistence from M34–M37, not the handicapped binary one. ⚠️ **NOT PRE-REGISTERED** — no committed prediction, no falsifier; the baseline predating the dossier and the support check remove the researcher degrees of freedom, but this is a comparison, not a test. ⚠️ **Gate only.** The same score CSVs show the clamp does **nothing** for the body in truth-referenced terms: `size_ratio` **0 → 0** at h18 and h36 in BOTH arms, and `crps_events` moves 7% at h18 and **0.6%** at h36. The one flattering body number (`pos_mcr` 32×) is the mean the glossary already warns is carried by a few large cells. **Freezing is a placement fix and there is no evidence it is a magnitude fix** — consistent with M32/M45 and the amount-ceiling wall. ⚠️ Seed 42, `sb`, AP. **Cost: zero GPU.** |

**What this closes and what it opens.** **#280's primary suspect survives, qualified** — M8 is
direction-confirmed and magnitude-retired, not withdrawn, which is the more useful outcome: the
floor-limited vehicle inflated a **headline**, it did not invent an **effect**. **#281 is answered** —
pair the design and it costs no GPU time. **What this does NOT do is fix the collapse.** +0.039 at h18
sits against an oracle ceiling near 0.50; freezing is a rollout-time intervention, not a trained fix;
and a frozen state is a **static risk map by construction**, which is exactly the degenerate-forecast
worry **C-293** raised. That the effect *grows* with horizon (+0.023 → +0.039 → +0.060) is consistent
with both "the state carries real information" and "a static map beats a degrading gate", and this
design does not separate them.

### INFERENCE-TIME FIXES — CLOSED 2026-08-17

`reports/2026-08-17_placement_intervention_dossier/`. All three candidate families are closed, each for a
different understood reason. **You cannot repair at inference time a gate that has stopped committing.**

| # | Claim | Evidence | Confidence |
|---|-------|----------|------------|
| M15 | **The model feeds back 2 cells where reality has 116** — a 57× shortfall. `thin:0.75` shows **29 well-placed cells recover 95%** of the oracle gap, so the model does not need to be nearly right; it needs to answer. ⚠️ **The second clause is CORRECTED by M32** — answering is necessary, not sufficient; answering in the wrong places is worse than staying quiet. | intervention EXP-02 | **High** — direct count, and it is what closes top-K. |
| M16 | **The gate keeps its shape and loses its nerve.** Moran's I dips to 0.458 at step 12 and **recovers to 0.593** by step 35, while `gate_mean` falls **12×** and committed cells fall 92 → 9. | intervention EXP-02 | **High** — direct, and it separates *zero collapse* from smearing. |
| M17 | **No marginal-preserving sampler can move a decided gate.** A 16× length-scale range plateaus at 25% of real clustering; AP flat and uniformly negative (best −0.0023). Mechanism established on a controlled synthetic sweep, not inferred. | intervention EXP-01 + `tools/marginal_skew_bound.py` | **Medium-high** — the null replicates M5 on a non-floor-limited vehicle *and* supplies its cause. |

**RETIRED by M16:** *"the gate's spatial structure diffuses during the rollout"* — load-bearing in the
reasoning since 2026-08-16 and the stated motivation for every coherent-sampling idea. It was measured on
`truncated_smoke` (0.409 → 0.178, stayed down). On the production vehicle the gate does **not** smear. M6
survives in its original form (the *oracle* holds structure) but its free-running half does not generalise.

**Two reasoning defects recorded with the results:**

* the first explanation for the copula's saturation was **backwards** — "too diffuse" when the bound is
  **skew** (too *decided*); a uniform gate reaches 13× the clustering the real run achieved;
* a build was nearly recommended on top-K's **14–19× headroom**, which is real as a ratio and worthless as
  a lever because it is measured on an essentially empty field. Same class as the floor-limited smoke
  measurements: a ratio between two numbers both near zero. **Check the absolute count under a percentage.**

---

### ROSTER CHECK — 2026-08-17 — gate structure is CLOSED as an explanatory axis

`reports/2026-08-17_placement_intervention_dossier/` EXP-03. Gate probe on all six roster models.

| # | Claim | Evidence | Confidence |
|---|-------|----------|------------|
| M18 | **No gate-structure metric predicts rollout retention.** Retention varies **11×** (0.02–0.54) while commitment (24× span), confidence decay (1.2–7×) and shape retention (70–86%) all vary independently of it. All six start within AP h1 0.38–0.47, so they differ almost entirely in retention. | EXP-03, 6 models | **High** — a negative across the whole roster, and the reason to stop looking at gate structure. |
| M19 | **The gate keeps its spatial shape in all six** (Moran's I 70–86% of h1). The one family-level property found. | EXP-03 | **High** — 6/6. |
| M20 | ⛔ **CONFOUNDED — do not read as evidence about scheduled sampling.** The retention separation is real (SS-off {0.54, 0.45} vs SS-on {0.33, 0.21, 0.05, 0.02}) but **all four SS-on models were trained 2026-08-12/13, before the C-259 validator landed 2026-08-14 04:19**, with `ss_feedback` unset → defaulting to `'mean'` → an **ungated** mean field in training against a gated sample at inference. That is the mismatch C-259 forbids (C-300). Two further weaknesses: `violet_visitor` is a config outlier on **six** axes, and dropping it leaves SS-off at **n=1** with p 0.067 → 0.2; the tightest pair (`purple_alien` vs `pink_pirate`, differing only in ε and seed) is confounded by the same mismatch. **The only SS data with a correct `ss_feedback` is `truncated_smoke`'s sweep, which is floor-limited — so there is no valid, non-floored SS measurement anywhere.** | EXP-03 + config/date audit | **NOT A LEAD ABOUT SS.** A sweep with `ss_feedback='sample'` tests a different intervention and cannot settle it. |

**RETIRED by M18 — two of my own claims from the same day:**

* *"the model doesn't need to be nearly right, it needs to answer"* (the quiet-gate diagnosis, EXP-02).
  Commitment spans 24× and the model committing **fewest** cells retains **best**; `pink_pirate` commits
  342 cells (3× reality) and retains worst at 0.02. True of `violet_visitor`, false of the family.
* *"the gate loses confidence"* as a family property — `pink_pirate`'s is **stable** (1.2×).

**Also recorded:** four of six roster models have configs that fail C-259 validation and could not be
loaded at all (`ss_epsilon_max: 0.5` with `ss_feedback` unset → defaults to `'mean'`). All four are in the
shipped `rescore.csv`, so **those rows are currently un-rerunnable**. Fixed in the working tree only,
uncommitted, raised as **views-models#404**.

---

### RETIRED — kept so they stay dead

| Claim | Retired | Why |
|-------|---------|-----|
| "It is the **cell state** — `cell` carries 89% of the freeze effect" | 2026-08-16 | Architecturally predetermined: `hs = o ⊙ tanh(hl)`, so hidden is a *readout* of cell and freezing cell constrains hidden by construction. No arm in the set separates them (C-292). |
| "The state result quantifies the mediator — **CONFIRMED AND QUANTIFIED**" | 2026-08-16 | Overclaimed against its own pre-registration, and no naive baseline existed. Persistence then beat every arm (M1). Downgraded to I-D. |
| "**The recurrence diffuses the gate**" (C-295's hypothesis) | 2026-08-17 | Falsified by M6 using data already on disk. Also backwards on its own terms: smoothing *raises* spatial autocorrelation — it is how `correlated_bernoulli` manufactures clustering from white noise — so diffusion predicts Moran's I going **up**, and it goes **down**. |
| "The target is **the spatial coherence** of the occurrence field" (EXP-03's framing) | 2026-08-16 | Too loose; corrected by M5. Superseded by I-B. |
| "Moran's I falls **0.50 → 0.16** by step 6" | 2026-08-17 | Not a trajectory. Paired the **oracle's** 0.507 with the **free-running** 0.16–0.19 as though one run produced both. Real free-running trajectory is 0.409 → 0.192 (M6). |
| "gated_NB beats climatology at h36" | 2026-08-15 | ARTIFACT 4/4 — zero-driven; 74.6% of the gap is true zeros, ΔAP −0.030, `size_ratio` 0.0 (Epic #263). |

### The persistence re-reference, 2026-08-17 — and a correction to this table

The first version of this section said, flatly, *"no arm anywhere beats persistence at any horizon ≥ 6."*
**That was unqualified and therefore wrong** — the sentence this table exists to prevent, written into the
table itself on day one. On `crps_all`, the **FAO-02 primary metric**, every arm beats persistence at every
horizon: CRPSS **+0.20 / +0.45 / +0.41 / +0.11** at h1/6/18/36.

So the honest question is not "does it beat persistence" but "**is the win real**" — and this repo already
has an audited, pre-registered rule for exactly that (`scripts/rollout_ruler_core.py`, Epic #263, 86 tests).
Applied to the best arm against persistence on `sb`:

| h | CRPSS | ΔAP | zero-share of the gap | verdict |
|--:|--:|--:|--:|---|
| 1 | +0.197 | **+0.1519** | 69.9% | **UNDECIDABLE** — no CI computed |
| 6 | +0.446 | −0.0257 | 79.9% | **ARTIFACT** |
| 18 | +0.409 | −0.0165 | 82.4% | **ARTIFACT** |
| 36 | +0.111 | −0.0141 | 73.4% | **ARTIFACT** |

**The model's CRPS win over persistence is an artifact at every horizon from h6** — the same verdict Epic
#263 reached for gated_NB against climatology, now reached against the naive baseline as well. The win is
bought by predicting near-zero on the 99% of cells that are zero, while occurrence gets *worse*.

**h1 is the only non-artifact signal in the entire programme.** It reads UNDECIDABLE solely because
`ci_excludes_zero=False` was passed — no bootstrap CI exists — while both point estimates say REAL
(CRPSS +0.197, ΔAP +0.152). It must not be read as a negative.

**Method caveat, stated rather than buried.** Persistence is scored as a **1-sample** forecast
(`_persistence_gathered` → `np.array([last])`) with `gate_source = frac(samples>0)`, i.e. a **binary**
ranking, while the arms use the continuous `gate-head`. That violates Epic #263's own method rule (*the
reference's S must equal the arms' cube width*). **Both biases run the same way as the conclusion:**

- On **AP**, persistence is *handicapped* — a binary ranking cannot order within its active set — and it
  still beats the model 15× at h18. Correcting would widen the gap.
- On **CRPS**, the arms' `2/(m*m)` estimator is biased *upward* at S>1, so the model's CRPS win is
  *understated*. Correcting would make the win bigger — and the ARTIFACT verdict does not depend on the
  win's size, only on ΔAP and zero-share.

The conclusion is therefore robust to the S mismatch. A matched-S re-score would sharpen the numbers, not
overturn them.

### What is NOT established

- ~~**On occurrence (AP), no arm beats persistence at any horizon ≥ 6.**~~ **SETTLED 2026-08-21 — see M34.** A 300-lesson model beats a *fairly-scored* persistence at **every** horizon out to h36 (worst of 4 seeds 2.0–2.3×, gaps 2.1–4.3× MDE). The old bullet was true of the 40-lesson vehicle it was written from. **What follows is that the long-horizon rollout is no longer to be described as having no demonstrated value on AP** — it has one, on four seeds and one vehicle.
- **On `crps_all` the arms do beat persistence, and that win is an ARTIFACT** at h6/18/36 by the audited rule. **Unchanged** — M34 is an **AP** result and makes no CRPS claim; a 1-sample reference is the degenerate path `assert_sample_cube` refuses (C-220).
- **No result here is multi-vehicle**, and only M34/M37 are multi-seed (n=4). Positive findings at n=1 have historically evaporated on proper runs; the same standard applies to everything above, and M34 was held to it (EXP-03) after M8 was demoted on it (#280).
- **Nothing has been shown to fix the collapse *by intervention*.** Two inference-time interventions have been tried (copula M5, state freeze M8); neither reaches persistence. **Training to 300 lessons did** — but as M27 shows that is the model being better to begin with, not surviving its own output better. Retention itself is still stuck at 0.69.

### Verification pass, 2026-08-17

Every headline number above was recomputed from the committed CSVs rather than copied from the dossier
prose. Results:

| figure | verdict |
|---|---|
| persistence vs arms, all horizons (M1) | ✅ exact |
| freeze-arm AP ladder (M8) | ✅ exact |
| `crps_all` blindness, 4 arms at h18 (M9) | ✅ exact |
| oracle / scramble / free-running AP (M3) | ✅ exact |
| copula sweep AP flatness (M5) | ✅ exact |
| E4 decomposition **89% / 43% / 8%** | ✅ 88.6% / 42.6% / 7.9% |
| Moran's I "0.50 → 0.16" | ❌ **two arms quoted as one trajectory** — retired |
| `thin` activation "0.33 vs 0.27" | ❌ **two horizons quoted as one comparison** — corrected |
| "25× vs 2.6×" | ⚠️ true but the **peak**, not the range (4.4×→27×→14×) — restated as a range |

Three of nine headline figures were wrong or misleading, and **all three failed the same way**: values
pulled from different cells of the results grid and presented as a single comparison. None of the three
changed a conclusion — the effects are large enough to survive — but each was quotable, and two had already
propagated into the risk register. Registered as **C-298**.

One near-miss worth recording: "occurrence merely being **plausible** — 43%" looked wrong against
`spatial_scramble` (0.9%) until traced; "plausible" means `wrong_month:-60`, a *real* field from the wrong
month, and the figure is correct. **A verification pass produces false positives too** — the fix is to
trace the number to its arm, not to "correct" it on suspicion.

### The only work that is not measured against a collapsed control

Follows directly from the table above.

**1. Establish whether the h1 win is real — the single highest-value open question.** It is the only
non-artifact signal in the programme and it is currently UNDECIDABLE for a purely procedural reason: no
bootstrap CI was computed. Both point estimates say REAL. Needs a re-run (score-then-delete removed the
cubes) emitting **per-origin** scores so an origin-bootstrap CI can be formed, at matched S. Cheap,
decisive, and it either establishes the project's first genuine win over a naive baseline or removes the
last one.

**2. Stop treating the long-horizon rollout as having demonstrated value.** h6–h36 are ARTIFACT against
persistence. Any further rollout work should be justified as *research toward a future capability*, not as
improving something that currently works. This is a framing change, not an experiment.

**3. Make the collapsed control unusable as a reference.** Persistence and climatology should be emitted by
the scorer on every run, and the artifact verdict applied automatically, so no future dossier can report a
win against `none` without the naive comparison sitting beside it. Structural, cheap, and it prevents the
exact error this ledger was created to catch.

**Deferred until the above:** the top-K feedback arm, and every other inference-time intervention. They
compete to improve a rollout whose value over persistence is currently negative on occurrence and artifactual
on CRPS. Worth doing *after* there is a reference worth beating — not before.

### Standing rule adopted 2026-08-17

Cheap single-seed probes here have been **reliable at killing hypotheses and unreliable at establishing
them** — every retirement above has held, while the one positive claim (I-A's 89%) is the least
trustworthy line in the table. So:

1. Design cheap experiments to **eliminate**, and read a null as the informative outcome.
2. A **positive** result from a single-seed probe triggers **escalation** (≥3 seeds, second vehicle),
   never a conclusion.
3. A claim enters this table as MEASUREMENT or INFERENCE **before** it is argued from.

---

## Evaluation & selection criteria — adopted from FAO Pre-Release Note 05 (Topics C & D)

Source: `~/brain/2_projects/fao02/_dev_materials/prerelease_notes/fao_02_pre_release_note_05/`
(Topic C = model validity/selection; Topic D = ensemble construction). **We use these going forward.**

**Metrics** (cell–month level, temporal backtest protocol):
| Metric | Role | Better |
|--------|------|--------|
| **CRPS** | **primary ranking metric** | lower |
| **QS99** (99th-pct quantile score) | guardrail — tail sanity (catches *timid* models) | lower |
| **Brier** | guardrail — onset/hurdle probability calibration (`y>0`) | lower |
| **MCR** (mean pred ÷ mean obs) | guardrail — magnitude calibration | **closest to 1** |
| *Bounded?* (ours, C-113) | sanity gate — does the 36-step rollout stay in range or `expm1`-explode? | bounded |

**Eligibility (strict conjunction — Topic C.5):** a model is **Eligible** iff
1. **Superiority:** CRPS ≥ **5%** better than the baseline (C.2), *and*
2. **All guardrails non-inferior:** QS99 & Brier within **1%** of baseline; MCR calibration-distance `|MCR−1|` at least 1% closer-to-1 than baseline's (C.3/C.4).

Fail either → **Ineligible** (regardless of ranking). Ranking (CRPS order) ≠ admissibility (guardrails).

**Ensemble (Topic D):** eligible models only → **equal-weight mixture** of predictive samples →
**greedy forward selection** (seed = lowest-CRPS eligible; add the model giving the greatest
*strict* ensemble-CRPS reduction — no 5% margin intra-ensemble; stop when none strictly improves).

---

## Baseline reference

> The baseline that superiority/guardrails are measured against. (FAO uses an empirical heuristic
> baseline, Topic B; for the C-113 program our working reference is the SS-off clean run below until
> a formal baseline is set.) **TBD — set once the first clean eval lands.**

---

## Run ledger

Config fingerprint columns capture the *variation axes* (what we change between runs).
Metrics filled from `--evaluate`; *Bounded?* from the 36-step rollout / `diagnose_io_gain`.

| # | Date | Model · arch | Key params (variation axes) | wandb run | CRPS | QS99 | Brier | MCR | Bounded? | Eligibility | Notes (incl. non-wandb) |
|---|------|--------------|------------------------------|-----------|------|------|-------|-----|----------|-------------|--------------------------|
| R1 | 2026-06-07 | violet · HydraBNUNet06_LSTM4 | loss=tobit; **ss_epsilon_max=0.0 (SS OFF, pure TF)**; balancer=active(unfrozen); seeds 42/42; dropout 0.15; sigma{1.0,0.75,0.5}; onset_bias −7.0; log1p; total_lessons=80 | train `serene-plant-49` · eval `ro537u46` | os 0.04 ✓ / sb 0.7 ✓ / **ns 5.9e8 💥** | logged ✓ | logged ✓ | os 0.13 / sb 64⚠ / **ns 1.1e11 💥** | **NO — `lr_ns_best` explodes** | **Baseline (PATHOLOGICAL zero-point)** | C-113 reproduces on the clean SS-off baseline → explosion is NOT caused by scheduled sampling. **Head-specific**: `os` healthy, `sb` bounded but over-predicts ~64×, `ns` explodes (worst in posterior-sample-mean; robust CRPS for ns ~55–129 already ~1000× `os`). Active balancer + log1p. All 4 PRN-05 metrics emit (CRPS/QS99/Brier/MCR). This is the reference every fix is measured against. |
| R2 | 2026-06-07 | violet · HydraBNUNet06_LSTM4 | **sweep trial: dropout=0.10, lr=0.0005**; else = R1 baseline (tobit, SS off, sigma{1,.75,.5}, seeds 42, 80 lessons, active balancer, log1p) | sweep `ih3fc9u9` | os 0.04 ✓ / **sb 1.4e8 💥** / ns 50 ⚠ | logged ✓ | logged ✓ | os 2.0 / **sb 2.3e10 💥** / ns 3.0e4 ⚠ | **NO — sb explodes** | Diagnostic (sweep 1/9) | Explosion persists at lower dropout+lr. **Worst head SHIFTED to `lr_sb_best`** (R1 was `ns`) → runaway is not tied to one head; both sb & ns unstable, os consistently healthy. |
| R3 | 2026-06-07 | violet · HydraBNUNet06_LSTM4 | **sweep trial: dropout=0.10, lr=0.001**; else = R1 baseline | sweep `esa59shz` | os 0.04 ✓ / sb 0.16 ✓ / **ns 8.2e4 💥** | logged ✓ | logged ✓ | os 0.43 / sb 0.09 ✓ / **ns 3.8e7 💥** | **NO — ns explodes** | Diagnostic (sweep 2/9) | Explodes on `ns` (sb bounded & well-calibrated here). max\|metric\|≈1.8e9 — **less extreme than R1/R2 (~1e11)**. Pattern holding: *something* always explodes (sb or ns), magnitude varies, `os` always healthy. |
| R4 | 2026-06-20 | violet · HydraBNUNet06_LSTM4 | **post-#115 clean no-coords baseline** (refactored channel-role code); loss=**hurdle_nb** (bounded); n_posterior=8; static_channels=[] (no-coords floor); dropout 0.15; seeds 42; saved data (`-sa`), no `-re` | offline `20260620_162128-hhegi8bw` | sb 46.9 / ns 49.1 / os 49.7 (time-series CRPS) | sb 0.88 / ns 0.62 / os 0.63 (QS) | by_sb 0.69 / by_ns 0.71 / by_os 0.69 | sb 126 / ns 1552 / os 1077 | **YES — all finite, preds max ~120, NO explosion** | **Baseline (clean bounded reference for the coord experiment)** | First run on the post-#115 refactored code; all three census fixes hold in a real run (sidecar persists `static_channels: []` ✓). Hurdle-NB bounds the C-113 explosion (contrast R1–R3 ~1e11). Remaining pathology is **over-firing/miscalibration** (MCR ≫ 1 on ns/os: 1552/1077) — exactly the spatial signal CoordConv (#110) targets. This R4 is the no-coords anchor the coords smoke/comparison runs are measured against. |
| R5 | 2026-06-20 | violet · HydraBNUNet06_LSTM4 | **#110 CoordConv smoke** — R4 config + `static_channels=[row_coord,col_coord]`, `input_channels=5` (ADR-060/061 top-skip). One variable vs R4: coords on/off. Same total_lessons=40, n_posterior=8, dropout 0.15, seed 42, `-sa`, no `-re` | offline `20260620_164252-evzwoql4` | sb 49.5 / ns **79.4** / os 55.0 (CRPS — **all worse vs R4**) | sb 0.93 / ns 0.98 / os 0.71 (QS — worse) | by_sb 0.86 / by_ns 0.88 / by_os 0.84 (worse) | sb 134 / ns **2498** / os 1199 (worse) | YES — finite, bounded (ns pred mean 49→85) | **Diagnostic (first coord comparison — NOT a verdict; n=1/arm)** | **Engineering smoke PASS:** coords train+eval+reload with no crash → all three census fixes (C-157/158/159) hold with coords genuinely on. **Scientific result: coords HURT** — 12/12 metrics regressed vs R4, ns CRPS +62% (49→79), ns MCR 1552→2498, ns pred-mean 49→85 (MORE over-firing, opposite of hypothesis). Unanimous + large-magnitude, but **single seed/arm — no CI yet**; needs the #110 within-run-bootstrap decision rule (C-162) or multi-seed before a binding verdict. Prior now clearly negative for CoordConv-as-configured. |

**Status legend:** `Eligible` · `Ineligible` · `Baseline` (the reference) · `Diagnostic` (not a candidate, e.g. a probe) · `Failed` (run errored/exploded).

---

## What we know / wins (running narrative)

- **2026-06-07 — wandb training logging restored (C-132).** Single-run `-t` training now opens a
  `job_type="train"` run and logs per-lesson curves (was silently dropped by a stale phase-template
  override). Fix = delete the override; pinning test + fail-loud guard added. → all training results
  from here are observable. See `reports/2026-06-07_wandb_falsification/`.
- **2026-06-07 — clean baseline established (R1).** violet retrained with scheduled sampling OFF
  (`ss_epsilon_max=0.0`) = pure teacher forcing — the unfixed-model zero-point for C-113. Training
  healthy (loss↓).
- **2026-06-07 — C-113 REPRODUCED on the clean baseline (R1 eval).** Explosion is **head-specific**:
  `lr_ns_best` explodes (CRPS_mean ~6e8, MCR ~1e11), `lr_sb_best` bounded but over-predicts ~64×,
  `lr_os_best` healthy (CRPS ~0.04, MCR ~0.13). **Key inference:** the runaway is NOT caused by
  scheduled sampling (it was off) — it persists with pure teacher forcing + active balancer + log1p.
  The explosion concentrates in the **posterior-sample-mean** (a few runaway draws dominate), and is
  worst on the **non-state head**. Leads to chase: why `ns` specifically; the active-balancer (C-111)
  interaction; sample-mean vs robust-aggregate divergence. All 4 PRN-05 metrics confirmed emitted.

- **2026-06-08 ~00:40 — sweep progress + a data-availability flag.** 4/9 trials trained. R2 (0.1/0.0005, sb-explode) & R3 (0.1/0.001, ns-explode) confirmed via each run's *own* config — mappings correct. **BUT trials 3 (0.1/0.002) & 4 (0.15/0.0005) wrote train-only wandb summaries (13 keys, NO CRPS/MCR/eval)** — unlike trials 1–2 (88-key with eval). So their explosion status is **unreadable from wandb so far**. Open question for the morning: *does the sweep reliably evaluate every trial, or only some?* If sweep trials are train-only, the sweep tells us about training stability but NOT the rollout explosion (which is an eval-phase phenomenon) — meaning a sweep may be the wrong tool for the explosion question, and we'd need per-config `-e` runs. Will recheck at completion (eval may flush late) and do a careful full pass then. **Confirmed pattern from R1–R3 (the trials that DID eval): something always explodes (sb or ns), `os` always healthy.**

- **2026-06-08 ~01:09 — overnight sweep OOM-KILLED at trial ~5/9 (NOT restarted — would re-die).**
  Kernel log: `Out of memory: Killed process 2097902 (python) total-vm:33.7GB anon-rss:13.2GB global_oom`.
  **Mechanism — CORRECTED 2026-06-08:** my first claim ("RAM accumulates ~2.6 GB/trial across the
  sweep") is **RETRACTED — it's wrong.** Counter-evidence: `pink_pirate` (a *healthy* model) ran dozens
  of sweep trials over hours without OOM, so sweep trials DO free memory between them; a sweep does not
  use more RAM than the single run it's made of. The real distinguishing factor: **this model EXPLODES
  (C-113)**, and it died **mid-eval** (posterior sampling) where preds hit ~1e11 (`expm1`→inf). Leading
  (UNVERIFIED) hypothesis: the explosion balloons a single trial's eval memory past the limit — an OOM
  that is a **symptom of C-113**, not a sweep property. (R1, a single exploding run, survived its
  explosion, so it's not deterministic per trial — depends on blow-up severity.) NOT a CUDA wedge, NOT
  an in-process crash (external SIGKILL) — a true RAM OOM, cause not yet measured.
  **Salvaged (clean evals): R1, R2, R3 only** — trials 4–5 trained but their eval didn't complete/flush
  before the kill (they evaluate *after* training; sweep trials DO eval — earlier "train-only" was just
  mid-eval). **Sweep produced NO new eval rows beyond R1–R3.**
  **DECISION: not auto-restarted** — a fresh `-s` re-runs 1→9 and re-OOMs at ~trial 5 (infinite re-fail
  loop unattended). → registered as a risk (C-135). **Morning options:** (a) run trials individually
  (`-t -e` per config) so RAM frees between trials; (b) fix the cross-trial RAM accumulation (cf. ADR-047
  streaming `del`+`gc.collect`); (c) lower `n_posterior_samples`; (d) cap the sweep grid to ≤3–4 combos
  per process. The OOM is itself a finding: **the explosion isn't just a metric artifact — it inflates
  eval memory enough to kill the process.**

**Confirmed findings (3 clean evals, R1–R3): C-113 reproduces, SS-independent; the exploding head varies
(ns/sb), `os` always healthy; magnitude ~1e8–1e11; and the explosion is severe enough to OOM eval.**

- **2026-06-08 (pm) — collapse baseline + sweep-crash correction + magnitude-calibration dossier opened.**
  Today's **40-lesson** calibration run (`calibration_model_20260608_165326.pt`, eval `ffldgbxf`)
  **COLLAPSED**: MCR ≈ 0.002–0.03 (predicts ~0 everywhere); CRPS small (sb 0.13 / ns 0.05 / os 0.04) **only
  because ~95%+ cells are zero** — CRPS rewards the collapse. This is the **opposite** mode to R1–R3
  (80-lesson, explode) → framed as **two distinct failure modes**: collapse = zero-inflation reward;
  explosion = no rollout training. Verified (git): **no probability-coupled hurdle was ever implemented**
  (only C-45 ground-truth masking `aba45bc` + Tobit censoring `56194d2`; neither couples magnitude to
  predicted probability). New program → `reports/2026-06-08_magnitude_calibration_dossier/` (candidate #1 =
  minimal hurdle; judge on twCRPS+Coverage, MCR diagnostic only).
  **Sweep-crash correction (distinct from the ~01:09 RAM OOM above):** the 2026-06-08 *afternoon* sweep
  failures (`ikvegy30`/`3vidg2tr`/`4atlhfw0`/`x9zk3ujt`/`ksubmpw3`, ~14:3x) were a **CUDA `unspecified launch
  failure` — kernel `Xid 62` → `Xid 45`** at `model.to(device)` (`make()`), i.e. a **GPU-context fault**
  (one long-lived sweep process; trial 1 ran, then the context was poisoned — likely by a concurrent
  memory-intensive GPU job — so every later trial died identically). **Not** the RAM OOM logged at ~01:09;
  two different mechanisms. Fixed by `rmmod/modprobe nvidia_uvm`; a fresh single run trained + evaluated
  cleanly afterward. (This corrects/extends the ~01:09 entry: there were *two* distinct sweep-failure
  events on 2026-06-08.)

- **2026-06-09 — Arm-1 (hurdle, lognormal_nll, 40 lessons): magnitude UN-COLLAPSED one-step (directional); the rollout exploded.** *(Originally filed "FAIL → ZITD"; reframed 2026-06-09 — see the ⟳ note below.)*
  One variable vs the 40-lesson Tobit baseline (`loss_reg` tobit→lognormal_nll, sigma dict→0.9 scalar, +`hurdle_threshold=0`).
  Training completed HEALTHY; **eval FAILED — `views-evaluation` rejected the predictions ("Input contains infinity").**
  Verified from saved preds (`predictions_calibration_20260609_051916`, origin_0): **NOT a collapse — an autoregressive
  runaway.** Step-1 magnitudes are non-zero (sb 61 / ns 580 / os 91 — the hurdle *un-collapsed* the head vs baseline ~0.02),
  then the 36-step free-running rollout grows exponentially per step → expm1 → **INF by step ~13–15**. **Key finding:
  un-collapsing the magnitude head directly TRIGGERS the C-113 explosion — magnitude calibration and rollout stability are
  coupled; the head can't be fixed in isolation.** Per the binding stopping rule (dossier `2026-06-08_magnitude_calibration`)
  → **commit to ZITD/structural, no more loss tweaks.** ZITD is doubly motivated: its sub-exponential softplus link makes
  drift linear, dissolving exactly this expm1 explosion. Artifact `calibration_model_20260609_051916`.
  **⟳ Reframe 2026-06-09 (step-1 read; C-136 / M-R1 / M-R2):** the "FAIL → ZITD" verdict conflated two axes. A
  teacher-forced **step-1 `MCR_pos`** (rollout not engaged) shows the magnitude head moved **off ~0**: sb 0.11→**0.19**,
  ns 0.02→**1.29**, os 0.03→**0.73** vs Tobit. The explosion is purely the **untrained rollout** (Axis B). ⚠ **Direction,
  not values** — MCR is a diagnostic ratio (not a proper score); single-draw, 1 origin/seed, small n (131/59/50) → "un-collapsed"
  holds, "calibrated" does not. Proper-subset score + CI + 2nd seed = R4's readout (#93). **Revised next step:** hurdle +
  **rollout training** (cheap SS-middle probe → GTF), NOT a count-likelihood rebuild. Stopping rule intact (rollout ≠ a new loss).
  *(Numbers refined 2026-06-10 by the durable `scripts/mcr_readout.py` — see dossier `07`: the all-origins aggregate MCR is
  mean-dominated, the median positive cell still under-predicts, and sb barely moved. The step-1 figures above are the
  superseded origin-0 `/tmp` throwaway.)*

- **2026-06-10 — R4 (hurdle + scheduled sampling `ss_epsilon_max=0.5`, 40 lessons) → EXPLODED at eval (same as Arm-1).**
  The cheap rollout-training probe — SS-middle on the hurdle config, one variable vs Arm-1. Trained clean; eval failed
  "Input contains infinity". Durable readout (`scripts/mcr_readout.py`, `predictions_calibration_20260610_010843`):
  FULL-rollout MCR ≈ **3.4e33 / 3.1e33 / 7.5e33** (sb/ns/os) — indistinguishable from Arm-1; step-1 magnitude no better
  (sb 0.21→0.088). **Completes the scheduled-sampling bracket: 0.25→explode, 0.5→explode, 1.0→collapse — plain per-step
  SS is EXHAUSTED (proven, not assumed), even on the un-collapsed hurdle head.** → the real fix is **GTF / B1 rollout
  training (#78, cross-step gradients)** or the **count-likelihood head** (ZINB/hurdle-NB). Rollout dossier `07` EXP-02.

*(Append wins/lessons here as runs land — especially negatives and "looked-right-but-wasn't".)*
