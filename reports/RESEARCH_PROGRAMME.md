# Research programme — the ranked queue

**Purpose.** The standing list of what this programme is trying to find out, ranked, so nothing is
forgotten and we work **one epic at a time**. Agreed with Simon 2026-10-27 from his own four-item
list plus five additions.

## ⭐⭐⭐ The terminal objective — stated by Simon 2026-10-28

The queue below is the means. The end is a **paper**:

> use **synthetic forecasts mirroring known failure modes and virtues** — and therefore carrying a
> **known ranking** — to hone in on a **minimal set of evaluation metrics which together rank the
> synthetic forecasts correctly every time**.

⇒ **Fixing HydraNet is a real goal and not the only one.** Every entry in this queue that measures what
a metric can or cannot see is simultaneously a **paper artifact**, and is to be logged as results
arrive (**D-6**) rather than at epic close, because the paper needs the provenance and not just the
number.

**Where it is written up in full:** `reports/2026-09-30_evaluation_learning_report.md` **§9** — what of
it is already built (the eleven-forecast gated population and the 11 × 20 separation matrix, from
RP-1 / epic #477), ⛔ what is missing (**a declared ground-truth ranking — it does not exist; the only
ordering constraint in the harness is "the oracle ranks first"**), and the three constraints the
minimality claim must respect.

⛔ **The next piece of work on the paper is not a measurement.** It is Simon declaring the correct
ranking of the eleven synthetic forecasts, pair by pair, **before** any subset search runs. Two pairs
are already known to be contested by measurement: `crps_events` and `placement_rho` both rank the
mass-preserving blur **above** the delivered model, and `spiky-1/30` outranks persistence on
`placement_rho`.

**Horizon.** ⚠️ Simon said *"a couple of months left"* — read as **to end-2026, ~9 weeks**. If he
meant end-2027 this register is under-scoped and the horizon line is the thing to correct, not the
ranking.

**Last updated** 2026-10-28 · **10 entries** · **1 live (RP-3 #468)** · **2 measured out** (RP-1 — epic #477, **merged** as PR #485; RP-2 — σ̂ measured, **C-391**) · 6 queued · 1 parked (RP-9)

## How this works

1. We take the **top queued entry**, turn it into an epic (epic issue + story issues + tracking),
   and run it.
2. An epic may run **all the way to a pod run**, or be **dropped** when a gate kills it. ⭐ Dropping
   early is the intended behaviour, not a failure — #458 was dropped after three stories and the
   three stories are still results.
3. ⛔ **The ranking is expected to change.** When an epic closes, re-rank before picking the next
   one. Record the re-rank here with its reason.
4. An entry leaves the queue only to **§Closed** or **§Dropped**, with the outcome written down.
   ⛔ Never delete an entry.

## ⛔ RE-RANK 2026-10-28, after RP-1 and RP-2 both closed

Required by rule 3 (*"when an epic closes, re-rank before picking the next one; record the re-rank
with its reason"*). **The order below is UNCHANGED, and that is the finding** — but two entries
changed shape and one new candidate appeared, so the reasoning is recorded rather than the ranking.

**What changed:**

1. ⭐⭐⭐ **A NEW CANDIDATE, and it is the cheapest thing on this register: `RP-2b` — measure the
   noise floor of the other 18 battery columns on the three seed replicates.** RP-1 measured what
   the 20 columns can *discriminate*; RP-2 measured what 2 of them can *resolve*. ⇒ **18 columns
   have no measured noise floor**, and the cubes already exist on disk (M75), so this is a
   **readout, not a pod run** — hours of CPU, no new training. It is also directly the paper's
   second criterion (evaluation report **§9.4b**). ⚠️ **Not inserted into the ranking without
   Simon** — it is a new entry and the queue is his.
2. ⛔ **RP-5 got harder, and knowing that is the point.** RP-5 asks why both heads sit below their
   own objectives. **C-391 says the body-side placement readout cannot support a single-run
   comparison at all** ⇒ RP-5 must either use the gated out-of-sample readout or replicate its arms.
3. ⭐ **RP-6 keeps its new first question** (from RP-1): whether its recorded 0.303-vs-0.367
   magnitude-ranking ordering is population-dependent, since `placement_rho` gives 0.629 vs 0.428.
4. ⭐ **RP-4 is now doubly constrained before it starts**: 14 of 20 columns are blind to dispersion
   (RP-1), and the noise floor of the 4 that are not is **unmeasured** (RP-2 covered B-1a/B-1b, not
   the CRPS splits).

**Why the order does not move.** RP-3 is already live (#468) and nothing measured today bears on it.
Everything below it is an *experiment*, and both closed entries were *instruments* — the discipline
is instruments before experiments, so closing two instruments does not promote any experiment past
another. ⇒ **No re-ranking is warranted on the evidence; the next pick is Simon's between RP-3
continuing, the new RP-2b readout, and RP-4.**

## ⭐ The queue, in order

| # | ID | challenge | status | cost | can it end in a pod run? |
|---|---|---|---|---|---|
| **1** | **RP-1** | The metric battery cannot see what we are trying to fix | ✅ **MEASURED OUT 2026-10-28** — epic **#477**, all seven stories closed; ⛔ awaiting merge only | **CPU, hours — spent: ~7 min of compute across six scripts** | no — it makes later ones readable |
| **2** | **RP-2** | No noise floor: run-to-run σ̂ has never been computed | ✅ **MEASURED OUT 2026-10-28** — σ̂ exists for B-1a and B-1b; ⛔ **one readout's noise exceeds its signal** | **CPU — spent: 37 min compute** | no — it makes later ones readable |
| **3** | **RP-10** | ⭐ **The truth is not ground truth — we have never estimated the Bayes error floor** | 🔵 queued | **CPU, minutes** (route A) | no |
| **4** | **RP-7** | Nothing scores spatial dependence or aggregates | 🔵 queued | CPU | no |
| **5** | **RP-4** | Is the body actually under-dispersed, or is it the draw count? | 🔵 queued | **CPU, an afternoon** | no |
| **6** | **RP-5** | Both heads sit BELOW their own objectives and nobody knows why | 🔵 queued | CPU first, then **GPU** | ⭐ **yes** |
| **7** | **RP-6** | Magnitude ranking loses to persistence (0.303 vs 0.367) | 🔵 queued | **GPU** | ⭐ **yes** |
| **8** | **RP-8** | Everything we know is month 1 | 🔵 queued | **GPU, large** | ⭐ **yes** |
| — | **RP-3** | The family-free recalibration beats our emission by 15% | 🟢 **LIVE in parallel** — epic **#468** | **CPU, minutes** | no GPU needed |
| — | **RP-9** | Optimisation pathologies (plasticity, edge of stability, ridge) | ⛔ **PARKED** | — | — |

⛔ **RE-RANKED 2026-10-27b, and the reason is a new entry.** Simon raised that the input data is **not ground truth** — `ged_sb_best` is a sum of geocoded point events whose sharpness is partly an artifact of measurement, reporting and aggregation ⇒ there is an irreducible **Bayes error floor** and we have never estimated it. That became **RP-10**, entered at **rank 3**, and it pulled **RP-7 from 7 to 4** because if the fine-scale target is partly noise then **aggregates may be the only well-posed target**. RP-4, RP-5 and RP-6 each moved down one; **RP-3 is unranked and runs in parallel.** Full argument: `reports/2026-10-27_measurement_floor/ARGUMENT.md`.

⚠️ **Why the enabling work sits above every experiment.** RP-1 and RP-2 are **hours of CPU
between them**, and the alternative is spending ~$2.25 and a day of pod time on an arm whose result
we cannot read — three of our instruments were measurably blind in the last two weeks, and we have
no noise floor at all. ⇒ They are ranked first because they are **cheap and they make everything
after them interpretable**, not because they are more interesting. **RP-3 runs in parallel** —
it is independent, already open, and needs no GPU.

---

## RP-1 · The metric battery cannot see what we are trying to fix

**Rank 1.** ✅ **MEASURED OUT 2026-10-28 — epic #477, stories #478–#484 all CLOSED.** ⛔ The only
item left is pushing and merging the branch, which is Simon's. CPU; ~7 minutes of compute total.

> ### ⭐⭐ THE ANSWER, and the thesis below was CONFIRMED, not refuted
>
> **The battery is partially blind and three blind spots are TOTAL.** `RESULTS_LEDGER` **M69–M73**
> carries the measured record; five `FINDING_*.md` files carry the detail.
>
> - ⛔ **13 of 18 scoring columns give the worst constructible magnitude forecast the ORACLE's
>   exact score** — and **14 of 20** including `MCR_field` and `act_freq`.
> - ⭐⭐⭐ **14 of 20 columns are functions of the MEAN FIELD alone**, derived from the source and
>   confirmed at exactly `0.000e+00` ⇒ the ensemble's dispersion is visible to **exactly four**
>   columns, three of which are one CRPS split.
> - ⭐⭐⭐ A **100% permutation of the event-cell magnitudes**, placement untouched, moves six
>   battery columns by **exactly 0** and the new instrument by **1.03** ⇒ **#444 confirmed, as a
>   curve.**
> - ⛔ **`matched_pred_thresh` keeps its documented promise in 75 of 143 forecast-months (52%)**,
>   failing in three distinct ways — **C-388**, and the delivered model is affected by one of them.
> - ⛔ **Of FAO-02's three locked guardrails the frozen ruler computes exactly ONE as defined** —
>   QS99 absent, MCR present only as two statistics that are not it — **C-389**.
> - ⛔ **The plug-in CRPS charges 9 of 11 forecasts nothing and the BASELINE 3.4× the model** ⇒
>   **18% skill inflation at m=16**, in the direction that flatters us.
> - ⭐ **The instrument built:** `scripts/placement_rho.py`, 42 tests, two independent guard
>   audits, swept, with two measured ways to misuse it written into its own docstring.
>
> ⚠️ **Three of my own published claims were corrected in place** (`W-91`, `W-92`, `W-94`), each by
> a gate written to reject my own explanation. ⇒ The epic's real output is not the instrument; it
> is that **four separately-published numbers of ours were wrong in the same direction** — toward
> the conclusion already drawn.
>
> **Hands to the queue:** RP-4 inherits the four-column dispersion constraint · RP-6 inherits a new
> first question (its `0.303` vs `0.367` ordering **reverses** on this population) · RP-10 gained
> **two** measured points (`crps_events` and `placement_rho` both rank the mass-preserving blur
> **above** the delivered model) · ⛔ **Gini is dead as a candidate instrument** (`W-89`).
>
> ⚠️ **Still owed, none of it in this entry's scope:** #481's count of estimator-sensitive recorded
> comparisons (rule written, cubes mostly deleted); C-389's validation-partition guardrail pass;
> C-388 is registered, **not fixed**; and whether C-388's worst mechanism fires on the B-1a/B-1b
> populations is **unmeasured**.

⛔⛔ **The investigation changed what this entry is, and the change is the finding.** This is **not a design problem.** `reports/2026-08-13_evaluation_pitfalls_and_metric_battery_DRAFT.md` is **31 KB**, catalogues failure modes A–I, and its **§5** already designs exactly this method — *"treat the metric suite itself as software under test, and unit-test it with adversarial/degenerate forecasts whose true quality is known"* — naming **seven** forecasts, the acceptance rule *"a metric that cannot separate these is blind and must not be used for selection"*, and **three stated results**. ⛔ **It was never written.** A repo-wide grep for a degenerate-forecast harness returns nothing. ⇒ **C-383's shape, third time this week** — and one of the unverified claims is *"CRPS fails on sparse (all collapse together)"*, about **FAO-02's PRIMARY metric**. The draft also says the harness *"would have flagged the #258 defect on day one, on synthetic data, before any model run."*

⭐ **And reading the frozen ruler gave a derived inventory before any measurement:** of its seven columns, **AP** and **Brier** depend on the gate alone (⇒ blind to all magnitude, and rank-based AP is blind to gate *calibration* too); **size_ratio** uses `cs.mean(1)` and a **median** (⇒ blind to the ensemble's spread *and* to the tail); **pos_mcr** is `mean(ey/truth)` (⇒ weights each cell by `1/truth`, so it is **dominated by the smallest events**); and all three CRPS columns use the **plug-in** (`2.0/(m*m)`, `lodestar_score.py:33`), which Ferro2014 proves favours **under-dispersed** ensembles. ⇒ **Nothing in the frozen ruler is both magnitude-sensitive and unbiased toward the defect** — that is the thesis #477 measures. ⭐ The ruler documents one of these itself and nobody drew the consequence: `apply_threshold_gate`'s comment reads *"body-only; p (AP/Brier) unchanged"*.

⚠️ Two further gaps found by reading, to be confirmed not asserted: **QS99** is a locked FAO-02 guardrail and is **absent from the frozen ruler's output** (#482); and **AFSS** = **1.000000** under count-matching ⇒ zero information, while **hit rate** is **identically** FSS(n=1) (R-27) ⇒ a duplicate, not an independent column.

⛔ **What would end it early, and it is a real possibility:** if **#479**'s separation matrix shows the battery **does** separate all seven, the premise is wrong — three measured blind spots do not imply a uniformly blind battery, and `crps_events` is tail-weighted and may do most of the work. ⇒ The epic closes at M2, **#444** closes with it, and ⭐ **that is the most valuable outcome available**: our instruments would be sound and the defect entirely in the model. ⚠️ **Prior, stated before measuring so it can be scored: partial blindness, not total.**

**The claim.** Our instruments are blind to the defect we are hunting, and this was measured three
independent times in two weeks.

| instrument | how it was blind | source |
|---|---|---|
| **AP** | ranks on the **gate**. Returned **0.736228**, identical to 6 d.p., across all four cells of a 2×2 while the forecast was destroyed. Ranked on what is actually emitted: **0.516 → 0.371**. | M67(1), `lodestar_score.py:158` |
| **FSS** | thresholds at a **count-matched quantile** ⇒ blind to monotone distortion. Survived a field wrong by **+1.96% per cell**, sign flipping between windows. | **C-386** |
| **plug-in CRPS** | **manufactured a turning point** in a swept curve and understated an arm margin **13×**. | **M68**, #459, #442 |

⇒ And **#444** is already open: *"Nothing in the metric battery measures PLACEMENT of the emitted
magnitude — a change can invert it and every FAO-02 column stays quiet."*

**What an epic does.** Audit every column in the battery for **what it cannot see**, as a measured
property not an opinion: for each metric, construct the perturbation it is invariant to and show the
column not moving. Then build or adopt one instrument that **can** see magnitude placement, and
validate it against a known-answer perturbation.

**What kills it.** Nothing, realistically — worst case the audit finds the remaining columns are
sound, which is itself the answer. ⚠️ The real risk is scope: ⛔ do **not** let this become an
FAO-02 amendment. FAO-02 is **LOCKED**; this epic measures what the locked battery is blind to and
reports it. Changing the framework is Simon's call, separately.

**Blocks.** RP-4, RP-5, RP-6, RP-7 — every one of them is a claim made with these rulers.
**Blocked by.** nothing.

---

## RP-2 · No noise floor: run-to-run σ̂ has never been computed

**Rank 2.** 🔵 queued. CPU ~37 min + gates. Issues **#462**, **#463** already filed.

## ✅ MEASURED OUT 2026-10-28 — the answer, and it changes which instrument arms are judged on

**σ̂ now exists for B-1a and B-1b** (`reports/2026-10-28_sigma_hat/FINDING_RP2_SIGMA.md`, **M75**),
from three true seed replicates, with all eight gates passing.

- ⛔⛔ **B-1b body-alone: CV > 1 at 11 of 14 scales** — σ̂ exceeds the mean. At out/n=33 the honest
  bar is **2.1312 against a value of 0.2947**, i.e. **7.2× the quantity itself** ⇒ **no single-run
  comparison on it can ever clear its own floor.** **C-391**, **C-320** instance 6.
- ⭐ **B-1b gated out-of-sample IS usable**: CV **0.024–0.054** ⇒ **≈0.16 clears the bar.**
  **This is the readout RP-5 / RP-6 / RP-8 must judge their arms on.**
- ⚠️ **B-1a is weakest at d=1**, the setting its published headline uses: σ̂ **0.5073**, bar
  **2.5214**, on a quantity whose range is ~0.4–1.5.
- ⭐⭐ **σ̂ is not flat in the scale parameter and moves in OPPOSITE directions** across the three
  (falls, falls, **rises**) ⇒ a single scalar σ̂ would have been wrong both ways. Spans 7.9× / 18.9×.
- ⭐ **The spread is purely the seed**: training *and* scoring are both bit-deterministic (M74, G-9),
  so there is no measurement-noise term to subtract and no way to argue the floor down.

⛔ **Scope, binding:** `truncated_nb` @ dropout 0.15. The published B-1a 0.391 / B-1b 0.6267 are
`nb` @ 0.15 on a **deleted** checkpoint and the live `nb` control is @ **0.125**; family alone shifts
B-1a **3.7×**. ⇒ discharged for `truncated_nb` @ 0.15, **provisionally** for `nb` as a transported CV.
⚠️ **n=3, 2 df**: 95% CI **[0.52σ̂, 6.28σ̂]**. Two more seeds take the bar 4.97σ̂ → ~3.0σ̂ at ~17 min each.

## ⛔ AMENDED 2026-10-28 — part of this entry was WRONG, and it is corrected rather than overwritten

**σ̂ is NOT unrun.** `/home/simon/Documents/kestrel_sigma_backup_20261021/READOUT.md`, dated
**2026-10-21**, reports σ̂ for the **gate-calibration slope `b`** over seeds 42/43/44, with M1–M4 and
a pre-registration. ⇒ **W-83's shape a second time** — a quantity recorded as missing while the
artifact sat on this machine. ⚠️ The fix that worked was not vigilance: it was **reading the
directory before planning the compute**, four minutes, now the first step of any "never measured"
claim.

**What RP-2 step 0 measured instead** (`reports/2026-10-28_sigma_hat/`, **M74**):

- ⭐⭐⭐ **The pipeline is bit-deterministic.** `amber` ≡ `silver` ≡ `silver_0_1_4` at **190 of 190**
  state-dict entries, across git **73c3f42 vs 2e73de3** and views-hydranet **0.1.2/0.1.3/0.1.4**.
  Every other pair agrees on **15/190**, and those 15 are exactly the BatchNorm
  `num_batches_tracked` counters ⇒ **0 of 175 float tensors agree between any two seeds.**
  ⇒ **This entry's falsification question is ANSWERED: rerun variance is exactly zero, so σ̂ across
  seeds is pure seed sensitivity and the seed axis is NOT decorative.** ⭐ That also means the
  planned **mask-pairing gate G-3** is superseded by stronger evidence for *training* — though not
  for the *scoring* forward pass, so it is narrowed rather than dropped.
- ⛔⛔ **And it refutes the readout's choice of σ̂.** `READOUT.md` selects the same-version pair
  (**0.0002…0.0196**, df=1) over the as-launched family (**0.0351/0.0932/0.0434**, df=2) by arguing
  *"by elimination the excess sits on the 0.1.3/0.1.4 boundary"*. That boundary is **numerically
  inert**, so there is nothing to eliminate onto it ⇒ the pair is **the same estimate with the
  outlying seed dropped**, understating the floor **3×–200×**. See **C-390**.

⇒ **What is still genuinely unrun: σ̂ for B-1a / B-1b**, which is what #462/#463 ask for. The
2026-10-21 readout measures a **different statistic** (GLOSSARY §5e). ⛔ Never quote one against
the other — that collision is what produced W-83.

**The claim.** Every arm this programme has run is **n=1**, and we cannot distinguish a real effect
from seed variance.

- ⚠️ **It is unrun for B-1a/B-1b, not blocked** — ⛔ *the original wording "it is unrun" was wrong as
  written and is corrected above.* Three true seed replicates sit on local disk —
  `{silver,bronze,cobalt}_kestrel`, seeds **42/43/44**, identical configs but the seed, git `2e73de3`.
- ⛔ **W-83**: I told Simon σ̂ *"doesn't exist"* and it was the load-bearing fact in a strategic
  recommendation. He asked *"are you sure? have you checked?"* and the data was on this machine.
  The cause was a naming collision — `e1_sigma.py` measures **across-draw σ** (MC-dropout noise
  inside one checkpoint), which is a different quantity (GLOSSARY §5e instance 4).
- ⚠️ The estimator is **already locked**: pooled sample sd, `ddof=1`, **2 df** ⇒ 95% CI
  **[0.52σ̂, 6.28σ̂]**, a **12.1-fold** span, and the honest bar against one unreplicated arm is
  **4.97σ̂**. ⛔ Most arms this programme has run would not clear it.

**What an epic does.** #462 then #463. Route is established: `d4_mu_on_the_grid.py --tag=` per
checkpoint, **sequentially**, 738.8 s / 4.01 GiB each. Four gates, G-1/G-2/G-4 reproducing published
numbers from committed `.out` files.

**What kills it.** ⛔ **G-3, the mask-pairing gate.** `lock_entropy(42,42)` runs before
`load_vehicle`, so all three runs *should* see the same four dropout masks, making this a **paired**
design. But `lock_entropy` sets `use_deterministic_algorithms(warn_only=True)`, so non-determinism
can fall through. If the masks differ, σ̂ contains draw noise and B-1a moves 0.755 → 0.529 → 0.400
over n = 1, 2, 4 — **draw noise wearing a seed-effect costume. Hard stop and re-plan.**

**⛔⛔ The limit to write down before it runs.** No two rows share both keys:

| | family | dropout | status |
|---|---|---|---|
| published B-1a **0.391** / B-1b **0.6267** | `nb` | 0.15 | checkpoint **DELETED** |
| the live `nb` control | `nb` | **0.125** | live |
| the three replicates | **`truncated_nb`** | 0.15 | live |

B-1a on `truncated_nb` returned **1.462** against `nb`'s **0.391** — a **3.7× shift from family
alone**. ⇒ σ̂ discharges the noise-floor requirement for `truncated_nb` @ 0.15 and only
**provisionally**, as a transported CV, for `nb`.

**Blocks.** RP-5, RP-6, RP-8 — any arm. **Blocked by.** nothing.

---

## RP-3 · The family-free recalibration beats our emission by 15%

**Unranked — runs in parallel.** 🟢 **LIVE — epic #468.** CPU, minutes. Runs in parallel with RP-1/RP-2.

**The claim.** Keep the model's `gate × mu` **ranking**, throw its predicted **amounts** away, look
up what history did at each rank. Fit 421–456, score 457–469, 842,634 held-out cell-months:
**0.039026** fair against the best parametric arm's **0.046023** = **−15.3%**, and **−23.1%**
against what ships. **5.3 s, CPU, nothing retrained.**

⭐ Reproduced independently to **0.36%** from a script committed before it ran (#460, **M68**). ⭐ And
**better than published** — M67's turnover argument was the plug-in's bias; under the fair estimator
the curve is monotone to 32,000 bins and the optimum is **unlocated**.

**Stories.** #471 generalise (⛔ may kill) · #473 spatial/aggregate (⛔ may kill) · #469 the ceiling ·
#470 the background gap · #472 the extrapolation limit · #474 ship or kill · #475 tracking.

**What kills it.** #471 — the win is `sb`-only. Or #473 — the per-cell win costs spatial coherence,
which would be Epic #263's shape repeating.

⚠️ **Honest framing, which must appear in every write-up:** this is an **emit-time recalibration**,
the last seven failures were also emit-time, and it earns its place on a **measured win on the
primary metric at zero training cost** — not on being upstream. ⛔ The defect in ADR-073 §0 is
**untouched**. This works around it.

---

## RP-10 · The truth is not ground truth — we have never estimated the Bayes error floor

**Rank 3.** 🔵 queued. **CPU, minutes** for route A. ⭐ **New 2026-10-27, raised by Simon.** Full
argument: `reports/2026-10-27_measurement_floor/ARGUMENT.md`.

**The claim.** `ged_sb_best` at `(priogrid_id, month_id)` is a **sum of geocoded point events**.
Many of those events are geocoded to an administrative centroid rather than to where the deaths
occurred, and many carry month-or-coarser dates. ⇒ Aggregating them into a 0.5° × 1-month bin makes
a **spiky field by construction**, and some of that spikiness is **our ignorance of where and when,
recorded as though it were knowledge.**

⇒ ⛔ **CRPS against that field is proper for predicting the OBSERVATION, not the process.** A model
that correctly predicted latent conflict intensity would score **worse** than one that also learned
to mimic the measurement noise. There is an **irreducible CRPS** no model can beat.

⇒ ⛔⛔ **We have never estimated it**, which means **"the model scores 0.039" is uninterpretable** —
near-optimal or terrible, we cannot say.

**⚠️ A correction made before building on it.** I claimed in conversation that GED's `where_prec` /
`date_prec` would let us perturb the truth within its own declared precision. **True of GED as a
source, false of the data we hold:** `pull_land_all3_121_504.parquet` has exactly
`['ged_sb_best', 'ged_ns_best', 'ged_os_best']` and **no precision fields** — they are event-level
and were summed away. ⇒ Two routes, not equivalent:

| route | gives | cost |
|---|---|---|
| **A · declared swept kernel** — blur the truth at a stated radius, total preserved, sweep r | a floor **conditional on an assumed** precision ⇒ bounds the **shape**, not the level | **CPU, minutes.** Available now. |
| **B · event-level GED** — perturb each event within its own coded precision, re-aggregate | the **actual** floor the data declares for itself | needs an event-level pull that does not exist here |

⛔ **Do not report A's number as "the floor."** Report it as *"the floor under an assumed blur of
radius r"*, with r swept (D-4).

**⭐⭐⭐ Why it ranks this high: it reframes D12, our most-quoted spatial result.** The body carries
**0.391** of its due spatial structure at **1 cell** and **0.870** at **32**. We have read that one
way — *"the model fails at fine scale."* The second reading is that **the fine scale is partly
unlearnable because at 1 cell we are scoring measurement noise.** ⛔ The two have never been
distinguished **and they imply opposite actions**: fix the model, or change the target. ⇒ The floor
**per spatial scale** settles it.

**⚠️ The bound, from our own record.** `thin_0.75` keeps **97%** of the oracle after deleting 25% of
events while **scrambling placement costs 81%** ⇒ if locations were mostly artifact, scrambling could
not cost 81%. **Placement carries real signal**; this argument is about **sharpness**, not about
whether location matters. ⭐ Compatible at different scales — which is D12's profile exactly.

**⛔ And the half that cuts the other way, which is the contribution.** Location and date error
**move** mass; the **total is preserved**. Under-reporting makes the observed total an **undercount**
⇒ the latent total is **≥ 50,881**, while the delivered model emits **16,612**.

⇒ ⛔⛔ **Measurement error excuses the smoothing and makes the mass deficit WORSE.** ⇒ The diagnosis
**splits in two**, and the split is the point:

| question | status |
|---|---|
| *is the field too smooth?* | ⚠️ possibly defensible, possibly even correct |
| *is the total one third of truth?* | ⛔ **not defensible by any measurement-error story** |

⚠️ These have been carried as one defect ("under-firing"). **They are not one defect.**

**⛔ What it does not touch.** `size_ratio` saturating is **arithmetic**; the plug-in bias is
**arithmetic**; AP ranking on the gate is a **code path**; C-387 is a **derivation**. Those are
properties of the **instruments**; this is a property of the **target**. Neither excuses the other.

**What an epic does.** Route A per spatial scale, swept; then route B if A says the question is live.
⭐ **Started already, inside RP-1:** an eighth red-team forecast — blurred truth, total preserved —
because ⛔ **if blurred truth scores WORSE than raw truth on our columns, the battery actively
penalises the forecast this argument says may be right.**

**Blocks.** RP-4, RP-5, RP-6, RP-7 — every magnitude claim is read against this floor.

---

## RP-4 · Is the body actually under-dispersed, or is it the draw count?

**Rank 5.** 🔵 queued. CPU, an afternoon. ⭐ The cheapest entry that could change what a standing
belief means.

**What is measured.** The body **under-fires**: emitted positive fraction **0.0009** against an
observed **0.0034** = **0.26× of truth** (M66); size-ratio **~0.02** (timid).

**What is not.** Whether its *ensemble* is under-dispersed, or whether we are failing to sample it.
At `θ = 3.6e-06` the body's nominal variance is `mu + mu²/θ` — enormous — but the realised
distribution is **mostly-zero with a rare heavy branch**, and ⛔ **at D=4 draws we essentially never
sample that branch.**

⇒ ⭐ **Head draws are free and uncapped** (W-68, #459). Sweeping `K` may dissolve "the body is
under-dispersed" without retraining anything — or confirm it, which is equally useful.

⛔ **And there is a loop to break.** Ferro 2014 proves the **plug-in CRPS estimator favours
under-dispersed ensembles**, and the record was scored at **m=16 plug-in**. So part of this
"pathology" is **our own ruler selecting for it**. RP-1 and #459 are the other half of this entry.

⚠️ **Do not conflate with the gate.** The **gate** has a measured dispersion defect — slope **b > 1**
in all six fits (1.146 ctrl / 1.419 arm), logits compressed. The body does not have that measurement.

**What an epic does.** Sweep `K` on a frozen cube, report the body's ensemble spread, firing rate and
CRPS as functions of draw count, under **both** estimators. State plainly whether "under-dispersed"
survives at converged `m`.

**What kills it.** It cannot be killed, only answered. ⚠️ The risk is that the answer is "yes, still
under-dispersed at m=1024," in which case the entry closes having removed a confound and RP-5 inherits it.

**Blocked by.** RP-1 (lightly — it uses CRPS, which RP-1 is auditing).

---

## RP-5 · Both heads sit BELOW their own objectives and nobody knows why

**Rank 6.** 🔵 queued. CPU diagnosis first, then possibly GPU. ⭐ **The most interesting unexplained
fact in the record** — the loss-balance method review's own words.

**The claim.** Each head stops short of its own best achievable loss:

- the **gate** sits below its BCE optimum (P2's verdict **D2**)
- the **body** sits below its NLL optimum — M6's `θ̂` **0.154** against a conditionally-honest **0.432**

⇒ Hard to explain by an allocation story, and **nobody has explained it.**

**⛔ What this entry is NOT.** It is **not** a hunt for loss of plasticity, edge of stability, mode
connectivity or a parameter ridge. ⛔ Those are **RP-9, parked, with zero evidence.** We already have
a *derived* explanation for one half of the apparent non-convergence, and it must be ruled out first:

> **ADR-073 §0.** The NB NLL of a zero is `θ·log(1+μ/θ)`, **monotone increasing in `θ`** ⇒ on an
> empty cell the optimum is `θ → 0` at **every** `μ`, with **no interior optimum**. 99.66% of cells
> are empty. ⇒ **`θ = 3.6e-06` is the objective's correct answer**, not a convergence failure.

⇒ **We do not fail to reach the minimum in `θ`; there is no interior minimum to reach.** The real
question is the **residue**: why does each head stop short on the parameters that *do* have an
interior optimum.

⚠️ **The precedent for why the exotic mechanisms wait.** M46's WideMemory null turned out **not** to
be vanishing gradients (BPTT reach is real — **1.6e-02** trained vs **2.8e-17** at init), and
**C-325** records two mechanism tests that measured the network **at initialisation** and were filed
as ruling a mechanism out. ⛔ Reaching for a sophisticated mechanism before ruling out a derived
mundane one is the shape that has cost us most.

**What an epic does.** Quantify the gap per head on a frozen checkpoint against a properly-computed
conditional optimum; separate the `θ`-boundary contribution (derived, expected) from the residue;
then, and only if a residue survives, design one measurement for it.

**Blocked by.** RP-1, RP-2, RP-4.

---

## RP-6 · Magnitude ranking loses to persistence

**Rank 7.** 🔵 queued. GPU. ⚠️ Revisits a direction previously closed as a wall.

**The claim, measured.** On **magnitude**, spearman **0.303** against persistence's **0.367**,
confound-clean (`project_amount_ceiling_wall`). ⇒ A dumb baseline ranks *how big* better than we do.

⭐ **And the half Simon's framing got right that I should not blur:** on **occurrence** we win — at
L=300 the *worst* of 4 seeds beats a fairly-scored persistence at **every** horizon (h36 **2.22×**),
and the gate is calibrated (median |gate−p| **0.0054** over 842,634 cell-months).

⇒ **It is a conversion failure, not a ranking failure** — which is exactly what RP-3 demonstrates
from the other direction: a lookup on *our own ranking* beats *our own amounts* by 15%.

**⚠️ Why this was closed and why it may reopen.** `project_amount_ceiling_wall` declared the amount
ceiling **irreducible** — *"data can't say HOW BIG a jump is."* Then Epic #230 measured the 2-NB
mixture null and found component 2 is **alive** (median w|act 0.71–0.92, μ2:μ1 14×–690×) while
magnitude stayed capped ⇒ the gap is **tail SHAPE (ξ>0), not mean-decoupling**. And
`project_volatility_ceiling_predictable` says jump **risk/spread** IS predictable (S2 0.79 vs 0.39;
quantiles 48% sharper and calibrated; tail ξ≈0.8).

⇒ ⛔ **Three prior results point in different directions and none of them was scored on the frozen
ruler with a noise floor.** That is the reason to reopen — not a new idea.

**What kills it.** RP-3 closing with the recalibration shipping would substantially *answer* this
entry without an arm. ⇒ ⭐ **Re-rank after RP-3.**

**Blocked by.** RP-1, RP-2, RP-3.

---

## RP-7 · Nothing scores spatial dependence or aggregates

**Rank 4** (up from 7, on RP-10). 🔵 queued. CPU. Partly covered by **#473** inside RP-3.

**The claim.** Everything we emit and everything we score is **per-cell and independent**.

- Measured: the body carries **0.391** of its due spatial structure at 1 cell and **0.870** at 32
  (D12). ⚠️ **11.2%** of the body field's spatial variance is our own 4-draw MC noise, and it is
  spatially correlated.
- ⛔ **Nobody has ever scored a country-month or region-month total** — the audit-facing number.
- Raised by Simon 2026-10-27 against the quantile head; ⚠️ it applies equally to the NB head and to
  the RP-3 lookup, so it does **not** discriminate between them. It is a property of per-cell
  emission in general.

**What an epic does.** Score aggregates at the scales the product is consumed at, and measure whether
per-cell improvements survive aggregation. ⛔ **Never run `d12_variogram.py` without `--tag`** — it
overwrites the irreplaceable curves for a **deleted** checkpoint.

**Blocked by.** RP-1. ⚠️ **Re-rank after #473** — if #473 answers it for the lookup, this entry
narrows to "and for the delivered model."

---

## RP-8 · Everything we know is month 1

**Rank 8.** 🔵 queued. GPU, large. ⚠️ Deliberately last.

**The claim.** The 36-step rollout explodes (**C-113**) and the response was to **score T=0 only**.
⇒ Every result in this programme is about **the first forecast month**. The product is **36 months**.

- The bloom itself is **fixed 9/9** by sample-feedback (Epic #193, ADR-070 active).
- ⛔ But *"the bounded rollout has NO magnitude skill"* — crps_events tied, AP below climatology by
  ~h6.
- ⛔ **Never pool the 36-step rollout into a T=0 read** — the trap that produced the retracted M1
  verdict.

**Why last.** It is the largest, most expensive entry, and every one of RP-1 … RP-7 improves our
ability to read its result. ⚠️ But it is also the only entry that bears on **what we actually
deliver**, so if the horizon is end-2027 rather than end-2026, it moves up.

---

## ⛔ RP-9 · Optimisation pathologies — PARKED

Loss of plasticity · capacity loss · implicit regularisation · edge of stability · mode connectivity ·
parameter ridge.

**Status: parked, with zero evidence for any of them.** Proposed by Simon 2026-10-27 as candidate
causes of apparent non-convergence.

⛔ **Why parked rather than queued.** The apparent non-convergence has a **derived** explanation for
its largest component (ADR-073 §0 — no interior optimum in `θ` on 99.66% of cells), and the measured
residue is RP-5. Investigating six exotic mechanisms before quantifying the residue is the C-325 /
M46 shape, and that shape has cost this programme more than any other.

**Unparks when:** RP-5 quantifies a residue that the `θ` boundary does not explain, **and** points at
one of these specifically. ⇒ Then it becomes a story inside RP-5, not an epic.

---

## Standing discipline — rules, not epics

These are not queued because they are not research. They apply to every entry above.

| | rule | why |
|---|---|---|
| **D-1** | ⛔ **Every published number has a committed script that produced it.** | **C-383.** Hit **twice in three stories** this week: M67 had no script (recoverable, #460); M1f/M1g had no artifact *and* no captured output (**not** recoverable, #461). ⇒ We cannot audit our own results. |
| **D-2** | ⛔ **Every number carries population + transform + estimator + whether normalised.** | **C-351**, GLOSSARY §5e. Twelve instances. |
| **D-3** | ⛔ **No new emitting family.** NB + ZINB only (scope lock 2026-07-22). | ADR-073 §0 forecloses the class; **C-387** disqualified the quantile head structurally. Two heads, one cause: neither can place an atom where 99.66% of the mass lives. |
| **D-4** | ⛔ **Sweep the free parameter before reporting a scalar.** | Three claims withdrawn in one programme; **M68** is the fourth — a bin-count sweep manufactured a turning point. |
| **D-5** | ⛔ **Measurement scripts are committed BEFORE they run.** | The only reason #460 and #461 are trustworthy. |
| **D-6** | ⛔ **Log, document and report as insights arise — not at the end.** An argument that exists only in a chat turn is lost to the next context window. | Simon, 2026-10-27: *"make sure you log and document and report frequently so the insights or thought we produce are not lost or diluted or truncated."* ⭐ RP-10 exists because this rule was applied the moment he raised it. |

---

## §Closed

| ID | outcome |
|---|---|
| **RP-1** | ✅ **MEASURED OUT 2026-10-28**, epic #477, seven stories. The thesis was **confirmed**: the battery is partially blind, three blind spots are total, and a 100% magnitude scramble leaves six columns bit-identical. Byproducts: **C-388**, **C-389**, C-303→23, `W-87`–`W-94`, `R-29`–`R-34`, `RESULTS_LEDGER` M69–M73, one new instrument, and locked `MCR`/`QS99` glossary rows. ⛔ Awaiting merge only. |

## §Dropped

| ID | outcome |
|---|---|
| **epic #458** | ⛔ **Deliverable withdrawn 2026-10-27.** A quantile-body arm. Killed by **W-86** (hypothesis H refuted by arithmetic already in hand), **C-387** (the head cannot place the atom; the "overshoot" is the forced monotone fan), and the 2026-07-22 scope lock. ⭐ **Three stories stand as results**: #459 the estimator, #460 the reproduction + M68, #461 H refuted. Repointed to **#468** = RP-3. |

---

## Superseded

`reports/research_directions_issues.md` (2026-06-02) — a literature-derived list of seven quick
wins/mediums/larges, filed as issues #57–#63. Pre-dates the foundation ruler, the glossary and every
ledger. ⛔ Not maintained; do not work from it.
