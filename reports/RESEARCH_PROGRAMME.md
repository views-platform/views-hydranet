# Research programme — the ranked queue

**Purpose.** The standing list of what this programme is trying to find out, ranked, so nothing is
forgotten and we work **one epic at a time**. Agreed with Simon 2026-10-27 from his own four-item
list plus five additions.

**Horizon.** ⚠️ Simon said *"a couple of months left"* — read as **to end-2026, ~9 weeks**. If he
meant end-2027 this register is under-scoped and the horizon line is the thing to correct, not the
ranking.

**Last updated** 2026-10-27b · **9 entries** · **2 live (RP-1 #477, RP-3 #468)** · 6 queued · 1 parked (RP-9)

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

## ⭐ The queue, in order

| # | ID | challenge | status | cost | can it end in a pod run? |
|---|---|---|---|---|---|
| **1** | **RP-1** | The metric battery cannot see what we are trying to fix | 🟢 **LIVE** — epic **#477** | **CPU, hours** | no — it makes later ones readable |
| **2** | **RP-2** | No noise floor: run-to-run σ̂ has never been computed | 🔵 queued | **CPU, ~1 h** | no — same |
| **3** | **RP-3** | The family-free recalibration beats our emission by 15% | 🟢 **LIVE** — epic **#468** | **CPU, minutes** | no GPU needed; ships or dies on CPU |
| **4** | **RP-4** | Is the body actually under-dispersed, or is it the draw count? | 🔵 queued | **CPU, an afternoon** | no |
| **5** | **RP-5** | Both heads sit BELOW their own objectives and nobody knows why | 🔵 queued | CPU first, then **GPU** | ⭐ **yes** |
| **6** | **RP-6** | Magnitude ranking loses to persistence (0.303 vs 0.367) | 🔵 queued | **GPU** | ⭐ **yes** |
| **7** | **RP-7** | Nothing scores spatial dependence or aggregates | 🔵 queued | CPU | no |
| **8** | **RP-8** | Everything we know is month 1 | 🔵 queued | **GPU, large** | ⭐ **yes** |
| — | **RP-9** | Optimisation pathologies (plasticity, edge of stability, ridge) | ⛔ **PARKED** | — | — |

⚠️ **Why two pieces of enabling work sit above every experiment.** RP-1 and RP-2 are **hours of CPU
between them**, and the alternative is spending ~$2.25 and a day of pod time on an arm whose result
we cannot read — three of our instruments were measurably blind in the last two weeks, and we have
no noise floor at all. ⇒ They are ranked first because they are **cheap and they make everything
after them interpretable**, not because they are more interesting. **RP-3 runs in parallel** —
it is independent, already open, and needs no GPU.

---

## RP-1 · The metric battery cannot see what we are trying to fix

**Rank 1.** 🟢 **LIVE — epic #477.** CPU, hours. Stories **#478**–**#484**.

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

**The claim.** Every arm this programme has run is **n=1**, and we cannot distinguish a real effect
from seed variance.

- ⭐ **It is unrun, not blocked.** Three true seed replicates sit on local disk —
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

**Rank 3.** 🟢 **LIVE — epic #468.** CPU, minutes. Runs in parallel with RP-1/RP-2.

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

## RP-4 · Is the body actually under-dispersed, or is it the draw count?

**Rank 4.** 🔵 queued. CPU, an afternoon. ⭐ The cheapest entry that could change what a standing
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

**Rank 5.** 🔵 queued. CPU diagnosis first, then possibly GPU. ⭐ **The most interesting unexplained
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

**Rank 6.** 🔵 queued. GPU. ⚠️ Revisits a direction previously closed as a wall.

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

**Rank 7.** 🔵 queued. CPU. Partly covered by **#473** inside RP-3.

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

---

## §Closed

| ID | outcome |
|---|---|
| — | *(none yet)* |

## §Dropped

| ID | outcome |
|---|---|
| **epic #458** | ⛔ **Deliverable withdrawn 2026-10-27.** A quantile-body arm. Killed by **W-86** (hypothesis H refuted by arithmetic already in hand), **C-387** (the head cannot place the atom; the "overshoot" is the forced monotone fan), and the 2026-07-22 scope lock. ⭐ **Three stories stand as results**: #459 the estimator, #460 the reproduction + M68, #461 H refuted. Repointed to **#468** = RP-3. |

---

## Superseded

`reports/research_directions_issues.md` (2026-06-02) — a literature-derived list of seven quick
wins/mediums/larges, filed as issues #57–#63. Pre-dates the foundation ruler, the glossary and every
ledger. ⛔ Not maintained; do not work from it.
