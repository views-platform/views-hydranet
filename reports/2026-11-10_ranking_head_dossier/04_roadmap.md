# 04 — Roadmap: from the loss to the one number

**Serves:** the *"how do we get there"* half — the gated sequence from an unwritten loss to the
finish line of `00 §⭐⭐`, cheapest-first, with a decision point at the end of every phase.

Written 2026-11-10. ⛔ **The governing constraint is the handover's:** *"Build the cheapest B that
can exist, and run it."* Every phase below is ordered so the cheapest thing that could change the
answer happens first.

---

## 1 The phases

### Phase 1 — the loss, and only the loss  ·  CPU, hours, nothing trained

**Build** the level-indexed listwise loss of `02 §3` with the dyadic transform of `02 §2`, as one
entry behind a default-off flag.

⛔ **The numerics are tested before anything trains.** These are the tests, and each needs a half
that fails against the defect — `03 §3.5` says why:

| # | the property | the defect it must refuse |
|---|---|---|
1 | `r = ceil(log2(1+y))` maps `y = 681 → r = 10`, `w = 1023` | raw counts: `w` is `inf` in float32 |
2 | `|R|` is the **a-priori** level set, fixed across folds | `|R|` read from observed levels (`C-423`) |
3 | an all-one-level list gives loss **exactly 0** | a non-zero loss on an unrankable list |
4 | the lowest level is the dropped final step | the zero block contributing gradient |
5 | an empty level contributes nothing and does not change `|R|` | level 9 being silently skipped |
6 | the criterion returns a **0-dim** tensor | a non-scalar that broadcasts silently (`03 §3.3`) |
7 | every weight is finite for every `y` in `[0, y_max]` | ⛔ `C-420`, the Tier-1 case |

**Gate to phase 2:** all seven green, plus the full suite and lint. ⛔ **If the loss cannot be made
to satisfy 3 and 4 simultaneously, that is a finding about Eq. 2 at our sparsity, and the reserve
option (`01 §3`, Blondel's differentiable sort) becomes the next thing tried** — not a verdict on B.

#### ⭐ Phase 1 is DONE · 2026-11-10 · gate PASSED

`views_hydranet/utils/listwise_level_loss.py` (`ListwiseLevelLoss`, registry key `listwise_level`,
config field `loss_reg_y_max`). `tests/test_listwise_level_loss.py`: **23 passed** — the seven
properties above, each with its refusing half, plus the registry entry, `needs_latent`, the empty
list, and the three config facts below. Full suite **3,036 passed**, 12 skipped, 34 xfailed;
`ruff check` and `format --check` clean over 371 files.

**Default-off is a measured property, not a docstring.** `loss_reg_y_max` defaults to `None`, so
every existing config is unchanged; the loss is reachable only by naming `listwise_level` in
`loss_reg`, which no config in the fleet does. Two refusals come for free and are asserted:
declaring the loss **without** `loss_reg_y_max` fails `validate_loss_reg_params` — the level set is
the objective's scale and a default would be a silent per-run objective (`C-423`) — and
`body_supervision='active'` fails `validate_body_supervision_latent` (`C-193`), which is the right
answer here for a new reason: the zero block **is** the list's lower levels, so masking it away
leaves a one-level list and a loss of exactly 0.
⚠️ `docs/CICs/HydraNetConfig.md` §3 moved 105 → **106** fields with it.

**Properties 3 and 4 hold together**, so the reserve option is not reached. Three implementation
notes, each one a measurement rather than a claim:

1. ⭐ **The dyadic level is computed by integer comparison, never a float `log2`.** Every exact
   power of two is a level boundary and the `expm1(log1p(y))` round trip leaves a count a few ULP
   off it, so `ceil(log2(1 + y))` in floating point returns the next level up: measured,
   `ceil(log2(1 + 1.0000001)) = 2`. The edge table `[2^0 - 1, …, 2^r - 1] = [0, 1, 3, 7, …]` with
   `searchsorted` is exact over the whole range.
2. ⚠️ **Property 4 is implied by the objective, not enforced by the code — found by mutation.**
   Giving the lowest level a step of its own, with a gain of 1 instead of `2^0 - 1 = 0`, leaves all
   20 tests green. That mutant is *equivalent*: the lowest level has nothing below it, so `s̃_t` is
   empty, `P_t(d) = 1`, and the step contributes exactly 0 whatever weight it carries. ⇒ Zhu's
   truncation at step `|R|` is a property of the objective at our data, and `02 §2.4`'s *"handled
   twice over, by construction"* is now verified rather than argued.
3. ⭐ **Seven further mutations were applied to the loss and all seven were caught**: the normaliser
   from the occupied levels (`C-423`, 6 failures) · a float `log2` transform (6) · the gain `2^r`
   for `2^r - 1` (4) · `s̃_t` as the levels *above* (6) · `s̃_t` including `c_t` itself (7) ·
   dropping the *highest* level instead of the lowest (2) · removing the `C-420` refusal (1).
   ⚠️ Mutations were applied against a scratchpad copy and restored by checksum, not by
   `git checkout` — the mechanism that has eaten uncommitted work here twice.

⚠️ **What phase 1 does NOT establish**, stated here rather than left to be discovered: the loss has
never seen a gradient step inside the training loop. `03 §3.1`'s dispatch question is untouched and
is phase 2's whole content; the `n_params = 1` score currently reaches the criterion through the
legacy per-target channel path, which is *where* phase 2 has to decide, not a decision already made.

### Phase 2 — the seam  ·  CPU, hours

Answer `03 §3.1`: either the score fits the family shape at `n_params = 1`, or the dispatcher is
widened deliberately. Then **prove inertness**: with the flag off, the baseline is byte-identical
against `tests/distributions/test_parity_anchor.py`'s golden — asserted, not claimed in a docstring.

**Gate to phase 3:** the parity anchor passes with the new code present and the flag off.

#### ⭐ Phase 2 is DONE · 2026-11-10 · gate PASSED

**The dispatch question of `03 §3.1` is answered, and the answer is *neither* of the two options it
offered.** The score needs no family shape and the seam needs no widening: the legacy point branch
at `training_engine.py:923` — `pred_j = t1_pred_for_loss[:, j, :, :]` — already slices **one channel
per target** and hands the criterion `[B, H, W]`, which *is* the `n_params = 1` score. With
`needs_latent = True`, `t1_pred_for_loss` is `output.reg_latent` (`:847`), so the score arrives
**pre-activation**, on the whole real line, which is what a scale-free score requires.
⇒ **`ADR-067`'s registry and the `isinstance` dispatcher are both untouched.**

⛔ **Asserted against the real `_process_sequence`, not reasoned about.** That is the `#308` /
`C-324` lesson: a change with 7 unit tests, 5/5 mutations and a green suite was a **no-op on the
path production runs**, and two arms trained to byte-identical weights over 276 minutes. Four
integration tests now drive the real loop with the real dispatcher and a spy criterion, and they
assert: the criterion is called with `[1,H,W]` · it sees **negative** scores, so it is being fed
`reg_latent` and not the ReLU-activated field · the model's gradient is finite and **non-zero** ·
and it is **exactly zero** on a single-level patch, which is the refusing half — the gradient
vanishes precisely when there is nothing to rank, which no per-cell magnitude loss does.
Three further mutations were applied and all three caught: `needs_latent = False`, an inert zero
return, and a non-scalar return.

⭐ **Inertness, measured:** `tests/distributions/test_parity_anchor.py` — **6 passed**, byte-exact
over all six legacy `output_distribution` values, with the new code present and the flag off. The
registry's 14 pre-existing entries are asserted unchanged **by class**, not by count, so an edited
`cls` on an existing key cannot hide behind a stable length. Full suite **3,043 passed**, 12
skipped, 34 xfailed; `ruff check` and `format --check` clean over 371 files.

⭐ **One property worth having found: the objective is shift-invariant.** `f -> f + c` multiplies
Eq. 1's numerator and denominator by `e^c`, so the loss cannot read the score's level at all — only
its order and its spacing. Asserted over four shifts, with MSE as the refusing half. That is the
mechanical difference between a ranking body and a magnitude body, in one line of test.

⛔⛔ **And one hazard, which phase 2 exists to catch and nearly did not.** The loss reads the latent,
so **training** is safe under any activation. **Emission is not.** `output_distribution='standard'`
resolves the head's activation to **ReLU** (`HydraBNrecurrentUnet_06_LSTM4.py:125-127`), and ReLU
maps the whole negative half-line to exactly 0 — **re-creating the atom at zero that `02 §5`
measures as `tie@0` and that this entire programme exists to remove**. `reg_activation='softplus'`
is strictly increasing: it preserves the ordering element for element and introduces no tie, both
halves measured. ⇒ **A ranking arm sets `reg_activation: 'softplus'`.** ⭐ **Ratified by Simon, 2026-11-10**, so it
is a constraint on every arm this programme builds rather than a preference: an arm whose config
omits it is mis-built, and `03 §5`'s box says so.
⚠️ Recorded as a RED box and as a test, **not** as a config validator — `C-411` already counts ~50
validators and 15 `reject_*` guards in `HydraNetConfig`, and this is a config-authoring choice
rather than an invalid combination. ⇒ The thing that catches an omission is the pre-flight box,
read before a pod is paid for; if that proves too weak in practice, the validator is one line.

⚠️ **What phase 2 does NOT establish:** nothing has been trained, and `03 §3.3`'s engine-level hole
— nothing asserts a criterion reduced to a scalar — is closed for *this* loss only and remains open
for anything else the dispatcher routes.

### Phase 3 — ⭐⭐ the cheap readout that bears on the finish line  ·  CPU/GPU, one short train

**The question:** does a patch-trained ordering score sort the **global** grid?

`02 §4.3` names this as the one real cost of the patch-local list, and it is answerable long before
convergence. Train the ranking body briefly — far short of the 300-lesson plateau — emit the score
on the global grid, and measure **`ρ_event`** against the body's `+0.3127 ± 0.0713`.

⭐ **Why this is the right cheap probe:** it is in the units of the finish line. It cannot settle the
programme, but it is the only measurement available before convergence that could move the estimate
of whether convergence is worth buying.

**Read-out order:** `ρ_event` global · then `ρ_event` within-patch · then the gap between them. The
gap **is** the quantity `02 §4.3` worries about, and it is a number rather than an argument.

**Gate to phase 4:** a score field exists on the global grid and `ρ_event` is computable on it.
⚠️ **Not** a go/no-go on the value of `ρ_event` — phase 3 is an instrument check and an estimate,
and `00 §6`'s kill rule applies to it as to everything else here.

#### ⛔⛔ PRE-COMMITMENT, written 2026-11-10 BEFORE phase 3 runs — read before citing its number

**Phase 3's `ρ_event` is an instrument reading. It may not be cited as evidence for or against B in
any verdict.** A verdict uses only a measurement taken against `05_analysis_plan`'s pre-registered
criterion, and `05` does not exist yet — it is blocked on `00 §5` items 1 and 2.

⇒ The reason this line exists, stated plainly: phase 3 produces a number **in the units of the
finish line**, before the rule for reading such a number has been written. Whoever writes that rule
afterwards knows the answer — which is exactly how `MCB` became unusable here, measured
(`MCB_A` 0.138 vs `MCB_B` 2.952) before it was proposed, against `C-3531`. Running the check anyway
is a deliberate trade: ⭐ it answers a **plumbing** question — *does a score trained on 32×32 patches
sort the 360×720 grid at all, or only within a patch?* — whose answer matters whatever `05` decides,
because a dead end there is a dead end under any criterion.

**What phase 3 may conclude:** that the patch-local list does, or does not, transfer to the global
grid, and the size of the gap. **What it may not conclude:** anything about whether B beats A.
⚠️ Nor may it be re-labelled later: if its number is wanted as evidence, the run is **repeated**
under `05`, on a converged arm, with the criterion declared first.

**Approved by Simon, 2026-11-10**, with that cost named and accepted.

#### ⭐⭐ The instrument exists and its control is PINNED · 2026-11-10

`tools/rho_event_readout.py` · `results/rho_event_control.out` · `tests/test_rho_event_readout.py`
(4 passed). Built to the one recipe in this repo that produced an instrument whose headline did not
have to be withdrawn — `2026-09-30_evaluation_learning_report.md` §9c, where `ordering_by_index.py`
was the **fifth** instrument of its dossier and the **first to survive**. All three of its named
differences are implemented, and two were mutation-tested:

1. ⭐⭐ **The estimator is IMPORTED, never copied** — `spearman` comes from
   `../2026-11-10_rank_coupling_verdict/tools/rank_coupling_verdict.py`, the module that published
   `+0.3127`, and the test asserts the **same function object**. ⛔ Mutation: replacing it with a
   byte-equivalent local copy fails that test. Both of §9c's HOLDs were about a baseline that moved.
2. ⭐⭐ **`G-1` pins the control by arithmetic, and it PASSES:** recomputed here from the committed
   artifacts, **BODY `+0.3127 ± 0.0713`** and **GATE `+0.4418 ± 0.0583`** — both published figures
   to **5e-05**, in 33 s of CPU. ⛔ If either misses, the tool prints no other number and exits 1.
   Mutation: editing an anchor fails the guard that ties it to `00_README`'s finish line.
3. ⭐ **The swept axis is the thing in question** — one row per index, no single number to move. A
   ranking arm's score becomes the third row via `--score`.

⛔ **`G-2` counts distinct VALUES, not files**, which is §0 item 2 of the evaluation report: a
delivery served **0.0 for all 2,330,712 cells** on global land in all three targets and **every
structural check passed** — valid manifest, right coverage, file size 98.1% of a real run's, and no
metric in the inventory flagged it. Measured on this firing-set population: **GATE 1.0000**, **BODY
0.7796**. ⚠️ Not the same population as §9c `N9`'s 64,732-of-64,818, which is the whole global grid.

⚠️ **Population, declared (`C-351`):** 7 folds, score months 463–469, `sb`, h1, truth in **raw
counts, no transform**, τ = 0.2 from the arm config, **1,034 event cell-months**. ⛔ That is *not*
the 2,905 figure quoted elsewhere in this programme — 2,905 is event cells over 13 origins on the
unrestricted global grid. Two populations, one metric name.

#### ⭐⭐ The score dump is BUILT and PROVEN · 2026-11-10 · 54 s, CPU, no pod

`tools/score_on_the_grid.py` + `tools/gate_alignment.py`, with
`results/score_grid_PROVE_GATE_v20261012.npz` and
`results/gate_alignment_PROVE_GATE_v20261012.out`.

⭐ **NO NEW INFRASTRUCTURE**, which is the discipline `2026-11-08_ordering_discrimination`'s
`mu_ordering.py` set. The vehicle loader, the volume builder, the grid mapping and the data pull are
all imported from `../2026-10-12_underfire_diagnosis/vehicle.py`; the forward loop is the same
sequential recurrent sweep `d4_mu_on_the_grid.py` runs. **What is new is which channel is collected
and the output contract.** `--channel latent` is the ranking score, `--channel mu` is the body's
mean field — ⭐ which is `S8`/#568, the thing §9c `N9` records as *"NOT ON DISK for any delivered
run"*. One tool, both jobs.

**Measured cost, not estimated:** 49 sequential steps over all **64,818** land cells, one
deterministic pass, **54 s wall · 3.04 GB peak RSS · 8 threads · CPU · no pod.** Determinism is
proven before the pass is spent: two forwards on identical input must be bit-identical, or the run
refuses — `locked = False` alone clears the flag without calling `.eval()`, so a module already in
train() would resample a fresh mask every forward, which is `C-324`'s inert-knob signature pointed
the other way.

⭐⭐ **Three things the proof run established, all by measurement:**

1. **The cell order is identical to `e1_data`'s** — asserted, `True`. On this grid the cell index
   **is geography** (`C-289`, `C-419`), so a permuted field would score a spatial sweep against the
   truth and return a plausible number. The readout now refuses on this too, element for element.
2. ⭐⭐ **`G-ALIGN` resolves at offset −1**: anomaly `r` **+0.8285** against the runner-up's
   **+0.5015**, margin **+0.3269**, past the thresholds `mu_ordering` pre-registered. ⇒ **Feeding
   month `t` yields the h1 forecast for month `t−1`**, established by measurement rather than by
   reading the code. ⚠️ The LEVEL correlation cannot resolve a month at all — every offset reads
   0.930–0.975 because the gate field is too temporally smooth — and is kept only as a diagnostic.
   ⭐ These figures reproduce the discarded `mu_ordering.out`'s (−1, r +0.8296, margin +0.3253)
   from a different tool, which retires the loose end left by discarding that file.
3. ⛔⛔ **And what the PASS does NOT establish, which is the part that matters.** The anomaly `r` is
   **not 1.0**, and two explanations are live and **not separated by anything on disk**: the dumping
   checkpoint may not be the delivered weights (`C-317` — `4acbf4d3…` is the *retrained control*),
   and the dump is **teacher-forced** while the delivery is autoregressive (`C-399`).

#### ⛔⛔ Teacher forcing: the mode is right, the carry is forbidden

`vehicle.build_volume` writes the **observed** fatality series into the model's input channels for
every month (`vehicle.py:398-404`), and the loop steps over that volume — so the model is handed the
observed history at every step, never its own predictions. **That is `C-399`, Tier 1, verbatim:** the
frozen cube every RP-3 number came from had exactly this property, measuring teacher-forced AUC
**0.99777** against the delivered autoregressive **0.72340**, with no decay in horizon.

⇒ **The defect `C-399` records is not the mode — it is the carry.** Teacher-forced at h1 is the same
input regime the body is *trained* in, so it is the right setting for the one question phase 3 asks:
it moves the patch→global axis alone and leaves the rollout axis fixed. ⛔ What may never happen is
comparing a field from this tool with the **delivered** figures — the body's `+0.3127`, the gate's
`+0.4418`. So `input_mode="teacher_forced"` is written **into every file**, and
`rho_event_readout.py` refuses a field whose stored mode is not `delivered`. The refusal reads the
label on disk, so it cannot be forgotten at a call site.

⭐ **Which fixes the comparator.** A teacher-forced ranking score is compared against a
**teacher-forced body `mu`** from the same dumper, same checkpoint lineage, one variable changed —
the loss. Not against the delivered rows. ⇒ Phase 3 needs **two** dumps, and `--channel mu` on the
control checkpoint is the one that can be taken today.

#### ⭐⭐ The comparator is MEASURED and PINNED · 2026-11-10 · `+0.5189 ± 0.0473`

`--channel mu` on the control checkpoint ran in **149 s** on CPU
(`results/score_grid_CONTROL_MU_v20261012.npz`, 14.3 MB, deliberately uncommitted and re-runnable;
`n_params 2`, determinism verified, `G-ALIGN` reproducing offset `-1` at anomaly `r +0.828477`,
margin `+0.326949`). ⭐ That closes `S8`/#568 — §9c `N9` records the body's `mu` field as *"NOT ON
DISK for any delivered run"*, and it is now on disk for a teacher-forced one.

`tools/rho_event_teacher_forced.py` reads it. **Population, per `C-351`:** 7 folds, score months
463–469, `sb`, h1, firing set = the **delivered** gate at `τ = 0.2` (identical for control and arm,
and identical to the delivered table's definition), truth in raw counts, `ddof=1`.

| index | `ρ_event` | SD | distinct share |
|---|---|---|---|
| **BODY `mu`, teacher-forced** | **+0.5189** | 0.0473 | **1.000000** |

⛔⛔ **This number may NOT be set beside the delivered `+0.3127`, and the bar it sets is its own.**
The same weights' delivered autoregressive `ρ_event` is `+0.3127 ± 0.0713`. The **`+0.2062`**
difference is the `C-399` teacher-forcing carry and the `C-317` checkpoint identity **together**,
and nothing on disk separates them. ⇒ A ranking arm in this table must beat **+0.5189**, not
`+0.3127`. The pin is hardcoded with a `5e-5` tolerance, so the tool refuses to print an arm row
unless the control reproduces; `tests/test_rho_event_teacher_forced.py` additionally asserts the
pin is **not** the delivered anchor, because pasting `+0.3127` there would hand an arm a free
`+0.2062`.

⭐ **One hypothesis died here.** The distinct share is **1.000000** — on the firing cells the body's
`mu` has **no ties at all**. So "the body ranks worse than the gate because it is less resolved"
(§9c `N9`'s explanation for why the index wins) is **not** available in this regime. Resolution is
not the body's ordering problem.

#### ⛔⛔ A sign error, caught by a population refusal and not by review · 2026-11-10

The fed→forecast month map was written **inverted**, and `G-ALIGN` passed anyway — the offset
sweep was right and the *use* of its answer was not. What caught it was the month-coverage
refusal: at the inverted direction the dump covers forecast months 420–468 and cannot score month
**469**, so the tool refused instead of returning a plausible table shifted by one month.

⚠️ **The prose was wrong in both tools and the code was right in one.** `mu_ordering.py` prints
*"Feeding month t yields the h1 forecast for month t+off"* and then does `src = fed_idx[month +
best]`, which is the opposite mapping and the correct one; `gate_alignment.py` inherited the
sentence. Both sentences are now corrected, and the map is a **named function**
(`forecast_index`) with its own test and an explicit discriminating half that proves the two
directions are distinguishable on this design's months.

⇒ Noted, not fixed: `mu_ordering.py`'s own sentence is in the `2026-11-08_ordering_discrimination`
dossier and its code is correct; this dossier does not edit another's tool.

⇒ **What remains for phase 3 is a trained ranking arm** — the dump, the alignment gate, the cell
contract, the readout **and now the pinned comparator** are all built and proven. ⚠️ `body_mean_dump_dir` could not serve — it
refuses a head with no registered family (`hydranet_inference.py:428-430`) and it lives inside the
posterior-pass loop (`:1738`), so it would have built the draw cube that OOM-killed both EXP-02 arms
on host RAM. ⚠️ §9c `N9` calls the same gap *"not a CPU afternoon"*; it was **54 seconds**, because
the expensive thing in that sentence is the posterior cube and this tool builds none.

### Phase 4 — train to the plateau  ·  GPU, remote

⛔ **Remote, not local.** The local route is **closed on measurement**: both smoke arms were
OOM-killed on **host RAM** at ~21 GiB on a 31 GiB box, dying in the backtest pass after origin 1's
draw. This is not re-litigated.

Train the ranking body to the 300-lesson plateau at the vehicle's settings, one variable changed
from the control: the body's loss. **`windows_per_lesson: 3`, `window_dim: 32`, `batch_size: 1`** —
the training path of `02 §4.1`, unchanged.

⚠️ **The watchdog from `01 §1`:** Jagerman 2017's failure signature for mishandled target-side ties
is **overfitting** — validation decay while training performance still improves. Log both curves
and read them, because that is the one failure mode the literature told us to expect.

**Gate to phase 5:** the arm converged, its manifest records what it is, and the validation curve
was read.

### Phase 5 — the emission  ·  CPU, minutes, nothing retrained

Only now compose `gate × ranking body × empirical draw` per `02 §1`, and only now does the
`tie@0` prediction of `02 §5` become checkable: a continuous score should drive it to **0.000**,
where A's count-mean index put it at 0.635 at a loose threshold.

⚠️ `body_mean_dump_dir` writes a **mean**; a scale-free score is not a mean. The dump seam needs
either a draws extension or a score-aware variant — named in `03 §5` as an open box.

**Gate to the verdict:** `05_analysis_plan`'s pre-registered criterion, which does not yet exist.

---

## 2 Dependency graph

```
phase 1  loss + numerics  ──►  phase 2  seam + inertness  ──►  phase 3  ⭐ global ordering probe
                                                                   │
                                                                   ▼
                                                        phase 4  train to plateau (remote GPU)
                                                                   │
                                                                   ▼
                                                        phase 5  rank-coupled emission
                                                                   │
                                                                   ▼
                                               ⛔ the verdict — BLOCKED on 05 (see below)
```

**Off the critical path, in parallel, owned elsewhere:** `00 §5` item 1 (Simon — the primary
quantity) · item 2 (`baseline` — its seed dispersion) · item 3 (`library` — the Mohler 2020 audit).
⇒ Phases 1–4 do **not** wait on any of them. **Phase 5's verdict does.**

---

## 3 Decision points

| after | the question | what makes it answerable |
|---|---|---|
phase 1 | does Eq. 2 survive our sparsity numerically? | the seven tests. If 3 and 4 cannot both hold, the reserve option is next (`01 §3`) |
phase 2 | is the baseline provably untouched? | the parity anchor, with the flag off |
⭐ phase 3 | does a patch-trained score sort the global grid? | `ρ_event` global, and the patch↔global gap |
phase 4 | did it converge without the tie-overfitting signature? | the two curves |
phase 5 | did `tie@0` go to zero as `02 §5` predicts? | the `tie@0` column, recomputed on the trained score |
⛔ **the verdict** | is the ranking body better at ordering than the magnitude body? | **the pre-registered criterion of `05`, which does not exist yet** |

---

## 4 Milestones — and what "done" means

⭐ **Done is the handover's definition, not a win:** the measurement ran and it was **diagnostic** —
some possible outcome would have sent the decision the other way, and the measurement had the
resolution to tell those outcomes apart. **Three** outcomes, all of them done:

| | |
|---|---|
the ranking body orders better than the magnitude body | done |
the magnitude body holds | done |
⭐ **not resolvable at this `n`** | **done.** Under-power is a result, not a loss for B |

⛔ The third is non-negotiable. Without it an under-powered run reads identically to a negative one,
and the idea ends for the wrong reason — which is exactly what happened on 2026-11-08.

---

## 5 Cost, stated up front

| phase | compute | cost |
|---|---|---|
1 | CPU | hours of writing; seconds to run |
2 | CPU | hours |
3 | one short train + CPU scoring | ⭐ the cheapest thing that bears on the finish line |
4 | remote GPU, 300 lessons | the only expensive phase |
5 | CPU, minutes | nothing retrained |

⇒ **Phases 1–3 are all CPU-or-cheap and all precede the only expensive step.** That ordering is the
point: the expensive phase is bought with phase 3's number, not with an argument.
