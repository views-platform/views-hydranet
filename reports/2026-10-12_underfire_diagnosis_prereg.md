# Pre-analysis plan — WHY does HydraNet under-fire?

**Date:** 2026-10-12 · **Status:** PRE-REGISTERED, NOT RUN · **Branch:** `probe/why-hydranet-underfires`
**Issue:** #396 · **Intervention it gates:** #394

---

## 0. The established fact this starts from

Measured 2026-10-11 (`reports/2026-10-11_conflictology_empty_mode_prereg.md`, commit `de4813d`),
on the 100 global land cells with the highest observed positive rate in months 421–456:

| | positive share |
|---|---|
| observed history (conflictology, closed form, n=36) | **0.7639** |
| HydraNet's own posterior (S=16, recovered by algebra) | **0.4062** |

Corroborated independently at **S=128** on the live FAO delivery by the `faoapi` session. Older and
broader: **M45** records step-1 `act_ratio` at **0.36× truth**. And **I-A**, replicated on 4
vehicles / 2 body families / 4 seeds: **occurrence carries 90–95% of the skill; magnitude carries
zero.** So this is the main axis, not a side one.

**Nobody has established why.** Six interventions have failed, and the ledger's own **M46** note
says all of them changed *what the model is fed* and none changed *the horizon of the loss* — the
signature of testing levers without a model of the cause.

⛔ **This project diagnoses. It changes nothing and retrains nothing.**

## 1. Three candidate causes, and why the order matters

The composed forecast is **gate × body** (`gated_NB`, ADR-067/069). Zero mass therefore has two
possible origins before training is even implicated.

| | cause | if it dominates, the lever is | cost |
|---|---|---|---|
| **C1** | **The family cannot express it.** Even at best fit, an NB with the right mean puts too much mass at zero on data this overdispersed. | the distribution head (ZINB, heavier tail) — **#394 is then not the lever** | ~1 h, **no model, no GPU** |
| **C2a** | **Dilution.** 75% of magnitude-loss terms are non-land structural zeros (#394), so the informative gradient is swamped. | the loss scope — **#394 is the lever** | ~3 h, one forward/backward |
| **C2b** | **The objective rewards timidity.** The timid predictive distribution actually scores better than the honest one against what happened. | the objective itself | ~3 h |
| **C3** | It can express it, the loss rewards it, training does not find it. | optimisation / architecture | **only if C1, C2a, C2b all come back clean** |

Cheapest first, because **C1 can settle it alone**: if the best-possible NB already implies ~60%
zero mass on these cells, then no loss change can fix this and #394 is the wrong week of work.

## 2. Method — four measurements

**T0 — gate or body?** (diagnostic regardless of outcome, ~2 h)
Run `violet_visitor`'s local artifact on the always-active cells; read the gate probability and the
body's own `P(Y=0)` separately. Composed zero mass is `(1 − g) + g·P_NB(0)`. Attribute it.

**T1 — C1, the family's ceiling.** (~1 h, no model)
For each of the 100 cells, fit **NB** and **ZINB** by maximum likelihood to that cell's own 36
observations (`views_hydranet.distributions.get_family`, the repo's own families — not scipy, so the
parameterisation is the production one). Compare each fitted distribution's implied `P(Y=0)` against
the observed `1 − 0.7639 = 0.2361`.

**T2 — C2a, where the gradient actually comes from.** (~3 h)
One real batch, one forward/backward on the artifact. Partition cells into **non-land** /
**zero-land** / **active-land** and report each partition's share of the magnitude-loss value **and**
of the gradient norm reaching the head. #394 asserts 75% of *terms*; this measures whether the
*gradient* follows.

**T3 — C2b, does honesty score worse?** (~3 h)
For each always-active cell, score two predictive distributions against that cell's actual observed
outcomes using the repo's own `FamilyLoss`/`family.nll`: (a) the model's own emitted parameters,
(b) the T1-fitted honest distribution. Report mean NLL for each.

## 3. Pre-registered predictions

Committed before looking. Two are against my own earlier framing, so they can score against me.

- **P1 (T1).** The best-fit **NB** will imply materially more zero mass than observed — median fitted
  `P(Y=0) ≥ 0.45` against the observed 0.236. Reason: these cells are heavily overdispersed, and
  NB's `P(0) = (θ/(θ+μ))^θ` rises as θ falls. **ZINB will fit materially better** (median `P(Y=0)`
  within 0.10 of observed).
- **P2 (T0).** The **gate** is the dominant source of composed zero mass, contributing more than the
  body's `P_NB(0)` in a majority of the 100 cells.
- **P3 (T2).** Non-land cells will account for **≥ 50%** of the magnitude-loss gradient norm.
- **P4 (T3).** The honest distribution will score **BETTER** (lower NLL) than the model's own — i.e.
  **the objective does NOT reward timidity per cell**, and the live mechanism is dilution (C2a), not
  scoring (C2b). ⛔ This predicts against "the loss pays it to be empty", which is the framing I gave
  Simon on 2026-10-11.

## 4. Falsifiers — pre-committed

| # | observation | verdict |
|---|---|---|
| **F1** | P1 holds for NB **and** ZINB also implies ≫ observed zero mass | ⛔ **C1 dominates.** The family is the ceiling. **STOP the project**; #394 is not the lever; route to the head. |
| **F2** | P1 fails — best-fit NB's `P(Y=0)` is already within 0.10 of observed | the family is not the constraint; C1 eliminated, continue to T2/T3 |
| **F3** | P3 fails — non-land is **< 20%** of the gradient norm despite 75% of terms | ⛔ **#394's mechanism is refuted.** The dilution is arithmetic without consequence. Say so on #394. |
| **F4** | P4 fails — the model's timid distribution scores **better** than honest | ⛔ **C2b confirmed:** the objective rewards emptiness. Outranks #394. |
| **F5** | T0 cannot attribute (gate and body contributions within 10% of each other across most cells) | record as **indiscriminate**, not as a finding |
| **F6** | the artifact's config does not match the delivery's, or inference is not byte-reproducible on a known anchor | ⛔ **VOID.** Fix the vehicle, do not reinterpret. |
| **F7** | any single measurement exceeds **2× its stated budget** | record **UNRESOLVED**, move on. ⛔ Do not chase. |

## 5. ⛔ Definition of done — and the stop rules

**Done when all four of these exist, and not before or after:**

1. A number for T0, T1, T2, T3 — or an explicit `UNRESOLVED` / `VOID` for any that failed a gate.
2. Each of P1–P4 marked held or failed.
3. **One sentence** naming which cause dominates, or stating plainly that the measurements cannot
   discriminate.
4. A `found, not chased` list for everything noticed along the way.

**Then stop.** Specifically:

- ⛔ **No fix is attempted and nothing is retrained in this project.** The output is a diagnosis and a
  recommendation about which lever to pull.
- ⛔ **No new questions are added mid-project.** Anything discovered goes on the `found, not chased`
  list. This rule exists because the last two weeks spiralled.
- ⛔ **F1 is a hard stop**: if the family is the ceiling, the remaining tests are not run.
- **Total budget: ~1.5 days wall clock, no training, no GPU hours beyond single forward/backward
  passes.** F7 enforces it per-measurement.

## 6. Out of scope

- Running #394's A/B, or any other intervention. That is the *next* decision, informed by this.
- Horizon-aware loss design (M46's gap). Named, not pursued.
- `ns` / `os`. **`sb` only.**
- The map / point-estimate question — owned by `faoapi` and by Simon under ADR-023.
- The magnitude ceiling (#241). I-A says magnitude carries ~0% of the skill; it is not the axis.

## 7. Vehicle

`violet_visitor` (4 local `.pt` artifacts), the R&D reference model and one of the 8 delivered.
Calibration partition, `REGION="land"`, 64,818 cells, months 457–504. Cell set: the 100 highest
observed positive rate in months 421–456, identical to the 2026-10-11 run. Config read-only from
views-models — ⛔ **nothing committed there.**

---

# OUTCOME — T1 complete, T0 partial, T2/T3 blocked. Run 2026-10-12.

## T1 — C1 ELIMINATED. The family is not the ceiling. (MEASUREMENT)

NB and ZINB fitted by MLE to each of the 100 cells' own 36 observations, using the repo's own
families (`get_family`), scored with the production `prob_positive`:

| | implied positive mass (median) | implied zero mass |
|---|---|---|
| **observed** | **0.7639** | 0.2361 |
| **best-fit NB** | **0.7695** | 0.2305 |
| best-fit ZINB | 0.7387 | 0.2613 |
| **HydraNet's actual posterior** | **0.4062** | 0.5938 |

NLL: NB 3.6953, ZINB 3.6819 — ZINB barely improves, i.e. there is **no excess zero spike beyond
what a plain NB already handles**, consistent with views-baseline's "falls back to plain nb when no
excess zeros."

**Convergence verified** (guards against an optimiser artifact): fitted `mu / sample mean` = **1.0000
at p5 and p95**, 0 of 100 cells off by >5% — the NB MLE's defining property. Per-cell agreement with
observed: mean abs error **0.0272**, max 0.1027, corr **0.9644**. The three cells at 100% observed
positive fit to 0.9931–1.0000. θ spans 0.179–3.431.

⇒ **P1 FAILS.** I predicted the best-fit NB would be forced to ≥0.45 zero mass by overdispersion. It
is not — the NB tracks the observed zero rate to within 0.006 at the median. **The production family,
at its own best fit, would emit 0.7695 positive mass. HydraNet emits 0.4062.** The gap is not the
distribution's expressiveness. **F2 fires; C1 is eliminated; the project continues.**

(P1's ZINB half HOLDS — within 0.0252 of observed — but that is now uninformative, since NB already
matches.)

## T0 — partial, via a cheaper route than planned, and it answers a DIFFERENT question

**Method substitution, declared:** rather than load an artifact, the 8 delivered models form a
natural **composition × family 2×2**, so the delivered frames alone can test whether either design
choice explains the under-firing. Same h1, same 100 cells.

| model | composition | family | positive fraction | cells entirely zero |
|---|---|---|---|---|
| bold_comet | threshold | mixture_nb | 0.6250 | 1 |
| blazing_meteor | threshold | mixture_nb | 0.5313 | 5 |
| pink_pirate | soft | mixture_nb | 0.5000 | 3 |
| blue_stranger | soft | mixture_nb | 0.4375 | 7 |
| purple_alien | soft | mixture_nb | 0.4375 | 6 |
| bright_starship | soft | nb | 0.4375 | 8 |
| **violet_visitor** | **threshold** | **nb** | **0.3750** | **8** |
| heavy_freighter | soft | nb | 0.2500 | 8 |

Grouped: **threshold 0.5313 vs soft 0.4375** · **mixture_nb 0.500 vs nb 0.375**.

⇒ **P2's premise is damaged.** I predicted the gate dominates. `threshold_gate` — which *can* hard-zero
a whole cell (`violet_visitor`: `gate_threshold = 0.20`) — is associated with a **higher**, not lower,
positive fraction. The composition type is not the cause, and the family contrasts more strongly than
the gate does.

⛔ **The real finding here is fleet-wide:** **all 8 delivered models under-fire** on these cells —
0.25 to 0.625 against an observed 0.7639 — and **every one of them emits an entirely zero posterior
in 1 to 8 of the 100 most persistently violent cells on earth.** This is not one model's defect, one
family's, or one composition's.

⚠️ **Confounded and not decisive:** 8 models, unbalanced 2×2, differing in architecture, seed and
training history as well as in the two design axes. And it does **not** answer the prereg's actual
T0 question — within a single model, is the zero mass from `(1 − g)` or from `P_body(0)`? That needs
the artifact. **T0 is recorded PARTIAL, not done.**

## ⛔ T2 / T3 / T0-proper — BLOCKED. F6 fires on the vehicle.

All four local `violet_visitor` artifacts predate the delivery *and the region switch*:

| artifact | date | |
|---|---|---|
| `calibration_model_20260812_191742.pt` | Aug 12 | |
| `validation_model_20260908_135014.pt` | Sep 8 | |
| `validation_model_20260916_032119.pt` | Sep 16 | |
| `validation_model_20260917_010418.pt` | **Sep 17** | newest |
| — | **Sep 19** | ⛔ `REGION` switched `africa_me_legacy` → `land` (runbook #499) |
| — | **Sep 28** | the delivery ran |

The newest local artifact is **two days older than the region switch**, so it was trained on
`africa_me_legacy` (13,110 cells, 180×180 grid) — **not** the global `land` model (64,818 cells,
360×720) that produced the delivery whose under-firing T1 just characterised. They load fine (plain
`OrderedDict` state dicts, ~3.7 MB), and the network is fully convolutional so it would *run* on the
global grid, but it is not the delivered model.

**This is the F6 condition** — the artifact's config does not match the delivery's. Per §4, that is
**VOID, not reinterpreted**. The three remaining measurements are therefore **not run**, pending a
decision on the vehicle.

## Found, not chased

- `violet_visitor`'s `gate_threshold` is **0.20**, not the 0.5 recorded for `th_gated` in the
  glossary/ledger. Not investigated; may be a legitimate per-model setting or may be drift.
- The `mixture_nb` vs `nb` contrast (0.500 vs 0.375) is the strongest signal in the T0 table and was
  not a pre-registered question. Not chased.
- `heavy_freighter` is the weakest of the 8 at **0.2500** and is the designated global PoC/template.
  Not chased.

---

# AMENDMENT 1 — the vehicle, 2026-10-12. Written BEFORE any measurement runs.

**F6 is satisfied differently, not waived.** §7 named the vehicle as `violet_visitor`'s local
artifacts. Those are VOID (all ≤ Sep 17; `REGION` switched to `land` on Sep 19; delivery Sep 28), and
no global-land artifact survives anywhere — the pod was destroyed and this codebase never uploads
weights (`log_artifact` has no non-test caller). See views-models#549, #550.

**New vehicle:** a freshly trained `violet_visitor` at `REGION="land"`, 300 lessons, calibration,
produced by one pod run (views-models#550, ~2 h 25 m to saved weights, ~$2.50, approved by Simon
2026-10-12).

⛔ **It is NOT the delivered model** and is not bit-reproducible (**C-317**). The views-models session
holds the sha256 prefixes of all eight 2026-09-28 artifacts, so the divergence is **provable rather
than assumed**, and the first thing recorded on the new artifact is its own sha256.

**Why that is acceptable, stated as a limit rather than a dismissal:** T2 and T3 ask about the
**loss** — where its gradient comes from, and whether it rewards timidity. Those are properties of
the objective and the grid, not of one weight set. **T0** (gate vs body attribution) *is* weight-
specific, so its result transfers to "a production-config HydraNet at global land", **not** to the
delivered forecast. Every T0 line will carry that scope.

The revised F6 condition: **VOID unless** the new artifact's sidecar confirms `REGION`-consistent
`input_channels`, `output_distribution: 'nb'`, `forecast_composition: 'threshold_gate'` and
`gate_threshold: 0.20`, matching the delivered config. Config identity is the gate; weight identity
is explicitly not.

## Three constraints added by the views-models review

1. ⛔ **The artifact is three files.** `train_model.py` writes the `.pt` (`:73`), `.pt.config.json`
   (`:138`) and `.pt.sha256` (`:144`). `model_artifact_fetcher.py:107` reconstructs the model **from
   the config sidecar**; without it the `else` branch at `:125-128` calls `.to()` on an
   `OrderedDict` and raises `AttributeError`. **All three must be pulled down**, or there is no
   loadable vehicle. This is why views-models#549's original one-line fix was insufficient.

2. ⚠️ **A `mixture_nb` vehicle would change what T3 scores.** `model_artifact_fetcher.py:117`
   attaches `hurdle_nb_theta` **from the sidecar**. For `violet_visitor` (`nb`, sidecar
   `hurdle_nb_theta: None`) no learned parameter sits outside the state dict. **If the vehicle is
   ever switched to one of the five `mixture_nb` models, the sidecar becomes part of T3's object**
   and T3 must be re-specified. Recorded now so it cannot be discovered mid-run.

3. ⚠️ **The local-fit claim gets measured, not assumed.** Postmortem §2.2's ~3 GB VRAM peak is
   **prose and the single source** — no run log carries an instrumented peak (the only VRAM string
   in `violet_visitor`'s log is a preflight `VRAM: 32`, the pod's capacity). It is nonetheless
   stronger evidence than first used: it was measured on a run that **trained**, so it already
   includes the backward pass and BPTT activations across the 32-month window and four ConvLSTM
   cells. ⇒ **`torch.cuda.max_memory_allocated()` is recorded on the first forward/backward and the
   run stops if it surprises us.** ⛔ `disk_guard` is not consulted — postmortem §2.5: it reads the
   host's RAM rather than the cgroup, and under-reports true peak 2–3× even corrected.

## Unchanged by all of the above

**T1 stands and needs no vehicle.** Best-fit NB implies 0.7695 positive mass against an observed
0.7639 while HydraNet emits 0.4062. **C1 is eliminated and the distribution head is not the lever**,
whatever the pod returns.

T0's cross-model 2×2 also stands: all 8 delivered models under-fire (0.25–0.625 vs 0.7639), and
neither composition nor family explains it.

---

# AMENDMENT 2 — the F6 gate was insufficient, 2026-10-12. Still before any measurement.

Found by smoke-testing Amendment 1's own F6 check against the local artifacts: **all four PASS the
config-identity test** despite being `africa_me_legacy`-trained. The sidecar records
`model`, channel counts, dropout, seeds, `output_distribution`, `reg_activation` and
`n_head_samples` — **no region and no grid size** — and the network is fully convolutional, so the
state-dict shapes are identical between a 13,110-cell africa model and a 64,818-cell land one.

⇒ **Config identity is necessary and NOT sufficient**, and Amendment 1's revised F6 would have
admitted exactly the vehicle it was written to exclude.

**Revised F6, third time:** VOID unless *all* of —

1. the sidecar matches the delivered config (Amendment 1), **and**
2. the artifact's sha256 is **not** one of the four known africa artifacts, denied by hash in
   `2026-10-12_underfire_diagnosis/vehicle.py:KNOWN_AFRICA_SHA256`, **and**
3. the operator asserts the artifact came from the views-models#550 pod run — which rests on
   `pod_run_model.sh:279-282` refusing to start unless `REGION == 'land'`.

⚠️ **(3) is an assertion, not a verification, and is recorded as such.** Nothing in a `.pt` or its
sidecar can prove the region. `load_vehicle` prints that limit on every run rather than implying a
check it cannot perform. The durable fix would be recording `region` in the sidecar — noted on the
**found, not chased** list, not pursued here.

## T3 gains a second, unregistered comparison — declared

The pre-registered T3 (model vs honest NB fitted to observed history) is **confounded**: the honest
fit is a per-cell constant while the model is conditional on inputs, so a model win could be
legitimate conditioning rather than rewarded timidity. **P4 is scored on the registered comparison
as written.**

Added alongside, and reported separately: the model against a **mean-preserving de-timidified
version of itself** — `mu` held, `theta` raised until `P(Y=0)` equals the observed zero rate. Same
mean, same conditioning, strictly less zero mass. **F4 is judged on this controlled comparison**,
because the registered one cannot carry it. The NB floor at fixed mean is `exp(-mu)` (Poisson
limit); cells below it are reported, not forced.

## Found, not chased (appended)

- The artifact sidecar does not record `region` or grid dimensions, so no artifact in this system
  can be audited for the extent it was trained on. The durable fix is a sidecar field.

---

# AMENDMENT 3 — F6 is now READ from the artifact, 2026-10-12. Still before any measurement.

Amendment 2 recorded that region provenance was *unverifiable* and handled it with a hash deny-list
plus an operator assertion. views-models closed that properly instead, so F6 is no longer an
assertion for most of what it covers.

**What changed upstream** (views-models `fix/549-collect-pod-artifact`, tip **`f99413ef`**): the pod
now writes **`artifact/PROVENANCE`** at collection time, carrying `model`, `artefact`, `sha256`,
`lessons`, `region`, `run_type`, `git`, `trained` and `mode`, plus an `artifact/REHEARSAL` marker on
a rehearsal run.

⛔ **My own suggested fix for this would have silently done nothing.** I proposed copying
`$OUT/REHEARSAL` into `artifact/` at collect time. `stage collect_artifact` is at
`pod_run_model.sh:318`; `$OUT/REHEARSAL` is written at **`:445`**. The copy would have copied a file
that did not exist yet — a no-op that looks like a guard, which is the same defect class this whole
thread is about. Caught by the views-models session; verified here from the pushed branch.

**Revised F6, fourth and current.** VOID unless *all* of:

1. the three files are present — `.pt`, `.pt.config.json` (required to load at all), `.pt.sha256`;
2. no `INCOMPLETE` or `REHEARSAL` marker sits beside the artifact;
3. `artifact/PROVENANCE` exists — **its absence is VOID**, because lessons and region cannot
   otherwise be established and §4 says VOID rather than reinterpret;
4. `PROVENANCE: mode` does not contain `REHEARSAL`;
5. `PROVENANCE: lessons` is numeric and **≥ 300** (`MIN_LESSONS`). A shorter run is a different
   model — M45's control was measured at L=300 and 40 lessons is smoke-only per the retention
   programme;
6. `PROVENANCE: sha256` agrees with the file's own digest;
7. the sidecar matches the delivered config (Amendment 1);
8. the sha256 is not one of the four known africa artifacts (Amendment 2) — kept as a backstop.

Verified by unit test against five scenarios: missing PROVENANCE → VOID; rehearsal → VOID;
40 lessons → VOID; `lessons: unknown` → VOID; `lessons: 300` production → ACCEPT. The parser
splits on the first colon only, so values containing colons (`trained: …T20:00:00Z to …`) survive.

⚠️ **`region:` is RECORDED AS ENFORCED, NOT MEASURED, and is printed that way.** It records that
`pod_run_model.sh:279-282` refuses any value but `land` — not a read-back of the grid the weights
saw. Nothing in a `.pt` or its sidecar records the region and the net is fully convolutional, so
this is weaker than a measurement and the gate says so on every run rather than implying a check it
cannot perform. views-models stated this limit themselves rather than letting the file imply more.

**Tonight's run is production, not a rehearsal** — `pod_run_model.sh violet_visitor` with no
`--rehearsal`, floor ≥ 300 lessons, per the views-models session. That will now be **readable off
PROVENANCE** rather than taken on anyone's word, which is the point of the file.

## Found, not chased (appended)

- `artifact/INCOMPLETE` still instructs the reader to use "the git sha in MANIFEST", and MANIFEST
  lives at `$OUT`, not in `artifact/`. PROVENANCE now carries the sha so the instruction is
  satisfiable, but the wording points at a file that does not travel. Cosmetic; views-models is
  taking it in tomorrow's PR rather than rebuilding the branch the pod is about to clone.

---

# OUTCOME — ALL FOUR MEASUREMENTS COMPLETE. 2026-10-12.

**Vehicle:** `calibration_model_20261012_033112.pt`, sha256 `4acbf4d3c2b457b0…`, PROVENANCE
`lessons: 300`, `mode: production`, `git f99413e`, trained 01:24:58Z–04:25:10Z. F6 **PASSED** on all
eight conditions. C-317 divergence confirmed against the delivered `d92dcff463bd310f…` — different
weights, **proven, not assumed.** Peak CUDA allocation **1.68 GiB** (prose figure was ~3 GB).

## ⛔ Every one of the four pre-registered predictions FAILED

That is the result, and it is why the predictions were committed first.

| | prediction | measured | |
|---|---|---|---|
| **P1** | best-fit NB forced to ≥0.45 zero mass | **0.2305** vs observed 0.2361 | ❌ |
| **P2** | the **gate** dominates the zero mass | gate 0.034, **body 0.511** | ❌ |
| **P3** | non-land ≥50% of the gradient norm | **12.49%** | ❌ |
| **P4** | the honest distribution scores better | model 4.21, honest 7.18 | ❌ |

## T2 — ⛔ #394's MECHANISM IS REFUTED (F3 fires)

One forward/backward, month 439→440, `eval()` mode (no dropout, no BN writes), loss 0.005487.

| partition | cells | % cells | % loss | % gradient norm |
|---|---|---|---|---|
| non-land | 194,382 | **74.99%** | 0.23% | **12.49%** |
| zero-land | 64,591 | 24.92% | 7.52% | 13.29% |
| **active-land** | **227** | **0.09%** | **92.25%** | **74.21%** |

#394's 75%-of-terms figure is **exactly right** (74.99% measured). Its *consequence* is not: the
gradient does **not** follow the term count. **227 cells — 0.09% of the grid — carry 92% of the
loss and 74% of the gradient.** The convolutional head concentrates on the live cells regardless of
how many structural zeros surround them.

⇒ **F3 FIRES.** The dilution is arithmetic without consequence. **The ocean mask is not the lever,
and #394 should not be run as an intervention.** This was my hypothesis and my issue; it is dead.

## T0 — the GATE KNOWS. The BODY is the problem. (P2 fails)

35 steps, months 421–455, inside the train period.

- gate probability on these cells: median **0.6945** (p10 0.3265, p90 0.9499) — the gate correctly
  identifies them as active
- cell-steps killed by the threshold (`g ≤ 0.20`): **3.4%**
- the body's own `P(Y=0)` where the gate lets it through: median **0.5196**
- **zero mass contributed by the gate 0.034 · by the body 0.511** → the body supplies **94%** of it

## T3 — the OBJECTIVE REWARDS THE CONFIGURATION THAT EMPTIES THE MAP (F4 fires)

⚠️ **The pre-registered comparison (a) is confounded and so was my own controlled comparison (b).
Both are reported; the verdict rests on (c), which I added after seeing (b)'s size.**

| | parameterisation | P(Y=0) | mean NLL | θ |
|---|---|---|---|---|
| | model: own μ, own θ | 0.5253 | **4.2121** | 0.1544 |
| **(c)** | model μ + **honestly fitted** θ | 0.2758 | **5.5552** worse | 0.4323 |
| | honest μ + model θ | 0.5556 | 4.7655 worse | 0.1544 |
| **(a)** | honest: own μ, own θ | 0.3345 | 7.1826 worse | 0.4323 |
| **(b)** | model, de-timidified to the observed zero rate | 0.2500 | 32.3723 worse | 0.5370 |

**(b) was confounded by my own instrument.** In a 2-parameter NB, θ controls **both** zero mass and
dispersion. Forcing `P(0)` down to 0.25 at fixed μ raised θ 4× and **cut the variance 4×** (to the
Poisson floor for 16% of cell-steps). The actuals are heavy-tailed — median 7, p90 90, **max
1528** — so (b) measured the penalty for destroying dispersion, not a reward for emptiness. Its
32.37 is not evidence of anything about timidity.

**(c) removes that.** It swaps in the *honestly fitted* θ (out-of-sample, fitted 385–420, scored
421–456) rather than one chosen to hit a zero rate. That brings `P(Y=0)` to **0.2758**, close to the
observed 0.2361 — and the loss gets **worse**, 4.21 → 5.56.

⇒ **F4 FIRES, de-confounded. C2b is confirmed and it outranks #394.**

## ⛔ The one sentence the definition of done asks for

**The loss pays the body to be extremely overdispersed in order to cover a very heavy tail, and in a
two-parameter NB you cannot be that overdispersed without putting half your mass on zero — so the
empty map is the objective working exactly as specified, in a distribution family that cannot
separate *how often* from *how big*.**

Supporting: the model's θ is **0.1544**, *below* the entire range of honest per-cell fits
(T1: 0.179–3.431). It is not a failure to converge — moving θ toward the honest value makes the loss
**worse**. The gate already knows where the violence is (median 0.69); the body cannot report it
without giving up the tail.

**What this implies about the lever** (INFERENCE, not measurement): the fix is a parameterisation
that separates occurrence from magnitude — not a loss-scope change (T2), not the head's
expressiveness in the sense T1 tested, and not the gate (T0). ⛔ **Not pursued here. This project
diagnoses and stops.**

## Found, not chased

- The model's **μ** is reasonable (median 10.2 against actual median 7); the defect is wholly in θ.
- `mixture_nb` models scored a higher positive fraction than `nb` ones in the earlier cross-model
  2×2 (0.500 vs 0.375) — consistent with a second component relieving the single-θ conflict, and
  **untested as such.**
- T1's ZINB barely improved on NB *when fitting an empirical per-cell distribution*; that is a
  different object from a conditional model's fit and does **not** bear on whether ZINB would help
  here.
- views-models corrected a claim I had adopted: `collect_artifact` runs **after** evaluation, not
  before, because `-t -e` is one process. The protection I repeated as "the stronger claim" does not
  exist yet. Theirs to fix; recorded because I had relied on it.
- `$OUT/FAILURE` survives a successful rerun (views-models, theirs).

## Definition of done — MET

Four numbers ✅ · four predictions scored ✅ · one sentence ✅ · found-not-chased list ✅.
**No fix attempted, nothing retrained, no new question added** — (c) is the same question as (b)
with the instrument corrected, declared as such.
