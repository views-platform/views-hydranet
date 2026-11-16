# Knowns, and the rules for reporting a result

**Written 2026-11-12 at Simon's instruction**, after the fourth or later research programme that
collapsed results across the horizon. It exists because the rules below were given verbally,
repeatedly, and were not written anywhere a session reads at startup.

Two parts, deliberately separate:

* **Part A — RULES.** How a result must be scoped and reported. These are Simon's, they are
  standing, and they do not expire.
* **Part B — KNOWNS.** Settled facts. Do not re-litigate; do not spend a programme rediscovering
  one.

Each line in Part B is tagged with how it is known: **[checked]** = measured in a session on the
thing the claim is about, with the artifact named; **[read]** = taken from a report, memory or
earlier session and *not* re-verified here, so it is as unreliable as any note until re-measured.

---

## Part A — RULES for scoping and reporting a result

### A1 · The object of work is the BODY — bulk and tail

The body is what we are trying to improve. **The gate is not the object of work.** It may appear as
a component (a firing set, a composition input, a context row), but it is never the thing being
improved, and a result about the gate is not progress on the question.

⛔ Specifically forbidden without being asked: gate thresholds, `pos_weight`, gate calibration,
gate resolution. See `K5` — that axis is closed.

### A2 · Three targets. Report all three, separately

| locked name | what it is |
|---|---|
| **sb** | state-based violence (government vs rebels) |
| **ns** | non-state violence (group vs group) |
| **os** | one-sided violence (against civilians) |

⛔ Never pool them. ⛔ Never average across them. ⛔ Never run `sb` alone and present it as the
result — `sb` alone is a result about `sb`, and must be labelled as covering **one of three**.

If an instrument can only reach one target, that is a limitation of the instrument and must be
stated in the same breath as its number.

### A3 · 36-step horizon. NEVER collapse across it

The forecast horizon is **36 steps, indexed from 0**: `t0` is the first forecast month, `t35` the
last. ⚠️ `h1` in the code is **`t0`**, not `t1` — the loaders are named for the 1-based horizon and
the reporting is 0-based, which is exactly the kind of off-by-one that has already cost a dossier.

**THE MAPPING RULE, stated once and used everywhere.** Reports say **`t`, 0-based**; the code says
**`step` / `h`, a 1-based LEAD TIME**. They are both correct and neither is an index into the other:

```
t_k  ==  step_(k+1)  ==  h(k+1)          so the default set t0, t17, t35  ==  step 1, 18, 36
```

⚠️ `step` is 1-based because it is a **lead time in months**, not an array position: the platform
gets the calendar month by `month = origin + step`
(`views-pipeline-core/managers/evaluation/stage.py:467`), and ADR-040 makes the orchestrator the
sole authority on lead times. A lead time of 0 would mean a nowcast, which is not what is produced.
⛔ So the platform contract is NOT to be zero-indexed; only the reporting vocabulary is 0-based, and
this line is the join between them.

⛔ **No average across steps.** Not a mean, not a pooled score, not "performance" unqualified.
⛔ **No single step standing in for the horizon**, including `t0`.

**Default reporting set, when nothing else is said: `t0`, `t17`, `t35` — each reported separately.**
Sometimes Simon wants more steps, sometimes fewer; he will say so. **Absence of an instruction means
the default set, not a free choice.** He will say explicitly if he ever wants an average, and that
has to be assumed not to happen.

### A4 · MSE and CRPS are not measures of goodness here

⛔ Neither MSE nor CRPS may carry a verdict. They may appear as diagnostics, labelled as such.

⛔ **The evaluation scheme in `views-pipeline-core`, and everything the FAO-02 release notes say
about what good evaluation is, are DATED and are not authoritative.** That includes the FAO-02
"locked eval framework" (CRPS primary, QS99/Brier/MCR guardrails). ⚠️ It is recorded in memory as
*locked*, and that record is now superseded by this line. We are still learning what the right
measures are; the paper's minimal metric set (`K19`) is the live work on that question.

### A6 · A result is a TENSOR, and it is reported as TABLES

A result has dimensions. The ones that recur here:

| dimension | values |
|---|---|
| violence type | `sb`, `ns`, `os` |
| horizon step | `t0` … `t35` (default `t0`, `t17`, `t35`) |
| task | classification (occurrence) vs regression (amount) |
| metric | whichever are defensible for the question — see `A4` |
| population | all open cells / event cells / tail / per-cell vs pooled |
| inference mode | teacher-forced vs delivered autoregressive |
| arm | control vs treatment, and seed |

**Not all dimensions are needed in any one result** — usually most are not, and choosing which to
hold fixed is part of the design. ⛔ **But if the output does not contain one or more tables laid
out along some of these dimensions, something is wrong.** A lone scalar, or prose with numbers
embedded in it, is the shape a result takes when a dimension has been collapsed without anyone
saying so.

Minimum acceptable shape: **one table whose rows and columns are two of the dimensions, with every
dimension held fixed named in the caption.** For example, rows = horizon step, columns = violence
type, caption naming the task, metric, population, inference mode and arm.

### A5 · Every number names its population, transform and estimator — and its inference mode

`C-351`: a number is its population, its transform and its estimator, in the same sentence.
`C-399`: **teacher-forced and delivered autoregressive numbers may never be compared.** Name the
inference mode on **both** sides of every comparison. The measured size of that confusion is
teacher-forced AUC **0.99777** against delivered **0.72340**, with no decay in horizon.

---

## Part B — KNOWNS. Settled; do not re-litigate

### B1 · What the body cannot do

* **The WALL — the data cannot say how big.** No feature predicts the size of a sudden jump.
  Spearman **0.303** against fair persistence **0.367**. **[read]** — `project_amount_ceiling_wall`.
* **The 2-NB tail-decoupled mixture does not crack it.** Tail *shape* (ξ > 0) is the gap, not
  mixture flexibility. **[read]** — `project_mixture_tail_decoupled_null`.
* **Jump risk IS predictable** (0.79 against 0.39) and the tail index is ξ ≈ 0.8 ⇒ a heavy-tailed
  per-cell head is the open avenue. **[read]** — `project_volatility_ceiling_predictable`.
* **The body is timid, and the deficit is scale-dependent.** It carries **0.391** of due spatial
  structure at 1 cell and **0.870** at 32 cells. ⛔ No single scalar may be quoted for this.
  **[read]** — `project_d12_scale_profile`.
* **Closed levers on the body**: a point body mask; the winsorized τ-pinball dial (lifts magnitude,
  explodes CRPS); the body loss as such (dead-ReLU + softplus, `C-178`). **[read]**

### B2 · What the rollout does

* **The 36-month explosion ("bloom") is autoregressive feedback**, not the loss and not the gate
  (`C-113`). **[read]**
* **Mitigated, not solved**: feeding back a *sample* rather than the mean fixes it 9/9 on 6
  retrained models (ADR-070, `rollout_feedback=sample`). **[read]**
* **There is no magnitude skill in rollout.** **[read]**
* **Retention levers are closed**: training plateaus at 300 lessons; scheduled sampling hurts AP.
  **[read]**
* ⚠️ **Measured 2026-11-12 and new**: at h1 on `sb`, the body's `mu` ranks event cells at
  [ERASED 2026-11-16 by Simon: teacher-forced number, not to be used] teacher-forced against **+0.3127 ± 0.0713** delivered, while the gate moves
  only **+0.4470 → +0.4418**. ⇒ The body's ranking deficit in the delivered numbers is largely a
  **rollout** deficit. ⛔ Not a clean decomposition: the delivered body row is the mean of 16 draws
  (distinct share 0.7796) and the teacher-forced row is `mu` (1.000000), so draw quantisation is
  confounded with rollout. **[checked]** —
  `2026-11-10_ranking_head_dossier/results/`, `calm_lander`, 7 folds, months 463..469,
  1,034 event cell-months, global land, `ddof=1`. ⛔ **`t0` (code `h1`) only, `sb` only — one step of 36, one
  target of three. It is not a result under `A2`/`A3`.**

### B3 · What the gate does — closed, and not the work

* The gate **is calibrated** at `pos_weight` ≈ 1–2. **[read]**
* `pos_weight` is a **pure intercept shift**, so that axis is closed. **[read]**
* **Firing is not the lever**: recall is free, placement is everything (`thin_0.75` keeps 97%;
  scramble costs 81%). **[read]**
* Logits are compressed (under-dispersed) in both arms. **[read]**

### B4 · Composition

* `threshold_gate` **discards** calibrated gate information; recomposing closes **76.3%** of the
  empty-map gap with nothing retrained. **[read]**
* Properly composed, `gated_NB ≈ th_gated_NB`; `forecast_composition` is a config axis (ADR-069).
  **[read]**

### B5 · Vocabulary and families — locked

* Only ADR-067 **NB** and **ZINB**, all-cell. ⛔ Never the legacy `hurdle_nb`. Say **`gated_NB`**,
  never `masked_NB`. **[read]** — `reports/GLOSSARY.md` is the only source of terms.
* ZINB is falsified and blooms in rollout; `gated_NB ≡ th_gated_NB` ships. **[read]**
* D-3 of the research queue: **no new emitting family.** **[read]**

### B6 · Data and population

* **Global `land` is 64,818 cells.** The 13,110-row Africa+ME scoped data must go; a full disk
  sweep is still owed. **[read]**
* ⛔ **Never validate on cached disk data — pull fresh.** **[read]**
* ⚠️ Open and unresolved: views-datafactory's spatial allocator writes `priogrid_gid` and the grid
  compiler never reads it, so placement numbers computed before that is fixed are wrong. Their
  advice is **do not re-pin anything yet** (views-hydranet#591). **[read]**

### B7 · Engineering facts that have cost real work

* `reports/` is in `.gitignore` but its files **are tracked**. ⇒ Use **`command grep`** under
  `reports/`; an absence proved by a recursive `grep` there proves nothing. **[read]** — `CLAUDE.md`
* **`git add -f` with NAMED FILES, never a directory** (`C-395`). **[read]** — `CLAUDE.md`
* Run with `conda run --no-capture-output -n views-hydranet-env python <script>`. **[read]**
* Releases stay **0.1.x**; verify a release by **downloading the wheel**, never site-packages
  (`C-347`, fired six times). **[read]**
* The fleet is **27** HydraNet models (the roster is 8). Count the fleet for blast radius.
  **[read]**
* **A config does not describe the artifact beside it.** 6 of 11 pod checkpoints carry a config
  naming a different architecture than their sidecar (`C-424`). ⇒ Name a comparator by its
  **checkpoint**, never re-derive it from a config. **[read]**
* Pod-returned weights land in
  `views-models/models/<name>/data/generated/calibration_delivery_*/artifact/`, **not** in
  `models/<name>/artifacts/`, which holds only `.gitkeep`. 11 checkpoints, 3.53 MB, 906,904
  parameters each. **[checked 2026-11-12]**
* EXP-02 runs **remote**; the local route is closed (both smoke arms OOM-killed on host RAM).
  **[read]**

### B8 · Where this is all going

* **The end goal is a paper.** Synthetic forecasts with a *known* ranking give the minimal metric
  set; declaring that ranking is Simon's call. **[read]**
* The ranked queue is `reports/RESEARCH_PROGRAMME.md`, RP-1..RP-9. Read it before proposing a
  direction. **[read]**

---

## ⛔ GLOSSARY §7's `T=0` line is a scar, not a settled fact

`reports/GLOSSARY.md` §7, under **"Settled facts — do NOT re-open"**, currently reads:

> **T=0** — the first forecast month only; the *only* thing we score.

**That is not a research finding.** Simon's account, 2026-11-12: it was written as a hard, simple
rule during a programme in which the same instruction as `A3` was given repeatedly and ignored
anyway, at a time when the work was deliberately zoomed in on `t0`. ⇒ It records a coping strategy,
not a fact about forecasting.

⛔ **And it did not work.** All 36 months were scored anyway. So the line neither reflects the
research position nor achieved what it was written for, while sitting in the section a session is
told never to re-open.

⇒ **`A3` governs.** `t0` may be the sole reported step again when the work is deliberately zoomed
in on `t0` — and that is then a stated scope for that piece of work, not a standing rule.

💬 **One line of a LOCKED file, so Simon says yes before I touch it.** Proposed replacement for
that bullet:

> **T=0 / `t0`** — the first forecast month. ⚠️ **NOT the only thing we score**: the standing rule
> is `t0`, `t17`, `t35` reported separately, for `sb`, `ns` and `os` separately — see
> `reports/KNOWNS_AND_REPORTING_RULES.md` §A2–A3. A programme may deliberately zoom in on `t0`,
> which is a stated scope for that programme and not a standing rule.
