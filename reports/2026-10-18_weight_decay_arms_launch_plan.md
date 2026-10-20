# LAUNCH PLAN — the views-hydranet half

**Date:** 2026-10-18 · **Seat:** views-hydranet · **Status:** PLAN ONLY. Nothing executed, nothing created, nothing merged.
**Why this exists:** Simon, 2026-10-18 — *"now this is getting too improvised and chaotic. Please plan out all steps you need to take to get us to a place where I can deploy pods."* **He is right.** This seat was also answering implementation questions as they arrived rather than planning the path.
**Pre-registration:** `2026-10-18_weight_decay_arms_prereg.md` (6 amendments) · **Companion:** the `views-models` seat is writing its half; these reconcile into one plan for him, not two.

---

## 0. ⚠️ FIRST: a label clash that must not reach him

The other seat reports *"he chose allocation A — `3 × weight_decay 0.1` (seeds 42, 43, 44) and `2 × weight_decay 0.0` (seeds 42, 43)"*. ⛔ **This seat labelled that same allocation `B`** in its message to Simon (where `A` was the originally authorised `2×0.1 + 2×0.01 + 1×0.0`).

**The CONTENT agrees exactly** and it is the allocation both seats recommended. ⛔ **But the letters do not, and a letter is a label whose meaning differs between seats — the provenance-mismatch class applied to a decision.** **Resolution: the allocation is referred to by its content everywhere from here, never by letter:**

> **`3 × weight_decay 0.1` at seeds 42, 43, 44 · `2 × weight_decay 0.0` at seeds 42, 43.**

⚠️ **The joint plan must confirm this content back to him in one line before anything is created**, because *"he chose A"* is currently true in one seat's vocabulary and false in the other's.

---

## 1 — ⛔ THE FIVE GATING QUESTIONS, ANSWERED FROM CODE. All five are settled.

The other seat asked these and was right that they are this seat's call, not its guesses.

### 1.1 ✅ Is there a library release gate? **NO.**

```
pod_run_model.sh:171-173          uv pip install ... "views-hydranet~=0.1.1"
models/copper_kestrel/requirements.txt:1   views-hydranet~=0.1.1
```

`~=0.1.1` means `>=0.1.1, <0.2.0`, so the pod installs **the latest 0.1.x — currently 0.1.3.** And `weight_decay`, `np_seed` and `torch_seed` are **long-standing config keys present throughout 0.1.x**, not new fields.

⛔ **So nothing in this allocation needs a views-hydranet change released to PyPI first.** ⚠️ **This is the question that produced W-22** — the `body_nll_denominator` arm *would* have been a silent no-op because its key existed only on an unpushed branch while the pod installed from PyPI. **The answer is different here, and it is different for a reason that was checked rather than assumed.**

### 1.2 ✅ Is `weight_decay: 0.0` reachable without a library change? **YES.**

```
config_initializer.py:150   weight_decay: float = Field(..., ge=0.0)
config_initializer.py:149   learning_rate: float = Field(..., gt=0.0)      <- contrast
```

⛔ **`ge`, not `gt`** — `0.0` validates. `choose_scheduler` (`utils/utils.py:277-282`) passes it straight to `AdamW(..., weight_decay=config["weight_decay"])`, and `weight_decay=0.0` is valid for AdamW. **Nothing coerces, floors or rejects it. No library change.**

⚠️ **Whether it TRAINS is a different question and is the risk Simon accepted knowingly.** Nothing in the record speaks to it; **F11 pre-registers a diverging/NaN/stability-gate-tripping zero-decay run as VOID for that arm (C-307, not negative)**, and the readout must state how many of the two survived **before any slope is read.**

### 1.3 ✅ The model-side pin each new config must carry: **`views-hydranet~=0.1.1`**

Verbatim from `models/copper_kestrel/requirements.txt:1`, matching the pod runner. **Copy it; do not "update" it to `~=0.1.3`** — the pod's own install line is `~=0.1.1`, and a model pin tighter than the runner's is a divergence with no benefit.

### 1.4 ⛔ `CORE_GENOME` is the authoritative reference for the F1 config-diff check

`infrastructure/reproducibility_gate.py:27-44` — *"Core genome: parameters every HydraNet config must declare. These are the knobs that affect training outcome."* It contains **`np_seed`, `torch_seed`, `learning_rate`, `weight_decay`, `total_lessons`, `windows_per_lesson`, `window_dim`, `steps`, `time_steps`, `clip_grad_norm`, `input_channels`, `output_channels`, `total_hidden_channels`, `dropout_rate`, `loss_reg`, `loss_class`.**

⛔ **Two consequences:**
1. **F1's check now has an authoritative list instead of this seat eyeballing a diff.** The five arms must differ from `amber_kestrel` **only** in `weight_decay`, `np_seed` and `torch_seed` — all three of which are **in** `CORE_GENOME`, i.e. all three are declared outcome-affecting knobs. **Any other `CORE_GENOME` difference voids that arm.**
2. ⚠️ **A genome difference BETWEEN arms is expected and correct, not a fault.** Stated because a reproducibility gate reporting different genomes across five arms will look alarming to anyone who has not read this line.

### 1.5 ⛔ No in-training instrumentation is required, and this seat DECLINES the trajectory-hook change

**The endpoints — `a`, `b`, and the held-out calibration error — are measured POST-HOC** by the local teacher-forced sweep from each arm's delivered weights (`b_before = 1.1875` in 554 s). **Nothing needs logging during training.**

⛔ **So the one-line `sd` addition to the trajectory hook is NOT wanted here, and this seat declines it.** It was identified for **C3** — *is the gate's logit spread still growing at lesson 300* — which is a **different question on a different endpoint** and is not on this plan's critical path. **Available, considered, declined for scope.** Recorded so it reads as a decision rather than an oversight.

⚠️ **One thing each arm SHOULD enable, and it is config-only:** `trajectory_log_path`, as `copper_kestrel` does (`config_hyperparameters.py:233`). It is free, it came home last time, and ⛔ **`amber_kestrel` did not set it, which is why Phase 1 had no control baseline for the level trajectory.** **Setting it on all five costs nothing and closes that gap.**

---

## 2 — ⛔ MY VIEW ON THE FOOTPRINT QUESTION, which they asked and which may change their plan

Their concern: **five permanent model directories, 17 tracked files each, plus five test updates, is a large and permanent footprint for a five-run diagnostic whose expected outcome is "inconclusive."** They ask whether `pod_run_model.sh:38`'s ephemeral-patch mechanism could carry `weight_decay` and a seed instead.

### What that mechanism actually is

⛔ **It is `--rehearsal <n>`, and it is narrower than hoped but its SHAPE is exactly right.** `pod_run_model.sh:37-39` — it *"patches the POD's clone of the config — never a tracked file — and marks the output as unfit to deliver"*, writing a `REHEARSAL` marker (`:26`) and a MANIFEST `mode:` line, with `:131-134` cleaning the markers so a production run cannot inherit an "undeliverable" mark or vice versa.

⛔ **And its stated rationale is our situation exactly** (`:44-47`):

> *"Before `--rehearsal` existed the only way to get one was to edit the guard out, which produced output indistinguishable from a real run: the cheap test and the accident looked the same on disk. That is the actual hazard."*

**Our hazard is the same one in a new place: five diagnostic arms that must never be deliverable, created as five permanent roster entries that look exactly like candidate models on disk.** ⛔ **The prereg's F6 forbids calling any arm shippable — in PROSE. The rehearsal mechanism exists because prose is not enough and the artifact must be branded.**

### ⛔ But I do NOT recommend extending the launcher, and the reason is the stronger one

`pod_run_model.sh` is **the path every future production run takes.** ⛔ **Five config directories are inert; a launcher change is load-bearing for everything that follows.** After nine corrections in one pre-registration, recommending a change to the most load-bearing script in the pipeline — for a five-run diagnostic — is the scope expansion this programme keeps getting bitten by.

### ✅ THE PATH I RECOMMEND: use the mechanism that already exists for exactly this

`EXPERIMENT_IN_PROGRESS` is already implemented and **test-guarded as a SET** so adding or removing an entry is caught and never silent — `tests/test_roster_conformance.py:279,286,295`: *"Without this, marking a model `EXPERIMENT_IN_PROGRESS` silently removes it from every [check]."* `violet_visitor`, `amber_kestrel` and `copper_kestrel` all reference it.

> ⛔ **Recommend: five directories, each declaring `EXPERIMENT_IN_PROGRESS = True`, plus a PRE-COMMITTED REMOVAL STEP after the readout.**

**Why this is the right trade:**
- It addresses the real objection — *these are not candidate models* — **with a declaration the test suite already understands**, rather than with a change to the launcher.
- `EXPERIMENT_IN_PROGRESS` is **roster-guarded as a set**, so the five cannot quietly become permanent: **removing them is a visible, tested change.**
- ⛔ **The footprint objection is about PERMANENCE, not existence.** A pre-committed removal makes it temporary without touching anything load-bearing.
- ⚠️ **It does NOT brand the output the way `--rehearsal` does**, so **F6 remains prose.** That is the cost, it is real, and this seat judges it smaller than changing the launcher. **If Simon disagrees, extending the launcher is the better long-term answer and should be its own proposal with its own review — not a side-effect of this experiment.**

⛔ **This is a recommendation to the joint plan, not a decision. The five directories and the tests are the other seat's repo and its call.**

---

## 3 — THIS SEAT'S STEPS, in order

| # | step | needs | status |
|---|---|---|---|
| **H1** | Answer the five gating questions from code | — | ✅ **DONE** (§1) |
| **H2** | Rule on the footprint question | — | ✅ **DONE** (§2), as a recommendation |
| **H3** | ⛔ **F1 config-diff review of all five configs against `amber_kestrel`, using `CORE_GENOME` as the reference list** | the five configs to exist | **blocked on their P1** |
| **H4** | Confirm each config sets `trajectory_log_path` and carries `views-hydranet~=0.1.1` | same | **blocked on their P1** |
| **H5** | Confirm no arm's config trips `reject_near_miss_keys` or a `model_validator` | same | **blocked on their P1** |
| **H6** | *(after the runs)* local sweep per arm → `a`, `b`, calibration error, bootstrap `σ̂` **per endpoint** | delivered weights | **after launch** |
| **H7** | *(after the runs)* evaluate F1–F11, read W-0…W-5, write the outcome | H6 | **after launch** |

⛔ **H3–H5 are cheap, they are this seat's, and they are the last gate before launch.** F1 voids an arm that differs anywhere but its named key and seed, and an arm voided *after* 2 h 15 m of GPU is a wasted pod.

### ⛔ What this seat does NOT do, stated so the joint plan has no gaps

- **No model directories, no test edits, no branch, no PR, no merge in `views-models`.** Standing instruction, unchanged.
- ⛔ **No library change**, because §1 establishes none is needed. **If one turns out to be needed, it stops the plan and goes to Simon** — it does not get made mid-launch.
- **No pod launch, no merge to `development`.** ⚠️ The other seat is right that the pod shallow-clones `-b development` (`pod_run_model.sh:236`), so **nothing can run until the configs are merged, and that merge needs Simon's explicit word.**

---

## 4 — What Simon has to decide, and it should stay at two

⛔ **This plan deliberately adds no new decisions.** From §1, three of the questions that looked like decisions are **settled facts** (no release gate, `0.0` is valid, the pin is `~=0.1.1`), and two are **this seat's calls already made** (declining the hook change, recommending `EXPERIMENT_IN_PROGRESS` over a launcher change).

**What remains his:**

1. ⛔ **Confirm the allocation by content** — `3 × 0.1` at seeds 42/43/44 and `2 × 0.0` at seeds 42/43 — because the two seats used different letters for it (§0).
2. ⛔ **The merge to `development`**, which is what unblocks every pod. It cannot be delegated and both seats are blocked on it.

⚠️ **And one thing he already accepted knowingly, restated because it is the main risk and not a decision:** nothing in the record says `weight_decay: 0.0` trains. If both zero-decay arms void, the five pods still deliver `σ̂` — **the first run-to-run variance measurement in this pipeline, and the Phase-1 audit (W-5)** — which is why the set is worth launching even in that case.

---

## 5 — ⛔ And the honest statement of what this experiment is

**Recorded here because the joint plan will be read by someone deciding whether to spend five pods:**

- **Expected outcome is W-2: INCONCLUSIVE.** Three independent reasons — a 2-df denominator, an unknown effect size, and a feasibility boundary (`σ̂ < 0.0356`) that sits inside the plausible range.
- ⛔ **It cannot tell anyone a model is better to serve.** No forecast stage ⇒ no `act_ratio`, no CRPS, no AP ⇒ **the firing-rate guardrail that caught the 37–83% false CRPS win is unavailable.** F6.
- ✅ **What it reliably delivers regardless: `σ̂`** — the first measurement of run-to-run variance in this training pipeline — **and with it an error bar on the amber/copper 0.17 gap this programme has been reasoning from for three days (W-5).**

⛔ **That is the honest case for spending the pods, and it is not the case the experiment was originally proposed on.**
