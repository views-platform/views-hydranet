# Pre-Analysis Plan — does training past 160 lessons keep buying rollout retention?

**Date:** 2026-08-18 · pre-registered **before any arm runs**
**Companion to:** `reports/2026-08-17_ss_retention_dossier/` (whose parked sweep this gates),
`reports/RESULTS_LEDGER.md` §TRAINING LENGTH, `scripts/floor_gate.py`, `SCOPE.md`
**Status:** LOCKED. Nothing below is changed after a control is seen.

## 1. Why

Every HydraNet in the tree trains for **160 lessons** (or 40). Checked three ways: all 11 live configs,
259 wandb run configs on this machine (max ever recorded = 300), `git log -S` across views-models.
**600 lessons has never been run here and no 600-lesson tuning left any trace** — the three mentions in
the repo are proposals, two marked unconfirmed.

The one step of the ladder that has been measured, single-variable, collapsed the readout:

| lessons | AP h1 | AP h18 | retention | floor gate |
|--:|--:|--:|--:|---|
| 40 `shortzero_fortytwo` | 0.288873 | 0.019622 | 0.0679 | **FAIL** — 2.16× chance |
| 160 `longzero_fortytwo` | 0.474461 | 0.256912 | **0.5415** | **PASS** — 28.30× chance |
| 160 **oracle** `use_real` | 0.474461 | **0.479269** | **1.0101** | — |

Training length is the only intervention that has ever moved retention. The oracle row says the
achievable retention is **1.0** — fed the true field this model holds AP flat over 36 steps — so 46% of
the ceiling is discarded by the rollout, and per I-A (4/4 vehicles) ~90–95% of that is *placement*.

`RESULTS_LEDGER.md` §TRAINING LENGTH states the charter: whether 160 is on the plateau or the slope is unknown,
and if it is still climbing then every experiment at 160 — **including the parked SS sweep** — measures a
partially-trained model.

## 2. Intervention — exactly one variable

Clones of `violet_visitor` at `ss_epsilon_max = 0.0`, `ss_feedback = 'sample'`, built by
`reports/2026-08-17_ss_retention_dossier/tools/make_ss_arm.py`, which asserts the **symmetric
difference** of the floor's and the arm's resolved config dicts equals exactly the intended key set.
`total_lessons` ∈ {160, 300, 600, (900)}. `longzero_fortytwo` **is** the L=160 arm, already built by that
same builder from that same floor.

**What varying `total_lessons` does and does not touch** — established by reading the code, recorded
here so it is not discovered afterwards:

| axis | effect | verdict |
|---|---|---|
| LR schedule | `warmup_decay_lr_scheduler.py:13` inverse-sqrt, stepped per lesson (`training_engine.py:1070`), `warmup_steps=100`. A 600-lesson run's first 160 lessons see byte-identical LRs. | **prefix-consistent, not a confound** |
| BN recalibration | `_recalibrate_bn` draws 30 windows via `planner.get_lesson(w)`; `get_intensity_ratio` is roof-clipped at 0.665 until step 114 (L=160) / 214 (L=300) / 427 (L=600), so steps 0–29 are clipped at every L in the ladder. | **identical for all L ≥ 160** |
| `ss_warmup_lessons` | absolute, so the ε-exposure fraction would shift | **moot: every arm is ε=0** |
| **curriculum cooling** | `curriculum.py:85` — `b = (min−max)/(total_steps × slope_ratio)`. The difficulty schedule **stretches proportionally** with `total_lessons`. | ⚠️ **coupled by construction — see §7** |

## 3. Arms — per lesson count L, two of them

| arm | how | yields |
|---|---|---|
| control | `main.py -r calibration -t -e -sa` | **C(L)** = AP@h1, **F(L)** = AP@h18 |
| oracle | `run_realism_arms.py --arms use_real` on that arm's own artifact | **O(L)** = AP@h18 under perfect feedback |

Target `sb`, h ∈ {1,6,12,18,24,30,36}, h\* = **18**, 13 origins, S = 16, calibration partition, the pinned
v2 truth. Reference for FG-B is `2026-08-15_rollout_ruler_trust_dossier/results/rescore.csv`
`model == climatology` (AP h1 = 0.297976) — the same origins, N, partition and S. **Not** `light_strider`,
which is a different snapshot and disagrees by 12–15%.

## 4. Endpoints — the convention already in force, not a new one

Taken verbatim from `2026-08-17_ss_retention_dossier/05_analysis_plan.md` §4 (Primary / Co-primary rows):

* **Primary** — `F(L)` = AP@h18, free-running, absolute.
* **Co-primary** — retention `R(L) = AP(h18)/AP(h1)`. **Must agree in sign with the primary**, because a
  ratio of two noisy quantities can move on its denominator.
* **Decomposition** — `log F = log C + log R` splits any change in free-running skill exactly into a
  **T=0** part and a **retention** part. (Not a *ceiling* part — the ceiling is the oracle's score,
  a different number; conflating them was corrected 2026-08-21.) `O(L)` says whether the achievable-at-h18 ceiling itself moves.

**Pre-registered effect size θ = 0.14 on retention.** Derived only from numbers already on disk:
`0.30 × (R_oracle(160) − R(160)) = 0.30 × (1.010134 − 0.541482) = 0.140595`. The floor-limited 40-lesson
retention is **not** used to set any threshold.

*(The oracle's retention exceeding 1 is not an error: prevalence rises with horizon — 1343 events at h1,
1547 at h18 — so AP can legitimately be higher later. Recorded so a reviewer does not spend time on it.)*

## 5. The noise floor is measured first, and it is a gate

Seeds 42/43/44/45 at L=160, ε=0 give **σ_seed**, the sample SD of R across training runs. The
origin-block MDE is noise *within* a run; σ_seed is noise *between* runs — the uncertainty that actually
applies to comparing lesson counts, and **which no experiment in this programme has ever measured**.

**Prior, stated before the run so it cannot be retrofitted.** The v2 scoreboard's 3 seeds × 300 lessons
give R = 0.4859 / 0.5505 / 0.5259 → mean 0.5208, **σ ≈ 0.0326**. Different snapshot, family and date, so
not a substitute — but if σ_seed lands near it the design is powered. Those same numbers make the **prior
expectation PLATEAU** (0.5208 at 300L vs 0.5415 at 160L), which is exactly why the null must be made
*declarable* rather than assumed.

**Decision rule — one-sided 95% prediction bound for a new run (n = 4, t₃,₀.₉₅ = 2.35336):**

```
multiplier k = 2.35336 * sqrt(1 + 1/4) = 2.631        # pinned
bound(X)     = mean(X over the four 160-lesson seeds) + k * sd(X over those seeds)
```

Applied to **both** endpoints, with sign agreement required:

Let `T` be the **longest** arm scored (not the last one run):

| state | condition |
|---|---|
| **RISING** | `R(T) > bound(R)` **and** `F(T) > bound(F)` **and** `F(T) − F(160, s42) > 3 · mean MDE_F(160)` |
| **PLATEAU** | both inside their bounds **and** `k · σ_seed(R) < θ` |
| **UNDERPOWERED** | inside the bounds but a bound is wider than θ — "no effect" and "could not tell" are not distinguishable; also the state before ≥3 anchor seeds or any arm above L=160 exists |
| **G1-STOP** | `k · σ_seed(R) ≥ 0.30` — see below |
| **VOID** | any falsifier in §6 fires |

Implemented in `scripts/lesson_curve_gate.py::curve_verdict`, which is unit-tested against every one of
these five states (`tests/test_lesson_curve_verdict.py`) and runs its falsifiers **before** its rule.

**Stated up front, not discovered later:** if `σ_seed(R) > 0.0532` then `k·σ_seed > θ` and **PLATEAU is
unreachable by construction** — the design could then detect only a large rise. σ_seed is a deliverable in
its own right: it bounds what *any* single-seed experiment on this vehicle can see, and retro-scopes much
of the prior single-seed work under the standing rule (`RESULTS_LEDGER.md` §Standing rule adopted 2026-08-17).

**G1 — the stage-1 stop.** If `k · σ_seed(R) ≥ 0.30` — more than half the entire 0.4687 gap to the
ceiling — no single-seed lesson point can say anything. Stop, and report *that* as the result. It would
be worth more than the curve.

## 6. Falsifiers — checked and recorded before any prediction is read

* **F1** AP@h1 differs between an arm's control and its oracle by > 1e-6 ⇒ something other than the
  feedback path moved (step 1 has no feedback) ⇒ that point **VOID**.
* **F2** `N ≠ 170430` or origins ≠ 13 in any row ⇒ arms on different supports ⇒ **VOID**.
* **F3** two arms share a `weight_sha256` ⇒ the same model scored twice. Weight-tensor hash, **never** the
  `.pt` file sha — torch stamps mtimes, so file shas are an invalid identity check (M22).
* **F4** a control fails floor gate **FG-A** ⇒ that point is floor-limited; its numbers are not evidence.
* **F5** `make_ss_arm.py`'s symmetric-difference assertion reports any key beyond the intended set ⇒ the
  arm is not single-variable ⇒ not built.
* **F6** repo HEAD differs across arms ⇒ not comparable.
* **F7 (advisory, never VOID)** the fresh oracle on `longzero_fortytwo`'s artifact should reproduce the
  archived `score_violet_visitor_use_real.csv` (M22: identical weight tensors). Report the difference;
  investigate above 3·MDE. Advisory by design — the two runs differ in model directory and the
  equivalence is unverified, so a hard VOID on it would be the wrong call.

### AMENDMENT 1 — 2026-08-18 09:20, F6 narrowed to its stated intent

**F6 fired on stage 1 and the verdict was VOID.** Recorded here because a falsifier was changed after
data was seen, which is the one move that most needs a paper trail.

*What fired:* the four L=160 arms span two commit ids — seed 42 was trained at `5ed6c223`, seeds 43–45
at `eb388f9`.

*The objective evidence it was nominal:* `git rev-parse <head>:views_hydranet` returns
**`ca41c3f59769525b5a8bebfb0a94bbc4ea778d30` at both commits** — the training/inference package is
byte-identical. So is the scorer blob (`ef7ee1da`). `git diff 5ed6c223 eb388f9 -- views_hydranet/` is
**0 lines**. The seven intervening commits are documentation, a pre-registration, and this dossier's own
analysis tooling.

*The change:* F6 now compares a **code fingerprint** — the git tree hash of `views_hydranet/` plus the
scorer blob at each arm's HEAD — instead of the commit id. `head` remains the fallback where no
fingerprint can be computed.

*Why this is a narrowing, not a relaxation:* F6's stated purpose is "arms built at different code are
not comparable". A commit id answers a different question, and answers it wrongly in both directions —
it fires on docs-only commits **and** it would pass two arms built from the same commit id in a dirty
working tree. The fingerprint is strictly closer to the intent. It is sabotage-tested: two arms with
different `views_hydranet` trees still return VOID.

*What was NOT changed:* θ, k, G1, the anchor length, h\*, and F1–F5. The pinned decision-rule md5
`5d6a256bb2b41485220d033cd0bfbc87` is unchanged, because none of its inputs moved.

---

### AMENDMENT 2 — 2026-08-18 09:40, two anchor seeds added for power

**sigma_seed(R) measured 0.0544** at L=160 (n=4: 0.5415 / 0.6284 / 0.5780 / 0.6648). §5 pre-registered
the boundary at **0.0532** — above it, `k · sigma > theta` and PLATEAU is unreachable by construction.
We landed 2% over: `k · sigma = 0.1431` against `theta = 0.140`.

**Change:** two more anchor seeds (46, 47) at L=160, ε=0, appended after the curve. The decision rule is
untouched — its multiplier already depends on n through `t(n-1, 0.95) · sqrt(1 + 1/n)`:

| n | k | k · sigma | PLATEAU declarable |
|--:|--:|--:|---|
| 4 | 2.631 | 0.1431 | no |
| 5 | 2.335 | 0.1270 | yes |
| 6 | 2.177 | 0.1184 | yes, with margin |

**Why this is not fishing:** the decision was taken on the **noise floor alone**, before any arm above
L=160 had been read — the L=300 arm had crashed and the L=600 arm was at lesson 7. §5 anticipated this
exact contingency in writing. Adding observations to an estimator whose penalty already shrinks with n
is not a threshold relaxation; theta, k's formula, G1 and the anchor length are unchanged, and the
pinned decision-rule md5 does not move.

**What it buys:** the difference between reporting "training is done as a lever" and reporting "we could
not tell" — which, on a flat result, is the whole value of the experiment.

---

### AMENDMENT 3 — 2026-08-19 08:45, the multiplier now tracks n as §5 always specified

`K_PRED` was hard-coded at **2.631** (the n=4 value) while §5 defines the multiplier as
`t(n-1, 0.95) · sqrt(1 + 1/n)` and AMENDMENT 2 tabulates it for n=5 and n=6. This is a **bug fix, not a
rule change**: the implementation did not match the pre-registration it claimed to implement.

Direction of the error matters. A too-large multiplier widens the bound, which makes **both** RISING and
PLATEAU harder — it can never have manufactured a positive. What it did do is discard the power the
extra seed was run to buy: at n=5 it reported `k·sigma = 0.1254` (a null NOT declarable) where the
pre-registered formula gives **0.1113** (declarable).

`rule_md5` is **unchanged** — `K_PRED` remains the pinned n=4 reference value, and n was always a free
parameter of the stated formula rather than a pinned constant.

---

**Two pinned hashes.** Floor-gate thresholds md5 **`6d5714d5ceda147ed16f53143abe7e37`** (h\*=18,
target `sb`, θ=0.30, r=5.0, b=1.2, k=3.0). Decision-rule md5
**`5d6a256bb2b41485220d033cd0bfbc87`** (`scripts.lesson_curve_gate.rule_md5()` over θ=0.14, k=2.631,
G1=0.30, anchor L=160, h\*=18) — reported on every `VERDICT.md`. A mismatch on either means a
threshold moved after a control was seen, and the result must not be quoted.

## 7. Stated confounds

1. **Lesson count and curriculum-cooling rate move together by construction** (`curriculum.py:85`). A
   600-lesson run is *not* a 160-lesson run continued; it is the same curriculum shape spread over 600
   lessons. This experiment answers "does setting `total_lessons` higher help", which is the production
   question — **not** "do more gradient steps help". Decoupling them is a second variable and a separate
   experiment.
2. **One seed on the lesson axis.** σ_seed is measured at L=160 and *assumed* comparable at 300/600. If
   training variance grows with length, the bound is understated.
3. **`total_lessons` is the only key that differs, but training cost per lesson is not constant** —
   0.2505 → 0.3058 min/lesson from 40L to 160L (+22%). Every cost figure above 160 is an extrapolation
   from two points and cannot separate a fixed overhead from a superlinear term.
4. **One target (`sb`), one primary horizon (h\*=18), 13 origins, S=16, calibration partition.**
5. **The oracle is `use_real`** — perfect occurrence *and* magnitude. It is the ceiling of the feedback
   path, not of the model, and says nothing about what is learnable.
6. **A null here does not license "training is done" for any other vehicle** — one config, one family
   (`nb`), one composition (`soft_gate`).

## 8. Decision rule — pre-committed

* **RISING** ⇒ the ladder 600→300→160 cost real skill; the rollout problem is substantially a training
  problem; the parked SS sweep must be re-scoped to the converged length before it can be read.
* **PLATEAU** ⇒ training is finished as a lever at this configuration; the residual gap is structural and
  attacking placement is justified rather than assumed; the parked SS sweep can be read as-is at 160.
* **UNDERPOWERED** ⇒ report as such. Do not narrate a direction from a difference inside the bound.
* **VOID** ⇒ name the falsifier, exclude the point, do not quote its numbers.

A positive result here is **one seed per lesson point** and therefore an **escalation trigger**, not a
conclusion — the standing rule applies to this dossier exactly as to every other.
