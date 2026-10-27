# S3 (#461) — Hypothesis H is REFUTED as stated. The quantile head's deficit is mostly the bulk, not the tail.

**Date** 2026-10-27 · **Issue** #461 (epic #458) · **Measured from the surviving record only** —
no new compute, because the artifact is gone (see §1).

⚠️ **This story's plan is not executable as written**, and the reason is itself a finding. Reported
either way, per the epic's risk #2: *"H is refuted in S3 — then the head needs more than a bound,
and S6/S7 change shape."*

## §1 ⛔ The artifact is gone and the scorer output was never captured

| asked for | found |
|---|---|
| the banked 40L quantile checkpoint | ⛔ **absent.** `find /home/simon -iname "*quantile*"` with `.pt`/`.npz`/`.pkl`/`.parquet`/`.csv` returns **nothing** |
| the T=0 scorer's per-cell decomposition | ⛔ **never captured.** `tools/t0_score.py:67` prints `CRPS on zero-truth` / `active` and `log1p(max)`, but **no log in `run_logs/` contains that output** (`grep -l "zero-truth"` and `"T=0 SCORER"` → empty) |
| per-cell or top-N data to sweep the cut | ⛔ **absent.** `m1_q40L.log` is 1,922 lines of training output; no `top-N`, no `per-cell`, no `cell_crps` |
| the surviving quantitative record | ✅ the M1f/M1g **markdown tables** (7.22 / 1.21 / 1.36) and **one narrative scalar**: *"top-10 cells ≈ 36% of CRPS"* |

⇒ S3's steps (b) per-cell decomposition, (c) same-cell comparison against the family-free fit, and
(d) apply the bound and re-score are **not executable**. Nothing can be recomputed.

⇒ ⛔ **This is the C-383 shape again** — a number quoted in a document with no artifact and no
captured output behind it — and it is now the *second* story in this epic to hit it (#460 was created
to close exactly this for M67). ⚠️ The epic rests on M1f/M1g the way it rested on M67, and M1f/M1g
are **less** recoverable: M67's cube survived, this checkpoint did not.

## §2 ⛔⛔ H is refuted by arithmetic on the numbers that do survive

**H, as the epic states it:** *"the quantile head's deficit is concentrated in the top quantile on
~100 cells, and bounding its tail by what the conditional law was observed to contain recovers most
of the family-free fit's advantage."*

M1f measures the concentration: **top-10 cells ≈ 36% of sb CRPS.** Take the most generous possible
bound — those cells contribute **exactly zero**:

| target | quantile head | conflictology | ratio now | perfect top-10 removal | ratio then |
|---|---|---|---|---|---|
| `lr_sb` | 7.22 | 0.276 | **26.2×** | 4.62 | **16.7×** |
| `lr_ns` | 1.21 | 0.108 | **11.2×** | 0.77 | **7.2×** |
| `lr_os` | 1.36 | 0.039 | **34.9×** | 0.87 | **22.3×** |

**What share the bounded cells would have to carry to reach parity:**

| target | required | measured |
|---|---|---|
| `lr_sb` | **96.2%** | **36%** |
| `lr_ns` | **91.1%** | — |
| `lr_os` | **97.1%** | — |

⇒ ⛔ **H is refuted as stated.** A bound on the top cells cannot close an 11–35× gap when those
cells carry roughly a third of the score. It would need to carry essentially all of it.

## §3 ⭐⭐ The reading that replaces it, available since 2026-07-16

Strip the top-10 cells out and the quantile head's **remaining** CRPS is **4.62**. Conflictology's
**entire** score is **0.276**.

⇒ ⭐⭐ **The quantile head's non-top-10 CRPS alone is 17× worse than the whole baseline.**
**64% of the deficit is in the bulk, not the tail.** The head is wrong about ordinary cells.

⇒ The epic made the overshoot its centrepiece — *"the property that costs the lookup its generality
is exactly the property whose absence cost the quantile head 11–35×"*. That symmetry was real but
it was **load-bearing on the minority of the effect.**

## §4 ⭐ The bound the epic proposed has nonetheless never been tried — two things were conflated

| | what it was | result |
|---|---|---|
| **15L "bounded" head** | a **training-time sigmoid** squashing the quantiles | ⛔ **339.6** — *"the sigmoid bound made permanent (gradient-death → stuck at ceiling)"*. Reverting to unbounded collapsed CRPS **47×** to 7.22. The bound **created** the overshoot. |
| **today's `QUANTILE_EMIT_CEIL`** | `hydranet_inference.py:43` = **13.0** in log1p space | permits **442,412 deaths in one cell-month** — **3.9×** the observed sb maximum of 113,000. Its own comment: *"so the 36-step rollout stays finite (C-113 bloom)"* and *"the scored T=0 tail is **untouched**"* ⇒ it is a **rollout-safety** device, **not** a scoring bound. |
| **S6's proposal** | an **emit-time** bound from the observed conditional law | ⚠️ **untried.** It is neither of the above, and gradient death cannot apply to it. |

⇒ So S6 was not proposing something already falsified — the 47× failure was a *different*
intervention. ⛔ But §2's arithmetic caps what it could ever buy at about a third of a 26× gap, so
**S6 does not earn a place on the critical path.**

## §5 ⚠️ Three axes on which the 11–35× is not comparable to anything in the current programme

1. **40 lessons**, not 300. `MEMORY.md` on the training axis: *"40L = smoke only; 160L not
   converged."*
2. **"views-evaluation CRPS" on the calibration partition**, not the frozen lodestar ruler
   (`reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py`) every current number uses.
3. ⛔ **An unnamed finite-ensemble estimator.** Per #442/#459/M68, that choice alone moved a
   headline **−5.94% → −16.27%**, inverted a conclusion, and manufactured a turning point in a
   swept curve. The quantile comparison names no estimator at all.

## §6 ⛔⛔ And the precedent: this repo has already had a 40-lesson verdict inverted at 300

`project_persistence_reference_l300`: *"**M1 was measured at 40 lessons and nobody re-ran the
race.** At L=300 the WORST of 4 seeds beats a fairly-scored persistence at EVERY horizon (h36
2.22×)… The ruler is stable, the model moved."*

The quantile head's 11–35× is the same shape — a 40-lesson verdict, never re-run — and M1f's own
closing words call the residual *"a refinement, not viability."*

⇒ ⚠️ **This does not rescue H**, which §2 refutes on its own terms at any lesson count. It bears on
something else: whether the head deserves the arm.

## §7 What this does to epic #458

**Untouched — the epic's destination stands on its own:**
- ADR-073 §0's proof that the NB head **cannot** represent the law (`theta·log(1+mu/theta)` monotone
  increasing ⇒ no interior optimum on an empty cell, 99.66% of cells empty). Not a measurement, a
  derivation.
- The quantile head is **fully wired** and `QuantileLoss` is multi-τ pinball ⇒ *"uniform grid ⇒
  pinball-sum ~ CRPS"*, and `hurdle_quantiles_to_samples` composes with a gate ⇒ **no `theta` to
  collapse.**
- The quantile head has **never been run at 300 lessons on the frozen ruler.** The arm remains a
  config change.

**Changed:**
- ⛔ **H: refuted.** S3 closes. Epic acceptance criterion 3 is **met** — *"H is measured and reported
  either way. A refutation closes this epic honourably and is a real result."*
- ⛔ **S6 (bound the tail): recommend PARK**, with §2 as the reason. It was the epic's only
  implementation story and the only thing on the critical path that needed code.
- ⭐ **The arm's justification is now simpler and does not depend on H at all:** the current head
  provably cannot represent the law; a wired alternative can; its sole negative verdict is a
  40-lesson measurement on a different partition under an unnamed estimator that has since been
  shown to move headlines by 10 points — and this repo has already inverted one 40-lesson verdict at
  300.

⇒ **Recommended path: S4/S5 (σ̂) → S7 (prereg) → S8 (the arm).** Dropping S6 removes the only code
change from the epic and shortens the critical path.

🔵 **Open, and honestly open:** whether the bulk deficit (§3) is a 40-lesson artifact or a real
property of the head. ⛔ **Nothing on disk can answer it** — the only route is the 300-lesson arm
itself, which makes the arm the measurement rather than the confirmation. That must be written into
S7's prereg as the arm's scope, before it runs.
