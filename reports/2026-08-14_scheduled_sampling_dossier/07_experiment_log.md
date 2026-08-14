# 07 — Experiment log (append-only; negatives first-class)

## EXP-SS-1 — ε dose-response sweep {0, 0.1, 0.25, 0.5} — 2026-08-14 — ❌ NEGATIVE (SS falsified as the rollout-collapse fix)

**Pre-registration:** `05_analysis_plan.md` (LOCKED 2026-08-14). One variable = `ss_epsilon_max`;
`ss_feedback='sample'`, seed 42, 40L, africa datafactory (fresh pull), `truncated_nb`+`soft_gate`.
**Driver/artifacts:** `scratchpad/ss_sweep_driver.sh` (setsid, config floor trap-restore, per-arm
score→delete). Cubes: ε0.0=`predictions_calibration_20260814_003058` (TF re-emit),
ε0.1=`…_061215`, ε0.25=`…_080018`, ε0.5=`…_082726`. Scores: `results/score_eps*.csv`.

### Baseline correction (logged before the verdict — this is why correctness-first mattered)
The **ε=0 (teacher-forced) arm re-scored on all 13 origins with the fixed sampler OVERTURNS the
earlier 2-origin "bloom" read.** Earlier (2 origins + the 128-round buggy sampler) I reported
act_ratio 1.6×→18×→44× (a bloom). The clean read is the opposite — a residual **collapse**:
act_ratio **1.41 → 0.29 → 0.27** (h1/18/36). So truncated_nb fixes T=0 (1.41 vs old plain-NB
0.03) and *mitigates* the rollout collapse (0.03→0.27 at h36) but does NOT eliminate it. The
earlier "bloom" was a slow-sampler + 2-origin artifact (C-259-class: invalid knowledge from a wrong
implementation, caught by the sampler perf-fix + full-origin re-score).

### Result vs pre-registration
| h | ε=0 act_ratio (AP) | ε=0.1 | ε=0.25 | ε=0.5 |
|---|---|---|---|---|
| 1  | 1.41 (0.298) | 1.48 (0.285) | 1.56 (0.278) | 1.60 (0.204) |
| 18 | 0.29 (0.007) | 0.34 (0.007) | 0.17 (0.010) | 0.23 (0.009) |
| 36 | 0.27 (0.008) | 0.30 (0.009) | 0.16 (0.010) | 0.21 (0.009) |

- **P1 (act_ratio flattens monotonically toward 1) — FALSIFIED.** No monotone trend
  (h36: 0.27→0.30→0.16→0.21); rollout AP flat at ~0.01 for every ε.
- **P2 (T=0 not collapsed) — holds on act_ratio, but T=0 gate AP DEGRADES with ε (0.30→0.20).**
- **F1 FIRES:** scheduled sampling does not move the gate's rollout precision at all → the wall is
  **gate precision**, not input-exposure bias.
- **F2 FIRES:** non-monotone; higher ε degrades T=0 with zero rollout gain (the undertrained-40L
  instability the skepticism ledger flagged).
- **F-DEGEN:** none — all 4 arms trained/emitted cleanly (ERROR.log empty throughout); the hardened
  SS piping (validator + parity, `c07a352`) ran end-to-end for the first time without a crash.

### Verdict + honest null-scoping
**Scheduled sampling (a *partial* (c): per-step true target) is falsified as the fix for the
`truncated_nb` rollout collapse at ε≤0.5 / 40 lessons.** It buys no rollout occurrence skill and
harms T=0 at dose. This does NOT prove exposure bias is irrelevant (skepticism ledger 1–3: 40L is
undertrained; a fuller (c) — rollout-level / distribution-matching objective — is untested). But it
redirects the program: the invariant is **the gate has no rollout precision (AP 0.30→0.01)**, and
neither the truncated body nor SS moved it.

## EXP-SS-2 — mechanism-split probes (oracle + feedback-content), same TF artifact — 2026-08-14

Emit-only (no retrain) from `calibration_model_20260814_003058.pt`, africa truth, sb, h=1/18/36.
Scores: `results/score_{oracle,meanfb}.csv`; the free-`sample` row is EXP-SS-1's ε=0 arm.

| rollout_feedback | h1 AP / act_ratio | h18 | h36 | rollout mode |
|---|---|---|---|---|
| **sample** (free) | 0.298 / 1.41 | 0.007 / 0.29 | 0.008 / 0.27 | **collapse** (under-fire) |
| **mean** | 0.298 / 1.41 | 0.009 / 110.2 | 0.010 / 95.8 | **bloom** (over-fire ~96×) |
| **teacher_forced** (oracle) | 0.298 / 1.41 | 0.301 / 1.26 | 0.271 / 1.22 | **works** (AP holds) |

### Verdict (mechanism — decisive, by elimination)
- **Oracle (real inputs) keeps AP high across the whole horizon (0.30→0.27) and activation calibrated
  (~1.2).** So the model IS skilful at h36 given in-distribution inputs ⇒ the free-rollout collapse is
  **NOT hidden-state / recurrent drift** (overturns the prior C-222-based bet) and **NOT input-exposure
  quantity** (SS null, EXP-SS-1).
- **Both naive feedbacks BRACKET the target oppositely** — `sample`→collapse (0.27), `mean`→bloom (96×) —
  and **neither recovers rollout AP** (~0.01). So it is not "pick a better naive feedback."
- **⇒ ROOT = the distributional gap.** The model cannot emit a fed-back field that *looks like real
  conflict history* (sparse, persistent, integer, spatially coherent). Confirmed by elimination:
  SS ✗, feedback-content ✗, hidden-state ✗, oracle ✓.

### Next lever (justified, not premature)
**Distribution-matching / rollout-aware training** (the genuine lever (c)): a discriminator/adversarial
term on the fed-back field, or a rollout-level realism objective — so the generated field stays
in-distribution. Every cheaper option is now empirically exhausted.
