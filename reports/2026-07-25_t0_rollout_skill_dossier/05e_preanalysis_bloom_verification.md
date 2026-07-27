# 05e — Pre-analysis plan: rigorous bloom-fix verification (Epic #193, S1)

**Pre-registered 2026-07-27, BEFORE the S5/S6 retrain+eval.** Locks the quantitative criterion for "the
bloom is fixed" and the production-default decision. Governs S6 (18-rollout eval) + S7 (verdict). Metrics =
frozen-ruler crps split + per-step magnitude trajectory (FAO-02: NO twCRPS/PIT).

## Hypothesis (H-BLOOM)

For every deployable family arm × seed, feeding back a **sample** (`rollout_feedback=sample`) keeps the
36-step rollout **bounded and in-distribution**, whereas feeding back the **mean** (`mean`) drives the
runaway (where that seed/arm is bloom-prone). The sample-feedback mitigation is universal even though the
bloom itself is model-dependent.

## Matrix (S5 trains, S6 evals)

- **Arms (valid family × composition, ADR-069):** gated_NB (`nb`+`soft_gate`), th_gated_NB
  (`nb`+`threshold_gate`, τ=baserate), ZINB (`zinb`+`self_zeroed`).
- **Seeds:** `torch_seed ∈ {42, 43, 44}` — **known & recorded in the artifact sidecar** (S5 fixes the
  seed-murk that confounded EXP-4).
- **Trained models:** `nb`×3 (serves gated_NB + th_gated_NB via emit composition) + `zinb`×3 = **6 models**.
- **Rollouts:** 3 arms × 3 seeds × {mean, sample} = **18**, scored on `tools/rollout_skill_score.py`.

## Quantitative criterion (locked — no post-hoc goalpost move)

Let `M(h)` = per-step **mean emitted magnitude** (log1p space) at horizon h, and `crps_all(h)` the ruler
score. Data in-range ceiling = `DATA_LOG_MAX` (the value `is_out_of_range` uses in
`test_rollout_stability_guard.py`).

- **"mean BLOOMS"** for an (arm, seed) cell ⇔ under `rollout_feedback=mean`, EITHER `M(36)` exceeds
  `DATA_LOG_MAX` (out-of-range), OR `crps_all(36) ≥ 5 × crps_all(1)` (order-of-magnitude runaway). (The
  EXP-1/hardening blooms were crps_all 0.14→5–15 and M→saturation — this threshold is deliberately
  conservative to avoid false "bloom" labels.)
- **"sample BOUNDED"** for an (arm, seed) cell ⇔ under `rollout_feedback=sample`, `M(h) ≤ DATA_LOG_MAX` for
  **all** h ≤ 36 AND `crps_all(h)` stays O(1) (no cell exceeds `2 × crps_all(1)` except the shared
  real-event spike at the terminal horizon, which is a truth feature, not a model runaway — flagged
  separately, not counted as unbounded).

## Pre-registered predictions

- **P1:** `sample` is BOUNDED in **9/9** cells.
- **P2:** `mean` BLOOMS in a **majority** of cells (≥5/9) — confirming the bug is real (model-dependent, so
  not necessarily 9/9; the ad-hoc hardening saw 3/4 models bloom).
- **P3:** T=0 (h=1) crps_all is **identical** between the mean and sample rollout of the same artifact
  (T=0-neutrality — a hard check, not a soft prediction).

## Pre-committed falsifiers

- **F-B1 (fix incomplete):** `sample` is NOT bounded in **any** cell → the mitigation does not universally
  hold → C-113 stays open; investigate that arm/seed. (P1 falsified.)
- **F-B2 (T=0 not neutral):** any cell's h=1 crps_all differs between mean and sample → the fix leaks into
  T=0, contradicting the ADR-070 construction argument → STOP and fix before any default ships.
- **F-B3 (bug not real):** `mean` blooms in **0/9** cells → the whole premise is unsupported on these
  seeds; the "fix" fixes nothing observable → re-scope. (Very unlikely given EXP-1/hardening.)
- **F-B4 (determinism):** re-scoring a cell changes any number → S2 #121 violated → fix before trusting.

## Decision rules

- **P1 (9/9 bounded) + P2 (majority bloom) + P3 (T=0 identical) ⇒ the bloom is FIXED** by
  `rollout_feedback=sample`; C-113 → resolved-with-evidence (S8).
- **Partial (sample bounded in most but not all) ⇒ evidenced-downgrade**, name the failing cells.
- **F-B2 fires ⇒ HARD STOP** (T=0 leakage) — the default must not ship.

## Production-default DECISION (S1, needs-decision — RESOLVED)

**`rollout_feedback` defaults to `sample` for registered family heads; `mean`/raw for legacy heads.**
Rationale: the fix is **T=0-neutral by construction** (ADR-070 §3 — the seed step emits before any feedback),
so sample-on for families has **zero cost to the scored T=0 product** and mitigates the bloom by default —
all upside, no downside on the shipped metric. Legacy heads cannot sample (no family) and stay byte-identical.
Overridable via explicit `rollout_feedback` for experiments. Wired in S4; verified T=0-identical in S6 (P3).
