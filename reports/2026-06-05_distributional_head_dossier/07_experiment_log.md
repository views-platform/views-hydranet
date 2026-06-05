# 07 — Experiment Log (append-only)

**Date opened:** 2026-06-05 · **Status:** seeded (skeleton) · **Dossier:** [00_README](00_README.md)

Append-only ledger of every distributional-head experiment — **including negative results** (per the session's falsification discipline). Newest at the bottom. One row per run; link out to the driver, the pre-analysis plan, and the results doc/postmortem. Do not edit past entries except to add a cross-link or correct a typo.

**Entry format**

```
### EXP-NN · <short title> · <date> · <status>
- **Plan (pre-reg):** <link to 05 / preanalysis_*.md>
- **Config / variable:** <the one thing changed>
- **Driver / artifact:** <scripts/run_*.sh · artifact ts · logs/*_RESULTS.txt>
- **Readout:** <diagnose_io_gain attractor> → <eval: CRPS / MCR / calibration / zero-rate>
- **Verdict vs falsifiers (05 §4):** <which fired / none> → <SUCCESS / FALSIFIED / INCONCLUSIVE>
- **Decision:** <next per 05 §8 / 04>
```

Status legend: `planned` · `running` · `done` · `falsified` · `inconclusive` · `superseded`.

---

## Precursors (C-113 work that motivated this program — not ZITD experiments)

Context only; full records in the parent `reports/`:
- **freeze_h ablation** (`results_freezeh_ablation.md`) — divergence rides the prediction→input loop, not the recurrent state.
- **io-gain diagnostic** (`results_io_gain_diagnostic.md`) — violet's free-running map settles out-of-range (log ~40); pink in-range; the retrain-free readout we'll reuse.
- **feedback clamp** (`results_feedback_clamp.md`) — bounding the feedback is a safety rail, not a fix (bounded-but-pinned) → motivates a structural output fix = this program.
- **C-111 balancer bisect** (`preanalysis_balancer_bisect.md`) — acute-regression test, in flight; orthogonal to ZITD (`02 §0.4`).

## Distributional-head experiments

### EXP-01 · ZITD MVP — violet, fixed ρ, mean rollout · TBD · **planned**
- **Plan (pre-reg):** [`05_analysis_plan.md`](05_analysis_plan.md)
- **Config / variable:** `output_distribution="zitd"`, fixed `ρ=1.5`, softplus `μ`, mean autoregressive feedback; one variable from baseline (the head).
- **Gated on:** [`04_roadmap`](04_roadmap.md) M1 (`ZITDLoss`+sampler tested) + M2 (head behind flag, baseline parity).
- **Readout (planned):** `diagnose_io_gain`(E[y]) in-range? → eval CRPS / MCR / PICP / zRMSE / `P(Y>0)` AUC / zero-rate, vs `s0` + Tobit.
- **Status:** not started — blocked on M1/M2 (Tweedie density implementation).

<!-- append EXP-02 … below as P4 ablations run (learned ρ · sampled rollout · per-target φ,ρ · classifier transition · multi-seed) -->
