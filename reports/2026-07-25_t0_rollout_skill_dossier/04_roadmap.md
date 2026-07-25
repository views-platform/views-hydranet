# 04 — Roadmap (phased, gated)

**Principle:** the free-running skill curve is GPU-free (re-score of data on disk) — so the *first result*
is cheap and comes *before* any re-inference. The expensive oracle re-run is gated on that first result
motivating it.

## Phase 0 — method-review + pre-register (no code)
- `expert-method-review` of `02_design` (DQ1–DQ4). → then `preregister` `05_analysis_plan`: predictions +
  falsifiers for Phase 2's free-running curve, *before looking*.
- **Gate:** `05` locked.

## Phase 1 — G1 loader + G4 origin set (pure Python, TDD) — the instrument
- Build `gather_all_horizons` (index by h = month − origin) + the identical-support origin set.
- **Red test first:** h=1 slice == current `gather_t0` byte-exact (faithfulness); origin-set count matches
  the pinned support; a synthetic per-horizon fixture scores as hand-computed.
- Reuse `crps_ensemble`/AP/Brier verbatim; wrap in a `rollout_skill_score.py` tool.
- **Gate:** pre-flight checklist §D green (minus the oracle-only rows).

## Phase 2 — the free-running skill curve (GPU-FREE) — the first result
- Re-score the existing lodestar/eval `origin_*` dirs (nb foundation + the 3 composition arms) at all h.
- Add climatology (reuse white_ranger) + persistence baselines (G3) on the same support.
- **Read-out:** skill-vs-horizon per target; locate crossover h_x (free-running vs climatology/mixture).
- **Log** (`07`) vs the Phase-0 falsifiers. This answers Q1 (depth) with zero new compute.
- **Multi-seed:** run across seeds 42/43/44 for any KEEP claim; single-seed = INDICATIVE.
- **Gate/decision:** does free-running beat climatology anywhere (h_x > 1)? If never (h_x = 1), the
  deployed rollout has no skill past T=0 — a decisive, publishable negative that reframes the whole epic.

## Phase 3 — G2 teacher-forced oracle (small GPU re-run) — bug vs ceiling
- Implement the `rollout_feedback` flag (default `predicted`; `teacher_forced` = realized-truth feedback),
  default-off, parity-proven; full suite + lint + determinism + floor-trap green.
- Re-infer the oracle rollout for the arms of interest (bounded eval; ask-before-long-batches honored by
  the pre-reg).
- **Read-out:** the bloom-cost gap `crps_free(h) − crps_oracle(h)`. → the **bug-vs-ceiling verdict** (§02.3).
- **Gate/decision:** gap large & growing ⇒ green-light the sample-feedback / GTF fixes (they have room to
  win). gap ≈ 0 ⇒ the ceiling dominates; pivot to "T=0 is the product" + document the predictability limit.

## Phase 4 — (conditional on Phase 3) score the fixes
- Only if Phase 3 says the bloom is a fixable bug: pre-register + score the fix ladder ON THIS RULER —
  sample-feedback (H-SAMPLE, cheap first), then τ-gated, then GTF (retrain). Each one variable, each
  scored for *skill* not boundedness. The ruler makes each a clean A/B.

## Phase 5 — promote
- Ruler validated ⇒ proposed ADR ("T>0 rollout skill evaluation") + archive the dossier; route any risks to
  `register-risk`. The fix that wins Phase 4 gets its own ADR citing this measurement ADR.

## Dependency graph
```
Phase0 (prereg) ─▶ Phase1 (loader,TDD) ─▶ Phase2 (free-running curve, GPU-FREE) ─▶ decision Q1
                                                                    │
                                             Phase3 (oracle flag+run) ─▶ bug-vs-ceiling verdict Q2
                                                                    │
                                              Phase4 (score the fixes) ─▶ Phase5 (promote/ADR)
```
