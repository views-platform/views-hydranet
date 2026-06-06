# 07 — Experiment Log (append-only)

**Date:** 2026-06-06 · **Status:** seeded (skeleton) · **Dossier:** [00_README](00_README.md)

Append-only ledger of every run + outcome — **including negatives/postmortems**. Each entry links its pre-registration (`05` or a `preanalysis_*.md`) and records its **verdict against the pre-registered falsifiers**. This is a Popperian record *and* a meta-evaluation corpus (later used to assess the workflow + the skills) — **not a highlight reel**. No ad hoc rescue.

## Entry format
```
### EXP-NN — <title>  (YYYY-MM-DD)
- Pre-registration: <link to 05 / preanalysis_*.md>
- One variable: <the single change vs baseline>
- Driver / artifact / results: <run_*.sh · model_*.pt · logs/*_RESULTS.txt>
- Readout: <diagnose_io_gain attractor · step-wise CRPS · MCR/PIT/coverage · sharpness>
- Verdict vs falsifiers: <which fired / none>
- Decision: <proceed / escalate (B2/ZITD) / postmortem link>
```
Legend: ✅ predictions held · 🔴 falsifier fired · ⚪ inconclusive.

---

## Precursors (done — the evidence base this program builds on)
- **`results_freezeh_ablation.md`** — freeze_h inert; divergence rides the prediction→input feedback, not the recurrent state.
- **`results_io_gain_diagnostic.md`** — violet's free-running map settles out-of-range (~log 40 → expm1 ~1e17); `diagnose_io_gain` is the fast readout.
- **`results_feedback_clamp.md`** — clamping the feedback is a safety rail, not a fix (ramps-to-ceiling).
- **`preanalysis_balancer_sweep.md` §RESULT** — F2 fired (seed4_frozen → inf): freezing is seed-fragile; exposure bias is the root → do rollout training.

## Planned

### EXP-01 — B1 pushforward MVP  (planned)
- Pre-registration: [`05_analysis_plan.md`](05_analysis_plan.md)
- One variable: `rollout_horizon` (pushforward stability term) on violet/seed-42, **active** balancer.
- Gated on: `03 §5` pre-flight green (the §3 harness build, P1/P2) + GPU.
- Baselines: `…233938` (active exploder, stability) · `…051634` (frozen healthy, calibration).
- *(awaiting execution — no outcome yet; this is the open pre-registration the dossier `status` tracks)*
