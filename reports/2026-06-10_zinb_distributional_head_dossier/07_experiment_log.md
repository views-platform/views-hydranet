# 07 — Experiment Log (append-only)

**Date:** 2026-06-10 · Negatives first-class. Each entry: one variable, pre-registration link, verdict
vs falsifiers. Canonical metrics live in `../RESULTS_LOG.md`; this is the narrative ledger.

## Entry format
```
### EXP-NN — <title> (YYYY-MM-DD) <✅ held / 🔴 falsifier fired / ⚪ inconclusive>
- Pre-registration: 05 §<n>
- One variable: <change vs baseline>
- Artifact / run / results:
- Readout: Bounded? (diagnose_io_gain) · CRPS/QS99/Brier (FAO) · MCR (diagnostic) · positive-subset
- Verdict vs falsifiers (05): <which fired / none>
- Decision: <next checklist box / escalate / revert>
```

---

## The "before" (baseline of record — `../RESULTS_LOG.md` row 1)
**Current model (2026-06-10), R4 = hurdle `lognormal_nll` + SS 0.5, 40 lessons, active balancer.**
Eval **FAILED — "Input contains infinity"**; the FAO pipeline produced no CRPS/QS99/Brier. Full-rollout
MCR ≈ 3.4e33 / 3.1e33 / 7.5e33 (our `mcr_readout`). **Bounded? = 💥. Eligible? = No.** This is the
exploded, unscoreable starting point the ZINB head must beat. SS bracket is exhausted (0.25/0.5/1.0 all fail).

*(First ZINB entry lands when #102 runs — not before.)*
