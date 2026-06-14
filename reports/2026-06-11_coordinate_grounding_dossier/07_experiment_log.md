# 07 — Experiment Log (append-only)

> ⚠️ **C-119 confound (2026-06-15).** Everything below was run on a **non-deterministic** training pipeline
> (~20% run-to-run variance at a fixed seed; root cause C-119, fixed in `daab1c1` —
> `../postmortem_training_nondeterminism_init_rng_drift.md`). The **6-run "baseline of record" is superseded**
> and the **coordinate experiment #110 "coords made it worse" verdict is VOID** — the no-coords baseline
> alone swings FULL MCR sb 2.99–3.69 run-to-run, comparable to the claimed coord effect. The coordinate
> question must be re-run on the deterministic pipeline before any verdict stands.

**Date:** 2026-06-11 · Negatives first-class. Each entry: one variable, pre-registration link, verdict vs
falsifiers. Canonical metrics live in `../RESULTS_LOG.md`; this is the narrative ledger.

## Entry format
```
### EXP-NN — <title> (YYYY-MM-DD) <✅ held / 🔴 falsifier fired / ⚪ inconclusive>
- Pre-registration: 05 §<n>
- One variable: <change vs baseline>
- Artifact / run / results:
- Readout: gate forensic · rollout biopsy · MCR (step-1/full) · CRPS/QS99/Brier (FAO)
- Verdict vs falsifiers (05): <which fired / none>
- Decision: <next roadmap box / escalate / ship>
```

---

## The "before" (baseline of record) — bounded hurdle-NB, 6-run sweep (2026-06-11)

**Config:** hurdle-NB head (softplus μ + class-weighted BCE gate, learned per-target θ), exact hurdle mean
at inference; 3 arms × 2 seeds, 40 lessons, frozen balancer, scheduled sampling off.

**Bounded? = ✅ (all 6).** No explosion to ∞ — C-113 is bounded. This is the foundation coordinates build on.

**MCR readout (`scripts/mcr_readout.py`), step-1 / full, sb·ns·os:**

| Arm (θ, pw, seed) | STEP-1 sb/ns/os | FULL sb/ns/os |
|---|---|---|
| S1 (1.0, 10, 42) | 0.41 / 0.40 / 0.45 | 2.55 / 3.38 / 4.58 |
| S1 (1.0, 10, 4)  | 0.58 / 0.40 / 0.70 | 5.46 / 5.44 / 7.42 |
| S2 (1.0, 25, 42) | 0.52 / 0.32 / 0.75 | 2.36 / 2.90 / 7.72 |
| S2 (1.0, 25, 4)  | 0.76 / 0.40 / 1.36 | 5.63 / 5.78 / 13.36 |
| S3 (0.3, 10, 42) | 0.54 / 0.29 / 0.55 | 3.31 / 2.98 / 6.63 |
| S3 (0.3, 10, 4)* | 0.67 / 0.62 / 1.67 | 3.32 / 4.41 / 7.81 |

*S3 seed4 on a partial prediction set (disk-full truncated its eval; ~30% fewer positive cells) — bounded
and consistent, lower-confidence.

**Diagnosis (from the diagnostic plots, `reports/plots/diagnostics/110626_*`):**
- **Gate over-fires 4–16×** (classification "Detection Bias Pulse"): well-calibrated for ~15 lessons, then
  the event-ratio climbs to 4–16× and **worsens through training** — **worse at higher `pos_weight`**
  (pw=25 → ~16×, pw=10 → ~12×).
- **Magnitude over-predicts ~40–50×** (regression forensic): true mean ≈ 0.03, predicted mean rises
  0.83 → 1.78 over the 40 lessons — over-prediction that **grows** with training.
- **Blob-bloom** (rollout biopsy): ground truth is sparse/stationary; the prediction blooms localized
  blobs of conflict in **structural-zero regions** over the rollout — the spatial face of FULL MCR 2.4–13.

**Read:** step-1 under-predicts (~0.4–0.7), the rollout over-predicts (2.4–13) — bounded but drifting up.
The lever is **spatial grounding + exposure bias**, not θ or `pos_weight` (the two HP knobs don't fix it;
higher pw makes the gate worse). This is the baseline the coordinate experiment must beat.

*(First coordinate entry — EXP-01 — lands when roadmap box 3 runs. Not before.)*
