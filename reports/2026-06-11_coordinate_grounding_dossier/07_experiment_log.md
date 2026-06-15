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

---

## Determinism validation (pre-registered, 2026-06-15) — gate before any coordinate run

**Why first:** the 6-run baseline above is C-119-confounded. Before a single coordinate verdict can stand, the
pipeline must be proven **bit-reproducible** at real scale (the C-119 fix `daab1c1` is validated on a tiny config in
`tests/test_training_engine.py`; this confirms it holds on the production violet config).

**Pre-registration.** Two no-coords runs from `views-models/models/violet_visitor/`, same data + same seed (42),
post-`daab1c1`:
- **Run 1:** `python main.py -r calibration -t -e -re` (fetches viewser data, caches parquet).
- **Run 2:** `python main.py -r calibration -t -e -re --saved` (**same cached data**, fresh independent training).

**Pass = bit-identical**, judged by the reliable signals only (procedure + rationale: `../reproducibility_runbook.md`):
1. weight-**TENSOR** hash of the two artifacts **identical** (NOT `.pt` file sha — that embeds non-deterministic zip
   mtimes and is unreliable), **and**
2. every `origin_*/{target}/y_pred.npy` **`np.array_equal`**, **and**
3. `scripts/mcr_readout.py` MCR/CRPS **match**.

Compare with `scripts/compare_run_determinism.py <artifact1> <artifact2> <preds1> <preds2>`.

**Falsifier:** any difference in weights, `y_pred`, or MCR → the fix does **not** hold at scale → **STOP**, do not
proceed to #113/#105, re-investigate. **Verdict + Run-1-as-baseline only on a clean pass.**

**RESULT (2026-06-15) — ✅ PASS, fixed-fixed.** Two no-coords runs, seed 42, Run 2 `--saved` (identical frozen
data), post-`daab1c1`:
- weight-**TENSOR** hash **identical**: `2a7667c5…` (both artifacts), **and**
- **78/78** `origin_*/{target}/y_pred.npy` `np.array_equal` (13 origins × 6 targets), **and**
- MCR/CRPS point estimates **identical** — STEP-1 sb/ns/os 0.369/0.351/0.589; **FULL 3.702/4.497/7.005**; CRPS to 4 dp.

No falsifier fired. The C-119 fix holds at **production scale**. Both runs exited 137 *after* everything was written
(known-spurious C-116 publish-OOM, verified on disk — not a failure). **Run 1
(`calibration_model_20260615_015644.pt`) is now the deterministic no-coords baseline of record** — the FULL MCR
sb·ns·os **3.70 · 4.50 · 7.00** any coordinate experiment must beat. The earlier 6-run sweep (C-119-confounded) is
superseded. → Cleared to resume **#113 Phase 2** on a trustworthy seam → **#105**. (Also logged:
`../reproducibility_runbook.md` §4.)
