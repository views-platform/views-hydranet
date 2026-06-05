# Results — C-111 Balancer-Freeze Bisect (C-113)

**Date:** 2026-06-05
**Pre-registered plan:** `reports/preanalysis_balancer_bisect.md`
**Verdict:** **H CORROBORATED — the C-111 fix (un-freezing the MultiTaskLoss balancer) is the cause of the C-113 acute runaway.** Freezing the balancer (reverting C-111) keeps violet's free-running map in-range; the active balancer drives it out of range. Both arms trained on GPU (device-matched — the earlier CPU confound is gone).

---

## 1. Data — free-running attractor (`diagnose_io_gain.py`, retrain-free, on each fresh artifact)

| Arm | balancer `log_vars` | artifact | rollout settles at (log) | `expm1` (counts) | verdict |
|-----|--------------------|----------|-------------------------:|-----------------:|---------|
| **CONTROL** (C-111 default) | trainable | `..._20260604_233938.pt` (GPU) | **~15–17** | ~1e6–1e7 | **PATHOLOGICAL (out-of-range)** |
| **TEST** (pre-C-111) | frozen at 0 | `..._20260605_051634.pt` (GPU) | **~4–5** | ~70–215 | **healthy (in-range)** |

Frozen arm stays in-range for **all three synthetic seeds** and under **both** `freeze_h='none'` and `'all'` (state-independent, consistent with the freeze_h ablation). Local operator norm `‖J‖₂ > 1` for both arms (as before — not the discriminator; the attractor level is).

## 2. Pre-registered verdict (plan §4–§5)

- **Prediction:** CONTROL out-of-range, TEST in-range. **Both confirmed.**
- **F1** (frozen still explodes) — did **not** fire (frozen settles at log ~4–5).
- **F2** (control fails to reproduce) — did **not** fire (control reproduced an out-of-range attractor on a clean GPU retrain).
- **F3** (diagnostic vs eval disagree) — n/a (diagnostic only; see caveats).

⇒ **The active balancer is the driver of the acute runaway.** The C-111 fix — adding `MultiTaskLoss.log_vars` to the optimizer so the Kendall–Gal homoscedastic weighting actually learns — destabilises the autoregressive dynamics for this seed. The pre-C-111 frozen balancer (equal weighting) was **accidentally protective**: "the bug was load-bearing." This explains why the model was stable for years and only exploded on the post-C-111 retrain (`memory: project-explosion-is-regression`).

## 3. Caveats (honest)

- **Single seed per arm** (C-112/C-119). The *qualitative* contrast is huge and robust to run-to-run variance — frozen log ~4–5 vs active log ~16, i.e. ~5 orders of magnitude apart after `expm1` — but a multi-seed confirmation is warranted before any production claim.
- **Synthetic-seed diagnostic**, not a full real-data eval. Mitigated: the same diagnostic previously matched the real eval for both pink (in-range) and violet (out-of-range, ~1e17 ⇄ observed CRPS). A confirmatory `--evaluate` on the frozen artifact would close the loop.
- **Magnitude variance:** the control here settled ~1e7 vs June-3's ~1e17 — run-to-run variance (C-119); qualitatively out-of-range either way.

## 4. Disposition — what this means

- **C-111 is the acute cause of C-113.** C-113's acute runaway is a **regression introduced by C-111**, not a fundamental architectural flaw.
- **Do NOT just freeze-and-forget.** Freezing reverts C-111's *intent* (a learnable balancer is desirable). The real fix is to **let the balancer learn but constrain it** — e.g. weight-decay / bound the `log_vars`, lower their learning rate, or clamp their range — so it cannot drive a head into the divergent regime. Freezing is the *diagnostic*; regularisation is the *fix*. (A frozen-balancer config is a safe immediate fallback if a stable model is needed now.)
- **The ZITD / distributional-head dossier is still justified — it targets the *chronic* problem** (MCR ≪ 1, no calibrated uncertainty), which is orthogonal to this acute regression (`02_design §0.4`). The bisect doesn't kill the dossier; it cleanly separates the two tracks: *acute = regularise the balancer*, *chronic = count-likelihood head*.
- **`freeze_h` and the feedback clamp** remain confirmed non-fixes (freeze_h inert; clamp a bounded-but-degenerate rail) — the balancer was upstream of both.

## 5. Next (proposed, not done)

1. Confirm on ≥1 more seed (frozen vs active) — cheap now that the GPU is healthy.
2. Design the balancer-regularisation fix (bound/decay `log_vars`), behind a flag; the `freeze_multitask_balancer` flag already exists as the extreme.
3. A confirmatory `--evaluate` on the frozen artifact (CRPS/MCR) to verify the in-range attractor translates to healthy real-data metrics.
4. Proceed with the ZITD dossier on its own track for the chronic problem.
