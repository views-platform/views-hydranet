# Results — In-Domain Feedback-Input Clamp (C-113)

**Date:** 2026-06-04
**Pre-analysis:** `reports/preanalysis_feedback_clamp.md` (pre-registered)
**Verdict:** **Partial — a useful safety rail, NOT a fix for the worst head.** The clamp prevents the astronomical runaway (violet `lr_sb` CRPS 2.13e17 → 798), fully recovers two of three heads and the early horizon, and is **benign on the healthy model** — but it **triggers pre-registered falsifier F2** for `lr_sb`: the prediction *ramps to the clamp ceiling over the horizon and pins there*, producing bounded-but-degenerate over-prediction (MCR ~56,000). The underlying pathological attractor is untouched; the clamp only caps where it lands.

---

## 1. Data (step-wise CRPS; MCR_sample in parens), baseline → clamped

Clamp = per-target log1p ceiling `[sb 10.78, ns 7.60, os 12.09]`, inference-only, n=16.

| Model | head | baseline CRPS | **clamped CRPS** | clamped MCR | verdict |
|-------|------|--------------:|-----------------:|------------:|---------|
| **violet** | lr_sb | 2.13e17 | **798.8** | 56,224 | bounded but **degenerate (F2)** |
| **violet** | lr_ns | 2.78e9 | **0.71** | 302 | recovered to healthy ✓ |
| **violet** | lr_os | 54.5 | **0.15** | 28.9 | recovered to healthy ✓ |
| pink (control) | lr_sb | 0.1325 | **0.1325** | 0.005 | unchanged ✓ |
| pink (control) | lr_ns | 0.031 | **0.0313** | 0.41 | unchanged ✓ |
| pink (control) | lr_os | 0.051 | **0.0511** | 0.015 | unchanged ✓ |

**Pink reproduced its baseline to 4 sig figs** — confirms (a) metrics valid despite the post-eval OOM (exit 137, same teardown as prior runs; wandb summary dumped first), and (b) the clamp is a **no-op on a healthy model** (it never approaches the ceiling on real data).

## 2. The decisive detail — violet `lr_sb` ramps to the ceiling (F2)

Per-step `lr_sb` under the clamp (from `eval_calibration_lr_sb_best_step_*.parquet`):

| step | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | … | plateau |
|------|----|----|----|----|----|----|----|----|---|---------|
| CRPS | 0.27 | 0.58 | 3.3 | 19.7 | 69 | 138 | 209 | 268 | ↗ | ~800 |
| MCR  | **0.90** | 5.7 | 70 | 557 | 2192 | 4229 | 6459 | 8722 | ↗ | ~56,000 |

**Step 1 is healthy (MCR ≈ 0.9, CRPS 0.27).** Then it degrades monotonically: the head keeps pushing toward its high attractor, gets capped at log 10.78 (`expm1` ≈ 48,000 counts) each step, so by mid-horizon predictions are pinned near the ceiling while truth ≈ 0 → CRPS in the hundreds, MCR in the thousands. This is exactly the pre-registered **F2 "bounded-but-degenerate (pinned at ceiling)"** failure — reached via a ramp, not an instant wall.

## 3. Pre-registered falsifiers — outcome

- **F1 (ineffective):** NOT triggered. Hugely effective at preventing catastrophe (lr_sb −14 orders; lr_ns −9 orders to healthy; lr_os to healthy).
- **F2 (bounded-but-degenerate):** **TRIGGERED for `lr_sb`** — ramp-to-ceiling, MCR ~56,000, predictions pinned at the in-domain max (~48k counts) over the horizon. `lr_ns`/`lr_os` do *not* trigger F2 (they stay below their ceilings → healthy).
- **F3 (not benign on healthy):** NOT triggered — pink identical to baseline.
- **F4 (MCR collapse → 0):** NOT triggered; the opposite — `lr_sb` MCR is pathologically *high* (over-prediction), which is the F2 pinning signature, not collapse.

## 4. Honest reading

The clamp is a **safety rail, not a cure**:
- ✅ **Averts catastrophe.** No more 1e17 / NaN-risk; every metric finite and computable. For a production guard against the `expm1` overflow, this alone has value.
- ✅ **Recovers most of the output.** `lr_ns` and `lr_os` return to fully healthy; `lr_sb` is healthy at step 1–2. The early forecast horizon is usable.
- ✅ **Benign + retrain-free.** No change to the healthy model; one inference-only line; emitted values never directly capped.
- ❌ **Does not fix the diverging head.** `lr_sb` is converted from *explosion* to *ceiling-pinned over-prediction* (MCR ~56k). The clamp caps **where** the runaway lands; it does not stop the map from running there. Pinning at the data **max** is itself a bad operating point (predict the worst-ever value everywhere → gross over-prediction) — confirming the ceiling is a *bound*, not a *calibrated target*.
- ⚠️ **Escalation cost stands** (pre-analysis §3.5): a head pinned at its historical max is exactly the "can't see beyond the worst observed event" failure — here it manifests as over-prediction rather than signal loss.

**Net:** the diagnostic prediction (clamping the feedback breaks the astronomical runaway) held — but "bounded" was necessary, not sufficient, exactly as the skepticism ledger warned. The clamp is worth keeping as an **optional guard rail** (default off; finite-output safety net beneath a real fix), not as the fix.

## 5. Disposition

- **C-113 stays OPEN.** The feedback clamp is logged as a validated *mitigation* (catastrophe-prevention + 2/3-head recovery, retrain-free) but **not** a resolution — `lr_sb`'s attractor is untouched.
- **Durable fix still required**, now better targeted: lower the input→output attractor so `lr_sb` doesn't ratchet to its ceiling — spectral-norm/Lipschitz on the input→output path, pushforward/GTF training, or a count-likelihood head (which also dissolves `expm1` and the MCR problem). The clamp can sit underneath any of these as a backstop.
- **Code:** `feedback_clamp_log1p` stays in the schema (default None → off); `_clamp_feedback` retained. Keep unmerged-as-default; a future ADR could enable it as a guard rail with the F2/over-prediction caveat documented.
- Catalogue and register updated.

## 6. Honesty notes

- Exit 137 on both runs = post-eval `queryset pg_metadata` OOM teardown (as in prior runs); metrics finalized/synced before the kill (pink's exact baseline reproduction proves validity).
- Clamp confirmed *active*: violet's `lr_sb` plateau (~CRPS 800, predictions ~`expm1(10.78)`) and the recovery of `lr_ns`/`lr_os` only make sense if the per-channel ceiling engaged; pink's no-op is the negative control.
- The `lr_sb` ramp (healthy step-1 → pinned late) is consistent with the diagnostic's "ratchet across steps" mechanism and the freeze_h ablation (input-loop driven).
