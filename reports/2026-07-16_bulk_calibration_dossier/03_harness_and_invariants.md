# 03 — Harness & Invariants (the crown jewel: the locked bulk-calibration metric)

Every time this project was fooled, a **measurement lied** (pooled-MCR hid the timid body ×3; rollout-
pooling faked the quantile "degeneration" ×1). So the bulk-calibration metric is locked here with
belt-suspenders-diapers rigour, BEFORE any run.

## A. Invariant taxonomy
**Hard (never break):** fail-loud / no-silent-clamp on the SCORED T=0 quantity (a clamp may bound the
rollout, never the scored bulk); output-contract `(N,S)` parity; reproducibility (seed lock); full suite
green; **the baseline stays byte-identical when the new flag is off**; the eval is T=0-only, positives-only,
per-cell (never pooled-mean).
**Deliberately changed (behind a default-off flag):** the body's TARGET (winsorized) and body LOSS (τ-pinball
lifter) — replacing plain MSE-on-log1p. Current behavior we intend to replace: the timid log-median body.
**Respect while changing:** the FROZEN gate (dense-mse+wBCE pw2) — its Brier must not move; the tail
handling (parked, not degraded); the (N,S) sample pipeline + Path A carrier.

## B. Standing harness (already exists — reuse)
- **OCP flags:** `LOSS_REG_REGISTRY` + `output_distribution`/`loss_reg` config validators — add the new loss
  here, default-off (baseline byte-identical).
- **Reproducibility:** `torch_seed/np_seed=42`, ReproducibilityGate/lock_entropy (GPU bitwise NOT guaranteed).
- **Fast retrain-free readout:** `reports/2026-07-15_quantile_head_build_dossier/tools/t0_score.py` (T=0-only,
  rollout NEVER pooled — validated: reproduces the audit CRPS) + honest per-cell metrics in
  `scripts/audit_run.py` / `.../tools/score.py` (`ratio_med`, `within2x`, `within2x_rescaled`, `spearman_pos`).
- **Locked baseline:** **white_ranger** (conflictology) T=0 scorecards `eval_calibration_<t>_step_*.parquet`
  (`step01`=T=0); + the same-config dense-mse pw2 A/B baseline.
- **Run discipline:** one-heavy-job-at-a-time; background+notify; config trap-restore; timestamped preds.
- **Negative discipline:** `07_experiment_log` + postmortems.

## C. New harness THIS program must build first (the gaps)
1. **THE BULK-CALIBRATION METRIC** (§D — extend `t0_score.py`; retrain-free; anchor it on the existing
   dense-mse pw2 predictions before any new run).
2. Winsorize target helper (per-cell robust cap) + unit tests.
3. τ-pinball log-space loss in `LOSS_REG_REGISTRY` (default-off) + NaN/Inf guards + unit tests (minimiser
   is the τ-quantile; gradient finite; off ⇒ baseline unchanged).
4. Same-seed A/B driver (baseline vs +dial, seed 42 + 2 more).

## D. ⭐ THE LOCKED BULK-CALIBRATION METRIC (do not change after the first run)

**Plain-English:** *On the first forecast month only, on cells that actually had conflict, in the normal
range only, what fraction of the true amount do we predict — and is it moving from "far too low" toward
"about right", without breaking the gate?*

**Scope — three mandatory filters (per target sb/ns/os):**
1. **T=0 ONLY** — first forecast month per origin; the 36-step rollout is NEVER pooled in.
2. **Positive-truth ONLY** — cells with `truth > 0` (zeros are the gate's metric, not the body's).
3. **BULK ONLY** — `truth ≤ cut`. The extreme-tail positives (`truth > cut`) are reported separately, PARKED.

**The cut (LOCKED, defined on TRUTH, on the TRAINING months, before eval):** a fixed count per target = the
**98th percentile of positive training truths**; ALSO report at **97 and 99** so the verdict is not sensitive
to the choice. Identical cut applied to baseline and test model. (The training winsorize may use a per-cell
adaptive cap; the METRIC uses this fixed global cut for interpretability + stability.)

**Headline — `ratio_med`** = **median over bulk-positive cells of (E[y] / truth)**, where `E[y]` = mean over
the S posterior samples. Baseline ≈ 0.05–0.11. **PASS bar (user-owned): `ratio_med ∈ [0.7, 1.3]`** at T=0.
🩹 *per-cell median → the pooled-mean/MCR cancellation cannot occur. MCR is BANNED as a headline.*

**Supporting reads (always reported, never a lone number):**
- `within2x`, `within2x_rescaled` — sharpness / is-it-genuine-vs-a-global-rescale.
- full **ratio spread** p10/p50/p90 of `E[y]/truth` on bulk positives — even vs lopsided lift.
- `spearman_pos` — ranking sanity.

**Guardrails — must NOT regress vs the un-lifted dense-mse pw2 (else it's over-firing, not calibration):**
gate **Brier**, overall **CRPS**, **QS99** — all at T=0, within noise.

**Comparison — strict same-seed A/B:** dense-mse+wBCE(pw2) WITHOUT the dial vs WITH; seed 42 (+2 seeds);
everything else identical. Effect = the delta; **month-block bootstrap CI on Δ`ratio_med` must exclude 0.**

**Reproducibility:** ≥3 seeds (floor is seed-bimodal); run-to-run variance is a disqualifier.

**Graduation:** iterate on the calibration split; the VERDICT requires the SAME metric on the **validation
partition** 🩹 *(count_mean lifted on calibration then collapsed OOS — the exact trap).*

## E. Pre-flight checklist (must be green before the FIRST training run)
- [ ] bulk-calibration metric implemented in `t0_score.py` + **anchored on dense-mse pw2 preds** (baseline
      `ratio_med` recorded) — **blocker**
- [ ] winsorize helper + τ-pinball loss unit-tested (minimiser, finite grad, NaN/Inf guard) — **blocker**
- [ ] loss registered via `LOSS_REG_REGISTRY` (OCP), behind default-off flag; baseline byte-identical off
- [ ] pre-analysis plan (`05`) pre-registered: metric + falsifiers + thresholds LOCKED
- [ ] same-seed A/B driver + ≥3-seed plan; validation-graduation step defined
- [ ] gate frozen (Brier unchanged) + tail parked — one variable
- [ ] new failure modes noted for the risk register
