# 05 — Analysis Plan (pre-registration)

**Date:** 2026-06-11 · Pre-registered **before** the coordinate train (roadmap box 3). Append-only.

> **⚙️ Operating-point update (2026-06-16) — supersedes the seed/sample lines below.** (GitHub #110.)
> - **`n_posterior_samples = 8`** (was 16) — interim workaround for the eval-stage OOM (C-116 / #124); restore
>   to 16 once C-116 is fixed. **First acceptance gate:** an 8-sample no-coords run completes **without OOM**.
> - **`WANDB_MODE=offline`** — avoid the `run.finish()` network-hang (#126).
> - **1 seed (42), not ≥2.** The multi-seed requirement (and the **F-volatility** falsifier below) existed to
>   fight run-to-run non-determinism — which **C-119 fixed** (validated *fixed-fixed* 2026-06-15, #119). Single
>   runs are now bit-reproducible, so a 1-seed A/B (baseline vs +coords, same seed/samples) is a clean
>   controlled comparison. Re-add a 2nd seed only if the 8-sample signal is ambiguous.
> - Comparator baseline = the **8-sample no-coords** run at this operating point (not the superseded 6-run sweep).
> - **Verdict rule (#129, corrected).** Coords **WIN** iff the coords-on FULL-MCR **95% bootstrap CI**
>   (`scripts/mcr_readout.py::_bootstrap_mcr_ci`) is **non-overlapping and lower** than the no-coords baseline's
>   CI on **≥2/3** targets **AND** CRPS is non-inferior; overlapping CIs → **inconclusive → escalate**. The noise
>   band is the **within-run** bootstrap CI from **one** run — NOT a run-to-run difference (which is ≡0 under the
>   determinism fix; that was the C-162 defect). The **F-volatility** falsifier below is reinterpreted accordingly:
>   judged by the within-run CI width, not multi-seed spread.

## One variable
The bounded hurdle-NB **S1** config (θ=1.0, `pos_weight`=10, frozen balancer, scheduled sampling off,
40 lessons) **+ coordinate channels** (ADR-061) **vs** the same config without them (the 6-run baseline of
record, `07` "before"). Nothing else changes.

> **Comparator prerequisites (falsify P5/C-155 + C-151).** Before this comparison is valid: pin the
> baseline to `config_hyperparameters.py` (hurdle_nb) + its recorded per-arm env + seed + the C-42
> reproducibility lock; **quarantine the stale `config_sweep.py` (tobit)** so coords are not benchmarked
> against a Tobit baseline; confirm `feedback_clamp` was **off** in the baseline (C-151) so "bounded" is
> intrinsic; and demonstrate I5 (toggle-off **bit-identical**) against a re-run, not just the recorded row.

> **Baseline provenance — PINNED (#107, 2026-06-13).**
> - **Config:** `views-models/models/violet_visitor/configs/config_hyperparameters.py` (`output_distribution`
>   = `loss_reg` = `hurdle_nb`, `loss_class=weighted_bce`, frozen balancer, SS off). The stale
>   `config_sweep.py` (tobit/focal) was **aligned to hurdle_nb** (C-155) so it is no longer a footgun.
> - **Per-arm env:** `HN_SEED∈{42,4}`, `HN_THETA_INIT∈{1.0,0.3}`, `HN_POS_WEIGHT∈{10,25}`, `HN_LESSONS=40`.
>   The **S1 comparator for the coords run = θ=1.0, pw=10, seeds {42,4}** (the two S1 runs).
> - **Reproducibility:** seeds locked via `ReproducibilityGate.lock_entropy(np_seed, torch_seed)` (C-42);
>   fresh process per run (`run.sh` → `main.py`), env read at startup.
> - **Clamp (C-151):** `feedback_clamp_log1p = None` in **all 11** baseline wandb runs ⇒ no clamping
>   occurred ⇒ the observed bound is **intrinsic, not clamp-masked**. C-151 resolved.
> - **Disk (C-154):** the coords run sets `min_free_disk_gb` so the manager aborts before writes if the
>   volume is short (the guard added in #107).

## Hypothesis
HydraNet's spatial over-firing is a symptom of **position-blindness**: a translation-invariant CNN cannot
represent that most cells are structural zeros. Injecting absolute coordinates lets the model learn a
**spatial base-rate prior**, so (a) the onset gate stops flooding structural-zero regions and (b) the
rollout stops blooming blobs there — moving full-horizon MCR toward 1 **without** any loss change.

## Pre-registered prediction (if coordinates are the lever)
- **Gate forensic:** the "Detection Bias Pulse" event-ratio **stops climbing** to 4–16× and flattens toward ≈1.
- **Rollout biopsy:** blobs **stop blooming in structural-zero regions** specifically (not merely shrink uniformly).
- **MCR:** **FULL MCR moves toward 1** (step-1 was already ~0.4–0.7; the drift is the target).

## Falsifiers (pre-committed)
- **F-persist:** blobs persist or merely **relocate**, and/or the gate still floods (event-ratio still
  climbs) ⇒ coordinates are **not** the lever ⇒ escalate to **static covariates** (ADR-060 enables) — **not**
  a return to loss-level tinkering, and **not** re-tuning coords in place.
- **F-smooth-proxy:** any gain is weak/ambiguous and concentrated where geography is smooth, failing on
  sharp settlement structure ⇒ the smooth-coordinate proxy is too weak (Tancik 2020) ⇒ escalate to
  Fourier features or covariates. *Distinguish from F-persist by where the residual blobs sit.*
- **F-volatility:** run-to-run variance across **≥2 seeds** is large ⇒ volatility disqualifies (the
  shrinkage lesson), regardless of mean.
- **F-offpath:** the toggle-off run is **not** bit-identical to baseline (ADR-060 I5) ⇒ the seam is wrong;
  the comparison is invalid ⇒ stop and fix the seam.

## Metrics (FAO PRN-05 — logged to `../RESULTS_LOG.md`)
- **CRPS** = primary ranking (superiority = ≥5% better than baseline).
- **QS99 / Brier / MCR** = guardrails (non-inferiority; MCR closest-to-1, **diagnostic only**).
- Per target (sb/ns/os), full grid **and** positive-cell subset.
- **Multi-seed** (≥2; ≥3 if promising).
- The three diagnostic instruments (`03`): gate forensic, rollout biopsy, MCR readout — read **side-by-side**
  against the baseline plots, since the diagnosis was visual.

## Decision bar
Prediction holds (gate flattens **and** blobs vacate structural-zero regions **and** FULL MCR toward 1),
CRPS non-inferior, multi-seed stable → **ship** (ADR-061 → Accepted). Any falsifier fires → log, escalate
per `04` box 4 — do **not** re-tune in place.
