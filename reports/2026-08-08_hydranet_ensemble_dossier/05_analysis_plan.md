# 05 — Pre-analysis plan 🔒 LOCKED (HydraNet 8-member Africa `concat` ensemble)

**LOCKED 2026-08-08, before any member run.** The two `needs-decision` items are resolved (below); predictions +
falsifiers here precede S5/S6. Roster = **3 gated_NB / 2 th_gated_NB / 3 mixture_NB**; **`S` = 16 (4×4)**.

> **AMENDMENT 2026-08-09 (deliberate, pre-run): `total_lessons` 300 → 160.** The local africa run is bounded by
> a **25–26h uninterrupted compute window**; the full 8×300 run is ~30 GPU-hours (does not fit; would leave
> 1–2 members unfinished → an unscoreable ensemble). 8×160 ≈ ~18h fits with margin. Valid because this is the
> **local dress-rehearsal** and the readout is **relative** (ensemble vs its own members vs `light_strider`,
> **all at 160 lessons**); it is NOT number-for-number comparable to the v2 **300**-lesson scoreboard (not the
> goal tonight). The production **global-on-server** run restores the full lesson budget. F1–F4 unchanged.

## Hypothesis
Pooling 8 independent HydraNet posteriors (the v2 `gated_NB` family, seed- and composition-diverse) into an
equal-weight `concat` ensemble on `africa_me_legacy` **reduces variance and improves occurrence calibration**
vs the best single member, and provides a **short/long-horizon hedge** (gated_NB short-horizon occurrence +
th_gated_NB long-horizon AP) and a **small structural hedge** (mixture_NB) — **without** cracking the ξ=0
magnitude ceiling.

## Intervention (the ONE variable)
**Ensembling** (8 members `concat`-pooled) vs **each single member**. Everything else — region, foundation
config, truth, ruler, `S` — held fixed. Members differ only on the pre-registered `(family, composition, seed)`.

## Roster — 🔒 LOCKED (decision 2026-08-08)
| Family | composition | output_distribution | seeds | count |
|---|---|---|---|---|
| gated_NB | soft_gate | nb | 42, 43, 44 | 3 |
| th_gated_NB | threshold_gate τ=0.5 | nb | 45, 46 | 2 |
| mixture_NB | soft_gate | mixture_nb | 42, 43, 44 | 3 |

= **8 members**. **ZINB EXCLUDED** (blooms in the free-running rollout — v2 scoreboard F1). **Decision: keep the
full 3/2/3 mix** — maximum family/composition diversity; the 3 mixture members are the structural hedge and the
#241 heavy-tail upgrade slot (their magnitude-NULL is accepted; the ξ=0 scope is honest).

## Fixed design
- **Region** `africa_me_legacy` (13,110 cells); **combiner** `concat` (equal-weight posterior sample-pool via
  `PredictionFrameEnsembleManager`; pooled draws = 8×`S`).
- **Foundation** (all members): `output_distribution` per roster · `forecast_composition` per roster ·
  `body_supervision=all` · `loss_reg=mse` · `loss_class_pos_weight=2` · `reg_activation=softplus` ·
  `bn_recalibrate=True` · `rollout_feedback=sample` · `total_lessons=300` · log1p on the 3 lr_ targets.
- **`S` = 16 (n_posterior_samples=4 × n_head_samples=4)** — 🔒 LOCKED. **8×16 = 128 pooled draws**, within the
  RTX 4070 (rusty_bucket OOM'd ~28.6 GB at 8×128 *raw*; the thinned 16 is the memory-safe target — 7×16=112
  pooled fine in the smoke). F3 re-checks memory at S4 before the 300-lesson run.
- **Truth** frozen v2 parquet; **ruler** `score_v2_horizons` (`crps_all`/`AP`/`crps_none` @ h=1/18/36 per target).

## Metrics (pre-committed)
- **Primary:** `gw_stratified.score_gw_v2` Giacomini–White differential — **ensemble vs best single member** and
  **ensemble vs `light_strider` climatology** — on the ex-ante high-risk stratum, per target, h=1/18/36.
- **Supporting:** v2 ruler `crps_all` (occurrence+magnitude), `AP` (occurrence), `crps_none` (bloom/off-support),
  per target × horizon; 3-seed spread as the uncertainty band.
- **Diagnostic only:** size_ratio (Goodhart guard — never select on it). NEVER select on `crps_events`
  (Forecaster's Dilemma).

## Predictions (committed)
- **P1** — ensemble `crps_all` ≤ best single member at **h=1** (variance reduction shows first at short horizon).
- **P2** — ensemble beats `light_strider` on **occurrence** (`crps_all`, `AP`) at short horizons; climatology
  overtakes by ~h18 (magnitude regime — expected, not a failure).
- **P3** — **no magnitude miracle:** size_ratio stays capped; ensemble does not beat climatology past ~h18.
- **P4** — bloom stays suppressed (`crps_none` does not degrade catastrophically per-horizon) — the gated arms +
  sample-feedback hold; ZINB's absence keeps F2 quiet.

## Falsifiers (pre-committed)
- **F1** — ensemble ≤ best single member on `crps_all` (GW not significant, or negative) at h=1 ⇒ the ensemble
  adds nothing over one model ⇒ **the pooling premise fails** (ship the single best member instead).
- **F2** — **bloom re-armed:** `crps_none` degrades materially per-horizon in the free-running rollout ⇒ a member
  (likely mixture or a mis-composed arm) blooms ⇒ **do NOT ship**; quarantine the offender.
- **F3** — **OOM** at 8×`S` on the target hardware ⇒ `S` too large ⇒ re-pick `S` (thin), do not ship at that S.
- **F4** — **member inconsistency:** any member differs on partition / queryset / scale / sample-count from the
  others ⇒ `concat` pools incomparable draws ⇒ **invalid**; fix the member, re-pool.

## Method
S1 foundation → S2 migrate+Tier-A → S3 roster configs → S4 wire ensemble + reconcile sample-count contract →
S5 run 8 members × 300 lessons + pool (setsid, trap-restore, disk-safe, manifest+sentinel — the proven smoke
harness scaled up) → S6 score (GW + ruler) vs members + climatology → verdict vs F1–F4 → S7 disposition.

## Decision rules
- **P1 holds + F1–F4 quiet** → ship the ensemble; `promote` → proposed ADR (ensemble as the delivery unit).
- **F1 fires** → ship the best single member; the ensemble is not worth its cost.
- **F2 fires** → do not ship; quarantine the blooming member, re-pool the rest.
- **F3 fires** → re-pick `S`, re-run S5.
- **F4 fires** → fix the inconsistent member (BLOCK), re-pool.
- **Honest close:** whatever ships, record the ξ=0 magnitude ceiling explicitly; the mixture members are the
  heavy-tail head's (#241) future upgrade slot.
