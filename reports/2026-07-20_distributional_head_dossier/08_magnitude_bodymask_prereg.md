# 08 — Pre-registration: body_mask magnitude sweep on the NB head (LOCKED before any run)

**Locked:** 2026-07-27 · **Scope:** the magnitude effort — can a body-loss cell mask lift the timid
body *now that composition is applied at emit time* (Epic #183)? · **Budget:** 1 seed (screen).

## Context / why re-open a settled negative
The old `body_mask` sweep (`[[project_body_mask_sweep_negative]]`) found `pos_cells` lifts size-ratio
×11–60 but **blows crps-all (ns 0.08→24.6)**. But that sweep was scored **before Epic #183**, when the
frozen ruler's crps-all was gate-independent and the emitted cube was the **raw (ungated) body** — so a
bigger `pos_cells` body landed on true-zero cells with nothing suppressing it. Post-#183 composition is
applied **in-model at emit** (ADR-069): the emitted cube IS the `gate × body` forecast, so the gate
zeros the body on true-zero cells *before* the ruler sees it. **The exact mechanism that blew crps-all
is removed.** Also: the old sweep was on the **legacy point head**; this is the **new NB distributional
head** (per-cell θ, sampled) — θ can carry spread the point head could not.

## The ONE variable
`body_mask ∈ {none, pos_cells, pos_timelines}` (ADR-065), **NB head**, seed 42, 40 lessons, BN-recal on.
Everything else held: architecture, features, gate (weighted_bce pw=10), `forecast_composition=soft_gate`
during training feedback, frozen lodestar months/cells/truth. `body_mask` masks the **body NLL** cells:
`none` = all cells (foundation, timid); `pos_cells` = per-step positives (`y>thr`) → no zero-pull on the
magnitude head; `pos_timelines` = full timeline of ever-active cells (adds zero-pull on the hard cells).

> **ZINB is deliberately EXCLUDED from the mask sweep.** `body_mask` enters ZINB's NLL as a weight on the
> whole per-cell term, so `pos_cells` (weight 0 on `y=0`) starves the structural-π zero term → π
> degenerates (trips F1). ZINB's magnitude lever is π/θ (+ heavy tail), not a body mask. `ZINB × none`
> (banked) is a loose reference only.

## Compositions scored (emit-time re-score of each trained model — NOT training variables)
All three are legal NB emit configs (NB cannot use `self_zeroed`; validator ADR-069):
1. **`threshold_gate τ=0.0`** — full body everywhere (gate ≥ 0 always) = the **ungated control** (the old
   raw-body scoring, reproduced within a legal config).
2. **`threshold_gate τ=0.5`** — full body where gate ≥ 0.5 = the **magnitude money arm** (unshrunk body
   on retained cells).
3. **`soft_gate`** — per-draw `Bernoulli(gate) × body` (matches how the banked baselines were emitted).

Grid: **3 body_mask × 3 compositions × 3 targets (sb/ns/os)** = 27 score cells from **3 NB trainings**.

## Ruler (C-167, run in tandem)
Frozen `../2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py` (crps-all/events/none, size-ratio,
AP/Brier, QS99) **+** `scripts/sharpness_scorecard.py` (FSS@1/conc1%/area_ratio), vs the STEP-1
foundation baseline (FSS@1 ≈ 0.00–0.01, area_ratio 0.1–0.2×, conc1% 0.49–0.58, MCR 0.004–0.015).

## Pre-registered predictions (committed before looking)
1. **`pos_cells` lifts size-ratio** on event cells vs `none`, under `threshold_gate τ=0.5` (magnitude
   moves toward 1).
2. **Ungated control reproduces the blowup:** under `threshold_gate τ=0.0`, `pos_cells` crps-all is far
   worse than `none` (confirms the old-negative mechanism).
3. **The gate saves it:** under `threshold_gate τ=0.5`, `pos_cells` composed crps-all is NOT blown the
   way the ungated (τ=0) is — the true-zero over-fire is suppressed.
4. **`pos_timelines` is MORE timid** than `none` (size-ratio ≤, crps-none ≤) — wrong-direction control.
5. **No smearing:** under `threshold_gate τ=0.5`, `pos_cells` FSS@1/conc1% do NOT degrade vs the STEP-1
   baseline.

## Falsifiers (pre-committed — C-167 magnitude rule)
- **F-MAG-1 (smearing, = F2b):** `pos_cells` lifts size-ratio BUT composed `crps_events` worsens AND
  FSS@1/conc1% degrade ⇒ the "magnitude gain" is a smear ⇒ the mask is not a real magnitude fix.
- **F-MAG-2 (gate can't save it):** `pos_cells` *gated* (τ=0.5) STILL blows composed crps-all like the
  ungated control ⇒ the over-fire is on gate-**retained** cells (real false magnitude, not true-zeros)
  ⇒ the mask alone is insufficient → robust-trendline / heavy tail needed.
- **F-DEGEN:** any training NaN, or θ collapses (std/mean < 0.02).

## Decision rule (1-seed SCREEN — advance/park only, NO hard kill)
- Predictions 1 + 3 + 5 hold ⇒ **magnitude life**; the old negative was the ungated-scoring artifact ⇒
  next lever = **robust-trendline target** on the masked body (fix small-positive over-fire).
- **F-MAG-2** fires ⇒ mask alone insufficient ⇒ go straight to robust-trendline or the heavy tail.
- **F-MAG-1** fires ⇒ the mask smears ⇒ drop it; reconsider heavy tail as the magnitude route.
- More seeds only *after* a 1-seed signal (BN-recal makes 1 seed defensible for a screen; a lone basin
  is still a caveat, so we do not KILL on it).

## Skepticism ledger
- 1 seed = basin risk (BN-recal mitigates C-184, does not eliminate).
- `pos_cells` may simply **move** the over-fire from true-zeros (gate-suppressed) to gate-**retained**
  false positives (F-MAG-2) — that is the honest failure mode to watch.
- The old negative was the legacy point head; a distributional θ may absorb some over-fire — or may not.
- Winsorizing "didn't save us," but that was a global point τ-dial, not a masked distributional head —
  not a clean refutation of the robust-trendline follow-up.
