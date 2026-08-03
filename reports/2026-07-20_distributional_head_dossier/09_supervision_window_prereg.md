# 09 — Pre-registration: the `body_supervision` window sweep (LOCKED before any run)

**Locked:** 2026-07-28 · **Scope:** the magnitude effort — does a graded supervision window around
conflict activity fix the **bulk-magnitude downward bias** (the timid body), where the hard endpoints
fail? · **Budget:** 3 seeds (serious, not a 1-seed screen). · **Governs:** ADR-065 amendment; C-224,
C-227, D-13; supersedes dossier `08`'s `pos_cells`/`pos_timelines` framing.

## Context / why this experiment
Dossier `08` (1-seed) + the corrected count-space read showed the body-supervision failure is a
**boundary problem**: `pos_cells` (=`active,0,0`) un-collapses magnitude but **over-cooks true-zero
cells** (the head is unsupervised off the positive support → drifts high, and never learns the
ramp-down to zero); `pos_timelines` (=`active,W,W`) is **timid** (supervises every dead-stretch zero
of active cells). The amendment (ADR-065, 2026-07-28) makes the supervised region a graded, asymmetric
**window**: `active` + `onset_lead` (months before onset) + `cessation_lag` (months after cessation).
This pre-registers the sweep that decides whether an **interior** window beats both endpoints.

## The ONE variable
`body_supervision` window radii `(onset_lead, cessation_lag)`, **NB head**, everything else held
(architecture, gate = weighted_bce pw=10, features, frozen-ruler months/cells/truth, BN-recal on).
`(0,0)` = per-step positives; `(≥W,≥W)` = active-cell timelines; `all` = every cell (foundation).

## Family scope
- **NB is the vehicle** (primary). The window supervises the point/NB body cleanly.
- **ZINB is a reference at `all` only, NOT swept over `active`.** `body_supervision=active` masks the
  whole per-cell NLL, which for ZINB starves the structural-π zero term on the excluded deep-zeros
  (π-degeneracy, the C-227/F1 family). ZINB's magnitude lever is π/θ, not the window. `ZINB × all`
  (3 seeds) is a loose reference for where the occurrence-native family sits.

## Radii grid (PROPOSED — lock before run)
Chosen to test the asymmetry hypothesis (decay is the body's job; pre-onset is more the gate's) and
to bracket both endpoints. **NB, 40 lessons, seeds {42, 43, 44}:**

| # | `(onset_lead, cessation_lag)` | role |
|---|---|---|
| 1 | `all` | timid foundation (control) |
| 2 | `(0, 0)` | per-step positives = old pos_cells (over-cook endpoint) |
| 3 | `(0, 2)` | **cessation-decay only** — teach the 2-mo ramp-down, ignore run-up (the lead hypothesis) |
| 4 | `(2, 2)` | symmetric small window |
| 5 | `(0, 6)` | long decay → toward pos_timelines (does more decay re-timidify?) |

= **5 NB settings × 3 seeds = 15 trainings** + `ZINB × all × 3` reference (or reuse banked). Emit-and-
score is a free re-score on top. (Grid is deliberately small/asymmetric-weighted; expand only if a
signal appears.)

## Compositions scored (emit-time re-score; not training variables)
`threshold_gate τ=0.5` (the deliverable, magnitude-preserving) + `soft_gate` (completeness). Both are
free re-scores of each trained model.

## Ruler (three instruments, run together — C-224 + C-227)
1. **Frozen lodestar** `../2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py` — crps-all/events/
   none, size_ratio, AP/Brier, QS99. **CRPS selects** (FAO-02).
2. **`scripts/tail_scorecard.py`** — reach%/cover90%/pin90 by truth-magnitude bin. **Diagnostic only**
   (the tail lens; never selection).
3. **Per-regime CRPS decomposition (C-227, folds in Option 5)** — report crps split by regime
   {stable-zero, active-stable, escalation, de-escalation} × {occurrence, magnitude} **before** any
   crps_all aggregation, so a stratum-trade cannot hide in the headline (as it did for the mask).

## Pre-registered predictions (committed before looking)
1. **An interior window beats both endpoints on composed crps-all** (`threshold_gate τ=0.5`): some
   `active,(a,b)` with `b>0` lifts bulk magnitude (MCR / size_ratio up on the truth `[3,100)` bins vs
   `all`) **without** the true-zero crps_none explosion that `(0,0)` shows.
2. **`cessation_lag>0` reduces the true-zero over-cook vs `(0,0)`** — the decay supervision anchors the
   ramp-down, so `E[y]>100` on true-zero cells (dossier 08: ~46 ns / 67 os) drops.
3. **Asymmetry:** `onset_lead>0` adds little over `onset_lead=0` at matched `cessation_lag` — the
   pre-onset quiet is largely the gate's job. (`(2,2)` ≈ `(0,2)` on composed crps-all.)
4. **The tail stays dead for EVERY radius** — `tail_scorecard` top-bin reach% ≈ 0 / q90 ≈ 0 across the
   grid. This is the scope guard: the window is a *bulk*-magnitude lever, orthogonal to the ξ≈0.8 tail
   (C-149/C-224). If the tail *does* move, something is wrong (investigate before trusting).
5. **No smearing:** FSS@1 / conc1% do not degrade vs the STEP-1 foundation baseline for the winning
   radius (C-167 anti-Goodhart guard).

## Falsifiers (pre-committed — any ⇒ the window is not the bulk-bias lever)
- **F-SUP-1 (no interior win):** no `active` setting beats `all` on composed crps_events (≥2/3 seeds)
  without degrading crps_none or FSS ⇒ the supervision region does not fix the downward bias ⇒ the
  lever is elsewhere (family/tail — the quantile-Δ head / GPD, D-13 / C-149).
- **F-SUP-2 (all radii ≈ pos_cells):** every `active` setting reproduces the `(0,0)` over-cook
  (crps_none explodes) regardless of `(a,b)` ⇒ boundary supervision does not anchor the drift ⇒ the
  fix is the gate/π, not the body window.
- **F-SEED (unstable winner):** the best radius rank-flips across the 3 seeds ⇒ the "win" is a basin
  artifact, not a real effect (the multi-seed discipline; the 1-seed-flips lesson).
- **F-DEGEN:** any training NaN.

## Decision rule (3-seed — advance/productionize, or park with a banked negative)
- **Win:** an interior radius, in ≥2/3 seeds, lifts bulk magnitude (MCR/size_ratio on the mid bins) +
  holds crps_none/FSS + beats **both** endpoints on composed crps_all ⇒ the window is a real
  bulk-magnitude lever ⇒ set that radius as the production default, add seeds + M3-validation graduation.
- **Park:** F-SUP-1 or F-SUP-2 fires ⇒ the body-supervision region is not the bulk-bias lever ⇒ the
  knob is built + productionized (not wasted — a clean, sweepable axis), the negative is banked, and
  the effort pivots to the family/tail axis (quantile-Δ head; C-224 governance amendment first).
- **The tail is out of scope for the decision** regardless of outcome (prediction 4).

## Skepticism ledger
- 3 seeds is better than 1 but still thin for a heavy-tailed field; a win is provisional pending M3.
- The banked dossier-08 configs used the retired keywords → all sweep configs are regenerated on the
  new `body_supervision` axis (the endpoints are byte-identical by construction, ADR-065 amend.).
- `cessation_lag` supervision uses future-relative truth to *select* training cells — a training-time
  choice, not an inference feature (ADR-065 §A5); the emitted forecast is causal.
- This is a *bulk*-bias experiment. Even a clean win leaves the surge tail (q90≈0 on 300+) untouched —
  do not narrate a tail fix from a bulk win.
