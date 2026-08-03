# Open threads — PARKED (do not let these vanish)

**Purpose:** a durable list of balls we deliberately put in the air and are *consciously* deferring — so
walking toward the next thing (th_gated proper build → the three-model base → the bloom) never silently
drops one. Written 2026-07-24 after the composition-arm comparison. Each item is tagged with *when* we
intend to return to it. This is a holding list, **not** a decision — decisions happen when we regroup.

## Legend
- **[NOW]** — the active focus (not really parked; listed for context).
- **[BEFORE-3-MODELS]** — settle before we call the three T=0 arms "done / a base to work off".
- **[BEFORE-BLOOM]** — do after the three arms are real, before pivoting to T>0.
- **[BEFORE-ADR-PROMOTION]** — required before any arm graduates to a committed ADR / production, but not
  blocking exploratory T=0/T>0 work.
- **[LATER / OPTIONAL]** — real value but explicitly deferred; revisit when we choose.

---

## 1. th_gated_NB — proper integration  **[DONE 2026-07-24 — Epic #183]**
✅ Composition is now a real config axis (`forecast_composition`, ADR-069) applied inside the model at
emit time; the three arms are honest model outputs. Emit-time re-eval (S8, 3 seeds) confirmed the
implementation is faithful (th_gated reproduces) AND falsified "th_gated uniquely best": properly
composed, **gated_NB ≈ th_gated_NB** (the score-time re-score had undersold gated_NB by never applying
the per-draw gate). See `07_experiment_log` S8 PASS. Ensemble note (item 6) is now sharper: the two
sharp-gate arms are nearly identical, so the real diversity is **nb-vs-zinb**, not soft-vs-threshold.

## 2. ZINB π-ridge on/off decision  **[BEFORE-3-MODELS]**
Memory: "π-ridge decision open (lean OFF first)." ZINB cannot be a *settled* third arm until the
π-penalty config (`pi_penalty_weight`/`pi_penalty_prior_logit`, C-200) is chosen and justified. The
3×300 ZINB run — was π-ridge on or off? Confirm and lock before treating ZINB as a real ensemble arm.

## 3. M3 / validation-partition graduation (the F3 falsifier)  **[BEFORE-ADR-PROMOTION]**
ALL T=0 numbers are on the CALIBRATION partition. The pre-registered **F3** kill: a calibration win that
does not survive the VALIDATION partition is invalid. Cost (corrected 2026-07-24): needs a **viewser
fetch to month 552** (local data ends at 504) **+ a fresh 3×300 retrain under `run_type=validation`**
(validation trains on 121–504, a different window) + a validation-partition ruler. NOT a cheap eval.
Deliberately deferred — but must run once, on all arms together, before any ADR promotion. Do not forget.

## 4. Magnitude / heavy-tail body — the epic's ORIGINAL goal  **[LATER / OPTIONAL]**
None of the three arms improved **crps-events** (the actual sizing of conflicts); they are occurrence /
aggregate-CRPS plays. lognormal / gamma / bulk+GPD-tail (volatility dossier: tail ξ≈0.8) was the intended
magnitude path. Tension to hold: the [amount-ceiling WALL] finding says point-magnitude is largely
irreducible (spearman 0.303 < persistence 0.367), so chasing this may re-hit the wall — what IS
predictable is spread/volatility (S2 0.79), i.e. calibrated *uncertainty*, which the head already
delivers. So: park for now; revisit only if we decide the magnitude prize is worth the wall risk.

## 5. os is a persistent weak target  **[BEFORE-ADR-PROMOTION]**
os loses crps-all to white_ranger under ZINB AND th_gated (register **C-170**, os under-firing gate).
Whatever we ensemble, os drags. Decide: accept it, or treat os as its own problem, before promotion.

## 6. Ensemble design note (not a task yet)  **[BEFORE-BLOOM]**
gated_NB and th_gated_NB are the SAME trained nb model, composed two ways → low mutual diversity. The real
diversity is **nb-model vs zinb-model**. Factor this into any ensemble weighting; don't over-count
correlated arms.

## 7. The BLOOM (T>0 autoregressive runaway, C-113)  **[ACTIVE — first pass done 2026-07-25]**
First pass run (inference-only). **See `bloom_investigation.md` (detailed + honest epistemic table) and
`plan_bloom_fix_sparse_feedback.md` §NEXT.** State: **sparsity IS the lever** — τ≥0.8 threshold feedback
keeps the 36-step rollout bounded (3 seeds) and beats the foundation on T=0 crps-all; but this is a
**conservative-point tool, NOT the solution** (⚠️ stability ≠ skill — we don't score the rollout; ZINB
blooms too; several results single-seed; the clamp rail was inert). **NEXT = the pre-registered
sample-feedback (ancestral) rollout** — feed back a *draw* not the mean (decalibration-proof like τ, but
keeps the distribution). Deeper ceiling (hypothesis): the model has only conflict-history features, so
long-horizon *skill* may be capped regardless of rollout method (features/world-model gap).

## 8. Minor / low-priority
- **C-215** — reg/cls training loss not logged numerically post-hoc (observability, Tier 4). [LATER]
- Reconcile the composition config with **archived ADR-066** (`output_distribution` → `body_family` ×
  `zero_handling`) — the design we archived is the shape we're now reviving for item 1. [NOW, within item 1]
