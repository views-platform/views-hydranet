# 05 — Analysis plan (pre-registration) — 🔒 LOCKED 2026-08-01

**STATUS: LOCKED.** Predictions + falsifiers committed *before* any 3-seed score is read. Preconditions
met: S1 decisiveness review = PROCEED; harness green (S2 family + S3 GW scorer, all tests green); S4
single-tile smoke = GREEN. Only the GPU run (S6) and scoring (S7) remain. Do not amend below this line
except to record that execution has begun.

**Builds on:** `02_design` · S1 (`00_README §5`, risks C-253/254) · the amended issues #232/#233 ·
`08_method_review` Exp 2a · C-MR1–C-MR6 / C-248–C-254.

## 1. Hypothesis
**H:** the per-cell 2-component mixture-density NB head improves **conditional predictive accuracy on
the ex-ante high-risk stratum** over the single gated NB — **if and only if** the amount-ceiling wall
is *within the NB family* (its mean-tied light tail). A null is *scoped* (see §7).

## 2. Intervention (the ONE variable)
`output_distribution="mixture_nb"` (soft_gate composition; sample-feedback rollout, ADR-070). Every
other knob held at the v2 `gated_NB` baseline (300 lessons, same features/targets/gate/seeds/foundation).
Comparator = the v2 `gated_NB` (the single-NB body), scored on identical intersected support.

## 3. Skepticism ledger
1. **Low-power regime** (Lerch2017) — mitigated but not eliminated; S1 showed the stratum is event-rich (155–169× lift), yet GW power still binds on the **origin count P** (C-254).
2. **`w→1` collapse** — the decisive-negative signal; must not crash (C-249 clamp-and-log) and must be *observed*, not a build artefact (S4 confirmed it trains + uses component 2).
3. **Tail-component starvation** on sparse positives — watched via `record_params`.
4. **Bloom re-arm** (C-MR4) — a heavier fed-back tail could re-arm exposure bias; `crps_none` guardrail live.
5. **Mask-graveyard / Goodhart** — `size_ratio` is diagnostic-only, never a target.
6. **Modest-effect prior (from S4, honest):** the single-tile smoke beat NB only by 0.007–0.048 nll and *least* on the heaviest cell — a 2-NB light tail may not crack ξ≈0.8. The pre-committed effect size (§4) reflects this; a null here is *expected-plausible*, not a surprise to rescue.

## 4. Pre-registered predictions (primary first)
| Endpoint | Prediction | Pass / fail threshold |
|---|---|---|
| **PRIMARY — GW conditional test, high-risk stratum** | mixture lowers stratified CRPS | **≥5% reduction in stratified mean CRPS on the last-12 high-risk stratum, origin-block-bootstrap 95% CI excludes 0, in ≥2/3 seeds** |
| Secondary — per-decile stratified `crps_all` | improvement concentrated in high-risk deciles | monotone-ish gain across the recent-intensity decile curve |
| Diagnostic — `record_params` | live component 2 | `w<0.995`, `μ2≫μ1`, non-degenerate tail occupancy (not a target) |
| Diagnostic — `MCR`, `size_ratio` | — | reported, NOT decision inputs (Goodhart guard) |

**Fixed method constants (locked):** stratum = recent realized intensity, **last-12-month** window
(ex-ante, model-independent); robustness = last-3. Variance = **origin-block bootstrap** (resample whole
origins), NOT plain HAC. **Report `P`** (origin count). NEVER select on `crps_events` (improper).

## 5. Falsifiers (pre-committed — any one fires ⇒ hypothesis rejected, not rescued)
- **F1 (within-family gain absent):** GW not significant on the high-risk stratum in ≥2/3 seeds (or <5% effect) ⇒ **wall is real *within-family*** (see §7 scope).
- **F2 (works-but-degenerate):** `size_ratio` moves but stratified `crps_all` does not ⇒ inflation → reject (Goodhart).
- **F3 (rollout bloom):** `crps_none` per-horizon degrades vs the gated_NB baseline ⇒ tail re-armed exposure bias → reject.
- **F4 (identifiability/collapse):** `record_params` shows `w→1` / dead component 2 across seeds ⇒ collapsed to NB. (S4 already disambiguated "can't train it" — so at 300 lessons this reads as *no tail signal*, i.e. F1.)

## 6. Method
3 seeds (42/43/44) × 300 lessons on the **v2 datafactory foundation** (frozen truth `calibration_datafactory_df.parquet`, sha 620f4aa3…). Emit the D×K rollout; score with `gw_stratified.py::score_gw_v2` (last-12 stratum, origin-block bootstrap, n_boot=2000). Hardened driver in `views-models` (one-at-a-time, trap-restore, inline-score-delete-cube, `crps_none` bloom guardrail). **Cheap-before-expensive already honoured:** S4 smoke GREEN precedes this run. NEVER score on stale data.

## 7. Decision rules & ⚠️ honest null-scoping
- **Positive** (PRIMARY passes, no falsifier fires) → **magnitude re-opens**; `/rnd-dossier promote` → proposed ADR `0xx_mixture_density_head`.
- **Negative** (F1 fires, or F2/F3/F4) → close the magnitude axis **within-family** and update the amount-ceiling record + C-MR2.
- **⚠️ SCOPE (non-negotiable):** a null means *"mean-decoupled **within-family** (light-tail) flexibility does not crack the wall,"* **NOT** *"the wall is real."* The **heavy-tail head (GPD/PIG/discrete-Pareto) remains explicitly untested** — its status stays *open*, not disproven. A small-`P` null is **suggestive, not decisive** (C-254).
