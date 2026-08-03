# Tail-decoupled mixture-density NB head — dossier (2026-08-01)

## 1. Purpose
Settle the last open **pre-ship science question**: is the amount-ceiling *magnitude wall*
**within the NB exponential family** (its mean-tied light tail) **or real** (the data genuinely can't
say how big)? We test this with the one structurally-different magnitude lever left — a per-cell
**2-component mixture-density negative binomial** head that gives the positives a **mean-decoupled**
tail. Outcome is **binary and ship-relevant**: it either re-opens the magnitude axis or closes it for
good (ship `gated_NB` with the ceiling *proven*, not assumed).

## 2. Relationship to prior work / ADRs
- **Direct parent:** `reports/2026-07-29_v2_scoreboard_dossier/08_method_review.md` **Experiment 2a**
  (the panel's "decisive test of whether the wall is within-family or real"). This dossier executes it.
- **Builds on:** ADR-067 distribution-family subsystem (the strangler-fig slot the new head plugs
  into), ADR-069 composition (`soft_gate`), ADR-070 sample-feedback rollout, the frozen lodestar /
  v2 horizon ruler.
- **Supersedes-as-attempt:** `10_zinbcore_prereg.md` (th_gated_ZINBcore was a *partial* tail-decoupled
  *body* — but the ZINB core is still `NB(μ,θ)`, mean-tied; **DROPPED** 2026-07-31). The mixture is the
  first head to decouple the *tail shape* from the mean.
- **Will become:** a proposed ADR (`0xx_mixture_density_head`) **iff** the experiment lands positive;
  a clean negative closes the magnitude axis and updates the amount-ceiling record instead.

## 3. Document index
| # | Doc | Status |
|---|-----|--------|
| 00 | README (this spine) | **living** |
| 01 | literature | seeded (10 papers; 4 pending `/library rebuild` for search) |
| 02 | design | **drafted** — ready for `expert-method-review` |
| 03 | harness_and_invariants | **drafted** — pre-flight checklist RED (build gaps open) |
| 04 | roadmap | drafted |
| 05 | analysis_plan (pre-registration) | **🔒 LOCKED** (F1–F4; PRIMARY ≥5%/CI>0/≥2/3) |
| 06 | glossary | seeded |
| 07 | experiment_log | **EXP-01 (S4 GREEN) + EXP-02 (S6/S7 → NULL) logged** |

## STATUS: ⛳ CLOSED — **NULL** (2026-08-02). The 2-NB mixture does **not** crack the magnitude wall
(PRIMARY GW significant 3/3 but sub-5%, h=1-only; F2/F3/F4 all clean). The *meaningful* "active-but-insufficient"
null: component-2 is **genuinely engaged** (median `w|active` 0.71–0.92, μ2:μ1 14×–690×) yet magnitude skill stays
capped ⇒ **mean-decoupling is not the missing ingredient, tail SHAPE is.** Magnitude axis closed at the
within-family boundary; **heavy-tail (GPD/PIG) head = the sole sharpened next lever.** `gated_NB` ships; mixture
NOT promoted (no ADR); code stays in-tree uncommitted. Byproduct: **C-255** (forensic ZINB-layout mislabel).

## 4. Harness at a glance (→ `03`)
**~75% already exists** (the good-news finding): ADR-067 family subsystem (one-line registry OCP seam;
ZINB's `compose-NBCore` + `logaddexp` NLL is the exact template), the v2 horizon ruler + frozen truth,
`record_params` forensics, config default-off-flag pattern, determinism/parity discipline.
**To build (gate the first run):** (a) the mixture family — NLL numerics (logsumexp/2×NBCore) + sampler
+ ordered-means; (b) the **stratified-proper + Giacomini–White** scorer extension to the ruler;
(c) the single-tile overfit faithfulness gate; (d) the parity anchor proving baseline byte-identical
with the new family off.

## 5. Current state & next actions
- [x] dossier scaffolded + harness audited (2026-08-01)
- [x] **upfront correctness review** (expert-code-review, S2/S3) — 2 Tier-1 silent-nullifiers caught → C-248/249; specs amended (GH #232/#233); risks C-248..C-252 + D-15 registered
- [x] **S1 #231 — decisiveness method review DONE → VERDICT: PROCEED (decisive for a moderate effect).** Empirical sketch (frozen v2 truth): the recent-intensity stratum is **event-rich, not starved** — turns a 0.46%-event field into a 17–42%-event stratum (155–169× density lift, 42–78% of events captured), so Lerch's low-power warning (extreme-threshold twCRPS) does NOT bind. Risks C-253/C-254 registered.
  - **Three binding constraints inherited by S3 (#233) + S5 (#235):**
    1. **Stratum = recent realized intensity, last-12-month window** (ex-ante, model-independent, 78% event capture); report last-3 as robustness. Not gate-prob (model-coupled), not climatology alone.
    2. **Variance = origin-block bootstrap** (resample whole origin-months), NOT plain HAC — the panel's serial (36-h) + spatial dependence would inflate false significance (C-253).
    3. **Min detectable effect (pre-commit in S5):** ≥5% reduction in stratified mean CRPS on the last-12 high-risk stratum, origin-block-bootstrap 95% CI excluding 0 in ≥2/3 seeds. Sub-5% conceded unresolvable. **Report the eval origin count `P`** — a small-`P` null is suggestive-not-decisive (C-254).
- [x] **S2 #232 DONE** — `MixtureNBFamily` + 1 registry line; 22 mixture + 215 distributions tests green, ruff clean; C-249/250/251 guards enforced; diff-gate passed. Impl findings: C-249/D-15 reversed → clamp-and-log (the trap is *unclamped* log(w), not "not-log-sigmoid"; empirically verified); ordered-means non-strict in fp32 (`mu2 >= mu1`, coincident components = valid collapse). Files: `views_hydranet/distributions/mixture_negative_binomial.py`, `registry.py` (+1 line), `tests/distributions/test_mixture_negative_binomial.py`. **Not committed** (working-tree only).
- [x] **S3 #233 DONE** — new `reports/2026-07-29_v2_scoreboard_dossier/tools/gw_stratified.py` (`exante_stratum` + `gw_stratified_test` + `score_gw_v2`); ADDITIVE (separate module, `_metric_row` untouched ⇒ h=1 anchor preserved by construction). 8 GW + 2 scorer tests green, ruff clean. Enforces C-248 (leakage-free), C-252 (per-cell vectors), C-253 (origin-block bootstrap), C-254 (reports P). Tests: `tests/test_gw_stratified.py`. **Not committed** (working-tree only).
- [x] **S4 #234 DONE — GREEN** (STOP-gate cleared). Single-tile overfit (head isolated, direct gradient fit) on 3 real heavy sb cells: mixture uses a live component 2 (`w<0.995`, `μ2≫μ1`) + beats single-NB on all. Honest hint: gain modest, smallest on the heaviest cell (2-NB light tail vs ξ≈0.8) — the pre-registered scope. Driver: scratch `single_tile_overfit.py`; logged `07` EXP-01. Data note: `lr_sb_best` is RAW counts (max 113,395), not log1p.
- [x] **S5 #235 DONE** — `05_analysis_plan.md` 🔒 **LOCKED** (predictions + F1–F4 committed before any score): PRIMARY = ≥5% stratified-CRPS reduction, origin-block-bootstrap 95% CI excludes 0, ≥2/3 seeds; honest null-scoping (a null = within-family, heavy-tail untested).
- [x] **S6 #236 DONE** — mixture_nb vs nb × seeds 42/43/44 × 300 lessons, retrain-both on the v2 foundation (views-models, GPU). 6 trainings + 6 emits, views-frames pinned 1.8.0. Recovered a cross-repo `'targets'` emit block (pipeline-core PR #381; verified moot — we re-score raw cubes) and re-emitted nb-43 via `--artifact_name`. See `07` EXP-02 for the full scar log.
- [x] **S7 #237 DONE — VERDICT: NULL.** PRIMARY GW (sb h=1 ex-ante stratum): significant 3/3 (p≤0.003) but sub-5% (1.5–3.4%), h=1-only. F2 clean (size_ratio *down*), F3 clean (no bloom), **F4 clean — component-2 directly measured alive** on active cells (`scratch/w_probe.py`; median `w|act` 0.71–0.92, μ2:μ1 14×–690×; target-major layout + truth-parquet alignment verified). Fixed 2 latent scorer bugs (`gw_stratified.py` `parents[3]` + stratum `reset_index`). Registered **C-255** (the param-health forensic mislabels the 5-param mixture as ZINB μ/θ/π — nearly hid F4).
- [x] **S8 #238 DONE — disposition:** magnitude axis **closed at the within-family boundary** with the honest scope (a null here = light-tail flexibility insufficient, **NOT** "wall is real"; heavy-tail head untested). `gated_NB` = ship candidate; **mixture_nb NOT promoted** (no ADR). Amount-ceiling story sharpened: the gap is **tail shape**, not mean-decoupling. Mixture family code + scorer stay in-tree, **uncommitted**.
- **Next lever (if magnitude is ever reopened):** a genuinely heavy-tailed head (GPD / Poisson-inverse-Gaussian) — its own epic. This dossier does not pursue it.

## 6. Conventions
Numbered dated docs; `00` living, rest point-in-time. Locked glossary `reports/GLOSSARY.md` governs
shared terms (this dossier's `06` only adds *new* ones). Git-tracked via `git add -f` (reports/ is
gitignored) — **not committed to `development` without explicit user sign-off.** Scope-lock: ONLY the
ADR-067 NB/ZINB head lineage (the mixture is a new family in that lineage, **not** a legacy hurdle).
NO population (deferred to its own epic).
