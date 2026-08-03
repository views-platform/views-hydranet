# Distributional Output Head — Research & Experimentation Dossier

**Opened:** 2026-06-05 · **Status:** **ESCALATION-ONLY (2026-06-09)** — the live direction is *hurdle + rollout training* (see `../2026-06-08_magnitude_calibration_dossier/`); this count-likelihood head is pursued **only if** that underdelivers · **Owner:** Simon Polichinel von der Maase / Claude
**Branch:** `fix/variational-dropout-autoregressive-stability` (work area; decision graduates to an ADR)
**Tracked:** yes (`git add -f` — this dossier is version-controlled despite living under gitignored `reports/`)

---

## 0. ⏭ 2026-06-09 — this count-likelihood head is now ESCALATION-ONLY (live direction = hurdle + rollout training)

After Arm-1's **step-1 read** (the lognormal hurdle **un-collapsed magnitude**; the failure was the **untrained rollout**, not the head — `../2026-06-08_magnitude_calibration_dossier/07`, risk C-136), the live next step is the **simplest grounded path: the existing hurdle + rollout training** (a cheap scheduled-sampling-middle probe → GTF). **A count-likelihood distributional head (this dossier) is now the ESCALATION — pursued only if hurdle + rollout underdelivers — NOT the immediate next build.**
- **If escalated, the head is a hurdle-NB:** the existing classifier *as the gate* `P(y>0)` × a **zero-truncated NB** positive part (softplus μ, count-space; closed-form, **no Tweedie-density blocker**). A plain **ZINB-mixture would mis-specify π** (the classifier learns *marginal* `P(y>0)`, not a structural zero-inflation gate — C-137); the proven count head on our data (DynAttn) uses a *dedicated* π. Tweedie/ZITD (§0–§11) is the tail-escalation beyond that.
- **Escalation triggers:** hurdle + rollout explodes-or-collapses with no stable-*nonzero* regime, OR un-collapses but leaves a quantified tail/calibration gap.
- **Design lives in `02 §0.0`; pre-registration in `05 §0`** — both retained, but as the *escalation* design, not the next action.
- **Lineage (honesty note):** re-scoped Tweedie-first → ZINB-first → hurdle-NB-first → (now) escalation-only as evidence accrued. The convergence: don't build a new count head until the cheaper hurdle+rollout path (using machinery we already have) is shown insufficient.

## 1. Purpose (one paragraph)

Replace HydraNet's current output side — a `log1p`-space point prediction inverted by `expm1`, with a separate BCE classifier — with a **single distributional / count-likelihood head**: the network emits the *parameters* of a distribution over each cell's conflict intensity (a **Zero-Inflated Tweedie**, `π, μ, φ, ρ`, is the lead candidate), trained by negative log-likelihood. Inputs stay `log1p` (good conditioning, no cached stats); the output is **never `expm1`'d into a point** — the forecast is the distribution's mean/quantiles, and the autoregressive feedback re-encodes `log1p(mean or sample)`. With a **sub-exponential link (softplus)** instead of the default log link, a drift becomes *linear* in counts, not exponential — retiring the `expm1` catastrophe at the source while giving calibrated magnitude (attacks chronic MCR ≪ 1) and principled aleatoric uncertainty.

## 2. How this relates to prior work (read before starting)

> **Gated behind the magnitude-calibration parent dossier** (`../2026-06-08_magnitude_calibration_dossier/`).
> ZITD is the *escalation* of that program — the heavy distributional fix — and runs only on a
> pre-registered trigger if the minimal hurdle there falls short. That program reuses this dossier's
> literature/harness by pointer (it does not duplicate them).

- **This is Path B, revived and sharpened.** `reports/path_b_zero_inflated_tweedie.md` (2026-05-27) already proposed the Zero-Inflated Tweedie head (grounded in Jiang 2023 STTD and Gao/Zhu 2024 STZITD-GNN, both on ~95% zero-inflated spatiotemporal data). That document is **absorbed** into this dossier (→ `02_design`, `01_literature`); the original will be marked superseded with a pointer here.
- **Path A shipped; this is complementary, not contradictory.** `reports/paths_forward.md` (2026-05-29) chose **Path A (Tobit censored regression)** as the first step → implemented as **ADR-054**. The distributional head is the next, deeper move on the same axis (catalogue "Axis C"). Its **design-constraints/invariants table** (fail-loud/ADR-003, multi-task heads/ADR-020, 36-step autoregression, log1p/expm1, ReLU output, curriculum sampling, MC-dropout eval, pandas-free/ADR-047) is inherited verbatim into `03_harness_and_invariants`.
- **Informed by this session's findings (2026-06-04/05):**
  - `results_freezeh_ablation.md` — divergence rides the prediction→input loop, not the recurrent state.
  - `results_io_gain_diagnostic.md` — violet's free-running map settles at an out-of-range attractor (log ~40 → expm1 ~1e17); the **retrain-free `scripts/diagnose_io_gain.py`** is our fast readout.
  - `results_feedback_clamp.md` — clamping the feedback is a safety rail, not a fix (ramps-to-ceiling); confirms the real lever is the output representation.
  - `project_explosion_is_regression` (memory) + the **C-111 balancer bisect** (in flight) — the acute explosion is likely a recent regression; this dossier targets the **chronic** magnitude/uncertainty bias regardless of the bisect outcome.

## 3. Document index

| # | File | Role | Status |
|---|------|------|--------|
| 00 | `00_README.md` | this index / status / navigation | **seeded** |
| 01 | `01_literature.md` | annotated bibliography (library papers + notes) + gaps to fetch | **seeded** |
| 02 | `02_design.md` | architecture: ZITD (π,μ,φ,ρ) head — **absorbs Path B verbatim** + §0 advances (expm1-runaway fix, autoregressive feedback, MC-dropout coexistence, Tweedie-density blocker) → graduates to ADR | **seeded** |
| 03 | `03_harness_and_invariants.md` | guardrails: hard vs intentionally-changed invariants, standing harness, new harness (Tweedie loss tests + sampling + calibration), pre-flight checklist | **seeded** |
| 04 | `04_roadmap.md` | phased gated build (P0 unblock → P1 loss → P2 head/flag → P3 MVP → P4 ablate → P5 decide), dep-graph, decision points, milestones | **seeded** |
| 05 | `05_analysis_plan.md` | pre-registered P3 MVP experiment (violet, fixed ρ, mean rollout): dual hypothesis, falsifiers (incl. **F5 zero-rate trap**), metrics, controls | **seeded** |
| 06 | `06_glossary.md` | grouped vocabulary — Tweedie params, links, aleatoric/epistemic, metrics, zero-inflation, rollout, program shorthand | **seeded** |
| 07 | `07_experiment_log.md` | append-only ledger (entry format + legend); precursors + EXP-01 (MVP, planned) | **seeded (skeleton)** |

Risks fold into `reports/technical_risk_register.md`. The **decision** graduates to `docs/ADRs/proposed/` when committed (working research lives here; governance lives in `docs/`).

## 4. The experimentation harness — at a glance (detail in `03`)

Much of this already exists (built/used during this session); the dossier codifies and completes it.

- **Default-off feature flags** — every new head/loss/link behind a config flag defaulting to current behavior (pattern: `feedback_clamp_log1p`, `freeze_multitask_balancer`). The stable `log1p`+`expm1` path is never disturbed.
- **Parity / regression gates** — full test suite green; `_df`/`_pf` parity held; CIC contracts synced; TDD with ADR-005 taxonomy before any run.
- **Fast cheap readout** — `scripts/diagnose_io_gain.py` (retrain-free attractor probe) filters before expensive evals.
- **Pre-registration** — a pre-analysis plan (hypotheses + falsifiers) before each run; negative results recorded honestly.
- **Reproducibility & run discipline** — `ReproducibilityGate` seed lock; conda `views-hydranet-env`; one model on GPU at a time; n ≤ 64; background + notify; artifacts preserved/timestamped.
- **Evaluation comparability** — fixed metric protocol (CRPS is *proper* for distributions; MCR; coverage/calibration; per-step trajectory) vs the locked `log1p` baseline (`reports/s0_baseline_metrics.md`).

## 5. Current state & next actions

- [x] Dossier created, scope fixed (distributional head / Path B revival), tracked in git.
- [x] `01_literature` — seeded (24 library papers annotated, grouped by role; gaps list incl. **Gao/Zhu 2024 STZITD verify/fetch** and **Dunn & Smyth Tweedie-density — implementation-critical**).
- [x] `02_design` — Path B migrated verbatim (§1–§11) + §0 advances (stability case, autoregressive feedback, MC-dropout coexistence, Tweedie-density blocker, open choices).
- [x] `03_harness_and_invariants` — invariants split hard vs intentionally-changed; standing harness catalogued; new harness scoped (**Tweedie NLL + tests is the blocker**); pre-flight checklist.
- [x] `04_roadmap` — phased gated plan (P0–P5), dependency graph, decision points, milestones M1–M5.
- [x] `05_analysis_plan` — P3 MVP pre-registered (dual stability+magnitude hypothesis; F1–F5 incl. zero-rate trap; metrics; pink/baseline/Tobit controls).
- [x] `06_glossary` + `07_experiment_log` (skeleton) — **scaffold complete (00–07)**, all git-tracked.
- [x] `reports/path_b_zero_inflated_tweedie.md` superseded → pointer to `02_design`.
- [ ] (gated) fold the C-111 bisect outcome into context once it lands.

## 6. Conventions

Numbered, dated where a doc is a point-in-time artifact; the README is living. Mirrors the repo's dossier convention (`reports/archived/2026-02-25_hydranet_hardening_dossier/`). Archive the whole directory under `reports/archived/` when the program closes (ADR adopted or abandoned).
