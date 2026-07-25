# T>0 Rollout Skill dossier — the bloom epic

**Opened:** 2026-07-25 · **Status:** SCAFFOLDED (harness audited; design drafted; not yet pre-registered).
**Container for:** the bloom (C-113) — promoted from a parked sub-thread to a first-class program.

## 1. Purpose

Build **ONE frozen, programmatic ruler that scores the 36-step autoregressive rollout against the realized
future** — so "fix the bloom" can mean *measurably more accurate*, not merely *numerically bounded*. The
existing bloom work found a knob (τ≥0.8) that keeps the rollout finite, but **we have no way to tell whether
a bounded rollout is a good forecast** (STABILITY ≠ SKILL). Without this ruler, every rollout fix
(sample-feedback, τ, GTF) is judged on boundedness — exactly the low-fidelity-probe trap that just produced
corrupted knowledge on gated_ZINBcore. This ruler is the precondition for the whole bloom epic.

## 2. Relationship to prior work / ADRs

- **Extends the frozen lodestar ruler** (`reports/2026-07-17_lodestar_eval_dossier`) from T=0 to T=1..36 —
  same scoring functions (`crps_ensemble`/`average_precision`/`brier`), same identical-support discipline,
  now indexed by horizon. **h=1 must reproduce the lodestar T=0 number** (faithfulness guardrail).
- **Absorbs** the two bloom docs from the distributional-head dossier as prior art (both marked superseded
  with a pointer here): `bloom_investigation.md` (the STABILITY≠SKILL confession + the τ measurements) and
  `plan_bloom_fix_sparse_feedback.md` (the escalation ladder + the H-SAMPLE sample-feedback probe).
- **Cross-refs:** C-113 (the bloom / AR feedback); ADR-058 (rollout training); ADR-056 (scheduled
  sampling); ADR-069 (the composition axis — τ / gated arms are rollout-feedback variants).
- **Exit ramp:** a validated ruler graduates to a proposed ADR ("T>0 rollout skill evaluation"); it is the
  measurement ADR that the sample-feedback / GTF fix ADRs will cite.

## 3. Document index

| # | File | Status |
|---|------|--------|
| 00 | `README` | living (this file) |
| 01 | `01_literature` | scaffold — exposure-bias / scheduled-sampling / rollout-eval anchors + gaps-to-fetch |
| 02 | `02_design` | **DRAFT** — the ruler: per-horizon re-score + free-running-vs-oracle decomposition + baselines |
| 03 | `03_harness_and_invariants` | **DONE (audit)** — ~70% already exists; gaps G1–G4 named |
| 04 | `04_roadmap` | **DRAFT** — phased build; the free-running curve is GPU-free (phase 1) |
| 05 | `05_analysis_plan` | **PRE-REGISTERED (EXP-1)** — current-behavior rollout-skill curve, GPU-free; P1–P5 + F1–F5 locked |
| 06 | `06_glossary` | scaffold — new terms (horizon-h, free-running, teacher-forced-oracle, skill-crossover, bloom-cost gap) |
| 07 | `07_experiment_log` | empty (append-only) |

## 4. Harness at a glance (see `03`)

**Already exists (~70%):** the frozen scoring core (`crps_ensemble`/AP/Brier), the determinism gate
(S2 #121), the config trap-restore pattern, the white_ranger climatology baseline dir, and — the decisive
find — **all 36 horizons of the free-running rollout are ALREADY PERSISTED on disk** (the lodestar loader
throws away h>1). **To build (gaps):** G1 per-horizon loader (pure Python; h=1 reproduces lodestar), G2
teacher-forced-oracle rollout (one feedback-source flag + small re-run), G3 per-horizon baselines
(climatology reuse + persistence construction), G4 identical-support origin set (origins with 36 future
months). **The free-running skill curve costs ZERO GPU** — it is a re-score of existing data.

## 5. Current state & next actions

- [x] Dossier scaffolded; harness audited; design drafted.
- [x] **expert-method-review done** (`02b`): 6-seat panel, DQ1–DQ4 resolved, chair rulings §6b (twCRPS/PIT
      OUT — FAO-02; direct-h parked, not a baseline), Salinas reframe (mean-feedback ≠ deployed skill),
      7 register risks C-a..C-g. Design revised (`02` §2/§6/§7).
- [x] Registered the 7 risks (C-217..C-223).
- [x] **BLOCKER CLEARED (C-217):** partition verified — calibration train (121,456) / test (457,504); all
      13 origins roll 36 steps inside the held-out 457–504 window. No leakage. (Also fixed H≈335 ⇒ the
      recursive-vs-direct cost is ~10%, not 36×; C-223 corrected.)
- [x] **EXP-1 pre-registered** (`05`): P1–P5 + F1–F5; metrics = crps_all/events/none split + Brier/MCR/QS99
      (NO twCRPS/PIT); CIs = block bootstrap over origins. *Awaiting user sign-off before build.*
- [ ] **NEXT: build G1** (loader, TDD, h=1==lodestar byte-exact = F1) → run EXP-1 (GPU-free current-behavior
      diagnostic + climatology/mixture/persistence baselines).
- [ ] EXP-2 ancestral (sample-feedback) = the **deployed skill verdict**; EXP-3 oracle = bug-vs-ceiling gap.

## 6. Conventions

Numbered dated docs; `00` living; git-tracked via `git add -f` (reports/ is gitignored); archived on close.
Vocabulary follows the LOCKED `reports/GLOSSARY.md` — new rollout terms are added there, never invented ad
hoc. STABILITY≠SKILL and the feature/world-model ceiling are first-class framing, restated in `02`/`05`.
