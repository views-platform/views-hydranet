# ZINB Distributional Head — Dossier (the one direction)

**Date:** 2026-06-10 · **Status:** live · **Epic:** #97 · **Checklist (single source of truth):** #104

## Purpose
Replace HydraNet's separate regression + classification heads with **one hurdle negative-binomial
(hurdle-NB) likelihood per target**. This gives calibrated magnitude + uncertainty, makes the onset
classifier the principled hurdle gate, and **dissolves the regression-vs-classification balancer — the
C-111 instability source** (one joint NLL per target → nothing to weight across families).

> **Stability is NOT claimed "by construction."** The hurdle-NB removes the *balancer* problem; whether
> it tames the **autoregressive** explosion (C-113) is *unproven* (C-148) and tested empirically by the
> explosion-check (`02_design §6`), never assumed.

> **Likelihood: hurdle-NB, not ZINB** (gate-resolved 2026-06-10, C-146) — the reused classifier learns the
> *marginal* `P(y>0)`, which is the hurdle gate; a ZINB needs a *structural* π and would mis-specify.

## Why this, why now
After ~3 weeks of loss/rollout experiments that all ended in autoregressive explosion (see the
since-February catalog, `../2026-06-10_since_february_catalog.md`), the program committed to **one
direction**. All other directions are parked or superseded — recorded in **C-139** of the risk
register. This dossier is the clean home for the chosen design.

## Prior art (archived — do not re-open the design from these)
- `../archived/2026-06-05_distributional_head_dossier/` — the original ZINB/ZITD/Tweedie design work (superseded; the live design is lifted into `02_design` here).
- `../archived/2026-06-08_magnitude_calibration_dossier/` — the hurdle/gate program (superseded).
- `../archived/2026-06-05_rollout_training_dossier/` — rollout training (**parked fallback** if ZINB fails).

## Document index
- `01_literature` — annotated cites (ZINB on VIEWS/PRIO, hurdle theory) + pointers.
- `02_design` — **the locked design** (head / loss / count-target bridge / inference / flag / explosion-gate). Decided; not re-opened.
- `03_harness_and_invariants` — parity, count-target containment tests, the explosion-check gate, standing invariants.
- `04_roadmap` — the linear sub-issues (#98–103), the checklist (#104), the two exits.
- `05_analysis_plan` — pre-registration of the first experiment.
- `06_glossary` — shared vocabulary.
- `07_experiment_log` — append-only; seeded with the "before" row.

## The harness (5 binding rules)
1. **One at a time** — work only the current unchecked checklist box; do → log to `../RESULTS_LOG.md` → tick → advance.
2. **No new scope mid-epic** — no new issues / re-scope / new directions. Ideas → a Parking Lot, never acted on during.
3. **Findings log, they don't steer** — log every result; advance regardless. Only pre-listed objective HARD-STOPs (compile-fail; a pre-registered falsifier) pause for the chair.
4. **No detours / no "let's try."** The roadmap is the only path.
5. **Only two exits:** ship the ZINB head, **or** the fallback fires → **revert to commit `e029e63`** (months-old stable) and ship that.

Brake word: **CIRCLE**.

## Next action
Checklist #104, box 1: **sub-issue #98 — the raw-count target provider.** No code until the box is live and the chair says go.

## Conventions
- Results → `../RESULTS_LOG.md` (FAO PRN-05 metrics). Risks → `../technical_risk_register.md`.
- MCR is a **diagnostic ratio**, never the optimization target. CRPS is the primary proper score.
