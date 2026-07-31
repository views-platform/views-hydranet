# Coordinate Grounding — Dossier (the next structural lever)

**Date:** 2026-06-11 · **Status:** live · **ADRs:** 060 (contract) + 061 (instance)

## Purpose
Make HydraNet **coordinate-aware**: inject static, geometry-derived coordinate channels at the model
input and the final full-resolution layer, so the network can condition on **absolute position**. This
attacks the **spatial over-firing** diagnosed in the bounded hurdle-NB sweep — the onset gate flooding
structural-zero regions and the rollout blooming conflict "blobs" where ground truth never has them.

> **This builds ON the ZINB/hurdle-NB head — it does not re-open it.** The hurdle-NB bounded the C-113
> explosion (the hard part); this dossier takes the *next* lever. Per the loss-hacking stopping rule,
> coordinate grounding is a **new structural (architectural) insight** — *not* loss-level tinkering — so
> it clears the rule rather than violating it.

> **Spatial grounding is NOT claimed "by construction."** CoordConv has no published validation for
> autoregressive/rollout stability or for conflict forecasting (the literature is single-pass medical
> segmentation). Whether coordinates stop the blob-bloom is **unproven** and tested empirically by the
> pre-registered experiment (`05`), never assumed.

## Why this, why now
The 6-run bounded hurdle-NB sweep (overnight 2026-06-11) diagnosed a *bounded-but-drifting* rollout: the
gate over-fires 4–16× (worse with higher `pos_weight`, worsening through training), magnitude
over-predicts and rises, and the rollout blooms localized blobs in structural-zero regions. The reframe:
C-113 is no longer a divergence to ~1e17 — it is **spatial over-firing + exposure bias**. HydraNet is a
**translation-invariant** ConvLSTM U-Net predicting a **spatially near-degenerate** process; a CNN cannot
represent absolute position (Liu et al. 2018), so it cannot learn "this place is structurally peaceful."

## Document index
- `01_literature` — the four cites (CoordConv, CoordConv-Unet, Fourier features, coordinate attention) + the domain argument.
- `02_design` — **the locked design** (= ADR-061 §2: two injection points / derived-from-geometry / `[-1,1]` / static / toggle). Decided; not re-opened.
- `03_harness_and_invariants` — the ADR-060 I1–I5 invariants, bit-identity-when-off, align-by-construction, and the gate-forensic / rollout-biopsy / MCR readout protocol.
- `04_roadmap` — the epic + linear sub-issues + the two exits.
- `05_analysis_plan` — pre-registration of the one-variable experiment.
- `06_glossary` — shared vocabulary.
- `07_experiment_log` — append-only; seeded with the 6-run "before" baseline.

## The harness (5 binding rules)
1. **One at a time** — work only the current unchecked roadmap box; do → log to `../RESULTS_LOG.md` → tick → advance.
2. **No new scope mid-epic** — no new directions / re-scope. Ideas → a Parking Lot, never acted on during.
3. **Findings log, they don't steer** — log every result; advance regardless. Only pre-listed objective HARD-STOPs pause for the chair.
4. **No detours / no "let's try."** The roadmap is the only path. **Coordinates first; covariates only as the pre-registered escalation.**
5. **Only two exits:**
   - **Ship coordinates** — the experiment validates → ADR-061 → Accepted, coordinate channels stay on.
   - **Drop coordinates** — the experiment falsifies → the toggle defaults off (bit-identical baseline preserved by ADR-060 I5), and **escalate to static covariates** (the next instance of the ADR-060 seam). The hurdle-NB baseline is **never** reverted by this dossier.

Brake word: **CIRCLE**.

## Next action
**Opened on GitHub (2026-06-13):** epic **#105**, master-tracker checklist **#112** (+ stories #106–#111).
Work the checklist top-down — prerequisites **#106** (count-space explosion-check, C-142) and **#107**
(ops + reproducibility prep) clear *before* the experiment **#110**; the seam **#108** is box 1.
No code until the box is live and the chair says go.

## Conventions
- Results → `../RESULTS_LOG.md` (FAO PRN-05 metrics). Risks → `../technical_risk_register.md`.
- MCR is a **diagnostic ratio**, never the optimization target. CRPS is the primary proper score.
- One variable per run: **+coordinates, nothing else**, vs the bounded hurdle-NB baseline of record.
