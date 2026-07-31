# 07 — Experiment Log (append-only)

**Date:** 2026-06-06 · **Status:** seeded (skeleton) · **Dossier:** [00_README](00_README.md)

Append-only ledger of every run + outcome — **including negatives/postmortems**. Each entry links its pre-registration (`05` or a `preanalysis_*.md`) and records its **verdict against the pre-registered falsifiers**. This is a Popperian record *and* a meta-evaluation corpus (later used to assess the workflow + the skills) — **not a highlight reel**. No ad hoc rescue.

## Entry format
```
### EXP-NN — <title>  (YYYY-MM-DD)
- Pre-registration: <link to 05 / preanalysis_*.md>
- One variable: <the single change vs baseline>
- Driver / artifact / results: <run_*.sh · model_*.pt · logs/*_RESULTS.txt>
- Readout: <diagnose_io_gain attractor · step-wise CRPS · MCR/PIT/coverage · sharpness>
- Verdict vs falsifiers: <which fired / none>
- Decision: <proceed / escalate (B2/ZITD) / postmortem link>
```
Legend: ✅ predictions held · 🔴 falsifier fired · ⚪ inconclusive.

---

## Precursors (done — the evidence base this program builds on)
- **`results_freezeh_ablation.md`** — freeze_h inert; divergence rides the prediction→input feedback, not the recurrent state.
- **`results_io_gain_diagnostic.md`** — violet's free-running map settles out-of-range (~log 40 → expm1 ~1e17); `diagnose_io_gain` is the fast readout.
- **`results_feedback_clamp.md`** — clamping the feedback is a safety rail, not a fix (ramps-to-ceiling).
- **`preanalysis_balancer_sweep.md` §RESULT** — F2 fired (seed4_frozen → inf): freezing is seed-fragile; exposure bias is the root → do rollout training.

## Planned

### EXP-01 — Scheduled-sampling always-feed (`ss_epsilon_max` 0.25 → 1.0)  (2026-06-06) ⚠️
- **What & why:** the "just try a fix" cheap stab — crank own-prediction training to the max. The cross-step gradient stays **detached**, so this is the crude *per-step* proxy, **not** the `rollout_horizon` B1 of `05` (which remains the real build). Surfaced via `02c`.
- **One variable:** `ss_epsilon_max` 0.25 → 1.0 on **violet / seed-42, active balancer**.
- **Driver / artifact / log:** `views-models/scripts/run_ss_alwaysfeed.sh` · `calibration_model_20260606_190457.pt` · `logs/ss_alwaysfeed_20260606_171806.log`. Config trap-restored to 0.25 after.
- **Readout (`diagnose_io_gain`, synthetic):** the free-running rollout **stops exploding** — but **collapses to 0.00** by step ~12 (step1 ≈1.0 → step12/24/48 = 0.00; expm1 = 0). Local operator gain still ‖J‖₂>1, but the attractor moved to the trivial **zero fixed point**.
- **Verdict vs falsifiers:** runaway **gone** (no out-of-range) — but this is the **"bounded but useless"** collapse the panel *and the chair* predicted (C-126 family): bounded by going degenerate, not by forecasting. ⚠️ **partial — cured the explosion by killing the forecast.**
- **Caveat:** synthetic probe; healthy pink settles nonzero-in-range, this flatlines to 0 → strong collapse signal, but real-data skill not yet confirmed (folded into EXP-02's readout).
- **Decision:** → method review `02d` → run the **middle ceiling** (EXP-02) with a **real eval**, then build the structured (c) GTF.
- **Evidence added to the axis:** ceiling **0.25 → explode**, **1.0 → collapse** — brackets the scheduled-sampling-intensity axis.

### EXP-02 — Scheduled-sampling middle ceiling (`ss_epsilon_max` ≈ 0.5) + real eval  (planned · pre-registered here)
- **▶ UN-PARKED 2026-06-09 — this IS the live R4 probe (#93).** After Arm-1 un-collapsed magnitude (the hurdle
  works one-step; the explosion is the **untrained rollout** — C-136), the SS-middle ceiling **on the hurdle config**
  is exactly the cheap rollout-training probe: does un-collapsing the head break the old explode(0.25)/collapse(1.0)
  bracket → a stable *nonzero* rollout? **Readout = step-1 + full-36 + positive-subset proper scores** (the R4 readout,
  #93). If it explodes/collapses → the B1-pushforward MVP (`04` P3). *(Was: ⏸️ PARKED 2026-06-08 as a proxy for the
  magnitude question — now un-parked, since the hurdle changed the head it runs on.)*
- **Hypothesis:** plain scheduled-sampling has a stable-*nonzero* regime somewhere between explode (0.25) and collapse (1.0). *(Prior is low — see the knife-edge risk RT-knifeedge + Huszár bias; this is a VoI **gate**, not a candidate endpoint.)*
- **One variable:** `ss_epsilon_max` → ~0.5 on violet/seed-42, active balancer. (The dial already ramps low→ceiling; we only lower the ceiling.)
- **Readout:** BOTH `diagnose_io_gain` (stability across 36) **and a real `--evaluate`** (CRPS/MCR — *skill*, not just boundedness). **Do not judge on the synthetic probe alone** (the 1.0 collapse read "healthy" at 0.00).
- **Pre-registered decision rule:**
  - **stable ∧ nonzero ∧ nontrivial skill** ⇒ plain-SS *has* a regime (cheap, surprising win) → confirm on a 2nd seed.
  - **explode ∨ collapse ∨ stable-but-zero-skill** ⇒ plain-SS has **no good regime** (confirmed, not assumed) → commit to **(c) GTF**.
- Baselines: `…233938` (explode) · `…051634` (frozen-healthy, calibration ref).

- **🔴 RESULT 2026-06-10 — EXPLODED (R4, #93).** `ss_epsilon_max=0.5` on the **hurdle** config (one variable vs Arm-1). Trained clean; **eval failed — "Input contains infinity"** (same as Arm-1). Durable readout (`scripts/mcr_readout.py`, artifact `predictions_calibration_20260610_010843`): **FULL-rollout MCR ≈ 3.4e33 / 3.1e33 / 7.5e33** (sb/ns/os) — indistinguishable from Arm-1's ≈ 2.5e33 / 2.5e33 / 6.3e33; SS=0.5 made **no meaningful difference**. STEP-1 magnitude no better (sb 0.21→0.088). **Verdict: `explode` falsifier fired.** ⇒ **The scheduled-sampling axis is now fully bracketed: 0.25→explode, 0.5→explode, 1.0→collapse — plain per-step SS has NO good regime (proven, not assumed), even on the un-collapsed hurdle head.** Per the decision rule → commit to **(c) the real GTF / B1 rollout training (#78)** with cross-step gradients, **or** the count-likelihood head (escalation). The cheap-proxy era on this axis is closed.

### Later — the real build (c): controlled unroll-K / GTF
- Pre-registration: [`05_analysis_plan.md`](05_analysis_plan.md) (the `rollout_horizon` B1/GTF MVP). Gated on the `03 §3` harness build. The chair's "scheduler, start-low-get-higher" instinct **is** GTF's α-anneal ⇒ build **B2-flavoured** (bounded cross-step gradient), not just a deeper pushforward.
