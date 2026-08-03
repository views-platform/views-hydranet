# 07 — Experiment Log (append-only)

### EXP-01 — C-147 gate reliability on R4/R5 (2026-06-20) ⚪→🔴 H2 falsified; H1 holds *only under rollout*
- **Pre-registration:** `05_analysis_plan.md` (written before this run; `pos_weight` read only after).
- **What ran:** `scripts/gate_reliability.py` (new) on R4 no-coords (`…_162127`) + R5 coords (`…_165915`),
  3 targets, n-boot=200. Read-only; C-136 join guards passed (unique index, every π-cell matched);
  F-alignment-artifact guard passed (`by_*` ∈ [0,1] on disk). Readout: `gate_reliability_readout.md`.
  Internal-consistency check passed (base-rate Brier = prevalence·(1−prevalence), e.g. ns 0.0029).
- **`loss_class_pos_weight` (read after pre-reg): `10.0`** — `≫1` but **far below** the O(100–700)
  inverse-base-rate I predicted in H2.

**Headline numbers** (mean π vs onset prevalence; ECE; Brier skill vs base-rate):

| run | target | prev | STEP-1 mean π / ECE / skill | FULL mean π / ECE / skill |
|-----|--------|------|----------------------------|----------------------------|
| R4 no-coords | sb | 0.77% | 0.012 / 0.007 / −0.06 | 0.693 / 0.685 / **−77** |
| R4 no-coords | ns | 0.34% | 0.011 / 0.007 / −0.88 | 0.702 / 0.699 / **−239** |
| R4 no-coords | os | 0.41% | 0.008 / 0.005 / −0.37 | 0.686 / 0.682 / **−145** |
| R5 coords | sb | 0.77% | 0.028 / 0.020 | 0.884 / 0.875 / −99 |
| R5 coords | ns | 0.34% | **0.164 / 0.161** | 0.902 / 0.899 / −308 |
| R5 coords | os | 0.41% | 0.097 / 0.093 | 0.868 / 0.863 / −184 |

At FULL, ~68% of cells pin at π≈1.0 while empirical onset is ~0.3% in **every** π bin (the gate is
uninformative by the end of the rollout).

**Verdict vs pre-registered falsifiers (`05`):**
- **H1 (gate miscalibrated-HIGH): ✅ HELD — but only under rollout.** FULL: mean π ≈ 0.69–0.70 (no-coords)
  vs prevalence ~0.3–0.8%, ECE ≈ 0.68–0.70, Brier skill −77 to −239 (worse than predicting the base rate).
- **H2 (the class weight drives it): 🔴 FALSIFIED — twice.** (1) `pos_weight=10`, not O(100–700); (2) at
  **STEP-1** (teacher-forced) the no-coords gate is **calibrated** (ECE 0.005–0.007, mean π ≈ prevalence,
  reliability near-diagonal in the dominant low bin). A class weight that broke calibration would show at
  step 1; it doesn't. The class weight is **not** the cause.
- **F-calibrated:** did NOT fire at FULL (predicted). It *partially* describes STEP-1 (the no-coords gate
  *is* ~calibrated teacher-forced) — and that is exactly the observation that falsified H2.
- **New (not pre-registered): coords degrade the gate even at STEP-1** (ECE 0.02–0.16 vs 0.005–0.007;
  ns mean π 0.164 ≫ 0.34% prevalence *before any rollout*) and worse at FULL. Coords add spatial capacity
  the gate uses to over-fire from the first step.

**Mechanism (the trustworthy conclusion):** the onset gate is **calibrated when teacher-forced** and
**saturates to π≈1 through the autoregressive rollout** — the miscalibration is *born in the feedback loop,
not the loss*. This is the C-113 autoregressive amplifier; it is precisely what the old model's
frozen-hidden-state hack suppressed. Coords amplify it (and start it earlier).

**Decision (findings log — they don't steer; this is the pre-committed read):**
- **Do NOT "fix the gate loss."** Dropping the class weight / adding a base-rate prior to the gate would
  treat a non-cause — the gate is fine at step 1. (My earlier instinct, now corrected by data.)
- The principled target is the **autoregressive feedback / rollout dynamics**: keep the gate calibrated
  *through* the rollout — the principled version of the frozen-state prior. Pre-register that separately.
- **CoordConv:** confirmed it worsens gate calibration (capacity misuse); park it until the rollout is fixed.
- This resolves the loss-vs-feedback question the session opened with: **it's the feedback, not the loss/gate.**
