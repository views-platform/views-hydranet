# Negative Result — Variational (Locked-Mask) Dropout Did NOT Fix the Autoregressive Runaway

**Date:** 2026-06-04
**Companion to:** `reports/preanalysis_autoregressive_stability.md`, `docs/ADRs/proposed/057_variational_dropout_autoregressive_stability.md`
**Status:** Hypothesis **FALSIFIED**. The pre-registered fix (ADR-057, I1/I2) does not resolve risk C-113. Back to the drawing board.
**Branch:** `fix/variational-dropout-autoregressive-stability` (code unmerged, retained — see §6)

---

## 1. What we tested (and pre-registered)

Pre-analysis §3.1 committed, *before* running: with consistent-mask (variational) dropout enabled at inference (ADR-057, Gal & Ghahramani 2016, RNN), the step-wise CRPS on the two exploding heads would be **bounded across all 36 steps with no magnitude capping** (raw predictions ≲ data max ≈ 1.8e5). Falsifier (§3.3): if the runaway persists, the driver is the deterministic recurrence, not dropout noise.

I2 method: re-evaluate the **existing June-3 artifacts** (no retrain; inference-only change) with the LockedDropout path, `n_posterior_samples=16`, one model at a time.

## 2. What happened

| Model (seed) | head | locked-mask now (n=16) | June-3 pre-fix (n=4) | verdict |
|--------------|------|------------------------|----------------------|---------|
| pink_pirate (4) | lr_sb | 0.1325 | 0.1326 | bounded ✓ |
| pink_pirate (4) | lr_ns | 0.0317 | 0.031 | bounded ✓ |
| pink_pirate (4) | lr_os | 0.0511 | 0.051 | bounded ✓ |
| **violet_visitor (42)** | **lr_sb** | **2.13e17** | 4.6e6 | **EXPLODED** |
| **violet_visitor (42)** | **lr_ns** | **2.78e9** | 0.04 | **EXPLODED** |
| **violet_visitor (42)** | **lr_os** | **54.5** | 0.05 | **elevated/diverged** |

(step-wise CRPS. blue_stranger not run — see §5.)

The locked-mask fix **did not bound the runaway.** Pre-registered acceptance failed for violet_visitor. **Hypothesis FALSIFIED.**

## 3. The decisive evidence (and what it points to)

The cleanest, least-confounded observation: **pink_pirate and violet_visitor ran the *identical* locked-mask inference code, at the *identical* sample count (n=16), and pink stayed bounded while violet exploded.** The only difference is the *trained weights* (different seed/config). Therefore the explosion is a property of the **trained recurrent dynamics** (effective spectral radius ≥ 1 for some weight configurations), not of the dropout-noise treatment. This is ADR-028 §1 (spectral-radius compounding) / §2 (cell-state accumulation) territory — exactly the pre-committed falsifier branch.

Corollary: dropout masking — locked *or* per-step — is **not the lever**. Gal-2016 consistent-mask dropout addresses *overfitting noise* in *teacher-forced* sequence models; it does nothing for a *deterministically divergent* free-running recurrence.

## 4. Biases weighed (intellectual-honesty audit)

- **Confirmation bias (mine):** I was confident in the Gal-2016 fix and pre-registered success. The result contradicts it. Per Popperian discipline (and our own §3.3), we accept the falsification rather than rescue the hypothesis. No ad hoc "it would have worked if…".
- **Sample-count confound (acknowledged):** the pre-fix baseline was n=4; this run is n=16, so the *magnitudes* are not directly comparable (more samples surface more divergent trajectories). This does **not** save the hypothesis: a working fix bounds trajectories at any n, and 2e17 is not bounded. The binary verdict (bounded vs not) is confound-free; the cross-run magnitude delta is not interpreted causally.
- **Counter-hypothesis we did NOT rule out — locking may HURT:** in a free-running recurrence a *fixed* mask drops the same channels for all 36 steps; if a stabilizing channel is dropped for the whole horizon, divergence could worsen vs per-step resampling (which lets every channel contribute intermittently). This is the opposite of Gal's teacher-forced finding. We did not run the clean control (unlocked@16) to distinguish "ineffective" from "counterproductive."
- **One-model caution:** violet falsifies the *universal* claim; blue_stranger (the other exploder) was not run, so we have not confirmed the pattern on a second diverging model. pink_pirate is uninformative about the fix (it never exploded — it only confirms no regression).
- **"Was the fix even active?"** Yes: the locked path initialized in both runs (inference log, set_locked_dropout site); pink ran the same code and stayed healthy. The fix engaged; it simply did not address the driver.

## 5. What is and isn't established

- **Established:** locked-mask dropout (inference-only, n=16) does **not** bound violet_visitor's runaway; pink_pirate is unaffected (no regression). The driver is most consistent with the trained recurrent dynamics, not dropout noise.
- **Not established:** whether locking *helps, hurts, or is neutral* vs per-step at matched n (no unlocked@16 control); whether blue_stranger follows violet (not run). These are cheap to close if we want certainty before fully abandoning the dropout angle — but they would only sharpen an already-clear falsification.

## 6. Disposition

- **ADR-057:** remains Proposed but is **empirically insufficient as the fix for C-113** — status note added pointing here. Do **not** merge LockedDropout as "the fix."
- **Code:** `LockedDropout` and the inference wiring stay on the branch, **unmerged**. The module is correct and harmless (training byte-identical; inference-only), and a learned-posterior direction may still want consistent masks — but it is not a stabilizer and must not be sold as one.
- **C-113** (autoregressive runaway) stays **OPEN**, now with a sharper diagnosis: deterministic recurrent divergence, surfaced per trained-weight configuration, amplified by `expm1`.

## 7. Back to the drawing board — literature directions (library inventory, 2026-06-04)

The problem, reframed: **a free-running autoregressive ConvLSTM forecaster whose trained recurrence diverges over the 36-step horizon for some weight configurations, amplified by the `expm1` inverse.** Dropout is not the lever. Candidate directions, with what the library already holds vs gaps:

| Direction | In library | Gap |
|-----------|-----------|-----|
| **A. Generalized/sparse teacher forcing** (control diverging trajectories via the system's Lyapunov exponent) | ✅ **Hess et al. 2023** (`incoming/deep_consored/hess23a.pdf`) — strongest direct hit; reshapes to pLRNN + sparse TF to keep recursive trajectories on the attractor | — |
| **B. Bounded-output / count-distribution heads** (avoid `expm1` overflow entirely) | ✅ Tweedie (Damato 2025, Jiang 2023), renewal (Turkmen 2020/21), **ZINB** (Iacus 2025), DeepAR (Salinas 2020), spline-RNN (Gasthaus 2019) | GP-Tweedie; Lambert 1992 ZIP |
| **C. Spectral-radius / Lipschitz control of the recurrence** | ⚠️ only LSTM (1997) CEC, Gal dropout | **MAJOR GAP:** Miyato 2018 spectral norm; Erichson Lipschitz-RNN; Chang antisymmetric-RNN; Arjovsky unitary-RNN — none present |
| **D. State/cell clamping, weight damping, grad clip** (ADR-028 §1/§2 — our own fallback I4) | ⚠️ LSTM, Neural ODE (implicit) | explicit recurrent clamping/damping studies |
| **E. Non-autoregressive / diffusion paradigms** (sidestep recurrence blow-up) | ✅ **GenCast** (Price 2023, diffusion ensemble weather), Switching-SSM (Xu 2021), Temporal Fusion Transformer (Lim 2021), STGformer | — |
| **F. Precedent: this exact failure** | ✅ **Radford 2022** (HighRes conflict ConvLSTM) *documents* forecast escalation/divergence as an open challenge — direct domain precedent | (descriptive, no fix) |

**Most promising near-term reads:** (A) Hess 2023 GTF — it is literally about stopping recursive predictors from diverging; (D) ADR-028 §2 cell-state clamp / in-domain feedback bound (our pre-committed I4, magnitude-neutral); and (C) spectral-norm / Lipschitz constraints on the ConvLSTM recurrent convolutions — the principled root fix, but we must fetch the papers. (B) and (E) are larger paradigm shifts that also dissolve the `expm1` amplifier, overlapping the registered research issues (#60 Tweedie, #63 ZINB) and the VAE arc (I10).

**To fetch (gap C/B):** Miyato 2018 (spectral norm); Erichson et al. (Lipschitz RNN); Chang et al. (antisymmetric RNN); Arjovsky et al. (unitary RNN); Lambert 1992 (ZIP).

---

*This document records a falsification. The fix was a reasonable, literature-grounded hypothesis; the data said no. The value is the sharpened diagnosis: the driver is the recurrence, not the dropout — which redirects the search toward dynamics control (GTF / Lipschitz / cell-state) and bounded-output heads.*
