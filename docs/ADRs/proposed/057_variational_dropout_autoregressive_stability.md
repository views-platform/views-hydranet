# ADR-057: Variational (Consistent-Mask) Dropout for Autoregressive Stability

**Status:** Proposed — **empirically insufficient** (see Outcome note)
**Date:** 2026-06-03

> **Outcome (2026-06-04):** I2 validation FALSIFIED this fix. With locked-mask dropout at inference (n=16), `violet_visitor` still exploded (lr_sb CRPS 2.13e17) while `pink_pirate` stayed bounded under identical code — implicating the trained recurrent dynamics (spectral radius), not dropout noise, as the driver of C-113. **Do not merge LockedDropout as the fix for C-113.** The module is correct and retained (training byte-identical, inference-only) but is not a stabilizer. Full analysis: `reports/postmortem_locked_dropout_negative_result.md`. Next directions: ADR-028 §2 (cell-state clamp / in-domain feedback bound), generalized teacher forcing (Hess 2023), spectral-radius/Lipschitz control, and bounded-output heads.
**Branch:** `fix/variational-dropout-autoregressive-stability`
**Depends on:** ADR-027 (autoregressive inference), ADR-028 (numerical stability guards), ADR-054 (Tobit loss / latent feedback), ADR-056 (scheduled sampling)
**Working detail:** `reports/preanalysis_autoregressive_stability.md` (mechanical findings, pre-registered hypotheses, full alternative analysis)

## Context

The post-C-111 clean retrain (2026-06-03) produced astronomically exploding evaluation metrics on two regression heads (`blue_stranger/lr_ns_best`, `violet_visitor/lr_sb_best`: raw predictions to ~1.66e22; CRPS to 1e15), while `pink_pirate` and all `lr_os_best` heads stayed healthy. The decisive datum is **step-wise growth**: step-1 CRPS is normal, the divergence appears and compounds across the 36-step autoregressive roll-forward. The error is generated *by the recursion*, not by a static output transform.

Root-cause analysis (see the pre-analysis plan) attributes this to **autoregressive recurrent runaway** — ADR-028 §1 (spectral-radius compounding) and §2 (additive cell-state explosion), both marked *Deferred* in ADR-028's status notes — with `expm1` (ADR-028 §3) as the final amplifier. A second, independent and immediately actionable driver is the **stochasticity mechanism**: our posterior is Monte-Carlo Dropout, and the inference loop resamples a **fresh dropout mask at every one of the 36 autoregressive steps**.

Gal & Ghahramani (2016, *A Theoretically Grounded Application of Dropout in RNNs*) state exactly this failure — *"noise added to recurrent layers will be amplified for long sequences, and drown the signal"* — and prescribe the remedy: *"use the same dropout mask at each time step."* Mechanical grounding (Appendix A of the pre-analysis plan): HydraNet has one shared `nn.Dropout` applied 16× on the U-Net **emission** path and **none** on the ConvLSTM recurrence; because inference is **free-running** (the prediction feeds back as next input), emission-path dropout — normally the "safe" placement — becomes a compounding noise source.

## Decision

Adopt **variational (consistent-mask) dropout** in the inference path, via a `LockedDropout` module that caches its Bernoulli mask (keyed by tensor shape) and exposes `reset()`. Three settled choices:

1. **Mask granularity = per posterior sample.** One mask held fixed across a full `predict()` trajectory (digest + seed + 36 steps); `reset()` called once per `sample_idx`. This is the literal Gal MC-dropout estimator (eq. 4): one weight realization `ω̂ ∼ q(ω)` per trajectory, K trajectories = the posterior.
2. **Inference-only, first.** Lock masks only in the posterior-sampling path; leave training unchanged. Testable with **zero retrain** by re-evaluating existing artifacts; keeps weights identical to the C-111 baseline.
3. **No recurrent dropout; no output clamp.** Lock only the existing emission-path dropout. Do **not** add dropout to the ConvLSTM cells (a modeling change) and do **not** clamp regression-head output (ADR-028 §3) — both are rejected here (see Alternatives).

This intervention is **magnitude-neutral**: it changes the temporal correlation of the dropout noise, not the values the heads may emit. That is essential because we are simultaneously fighting under-prediction (MCR ≪ 1); any magnitude cap would worsen MCR.

## Alternatives considered (preserved, not discarded)

| Alt | Description | Disposition | Revisit when |
|-----|-------------|-------------|--------------|
| 1b | Per-step mask (status quo) | **Rejected** — this is the bug | never |
| 1c | One mask across many samples | **Rejected** — posterior collapses | never |
| 2b | Consistent mask at train **and** inference | **Deferred** — strict Bayesian faithfulness; needs retrain; couples with ADR-056 | if we keep dropout as the long-term posterior, or need rigor for a production train |
| 3b | Add consistent-mask **recurrent** dropout | **Deferred** — a regularization/modeling change | only after the provenance question below is resolved |
| ADR-028 §3 | Clamp `out_reg` before `expm1` | **Rejected as primary** — caps the upper tail, fights MCR | only as a last-resort safety net beneath a dynamics fix |
| §2 fallback | Cell-state clamp / in-domain feedback bound | **Held in reserve** — magnitude-neutral | if consistent masks do **not** stop the runaway (→ deterministic recurrence is the driver) |

## Consequences

- **Posterior spread will narrow.** Per-step resampling was injecting fresh white noise every step; locking removes it. This must **not** be read as a regression: the pre-fix spread was *right for the wrong reason* (an artifact of the same per-step noise that ran away). The correct post-fix test is **calibration** (coverage vs nominal; MCR near 1), not raw spread width.
- **Sample count.** Each posterior sample is now one coherent draw rather than a 36-step noise average; we expect to need `n_posterior_samples` back at 64 (production) for a smooth posterior. Validation re-eval should use 64.
- **MCR is not addressed by this ADR.** Stabilization ≠ calibration. The principled fix for both is a learned latent-variable posterior (see Open Questions).
- **Mild train/test mismatch** under inference-only (2a) — accepted; reversible; a strict subset of 2b.
- **Falsifiable:** if the runaway persists with locked masks and no clamping, the driver is the deterministic recurrence, not dropout noise → pivot to the §2 fallback (pre-committed).

## Open questions

- **Provenance of "no recurrent dropout."** Why HydraNet never applied dropout to the ConvLSTM connections is not remembered or recorded. Until established (deliberate stability choice vs oversight vs inherited), this is an *undocumented assumption*, not a justified decision, and blocks any 3b experiment. → candidate risk-register entry.
- **The VAE arc (near-term, not someday).** The durable replacement for the dropout-induced posterior is a learned `q_φ(z|x)` with the reparameterization trick and an ELBO objective (Kingma & Welling 2019), optionally with flexible/autoregressive posteriors (IAF) for the conflict-count tail — the only route that addresses explosion *and* MCR together. Warrants its own ADR when committed.

## References

- Y. Gal, Z. Ghahramani (2016). *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks.* NeurIPS. `papers/Gal2016_RecurrentDropout.pdf`.
- Y. Gal, Z. Ghahramani (2016). *Dropout as a Bayesian Approximation.* ICML. `papers/Gal2016_DropoutBayesian.pdf`.
- D. P. Kingma, M. Welling (2019). *An Introduction to Variational Autoencoders.* Found. & Trends ML. `incoming/vea/1906.02691v3.pdf`.
- `reports/preanalysis_autoregressive_stability.md` — full mechanical findings, pre-registered hypotheses, and decision rationale.
