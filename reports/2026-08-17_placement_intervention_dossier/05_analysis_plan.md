# Pre-Analysis Plan — coherent feedback sampling, re-tested on a vehicle that has skill (M5 / C-290)

**Date:** 2026-08-17 (pre-registered **before** execution)
**Dossier:** `reports/2026-08-17_placement_intervention_dossier/`
**Builds on:** `reports/2026-08-17_vehicle_replication_dossier/` (EXP-02, GREEN) ·
`reports/2026-08-16_feedback_realism_dossier/` EXP-05 (the null being re-tested) ·
`reports/RESULTS_LEDGER.md` M5, M10–M14, C-290, C-296

## 1. Hypothesis

**H:** The independent per-cell Bernoulli draw that produces the fed-back occurrence field destroys
spatial information the gate still holds, and replacing it with a **spatially-correlated draw of the
same marginals** recovers rollout skill on a vehicle whose control is not floor-limited.

**H0 (the standing claim this attacks):** M5 — "fed clustering spanning 100× moves AP not at all."

## 2. Intervention (the ONE variable)

`feedback_length_scale` ∈ {0.5, 1.0, 3.0} on `HydraNetInference`, i.e. `correlated_bernoulli`
(Gaussian copula, marginals exact by construction) replaces `compose_samples`' independent Bernoulli
**on the feedback path only**. The scored cube keeps independent sampling. Everything else — artifact,
seed, origins, S, composition, targets — is held identical to the control.

Diagnostic argument, **no config key**; `None` is byte-identical to production.

## 3. Skepticism ledger

1. **This was already run and was null.** EXP-05 swept the same three length scales on
   `truncated_smoke` and found AP flat at ~0.007. The *only* reason to re-run is that the vehicle was
   **floor-limited**: its control sat at AP 0.0070, and last night proved that floor made
   `spatial_scramble` read +0.9% where it is truly −94%. If the re-test is also null on a control at
   0.2569, M5 becomes *much* stronger, not weaker. **I must not present a second null as a failure of
   the experiment.**
2. **Clustering is not placement.** Last night's decisive result is that *correct* placement carries
   ~95% of the gap and *wrong* placement is worse than the model's own output. A copula makes the field
   clumpy in whatever places the gate already favours; if the gate's favoured places are wrong, coherent
   clumps in wrong places could plausibly be **worse** than scattered draws. Predicted in F2.
3. **Not byte-paired (C-296).** The copula consumes a different number of RNG variates than the
   control's Bernoulli, so later steps' body draws come from a different stream. Read at **one
   significant figure**; a difference under ~0.01 AP is not interpretable as an effect.
4. **The gate itself diffuses.** M6/M7: the gate's own field smears during the rollout, and by step 12
   its top-K is 27× more clustered than the draw. A copula imposes *generic* structure, not the gate's
   *own* ranking — so even a positive result would not establish that we had used the model's
   information, only that clumpiness helps.
5. **Success could be a firing-rate artifact.** More clustering at fixed marginals changes the spatial
   distribution but the copula also changes which cells fire. `act_ratio` is recorded; a "win" that
   comes with a large activation shift is a confound, not a fix (F3).
6. **One seed, one vehicle.** Same limitation as everything else in this programme.

## 4. Pre-registered predictions

| # | Endpoint (primary first) | Prediction | Threshold |
|---|---|---|---|
| **P1** (primary) | gate AP at h18, target `sb`, best copula arm vs control 0.2569 | the copula **does not** materially help | pass if \|ΔAP\| < 0.01; a gain ≥ +0.01 refutes M5 |
| **P2** | fed-field clustering (`neighbour_pairs_per_active`) | rises monotonically with ℓ and brackets the oracle's | ℓ=3.0 ≥ oracle's value |
| **P3** | direction if any effect exists | any AP change is **negative or null**, per skepticism §2 | — |

**Primary endpoint is P1 with a numeric threshold, chosen so that the informative outcome is a null.**

## 5. Falsifiers (pre-committed — any one fires ⇒ the arm is void, not rescued)

- **F1 — silent no-op.** Fed-field clustering does not move with ℓ (ℓ=3.0 within 20% of the control's)
  ⇒ the copula never engaged; the arm's score is **void**, not evidence that clustering does not matter.
  (This is exactly how EXP-05's validity was established; the same check transfers.)
- **F2 — h=1 not identical.** h=1 `AP` differs across arms or from the control by > 1e-6 ⇒ something
  other than the feedback path moved (step 1 has no feedback) ⇒ **void**.
- **F3 — support mismatch.** `N` ≠ 170430 in any scored row ⇒ arms scored on different supports ⇒
  **void**.
- **F4 — firing-rate confound.** An arm shows ΔAP ≥ +0.01 *and* `act_ratio` differs from the control by
  > 25% ⇒ the gain is a firing-rate change, **not** a placement fix ⇒ inconclusive, not a win.

## 6. Method

`violet_visitor` · artifact `calibration_model_20260812_191742.pt` (sha `909f44c0…`) · target `sb` ·
h = 1,6,12,18,24,30,36 · 13 origins (457–504) · **S = 16** · seed 42 · calibration partition ·
eval-only, no training, no config mutation.

**Control:** the **already-scored** `identity` arm at
`reports/2026-08-17_vehicle_replication_dossier/results/score_violet_visitor_identity.csv`
(AP 0.4745 / 0.3924 / 0.2569 / 0.1370 at h1/6/18/36), which was measured last night on today's code and
is bit-identical to the preserved 2026-08-12 production cubes. **No control is re-run.**

**Oracle ceiling for context:** the `use_real` arm from the same dossier (AP ≈ 0.479 at h18).

**Readout order, cheap before expensive:** ℓ=1.0 first (the value that hit the real clustering target on
smoke). If F1 fires there, stop — the seam is broken and the other two arms are pointless.

**Run discipline:** reuse `reports/2026-08-17_vehicle_replication_dossier/tools/overnight_run.sh`
unchanged (per-arm sentinel, disk preflight, refuse-on-leftover, score-then-delete, artifact-sha and
config-md5 guards). ~10 min/arm on this vehicle.

## 7. Decision rules

| outcome | what we do |
|---|---|
| **P1 holds (null) and F1–F4 clear** | M5 is **confirmed on a non-floor-limited vehicle** and upgraded from "medium-high, read at one sig fig" to a robust null. **The inference-time sampling family is closed**, and the programme moves to the training side. This is the expected and useful result. |
| **ΔAP ≥ +0.01 with F4 clear** | M5 is **overturned** — it was a floor artifact like `spatial_scramble`'s +0.9%. Escalate immediately: second seed, third vehicle, and build top-K feedback (deferred below) as the stronger version of the same idea. |
| **ΔAP ≤ −0.01** | coherent clumps in wrong places are actively harmful — corroborates skepticism §2 and last night's `spatial_scramble` result. Same conclusion as the null for the fix, but a stronger statement about *why*. |
| **any falsifier fires** | that arm is void; fix and re-run before reading anything. |

**Explicitly deferred to a later phase, not part of this plan:** **top-K feedback** — feeding back the
gate's *own* top-K cells rather than generic clumps. It is the better-motivated intervention (M7: the
gate's ranking is 4–27× more clustered than the draw) but **it is not implemented**; `topk_mask` exists
only as a measurement. Building it requires a new draw branch in `_sample_feedback` with the C-296
RNG-pairing discipline, tests, and a review. Pre-registering it here would be pre-registering a plan for
code that does not exist.

## 8. Scope

One seed, one vehicle, one target, 13 origins. Per the standing rule of 2026-08-17, a positive is an
**escalation trigger**, not a conclusion. Not byte-paired (C-296) — one significant figure.
