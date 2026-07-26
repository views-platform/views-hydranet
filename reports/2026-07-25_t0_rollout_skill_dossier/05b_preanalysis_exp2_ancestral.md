# 05b — Pre-analysis plan: EXP-2, the ancestral (sample-feedback) rollout

**Pre-registered 2026-07-26, BEFORE running.** The deployed-skill verdict (EXP-1 was a diagnostic of the
*broken* mean-feedback object, C-218). One variable: **feedback content** (sample vs mean). Scored on the
frozen per-horizon ruler (`tools/rollout_skill_score.py`); metrics = crps_all/events/none split + AP + Brier
+ size_ratio (no twCRPS/PIT). Same 13 origins, N=170,430, calibration 457–504.

## Hypothesis (H-SAMPLE)

Feeding back a per-cell **sample** (a sparse count draw) instead of the diffuse emit-mean keeps each
trajectory sparse and in-distribution, so (a) the learned-π arm (zinb) that bloomed under mean-feedback
stays **bounded**, and (b) the rollout stops being **timid-zero** — it feeds back real magnitudes, so
size_ratio > 0 and event skill (crps_events) should be no worse and plausibly better than the timid
mean-feedback rollout, out to the horizon where the intrinsic ceiling dominates.

## The ONE change (behind a default-off flag)

`hydranet_inference.py::predict()` AR loop (~line 432): a `rollout_feedback` config field ∈
{`mean` (default = today, byte-identical), `sample`}. When `sample`, the fed-back magnitude is a **single
seeded family draw** per cell from that step's activated params (`family.sample`, k=1, log1p space),
composed as the arm dictates — instead of `_emit_magnitude`'s `compose_mean`. The emitted per-horizon D×K
cube (what the ruler scores) is unchanged; only the *feedback copy* changes. Determinism via the S2 #121
seeded generator. Legacy point heads: flag inert (no family) — fail-loud if set.

## Arms

- **nb** (`…102130`) sample-feedback — the primary (was timid-zero bounded under mean-feedback).
- **zinb** (`…063927`) sample-feedback — the bloom test (bloomed catastrophically under mean-feedback).
- Compared **against their own EXP-1 mean-feedback curves** (same artifact, same ruler — a clean A/B on the
  one variable).

## Pre-registered predictions

- **P1 (zinb un-blooms):** zinb sample-feedback crps_all stays **bounded** across h (no 0.14→5.4 runaway);
  crps_none does not explode (the sparse draws don't smear onto zero cells).
- **P2 (nb less timid):** nb sample-feedback **size_ratio > 0** (vs 0.0 under mean-feedback) — it feeds back
  real magnitudes, so the median event-cell forecast is no longer zero.
- **P3 (event skill ≥ mean-feedback):** nb sample-feedback crps_events ≤ its mean-feedback crps_events at
  most horizons (feeding back magnitude helps, or at worst ties). Occurrence (AP) unchanged-to-better.
- **P4 (honest diffuseness, not runaway):** the *spread across sample-paths* grows with h (correct marginal
  uncertainty) while each single path stays sparse — diffuse-in-aggregate, sharp-per-path.

## Pre-committed falsifiers

- **F-S1 (sample-feedback ALSO runs away):** if crps_all blooms under sample-feedback too → the instability
  is intrinsic to the input→output map (io-gain>1 even on in-distribution inputs), NOT a mean-feedback
  artifact → escalate to rung-4 (bound the gain) / accept the model can't roll out. Kills H-SAMPLE.
- **F-S2 (still timid / no magnitude gain):** if nb sample-feedback size_ratio stays ≈0 AND crps_events
  doesn't improve → the timidity is a **body** defect, not a feedback artifact → sample-feedback is not the
  lever (the fix is the per-cell head, not the rollout).
- **F-S3 (determinism breaks):** re-run changes any number → S2 #121 violated → fix before trusting.
- **F-S4 (paths not sparse):** a single sample-path at T=8/17/35 is still a dense blob (not a plausible
  sparse conflict map) → sampling didn't buy in-distribution-ness → diagnose.

## Method

1. **Implement** the `rollout_feedback` flag (TDD, default-off). Red tests: flag off ⇒ byte-identical
   rollout to today (parity); flag on ⇒ feedback differs; determinism (seeded, reproducible); legacy head
   ⇒ fail-loud. Full suite + lint + determinism gate green.
2. **Stealth:** trap-restore the violet floor md5 in the driver; never commit views-models.
3. **Run** eval nb + zinb with `rollout_feedback=sample` → persist ancestral `origin_*` dirs.
4. **Score** at all h with the frozen ruler; **A/B vs the EXP-1 mean-feedback curves** (same artifacts).
5. **Render** one sample-path map at h=8/17/35 (P4/F-S2 visual).
6. **Log** (`07`) vs F-S1..F-S4. Single-seed INDICATIVE; 3-seed + block-bootstrap for any KEEP.

## Decision rules

- **F-S1 fires ⇒** the bloom is intrinsic to recursion → EXP-3 oracle becomes the priority (is ANY rollout
  skillful?), and C-223 (direct-multi-horizon) escalates.
- **F-S2 fires ⇒** stop chasing the rollout for magnitude; the lever is the body/head (redirect the epic).
- **H-SAMPLE holds (P1–P3) ⇒** sample-feedback is the deployed rollout; proceed to EXP-3 (oracle gap =
  how much of the residual is exposure-bias vs the predictability ceiling), then the τ / GTF ladder.
