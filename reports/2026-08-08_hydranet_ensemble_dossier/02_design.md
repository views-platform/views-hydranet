# 02 — Design: the 8-member Africa `concat` ensemble (2026-08-08)

## The approach
Take the validated **`gated_NB`** posterior (v2 scoreboard ship candidate) and deliver it as an **8-member,
equal-weight `concat` ensemble** on `africa_me_legacy`. Each member is one `(family, composition, seed)` on the
identical v2 foundation config, trained 300 lessons; `PredictionFrameEnsembleManager` `concat`-pools their D×K
posterior cubes on the sample axis (pooled draws = 8×`S`). The ensemble — not any single member — is the
delivery unit.

## Why this design
- **Variance reduction + occurrence calibration** — averaging independent posteriors of a seed-stable,
  occurrence-strong model is the classic, low-risk ensemble win; it shows first at short horizons (P1).
- **Horizon hedge** — gated_NB owns short-horizon occurrence; th_gated_NB (τ=0.5) buys a marginal long-horizon
  AP edge (v2 scoreboard). Mixing them hedges the horizon regime.
- **Structural hedge** — mixture_NB is a different head family (2-component NB); even though it's NULL on
  magnitude (tail_decoupled_head dossier), its *different* error structure adds diversity to the pool. It is a
  **placeholder** the heavy-tail head (#241) will replace when it lands.
- **`concat` (not weighted)** — the PF ensemble path is equal-weight sample-pooling by construction; weighted
  pooling is out of scope (and there's no validated per-member weight).

## Why NOT the alternatives
- **ZINB excluded** — self-zeroed ZINB blooms in the free-running rollout (v2 scoreboard F1); sample-feedback
  stabilises the *gated* arms but not self-zeroed ZINB. Including it would re-arm the bloom.
- **No magnitude claim** — every member shares the ξ=0 ceiling; the ensemble cannot manufacture magnitude skill
  the members lack. This is a hedging/calibration play, honestly scoped.
- **No population, no global scale-up** — deferred (population inherited an AP-collapse negative; global `land`
  is a follow-on that hands off `rusty_bucket`).

## Member → dir mapping (recorded at S3; smoke used a provisional mapping)
8 existing HydraNet dirs (violet_visitor, bright_starship, bold_comet, blazing_meteor, heavy_freighter,
pink_pirate, blue_stranger, purple_alien) each become one roster member. 5 are already on datafactory; 3
(pink_pirate, blue_stranger, purple_alien) migrate at S2. heavy_freighter switches its global `land` grid →
africa. The exact name→`(family,composition,seed)` mapping is locked in the S3 story + logged in 07.

## Open design decisions (→ 05 needs-decision)
1. **Roster composition** — 3 gated / 2 th_gated / 3 mixture (default) vs fewer mixture + more gated_NB.
2. **`S`** — the per-member D×K sample count (memory-safe at 8×`S`).

**Graduates to a proposed ADR on `promote`** (the ensemble as the datafactory-consistent delivery unit),
cross-linking this dossier as the evidence trail.
