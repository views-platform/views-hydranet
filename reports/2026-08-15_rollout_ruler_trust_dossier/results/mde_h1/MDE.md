# Minimum detectable effect at P = 13

Epic #263 / S4 (#268). Pair: `violet_visitor` − `climatology`, target `sb`, h = 1. 13 origins, N = 170,430 cell-months. 2000 bootstrap replicates, 90% CI, seed 0.

| quantity | value |
|---|---:|
| observed mean CRPS differential | -0.0589839 |
| origin-block CI | [-0.0743686, -0.0432933] |
| **MDE (origin-block, half-width)** | **0.0155376** |
| iid-over-cells CI (for contrast only) | [-0.0717186, -0.0468652] |
| iid half-width | 0.0124267 |
| **origin-block ÷ iid width** | **1.25×** |
| separable from 0 at 90%? | YES |

## Per-origin mean differential

| origin (m0) | mean Δcrps |
|---:|---:|
| 457 | -0.00134421 |
| 458 | -0.016971 |
| 459 | -0.0231373 |
| 460 | -0.00406547 |
| 461 | -0.0401856 |
| 462 | -0.0610353 |
| 463 | -0.0704127 |
| 464 | -0.0886922 |
| 465 | -0.0887262 |
| 466 | -0.101892 |
| 467 | -0.0868345 |
| 468 | -0.093816 |
| 469 | -0.0896776 |

## Reading this

An effect smaller than **0.0155376** cannot be distinguished from zero at 90% with 13 origins, however many cells there are. A null on this pair therefore means *either* no difference *or* no power — the distinction C-254 exists to preserve.

The iid-over-cells bootstrap gives a CI **1.25× narrower**. That is the overconfidence C-221/C-253 warn about: it treats ~13k spatially co-active cells per origin as independent observations.

**Neither reading licenses using the iid bootstrap.** It is reported as a contrast only. The origin block is the correct unit because the 36-month futures of adjacent origins overlap; that is a structural fact about the design, not an empirical result about which interval happens to be wider on one pair.

## Why this is not called a Giacomini–White test

The models are trained **once** and scored at 13 rolling origins, so the forecasting scheme is **fixed**, satisfying Giacomini & White 2006 §3.2 Comment 3 (*"Expanding window forecasting schemes are ruled out by assumption"*). That holds **by construction**: one prediction directory is the output of one training run, so its 13 origins necessarily share an artifact — nothing checks them pairwise, because there is nothing that could differ. What `partition_audit.py` adds is that the artifact's sha256 is resolved and recorded, and now fails loud when it cannot be, so the claim is auditable after the fact rather than merely asserted.

But `gw_stratified` is a **bootstrap on the mean loss differential**, not a GW conditional-predictive-ability regression: there is no analytic variance estimator and no conditioning test function. Calling it "the GW test" would be overclaiming. At P = 13 an HAC/Newey–West regression would be *less* honest, not more, which is why upgrading it is out of scope (`SCOPE.md` #12).

Residual, stated plainly: at a fixed horizon the 13 origins are **adjacent months**, so the origin blocks are not independent either. The block bootstrap fixes within-origin dependence, not between-origin serial dependence. **Report the MDE; do not try to fix it.**
