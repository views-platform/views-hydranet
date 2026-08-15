# Minimum detectable effect at P = 13

Epic #263 / S4 (#268). Pair: `violet_visitor` − `climatology`, target `sb`, h = 36. 13 origins, N = 170,430 cell-months. 2000 bootstrap replicates, 90% CI, seed 0.

| quantity | value |
|---|---:|
| observed mean CRPS differential | -0.0874845 |
| origin-block CI | [-0.0900609, -0.0844634] |
| **MDE (origin-block, half-width)** | **0.00279874** |
| iid-over-cells CI (for contrast only) | [-0.100316, -0.0754892] |
| iid half-width | 0.0124133 |
| **origin-block ÷ iid width** | **0.23×** |
| separable from 0 at 90%? | YES |

## Per-origin mean differential

| origin (m0) | mean Δcrps |
|---:|---:|
| 457 | -0.0880799 |
| 458 | -0.0721519 |
| 459 | -0.0855434 |
| 460 | -0.0776534 |
| 461 | -0.0884774 |
| 462 | -0.0878556 |
| 463 | -0.0886303 |
| 464 | -0.0923607 |
| 465 | -0.0881941 |
| 466 | -0.0945084 |
| 467 | -0.0868592 |
| 468 | -0.0930043 |
| 469 | -0.0939792 |

## Reading this

An effect smaller than **0.00279874** cannot be distinguished from zero at 90% with 13 origins, however many cells there are. A null on this pair therefore means *either* no difference *or* no power — the distinction C-254 exists to preserve.

The iid-over-cells bootstrap gives a CI **4.44× WIDER** here, not narrower. That is the opposite of the direction C-221/C-253 anticipate, and it is worth stating plainly rather than asserting the expected sign: the per-cell CRPS differential on this pair is dominated by a handful of heavy-tail conflict cells, so resampling 170k individual cells shakes those extremes in and out and inflates the spread, while the 13 per-origin means are comparatively stable. Which bootstrap is wider depends on the balance between within-origin correlation (favours origin-block being wider, the C-253 synthetic case) and tail heaviness (favours iid being wider, this case).

**Neither reading licenses using the iid bootstrap.** It is reported as a contrast only. The origin block is the correct unit because the 36-month futures of adjacent origins overlap; that is a structural fact about the design, not an empirical result about which interval happens to be wider on one pair.

## Why this is not called a Giacomini–White test

The models are trained **once** and scored at 13 rolling origins, so the forecasting scheme is **fixed**, satisfying Giacomini & White 2006 §3.2 Comment 3 (*"Expanding window forecasting schemes are ruled out by assumption"*). That holds **by construction**: one prediction directory is the output of one training run, so its 13 origins necessarily share an artifact — nothing checks them pairwise, because there is nothing that could differ. What `partition_audit.py` adds is that the artifact's sha256 is resolved and recorded, and now fails loud when it cannot be, so the claim is auditable after the fact rather than merely asserted.

But `gw_stratified` is a **bootstrap on the mean loss differential**, not a GW conditional-predictive-ability regression: there is no analytic variance estimator and no conditioning test function. Calling it "the GW test" would be overclaiming. At P = 13 an HAC/Newey–West regression would be *less* honest, not more, which is why upgrading it is out of scope (`SCOPE.md` #12).

Residual, stated plainly: at a fixed horizon the 13 origins are **adjacent months**, so the origin blocks are not independent either. The block bootstrap fixes within-origin dependence, not between-origin serial dependence. **Report the MDE; do not try to fix it.**
