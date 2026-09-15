# Wave 1 — cross-seed findings

Decision rule: 4/4 sign agreement AND |mean| > seed sd. No p-values are claimed — a
paired sign-flip at n=4 floors at 1/16 = 0.0625 and cannot reach 0.05.

Seeds with all four arms complete: **4/4** — fullzero_fortytwo, fullzero_fortythree, fullzero_fortyfour, fullzero_fortyfive

## AP (higher is better) — each freeze arm minus `none`

| arm | h | fortytwo | fortythree | fortyfour | fortyfive | mean | sd | verdict |
|---|---|---|---|---|---|---|---|---|
| hidden | 18 | -0.0090 | -0.0002 | +0.0056 | +0.0032 | -0.0001 | 0.0064 | CONTESTED — signs split 2+/2- |
| hidden | 36 | +0.0173 | +0.0092 | +0.0231 | +0.0249 | +0.0186 | 0.0070 | SUPPORTED (4/4 positive) |
| cell | 18 | +0.0323 | +0.0391 | +0.0460 | +0.0292 | +0.0367 | 0.0075 | SUPPORTED (4/4 positive) |
| cell | 36 | +0.0620 | +0.0604 | +0.0646 | +0.0493 | +0.0591 | 0.0067 | SUPPORTED (4/4 positive) |
| all | 18 | +0.0316 | +0.0424 | +0.0472 | +0.0287 | +0.0375 | 0.0088 | SUPPORTED (4/4 positive) |
| all | 36 | +0.0580 | +0.0592 | +0.0660 | +0.0480 | +0.0578 | 0.0074 | SUPPORTED (4/4 positive) |

## Brier (lower is better) — each freeze arm minus `none`

| arm | h | fortytwo | fortythree | fortyfour | fortyfive | mean | sd | verdict |
|---|---|---|---|---|---|---|---|---|
| hidden | 18 | -0.0004 | -0.0004 | -0.0004 | -0.0008 | -0.0005 | 0.0002 | SUPPORTED (0/4 positive) |
| hidden | 36 | -0.0007 | -0.0008 | -0.0009 | -0.0011 | -0.0009 | 0.0002 | SUPPORTED (0/4 positive) |
| cell | 18 | -0.0015 | -0.0014 | -0.0013 | -0.0015 | -0.0014 | 0.0001 | SUPPORTED (0/4 positive) |
| cell | 36 | -0.0017 | -0.0017 | -0.0016 | -0.0017 | -0.0017 | 0.0000 | SUPPORTED (0/4 positive) |
| all | 18 | -0.0015 | -0.0015 | -0.0013 | -0.0015 | -0.0015 | 0.0001 | SUPPORTED (0/4 positive) |
| all | 36 | -0.0016 | -0.0017 | -0.0017 | -0.0017 | -0.0017 | 0.0000 | SUPPORTED (0/4 positive) |

## crps_events (lower is better) — each freeze arm minus `none`

| arm | h | fortytwo | fortythree | fortyfour | fortyfive | mean | sd | verdict |
|---|---|---|---|---|---|---|---|---|
| hidden | 18 | -0.0766 | -0.0441 | -0.0647 | -0.1713 | -0.0892 | 0.0563 | SUPPORTED (0/4 positive) |
| hidden | 36 | -0.0209 | -0.0252 | -0.0872 | -0.1263 | -0.0649 | 0.0509 | SUPPORTED (0/4 positive) |
| cell | 18 | -0.9873 | -0.9072 | -0.9583 | -0.6119 | -0.8662 | 0.1727 | SUPPORTED (0/4 positive) |
| cell | 36 | -0.5213 | -0.4637 | -0.5584 | -0.4094 | -0.4882 | 0.0654 | SUPPORTED (0/4 positive) |
| all | 18 | -1.0742 | -0.8513 | -0.9692 | -0.6383 | -0.8832 | 0.1870 | SUPPORTED (0/4 positive) |
| all | 36 | -0.4929 | -0.4709 | -0.5856 | -0.3968 | -0.4865 | 0.0778 | SUPPORTED (0/4 positive) |

## size_ratio (higher is better) — each freeze arm minus `none`

| arm | h | fortytwo | fortythree | fortyfour | fortyfive | mean | sd | verdict |
|---|---|---|---|---|---|---|---|---|
| hidden | 18 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | NO EFFECT (all 4 deltas exactly 0) |
| hidden | 36 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | NO EFFECT (all 4 deltas exactly 0) |
| cell | 18 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | NO EFFECT (all 4 deltas exactly 0) |
| cell | 36 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | NO EFFECT (all 4 deltas exactly 0) |
| all | 18 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | NO EFFECT (all 4 deltas exactly 0) |
| all | 36 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | NO EFFECT (all 4 deltas exactly 0) |

## C.4 — onset vs continuation AP, freeze arm minus `none`

| arm | h | universe | fortytwo | fortythree | fortyfour | fortyfive | mean | verdict |
|---|---|---|---|---|---|---|---|
| hidden | 18 | cont | -0.0107 | +0.0128 | +0.0108 | -0.0056 | +0.0018 | CONTESTED — signs split 2+/2- |
| hidden | 18 | onset | +0.0313 | +0.0214 | +0.0229 | +0.0144 | +0.0225 | SUPPORTED (4/4 positive) |
| hidden | 36 | cont | +0.0471 | +0.0948 | +0.1453 | +0.0392 | +0.0816 | SUPPORTED (4/4 positive) |
| hidden | 36 | onset | +0.0412 | +0.0425 | +0.0561 | +0.0247 | +0.0411 | SUPPORTED (4/4 positive) |
| cell | 18 | cont | +0.0738 | +0.0971 | +0.0097 | +0.0213 | +0.0505 | SUPPORTED (4/4 positive) |
| cell | 18 | onset | +0.0366 | +0.0516 | +0.0300 | +0.0229 | +0.0353 | SUPPORTED (4/4 positive) |
| cell | 36 | cont | +0.1655 | +0.1996 | +0.1866 | +0.0812 | +0.1582 | SUPPORTED (4/4 positive) |
| cell | 36 | onset | +0.0519 | +0.0680 | +0.0617 | +0.0500 | +0.0579 | SUPPORTED (4/4 positive) |
| all | 18 | cont | +0.0803 | +0.1026 | +0.0140 | +0.0197 | +0.0542 | SUPPORTED (4/4 positive) |
| all | 18 | onset | +0.0330 | +0.0532 | +0.0359 | +0.0256 | +0.0369 | SUPPORTED (4/4 positive) |
| all | 36 | cont | +0.1391 | +0.2093 | +0.2079 | +0.0871 | +0.1608 | SUPPORTED (4/4 positive) |
| all | 36 | onset | +0.0465 | +0.0683 | +0.0623 | +0.0493 | +0.0566 | SUPPORTED (4/4 positive) |

## C.2/C.3 — dispersion of predicted change, and direction skill

| arm | h | measure | fortytwo | fortythree | fortyfour | fortyfive | mean | verdict |
|---|---|---|---|---|---|---|---|
| hidden | 18 | dispersion | -0.3934 | -0.1985 | -0.3689 | -0.3797 | -0.3351 | SUPPORTED (0/4 positive) |
| hidden | 18 | rho | +0.0047 | +0.0042 | +0.0094 | +0.0239 | +0.0105 | SUPPORTED (4/4 positive) |
| hidden | 36 | dispersion | -0.5771 | -0.2172 | -0.5924 | -0.6606 | -0.5118 | SUPPORTED (0/4 positive) |
| hidden | 36 | rho | -0.0026 | +0.0014 | +0.0173 | +0.0189 | +0.0087 | CONTESTED — signs split 3+/1- |
| cell | 18 | dispersion | -0.6546 | -0.8802 | -0.5547 | -0.6867 | -0.6940 | SUPPORTED (0/4 positive) |
| cell | 18 | rho | +0.0297 | +0.0330 | +0.0335 | +0.0528 | +0.0372 | SUPPORTED (4/4 positive) |
| cell | 36 | dispersion | -0.8354 | -0.9238 | -0.7881 | -0.9418 | -0.8723 | SUPPORTED (0/4 positive) |
| cell | 36 | rho | +0.0443 | +0.0265 | +0.0856 | +0.0826 | +0.0598 | SUPPORTED (4/4 positive) |
| all | 18 | dispersion | -0.6747 | -0.8814 | -0.5713 | -0.7130 | -0.7101 | SUPPORTED (0/4 positive) |
| all | 18 | rho | +0.0475 | +0.0289 | +0.0501 | +0.0584 | +0.0462 | SUPPORTED (4/4 positive) |
| all | 36 | dispersion | -0.8546 | -0.9147 | -0.8113 | -0.9739 | -0.8886 | SUPPORTED (0/4 positive) |
| all | 36 | rho | +0.0265 | +0.0453 | +0.0892 | +0.0477 | +0.0522 | SUPPORTED (4/4 positive) |

