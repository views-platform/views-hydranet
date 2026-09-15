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
| hidden | 18 | cont | +0.0084 | +0.0164 | -0.0004 | +0.0066 | +0.0078 | CONTESTED — signs split 3+/1- |
| hidden | 18 | onset | -0.0092 | +0.0006 | +0.0159 | -0.0021 | +0.0013 | CONTESTED — signs split 2+/2- |
| hidden | 36 | cont | +0.0069 | -0.0015 | +0.0061 | +0.0164 | +0.0070 | CONTESTED — signs split 3+/1- |
| hidden | 36 | onset | +0.0187 | +0.0130 | +0.0266 | +0.0146 | +0.0183 | SUPPORTED (4/4 positive) |
| cell | 18 | cont | +0.0464 | +0.0567 | +0.0530 | +0.0186 | +0.0437 | SUPPORTED (4/4 positive) |
| cell | 18 | onset | +0.0172 | +0.0239 | +0.0279 | +0.0205 | +0.0224 | SUPPORTED (4/4 positive) |
| cell | 36 | cont | +0.0745 | +0.0608 | +0.0517 | +0.0588 | +0.0614 | SUPPORTED (4/4 positive) |
| cell | 36 | onset | +0.0480 | +0.0496 | +0.0562 | +0.0299 | +0.0459 | SUPPORTED (4/4 positive) |
| all | 18 | cont | +0.0389 | +0.0594 | +0.0538 | +0.0197 | +0.0429 | SUPPORTED (4/4 positive) |
| all | 18 | onset | +0.0206 | +0.0289 | +0.0296 | +0.0185 | +0.0244 | SUPPORTED (4/4 positive) |
| all | 36 | cont | +0.0660 | +0.0590 | +0.0604 | +0.0557 | +0.0603 | SUPPORTED (4/4 positive) |
| all | 36 | onset | +0.0455 | +0.0485 | +0.0546 | +0.0289 | +0.0444 | SUPPORTED (4/4 positive) |

## C.2/C.3 — dispersion of predicted change, and direction skill

| arm | h | measure | fortytwo | fortythree | fortyfour | fortyfive | mean | verdict |
|---|---|---|---|---|---|---|---|
| hidden | 18 | dispersion | +0.1827 | +0.0361 | +0.0318 | -0.2177 | +0.0082 | CONTESTED — signs split 3+/1- |
| hidden | 18 | rho | +0.0011 | +0.0019 | +0.0068 | +0.0093 | +0.0048 | SUPPORTED (4/4 positive) |
| hidden | 36 | dispersion | +0.0629 | -0.1085 | -0.1162 | -0.3824 | -0.1360 | CONTESTED — signs split 1+/3- |
| hidden | 36 | rho | -0.0034 | -0.0023 | +0.0014 | +0.0032 | -0.0003 | CONTESTED — signs split 2+/2- |
| cell | 18 | dispersion | -0.4471 | -0.5050 | -0.4076 | -0.4654 | -0.4563 | SUPPORTED (0/4 positive) |
| cell | 18 | rho | +0.0132 | +0.0044 | +0.0238 | +0.0132 | +0.0137 | SUPPORTED (4/4 positive) |
| cell | 36 | dispersion | -0.6840 | -0.7422 | -0.6164 | -0.5946 | -0.6593 | SUPPORTED (0/4 positive) |
| cell | 36 | rho | -0.0036 | -0.0063 | -0.0044 | +0.0063 | -0.0020 | CONTESTED — signs split 1+/3- |
| all | 18 | dispersion | -0.4388 | -0.4976 | -0.4093 | -0.4770 | -0.4557 | SUPPORTED (0/4 positive) |
| all | 18 | rho | +0.0198 | +0.0102 | +0.0340 | +0.0143 | +0.0196 | SUPPORTED (4/4 positive) |
| all | 36 | dispersion | -0.6716 | -0.7615 | -0.6262 | -0.6095 | -0.6672 | SUPPORTED (0/4 positive) |
| all | 36 | rho | -0.0085 | -0.0030 | -0.0113 | +0.0116 | -0.0028 | CONTESTED — signs split 1+/3- |

