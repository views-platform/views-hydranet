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
| hidden | 18 | cont | +0.0918 | -0.0361 | -0.0286 | +0.1143 | +0.0353 | CONTESTED — signs split 2+/2- |
| hidden | 18 | onset | +0.0100 | -0.0146 | +0.0088 | +0.0132 | +0.0043 | CONTESTED — signs split 3+/1- |
| hidden | 36 | cont | +0.0740 | +0.0060 | +0.0318 | +0.1223 | +0.0585 | SUPPORTED (4/4 positive) |
| hidden | 36 | onset | +0.0220 | +0.0068 | +0.0165 | +0.0361 | +0.0203 | SUPPORTED (4/4 positive) |
| cell | 18 | cont | +0.2571 | +0.1277 | +0.1156 | +0.1795 | +0.1700 | SUPPORTED (4/4 positive) |
| cell | 18 | onset | +0.0387 | +0.0280 | +0.0544 | +0.0480 | +0.0423 | SUPPORTED (4/4 positive) |
| cell | 36 | cont | +0.1826 | +0.1179 | +0.1576 | +0.1024 | +0.1401 | SUPPORTED (4/4 positive) |
| cell | 36 | onset | +0.0415 | +0.0236 | +0.0388 | +0.0388 | +0.0357 | SUPPORTED (4/4 positive) |
| all | 18 | cont | +0.2380 | +0.1334 | +0.0778 | +0.1971 | +0.1616 | SUPPORTED (4/4 positive) |
| all | 18 | onset | +0.0384 | +0.0278 | +0.0624 | +0.0483 | +0.0442 | SUPPORTED (4/4 positive) |
| all | 36 | cont | +0.2024 | +0.1022 | +0.1716 | +0.0993 | +0.1439 | SUPPORTED (4/4 positive) |
| all | 36 | onset | +0.0402 | +0.0244 | +0.0409 | +0.0381 | +0.0359 | SUPPORTED (4/4 positive) |

## C.2/C.3 — dispersion of predicted change, and direction skill

| arm | h | measure | fortytwo | fortythree | fortyfour | fortyfive | mean | verdict |
|---|---|---|---|---|---|---|---|
| hidden | 18 | dispersion | -0.3377 | -0.1411 | -0.2520 | -0.3920 | -0.2807 | SUPPORTED (0/4 positive) |
| hidden | 18 | rho | +0.0081 | +0.0039 | +0.0089 | +0.0104 | +0.0078 | SUPPORTED (4/4 positive) |
| hidden | 36 | dispersion | -0.4055 | -0.1431 | -0.3468 | -0.4772 | -0.3432 | SUPPORTED (0/4 positive) |
| hidden | 36 | rho | +0.0108 | -0.0004 | +0.0047 | +0.0123 | +0.0069 | CONTESTED — signs split 3+/1- |
| cell | 18 | dispersion | -0.8639 | -0.6307 | -0.9769 | -0.8173 | -0.8222 | SUPPORTED (0/4 positive) |
| cell | 18 | rho | +0.0080 | +0.0171 | +0.0505 | +0.0100 | +0.0214 | SUPPORTED (4/4 positive) |
| cell | 36 | dispersion | -0.9547 | -0.6830 | -1.0950 | -0.8744 | -0.9018 | SUPPORTED (0/4 positive) |
| cell | 36 | rho | +0.0091 | +0.0010 | +0.0252 | +0.0353 | +0.0177 | SUPPORTED (4/4 positive) |
| all | 18 | dispersion | -0.8758 | -0.6175 | -0.9594 | -0.8421 | -0.8237 | SUPPORTED (0/4 positive) |
| all | 18 | rho | +0.0200 | +0.0247 | +0.0060 | +0.0326 | +0.0208 | SUPPORTED (4/4 positive) |
| all | 36 | dispersion | -0.9735 | -0.6893 | -1.1308 | -0.8817 | -0.9188 | SUPPORTED (0/4 positive) |
| all | 36 | rho | +0.0155 | +0.0198 | +0.0009 | +0.0276 | +0.0159 | SUPPORTED (4/4 positive) |

