# Morning report — vehicle replication on `violet_visitor`

**VERDICT: GREEN**

Pre-registration: `05_analysis_plan.md` (LOCKED before the run). Falsifiers above are recorded before predictions P1-P4 are read.

## Run state

- **control in use: identity (tonight's code)**
- preserved 2026-08-12 cubes: scored (F1 reference; reproduced rescore.csv bit-for-bit)
- `identity` (tonight's code): scored
- `use_real`: done
- `spatial_scramble`: done
- `occurrence_real_magnitude_model`: done
- `occurrence_model_magnitude_real`: done
- `thin_0.75`: done

## F5 — support (`N`) identical across arms

pass — every scored row has N=170430

## F4 — h=1 identical across arms (step 1 has no feedback)

| arm | AP@h1 | dAP vs control |
|---|--:|--:|
| control | 0.474461375 | — |
| `use_real` | 0.474461375 | 0.00e+00 |
| `spatial_scramble` | 0.474461375 | 0.00e+00 |
| `occurrence_real_magnitude_model` | 0.474461375 | 0.00e+00 |
| `occurrence_model_magnitude_real` | 0.474461375 | 0.00e+00 |
| `thin_0.75` | 0.474461375 | 0.00e+00 |

pass — worst |dAP@h1| = 0.00e+00

## F6 — arm separation on the real field

| relation | expected | observed | |
|---|---|---|---|
| af(spatial_scramble) == af(use_real) | < 1e-6 | 0.00e+00 | ok |
| clustering destroyed | < 0.5 | 0.025 | ok |
| af(occ_real_mag_model) == af(use_real) | < 1e-6 | 0.00e+00 | ok |
| magnitudes swapped | > 5% | 71.4% | ok |
| af(thin:0.75) == 0.25 x af(use_real) | within 5% | 1.0% | ok |

## The comparison — gate AP, target sb

| h | control | `use_real` | `spatial_scramble` | `occurrence_real_magnitude_model` | `occurrence_model_magnitude_real` | `thin_0.75` |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.4745 | 0.4745 | 0.4745 | 0.4745 | 0.4745 | 0.4745 |
| 6 | 0.3924 | 0.4774 | 0.2110 | 0.4652 | 0.3926 | 0.4452 |
| 12 | 0.3226 | 0.4790 | 0.0945 | 0.4674 | 0.3141 | 0.4618 |
| 18 | 0.2569 | 0.4793 | 0.0486 | 0.4689 | 0.2600 | 0.4692 |
| 24 | 0.2060 | 0.4729 | 0.0300 | 0.4704 | 0.2031 | 0.4583 |
| 30 | 0.1699 | 0.4744 | 0.0230 | 0.4630 | 0.1631 | 0.4609 |
| 36 | 0.1370 | 0.4577 | 0.0188 | 0.4626 | 0.1288 | 0.4281 |

**Oracle-control gap at h18 = 0.2224** (oracle 0.4793, control 0.2569); on `truncated_smoke` it was 0.2938.

| component | share of the gap | truncated_smoke |
|---|--:|--:|
| `occurrence_real_magnitude_model` |  95.3% | 88.6% |
| `occurrence_model_magnitude_real` |   1.4% | 7.9% |
| `spatial_scramble` | -93.7% | 0.9% |

## Byproduct — `identity` (tonight) vs preserved 2026-08-12 cubes

Same artifact, same seed, same origins; different code. The gap below IS the
effect of d3a2626 / c07a352 / a2eabeb (per-site LockedDropout) on the
free-running path.

| h | identity (today) | preserved (08-12) | diff |
|--:|--:|--:|--:|
| 1 | 0.4745 | 0.4745 | +0.0000 |
| 6 | 0.3924 | 0.3924 | +0.0000 |
| 12 | 0.3226 | 0.3226 | +0.0000 |
| 18 | 0.2569 | 0.2569 | +0.0000 |
| 24 | 0.2060 | 0.2060 | +0.0000 |
| 30 | 0.1699 | 0.1699 | +0.0000 |
| 36 | 0.1370 | 0.1370 | +0.0000 |
