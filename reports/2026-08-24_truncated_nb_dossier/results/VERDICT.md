# EFFECT (NEGATIVE)

p=0.0143 (one-sided in the OBSERVED direction; floor 0.0143), mean ΔAP=-0.2376, |mean ΔAP| ≥ 3×MDE=0.0317, all four seeds agree in sign.

**4v4 — exact one-sided permutation floor `1/C(8,4)` = 0.0143.**

| seed | arm | control | truncated_nb | Δ AP@h18 | Δ AP@h1 | floor |
|--:|---|--:|--:|--:|--:|---|
| 42 | `truncfullzero_fortytwo` | 0.3298 | 0.1113 | -0.2186 | -0.0212 | PASS |
| 43 | `truncfullzero_fortythree` | 0.3318 | 0.0970 | -0.2348 | -0.0277 | PASS |
| 44 | `truncfullzero_fortyfour` | 0.3058 | 0.0497 | -0.2561 | -0.0273 | PASS |
| 45 | `truncfullzero_fortyfive` | 0.3352 | 0.0943 | -0.2408 | -0.0351 | PASS |

## Magnitude guardrails (§5) — reported, never summarised away

| seed | h | Δ crps_all | Δ size_ratio | Δ mag_on_false_pos | Δ n_false_pos |
|--:|--:|--:|--:|--:|--:|
| 42 | 1 | +0.0071 | +0.3601 | +5.3592 | +31.0000 |
| 42 | 6 | +0.0439 | +0.6969 | +20.2989 | +87.0000 |
| 42 | 12 | +0.1715 | +1.3125 | +43.7013 | +234.0000 |
| 42 | 18 | +0.3857 | +1.9013 | +51.9147 | +339.0000 |
| 42 | 24 | +0.7618 | +2.3188 | +61.8242 | +444.0000 |
| 42 | 30 | +1.2370 | +2.0054 | +70.8865 | +462.0000 |
| 42 | 36 | +1.7879 | +1.5313 | +79.0742 | +460.0000 |
| 43 | 1 | +0.0059 | +0.3040 | +5.4379 | +15.0000 |
| 43 | 6 | +0.0850 | +0.6689 | +44.9885 | +154.0000 |
| 43 | 12 | +0.3391 | +1.7483 | +91.0091 | +324.0000 |
| 43 | 18 | +1.0161 | +3.7723 | +109.1628 | +396.0000 |
| 43 | 24 | +2.6031 | +7.6992 | +102.4643 | +408.0000 |
| 43 | 30 | +5.0865 | +9.6875 | +103.2985 | +481.0000 |
| 43 | 36 | +7.7233 | +10.6607 | +100.2118 | +456.0000 |
| 44 | 1 | +0.0021 | +0.1709 | +3.5154 | +14.0000 |
| 44 | 6 | +0.0397 | +0.3125 | +20.5188 | +211.0000 |
| 44 | 12 | +0.1922 | +0.6875 | +36.9519 | +401.0000 |
| 44 | 18 | +0.4692 | +1.1250 | +33.8104 | +464.0000 |
| 44 | 24 | +0.8155 | +1.6875 | +35.7814 | +494.0000 |
| 44 | 30 | +1.1840 | +1.7812 | +34.1326 | +493.0000 |
| 44 | 36 | +1.5575 | +1.7955 | +33.5579 | +487.0000 |
| 45 | 1 | +0.0071 | +0.3643 | +6.1474 | +25.0000 |
| 45 | 6 | +0.0988 | +1.0212 | +34.0395 | +167.0000 |
| 45 | 12 | +0.4550 | +3.3393 | +63.9673 | +330.0000 |
| 45 | 18 | +1.3492 | +5.5250 | +63.9422 | +402.0000 |
| 45 | 24 | +3.0743 | +6.9931 | +66.1055 | +493.0000 |
| 45 | 30 | +5.6132 | +8.7500 | +70.9498 | +525.0000 |
| 45 | 36 | +8.3689 | +9.2695 | +71.5015 | +528.0000 |

⚠️ **A gain in AP accompanied by a regression in `crps_all` is a TRADE, not a win** (§5). The family's author named this risk himself: a truncated body gives the gate's false positives full magnitude, and `crps_all` is blind to it.

⚠️ **Registered false-negative mode (§7):** the gate is retrained alongside the body, so a NULL closes *'swapping the body fixes rollout skill'*, NOT *'the double-zero diagnosis was wrong'* — M44's decomposition stands on its own measurement either way.
