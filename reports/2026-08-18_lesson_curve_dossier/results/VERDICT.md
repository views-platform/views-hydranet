# UNDERPOWERED

the endpoints disagree (R inside its bound, F over its), or the rise does not clear 3 x MDE. 'No effect' and 'could not tell' are not distinguishable here.

## Notes (not blocking)

- 2 scheduled-sampling arm(s) (eps>0) scoped out — the lesson curve is the eps=0 axis: fullhalf_fortythree, fullhalf_fortytwo

Pre-registration: `05_analysis_plan.md` (LOCKED), rule md5 `5d6a256bb2b41485220d033cd0bfbc87`. Invariants run before the verdict; a null is declared only when the prediction bound excludes the pre-registered effect theta = 0.14.

⚠️ One seed per lesson point above 160. Per the standing rule a positive here is an escalation trigger, not a conclusion.

| arm | L | seed | source | C=AP h1 | F=AP h18 | O=oracle h18 | R=F/C | ±MDE(R) | FG-A |
|---|--:|--:|---|--:|--:|--:|--:|--:|---|
| `longzero_fortytwo` | 160 | 42 | 17_ss_retention | 0.4745 | 0.2569 | 0.4793 | 0.5415 | — | PASS |
| `longzero_fortythree` | 160 | 43 | 17_ss_retention | 0.4510 | 0.2834 | — | 0.6284 | — | PASS |
| `longzero_fortyfour` | 160 | 44 | 17_ss_retention | 0.4641 | 0.2683 | — | 0.5780 | — | PASS |
| `longzero_fortyfive` | 160 | 45 | 17_ss_retention | 0.4591 | 0.3052 | — | 0.6648 | — | PASS |
| `longzero_fortysix` | 160 | 46 | 17_ss_retention | 0.4648 | 0.2880 | — | 0.6196 | — | PASS |
| `longzero_fortyseven` | 160 | 47 | 18_lesson_curve | 0.4607 | 0.2605 | 0.4740 | 0.5654 | 0.0415 | PASS |
| `fullzero_fortytwo` | 300 | 42 | 18_lesson_curve | 0.4779 | 0.3298 | 0.4974 | 0.6902 | 0.0464 | PASS |
| `fullzero_fortythree` | 300 | 43 | 17_ss_retention | 0.4774 | 0.3318 | — | 0.6951 | 0.0453 | PASS |
| `fullzero_fortyfour` | 300 | 44 | 17_ss_retention | 0.4799 | 0.3058 | — | 0.6372 | 0.0503 | PASS |
| `sixhundredzero_fortytwo` | 600 | 42 | 18_lesson_curve | 0.4991 | 0.3452 | 0.5072 | 0.6916 | 0.0291 | PASS |

**Decomposition** `log F = log C + log R`, against L=160 seed 42:

| L | dlog F | from T=0 skill | from retention | the ceiling O(L) |
|--:|--:|--:|--:|--:|
| 300 | +0.1741 | +0.0114 | +0.1627 | not measured |
| 300 | +0.2559 | +0.0061 | +0.2498 | not measured |
| 300 | +0.2499 | +0.0072 | +0.2427 | 0.4974 |
| 600 | +0.2955 | +0.0507 | +0.2448 | 0.5072 |

### Ship battery (FAO-02) — does training move MAGNITUDE, or only occurrence?

FAO-02 selects on **CRPS** with a **strict conjunction** of QS99 / Brier / **MCR** guardrails, on the **validation** partition. These arms are scored on **calibration**, so nothing here is a ship decision — but `size_ratio` and `MCR` are the blocker, and they are free to read off arms already produced.

| arm | L | crps_all h18 | crps_events h18 | size_ratio h18 | MCR h18 | |MCR-1| |
|---|--:|--:|--:|--:|--:|--:|
| `longzero_fortytwo` | 160 | 0.13455 | 14.82 | 0.0000 | 0.00124 | 0.9988 |
| `longzero_fortythree` | 160 | 0.13438 | 14.80 | 0.0000 | 0.01375 | 0.9863 |
| `longzero_fortyfour` | 160 | 0.13321 | 14.61 | 0.0000 | 0.06178 | 0.9382 |
| `longzero_fortyfive` | 160 | 0.13452 | 14.82 | 0.0000 | 0.00635 | 0.9936 |
| `longzero_fortysix` | 160 | 0.13293 | 14.61 | 0.0000 | 0.04453 | 0.9555 |
| `longzero_fortyseven` | 160 | 0.13453 | 14.82 | 0.0000 | 0.00240 | 0.9976 |
| `fullzero_fortytwo` | 300 | 0.13412 | 14.77 | 0.0000 | 0.00718 | 0.9928 |
| `fullzero_fortythree` | 300 | 0.13408 | 14.77 | 0.0000 | 0.01135 | 0.9886 |
| `fullzero_fortyfour` | 300 | 0.13402 | 14.75 | 0.0000 | 0.01388 | 0.9861 |
| `sixhundredzero_fortytwo` | 600 | 0.13305 | 14.65 | 0.0000 | 0.02274 | 0.9773 |

**MCR = mean(prediction) / mean(truth); 1.0 is right-sized.** `size_ratio` is the median guess on cells that truly had conflict. A model can win `crps_all` on a 99%-zero field by emitting almost nothing — which is why FAO-02 makes the magnitude guardrail a hard conjunct and why Epic #263 called that pattern an ARTIFACT.

**sigma_seed at L=160** (n=6): retention 0.0458, AP@h18 0.0185

- mean R 0.5996 → prediction bound **0.6994** (mean + 2.176 x sigma, n=6)
- mean F 0.2771 → prediction bound **0.3173**
- k x sigma(R) = **0.0997** vs theta = 0.14  ← a null is declarable
- mean MDE_F over the anchor: 0.0195

