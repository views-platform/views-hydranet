# ITF FAILS TOO

both seeds <= control - 1σ — direction is not the answer

⚠️ **This is a 2v2 screen, not a significance test.** The exact one-sided permutation floor at 2v2 is 0.167; a p-value here would be theatre.

σ (control seed sd @h18, n=4) = **0.0134**

| seed | arm | control | ITF | Δ | σ | floor |
|--:|---|--:|--:|--:|--:|---|
| 42 | `itffullhalf_fortytwo` | 0.3298 | 0.3125 | -0.0174 | -1.30 | PASS |
| 43 | `itffullhalf_fortythree` | 0.3318 | 0.3105 | -0.0213 | -1.59 | PASS |

⚠️ Per §7 (C-307): ε starts at **0.5, not 1.0** — a softened ITF. A null cannot distinguish *'ITF fails'* from *'we did not run real ITF'*. Reopen triggers are in §7.

