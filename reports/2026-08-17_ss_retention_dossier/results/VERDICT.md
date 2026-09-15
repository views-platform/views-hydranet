# UNDERPOWERED

p=0.0286 IS significant, but the drop of 0.0426 does not clear 3.0 x MDE = 0.0541. The direction is established; the MAGNITUDE is inside the measurement resolution this design can assert. 'No effect' and 'could not tell' are not distinguishable here.

## Notes (not blocking)

- 5 arm(s) at another lesson count ignored — this sweep is L=300: longzero_fortyfive, longzero_fortyfour, longzero_fortysix, longzero_fortythree, longzero_fortytwo

Pre-registration: `05_analysis_plan.md` (LOCKED, AMENDMENT 1 → L=300), rule md5 `d1432db9a7611cf349f1009225365027`. Falsifiers run before the verdict. Direction is pre-registered: **SS lowers AP@h18**, one-sided, alpha=0.05.

| arm | eps | seed | L | AP h1 | AP h18 | retention | size_ratio | src |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| `fullzero_fortytwo` | 0.0 | 42 | 300 | 0.4779 | 0.3298 | 0.6902 | 0.0000 | curve |
| `fullzero_fortythree` | 0.0 | 43 | 300 | 0.4774 | 0.3318 | 0.6951 | 0.0000 | sweep |
| `fullzero_fortyfour` | 0.0 | 44 | 300 | 0.4799 | 0.3058 | 0.6372 | 0.0000 | sweep |
| `fullzero_fortyfive` | 0.0 | 45 | 300 | 0.4716 | 0.3352 | 0.7107 | 0.0000 | sweep |
| `fullhalf_fortytwo` | 0.5 | 42 | 300 | 0.4502 | 0.3064 | 0.6805 | 0.0000 | sweep |
| `fullhalf_fortythree` | 0.5 | 43 | 300 | 0.4435 | 0.2687 | 0.6060 | 0.0000 | sweep |
| `fullhalf_fortyfour` | 0.5 | 44 | 300 | 0.4496 | 0.2528 | 0.5622 | 0.0000 | sweep |
| `fullhalf_fortyfive` | 0.5 | 45 | 300 | 0.4529 | 0.3044 | 0.6723 | 0.0000 | sweep |

**4 control vs 4 treated.**

- mean AP@h18: control 0.3257 → treated 0.2831  (**-0.0426**)
- mean retention difference: -0.0530  (endpoints agree: True)
- exact one-sided permutation **p = 0.0286**
- mean MDE_AP(h18) = 0.0180
- **anchor guard**: mean dAP(h1) = -0.0277 against 3.0 x MDE_AP(h1) = 0.0440 → OK

⚠️ Per §3.1 this cannot settle what the roster showed — those models trained with `ss_feedback='mean'`, which C-259 forbids. A null here answers the forward-looking question only. A NULL requires the interval to exclude theta = 30% of the control mean.

