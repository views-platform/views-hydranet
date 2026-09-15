# Wave 1 — morning report

- run state: **IN PROGRESS**
- last heartbeat: 1788406612 2026-09-03 05:36:52
- phase: 2026-09-03 05:37:12 FINISHING

## Arms

| seed | arm | scored | origins | n_passes | AP@h18 |
|---|---|---|---|---|---|
| fullzero_fortytwo | none | yes | 13/13 | 4 | 0.329840 |
| fullzero_fortytwo | hidden | yes | 13/13 | 4 | 0.320877 |
| fullzero_fortytwo | cell | yes | 13/13 | 4 | 0.362189 |
| fullzero_fortytwo | all | yes | 13/13 | 4 | 0.361445 |
| fullzero_fortythree | none | yes | 13/13 | 4 | 0.331825 |
| fullzero_fortythree | hidden | yes | 13/13 | 4 | 0.331593 |
| fullzero_fortythree | cell | yes | 13/13 | 4 | 0.370916 |
| fullzero_fortythree | all | yes | 13/13 | 4 | 0.374253 |
| fullzero_fortyfour | none | yes | 13/13 | 4 | 0.305766 |
| fullzero_fortyfour | hidden | yes | 13/13 | 4 | 0.311400 |
| fullzero_fortyfour | cell | yes | 13/13 | 4 | 0.351768 |
| fullzero_fortyfour | all | yes | 13/13 | 4 | 0.353009 |
| fullzero_fortyfive | none | yes | 13/13 | 4 | 0.335170 |
| fullzero_fortyfive | hidden | yes | 13/13 | 4 | 0.338397 |
| fullzero_fortyfive | cell | yes | 13/13 | 4 | 0.364371 |
| fullzero_fortyfive | all | yes | 13/13 | 4 | 0.363836 |

## Reproduction falsifier (the cube path was NOT changed)

- fullzero_fortytwo/identity AP@h18 = 0.3298395823400329 vs archived 0.3298395823400329 — **EXACT**
- fullzero_fortytwo/identity_freezecell AP@h18 = 0.3621885544392029 vs archived 0.3621885544392029 — **EXACT**

## Gate and body, by horizon (seed-42 first)

### fullzero_fortytwo

| h | AP none | AP hidden | AP cell | AP all | sizeR none | sizeR hidden | sizeR cell | sizeR all |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4779 | 0.4779 | 0.4779 | 0.4779 | 0.0462 | 0.0462 | 0.0462 | 0.0462 |
| 6 | 0.4008 | 0.4021 | 0.4118 | 0.4142 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | 0.3770 | 0.3699 | 0.3944 | 0.3945 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 18 | 0.3298 | 0.3209 | 0.3622 | 0.3614 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | 0.2967 | 0.2907 | 0.3350 | 0.3343 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 30 | 0.2631 | 0.2673 | 0.3142 | 0.3162 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 36 | 0.2208 | 0.2381 | 0.2828 | 0.2788 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### fullzero_fortythree

| h | AP none | AP hidden | AP cell | AP all | sizeR none | sizeR hidden | sizeR cell | sizeR all |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4774 | 0.4774 | 0.4774 | 0.4774 | 0.0437 | 0.0437 | 0.0437 | 0.0437 |
| 6 | 0.4071 | 0.4109 | 0.4300 | 0.4229 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | 0.3596 | 0.3634 | 0.3907 | 0.3911 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 18 | 0.3318 | 0.3316 | 0.3709 | 0.3743 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | 0.3027 | 0.2966 | 0.3425 | 0.3420 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 30 | 0.2748 | 0.2668 | 0.3194 | 0.3168 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 36 | 0.2287 | 0.2379 | 0.2891 | 0.2879 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### fullzero_fortyfour

| h | AP none | AP hidden | AP cell | AP all | sizeR none | sizeR hidden | sizeR cell | sizeR all |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4799 | 0.4799 | 0.4799 | 0.4799 | 0.0848 | 0.0848 | 0.0848 | 0.0848 |
| 6 | 0.3786 | 0.3811 | 0.3955 | 0.3958 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | 0.3312 | 0.3334 | 0.3631 | 0.3653 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 18 | 0.3058 | 0.3114 | 0.3518 | 0.3530 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | 0.2881 | 0.2922 | 0.3263 | 0.3283 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 30 | 0.2528 | 0.2620 | 0.3045 | 0.3108 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 36 | 0.2108 | 0.2339 | 0.2755 | 0.2769 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### fullzero_fortyfive

| h | AP none | AP hidden | AP cell | AP all | sizeR none | sizeR hidden | sizeR cell | sizeR all |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.4716 | 0.4716 | 0.4716 | 0.4716 | 0.0469 | 0.0469 | 0.0469 | 0.0469 |
| 6 | 0.3979 | 0.3989 | 0.4099 | 0.4100 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 12 | 0.3583 | 0.3617 | 0.3868 | 0.3858 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 18 | 0.3352 | 0.3384 | 0.3644 | 0.3638 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 24 | 0.3098 | 0.3223 | 0.3467 | 0.3461 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 30 | 0.2718 | 0.2975 | 0.3160 | 0.3173 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 36 | 0.2396 | 0.2645 | 0.2889 | 0.2876 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## AMBER

- ANOMALIES.txt is non-empty:
-   2026-09-03 03:26:47 repo.head re-baselined by the operator: commits since 0a76a6a touch only dossier tools and tests, not the inference path
-   2026-09-03 03:33:42 repo.head re-baselined after committing escalation.py (dossier tools only)
-   2026-09-03 03:36:50 repo.head re-baselined after assemble.py commit
-   2026-09-03 03:39:30 repo.head re-baselined after dossier scaffold commit (docs only)
-   2026-09-03 03:40:10 repo.head re-baselined after D.2 harness note (docs only)
