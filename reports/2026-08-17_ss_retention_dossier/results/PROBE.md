# Placement probe — where did scheduled sampling do its damage?

Inference-only, on two frozen artifacts differing in exactly one config key (`ss_epsilon_max`). Control and oracle already existed and were not re-run.

| model | control AP h18 | oracle (ceiling) | `occurrence_real_magnitude_model` | `spatial_scramble` | `thin_0.75` |
|---|--:|--:|--:|--:|--:|
| `fullzero_fortytwo` (eps=0.0) | 0.3298 | 0.4974 | 0.4888 | 0.0925 | 0.4807 |
| `fullhalf_fortytwo` (eps=0.5) | 0.3064 | 0.4825 | 0.4756 | 0.0953 | 0.4709 |

## Pre-registered predictions

- **P1** Handing both models perfect occurrence closes MOST of the gap between them (>60%) ⇒ SS's damage is in the occurrence field it emits
  - observed: gap +0.0235 → +0.0132, **44% closed** → **FAILS**
- **P2** If instead the gap SURVIVES perfect occurrence (<30% closed) ⇒ SS damaged the model's use of its input, not its placement
  - observed: 44% closed → **FAILS**
- **P3** `spatial_scramble` falls below BOTH controls — destroying placement is worse than either model's own output
  - observed: eps=0 -0.2374, eps=0.5 -0.2111 → **HOLDS**
- **P4** `thin:0.75` recovers ≥60% of each model's own gap — a quarter of the true events, correctly placed, is still enough (M4/M15)
  - observed: eps=0 90%, eps=0.5 93% → **HOLDS**

⚠️ **One seed, one vehicle, one dose, one target (`sb`), h\*=18.** `spatial_scramble` carries C-291's confound: destroying clustering also breaks alignment with the statics. The share statistic `(arm − control)/(oracle − control)` is meaningless for an arm that falls OUTSIDE that interval, which `spatial_scramble` does — quote its sign, never its share.

