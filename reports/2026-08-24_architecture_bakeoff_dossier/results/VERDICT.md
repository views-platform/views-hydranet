# IN PROGRESS (0/12)

| architecture | seed | control AP@h18 | arm AP@h18 | Δ | oracle | floor |
|---|--:|--:|--:|--:|--:|---|
| AntiAliasedPool | 42 | 0.3298 | — | — | — | — |
| AntiAliasedPool | 43 | 0.3318 | — | — | — | — |
| DynamicTopSkip | 42 | 0.3298 | — | — | — | — |
| DynamicTopSkip | 43 | 0.3318 | — | — | — | — |
| FiLMSkip | 42 | 0.3298 | — | — | — | — |
| FiLMSkip | 43 | 0.3318 | — | — | — | — |
| ShallowPool | 42 | 0.3298 | — | — | — | — |
| ShallowPool | 43 | 0.3318 | — | — | — | — |
| DualStream | 42 | 0.3298 | — | — | — | — |
| DualStream | 43 | 0.3318 | — | — | — | — |
| WideMemory | 42 | 0.3298 | — | — | — | — |
| WideMemory | 43 | 0.3318 | — | — | — | — |

⚠️ Δ alone does not promote a candidate: the pre-registration requires the body metrics (`crps_all`, `size_ratio`, `mag_on_false_pos`) to be read beside it, and parameter counts reported — `ShallowPool` has 16% FEWER parameters, so a loss there may be capacity.
