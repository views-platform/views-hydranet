# IN PROGRESS (11/12)

| architecture | seed | control AP@h18 | arm AP@h18 | Δ | oracle | floor |
|---|--:|--:|--:|--:|--:|---|
| AntiAliasedPool | 42 | 0.3298 | 0.3214 | -0.0084 | 0.4991 | PASS |
| AntiAliasedPool | 43 | 0.3318 | 0.3326 | 0.0008 | 0.5017 | PASS |
| DynamicTopSkip | 42 | 0.3298 | 0.3309 | 0.0011 | 0.4993 | PASS |
| DynamicTopSkip | 43 | 0.3318 | 0.3350 | 0.0031 | 0.4944 | PASS |
| FiLMSkip | 42 | 0.3298 | 0.3141 | -0.0157 | 0.5004 | PASS |
| FiLMSkip | 43 | 0.3318 | 0.3400 | 0.0082 | 0.4990 | PASS |
| ShallowPool | 42 | 0.3298 | 0.3120 | -0.0178 | 0.4909 | PASS |
| ShallowPool | 43 | 0.3318 | 0.3169 | -0.0150 | 0.4991 | PASS |
| DualStream | 42 | 0.3298 | 0.3175 | -0.0124 | 0.4928 | PASS |
| DualStream | 43 | 0.3318 | — | — | — | — |
| WideMemory | 42 | 0.3298 | 0.3188 | -0.0111 | 0.5077 | PASS |
| WideMemory | 43 | 0.3318 | 0.3407 | 0.0089 | 0.5094 | PASS |

⚠️ Δ alone does not promote a candidate: the pre-registration requires the body metrics (`crps_all`, `size_ratio`, `mag_on_false_pos`) to be read beside it, and parameter counts reported — `ShallowPool` has 16% FEWER parameters, so a loss there may be capacity.
