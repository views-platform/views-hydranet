# State-freeze at L=300 — auto-assembled

**Falsifiers pass:** h1 identical across arms (no feedback at step 1), and every `none` arm reproduces its published free-running value (M34).

## `fullzero_fortythree`

| arm | h1 | h6 | h18 | h36 | h18 vs none |
|---|--:|--:|--:|--:|--:|
| `none` | 0.4774 | 0.4071 | 0.3318 | 0.2287 | — |
| `hidden` | 0.4774 | 0.4109 | 0.3316 | 0.2379 | -0.0002 |
| `cell` | 0.4774 | 0.4300 | 0.3709 | 0.2891 | +0.0391 |
| `all` | 0.4774 | 0.4229 | 0.3743 | 0.2879 | +0.0424 |
| `cell@0.1` | 0.4774 | 0.4172 | 0.3643 | 0.2834 | +0.0325 |
| `cell@0.25` | 0.4774 | 0.4200 | 0.3678 | 0.2852 | +0.0360 |
| `cell@0.5` | 0.4774 | 0.4238 | 0.3716 | 0.2886 | +0.0398 |
| `cell@0.75` | 0.4774 | 0.4247 | 0.3731 | 0.2916 | +0.0413 |

## `fullzero_fortytwo`

| arm | h1 | h6 | h18 | h36 | h18 vs none |
|---|--:|--:|--:|--:|--:|
| `none` | 0.4779 | 0.4008 | 0.3298 | 0.2208 | — |
| `hidden` | 0.4779 | 0.4021 | 0.3209 | 0.2381 | -0.0090 |
| `cell` | 0.4779 | 0.4118 | 0.3622 | 0.2828 | +0.0323 |
| `all` | 0.4779 | 0.4142 | 0.3614 | 0.2788 | +0.0316 |

## Mean over seeds (h18)

| arm | mean | seeds |
|---|--:|---|
| `none` | 0.3308 | 0.3318, 0.3298 |
| `hidden` | 0.3262 | 0.3316, 0.3209 |
| `cell` | 0.3666 | 0.3709, 0.3622 |
| `all` | 0.3678 | 0.3743, 0.3614 |
| `cell@0.1` | 0.3643 | 0.3643 |
| `cell@0.25` | 0.3678 | 0.3678 |
| `cell@0.5` | 0.3716 | 0.3716 |
| `cell@0.75` | 0.3731 | 0.3731 |

*Auto-generated. Falsifiers only — no paired CI, no verdict. See `07_experiment_log.md`.*
