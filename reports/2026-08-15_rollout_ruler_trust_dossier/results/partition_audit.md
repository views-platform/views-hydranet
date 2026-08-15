# Partition & provenance audit

Epic #263 / S2 (#266). Run type: `calibration`. 6 arm(s).

| arm | train | origins | emitted | leak | rollout_feedback | S | truth pin |
|---|---|--:|---|:--:|---|--:|:--:|
| `violet_visitor` | 121–456 | 13 | 457–504 | no | `sample` | 16 | ok |
| `bright_starship` | 121–456 | 13 | 457–504 | no | `sample` | 16 | ok |
| `blazing_meteor` | 121–456 | 13 | 457–504 | no | `sample` | 16 | ok |
| `pink_pirate` | 121–456 | 13 | 457–504 | no | `sample` | 16 | ok |
| `blue_stranger` | 121–456 | 13 | 457–504 | no | `sample` | 16 | ok |
| `purple_alien` | 121–456 | 13 | 457–504 | no | `sample` | 16 | ok |

**C-217** no origin emits at or before the train window's end. **C-218** every scored arm feeds back samples, not the mean. **C-220** every cube is 2-D with S > 1. **Giacomini fixed-scheme** each arm resolves to one artifact across all origins.
