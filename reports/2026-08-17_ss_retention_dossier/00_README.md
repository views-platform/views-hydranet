# Scheduled sampling and rollout retention — CLOSED 2026-08-21

**Question.** The gate keeps its shape and loses its nerve in free-running rollout, and training does
not fix it (M26–M29). The model has never once seen its own output during training — every arm in this
programme is `ss_epsilon_max = 0.0`. Textbook exposure bias, textbook fix. Does it work?

**Answer: no. It makes the rollout worse, and the probe says why.**

| | |
|---|---|
| **direction** | ε=0.5 lowers AP@h18 by **0.0426** (0.3257 → 0.2831), retention by 0.053, **p = 0.0286** exact one-sided, 4v4 seeds, all four pairs down on both endpoints |
| **not a trade** | the §4 anchor guard **passes** — ΔAP(h1) −0.0277 against a 0.0440 limit |
| **formally** | **UNDERPOWERED** — significant, but 0.0426 < 3×MDE (0.0541). Direction established; magnitude not |
| **the twist** | SS largely **fixed** the zero collapse (`act_ratio` h18 9.4×, h36 28×) and AP fell at every horizon anyway |
| **where the damage is** | **56%** the model itself, **44%** the field it emits. The oracle ceiling drop (−0.0149) reproduces the residual (+0.0132) independently |
| **what is falsified** | "SS damaged the model's ability to use its input" — `thin:0.75` recovers 90% (ε=0) / **93%** (ε=0.5). It uses a good field *better* |

## Documents

| file | what it is |
|---|---|
| `05_analysis_plan.md` | pre-registration, **LOCKED** 2026-08-17. AMENDMENT 1 (L=160 → 300, forced by M29), AMENDMENT 2 (the §4 guard implemented) |
| `07_experiment_log.md` | **EXP-01** the 4v4 sweep · **EXP-02** the placement probe |
| `results/VERDICT.md` | rendered by `tools/verify_sweep.py`; regenerated after every arm |
| `results/PROBE.md` | the probe decomposition, rendered by `tools/probe_report.py` |
| `LAUNCH.md` | how the run was launched and what it cost |

## Tools

The decision rule is **`scripts/ss_sweep_gate.py`** — tracked, unit-tested (22 tests), md5
`d1432db9a7611cf349f1009225365027`. It lives in `scripts/` and not here because a tracked test may not
runtime-load the gitignored `reports/` tree; a rule that lived only in the dossier would be a rule with
no test in CI. `tools/verify_sweep.py` is thin I/O over it. The scheduler is the lesson curve's audited
`run_queue.sh`; this dossier's own `run_sweep.sh` was **deleted** — keeping two schedulers is what caused
the 2026-08-19 audit in the first place.

## Scope — what this does NOT settle

* **Not the roster.** Per §3.1, M20's roster separation stays confounded: those models trained with
  `ss_feedback='mean'`, which C-259 forbids. This answers the forward-looking question only.
* **One dose** (ε_max 0.5), **one vehicle**, **one training length** (300), target `sb`, h\*=18,
  calibration partition.
* **The probe is one seed** (42). The sweep's significance rests on four; the probe's decomposition
  does not.
* ε>0 arms are **not bit-reproducible** — SS training feedback uses the global RNG in production (M22).
  This adds variance to the treated side only, which makes the one-sided test *harder* to pass. The
  result is therefore conservative, not flattered.
* `spatial_scramble` inherits **C-291**'s confound.
