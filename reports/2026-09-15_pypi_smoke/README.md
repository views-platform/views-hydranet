# PyPI smoke — 2026-09-15/16: the consumer path, end to end, from the wheel

**Question.** Does `views-hydranet` **0.1.0 from PyPI** — not the checkout — train and evaluate a
roster model the way the checkout did?

**Method.** A fresh env built the way a model's `run.sh` builds it (`pip install -r
requirements.txt`, which resolves `views-hydranet~=0.1.0` from pypi.org), then
`main.py --run_type validation --train --evaluate --saved` for four models, one after another,
on the RTX 4070. `run_smoke.sh` is the driver; `SMOKE.log` the record; `log_<model>.txt` each run.

**Answer: yes, bit for bit.** All four retrained artifacts are **byte-identical** to the ones
trained from the checkout on 2026-09-07/08 — every tensor, BatchNorm buffers included — and the
frozen scorer gives identical AP / Brier / CRPS on the new cubes.

| model | train | lessons/h | eval | artifact vs 09-07/08 |
|---|---|---|---|---|
| bold_comet | 3 h 23 m | 89 | 24 min | byte-identical |
| purple_alien | 4 h 56 m | 76 | 17 min | byte-identical |
| heavy_freighter | 3 h 01 m | 99 | 17 min | byte-identical |
| violet_visitor | 4 h 53 m | 77 | 20 min | byte-identical |

**Two things found on the way.**

1. **The first pass trained on CPU** (`log_violet_visitor_CPU.txt`, 6 h 46 m). The fresh env
   resolved torch 2.14+cu130, which the 535 driver cannot run; torch fell back silently and the
   only trace was a DEBUG line. `views-hydranet#377`. torch 2.6.0+cu124 was installed by hand
   before the GPU pass.
2. **`views-models/envs/views-hydranet` is a symlink to the dev env** on this machine, so the
   models' `run.sh` never installs from PyPI here — and the first attempt of this smoke pip-upgraded
   the dev env through it (restored; the suite was green afterwards).

**What changed in the forecasts, and why it is not a regression.** bold_comet's new cube fires on
~5× more cells than the 09-08 dossier cube (act_ratio 0.07 → 0.42 at h18) with the gate unchanged.
Cause: views-models commit `4f536fdb` (2026-09-15 10:35) set `gate_threshold` 0.5 → 0.14 / 0.16 /
0.20 on the three hard-gate models, applying M75 as priors for the sweep in views-models#466. The
model is the same; the cut is lower. No bloom: cells fired / observed is still 0.42 (< 1), and the
quiet-cell cost is flat across horizons.
