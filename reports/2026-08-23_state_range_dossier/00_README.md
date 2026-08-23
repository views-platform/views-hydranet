# Dossier — state range at the rollout origin (2026-08-23)

**Question.** Does the recurrent state the model must hold when free-running begins lie outside the
range of states it produces on training-distribution input?

**Answer: NO.** `f ≈ 0.003` against a chance rate of `0.02` — ~7× below chance, both seeds, both state
halves. See `07_experiment_log.md`; ledger **M43**; PR #298.

**Consequence.** The state is in-distribution at the origin and **decays ~36× over the forecast horizon**.
The failure is in the rollout's *dynamics*, not its starting point — and the out-of-range explanation for
**M38** (freezing the cell state, +0.039 AP@h18) is **excluded**, so M38 has no mechanism again.

## Documents

| file | what |
|---|---|
| `05_analysis_plan.md` | pre-registration — **LOCKED `cbb1e5b` with `tools/` empty**, + AMENDMENTS 1–5, each committed before the step it governs |
| `07_experiment_log.md` | the result, the decay trajectory, all five falsifiers, two wrong predictions scored |
| `tools/capture_regimes.py` | captures R1 / R2 / F3; drives its own forward passes so each state's regime is a function argument, not inferred from call order |
| `tools/state_range.py` | computes `f`, renders the §4 verdict |
| `tools/run_capture.sh` | runner — exists because `ModelPathManager` resolves from CWD, so every path must be absolute |
| `results/STATE_RANGE.json` | the verdict |
| `results/regimes_*.json` | per-patch audit trail |
| `results/state_decay_fortythree.csv` | the free-running decay trajectory |

Raw state tensors (`r1_state_*.pt`, `r2_state_*.pt`, ~38 MB) are **deliberately not committed** — they
are regenerable via `run_capture.sh`.

## Carried forward

* **`f` measures excursion, not degeneracy.** A state collapsed toward zero reads trivially "in range"
  of an interval spanning zero. Any follow-up needs a distributional distance.
* **The curriculum barely separates the training diet** — mean event density 4.548 / 5.985 / 3.966
  across thresholds 143 / 75 / 10: **+15% and non-monotone** over a nominal ratio range of 0.665 → 0.05.
* **C-308 second occurrence** — the probe inherited `predict()`'s `seq_len-1` origin fallback instead of
  the production origin 335, making a headline number 3× wrong while every falsifier stayed green.
* **The decay curve and the #290 damage curve do not line up** (damage is front-loaded; collapse is
  slowest early), so the collapse is not yet an explanation for the early skill loss.

## Status

**CLOSED** — question answered, negative, merged. Not promoted to an ADR: a refuted hypothesis has no
design to graduate.
