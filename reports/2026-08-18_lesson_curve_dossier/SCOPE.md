# Scope — what this dossier does NOT establish

Written before the run, so nothing here is a retrofit.

## Exclusions

1. **One seed per lesson point.** Seeds 42–45 exist only at L=160, to measure σ_seed. Every point at
   300/600/900 is seed 42 alone. A positive result is an **escalation trigger**, not a conclusion
   (`RESULTS_LEDGER.md` §Standing rule adopted 2026-08-17).

2. **σ_seed is measured at 160 and assumed comparable above it.** If training variance grows with
   length — which nothing here tests — the prediction bound is too narrow and RISING is too easy.

3. **Lesson count and curriculum cooling are confounded by construction.** `curriculum.py:85` divides the
   difficulty slope by `total_lessons × windows_per_lesson`, so a longer run gets a *stretched* schedule,
   not a *continued* one. Every statement here reads "setting `total_lessons` higher", **never** "more
   gradient steps". Two runs at the same L are unaffected; only the across-L comparison carries it.

4. **The 40-lesson point is excluded from the ladder, not merely uninteresting.** Its control fails FG-A
   (2.16× chance) *and* it is the one L at which the BN-recalibration windows genuinely differ (its
   `get_intensity_ratio` at steps 0–29 is not roof-clipped, unlike every L ≥ 160). It appears in the
   framing table as the reason to ask the question, and is used for no threshold and no comparison.

5. **One target (`sb`), one primary horizon (h\* = 18), one partition (calibration), 13 origins, S = 16.**
   `ns`/`os` are not scored. Nothing here is a validation-partition result.

6. **One configuration.** `nb` body, `soft_gate` composition, `violet_visitor`'s hyperparameters. A
   plateau here is a plateau for *this* vehicle; the 4/4 multivehicle replication of I-A does not transfer
   to this claim and is not evidence for it.

7. **The oracle is `use_real` — perfect occurrence AND magnitude.** It bounds the feedback path, not the
   model. "The ceiling" in this dossier means that and nothing more; it is not an estimate of what is
   learnable from the data.

8. **Cost figures above 160 lessons are extrapolations from two points.** Measured per-lesson cost rose
   22% from 40L to 160L; two points cannot separate a fixed overhead from a superlinear term. The
   timeouts are sized for the pessimistic reading, and a stage that overruns is a scheduling failure, not
   a result.

9. **No fix is proposed or tested.** This measures whether a lever still has travel. It repairs nothing.

10. **MDE on retention is unavailable for the L=160 seed-42 arm** — its cube was deleted after scoring by
    design. The decision rule in §5 was written not to need it; every *new* arm computes its own via
    `ap_block_bootstrap.py --ratio`.

## Known couplings carried in

- **The three L=160 seed arms are arms 1–3 of the parked SS sweep** (`longzero_fortythree`, `_fortyfour`,
  `_fortyfive`) and are written into **that** dossier's results directory by its **unmodified**
  `run_arm.sh`. This pre-pays ~3.5 h of the parked sweep instead of duplicating it, and
  `run_sweep.sh`'s score-CSV resumability will skip them later. It also means the sweep is partly
  consumed by this dossier — recorded here rather than discovered at relaunch.
- **`violet_visitor` is the floor for every arm**, so all six of its config idiosyncrasies (dropout 0.15,
  `sampling_strategy: sigmoid`, `freeze_multitask_balancer`, no `loss_class_alpha/gamma`, `id_col`,
  `ss_warmup_lessons`) are held fixed and inherited, not controlled.
- **C-61 queryset pinning**: each clone writes `model_name = "violet_visitor"` into `config_queryset.py`
  so the provenance digest matches the cached parquet. The arms therefore share one data identity by
  construction — which is what makes them comparable, and what makes them useless for any question about
  the data.
