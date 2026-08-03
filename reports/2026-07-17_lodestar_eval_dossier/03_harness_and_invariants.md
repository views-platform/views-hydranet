# 03 — Harness & invariants (the LOCKED ruler + guarantees)

The crown jewel. This codifies the frozen ruler and the guardrails that make the table trustworthy.

## A. Existing guardrails we RELY ON (discovered, not built here)
- **Reproducibility gate** — `infrastructure/reproducibility_gate.py` locks numpy/torch seeds +
  deterministic algorithms per run (logged "Entropy locked"). ⇒ the seed axis of the grid is meaningful.
- **BatchNorm-recalibration** (`training_engine.py`, C-184) — ON by default; recomputes BN stats post-train.
  Confirmed firing in every grid run's log ("BN-recalibrating 15 layers"). ⇒ removes the seed-bimodal
  collapse; the grid measures real config effects, not the BN lottery.
- **Partition config** — `config_partitions.py` calibration test = (457, 504), IDENTICAL in baseline and
  hydranet configs. ⇒ re-running any model now lands on the same window. THIS is the fix for the
  months-misaligned bug; do not alter partitions to force alignment.
- **Stealth trap-restore** — views-models configs patched in place + restored (md5-checked) via a trap. Never
  commit/push views-models.
- **`bulk_score.py::crps_ensemble`** — validated CRPS energy-form; the lodestar scorer reuses it verbatim so
  numbers are consistent with prior work.
- **Grid-name agnostic join** — `grid_naming.grid_id_col` / name-set membership (#144). The scorer must
  derive the grid column from data, never hardcode.
- **Glossary** — `reports/GLOSSARY.md`: terminology guardrail (no synonym drift).

## B. Invariants
- **HARD (must hold):** T=0-only; one truth parquet; identical (month,cell) support across compared models;
  metric functions unit-tested; frozen scorer unchanged mid-analysis.
- **Intentionally set BY this program:** the ruler itself (AP+Brier for switch; all/event/zero-CRPS + size
  ratio for size); the common-support intersection rule; the switch-score derivation for count-only
  baselines (fraction of samples > 0).
- **Respect-while-using:** the stealth rule; BN-recal default-on; the partition window (457–504).

## C. THE FROZEN RULER — `tools/lodestar_score.py` (spec)
One script, self-tested, then **frozen** (a change ⇒ new version ⇒ re-score everything).

**Inputs:** a canonical truth parquet + a model registry `{label → (pred_dir, {target → (lr_name, by_name|None)})}`.

**Per target (sb/ns/os):**
1. **Gather T=0** for each model: first forecast month per origin; collect (month, unit) → count-samples,
   and switch-samples if `by_name` present.
2. **Common support:** intersect the (month, unit) keys across ALL models that have this target, AND keep
   only keys present in the truth parquet. This is THE fairness step. Report |support|, N_event, N_zero.
3. **Truth:** join count truth by (month_id, grid-id) (grid-name-agnostic). binary truth = (count > 0).
4. **Switch:** P(conflict) = mean(switch samples) [hydranet] or mean(count samples > 0) [baselines]. Compute
   `AP = average_precision_score`, `Brier = mean((p - binary)^2)`.
5. **Size:** `crps = crps_ensemble(truth, count_samples)` (energy form, sorted). Report all/event/zero means +
   `size_ratio = median(mean_samples / truth)` over event cells.

**Fail-loud (raise, never silently coerce):** missing target dir; empty common support; non-finite metric;
a model missing > 20% of the common support (coverage mismatch); truth join yielding all-NaN.

**Determinism:** no randomness. Same inputs → identical output. Prints a table + writes `results/lodestar.csv`.

## D. Self-test (gate before any real scoring)
`tools/test_lodestar_score.py` (or an inline `--selftest`): synthetic 2-model, 3-cell fixtures where AP,
Brier, CRPS, and the event/zero split are hand-computable → assert the scorer reproduces them. MUST pass
before the ruler is frozen and used.

## E. Pre-flight checklist (gate to "ready to run")
- [ ] partition confirmed 457–504 (both families) ✔ (already verified)
- [ ] one baseline re-runs cleanly (feasibility) — pending
- [ ] scorer written + self-test green + FROZEN — pending
- [ ] common-support intersection verified non-trivial for all three targets — pending
- [ ] grid architecture fixed & config-validated — pending
- [ ] stale baseline preds deleted; baselines re-run on 457–504 — pending

## F. Known gaps / risks to watch
- **Mixture is sparse** (~hundreds of cells vs ~170k) ⇒ common support may be small; report N so it's visible.
  A small N is honest, not a bug — but flag it.
- **Sample counts differ** (climatology 64, mixture 256, hydranet 8) ⇒ CRPS is sample-count sensitive at
  small N; note hydranet's 8 samples as a caveat (not corrected here — it's the deployed operating point).
- **Truth vintage:** baselines were trained on data as-of their run; re-running now uses current data. Fine
  for calibration truth (historical, stable) but assert the join covers the eval months.
