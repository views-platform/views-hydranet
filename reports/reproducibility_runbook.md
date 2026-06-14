# Reproducibility Runbook — how to verify HydraNet training/eval is deterministic

**Created:** 2026-06-15 · **Owner:** chair + pair
**Why this exists:** training was silently non-deterministic at a fixed seed (~20% run-to-run variance), which
confounded every single-run comparison (root cause **C-119**, fixed in `daab1c1`; post-mortem
`postmortem_training_nondeterminism_init_rng_drift.md`). This runbook is the **single trustworthy reference** for
checking determinism so we never lose the method or repeat the mistake. Supersedes, for determinism purposes, the old
`views-models/reports/parity_runbook.md` (which is unreliable — see below).

---

## 1. The reliable signal — and the trap to avoid

**✅ Judge determinism by the weight-TENSOR hash and exact prediction equality. ❌ Never by the `.pt` file sha.**

- **`.pt` FILE sha256 is UNRELIABLE.** `torch.save` writes a zip archive that embeds non-deterministic metadata
  (mtimes), so the **same** `state_dict` saved twice yields **different** file shas. Proven:
  ```python
  sd = {"w": torch.ones(10)}; # save twice -> sha 'fe07ce3d…' vs '1fc215d0…'  (DIFFERENT, identical weights)
  ```
  `sha256sum model.pt` therefore **cannot** distinguish identical weights from different weights. Do not use it.
- **Weight-TENSOR hash IS reliable.** Hash the `state_dict` tensors' bytes directly:
  ```python
  import hashlib
  def state_dict_hash(sd):
      h = hashlib.sha256()
      for k in sorted(sd):
          h.update(k.encode()); h.update(sd[k].detach().cpu().numpy().tobytes())
      return h.hexdigest()
  ```
  Identical weights → identical hash; any difference → different hash. (Same method as the S2 regression test
  `tests/test_training_engine.py::_state_dict_hash`.)
- **Predictions:** compare with `np.array_equal` on every `origin_*/{target}/y_pred.npy` (exact identity), and use
  `scripts/mcr_readout.py` for the human-readable MCR/CRPS cross-check.

## 2. Do NOT use the old views-models parity tooling for determinism

- **`investigations/compare_parity.py`** measures *similarity* (pearson `r`, grades "EXCELLENT" at `r>0.99`),
  **collapses posterior samples to their mean**, and only compares two *different* models (viewser↔datafactory),
  picking each model's *latest* predictions dir. It **structurally cannot** compare two runs of the same model, and
  `r>0.99` would *mask* small non-determinism. Similarity ≠ identity.
- **`views-models/reports/parity_runbook.md`** logs a `sha256sum` of the `.pt` file — unreliable per §1.
- These were built for cross-implementation parity (a different question), not bit-reproducibility. Use the §1 method
  and `scripts/compare_run_determinism.py` instead.

## 3. The determinism check (procedure)

Ground rules (kept from the old runbook — these parts are sound): **one model on the GPU at a time**, **frozen data**,
**log every run here**.

1. From `views-models/models/violet_visitor/`:
   - **Run 1:** `python main.py -r calibration -t -e -re` — fetches viewser data, trains, evals, report; caches parquet.
   - **Run 2:** `python main.py -r calibration -t -e -re --saved` — **same cached data**, a fresh *independent* training.
     (`--saved` pins the *data*, not the model — both runs train from scratch; we compare them.)
2. Compare with `scripts/compare_run_determinism.py <artifact1.pt> <artifact2.pt> <preds_dir1> <preds_dir2>`:
   - weight-tensor hashes **identical**, and
   - every `y_pred.npy` **`np.array_equal`**, and
   - `mcr_readout.py` MCR/CRPS **match**.
3. **Verdict:** all identical → **deterministic ("fixed-fixed")**; Run 1 is the deterministic baseline. **Any**
   difference → **STOP and investigate** — the fix doesn't hold at scale.

**Caveats:** ~50 min/run (train+eval); sequential (one GPU); the `min_free_disk_gb` guard protects the ~2.5 GB writes;
an **exit-137-after-metrics** is the known-spurious **C-116** (metrics + artifact already written), not a failure.

## 4. Run log (append-only)

| Date | Runs (config) | Weight-hash identical? | `y_pred` array-equal? | MCR match? | Verdict |
|------|---------------|------------------------|------------------------|------------|---------|
| _pending_ | violet no-coords ×2 (`-t -e`, seed 42; run2 `--saved`), post-`daab1c1` | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

## 5. Cross-refs
- **C-119** (register) — the non-determinism bug + fix. **C-79** — the regression test that guards it.
- `reports/postmortem_training_nondeterminism_init_rng_drift.md` — full investigation + the `.pt`-sha trap (§4 there).
- `tests/test_training_engine.py` — `test_init_deterministic_regardless_of_prior_rng_state`,
  `test_training_run_is_reproducible` (the committed guards).
