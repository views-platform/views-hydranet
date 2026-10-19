# Class Intent Contract: `VisualDiagnostics`

**Status:** Active
**Owner:** Custodian (Visual Truth Engine)
**Last Reviewed:** 19.10.2026
**Related ADRs:** ADR-005 (Testing), ADR-006 (CIC), ADR-008 (Failure Loudness), ADR-034 (Prediction Diagnostics), ADR-035 (Training Health Audit), ADR-037 (Geometric Health Visualization)

---

## 1. Purpose
`VisualDiagnostics` is the **Visual Truth Engine** of the HydraNet pipeline. It is a unified, passive observer that generates "Visual Biopsies" — structured PNG diagnostics — at every stage of the data lifecycle. It covers 7 pipeline stages from raw ingestion (Stage 1) through final reconstruction (Stage 7), providing a human-interpretable audit trail that no scalar metric can replace.

It implements the **Null Object pattern**: when `diagnostic_visualizations: False` in config, every public method is a silent no-op. This ensures zero performance cost and zero code-path changes when diagnostics are disabled.

---

## 2. Non-Goals (Explicit Exclusions)
- This class does **not** compute metrics, gradients, or model weights.
- This class does **not** mutate input data, tensors, or VolumeHandler state.
- This class does **not** raise exceptions to callers — all active methods wrap their bodies in `try/except` and log errors.
- This class does **not** control curriculum strategy, sampling, or optimization.
- This class does **not** own the authoritative metric history (that is `TrainingForensics`).

---

## 3. Responsibilities and Guarantees
- **Null Object Contract:** When inactive (`self.active == False`), ALL 9 public methods return immediately without side effects. No files are created, no exceptions are raised, no logs are emitted beyond DEBUG level.
- **Resilient Active Mode:** When active, every public method wraps its body in `try/except Exception`. Failures are logged at ERROR level; the exception is NEVER propagated to the caller.
- **PNG File Output:** Each biopsy call saves exactly one PNG to `self.save_dir`, using the sanitized `stage_label` as the filename stem. Exception: `biopsy_loss_curves` saves two PNGs (linear + log scale).
- **Save Directory Isolation:** The save directory `reports/plots/diagnostics/{timestamp}/` is created at construction time. The timestamp is resolved in priority order: `run_timestamp` argument → `config['model_time_stamp']` → `datetime.now()`.
- **9-Method Public Interface:**
  1. `biopsy_dataframe(df, stage_label, features)` — Stage 1 (raw ingestion) and Stage 2 (scaled) and Stage 7 (final reconstruction)
  2. `biopsy_volume(vh, stage_label)` — Stage 3 (global volume) and Stage 6 (predicted volume)
  3. `biopsy_tensor(tensor, stage_label, channel_names)` — raw PyTorch tensor [B,T,C,H,W] or [T,C,H,W]
  4. `biopsy_sample(sample_vh, global_vh, stage_label)` — Stage 4 (training samples with global context)
  5. `biopsy_autoregressive(truth_seq, pred_seq, stage_label, channel_names, time_indices)` — Stage 5 inference (3×6 Truth/Pred/Delta grid)
  6. `biopsy_training_performance(y_reg, y_hat_reg, y_cls, y_hat_cls, stage_label, *, y_hat_forecast, forecast_label, time_indices)` — Stage 5 training (5×6 grid: truth Reg / body log1p(E[y]) / truth Cls / cls gate / THE FORECAST). **Also persists the five fields as `.npz`.**
  7. `biopsy_loss_curves(history_reg, history_cls, history_total, stage_label)` — Live loss evolution (linear + log)
  8. `biopsy_feature_dossier(target_name, dossier, stage_label)` — Per-target forensic dossier (fed by `TrainingForensics`)
  9. `biopsy_health_constellation(weight_norms, stage_label)` — ADR-037 radar plot of mean L2 norms per functional block (Encoder, Bottleneck, Decoder, MultiTaskHead); saved to `02_training_dynamics/` as `constellation_{safe_label}.png`
- **North-Up Convention:** All spatial grids are rendered with `origin='upper'` (index 0 = North), consistent with the VolumeHandler flip convention.
- **Stats Overlay:** `_calculate_stats(data)` computes μ, σ, min, max from finite values only. Returns `"EMPTY"` for all-NaN or zero-element inputs.
- **⛔ Units are named on every row label.** Regression rows carry **log1p counts**, not counts: the targets are trained in log1p space and a family head's biopsy value is already log1p'd (`training_engine._family_target_log1p_mean`). The row previously read `PREDICTION: body E[y]`, and a reader applied `log1p` to an already-log1p quantity and understated the body by an exponential. The label now says `log1p(E[y])`.
- **⛔ One colour scale per UNIT, not per row** (`_shared_count_scale`). Rows 0/1/4 (log1p counts) share `(0.0, max)`; rows 2/3 (probabilities) are pinned to `(0.0, 1.0)`; the shared range is printed on the label. Independent per-row min–max stretching made "bright prediction over dark truth" the rendering a sparse panel *always* produces — an identically-zero truth row collapses to `vmin == vmax` and renders flat while a narrow prediction band saturates the colormap. Cross-row brightness comparisons taken from the old figure were unfalsifiable by construction. A degenerate span falls back to `vmax = 1.0`.
- **⛔ `biopsy_training_performance` does NOT compose the forecast.** `y_hat_forecast` is a **required keyword** supplied by the caller, which composes with `training_engine._family_composed_mean_log1p` — the same helper deployment mirrors (`hydranet_inference._emit_magnitude`): `log1p(compose_mean(E[Y], gate))`. The method used to form `gate × y_hat_reg` itself, i.e. `gate · log1p(μ)`, which is a different function from `log1p(gate · μ)` and agrees only where the gate is 1 — under a docstring calling it "the ACTUAL scored forecast". Required rather than defaulted so a shadow default cannot silently restore it (C-85).

---

## 4. Inputs and Assumptions
- **Constructor config keys consumed:** `diagnostic_visualizations`, `spatial_cols`, `time_col`, `height`, `width`, `row_offset`, `col_offset`, `regression_metrics`, `classification_metrics`, `loss_reg`, `loss_class`, `model_time_stamp` (optional).
- **VolumeHandler inputs:** Assumes data is in `[T, H, W, C]` logical order (axis permutation is applied internally via `get_axis_idx`).
- **Array inputs:** `biopsy_autoregressive` and `biopsy_training_performance` accept lists of `[H, W, C]` numpy arrays (one per time step).
- **Dossier inputs:** `biopsy_feature_dossier` consumes the dict returned by `TrainingForensics.get_dossier()`. Keys: `bias_instant`, `bias_running`, `y_bar`, `y_hat_bar`, plus metric names (e.g. `mse`, `ap`).
- **Precondition:** The class assumes that `diagnostic_visualizations` is the single authoritative switch for all diagnostic output. No other checks are needed by callers.

---

## 5. Outputs and Side Effects
- **PNG files** written to `reports/plots/diagnostics/{timestamp}/biopsy_{safe_label}.png` (relative to cwd at construction time).
- **`biopsy_{safe_label}.npz`** — written by `biopsy_training_performance` **before** the figure is built, holding `y_reg`, `y_hat_reg`, `y_cls`, `y_hat_cls`, `y_hat_forecast` and a `units` string. `TrainingForensics` persists per-lesson **scalars only** (no `savez`/`np.save` anywhere in it), so before this the field existed solely as rendered pixels and any later question about it required a fresh training run. The write precedes the drawing because matplotlib is the part that fails and the array is the part that cannot be recovered; it never raises (a diagnostic must not abort a multi-hour run). **Measured cost at production shapes** (`window_dim: 32`, 6 biopsy steps, 3 targets): **196 KiB and 20 ms per biopsy** → **172 MiB and 18 s** across a 300-lesson × 3-window run, against a measured ≥50 min (≈13% of training) already spent on the PNGs. ⚠️ Gated on `diagnostic_visualizations` like every other method, so a diagnostics-off run still produces no field record.
- **`loss_evolution.png`** and **`loss_evolution_log.png`** — written by `biopsy_loss_curves`, keyed by fixed name (not stage_label), so they are overwritten on each call (accumulating the latest view).
- **`forensic_{target_name}.png`** — written by `biopsy_feature_dossier`, keyed by target name.
- **`self.save_dir`** directory is created at construction (side effect of `os.makedirs`).
- No mutation of input data, tensors, or VolumeHandler objects.

---

## 6. Failure Modes and Loudness
- **Active method failure:** Caught by `try/except Exception`. Logged at `logger.error`. Never raised. The pipeline continues.
- **Unsupported tensor ndim:** `biopsy_tensor` logs a `logger.warning` and returns silently (does not save a PNG).
- **Inactive method:** Silently returns (no log, no side effect, no exception).
- **All-NaN/empty data:** `_calculate_stats` returns `"EMPTY"` string; no exception is raised.
- **Missing config keys:** All config keys have fallback defaults (e.g. `height` defaults to 180). Missing keys do not raise.
- **NO exception propagation is permitted from any public method.**

---

## 7. Boundaries and Interactions
- **Producers (call VisualDiagnostics):**
  - `train_model.py`: calls `biopsy_sample`, `biopsy_training_performance`, `biopsy_loss_curves`, `biopsy_feature_dossier`
  - `run_predict.py` (inference path): calls `biopsy_autoregressive`, `biopsy_volume`, `biopsy_dataframe`
  - `data_fetcher.py` (ingestion path): calls `biopsy_dataframe` at Stages 1 and 2
- **Consumers of VisualDiagnostics outputs:**
  - Human operators reviewing `reports/plots/diagnostics/{timestamp}/` for pipeline health
- **Sibling classes:**
  - `TrainingForensics`: provides dossier dicts consumed by `biopsy_feature_dossier`
  - `VolumeHandler`: provides the data substrate consumed by `biopsy_volume` and `biopsy_sample`
- **Import contract (issue #215):** `matplotlib` is imported **lazily** (via the module-level `_load_mpl()`
  helper, Agg backend), **only inside the plotting methods** — never at module level. Because this class is
  a Null Object (inactive by default), importing this module or constructing/using an inactive instance
  requires **no plotting stack**. `matplotlib` is therefore a runtime-**optional** `viz` extra
  (`pip install views-hydranet[viz]`); models that set `diagnostic_visualizations=True` must provide it.

---

## 10. Test Alignment

| Test File | Coverage |
|-----------|----------|
| `tests/test_visual_diagnostics.py` | 39 tests: 8 beige null-object, 15 green active-mode with PNG output, 8 red error-logging with exc_info, 8 adversarial NaN/empty inputs |

---

## End of Contract
This document defines the **intended meaning** of `VisualDiagnostics`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
