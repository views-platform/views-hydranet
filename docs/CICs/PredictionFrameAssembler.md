# Class Intent Contract: PredictionFrameAssembler

**Status:** Active
**Owner:** Interface Adapter
**Last reviewed:** 11.04.2026
**Related ADRs:** ADR-022 (Layered Architecture), ADR-039 (Symmetry Engine), ADR-047 (Pandas-free output)

---

## 1. Purpose

The `PredictionFrameAssembler` is an **Interface Adapter** between the Entity-layer `VolumeHandler` and the Framework-layer `PredictionFrame` (from `views_pipeline_core`). Its primary purpose is to convert a wrapped prediction VolumeHandler (signal) plus a history scaffold (provider) into a `dict[str, PredictionFrame]` keyed by target name, without ever materializing a pandas DataFrame.

It exists as a separate class — rather than as methods on `VolumeHandler` — so that the Entity-layer `VolumeHandler` does not import a Framework type. This was the D-01 partial split decision, executed on 2026-04-11.

---

## 2. Non-Goals (Explicit Exclusions)

- This class does **not** modify the input volumes — `signal` and `history` are read-only.
- This class does **not** perform inference, scaling, or temporal alignment — those are upstream responsibilities of `InferenceOrchestrator`.
- This class does **not** materialize pandas DataFrames at any stage. The output is pure numpy arrays wrapped in `PredictionFrame`.
- This class does **not** maintain state between calls. It is stateless; instances are interchangeable.

---

## 3. Responsibilities and Guarantees

- **Pandas-free output:** Guarantees that no pandas DataFrame is constructed during PF assembly. The output path is numpy → `PredictionFrame` directly.
- **Bounds checking:** `assemble_evaluation()` raises `ValueError` if `start_idx + duration > history_duration`, preventing silent out-of-range slicing.
- **North-Up convention:** Applies the same `np.flip(..., axis=0)` as `VolumeHandler.from_df()` to ensure spatial parity between input and output (verified by `test_volume_handler_hard_gates.py::test_gate_flip_symmetry_from_df_to_output`).
- **Stochastic preservation:** When the signal has an `S` axis (stochastic mode), `y_pred.shape == (N, S)`. When the signal is 4D (point mode), `y_pred.shape == (N, 1)` — never `(N,)` or scalar.
- **Identifier alignment:** Each `PredictionFrame` carries `identifiers = {"time": time_flat, "unit": unit_flat}` derived from the provider's `time_col` and `id_col` channels. Length matches `y_pred.shape[0]`.
- **Lazy framework import:** `PredictionFrame` is imported inside `assemble_evaluation()` (not at module level) so consumers can import this module without `views_pipeline_core` installed.
- **Stateless:** No instance state. Construct once per inference run or once per call — both are valid.

---

## 4. Inputs and Assumptions

- **`signal: VolumeHandler`** — A wrapped prediction volume produced by `VolumeHandler.wrap_predictions()` and (optionally) `inverse_transform_volume()` / `collapse_to_point()`. Must have `pred_{target}` channels for every target name in `all_targets`.
- **`history: VolumeHandler`** — A history scaffold providing geographic identity and time keys. Must have `id_col` and `time_col` channels.
- **`start_idx: int`** — Temporal offset within `history` where the signal starts. Must satisfy `start_idx + signal.duration <= history.duration`.
- **`all_targets: List[str]`** — Target names. The assembler looks up `pred_{target}` in `signal.channel_map`.
- **Mask convention:** Cells with `priogrid_gid > 0` are valid; cells with `priogrid_gid == 0` are dropped (ocean cells, per `DataFetcher.standardize_raw_df`).

---

## 5. Outputs and Side Effects

- **Returns `dict[str, PredictionFrame]`** — One entry per target name. Each `PredictionFrame` has `y_pred.ndim == 2` always.
- **No file I/O.** Pure in-memory transformation.
- **No mutation.** Inputs are not modified.
- **No logging on success.** Only logs on the bounds-check failure path (`logger.error` before raising).

---

## 6. Failure Modes and Loudness

- **Convention mismatch:** Raises `ValueError("Signal must be North-Up. Got: ...")` or `ValueError("Provider must be North-Up. Got: ...")` if either input has `spatial_convention != SpatialConvention.NORTH_UP`. Guards use `raise ValueError` (not `assert`) so they survive `python -O`.
- **Bounds Violation:** Raises `ValueError("PredictionFrameAssembler Contract Violation: ...")` if `start_idx + duration > history_duration`. Logged at ERROR level before raising.
- **Missing target channel:** Raises `ValueError` (from `signal.channel_map.index(pred_col)`) if a target name's `pred_{target}` column is missing from the signal's channel map.
- **Missing `views_pipeline_core`:** Raises `ImportError` at the lazy import site inside `assemble_evaluation()`. Module-level import (`from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler`) succeeds without the framework installed.

---

## 7. Boundaries and Interactions

- **Upstream (callers):** `InferenceOrchestrator.generate_prediction_frames()` and `InferenceOrchestrator.generate_prediction_frames_streaming()` are the only production callers. Each constructs a `PredictionFrameAssembler()` instance once per invocation and reuses it across origins.
- **Read-only access to VolumeHandler:** The assembler uses public properties (`.data`, `.axes`, `.channel_map`, `.id_col`, `.time_col`, `.get_axis_idx()`) and the public method `.slice_time()`. It does not reach into `_metadata` or `_data` directly (small encapsulation win over the original implementation).
- **Framework boundary:** `PredictionFrame` is imported lazily inside `assemble_evaluation()`. This is the only contact point with `views_pipeline_core`.

---

## 8. Examples of Correct Usage

```python
from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

assembler = PredictionFrameAssembler()
pf_dict = assembler.assemble_evaluation(
    signal=pred_handler,        # Wrapped, inverse-transformed (and optionally collapsed) volume
    history=window_handler,     # The matching history scaffold
    start_idx=0,                # Temporal offset within history
    all_targets=all_targets,    # Target names — pred_{target} columns must exist
)
# pf_dict["lr_sb_best"].y_pred.shape == (N, S)  # stochastic
# pf_dict["lr_sb_best"].y_pred.shape == (N, 1)  # point
```

---

## 9. Examples of Incorrect Usage

- **Passing a non-wrapped VolumeHandler:** The signal must have `pred_{target}` channels. Passing a raw history volume will raise `ValueError` on the channel lookup.
- **Reusing a stale `start_idx`:** `start_idx` must be valid for the *current* `history` argument. The assembler does no caching — each call is independent.
- **Constructing per-origin:** Acceptable but wasteful. The assembler is stateless; one instance suffices for an entire inference run.
- **Calling `_valid_cell_indices()` or `_reconstruct_as_pf_dict()` directly:** Private helpers. Use `assemble_evaluation()` as the entry point.

---

## 10. Test Alignment

- **🟩 Green Team:** Round-trip and shape tests in `tests/test_prediction_frame_assembler.py` (`TestAssembleEvaluation`):
  - `test_returns_dict`, `test_all_targets_present`
  - `test_stochastic_shape`, `test_point_shape`
  - `test_identifiers_populated`
- **🟫 Beige Team:** Integration tests via `InferenceOrchestrator` in `tests/test_inference_orchestrator_pf.py`.
- **🟥 Red Team:** `test_bounds_check_raises_on_bad_start_idx` (verifies the contract violation path).
- **Memory hygiene:** `tests/test_inference_memory_hygiene.py::test_valid_cell_indices_does_not_copy_signal_data` and `test_valid_cell_indices_does_not_copy_provider_data` enforce that `np.transpose` and `np.flip` produce views, not copies (critical for pgm-scale memory).

---

## End of Contract

This document defines the **intended meaning** of `PredictionFrameAssembler`.
Changes to behavior that violate this intent are bugs.
Changes to intent must update this contract.
