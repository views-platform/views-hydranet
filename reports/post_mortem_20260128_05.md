# Post-Mortem: HydraNet Evaluation Stabilization & Orchestration
**Date:** 2026-01-28
**Status:** Completed & Verified

## 1. Executive Summary
This session involved the recovery and modernization of the `views-hydranet` forecasting and evaluation pipeline following a "crash site" discovery. The system was in a "split-brain" state with redundant experimental files and broken utility functions. By applying a "Rust-like" robustness plan and the **Chain-of-Verification** method, we transitioned the pipeline from a fragile, dictionary-based output to a strictly validated, contract-compliant Producer for the `views-evaluation` ecosystem.

## 2. Problem Diagnosis (Triage)
Before stabilization, the system suffered from several critical failures:
- **Redundancy:** `evaluate_model.py` and `generate_forecast.py` were 99% identical copies, causing ambiguity.
- **KeyError ('model_time_stamp'):** Mutation of the `self.config` object in `HydranetManager` was volatile, leading to crashes during path construction.
- **TypeError ('NoneType' not iterable):** The manager's evaluation entry point returned `None`, crashing the downstream pipeline which expected a list of DataFrames.
- **Orchestration Gap:** Evaluation was hardcoded to a single window, failing the standard 12-window rolling-origin requirement for offline evaluation.
- **Stale Logic:** Reliance on deleted utility functions from 2025.

## 3. Implemented Solutions

### A. Architectural Consolidation
- **Canonical Path:** Deleted `evaluate_model.py` and promoted `views_hydranet/forecast/execution.py` as the unified forecasting core.
- **Contract Utility:** Authored `zstack_to_contract_df` to handle the flattening of 5D/6D zstacks into the 2D MultiIndex DataFrames required by the consumer.
- **Explicit Scaling:** Adopted "Option 2" (Explicit Raw Scale). The Producer (HydraNet) is now solely responsible for inverse-transforming log-fatalities to raw counts (`exp(x)-1`) before emission.

### B. Manager Robustness
- **Local State:** Refactored `HydranetManager` to use local variables for timestamps and paths, ensuring immunity to pipeline configuration volatility.
- **Rolling Origin Orchestration:** Integrated `get_rolling_origin_indices` utility. The manager now automatically loops 12 times for validation/calibration partitions, slicing the input volume into the "Predictive Parallelogram" required by `views-evaluation`.
- **Dynamic Targets:** Integrated `target_variable` ("sb", "ns", "os") selection from hyperparameters/sweeps.

### C. Verification Framework
A multi-layered test suite was established to provide permanent protection:
1. **Reversibility Proof:** `contract_df_to_zstack` proves that we can recover the original model output from the contract DataFrame (lossless roundtrip).
2. **Integration Proof:** End-to-end flow from `.pt` artifact loading to contract emission verified via toy model.
3. **Adversarial Defense:** `validate_contract_dataframes` blocks NaNs, Infs, and Ocean Cell violations.
4. **Golden Regression:** Deterministic CRPS values are pinned to ensure numerical stability across environments.

## 4. Final Status
- **Resolved Bugs:** All reported KeyErrors and TypeErrors are fixed.
- **Contract Fulfillment:** HydraNet is now a verified "Producer" for `views-evaluation`.
- **Test State:** **73 tests passing green.**
- **Version Control:** All changes committed and pushed to `migrate_stuff_from_old_repo`.

## 5. Next Steps
- **Performance:** For high-resolution grids (180x180), the DataFrame conversion is functional but could be optimized with vectorization if execution time becomes a bottleneck in massive sweeps.
- **Security:** Consider re-enabling `weights_only=True` in `torch.load` by allowlisting the `architectures` classes in safe globals.
- **Phase 2 (Config):** Proceed with full transition from dict-based config to typed `PipelineConfig` objects.
