# Exhaustive HydraNet Configuration Audit (03-02-2026)

This report lists every configuration key currently accessed by the `views_hydranet` package, identified through recursive static analysis of dictionary access, `.get()` calls, and attribute-style lookups.

## 1. Topographical & Structural Invariants
*Used by VolumeHandler, DataSniffer, and DataFetcher to map the physical layout.*

*   `height`
*   `width`
*   `time_col`
*   `id_col`
*   `spatial_cols`
*   `row_offset`
*   `col_offset`
*   `identity_cols`
*   `features`
*   `index_names`

## 2. Partitioning & Run-Time Metadata
*Governs the temporal slicing and experiment tracking.*

*   `run_type`
*   `steps`
*   `time_steps`
*   `model_time_stamp`
*   `sweep`

## 3. Training Architecture & Weights
*Directly injected into the Torch model and initialization routines.*

*   `model`
*   `input_channels`
*   `output_channels`
*   `total_hidden_channels`
*   `window_dim`
*   `dropout_rate`
*   `weight_init`
*   `weight_decay`

## 4. Optimization & Gradients
*Used by the Trainer and Curriculum Learner.*

*   `learning_rate`
*   `scheduler`
*   `warmup_steps`
*   `windows_per_lesson`
*   `clip_grad_norm`

## 5. Multi-Task Physics (The Heads)
*Defines the naming and mapping of semantic signals.*

*   `target_variable`
*   `targets`
*   `classification_outputs`

## 6. Loss Function Parameters
*Parameters for Focal and Shrinkage loss variants.*

*   `loss_reg`
*   `loss_reg_a`
*   `loss_reg_c`
*   `loss_class`
*   `loss_class_alpha`
*   `loss_class_gamma`

## 7. Curriculum & Spatial Filtering
*Governs lesson-level data selection.*

*   `total_lessons`
*   `min_events`
*   `slope_ratio`
*   `roof_ratio`
*   `max_ratio`
*   `min_ratio`
*   `freeze_h`
*   `random_flips` (Optional/Experimental)

## 8. Outbound Evaluation & Aggregation
*Governs the transition from tensors back to DataFrames.*

*   `n_posterior_samples`
*   `evalution_mode` (Includes typo normalize logic)
*   `aggregate_method`

## 9. Reproducibility
*Ensures bit-perfect deterministic runs.*

*   `np_seed`
*   `torch_seed`

## 10. Legacy / Residual Artifacts
*Accessed in legacy_code or experimental modules.*

*   `test_samples`
