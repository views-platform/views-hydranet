# Plan: Exhaustive HydraNet Test Expansion

## Objective
To achieve "Rust-like" robustness by ensuring every transformation, orchestration, and configuration state in HydraNet is verifiable, predictable, and protected against regressions.

---

## Phase 1: The "Translation & Augmentation" Suite (Unit)
The monkey-patching and target translation are the most "magical" parts of the system. We must de-mystify them with unit tests.
*   **Target Translation:** Verify `lr_sb_best` -> `lr_sb_best` and `sb` -> `lr_sb` across all edge cases.
*   **JIT Augmentor:** Verify that the augmented data loader correctly:
    *   Unlogs `lr_` columns into `lr_` columns.
    *   Binarizes magnitude columns into `_binarized` columns.
    *   Does nothing if columns are already present.
*   **Verification Target:** `tests/test_manager_augmentation.py`

## Phase 2: The "Multi-Task Contract" Suite (Unit)
Verify that HydraNet can emit its **Full 6-Head Output** correctly and satisfies the consumer contract.
*   **Parallel Targets:** Test `zstack_to_contract_df` with a list of 6 targets (3 mag, 3 binary).
*   **Channel Fidelity:** Ensure that when asking for "ns_best", the data is pulled from channel 1, and for "os_best" from channel 2.
*   **Binary Contract:** Ensure `_binarized` predictions are strictly `0.0` or `1.0`.
*   **Verification Target:** Expand `tests/test_forecast_contract.py`

## Phase 3: The "Manager Lifecycle" Suite (Integration)
Verify that the `HydranetManager` orchestrates the pipeline core correctly without crashing.
*   **Intercept Proof:** Mock `super()._execute_model_evaluation` and verify that the `self.configs["targets"]` it sees are the translated `lr_` versions.
*   **Restoration Proof:** Verify that after evaluation fails or succeeds, the `targets` config and `read_dataframe` are restored to their original state.
*   **Verification Target:** `tests/test_manager_lifecycle.py`

## Phase 4: The "Stochastic Integrity" Suite (Statistical)
Since HydraNet is stochastic (MC Dropout), we must prove the statistics are stable.
*   **Sample Variance:** Test that with 100 samples, the mean prediction is stable across multiple runs (Seed check).
*   **Dropout Check:** Prove that predictions *differ* when dropout is on, and are *identical* when dropout is off.
*   **Verification Target:** `tests/test_stochastic_stability.py`

## Phase 5: The "Memory & Scale" Suite (Performance)
Prevent the 32GB RAM death before it happens.
*   **Linear Growth Test:** Measure RAM for 10, 50, and 100 samples. Assert that the growth is linear.
*   **Large Grid Test:** Run a conversion on a $360 \times 360$ grid (simulated) to ensure the vectorized logic scales spatially.
*   **Verification Target:** `tests/test_scale_limits.py`
