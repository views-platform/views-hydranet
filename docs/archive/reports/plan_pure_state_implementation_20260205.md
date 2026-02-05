# Plan: Pure State Output Implementation & Verification

| Plan Info           | Details |
|---------------------|---------|
| Subject             | Implementing ADR 032 and Sniffer Verification |
| Status              | Proposed |
| Author              | Gemini CLI |
| Date                | 05.02.2026 |

## 1. Objective (The Bit-Perfect Promise)
We will implement the "Pure State" output schema (ADR 032) to ensure that the HydraNet pipeline delivers bit-perfect, losslessly restored DataFrames. The output must be indistinguishable from the input data, augmented only by model predictions.

## 2. Implementation Steps

### 2.1 VolumeHandler Refactor
- **Naming Engine:** Update `wrap_predictions` to enforce `pred_lr_` and `pred_by_` prefixes and remove `_raw`/`_prob` suffixes.
- **Identity Restoration:** Hardwire the carrying of `c_id`, `row`, and `col` during the `to_evaluation_df` and `to_forecast_df` calls.
- **Binary Actuals:** Ensure `by_{target}` is generated via a deterministic `> 0` check during reconstruction if not already present in the history scaffold.

### 2.2 DataSniffer (The Sentinel) Upgrades
The `DataSniffer` will receive two new mandatory check methods:
1.  **`sniff_pure_state_parity(df_input, df_output)`**:
    - **Logic:** `temp_df = df_output.drop(columns=[c for c in df_output.columns if c.startswith("pred_")])`.
    - **Assert:** `temp_df.equals(df_input)`.
    - **Goal:** Prove zero information loss/corruption during the volume round-trip.
2.  **`sniff_pure_state_schema(df_output, config)`**:
    - **Logic:** Checks MultiIndex names, presence of `c_id`, and exact prefix matching for all requested targets.
    - **Goal:** Enforce the ADR 032 contract before the data reaches the Adapter layer.

## 3. Verification Protocol (The Tests)
We will implement `tests/test_pure_state_integrity.py` with two core mission-critical tests:

### Test 1: The Bit-Wise Parity Audit
- **Scenario:** Ingest a DataFrame, convert to Volume, reconstruct to DataFrame.
- **Assertion:** `Sniffer.sniff_pure_state_parity` must pass. Any shift in indices, coordinates, or bookkeeping IDs must trigger a "Fail Loud and Proud" event.

### Test 2: The Structural Schema Audit
- **Scenario:** Verify the output of a mock evaluation.
- **Assertion:** `Sniffer.sniff_pure_state_schema` must pass. 
- **Checks:**
    - MultiIndex levels are `(month_id, priogrid_gid)`.
    - Column types are correct (scalars for actuals, lists for stochastic preds).
    - Redundant suffixes (`_raw`, `_prob`) are absent.

## 4. Integration
The `ModelArtifactEvaluator` will be the primary orchestrator of these tests, calling the Sniffer immediately before returning the final DataFrame to the Manager.

## 4. The Team Audit Verification Suite

To ensure absolute reliability, the implementation will be verified against the following three psychological roles:

### 4.1 Green Team (The Happy Path)
*   **Goal:** Prove the system works perfectly under ideal conditions.
*   **Verification:** 
    - Perform a full `DF -> Volume -> DF` round-trip.
    - Assert `c_id` (country_id) is preserved bit-perfectly across all time steps.
    - Assert coordinate alignment: `geographic_row` in DF must exactly match `array_index_y + row_offset`.

### 4.2 Beige Team (The Robustness Path)
*   **Goal:** Prove the system fails "Loud and Proud" when contracts are violated.
*   **Verification:** 
    - Attempt reconstruction with a meta-volume missing the `c_id` channel; assert immediate `KeyError`.
    - Provide an input DF with a misnamed index (e.g., `month` instead of `month_id`); assert immediate `ContractViolation`.
    - Verify that if `n_posterior_samples > 1` but the reconstruction produces scalars, the Sniffer triggers a type-mismatch error.

### 4.3 Red Team (The Invincibility Path)
*   **Goal:** Prove that malicious or catastrophic data states cannot corrupt identities.
*   **Verification:**
    - **The Shuffle Test:** Randomly shuffle the rows of the input DataFrame before Volume construction. Assert that the reconstructed DataFrame is bit-identical to the *pre-shuffled* input. This proves identity is restored via coordinate math, not positional luck.
    - **The Ocean Breach Test:** Inject non-zero prediction values into geographic coordinates where `priogrid_gid == 0` (Ocean). Assert that the reconstruction bridge strictly filters these out, maintaining a "Land-Only" output.
