# Plan: ModelArtifactEvaluator Implementation

## 1. Objective
Implement the `ModelArtifactEvaluator` class in `views_hydranet/utils/model_artifact_evaluator.py` as defined in ADR 024.

## 2. Component Handshake
*   **Input Dependencies:**
    *   `config`: Full operational configuration dictionary.
    *   `model`: Trained `torch.nn.Module`.
    *   `device`: Target execution device (CPU/CUDA).
    *   `handler`: Canonical Inbound `VolumeHandler`.
    *   `scaler`: `FeatureScaler` with fitted transformation parameters.

## 3. Implementation Steps

### Step 3.1: Constructor and Orchestration
*   Initialize internal state.
*   Resolve `origins` indices using `get_rolling_origin_indices`.

### Step 3.2: The Inference Loop
*   Iterate over `origins`.
*   Call `HydraNetInference.generate_posterior_samples`.

### Step 3.3: The Reconstruction Bridge
*   Apply `wrap_predictions` (Watermarking).
*   Apply `scaler.inverse_transform_volume`.
*   Apply `collapse_to_point` (if Point Mode).
*   Invoke `to_evaluation_df` (Vader Bridge).

### Step 3.4: The Subsetting Gate
*   Filter resulting DataFrame columns to requested targets only.
*   Enforce MultiIndex restoration.

### Step 3.5: Validation Gate
*   Invoke `validate_contract_dataframes`.

## 4. Verification Protocol

### Green Team (The Happy Path)
*   Prove that a mock volume returns bit-perfect DataFrames.
*   Verify that `month_id` and `pg_id` indices are correct.

### Beige Team (The Robustness Path)
*   Verify failure on mismatched temporal durations.
*   Verify failure on missing target columns.

### Red Team (The Invincibility Path)
*   Verify that the evaluator maintains integrity even if the model returns shuffled tensors (relying on Vader Bridge).
