# ViEWS HydraNet Architectural Contracts & Schemes

> **Status:** Draft / Recovered  
> **Date:** 2026-01-30  
> **Scope:** `views_hydranet` library

## 1. The Configuration Handshake Protocol

HydraNet employs a "Strict Handshake" mechanism to ensure stability before any heavy computation begins.

- **Trigger:** `HydranetManager._perform_strict_handshake()`
- **Mechanism:** 
  1. Reads configuration from `self._hydranet_config` (primary) or `self.configs` (legacy base).
  2. Validates against `HydraNetConfig` Pydantic schema.
  3. Updates both local and base storage with the *validated* and *healed* values.
- **Critical Fields:**
  - `run_type`: Must be one of `calibration`, `validation`, `forecasting`.
  - `steps`: List of integers defining the look-ahead horizon (e.g., `[1, ..., 36]`).
  - `target_variable`: The specific conflict type (e.g., `sb`, `ns`, `os`).

## 2. The "Safe-Mode" Manager Pattern

To survive in hostile environments (like mocked tests or partial initializations), the Manager uses a "Safe-Mode" pattern for state access.

- **Model Path:** 
  - **Unsafe:** `self.model_path` (Base class property, may not exist).
  - **Safe:** `self._model_path` (Local attribute, guaranteed at `__init__`).
- **Config Access:**
  - **Unsafe:** `self.configs` (Base class property, may trigger attribute errors).
  - **Safe:** `self.config` (Property that falls back to `self._hydranet_config`).

## 3. Data Contracts

### 3.1 Input Volume Contract (The 4D Cube)
HydraNet operates on a strict 4D Spatiotemporal Volume format: `[Time, Height, Width, Channels]`

- **Channels 0-4 (Metadata):** strictly reserved for `priogrid_gid`, `col`, `row`, `month_id`, `c_id`.
- **Channels 5+ (Features):** The actual input features (e.g., `lr_sb_best`).
- **Spatial Orientation:** "North is Up". The conversion process applies `np.flip(vol, axis=0)` to match CNN expectations.

### 3.2 Evaluation Target Scheme
The system enforces a specific naming convention for targets during evaluation.

- **Translation:** `lr_` (log-normalized) prefixes are automatically translated to `lr_` (level-raw).
  - Example: `lr_sb_best` -> `lr_sb_best`
- **Derivation:** If a target is requested (e.g., `lr_sb_best`) but only the log version exists, the system automatically:
  1. Detects `lr_sb_best`.
  2. Applies `expm1` (inverse log1p).
  3. Creates `lr_sb_best`.
- **Binarization:** Targets containing `binarized` trigger an automatic thresholding:
  - `val > 0 ? 1.0 : 0.0`

### 3.3 Output Contract (The Producer Interface)
Predictions are returned as a list of DataFrames, adhering to the ViEWS Producer Contract.

- **Index:** `MultiIndex(month_id, priogrid_gid)`
- **Columns:** `pred_lr_{target}`
- **Content:** List of posterior samples (for stochastic evaluation).

## 4. Numerical Healing & Stability (TEMPORARY SHIM)

To prevent pipeline crashes during debugging (short training runs), HydraNet implements a temporary **Stability Shim**.

**CRITICAL:** This is not a fix for the underlying modeling problems. Overflows to non-finite or unrealistic values are considered architectural failures.

- **Automatic Sanitization:** All non-finite values (`NaN`, `Inf`) are automatically substituted with `0.0` at the contract boundaries.
- **Safety Clamping:** Values are clamped to a maximum of `20.0` in log-space (approx. 485 million in raw-space) before inverse transformation.
- **Two-Pass Implementation:** Sanitization occurs both **before** and **after** mathematical transformations.
- **Scope:** This healing applies to **both** Ground-Truth (Actuals) augmentation and Model Predictions.

## 5. Shadow Environment (Evaluation)
To evaluate without polluting the production data state, HydraNet uses a "Shadow Environment".

1. **Create:** A temporary folder `artifacts/tmp_eval_data`.
2. **Augment:** Generates derived ground-truth columns (e.g., `lr_` from `lr_`) and saves a new parquet file there.
3. **Link:** Symlinks other required files (logs, etc.) from the real raw directory.
4. **Redirect:** Points `self._model_path.data_raw` to this shadow directory.
5. **Execute:** Runs evaluation.
6. **Restore:** Points `data_raw` back to original and deletes the shadow directory.

## 5. Logging & Observability
- **WandB:** Initialization is **not assumed**. Logging calls are wrapped in checks (`if wandb.run is not None`) to prevent crashes in offline/test modes.
- **Fail-Fast:** The system is designed to crash early (during the Handshake) if the configuration is invalid, rather than failing deep in the training loop.
