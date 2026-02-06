# Post-Mortem: Data Pipeline Liberation & Stateful Scaling
**Date:** 2026-02-01
**Status:** Completed (Infrastructure Established)

## 1. The Challenge
The HydraNet pipeline suffered from "Russian Doll" complexity and "Primitive Obsession." Specifically:
- **Hidden Caching:** Training data was silently cached as `.npy` volumes, masking upstream changes.
- **Double Unlogging:** String prefixes like `ln_` triggered automatic mathematical operations in multiple nested layers, causing evaluation-time overflows.
- **Leaky Boundaries:** Math (log1p/expm1) was scattered throughout the codebase, making it impossible to know the exact scale of data at any given point.

## 2. The Solution: The Gateway Pattern
We enforced a strict boundary between **Raw Space** (Ingestion/Evaluation) and **Semantic Space** (Model/Tensors).

### Infrastructure Introduced:
- **`DataFetcher` (`utils/data_fetcher.py`)**: A dedicated class for literal ingestion. No more magic; it loads the parquet, prints the columns, and hands it off.
- **`FeatureScaler` (`utils/feature_scaler.py`)**: A stateful, declarative gateway. 
    - It interpreted `log1p`, `asinh`, and `identity` from the config.
    - It enforces a "one-shot" lifecycle (locked after fit).
    - It provides bit-identical `inverse_transform` capabilities.
- **Flattened Orchestration**: `HydranetManager` was refactored to show the linear flow: `Fetch -> Scale -> Transform -> Train/Infer`.

## 3. Current State of the Codebase
- **Prefix Stability:** The strings `lr_` and `lr_` are now treated as literal markers. The pipeline no longer attempts to "smartly" change them.
- **Internal Invariant:** Everything inside the "Tensor Zone" is semantic. No JIT scaling happens inside `get_full_tensor` or the Trainer.
- **Authoritative Gateway:** The `FeatureScaler` is the only place where math is performed.

## 4. Known Issues & Residual Debt
The following items were identified during the post-refactor audit:
- **Source Bug:** A `NameError` exists in `_execute_model_evaluation` (Line 256: `raw_targets` vs `standard_targets`).
- **Source Bug:** A `SyntaxError` exists in `utils/data_loader.py` (Line 29: empty `if __name__` block).
- **Test Alignment:** The test suite has been 95% aligned. 4 failures remain, blocked by the `NameError` in the source.

## 5. Next Steps
- Implement the "Metadata Registry" (The Ledger) to replace the remaining string-parsing logic in `_augment_dataframe`.
- Resolve the identified source-code bugs.
- Perform a final validation run with real `purple_alien` data to confirm the overflow issue is resolved.
