# Audit: Redundant Code Purge List

This list identifies artifacts that are no longer essential to the HydraNet "Boring Architecture."

## 1. [RED] Safe to Delete (Verified Unused)

### Core Utilities (`views_hydranet/utils/`)
- **`utils_loss.py`**: Contains duplicate definitions of loss functions that now live in dedicated files (`focal_loss.py`, `mtloss.py`, `shringkage_loss.py`).
- **`utils_topology.py`**: Contains `SpatialLayout` enum and invariants that are not consumed by any active component.
- **`utils.py` -> `my_decay`**: superseded by the logic in `CurriculumLearner`.

### Legacy Tree (`views_hydranet/legacy_code/`)
- **Entire Directory**: Verified that no active component in `manager/`, `train/`, or `utils/` imports from this tree. It represents significant linguistic and technical debt.

### Configuration (`HydraNetConfig`)
- **`h_init`**: Validated at the gate but never used to initialize hidden states.
- **`output_channels`**: Redundant; neural topology is determined by `classification_targets`.
- **`time_steps`**: Redundant checksum; derived from `len(steps)`.

### Dependencies (`pyproject.toml`)
- **`torchvision`**: Only referenced in comments and legacy wandb logs.
- **`torchaudio`**: Only referenced in legacy wandb logs.

## 2. [YELLOW] Deprecated (Refactor Recommended)

### `views_hydranet/utils/utils.py`
- **`execute_freeze_h_option`**: This is duplicated in `HydraNetInference`. The version in `utils.py` is only used by legacy code. Once `legacy_code/` is purged, this should be deleted.

## 3. [GREEN] Active (Essential)
- All other components in `views_hydranet/utils/`.
- `HydranetManager`, `InferenceOrchestrator`, `PureStateAdapter`.
- `train_model.py`.
