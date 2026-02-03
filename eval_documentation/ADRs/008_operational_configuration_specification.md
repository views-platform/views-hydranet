# ADR 008: Operational Configuration Specification

**Status:** Hardened
**Context:** Complexity arises when architectural assumptions are hidden in the code. To maintain a "Boring Architecture," we require an explicit, readable, and immutable configuration that serves as the single source of truth for hyperparameters, data topology, and model behavior.

---

## 1. Decision: Configuration as the "Source of Truth"
The configuration dictionary is the authoritative record for a run. 

### 1.1 The "No-Defaults" Policy (Zero-Magic)
To prevent silent configuration drift and hidden "Physics," the system enforces a strict **No-Defaults Policy**:
*   **Mandatory Fields:** Every parameter that affects the data topology, model architecture, or strategic trajectory must be explicitly defined in the config.
*   **Loud Failure:** Any missing field must trigger a `ValidationError` (via Pydantic) at startup. The implementation is prohibited from injecting fallback values.

### 1.2 The Checksum Law (Redundancy for Safety)
Where parameters are mathematically coupled, we favor **Explicit Redundancy** over silent derivation. The user must provide both values, and the pipeline will assert their equality at the handshake gate. This prevents silent logical failures if lists are misconfigured.

*   **Coupling 1:** `input_channels` MUST equal `len(features)`.
*   **Coupling 2:** `time_steps` MUST equal `len(steps)`.
*   **Coupling 3:** `n_classification_outputs` (implicit) MUST equal `len(classification_outputs)`.

### 1.3 Internalization of Architectural Invariants (ADR 020)
Naming prefixes (`pred_`) and suffixes (`_raw`, `_prob`) are fixed architectural invariants. Including them in the configuration is prohibited. They are enforced internally by the `VolumeHandler` Symmetry Engine.

---

## 2. Structural Zones

### 2.1 Spatiotemporal Topology (The Physics)
*   `height`, `width`, `time_col`, `id_col`, `spatial_cols`: Structural role mapping.
*   `row_offset`, `col_offset`: Geographic anchoring.
*   `identity_cols`: Non-predictive metadata to be stripped by name.
*   `features`: The exhaustive list of predictive signals.

### 2.2 Training & Optimization (The Torch Gate)
*   `model`, `total_hidden_channels`, `window_dim`, `dropout_rate`, `weight_init`.
*   `learning_rate`, `weight_decay`, `scheduler`, `warmup_steps`, `clip_grad_norm`.
*   `windows_per_lesson`: The mini-batch accumulation depth (ADR 014).

### 2.3 Loss & Strategy (The Trajectory)
*   `loss_reg`, `loss_class`: Loss variants ('b' for balanced, etc.).
*   `loss_reg_a`, `loss_reg_c`, `loss_class_alpha`, `loss_class_gamma`.
*   `total_lessons`, `max_ratio`, `min_ratio`, `slope_ratio`, `roof_ratio`, `min_events`.

---

## 3. Canonical Template (The Standard)

```python
hyperparameters = {
    # Topology & Physics (ADR 007 Compliance)
    'time_col': 'month_id',
    'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'row_offset': 0,
    'col_offset': 0,
    'height': 180,
    'width': 180,
    'identity_cols': ['month_id', 'priogrid_gid', 'row', 'col', 'c_id'],
    'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'input_channels': 3, # Checksum for 'features'

    # Model Architecture
    'model': 'HydraBNUNet06_LSTM4',
    'total_hidden_channels': 32,
    'dropout_rate': 0.1,
    'window_dim': 32,
    'weight_init': 'xavier_norm',
    'freeze_h': "hl",

    # Optimization (ADR 014 Compliance)
    'windows_per_lesson': 3,     
    'learning_rate': 0.001,
    'weight_decay': 0.1,
    'scheduler': 'WarmupDecay',
    'warmup_steps': 100,
    'clip_grad_norm': True,
    'torch_seed': 4,
    'np_seed': 4,

    # Multi-Task Signals (ADR 020 Compliance)
    'target_variable': 'lr_sb_best',
    'classification_outputs': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'targets': ['lr_sb_best'], # Inbound requested targets
    'transform': 'log1p',
    'steps': list(range(1, 37)),
    'time_steps': 36, # Checksum for 'steps'

    # Loss Functions
    'loss_reg': 'b',
    'loss_class': 'b',
    'loss_reg_a': 16,
    'loss_reg_c': 0.05,
    'loss_class_alpha': 0.75,
    'loss_class_gamma': 1.5,

    # Strategy (Curriculum ADR 011/012 Compliance)
    'total_lessons': 300,        
    'max_ratio': 0.95,           
    'min_ratio': 0.05,           
    'slope_ratio': 0.75,         
    'roof_ratio': 0.7,           
    'min_events': 5,             

    # Outbound / Evaluation
    'n_posterior_samples': 128,
    'evalution_mode': 'stochastic', # 'point' or 'stochastic'
    'aggregate_method': 'geometric_mean',
    'run_type': 'calibration',
}
```
