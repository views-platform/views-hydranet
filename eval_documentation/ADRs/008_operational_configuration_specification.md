# ADR 008: Operational Configuration Specification

**Status:** Proposed  
**Context:** Complexity arises when architectural assumptions are hidden in the code. To maintain a "Boring Architecture," we require an explicit, readable, and immutable configuration that serves as the single source of truth for hyperparameters, data topology, and model behavior.

---

## 1. Decision: Configuration as the "Source of Truth"
The configuration dictionary returned by `get_hp_config()` is the authoritative record for a run. 

### 1.1 Immutability
Once the configuration is passed to a component (Manager, Handler, Sampler), it must be treated as read-only.

### 1.2 The "No-Defaults" Policy (Zero-Magic)
To prevent silent configuration drift and hidden "Physics," the system enforces a strict **No-Defaults Policy**:
*   **Mandatory Fields:** Every parameter that affects the data topology, model architecture, or strategic trajectory must be explicitly defined in the config.
*   **Loud Failure:** Any missing field must trigger a `ValidationError` or `KeyError` at startup. The implementation is prohibited from injecting fallback values (e.g., assuming a default learning rate or spatial offset).
*   **Explicit Topology:** Column roles and spatiotemporal anchors must be defined in the configuration to allow "Zero-Magic" operation of data components.

---

## 2. Structural Zones

### 2.1 Ledger / Topology (The Physics)
*   `time_col`, `id_col`, `spatial_cols`: Structural role mapping.
*   `row_offset`, `col_offset`, `height`, `width`: Geographic anchoring.

### 2.2 Curriculum Learning (The Strategy)
*   `total_lessons`: Number of curriculum stages (optimization cycles).
*   `max_ratio`, `min_ratio`: Target-relative intensity range for signal anchorage.
*   `slope_ratio`, `roof_ratio`, `min_events`: Trajectory scheduling and floors.

### 2.3 Optimization & Initialization (The Mechanics)
*   `windows_per_lesson`: Mini-batch size (The Mixed Salad).
*   `learning_rate`, `weight_decay`, `scheduler`, `warmup_steps`: Optimizer constraints.
*   `weight_init`, `clip_grad_norm`, `np_seed`, `torch_seed`: Determinism and stability.

---

## 3. Canonical Example (The Standard)

```python
def get_hp_config():
    hyperparameters = {

        # ============================================================
        # Ledger / Topology (ADR 007 Compliance)
        # ============================================================
        'time_col': 'month_id',
        'id_col': 'priogrid_gid',
        'spatial_cols': ['row', 'col'],
        'row_offset': 87,
        'col_offset': 310,
        'height': 180,
        'width': 180,

        # ============================================================
        # Curriculum Learning (ADR 011/012 Compliance)
        # ============================================================
        'total_lessons': 300,        
        'max_ratio': 0.95,           
        'min_ratio': 0.05,           
        'slope_ratio': 0.75,         
        'roof_ratio': 0.7,           
        'min_events': 5,             

        # ============================================================
        # Optimization (ADR 014 Compliance)
        # ============================================================
        'windows_per_lesson': 3,     
        'learning_rate': 0.001,
        'weight_decay': 0.1,
        'scheduler': 'WarmupDecay',
        'warmup_steps': 100,
        'weight_init': 'xavier_norm',
        'clip_grad_norm': True,
        'torch_seed': 4,
        'np_seed': 4,

        # ============================================================
        # Model Architecture
        # ============================================================
        'model': 'HydraBNUNet06_LSTM4',
        'input_channels': 3,
        'output_channels': 1,
        'total_hidden_channels': 32,
        'dropout_rate': 0.125,
        'window_dim': 32,
        'h_init': 'abs_rand_exp-100',
        'freeze_h': "hl",

        # ============================================================
        # Loss Functions
        # ============================================================
        'loss_reg': 'b',
        'loss_class': 'b',
        'loss_reg_a': 16,
        'loss_reg_c': 0.05,
        'loss_class_gamma': 1.5,
        'loss_class_alpha': 0.75,

        # ============================================================
        # Inbound Data Handling
        # ============================================================
        'target_variable': 'sb_best',
        'transform': 'log1p',
        'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'asinh': [],
        'identity': [],
        'steps': list(range(1, 37)),
        'time_steps': 36,

        # ============================================================
        # Outbound / Evaluation
        # ============================================================
        'n_posterior_samples': 128,
        'evalution_mode': 'stochastic',
        'aggregate_method': 'geometric_mean',
        'aggregate_space': 'raw',
        'run_type': 'calibration',
    }

    return hyperparameters
```