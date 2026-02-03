# ADR 008: Operational Configuration Specification

**Status:** Proposed  
**Context:** Complexity arises when architectural assumptions are hidden in the code. To maintain a "Boring Architecture," we require an explicit, readable, and immutable configuration that serves as the single source of truth for hyperparameters, data topology, and model behavior.

---

## 1. Decision: Configuration as the "Source of Truth"
The configuration dictionary returned by `get_hp_config()` is the authoritative record for a run. 
*   **Immutability:** Once the configuration is passed to a component (Manager, Handler, Sampler), it must be treated as read-only.
*   **Explicit Topology:** Column roles and spatiotemporal anchors must be defined in the configuration to allow "Zero-Magic" operation of data components.

---

## 2. Structural Zones

### 2.1 Ledger / Topology (Structural Truth)
This section defines the "Physics" of the dataset. It allows the `VolumeHandler` to operate without hardcoded column names.
*   `time_col`: Defines the temporal axis.
*   `id_col`: Defines the unit/node identity axis.
*   `spatial_cols`: Defines the geographic [y, x] grid coordinates.

---

## 3. Canonical Example (The Standard)

Below is the definitive, unabridged example of a compliant configuration. All future hyperparameter files must adhere to this structure.

```python
def get_hp_config():
    """
    Contains the hyperparameter configurations for model training.
    """

    hyperparameters = {

        # ============================================================
        # Model identity / high-level
        # ============================================================
        'model': 'HydraBNUNet06_LSTM4',

        # ============================================================
        # Ledger / Topology (ADR 007 Compliance)
        # ============================================================
        'time_col': 'month_id',
        'id_col': 'priogrid_gid',
        'spatial_cols': ['row', 'col'],
        'row_offset': 87,            # Global geographic anchor (Y)
        'col_offset': 310,           # Global geographic anchor (X)
        'height': 180,               # Global grid resolution (Y)
        'width': 180,                # Global grid resolution (X)

        # ============================================================
        # Target / data handling
        # ============================================================
        'target_variable': 'sb_best',
        'min_events': 5,
        'window_dim': 32,
        'time_steps': 36, 
        "steps": list(range(1, 37)),

        'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'asinh': [],
        'identity': [],

        # ============================================================
        # Model architecture
        # ============================================================
        'input_channels': 3,
        'output_channels': 1,
        'total_hidden_channels': 32,
        'dropout_rate': 0.125,
        'h_init': 'abs_rand_exp-100',
        'freeze_h': "hl",

        # ============================================================
        # Initialization / regularization
        # ============================================================
        'weight_init': 'xavier_norm',
        'clip_grad_norm': True,

        # ============================================================
        # Optimization
        # ============================================================
        'learning_rate': 0.001,
        'weight_decay': 0.1,
        'scheduler': 'WarmupDecay',
        'warmup_steps': 100,

        # ============================================================
        # Curriculum & Optimization (The Lesson Specification)
        # ============================================================
        'total_lessons': 300,        # Number of curriculum stages
        'windows_per_lesson': 3,     # True Mini-Batch size (The Mixed Salad)
        'n_posterior_samples': 10,   # Resolution of uncertainty (MC Dropout)

        # ============================================================
        # Loss: classification & regression
        # ============================================================
        'loss_class': 'b',
        'loss_class_gamma': 1.5,
        'loss_class_alpha': 0.75,
        'loss_reg': 'b',
        'loss_reg_a': 16,
        'loss_reg_c': 0.05,

        # ============================================================
        # Ratios / heuristics
        # ============================================================
        'slope_ratio': 0.75,
        'roof_ratio': 0.7,

        # ============================================================
        # Reproducibility
        # ============================================================
        'np_seed': 4,
        'torch_seed': 4,

        # ============================================================
        # Evaluation
        # ============================================================
        'evalution_mode': 'point',
        'aggregate_method': 'geometric_mean',
        'aggregate_space': 'raw',
    }

    return hyperparameters
```

---

## 4. Operational Invariants
1.  **Visibility:** Any parameter that affects the data transformation or model outcome must be visible in the config.
2.  **No Dynamic Augmentation:** The configuration passed to the `VolumeHandler` must not be modified by "clever" helper functions during runtime.
3.  **Traceability:** The configuration should be logged (e.g., to W&B) to ensure bit-perfect reproducibility of any given run.