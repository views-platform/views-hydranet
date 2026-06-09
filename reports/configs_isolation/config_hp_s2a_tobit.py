
def get_hp_config():
    """
    S2a — BASELINE + Tobit censored-normal loss
    Delta: replaces shrinkage with Tobit NLL (loss_reg='tobit', sigma=1.0).
    No hurdle_threshold — Tobit handles zero-inflation internally via
    censored likelihood. This is the validation that Path A (ADR-054) fixes
    the gradient starvation that S2 (hurdle mask) caused.

    Expected: PASS. Dense gradient from all cells (including y=0 via censored
    likelihood) should eliminate the divergence seen in S2.
    """

    hyperparameters = {

        # ============================================================
        # Ledger / Topology (ADR 007 Compliance)
        # ============================================================
        'time_col': 'month_id',
        'id_col': 'priogrid_gid',
        'spatial_cols': ['row', 'col'],
        'identity_cols': ['month_id', 'priogrid_gid', 'c_id', 'row', 'col'],
        "index_names": ['month_id', 'priogrid_gid'],
        'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'input_channels': 3,
        'row_offset': 87,
        'col_offset': 310,
        'height': 180,
        'width': 180,

        # ============================================================
        # Model Architecture
        # ============================================================
        'model': 'HydraBNUNet06_LSTM4',
        'total_hidden_channels': 32,
        'dropout_rate': 0.125,
        'window_dim': 32,
        'output_channels': 1,
        'weight_init': 'xavier_norm',
        'freeze_h': "hl",
        'h_init': 'abs_rand_exp-100',

        # ============================================================
        # Optimization (ADR 014 Compliance)
        # ============================================================
        'windows_per_lesson': 3,
        'learning_rate': 0.001,
        'weight_decay': 0.1,
        'scheduler': 'WarmupDecay',
        'warmup_steps': 100,
        'clip_grad_norm': True,
        'torch_seed': 4,
        'np_seed': 4,

        # ============================================================
        # Multi-Task Signals (ADR 020 Compliance)
        # ============================================================
        'classification_targets': ['by_sb_best', 'by_ns_best', 'by_os_best'],
        'regression_targets': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],

        'transformations': {
            'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
            'asinh': [],
            'identity': []
        },

        'derivations': {
            'binary': [
                {'from': 'lr_sb_best', 'to': 'by_sb_best', 'threshold': 0},
                {'from': 'lr_ns_best', 'to': 'by_ns_best', 'threshold': 0},
                {'from': 'lr_os_best', 'to': 'by_os_best', 'threshold': 0},
            ],
        },

        'steps': list(range(1, 37)),
        'time_steps': 36,

        # ============================================================
        # Loss Functions
        # ============================================================
        # ── S2a DELTA: Tobit censored-normal likelihood (ADR-054) ──
        'loss_reg': 'tobit',
        'loss_reg_sigma': 1.0,
        'loss_class': 'focal',
        'loss_class_alpha': 0.75,
        'loss_class_gamma': 1.5,
        'onset_bias_init': -7.0,

        # ============================================================
        # Strategy (Curriculum ADR 011/012 Compliance)
        # ============================================================
        'total_lessons': 20,
        'max_ratio': 0.95,
        'min_ratio': 0.05,
        'slope_ratio': 0.75,
        'roof_ratio': 0.7,
        'min_events': 5,
        'sampling_strategy': 'threshold',

        # ============================================================
        # Outbound / Evaluation
        # ============================================================
        'n_posterior_samples': 3,
        'evaluation_mode': 'stochastic',
        'aggregate_method': 'arithmetic_mean',
        'skip_predictions_delivery': True,
    }

    return hyperparameters
