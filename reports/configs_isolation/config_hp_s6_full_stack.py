
def get_hp_config():
    """
    S6 — FULL NEW STACK (basu_dpd + hurdle + qs99 + target_weights)
    Delta: everything the sweep had, in a single run.
    Only run this if BOTH S4 (combined features) and S5 (basu_dpd alone) pass.
    This is the config that would prove the full vision works.
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
        # ── S6 DELTA: basu_dpd + all new features ──
        # ============================================================
        'loss_reg': 'basu_dpd',
        'loss_reg_alpha': 0.5,
        'loss_reg_sigma': 1.0,
        'loss_class': 'focal',
        'loss_class_alpha': 0.75,
        'loss_class_gamma': 1.5,
        'onset_bias_init': -7.0,
        'hurdle_threshold': 0.0,
        'qs99_weight': 0.1,
        'qs99_tau': 0.99,
        'target_weights': {
            'lr_sb_best': 2.0,
            'lr_ns_best': 1.0,
            'lr_os_best': 1.0,
        },

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
