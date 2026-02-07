
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.utils_config import HydraNetConfig

# BIT-PERFECT ROOT AUDIT STANDARDS
BASE_CONFIG = {
    'run_type': 'calibration',
    'steps': [1, 2],
    'time_steps': 2,
    'input_channels': 3,
    'output_channels': 1,
    'regression_targets': ['lr_sb_best'],
    'classification_targets': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'identity_cols': ['month_id', 'priogrid_gid'],
    'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'transform': {
        'log1p': ['lr_sb_best'],
        'asinh': ['lr_ns_best'],
        'identity': ['lr_os_best']
    },
    'height': 4, 'width': 4,
    'time_col': 'month_id', 'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'row_offset': 0, 'col_offset': 0,
    'model': 'Dummy', 'window_dim': 1, 'total_hidden_channels': 8,
    'dropout_rate': 0.0, 'weight_init': 'norm', 'h_init': 'zero',
    'learning_rate': 0.01, 'weight_decay': 0.0, 'windows_per_lesson': 1,
    'scheduler': 'none', 'warmup_steps': 1, 'clip_grad_norm': True,
    'loss_reg':  'lr_b', 'loss_class':  'lr_b', 'loss_reg_a': 1, 'loss_reg_c': 1,
    'loss_class_gamma': 1, 'loss_class_alpha': 1,
    'total_lessons': 1, 'n_posterior_samples': 1, 'np_seed': 1, 'torch_seed': 1,
    'min_events': 0, 'slope_ratio': 0.1, 'roof_ratio': 0.1, 'max_ratio': 0.9, 'min_ratio': 0.1,
    'freeze_h': 'none', 'evalution_mode': 'point', 'aggregate_method': 'mean'
}

def test_root_scaling_validation():
    """Assert that the scaler fails if a feature is missing from the 'transform' dict."""
    bad = BASE_CONFIG.copy()
    bad['transform'] = {'log1p': ['lr_sb_best']} # Missing ns and os
    with pytest.raises(ValidationError, match="not assigned a transform in the 'transform' dict"):
        HydraNetConfig(**bad)

def test_root_checksum_input_channels():
    """Assert input_channels checksum."""
    bad = BASE_CONFIG.copy()
    bad['input_channels'] = 99
    with pytest.raises(ValidationError, match="Checksum Law Violation: input_channels"):
        HydraNetConfig(**bad)

def test_scaler_root_consumption():
    """Assert FeatureScaler correctly uses the 'transform' dict."""
    scaler = FeatureScaler(BASE_CONFIG)
    df = pd.DataFrame({
        'lr_sb_best': [10.0], 'lr_ns_best': [10.0], 'lr_os_best': [10.0]
    })
    semantic = scaler.fit_transform(df)
    # log1p(10) is ~2.39
    assert semantic['lr_sb_best'].iloc[0] < 3.0

    recovered = scaler.inverse_transform(semantic)
    np.testing.assert_allclose(df['lr_sb_best'], recovered['lr_sb_best'])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
