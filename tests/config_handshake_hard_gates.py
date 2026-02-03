
import pytest
from pydantic import ValidationError
from views_hydranet.utils.utils_config import HydraNetConfig

# SHARED MINIMUM SET
MIN_CFG = {
    'run_type': 'calibration',
    'steps': [1, 2],
    'time_steps': 2,
    'input_channels': 1,
    'output_channels': 1,
    'target_variable': 'sb',
    'targets': ['sb'],
    'classification_outputs': ['sb'],
    'identity_cols': ['month_id', 'priogrid_gid'],
    'features': ['sb'],
    'transforms': {'log1p': ['sb']},
    'height': 1, 'width': 1,
    'time_col': 'month_id', 'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'row_offset': 0, 'col_offset': 0,
    'model': 'Dummy', 'window_dim': 1, 'total_hidden_channels': 8,
    'dropout_rate': 0.0, 'weight_init': 'norm', 'h_init': 'zero',
    'learning_rate': 0.01, 'weight_decay': 0.0, 'windows_per_lesson': 1,
    'scheduler': 'none', 'warmup_steps': 1, 'clip_grad_norm': True,
    'loss_reg': 'b', 'loss_class': 'b', 'loss_reg_a': 1, 'loss_reg_c': 1,
    'loss_class_gamma': 1, 'loss_class_alpha': 1,
    'total_lessons': 1, 'n_posterior_samples': 1, 'np_seed': 1, 'torch_seed': 1,
    'min_events': 0, 'slope_ratio': 0.1, 'roof_ratio': 0.1, 'max_ratio': 0.9, 'min_ratio': 0.1,
    'freeze_h': 'none', 'evalution_mode': 'point', 'aggregate_method': 'mean'
}

def test_gate_6_input_checksum():
    bad = MIN_CFG.copy()
    bad['input_channels'] = 99 # features has 1
    with pytest.raises(ValidationError, match="Checksum Law Violation: input_channels"):
        HydraNetConfig(**bad)

def test_gate_7_time_checksum():
    bad = MIN_CFG.copy()
    bad['time_steps'] = 99 # steps has 2
    with pytest.raises(ValidationError, match="Checksum Law Violation: time_steps"):
        HydraNetConfig(**bad)

def test_gate_8_scaling_missing():
    bad = MIN_CFG.copy()
    bad['transforms'] = {} # features has 'sb'
    with pytest.raises(ValidationError, match="missing from 'transforms'"):
        HydraNetConfig(**bad)

def test_gate_9_scaling_dual():
    bad = MIN_CFG.copy()
    bad['transforms'] = {'log1p': ['sb'], 'asinh': ['sb']}
    with pytest.raises(ValidationError, match="mapped multiple times"):
        HydraNetConfig(**bad)

    cfg = HydraNetConfig(**MIN_CFG)
    assert 'log1p' in cfg.transforms
    assert cfg.transforms['log1p'] == ['sb']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
