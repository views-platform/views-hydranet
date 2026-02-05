
import numpy as np
import pandas as pd
import pytest
import torch

from views_hydranet.utils.volume_handler import VolumeHandler

# SHARED PHYSICS CONFIG
PHYSICS_CFG = {
    'time_col': 'month_id',
    'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'identity_cols': ['month_id', 'priogrid_gid'],
    'features': ['feature_a', 'feature_b'],
    'row_offset': 10,
    'col_offset': 20,
    'height': 4,
    'width': 4
}

def test_gate_11_identity_striping():
    """Assert to_pytorch strips identities by name."""
    df = pd.DataFrame({
        'month_id': [1, 1], 'priogrid_gid': [1, 2],
        'row': [10, 10], 'col': [20, 21],
        'feature_a': [1.0, 2.0], 'feature_b': [0.0, 0.0]
    })
    handler = VolumeHandler.from_df(df, PHYSICS_CFG)
    tensor = handler.to_pytorch(torch.device('cpu'), include_identities=False)

    # 2 features, T=1, H=4, W=4
    assert tensor.shape == (1, 1, 2, 4, 4)

def test_gate_12_6_head_dressing():
    """Assert wrap_predictions correctly dresses 6 semantic heads."""
    posterior = torch.ones((1, 1, 4, 4, 4)) # T=1, C=4, H=4, W=4 (2 reg, 2 class)
    base_names = ['a', 'b']

    handler = VolumeHandler(
        data=np.zeros((1, 4, 4, 2)), axes=('T', 'H', 'W', 'C'),
        channel_map=['month_id', 'priogrid_gid'],
        time_col='month_id', id_col='priogrid_gid',
        spatial_cols=['row', 'col']
    )

    pred_handler = handler.wrap_predictions(posterior, base_names=base_names)
    
    # Check internal signal names
    assert "pred_lr_a" in pred_handler.channel_map
    assert "pred_by_a" in pred_handler.channel_map
    assert "pred_lr_b" in pred_handler.channel_map
    assert "pred_by_b" in pred_handler.channel_map

def test_gate_13_14_topography_restoration():
    """Assert full symmetry recovery and MultiIndex restoration."""
    df_hist = pd.DataFrame({
        'month_id': [1, 1], 'priogrid_gid': [1, 2],
        'row': [10, 10], 'col': [20, 21],
        'feature_a': [10.0, 20.0], 'feature_b': [5.0, 5.0]
    })
    handler = VolumeHandler.from_df(df_hist, PHYSICS_CFG)

    # Simulate a 4-channel prediction (2 reg, 2 class)
    # feature_a_raw, feature_b_raw, feature_a_prob, feature_b_prob
    posterior = torch.zeros((1, 1, 4, 4, 4))

    pred_handler = handler.wrap_predictions(posterior, base_names=['feature_a', 'feature_b'])
    df_res = pred_handler.to_evaluation_df(history=handler, start_idx=0)

    # GATES
    assert isinstance(df_res.index, pd.MultiIndex)
    assert df_res.index.names == ['month_id', 'priogrid_gid']
    assert "feature_a" in df_res.columns # The Actual
    assert "pred_lr_feature_a" in df_res.columns # The Prediction
    assert not any("INTERNAL" in col for col in df_res.columns)

def test_gate_15_geographic_anchoring():
    """Assert row_offset correctly anchors the grid."""
    df = pd.DataFrame({
        'month_id': [1], 'priogrid_gid': [1],
        'row': [10], 'col': [20], # Matches offsets exactly
        'feature_a': [1.0], 'feature_b': [1.0]
    })
    handler = VolumeHandler.from_df(df, PHYSICS_CFG)
    data = handler.data # [T, H, W, C]

    # Coordinates (10, 20) with offsets (10, 20) should map to grid index (0, 0)
    # But wait, it is flipped (North-Up)
    # Row 0 in input (bottom) becomes Row 3 in North-Up (top)
    # So we check index [0, 3, 0, :]
    assert data[0, 3, 0, handler.channel_map.index('feature_a')] == 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
