
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
    'features': [ 'lr_feature_a',  'lr_feature_b'],
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
         'lr_feature_a': [1.0, 2.0],  'lr_feature_b': [0.0, 0.0]
    })
    handler = VolumeHandler.from_df(df, PHYSICS_CFG)
    tensor = handler.to_pytorch(torch.device('cpu'), include_identities=False)

    # 2 features, T=1, H=4, W=4
    assert tensor.shape == (1, 1, 2, 4, 4)

def test_gate_12_6_head_dressing():
    """Assert wrap_predictions correctly dresses 6 semantic heads."""
    posterior = torch.ones((1, 1, 4, 4, 4)) # T=1, C=4, H=4, W=4 (2 reg, 2 class)
    base_names = [ 'lr_a',  'lr_b']

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
         'lr_feature_a': [10.0, 20.0],  'lr_feature_b': [5.0, 5.0]
    })
    handler = VolumeHandler.from_df(df_hist, PHYSICS_CFG)

    # Simulate a 4-channel prediction (2 reg, 2 class)
    # feature_a_raw, feature_b_raw, feature_a_prob, feature_b_prob
    posterior = torch.zeros((1, 1, 4, 4, 4))

    pred_handler = handler.wrap_predictions(posterior, base_names=[ 'lr_feature_a',  'lr_feature_b'])
    df_res = pred_handler.to_evaluation_df(history=handler, start_idx=0)

    # GATES
    assert isinstance(df_res.index, pd.MultiIndex)
    assert df_res.index.names == ['month_id', 'priogrid_gid']
    assert "lr_feature_a" in df_res.columns # The Actual
    assert "pred_lr_feature_a" in df_res.columns # The Prediction
    assert not any("INTERNAL" in col for col in df_res.columns)

def test_gate_15_geographic_anchoring():
    """Assert row_offset correctly anchors the grid."""
    df = pd.DataFrame({
        'month_id': [1], 'priogrid_gid': [1],
        'row': [10], 'col': [20], # Matches offsets exactly
         'lr_feature_a': [1.0],  'lr_feature_b': [1.0]
    })
    handler = VolumeHandler.from_df(df, PHYSICS_CFG)
    data = handler.data # [T, H, W, C]

    # Coordinates (10, 20) with offsets (10, 20) should map to grid index (0, 0)
    # But wait, it is flipped (North-Up)
    # Row 0 in input (bottom) becomes Row 3 in North-Up (top)
    # So we check index [0, 3, 0, :]
    assert data[0, 3, 0, handler.channel_map.index( 'lr_feature_a')] == 1.0

def test_gate_17_negative_offset_rejection():
    """
    RED GATE: VolumeHandler must raise BEFORE writing when row/col offsets
    produce negative indices. Two cases must both be caught:

    Case A — "Loud": negative index exceeds numpy array bounds → currently
              raises IndexError deep in numpy (wrong layer, wrong type).
              After the guard: raises ValueError early with a clear message.

    Case B — "Silent" (the real danger): negative index is within numpy's
              valid wrap-around range (e.g., -87 in a height=180 array).
              numpy writes to the WRONG location and raises nothing.
              After the guard: raises ValueError early.

    This is Phase 1, Step 1 of the Test Remediation Plan (2026-02-19).
    """
    # --- Case A: Loud failure (small grid, index well out of bounds) ---
    # row=20, row_offset=50 → r_idx = 20-50 = -30. height=4, so -30 < -4 → IndexError
    bad_row_cfg = dict(PHYSICS_CFG, row_offset=50, col_offset=20)
    df_bad_row = pd.DataFrame({
        'month_id': [1], 'priogrid_gid': [1],
        'row': [20], 'col': [20],
        'lr_feature_a': [1.0], 'lr_feature_b': [1.0],
    })
    with pytest.raises(ValueError, match="row"):
        VolumeHandler.from_df(df_bad_row, bad_row_cfg)

    # --- Case B: Silent failure (large grid, index wraps without error) ---
    # Mimics the production scenario: height=180, row_offset=87, but data
    # starts at row=0 (local coords). r_idx = 0-87 = -87. In a 180-tall array,
    # -87 is a VALID numpy index (wraps to 93). No IndexError. Data is silently
    # written 87 rows from the bottom instead of the top. Map is inverted.
    silent_cfg = {
        'time_col': 'month_id', 'id_col': 'priogrid_gid',
        'spatial_cols': ['row', 'col'],
        'identity_cols': ['month_id', 'priogrid_gid'],
        'features': ['lr_feature_a'],
        'row_offset': 87,   # <-- typical Africa offset
        'col_offset': 310,
        'height': 180,
        'width': 180,
    }
    df_silent = pd.DataFrame({
        'month_id': [1], 'priogrid_gid': [1],
        'row': [0],   # local coord — r_idx = 0-87 = -87, wraps to 93 silently
        'col': [310],
        'lr_feature_a': [1.0],
    })
    with pytest.raises(ValueError, match="row"):
        VolumeHandler.from_df(df_silent, silent_cfg)

    # --- Case C: Col offset produces negative c_idx ---
    bad_col_cfg = dict(PHYSICS_CFG, row_offset=10, col_offset=50)
    df_bad_col = pd.DataFrame({
        'month_id': [1], 'priogrid_gid': [1],
        'row': [10], 'col': [10],  # col=10, offset=50 → c_idx=-40
        'lr_feature_a': [1.0], 'lr_feature_b': [1.0],
    })
    with pytest.raises(ValueError, match="col"):
        VolumeHandler.from_df(df_bad_col, bad_col_cfg)

    # --- Case D: Span violation (Positive index out of bounds) ---
    # row=15, row_offset=10 → r_idx=5. height=4, so 5 >= 4 → ValueError
    span_cfg = dict(PHYSICS_CFG, height=4)
    df_span = pd.DataFrame({
        'month_id': [1], 'priogrid_gid': [1],
        'row': [15], 'col': [20],
        'lr_feature_a': [1.0], 'lr_feature_b': [1.0],
    })
    with pytest.raises(ValueError, match="Span Violation"):
        VolumeHandler.from_df(df_span, span_cfg)

    # --- Boundary: exact match must NOT raise ---
    # row_offset == df.row.min() → r_idx=0, valid.
    exact_cfg = dict(PHYSICS_CFG, row_offset=10, col_offset=20)
    df_exact = pd.DataFrame({
        'month_id': [1], 'priogrid_gid': [1],
        'row': [10], 'col': [20],
        'lr_feature_a': [1.0], 'lr_feature_b': [1.0],
    })
    VolumeHandler.from_df(df_exact, exact_cfg)  # must not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
