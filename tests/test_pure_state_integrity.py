"""
Team Audit Suite for ADR 032: Authoritative Spatiotemporal Output Schema.
Verifies Accuracy (Green), Robustness (Beige), and Invincibility (Red).
"""

import numpy as np
import pandas as pd
import pytest
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.data_sniffer import DataSniffer

@pytest.fixture
def mock_config():
    return {
        "height": 10,
        "width": 10,
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "row_offset": 0,
        "col_offset": 0,
        "identity_cols": ["month_id", "priogrid_gid", "row", "col", "c_id"],
        "features": ["lr_sb_best"]
    }

@pytest.fixture
def input_df():
    """Bit-perfect input data scaffold."""
    data = {
        "month_id": [500, 500, 501],
        "priogrid_gid": [100, 101, 100],
        "row": [1, 1, 1],
        "col": [1, 2, 1],
        "c_id": [42, 42, 42],
        "lr_sb_best": [10.0, 0.0, 5.0]
    }
    return pd.DataFrame(data)

# --- GREEN TEAM: ACCURACY ---

def test_green_roundtrip_parity(input_df, mock_config):
    """Prove bit-perfect round-trip: DF -> Volume -> DF."""
    # 1. Ingest
    handler = VolumeHandler.from_df(input_df, mock_config)
    
    # 2. Reconstruct (Historical)
    output_df = handler.to_historical_df()
    
    # 3. Verify via Sniffer
    sniffer = DataSniffer(mock_config)
    # This should pass without error
    sniffer.sniff_pure_state_parity(input_df, output_df)
    
    # Explicit check for bookkeeping identity
    # Use .loc for precise index-based lookup (month_id, priogrid_gid)
    assert output_df.loc[(500, 100), "c_id"] == 42
    assert "by_sb_best" in output_df.columns
    assert output_df.loc[(500, 100), "by_sb_best"] == 1.0 # 10.0 > 0
    assert output_df.loc[(500, 101), "by_sb_best"] == 0.0 # 0.0 > 0 is False

def test_green_schema_conformance(input_df, mock_config):
    """Verify prefix-based naming and suffix retirement."""
    handler = VolumeHandler.from_df(input_df, mock_config)
    
    # Create mock predictions: [T, H, W, C]
    # For a 10x10x2 grid, 1 task (sb) -> 2 heads (lr, by)
    mock_preds = np.random.rand(2, 10, 10, 2)
    
    # Wrap
    pred_handler = handler.wrap_predictions(mock_preds, ["sb_best"])
    
    # Reconstruct
    output_df = pred_handler.to_historical_df()
    
    # Verify Schema
    sniffer = DataSniffer(mock_config)
    sniffer.sniff_pure_state_schema(output_df, mock_config)
    
    assert "pred_lr_sb_best" in output_df.columns
    assert "pred_by_sb_best" in output_df.columns
    assert "pred_lr_sb_best_raw" not in output_df.columns # retired
    assert "pred_by_sb_best_prob" not in output_df.columns # retired

# --- BEIGE TEAM: ROBUSTNESS ---

def test_beige_missing_id_fails(input_df, mock_config):
    """Verify that providing a DF with missing role columns triggers an immediate crash."""
    bad_df = input_df.drop(columns=["month_id"])
    with pytest.raises(ValueError, match="Missing columns"):
        VolumeHandler.from_df(bad_df, mock_config)

def test_beige_misnamed_index_fails(input_df, mock_config):
    """Verify that misnamed index in Sniffer triggers error."""
    handler = VolumeHandler.from_df(input_df, mock_config)
    output_df = handler.to_historical_df()
    
    # Corrupt config for Sniffer
    bad_config = mock_config.copy()
    bad_config["time_col"] = "wrong_month"
    
    sniffer = DataSniffer(bad_config)
    with pytest.raises(ValueError, match="Index Mismatch"):
        sniffer.sniff_pure_state_schema(output_df, bad_config)

# --- RED TEAM: INVINCIBILITY ---

def test_red_shuffle_immunity(input_df, mock_config):
    """Prove that row-shuffling has zero impact on identity restoration."""
    # 1. Shuffle Input
    shuffled_df = input_df.sample(frac=1).reset_index(drop=True)
    assert not shuffled_df.equals(input_df) # verify shuffle
    
    # 2. Convert to Volume
    handler = VolumeHandler.from_df(shuffled_df, mock_config)
    
    # 3. Reconstruct
    output_df = handler.to_historical_df()
    
    # 4. Verify parity against ORIGINAL (pre-shuffle) input
    sniffer = DataSniffer(mock_config)
    sniffer.sniff_pure_state_parity(input_df, output_df)

def test_red_ocean_breach_prevention(input_df, mock_config):
    """Verify that non-land pixels are strictly excluded from output DF."""
    handler = VolumeHandler.from_df(input_df, mock_config)
    
    # Injected data at [0,0] which is NOT in input_df (indices were 1,1 and 1,2)
    # Even if predictions exist there, to_historical_df must mask it via the scaffold
    mock_preds = np.ones((2, 10, 10, 2))
    pred_handler = handler.wrap_predictions(mock_preds, ["sb_best"])
    
    output_df = pred_handler.to_historical_df()
    
    # The output row count must match input row count (Land only)
    assert len(output_df) == len(input_df)
    
    # Verify that no priogrid_gid=0 (default np.zeros value) exists in output
    assert (output_df.index.get_level_values("priogrid_gid") > 0).all()
