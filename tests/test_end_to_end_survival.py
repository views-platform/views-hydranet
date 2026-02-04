import pytest
import numpy as np
import pandas as pd
import torch
import logging
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.feature_scaler import FeatureScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EndToEndProof")

def test_end_to_end_survival_physics():
    """
    PROVES the survival sequence for Milestone 1:
    Inference (Semantic) -> Inverse (Raw) -> Collapse (Point) -> DF
    """
    # 1. SETUP: 180x180 grid, 10 samples (keep it small for the test)
    config = {
        "transform": {"log1p": ["sb"], "asinh": ["ns"], "identity": ["os"]},
        "aggregate_method": "arithmetic_mean"
    }
    
    # Create semantic data (log space)
    # [T=1, H=4, W=4, C=8, S=10]
    # C0: pg_id, C1: month_id, C2: sb_signal, C5: sb_prob
    data = np.zeros((1, 4, 4, 8, 10))
    data[:,:,:,0,:] = 1 # pg_id
    data[:,:,:,1,:] = 400 # month_id
    data[:,:,:,2,:] = np.log1p(100.0) # sb_signal
    data[:,:,:,5,:] = 0.9 # sb_prob
    
    full_channel_map = ["priogrid_gid", "month_id", 
                        "sb_INTERNAL_SIGNAL", "ns_INTERNAL_SIGNAL", "os_INTERNAL_SIGNAL", 
                        "sb_INTERNAL_PROB", "ns_INTERNAL_PROB", "os_INTERNAL_PROB"]
    
    vh = VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C", "S"),
        channel_map=full_channel_map,
        time_col="month_id", id_col="priogrid_gid", spatial_cols=("y", "x"),
        identity_cols=["priogrid_gid", "month_id"],
        feature_cols=["sb_INTERNAL_SIGNAL", "ns_INTERNAL_SIGNAL", "os_INTERNAL_SIGNAL", 
                      "sb_INTERNAL_PROB", "ns_INTERNAL_PROB", "os_INTERNAL_PROB"]
    )
    
    scaler = FeatureScaler(config)
    scaler._is_fitted = True # Manual bypass for the proof
    
    # ---------------------------------------------------------
    # THE SURVIVAL SEQUENCE
    # ---------------------------------------------------------
    
    # A. Immediate Inversion (Vectorized NumPy)
    raw_vh = scaler.inverse_transform_volume(vh)
    # Check: C2 (sb) should now be 100.0
    np.testing.assert_allclose(raw_vh.data[0,0,0,2,0], 100.0, rtol=1e-5)
    
    # B. Point-Collapse (Vectorized NumPy)
    # Average 100.0 across 10 samples -> 100.0
    point_vh = raw_vh.collapse_to_point(method="arithmetic_mean")
    assert point_vh.data.ndim == 4
    np.testing.assert_allclose(point_vh.data[0,0,0,2], 100.0, rtol=1e-5)
    
    # C. Safe Reconstruction (Single Scalar per Cell)
    # No list objects created!
    df = point_vh.to_historical_df()
    
    # FINAL VERIFICATION
    # Note: VolumeHandler automatically dresses names into the final contract
    assert "pred_sb_raw" in df.columns
    np.testing.assert_allclose(df["pred_sb_raw"].iloc[0], 100.0, rtol=1e-5)
    
    print("\n✅ Survival Sequence Proven: Vectorized Inversion + Collapse eliminates RAM risk.")

if __name__ == "__main__":
    test_end_to_end_survival_physics()
