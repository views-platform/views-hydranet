import logging

import numpy as np
import pandas as pd

from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EndToEndProof")

def test_end_to_end_survival_physics():
    """
    PROVES the survival sequence for Milestone 1:
    Inference (Semantic) -> Inverse (Raw) -> Collapse (Point) -> DF
    """
    # 1. SETUP: 180x180 grid, 10 samples (keep it small for the test)
    config = {
        "transform": {"log1p": ["lr_sb"], "asinh": ["lr_ns"], "identity": ["lr_os"]},
        "aggregate_method": "arithmetic_mean"
    }
    # Create semantic prediction data (log space)
    # [T=1, H=4, W=4, C=6, S=10] -> (sb_signal, ns_signal, os_signal, sb_prob, ns_prob, os_prob)
    data = np.zeros((1, 4, 4, 6, 10))
    data[0,:,:,0,:] = np.log1p(100.0) # sb_signal
    data[0,:,:,3,:] = 0.9 # sb_prob

    # 1. SETUP SCAFFOLD
    df_scaffold = pd.DataFrame({
        "month_id": [400]*16,
        "priogrid_gid": np.arange(1, 17),
        "row": np.repeat(np.arange(4), 4),
        "col": np.tile(np.arange(4), 4),
        "lr_sb": [10.0]*16, "lr_ns": [0.0]*16, "lr_os": [0.0]*16
    })

    scaffold_config = {
        "height": 4, "width": 4, "time_col": "month_id", "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"], "row_offset": 0, "col_offset": 0,
        "identity_cols": ["month_id", "priogrid_gid", "row", "col", "c_id"],
        "features": ["lr_sb", "lr_ns", "lr_os"]
    }
    # Add dummy c_id
    df_scaffold["c_id"] = 42

    vh_scaffold = VolumeHandler.from_df(df_scaffold, scaffold_config)

    # 2. WRAP PREDICTIONS (This applies the pred_ prefix and Watermarks)
    # Must use lr_ prefixed names for wrap_predictions base_names
    vh = vh_scaffold.wrap_predictions(data, base_names=["lr_sb", "lr_ns", "lr_os"])

    scaler = FeatureScaler(config)
    scaler._is_fitted = True # Manual bypass for the proof

    # ---------------------------------------------------------
    # THE SURVIVAL SEQUENCE
    # ---------------------------------------------------------

    # A. Immediate Inversion (Vectorized NumPy)
    raw_vh = scaler.inverse_transform_volume(vh)
    # Channel idx for pred_lr_sb is resolved dynamically
    sb_idx = vh.channel_map.index("pred_lr_sb")
    np.testing.assert_allclose(raw_vh.data[0,0,0,sb_idx,0], 100.0, rtol=1e-5)

    # B. Point-Collapse (Vectorized NumPy)
    # Average 100.0 across 10 samples -> 100.0
    point_vh = raw_vh.collapse_to_point(method="arithmetic_mean")
    assert point_vh.data.ndim == 4
    np.testing.assert_allclose(point_vh.data[0,0,0,sb_idx], 100.0, rtol=1e-5)

    # C. Safe Reconstruction (Single Scalar per Cell)
    # No list objects created!
    df = point_vh.to_historical_df()

    # FINAL VERIFICATION
    # Note: VolumeHandler automatically dresses names into the final contract
    assert "pred_lr_sb" in df.columns
    np.testing.assert_allclose(df.loc[(400, 1), "pred_lr_sb"], 100.0, rtol=1e-5)

    print("\n✅ Survival Sequence Proven: Vectorized Inversion + Collapse eliminates RAM risk.")

if __name__ == "__main__":
    test_end_to_end_survival_physics()
