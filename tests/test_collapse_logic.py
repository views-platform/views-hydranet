
import numpy as np
import pytest
from views_hydranet.utils.volume_handler import VolumeHandler

def test_jensens_inequality_sequence():
    """
    Formal Proof: Verifies that 'Inverse -> Collapse' (Arithmetic Mean) 
    yields a different (and scientifically correct) result than 
    'Collapse -> Inverse' (Geometric Mean).    
    Data: Two samples, 10 and 100 (Raw Count Space).
    Transformation: Log1p (Natural Log(x + 1)).
    """
    
    # 1. Setup Data
    raw_samples = np.array([10.0, 100.0])
    # Model predicts in Log Space
    model_output_log_space = np.log1p(raw_samples) 
    # [2.397, 4.615]
    
    # Create VolumeHandler with this data (1 Time, 1 Height, 1 Width, 1 Channel, 2 Samples)
    # Shape: (T=1, H=1, W=1, C=1, S=2)
    data_5d = model_output_log_space.reshape(1, 1, 1, 1, 2)
    
    vh = VolumeHandler(
        data=data_5d,
        axes=("T", "H", "W", "C", "S"),
        channel_map=["pred_sb_INTERNAL_SIGNAL"],
        time_col="month_id",
        id_col="pg_id",
        spatial_cols=("row", "col"),
        spatial_offset=(0,0)
    )

    # --- PATH A: The HydraNet Manager Path (Correct) ---
    # 1. Inverse Transform FIRST (Simulated)
    # The Manager calls scaler.inverse_transform_volume(vh) here.
    # We simulate this by modifying the data in place (or creating new VH).
    data_raw = np.expm1(vh.data) # Inverse of log1p
    
    vh_raw = VolumeHandler(
        data=data_raw,
        axes=vh.axes,
        channel_map=vh.channel_map,
        time_col=vh.time_col,
        id_col=vh.id_col,
        spatial_cols=vh.spatial_cols,
        spatial_offset=vh.spatial_offset
    )
    
    # 2. Collapse SECOND
    vh_collapsed_A = vh_raw.collapse_to_point(method="mean")
    result_A = vh_collapsed_A.data.flatten()[0]
    
    # Expected: (10 + 100) / 2 = 55.0
    expected_A = 55.0
    np.testing.assert_allclose(result_A, expected_A, rtol=1e-5)
    
    
    # --- PATH B: The Wrong Path (Geometric Mean) ---
    # 1. Collapse FIRST (in Log Space)
    vh_collapsed_B_log = vh.collapse_to_point(method="mean")
    
    # 2. Inverse Transform SECOND
    result_B_log = vh_collapsed_B_log.data.flatten()[0]
    result_B = np.expm1(result_B_log)
    
    # Expected: expm1( (log1p(10) + log1p(100)) / 2 ) 
    # = expm1( 3.506 ) = 32.32
    # This is roughly the geometric mean of 11 and 101, minus 1.
    expected_B = np.expm1(np.mean(model_output_log_space))
    np.testing.assert_allclose(result_B, expected_B, rtol=1e-5)
    
    # --- CONCLUSION ---
    # The results must be significantly different.
    # 55.0 vs 32.32
    assert abs(result_A - result_B) > 10.0
    print(f"\nSUCCESS: Jensen's Inequality Proven.")
    print(f"Path A (Inverse->Collapse): {result_A:.4f} (Arithmetic Mean)")
    print(f"Path B (Collapse->Inverse): {result_B:.4f} (Geometric Mean)")

if __name__ == "__main__":
    test_jensens_inequality_sequence()
