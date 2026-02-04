import pytest
import numpy as np
import pandas as pd
import torch
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.feature_scaler import FeatureScaler

# FALSIFICATION CONFIG
CFG = {
    "transform": {"log1p": ["sb"], "asinh": ["ns"]}, # 'os' is intentionally identity
    "aggregate_method": "arithmetic_mean"
}

class TestNukeProofAudit:
    """Popperian Audit: Attempting to falsify the hardening logic."""

    def test_gate_a_heterogeneous_collision(self):
        """
        Gate A: Mixed Scale Collision.
        Verifies that log1p and asinh don't interfere when inverting a volume.
        """
        # [T=1, H=1, W=1, C=3, S=1]
        data = np.zeros((1, 1, 1, 3, 1))
        data[0,0,0,0,0] = np.log1p(10.0)   # SB: 10
        data[0,0,0,1,0] = np.arcsinh(10.0) # NS: 10
        data[0,0,0,2,0] = 10.0             # OS: 10 (Identity)
        
        vh = VolumeHandler(
            data=data, axes=("T", "H", "W", "C", "S"),
            channel_map=["sb_INTERNAL_SIGNAL", "ns_INTERNAL_SIGNAL", "os_INTERNAL_SIGNAL"],
            time_col="t", id_col="i", spatial_cols=("y", "x")
        )
        
        scaler = FeatureScaler(CFG)
        scaler._is_fitted = True
        
        raw_vh = scaler.inverse_transform_volume(vh)
        
        # All should be 10.0 now
        np.testing.assert_allclose(raw_vh.data[0,0,0,0,0], 10.0, rtol=1e-5)
        np.testing.assert_allclose(raw_vh.data[0,0,0,1,0], 10.0, rtol=1e-5)
        np.testing.assert_allclose(raw_vh.data[0,0,0,2,0], 10.0, rtol=1e-5)

    def test_gate_b_actuals_corruption(self):
        """
        Gate B: Actuals Corruption.
        Prove that ACTUAL_INTERNAL_ columns are NOT double-inversed.
        """
        data = np.zeros((1, 1, 1, 2, 1))
        data[0,0,0,0,0] = 5.0 # This is an Actual (already raw)
        data[0,0,0,1,0] = np.log1p(100.0) # This is a prediction
        
        vh = VolumeHandler(
            data=data, axes=("T", "H", "W", "C", "S"),
            channel_map=["ACTUAL_INTERNAL_sb", "sb_INTERNAL_SIGNAL"],
            time_col="t", id_col="i", spatial_cols=("y", "x")
        )
        
        scaler = FeatureScaler(CFG)
        scaler._is_fitted = True
        
        raw_vh = scaler.inverse_transform_volume(vh)
        
        # Actual must remain 5.0. If it was double-inversed (expm1(5)), it would be ~147
        assert raw_vh.data[0,0,0,0,0] == 5.0
        # Prediction must be 100.0
        np.testing.assert_allclose(raw_vh.data[0,0,0,1,0], 100.0, rtol=1e-5)

    def test_gate_c_axis_permutation_robustness(self):
        """
        Gate C: Axis Permutation.
        Prove that collapse_to_point uses semantic labels, not positions.
        """
        # Permuted layout: [S, C, T, H, W]
        data = np.random.rand(10, 2, 1, 4, 4)
        vh = VolumeHandler(
            data=data, axes=("S", "C", "T", "H", "W"), # CRAZY LAYOUT
            channel_map=["a", "b"],
            time_col="t", id_col="i", spatial_cols=("y", "x")
        )
        
        # This will fail if I used 'axis=4' or 'axis=-1' hardcoded
        point_vh = vh.collapse_to_point(method="mean")
        
        assert point_vh.data.shape == (2, 1, 4, 4)
        assert "S" not in point_vh._metadata.axes

    def test_gate_e_symmetry_leak(self):
        """
        Gate E: Multi-Task Symmetry Leak.
        Prove that probabilities (_INTERNAL_PROB) are NEVER inverse-transformed.
        """
        data = np.zeros((1, 1, 1, 2, 1))
        data[0,0,0,0,0] = np.log1p(10.0) # Signal
        data[0,0,0,1,0] = 0.8            # Probability
        
        vh = VolumeHandler(
            data=data, axes=("T", "H", "W", "C", "S"),
            channel_map=["sb_INTERNAL_SIGNAL", "sb_INTERNAL_PROB"],
            time_col="t", id_col="i", spatial_cols=("y", "x")
        )
        
        scaler = FeatureScaler(CFG)
        scaler._is_fitted = True
        
        raw_vh = scaler.inverse_transform_volume(vh)
        
        # Prob must stay 0.8. If expm1 was applied, it would be ~1.22 (illegal)
        assert raw_vh.data[0,0,0,1,0] == 0.8

if __name__ == "__main__":
    pytest.main([__file__, "-v"])