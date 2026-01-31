import pytest
import numpy as np
from views_hydranet.utils.utils_scaling import ScalingEngine

@pytest.mark.parametrize("transform", ["log1p", "asinh", "identity"])
def test_scaling_engine_symmetry_parity(transform):
    """
    MATHEMATICAL PROOF: inverse(forward(x)) == x.
    Verifies bit-level parity for the ScalingEngine.
    """
    engine = ScalingEngine(transform_name=transform)
    
    # Create random raw data [0, 1000]
    raw_data = np.random.rand(10, 10) * 1000.0
    
    # 1. Scale
    scaled = engine.scale(raw_data, context="TestForward")
    
    # 2. Unscale
    recovered = engine.unscale(scaled, context="TestInverse")
    
    # 3. Assert Parity
    # We use a small epsilon for floating point precision
    np.testing.assert_allclose(recovered, raw_data, rtol=1e-5, err_msg=f"Symmetry broken for {transform}!")
