
import pytest
import pandas as pd
import numpy as np
from views_hydranet.utils.feature_scaler import FeatureScaler

def test_gate_1_math_parity():
    """Assert bit-perfect reversibility for asinh/sinh."""
    config = {
        "transforms": {
            "asinh": ["feature_a"]
        }
    }
    scaler = FeatureScaler(config)
    df = pd.DataFrame({"feature_a": [0.0, 10.0, 1000.0]})
    
    semantic = scaler.fit_transform(df)
    # asinh(10) is ~2.99
    assert semantic["feature_a"].iloc[1] < 10.0
    
    recovered = scaler.inverse_transform(semantic)
    np.testing.assert_allclose(df["feature_a"], recovered["feature_a"], rtol=1e-7)

def test_gate_2_loud_failure():
    """Assert ValueError on missing columns."""
    config = {
        "transforms": {
            "log1p": ["missing_feature"]
        }
    }
    scaler = FeatureScaler(config)
    df = pd.DataFrame({"other": [1.0]})
    
    with pytest.raises(ValueError, match="Requested feature 'missing_feature' missing"):
        scaler.fit_transform(df)

def test_gate_3_state_lock():
    """Assert RuntimeError if inverse called before fit."""
    scaler = FeatureScaler({"transforms": {"identity": ["a"]}})
    df = pd.DataFrame({"a": [1.0]})
    
    with pytest.raises(RuntimeError, match="Must be FITTED"):
        scaler.inverse_transform(df)

def test_gate_4_one_shot_law():
    """Assert RuntimeError if fit called twice."""
    scaler = FeatureScaler({"transforms": {"identity": ["a"]}})
    df = pd.DataFrame({"a": [1.0]})
    
    scaler.fit_transform(df)
    with pytest.raises(RuntimeError, match="already fitted"):
        scaler.fit_transform(df)

def test_gate_5_subset_tolerance():
    """Assert inverse_transform skips missing columns (required by Manager sub-setting)."""
    config = {
        "transforms": {
            "log1p": ["feature_a", "feature_b"]
        }
    }
    scaler = FeatureScaler(config)
    df = pd.DataFrame({"feature_a": [1.0], "feature_b": [1.0]})
    
    scaler.fit_transform(df)
    
    # Simulate a downstream DF that only has feature_a
    subset_df = pd.DataFrame({"feature_a": [0.693147]}) # log1p(1)
    
    # Should NOT crash, should skip feature_b
    recovered = scaler.inverse_transform(subset_df)
    assert "feature_a" in recovered.columns
    np.testing.assert_allclose(recovered["feature_a"], [1.0], rtol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
