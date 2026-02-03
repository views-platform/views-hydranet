
import pytest
import pandas as pd
import numpy as np
from views_hydranet.utils.feature_scaler import FeatureScaler

class TestFeatureScaler:
    """
    Rigorously verifies ADR 019: FeatureScaler Specification.
    Ensures stateful transformations and bit-perfect reversibility.
    """

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "sb": [0.0, 10.0, 100.0],
            "ns": [1.0, 2.0, 3.0],
            "id": [1, 2, 3]
        })

    @pytest.fixture
    def scaler_config(self):
        return {
            "log1p": ["sb"],
            "asinh": ["ns"]
        }

    def test_forward_transform_math(self, sample_df, scaler_config):
        """Verify that forward math matches numpy standards."""
        scaler = FeatureScaler(scaler_config)
        df_semantic = scaler.fit_transform(sample_df)
        
        # sb: log1p(100) approx 4.6151
        assert np.isclose(df_semantic["sb"].iloc[2], np.log1p(100.0))
        # ns: arcsinh(3) approx 1.8184
        assert np.isclose(df_semantic["ns"].iloc[2], np.arcsinh(3.0))
        # id: should remain untouched
        assert df_semantic["id"].iloc[2] == 3

    def test_bit_perfect_reversibility(self, sample_df, scaler_config):
        """Verify that Semantic -> Raw roundtrip is lossless."""
        scaler = FeatureScaler(scaler_config)
        
        df_semantic = scaler.fit_transform(sample_df)
        df_raw = scaler.inverse_transform(df_semantic)
        
        pd.testing.assert_frame_equal(sample_df, df_raw)

    def test_stateful_gate_enforcement(self, sample_df, scaler_config):
        """Falsification: inverse_transform must fail if not fitted."""
        scaler = FeatureScaler(scaler_config)
        
        with pytest.raises(RuntimeError, match="Contract Violation: Must be FITTED"):
            scaler.inverse_transform(sample_df)

    def test_fail_loud_on_missing_columns(self, sample_df):
        """Falsification: Scaler must crash if a configured column is missing."""
        bad_config = {"log1p": ["ghost_column"]}
        scaler = FeatureScaler(bad_config)
        
        with pytest.raises(ValueError, match="Requested column 'ghost_column' missing"):
            scaler.fit_transform(sample_df)

    def test_one_shot_gate(self, sample_df, scaler_config):
        """Falsification: fit_transform cannot be called twice."""
        scaler = FeatureScaler(scaler_config)
        scaler.fit_transform(sample_df)
        
        with pytest.raises(RuntimeError, match="already fitted"):
            scaler.fit_transform(sample_df)
