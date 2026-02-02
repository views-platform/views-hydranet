
import pytest
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

class TestStochasticIntegrity:
    """
    Rigorously verifies ADR 007 Section 3.4: Stochastic Integrity.
    Ensures that 5D posterior samples are preserved as lists in the DataFrame.
    """

    @pytest.fixture
    def stochastic_vh(self):
        # [T=1, H=2, W=2, C=1, S=10]
        # Data: A 5D volume with 10 samples per cell
        data = np.zeros((1, 2, 2, 1, 10))
        # Set specific values for Sample 0 and Sample 9 at cell (0,0)
        data[0, 0, 0, 0, 0] = 1.1
        data[0, 0, 0, 0, 9] = 9.9
        
        # We need a dummy ledger
        return VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C", "S"),
            channel_map=["pred_feat"],
            time_col="t", id_col="i", spatial_cols=["y", "x"],
            identity_cols=["t", "i"],
            feature_cols=["pred_feat"]
        )

    def test_reconstruction_preserves_samples_as_lists(self, stochastic_vh):
        """
        Falsification Test: Does the system collapse the 10 samples?
        Expected: A DataFrame where the 'pred_feat' column contains lists of length 10.
        """
        # Manually set the ID channel to 1 at (0,0) so it counts as Land
        # We need a provider or we can just mock the identity logic
        # For simplicity, let's use the handler itself as provider (Historical mode)
        
        # We need to hack the unit test because our __init__ validated channel map vs data.
        # stochastic_vh has 5 channels mapped? No, it has 1 mapped to 5D. 
        # Wait, the __init__ check: actual_channels = self._data.shape[c_idx]
        # c_idx for ("T", "H", "W", "C", "S") is 3. 
        # data.shape[3] is 1. channel_map is ["pred_feat"] (len 1).
        # THIS PASSES INIT.
        
        # Manually inject land identity
        # We need to add an id channel to the 5D data? No, reconstruction uses 
        # the provider's 4D identity scaffold.
        
        # 1. Create Identity Scaffold (4D)
        # [T=1, H=2, W=2, C=2] (t, i)
        id_data = np.zeros((1, 2, 2, 2))
        id_data[0, 0, 0, 1] = 555 # priogrid_gid at (0,0)
        scaffold = VolumeHandler(
            data=id_data,
            axes=("T", "H", "W", "C"),
            channel_map=["t", "i"],
            time_col="t", id_col="i", spatial_cols=["y", "x"],
            identity_cols=["t", "i"]
        )
        
        # 2. Reconstruct 5D Signal using 4D Scaffold
        df = stochastic_vh.to_evaluation_df(scaffold, start_idx=0)
        
        # 3. Assertions
        assert len(df) == 1
        val = df.iloc[0]["pred_feat"]
        
        # SPIRIT TEST: Is it a list?
        assert isinstance(val, list), f"FAIL: Expected list, got {type(val)}"
        
        # LENGTH TEST: Are all samples there?
        assert len(val) == 10, f"FAIL: Expected 10 samples, got {len(val)}"
        
        # VALUE TEST: No averaging occurred
        assert val[0] == 1.1
        assert val[9] == 9.9
        
    def test_wrap_predictions_no_collapse(self):
        """Verifies that wrap_predictions does not use np.mean()."""
        # [T=1, H=2, W=2, C=1, S=5]
        raw_5d = np.random.rand(1, 2, 2, 1, 5)
        
        parent = VolumeHandler(
            data=np.zeros((1, 2, 2, 1)),
            axes=("T", "H", "W", "C"),
            channel_map=["f"],
            time_col="t", id_col="i", spatial_cols=["y", "x"]
        )
        
        # Wrap it
        wrapped = parent.wrap_predictions(raw_5d, ["pred_f"])
        
        # Assert 5D preservation
        assert wrapped.data.ndim == 5
        assert wrapped.data.shape[-1] == 5
        assert "S" in wrapped.axes
