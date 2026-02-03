
import pytest
import torch
import torch.nn as nn
import numpy as np
from views_hydranet.utils.hydranet_inference import HydraNetInference

class MockModel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.base = channels
        # Output 6 channels (3 reg, 3 class)
        self.conv = nn.Conv2d(channels, channels * 2, 3, padding=1)
    def forward(self, x, h):
        out = self.conv(x)
        # Split into 3+3
        reg, cl = out[:, :3], out[:, 3:]
        return reg, cl, h
    def init_hTtime(self, hidden_channels, H, W):
        return torch.zeros((1, hidden_channels, H, W))

def test_gate_16_bootstrap_logic():
    """Assert model bootstraps from history when in_sample_seq_len is 0."""
    config = {"time_steps": 1, "input_channels": 3, "n_posterior_samples": 1}
    model = MockModel(3)
    inference = HydraNetInference(model, config, device="cpu")
    
    # [Batch=1, Time=1, Channel=3, H=4, W=4]
    full_tensor = torch.ones((1, 1, 3, 4, 4))
    
    # is_evaluation=True, full_seq_len = 1-1 = 0? 
    # No, let's use is_evaluation=False
    # full_seq_len = 1-1 + 1 = 1. in_sample_seq_len = 1-1 = 0.
    mag, prob = inference.predict(full_tensor, sample_idx=0, is_evaluation=False)
    
    assert mag.shape == (1, 3, 4, 4)
    assert not np.isnan(mag).any()

def test_gate_17_graph_detachment():
    """Verify code uses .detach() to prevent graph accumulation (OOM fix)."""
    import inspect
    import views_hydranet.utils.hydranet_inference as hi
    source = inspect.getsource(hi.HydraNetInference.predict)
    assert "t0 = t1_pred.detach()" in source

def test_gate_18_19_explosion_and_shapes():
    """Assert panic check triggers on infinity."""
    config = {"time_steps": 2, "input_channels": 3, "n_posterior_samples": 1}
    
    class ExplodingModel(MockModel):
        def forward(self, x, h):
            return torch.tensor([float('inf')]), torch.tensor([0.0]), h
            
    inference = HydraNetInference(ExplodingModel(3), config, device="cpu")
    full_tensor = torch.ones((1, 1, 3, 4, 4))
    
    mag, prob = inference.predict(full_tensor, sample_idx=0, is_evaluation=False)
    assert np.isnan(mag).all()

def test_gate_20_posterior_depth():
    """Assert 6-channel concatenation (ADR 020)."""
    config = {
        "time_steps": 1, "input_channels": 3, "n_posterior_samples": 2,
        "steps": [1]
    }
    from views_hydranet.utils.volume_handler import VolumeHandler
    handler = VolumeHandler(
        data=np.zeros((2, 4, 4, 3)), axes=("T", "H", "W", "C"),
        channel_map=["a", "b", "c"], time_col="t", id_col="i", spatial_cols=("r","c"),
        feature_cols=("a","b","c")
    )
    
    model = MockModel(3)
    inference = HydraNetInference(model, config, device="cpu")
    
    # 2 samples, 1 time step
    post, _ = inference.generate_posterior_samples(handler, is_evaluation=True)
    
    # Shape: (Time, H, W, Channels, Samples)
    # Channels should be 6 (3 reg + 3 class)
    assert post.shape == (1, 4, 4, 6, 2)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
