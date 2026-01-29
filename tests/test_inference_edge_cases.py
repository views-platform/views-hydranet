import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from views_hydranet.utils.hydranet_inference import HydraNetInference

@pytest.fixture
def inference_engine():
    config = {
        "time_steps": 3,
        "test_samples": 1,
        "input_channels": 8,
        "freeze_h": "none"
    }
    # Create a mock that pretends to be a Module
    model = MagicMock()
    model.__class__ = torch.nn.Module
    model.base = 32
    # Standard forward return: out_reg, out_class, h
    model.return_value = (torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 16), torch.zeros(1, 32, 16, 16))
    model.init_hTtime.return_value = torch.zeros(1, 32, 16, 16)
    
    return HydraNetInference(model, config, device="cpu")

def test_inference_panic_switch_on_nan(inference_engine):
    """Verify that the panic switch catches NaNs and fills the rest with NaNs."""
    # Force the model to return NaN during the autoregressive step
    # We need to distinguish between in-sample and out-of-sample steps
    # seq_len=5, time_steps=3 -> in_sample=1, out_of_sample=3
    input_tensor = torch.zeros(1, 5, 8, 16, 16)
    
    # Create a side effect that returns NaN only on the 2nd call (first out-of-sample step)
    nan_tensor = torch.full((1, 3, 16, 16), float('nan'))
    standard_return = (torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 16), torch.zeros(1, 32, 16, 16))
    nan_return = (nan_tensor, torch.zeros(1, 3, 16, 16), torch.zeros(1, 32, 16, 16))
    
    inference_engine.model.side_effect = [standard_return, nan_return, standard_return, standard_return]
    
    mags, probs = inference_engine.predict(input_tensor, sample_idx=0, is_evaluation=True)
    
    # All months from the explosion point onwards should be NaN
    assert np.isnan(mags).any()
    # The first month was fine (in-sample), the second month exploded (nan_return)
    # mags shape is [time_steps, channels, H, W] -> [3, 8, 16, 16]
    assert np.isnan(mags[0]).all() # The first month out-of-sample is the one that returned NaN

def test_freeze_h_all_strategy(inference_engine):
    """Verify that 'all' strategy does not update the hidden state."""
    inference_engine.config["freeze_h"] = "all"
    t0 = torch.randn(1, 8, 16, 16)
    h_init = torch.randn(1, 32, 16, 16)
    
    # Forward returns a NEW h, but execute_freeze_h_option should discard it
    new_h_from_model = torch.randn(1, 32, 16, 16)
    inference_engine.model.return_value = (torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 16), new_h_from_model)
    
    _, _, h_final = inference_engine.execute_freeze_h_option(t0, h_init)
    
    # h_final should be EQUAL to h_init, NOT new_h_from_model
    assert torch.equal(h_final, h_init)

def test_freeze_h_hs_strategy(inference_engine):
    """Verify that 'hs' strategy only freezes the first half of the hidden state."""
    inference_engine.config["freeze_h"] = "hs"
    num_channels = 32
    t0 = torch.randn(1, 8, 16, 16)
    
    # h = [hs (16), hl (16)]
    h_init = torch.zeros(1, num_channels, 16, 16)
    h_init[:, :16, :, :] = 1.0 # Frozen hs
    h_init[:, 16:, :, :] = 2.0 # hl to be updated
    
    # Model returns 9.0 for everything
    h_new = torch.full((1, num_channels, 16, 16), 9.0)
    inference_engine.model.return_value = (torch.zeros(1, 3, 16, 16), torch.zeros(1, 3, 16, 16), h_new)
    
    _, _, h_final = inference_engine.execute_freeze_h_option(t0, h_init)
    
    # First 16 channels (hs) should remain 1.0
    assert torch.all(h_final[:, :16, :, :] == 1.0)
    # Last 16 channels (hl) should be the NEW values (9.0)
    assert torch.all(h_final[:, 16:, :, :] == 9.0)
