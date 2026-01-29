import pytest
import torch
import numpy as np
from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4

def test_architecture_instantiation():
    """Verify that the model can be instantiated with valid parameters."""
    model = HydraBNUNet06_LSTM4(
        input_channels=8,
        total_hidden_channels=32,
        output_channels=1,
        dropout_rate=0.1
    )
    assert isinstance(model, torch.nn.Module)
    assert model.base == 32

def test_architecture_hidden_state_init():
    """Verify that hidden state initialization produces correct shapes and types."""
    model = HydraBNUNet06_LSTM4(8, 32, 1, 0.1)
    H, W = 16, 16
    h = model.init_hTtime(hidden_channels=32, H=H, W=W)
    
    assert h.shape == (1, 32, H, W)
    assert h.dtype == torch.float64
    assert torch.all(h == 0)

def test_architecture_forward_pass_shapes():
    """Verify that the forward pass produces correctly shaped outputs for regression and classification."""
    input_ch = 8
    hidden_ch = 32
    out_ch = 1 # Per head, total 3
    H, W = 16, 16
    
    model = HydraBNUNet06_LSTM4(input_ch, hidden_ch, out_ch, 0.1).float()
    x = torch.randn((1, input_ch, H, W)).float()
    h = model.init_hTtime(hidden_ch, H, W).float()
    
    out_reg, out_class, new_h = model(x, h)
    
    # 3 heads (sb, ns, os) concatenated
    assert out_reg.shape == (1, 3, H, W)
    assert out_class.shape == (1, 3, H, W)
    assert new_h.shape == (1, hidden_ch, H, W)

def test_architecture_recurrent_state_evolution():
    """Verify that the hidden state actually changes after a forward pass."""
    model = HydraBNUNet06_LSTM4(8, 32, 1, 0.0).float() # No dropout for determinism
    x = torch.randn((1, 8, 16, 16)).float()
    h_init = model.init_hTtime(32, 16, 16).float()
    
    _, _, h_next = model(x, h_init)
    
    # Hidden state should no longer be zeros
    assert not torch.all(h_next == 0)
    assert h_next.shape == h_init.shape

def test_architecture_multi_batch_support():
    """Verify that the architecture supports batch sizes > 1."""
    # Note: init_hTtime currently hardcodes batch size 1 in its return shape
    # We test if the forward pass itself is batch-agnostic if we provide the right h
    batch_size = 4
    input_ch = 8
    hidden_ch = 32
    H, W = 16, 16
    
    model = HydraBNUNet06_LSTM4(input_ch, hidden_ch, 1, 0.1).float()
    x = torch.randn((batch_size, input_ch, H, W)).float()
    h = torch.zeros((batch_size, hidden_ch, H, W)).float()
    
    out_reg, out_class, new_h = model(x, h)
    
    assert out_reg.shape[0] == batch_size
    assert new_h.shape[0] == batch_size
