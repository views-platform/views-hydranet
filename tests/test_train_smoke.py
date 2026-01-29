import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from views_hydranet.train.train_model import training_loop

def test_training_loop_smoke():
    """
    Verify that the training loop can run a single sample/batch without crashing.
    This test mocks data loading and logging to isolate the loop logic.
    """
    # 1. Mock Config
    config = {
        "samples": 1,
        "batch_size": 1,
        "np_seed": 42,
        "torch_seed": 42,
        "input_channels": 8,
        "total_hidden_channels": 32,
        "output_channels": 1,
        "dropout_rate": 0.0,
        "clip_grad_norm": True
    }
    
    # 2. Mock Model and Criterion
    model = MagicMock()
    model.base = 32
    # Mock forward pass: out_reg, out_class, h
    model.return_value = (torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16), torch.randn(1, 32, 16, 16))
    # Mock hidden state init
    model.init_h.return_value = torch.zeros(1, 32, 16, 16)
    
    optimizer = MagicMock()
    scheduler = MagicMock()
    
    # Criterion tuple: reg, class, mt
    criterion_reg = MagicMock()
    criterion_reg.return_value = torch.tensor(0.5, requires_grad=True)
    criterion_class = MagicMock()
    criterion_class.return_value = torch.tensor(0.5, requires_grad=True)
    
    multitaskloss_instance = MagicMock()
    multitaskloss_instance.return_value = torch.tensor(1.0, requires_grad=True) # Combined loss
    
    criterion = (criterion_reg, criterion_class, multitaskloss_instance)
    
    views_vol = np.zeros((10, 16, 16, 8))
    device = torch.device("cpu")
    
    # 3. Patch dependencies
    # We mock get_train_tensors to return a small sequence [batch, time, channels, H, W]
    mock_tensor = torch.randn(1, 2, 8, 16, 16) 
    
    with patch("views_hydranet.train.train_model.get_train_tensors", return_value=mock_tensor), \
         patch("views_hydranet.train.train_model.train_log"), \
         patch("views_hydranet.train.train_model.wandb"):
        
        # 4. Execute
        training_loop(config, model, criterion, optimizer, scheduler, views_vol, device)
        
    # 5. Verify Interations
    assert optimizer.zero_grad.called
    assert optimizer.step.called
    assert scheduler.step.called
