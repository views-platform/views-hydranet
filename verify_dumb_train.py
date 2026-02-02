
import logging
import sys
import numpy as np
import torch
from unittest.mock import MagicMock
from views_hydranet.train.train_model import train
from views_hydranet.utils.volume_handler import VolumeHandler

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("VerifyDumbTrain")

def test_train_is_dumb():
    logger.info("--- AUDIT: train() is Dumb (No Step) ---")

    # 1. Setup minimal components
    device = torch.device("cpu")
    model = torch.nn.Linear(1, 1).to(device) # Tiny model
    model.base = 1 # Satisfy model.base access
    orig_weight = model.weight.clone().detach()
    
    optimizer = MagicMock() # We expect NO calls to this
    scheduler = MagicMock()
    pbar = MagicMock()
    
    # Mock losses
    criterion_reg = MagicMock(return_value=torch.tensor(0.1, requires_grad=True))
    criterion_class = MagicMock(return_value=torch.tensor(0.1, requires_grad=True))
    multitaskloss = MagicMock(return_value=torch.tensor(0.2, requires_grad=True))
    multitaskloss.train = lambda: None 
    
    # Mock handler
    config = {
        "time_col": "t", "id_col": "i", "spatial_cols": ["y", "x"],
        "identity_cols": ["t", "i"], "features": ["f1"],
        "row_offset": 0, "col_offset": 0, "height": 1, "width": 1,
        "window_dim": 1, "steps": [1],
        "random_flips": False # Disable for simplicity
    }
    data = np.random.rand(2, 1, 1, 2) # T=2, H=1, W=1, C=2
    handler = VolumeHandler(
        data=data, axes=("T", "H", "W", "C"), channel_map=["t", "f1"],
        time_col="t", id_col="i", spatial_cols=["y", "x"],
        identity_cols=["t"], feature_cols=["f1"]
    )

    # 2. Execution
    logger.info("Calling train()...")
    
    # We must mock model.init_h 
    model.init_h = lambda hidden_channels, dim: torch.zeros((1, 1, 1, 1), requires_grad=True) 
    
    # Monkeypatch the forward to return (pred, pred_class, h)
    # pred shape must be [B, C, H, W] -> [1, 1, 1, 1]
    def mock_forward(t0, h):
        pred = torch.ones((1, 1, 1, 1), requires_grad=True)
        return pred, pred, h
    model.forward = mock_forward
    
    loss = train(
        model, optimizer, scheduler, 
        criterion_reg, criterion_class, multitaskloss,
        handler, config, device, pbar
    )

    # 3. Verification
    if not isinstance(loss, torch.Tensor):
        logger.error(f"FAIL: train() did not return a Tensor. Got {type(loss)}")
        sys.exit(1)
        
    if not torch.allclose(model.weight, orig_weight):
        logger.error("FALSIFIED: model weights changed! train() stepped the optimizer.")
        sys.exit(1)
    else:
        logger.info("PASS: Model weights are untouched.")

    if optimizer.step.called:
        logger.error("FALSIFIED: optimizer.step() was called inside train().")
        sys.exit(1)
    else:
        logger.info("PASS: optimizer.step() was NOT called.")

    logger.info("--- DUMB TRAIN AUDIT COMPLETE ---")

if __name__ == "__main__":
    test_train_is_dumb()
