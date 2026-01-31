from unittest.mock import MagicMock, patch

import numpy as np
import torch

from views_hydranet.train.train_model import train_model_artifact, training_loop


def test_training_loop_smoke(valid_config_dict):
    """
    Verify the loop logic itself.
    """
    # 1. Mock Model
    model = MagicMock()
    model.base = 8
    model.return_value = (torch.randn(1, 3, 16, 16), torch.randn(1, 3, 16, 16), torch.randn(1, 8, 16, 16))
    model.init_h.return_value = torch.zeros(1, 8, 16, 16)

    optimizer = MagicMock()
    scheduler = MagicMock()
    criterion = (MagicMock(), MagicMock(), MagicMock())
    for c in criterion: c.return_value = torch.tensor(1.0, requires_grad=True)

    views_vol = np.zeros((10, 16, 16, 8))
    mock_tensor = torch.randn(1, 2, 3, 16, 16)

    # Ensure config has what loop needs
    valid_config_dict["batch_size"] = 1
    valid_config_dict["samples"] = 1

    with patch("views_hydranet.train.train_model.get_train_tensors", return_value=mock_tensor), \
         patch("views_hydranet.train.train_model.train_log"):

        training_loop(valid_config_dict, model, criterion, optimizer, scheduler, views_vol, torch.device("cpu"))

    assert optimizer.step.called

def test_train_model_artifact_smoke_resilience(valid_config_dict, mock_mpm):
    """
    STRICT SMOKE TEST:
    Verifies the entire stack with a valid config fixture.
    """
    with patch("views_hydranet.utils.utils.wandb"), \
         patch("views_hydranet.train.train_model.init_weights"), \
         patch("views_hydranet.train.train_model.torch.save"), \
         patch("views_hydranet.train.train_model.os.makedirs"):

        # 8 channels: 5 ID + 3 feature
        mock_vol = np.zeros((10, 4, 4, 8))
        mock_vol[:, :, :, 5:] = 1.0

        # We simulate the small sample count for the smoke test
        valid_config_dict["samples"] = 1
        valid_config_dict["batch_size"] = 1

        # Execute REAL artifact training
        train_model_artifact(mock_mpm, valid_config_dict, torch.device("cpu"), mock_vol)
        print("\n✅ Training path is robust!")
