from unittest.mock import MagicMock, patch

import pytest
import torch

from views_hydranet.utils.utils import choose_sheduler, init_weights


@pytest.fixture
def mock_unet():
    """Fixture for a mock model."""
    return MagicMock()

@pytest.fixture
def mock_config():
    """Fixture for a mock config dictionary."""
    return {
        "scheduler": "plateau",
        "learning_rate": 0.001,
        "weight_init": "xavier_uni",
    }

@patch('views_hydranet.utils.utils.ReduceLROnPlateau')
@patch('views_hydranet.utils.utils.torch.optim.AdamW')
def test_choose_scheduler_plateau(mock_adamw, mock_plateau_scheduler, mock_unet, mock_config):
    """
    Tests that choose_sheduler correctly selects and instantiates the ReduceLROnPlateau scheduler.
    """
    # Arrange
    mock_optimizer_instance = MagicMock(spec=torch.optim.Optimizer)
    mock_optimizer_instance.param_groups = [{'lr': mock_config['learning_rate']}]
    mock_scheduler_instance = MagicMock()
    mock_adamw.return_value = mock_optimizer_instance
    mock_plateau_scheduler.return_value = mock_scheduler_instance

    # Act
    optimizer, scheduler = choose_sheduler(mock_config, mock_unet)

    # Assert
    mock_adamw.assert_called_once_with(mock_unet.parameters(), lr=0.001, betas=(0.9, 0.999))
    mock_plateau_scheduler.assert_called_once_with(mock_optimizer_instance)
    assert optimizer is mock_optimizer_instance
    assert scheduler is mock_scheduler_instance

# New test for init_weights
@patch('torch.nn.init.xavier_uniform_')
def test_init_weights_xavier_uni(mock_xavier_uniform_, mock_config):
    """
    Tests that init_weights correctly applies xavier_uniform_ initialization to Conv2d and Linear layers.
    """
    # Arrange
    conv_layer = MagicMock(spec=torch.nn.Conv2d)
    conv_layer.weight = MagicMock()

    linear_layer = MagicMock(spec=torch.nn.Linear)
    linear_layer.weight = MagicMock()

    bn_layer = MagicMock(spec=torch.nn.BatchNorm2d)
    bn_layer.weight = MagicMock()

    # Act
    init_weights(conv_layer, mock_config)
    init_weights(linear_layer, mock_config)
    init_weights(bn_layer, mock_config) # Should not be initialized

    # Assert
    mock_xavier_uniform_.assert_any_call(conv_layer.weight)
    mock_xavier_uniform_.assert_any_call(linear_layer.weight)
    assert mock_xavier_uniform_.call_count == 2
