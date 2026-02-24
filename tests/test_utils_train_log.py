from unittest.mock import patch

import numpy as np

from views_hydranet.utils.utils import train_log


@patch("views_hydranet.utils.utils.wandb")
def test_train_log_basic(mock_wandb):
    """
    Tests that train_log calculates average losses and logs them to wandb.
    """
    # Arrange
    avg_loss_list = [0.1, 0.2, 0.3]
    avg_loss_reg_list = [0.05, 0.1, 0.15]
    avg_loss_class_list = [0.02, 0.03, 0.04]

    expected_avg_loss = np.mean(avg_loss_list)
    expected_avg_loss_reg = np.mean(avg_loss_reg_list)
    expected_avg_loss_class = np.mean(avg_loss_class_list)

    # Act
    train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list)

    # Assert
    mock_wandb.log.assert_called_once_with(
        {
            "avg_loss": expected_avg_loss,
            "avg_loss_reg": expected_avg_loss_reg,
            "avg_loss_class": expected_avg_loss_class,
        }
    )


@patch("views_hydranet.utils.utils.wandb")
def test_train_log_empty_lists(mock_wandb):
    """
    Tests that train_log handles empty lists gracefully (though np.mean on empty lists
    will raise a RuntimeWarning and return NaN, which wandb should handle).
    """
    # Arrange
    avg_loss_list = []
    avg_loss_reg_list = []
    avg_loss_class_list = []

    # Act
    train_log(avg_loss_list, avg_loss_reg_list, avg_loss_class_list)

    # Assert
    mock_wandb.log.assert_called_once()
    # Check that NaN values are passed to wandb (np.mean([]) is NaN)
    logged_args = mock_wandb.log.call_args[0][0]
    assert np.isnan(logged_args["avg_loss"])
    assert np.isnan(logged_args["avg_loss_reg"])
    assert np.isnan(logged_args["avg_loss_class"])
