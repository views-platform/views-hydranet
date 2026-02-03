
from unittest.mock import MagicMock

import torch.nn as nn

from views_hydranet.utils.utils import apply_dropout


def test_apply_dropout_to_dropout_layer():
    """
    Tests that apply_dropout sets a Dropout layer to training mode.
    """
    # Arrange
    dropout_layer = nn.Dropout(p=0.5)
    mock_train = MagicMock()
    dropout_layer.train = mock_train

    # Act
    apply_dropout(dropout_layer)

    # Assert
    mock_train.assert_called_once()

def test_apply_dropout_to_non_dropout_layer():
    """
    Tests that apply_dropout does nothing to a non-Dropout layer.
    """
    # Arrange
    linear_layer = nn.Linear(10, 10)
    mock_train = MagicMock()
    linear_layer.train = mock_train

    # Act
    apply_dropout(linear_layer)

    # Assert
    mock_train.assert_not_called()
