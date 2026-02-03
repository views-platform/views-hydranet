import sys
from io import StringIO
from unittest.mock import patch

import torch

from views_hydranet.utils.utils_device import setup_device


def test_setup_device_cuda_available():
    """
    Tests setup_device when CUDA is available.
    """
    with patch('torch.cuda.is_available', return_value=True):
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        device = setup_device()

        sys.stdout = sys.__stdout__  # Reset stdout

        assert device == torch.device('cuda')
        assert "Using device: cuda" in captured_output.getvalue()

def test_setup_device_cuda_not_available():
    """
    Tests setup_device when CUDA is not available.
    """
    with patch('torch.cuda.is_available', return_value=False):
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        device = setup_device()

        sys.stdout = sys.__stdout__  # Reset stdout

        assert device == torch.device('cpu')
        assert "Using device: cpu" in captured_output.getvalue()

