import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import torch

from views_hydranet.utils.utils_device import setup_device
from views_hydranet.utils.utils_logging import log_device_report


def test_setup_device_cuda_available():
    """setup_device returns cuda device when CUDA is available."""
    with patch("torch.cuda.is_available", return_value=True):
        device = setup_device()
        assert device == torch.device("cuda")


def test_setup_device_cuda_not_available():
    """setup_device returns cpu device when CUDA is not available."""
    with patch("torch.cuda.is_available", return_value=False):
        device = setup_device()
        assert device == torch.device("cpu")


def test_log_device_report_cuda():
    """log_device_report prints a GPU banner when device is cuda."""
    fake_props = MagicMock()
    fake_props.total_memory = 24 * 1024**3  # 24 GiB

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.get_device_name", return_value="NVIDIA Test GPU"),
        patch("torch.cuda.get_device_properties", return_value=fake_props),
    ):
        captured = StringIO()
        sys.stdout = captured
        try:
            log_device_report(torch.device("cuda"), "training")
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "DEVICE REPORT" in output
        assert "TRAINING" in output
        assert "GPU" in output
        assert "NVIDIA Test GPU" in output


def test_log_device_report_cpu():
    """log_device_report prints a WARNING banner when device is cpu."""
    captured = StringIO()
    sys.stdout = captured
    try:
        log_device_report(torch.device("cpu"), "forecasting")
    finally:
        sys.stdout = sys.__stdout__

    output = captured.getvalue()
    assert "WARNING" in output
    assert "CPU" in output
    assert "FORECASTING" in output
    assert "hard stop" in output.lower()
