import torch


def setup_device():
    """Sets up the device for PyTorch operations (CUDA if available, otherwise CPU).

    Prints the selected device to standard output.

    .. warning::
        This function's behavior depends on the system environment and the availability of CUDA.
        It uses `torch.cuda.is_available()` which can vary between execution environments.

    Returns:
        torch.device: The selected device ('cuda' or 'cpu').

    Example:
        >>> # Assuming CUDA is available
        >>> import torch
        >>> from unittest.mock import patch
        >>> from io import StringIO
        >>> import sys
        >>> with patch('torch.cuda.is_available', return_value=True):
        ...     captured_output = StringIO()
        ...     sys.stdout = captured_output
        ...     device = setup_device()
        ...     sys.stdout = sys.__stdout__
        ...     assert device == torch.device('cuda')
        ...     assert "Using device: cuda" in captured_output.getvalue()

        >>> # Assuming CUDA is not available
        >>> with patch('torch.cuda.is_available', return_value=False):
        ...     captured_output = StringIO()
        ...     sys.stdout = captured_output
        ...     device = setup_device()
        ...     sys.stdout = sys.__stdout__
        ...     assert device == torch.device('cpu')
        ...     assert "Using device: cpu" in captured_output.getvalue()
    """
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device  # not sure you need to return it, but it might be useful for debugging
