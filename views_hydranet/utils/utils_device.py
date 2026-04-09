import logging

import torch

logger = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """Sets up the device for PyTorch operations (CUDA if available, otherwise CPU).

    Device selection is logged at DEBUG level. Call `log_device_report` from
    `utils_logging` to emit a visible banner at operation start.

    .. warning::
        This function's behavior depends on the system environment and the availability of CUDA.
        It uses `torch.cuda.is_available()` which can vary between execution environments.

    Returns:
        torch.device: The selected device ('cuda' or 'cpu').
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.debug("Device selected: %s", device)
    return device
