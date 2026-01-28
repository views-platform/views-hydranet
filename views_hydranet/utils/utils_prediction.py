import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from views_pipeline_core.managers.model import ModelPathManager

from views_hydranet.utils.utils import apply_dropout, execute_freeze_h_option, get_full_tensor

logger = logging.getLogger(__name__)

def predict(
    model: nn.Module,
    full_tensor: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device,
    sample_i: int,
    is_evalutaion: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Function to create predictions for the Hydranet model.
    The function takes the model, the test tensor, the number of time steps to predict, the config, and the device as input.
    The function returns **two lists of numpy arrays**. One list of the predicted magnitudes and one list of the predicted probabilities.
    Each array is of the shap **fx180x180**, where f is the number of features (currently 3 types of violence).
    """

    logger.debug(f"Predicting sample {sample_i + 1}/{config['test_samples']}")

    # Set the model to evaluation mode
    model.eval()

    # Apply dropout which is otherwise not applied during eval mode
    model.apply(apply_dropout)

    # create empty lists to store the predictions both counts and probabilities
    pred_np_list = []
    pred_class_np_list = []

    # initialize the hidden state
    h_tt = model.init_hTtime(hidden_channels=model.base, H=180, W=180).float().to(device)

    # get the sequence length
    seq_len = full_tensor.shape[1]

    if is_evalutaion:
        full_seq_len = seq_len - 1
        in_sample_seq_len = seq_len - 1 - config["time_steps"]
    else:
        full_seq_len = seq_len - 1 + config["time_steps"]
        in_sample_seq_len = seq_len - 1

    for i in range(full_seq_len):
        if i < in_sample_seq_len:
            # get the tensor for the current month
            t0 = full_tensor[:, i, :, :, :].to(device)
            # predict the next month
            t1_pred, t1_pred_class, h_tt = model(t0, h_tt)
        else:
            t0 = t1_pred.detach()
            # Execute whatever freeze option you have set in the config out of sample
            t1_pred, t1_pred_class, h_tt = execute_freeze_h_option(config, model, t0, h_tt)

            # Only save the out-of-sample predictions
            t1_pred_class = torch.sigmoid(t1_pred_class)
            pred_np_list.append(t1_pred.cpu().detach().numpy().squeeze())
            pred_class_np_list.append(t1_pred_class.cpu().detach().numpy().squeeze())

    return pred_np_list, pred_class_np_list


def sample_posterior(
    model: nn.Module, views_vol: np.ndarray, config: Dict[str, Any], device: torch.device
) -> Tuple[List, List, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples from the posterior distribution of Hydranet.

    Args:
    - model: HydraNet
    - views_vol: Input views data.
    - config: Configuration file
    - device: Device for computations.

    Returns:
    - tuple: (posterior_magnitudes, posterior_probabilities, out_of_sample_vol, out_of_sample_meta_vol, full_tensor, metadata_tensor)
    """

    logger.info(f"Drawing {config['test_samples']} posterior samples...")

    full_tensor, metadata_tensor = get_full_tensor(views_vol, config)

    # these two are only used for calibration and testing - not for forecasting
    out_of_sample_vol = full_tensor[:, -config["time_steps"] :, :, :, :].cpu().numpy()
    out_of_sample_meta_vol = metadata_tensor[:, -config["time_steps"] :, :, :, :]

    posterior_list = []
    posterior_list_class = []

    for sample_i in range(config["test_samples"]):
        pred_np_list, pred_class_np_list = predict(model, full_tensor, config, device, sample_i)
        posterior_list.append(pred_np_list)
        posterior_list_class.append(pred_class_np_list)

    return (
        posterior_list,
        posterior_list_class,
        out_of_sample_vol,
        out_of_sample_meta_vol,
        full_tensor,
        metadata_tensor,
    )

