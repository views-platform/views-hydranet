import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn

from views_hydranet.utils.utils_contract_converters import (
    predictions_to_contract_df,
)
from views_hydranet.utils.utils_prediction import sample_posterior

logger = logging.getLogger(__name__)

def forecast_posterior(
    model: nn.Module,
    views_vol: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, List[pd.DataFrame]]:
    """
    Retrieve forecasts and generate DataFrames adhering to the views-evaluation contract.

    This function unifies the forecasting logic for both evaluation and true future forecasting.
    It returns a dictionary where keys are targets (sb, ns, os) and values are lists
    of contract-compliant DataFrames.

    Args:
        model: Trained model.
        views_vol: Input volume tensor.
        config: Configuration dictionary.
        device: Computation device.

    Returns:
        Dict[str, List[pd.DataFrame]]: Contract-compliant predictions per target.
    """
    # 1. Sample Posterior
    posterior_list, _, _, out_of_sample_meta_vol, _, _ = sample_posterior(model, views_vol, config, device)

    # 2. Convert to Contract DataFrames for each target
    # out_of_sample_meta_vol is [batch, steps, channels, H, W]
    vol_np = out_of_sample_meta_vol.cpu().numpy()

    results = {}
    for target in ["sb", "ns", "os"]:
        results[target] = predictions_to_contract_df(
            posterior_list=posterior_list,
            forecast_storage_vol=vol_np,
            target=target
        )

    return results

def forecast_with_model_artifact(
    config: Any,
    device: torch.device,
    views_vol: torch.Tensor,
    PATH_ARTIFACTS: Path,
    artifact_name: Optional[str] = None,
) -> Dict[str, List[pd.DataFrame]]:
    """
    Load a model artifact and produce contract-compliant forecasts.
    """
    if artifact_name:
        logger.info(f"Using artifact: {artifact_name}")
        if not artifact_name.endswith(".pt"):
            artifact_name += ".pt"
        PATH_MODEL_ARTIFACT = PATH_ARTIFACTS / artifact_name
    else:
        raise NotImplementedError("Automatic artifact selection requires ModelPathManager integration.")

    if not PATH_MODEL_ARTIFACT.exists():
        raise FileNotFoundError(f"Model artifact not found at {PATH_MODEL_ARTIFACT}")

    # load the model
    # TODO: In the future, use weights_only=True and add model classes to safe globals for better security.
    model = torch.load(PATH_MODEL_ARTIFACT, weights_only=False)
    model.to(device)
    model.eval()

    # Generate forecasts
    forecast_results = forecast_posterior(model, views_vol, config, device)

    logger.info("Forecasting complete. Results adhere to views-evaluation contract.")
    return forecast_results


