"""
Utilities for 'True Forecasting' (Future-Prediction) Workflows.

This module provides tools for generating metadata volumes for future time-steps,
managing fake prediction volumes for testing, and ensuring month_id alignment 
between observed data and forecast horizons.
"""

import logging
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from views_hydranet.utils.utils_df_to_vol_conversion import (
    calculate_absolute_indices,
    get_requried_columns_for_vol,
)

logger = logging.getLogger(__name__)


def generate_fake_vol(vol: np.ndarray, month_range: int = 36) -> np.ndarray:
    """
    Generates a fake prediction volume for testing purposes.

    Extracts the last three features from the input volume (SB, NS, OS).

    Args:
        vol: Input 4D volume array [n_months, height, width, n_features].
        month_range: Number of months to include in the fake volume.

    Returns:
        np.ndarray: Fake prediction volume [month_range, height, width, 3].
    """
    # Extract the last three features from the volume
    fake_vol = vol[-month_range:, :, :, 5:]

    return fake_vol


def make_forecast_storage_vol(
    df: pd.DataFrame,
    height: int = 180,
    width: int = 180,
    month_range: int = 36,
    to_tensor: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Creates a forecast metadata volume based on the last observed month.

    The spatial state (Grid IDs, Coordinates) of the last month is repeated 
    forward into the future for the specified month_range. Month IDs are 
    automatically incremented.

    Args:
        df: Input DataFrame containing observed data.
        height: Spatial grid height.
        width: Spatial grid width.
        month_range: Number of future months to prepare for.
        to_tensor: If True, returns a PyTorch tensor in (B, T, C, H, W) format.

    Returns:
        Union[np.ndarray, torch.Tensor]: Forecast storage volume.

    Invariants:
        - The first month of the forecast is exactly last_month_id + 1.
        - Spatial orientation matches the training volume (flipped vertically).
    """
    # 1. Index Calculation
    df = calculate_absolute_indices(df)

    # Infer the last month_id from the DataFrame
    last_month_id = df["month_id"].max()

    # Create a sub DataFrame of only the last month
    sub_df = df[df["month_id"] == last_month_id].copy()

    # Initialize the volume array
    required_columns = get_requried_columns_for_vol()
    features_num = len(required_columns)

    # Create the zero array with only the last month
    vol = np.zeros([height, width, 1, features_num])

    # Adjust abs_month to 0 for the initial volume
    sub_df["adjusted_abs_month"] = 0

    for i, col_name in enumerate(required_columns):
        vol[sub_df["abs_row"], sub_df["abs_col"], sub_df["adjusted_abs_month"], i] = sub_df[
            col_name
        ]

    # Stack the volume to the desired month range
    vol = np.repeat(vol, month_range, axis=2)

    # Adjust the month_id with an increment of 1
    for i in range(month_range):
        vol[:, :, i, 3] = (
            last_month_id + i + 1
        )

    # Reorient and transpose
    vol = np.flip(vol, axis=0)
    vol = np.transpose(vol, (2, 0, 1, 3))

    logger.debug(f"Forecast volume created with shape {vol.shape}")

    # Convert to tensor and permute dimensions.
    if to_tensor:
        # Align with out_of_sample_meta_vol: (batch, time, feature, height, width)
        vol = (
            torch.from_numpy(vol.copy())
            .float()
            .unsqueeze(dim=0)
            .permute(0, 1, 4, 2, 3)
        )

    return vol


def merge_vol(forecast_storage_vol: np.ndarray, vol_fake: np.ndarray) -> np.ndarray:
    """
    Merges a metadata volume with a prediction volume.

    Args:
        forecast_storage_vol: Spatial metadata [T, H, W, C_meta].
        vol_fake: Model predictions [T, H, W, C_pred].

    Returns:
        np.ndarray: Consolidated volume [T, H, W, C_meta + C_pred].
    """
    full_vol = np.concatenate([forecast_storage_vol, vol_fake], axis=-1)
    return full_vol


def plot_vol_comparison(
    vol: np.ndarray, new_vol: np.ndarray, month_range: int = 36
) -> None:
    """Plots a visual comparison of two volumes step-by-step."""
    features_titles = [
        "pg_id", "col", "row", "month_id", "c_id",
        "lr_sb_best", "lr_ns_best", "lr_os_best",
    ]
    n_features = vol.shape[-1]

    vol = vol[-month_range:, :, :, :]
    new_vol = new_vol[-month_range:, :, :, :]

    for i in range(month_range):
        fig, ax = plt.subplots(2, n_features, figsize=(20, 7))
        for j in range(n_features):
            ax[0, j].imshow(vol[i, :, :, j], cmap="rainbow")
            ax[1, j].imshow(new_vol[i, :, :, j], cmap="rainbow")
        plt.show()