import logging
from datetime import datetime
from typing import List, Optional, Tuple

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
    """Generates a fake prediction volume for testing purposes.

    Extracts the last three features from the input volume.
    Assumes the last three features represent `sb`, `ns`, and `os`.

    Args:
        vol: The input 4D volume array with shape [n_months, height, width, n_features].
        month_range: The number of months to include in the fake volume. Default is 36.

    Returns:
        A volume array with the last three features, shape [month_range, height, width, 3].
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
) -> np.ndarray | torch.Tensor:
    """Creates a forecast storage volume based on the last month of data in the DataFrame.

    The volume is repeated for the specified `month_range` with incrementally
    adjusted month IDs.

    Args:
        df: The input DataFrame containing spatial-temporal data.
            Expected columns include 'row', 'col', 'month_id', 'c_id', 'priogrid_gid'.
        height: The height of the spatial grid. Defaults to 180.
        width: The width of the spatial grid. Defaults to 180.
        month_range: The number of months to forecast into the future. Default is 36.
        to_tensor: Whether to return the result as a torch.Tensor.

    Returns:
        The forecast storage volume with shape [month_range, height, width, 5]
        (if ndarray) or [1, month_range, 5, height, width] (if tensor).
    """

    df = calculate_absolute_indices(df)  # abs_row, abs_col, abs_month needed for the volume

    # Infer the last month_id from the DataFrame
    last_month_id = df["month_id"].max()

    # Create a sub DataFrame of only the last month
    sub_df = df[df["month_id"] == last_month_id].copy()

    # required features
    required_columns = get_requried_columns_for_vol()

    # check if the required columns are in the df
    for col in required_columns:
        if col not in sub_df.columns:
            raise ValueError(
                f"Column '{col}' not found in the DataFrame. "
                "Check the input DataFrame and try again."
            )

    # Initialize the volume array
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
        )  # to get one month after the last observed month

    # Reorient and transpose
    vol = np.flip(vol, axis=0)
    vol = np.transpose(vol, (2, 0, 1, 3))

    logger.info(
        f"Volume of shape {vol.shape} created. Should be ({month_range}, 180, 180, {features_num})"
    )

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
    """Merges a forecast volume with an existing forecast storage volume.

    Combines the features from `vol_fake` with `forecast_storage_vol` along the feature axis.

    Args:
        forecast_storage_vol: The forecast storage volume with shape
            [n_months, height, width, n_features].
        vol_fake: The forecast volume to be merged with shape
            [n_months, height, width, n_features_fake].

    Returns:
        The merged volume with shape
        [n_months, height, width, n_features + n_features_fake].
    """
    # Merge the forecast volume with the storage volume along the feature axis
    full_vol = np.concatenate([forecast_storage_vol, vol_fake], axis=-1)

    logger.info(
        f"Volume of shape {full_vol.shape} created. "
        f"Should be ({forecast_storage_vol.shape[0]}, 180, 180, "
        f"{forecast_storage_vol.shape[3] + vol_fake.shape[3]})"
    )

    return full_vol


def check_vol_equal(vol: np.ndarray, full_vol: np.ndarray) -> None:
    """Verifies the merging of two volumes.

    Checks if the original volume and the merged volume are equivalent for the
    overlapping time steps and features.

    Args:
        vol: The original volume.
        full_vol: The merged volume.
    """

    logger.debug(f"Original vol shape: {vol.shape}")
    logger.debug(f"Full vol shape: {full_vol.shape}")

    # trim original volume to the same shape as the full volume - ie. the last n months
    month_range = full_vol.shape[0]
    vol_trimmed = vol[-month_range:, :, :, :]

    logger.debug(f"Trimmed original vol shape: {vol_trimmed.shape}")

    # now go through each feature individually and check if they are the same
    list_features = [
        "pg_id",
        "col",
        "row",
        "month_id",
        "c_id",
        "ln_sb_best",
        "ln_ns_best",
        "ln_os_best",
    ]

    for i in range(vol_trimmed.shape[-1]):
        feature_name = list_features[i] if i < len(list_features) else f"feature_{i}"
        is_equal = np.array_equal(vol_trimmed[:, :, :, i], full_vol[:, :, :, i])
        logger.info(f"Feature {i} ({feature_name}) equal: {is_equal}")


def check_month_id_consistency(
    forecast_storage_vol: torch.Tensor, df: pd.DataFrame, month_range: int = 36
) -> None:
    """Checks consistency of month_id values between forecast storage and DataFrame.

    Args:
        forecast_storage_vol: The forecast storage volume with shape
            [batch, time, feature, height, width].
        df: The DataFrame containing reference month_id values.
        month_range: The expected range of months in the forecast storage volume.

    Raises:
        ValueError: If there is a mismatch in month_id values.
    """
    logger.debug(f"Forecast storage vol shape: {forecast_storage_vol.shape}")

    # Retrieve month_id values
    min_month_id_df = df["month_id"].min()
    max_month_id_df = df["month_id"].max()

    # month_id is the 4th feature (index 3)
    min_month_id_vol = forecast_storage_vol[:, :, 3, :, :].min().item()
    max_month_id_vol = forecast_storage_vol[:, :, 3, :, :].max().item()

    logger.info(f"Min month_id in df: {min_month_id_df}")
    logger.info(f"Max month_id in df: {max_month_id_df}")
    logger.info(f"Min month_id in forecast storage: {min_month_id_vol}")
    logger.info(f"Max month_id in forecast storage: {max_month_id_vol}")

    logger.info(f"Months forecasted ahead: {int(max_month_id_vol - min_month_id_vol + 1)}")

    # Check if min month_id in the forecast storage volume is 1 above the max month_id in the df
    if min_month_id_vol != max_month_id_df + 1:
        raise ValueError(
            f"Mismatch in month_id: Expected min {max_month_id_df + 1}, got {min_month_id_vol}."
        )

    # Check if max month_id in the forecast storage volume is month_range above the max month_id in the df
    if max_month_id_vol != max_month_id_df + month_range:
        raise ValueError(
            f"Mismatch in month_id: Expected max {max_month_id_df + month_range}, got {max_month_id_vol}."
        )


def plot_vol_comparison(
    vol: np.ndarray, new_vol: np.ndarray, month_range: int = 36
) -> None:
    """Plots a comparison of slices from two 4D volume arrays.

    Args:
        vol: The original 4D volume array.
        new_vol: The new 4D volume array to compare with.
        month_range: The number of slices (time steps) to plot. Default is 36.
    """
    features_titles = [
        "pg_id",
        "col",
        "row",
        "month_id",
        "c_id",
        "ln_sb_best",
        "ln_ns_best",
        "ln_os_best",
    ]
    n_features = vol.shape[-1]

    # Ensure the volumes cover the last month_range months
    vol = vol[-month_range:, :, :, :]
    new_vol = new_vol[-month_range:, :, :, :]

    for i in range(month_range):
        fig, ax = plt.subplots(2, n_features, figsize=(20, 7))

        for j in range(n_features):
            # Plot the original volume in the first row
            ax[0, j].imshow(
                vol[i, :, :, j],
                cmap="rainbow",
                vmin=vol[:, :, :, j].min(),
                vmax=vol[:, :, :, j].max(),
            )
            ax[0, j].set_title(
                features_titles[j] if j < len(features_titles) else f"Feature {j}"
            )

            # Plot the new volume in the second row
            ax[1, j].imshow(
                new_vol[i, :, :, j],
                cmap="rainbow",
                vmin=new_vol[:, :, :, j].min(),
                vmax=new_vol[:, :, :, j].max(),
            )
            ax[1, j].set_title(
                f"New {features_titles[j]}"
                if j < len(features_titles)
                else f"New Feature {j}"
            )

        fig.suptitle(f"Time Step {i + 1}", fontsize=16, y=1.05)

        # Remove ticks
        for a in ax.flat:
            a.set_xticks([])
            a.set_yticks([])

        plt.subplots_adjust(
            left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.2, hspace=0.4
        )
        plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.95])

        plt.show()