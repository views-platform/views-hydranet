import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt  # get this one outta here
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from views_pipeline_core.configs.pipeline import PipelineConfig
from views_pipeline_core.files.utils import read_dataframe


def get_requried_columns_for_vol() -> list[str]:
    """
    Returns the list of required columns for constructing the volume array.

    These columns are necessary for creating the spatial-temporal volume format used by models
    such as HydraNet and other CNN-based models. The minimum volume array includes data for
    spatial coordinates, temporal indices, and identifiers for grid cells.
    Beynd these columns, the volume typically includes "forecast feeatures",
    for instance "ln_sb_best", "ln_ns_best", and "ln_os_best" for the three event types.

    Returns:
        list of str: A list of column names required to create the volume array, specifically:
                     - 'priogrid_gid': Priogrid ID, a unique identifier for grid cells.
                     - 'col': Column index in the spatial grid.
                     - 'row': Row index in the spatial grid.
                     - 'month_id': Temporal index for months.
                     - 'c_id': Country ID or relevant identifier.

    Example:
        >>> get_requried_columns_for_vol()
        ['priogrid_gid', 'col', 'row', 'month_id', 'c_id']
    """

    required_columns = ["priogrid_gid", "col", "row", "month_id", "c_id"]

    return required_columns


def calculate_absolute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Computes absolute indices for 'row', 'col', and 'month_id'.

    This function calculates indices starting from 0 for the row, column, and
    month dimensions based on the minimum values present in the input DataFrame.
    These absolute indices are required for correctly placing data points into
    a NumPy volume array.

    Args:
        df: The input DataFrame. Must contain 'row', 'col',
            and 'month_id' columns.

    Returns:
        A new DataFrame instance with 'abs_row', 'abs_col', and 'abs_month' columns
        added.

    Example:
        >>> import pandas as pd
        >>> df_in = pd.DataFrame({
        ...     'row': [5, 15, 25, 35],
        ...     'col': [10, 20, 30, 40],
        ...     'month_id': [100, 100, 101, 101]
        ... })
        >>> df_out = calculate_absolute_indices(df_in)
        >>> df_out[['abs_row', 'abs_col', 'abs_month']]
           abs_row  abs_col  abs_month
        0        0        0          0
        1       10       10          0
        2       20       20          1
        3       30       30          1
    """

    df_abs = df.copy()

    # get the first month_id
    month_first = df_abs["month_id"].min()

    # calculate the absolute indices
    df_abs["abs_row"] = df_abs["row"] - df_abs["row"].min()
    df_abs["abs_col"] = df_abs["col"] - df_abs["col"].min()
    df_abs["abs_month"] = df_abs["month_id"] - month_first

    # insure the data types are integers
    df_abs["abs_row"] = df_abs["abs_row"].astype(int)
    df_abs["abs_col"] = df_abs["abs_col"].astype(int)
    df_abs["abs_month"] = df_abs["abs_month"].astype(int)

    return df_abs


def df_to_vol(
    df: pd.DataFrame,
    height: int = 180,
    width: int = 180,
    forecast_features: list[str] = ["ln_sb_best", "ln_ns_best", "ln_os_best"],
) -> np.ndarray:
    """Converts a DataFrame into a 4D numpy array (volume) for spatial-temporal data representation.

    This volume format is used by models like HydraNet and other CNN-based models. The resulting
    volume array has dimensions [n_months, height, width, n_features].

    Args:
        df: The input DataFrame containing spatial-temporal data. Must include columns:
            - 'priogrid_gid': Priogrid ID.
            - 'col': Column index in the spatial grid.
            - 'row': Row index in the spatial grid.
            - 'month_id': Temporal index for months.
            - 'c_id': Country ID or relevant identifier.
        height: The height of the spatial grid. Defaults to 180.
        width: The width of the spatial grid. Defaults to 180.
        forecast_features: List of forecast feature columns to include in the volume.
            Defaults to ['ln_sb_best', 'ln_ns_best', 'ln_os_best'].

    Returns:
        A 4D volume array with shape [n_months, height, width, n_features].
        Where n_features is the total number of required and forecast features combined.
        Given the default settings, the default shape is [n_months, 180, 180, 8].

    Raises:
        ValueError: If any of the required columns are missing or if indices are out of bounds.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> mock_data = {
        ...     "priogrid_gid": [1, 2, 3, 4],
        ...     "col": [10, 20, 30, 40],
        ...     "row": [5, 15, 25, 35],
        ...     "month_id": [100, 100, 101, 101],
        ...     "c_id": [1, 1, 1, 1],
        ...     "ln_sb_best": [0.1, 0.2, 0.3, 0.4],
        ...     "ln_ns_best": [0.2, 0.3, 0.4, 0.5],
        ...     "ln_os_best": [0.3, 0.4, 0.5, 0.6],
        ... }
        >>> mock_df = pd.DataFrame(mock_data)
        >>> vol = df_to_vol(mock_df, height=40, width=40)
        >>> vol.shape
        (2, 40, 40, 8)
    """
    # --- INPUT VALIDATION: Check for empty DataFrame ---
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")
    # --- END INPUT VALIDATION ---

    # to get prio grid id out of the index
    df = df.reset_index()

    # --- INPUT VALIDATION: Check for duplicate priogrid_gid and month_id combinations ---
    if df.duplicated(subset=["priogrid_gid", "month_id"]).any():
        duplicate_entries = df[df.duplicated(subset=["priogrid_gid", "month_id"], keep=False)]
        raise ValueError(
            "Duplicate entries found for 'priogrid_gid' and 'month_id'. "
            "Each priogrid_gid must have a unique month_id. "
            f"Duplicated entries:\n{duplicate_entries}"
        )
    # --- END INPUT VALIDATION ---

    required_columns = get_requried_columns_for_vol()

    for col in required_columns:
        if col not in df.columns.tolist():
            raise ValueError(
                f"Column {col} not found in the DataFrame. "
                "Please check your viewser query set in 'model'/configs/config_input_data.py"
            )

    vol_features = required_columns + forecast_features
    n_features = len(vol_features)

    month_first = df["month_id"].min()
    month_last = df["month_id"].max()
    month_range = month_last - month_first + 1

    df_abs = calculate_absolute_indices(df)  # abs_row, abs_col, abs_month needed for the volume

    # --- INPUT VALIDATION ---
    if df_abs["abs_row"].max() >= height:
        raise ValueError(
            f"Maximum row index ({df_abs['abs_row'].max()}) is out of bounds for height {height}."
        )
    if df_abs["abs_col"].max() >= width:
        raise ValueError(
            f"Maximum column index ({df_abs['abs_col'].max()}) is out of bounds for width {width}."
        )
    # --- END INPUT VALIDATION ---

    vol = np.zeros([height, width, month_range, n_features])  # Create the volume array.

    for i, feature in enumerate(vol_features):
        vol[df_abs["abs_row"], df_abs["abs_col"], df_abs["abs_month"], i] = df_abs[feature]

    # vol = np.flip(vol, axis=0)  # REMOVED: Orientation now handled at CNN boundary
    vol = np.transpose(vol, (2, 0, 1, 3))  # Move the month dimension to the front. [T, H, W, C]

    logger.info(f"Volume of shape {vol.shape} created. Should be (n_months, 180, 180, 8)")

    return vol


def vol_to_df(
    vol: np.ndarray,
    forecast_features: list[str] = ["ln_sb_best", "ln_ns_best", "ln_os_best"],
) -> pd.DataFrame:
    """Converts a 4D numpy array (volume) back into a DataFrame.

    This function is used to transform the 4D volume format used by models like HydraNet back into
    a DataFrame. The purpose is to check that the conversion between DataFrame and volume does not alter data,
    thus verifying consistency between df_to_vol and vol_to_df operations.

    Args:
        vol: The input 4D volume array (created with df_to_vol()) to be converted, with shape
             [n_months, height, width, n_features].
        forecast_features: List of feature names corresponding to the forecast features in the volume.
             Defaults to ['ln_sb_best', 'ln_ns_best', 'ln_os_best'].

    Returns:
        The DataFrame representation of the volume array containing columns:
        'priogrid_gid', 'col', 'row', 'month_id', 'c_id', followed by forecast features.
        Rows where 'priogrid_gid' is 0 are removed.

    Raises:
        ValueError: If the number of features in the volume does not match the expected number.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> mock_vol = np.zeros((2, 10, 10, 8))
        >>> mock_vol[:, :, :, 0] = np.arange(200).reshape(2, 10, 10) # priogrid_gid
        >>> mock_vol[:, :, :, 3] = np.array([100, 101]).reshape(2, 1, 1) # month_id
        >>> df_recreated = vol_to_df(mock_vol)
        >>> df_recreated.shape
        (200, 8)
    """

    required_columns = get_requried_columns_for_vol()

    vol_features = required_columns + forecast_features
    n_features = len(vol_features)

    # check that the n_features is the same as the last dimension of the volume
    if n_features != vol.shape[3]:
        raise ValueError(
            f"Number of features in the volume array ({vol.shape[3]}) does not match the number of features expected ({n_features})."
        )

    feature_dict = {}
    for i, feature in enumerate(vol_features):
        feature_dict[feature] = vol[:, :, :, i].flatten()

    df = pd.DataFrame(feature_dict)

    # Correct the data types for required columns
    for col in required_columns:
        df[col] = df[col].astype(int)

    # Remove rows where 'priogrid_gid' is 0 - these are ocean cells and not PRIO grid cells as such.
    df = df[df["priogrid_gid"] != 0]

    logger.info(f"DataFrame of shape {df.shape} created. Should be (n_months * 180 * 180, 8)")

    return df

    required_columns = get_requried_columns_for_vol()

    vol_features = required_columns + forecast_features
    n_features = len(vol_features)

    # check that the n_features is the same as the last dimension of the volume
    if n_features != vol.shape[3]:
        raise ValueError(
            f"Number of features in the volume array ({vol.shape[3]}) does not match the number of features expected ({n_features})."
        )

    feature_dict = {}
    for i, feature in enumerate(vol_features):
        feature_dict[feature] = vol[:, :, :, i].flatten()

    df = pd.DataFrame(feature_dict)

    # Correct the data types for required columns
    for col in required_columns:
        df[col] = df[col].astype(int)

    # Remove rows where 'priogrid_gid' is 0 - these are ocean cells and not PRIO grid cells as such.
    df = df[df["priogrid_gid"] != 0]

    logger.info(f"DataFrame of shape {df.shape} created. Should be (n_months * 180 * 180, 8)")

    return df


def df_vol_conversion_test(
    df: pd.DataFrame,
    vol: np.ndarray,
    forecast_features: list[str] = ["ln_sb_best", "ln_ns_best", "ln_os_best"],
) -> None:
    """Tests the consistency of DataFrame and volume array conversions.

    This unit test verifies that converting a DataFrame to a 4D volume array and back to a DataFrame
    results in the original data. It ensures the `df_to_vol` and `vol_to_df` functions are consistent
    and that data integrity is maintained during the transformations.

    Args:
        df: The original DataFrame containing the spatial-temporal data.
        vol: The 4D volume array obtained from the DataFrame conversion via df_to_vol().
        forecast_features: List of feature names included in the volume.
            Defaults to ['ln_sb_best', 'ln_ns_best', 'ln_os_best'].
    """

    # Make a copy of the original DataFrame
    df_copy = df.copy()

    # Proof of concept: Check if the copy is the same as the original
    logger.info(f"Original DataFrame equals its copy: {df.equals(df_copy)}")

    # Convert the volume back into a DataFrame
    df_recreated = vol_to_df(vol)

    # Trim the original DataFrame to match the features of the recreated DataFrame
    required_columns = ["priogrid_gid", "col", "row", "month_id", "c_id"]
    vol_features = required_columns + forecast_features
    df_trimmed = df[vol_features]

    # Sort both DataFrames by 'priogrid_gid' and 'month_id'
    df_trimmed = df_trimmed.sort_values(by=["priogrid_gid", "month_id"])
    df_recreated = df_recreated.sort_values(by=["priogrid_gid", "month_id"])

    # Reset the index to ensure alignment
    df_trimmed = df_trimmed.reset_index(drop=True)
    df_recreated = df_recreated.reset_index(drop=True)

    # Check if the two DataFrames are the same
    is_equal = df_trimmed.equals(df_recreated)
    logger.info(f"Trimmed original DataFrame equals recreated DataFrame from volume: {is_equal}")


def plot_vol(
    vol: np.ndarray,
    month_range: int,
    forecast_features: list[str] = ["ln_sb_best", "ln_ns_best", "ln_os_best"],
) -> None:
    """Plots feature maps from a 4D volume array over a specified range of months.

    This function generates and displays plots for each feature in the volume array for the last
    `month_range` time steps. Each subplot corresponds to a different feature map at each time step,
    allowing visualization of spatial-temporal data.

    Args:
        vol: The input 4D volume array with shape [n_months, height, width, n_features].
        month_range: The number of recent time steps (months) to plot.
        forecast_features: List of additional feature names to include in the plots.
            Defaults to ['ln_sb_best', 'ln_ns_best', 'ln_os_best'].

    Raises:
        ValueError: If `month_range` exceeds the number of time steps in `vol`.

    .. warning::
        This function calls `matplotlib.pyplot.show()`, which will block execution.

    Example:
        >>> import numpy as np
        >>> mock_vol_example = np.random.rand(3, 10, 10, 8)
        >>> # plot_vol(mock_vol_example, month_range=1) # Uncomment to see plot
    """

    # check if the month_range is valid:
    if month_range > vol.shape[0]:
        raise ValueError(
            f"month_range ({month_range}) exceeds the number of time steps in the volume ({vol.shape[0]})."
        )

    # get the required columns and feature titles
    required_columns = get_requried_columns_for_vol()
    features_titles = required_columns + forecast_features
    n_features = vol.shape[-1]

    # get sub_df of the lasst month_range months
    vol = vol[-month_range:, :, :, :]

    for i in range(month_range):
        fig, ax = plt.subplots(1, n_features, figsize=(15, 4))

        for j in range(
            min(n_features, vol.shape[-1])
        ):  # Handle cases where there are fewer than 7 features
            im = ax[j].imshow(
                vol[i, :, :, j],
                cmap="rainbow",
                vmin=vol[:, :, :, j].min(),
                vmax=vol[:, :, :, j].max(),
            )

            # if the feature does not have a name, use a generic numbered title
            features_title = features_titles[j] if j < len(features_titles) else f"Feature {j}"

            # Change in this feature is hard to see and uniform across each month, so we add the month id to the title
            if features_title == "month_id":
                features_title = "month_id" + f" ({np.unique(vol[i, :, :, j])})"

            ax[j].set_title(features_title)

        # Adding title with specific adjustment
        fig.suptitle(f"Time Step {i + 1}", fontsize=16, y=0.75)  # Adjust `y` for title position

        # remove ticks
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

        # Adjust layout
        plt.subplots_adjust(left=0.1, right=1, top=0.85, bottom=0.55, wspace=0.2, hspace=-0)
        plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.9])  # `rect` adjusts the position of subplots

        plt.show()


def create_or_load_views_vol(
    partition: str, PATH_PROCESSED: str | Path, PATH_RAW: str | Path
) -> np.ndarray:
    """Creates or loads a volume from a DataFrame for a specified partition.

    This function manages the creation or loading of a 4D volume array based on the DataFrame
    associated with the given partition. It ensures that the volume file is available locally,
    either by loading it if it exists or creating it from the DataFrame if it does not.

    Args:
        partition: The partition to process. Valid options are 'calibration', 'forecasting', 'testing'.
        PATH_PROCESSED: The path to the directory where processed volume data should be stored.
        PATH_RAW: The path to the directory where raw data is located.

    Returns:
        The 4D volume array created or loaded from the DataFrame, with shape
        [n_months, height, width, n_features].

    Example:
        >>> from pathlib import Path
        >>> # Mocking would be required for a full example
    """

    path_vol = os.path.join(str(PATH_PROCESSED), f"{partition}_vol.npy")

    # Create the folders if they don't exist
    os.makedirs(str(PATH_PROCESSED), exist_ok=True)

    # Check if the volume exists
    if os.path.isfile(path_vol):
        logger.info("Volume already created")
        vol = np.load(path_vol)
    else:
        logger.info("Creating volume...")
        path_raw = os.path.join(
            str(PATH_RAW), f"{partition}_viewser_df{PipelineConfig().dataframe_format}"
        )
        vol = df_to_vol(read_dataframe(path_raw))
        logger.info(f"shape of volume: {vol.shape}")
        logger.info(f"Saving volume to {path_vol}")
        np.save(path_vol, vol)

    logger.info("Done")

    return vol
