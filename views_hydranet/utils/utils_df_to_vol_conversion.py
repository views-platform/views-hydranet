import logging
import os

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

    .. warning::
        This function modifies the input DataFrame in-place by adding
        'abs_row', 'abs_col', and 'abs_month' columns. The caller should
        be aware of this side-effect. A future refactor should change this
        to return a new DataFrame instead.

    Args:
        df (pd.DataFrame): The input DataFrame. Must contain 'row', 'col',
                           and 'month_id' columns.

    Returns:
        pd.DataFrame: The same DataFrame instance that was passed as input, but
                      now with 'abs_row', 'abs_col', and 'abs_month' columns
                      added.

    Example:
        >>> import pandas as pd
        >>> df_in = pd.DataFrame({
        ...     'row': [5, 15, 25, 35],
        ...     'col': [10, 20, 30, 40],
        ...     'month_id': [100, 100, 101, 101]
        ... })
        >>> df_out = calculate_absolute_indices(df_in.copy()) # Use copy to avoid in-place modification in example
        >>> df_out[['abs_row', 'abs_col', 'abs_month']]
           abs_row  abs_col  abs_month
        0        0        0          0
        1       10       10          0
        2       20       20          1
        3       30       30          1
    """

    # get the first month_id
    month_first = df["month_id"].min()

    # calculate the absolute indices
    df["abs_row"] = df["row"] - df["row"].min()
    df["abs_col"] = df["col"] - df["col"].min()
    df["abs_month"] = df["month_id"] - month_first

    # insure the data types are integers
    df["abs_row"] = df["abs_row"].astype(int)
    df["abs_col"] = df["abs_col"].astype(int)
    df["abs_month"] = df["abs_month"].astype(int)

    return df


def df_to_vol(
    df: pd.DataFrame,
    height: int = 180,
    width: int = 180,
    forecast_features: list[str] = ["ln_sb_best", "ln_ns_best", "ln_os_best"],
) -> np.ndarray:


    """


    Converts a DataFrame into a 4D numpy array (volume) for spatial-temporal data representation.





    This volume format is used by models like HydraNet and other CNN-based models. The resulting


    volume array has dimensions [n_months, height, width, n_features].





    Args:


        df (pd.DataFrame): The input DataFrame containing spatial-temporal data. Must include columns:


                           - 'priogrid_gid': Priogrid ID.


                           - 'col': Column index in the spatial grid.


                           - 'row': Row index in the spatial grid.


                           - 'month_id': Temporal index for months.


                           - 'c_id': Country ID or relevant identifier.





        height (int, optional): The height of the spatial grid. Defaults to 180 which fits Africa and the Middle East.





        width (int, optional): The width of the spatial grid. Defaults to 180 which fits Africa and the Middle East.





        forecast_features (list of str, optional): List of forcast feature columns to include in the volume.


                                                   Defaults to ['ln_sb_best', 'ln_ns_best', 'ln_os_best'].





    Returns:


        np.ndarray: A 4D volume array with shape [n_months, height, width, n_features].


                    Where n_features is the total number of required and forecast features combined. Given the default settings the default shape is [n_months, 180, 180, 8].





    Raises:


        ValueError: If any of the required columns ('priogrid_gid', 'col', 'row', 'month_id', 'c_id') are missing from the DataFrame.





    .. warning::


        This function internally calls `calculate_absolute_indices` which modifies


        the input DataFrame `df` in-place by adding 'abs_row', 'abs_col', and 'abs_month' columns.


        It also prints the shape of the created volume to standard output.





    Returns:


        np.ndarray: A 4D volume array with shape [n_months, height, width, n_features].


                    Where n_features is the total number of required and forecast features combined.


                    Given the default settings, the default shape is [n_months, 180, 180, 8].





    Example:


        >>> import pandas as pd


        >>> import numpy as np


        >>> from io import StringIO


        >>> import sys


        >>> from unittest.mock import patch


        >>> # Create a mock DataFrame


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


        >>> # Capture stdout


        >>> captured_output = StringIO()


        >>> sys.stdout = captured_output


        >>> vol = df_to_vol(mock_df.copy(), height=40, width=40) # Use copy for example, use smaller height/width for brevity


        >>> sys.stdout = sys.__stdout__


        >>> assert isinstance(vol, np.ndarray)


        >>> assert vol.shape == (2, 40, 40, 8) # n_months (101-100+1), height, width, n_features (5 req + 3 forecast)


        >>> "Volume of shape (2, 40, 40, 8) created. Should be (n_months, 180, 180, 8)" in captured_output.getvalue()


        True


    """


    # --- INPUT VALIDATION: Check for empty DataFrame ---


    if df.empty:


        raise ValueError("Input DataFrame cannot be empty.")


    # --- END INPUT VALIDATION ---





    # to get prio grid id out of the index


    df = df.reset_index()

    # --- INPUT VALIDATION: Check for duplicate priogrid_gid and month_id combinations ---
    if df.duplicated(subset=['priogrid_gid', 'month_id']).any():
        duplicate_entries = df[df.duplicated(subset=['priogrid_gid', 'month_id'], keep=False)]
        raise ValueError(
            "Duplicate entries found for 'priogrid_gid' and 'month_id'. "
            "Each priogrid_gid must have a unique month_id. "
            f"Duplicated entries:\n{duplicate_entries}"
        )
    # --- END INPUT VALIDATION ---

    # required_columns = ['priogrid_gid', 'col', 'row', 'month_id', 'c_id']
    required_columns = get_requried_columns_for_vol()

    # print(f'\033[91mRequired columns: {required_columns}\033[0m')
    # print(f'\033[91mDataFrame columns: {df.columns.tolist()}\033[0m')

    for col in required_columns:
        if col not in df.columns.tolist():
            raise ValueError(
                f'Column {col} not found in the DataFrame. Please check your viewser query set in "model"/configs/config_input_data.py'
            )

    vol_features = required_columns + forecast_features

    n_features = len(vol_features)

    month_first = df["month_id"].min()
    month_last = df["month_id"].max()
    month_range = month_last - month_first + 1

    # DANGER! Right now this changes (adds columns) to the input DataFrame. Bad practice change later...
    # You could just do df_abs = calculate_absolute_indices(df) and then use df_abs in the rest of the function.
    # But I dont want to break anything now...
    df = calculate_absolute_indices(df)  # abs_row, abs_col, abs_month needed for the volume

    # --- INPUT VALIDATION ---
    if df["abs_row"].max() >= height:
        raise ValueError(
            f"Maximum row index ({df['abs_row'].max()}) is out of bounds for height {height}."
        )
    if df["abs_col"].max() >= width:
        raise ValueError(
            f"Maximum column index ({df['abs_col'].max()}) is out of bounds for width {width}."
        )
    # --- END INPUT VALIDATION ---

    vol = np.zeros([height, width, month_range, n_features])  # Create the volume array.

    for i, feature in enumerate(vol_features):
        vol[df["abs_row"], df["abs_col"], df["abs_month"], i] = df[feature]

    vol = np.flip(vol, axis=0)  # Flip the rows, so north is up.
    vol = np.transpose(vol, (2, 0, 1, 3))  # Move the month dimension to the front.

    print(f"Volume of shape {vol.shape} created. Should be (n_months, 180, 180, 8)")

    return vol


def vol_to_df(
    vol: np.ndarray,
    forecast_features: list[str] = ["ln_sb_best", "ln_ns_best", "ln_os_best"],
) -> pd.DataFrame:
    """
    Converts a 4D numpy array (volumne) back into a DataFrame.

    This function is used to transform the 4D volume format used by models like HydraNet back into
    a DataFrame. Th purpose is to cehck that the conversion between DataFrame and volume does not alter data,
    thus verifying consistency between df_to_vol and vol_to_df operations.

    Args:
        vol (np.ndarray): The input 4D volume array (created with df_to_vol()) to be converted, with shape
                          [n_months, height, width, n_features].
                          - n_months: Number of temporal steps (months).
                          - height: Height of the spatial grid.
                          - width: Width of the spatial grid.
                          - n_features: Number of features per grid cell.

        forecast_features (list of str, optional): List of feature names corresponding to
                                                   the forecast features in the volume.
                                                   Defaults to ['ln_sb_best', 'ln_ns_best', 'ln_os_best'].

    Returns:
        pd.DataFrame: The DataFrame representation of the volume array containing columns:
                      'priogrid_gid', 'col', 'row', 'month_id', 'c_id', followed by forecast features.
                      Rows where 'priogrid_gid' is 0 are removed. This datafreame should be identical to the original DataFrame used to create the volume via df_to_vol().

    Raises:
        ValueError: If the number of features in the volume does not match the expected number
                    of features (length of required + forecast features)

    .. warning::
        This function prints the shape of the created DataFrame to standard output.

    Returns:
        pd.DataFrame: The DataFrame representation of the volume array containing columns:
                      'priogrid_gid', 'col', 'row', 'month_id', 'c_id', followed by forecast features.
                      Rows where 'priogrid_gid' is 0 are removed. This DataFrame should be identical to the original
                      DataFrame used to create the volume via `df_to_vol()`.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from io import StringIO
        >>> import sys
        >>> # Create a mock volume (e.g., from a previous df_to_vol operation)
        >>> mock_vol = np.zeros((2, 10, 10, 8)) # 2 months, 10x10 grid, 8 features
        >>> # Populate with some dummy data for priogrid_gid, col, row, month_id, c_id
        >>> # and some feature data.
        >>> mock_vol[:, :, :, 0] = np.arange(200).reshape(2, 10, 10) # priogrid_gid
        >>> mock_vol[:, :, :, 1] = 5 # col
        >>> mock_vol[:, :, :, 2] = 5 # row
        >>> mock_vol[:, :, :, 3] = np.array([100, 101]).reshape(2, 1, 1, 1) # month_id
        >>> mock_vol[:, :, :, 4] = 1 # c_id
        >>> mock_vol[:, :, :, 5] = 0.5 # ln_sb_best
        >>> mock_vol[:, :, :, 6] = 0.6 # ln_ns_best
        >>> mock_vol[:, :, :, 7] = 0.7 # ln_os_best
        >>> # Capture stdout
        >>> captured_output = StringIO()
        >>> sys.stdout = captured_output
        >>> df_recreated = vol_to_df(mock_vol)
        >>> sys.stdout = sys.__stdout__
        >>> assert isinstance(df_recreated, pd.DataFrame)
        >>> assert df_recreated.shape == (200, 8) # 2 months * 10 * 10 cells, 8 features
        >>> "DataFrame of shape (200, 8) created. Should be (n_months * 180 * 180, 8)" in captured_output.getvalue()
        True
        >>> # Check removal of priogrid_gid=0 cells (if any were 0 after conversion)
        >>> assert 0 not in df_recreated['priogrid_gid'].values
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

    print(f"DataFrame of shape {df.shape} created. Should be (n_months * 180 * 180, 8)")

    return df


def df_vol_conversion_test(
    df: pd.DataFrame,
    vol: np.ndarray,
    forecast_features: list[str] = ["ln_sb_best", "ln_ns_best", "ln_os_best"],
) -> None:
    """
    Tests the consistency of DataFrame and volume array conversions.

    This unit test verifies that converting a DataFrame to a 4D volume array and back to a DataFrame
    results in the original data. It ensures the `df_to_vol` and `vol_to_df` functions are consistent
    and that data integrity is maintained during the transformations.

    Args:
        df (pd.DataFrame): The original DataFrame containing the spatial-temporal data.
                           Must include columns: 'priogrid_gid', 'col', 'row', 'month_id', 'c_id', and forecast features.

        vol (np.ndarray): The 4D volume array obtained from the DataFrame conversion via df_to_vol().
                          Shape should be [n_months, height, width, n_features].

        forecast_features (list of str, optional): List of feature names included in the volume. Defaults to
                                                   ['ln_sb_best', 'ln_ns_best', 'ln_os_best'].

    Returns:
        None: This function does not return a value. It prints the result of the equivalence test between
              the original DataFrame and the DataFrame recreated from the volume array.
    """

    # Make a copy of the original DataFrame
    df_copy = df.copy()

    # Proof of concept: Check if the copy is the same as the original
    print("Original DataFrame equals its copy:", df.equals(df_copy))

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
    print("Trimmed original DataFrame equals recreated DataFrame from volume:", is_equal)


def plot_vol(vol, month_range, forecast_features=["ln_sb_best", "ln_ns_best", "ln_os_best"]):
    """
    Plots feature maps from a 4D volume array over a specified range of months.

    This function generates and displays plots for each feature in the volume array for the last `month_range` time steps.
    Each subplot corresponds to a different feature map at each time step, allowing visualization of spatial-temporal data.
    The main purpose of this function is to provide a visual representation of the data in the volume array to check that it is sound and as expected.

    Args:
        vol (np.ndarray): The input 4D volume array with shape [n_months, height, width, n_features].
                          - n_months: Number of time steps (months).
                          - height: Height of the spatial grid.
                          - width: Width of the spatial grid.
                          - n_features: Number of features per grid cell.

        month_range (int): The number of recent time steps (months) to plot. This should be less than or equal to the number of months in `vol`.

        forecast_features (list of str, optional): List of additional feature names to include in the plots.
                                                   Defaults to ['ln_sb_best', 'ln_ns_best', 'ln_os_best'].

    Returns:
        None: The function displays the plots for each time step and feature, but does not return any value.

    Raises:
        ValueError: If `month_range` exceeds the number of time steps in `vol`.

    .. warning::
        This function calls `matplotlib.pyplot.show()`, which will block the execution
        of the program until the plot window is closed. When running in an automated
        environment or script, this may require special handling (e.g., running in
        a non-interactive backend or redirecting output).

    Returns:
        None: The function displays the plots for each time step and feature, but does not return any value.

    Example:
        >>> import numpy as np
        >>> # Create a mock volume for demonstration purposes
        >>> mock_vol_example = np.random.rand(3, 10, 10, 8) # 3 months, 10x10 grid, 8 features
        >>> plot_vol(mock_vol_example, month_range=1) # This will open a plot window
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


def create_or_load_views_vol(partition, PATH_PROCESSED, PATH_RAW):
    """
    Creates or loads a volume from a DataFrame for a specified partition.

    This function manages the creation or loading of a 4D volume array based on the DataFrame
    associated with the given partition. It ensures that the volume file is available locally,
    either by loading it if it exists or creating it from the DataFrame if it does not.
    This volume array is used as input data for CNN-based models such as HydraNet.

    Args:
        partition (str): The partition to process. Valid options are 'calibration', 'forecasting', 'testing'.
        PATH_PROCESSED (str or Path): The path to the directory where processed volume data should be stored.

    Returns:
        np.ndarray: The 4D volume array created or loaded from the DataFrame, with shape
                    [n_months, height, width, n_features].

    .. warning::
        This function performs file system operations (creating directories, checking for file existence,
        loading and saving NumPy arrays). It also logs information to the logger. Its behavior depends
        on the existence of files and directories at the specified paths.

    Returns:
        np.ndarray: The 4D volume array created or loaded from the DataFrame, with shape
                    [n_months, height, width, n_features].

    Example:
        >>> from unittest.mock import patch, MagicMock
        >>> from pathlib import Path
        >>> import numpy as np
        >>> import pandas as pd
        >>> # Mock dependencies for the example
        >>> with patch('views_hydranet.utils.utils_df_to_vol_conversion.os.makedirs'), \
        ...      patch('views_hydranet.utils.utils_df_to_vol_conversion.os.path.isfile', return_value=False), \
        ...      patch('views_hydranet.utils.utils_df_to_vol_conversion.np.save'), \
        ...      patch('views_hydranet.utils.utils_df_to_vol_conversion.np.load', return_value=np.zeros((2,10,10,8))), \
        ...      patch('views_hydranet.utils.utils_df_to_vol_conversion.read_dataframe', return_value=MagicMock(spec=pd.DataFrame)), \
        ...      patch('views_hydranet.utils.utils_df_to_vol_conversion.df_to_vol', return_value=np.zeros((2,10,10,8))), \
        ...      patch('views_pipeline_core.configs.pipeline.PipelineConfig') as MockPipelineConfig:
        ...     
        ...     mock_pipeline_config_instance = MagicMock()
        ...     mock_pipeline_config_instance.dataframe_format = ".parquet"
        ...     MockPipelineConfig.return_value = mock_pipeline_config_instance
        ...     
        ...     partition = "testing"
        ...     path_processed = Path("/tmp/processed")
        ...     path_raw = Path("/tmp/raw")
        ...     
        ...     vol = create_or_load_views_vol(partition, path_processed, path_raw)
        ...     assert isinstance(vol, np.ndarray)
        >>> # Example where file exists
        >>> with patch('views_hydranet.utils.utils_df_to_vol_conversion.os.makedirs'), \
        ...      patch('views_hydranet.utils.utils_df_to_vol_conversion.os.path.isfile', return_value=True), \
        ...      patch('views_hydranet.utils.utils_df_to_vol_conversion.np.save'), \
        ...      patch('views_hydranet.utils.utils_df_to_vol_conversion.np.load', return_value=np.ones((2,10,10,8))) as mock_np_load:
        ...     
        ...     partition = "calibration"
        ...     path_processed = Path("/tmp/processed")
        ...     path_raw = Path("/tmp/raw")
        ...     
        ...     vol = create_or_load_views_vol(partition, path_processed, path_raw)
        ...     mock_np_load.assert_called_once()
        ...     assert np.all(vol == 1) # Check content from mock_np_load
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
