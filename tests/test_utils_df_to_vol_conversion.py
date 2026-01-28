from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from views_hydranet.utils.utils_df_to_vol_conversion import (
    calculate_absolute_indices,
    create_or_load_views_vol,
    df_to_vol,
    df_vol_conversion_test,
    get_requried_columns_for_vol,
    plot_vol,
    vol_to_df,
)


@pytest.fixture
def mock_df():
    """
    Fixture to create a mock DataFrame.

    This fixture creates a mock DataFrame with predefined data for testing purposes.
    The DataFrame contains columns for pg_id, col, row, month_id, c_id, ln_sb_best,
    ln_ns_best, and ln_os_best. The DataFrame is used to simulate input data for
    various functions that require a DataFrame as input.

    Returns:
        pd.DataFrame: A mock DataFrame with predefined data.
    """
    data = {
        "priogrid_gid": [1, 2, 3, 4],
        "col": [10, 20, 30, 40],
        "row": [5, 15, 25, 35],
        "month_id": [100, 100, 101, 101],
        "c_id": [1, 1, 1, 1],
        "ln_sb_best": [0.1, 0.2, 0.3, 0.4],
        "ln_ns_best": [0.2, 0.3, 0.4, 0.5],
        "ln_os_best": [0.3, 0.4, 0.5, 0.6],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_vol():
    """
    Fixture to create a mock volume.

    This fixture creates a mock 4D numpy array (volume) with random values for testing
    purposes. The volume has dimensions (2, 180, 180, 8), simulating a 3D spatial grid
    over 2 time steps with 8 features.

    Returns:
        np.ndarray: A mock 4D numpy array with random values.
    """
    return np.random.rand(2, 180, 180, 8)


def test_get_requried_columns_for_vol():
    """
    Test the get_requried_columns_for_vol function.

    This test verifies that the get_requried_columns_for_vol function returns a list
    of required column names for volume conversion. It checks that the returned value
    is a list of strings.

    Raises:
        AssertionError: If the returned value is not a list or if any element in the list
                        is not a string.
    """
    required_columns = get_requried_columns_for_vol()
    assert isinstance(required_columns, list)
    assert all(isinstance(col, str) for col in required_columns)


def test_calculate_absolute_indices(mock_df):
    """
    Test the calculate_absolute_indices function.

    This test verifies that the calculate_absolute_indices function correctly calculates
    the absolute indices for rows, columns, and months. It checks that the output DataFrame
    contains the new columns abs_row, abs_col, and abs_month, and that these columns are
    correctly calculated based on the input DataFrame.

    Args:
        mock_df (pd.DataFrame): A mock DataFrame with predefined data.

    Raises:
        AssertionError: If the output is not a DataFrame, if the new columns are not present,
                        or if the values in the new columns are not correctly calculated.
    """
    df = calculate_absolute_indices(mock_df)
    assert isinstance(df, pd.DataFrame)
    assert "abs_row" in df.columns
    assert "abs_col" in df.columns
    assert "abs_month" in df.columns
    assert df["abs_row"].equals(pd.Series([0, 10, 20, 30]))
    assert df["abs_col"].equals(pd.Series([0, 10, 20, 30]))
    assert df["abs_month"].equals(pd.Series([0, 0, 1, 1]))


def test_calculate_absolute_indices_modifies_in_place(mock_df):
    """
    Tests that calculate_absolute_indices modifies the DataFrame in-place.
    This is undesirable behavior that should be flagged.
    """
    # Make a copy before calling the function
    df_original_copy = mock_df.copy()

    # Call the function
    calculate_absolute_indices(mock_df)

    # Assert that the original DataFrame has been modified and is no longer
    # equal to the copy it was before the function call.
    assert not df_original_copy.equals(mock_df)


def test_df_to_vol(mock_df):
    """
    Test the df_to_vol function.

    This test verifies that the df_to_vol function correctly converts a DataFrame into
    a 4D numpy array (volume). It checks that the output is a numpy array with the expected
    shape and that the volume contains the correct data based on the input DataFrame.

    Args:
        mock_df (pd.DataFrame): A mock DataFrame with predefined data.

    Raises:
        AssertionError: If the output is not a numpy array or if the shape of the array
                        is not as expected.
    """
    vol = df_to_vol(mock_df)
    assert isinstance(vol, np.ndarray)
    assert vol.shape == (2, 180, 180, 8)


def test_vol_to_df(mock_vol):
    """
    Test the vol_to_df function.

    This test verifies that the vol_to_df function correctly converts a 4D numpy array
    (volume) back into a DataFrame. It checks that the output is a DataFrame with the
    expected columns and that the DataFrame contains the correct data based on the input
    volume.

    Args:
        mock_vol (np.ndarray): A mock 4D numpy array with random values.

    Raises:
        AssertionError: If the output is not a DataFrame or if the columns of the DataFrame
                        do not match the expected columns.
    """
    df = vol_to_df(mock_vol)
    assert isinstance(df, pd.DataFrame)
    required_columns = get_requried_columns_for_vol()
    forecast_features = ["ln_sb_best", "ln_ns_best", "ln_os_best"]
    expected_columns = required_columns + forecast_features
    assert set(df.columns) == set(expected_columns)


def test_df_vol_conversion_test(mock_df):
    """
    Test the df_vol_conversion_test function.

    This test verifies that the df_vol_conversion_test function correctly converts a
    DataFrame to a volume and back to a DataFrame, and that the original DataFrame and
    the recreated DataFrame are equal. It checks that the DataFrame contains the expected
    columns and that the data is correctly preserved through the conversion process.

    Args:
        mock_df (pd.DataFrame): A mock DataFrame with predefined data.

    Raises:
        AssertionError: If the original DataFrame and the recreated DataFrame are not equal.
    """
    vol = df_to_vol(mock_df)
    df_vol_conversion_test(mock_df, vol)
    df_recreated = vol_to_df(vol)
    required_columns = get_requried_columns_for_vol()
    forecast_features = ["ln_sb_best", "ln_ns_best", "ln_os_best"]
    vol_features = required_columns + forecast_features
    df_trimmed = mock_df[vol_features]
    df_trimmed = df_trimmed.sort_values(by=["priogrid_gid", "month_id"]).reset_index(drop=True)
    df_recreated = df_recreated.sort_values(by=["priogrid_gid", "month_id"]).reset_index(drop=True)
    assert df_trimmed.equals(df_recreated)


def test_plot_vol(mock_vol):
    """
    Test the plot_vol function.

    This test verifies that the plot_vol function runs without errors and correctly
    generates plots for the given volume. It checks that the function does not raise
    any exceptions during execution.

    Args:
        mock_vol (np.ndarray): A mock 4D numpy array with random values.

    Raises:
        pytest.fail: If the plot_vol function raises any exceptions.
    """
    try:
        plot_vol(mock_vol, month_range=1)
    except Exception as e:
        pytest.fail(f"plot_vol raised an exception: {e}")


def test_df_to_vol_out_of_bounds_raises_error(mock_df):
    """
    Tests that df_to_vol raises a ValueError if the span of row/col indices
    is larger than the provided height/width.
    """
    # 1. Test ROW out of bounds
    # Create a span of rows (5 to 205) greater than height (180)
    mock_df_copy = mock_df.copy()
    mock_df_copy.loc[0, 'row'] = 205 # max
    mock_df_copy.loc[1, 'row'] = 5   # min
    # abs_row max will be 205-5=200, which is > 180

    with pytest.raises(ValueError, match="Maximum row index .* is out of bounds for height"):
        df_to_vol(mock_df_copy)

    # 2. Test COL out of bounds
    # Create a span of cols (10 to 60) greater than width (40)
    mock_df_copy = mock_df.copy()
    mock_df_copy.loc[0, 'col'] = 60 # max
    mock_df_copy.loc[1, 'col'] = 10 # min
    # abs_col max will be 60-10=50, which is > 40

    with pytest.raises(ValueError, match="Maximum column index .* is out of bounds for width"):
        df_to_vol(mock_df_copy, width=40)


def test_df_to_vol_duplicate_indices_raises_error(mock_df):
    """
    Tests that df_to_vol raises a ValueError if there are duplicate
    (priogrid_gid, month_id) pairs in the input DataFrame.
    """
    # Create a DataFrame with a duplicate (priogrid_gid, month_id)
    # The original mock_df has:
    # priogrid_gid: [1, 2, 3, 4]
    # month_id:     [100, 100, 101, 101]
    # To create a duplicate, we can change the third row to have priogrid_gid=1 and month_id=100
    duplicate_df = mock_df.copy()
    duplicate_df.loc[2, 'priogrid_gid'] = 1
    duplicate_df.loc[2, 'month_id'] = 100 # Now (1, 100) is duplicated

    with pytest.raises(ValueError, match="Duplicate entries found for 'priogrid_gid' and 'month_id'"):
        df_to_vol(duplicate_df)


@pytest.fixture
def mock_df_with_nan():
    """
    Fixture to create a mock DataFrame with NaN values in forecast features.
    """
    data = {
        "priogrid_gid": [1, 2, 3, 4],
        "col": [10, 20, 30, 40],
        "row": [5, 15, 25, 35],
        "month_id": [100, 100, 101, 101],
        "c_id": [1, 1, 1, 1],
        "ln_sb_best": [0.1, np.nan, 0.3, 0.4], # NaN here
        "ln_ns_best": [0.2, 0.3, np.nan, 0.5], # NaN here
        "ln_os_best": [0.3, 0.4, 0.5, np.nan], # NaN here
    }
    return pd.DataFrame(data)


def test_df_vol_conversion_with_nan(mock_df_with_nan):
    """
    Tests the consistency of DataFrame and volume array conversions when
    the input DataFrame contains NaN values in forecast features.
    """
    vol = df_to_vol(mock_df_with_nan)
    df_recreated = vol_to_df(vol)

    required_columns = get_requried_columns_for_vol()
    forecast_features = ["ln_sb_best", "ln_ns_best", "ln_os_best"]
    vol_features = required_columns + forecast_features
    df_trimmed = mock_df_with_nan[vol_features]

    # Sort both DataFrames by 'priogrid_gid' and 'month_id'
    df_trimmed = df_trimmed.sort_values(by=["priogrid_gid", "month_id"]).reset_index(drop=True)
    df_recreated = df_recreated.sort_values(by=["priogrid_gid", "month_id"]).reset_index(drop=True)

    # pandas.DataFrame.equals() handles NaN values correctly.
    assert df_trimmed.equals(df_recreated)


def test_df_to_vol_empty_dataframe_raises_error():
    """
    Tests that df_to_vol raises a ValueError when an empty DataFrame is provided.
    """
    empty_df = pd.DataFrame(columns=[
        "priogrid_gid", "col", "row", "month_id", "c_id",
        "ln_sb_best", "ln_ns_best", "ln_os_best"
    ])
    with pytest.raises(ValueError, match="Input DataFrame cannot be empty."):
        df_to_vol(empty_df)


# New test for create_or_load_views_vol
@patch('views_hydranet.utils.utils_df_to_vol_conversion.os.makedirs')
@patch('views_hydranet.utils.utils_df_to_vol_conversion.np.save')
@patch('views_hydranet.utils.utils_df_to_vol_conversion.logger')
def test_create_or_load_views_vol_creates_if_not_exists(
    mock_logger, mock_np_save, mock_os_makedirs, mock_df, mock_vol
):
    """
    Tests that create_or_load_views_vol creates a volume if it does not exist.
    """
    partition = "testing"
    path_processed = Path("/mock/processed")
    path_raw = Path("/mock/raw")

    with patch('views_hydranet.utils.utils_df_to_vol_conversion.os.path.isfile', return_value=False), \
         patch('views_hydranet.utils.utils_df_to_vol_conversion.read_dataframe', return_value=mock_df), \
         patch('views_hydranet.utils.utils_df_to_vol_conversion.df_to_vol', return_value=mock_vol), \
         patch('views_pipeline_core.configs.pipeline.PipelineConfig') as MockPipelineConfig:

        # Mock the PipelineConfig instance and its attribute
        mock_pipeline_config_instance = MagicMock()
        mock_pipeline_config_instance.dataframe_format = ".parquet"
        MockPipelineConfig.return_value = mock_pipeline_config_instance

        result_vol = create_or_load_views_vol(partition, path_processed, path_raw)

        # Assertions
        mock_os_makedirs.assert_called_once_with(str(path_processed), exist_ok=True)
        mock_np_save.assert_called_once_with(str(path_processed / f"{partition}_vol.npy"), mock_vol)
        assert result_vol is mock_vol # Should return the created mock_vol

        # Check logger calls
        mock_logger.info.assert_any_call("Creating volume...")
        mock_logger.info.assert_any_call(f"shape of volume: {mock_vol.shape}")
        mock_logger.info.assert_any_call(f"Saving volume to {path_processed / f'{partition}_vol.npy'}")
        mock_logger.info.assert_any_call("Done")




...
@patch('views_hydranet.utils.utils_df_to_vol_conversion.os.makedirs')
@patch('views_hydranet.utils.utils_df_to_vol_conversion.np.load')
@patch('views_hydranet.utils.utils_df_to_vol_conversion.logger')
def test_create_or_load_views_vol_loads_if_exists(
    mock_logger, mock_np_load, mock_os_makedirs, mock_vol
):
    """
    Tests that create_or_load_views_vol loads a volume if it already exists.
    """
    partition = "testing"
    path_processed = Path("/mock/processed")
    path_raw = Path("/mock/raw")

    mock_np_load.return_value = mock_vol # np.load returns the mock_vol

    with patch('views_hydranet.utils.utils_df_to_vol_conversion.os.path.isfile', return_value=True):
        result_vol = create_or_load_views_vol(partition, path_processed, path_raw)

        # Assertions
        mock_os_makedirs.assert_called_once_with(str(path_processed), exist_ok=True)
        mock_np_load.assert_called_once_with(str(path_processed / f"{partition}_vol.npy"))
        assert result_vol is mock_vol # Should return the loaded mock_vol

        # Check logger calls
        expected_calls = [call("Volume already created"), call("Done")]
        mock_logger.info.assert_has_calls(expected_calls, any_order=False)


def test_df_vol_conversion_data_point_integrity(mock_df):
    """
    Tests that a specific data point from the DataFrame is correctly mapped to the volume
    and back after df_to_vol and vol_to_df operations, verifying positional and value integrity.
    """
    # Use a copy to ensure the original mock_df is not modified by df_to_vol
    df_copy = mock_df.copy()

    # --- Step 1: Trace a specific data point from df_copy to vol ---
    # Pick a data point: let's use the first row's 'ln_sb_best' value
    original_row = df_copy.iloc[0]
    original_value = original_row['ln_sb_best'] # 0.1
    original_pg_id = original_row['priogrid_gid'] # 1
    original_month_id = original_row['month_id'] # 100

    # Calculate expected absolute indices
    min_row = df_copy['row'].min() # 5
    min_col = df_copy['col'].min() # 10
    min_month = df_copy['month_id'].min() # 100

    abs_row = int(original_row['row'] - min_row) # 5 - 5 = 0
    abs_col = int(original_row['col'] - min_col) # 10 - 10 = 0
    abs_month = int(original_row['month_id'] - min_month) # 100 - 100 = 0

    # Define height, width, and forecast features as used in df_to_vol
    height = 180
    width = 180
    forecast_features = ["ln_sb_best", "ln_ns_best", "ln_os_best"]
    required_columns = get_requried_columns_for_vol()
    vol_features = required_columns + forecast_features

    # Find the feature index for 'ln_sb_best'
    feature_index_ln_sb_best = vol_features.index('ln_sb_best') # 5

    # Generate the volume
    vol = df_to_vol(df_copy, height=height, width=width, forecast_features=forecast_features)

    # After np.flip(vol, axis=0)
    # The original abs_row=0 (top-most in original orientation) becomes height - 1 - abs_row = 180 - 1 - 0 = 179
    flipped_abs_row = int(height - 1 - abs_row) # 179

    # After np.transpose(vol, (2, 0, 1, 3))
    # New order: (month_range, height, width, n_features)
    # So, vol_transposed[abs_month, flipped_abs_row, abs_col, feature_index]
    assert np.isclose(vol[abs_month, flipped_abs_row, abs_col, feature_index_ln_sb_best], original_value)

    # --- Step 2: Trace the data point back from vol to df_recreated ---
    df_recreated = vol_to_df(vol, forecast_features=forecast_features)

    # Find the corresponding row in the recreated DataFrame
    # Note: vol_to_df removes priogrid_gid=0, so filter for the correct pg_id and month_id
    recreated_row = df_recreated[
        (df_recreated['priogrid_gid'] == original_pg_id) &
        (df_recreated['month_id'] == original_month_id)
    ]

    # Assert that the value is preserved and correctly located
    assert not recreated_row.empty
    assert np.isclose(recreated_row['ln_sb_best'].iloc[0], original_value)


@patch("matplotlib.pyplot.show")
def test_plot_vol_invalid_month_range_raises_error(mock_show, mock_vol):
    """
    Tests that plot_vol raises a ValueError if month_range is greater than
    the number of time steps in the volume.
    """
    # The mock_vol has 2 months (time steps)
    invalid_month_range = 3

    with pytest.raises(ValueError, match="month_range .* exceeds the number of time steps"):
        plot_vol(mock_vol, month_range=invalid_month_range)




