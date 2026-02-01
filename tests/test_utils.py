import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from views_hydranet.utils.utils import (
    choose_model,
    get_full_tensor,
    get_train_tensors,
    my_decay,
    norm,
    norm_features,
    standard,
    unit_norm,
)


@pytest.fixture
def mock_views_vol():
    """Fixture to create a mock 4D numpy array for views_vol."""
    n_months, height, width, n_features = 36, 180, 180, 8
    return np.random.rand(n_months, height, width, n_features)

@pytest.fixture
def mock_config_train_tensors():
    """
    Fixture for a mock config dictionary used by get_train_tensors.
    """
    return {
        "time_steps": 36,
        "first_feature_idx": 5, # Corresponds to lr_best_sb_idx in utils.py
        "input_channels": 3,
        "window_dim": 16,
        "min_events": 10,
        "samples": 100,
        "slope_ratio": 1.0,
        "roof_ratio": 1.0,
    }

def test_norm_default_range():
    """
    Tests the norm function with default range [0, 1].
    """
    x = np.array([1, 2, 3, 4, 5])
    expected_norm = np.array([0., 0.25, 0.5, 0.75, 1.])
    result = norm(x)
    assert np.allclose(result, expected_norm)

def test_norm_custom_range():
    """
    Tests the norm function with a custom range [a, b].
    """
    x = np.array([10, 20, 30])
    a, b = -1, 1
    expected_norm = np.array([-1., 0., 1.])
    result = norm(x, a, b)
    assert np.allclose(result, expected_norm)

def test_unit_norm_no_noise():
    """
    Tests the unit_norm function without noise.
    """
    x = torch.tensor([3.0, 4.0])
    expected_norm = torch.tensor([0.6, 0.8])
    result = unit_norm(x, noise=False)
    assert torch.allclose(result, expected_norm)

def test_standard_no_noise():
    """
    Tests the standard function without noise.
    """
    x = np.array([1, 2, 3, 4, 5])
    expected_standard = np.array([-1.41421356, -0.70710678,  0.        ,  0.70710678,  1.41421356])
    result = standard(x, noise=False)
    assert np.allclose(result, expected_standard)

def test_my_decay_normal_case():
    """
    Tests my_decay function in a normal scenario where y is between min_events and roof_ratio*max_events.
    """
    sample = 0
    samples = 100
    min_events = 10
    max_events = 100
    slope_ratio = 1.0
    roof_ratio = 0.8 # roof at 80

    # b = ((-100 + 10) / (100 * 1.0)) = -90 / 100 = -0.9
    # y = (100 + (-0.9) * 0) = 100
    # y = min(100, 100 * 0.8) = min(100, 80) = 80
    # y = max(80, 10) = 80
    expected_y = 80
    result = my_decay(sample, samples, min_events, max_events, slope_ratio, roof_ratio)
    assert result == expected_y

def test_my_decay_at_min_events():
    """
    Tests my_decay function when y hits the min_events floor.
    """
    sample = 99 # large sample number to push y towards min_events
    samples = 100
    min_events = 10
    max_events = 100
    slope_ratio = 1.0
    roof_ratio = 1.0 # no roof constraint

    # b = ((-100 + 10) / (100 * 1.0)) = -0.9
    # y = (100 + (-0.9) * 99) = 100 - 89.1 = 10.9
    # y = min(10.9, 100 * 1.0) = 10.9
    # y = max(10.9, 10) = 10.9
    expected_y = 10 # int(10.9) is 10
    result = my_decay(sample, samples, min_events, max_events, slope_ratio, roof_ratio)
    assert result == expected_y

def test_my_decay_at_max_events():
    """
    Tests my_decay function when y is high and not constrained by min_events or roof_ratio.
    """
    sample = 0
    samples = 100
    min_events = 10
    max_events = 100
    slope_ratio = 0.1 # very small slope, so decay is slow
    roof_ratio = 1.0 # no roof constraint

    # b = ((-100 + 10) / (100 * 0.1)) = -90 / 10 = -9
    # y = (100 + (-9) * 0) = 100
    # y = min(100, 100 * 1.0) = 100
    # y = max(100, 10) = 100
    expected_y = 100
    result = my_decay(sample, samples, min_events, max_events, slope_ratio, roof_ratio)
    assert result == expected_y

def test_get_full_tensor_basic_config(mock_views_vol, caplog):
    """
    Tests get_full_tensor with a basic config, verifying output types and shapes.
    """
    mock_config = {"input_channels": 3}

    with caplog.at_level(logging.DEBUG, logger='views_hydranet.utils.utils'): # Specify logger name
        full_tensor, metadata_tensor = get_full_tensor(mock_views_vol, mock_config)

    # Assert types
    assert isinstance(full_tensor, torch.Tensor)
    assert isinstance(metadata_tensor, torch.Tensor)

    # Assert shapes
    n_months, height, width, n_features = mock_views_vol.shape
    expected_full_tensor_shape = (1, n_months, mock_config["input_channels"], height, width)
    expected_metadata_tensor_shape = (1, n_months, 5, height, width) # 5 = default/legacy fallback

    assert full_tensor.shape == expected_full_tensor_shape
    assert metadata_tensor.shape == expected_metadata_tensor_shape

def test_get_full_tensor_data_integrity():
    """
    Tests get_full_tensor for data integrity, ensuring specific values are correctly
    mapped to full_tensor and metadata_tensor after transformations.
    """
    n_months, height, width, n_features = 2, 2, 2, 8
    # Create a mock_views_vol with unique, predictable values for tracing
    mock_views_vol_data = np.arange(n_months * height * width * n_features).reshape(
        n_months, height, width, n_features
    )
    mock_config = {"input_channels": 3}

    # Call the function
    full_tensor, metadata_tensor = get_full_tensor(mock_views_vol_data, mock_config)

    # Hardcoded lr_best_sb_idx from get_full_tensor
    lr_best_sb_idx = 5

    # --- Verify a value in full_tensor ---
    # Choose a value that should go into full_tensor (e.g., from feature index 5, which is lr_best_sb_idx)
    original_month, original_row, original_col = 0, 0, 0
    original_feature_idx_full = lr_best_sb_idx # This is the 6th feature (index 5)
    original_value_full = mock_views_vol_data[original_month, original_row, original_col, original_feature_idx_full]

    # Expected position in full_tensor after unsqueeze(0) and permute(0,1,4,2,3)
    # The permute changes (N, H, W, F) to (N, F, H, W) effectively if only 4D.
    # But it's (N_batch, N_months, N_features, H, W) after unsqueeze and permute
    # new_tensor[batch_dim, month_dim, feature_dim, height_dim, width_dim]
    expected_full_tensor_month_idx = original_month
    expected_full_tensor_feature_idx = original_feature_idx_full - lr_best_sb_idx # Relative index within selected features

    # NOTE: df_to_vol FLIPS the volume. Row 0 (South) -> Row 1 (North) in 2x2 grid
    expected_full_tensor_height_idx = height - 1 - original_row
    expected_full_tensor_width_idx = original_col

    # Pull the value that will actually be at this flipped position
    actual_source_value = mock_views_vol_data[original_month, height - 1 - original_row, original_col, original_feature_idx_full]
    expected_value = np.log1p(actual_source_value)

    # Assert value in full_tensor
    assert torch.isclose(
        full_tensor[
            0, # Batch dimension
            expected_full_tensor_month_idx,
            expected_full_tensor_feature_idx,
            expected_full_tensor_height_idx,
            expected_full_tensor_width_idx,
        ],
        torch.tensor(expected_value, dtype=torch.float32),
    )

    # --- Verify a value in metadata_tensor ---
    # Choose a value that should go into metadata_tensor (e.g., from feature index 0)
    original_feature_idx_meta = 0
    original_value_meta = mock_views_vol_data[original_month, original_row, original_col, original_feature_idx_meta]

    # Expected position in metadata_tensor
    expected_metadata_tensor_month_idx = original_month
    expected_metadata_tensor_feature_idx = original_feature_idx_meta # Absolute index within metadata features
    expected_metadata_tensor_height_idx = original_row
    expected_metadata_tensor_width_idx = original_col

    # Assert value in metadata_tensor
    assert torch.isclose(
        metadata_tensor[
            0, # Batch dimension
            expected_metadata_tensor_month_idx,
            expected_metadata_tensor_feature_idx,
            expected_metadata_tensor_height_idx,
            expected_metadata_tensor_width_idx,
        ],
        torch.tensor(original_value_meta, dtype=torch.float32),
    )

def test_get_full_tensor_none_config(mock_views_vol):
    """
    Tests get_full_tensor when config is None, verifying output shapes and default input_channels.
    """
    # config is None by default
    full_tensor, metadata_tensor = get_full_tensor(mock_views_vol, config=None)

    # Assert types
    assert isinstance(full_tensor, torch.Tensor)
    assert isinstance(metadata_tensor, torch.Tensor)

    # Assert shapes
    n_months, height, width, n_features = mock_views_vol.shape
    # When config is None, input_channels defaults to 3 (hardcoded in get_full_tensor)
    expected_full_tensor_shape = (1, n_months, 3, height, width)
    expected_metadata_tensor_shape = (1, n_months, 5, height, width) # 5 = lr_best_sb_idx (hardcoded)

    assert full_tensor.shape == expected_full_tensor_shape
    assert metadata_tensor.shape == expected_metadata_tensor_shape


@patch('views_hydranet.utils.utils.get_window_index', return_value={'row_indx': 5, 'col_indx': 5}) # Mocking window_index selection
@patch('views_hydranet.utils.utils.get_window_coords', return_value={'min_row_indx': 5, 'max_row_indx': 21, 'min_col_indx': 5, 'max_col_indx': 21, 'dim': 16}) # Mocking window_coords
@patch('views_hydranet.utils.utils.torch.cuda.is_available', return_value=False) # Mock CUDA availability
def test_get_train_tensors_basic(
    mock_cuda_available,
    mock_get_window_coords,
    mock_get_window_index,
    mock_views_vol,
    mock_config_train_tensors
):
    """
    Tests the basic functionality of get_train_tensors, verifying output type and shape.
    """
    # Arrange
    sample = 0
    device = torch.device("cpu") # Since CUDA is mocked to be unavailable

    # Act
    train_tensor = get_train_tensors(mock_views_vol, sample, mock_config_train_tensors, device)

    # Assert
    assert isinstance(train_tensor, torch.Tensor)

    # Expected shape calculation
    time_steps = mock_config_train_tensors["time_steps"]
    input_channels = mock_config_train_tensors["input_channels"]
    window_dim = mock_config_train_tensors["window_dim"]

    # train_views_vol excludes the last 'time_steps' months
    expected_n_months_after_slice = mock_views_vol.shape[0] - time_steps

    # The shape of input_window will be (expected_n_months_after_slice, window_dim, window_dim, original_n_features)
    # After unsqueeze(0), permute(0,1,4,2,3), and slicing lr_best_sb_idx:last_feature_idx
    expected_shape = (
        1, # Batch dim
        expected_n_months_after_slice, # Months
        input_channels, # Sliced features
        window_dim, # Height
        window_dim # Width
    )
    assert train_tensor.shape == expected_shape


@patch('views_hydranet.utils.utils.get_window_index', return_value={'row_indx': 5, 'col_indx': 5}) # Mocking window_index selection
@patch('views_hydranet.utils.utils.get_window_coords', return_value={'min_row_indx': 5, 'max_row_indx': 21, 'min_col_indx': 5, 'max_col_indx': 21, 'dim': 16}) # Mocking window_coords
@patch('views_hydranet.utils.utils.torch.cuda.is_available', return_value=False) # Mock CUDA availability
@patch('torchvision.transforms.RandomHorizontalFlip', return_value=MagicMock(side_effect=lambda x: x)) # Disable flip
@patch('torchvision.transforms.RandomVerticalFlip', return_value=MagicMock(side_effect=lambda x: x)) # Disable flip
def test_get_train_tensors_data_integrity(
    mock_vf,
    mock_hf,
    mock_cuda_available,
    mock_get_window_coords,
    mock_get_window_index,
    mock_config_train_tensors
):
    """
    Tests get_train_tensors for data integrity using uniform values.
    """
    # Arrange
    sample = 0
    device = torch.device("cpu")

    n_months_full, height, width, n_features = 40, 180, 180, 8
    mock_views_vol_data = np.zeros((n_months_full, height, width, n_features))

    # Fill target channel with 50.0
    target_val = 50.0
    mock_views_vol_data[:, :, :, 5] = target_val

    # Act
    train_tensor = get_train_tensors(mock_views_vol_data, sample, mock_config_train_tensors, device)

    # Expect Unconditional Scaling: log1p(50)
    expected_value = np.log1p(target_val)

    # Assert: All pixels in the first feature channel should be the scaled value
    assert torch.allclose(
        train_tensor[0, :, 0, :, :],
        torch.tensor(expected_value, dtype=torch.float32)
    )

@patch('views_hydranet.utils.utils.get_window_index', return_value={'row_indx': 0, 'col_indx': 0}) # Mocking window_index selection
@patch('views_hydranet.utils.utils.get_window_coords', return_value={'min_row_indx': 0, 'max_row_indx': 4, 'min_col_indx': 0, 'max_col_indx': 4, 'dim': 4}) # Mocking window_coords
@patch('views_hydranet.utils.utils.torch.cuda.is_available', return_value=False)
@patch('torchvision.transforms.RandomVerticalFlip', return_value=MagicMock(side_effect=lambda x: x)) # Disable vertical flip
def test_get_train_tensors_spatial_transforms(
    mock_vertical_flip,
    mock_cuda_available,
    mock_get_window_coords,
    mock_get_window_index,
    mock_config_train_tensors
):
    """
    Tests get_train_tensors for correct application of spatial transformations (horizontal/vertical flips).
    Mocks the RandomHorizontalFlip and RandomVerticalFlip to ensure predictable behavior for testing.
    """
    # Arrange
    sample = 0
    device = torch.device("cpu")

    # Create a small, predictable views_vol for easier verification
    n_months_full, original_height, original_width, n_features = 40, 180, 180, 8

    # Create a 4x4 window to test flips
    mock_views_vol_data = np.zeros((n_months_full, original_height, original_width, n_features), dtype=np.float32)
    # Put a distinct pattern in a 4x4 area that will be picked as the window
    # Window (slice from original_height/width): 0:16 for rows, 0:16 for cols
    # Feature 5 (lr_best_sb_idx)
    # Month 0 (from train_views_vol[0])

    # Example pattern for a 4x4 section of feature 5, month 0:
    # 1 2 3 4
    # 5 6 7 8
    # 9 A B C
    # D E F G
    pattern_start_row = 0
    pattern_start_col = 0
    pattern_dim = 4

    # Values that will be in the input_window (feature 5, month 0, rows 0-3, cols 0-3)
    # Using small values for easier numpy array creation and verification
    for r in range(pattern_dim):
        for c in range(pattern_dim):
            mock_views_vol_data[0, r, c, 5] = (r * pattern_dim + c + 1) # Values 1 to 16

    # Configure mock_config_train_tensors
    mock_config_train_tensors["window_dim"] = pattern_dim
    mock_config_train_tensors["first_feature_idx"] = 5
    mock_config_train_tensors["input_channels"] = 1 # Only care about one feature for simplicity

    # The get_window_index and get_window_coords mocks are set at the patch decorator level.

    # Expected transformed pattern (horizontally flipped AND SCALED)
    # 4 3 2 1
    # 8 7 6 5
    # C B A 9
    # G F E D
    # Apply log1p
    full_tensor = torch.tensor([
        [[50, 60, 70], [80, 90, 100], [110, 120, 130]],
        [[140, 150, 160], [170, 180, 190], [200, 210, 220]]
    ], dtype=torch.float32)

    # --- Mock the transforms to ensure a horizontal flip happens ---
    class MockHorizontalFlip(torch.nn.Module):
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p
        def forward(self, img):
            # Simulate a horizontal flip always happening
            return torch.flip(img, dims=[-1]) # Flip along the last spatial dimension (width)

    # Use patch.object to mock the constructor returned by transforms.Compose
    with patch('torchvision.transforms.RandomHorizontalFlip', side_effect=MockHorizontalFlip):
        # Act
        train_tensor = get_train_tensors(mock_views_vol_data, sample, mock_config_train_tensors, device)

        # Assert that the specific feature channel in the window is horizontally flipped
        # train_tensor shape: (1, months, selected_features, H, W)
        # We selected input_channels = 1, so the feature index in train_tensor is 0
        flipped_window_slice = train_tensor[0, 0, 0, :, :] # Batch 0, Month 0, Feature 0, all H, W

        # Check if the pattern is correctly flipped
        assert torch.allclose(flipped_window_slice, expected_flipped_pattern)




@patch('views_hydranet.utils.utils.get_window_index', return_value={'row_indx': 0, 'col_indx': 0})
@patch('views_hydranet.utils.utils.get_window_coords', return_value={'min_row_indx': 0, 'max_row_indx': 4, 'min_col_indx': 0, 'max_col_indx': 4, 'dim': 4})
@patch('views_hydranet.utils.utils.torch.cuda.is_available', return_value=False)
def test_get_train_tensors_spatial_temporal_alignment(
    mock_cuda_available,
    mock_get_window_coords,
    mock_get_window_index,
    mock_config_train_tensors
):
    """
    Tests get_train_tensors to ensure spatial transformations maintain temporal and feature alignment.
    Verifies that flips (horizontal, vertical, both) are applied consistently across months and features.
    """
    # Arrange
    device = torch.device("cpu")
    n_months, height, width, n_features = 2, 4, 4, 2  # Small dimensions for clear tracing
    sample = 0 # Not relevant for fixed window

    # Create a views_vol where each cell (month, row, col, feature) has a unique, traceable value
    # Value = m * 1000 + r * 100 + c * 10 + f
    mock_views_vol_data = np.zeros((n_months, height, width, n_features), dtype=np.float32)
    for m in range(n_months):
        for r in range(height):
            for c in range(width):
                for f in range(n_features):
                    mock_views_vol_data[m, r, c, f] = m * 1000 + r * 100 + c * 10 + f

    # Configure mock_config_train_tensors
    mock_config_train_tensors["window_dim"] = width # Use full width/height for simplicity
    mock_config_train_tensors["first_feature_idx"] = 0 # Start from first feature
    mock_config_train_tensors["input_channels"] = n_features # Use all features
    mock_config_train_tensors["time_steps"] = 0 # Use all months

    # Classes for mocking transforms
    class MockHorizontalFlipAlways:
        def __init__(self, p=0.5):
            pass # Consume p argument
        def __call__(self, img):
            return torch.flip(img, dims=[-1])
    class MockVerticalFlipAlways:
        def __init__(self, p=0.5):
            pass # Consume p argument
        def __call__(self, img):
            return torch.flip(img, dims=[-2])

    # -------------------------------------------------------------------------
    # Helper to prepare expected tensor from original data
    # Input window is always mock_views_vol_data as window_dim=height/width, time_steps=0
    def prepare_expected_tensor(data, h_flip=False, v_flip=False):
    # The data is already in semantic space before reaching this utility
    data = data.copy()

        temp_tensor = torch.tensor(data).float().unsqueeze(dim=0).permute(0,1,4,2,3)
        # Apply permutations and slicing as in get_train_tensors
        # [:, :, lr_best_sb_idx:last_feature_idx, :, :]
        # Since first_feature_idx=0 and input_channels=n_features, this slice is effectively all features

        # Reshape for torchvision transforms
        N, C, D, H, W = temp_tensor.shape
        temp_tensor_reshaped = temp_tensor.reshape(N, C*D, H, W)

        if h_flip:
            temp_tensor_reshaped = torch.flip(temp_tensor_reshaped, dims=[-1])
        if v_flip:
            temp_tensor_reshaped = torch.flip(temp_tensor_reshaped, dims=[-2])

        return temp_tensor_reshaped.reshape(N, C, D, H, W)

    # -------------------------------------------------------------------------
    # Scenario 1: No Flip
    with patch('torchvision.transforms.RandomHorizontalFlip', return_value=MagicMock(side_effect=lambda x: x)), \
         patch('torchvision.transforms.RandomVerticalFlip', return_value=MagicMock(side_effect=lambda x: x)):

        train_tensor = get_train_tensors(mock_views_vol_data, sample, mock_config_train_tensors, device)
        expected_tensor = prepare_expected_tensor(mock_views_vol_data)
        assert torch.allclose(train_tensor, expected_tensor)

    # -------------------------------------------------------------------------
    # Scenario 2: Horizontal Flip Only
    with patch('torchvision.transforms.RandomHorizontalFlip', side_effect=MockHorizontalFlipAlways), \
         patch('torchvision.transforms.RandomVerticalFlip', return_value=MagicMock(side_effect=lambda x: x)):

        train_tensor = get_train_tensors(mock_views_vol_data, sample, mock_config_train_tensors, device)
        expected_tensor = prepare_expected_tensor(mock_views_vol_data, h_flip=True)
        assert torch.allclose(train_tensor, expected_tensor)

    # -------------------------------------------------------------------------
    # Scenario 3: Vertical Flip Only
    with patch('torchvision.transforms.RandomHorizontalFlip', return_value=MagicMock(side_effect=lambda x: x)), \
         patch('torchvision.transforms.RandomVerticalFlip', side_effect=MockVerticalFlipAlways):

        train_tensor = get_train_tensors(mock_views_vol_data, sample, mock_config_train_tensors, device)
        expected_tensor = prepare_expected_tensor(mock_views_vol_data, v_flip=True)
        assert torch.allclose(train_tensor, expected_tensor)

    # -------------------------------------------------------------------------
    # Scenario 4: Both Horizontal and Vertical Flip
    with patch('torchvision.transforms.RandomHorizontalFlip', side_effect=MockHorizontalFlipAlways), \
         patch('torchvision.transforms.RandomVerticalFlip', side_effect=MockVerticalFlipAlways):

        train_tensor = get_train_tensors(mock_views_vol_data, sample, mock_config_train_tensors, device)
        expected_tensor = prepare_expected_tensor(mock_views_vol_data, h_flip=True, v_flip=True)
        assert torch.allclose(train_tensor, expected_tensor)

@pytest.fixture
def mock_config_norm_features():
    """
    Fixture for a mock config dictionary used by norm_features.
    """
    return {
        "first_feature_idx": 1,
        "input_channels": 2,
        "un_log": False,
    }

def test_norm_features_basic(mock_config_norm_features):
    """
    Tests the basic functionality of norm_features, ensuring correct normalization
    of specified features and that other features are untouched.
    """
    # Arrange
    full_vol = np.ones((2, 2, 2, 4), dtype=np.float64)
    # Feature 0: Should be untouched
    full_vol[:, :, :, 0] = 100
    # Feature 1: Should be normalized. Values from 0 to 7
    full_vol[:, :, :, 1] = np.arange(8).reshape(2, 2, 2)
    # Feature 2: Should be normalized. Values from 0 to 14
    full_vol[:, :, :, 2] = np.arange(8).reshape(2, 2, 2) * 2
    # Feature 3: Should be untouched
    full_vol[:, :, :, 3] = 200

    # Create a copy to check for in-place modification
    original_full_vol = full_vol.copy()

    # Act
    result_vol = norm_features(full_vol, mock_config_norm_features)

    # Assert
    # 1. Check for in-place modification
    assert result_vol is full_vol

    # 2. Check untouched features
    assert np.all(result_vol[:, :, :, 0] == 100)
    assert np.all(result_vol[:, :, :, 3] == 200)

    # 3. Check normalized feature 1
    # feature_max = 7, feature_min = 0
    # expected = (1 - 0) * (original - 0) / (7 - 0) + 0 = original / 7
    expected_feature_1 = original_full_vol[:, :, :, 1] / 7.0
    assert np.allclose(result_vol[:, :, :, 1], expected_feature_1)

    # 4. Check normalized feature 2
    # feature_max = 14, feature_min = 0
    # expected = (1 - 0) * (original - 0) / (14 - 0) + 0 = original / 14
    expected_feature_2 = original_full_vol[:, :, :, 2] / 14.0
    assert np.allclose(result_vol[:, :, :, 2], expected_feature_2)




@patch('views_hydranet.utils.utils.HydraBNUNet06_LSTM4')
def test_choose_model_hydra(mock_hydra_model):
    """
    Tests that choose_model correctly selects and instantiates the HydraBNUNet06_LSTM4 model.
    """
    # Arrange
    mock_config = {
        "model": "HydraBNUNet06_LSTM4",
        "input_channels": 3,
        "total_hidden_channels": 64,
        "output_channels": 1,
        "dropout_rate": 0.5
    }
    device = torch.device("cpu")
    mock_model_instance = MagicMock()
    # Configure the mock's 'to' method to return itself
    mock_model_instance.to.return_value = mock_model_instance
    mock_hydra_model.return_value = mock_model_instance

    # Act
    model = choose_model(mock_config, device)

    # Assert
    mock_hydra_model.assert_called_once_with(
        mock_config["input_channels"],
        mock_config["total_hidden_channels"],
        mock_config["output_channels"],
        mock_config["dropout_rate"]
    )
    mock_model_instance.to.assert_called_once_with(device)
    assert model is mock_model_instance


def test_choose_model_unknown_raises_error(caplog):
    """
    Tests that choose_model raises a ValueError for an unknown model name.
    """
    # Arrange
    mock_config = {"model": "unknown_model"}
    device = torch.device("cpu")
    # Act & Assert
    with caplog.at_level(logging.ERROR, logger="views_hydranet.utils.utils"):
        with pytest.raises(ValueError, match="Unknown model type"):
            choose_model(mock_config, device)

    # Assert that the error message was logged
    assert "Unknown model type: unknown_model" in caplog.text
