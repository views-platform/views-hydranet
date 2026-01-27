import numpy as np
import pytest
import torch

from views_hydranet.utils.utils import norm, unit_norm, standard, my_decay, get_full_tensor

@pytest.fixture
def mock_views_vol():
    """
    Fixture to create a mock 4D numpy array for views_vol.
    Shape: [n_months, height, width, n_features]
    """
    n_months, height, width, n_features = 36, 180, 180, 8
    return np.random.rand(n_months, height, width, n_features)

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

def test_get_full_tensor_basic_config(mock_views_vol):
    """
    Tests get_full_tensor with a basic config, verifying output types and shapes.
    """
    mock_config = {"input_channels": 3}
    
    # Capture stdout
    from io import StringIO
    import sys
    captured_output = StringIO()
    sys.stdout = captured_output

    full_tensor, metadata_tensor = get_full_tensor(mock_views_vol, mock_config)

    sys.stdout = sys.__stdout__ # Reset stdout

    # Assert types
    assert isinstance(full_tensor, torch.Tensor)
    assert isinstance(metadata_tensor, torch.Tensor)

    # Assert shapes
    n_months, height, width, n_features = mock_views_vol.shape
    expected_full_tensor_shape = (1, n_months, mock_config["input_channels"], height, width)
    expected_metadata_tensor_shape = (1, n_months, 5, height, width) # 5 = ln_best_sb_idx (hardcoded)

    assert full_tensor.shape == expected_full_tensor_shape
    assert metadata_tensor.shape == expected_metadata_tensor_shape
    
    # Assert print messages
    assert f"views_vol shape {mock_views_vol.shape}" in captured_output.getvalue()
    assert f"full_tensor shape {full_tensor.shape}" in captured_output.getvalue()


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

    # Hardcoded ln_best_sb_idx from get_full_tensor
    ln_best_sb_idx = 5

    # --- Verify a value in full_tensor ---
    # Choose a value that should go into full_tensor (e.g., from feature index 5, which is ln_best_sb_idx)
    original_month, original_row, original_col = 0, 0, 0
    original_feature_idx_full = ln_best_sb_idx # This is the 6th feature (index 5)
    original_value_full = mock_views_vol_data[original_month, original_row, original_col, original_feature_idx_full]

    # Expected position in full_tensor after unsqueeze(0) and permute(0,1,4,2,3)
    # The permute changes (N, H, W, F) to (N, F, H, W) effectively if only 4D.
    # But it's (N_batch, N_months, N_features, H, W) after unsqueeze and permute
    # new_tensor[batch_dim, month_dim, feature_dim, height_dim, width_dim]
    expected_full_tensor_month_idx = original_month
    expected_full_tensor_feature_idx = original_feature_idx_full - ln_best_sb_idx # Relative index within selected features
    expected_full_tensor_height_idx = original_row
    expected_full_tensor_width_idx = original_col

    # Assert value in full_tensor
    assert torch.isclose(
        full_tensor[
            0, # Batch dimension
            expected_full_tensor_month_idx,
            expected_full_tensor_feature_idx,
            expected_full_tensor_height_idx,
            expected_full_tensor_width_idx,
        ],
        torch.tensor(original_value_full, dtype=torch.float32),
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
    expected_metadata_tensor_shape = (1, n_months, 5, height, width) # 5 = ln_best_sb_idx (hardcoded)

    assert full_tensor.shape == expected_full_tensor_shape
    assert metadata_tensor.shape == expected_metadata_tensor_shape


