import numpy as np
import pytest
import torch

from views_hydranet.utils.utils import norm, unit_norm, standard, my_decay

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
