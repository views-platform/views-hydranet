import numpy as np
import pytest
import torch

from views_hydranet.utils.utils import norm, unit_norm, standard

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
