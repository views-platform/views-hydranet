import numpy as np
import pytest

from views_hydranet.utils.utils_contract_converters import zstack_to_contract_df


@pytest.fixture
def base_data():
    steps, H, W, channels, samples = 1, 1, 1, 3, 2
    # Two samples in log-space: 1.0 and 3.0
    posterior_zstack = np.zeros((steps, H, W, channels, samples))
    posterior_zstack[:, :, :, 0, 0] = 1.0
    posterior_zstack[:, :, :, 0, 1] = 3.0

    meta_zstack = np.zeros((steps, H, W, 8, 1))
    meta_zstack[:, :, :, 0, 0] = 1.0 # Land
    meta_zstack[:, :, :, 3, 0] = 100 # month_id
    return posterior_zstack, meta_zstack

def test_geometric_mean_parity(base_data):
    """Verify: exp(mean(1, 3)) - 1 = exp(2) - 1"""
    post, meta = base_data
    config = {"evalution_mode": "point", "aggregate_method": "geometric_mean"}
    res = zstack_to_contract_df(post, meta, "sb", config=config)[0]

    expected = 2.0
    assert pytest.approx(res.iloc[0]["sb"][0]) == expected

def test_arithmetic_mean_parity(base_data):
    """Verify: mean(exp(1)-1, exp(3)-1)"""
    post, meta = base_data
    config = {"evalution_mode": "point", "aggregate_method": "arithmetic_mean"}
    res = zstack_to_contract_df(post, meta, "sb", config=config)[0]

    val1 = 1.0
    val2 = 3.0
    expected = (val1 + val2) / 2.0
    assert pytest.approx(res.iloc[0]["sb"][0]) == expected

def test_median_parity(base_data):
    """Verify: median is invariant to the transform space."""
    post, meta = base_data
    config = {"evalution_mode": "point", "aggregate_method": "median"}
    res = zstack_to_contract_df(post, meta, "sb", config=config)[0]

    # Median of 1.0 and 3.0 is 2.0. exp(2.0)-1
    expected = 2.0
    assert pytest.approx(res.iloc[0]["sb"][0]) == expected
