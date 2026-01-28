import pytest
from views_hydranet.utils.utils_orchestration import get_rolling_origin_indices

def test_get_rolling_origin_indices_standard():
    """
    48 months, 36 steps, 12 windows -> Origins should be 0..11.
    """
    origins = get_rolling_origin_indices(total_months=48, time_steps=36, num_windows=12)
    assert origins == list(range(12))

def test_get_rolling_origin_indices_forecast():
    """
    True forecasting: 1 window from the very end.
    """
    # If we have 100 months and TS=36. Last origin should be 99 - 36 = 63? 
    # Wait, month 63 is the last input, 64..99 are the 36 steps.
    # total_months = 100. last_origin = 100 - 36 - 1 = 63.
    origins = get_rolling_origin_indices(total_months=100, time_steps=36, num_windows=1)
    assert origins == [63]

def test_get_rolling_origin_indices_limited_data():
    """
    If we ask for 12 but only have 5 valid ones.
    """
    # total=40, TS=36. last_origin = 40 - 36 - 1 = 3.
    # Origins possible: 0, 1, 2, 3. Total 4.
    origins = get_rolling_origin_indices(total_months=40, time_steps=36, num_windows=12)
    assert origins == [0, 1, 2, 3]

def test_get_rolling_origin_indices_insufficient_data():
    """
    Not even enough for one window.
    """
    origins = get_rolling_origin_indices(total_months=30, time_steps=36, num_windows=1)
    assert origins == []
