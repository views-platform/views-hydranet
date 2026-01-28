from views_hydranet.utils.utils_date_index import calculate_date_from_index


def test_calculate_date_from_index_next_month():
    """
    Tests that calculating the date for the next index returns the next month.
    """
    # Given the default start_index=121 and start_date='01.1990'
    target_index = 122

    # When we calculate the date
    result = calculate_date_from_index(target_index)

    # Then the result should be the next month
    expected_date = '02.1990'
    assert result == expected_date

def test_calculate_date_from_index_explicit_defaults():
    """
    Tests calculate_date_from_index when default start_index and start_date are explicitly provided.
    """
    target_index = 122
    start_index = 121
    start_date = '01.1990'

    result = calculate_date_from_index(target_index, start_index=start_index, start_date=start_date)

    expected_date = '02.1990'
    assert result == expected_date

def test_calculate_date_from_index_same_index():
    """
    Tests calculate_date_from_index when the target_index is the same as the start_index.
    """
    target_index = 121
    start_index = 121
    start_date = '01.1990'

    result = calculate_date_from_index(target_index, start_index=start_index, start_date=start_date)

    expected_date = '01.1990'
    assert result == expected_date

