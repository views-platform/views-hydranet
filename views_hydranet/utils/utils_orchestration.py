import logging
from typing import List

logger = logging.getLogger(__name__)

def get_rolling_origin_indices(
    total_months: int, 
    time_steps: int, 
    num_windows: int
) -> List[int]:
    """
    Calculates the origin indices for a rolling-origin evaluation.
    
    The origins are calculated by rolling backwards from the end of the data,
    ensuring each origin has 'time_steps' months of ground truth following it.
    
    Args:
        total_months: Total number of months in the volume.
        time_steps: Number of steps to forecast.
        num_windows: Number of rolling sequences desired (e.g., 12).
        
    Returns:
        List[int]: Ascending list of origin indices.
    """
    # Last valid origin index where we still have 'time_steps' remaining
    # Example: N=48, TS=36. Last origin is 11. (48 - 36 - 1)
    # Month 11 is the last input, months 12..47 (36 steps) are the output.
    last_origin = total_months - time_steps - 1
    
    if last_origin < 0:
        logger.warning(f"Not enough data for even one window! N={total_months}, TS={time_steps}")
        return []
        
    # We want num_windows, but we might have less data than that.
    actual_windows = min(num_windows, last_origin + 1)
    
    origins = [last_origin - i for i in range(actual_windows)]
    return sorted(origins)
