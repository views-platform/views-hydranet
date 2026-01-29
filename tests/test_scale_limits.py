import pytest
import numpy as np
import os
import psutil
import time
from views_hydranet.utils.utils_contract_converters import zstack_to_contract_df

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

@pytest.mark.performance
def test_conversion_memory_linear_scaling():
    """
    Verify that RAM usage scales linearly and doesn't explode for 100 samples.
    """
    # 180x180 is the real grid size
    steps, H, W = 1, 180, 180
    
    # Baseline with 10 samples
    samples_low = 10
    zstack_low = np.zeros((steps, H, W, 3, samples_low), dtype=np.float32)
    meta = np.zeros((steps, H, W, 8, 1), dtype=np.float32)
    meta[:, :, :, 0, 0] = 1.0 # Land
    
    mem_start = get_process_memory()
    zstack_to_contract_df(zstack_low, meta, "sb")
    mem_low = get_process_memory() - mem_start
    
    # Scale to 100 samples
    samples_high = 100
    zstack_high = np.zeros((steps, H, W, 3, samples_high), dtype=np.float32)
    
    mem_start_high = get_process_memory()
    zstack_to_contract_df(zstack_high, meta, "sb")
    mem_high = get_process_memory() - mem_start_high
    
    print(f"\nRAM Usage (10 samples): {mem_low:.2f} MB")
    print(f"RAM Usage (100 samples): {mem_high:.2f} MB")
    
    # A linear-ish increase is expected. If it's more than 15x, 
    # something is wrong with object overhead.
    assert mem_high < (mem_low * 15), "Memory scaling is non-linear! Potential object overhead leak."
    # With 32GB ram, we want to stay well under 10GB per sequence. 
    # mem_high for 180x180x1 at 100 samples should be ~500MB.
    assert mem_high < 2000, f"Memory usage for 100 samples too high: {mem_high:.2f} MB"

def test_large_grid_scalability():
    """
    Simulate a higher resolution grid (e.g. 360x360) to ensure vectorized 
    logic doesn't hit a wall.
    """
    steps, H, W, samples = 1, 360, 360, 10
    zstack = np.zeros((steps, H, W, 3, samples), dtype=np.float32)
    meta = np.zeros((steps, H, W, 8, 1), dtype=np.float32)
    meta[:, :, :, 0, 0] = 1.0
    
    start_time = time.time()
    zstack_to_contract_df(zstack, meta, "sb")
    duration = time.time() - start_time
    
    print(f"\nDuration (360x360 grid): {duration:.2f} s")
    # Should be very fast (< 2 seconds) even for 4x more data
    assert duration < 2.0
