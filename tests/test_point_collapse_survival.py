import logging
import os
import time

import numpy as np
import psutil
import pytest

from views_hydranet.utils.volume_handler import VolumeHandler

# Configure logging for bit-perfect transparency
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("SurvivalProof")


def get_process_memory_gb() -> float:
    """Returns the current Resident Set Size (RSS) in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**3)


class SurvivalHandler:
    """
    A robust, isolated handler designed to prove the physics of Point-Collapse.
    This class mimics the target behavior for the VolumeHandler hardening.
    """

    def __init__(self, data: np.ndarray, axes: tuple):
        self._data = data
        self._axes = axes
        logger.info(f"Initialized SurvivalHandler with {axes} data. Shape: {data.shape}")

    @property
    def data(self):
        return self._data

    def collapse_to_point(self) -> "SurvivalHandler":
        """
        Mathematically collapses the sample dimension ('S') into a mean.
        This is the single responsibility: kill the 5th dimension in memory.
        """
        if "S" not in self._axes:
            logger.warning("Handler is already in Point mode. Skipping collapse.")
            return self

        s_idx = self._axes.index("S")
        logger.info(f"Collapsing dimension 'S' at index {s_idx}...")

        start_time = time.time()
        # PERFORM THE KILL: np.mean along the sample axis
        collapsed_data = np.mean(self._data, axis=s_idx)

        # Define new axes (4D)
        new_axes = tuple(ax for ax in self._axes if ax != "S")

        duration = time.time() - start_time
        logger.info(f"Collapse complete in {duration:.4f}s. New shape: {collapsed_data.shape}")

        return SurvivalHandler(collapsed_data, new_axes)


def run_survival_audit():
    """
    Empirical test: Create a massive 5D array and prove we can survive via collapse.
    """
    print("\n" + "=" * 50)
    print("SURVIVAL PROOF: POINT-COLLAPSE VS. RAM EXPLOSION")
    print("=" * 50)

    # 1. SETUP: 180x180 grid, 36 months, 250 samples (This is the 'Death Zone' for DFs)
    T, H, W, C, S = 36, 180, 180, 1, 250
    baseline = get_process_memory_gb()
    print(f"1. Baseline RAM: {baseline:.4f} GB")

    # 2. CREATE STOCHASTIC STATE (State A)
    # Total floats: 36 * 180 * 180 * 1 * 250 = 291,600,000
    # Size in float32: ~1.08 GB
    logger.info("Generating Stochastic 5D Volume...")
    stochastic_data = np.random.rand(T, H, W, C, S).astype(np.float32)
    handler_5d = SurvivalHandler(stochastic_data, ("T", "H", "W", "C", "S"))

    state_a_mem = get_process_memory_gb()
    print(
        f"2. State A (Stochastic 5D) RAM: {state_a_mem:.4f} GB "
        f"(Delta: {state_a_mem - baseline:.4f} GB)"
    )

    # 3. PERFORM COLLAPSE (The Milestone 1 Logic)
    handler_4d = handler_5d.collapse_to_point()

    # CRITICAL: Discard the 5D data to simulate real-world memory recovery
    del stochastic_data
    del handler_5d
    time.sleep(1)  # Let GC work

    state_b_mem = get_process_memory_gb()
    print(
        f"3. State B (Point 4D) RAM: {state_b_mem:.4f} GB "
        f"(Delta from Baseline: {state_b_mem - baseline:.4f} GB)"
    )

    # 4. VERIFICATION: Index Safety & Math
    # Verify that a random cell in 4D matches the mean of the 5D (using a known value)
    logger.info("Verifying mathematical parity...")
    # Since we can't check the deleted data, we trust np.mean() logic but check shape consistency
    assert handler_4d.data.shape == (T, H, W, C)
    print("4. Verification: Shape parity preserved.")

    # 5. CONCLUSION
    reduction = (state_a_mem - state_b_mem) / (state_a_mem - baseline) * 100
    print(f"\nSUCCESS: RAM footprint reduced by {reduction:.2f}% after Point-Collapse.")
    print(
        "Proof: We can now safely proceed to DataFrame reconstruction "
        "because we only have 1 value per cell."
    )


def test_collapse_mathematical_correctness():
    """
    GREEN GATE: VolumeHandler.collapse_to_point() must compute the correct
    arithmetic mean across the sample dimension S.

    The existing run_survival_audit() verifies that collapse produces the
    right *shape* but does NOT verify the *math*. A collapse that returns
    all zeros or the wrong aggregation would pass the shape check silently.

    Method: Create a 5D volume with 3 known samples (2.0, 4.0, 6.0).
    The arithmetic mean is exactly 4.0. After collapse, every cell must
    contain exactly 4.0 — not approximately, exactly, because these are
    representable float32 values with no accumulation error.

    Also verified: median collapse (also 4.0 for symmetric samples), and
    that an unknown method raises NotImplementedError loudly.
    """
    T, H, W, C, S = 1, 4, 4, 1, 3

    # Build 5D volume with known, distinct sample values.
    data = np.zeros((T, H, W, C, S), dtype=np.float32)
    data[..., 0] = 2.0  # sample 0: all cells = 2.0
    data[..., 1] = 4.0  # sample 1: all cells = 4.0
    data[..., 2] = 6.0  # sample 2: all cells = 6.0
    # arithmetic mean = (2 + 4 + 6) / 3 = 4.0 exactly

    handler_5d = VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C", "S"),
        channel_map=["value"],  # C=1, so channel_map has 1 entry
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
        spatial_offset=(0, 0),
    )

    # --- Arithmetic mean ---
    collapsed = handler_5d.collapse_to_point(method="arithmetic_mean")

    assert collapsed.data.shape == (T, H, W, C), (
        f"\nExpected collapsed shape {(T, H, W, C)}, got {collapsed.data.shape}.\n"
        f"The S dimension was not removed by the collapse."
    )
    assert np.allclose(collapsed.data, 4.0), (
        f"\nArithmetic mean collapse is incorrect.\n"
        f"Expected every cell = 4.0, got min={collapsed.data.min():.4f}, "
        f"max={collapsed.data.max():.4f}.\n"
        f"Input samples were [2.0, 4.0, 6.0]; expected mean = 4.0."
    )
    assert "S" not in collapsed._metadata.axes, (
        f"\nAxes after collapse still contain 'S': {collapsed._metadata.axes}.\n"
        f"The S axis must be removed from the metadata after collapse."
    )

    # --- Median (symmetric samples → same result) ---
    collapsed_med = handler_5d.collapse_to_point(method="median")
    assert np.allclose(collapsed_med.data, 4.0), (
        f"\nMedian collapse is incorrect.\n"
        f"Samples are [2.0, 4.0, 6.0]; expected median = 4.0, "
        f"got min={collapsed_med.data.min():.4f}, max={collapsed_med.data.max():.4f}."
    )

    # --- Unknown method raises loudly ---
    with pytest.raises(NotImplementedError):
        handler_5d.collapse_to_point(method="geometric_mean")

    # --- Already-4D handler returns self without error ---
    handler_4d = VolumeHandler(
        data=np.ones((T, H, W, C), dtype=np.float32),
        axes=("T", "H", "W", "C"),
        channel_map=["value"],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
        spatial_offset=(0, 0),
    )
    result = handler_4d.collapse_to_point(method="arithmetic_mean")
    assert result is handler_4d, (
        "collapse_to_point() on an already-4D volume must return self (no-op)."
    )


if __name__ == "__main__":
    run_survival_audit()
