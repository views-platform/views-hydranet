"""
VolumeSampler: The Pure Lens for spatiotemporal window extraction.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from views_hydranet.utils.sampling_strategies import SAMPLING_STRATEGY_REGISTRY
from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)


class VolumeSampler:
    """
    Mechanical tool for extracting spatiotemporal windows from a global VolumeHandler.
    Strictly follows instructions (target, threshold) provided by the strategic planner.
    """

    def __init__(self, handler: VolumeHandler, config: Dict[str, Any]):
        self.handler = handler
        self.config = config

        # --- THE HANDSHAKE (ADR 013 Section 1) ---
        dim = config["window_dim"]

        h_max = handler.data.shape[handler.get_axis_idx("H")]
        w_max = handler.data.shape[handler.get_axis_idx("W")]

        if dim > h_max or dim > w_max:
            err_msg = (
                f"VolumeSampler Contract Violation: window_dim ({dim}) exceeds "
                f"handler spatial bounds ({h_max}x{w_max})."
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        # Stateful Reproducibility: Use a local generator
        seed = config["np_seed"]
        self.rng = np.random.default_rng(seed)
        logger.info(f"VolumeSampler: Initialized with np_seed={seed}")

        strategy_name = config["sampling_strategy"]
        entry = SAMPLING_STRATEGY_REGISTRY.get(strategy_name)
        if entry is None:
            raise ValueError(
                f"Unknown sampling_strategy='{strategy_name}'. "
                f"Available: {list(SAMPLING_STRATEGY_REGISTRY.keys())}"
            )
        self._select_anchor = entry["fn"]
        self.min_events = config["min_events"]

    def get_train_volume(self) -> VolumeHandler:
        """Slices off the test horizon while preserving the Ledger."""
        steps = self.config["steps"]
        if not steps:
            return self.handler

        total_t = self.handler.data.shape[self.handler.get_axis_idx("T")]
        return self.handler.slice_time(0, total_t - len(steps))

    def _generate_window(self, target_name: str, threshold: int) -> VolumeHandler:
        """Internal: Extracts a single spatial window based on explicit instructions."""
        train_vh = self.get_train_volume()
        vol_data = train_vh.data
        dim = self.config["window_dim"]
        h_max, w_max = vol_data.shape[1], vol_data.shape[2]

        # Zero-Magic: Resolve target index from Ledger
        try:
            target_idx = train_vh.channel_map.index(target_name)
        except ValueError:
            err_msg = f"VolumeSampler: target_name '{target_name}' not found in Ledger."

            logger.error(err_msg)

            raise ValueError(err_msg)

        # Activity Search (Importance Sampling) — strategy-based anchor selection
        activity = np.count_nonzero(vol_data[..., target_idx], axis=0)
        r_anc, c_axc = self._select_anchor(
            activity, threshold, self.min_events, self.rng, self.config
        )

        # Spatial Jitter (prevent center-bias)
        r0 = np.clip(r_anc - self.rng.integers(0, dim), 0, h_max - dim)
        c0 = np.clip(c_axc - self.rng.integers(0, dim), 0, w_max - dim)

        # Atomic Extraction
        data = vol_data[:, r0 : r0 + dim, c0 : c0 + dim, :].copy()

        # Absolute Anchoring: Propagate geographic truth
        p_row, p_col = train_vh.spatial_offset
        h_max, _ = train_vh.shape[1], train_vh.shape[2]  # Global Height

        # r0 is index from the NORTH (top).
        # The South-most raw row index of the patch is:
        # global_south + (global_height - patch_height - r0)
        new_row_offset = p_row + (h_max - dim - r0)

        vh = VolumeHandler(
            data=data,
            axes=train_vh.axes,
            channel_map=train_vh.channel_map,
            time_col=train_vh.time_col,
            id_col=train_vh.id_col,
            spatial_cols=train_vh.spatial_cols,
            identity_cols=train_vh.identity_cols,
            feature_cols=train_vh.feature_cols,
            spatial_offset=(new_row_offset, p_col + c0),
        )

        mem_mb = data.nbytes / (1024**2)
        logger.debug(f"🔍 VolumeSampler: Extracted Window {data.shape} | Memory: {mem_mb:.2f} MB")
        return vh

    def get_batch(
        self, target_name: str, threshold: int, batch_size: int = 1
    ) -> Tuple[List[VolumeHandler], int]:
        """
        Returns a batch of VolumeHandlers based on strategic instructions.
        Also returns the count of available 'busy' cells for logging.
        """
        # 1. Peek at qualified count for transparency
        train_vh = self.get_train_volume()
        try:
            target_idx = train_vh.channel_map.index(target_name)
        except ValueError:
            err_msg = f"VolumeSampler: target_name '{target_name}' not found in Ledger."

            logger.error(err_msg)

            raise ValueError(err_msg)

        activity = np.count_nonzero(train_vh.data[..., target_idx], axis=0)
        qualified_count = len(np.argwhere(activity >= threshold))

        # 2. Generate the batch
        batch = []
        for _ in range(batch_size):
            batch.append(self._generate_window(target_name, threshold))

        return batch, qualified_count
