"""
VolumeSampler: The Pure Lens for spatiotemporal window extraction.
"""
import logging
from typing import Any, Dict, List, Tuple
import numpy as np
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
            raise ValueError(
                f"VolumeSampler Contract Violation: window_dim ({dim}) exceeds "
                f"handler spatial bounds ({h_max}x{w_max})."
            )

        # Batching state
        self.windows_per_lesson = config["windows_per_lesson"]
        self._buffer: List[VolumeHandler] = []

        # Stateful Reproducibility: Use a local generator
        seed = config["np_seed"]
        self.rng = np.random.default_rng(seed)
        logger.info(f"VolumeSampler: Initialized with np_seed={seed}")

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
            raise ValueError(f"VolumeSampler: target_name '{target_name}' not found in Ledger.")
        
        # Activity Search (Importance Sampling)
        activity = np.count_nonzero(vol_data[..., target_idx], axis=0)
        busy_cells = np.argwhere(activity >= threshold)
        
        if busy_cells.size > 0:
            # Pick a busy anchor
            r_anc, c_axc = busy_cells[self.rng.choice(len(busy_cells))]
        else:
            # Fallback to random anchor
            r_anc, c_axc = self.rng.integers(0, h_max), self.rng.integers(0, w_max)

        # Spatial Jitter (prevent center-bias)
        r0 = np.clip(r_anc - self.rng.integers(0, dim), 0, h_max - dim)
        c0 = np.clip(c_axc - self.rng.integers(0, dim), 0, w_max - dim)

        # Atomic Extraction
        data = vol_data[:, r0:r0+dim, c0:c0+dim, :].copy()
        
        # Absolute Anchoring: Propagate geographic truth
        p_row, p_col = train_vh.spatial_offset
        
        return VolumeHandler(
            data=data,
            axes=train_vh.axes,
            channel_map=train_vh.channel_map,
            time_col=train_vh.time_col,
            id_col=train_vh.id_col,
            spatial_cols=train_vh.spatial_cols,
            identity_cols=train_vh._metadata.identity_cols,
            feature_cols=train_vh._metadata.feature_cols,
            spatial_offset=(p_row + r0, p_col + c0)
        )

    def get_batch(self, target_name: str, threshold: int, batch_size: int = 1) -> Tuple[List[VolumeHandler], int]:
        """
        Returns a batch of VolumeHandlers based on strategic instructions.
        Also returns the count of available 'busy' cells for logging.
        """
        # 1. Peek at qualified count for transparency
        train_vh = self.get_train_volume()
        try:
            target_idx = train_vh.channel_map.index(target_name)
        except ValueError:
            raise ValueError(f"VolumeSampler: target_name '{target_name}' not found in Ledger.")
            
        activity = np.count_nonzero(train_vh.data[..., target_idx], axis=0)
        qualified_count = len(np.argwhere(activity >= threshold))

        # 2. Generate the batch
        batch = []
        for _ in range(batch_size):
            batch.append(self._generate_window(target_name, threshold))
        
        return batch, qualified_count
