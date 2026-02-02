"""
VolumeSampler: Stochastic windowing and batch management for HydraNet Volumes.
"""
import logging
from typing import Any, Dict, List
import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler

logger = logging.getLogger(__name__)

class VolumeSampler:
    def __init__(self, handler: VolumeHandler, config: Dict[str, Any]):
        """
        Lensing engine for spatiotemporal volumes.
        Binds a specific strategy to a global handler.
        """
        self.handler = handler
        self.config = config
        
        # --- THE HANDSHAKE (ADR 009 Section 1) ---
        dim = config.get("window_dim")
        if not dim:
            raise ValueError("VolumeSampler Contract Violation: 'window_dim' missing from config.")
            
        h_max = handler.data.shape[handler.get_axis_idx("H")]
        w_max = handler.data.shape[handler.get_axis_idx("W")]
        
        if dim > h_max or dim > w_max:
            raise ValueError(
                f"VolumeSampler Contract Violation: window_dim ({dim}) exceeds "
                f"handler spatial bounds ({h_max}x{w_max})."
            )

        # Batching state
        self.batch_size = config.get("batch_size", 1)
        self._buffer: List[VolumeHandler] = []

        # Stateful Reproducibility: Use a local generator
        seed = config.get("np_seed", 42)
        self.rng = np.random.default_rng(seed)
        logger.info(f"VolumeSampler: Initialized with np_seed={seed}")

    def get_train_volume(self) -> VolumeHandler:
        """Slices off the test horizon while preserving the Ledger."""
        steps = len(self.config.get("steps", []))
        if steps <= 0:
            return self.handler
        
        # We use the handler's internal slicing capability
        total_t = self.handler.data.shape[self.handler.get_axis_idx("T")]
        return self.handler.slice_time(0, total_t - steps)

    def _generate_window(self, sample_idx: int) -> VolumeHandler:
        """Internal: Core logic for extracting a single spatial window."""
        train_vh = self.get_train_volume()
        vol_data = train_vh.data
        dim = self.config["window_dim"]
        h_max, w_max = vol_data.shape[1], vol_data.shape[2]

        # Zero-Magic: Resolve target channel from Ledger
        features = train_vh._metadata.feature_cols
        target_name = features[sample_idx % len(features)]
        target_idx = train_vh.channel_map.index(target_name)
        
        # Activity Search
        activity = np.count_nonzero(vol_data[..., target_idx], axis=0)
        
        busy_cells = np.argwhere(activity >= self.config.get("min_events", 5))
        
        if busy_cells.size > 0:
            r_anc, c_axc = busy_cells[self.rng.choice(len(busy_cells))]
        else:
            r_anc, c_axc = self.rng.integers(0, h_max), self.rng.integers(0, w_max)

        # Spatial Jitter
        r0 = np.clip(r_anc - self.rng.integers(0, dim), 0, h_max - dim)
        c0 = np.clip(c_axc - self.rng.integers(0, dim), 0, w_max - dim)

        # Extraction
        data = vol_data[:, r0:r0+dim, c0:c0+dim, :].copy()
        
        # Absolute Anchoring
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

    def get_next_batch(self, sample_idx: int) -> List[VolumeHandler]:
        """
        Returns a full batch of VolumeHandlers.
        Resets the internal buffer each time.
        """
        self._buffer = []
        for i in range(self.batch_size):
            # We vary the internal seed slightly per batch element 
            # while keeping the base sample_idx for reproducibility
            self._buffer.append(self._generate_window(sample_idx + i))
        
        return self._buffer