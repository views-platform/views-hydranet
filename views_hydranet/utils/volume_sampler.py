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
        self.handler = handler
        self.config = config
        self.feature_start_idx = len(config["identity_cols"])
        
        # Batching state
        self.batch_size = config.get("batch_size", 1)
        self._buffer: List[VolumeHandler] = []

        # Stateful Reproducibility: Use a local generator
        seed = config.get("np_seed", 42)
        self.rng = np.random.default_rng(seed)
        logger.info(f"VolumeSampler: Initialized with np_seed={seed}")

    def get_train_volume(self) -> np.ndarray:
        """Slices off the test horizon."""
        steps = len(self.config.get("steps", []))
        return self.handler.data[:-steps] if steps > 0 else self.handler.data

    def _generate_window(self, sample_idx: int) -> VolumeHandler:
        """Internal: Core logic for extracting a single spatial window."""
        vol = self.get_train_volume()
        dim = self.config["window_dim"]
        h_max, w_max = vol.shape[1], vol.shape[2]

        feats = self.config["features"]
        target_idx = self.feature_start_idx + (sample_idx % len(feats))
        activity = np.count_nonzero(vol[..., target_idx], axis=0)
        
        busy_cells = np.argwhere(activity >= self.config.get("min_events", 5))
        if busy_cells.size > 0:
            # Use local RNG
            r_anc, c_axc = busy_cells[self.rng.choice(len(busy_cells))]
        else:
            # Use local RNG
            r_anc, c_axc = self.rng.integers(0, h_max), self.rng.integers(0, w_max)

        # Use local RNG for spatial jitter
        r0 = np.clip(r_anc - self.rng.integers(0, dim), 0, h_max - dim)
        c0 = np.clip(c_axc - self.rng.integers(0, dim), 0, w_max - dim)

        data = vol[:, r0:r0+dim, c0:c0+dim, :].copy()
        p_row, p_col = self.handler.spatial_offset
        return VolumeHandler(
            data=data,
            axes=self.handler.axes,
            channel_map=self.handler.channel_map,
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