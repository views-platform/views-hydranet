"""
VolumeSampler: Stochastic windowing for HydraNet Volumes.
"""
from typing import Any, Dict
import numpy as np
from views_hydranet.utils.volume_handler import VolumeHandler

class VolumeSampler:
    def __init__(self, handler: VolumeHandler, config: Dict[str, Any]):
        self.handler = handler
        self.config = config
        self.feature_start_idx = len(config["identity_cols"])

    def get_train_volume(self) -> np.ndarray:
        """Slices off the test horizon."""
        steps = len(self.config.get("steps", []))
        return self.handler.data[:-steps] if steps > 0 else self.handler.data

    def sample_window(self, sample_idx: int) -> VolumeHandler:
        """Extracts a 32x32 window and wraps it in a new Handler."""
        vol = self.get_train_volume()
        dim = self.config["window_dim"]
        h_max, w_max = vol.shape[1], vol.shape[2]

        # 1. Select anchor based on feature activity
        feats = self.config["features"]
        target_idx = self.feature_start_idx + (sample_idx % len(feats))
        activity = np.count_nonzero(vol[..., target_idx], axis=0)
        
        busy_cells = np.argwhere(activity >= self.config.get("min_events", 5))
        if busy_cells.size > 0:
            r_anc, c_axc = busy_cells[np.random.choice(len(busy_cells))]
        else:
            r_anc, c_axc = np.random.randint(0, h_max), np.random.randint(0, w_max)

        # 2. Define spatial bounds
        r0 = np.clip(r_anc - np.random.randint(0, dim), 0, h_max - dim)
        c0 = np.clip(c_axc - np.random.randint(0, dim), 0, w_max - dim)

        # 3. Extract and wrap
        data = vol[:, r0:r0+dim, c0:c0+dim, :].copy()
        
        # Inherit anchor from parent and shift by slice start
        p_row, p_col = self.handler._metadata.spatial_offset
        return VolumeHandler(
            data=data,
            axes=self.handler.axes,
            channel_map=self.handler.channel_map,
            spatial_offset=(p_row + r0, p_col + c0)
        )
