"""
VolumeHandler: Authoritative Layout Management for Spatiotemporal Volumes.
"""

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class VolumeMetadata:
    """
    The immutable ledger for a volume's layout.
    Tracks axis meaning, channel mapping, and transformation history.
    """
    axes: Tuple[str, ...]  # e.g., ("T", "H", "W", "C")
    channel_map: Tuple[str, ...]  # Canonical names for the "C" dimension indices
    spatial_offset: Tuple[int, int] # (row_offset, col_offset) for absolute anchoring
    history: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

class VolumeHandler:
    """
    Authority for Volume Layout and Geometric transformations.
    
    This class manages the 'Where' of the data. It ensures that 
    mechanical operations (flip, permute) are tracked and reversible.
    """

    def __init__(
        self, 
        data: Union[np.ndarray, torch.Tensor], 
        axes: Union[List[str], Tuple[str, ...]], 
        channel_map: Union[List[str], Tuple[str, ...]],
        spatial_offset: Tuple[int, int] = (0, 0)
    ) -> None:
        """
        Low-level constructor. Use from_df() for canonical construction.
        """
        self._data = data
        self._metadata = VolumeMetadata(
            axes=tuple(axes),
            channel_map=tuple(channel_map),
            spatial_offset=spatial_offset,
            history=()
        )

    @classmethod
    def from_df(
        cls, 
        df: pd.DataFrame, 
        config: Dict[str, Any], 
        height: int = 180, 
        width: int = 180
    ) -> 'VolumeHandler':
        """
        Factory: Constructs a VolumeHandler from a standardized DataFrame.
        Enforces Absolute Anchoring and North-Up orientation.
        """
        # 1. Identity and Feature Lists from Config
        identity_cols = config["identity_cols"]
        feature_cols = config["features"]
        channel_map = identity_cols + feature_cols
        
        # 2. Extract Offsets (Fixed anchors for Absolute Mapping)
        # If not provided, we anchor to the minimum of the current dataset
        row_offset = config.get("row_offset", df["row"].min())
        col_offset = config.get("col_offset", df["col"].min())
        
        # 3. Time Coordinate
        month_min = df["month_id"].min()
        month_max = df["month_id"].max()
        month_range = int(month_max - month_min + 1)
        
        # 4. Absolute Placement Coordinates
        r_idx = (df["row"] - row_offset).astype(int).values
        c_idx = (df["col"] - col_offset).astype(int).values
        m_idx = (df["month_id"] - month_min).astype(int).values

        # 5. Allocation
        n_channels = len(channel_map)
        vol = np.zeros([height, width, month_range, n_channels], dtype=np.float64)

        # 6. Rasterize
        for i, col_name in enumerate(channel_map):
            vol[r_idx, c_idx, m_idx, i] = df[col_name].values

        # 7. Orientation & Transposition
        vol = np.flip(vol, axis=0) # North-Up
        vol = np.transpose(vol, (2, 0, 1, 3)) # [T, H, W, C]

        logger.info(f"VolumeHandler: Created volume {vol.shape} anchored at ({row_offset}, {col_offset})")
        return cls(
            data=vol, 
            axes=("T", "H", "W", "C"), 
            channel_map=channel_map,
            spatial_offset=(row_offset, col_offset)
        )

    def to_df(self) -> pd.DataFrame:
        """
        Inverts the layout and returns a reconstructed DataFrame.
        Restores absolute coordinates using the stored spatial offsets.
        """
        t_idx = self.get_axis_idx("T")
        h_idx = self.get_axis_idx("H")
        w_idx = self.get_axis_idx("W")
        c_idx = self.get_axis_idx("C")
        
        temp_data = self._data.copy()
        if torch.is_tensor(temp_data):
            temp_data = temp_data.detach().cpu().numpy()
            
        # [T, H, W, C] -> [H, W, T, C]
        temp_data = np.transpose(temp_data, (h_idx, w_idx, t_idx, c_idx))
        # Reverse Flip
        temp_data = np.flip(temp_data, axis=0)
        
        mask = temp_data[:, :, :, 0] > 0 # priogrid_gid
        indices = np.where(mask)
        
        row_offset, col_offset = self._metadata.spatial_offset
        
        reconstructed_data = {}
        for i, col_name in enumerate(self._metadata.channel_map):
            raw_values = temp_data[indices[0], indices[1], indices[2], i]
            
            # Apply offsets to restore absolute coordinates
            # (Note: Month_id is handled naturally because we don't store its offset)
            if i < 5:
                reconstructed_data[col_name] = raw_values.astype(int)
            else:
                reconstructed_data[col_name] = raw_values.astype(np.float32)
                
        return pd.DataFrame(reconstructed_data)

    def visual_audit(self, start_month: int = 0, n_months: int = 5) -> None:
        """Renders visual grid for manual inspection."""
        import matplotlib.pyplot as plt
        plot_data = self._data.detach().cpu().numpy() if torch.is_tensor(self._data) else self._data.copy()
        
        c_axis = self.get_axis_idx("C")
        t_axis = self.get_axis_idx("T")
        
        actual_n_months = min(n_months, plot_data.shape[t_axis] - start_month)
        n_channels = plot_data.shape[c_axis]
        
        fig, axes = plt.subplots(actual_n_months, n_channels, figsize=(20, 2 * actual_n_months))
        fig.suptitle(f"VolumeHandler Visual Audit: Anchor={self._metadata.spatial_offset}", fontsize=16)

        for t in range(actual_n_months):
            for c in range(n_channels):
                ax = axes[t, c]
                ax.imshow(plot_data[start_month + t, :, :, c], cmap="rainbow")
                if t == 0: ax.set_title(self._metadata.channel_map[c], fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    @property
    def data(self) -> Union[np.ndarray, torch.Tensor]:
        return self._data

    @property
    def axes(self) -> Tuple[str, ...]:
        return self._metadata.axes

    @property
    def channel_map(self) -> Tuple[str, ...]:
        """Returns the canonical ordered list of channel names."""
        return self._metadata.channel_map

    @property
    def spatial_offset(self) -> Tuple[int, int]:
        """Returns the (row_offset, col_offset) of the volume."""
        return self._metadata.spatial_offset

    def get_axis_idx(self, label: str) -> int:
        try:
            return self._metadata.axes.index(label)
        except ValueError:
            raise KeyError(f"Axis '{label}' not found in current layout: {self._metadata.axes}")

    def permute(self, dims: Union[List[int], Tuple[int, ...]]) -> 'VolumeHandler':
        dims_tuple = tuple(dims)
        self._data = self._data.permute(*dims_tuple) if torch.is_tensor(self._data) else np.transpose(self._data, dims_tuple)
        new_axes = tuple(self._metadata.axes[i] for i in dims_tuple)
        self._metadata = replace(self._metadata, axes=new_axes, history=self._metadata.history + (("permute", dims_tuple),))
        return self

    def flip(self, axis_label: str) -> 'VolumeHandler':
        idx = self.get_axis_idx(axis_label)
        self._data = torch.flip(self._data, dims=[idx]) if torch.is_tensor(self._data) else np.flip(self._data, axis=idx)
        self._metadata = replace(self._metadata, history=self._metadata.history + (("flip", axis_label),))
        return self