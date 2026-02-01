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
        spatial_offset: Tuple[int, int] = (0, 0),
        history: Tuple[Tuple[str, Any], ...] = ()
    ) -> None:
        """
        Low-level constructor. Use from_df() for canonical construction.
        """
        self._data = data
        self._metadata = VolumeMetadata(
            axes=tuple(axes),
            channel_map=tuple(channel_map),
            spatial_offset=spatial_offset,
            history=tuple(history)
        )

    @classmethod
    def from_df(
        cls, 
        df: pd.DataFrame, 
        config: Dict[str, Any], 
        height: int = 180, 
        width: int = 180
    ) -> 'VolumeHandler':
        """Factory: Constructs a VolumeHandler from a standardized DataFrame."""
        identity_cols = config["identity_cols"]
        feature_cols = config["features"]
        channel_map = identity_cols + feature_cols
        
        row_offset = config.get("row_offset", df["row"].min())
        col_offset = config.get("col_offset", df["col"].min())
        
        month_min = df["month_id"].min()
        month_max = df["month_id"].max()
        month_range = int(month_max - month_min + 1)
        
        r_idx = (df["row"] - row_offset).astype(int).values
        c_idx = (df["col"] - col_offset).astype(int).values
        m_idx = (df["month_id"] - month_min).astype(int).values

        n_channels = len(channel_map)
        vol = np.zeros([height, width, month_range, n_channels], dtype=np.float64)

        for i, col_name in enumerate(channel_map):
            vol[r_idx, c_idx, m_idx, i] = df[col_name].values

        vol = np.flip(vol, axis=0) # Canonical North-Up
        vol = np.transpose(vol, (2, 0, 1, 3)) # [T, H, W, C]

        logger.info(f"VolumeHandler: Created volume {vol.shape} anchored at ({row_offset}, {col_offset})")
        return cls(data=vol, axes=("T", "H", "W", "C"), channel_map=channel_map, spatial_offset=(row_offset, col_offset))

    def to_pytorch(
        self, 
        device: torch.device, 
        include_identities: bool = False
    ) -> torch.Tensor:
        """
        Transforms the volume into a model-ready PyTorch tensor.
        Canonical Output Layout: [Batch=1, Time, Channel, Height, Width]
        
        Args:
            device: Target torch device.
            include_identities: If False, slices off the first 5 identity channels.
        """
        # 1. Standardize to float32 NumPy
        if torch.is_tensor(self._data):
            np_data = self._data.detach().cpu().numpy().astype(np.float32)
        else:
            np_data = self._data.astype(np.float32)

        # 2. Slice Channels if requested
        if not include_identities:
            # We assume canonical layout [T, H, W, C] at this gate
            np_data = np_data[..., 5:]

        # 3. Convert to Tensor
        tensor = torch.from_numpy(np_data).to(device)

        # 4. Reshape to PyTorch Sequence Format [B, T, C, H, W]
        # Current (from canonical): [T, H, W, C]
        # We need to move C to dim 1 (relative to T)
        # [T, H, W, C] -> [T, C, H, W]
        tensor = tensor.permute(0, 3, 1, 2)
        # Add Batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor

    def wrap_posterior(
        self, 
        posterior_data: Union[np.ndarray, torch.Tensor], 
        feature_names: List[str]
    ) -> 'VolumeHandler':
        """
        Creates a new VolumeHandler for model outputs (predictions).
        """
        if posterior_data.ndim != self._data.ndim:
            # Handle PyTorch [B, T, C, H, W] to NumPy [T, H, W, C] conversion if needed
            if posterior_data.ndim == 5:
                # Squeeze batch and permute back to canonical
                if torch.is_tensor(posterior_data):
                    posterior_data = posterior_data.squeeze(0).permute(0, 2, 3, 1)
                else:
                    posterior_data = np.squeeze(posterior_data, axis=0).transpose(0, 2, 3, 1)

        return VolumeHandler(
            data=posterior_data,
            axes=("T", "H", "W", "C"),
            channel_map=feature_names,
            spatial_offset=self._metadata.spatial_offset,
            history=self._metadata.history
        )

    def to_df(self, identity_provider: Optional['VolumeHandler'] = None) -> pd.DataFrame:
        """Inverts layout and returns coordinate-aware DataFrame."""
        temp_data = self._data.detach().cpu().numpy() if torch.is_tensor(self._data) else self._data.copy()
        
        t_idx, h_idx, w_idx, c_idx = self.get_axis_idx("T"), self.get_axis_idx("H"), self.get_axis_idx("W"), self.get_axis_idx("C")
        temp_data = np.transpose(temp_data, (h_idx, w_idx, t_idx, c_idx))
        temp_data = np.flip(temp_data, axis=0)
        
        if identity_provider:
            provider_data = identity_provider._data.detach().cpu().numpy() if torch.is_tensor(identity_provider._data) else identity_provider._data.copy()
            p_t, p_h, p_w, p_c = identity_provider.get_axis_idx("T"), identity_provider.get_axis_idx("H"), identity_provider.get_axis_idx("W"), identity_provider.get_axis_idx("C")
            provider_data = np.transpose(provider_data, (p_h, p_w, p_t, p_c))
            provider_data = np.flip(provider_data, axis=0)
            
            mask = provider_data[:, :, :, 0] > 0
            indices = np.where(mask)
            
            reconstructed = {}
            for i in range(5):
                name = identity_provider.channel_map[i]
                reconstructed[name] = provider_data[indices[0], indices[1], indices[2], i].astype(int)
            
            for i, name in enumerate(self.channel_map):
                reconstructed[name] = temp_data[indices[0], indices[1], indices[2], i].astype(np.float32)
        else:
            mask = temp_data[:, :, :, 0] > 0
            indices = np.where(mask)
            reconstructed = {}
            for i, name in enumerate(self.channel_map):
                vals = temp_data[indices[0], indices[1], indices[2], i]
                reconstructed[name] = vals.astype(int) if i < 5 else vals.astype(np.float32)
                
        return pd.DataFrame(reconstructed)

    def visual_audit(self, start_month: int = 0, n_months: int = 5) -> None:
        """Renders visual grid for manual inspection."""
        import matplotlib.pyplot as plt
        plot_data = self._data.detach().cpu().numpy() if torch.is_tensor(self._data) else self._data.copy()
        c_axis, t_axis = self.get_axis_idx("C"), self.get_axis_idx("T")
        actual_n_months = min(n_months, plot_data.shape[t_axis] - start_month)
        n_channels = plot_data.shape[c_axis]
        fig, axes = plt.subplots(actual_n_months, n_channels, figsize=(20, 2 * actual_n_months))
        fig.suptitle(f"Volume Audit: {self._metadata.axes} | Anchor={self._metadata.spatial_offset}", fontsize=16)
        for t in range(actual_n_months):
            for c in range(n_channels):
                ax = axes[t, c] if actual_n_months > 1 else axes[c]
                ax.imshow(plot_data[start_month + t, :, :, c], cmap="rainbow")
                if t == 0: ax.set_title(self.channel_map[c], fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

    @property
    def data(self) -> Union[np.ndarray, torch.Tensor]: return self._data
    @property
    def axes(self) -> Tuple[str, ...]: return self._metadata.axes
    @property
    def channel_map(self) -> Tuple[str, ...]: return self._metadata.channel_map
    @property
    def spatial_offset(self) -> Tuple[int, int]: return self._metadata.spatial_offset
    def get_axis_idx(self, label: str) -> int: return self._metadata.axes.index(label)

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