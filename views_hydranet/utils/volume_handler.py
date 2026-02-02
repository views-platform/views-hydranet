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
    """
    axes: Tuple[str, ...]
    channel_map: Tuple[str, ...]
    identity_cols: Tuple[str, ...]
    feature_cols: Tuple[str, ...]
    spatial_offset: Tuple[int, int]
    history: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

class VolumeHandler:
    def __init__(
        self, 
        data: Union[np.ndarray, torch.Tensor], 
        axes: Union[List[str], Tuple[str, ...]], 
        channel_map: Union[List[str], Tuple[str, ...]],
        identity_cols: Union[List[str], Tuple[str, ...]] = (),
        feature_cols: Union[List[str], Tuple[str, ...]] = (),
        spatial_offset: Tuple[int, int] = (0, 0),
    ) -> None:
        self._data = data
        self._metadata = VolumeMetadata(
            axes=tuple(axes),
            channel_map=tuple(channel_map),
            identity_cols=tuple(identity_cols),
            feature_cols=tuple(feature_cols),
            spatial_offset=spatial_offset
        )

    @classmethod
    def from_df(
        cls, 
        df: pd.DataFrame, 
        config: Dict[str, Any], 
        height: int = 180, 
        width: int = 180
    ) -> 'VolumeHandler':
        identity_cols = config.get("identity_cols", [])
        feature_cols = config.get("features", [])
        channel_map = list(identity_cols) + list(feature_cols)
        
        row_offset = config.get("row_offset", df["row"].min())
        col_offset = config.get("col_offset", df["col"].min())
        month_min = df["month_id"].min()
        month_max = df["month_id"].max()
        month_range = int(month_max - month_min + 1)
        
        r_idx = (df["row"] - row_offset).astype(int).values
        c_idx = (df["col"] - col_offset).astype(int).values
        m_idx = (df["month_id"] - month_min).astype(int).values

        vol = np.zeros([height, width, month_range, len(channel_map)], dtype=np.float64)

        m_chan_idx = channel_map.index("month_id")
        m_vals_global = np.arange(month_min, month_max + 1)
        vol[..., m_chan_idx] = m_vals_global.reshape(1, 1, month_range)

        for i, col_name in enumerate(channel_map):
            vol[r_idx, c_idx, m_idx, i] = df[col_name].values

        vol = np.flip(vol, axis=0) # North-Up
        vol = np.transpose(vol, (2, 0, 1, 3)) # [T, H, W, C]

        return cls(
            data=vol, 
            axes=("T", "H", "W", "C"), 
            channel_map=channel_map,
            identity_cols=identity_cols,
            feature_cols=feature_cols,
            spatial_offset=(row_offset, col_offset)
        )

    def to_historical_df(self) -> pd.DataFrame:
        """
        Converts the internal volume back to a sparse DataFrame.
        Strictly adheres to the ledger for masking and identity.
        """
        # 1. Move to NumPy and restore [H, W, T, C] orientation
        # We need the original spatial layout to apply the North-Up flip correctly
        work_data = self._data.detach().cpu().numpy() if torch.is_tensor(self._data) else self._data.copy()
        
        # Current: [T, H, W, C] -> Target: [H, W, T, C]
        t_idx, h_idx, w_idx, c_idx = self.get_axis_idx("T"), self.get_axis_idx("H"), self.get_axis_idx("W"), self.get_axis_idx("C")
        work_data = np.transpose(work_data, (h_idx, w_idx, t_idx, c_idx))
        
        # 2. Revert North-Up flip to match raw DataFrame coordinates
        work_data = np.flip(work_data, axis=0)
        
        # 3. Resolve the Mask (The 'Where')
        # We find 'priogrid_gid' from the ledger. Only pixels > 0 are Land.
        try:
            pg_idx = self.channel_map.index("priogrid_gid")
        except ValueError:
            raise ValueError("VolumeHandler Ledger Error: 'priogrid_gid' missing. Cannot mask Ocean.")
            
        mask = work_data[:, :, :, pg_idx] > 0
        indices = np.where(mask)
        
        # 4. Reconstruct columns
        df_dict = {}
        for i, col_name in enumerate(self.channel_map):
            values = work_data[indices[0], indices[1], indices[2], i]
            # Ensure identities are clean integers
            if col_name in ["priogrid_gid", "month_id", "row", "col", "c_id"]:
                df_dict[col_name] = values.astype(int)
            else:
                df_dict[col_name] = values.astype(np.float32)
                
        return pd.DataFrame(df_dict)

    def slice_time(self, start_idx: int, end_idx: int) -> 'VolumeHandler':
        """
        Returns a new VolumeHandler containing a temporal subset of the data.
        """
        t_idx = self.get_axis_idx("T")
        slices = [slice(None)] * self._data.ndim
        slices[t_idx] = slice(start_idx, end_idx)
        new_data = self._data[tuple(slices)]

        return VolumeHandler(
            data=new_data,
            axes=self._metadata.axes,
            channel_map=self._metadata.channel_map,
            identity_cols=self._metadata.identity_cols,
            feature_cols=self._metadata.feature_cols,
            spatial_offset=self._metadata.spatial_offset
        )

    def to_evaluation_df(self, history: 'VolumeHandler', start_idx: int) -> pd.DataFrame:
        """
        Converts predictions to DF by slicing a history provider.
        """
        duration = self.data.shape[self.get_axis_idx("T")]
        # 1. Explicitly slice the provider
        provider_slice = history.slice_time(start_idx, start_idx + duration)
        
        # 2. Reconstruct using provider's identities
        work_data = self._data.detach().cpu().numpy() if torch.is_tensor(self._data) else self._data.copy()
        t_idx, h_idx, w_idx, c_idx = self.get_axis_idx("T"), self.get_axis_idx("H"), self.get_axis_idx("W"), self.get_axis_idx("C")
        
        # Transform self to [H, W, T, C] and revert flip
        work_data = np.transpose(work_data, (h_idx, w_idx, t_idx, c_idx))
        work_data = np.flip(work_data, axis=0)

        # Transform provider to [H, W, T, C] and revert flip
        provider_data = provider_slice.data.detach().cpu().numpy() if torch.is_tensor(provider_slice.data) else provider_slice.data.copy()
        provider_data = np.transpose(provider_data, (h_idx, w_idx, t_idx, c_idx))
        provider_data = np.flip(provider_data, axis=0)

        pg_idx = provider_slice.channel_map.index("priogrid_gid")
        mask = provider_data[:, :, :, pg_idx] > 0
        indices = np.where(mask)

        df_dict = {}
        # Identities from provider
        for i, col_name in enumerate(provider_slice.channel_map):
            if col_name in provider_slice._metadata.identity_cols:
                vals = provider_data[indices[0], indices[1], indices[2], i]
                if col_name in ["priogrid_gid", "month_id", "row", "col", "c_id"]:
                    df_dict[col_name] = vals.astype(int)
                else:
                    df_dict[col_name] = vals

        # Signals from self
        for i, col_name in enumerate(self.channel_map):
            if col_name not in df_dict:
                df_dict[col_name] = work_data[indices[0], indices[1], indices[2], i].astype(np.float32)

        return pd.DataFrame(df_dict)

    def extrapolate_time(self, steps: int) -> 'VolumeHandler':
        """
        Creates a future Identity Scaffold by extending the last time step.
        """
        t_idx = self.get_axis_idx("T")
        
        # 1. Get the last frame
        slices = [slice(None)] * self._data.ndim
        slices[t_idx] = slice(-1, None)
        last_frame = self._data[tuple(slices)]
        
        # 2. Repeat for 'steps'
        repeat_shape = [1] * self._data.ndim
        repeat_shape[t_idx] = steps
        
        if torch.is_tensor(self._data):
            future_vol = last_frame.repeat(*repeat_shape)
        else:
            future_vol = np.tile(last_frame, repeat_shape)
            
        # 3. Increment month_id
        try:
            m_idx = self.channel_map.index("month_id")
            if torch.is_tensor(self._data):
                increments = torch.arange(1, steps + 1, device=self._data.device).view(steps, 1, 1)
                future_vol[..., m_idx] += increments
            else:
                increments = np.arange(1, steps + 1).reshape(steps, 1, 1)
                future_vol[..., m_idx] += increments
        except ValueError:
            pass

        return VolumeHandler(
            data=future_vol,
            axes=self._metadata.axes,
            channel_map=self._metadata.channel_map,
            identity_cols=self._metadata.identity_cols,
            feature_cols=self._metadata.feature_cols,
            spatial_offset=self._metadata.spatial_offset
        )

    def to_forecast_df(self, history: 'VolumeHandler') -> pd.DataFrame:
        """
        Converts predictions to DF by extrapolating a history provider.
        """
        duration = self.data.shape[self.get_axis_idx("T")]
        # 1. Explicitly extrapolate history into future
        provider_future = history.extrapolate_time(duration)
        
        # 2. Reconstruct using synthetic identities
        work_data = self._data.detach().cpu().numpy() if torch.is_tensor(self._data) else self._data.copy()
        t_idx, h_idx, w_idx, c_idx = self.get_axis_idx("T"), self.get_axis_idx("H"), self.get_axis_idx("W"), self.get_axis_idx("C")
        work_data = np.transpose(work_data, (h_idx, w_idx, t_idx, c_idx))
        work_data = np.flip(work_data, axis=0)

        provider_data = provider_future.data.detach().cpu().numpy() if torch.is_tensor(provider_future.data) else provider_future.data.copy()
        provider_data = np.transpose(provider_data, (h_idx, w_idx, t_idx, c_idx))
        provider_data = np.flip(provider_data, axis=0)

        pg_idx = provider_future.channel_map.index("priogrid_gid")
        mask = provider_data[:, :, :, pg_idx] > 0
        indices = np.where(mask)

        df_dict = {}
        for i, col_name in enumerate(provider_future.channel_map):
            if col_name in provider_future._metadata.identity_cols:
                vals = provider_data[indices[0], indices[1], indices[2], i]
                if col_name in ["priogrid_gid", "month_id", "row", "col", "c_id"]:
                    df_dict[col_name] = vals.astype(int)
                else:
                    df_dict[col_name] = vals

        for i, col_name in enumerate(self.channel_map):
            if col_name not in df_dict:
                df_dict[col_name] = work_data[indices[0], indices[1], indices[2], i].astype(np.float32)

        return pd.DataFrame(df_dict)

    def get_axis_idx(self, label: str) -> int:
        return self._metadata.axes.index(label)
    
    @property
    def data(self): return self._data
    @property
    def channel_map(self): return self._metadata.channel_map
