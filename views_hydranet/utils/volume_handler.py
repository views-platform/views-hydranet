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
    
    # Structural Roles (The names of the columns providing the scaffold)
    time_col: str
    id_col: str
    spatial_cols: Tuple[str, str] # (row_col, col_col)
    
    # Classification
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
        time_col: str,
        id_col: str,
        spatial_cols: Union[List[str], Tuple[str, str]],
        identity_cols: Union[List[str], Tuple[str, ...]] = (),
        feature_cols: Union[List[str], Tuple[str, ...]] = (),
        spatial_offset: Tuple[int, int] = (0, 0),
    ) -> None:
        self._data = data
        self._metadata = VolumeMetadata(
            axes=tuple(axes),
            channel_map=tuple(channel_map),
            time_col=time_col,
            id_col=id_col,
            spatial_cols=tuple(spatial_cols),
            identity_cols=tuple(identity_cols),
            feature_cols=tuple(feature_cols),
            spatial_offset=spatial_offset
        )
        
        # Validation: Channel dimension must match channel_map
        c_idx = self.get_axis_idx("C")
        actual_channels = self._data.shape[c_idx]
        expected_channels = len(self.channel_map)
        if actual_channels != expected_channels:
             raise ValueError(
                 f"VolumeHandler: Channel mismatch! Data has {actual_channels} channels, "
                 f"but channel_map has {expected_channels} names."
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
        # 1. Resolve Ledger Roles from Config (ADR 007 Section 1.1)
        # We enforce that these keys exist to ensure Zero-Magic operation
        try:
            time_col = config["time_col"]
            id_col = config["id_col"]
            y_col, x_col = config["spatial_cols"]
        except KeyError as e:
            raise KeyError(
                f"VolumeHandler Contract Violation: Missing Ledger Role {e} in config.\n"
                f"To comply with ADR 007, your config must define:\n"
                f"  'time_col': The temporal index (e.g., 'month_id')\n"
                f"  'id_col':   The unit index (e.g., 'priogrid_gid')\n"
                f"  'spatial_cols': ['row_col', 'col_col']\n"
            )
        
        identity_cols = config.get("identity_cols", [])
        feature_cols = config.get("features", [])
        
        # --- THE STRICT HANDSHAKE (ADR 007 Section 1.2) ---
        required_roles = [time_col, id_col, y_col, x_col]
        all_required = list(set(required_roles + list(identity_cols) + list(feature_cols)))
        
        missing = [c for c in all_required if c not in df.columns]
        if missing:
            raise ValueError(
                f"VolumeHandler Handshake Failed! Missing columns: {missing}"
            )
        
        channel_map = list(identity_cols) + list(feature_cols)
        
        # 2. Structural Anchoring
        row_offset = config.get("row_offset", df[y_col].min())
        col_offset = config.get("col_offset", df[x_col].min())
        
        month_min = df[time_col].min()
        month_max = df[time_col].max()
        month_range = int(month_max - month_min + 1)
        
        # 3. Coordinate Calculation
        r_idx = (df[y_col] - row_offset).astype(int).values
        c_idx = (df[x_col] - col_offset).astype(int).values
        m_idx = (df[time_col] - month_min).astype(int).values

        # 4. Allocation & Population
        vol = np.zeros([height, width, month_range, len(channel_map)], dtype=np.float64)

        # Dense Identity Population (Temporal)
        try:
            m_chan_idx = channel_map.index(time_col)
            m_vals_global = np.arange(month_min, month_max + 1)
            vol[..., m_chan_idx] = m_vals_global.reshape(1, 1, month_range)
        except ValueError:
            # Time might not be a channel, but a structural role only
            pass

        for i, col_name in enumerate(channel_map):
            vol[r_idx, c_idx, m_idx, i] = df[col_name].values

        # 5. Flip & Layout
        vol = np.flip(vol, axis=0) # North-Up
        vol = np.transpose(vol, (2, 0, 1, 3)) # [T, H, W, C]

        return cls(
            data=vol, 
            axes=("T", "H", "W", "C"), 
            channel_map=channel_map,
            time_col=time_col,
            id_col=id_col,
            spatial_cols=(y_col, x_col),
            identity_cols=identity_cols,
            feature_cols=feature_cols,
            spatial_offset=(row_offset, col_offset)
        )

    def to_pytorch(
        self, 
        device: torch.device, 
        include_identities: bool = False
    ) -> torch.Tensor:
        """
        Transforms the volume into a model-ready PyTorch tensor.
        Canonical Output Layout: [Batch=1, Time, Channel, Height, Width]
        """
        if torch.is_tensor(self._data):
            np_data = self._data.detach().cpu().numpy().astype(np.float32)
        else:
            np_data = self._data.astype(np.float32)

        if not include_identities:
            # Strip identities based on ledger classifications
            # Channels is at index 3 in [T, H, W, C]
            n_identities = len(self._metadata.identity_cols)
            print(f"DEBUG to_pytorch: n_identities={n_identities}, shape_before={np_data.shape}")
            np_data = np_data[:, :, :, n_identities:]
            print(f"DEBUG to_pytorch: shape_after={np_data.shape}")

        tensor = torch.from_numpy(np_data).to(device)
        
        # Current: [Time, Height, Width, Channel]
        # Target:  [Batch=1, Time, Channel, Height, Width]
        tensor = tensor.permute(0, 3, 1, 2) # [T, C, H, W]
        tensor = tensor.unsqueeze(0) # [B, T, C, H, W]

        return tensor

    def wrap_predictions(
        self, 
        posterior_data: Union[np.ndarray, torch.Tensor], 
        feature_names: List[str]
    ) -> 'VolumeHandler':
        """
        Creates a new VolumeHandler for model outputs, anchored to this handler's ledger.
        
        Args:
            posterior_data: Raw model output. Supports [Batch, Time, Channel, H, W] 
                            or [Time, H, W, Channel, Samples].
            feature_names: Names for the prediction channels.
        """
        
        # 1. Handle Dimensionality (Bridge from Model -> Handler)
        # Our internal standard is [T, H, W, C]
        
        if posterior_data.ndim == 5:
            if torch.is_tensor(posterior_data):
                # Assume [B=1, T, C, H, W] -> [T, H, W, C]
                work_data = posterior_data.squeeze(0).permute(0, 2, 3, 1)
            else:
                # Assume [T, H, W, C, S] -> [T, H, W, C] (Take mean of samples)
                work_data = np.mean(posterior_data, axis=-1)
        else:
            work_data = posterior_data

        return VolumeHandler(
            data=work_data,
            axes=("T", "H", "W", "C"),
            channel_map=feature_names,
            time_col=self._metadata.time_col,
            id_col=self._metadata.id_col,
            spatial_cols=self._metadata.spatial_cols,
            identity_cols=self._metadata.identity_cols,
            feature_cols=tuple(feature_names),
            spatial_offset=self._metadata.spatial_offset
        )

    def to_historical_df(self) -> pd.DataFrame:
        """
        Converts the internal volume back to a sparse DataFrame.
        Strictly adheres to the ledger for masking and identity.
        """
        # 1. Move to NumPy and restore [H, W, T, C] orientation
        work_data = self._data.detach().cpu().numpy() if torch.is_tensor(self._data) else self._data.copy()
        
        t_idx, h_idx, w_idx, c_idx = self.get_axis_idx("T"), self.get_axis_idx("H"), self.get_axis_idx("W"), self.get_axis_idx("C")
        work_data = np.transpose(work_data, (h_idx, w_idx, t_idx, c_idx))
        
        # 2. Revert North-Up flip to match raw coordinates
        work_data = np.flip(work_data, axis=0)
        
        # 3. Resolve the Mask (The 'Where') using Ledger roles
        id_col = self._metadata.id_col
        try:
            pg_idx = self.channel_map.index(id_col)
        except ValueError:
            raise ValueError(f"VolumeHandler Ledger Error: ID column '{id_col}' missing from map.")
            
        mask = work_data[:, :, :, pg_idx] > 0
        indices = np.where(mask)
        
        # 4. Reconstruct columns
        df_dict = {}
        for i, col_name in enumerate(self.channel_map):
            values = work_data[indices[0], indices[1], indices[2], i]
            
            # Cast known discrete identities to clean integers
            is_discrete_id = (col_name in self._metadata.identity_cols)
            if is_discrete_id:
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
            time_col=self._metadata.time_col,
            id_col=self._metadata.id_col,
            spatial_cols=self._metadata.spatial_cols,
            identity_cols=self._metadata.identity_cols,
            feature_cols=self._metadata.feature_cols,
            spatial_offset=self._metadata.spatial_offset
        )

    def to_evaluation_df(self, history: 'VolumeHandler', start_idx: int) -> pd.DataFrame:
        """
        Converts predictions to DF by slicing a history provider.
        Strictly enforces that the prediction window exists within history.
        """
        t_idx_self = self.get_axis_idx("T")
        duration = self._data.shape[t_idx_self]
        
        # 1. Strict Contract Validation (ADR 007 Section 3.3)
        history_duration = history.data.shape[history.get_axis_idx("T")]
        if start_idx + duration > history_duration:
            raise ValueError(
                f"VolumeHandler Contract Violation: Evaluation window [index {start_idx} : {start_idx + duration}] "
                f"exceeds history duration ({history_duration}). Use to_forecast_df for the future."
            )

        # 2. Slice the scaffold
        provider_slice = history.slice_time(start_idx, start_idx + duration)
        
        # 3. Align Self (Signal) to [H, W, T, C] and revert flip
        work_data = self._data.detach().cpu().numpy() if torch.is_tensor(self._data) else self._data.copy()
        t_idx, h_idx, w_idx, c_idx = self.get_axis_idx("T"), self.get_axis_idx("H"), self.get_axis_idx("W"), self.get_axis_idx("C")
        work_data = np.transpose(work_data, (h_idx, w_idx, t_idx, c_idx))
        work_data = np.flip(work_data, axis=0)

        # 4. Align Provider (Scaffold) to [H, W, T, C] and revert flip
        provider_data = provider_slice.data.detach().cpu().numpy() if torch.is_tensor(provider_slice.data) else provider_slice.data.copy()
        p_t, p_h, p_w, p_c = provider_slice.get_axis_idx("T"), provider_slice.get_axis_idx("H"), provider_slice.get_axis_idx("W"), provider_slice.get_axis_idx("C")
        provider_data = np.transpose(provider_data, (p_h, p_w, p_t, p_c))
        provider_data = np.flip(provider_data, axis=0)

        # 5. Mask via Scaffold ID (Looked up from Ledger)
        id_col = provider_slice._metadata.id_col
        try:
            pg_idx = provider_slice.channel_map.index(id_col)
        except ValueError:
            raise ValueError(f"VolumeHandler Ledger Error: ID column '{id_col}' missing from scaffold map.")
            
        mask = provider_data[:, :, :, pg_idx] > 0
        indices = np.where(mask)

        # 6. Reconstruct
        df_dict = {}
        # Identities from provider (authoritative scaffold)
        for i, col_name in enumerate(provider_slice.channel_map):
            if col_name in provider_slice._metadata.identity_cols:
                vals = provider_data[indices[0], indices[1], indices[2], i]
                # Clean integer casting for known types
                if col_name in ["priogrid_gid", "month_id", "row", "col", "c_id"]:
                    df_dict[col_name] = vals.astype(int)
                else:
                    df_dict[col_name] = vals

        # Signals from self (authoritative predictions)
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
            
        # 3. Increment month_id (found dynamically from Ledger)
        try:
            m_col = self._metadata.time_col
            m_idx = self.channel_map.index(m_col)
            
            if torch.is_tensor(self._data):
                increments = torch.arange(1, steps + 1, device=self._data.device).view(steps, 1, 1)
                future_vol[..., m_idx] += increments
            else:
                increments = np.arange(1, steps + 1).reshape(steps, 1, 1)
                future_vol[..., m_idx] += increments
        except ValueError:
            # Time column not in channel map
            pass

        return VolumeHandler(
            data=future_vol,
            axes=self._metadata.axes,
            channel_map=self._metadata.channel_map,
            time_col=self._metadata.time_col,
            id_col=self._metadata.id_col,
            spatial_cols=self._metadata.spatial_cols,
            identity_cols=self._metadata.identity_cols,
            feature_cols=self._metadata.feature_cols,
            spatial_offset=self._metadata.spatial_offset
        )

    def to_forecast_df(self, history: 'VolumeHandler') -> pd.DataFrame:
        """
        Converts predictions to DF by extrapolating a history provider.
        """
        duration = self._data.shape[self.get_axis_idx("T")]
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

        # Mask via Scaffold ID (Looked up from Ledger)
        id_col = provider_future._metadata.id_col
        try:
            pg_idx = provider_future.channel_map.index(id_col)
        except ValueError:
            raise ValueError(f"VolumeHandler Ledger Error: ID column '{id_col}' missing from scaffold map.")
            
        mask = provider_data[:, :, :, pg_idx] > 0
        indices = np.where(mask)

        df_dict = {}
        # Identities from provider (authoritative scaffold)
        for i, col_name in enumerate(provider_future.channel_map):
            if col_name in provider_future._metadata.identity_cols:
                vals = provider_data[indices[0], indices[1], indices[2], i]
                if col_name in ["priogrid_gid", "month_id", "row", "col", "c_id"] or col_name == provider_future._metadata.time_col:
                    df_dict[col_name] = vals.astype(int)
                else:
                    df_dict[col_name] = vals

        # Signals from self
        for i, col_name in enumerate(self.channel_map):
            if col_name not in df_dict:
                df_dict[col_name] = work_data[indices[0], indices[1], indices[2], i].astype(np.float32)

        return pd.DataFrame(df_dict)

    @property
    def data(self): return self._data
    @property
    def channel_map(self): return self._metadata.channel_map
    @property
    def id_col(self): return self._metadata.id_col
    @property
    def time_col(self): return self._metadata.time_col
    @property
    def spatial_offset(self): return self._metadata.spatial_offset

    def get_axis_idx(self, label: str) -> int:
        return self._metadata.axes.index(label)
