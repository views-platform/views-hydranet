"VolumeHandler: Authoritative Layout Management for Spatiotemporal Volumes."

import gc
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import torch

logger = logging.getLogger(__name__)

# Internal Naming Invariants (ADR 020/032)
LINEAR_PREFIX = "lr_"
BINARY_PREFIX = "by_"
PRED_PREFIX = "pred_"

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
    ) -> 'VolumeHandler':
        """
        Factory: Constructs a VolumeHandler from a standardized DataFrame.
        Enforces Absolute Anchoring and North-Up orientation.
        """
        # 1. Resolve Ledger Roles from Config (ADR 007 Section 1.1)

        height, width = config["height"], config["width"]

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
        # Offsets are also strictly required now
        try:
            row_offset = config["row_offset"]
            col_offset = config["col_offset"]
        except KeyError as e:
             raise KeyError(f"VolumeHandler Contract Violation: Missing mandatory offset {e} in config.")

        all_required = list(set(required_roles + list(identity_cols) + list(feature_cols)))

        missing = [c for c in all_required if c not in df.columns]
        if missing:
            raise ValueError(f"VolumeHandler Handshake Failed! Missing columns: {missing}")

        channel_map = list(identity_cols) + list(feature_cols)

        # 2. Structural Anchoring
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
            pass

        for i, col_name in enumerate(channel_map):
            vol[r_idx, c_idx, m_idx, i] = df[col_name].values

        # 5. Flip & Layout
        vol = np.flip(vol, axis=0) # North-Up
        vol = np.transpose(vol, (2, 0, 1, 3)) # [T, H, W, C]

        mem_mb = vol.nbytes / (1024**2)
        logger.debug(f"💠 VolumeHandler: Created Global Volume {vol.shape} | Memory: {mem_mb:.2f} MB")

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
            # ADR 007 hardening: Strip identity channels by checking the channel map.
            # This ensures only feature_cols reach the model.
            feature_indices = [i for i, name in enumerate(self.channel_map) if name in self._metadata.feature_cols]
            if not feature_indices:
                 # Fallback to legacy count-based stripping if feature_cols is empty
                 # (Protects against un-annotated handlers)
                 n_identities = len(self._metadata.identity_cols)
                 np_data = np_data[:, :, :, n_identities:]
            else:
                 np_data = np_data[:, :, :, feature_indices]

        tensor = torch.from_numpy(np_data).to(device)
        tensor = tensor.permute(0, 3, 1, 2) # [T, C, H, W]
        tensor = tensor.unsqueeze(0) # [B, T, C, H, W]

        return tensor

    def wrap_predictions(
        self,
        posterior_data: Union[np.ndarray, torch.Tensor],
        base_names: List[str]
    ) -> 'VolumeHandler':
        """
        Creates a new VolumeHandler for model outputs, anchored to this handler's ledger.
        Automatically applies ADR 032 naming Engine and Watermarks the volume with IDs.
        """
        # 1. Automated Naming (Internal Symmetry Gate)
        # ADR 032: Naming is derived literally from base features.
        # Rule: reg = pred_{feature}, prob = pred_by_{base} (where feature is lr_{base})
        reg_names = []
        prob_names = []
        for n in base_names:
            if not n.startswith(LINEAR_PREFIX):
                raise ValueError(
                    f"VolumeHandler Contract Violation: Feature '{n}' must start with '{LINEAR_PREFIX}' "
                    f"to conform to ADR 032 naming conventions."
                )
            reg_names.append(f"{PRED_PREFIX}{n}")
            prob_names.append(f"{PRED_PREFIX}{n.replace(LINEAR_PREFIX, BINARY_PREFIX, 1)}")
        
        # 2. THE WATERMARK (Red Team Hardening)
        # We prepend ALL identity channels from the parent to the prediction data
        # so the prediction volumes are "Self-Describing" and Join-Safe.
        time_col = self._metadata.time_col
        id_col = self._metadata.id_col
        
        # Identify all non-feature channels from the parent
        identity_names = [n for n in self.channel_map if n in self._metadata.identity_cols]
        identity_idxs = [self.channel_map.index(n) for n in identity_names]
        
        # Normalize Data Layout to [T, H, W, C, (S)]
        if torch.is_tensor(posterior_data):
            # Input is [B=1, T, C, H, W] -> [T, H, W, C]
            work_data = posterior_data.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        else:
            work_data = posterior_data

        # 3. DURATION GUARD (ADR 015 - Fail Loud)
        # The signal must match the temporal duration of its carrier handler.
        if work_data.shape[0] != self.shape[0]:
            raise ValueError(
                f"VolumeHandler Contract Violation: Signal duration ({work_data.shape[0]}) "
                f"does not match Handler duration ({self.shape[0]})."
            )

        if work_data.ndim == 5:
            # Stochastic: [T, H, W, C, S]
            axes = ("T", "H", "W", "C", "S")
            n_samples = work_data.shape[-1]
            
            # Extract and repeat identity watermarks
            id_vols = []
            for idx in identity_idxs:
                slice_data = self.data[..., idx : idx + 1] # [T, H, W, 1]
                watermark = np.expand_dims(slice_data, axis=-1) # [T, H, W, 1, 1]
                watermark = np.repeat(watermark, n_samples, axis=-1) # [T, H, W, 1, S]
                id_vols.append(watermark)
            
            full_data = np.concatenate(id_vols + [work_data], axis=-2)
        else:
            # Point: [T, H, W, C]
            axes = ("T", "H", "W", "C")
            id_vols = [self.data[..., idx : idx + 1] for idx in identity_idxs]
            full_data = np.concatenate(id_vols + [work_data], axis=-1)

        full_signal_names = list(identity_names) + reg_names + prob_names

        return VolumeHandler(
            data=full_data,
            axes=axes,
            channel_map=full_signal_names,
            time_col=time_col,
            id_col=id_col,
            spatial_cols=self._metadata.spatial_cols,
            identity_cols=tuple(identity_names),
            feature_cols=tuple(reg_names + prob_names),
            spatial_offset=self._metadata.spatial_offset
        )

    def collapse_to_point(self, method: str) -> 'VolumeHandler':
        """
        Mathematically collapses the sample dimension ('S') into a point estimate.
        Governed by ADR 021: Volume Dimension Reduction.
        """
        if "S" not in self._metadata.axes:
            logger.warning("VolumeHandler: collapse_to_point() called on a volume that is already 4D. Skipping.")
            return self

        s_idx = self.get_axis_idx("S")
        logger.info(f"💠 VolumeHandler: Collapsing dimension 'S' via {method} (ADR 021 Survival Gate)")

        if torch.is_tensor(self._data):
            work_data = self._data.detach().cpu().numpy()
        else:
            work_data = self._data

        if method in ["arithmetic_mean", "mean"]:
            collapsed_data = np.mean(work_data, axis=s_idx)
        elif method == "median":
            collapsed_data = np.median(work_data, axis=s_idx)
        else:
            raise NotImplementedError(f"Collapse method '{method}' is not defined in ADR 021. Must be 'arithmetic_mean' or 'median'.")

        # Update axes: Filter out 'S'
        new_axes = tuple(ax for ax in self._metadata.axes if ax != "S")

        return VolumeHandler(
            data=collapsed_data,
            axes=new_axes,
            channel_map=self.channel_map,
            time_col=self._metadata.time_col,
            id_col=self._metadata.id_col,
            spatial_cols=self._metadata.spatial_cols,
            identity_cols=self._metadata.identity_cols,
            feature_cols=self._metadata.feature_cols,
            spatial_offset=self._metadata.spatial_offset
        )

    def to_historical_df(self) -> pd.DataFrame:
        """
        Converts the internal volume back to a sparse DataFrame.
        """
        return self._reconstruct_from_provider(self)

    def to_evaluation_df(self, history: 'VolumeHandler', start_idx: int) -> pd.DataFrame:
        """
        Converts predictions to DF by slicing a history provider.
        """
        duration = self.data.shape[self.get_axis_idx("T")]

        # Contract Validation
        history_duration = history.data.shape[history.get_axis_idx("T")]
        if start_idx + duration > history_duration:
            raise ValueError(
                f"VolumeHandler Contract Violation: Evaluation window [index {start_idx} : {start_idx + duration}] "
                f"exceeds history duration ({history_duration})."
            )

        provider_slice = history.slice_time(start_idx, start_idx + duration)
        return self._reconstruct_from_provider(provider_slice)

    def to_forecast_df(self, history: 'VolumeHandler') -> pd.DataFrame:
        """
        Converts predictions to DF by extrapolating a history provider.
        """
        duration = self._data.shape[self.get_axis_idx("T")]
        provider_future = history.extrapolate_time(duration)
        return self._reconstruct_from_provider(provider_future)

    def _reconstruct_from_provider(self, provider: 'VolumeHandler') -> pd.DataFrame:
        """
        Shared logic: Align, Mask, Flatten, and Combine.
        Handles both Point (4D) and Stochastic (5D) volumes.
        Uses an Iterative Watermarked Bridge (ADR 023) to ensure 
        absolute topographic integrity and RAM scalability.
        """
        # 1. Align Self (Signal)
        temp_data = self._data.detach().cpu().numpy() if torch.is_tensor(self._data) else self._data.copy()
        has_samples = "S" in self._metadata.axes

        if has_samples:
            t_idx, h_idx, w_idx, c_idx, s_idx = self.get_axis_idx("T"), self.get_axis_idx("H"), self.get_axis_idx("W"), self.get_axis_idx("C"), self.get_axis_idx("S")
            temp_data = np.transpose(temp_data, (h_idx, w_idx, t_idx, c_idx, s_idx))
            temp_data = np.flip(temp_data, axis=0)
        else:
            t_idx, h_idx, w_idx, c_idx = self.get_axis_idx("T"), self.get_axis_idx("H"), self.get_axis_idx("W"), self.get_axis_idx("C")
            temp_data = np.transpose(temp_data, (h_idx, w_idx, t_idx, c_idx))
            temp_data = np.flip(temp_data, axis=0)

        # 2. Align Provider (Scaffold)
        p_data = provider.data.detach().cpu().numpy() if torch.is_tensor(provider.data) else provider.data.copy()
        p_t, p_h, p_w, p_c = provider.get_axis_idx("T"), provider.get_axis_idx("H"), provider.get_axis_idx("W"), provider.get_axis_idx("C")
        p_data = np.transpose(p_data, (p_h, p_w, p_t, p_c))
        p_data = np.flip(p_data, axis=0)

        # 3. Mask via Scaffold ID
        id_col = provider._metadata.id_col
        pg_idx = provider.channel_map.index(id_col)
        time_col = provider._metadata.time_col
        time_idx = provider.channel_map.index(time_col)
        
        mask = p_data[:, :, :, pg_idx] > 0
        indices = np.where(mask)

        # 4. Initialize Polars Scaffold (The Source of Truth)
        # ADR 032: We MUST carry month_id, priogrid_gid, row, col, and c_id
        scaffold_cols = {
            time_col: p_data[indices[0], indices[1], indices[2], time_idx].astype(np.int32),
            id_col: p_data[indices[0], indices[1], indices[2], pg_idx].astype(np.int32)
        }
        
        # Add Actuals and Identities to Scaffold
        # We explicitly generate binary actuals (by_) from linear actuals (lr_)
        for i, name in enumerate(provider.channel_map):
            if name in [time_col, id_col]: continue
            
            # Carry Identity Columns (row, col, c_id, etc) from provider
            if name in provider._metadata.identity_cols:
                scaffold_cols[name] = p_data[indices[0], indices[1], indices[2], i].astype(np.float32)
            
            # Carry Linear Actuals and Generate Binary Derivatives
            elif name in provider._metadata.feature_cols:
                # Store the Linear Actual
                scaffold_cols[name] = p_data[indices[0], indices[1], indices[2], i].astype(np.float32)
                
                # ADR 032: Generate Binary Derivative (by_) if name is linear (lr_)
                if name.startswith(LINEAR_PREFIX):
                    binary_name = name.replace(LINEAR_PREFIX, BINARY_PREFIX, 1)
                    if binary_name not in provider.channel_map: # Don't overwrite if it exists
                        scaffold_cols[binary_name] = (p_data[indices[0], indices[1], indices[2], i] > 0).astype(np.float32)

        pl_master = pl.DataFrame(scaffold_cols)
        del p_data, scaffold_cols
        gc.collect()

        # 5. Iterative Watermarked Join (One Column at a Time)
        try:
            head_id_idx = self.channel_map.index(id_col)
            head_time_idx = self.channel_map.index(time_col)
        except ValueError:
            head_id_idx = None 
            head_time_idx = None

        for i, name in enumerate(self.channel_map):
            if name in pl_master.columns: continue
            
            # Create Watermarked Head
            if head_id_idx is not None:
                if has_samples:
                    # STOCHASTIC WATERMARK: Extract first sample only for IDs (Join keys must be scalar)
                    head_dict = {
                        time_col: temp_data[indices[0], indices[1], indices[2], head_time_idx, 0].astype(np.int32),
                        id_col: temp_data[indices[0], indices[1], indices[2], head_id_idx, 0].astype(np.int32),
                        name: temp_data[indices[0], indices[1], indices[2], i, :]
                    }
                else:
                    # POINT WATERMARK: Standard extraction
                    head_dict = {
                        time_col: temp_data[indices[0], indices[1], indices[2], head_time_idx].astype(np.int32),
                        id_col: temp_data[indices[0], indices[1], indices[2], head_id_idx].astype(np.int32),
                        name: temp_data[indices[0], indices[1], indices[2], i]
                    }
            else:
                # POSITION FALLBACK: Re-use scaffold IDs (Vulnerable to shuffle)
                head_dict = {
                    time_col: pl_master[time_col], 
                    id_col: pl_master[id_col],
                    name: temp_data[indices[0], indices[1], indices[2], i, :] if has_samples else temp_data[indices[0], indices[1], indices[2], i]
                }

            df_head = pl.DataFrame(head_dict)
            
            # Explicit Join (Red Team Proof)
            pl_master = pl_master.join(df_head, on=[time_col, id_col], how="left")
            del df_head
            gc.collect()

        # 6. Iterative Safe Handshake to Pandas (Legacy Compatibility)
        # We build the Pandas DataFrame column by column to keep the 'Object Tax'
        # limited to exactly one column's worth of Python lists.
        df_out = pl_master.select([time_col, id_col]).to_pandas()
        
        for col in pl_master.columns:
            if col in [time_col, id_col]: continue
            
            # ADR 032: Prefixes pred_lr_ and pred_by_ are already applied in wrap_predictions.
            # Actuals (lr_, by_) and Identities (row, col, c_id) carry their literal names.
            new_name = col

            # Safe Export: Use to_list() for stochastic channels to ensure compatibility
            if has_samples and col in self.channel_map:
                df_out[new_name] = pl_master[col].to_list()
            else:
                df_out[new_name] = pl_master[col].to_pandas()
            
            gc.collect()

        # 7. Final Topographical Restoration
        if time_col in df_out.columns and id_col in df_out.columns:
            df_out = df_out.set_index([time_col, id_col])

        return df_out

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

    def extrapolate_time(self, steps: int) -> 'VolumeHandler':
        """
        Creates a future Identity Scaffold by extending the last time step.
        """
        t_idx = self.get_axis_idx("T")
        slices = [slice(None)] * self._data.ndim
        slices[t_idx] = slice(-1, None)
        last_frame = self._data[tuple(slices)]

        repeat_shape = [1] * self._data.ndim
        repeat_shape[t_idx] = steps

        if torch.is_tensor(self._data):
            future_vol = last_frame.repeat(*repeat_shape)
        else:
            future_vol = np.tile(last_frame, repeat_shape)

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

    def permute(self, dims: Union[List[int], Tuple[int, ...]]) -> 'VolumeHandler':
        """
        Reorders the axes of the volume and updates the Ledger.
        NOTE: Review needed - primarily used in geometric tests.
        """
        dims_tuple = tuple(dims)
        self._data = self._data.permute(*dims_tuple) if torch.is_tensor(self._data) else np.transpose(self._data, dims_tuple)

        # Update Ledger
        new_axes = tuple(self._metadata.axes[i] for i in dims_tuple)
        self._metadata = replace(
            self._metadata,
            axes=new_axes,
            history=self._metadata.history + (("permute", dims_tuple),)
        )
        return self

    def flip(self, axis_label: str) -> 'VolumeHandler':
        """
        Flips the volume along a specific named axis and updates the Ledger history.
        NOTE: Critical for data augmentation in training loop.
        """
        idx = self.get_axis_idx(axis_label)
        self._data = torch.flip(self._data, dims=[idx]) if torch.is_tensor(self._data) else np.flip(self._data, axis=idx)

        self._metadata = replace(
            self._metadata,
            history=self._metadata.history + (("flip", axis_label),)
        )
        return self

    @property
    def data(self):
        return self._data
    @property
    def shape(self):
        return self._data.shape
    def __len__(self):
        return self._data.shape[self.get_axis_idx("T")]
    @property
    def axes(self):
        return self._metadata.axes
    @property
    def channel_map(self):
        return self._metadata.channel_map
    @property
    def id_col(self):
        return self._metadata.id_col
    @property
    def time_col(self):
        return self._metadata.time_col
    @property
    def spatial_cols(self):
        return self._metadata.spatial_cols
    @property
    def spatial_offset(self):
        return self._metadata.spatial_offset

    def get_axis_idx(self, label: str) -> int:
        return self._metadata.axes.index(label)
