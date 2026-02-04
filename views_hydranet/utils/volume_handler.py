"VolumeHandler: Authoritative Layout Management for Spatiotemporal Volumes."

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

# Internal Naming Invariants (ADR 020)
PRED_PREFIX = "pred_"
REG_SUFFIX = "_raw"
PROB_SUFFIX = "_prob"

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
        Automatically applies ADR 020 naming Engine.
        """
        # 1. Automated Naming (Internal Symmetry Gate)
        reg_names = [f"{n}_INTERNAL_SIGNAL" for n in base_names]
        prob_names = [f"{n}_INTERNAL_PROB" for n in base_names]
        full_signal_names = reg_names + prob_names

        if posterior_data.ndim == 5:
            if torch.is_tensor(posterior_data):
                # Assume [B=1, T, C, H, W] -> [T, H, W, C]
                work_data = posterior_data.squeeze(0).permute(0, 2, 3, 1)
                axes = ("T", "H", "W", "C")
            else:
                # Assume [T, H, W, C, S] -> Preserve Samples
                work_data = posterior_data
                axes = ("T", "H", "W", "C", "S")
        else:
            work_data = posterior_data
            axes = ("T", "H", "W", "C")

        return VolumeHandler(
            data=work_data,
            axes=axes,
            channel_map=full_signal_names,
            time_col=self._metadata.time_col,
            id_col=self._metadata.id_col,
            spatial_cols=self._metadata.spatial_cols,
            identity_cols=self._metadata.identity_cols,
            feature_cols=tuple(full_signal_names),
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
            raise NotImplementedError(f"Collapse method '{method}' is not defined in ADR 021.")

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
        mask = p_data[:, :, :, pg_idx] > 0
        indices = np.where(mask)

        reconstructed = {}
        # 4. Identities & Actuals from Provider
        for i, name in enumerate(provider.channel_map):
            if name in provider._metadata.identity_cols:
                vals = p_data[indices[0], indices[1], indices[2], i]
                if name in ["priogrid_gid", "month_id", "row", "col", "c_id"] or name == provider._metadata.time_col:
                    reconstructed[name] = vals.astype(int)
                else:
                    reconstructed[name] = vals
            elif name in provider._metadata.feature_cols:
                # ADR 007 Hardening: Prefix Actuals to prevent collision with Predictions
                vals = p_data[indices[0], indices[1], indices[2], i]
                reconstructed[f"ACTUAL_INTERNAL_{name}"] = vals.astype(np.float32)

        # 5. Extract Features/Predictions from Self
        for i, name in enumerate(self.channel_map):
            if name in reconstructed:
                continue
            if has_samples:
                vals = temp_data[indices[0], indices[1], indices[2], i, :]
                reconstructed[name] = [row.tolist() for row in vals]
            else:
                reconstructed[name] = temp_data[indices[0], indices[1], indices[2], i].astype(np.float32)

        df_out = pd.DataFrame(reconstructed)

        # 6. Automatic Symmetry Recovery (ADR 020)
        final_rename = {}
        for col in df_out.columns:
            if "INTERNAL_SIGNAL" in col:
                base = col.replace("_INTERNAL_SIGNAL", "")
                final_rename[col] = f"{PRED_PREFIX}{base}{REG_SUFFIX}"
            elif "INTERNAL_PROB" in col:
                base = col.replace("_INTERNAL_PROB", "")
                final_rename[col] = f"{PRED_PREFIX}{base}{PROB_SUFFIX}"
            elif "ACTUAL_INTERNAL_" in col:
                base = col.replace("ACTUAL_INTERNAL_", "")
                final_rename[col] = base

        df_out = df_out.rename(columns=final_rename)

        # 7. Automated Topographical Restoration (ADR 007)
        # Restore the MultiIndex using the authoritative Ledger roles
        time_col, id_col = self._metadata.time_col, self._metadata.id_col
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
        """
        idx = self.get_axis_idx(axis_label)
        self._data = torch.flip(self._data, dims=[idx]) if torch.is_tensor(self._data) else np.flip(self._data, axis=idx)

        self._metadata = replace(
            self._metadata,
            history=self._metadata.history + (("flip", axis_label),)
        )
        return self

    @property
    def data(self): return self._data
    @property
    def shape(self): return self._data.shape
    def __len__(self): return self._data.shape[self.get_axis_idx("T")]
    @property
    def axes(self): return self._metadata.axes
    @property
    def channel_map(self): return self._metadata.channel_map
    @property
    def id_col(self): return self._metadata.id_col
    @property
    def time_col(self): return self._metadata.time_col
    @property
    def spatial_cols(self): return self._metadata.spatial_cols
    @property
    def spatial_offset(self): return self._metadata.spatial_offset

    def get_axis_idx(self, label: str) -> int:
        return self._metadata.axes.index(label)
